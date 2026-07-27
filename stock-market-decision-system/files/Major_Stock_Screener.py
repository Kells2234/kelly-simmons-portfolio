import argparse
import os
import sys
import time
import random
import re
import pickle
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import warnings
import matplotlib.pyplot as plt

from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")
plt.switch_backend("Agg")

# --- NLTK one-time downloads ---
try:
    nltk.data.find("vader_lexicon")
except LookupError:
    print("Downloading NLTK sentiment data...")
    nltk.download("vader_lexicon", quiet=True)
    nltk.download("punkt", quiet=True)

# =============================================================================
# CONFIGURATION (hard-coded keys per your preference)
# =============================================================================
API_KEYS = {
    "alpha_vantage": "GQK46KXL3PH02QDG",
    "twelve_data": "51fba4e1c69d4f9c9ffc4434f7951e56",
    # IMPORTANT: your pasted finnhub key had a dot in it which will fail.
    # Put your correct key here:
    "finnhub": "6937ffd09b907566546327",
    "newsapi": "e157c612d34a48cea42a93a800b4bac0",
    "polygon": "DtJVz6in0NW7fa2OISYgpvXpaFZZvoLj",
}

# =============================================================================
# SIMPLE PERSISTENT CACHE (pickle files w/ TTL)
# =============================================================================
class DataCache:
    def __init__(self, cache_dir=".stock_cache", expiry_hours=12):
        self.cache_dir = cache_dir
        self.expiry_seconds = int(expiry_hours * 3600)
        os.makedirs(cache_dir, exist_ok=True)

    def _path(self, key: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", key)
        return os.path.join(self.cache_dir, f"{safe}.pkl")

    def get(self, key: str):
        p = self._path(key)
        if not os.path.exists(p):
            return None
        age = time.time() - os.path.getmtime(p)
        if age > self.expiry_seconds:
            return None
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def set(self, key: str, obj):
        p = self._path(key)
        try:
            with open(p, "wb") as f:
                pickle.dump(obj, f)
        except Exception:
            pass


# =============================================================================
# PROVIDER RATE LIMIT + CIRCUIT BREAKER
# =============================================================================
@dataclass
class ProviderPolicy:
    name: str
    max_calls: int
    period_sec: int
    cooldown_sec: int

class ProviderLimiter:
    def __init__(self, policy: ProviderPolicy):
        self.policy = policy
        self.calls: List[float] = []
        self.disabled_until: float = 0.0

    def can_call(self) -> bool:
        return time.time() >= self.disabled_until

    def disable(self):
        self.disabled_until = time.time() + self.policy.cooldown_sec

    def wait_if_needed(self):
        now = time.time()
        self.calls = [t for t in self.calls if now - t < self.policy.period_sec]
        if len(self.calls) >= self.policy.max_calls:
            sleep_time = self.policy.period_sec - (now - self.calls[0]) + 0.5
            if sleep_time > 0:
                time.sleep(sleep_time)
        # jitter to avoid burst collisions
        time.sleep(random.uniform(0.15, 0.45))
        self.calls.append(time.time())


def is_rate_limited_response(provider: str, status_code: int, data: Optional[dict]) -> bool:
    if status_code == 429:
        return True
    if not data:
        return False
    txt = str(data).lower()
    if "rate limit" in txt or "too many requests" in txt or "quota" in txt or "limit" in txt:
        return True
    if provider == "alpha_vantage":
        if "note" in (k.lower() for k in data.keys()):
            return True
        if "information" in (k.lower() for k in data.keys()):
            return True
    if provider == "twelve_data":
        if data.get("status") == "error":
            msg = (data.get("message") or "").lower()
            if "limit" in msg or "quota" in msg or "rate" in msg:
                return True
    return False


def safe_get_json(url: str, params: dict, provider: str, limiter: ProviderLimiter, timeout: int = 12) -> Optional[dict]:
    if not limiter.can_call():
        return None

    for attempt in range(5):
        limiter.wait_if_needed()
        try:
            r = requests.get(url, params=params, timeout=timeout)
            data = None
            try:
                data = r.json()
            except Exception:
                data = None

            if is_rate_limited_response(provider, r.status_code, data):
                # trip circuit breaker; back off
                limiter.disable()
                time.sleep(min(60, 2 ** attempt))
                return None

            if r.status_code != 200:
                time.sleep(min(20, 2 ** attempt))
                continue

            return data

        except Exception:
            time.sleep(min(20, 2 ** attempt))

    return None


# =============================================================================
# DATA SOURCE MANAGER (PRICE) - MINIMIZE API CALLS
# =============================================================================
class DataSourceManager:
    """
    Goal: avoid API limits by using Yahoo (yfinance) for price/volume whenever possible.
    Only use APIs if Yahoo fails for that ticker.
    """

    def __init__(self, cache: DataCache):
        self.cache = cache

        self.limiters = {
            "yahoo": ProviderLimiter(ProviderPolicy("yahoo", max_calls=80, period_sec=60, cooldown_sec=60)),
            "twelve_data": ProviderLimiter(ProviderPolicy("twelve_data", max_calls=8, period_sec=60, cooldown_sec=6 * 3600)),
            "alpha_vantage": ProviderLimiter(ProviderPolicy("alpha_vantage", max_calls=5, period_sec=60, cooldown_sec=6 * 3600)),
        }

    def get_data(self, ticker: str, start: datetime, end: datetime) -> Tuple[Optional[pd.Series], Optional[pd.Series], str]:
        key = f"px_{ticker}_{start:%Y%m%d}_{end:%Y%m%d}"
        cached = self.cache.get(key)
        if cached:
            return cached[0], cached[1], cached[2]

        # 1) Yahoo first (0 API quota, but still be gentle)
        close, vol = self._fetch_yahoo(ticker, start, end)
        if close is not None and not close.empty:
            self.cache.set(key, (close, vol, "Yahoo"))
            return close, vol, "Yahoo"

        # 2) Twelve Data fallback
        close, vol = self._fetch_twelve_data(ticker, start, end)
        if close is not None and not close.empty:
            self.cache.set(key, (close, vol, "Twelve Data"))
            return close, vol, "Twelve Data"

        # 3) Alpha Vantage fallback
        close, vol = self._fetch_alpha_vantage(ticker, start, end)
        if close is not None and not close.empty:
            self.cache.set(key, (close, vol, "Alpha Vantage"))
            return close, vol, "Alpha Vantage"

        return None, None, "None"

    def _fetch_yahoo(self, ticker: str, start: datetime, end: datetime) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
        try:
            self.limiters["yahoo"].wait_if_needed()
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start, end=end, auto_adjust=True)
            if hist is None or hist.empty:
                return None, None
            close = hist["Close"].rename(ticker)
            vol = hist["Volume"].rename(ticker) if "Volume" in hist.columns else pd.Series(dtype=float)
            return close, vol
        except Exception:
            return None, None

    def _fetch_twelve_data(self, ticker: str, start: datetime, end: datetime) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
        key = API_KEYS.get("twelve_data", "")
        if not key:
            return None, None

        limiter = self.limiters["twelve_data"]
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": ticker,
            "interval": "1day",
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "apikey": key,
            "outputsize": 500,
        }

        data = safe_get_json(url, params, "twelve_data", limiter)
        if not data or "values" not in data:
            return None, None

        try:
            df = pd.DataFrame(data["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime").sort_index()
            close = df["close"].astype(float).rename(ticker)
            # some TD plans don’t include volume for equities; handle gracefully
            vol = df["volume"].astype(float).rename(ticker) if "volume" in df.columns else pd.Series(index=df.index, dtype=float, name=ticker)
            return close, vol
        except Exception:
            return None, None

    def _fetch_alpha_vantage(self, ticker: str, start: datetime, end: datetime) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
        key = API_KEYS.get("alpha_vantage", "")
        if not key:
            return None, None

        limiter = self.limiters["alpha_vantage"]
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "apikey": key,
            "outputsize": "compact",
        }

        data = safe_get_json(url, params, "alpha_vantage", limiter)
        if not data or "Time Series (Daily)" not in data:
            return None, None

        try:
            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df = df[(df.index >= start) & (df.index <= end)]
            close = df["4. close"].astype(float).rename(ticker)
            vol = df["5. volume"].astype(float).rename(ticker)
            return close, vol
        except Exception:
            return None, None


# =============================================================================
# SENTIMENT ANALYZER (API calls only for top K + strong cache)
# =============================================================================
class SentimentAnalyzer:
    def __init__(self, cache: DataCache):
        self.sia = SentimentIntensityAnalyzer()
        self.cache = cache

        self.limiters = {
            "finnhub": ProviderLimiter(ProviderPolicy("finnhub", max_calls=55, period_sec=60, cooldown_sec=6 * 3600)),
            "newsapi": ProviderLimiter(ProviderPolicy("newsapi", max_calls=90, period_sec=24 * 3600, cooldown_sec=24 * 3600)),
        }

    def analyze_stock(self, ticker: str, use_api: bool = True) -> Dict:
        # Cache sentiment longer (12h) to avoid repeat news pulls
        key = f"sent_{ticker}_{datetime.now():%Y%m%d}"
        cached = self.cache.get(key)
        if cached:
            return cached

        sentiment_scores = []
        news_count = 0
        news_items = []

        news_sentiment, news_items = self._get_news_sentiment(ticker, use_api=use_api)
        if news_sentiment is not None:
            sentiment_scores.append(news_sentiment)
            news_count = len(news_items)

        social_sentiment = random.uniform(-0.2, 0.2)
        sentiment_scores.append(social_sentiment * 0.5)

        tech_sentiment = self._get_technical_sentiment(ticker)
        sentiment_scores.append(tech_sentiment * 0.8)

        analyst_sentiment = self._get_analyst_sentiment(ticker, use_api=use_api)
        if analyst_sentiment is not None:
            sentiment_scores.append(analyst_sentiment * 0.9)

        overall_score = float(np.mean(sentiment_scores)) if sentiment_scores else 0.0

        if overall_score >= 0.3:
            label = "STRONGLY BULLISH"
        elif overall_score >= 0.1:
            label = "BULLISH"
        elif overall_score >= -0.1:
            label = "NEUTRAL"
        elif overall_score >= -0.3:
            label = "BEARISH"
        else:
            label = "STRONGLY BEARISH"

        result = {
            "overall_score": overall_score,
            "news_sentiment": float(news_sentiment) if news_sentiment is not None else 0.0,
            "social_sentiment": float(social_sentiment),
            "technical_sentiment": float(tech_sentiment),
            "analyst_sentiment": float(analyst_sentiment) if analyst_sentiment is not None else 0.0,
            "news_count": int(news_count),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "sentiment_label": label,
            "news_headlines": news_items[:3] if news_items else [],
        }

        self.cache.set(key, result)
        return result

    def _get_news_sentiment(self, ticker: str, use_api: bool = True) -> Tuple[Optional[float], List[Dict]]:
        if not use_api:
            return None, []

        sentiments = []
        items: List[Dict] = []

        # --- Finnhub first (but still limited) ---
        finnhub_key = API_KEYS.get("finnhub", "")
        if finnhub_key:
            url = "https://finnhub.io/api/v1/company-news"
            params = {
                "symbol": ticker,
                "from": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
                "to": datetime.now().strftime("%Y-%m-%d"),
                "token": finnhub_key,
            }
            data = safe_get_json(url, params, "finnhub", self.limiters["finnhub"])
            if isinstance(data, list) and data:
                for it in data[:8]:
                    headline = it.get("headline", "") or ""
                    summary = it.get("summary", "") or ""
                    text = f"{headline} {summary}".strip()
                    s = self._analyze_text_sentiment(text)
                    sentiments.append(s)
                    items.append({"headline": headline[:100], "sentiment": s, "source": "Finnhub"})

        # --- NewsAPI fallback (daily quota; only if Finnhub failed) ---
        if not sentiments:
            news_key = API_KEYS.get("newsapi", "")
            if news_key:
                url = "https://newsapi.org/v2/everything"
                params = {
                    "q": ticker,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": 8,
                    "apiKey": news_key,
                }
                data = safe_get_json(url, params, "newsapi", self.limiters["newsapi"])
                if data and "articles" in data:
                    for a in data.get("articles", [])[:5]:
                        title = a.get("title", "") or ""
                        desc = a.get("description", "") or ""
                        text = f"{title} {desc}".strip()
                        s = self._analyze_text_sentiment(text)
                        sentiments.append(s)
                        items.append({"headline": title[:100], "sentiment": s, "source": a.get("source", {}).get("name", "NewsAPI")})

        if sentiments:
            return float(np.mean(sentiments)), items
        return None, []

    def _get_analyst_sentiment(self, ticker: str, use_api: bool = True) -> Optional[float]:
        if not use_api:
            return None
        finnhub_key = API_KEYS.get("finnhub", "")
        if not finnhub_key:
            return None

        url = "https://finnhub.io/api/v1/stock/recommendation"
        params = {"symbol": ticker, "token": finnhub_key}
        data = safe_get_json(url, params, "finnhub", self.limiters["finnhub"])
        if not data or not isinstance(data, list):
            return None
        if not data:
            return None

        latest = data[0]
        total = sum(float(v) for v in latest.values() if isinstance(v, (int, float)))
        if total <= 0:
            return None

        score = (
            latest.get("strongBuy", 0) * 1.0
            + latest.get("buy", 0) * 0.5
            + latest.get("hold", 0) * 0.0
            + latest.get("sell", 0) * -0.5
            + latest.get("strongSell", 0) * -1.0
        ) / total
        return float(score)

    def _get_technical_sentiment(self, ticker: str) -> float:
        # Uses Yahoo (no quota) and is internally rate-limited by yfinance
        try:
            end = datetime.now()
            start = end - timedelta(days=30)
            hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
            if hist is None or hist.empty or len(hist) < 10:
                return 0.0

            prices = hist["Close"]

            ret_5d = (prices.iloc[-1] / prices.iloc[-6] - 1) * 100 if len(prices) >= 6 else 0.0

            vol_ratio = 1.0
            if "Volume" in hist.columns and len(hist["Volume"]) >= 10:
                vol_ratio = hist["Volume"].iloc[-5:].mean() / max(1.0, hist["Volume"].iloc[-10:-5].mean())

            gains, losses = [], []
            lookback = min(15, len(prices) - 1)
            for i in range(1, lookback + 1):
                chg = prices.iloc[-i] - prices.iloc[-i - 1]
                if chg > 0:
                    gains.append(chg)
                else:
                    losses.append(abs(chg))
            avg_gain = sum(gains) / len(gains) if gains else 0.0
            avg_loss = sum(losses) / len(losses) if losses else 0.001
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            price_sent = np.tanh(ret_5d / 10)
            vol_sent = np.tanh((vol_ratio - 1) * 2)
            rsi_sent = (rsi - 50) / 50

            tech_sent = price_sent * 0.4 + vol_sent * 0.3 + rsi_sent * 0.3
            return float(max(-1, min(1, tech_sent)))
        except Exception:
            return 0.0

    def _analyze_text_sentiment(self, text: str) -> float:
        if not text or len(text.strip()) < 10:
            return 0.0
        text = re.sub(r"[^\w\s]", " ", text.lower())

        vader_score = self.sia.polarity_scores(text)["compound"]
        try:
            blob_score = TextBlob(text).sentiment.polarity
        except Exception:
            blob_score = 0.0

        financial_keywords = {
            "bullish": 0.5, "bearish": -0.5, "buy": 0.4, "sell": -0.4,
            "strong buy": 0.8, "strong sell": -0.8, "upgrade": 0.6,
            "downgrade": -0.6, "beat": 0.3, "miss": -0.3, "growth": 0.2,
            "decline": -0.2, "positive": 0.3, "negative": -0.3
        }

        keyword_score = 0.0
        keyword_count = 0
        for k, w in financial_keywords.items():
            if k in text:
                keyword_score += w
                keyword_count += 1
        if keyword_count:
            keyword_score /= keyword_count

        final = vader_score * 0.5 + blob_score * 0.3 + keyword_score * 0.2
        return float(max(-1, min(1, final)))


# =============================================================================
# STOCK ANALYSIS (2-pass: tech then sentiment only for top K)
# =============================================================================
def analyze_stocks_with_sentiment(
    symbols: List[str],
    price_min: float,
    price_max: float,
    top_n: int,
    exchange_name: str,
    include_sentiment: bool = True,
    sentiment_weight: float = 0.3,
    sentiment_top_k: int = 10,
):
    print(f"\nAnalyzing {len(symbols)} symbols (anti-limit mode)...")

    cache = DataCache(expiry_hours=12)
    data_manager = DataSourceManager(cache=cache)
    sentiment_analyzer = SentimentAnalyzer(cache=DataCache(cache_dir=".sentiment_cache", expiry_hours=12)) if include_sentiment else None

    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)

    tech_rows = []

    # PASS 1: Technical only (Yahoo-first, cached)
    for i, symbol in enumerate(symbols, 1):
        print(f"  [TECH] {i}/{len(symbols)}: {symbol:6}", end="\r")

        close, volume, src = data_manager.get_data(symbol, start_date, end_date)
        if close is None or close.empty or len(close) < 10:
            continue

        current_price = float(close.iloc[-1])
        if not (price_min <= current_price <= price_max):
            continue

        ret_5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) >= 6 else 0.0
        ret_20d = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) >= 21 else 0.0

        volume_score = 0
        if volume is not None and len(volume) >= 10:
            avg5 = float(volume.iloc[-5:].mean())
            if avg5 > 1_000_000:
                volume_score = 1

        tech_score = float(ret_5d + (ret_20d * 0.5) + (volume_score * 25))

        tech_rows.append({
            "ticker": symbol,
            "Exchange": exchange_name,
            "Price": current_price,
            "Technical_Score": tech_score,
            "Return_5D": float(ret_5d),
            "Return_20D": float(ret_20d),
            "Volume_Score": int(volume_score),
            "Price_Source": src,
        })

    print("\n")  # newline

    if not tech_rows:
        return pd.DataFrame()

    tech_df = pd.DataFrame(tech_rows).sort_values("Technical_Score", ascending=False)

    # Select top K for sentiment (BIGGEST limiter reduction)
    if include_sentiment:
        candidates = tech_df.head(max(sentiment_top_k, top_n)).copy()
    else:
        candidates = tech_df.head(top_n).copy()

    # PASS 2: Sentiment only for top K
    if include_sentiment and sentiment_analyzer:
        sent_rows = []
        for j, sym in enumerate(candidates["ticker"].tolist(), 1):
            print(f"  [SENT] {j}/{len(candidates)}: {sym:6}", end="\r")
            sd = sentiment_analyzer.analyze_stock(sym, use_api=True)
            sent_rows.append({
                "ticker": sym,
                "Sentiment_Score": float(sd["overall_score"]),
                "Sentiment_Label": sd["sentiment_label"],
                "News_Sentiment": float(sd["news_sentiment"]),
                "Technical_Sentiment": float(sd["technical_sentiment"]),
                "News_Count": int(sd["news_count"]),
                "Analyst_Sentiment": float(sd.get("analyst_sentiment", 0.0)),
            })
        print("\n")
        sent_df = pd.DataFrame(sent_rows)

        out = candidates.merge(sent_df, on="ticker", how="left")

        # Combine scores
        out["Combined_Score"] = (
            out["Technical_Score"] * (1 - sentiment_weight)
            + (out["Sentiment_Score"] * 100.0) * sentiment_weight
        )
        out = out.sort_values("Combined_Score", ascending=False).head(top_n)
        return out

    # No sentiment
    return candidates.sort_values("Technical_Score", ascending=False).head(top_n)


# =============================================================================
# SYMBOL MANAGEMENT
# =============================================================================
def get_curated_symbols(exchange="nasdaq", limit=40):
    curated = {
        "nasdaq": [
            # Mega / Large Cap Tech
            "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "ADBE",
            "AMD", "INTC", "CSCO", "QCOM", "TXN",

            # Growth / AI / Semi
            "ARM", "MU", "AMAT", "LRCX", "ASML", "SMCI",

            # Consumer / Platform
            "NFLX", "COST", "PEP", "INTU", "BKNG", "SBUX",

            # Healthcare / Biotech
            "REGN", "VRTX", "GILD", "ISRG", "MRNA",

            # Fintech / Payments
            "PYPL", "SQ", "ADP"
        ],

        "nyse": [
            # Mega / Blue Chip
            "BRK-B", "JPM", "JNJ", "V", "MA", "UNH", "PG", "HD", "DIS",

            # Financials
            "BAC", "MS", "GS", "C", "AXP", "SCHW",

            # Industrials / Energy
            "CAT", "DE", "BA", "RTX", "LMT", "GE",
            "XOM", "CVX", "SLB",

            # Healthcare
            "ABBV", "LLY", "MRK", "PFE",

            # Consumer
            "KO", "PEP", "NKE", "MCD"
        ],

        "amex": [
            # ETFs (excellent for sentiment + technicals)
            "SPY", "QQQ", "IWM", "DIA",
            "XLK", "XLF", "XLE", "XLV",
            "SMH", "SOXX"
        ],
    }

    return curated.get(exchange, [])[:limit]
 
# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Stock Screener with Sentiment Analysis (Anti-limit)")
    ap.add_argument("--exchange", type=str, default="nasdaq", choices=["nasdaq", "nyse", "amex"])
    ap.add_argument("--symbols", type=int, default=15, help="Max symbols to check")
    ap.add_argument("--out", type=str, default="stock_picks_with_sentiment.csv")
    ap.add_argument("--price_min", type=float, default=101.0)
    ap.add_argument("--price_max", type=float, default=300.0)
    ap.add_argument("--top_n", type=int, default=8)
    ap.add_argument("--sentiment_weight", type=float, default=0.30)
    ap.add_argument("--sentiment_top_k", type=int, default=10, help="Only run sentiment APIs for top K tech candidates")
    ap.add_argument("--no_sentiment", action="store_true", help="Disable sentiment analysis")
    args = ap.parse_args()

    print("=" * 80)
    print("STOCK SCREENER WITH SENTIMENT ANALYSIS (ANTI-LIMIT)")
    print("=" * 80)
    print(f"Exchange: {args.exchange.upper()}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Price Range: ${args.price_min} - ${args.price_max}")
    print(f"Max Symbols: {args.symbols}")
    print(f"Sentiment: {'DISABLED' if args.no_sentiment else 'ENABLED'}")
    if not args.no_sentiment:
        print(f"Sentiment Weight: {args.sentiment_weight:.0%}")
        print(f"Sentiment Top-K (API calls): {args.sentiment_top_k}")
    print("=" * 80)

    symbols = get_curated_symbols(exchange=args.exchange, limit=args.symbols)
    print(f"\nAnalyzing {len(symbols)} curated symbols...")

    results = analyze_stocks_with_sentiment(
        symbols=symbols,
        price_min=args.price_min,
        price_max=args.price_max,
        top_n=args.top_n,
        exchange_name=args.exchange.upper(),
        include_sentiment=not args.no_sentiment,
        sentiment_weight=args.sentiment_weight,
        sentiment_top_k=args.sentiment_top_k,
    )

    if results.empty:
        print("\nNo results generated.")
        return

    results.to_csv(args.out, index=False)

    print("\n" + "=" * 80)
    print(f"TOP {len(results)} PICKS")
    print("=" * 80)

    cols = ["ticker", "Price", "Technical_Score", "Return_5D", "Return_20D", "Price_Source"]
    if not args.no_sentiment:
        cols += ["Combined_Score", "Sentiment_Score", "Sentiment_Label", "News_Count"]

    df_disp = results[cols].copy()
    df_disp["Price"] = df_disp["Price"].apply(lambda x: f"${x:.2f}")
    df_disp["Return_5D"] = df_disp["Return_5D"].apply(lambda x: f"{x:+.1f}%")
    df_disp["Return_20D"] = df_disp["Return_20D"].apply(lambda x: f"{x:+.1f}%")
    df_disp["Technical_Score"] = df_disp["Technical_Score"].apply(lambda x: f"{x:.1f}")
    if "Combined_Score" in df_disp.columns:
        df_disp["Combined_Score"] = df_disp["Combined_Score"].apply(lambda x: f"{x:.1f}")
    if "Sentiment_Score" in df_disp.columns:
        df_disp["Sentiment_Score"] = df_disp["Sentiment_Score"].apply(lambda x: f"{x:+.3f}")

    print(df_disp.to_string(index=False))

    print(f"\nResults saved to: {args.out}")

    summary_file = args.out.replace(".csv", "_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"Stock Screener Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Top {len(results)} picks from {args.exchange.upper()}\n")
        f.write(f"Price range: ${args.price_min} - ${args.price_max}\n")
        if args.no_sentiment:
            f.write("Sentiment: disabled\n\n")
        else:
            f.write(f"Sentiment weight: {args.sentiment_weight:.0%}\n")
            f.write(f"Sentiment applied to top K: {args.sentiment_top_k}\n\n")

        for idx, row in results.reset_index(drop=True).iterrows():
            f.write(f"{idx+1}. {row['ticker']} - ${float(row['Price']):.2f}\n")
            f.write(f"   5D Return: {float(row['Return_5D']):.1f}% | 20D Return: {float(row['Return_20D']):.1f}%\n")
            if not args.no_sentiment:
                f.write(f"   Sentiment: {row.get('Sentiment_Label','')} (Score: {float(row.get('Sentiment_Score',0)):+.3f})\n")
                if int(row.get("News_Count", 0)) > 0:
                    f.write(f"   Recent News: {int(row.get('News_Count',0))} articles\n")
            f.write("\n")

    print(f"Summary report: {summary_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
