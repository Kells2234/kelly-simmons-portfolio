# api_forex_discovery_cached.py
# Cached + fallback version, with hard-coded API keys (per your preference).
#
# Key improvements vs your original:
# - Eliminates per-pair "real-time price" API calls (current price = last OHLC close)
# - Fetches OHLC once per pair, computes indicators locally
# - Caches discovery + OHLC responses in SQLite with TTL
# - Fallback: Twelve Data -> Alpha Vantage -> Yahoo Finance
# - Retry/backoff for rate limit messages / 429

from __future__ import annotations

import os
import sys
import time
import json
import math
import hashlib
import sqlite3
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib import rcParams

warnings.filterwarnings("ignore")

# =============================================================================
# API CONFIGURATION (HARDCODED KEYS)
# =============================================================================
API_KEYS = {
    "alpha_vantage": "GQK46KXL3PH02QDG",
    "twelve_data": "51fba4e1c69d4f9c9ffc4434f7951e56",
    "finnhub": "6937ffd09b907566546327",
    "exchangerate_api": "8f9f7b1c7a9e4c6a8b5d3e2f1",
}

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================
CACHE_DB_PATH = "forex_cache.sqlite"
CACHE_TTL_SECONDS = 23 * 3600  # ~23 hours
DISCOVERY_CACHE_SECONDS = 7 * 24 * 3600

HTTP_TIMEOUT = 20
MAX_RETRIES = 4
MIN_SECONDS_BETWEEN_CALLS = 1.2

ANALYSIS_PERIOD_DAYS = 20
ANALYSIS_INTERVAL = "1h"  # primary analysis interval

DISCOVERY_MAX_PAIRS = 30  # analyze first N after dedupe
REPORT_PERIOD_FOR_CHARTS = "5d"
CHART_INTERVAL = "15m"


# =============================================================================
# SQLITE CACHE
# =============================================================================
class SqliteCache:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path)
        self._init()

    def _init(self):
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL,
                ttl INTEGER NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def get(self, key: str):
        cur = self.conn.cursor()
        cur.execute("SELECT created_at, ttl, payload FROM cache WHERE key = ?", (key,))
        row = cur.fetchone()
        if not row:
            return None
        created_at, ttl, payload = row
        now = int(time.time())
        if now > created_at + ttl:
            return None
        try:
            return json.loads(payload)
        except Exception:
            return None

    def set(self, key: str, payload, ttl: int):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO cache(key, created_at, ttl, payload) VALUES(?,?,?,?)",
            (key, int(time.time()), int(ttl), json.dumps(payload)),
        )
        self.conn.commit()

    def close(self):
        self.conn.close()


def cache_key(namespace: str, params: dict) -> str:
    raw = json.dumps({"ns": namespace, "params": params}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class RateLimiter:
    def __init__(self, min_interval_sec: float):
        self.min_interval = float(min_interval_sec)
        self.last_call = 0.0

    def wait(self):
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()


# =============================================================================
# REQUEST WRAPPER (retries + backoff + rate-limit detection)
# =============================================================================
def looks_rate_limited(provider: str, data: dict) -> bool:
    txt = json.dumps(data).lower()
    if "rate limit" in txt or "too many requests" in txt or "quota" in txt or "limit" in txt:
        return True
    if provider == "alpha_vantage":
        # Alpha often returns {"Note": "..."} or {"Information": "..."}
        if "note" in data or "information" in data:
            return True
    if provider == "twelve_data":
        if data.get("status") == "error":
            msg = (data.get("message") or "").lower()
            if "limit" in msg or "quota" in msg or "rate" in msg:
                return True
    return False


def request_json(url: str, params: dict, provider: str, limiter: RateLimiter) -> dict:
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            limiter.wait()
            resp = requests.get(url, params=params, timeout=HTTP_TIMEOUT)

            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue

            resp.raise_for_status()
            data = resp.json()

            if looks_rate_limited(provider, data):
                time.sleep(2 ** attempt)
                continue

            return data

        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)

    raise RuntimeError(f"{provider} request failed after retries: {last_err}")


# =============================================================================
# PAIR UTILITIES
# =============================================================================
def normalize_pair(display_symbol: str) -> tuple[str, str]:
    base, quote = display_symbol.replace("-", "/").split("/")
    return base.strip().upper(), quote.strip().upper()


def compact_symbol(display_symbol: str) -> str:
    b, q = normalize_pair(display_symbol)
    return f"{b}{q}"


def classify_pair(display_symbol: str) -> str:
    majors = {("EUR", "USD"), ("GBP", "USD"), ("USD", "JPY"), ("USD", "CHF"), ("AUD", "USD"), ("USD", "CAD"), ("NZD", "USD")}
    b, q = normalize_pair(display_symbol)
    if (b, q) in majors or (q, b) in majors:
        return "Major"
    exotics = {"TRY", "HKD", "SGD", "ZAR", "MXN", "CNH"}
    if b in exotics or q in exotics:
        return "Exotic"
    return "Minor"


# =============================================================================
# DISCOVERY (cached, mostly static to avoid discovery API costs)
# =============================================================================
def discover_forex_pairs_cached(cache: SqliteCache, limiter: RateLimiter) -> list[dict]:
    ck = cache_key("discovery_pairs_v2", {"v": 2})
    cached = cache.get(ck)
    if cached:
        return cached

    print("\n[Discovery] Building pair universe (cached)...")

    major_pairs = [
        "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
        "AUD/USD", "USD/CAD", "NZD/USD", "EUR/GBP",
        "EUR/JPY", "GBP/JPY", "EUR/CHF", "AUD/JPY"
    ]
    minor_exotic_pairs = [
        "EUR/AUD", "GBP/AUD", "EUR/CAD", "GBP/CAD",
        "EUR/NZD", "GBP/NZD", "AUD/CAD", "AUD/NZD",
        "CAD/JPY", "NZD/JPY", "USD/SGD", "USD/HKD",
        "USD/MXN", "USD/ZAR", "USD/TRY", "USD/CNH",
        "EUR/SGD", "GBP/CHF", "CHF/JPY", "AUD/CHF", "NZD/CHF"
    ]

    pairs = []
    for sym in major_pairs + minor_exotic_pairs:
        pairs.append({
            "symbol": compact_symbol(sym),
            "display_symbol": sym,
            "base": sym.split("/")[0],
            "quote": sym.split("/")[1],
            "type": classify_pair(sym),
            "source": "Static Universe (Cached)"
        })

    # Optional: ping discovery endpoints ONCE just to confirm connectivity (does not expand list)
    if API_KEYS.get("twelve_data"):
        try:
            url = "https://api.twelvedata.com/forex_pairs"
            params = {"apikey": API_KEYS["twelve_data"]}
            data = request_json(url, params, "twelve_data", limiter)
            if "data" in data:
                print(f"   Twelve Data discovery OK (reported {len(data['data'])} pairs).")
        except Exception as e:
            print(f"   Twelve Data discovery skipped: {str(e)[:120]}")

    if API_KEYS.get("finnhub"):
        try:
            url = "https://finnhub.io/api/v1/forex/symbols"
            params = {"exchange": "oanda", "token": API_KEYS["finnhub"]}
            data = request_json(url, params, "finnhub", limiter)
            if isinstance(data, list) and data:
                print(f"   Finnhub discovery OK (reported {len(data)} symbols).")
        except Exception as e:
            print(f"   Finnhub discovery skipped: {str(e)[:120]}")

    # Deduplicate
    uniq, seen = [], set()
    for p in pairs:
        if p["symbol"] not in seen:
            seen.add(p["symbol"])
            uniq.append(p)

    cache.set(ck, uniq, ttl=DISCOVERY_CACHE_SECONDS)
    return uniq


# =============================================================================
# OHLC FETCH (cached + fallback)
# =============================================================================
def ohlc_from_twelve_data(display_symbol: str, interval: str, outputsize: int,
                         cache: SqliteCache, limiter: RateLimiter) -> pd.DataFrame:
    if not API_KEYS.get("twelve_data"):
        raise RuntimeError("Missing Twelve Data API key")

    b, q = normalize_pair(display_symbol)
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": f"{b}/{q}",
        "interval": interval,
        "outputsize": int(outputsize),
        "apikey": API_KEYS["twelve_data"],
        "format": "JSON",
    }

    ck = cache_key("twelve_time_series", params)
    cached = cache.get(ck)
    if cached:
        data = cached
    else:
        data = request_json(url, params, "twelve_data", limiter)
        cache.set(ck, data, CACHE_TTL_SECONDS)

    if data.get("status") == "error":
        raise RuntimeError(f"Twelve Data error: {data.get('message')}")

    values = data.get("values") or []
    if not values:
        raise RuntimeError("Twelve Data returned no values")

    values = list(reversed(values))  # newest-first -> oldest-first
    df = pd.DataFrame(values)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)

    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    df["Volume"] = 0.0
    return df[["Open", "High", "Low", "Close", "Volume"]]


def ohlc_from_alpha_vantage(display_symbol: str, interval: str,
                           cache: SqliteCache, limiter: RateLimiter) -> pd.DataFrame:
    if not API_KEYS.get("alpha_vantage"):
        raise RuntimeError("Missing Alpha Vantage API key")

    b, q = normalize_pair(display_symbol)
    interval_map = {"1h": "60min", "15m": "15min", "30m": "30min", "5m": "5min", "1m": "1min"}
    av_interval = interval_map.get(interval, "60min")

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "FX_INTRADAY",
        "from_symbol": b,
        "to_symbol": q,
        "interval": av_interval,
        "outputsize": "compact",
        "apikey": API_KEYS["alpha_vantage"],
    }

    ck = cache_key("av_fx_intraday", params)
    cached = cache.get(ck)
    if cached:
        data = cached
    else:
        data = request_json(url, params, "alpha_vantage", limiter)
        cache.set(ck, data, CACHE_TTL_SECONDS)

    ts_key = None
    for k in data.keys():
        if "Time Series FX" in k:
            ts_key = k
            break

    if not ts_key:
        if "Note" in data or "Information" in data:
            raise RuntimeError(data.get("Note") or data.get("Information"))
        raise RuntimeError(f"Alpha Vantage unexpected response keys: {list(data.keys())}")

    series = data[ts_key]
    rows = []
    for ts, row in series.items():
        rows.append({
            "datetime": pd.to_datetime(ts),
            "Open": float(row["1. open"]),
            "High": float(row["2. high"]),
            "Low": float(row["3. low"]),
            "Close": float(row["4. close"]),
            "Volume": 0.0,
        })

    df = pd.DataFrame(rows).sort_values("datetime").set_index("datetime")
    return df[["Open", "High", "Low", "Close", "Volume"]]


def ohlc_from_yahoo(display_symbol: str, period: str, interval: str) -> pd.DataFrame:
    sym = compact_symbol(display_symbol)
    yahoo_symbol = f"{sym}=X"
    t = yf.Ticker(yahoo_symbol)
    hist = t.history(period=period, interval=interval)
    if hist is None or hist.empty:
        raise RuntimeError("Yahoo returned no data")
    hist = hist.rename(columns=str.title)
    if "Volume" not in hist.columns:
        hist["Volume"] = 0.0
    return hist[["Open", "High", "Low", "Close", "Volume"]]


def get_ohlc_with_fallback(display_symbol: str, period_days: int, interval: str,
                           cache: SqliteCache, limiter: RateLimiter) -> tuple[pd.DataFrame, str]:
    outputsize = int(period_days * 24 + 250)
    errors = []

    try:
        df = ohlc_from_twelve_data(display_symbol, interval, outputsize, cache, limiter)
        return df, "Twelve Data"
    except Exception as e:
        errors.append(f"Twelve Data: {e}")

    try:
        df = ohlc_from_alpha_vantage(display_symbol, interval, cache, limiter)
        return df, "Alpha Vantage"
    except Exception as e:
        errors.append(f"Alpha Vantage: {e}")

    try:
        period = f"{max(5, period_days)}d"
        df = ohlc_from_yahoo(display_symbol, period=period, interval=interval)
        return df, "Yahoo Finance"
    except Exception as e:
        errors.append(f"Yahoo Finance: {e}")

    raise RuntimeError("All sources failed | " + " | ".join(errors))


# =============================================================================
# INDICATORS (your logic preserved)
# =============================================================================
def calculate_rsi(prices: np.ndarray, period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def calculate_sma(prices: np.ndarray, period: int) -> float:
    if len(prices) < period:
        return float(prices[-1]) if len(prices) else 0.0
    return float(np.mean(prices[-period:]))


def calculate_ema(prices: np.ndarray, period: int) -> float:
    if len(prices) < period:
        return float(prices[-1]) if len(prices) else 0.0
    weights = np.exp(np.linspace(-1.0, 0.0, period))
    weights /= weights.sum()
    tail = prices[-period * 2:] if len(prices) >= period * 2 else prices
    return float(np.convolve(tail, weights, mode="valid")[-1])


def calculate_macd(prices: np.ndarray) -> tuple[float, float, float]:
    if len(prices) < 26:
        return 0.0, 0.0, 0.0
    ema12 = calculate_ema(prices, 12)
    ema26 = calculate_ema(prices, 26)
    macd_line = ema12 - ema26
    signal_line = calculate_ema(prices[-9:], 9) if len(prices) >= 9 else 0.0
    hist = macd_line - signal_line
    return float(macd_line), float(signal_line), float(hist)


# =============================================================================
# ANALYSIS (one OHLC pull per pair)
# =============================================================================
def analyze_forex_pair(pair_info: dict, cache: SqliteCache, limiter: RateLimiter):
    display_symbol = pair_info["display_symbol"]
    pair_type = pair_info["type"]
    symbol = pair_info["symbol"]

    try:
        print(f"  Analyzing {display_symbol} ({pair_type})...", end="\r")

        hist, source = get_ohlc_with_fallback(
            display_symbol=display_symbol,
            period_days=ANALYSIS_PERIOD_DAYS,
            interval=ANALYSIS_INTERVAL,
            cache=cache,
            limiter=limiter,
        )

        if hist is None or hist.empty or len(hist) < 80:
            return None

        prices = hist["Close"].values.astype(float)
        price = float(prices[-1])  # current price = last close

        rsi = calculate_rsi(prices, 14)
        sma_20 = calculate_sma(prices, 20)
        sma_50 = calculate_sma(prices, 50)
        macd_line, signal_line, macd_histogram = calculate_macd(prices)

        hourly_change = ((prices[-1] - prices[-2]) / prices[-2]) * 100 if len(prices) >= 2 else 0.0
        daily_change = ((prices[-1] - prices[-24]) / prices[-24]) * 100 if len(prices) >= 24 else 0.0
        weekly_change = ((prices[-1] - prices[-168]) / prices[-168]) * 100 if len(prices) >= 168 else 0.0

        recent_high = float(np.max(prices[-24:])) if len(prices) >= 24 else price
        recent_low = float(np.min(prices[-24:])) if len(prices) >= 24 else price
        resistance = recent_high * 1.001
        support = recent_low * 0.999

        score = 50

        # RSI scoring
        if rsi < 30:
            score += 20
        elif rsi > 70:
            score -= 20
        elif 30 <= rsi <= 50:
            score += 5
        elif 50 <= rsi <= 70:
            score -= 5

        # Trend scoring
        if price > sma_20 > sma_50:
            score += 15
        elif price < sma_20 < sma_50:
            score -= 15

        # MACD scoring
        if macd_line > signal_line and macd_histogram > 0:
            score += 10
        elif macd_line < signal_line and macd_histogram < 0:
            score -= 10

        # Momentum scoring
        if hourly_change > 0.1:
            score += 5
        elif hourly_change < -0.1:
            score -= 5

        # Support/Resistance scoring
        distance_to_resistance = (resistance - price) / price * 100
        distance_to_support = (price - support) / price * 100
        if distance_to_resistance < 0.2:
            score -= 10
        if distance_to_support < 0.2:
            score += 10

        # Volatility adjustment
        volatility = (np.std(prices[-24:]) / np.mean(prices[-24:]) * 100) if len(prices) >= 24 else 1.0
        if volatility > 0.5:
            score -= 5

        # Pair type adjustment
        if pair_type == "Major":
            score += 5
        elif pair_type == "Exotic":
            score -= 5

        score = max(0, min(100, score))

        if score >= 75:
            signal = "STRONG BUY"
            signal_color = "GREEN"
        elif score >= 60:
            signal = "BUY"
            signal_color = "LIGHT_GREEN"
        elif score <= 25:
            signal = "STRONG SELL"
            signal_color = "RED"
        elif score <= 40:
            signal = "SELL"
            signal_color = "LIGHT_RED"
        else:
            signal = "NEUTRAL/HOLD"
            signal_color = "YELLOW"

        if "BUY" in signal:
            stop_loss = price * 0.995
            take_profit_1 = price * 1.005
            take_profit_2 = price * 1.01
        elif "SELL" in signal:
            stop_loss = price * 1.005
            take_profit_1 = price * 0.995
            take_profit_2 = price * 0.99
        else:
            stop_loss = take_profit_1 = take_profit_2 = price

        pip_value = 0.0001 if "JPY" not in display_symbol else 0.01
        atr = float(np.mean(np.abs(prices[-14:] - np.roll(prices[-14:], 1)))) if len(prices) >= 14 else 0.001

        return {
            "Symbol": symbol,
            "Display_Symbol": display_symbol,
            "Base_Currency": pair_info["base"],
            "Quote_Currency": pair_info["quote"],
            "Pair_Type": pair_type,
            "Current_Price": price,
            "Signal": signal,
            "Signal_Color": signal_color,
            "Score": score,
            "RSI": float(rsi),
            "Hourly_Change_%": float(hourly_change),
            "Daily_Change_%": float(daily_change),
            "Weekly_Change_%": float(weekly_change),
            "SMA_20": float(sma_20),
            "SMA_50": float(sma_50),
            "MACD_Line": float(macd_line),
            "Signal_Line": float(signal_line),
            "Resistance": float(resistance),
            "Support": float(support),
            "Stop_Loss": float(stop_loss),
            "Take_Profit_1": float(take_profit_1),
            "Take_Profit_2": float(take_profit_2),
            "ATR": float(atr),
            "Pip_Value": float(pip_value),
            "Volatility_%": float(volatility),
            "Discovery_Source": pair_info["source"],
            "Price_Source": source,
            "Analysis_Time": datetime.now().strftime("%H:%M:%S"),
        }

    except Exception as e:
        print(f"Error analyzing {display_symbol}: {str(e)[:180]}")
        return None


# =============================================================================
# OUTPUT (kept very close to your original structure)
# =============================================================================
def create_forex_text_summary(df, base_filename):
    txt_filename = f"{base_filename}_FOREX_SUMMARY.txt"

    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("FOREX TRADING ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Pairs Analyzed: {len(df)}\n")
        f.write(f"Analysis Period: Last {ANALYSIS_PERIOD_DAYS} days ({ANALYSIS_INTERVAL} intervals)\n")
        f.write("=" * 80 + "\n\n")

        f.write("MARKET OVERVIEW:\n")
        f.write("-" * 80 + "\n")

        strong_buy = int((df["Signal"] == "STRONG BUY").sum())
        buy = int((df["Signal"] == "BUY").sum())
        neutral = int((df["Signal"] == "NEUTRAL/HOLD").sum())
        sell = int((df["Signal"] == "SELL").sum())
        strong_sell = int((df["Signal"] == "STRONG SELL").sum())

        f.write(f"Bullish (BUY/STRONG BUY): {strong_buy + buy} pairs\n")
        f.write(f"Neutral: {neutral} pairs\n")
        f.write(f"Bearish (SELL/STRONG SELL): {sell + strong_sell} pairs\n")
        f.write(f"Average Score: {df['Score'].mean():.1f}/100\n")
        f.write(f"Average RSI: {df['RSI'].mean():.1f}\n")
        f.write(f"Average Daily Change: {df['Daily_Change_%'].mean():+.2f}%\n\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("TOP 10 FOREX PAIRS RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n\n")

        top_pairs = df.sort_values("Score", ascending=False).head(10)

        for i, (_, pair) in enumerate(top_pairs.iterrows(), 1):
            f.write(f"{i}. {pair['Display_Symbol']}\n")
            f.write("   " + "-" * 50 + "\n")
            f.write(f"   Signal: {pair['Signal']}\n")
            f.write(f"   Score: {pair['Score']:.0f}/100\n")
            f.write(f"   Current Price: {pair['Current_Price']:.5f}\n")
            f.write(f"   RSI: {pair['RSI']:.1f} | 1H Change: {pair['Hourly_Change_%']:+.2f}%\n")
            f.write(f"   24H Change: {pair['Daily_Change_%']:+.2f}% | Weekly: {pair['Weekly_Change_%']:+.2f}%\n")
            f.write(f"   Entry: {pair['Current_Price']:.5f}\n")
            f.write(f"   Stop Loss: {pair['Stop_Loss']:.5f}\n")
            f.write(f"   Take Profit 1: {pair['Take_Profit_1']:.5f}\n")
            f.write(f"   Take Profit 2: {pair['Take_Profit_2']:.5f}\n")
            f.write(f"   Support: {pair['Support']:.5f} | Resistance: {pair['Resistance']:.5f}\n")
            f.write(f"   Volatility: {pair['Volatility_%']:.2f}% | Pair Type: {pair['Pair_Type']}\n")
            f.write(f"   Price Source: {pair.get('Price_Source', 'N/A')}\n\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("TRADING STATISTICS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total Pairs Analyzed: {len(df)}\n")
        f.write(f"Strong Buy Recommendations: {strong_buy}\n")
        f.write(f"Buy Recommendations: {buy}\n")
        f.write(f"Sell Recommendations: {sell}\n")
        f.write(f"Strong Sell Recommendations: {strong_sell}\n")

        avg_volatility = df["Volatility_%"].mean()
        max_volatility = df["Volatility_%"].max()
        min_volatility = df["Volatility_%"].min()

        f.write(f"\nVolatility Analysis:\n")
        f.write(f"  Average: {avg_volatility:.2f}%\n")
        f.write(f"  Maximum: {max_volatility:.2f}% ({df.loc[df['Volatility_%'].idxmax(), 'Display_Symbol']})\n")
        f.write(f"  Minimum: {min_volatility:.2f}% ({df.loc[df['Volatility_%'].idxmin(), 'Display_Symbol']})\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("RISK DISCLAIMER\n")
        f.write("=" * 80 + "\n")
        f.write("Forex trading involves substantial risk of loss and is not suitable for all investors.\n")
        f.write("The recommendations provided are for informational purposes only.\n")
        f.write("Past performance is not indicative of future results.\n")
        f.write("Always use proper risk management and never risk more than you can afford to lose.\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    return txt_filename


def create_forex_charts(df_top5, base_filename, cache: SqliteCache, limiter: RateLimiter):
    if df_top5.empty:
        return

    try:
        plt.style.use("seaborn-v0_8-darkgrid")
        rcParams["figure.figsize"] = [12, 8]

        charts_dir = f"{base_filename}_charts"
        os.makedirs(charts_dir, exist_ok=True)

        print(f"\nCreating charts in {charts_dir}/ directory...")

        for _, pair in df_top5.iterrows():
            display_symbol = pair["Display_Symbol"]

            try:
                hist, _ = get_ohlc_with_fallback(
                    display_symbol=display_symbol,
                    period_days=5,
                    interval=CHART_INTERVAL,
                    cache=cache,
                    limiter=limiter,
                )
            except Exception:
                hist = ohlc_from_yahoo(display_symbol, period=REPORT_PERIOD_FOR_CHARTS, interval=CHART_INTERVAL)

            if hist is None or hist.empty:
                continue

            fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [3, 1]})

            axes[0].plot(hist.index, hist["Close"], label="Price", linewidth=2, alpha=0.7)
            sma_20 = hist["Close"].rolling(window=20).mean()
            sma_50 = hist["Close"].rolling(window=50).mean()
            axes[0].plot(hist.index, sma_20, label="SMA 20", linewidth=1.5, alpha=0.7)
            axes[0].plot(hist.index, sma_50, label="SMA 50", linewidth=1.5, alpha=0.7)

            axes[0].axhline(y=pair["Support"], linestyle="--", alpha=0.5, label="Support")
            axes[0].axhline(y=pair["Resistance"], linestyle="--", alpha=0.5, label="Resistance")
            axes[0].axhline(y=pair["Current_Price"], linestyle="-", alpha=0.3, label="Current Price")

            axes[0].set_title(f"{display_symbol} - {pair['Signal']} (Score: {pair['Score']:.0f}/100)")
            axes[0].set_ylabel("Price")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # RSI
            delta = hist["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi_series = 100 - (100 / (1 + rs))

            axes[1].plot(hist.index, rsi_series, label="RSI", linewidth=2, alpha=0.7)
            axes[1].axhline(y=70, linestyle="--", alpha=0.5, label="Overbought (70)")
            axes[1].axhline(y=30, linestyle="--", alpha=0.5, label="Oversold (30)")
            axes[1].axhline(y=50, linestyle="--", alpha=0.3)

            axes[1].set_title("RSI (14 periods)")
            axes[1].set_xlabel("Date/Time")
            axes[1].set_ylabel("RSI")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylim([0, 100])

            plt.tight_layout()
            chart_filename = f"{charts_dir}/{display_symbol.replace('/', '_')}_chart.png"
            plt.savefig(chart_filename, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"  Chart saved: {chart_filename}")

        print(f"All charts saved to {charts_dir}/ directory")

    except Exception as e:
        print(f"Error creating charts: {e}")


def create_forex_print_friendly_output(df, base_filename, cache: SqliteCache, limiter: RateLimiter):
    if df.empty:
        return None

    print_df = df.copy()

    print_df["Formatted_Price"] = print_df["Current_Price"].apply(lambda x: f"{x:.5f}")
    print_df["Formatted_Score"] = print_df["Score"].apply(lambda x: f"{int(x)}/100")
    print_df["Formatted_Hourly"] = print_df["Hourly_Change_%"].apply(lambda x: f"{x:+.2f}%")
    print_df["Formatted_Daily"] = print_df["Daily_Change_%"].apply(lambda x: f"{x:+.2f}%")
    print_df["Formatted_RSI"] = print_df["RSI"].apply(lambda x: f"{x:.1f}")
    print_df["Formatted_Stop_Loss"] = print_df["Stop_Loss"].apply(lambda x: f"{x:.5f}")
    print_df["Formatted_TP1"] = print_df["Take_Profit_1"].apply(lambda x: f"{x:.5f}")
    print_df["Formatted_TP2"] = print_df["Take_Profit_2"].apply(lambda x: f"{x:.5f}")

    def color_signal(signal):
        if "STRONG BUY" in signal:
            return "🟢 STRONG BUY"
        elif signal == "BUY":
            return "🟡 BUY"
        elif "STRONG SELL" in signal:
            return "🔴 STRONG SELL"
        elif signal == "SELL":
            return "🟠 SELL"
        return "⚪ NEUTRAL"

    print_df["Colored_Signal"] = print_df["Signal"].apply(color_signal)

    print_columns = {
        "Display_Symbol": "Currency Pair",
        "Colored_Signal": "Signal",
        "Formatted_Price": "Price",
        "Formatted_Score": "Score",
        "Formatted_RSI": "RSI",
        "Formatted_Hourly": "1H Change",
        "Formatted_Daily": "24H Change",
        "Pair_Type": "Type",
        "Formatted_Stop_Loss": "Stop Loss",
        "Formatted_TP1": "TP1 (0.5%)",
        "Formatted_TP2": "TP2 (1.0%)",
        "Support": "Support",
        "Resistance": "Resistance",
        "Volatility_%": "Volatility",
        "Price_Source": "Price Source",
        "Analysis_Time": "Analysis Time",
    }

    available_columns = [col for col in print_columns.keys() if col in print_df.columns]
    print_df = print_df[available_columns].rename(columns=print_columns)

    print_filename = f"{base_filename}_FOREX_PRINT_READY.csv"
    print_df.to_csv(print_filename, index=False)

    create_forex_text_summary(df, base_filename)
    create_forex_charts(df.head(5), base_filename, cache, limiter)

    return print_filename


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 80)
    print("FOREX TRADING ANALYSIS - CACHED + FALLBACK (HARDCODED KEYS)")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis version reduces API calls by:")
    print("• Using ONE OHLC pull per pair (current price = last close)")
    print("• Caching discovery + candles (SQLite)")
    print("• Fallback sources if limits are hit")
    print("=" * 80)

    cache = SqliteCache(CACHE_DB_PATH)
    limiter = RateLimiter(MIN_SECONDS_BETWEEN_CALLS)

    # STEP 1: discovery (cached)
    print("\nSTEP 1: DISCOVERING FOREX PAIRS")
    print("-" * 50)
    all_pairs = discover_forex_pairs_cached(cache, limiter)
    print(f"   Total discovered (curated): {len(all_pairs)} pairs")

    # STEP 2: analyze
    print("\nSTEP 2: ANALYZING FOREX PAIRS")
    print("-" * 50)

    pairs_to_analyze = all_pairs[:DISCOVERY_MAX_PAIRS]
    analyzed_pairs = []

    for i, pair in enumerate(pairs_to_analyze, 1):
        display_symbol = pair["display_symbol"]
        print(f"{i:3d}/{len(pairs_to_analyze)}: {display_symbol:10}", end="")
        analysis = analyze_forex_pair(pair, cache, limiter)

        if analysis:
            analyzed_pairs.append(analysis)
            sig = analysis["Signal"]
            if "STRONG BUY" in sig:
                color = "\033[92m"
            elif sig == "BUY":
                color = "\033[93m"
            elif "SELL" in sig:
                color = "\033[91m"
            else:
                color = "\033[90m"
            print(f" - {color}{analysis['Current_Price']:.5f} [{analysis['Signal']}]\033[0m")
        else:
            print(" - No data or analysis failed")

        if i % 5 == 0:
            time.sleep(0.8)

    print(f"\nPAIRS SUCCESSFULLY ANALYZED: {len(analyzed_pairs)}")
    if not analyzed_pairs:
        print("\nNo forex pairs could be analyzed.")
        cache.close()
        return

    # STEP 3: output
    print("\nSTEP 3: CREATING OUTPUT FILES")
    print("-" * 50)

    df = pd.DataFrame(analyzed_pairs).sort_values("Score", ascending=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"forex_analysis_{timestamp}"

    raw_filename = f"{base_filename}_RAW.csv"
    df.to_csv(raw_filename, index=False)
    print(f"Raw data saved: {raw_filename}")

    print_friendly_file = create_forex_print_friendly_output(df, base_filename, cache, limiter)
    if print_friendly_file:
        print(f"Print-friendly CSV: {print_friendly_file}")

    print("\n" + "=" * 80)
    print("FOREX ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"  • Total pairs discovered: {len(all_pairs)}")
    print(f"  • Successfully analyzed: {len(df)}")
    print(f"  • Average score: {df['Score'].mean():.1f}/100")
    print(f"  • Average RSI: {df['RSI'].mean():.1f}")

    print("\n" + "=" * 80)
    print("TOP 10 FOREX TRADING SUGGESTIONS")
    print("=" * 80)
    top_10 = df.head(10)
    for i, (_, pair) in enumerate(top_10.iterrows(), 1):
        print(f"\n{i:2d}. {pair['Display_Symbol']:10} ({pair['Pair_Type']})")
        print(f"    {pair['Signal']} | Score: {pair['Score']:.0f}/100 | Price: {pair['Current_Price']:.5f}")
        print(f"    RSI: {pair['RSI']:.1f} | 1H: {pair['Hourly_Change_%']:+.2f}% | 24H: {pair['Daily_Change_%']:+.2f}%")
        print(f"    Support: {pair['Support']:.5f} | Resistance: {pair['Resistance']:.5f}")
        print(f"    Price Source: {pair['Price_Source']}")

    print("\n" + "=" * 80)
    print("FILES CREATED:")
    print("=" * 80)
    print(f"1. {raw_filename} - Raw analysis data")
    if print_friendly_file:
        print(f"2. {print_friendly_file} - Print-ready CSV (open in Excel)")
        print(f"3. {base_filename}_FOREX_SUMMARY.txt - Detailed text summary")
        print(f"4. {base_filename}_charts/ - Technical analysis charts")

    cache.close()


if __name__ == "__main__":
    try:
        print("API Keys Status (hard-coded):")
        for api, key in API_KEYS.items():
            print(f"  {api}: {'Configured' if key and len(key) > 5 else 'Missing/Invalid'}")
        main()
    except KeyboardInterrupt:
        print("\n\nScript stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)