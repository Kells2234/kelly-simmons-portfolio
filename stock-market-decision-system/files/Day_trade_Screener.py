# Day Trading code (REVISED WITH RATE-LIMIT + CACHING + FALLBACK)
# api_stock_discovery_print_friendly_rate_safe.py

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import requests
import warnings
import sys
import time
import os
import pickle
import random
import re

warnings.filterwarnings("ignore")

# ============================================================================
# API CONFIGURATION
# ============================================================================
API_KEYS = {
    "alpha_vantage": "GQK46KXL3PH02QDG",
    "twelve_data": "51fba4e1c69d4f9c9ffc4434f7951e56",
    "finnhub": "6937ffd09b907566546327",
}

# ============================================================================
# CACHING (PERSISTENT)
# ============================================================================
class DataCache:
    """
    Simple persistent file cache with TTL. Stores pickled Python objects.
    """
    def __init__(self, cache_dir=".api_cache", expiry_hours=12):
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


# ============================================================================
# RATE LIMITING + CIRCUIT BREAKER
# ============================================================================
class ProviderLimiter:
    """
    Per-provider rate limiter + cooldown ("circuit breaker") when we see 429/quota.
    """
    def __init__(self, name: str, max_calls: int, period_sec: int, cooldown_sec: int):
        self.name = name
        self.max_calls = max_calls
        self.period_sec = period_sec
        self.cooldown_sec = cooldown_sec
        self.calls = []
        self.disabled_until = 0.0

    def can_call(self) -> bool:
        return time.time() >= self.disabled_until

    def disable(self):
        self.disabled_until = time.time() + self.cooldown_sec

    def wait_if_needed(self):
        now = time.time()
        self.calls = [t for t in self.calls if now - t < self.period_sec]
        if len(self.calls) >= self.max_calls:
            sleep_time = self.period_sec - (now - self.calls[0]) + 0.5
            if sleep_time > 0:
                print(f"[{self.name}] Throttling: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
        # jitter helps reduce bursts
        time.sleep(random.uniform(0.15, 0.45))
        self.calls.append(time.time())


def _looks_rate_limited(provider: str, status_code: int, data) -> bool:
    if status_code == 429:
        return True
    if not data:
        return False
    txt = str(data).lower()
    if "rate limit" in txt or "too many requests" in txt or "quota" in txt or "limit" in txt:
        return True
    # Alpha Vantage sometimes returns "Note" when throttled
    if provider == "alpha_vantage" and isinstance(data, dict):
        if any(k.lower() in ["note", "information"] for k in data.keys()):
            return True
    # TwelveData can return status:error + message about limit
    if provider == "twelve_data" and isinstance(data, dict):
        if data.get("status") == "error" and "limit" in (data.get("message", "") or "").lower():
            return True
    return False


def safe_get_json(url: str, params: dict, provider: str, limiter: ProviderLimiter, timeout: int = 12):
    """
    Requests wrapper:
    - respects limiter
    - retries with exponential backoff
    - disables provider on 429/quota detection
    """
    if not limiter.can_call():
        return None

    for attempt in range(5):
        limiter.wait_if_needed()
        try:
            r = requests.get(url, params=params, timeout=timeout)
            try:
                data = r.json()
            except Exception:
                data = None

            if _looks_rate_limited(provider, r.status_code, data):
                print(f"[{provider}] Rate limited / quota hit. Cooling down...")
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


# ============================================================================
# GLOBALS (cache + provider policies)
# ============================================================================
cache = DataCache(cache_dir=".api_cache", expiry_hours=12)

limiters = {
    # Finnhub: free tier often 60/min, but be conservative
    "finnhub": ProviderLimiter("finnhub", max_calls=45, period_sec=60, cooldown_sec=6 * 3600),
    # Twelve Data: typical free tier is low per minute; keep small
    "twelve_data": ProviderLimiter("twelve_data", max_calls=6, period_sec=60, cooldown_sec=6 * 3600),
    # Alpha Vantage: 5/min free; set exactly
    "alpha_vantage": ProviderLimiter("alpha_vantage", max_calls=5, period_sec=60, cooldown_sec=6 * 3600),
    # Yahoo isn't a paid API but still can throttle; keep gentle
    "yahoo": ProviderLimiter("yahoo", max_calls=60, period_sec=60, cooldown_sec=600),
}

# ============================================================================
# REAL API STOCK DISCOVERY (CACHED)
# ============================================================================
def discover_stocks_from_finnhub():
    """Pull US stock list from Finnhub (cached)."""
    print("\n[API] Discovering stocks from Finnhub...")

    cache_key = f"finnhub_symbols_US_{datetime.now():%Y%m%d}"  # cache per day
    cached = cache.get(cache_key)
    if cached is not None:
        print(f"   Using cached Finnhub symbols: {len(cached)}")
        return cached

    discovered_stocks = []
    try:
        url = "https://finnhub.io/api/v1/stock/symbol"
        params = {"exchange": "US", "token": API_KEYS["finnhub"]}

        data = safe_get_json(url, params, "finnhub", limiters["finnhub"], timeout=20)
        if not data:
            print("   Finnhub returned no data (rate-limited or error).")
            return discovered_stocks

        print(f"   Retrieved {len(data)} symbols from Finnhub")

        for stock in data:
            symbol = stock.get("symbol", "")
            description = stock.get("description", "")
            mic = stock.get("mic", "")
            type_ = stock.get("type", "")

            if (
                type_ == "Common Stock"
                and mic in ["XNAS", "XNYS"]
                and symbol
                and len(symbol) <= 5
                and not any(x in symbol for x in [".", "-", "^"])
                and not description.startswith("TEST")
            ):
                discovered_stocks.append(
                    {
                        "symbol": symbol,
                        "name": description,
                        "exchange": "NASDAQ" if mic == "XNAS" else "NYSE",
                        "source": "Finnhub",
                    }
                )

        print(f"   Filtered to {len(discovered_stocks)} valid common stocks")
        cache.set(cache_key, discovered_stocks)
        return discovered_stocks

    except Exception as e:
        print(f"   Error: {str(e)[:80]}")
        return discovered_stocks


def discover_stocks_from_twelve_data():
    """Pull US stocks list from Twelve Data (cached)."""
    print("\n[API] Discovering stocks from Twelve Data...")

    cache_key = f"twelvedata_stocks_US_{datetime.now():%Y%m%d}"  # cache per day
    cached = cache.get(cache_key)
    if cached is not None:
        print(f"   Using cached Twelve Data stocks: {len(cached)}")
        return cached

    discovered_stocks = []
    try:
        url = "https://api.twelvedata.com/stocks"
        params = {"country": "United States", "apikey": API_KEYS["twelve_data"]}

        data = safe_get_json(url, params, "twelve_data", limiters["twelve_data"], timeout=20)
        if not data or "data" not in data:
            print("   Twelve Data returned no data (rate-limited or error).")
            return discovered_stocks

        for stock in data["data"]:
            symbol = stock.get("symbol", "")
            name = stock.get("name", "")
            exchange = stock.get("exchange", "")

            if (
                symbol
                and exchange in ["NASDAQ", "NYSE", "NYSE ARCA"]
                and len(symbol) <= 5
                and not any(x in symbol for x in [".", "-", "^"])
            ):
                discovered_stocks.append(
                    {"symbol": symbol, "name": name, "exchange": exchange, "source": "Twelve Data"}
                )

        print(f"   Retrieved {len(discovered_stocks)} stocks from Twelve Data")
        cache.set(cache_key, discovered_stocks)
        return discovered_stocks

    except Exception as e:
        print(f"   Error: {str(e)[:80]}")
        return discovered_stocks


def get_current_market_actives():
    """Hard-coded list (no API)."""
    print("\n[Market] Getting currently active stocks...")

    active_stocks = []
    common_active_symbols = [
        "SIRI", "F", "PLUG", "ACB", "SNDL", "FCEL", "NIO",
        "T", "VZ", "INTC", "BAC", "C", "WFC", "PFE",
        "MRO", "XOM", "CVX", "SLB",
        "AAL", "DAL", "LUV", "UAL",
        "M", "KSS", "GPS",
        "SPWR", "RUN", "FSLR",
        "SOFI", "UPST",
        "TLRY", "CGC",
    ]

    for symbol in common_active_symbols:
        active_stocks.append(
            {
                "symbol": symbol,
                "name": f"Active Stock {symbol}",
                "exchange": "NASDAQ" if symbol in ["SIRI", "PLUG", "ACB", "SNDL", "FCEL", "NIO"] else "NYSE",
                "source": "Market Active",
            }
        )

    print(f"   Added {len(active_stocks)} commonly active stocks")
    return active_stocks


# ============================================================================
# PRICE CHECK (CACHED + FALLBACK ORDER: Yahoo -> Finnhub -> TwelveData)
# ============================================================================
def check_stock_price_real_time(symbol: str):
    """
    Avoid burning API calls:
    - cache price for 15 minutes per symbol
    - use Yahoo first (no API quota)
    - then Finnhub
    - then Twelve Data
    """
    cache_key = f"rt_price_{symbol}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached["price"], cached["source"]

    # 1) Yahoo (fast, no API quota)
    try:
        limiters["yahoo"].wait_if_needed()
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1d", interval="1m")
        if hist is not None and not hist.empty:
            price = float(hist["Close"].iloc[-1])
            if price > 0:
                cache_short = DataCache(cache_dir=".api_cache", expiry_hours=0.25)  # 15 min
                cache_short.set(cache_key, {"price": price, "source": "Yahoo Finance"})
                return price, "Yahoo Finance"
    except Exception:
        pass

    # 2) Finnhub (API)
    try:
        url = "https://finnhub.io/api/v1/quote"
        params = {"symbol": symbol, "token": API_KEYS["finnhub"]}
        data = safe_get_json(url, params, "finnhub", limiters["finnhub"], timeout=8)
        if data and data.get("c"):
            price = float(data["c"])
            if price > 0:
                cache_short = DataCache(cache_dir=".api_cache", expiry_hours=0.25)
                cache_short.set(cache_key, {"price": price, "source": "Finnhub"})
                return price, "Finnhub"
    except Exception:
        pass

    # 3) Twelve Data (API)
    try:
        url = "https://api.twelvedata.com/price"
        params = {"symbol": symbol, "apikey": API_KEYS["twelve_data"]}
        data = safe_get_json(url, params, "twelve_data", limiters["twelve_data"], timeout=8)
        if data and data.get("price"):
            price = float(data["price"])
            if price > 0:
                cache_short = DataCache(cache_dir=".api_cache", expiry_hours=0.25)
                cache_short.set(cache_key, {"price": price, "source": "Twelve Data"})
                return price, "Twelve Data"
    except Exception:
        pass

    return None, None


# ============================================================================
# ANALYSIS FUNCTIONS (uses Yahoo for history, cached by yfinance + gentle limiter)
# ============================================================================
def analyze_discovered_stock(stock_info):
    symbol = stock_info["symbol"]

    try:
        print(f"  Analyzing {symbol}...", end="\r")

        price, price_source = check_stock_price_real_time(symbol)
        if price is None:
            return None

        if not (1.0 <= price <= 15.0):
            return None

        # Get historical data (Yahoo)
        try:
            limiters["yahoo"].wait_if_needed()
            stock = yf.Ticker(symbol)
            hist = stock.history(period="10d")
        except Exception:
            return None

        if hist is None or hist.empty or len(hist) < 5:
            return None

        prices = hist["Close"].values

        def calculate_rsi(prices_arr, period=14):
            if len(prices_arr) < period:
                return 50
            deltas = np.diff(prices_arr)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
            if avg_loss == 0:
                return 100
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))

        def calculate_sma(prices_arr, period):
            if len(prices_arr) < period:
                return prices_arr[-1] if len(prices_arr) > 0 else 0
            return float(np.mean(prices_arr[-period:]))

        rsi = float(calculate_rsi(prices, 14))
        sma_20 = float(calculate_sma(prices, 20))

        daily_change = float(((prices[-1] - prices[-2]) / prices[-2]) * 100) if len(prices) >= 2 else 0.0
        weekly_change = float(((prices[-1] - prices[-5]) / prices[-5]) * 100) if len(prices) >= 5 else 0.0

        volumes = hist["Volume"].values
        current_volume = float(volumes[-1]) if len(volumes) > 0 else 0.0
        avg_volume = float(np.mean(volumes[-5:])) if len(volumes) >= 5 else current_volume
        volume_ratio = float(current_volume / avg_volume) if avg_volume > 0 else 1.0

        score = 50
        if rsi < 30:
            score += 20
        elif rsi > 70:
            score -= 20

        if volume_ratio > 2.0:
            score += 15
        elif volume_ratio > 1.5:
            score += 5

        if daily_change > 2:
            score += 10
        elif daily_change < -2:
            score -= 10

        score = max(0, min(100, score))

        if score >= 70:
            action = "STRONG BUY"
        elif score >= 60:
            action = "BUY"
        elif score <= 30:
            action = "STRONG SELL"
        elif score <= 40:
            action = "SELL"
        else:
            action = "HOLD"

        # Company info: expensive sometimes; cache it for a week
        info_cache = DataCache(cache_dir=".api_cache", expiry_hours=24 * 7)
        info_key = f"yf_info_{symbol}"
        info = info_cache.get(info_key)

        if info is None:
            try:
                info = yf.Ticker(symbol).info
                info_cache.set(info_key, info)
            except Exception:
                info = {}

        company_name = (info or {}).get("longName", stock_info.get("name", symbol))
        sector = (info or {}).get("sector", "Unknown")
        market_cap = (info or {}).get("marketCap", 0)

        return {
            "Symbol": symbol,
            "Company": str(company_name)[:40],
            "Sector": sector,
            "Price": float(price),
            "Action": action,
            "Score": float(score),
            "RSI": float(rsi),
            "Daily_Change_%": float(daily_change),
            "Weekly_Change_%": float(weekly_change),
            "Volume_Ratio": float(volume_ratio),
            "SMA_20": float(sma_20),
            "Market_Cap": int(market_cap) if market_cap else 0,
            "Exchange": stock_info.get("exchange", "N/A"),
            "Discovery_Source": stock_info.get("source", "N/A"),
            "Price_Source": price_source,
            "Analysis_Time": datetime.now().strftime("%H:%M:%S"),
        }

    except Exception:
        return None


# ============================================================================
# PRINT-FRIENDLY OUTPUT (unchanged)
# ============================================================================
def create_print_friendly_output(df, base_filename):
    if df.empty:
        return None

    print_df = df.copy()

    print_df["Formatted_Price"] = print_df["Price"].apply(lambda x: f"${x:,.2f}")
    print_df["Formatted_Score"] = print_df["Score"].apply(lambda x: f"{int(x)}/100")
    print_df["Formatted_Change"] = print_df["Daily_Change_%"].apply(lambda x: f"{x:+.1f}%")
    print_df["Formatted_RSI"] = print_df["RSI"].apply(lambda x: f"{x:.1f}")
    print_df["Formatted_Volume"] = print_df["Volume_Ratio"].apply(lambda x: f"{x:.1f}x")

    if "Market_Cap" in print_df.columns:
        print_df["Formatted_Market_Cap"] = print_df["Market_Cap"].apply(
            lambda x: f"${x:,.0f}" if pd.notnull(x) and x > 1_000_000 else "N/A"
        )

    print_columns = {
        "Action": "Recommendation",
        "Symbol": "Ticker",
        "Company": "Company",
        "Formatted_Price": "Price",
        "Formatted_Score": "Score",
        "Formatted_Change": "Daily Change",
        "Formatted_RSI": "RSI",
        "Formatted_Volume": "Volume",
        "Sector": "Sector",
        "Exchange": "Exchange",
        "Discovery_Source": "Discovered Via",
        "Price_Source": "Price Source",
    }

    print_df = print_df[[c for c in print_columns.keys() if c in print_df.columns]]
    print_df = print_df.rename(columns=print_columns)

    print_filename = f"{base_filename}_PRINT_READY.csv"
    print_df.to_csv(print_filename, index=False)

    create_text_summary(df, base_filename)

    return print_filename


def create_text_summary(df, base_filename):
    txt_filename = f"{base_filename}_SUMMARY.txt"
    with open(txt_filename, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("API-DISCOVERED STOCKS ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Price Range: $1.00 - $15.00\n")
        f.write(f"Total Stocks Discovered: {len(df)}\n")
        f.write("=" * 70 + "\n\n")

        for action_type in ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]:
            action_df = df[df["Action"] == action_type]
            if not action_df.empty:
                f.write(f"\n{action_type} ({len(action_df)} stocks):\n")
                f.write("-" * 60 + "\n")
                for _, row in action_df.iterrows():
                    f.write(
                        f"{row['Symbol']:6} | Price: ${row['Price']:6.2f} | "
                        f"Score: {row['Score']:3.0f}/100 | "
                        f"Change: {row['Daily_Change_%']:+.1f}% | "
                        f"RSI: {row['RSI']:5.1f}\n"
                    )
                    f.write(f"      {row['Company'][:50]}\n")
                    f.write(f"      Source: {row.get('Discovery_Source', 'N/A')} | Exchange: {row.get('Exchange', 'N/A')}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("REPORT STATISTICS\n")
        f.write("=" * 70 + "\n")

        if len(df) > 0:
            f.write(f"Average Price: ${df['Price'].mean():.2f}\n")
            f.write(f"Price Range: ${df['Price'].min():.2f} - ${df['Price'].max():.2f}\n")
            f.write(f"Average Score: {df['Score'].mean():.1f}/100\n")
            f.write(f"Average RSI: {df['RSI'].mean():.1f}\n")

            f.write("\nDiscovery Sources:\n")
            sources = df["Discovery_Source"].value_counts()
            for source, count in sources.items():
                f.write(f"  {source}: {count} stocks\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")

    return txt_filename


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 80)
    print("REAL API STOCK DISCOVERY (RATE-SAFE) - $1.00 to $15.00")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis script pulls from APIs with protections:")
    print("• Caching discovery lists (daily)")
    print("• Caching real-time prices (15 min)")
    print("• Per-provider rate limiting + cooldown on 429/quota")
    print("• Yahoo-first price checks to avoid API burn")
    print("=" * 80)

    print("\nSTEP 1: DISCOVERING STOCKS FROM APIS")
    print("-" * 50)

    all_discovered_stocks = []

    finnhub_stocks = discover_stocks_from_finnhub()
    all_discovered_stocks.extend(finnhub_stocks)
    print(f"   Finnhub: {len(finnhub_stocks)} stocks")

    twelvedata_stocks = discover_stocks_from_twelve_data()
    all_discovered_stocks.extend(twelvedata_stocks)
    print(f"   Twelve Data: {len(twelvedata_stocks)} stocks")

    market_stocks = get_current_market_actives()
    all_discovered_stocks.extend(market_stocks)
    print(f"   Market Active: {len(market_stocks)} stocks")

    unique_stocks = []
    seen = set()
    for s in all_discovered_stocks:
        sym = s["symbol"]
        if sym not in seen:
            seen.add(sym)
            unique_stocks.append(s)

    print(f"\nTOTAL UNIQUE STOCKS DISCOVERED: {len(unique_stocks)}")

    if not unique_stocks:
        print("\nERROR: No stocks discovered (rate limits / invalid keys / network).")
        return

    print("\nSTEP 2: CHECKING PRICES & FILTERING TO $1-$15")
    print("-" * 50)

    stocks_in_range = []

    # IMPORTANT: still cap checks to avoid yfinance throttling on huge lists
    stocks_to_check = unique_stocks[:50]

    for i, stock in enumerate(stocks_to_check, 1):
        symbol = stock["symbol"]
        print(f"{i:3d}/{len(stocks_to_check)}: {symbol:8}", end="")

        analysis = analyze_discovered_stock(stock)

        if analysis:
            stocks_in_range.append(analysis)
            print(f" - ${analysis['Price']:6.2f} [{analysis['Action']}] ({analysis['Price_Source']})")
        else:
            print(" - Not in range or no data")

        # Gentle pacing (plus internal limiter/jitter)
        if i % 12 == 0:
            time.sleep(0.8)

    print(f"\nSTOCKS IN $1-$15 RANGE: {len(stocks_in_range)}")

    if len(stocks_in_range) == 0:
        print("\nNo stocks found in $1-$15 range.")
        return

    print("\nSTEP 3: CREATING OUTPUT FILES")
    print("-" * 50)

    df = pd.DataFrame(stocks_in_range).sort_values("Score", ascending=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"api_discovered_stocks_{timestamp}"

    raw_filename = f"{base_filename}_RAW.csv"
    df.to_csv(raw_filename, index=False)
    print(f"Raw data saved: {raw_filename}")

    print_friendly_file = create_print_friendly_output(df, base_filename)
    if print_friendly_file:
        print(f"Print-friendly CSV: {print_friendly_file}")

    print("\n" + "=" * 80)
    print("DISCOVERY SUMMARY")
    print("=" * 80)
    print(f"  • Total discovered: {len(unique_stocks)}")
    print(f"  • In $1-$15 range: {len(stocks_in_range)}")
    print(f"  • Average price: ${df['Price'].mean():.2f}")
    print(f"  • Average score: {df['Score'].mean():.1f}/100")

    print("\nTop 5 Recommendations:")
    for i, (_, row) in enumerate(df.head(5).iterrows(), 1):
        print(f"  {i}. {row['Symbol']}: {row['Action']} at ${row['Price']:.2f} (Score: {row['Score']:.0f})")

    print("\n" + "=" * 80)
    print("FILES CREATED:")
    print("=" * 80)
    print(f"1. {raw_filename} - Raw discovered data")
    if print_friendly_file:
        print(f"2. {print_friendly_file} - Print-ready CSV")
        print(f"3. {base_filename}_SUMMARY.txt - Text summary")

    print("\nCache folders used:")
    print("• .api_cache/ (discovery lists, prices, company info)")
    print("=" * 80)


if __name__ == "__main__":
    try:
        print("API Keys Status:")
        for api, key in API_KEYS.items():
            if key and len(key) > 10:
                print(f"  {api}: Configured")
            else:
                print(f"  {api}: Missing or invalid")
        main()
    except KeyboardInterrupt:
        print("\n\nScript stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)