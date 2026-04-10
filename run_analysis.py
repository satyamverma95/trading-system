# =================================================================
# run_analysis.py
# Reads watchlist.csv and runs full analysis for each symbol
# Usage: python run_analysis.py
#        python run_analysis.py --watchlist data/input/my_list.csv
# =================================================================

import argparse
import pandas as pd
import time
from utils.helpers import load_config, save_to_parquet
from providers.yfinance_provider import YFinanceProvider
from analysis.indicators import Indicators
from utils.logger import get_logger

logger = get_logger(__name__)

WATCHLIST_FILE = "data/input/watchlist.csv"


# ----------------------------------------------------------
# SAFE FORMATTERS
# ----------------------------------------------------------

def safe(val, fmt=".2f"):
    """Safely format a value that might be NaN or None."""
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "N/A"
        return f"{val:{fmt}}"
    except:
        return "N/A"


def safe_compare(val, threshold, above="✅ Above", below="❌ Below"):
    """Safely compare two values that might be NaN."""
    try:
        if val is None or threshold is None:
            return "N/A"
        if pd.isna(val) or pd.isna(threshold):
            return "N/A"
        return above if val > threshold else below
    except:
        return "N/A"


# ----------------------------------------------------------
# SUMMARY PRINTER
# ----------------------------------------------------------

def print_summary(df: pd.DataFrame, symbol: str, period: str, interval: str):
    latest = df.iloc[-1]
    prev   = df.iloc[-2]

    print("\n" + "=" * 60)
    print(f"  {symbol}  |  {period}  |  {interval}")
    print("=" * 60)
    print(f"  Data     : {len(df)} candles  |  {df.index[0].date()} → {df.index[-1].date()}")

    print(f"\n  PRICE")
    print(f"    Close    : ₹{safe(latest['Close'])}  (prev: ₹{safe(prev['Close'])})")
    try:
        change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
        print(f"    Change   : {'+' if change >= 0 else ''}{change:.2f}%")
    except:
        print(f"    Change   : N/A")
    print(f"    High/Low : ₹{safe(latest['High'])} / ₹{safe(latest['Low'])}")
    try:
        print(f"    Volume   : {int(latest['Volume']):,}")
    except:
        print(f"    Volume   : N/A")

    print(f"\n  TREND")
    print(f"    SMA 50   : ₹{safe(latest.get('SMA_50'))}  {safe_compare(latest['Close'], latest.get('SMA_50'))}")
    print(f"    SMA 200  : ₹{safe(latest.get('SMA_200'))}  {safe_compare(latest['Close'], latest.get('SMA_200'))}")
    print(f"    EMA 21   : ₹{safe(latest.get('EMA_21'))}  {safe_compare(latest['Close'], latest.get('EMA_21'))}")

    print(f"\n  MOMENTUM")
    rsi = latest.get('RSI')
    try:
        if rsi is not None and not pd.isna(rsi):
            rsi_label = "🟢 Oversold" if rsi < 30 else ("🔴 Overbought" if rsi > 70 else "⚪ Neutral")
            print(f"    RSI      : {rsi:.1f}  {rsi_label}")
        else:
            print(f"    RSI      : N/A")
    except:
        print(f"    RSI      : N/A")

    macd     = latest.get('MACD')
    macd_sig = latest.get('MACD_Signal')
    try:
        macd_label = "🟢 Bullish" if macd > macd_sig else "🔴 Bearish"
        print(f"    MACD     : {safe(macd)}  |  Signal: {safe(macd_sig)}  {macd_label}")
    except:
        print(f"    MACD     : N/A")

    print(f"\n  VOLATILITY")
    print(f"    BB Upper : ₹{safe(latest.get('BB_Upper'))}")
    print(f"    BB Lower : ₹{safe(latest.get('BB_Lower'))}")
    print(f"    ATR      : ₹{safe(latest.get('ATR'))}")

    print(f"\n  VOLUME")
    vol_ratio = latest.get('Vol_Ratio')
    try:
        vol_label = "🔥 High" if vol_ratio > 1.5 else "Normal"
        print(f"    Vol Ratio: {vol_ratio:.2f}x  {vol_label}")
    except:
        print(f"    Vol Ratio: N/A")

    print(f"\n  SIGNALS")
    signals = {
        "RSI Oversold"   : latest.get('RSI_Oversold'),
        "RSI Overbought" : latest.get('RSI_Overbought'),
        "MACD Crossover" : latest.get('MACD_Crossover'),
        "MACD Crossunder": latest.get('MACD_Crossunder'),
        "Golden Cross"   : latest.get('Golden_Cross'),
        "Death Cross"    : latest.get('Death_Cross'),
    }
    active = [k for k, v in signals.items() if v is True or v == True]
    if active:
        for s in active:
            print(f"    ✅ {s}")
    else:
        print(f"    No active signals")


# ----------------------------------------------------------
# PROCESS SINGLE SYMBOL
# ----------------------------------------------------------

def process_symbol(row: pd.Series, config: dict,
                   provider: YFinanceProvider,
                   ind: Indicators) -> bool:
    symbol   = row["SYMBOL"].strip()
    period   = str(row["PERIOD"]).strip()
    interval = str(row["INTERVAL"]).strip()
    save     = str(row["SAVE"]).strip().lower() == "true"

    try:
        logger.info(f"Processing {symbol} | {period} | {interval}")
        df = provider.get_historical_data(symbol, period=period, interval=interval)
        df = ind.add_all(df)
        print_summary(df, symbol, period, interval)

        if save:
            path = save_to_parquet(df, symbol, config["paths"]["processed_data"], interval)
            logger.info(f"Saved → {path}")

        return True

    except Exception as e:
        logger.error(f"Failed for {symbol}: {e}")
        print(f"\n  ❌ SKIPPED {symbol} — {e}")
        return False


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

def run(watchlist_file: str):

    # Load watchlist CSV
    try:
        watchlist = pd.read_csv(watchlist_file)
    except FileNotFoundError:
        logger.error(f"Watchlist file not found: {watchlist_file}")
        print(f"❌ Could not find {watchlist_file}")
        print(f"   Create it with columns: SYMBOL, PERIOD, INTERVAL, SAVE")
        return

    # Validate columns
    required_cols = {"SYMBOL", "PERIOD", "INTERVAL", "SAVE"}
    if not required_cols.issubset(watchlist.columns):
        logger.error(f"CSV missing required columns. Need: {required_cols}")
        print(f"❌ CSV must have columns: {required_cols}")
        return

    logger.info(f"Loaded watchlist: {len(watchlist)} symbols from {watchlist_file}")

    # Init config, provider, indicators
    config   = load_config()
    provider = YFinanceProvider(config)
    ind      = Indicators(config)

    # Track results
    success, failed = [], []

    print(f"\n🚀 Running analysis for {len(watchlist)} symbols...\n")

    for _, row in watchlist.iterrows():
        ok = process_symbol(row, config, provider, ind)
        if ok:
            success.append(row["SYMBOL"])
        else:
            failed.append(row["SYMBOL"])
        time.sleep(1)  # polite delay to avoid rate limiting

    # Final run summary
    print("\n" + "=" * 60)
    print(f"  RUN COMPLETE")
    print(f"  ✅ Success : {len(success)} symbols")
    if success:
        print(f"  Processed  : {', '.join(success)}")
    print(f"  ❌ Failed  : {len(failed)} symbols")
    if failed:
        print(f"  Failed list: {', '.join(failed)}")
    print("=" * 60 + "\n")


# ----------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading System — Batch Analysis")
    parser.add_argument(
        "--watchlist",
        type=str,
        default=WATCHLIST_FILE,
        help="Path to watchlist CSV (default: data/input/watchlist.csv)"
    )
    args = parser.parse_args()
    run(args.watchlist)