# =================================================================
# run_screener.py
# Run the composite screener across all symbols
# Usage: python run_screener.py
#        python run_screener.py --interval 1wk
# =================================================================

import argparse
import pandas as pd
from analysis.screener import Screener
from utils.helpers import load_config
from utils.logger import get_logger

logger = get_logger(__name__)

SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS",
    "ICICIBANK.NS", "HINDUNILVR.NS", "SBIN.NS", "BAJFINANCE.NS",
    "BHARTIARTL.NS", "KOTAKBANK.NS", "MARUTI.NS", "WIPRO.NS",
    "AXISBANK.NS"
]


def run(interval: str):
    config   = load_config()
    screener = Screener(config)

    print(f"\n🔍 Running screener | Interval: {interval}\n")
    df = screener.screen(SYMBOLS, interval)

    if df.empty:
        print("❌ No data found. Run python run_analysis.py first.")
        return

    # Display full table
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:.1f}".format)

    print("=" * 100)
    print(f"  SCREENER RESULTS — {interval.upper()}  |  Ranked by Composite Score")
    print("=" * 100)
    print(df[["Symbol", "Close", "Change%", "RSI",
              "RSI_Score", "MACD_Score", "Trend_Score",
              "Volume_Score", "Composite", "Signal"]].to_string())
    print("=" * 100)

    # Top picks
    print(f"\n  🏆 TOP 3 BULLISH:")
    for _, row in df.head(3).iterrows():
        print(f"     {row.name}. {row['Symbol']:20s} Score: {row['Composite']:.1f}  {row['Signal']}")

    print(f"\n  ⚠️  BOTTOM 3 (Weakest):")
    for _, row in df.tail(3).iterrows():
        print(f"     {row.name}. {row['Symbol']:20s} Score: {row['Composite']:.1f}  {row['Signal']}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading System — Screener")
    parser.add_argument("--interval", type=str, default="1d",
                        help="Interval to screen: 1d or 1wk (default: 1d)")
    args = parser.parse_args()
    run(args.interval)