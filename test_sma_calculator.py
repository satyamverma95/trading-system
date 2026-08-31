# =================================================================
# test_sma_calculator.py
# Test script: Fetch real Zerodha data and test SMA Calculator
# 
# Usage:
#   python test_sma_calculator.py
#   python test_sma_calculator.py --symbols RELIANCE HDFCBANK --period 1y --interval 1d
#   python test_sma_calculator.py --timeframe 5m
# =================================================================

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

# Setup path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source_code.common.config_loader import load_config
from source_code.common.logger import get_logger
from source_code.ingestion.batch_fetcher import BatchCandleFetcher
from source_code.processing.analysis.sma_calculator import SMACalculator

logger = get_logger(__name__)

# Default test symbols
DEFAULT_SYMBOLS = ["RELIANCE", "HDFCBANK", "INFY"]

# Test scenarios with different timeframes
TEST_SCENARIOS = {
    "intraday_1min": {
        "symbols": ["RELIANCE", "HDFCBANK"],
        "period": "5d",
        "interval": "1m",
        "sma_windows": [5, 10, 20]
    },
    "intraday_5min": {
        "symbols": ["RELIANCE", "HDFCBANK"],
        "period": "1mo",
        "interval": "5m",
        "sma_windows": [12, 24, 60]
    },
    "intraday_15min": {
        "symbols": ["RELIANCE", "HDFCBANK"],
        "period": "2mo",
        "interval": "15m",
        "sma_windows": [20, 40, 60]
    },
    "daily": {
        "symbols": ["RELIANCE", "HDFCBANK", "INFY", "ICICIBANK"],
        "period": "2y",
        "interval": "1d",
        "sma_windows": [20, 50, 100, 200]
    },
    "weekly": {
        "symbols": ["RELIANCE", "HDFCBANK", "INFY"],
        "period": "5y",
        "interval": "1wk",
        "sma_windows": [13, 26, 52]
    }
}


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 100)
    print(f"  {title}")
    print("=" * 100)


def print_section(title: str):
    """Print a formatted section."""
    print(f"\n{title}")
    print("-" * 100)


def test_single_scenario(
    scenario_name: str,
    symbols: list,
    period: str,
    interval: str,
    sma_windows: list,
    fetcher: BatchCandleFetcher,
    calculator: SMACalculator
):
    """Test a single scenario: fetch data and calculate SMAs."""
    
    print_section(f"🔍 Testing: {scenario_name} | Period={period} | Interval={interval}")
    
    try:
        # Step 1: Fetch data
        logger.info(f"Fetching {len(symbols)} symbols with interval={interval}, period={period}")
        print(f"\n📥 Fetching data for symbols: {', '.join(symbols)}")
        print(f"   Period: {period} | Interval: {interval}")
        
        raw_data = fetcher.fetch_batch(symbols, period=period, interval=interval)
        
        if not raw_data:
            print("❌ No data fetched!")
            return False
        
        print(f"✅ Fetched {len(raw_data)} symbols successfully")
        for symbol, df in raw_data.items():
            print(f"   {symbol}: {len(df)} rows")
        
        # Step 2: Calculate SMAs
        print(f"\n📊 Calculating SMAs: {sma_windows}")
        enriched_data = calculator.process_batch(raw_data, windows=sma_windows)
        
        # Step 3: Display results
        print(f"✅ SMA calculation complete")
        
        for symbol in symbols:
            if symbol not in enriched_data:
                print(f"⚠️  {symbol} not in results")
                continue
            
            df = enriched_data[symbol]
            print(f"\n📋 {symbol} (Last 5 rows):")
            print(f"   Rows: {len(df)} | Columns: {list(df.columns)}")
            
            # Prepare display columns
            display_cols = ["Open", "High", "Low", "Close", "Volume"]
            display_cols += [f"SMA_{w}" for w in sma_windows if f"SMA_{w}" in df.columns]
            
            # Filter to available columns
            display_cols = [col for col in display_cols if col in df.columns]
            
            # Format and display
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", 150)
            pd.set_option("display.float_format", "{:.2f}".format)
            
            print(df[display_cols].tail(5).to_string())
            
            # Get summary statistics
            summary = calculator.get_sma_summary(df)
            print(f"\n   SMA Summary:")
            for sma_col, stats in summary.items():
                print(f"     {sma_col}: Latest={stats['latest']:.2f} | "
                      f"Min={stats['min']:.2f} | Max={stats['max']:.2f} | "
                      f"NaN_Count={stats['nan_count']}")
        
        return True
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.exception(f"Exception in {scenario_name}:")
        return False


def test_all_scenarios(fetcher: BatchCandleFetcher, calculator: SMACalculator):
    """Test all predefined scenarios."""
    print_header("🧪 TESTING ALL SCENARIOS")
    
    results = {}
    for scenario_name, config in TEST_SCENARIOS.items():
        success = test_single_scenario(
            scenario_name=scenario_name,
            symbols=config["symbols"],
            period=config["period"],
            interval=config["interval"],
            sma_windows=config["sma_windows"],
            fetcher=fetcher,
            calculator=calculator
        )
        results[scenario_name] = "✅ PASS" if success else "❌ FAIL"
    
    # Summary
    print_header("📊 TEST SUMMARY")
    for scenario, result in results.items():
        print(f"  {result} | {scenario}")
    
    passed = sum(1 for r in results.values() if "PASS" in r)
    total = len(results)
    print(f"\nTotal: {passed}/{total} passed")


def test_custom(
    symbols: list,
    period: str,
    interval: str,
    sma_windows: list,
    fetcher: BatchCandleFetcher,
    calculator: SMACalculator
):
    """Test custom parameters."""
    print_header("🎯 CUSTOM TEST")
    
    success = test_single_scenario(
        scenario_name="custom",
        symbols=symbols,
        period=period,
        interval=interval,
        sma_windows=sma_windows,
        fetcher=fetcher,
        calculator=calculator
    )
    
    if success:
        print_section("✅ Custom test completed successfully!")
    else:
        print_section("❌ Custom test failed!")


def main():
    parser = argparse.ArgumentParser(
        description="Test SMA Calculator with real Zerodha data"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all predefined test scenarios"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Stock symbols to test (e.g., RELIANCE HDFCBANK INFY)"
    )
    parser.add_argument(
        "--period",
        default="1y",
        help="Period for data fetch (e.g., 1d, 5d, 1mo, 1y)"
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="Candle interval (e.g., 1m, 5m, 15m, 1h, 1d, 1wk)"
    )
    parser.add_argument(
        "--sma",
        nargs="+",
        type=int,
        default=[20, 50, 100],
        help="SMA windows to calculate (e.g., 20 50 100)"
    )
    parser.add_argument(
        "--scenario",
        choices=list(TEST_SCENARIOS.keys()),
        help="Run a specific predefined scenario"
    )
    
    args = parser.parse_args()
    
    # Initialize
    config = load_config()
    fetcher = BatchCandleFetcher(config, provider="yfinance")  # Use yfinance for testing
    calculator = SMACalculator(config)
    
    print_header("🚀 SMA CALCULATOR TEST SUITE")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: {config.get('project', {}).get('name')} v{config.get('project', {}).get('version')}")
    
    # Determine what to test
    if args.all:
        test_all_scenarios(fetcher, calculator)
    
    elif args.scenario:
        scenario_config = TEST_SCENARIOS[args.scenario]
        test_single_scenario(
            scenario_name=args.scenario,
            symbols=scenario_config["symbols"],
            period=scenario_config["period"],
            interval=scenario_config["interval"],
            sma_windows=scenario_config["sma_windows"],
            fetcher=fetcher,
            calculator=calculator
        )
    
    else:
        # Custom test
        symbols = args.symbols or DEFAULT_SYMBOLS
        test_custom(
            symbols=symbols,
            period=args.period,
            interval=args.interval,
            sma_windows=args.sma,
            fetcher=fetcher,
            calculator=calculator
        )
    
    print_header("✅ Test Complete")


if __name__ == "__main__":
    main()
