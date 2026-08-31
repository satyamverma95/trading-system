# =================================================================
# test_crossover_detector.py
# Test script: Validate Crossover Detector with real SMA data
# 
# Usage:
#   python test_crossover_detector.py
#   python test_crossover_detector.py --symbols RELIANCE HDFCBANK --period 1y --interval 1d
#   python test_crossover_detector.py --all
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
from source_code.processing.analysis.crossover_detector import CrossoverDetector

logger = get_logger(__name__)

# Default test symbols
DEFAULT_SYMBOLS = ["RELIANCE", "HDFCBANK"]

# Test scenarios
TEST_SCENARIOS = {
    "daily_standard": {
        "symbols": ["RELIANCE", "HDFCBANK", "INFY"],
        "period": "1y",
        "interval": "1d",
        "sma_fast": 20,
        "sma_slow": 50,
    },
    "daily_long": {
        "symbols": ["RELIANCE", "HDFCBANK"],
        "period": "2y",
        "interval": "1d",
        "sma_fast": 50,
        "sma_slow": 100,
    },
    "intraday_5min": {
        "symbols": ["RELIANCE", "HDFCBANK"],
        "period": "1mo",
        "interval": "5m",
        "sma_fast": 12,
        "sma_slow": 24,
    },
    "weekly": {
        "symbols": ["RELIANCE", "HDFCBANK", "INFY"],
        "period": "2y",
        "interval": "1wk",
        "sma_fast": 13,
        "sma_slow": 26,
    }
}


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 120)
    print(f"  {title}")
    print("=" * 120)


def print_section(title: str):
    """Print a formatted section."""
    print(f"\n{title}")
    print("-" * 120)


def test_scenario(
    scenario_name: str,
    symbols: list,
    period: str,
    interval: str,
    sma_fast: int,
    sma_slow: int,
    fetcher: BatchCandleFetcher,
    sma_calc: SMACalculator,
    crossover_calc: CrossoverDetector
):
    """Test a complete scenario: fetch → SMA → crossover detection"""
    
    print_section(f"🔍 Testing: {scenario_name}")
    print(f"Period={period} | Interval={interval} | SMA_{sma_fast}/{sma_slow}")
    
    try:
        # ============================================================
        # Step 1: Fetch Data
        # ============================================================
        logger.info(f"Fetching {len(symbols)} symbols (interval={interval}, period={period})")
        print(f"\n📥 Step 1: Fetching Data")
        print(f"   Symbols: {', '.join(symbols)}")
        
        raw_data = fetcher.fetch_batch(symbols, period=period, interval=interval)
        
        if not raw_data:
            print("❌ No data fetched!")
            return False
        
        print(f"✅ Fetched {len(raw_data)} symbols")
        for symbol, df in raw_data.items():
            print(f"   {symbol}: {len(df)} rows")
        
        # ============================================================
        # Step 2: Calculate SMAs
        # ============================================================
        print(f"\n📊 Step 2: Calculating SMAs")
        print(f"   Windows: SMA_{sma_fast}, SMA_{sma_slow}")
        
        enriched_sma = sma_calc.process_batch(raw_data, windows=[sma_fast, sma_slow])
        print(f"✅ SMA calculation complete")
        
        # ============================================================
        # Step 3: Detect Crossovers
        # ============================================================
        print(f"\n🎯 Step 3: Detecting Crossovers")
        
        crossover_calc_instance = CrossoverDetector(
            config=None,
            sma_fast_period=sma_fast,
            sma_slow_period=sma_slow
        )
        
        enriched_crossover = crossover_calc_instance.process_batch(enriched_sma)
        print(f"✅ Crossover detection complete")
        
        # ============================================================
        # Step 4: Display Results
        # ============================================================
        print(f"\n📋 Results by Symbol:")
        
        for symbol in symbols:
            if symbol not in enriched_crossover:
                print(f"\n⚠️  {symbol} not in results")
                continue
            
            df = enriched_crossover[symbol]
            
            print(f"\n{symbol.upper()}")
            print(f"  Rows: {len(df)}")
            
            # Summary statistics
            summary = crossover_calc_instance.get_crossover_summary(df)
            print(f"  Summary:")
            print(f"    Current State: {summary['current_state']}")
            print(f"    Days Since Crossover: {summary['days_since_last_cross']:.0f}")
            print(f"    Last Crossover: {summary['last_crossover_type']}")
            print(f"    Bullish Crosses: {summary['total_bullish_crosses']}")
            print(f"    Bearish Crosses: {summary['total_bearish_crosses']}")
            print(f"    Avg Score: {summary['avg_score']:.1f} | Latest Score: {summary['latest_score']:.1f}")
            
            # Display last 8 rows with key columns
            display_cols = [
                'Close', f'SMA_{sma_fast}', f'SMA_{sma_slow}',
                'Crossover_Signal', 'Crossover_State', 'Days_Since_Crossover', 'Crossover_Score'
            ]
            display_cols = [col for col in display_cols if col in df.columns]
            
            print(f"\n  Last 8 rows:")
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", 160)
            pd.set_option("display.float_format", "{:.2f}".format)
            
            print(df[display_cols].tail(8).to_string())
            
            # ========================================================
            # Show Recent Crossovers
            # ========================================================
            recent = crossover_calc_instance.get_recent_signals(df, days=90, signal_type='BULLISH')
            if len(recent) > 0:
                print(f"\n  Recent Bullish Crossovers (last 90 days): {len(recent)}")
                for idx, (date, row) in enumerate(recent.tail(5).iterrows(), 1):
                    print(f"    {idx}. {date.strftime('%Y-%m-%d')} - "
                          f"Close={row['Close']:.2f}, "
                          f"Score={row['Crossover_Score']:.1f}, "
                          f"Days={row['Days_Since_Crossover']:.0f}")
            
            recent = crossover_calc_instance.get_recent_signals(df, days=90, signal_type='BEARISH')
            if len(recent) > 0:
                print(f"\n  Recent Bearish Crossovers (last 90 days): {len(recent)}")
                for idx, (date, row) in enumerate(recent.tail(5).iterrows(), 1):
                    print(f"    {idx}. {date.strftime('%Y-%m-%d')} - "
                          f"Close={row['Close']:.2f}, "
                          f"Score={row['Crossover_Score']:.1f}, "
                          f"Days={row['Days_Since_Crossover']:.0f}")
        
        return True
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.exception(f"Exception in {scenario_name}:")
        return False


def test_all_scenarios(
    fetcher: BatchCandleFetcher,
    sma_calc: SMACalculator,
    crossover_calc: CrossoverDetector
):
    """Test all predefined scenarios"""
    print_header("🧪 TESTING ALL SCENARIOS")
    
    results = {}
    for scenario_name, config in TEST_SCENARIOS.items():
        success = test_scenario(
            scenario_name=scenario_name,
            symbols=config["symbols"],
            period=config["period"],
            interval=config["interval"],
            sma_fast=config["sma_fast"],
            sma_slow=config["sma_slow"],
            fetcher=fetcher,
            sma_calc=sma_calc,
            crossover_calc=crossover_calc
        )
        results[scenario_name] = "✅ PASS" if success else "❌ FAIL"
    
    # Summary
    print_header("📊 TEST SUMMARY")
    for scenario, result in results.items():
        print(f"  {result} | {scenario}")
    
    passed = sum(1 for r in results.values() if "PASS" in r)
    total = len(results)
    print(f"\nTotal: {passed}/{total} passed")


def main():
    parser = argparse.ArgumentParser(
        description="Test Crossover Detector with SMA-enriched data"
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
        "--sma-fast",
        type=int,
        default=20,
        help="Fast SMA period (default: 20)"
    )
    parser.add_argument(
        "--sma-slow",
        type=int,
        default=50,
        help="Slow SMA period (default: 50)"
    )
    parser.add_argument(
        "--scenario",
        choices=list(TEST_SCENARIOS.keys()),
        help="Run a specific predefined scenario"
    )
    
    args = parser.parse_args()
    
    # Initialize
    config = load_config()
    fetcher = BatchCandleFetcher(config, provider="yfinance")
    sma_calc = SMACalculator(config)
    crossover_calc = CrossoverDetector(config)
    
    print_header("🚀 CROSSOVER DETECTOR TEST SUITE")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: {config.get('project', {}).get('name')} v{config.get('project', {}).get('version')}")
    
    # Determine what to test
    if args.all:
        test_all_scenarios(fetcher, sma_calc, crossover_calc)
    
    elif args.scenario:
        scenario_config = TEST_SCENARIOS[args.scenario]
        test_scenario(
            scenario_name=args.scenario,
            symbols=scenario_config["symbols"],
            period=scenario_config["period"],
            interval=scenario_config["interval"],
            sma_fast=scenario_config["sma_fast"],
            sma_slow=scenario_config["sma_slow"],
            fetcher=fetcher,
            sma_calc=sma_calc,
            crossover_calc=crossover_calc
        )
    
    else:
        # Custom test
        symbols = args.symbols or DEFAULT_SYMBOLS
        test_scenario(
            scenario_name="custom",
            symbols=symbols,
            period=args.period,
            interval=args.interval,
            sma_fast=args.sma_fast,
            sma_slow=args.sma_slow,
            fetcher=fetcher,
            sma_calc=sma_calc,
            crossover_calc=crossover_calc
        )
    
    print_header("✅ Test Complete")


if __name__ == "__main__":
    main()
