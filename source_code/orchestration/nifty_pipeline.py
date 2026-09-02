"""CLI orchestration for the Nifty signal detection pipeline."""

import argparse
import logging
from typing import Dict, List, Optional

import pandas as pd

from source_code.common.config_loader import load_config
from source_code.ingestion.nifty_loader import load_nifty100_universe
from source_code.ingestion.batch_fetcher import BatchCandleFetcher
from source_code.processing.analysis.sma_calculator import SMACalculator
from source_code.processing.analysis.crossover_detector import CrossoverDetector
from source_code.processing.analysis.ranking_engine import RankingEngine
from source_code.ingestion.data.result_writer import ResultWriter

logger = logging.getLogger(__name__)


class NiftyPipeline:
    """Coordinate universe loading, analysis, ranking, and persistence."""

    def __init__(self, config: Optional[dict] = None, provider: Optional[str] = None):
        self.config = config or load_config()
        provider_name = provider or self.config.get("data_provider", "yfinance")
        self.fetcher = BatchCandleFetcher(self.config, provider=provider_name)
        self.sma_calculator = SMACalculator(self.config)
        self.result_writer = ResultWriter(self.config)

    def run(
        self,
        symbols: Optional[List[str]] = None,
        universe_csv: Optional[str] = None,
        period: Optional[str] = None,
        interval: Optional[str] = None,
        sma_fast: int = 20,
        sma_slow: int = 50,
        decay_factor: float = 1.0,
        rank_by: str = "days_since",
        state_filter: str = "ALL",
        days_threshold: int = 0,
        score_threshold: float = 0.0,
        top_n: Optional[int] = None,
        output_format: str = "csv",
        output_path: Optional[str] = None,
    ) -> Dict[str, object]:
        """Run the complete pipeline and return data plus output metadata."""
        selected_symbols = symbols or load_nifty100_universe(universe_csv)
        if not selected_symbols:
            raise ValueError("No symbols available for pipeline run")

        logger.info("Pipeline started for %d symbols", len(selected_symbols))
        raw_data = self.fetcher.fetch_batch(
            selected_symbols,
            period=period,
            interval=interval,
            skip_missing=True,
        )
        if not raw_data:
            raise RuntimeError("No market data was fetched")

        sma_data = self.sma_calculator.process_batch(
            raw_data,
            windows=sorted(set([sma_fast, sma_slow])),
        )
        crossover_data = CrossoverDetector(
            self.config,
            sma_fast_period=sma_fast,
            sma_slow_period=sma_slow,
        ).process_batch(sma_data, decay_factor=decay_factor)

        ranking_engine = RankingEngine(
            self.config,
            rank_by=rank_by,
            state_filter=state_filter,
            days_threshold=days_threshold,
            score_threshold=score_threshold,
        )
        ranked_results = ranking_engine.rank_batch(crossover_data, top_n=top_n)
        saved_path = self.result_writer.save_results(
            ranked_results,
            output_path=output_path,
            format=output_format,
        )

        logger.info("Pipeline complete: %d ranked symbols", len(ranked_results))
        return {
            "symbols_requested": selected_symbols,
            "symbols_fetched": list(raw_data),
            "raw_data": raw_data,
            "sma_data": sma_data,
            "crossover_data": crossover_data,
            "ranked_results": ranked_results,
            "output_path": saved_path,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Nifty crossover signal pipeline")
    parser.add_argument("--symbols", nargs="+", help="Symbols to process")
    parser.add_argument("--universe-csv", help="CSV containing the Nifty universe")
    parser.add_argument("--provider", choices=["yfinance", "zerodha"], help="Data provider")
    parser.add_argument("--period", default=None, help="Relative period such as 1y or 1mo")
    parser.add_argument("--interval", default=None, help="Candle interval such as 1d or 5m")
    parser.add_argument("--sma-fast", type=int, default=20)
    parser.add_argument("--sma-slow", type=int, default=50)
    parser.add_argument("--decay-factor", type=float, default=1.0)
    parser.add_argument("--rank-by", choices=["days_since", "score", "hybrid"], default="days_since")
    parser.add_argument("--state-filter", choices=["ALL", "BULLISH", "BEARISH"], default="ALL")
    parser.add_argument("--days-threshold", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--top-n", type=int)
    parser.add_argument("--output-format", choices=["csv", "json", "parquet", "html"], default="csv")
    parser.add_argument("--output-path", help="Output path; defaults to data/gold/nifty100_signals.<format>")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pipeline = NiftyPipeline(provider=args.provider)
    result = pipeline.run(
        symbols=args.symbols,
        universe_csv=args.universe_csv,
        period=args.period,
        interval=args.interval,
        sma_fast=args.sma_fast,
        sma_slow=args.sma_slow,
        decay_factor=args.decay_factor,
        rank_by=args.rank_by,
        state_filter=args.state_filter,
        days_threshold=args.days_threshold,
        score_threshold=args.score_threshold,
        top_n=args.top_n,
        output_format=args.output_format,
        output_path=args.output_path,
    )
    print(result["ranked_results"].to_string(index=False))
    print(f"\nSaved results to: {result['output_path']}")


if __name__ == "__main__":
    main()
