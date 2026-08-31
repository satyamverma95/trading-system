# =================================================================
# source_code/processing/analysis/sma_calculator.py
# LAYER 4: PROCESSING - SMA (Simple Moving Average) Calculator
# Timeframe-agnostic SMA calculation for any interval (1m, 5m, 1d, etc.)
# =================================================================

import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from source_code.common.config_loader import load_config

logger = logging.getLogger(__name__)


class SMACalculator:
    """
    Calculate Simple Moving Averages (SMA) for stock price data.
    
    Features:
    - Timeframe-agnostic: Works with any OHLCV interval (1m, 5m, 15m, 1h, 1d, etc.)
    - Multi-period: Calculate SMA 20, 40, 60, 100, etc. simultaneously
    - Batch processing: Process multiple symbols in one call
    - Configuration-driven: SMA periods from settings.yaml
    - NaN-safe: Handles edge cases gracefully
    
    Example:
        # Fetch 1-minute data
        data = fetcher.fetch_batch(["RELIANCE", "HDFCBANK"], period="5d", interval="1m")
        
        # Calculate SMAs (same code works for any timeframe!)
        calc = SMACalculator()
        enriched = calc.process_batch(data, windows=[5, 10, 20])
        
        # Result: Dict[symbol] = DataFrame with SMA_5, SMA_10, SMA_20 columns
        print(enriched["RELIANCE"].columns)
        # ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_10', 'SMA_20']
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize SMA Calculator.
        
        Args:
            config: Config dict. If None, loads from settings.yaml
        """
        self.config = config or load_config()
        
        # Get SMA periods from config, fallback to defaults
        indicators_config = self.config.get("indicators", {})
        self.default_sma_periods = indicators_config.get("sma_periods", [20, 50, 100])
        
        logger.info(f"SMACalculator initialized with default periods: {self.default_sma_periods}")

    def compute_sma(self, series: pd.Series, window: int) -> pd.Series:
        """
        Calculate Simple Moving Average for a single Series.
        
        Pure function - no side effects.
        
        Args:
            series: pd.Series of prices (usually Close prices)
            window: SMA window/period (e.g., 20, 50, 100)
            
        Returns:
            pd.Series with SMA values. First (window-1) rows will be NaN.
            
        Example:
            >>> close_prices = pd.Series([100, 101, 102, 103, 104, 105])
            >>> sma = calc.compute_sma(close_prices, window=3)
            >>> sma
            0      NaN
            1      NaN
            2    101.0
            3    102.0
            4    103.0
            5    104.0
        """
        if window < 1:
            raise ValueError(f"Window must be >= 1, got {window}")
        
        if len(series) < window:
            logger.warning(
                f"Series length ({len(series)}) < window ({window}). "
                f"Returning all NaN."
            )
            return pd.Series(np.nan, index=series.index)
        
        try:
            sma = series.rolling(window=window, min_periods=1).mean()
            return sma
        except Exception as e:
            logger.error(f"Error computing SMA with window={window}: {e}")
            raise

    def add_sma_columns(
        self, 
        df: pd.DataFrame, 
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Add SMA columns to a single OHLCV DataFrame.
        
        Args:
            df: OHLCV DataFrame with at least a 'Close' column
            windows: List of SMA periods (e.g., [20, 50, 100])
                     If None, uses default from config
            
        Returns:
            New DataFrame with original columns + SMA columns
            Column names: SMA_20, SMA_50, SMA_100, etc.
            
        Raises:
            ValueError: If 'Close' column not found in DataFrame
            
        Example:
            >>> df = pd.DataFrame({
            ...     'Open': [100, 101, 102],
            ...     'Close': [100.5, 101.5, 102.5]
            ... })
            >>> enriched = calc.add_sma_columns(df, windows=[2])
            >>> enriched.columns.tolist()
            ['Open', 'Close', 'SMA_2']
        """
        if windows is None:
            windows = self.default_sma_periods
        
        if "Close" not in df.columns:
            raise ValueError(
                f"DataFrame must have 'Close' column. "
                f"Found: {df.columns.tolist()}"
            )
        
        # Create a copy to avoid modifying input
        result_df = df.copy()
        close_prices = df["Close"].astype(float)
        
        logger.debug(
            f"Adding SMA columns to DataFrame with {len(df)} rows. "
            f"Windows: {windows}"
        )
        
        for window in windows:
            col_name = f"SMA_{window}"
            try:
                result_df[col_name] = self.compute_sma(close_prices, window=window)
                logger.debug(f"  Added {col_name}")
            except Exception as e:
                logger.error(f"  Failed to add {col_name}: {e}")
                raise
        
        logger.debug(f"Final columns: {result_df.columns.tolist()}")
        return result_df

    def process_batch(
        self, 
        symbol_data: Dict[str, pd.DataFrame], 
        windows: Optional[List[int]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Process multiple symbols and add SMA columns to all.
        
        Main entry point for batch processing.
        
        Args:
            symbol_data: Dict mapping symbol name to OHLCV DataFrame
                        Example: {
                            "RELIANCE": DataFrame(1000 rows, 5 cols),
                            "HDFCBANK": DataFrame(800 rows, 5 cols)
                        }
            windows: List of SMA periods. If None, uses default from config
            
        Returns:
            Dict[symbol] = enriched DataFrame with SMA columns
            
        Example:
            >>> from source_code.ingestion.batch_fetcher import BatchCandleFetcher
            >>> fetcher = BatchCandleFetcher()
            >>> data = fetcher.fetch_batch(["RELIANCE", "HDFCBANK"], 
            ...                            period="1y", interval="1d")
            >>> calc = SMACalculator()
            >>> enriched = calc.process_batch(data, windows=[20, 50])
            >>> print(enriched["RELIANCE"].tail(2)[["Close", "SMA_20", "SMA_50"]])
        """
        if windows is None:
            windows = self.default_sma_periods
        
        logger.info(
            f"Processing batch for {len(symbol_data)} symbols. "
            f"SMA windows: {windows}"
        )
        
        enriched_data = {}
        
        for symbol, df in symbol_data.items():
            try:
                logger.info(f"  Processing {symbol}: {len(df)} rows")
                enriched_df = self.add_sma_columns(df, windows=windows)
                enriched_data[symbol] = enriched_df
                
                # Log sample of results
                if len(enriched_df) > 0:
                    last_row = enriched_df.iloc[-1]
                    close_val = last_row.get('Close', 'N/A')
                    sma_val = last_row.get('SMA_20', 'N/A')
                    
                    # Format safely
                    close_str = f"{close_val:.2f}" if isinstance(close_val, (int, float)) else str(close_val)
                    sma_str = f"{sma_val:.2f}" if isinstance(sma_val, (int, float)) else str(sma_val)
                    
                    logger.debug(
                        f"    Last row - Close: {close_str}, SMA_20: {sma_str}"
                    )
            
            except Exception as e:
                logger.error(f"  Failed to process {symbol}: {e}")
                raise
        
        logger.info(f"Batch processing complete. Enriched {len(enriched_data)} symbols.")
        return enriched_data

    def get_sma_summary(self, df: pd.DataFrame) -> dict:
        """
        Get summary statistics for SMA columns in a DataFrame.
        
        Useful for quick verification of SMA values.
        
        Args:
            df: DataFrame with SMA columns
            
        Returns:
            Dict with summary stats for each SMA column
            
        Example:
            >>> summary = calc.get_sma_summary(enriched["RELIANCE"])
            >>> print(summary)
            {
                'SMA_20': {'latest': 1500.25, 'min': 1420.0, 'max': 1520.0},
                'SMA_50': {'latest': 1480.50, 'min': 1400.0, 'max': 1510.0}
            }
        """
        summary = {}
        
        # Find all SMA columns
        sma_cols = [col for col in df.columns if col.startswith("SMA_")]
        
        for col in sma_cols:
            sma_series = pd.to_numeric(df[col], errors="coerce")
            summary[col] = {
                "latest": sma_series.iloc[-1] if len(df) > 0 else None,
                "min": sma_series.min(),
                "max": sma_series.max(),
                "mean": sma_series.mean(),
                "nan_count": sma_series.isna().sum()
            }
        
        return summary
