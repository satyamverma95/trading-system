# =================================================================
# source_code/processing/analysis/crossover_detector.py
# LAYER 5: PROCESSING - Crossover Detector
# Identifies bullish and bearish SMA crossovers for trading signals
# =================================================================

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from source_code.common.config_loader import load_config

logger = logging.getLogger(__name__)


class CrossoverDetector:
    """
    Detect Simple Moving Average (SMA) crossover signals.
    
    Features:
    - Identifies bullish crossovers (SMA_fast crosses above SMA_slow)
    - Identifies bearish crossovers (SMA_fast crosses below SMA_slow)
    - Tracks current state (bullish/bearish) for each row
    - Calculates days since last crossover for ranking
    - Generates crossover scores for signal strength
    - Batch processes multiple symbols
    
    Example:
        # With SMA-enriched data
        enriched_data = {
            "RELIANCE": DataFrame(251 rows, cols: O,H,L,C,V,SMA_20,SMA_50),
            "HDFCBANK": DataFrame(251 rows, cols: O,H,L,C,V,SMA_20,SMA_50)
        }
        
        detector = CrossoverDetector()
        signals = detector.process_batch(enriched_data)
        
        # Result: DataFrames with additional columns:
        # - Crossover_Signal: 'BULLISH', 'BEARISH', or 'NONE'
        # - Crossover_State: Current state ('BULLISH' or 'BEARISH')
        # - Days_Since_Crossover: Days since last crossover
        # - Crossover_Score: Signal strength score (0-100)
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        sma_fast_period: int = 20,
        sma_slow_period: int = 50
    ):
        """
        Initialize Crossover Detector.
        
        Args:
            config: Config dict. If None, loads from settings.yaml
            sma_fast_period: Fast SMA period (default: 20)
            sma_slow_period: Slow SMA period (default: 50)
        """
        self.config = config or load_config()
        self.sma_fast_period = sma_fast_period
        self.sma_slow_period = sma_slow_period
        
        logger.info(
            f"CrossoverDetector initialized: "
            f"SMA_{sma_fast_period} vs SMA_{sma_slow_period}"
        )

    def _get_sma_column_names(self) -> Tuple[str, str]:
        """Get the SMA column names based on periods."""
        return (
            f"SMA_{self.sma_fast_period}",
            f"SMA_{self.sma_slow_period}"
        )

    def detect_crossovers(
        self,
        df: pd.DataFrame,
        decay_factor: float = 1.0
    ) -> pd.DataFrame:
        """
        Detect SMA crossover signals in a single DataFrame.
        
        Args:
            df: OHLCV DataFrame with SMA columns
            decay_factor: How much to decay the score per day
                         (1.0 = 1 point per day, 0.5 = 0.5 points per day)
            
        Returns:
            DataFrame with additional columns:
            - Crossover_Signal: 'BULLISH', 'BEARISH', or 'NONE'
            - Crossover_State: 'BULLISH' or 'BEARISH'
            - Days_Since_Crossover: Days since last crossover
            - Crossover_Score: 0-100 score based on recency
            
        Raises:
            ValueError: If required SMA columns not found
        """
        sma_fast_col, sma_slow_col = self._get_sma_column_names()
        
        # Validate columns exist
        if sma_fast_col not in df.columns:
            raise ValueError(f"Column '{sma_fast_col}' not found in DataFrame")
        if sma_slow_col not in df.columns:
            raise ValueError(f"Column '{sma_slow_col}' not found in DataFrame")
        
        df_copy = df.copy()
        
        # Extract SMA columns as numeric
        sma_fast = pd.to_numeric(df_copy[sma_fast_col], errors='coerce')
        sma_slow = pd.to_numeric(df_copy[sma_slow_col], errors='coerce')
        
        logger.debug(
            f"Detecting crossovers: {len(df)} rows, "
            f"{sma_fast.notna().sum()} valid SMA values"
        )
        
        # =====================================
        # STEP 1: Calculate Current State
        # =====================================
        # True = SMA_fast > SMA_slow (bullish condition)
        is_above = (sma_fast > sma_slow).fillna(False)
        
        # =====================================
        # STEP 2: Detect Crossover Events
        # =====================================
        # Shift to get previous row's state
        was_above = is_above.shift(1).fillna(False)
        
        # Bullish crossover: was NOT above, now IS above
        bullish_cross = (~was_above) & (is_above)
        
        # Bearish crossover: was above, now is NOT above
        bearish_cross = (was_above) & (~is_above)
        
        # Create crossover signal column
        df_copy['Crossover_Signal'] = 'NONE'
        df_copy.loc[bullish_cross, 'Crossover_Signal'] = 'BULLISH'
        df_copy.loc[bearish_cross, 'Crossover_Signal'] = 'BEARISH'
        
        # =====================================
        # STEP 3: Track Current State
        # =====================================
        df_copy['Crossover_State'] = 'BEARISH'
        df_copy.loc[is_above, 'Crossover_State'] = 'BULLISH'
        
        # Forward fill state for first row (if NaN in SMAs)
        df_copy['Crossover_State'] = df_copy['Crossover_State'].bfill()
        
        # =====================================
        # STEP 4: Calculate Days Since Crossover
        # =====================================
        days_since = self._calculate_days_since_crossover(
            df_copy['Crossover_Signal'],
            df_copy.index
        )
        df_copy['Days_Since_Crossover'] = days_since
        
        # =====================================
        # STEP 5: Calculate Crossover Score
        # =====================================
        crossover_scores = self._calculate_crossover_scores(
            df_copy['Crossover_State'],
            df_copy['Days_Since_Crossover'],
            decay_factor=decay_factor
        )
        df_copy['Crossover_Score'] = crossover_scores
        
        # =====================================
        # STEP 6: Track Last Crossover Type
        # =====================================
        last_crossover = self._get_last_crossover_type(df_copy['Crossover_Signal'])
        df_copy['Last_Crossover_Type'] = last_crossover
        
        logger.debug(
            f"Crossover detection complete: "
            f"{bullish_cross.sum()} bullish, {bearish_cross.sum()} bearish"
        )
        
        return df_copy

    def _calculate_days_since_crossover(
        self,
        signal_series: pd.Series,
        index: pd.Index
    ) -> pd.Series:
        """
        Calculate days since the last crossover event.
        
        Returns:
            Series with days since last crossover
            (NaN if no crossover yet, 0 on crossover day)
        """
        days_since = pd.Series(np.nan, index=index)
        
        # Find all crossover events
        crossover_mask = signal_series != 'NONE'
        crossover_indices = index[crossover_mask]
        
        if len(crossover_indices) == 0:
            return days_since
        
        # For each row, calculate days since last crossover
        for i, current_date in enumerate(index):
            # Find last crossover before or at current date
            last_crossovers = crossover_indices[crossover_indices <= current_date]
            
            if len(last_crossovers) > 0:
                last_date = last_crossovers[-1]
                
                # Calculate days difference
                if hasattr(index, 'day'):  # datetime index
                    days = (current_date - last_date).days
                else:
                    days = i - list(index).index(last_date)
                
                days_since.iloc[i] = max(0, days)
        
        return days_since

    def _calculate_crossover_scores(
        self,
        state_series: pd.Series,
        days_since_series: pd.Series,
        decay_factor: float = 1.0
    ) -> pd.Series:
        """
        Calculate crossover score based on state and recency.
        
        Score Logic:
        - Bullish state: 100 - (days_since * decay_factor)
        - Bearish state: 0 + (days_since * decay_factor * 0.5)
        
        Score ranges from 0-100 where higher = more bullish
        """
        scores = pd.Series(0.0, index=state_series.index)
        
        for i, state in enumerate(state_series):
            days = days_since_series.iloc[i]
            
            # Skip if NaN days
            if pd.isna(days):
                scores.iloc[i] = 0.0
                continue
            
            if state == 'BULLISH':
                # Bullish: start at 100, decay with time
                score = max(0, 100 - (days * decay_factor))
            else:  # BEARISH
                # Bearish: start at 0, slight bonus for recent signal
                score = min(100, days * decay_factor * 0.5)
            
            scores.iloc[i] = score
        
        return scores

    def _get_last_crossover_type(self, signal_series: pd.Series) -> pd.Series:
        """
        Track the type of the last crossover (BULLISH or BEARISH).
        Forward-fill to current row.
        """
        last_crossover = pd.Series('NONE', index=signal_series.index)
        
        current_type = 'NONE'
        for i, signal in enumerate(signal_series):
            if signal != 'NONE':
                current_type = signal
            last_crossover.iloc[i] = current_type
        
        return last_crossover

    def process_batch(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        decay_factor: float = 1.0
    ) -> Dict[str, pd.DataFrame]:
        """
        Process multiple symbols and detect crossovers for all.
        
        Args:
            symbol_data: Dict mapping symbol to OHLCV DataFrame
                        (must already have SMA columns from Module 4)
            decay_factor: Score decay per day (higher = slower decay)
            
        Returns:
            Dict[symbol] = DataFrame with crossover columns added
            
        Example:
            >>> symbols_with_sma = {
            ...     "RELIANCE": df_with_sma_20_50,
            ...     "HDFCBANK": df_with_sma_20_50
            ... }
            >>> detector = CrossoverDetector()
            >>> signals = detector.process_batch(symbols_with_sma)
            >>> print(signals["RELIANCE"].tail()[["Close", "SMA_20", "SMA_50", "Crossover_Signal"]])
        """
        logger.info(
            f"Processing batch for {len(symbol_data)} symbols"
        )
        
        enriched_data = {}
        
        for symbol, df in symbol_data.items():
            try:
                logger.info(f"  Processing {symbol}: {len(df)} rows")
                enriched_df = self.detect_crossovers(df, decay_factor=decay_factor)
                enriched_data[symbol] = enriched_df
                
                # Log sample statistics
                bullish_count = (enriched_df['Crossover_Signal'] == 'BULLISH').sum()
                bearish_count = (enriched_df['Crossover_Signal'] == 'BEARISH').sum()
                logger.debug(
                    f"    Crossovers: {bullish_count} bullish, {bearish_count} bearish"
                )
            
            except Exception as e:
                logger.error(f"  Failed to process {symbol}: {e}")
                raise
        
        logger.info(f"Batch processing complete. Enriched {len(enriched_data)} symbols.")
        return enriched_data

    def get_crossover_summary(self, df: pd.DataFrame) -> dict:
        """
        Get summary of crossovers in a DataFrame.
        
        Useful for quick verification and analysis.
        
        Args:
            df: DataFrame with crossover columns (output of detect_crossovers)
            
        Returns:
            Dict with summary statistics
            
        Example:
            >>> summary = detector.get_crossover_summary(enriched_df)
            >>> print(summary)
            {
                'total_bullish_crosses': 12,
                'total_bearish_crosses': 11,
                'current_state': 'BULLISH',
                'days_since_last_cross': 3,
                'last_crossover_type': 'BULLISH',
                'avg_score': 65.4,
                'latest_score': 85.2
            }
        """
        summary = {
            'total_bullish_crosses': (df['Crossover_Signal'] == 'BULLISH').sum(),
            'total_bearish_crosses': (df['Crossover_Signal'] == 'BEARISH').sum(),
            'current_state': df['Crossover_State'].iloc[-1] if len(df) > 0 else 'UNKNOWN',
            'days_since_last_cross': df['Days_Since_Crossover'].iloc[-1] if len(df) > 0 else np.nan,
            'last_crossover_type': df['Last_Crossover_Type'].iloc[-1] if len(df) > 0 else 'NONE',
            'avg_score': df['Crossover_Score'].mean(),
            'latest_score': df['Crossover_Score'].iloc[-1] if len(df) > 0 else 0.0,
            'min_score': df['Crossover_Score'].min(),
            'max_score': df['Crossover_Score'].max(),
        }
        return summary

    def get_recent_signals(
        self,
        df: pd.DataFrame,
        days: int = 30,
        signal_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get recent crossover signals.
        
        Args:
            df: DataFrame with crossover columns
            days: Look back period
            signal_type: 'BULLISH', 'BEARISH', or None for all
            
        Returns:
            Filtered DataFrame with recent signals
            
        Example:
            >>> recent = detector.get_recent_signals(df, days=7, signal_type='BULLISH')
            >>> print(recent[['Date', 'Close', 'Crossover_Signal', 'Crossover_Score']])
        """
        # Find recent crossovers
        recent_signals = df[df['Crossover_Signal'] != 'NONE'].copy()
        
        if signal_type:
            recent_signals = recent_signals[
                recent_signals['Crossover_Signal'] == signal_type
            ]
        
        # Filter by days if possible
        if len(recent_signals) > 0 and hasattr(df.index, 'day'):
            cutoff_date = df.index[-1] - pd.Timedelta(days=days)
            recent_signals = recent_signals[recent_signals.index >= cutoff_date]
        
        return recent_signals
