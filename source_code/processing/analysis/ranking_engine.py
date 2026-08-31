# =================================================================
# ranking_engine.py
# Module 6: Ranking & Scoring Engine
#
# Purpose:
#   Convert crossover signals into ranked watchlist.
#   Identifies most promising trading candidates based on:
#   - Signal state (BULLISH > BEARISH)
#   - Signal recency (fresher signals higher priority)
#   - Signal strength (crossover score as tie-breaker)
#
# Input: Dict[symbol] = DataFrame with crossover columns
# Output: Ranked DataFrame sorted by trading opportunity
#
# Author: Trading System
# =================================================================

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class RankingEngine:
    """
    Ranks stocks by crossover signal quality and recency.
    
    Ranking Strategy:
    1. Separate BULLISH and BEARISH signals
    2. Sort by Days_Since_Crossover (ascending) - fresher first
    3. Use Crossover_Score as tie-breaker (descending) - stronger first
    4. Concatenate groups with BULLISH first
    
    Attributes:
        config (dict): Application configuration
        rank_by (str): Primary ranking method ('days_since', 'score', 'hybrid')
        state_filter (str): Filter signals ('BULLISH', 'BEARISH', 'ALL')
        days_threshold (int): Ignore signals older than N days (0 = no filter)
        score_threshold (float): Ignore signals with score < threshold
    """
    
    def __init__(
        self,
        config: Optional[dict] = None,
        rank_by: str = 'days_since',
        state_filter: str = 'ALL',
        days_threshold: int = 0,
        score_threshold: float = 0.0
    ):
        """
        Initialize Ranking Engine.
        
        Args:
            config: Configuration dictionary
            rank_by: 'days_since' (default) | 'score' | 'hybrid'
            state_filter: 'ALL' (default) | 'BULLISH' | 'BEARISH'
            days_threshold: Exclude signals older than N days (0 = no limit)
            score_threshold: Exclude signals with score < threshold
        """
        self.config = config or {}
        self.rank_by = rank_by
        self.state_filter = state_filter
        self.days_threshold = days_threshold
        self.score_threshold = score_threshold
        
        logger.info(
            f"RankingEngine initialized: rank_by={rank_by}, "
            f"state_filter={state_filter}, days_threshold={days_threshold}"
        )
    
    def rank_batch(
        self,
        enriched_data: Dict[str, pd.DataFrame],
        top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Rank all symbols based on their latest crossover signals.
        
        Args:
            enriched_data: {symbol: DataFrame} with crossover columns
            top_n: Return only top N symbols (None = all)
        
        Returns:
            DataFrame with columns: [Rank, Symbol, State, Days_Since, Score, Close, 
                                     SMA_Fast, SMA_Slow, Last_Crossover_Type]
        
        Raises:
            ValueError: If enriched_data is empty
            KeyError: If required columns missing from DataFrame
        """
        if not enriched_data:
            raise ValueError("enriched_data is empty")
        
        try:
            # ====================================================
            # STEP 1: Extract latest row for each symbol
            # ====================================================
            rankings = []
            
            for symbol, df in enriched_data.items():
                if df.empty:
                    logger.warning(f"Skipping {symbol}: empty DataFrame")
                    continue
                
                try:
                    rank_dict = self.rank_symbol(symbol, df)
                    rankings.append(rank_dict)
                except KeyError as e:
                    logger.error(f"Missing column in {symbol}: {e}")
                    raise KeyError(f"Missing required column in {symbol}: {e}")
            
            if not rankings:
                raise ValueError("No valid rankings extracted from enriched_data")
            
            rankings_df = pd.DataFrame(rankings)
            logger.info(f"Extracted rankings for {len(rankings_df)} symbols")
            
            # ====================================================
            # STEP 2: Apply state filter
            # ====================================================
            if self.state_filter == 'BULLISH':
                rankings_df = rankings_df[rankings_df['State'] == 'BULLISH']
                logger.info(f"Filtered to {len(rankings_df)} BULLISH signals")
            
            elif self.state_filter == 'BEARISH':
                rankings_df = rankings_df[rankings_df['State'] == 'BEARISH']
                logger.info(f"Filtered to {len(rankings_df)} BEARISH signals")
            
            # ====================================================
            # STEP 3: Apply days threshold
            # ====================================================
            if self.days_threshold > 0:
                initial_count = len(rankings_df)
                rankings_df = rankings_df[
                    rankings_df['Days_Since'] <= self.days_threshold
                ]
                filtered_count = initial_count - len(rankings_df)
                logger.info(
                    f"Removed {filtered_count} signals older than "
                    f"{self.days_threshold} days"
                )
            
            # ====================================================
            # STEP 4: Apply score threshold
            # ====================================================
            if self.score_threshold > 0:
                initial_count = len(rankings_df)
                rankings_df = rankings_df[
                    rankings_df['Score'] >= self.score_threshold
                ]
                filtered_count = initial_count - len(rankings_df)
                logger.info(
                    f"Removed {filtered_count} signals with score < "
                    f"{self.score_threshold}"
                )
            
            # ====================================================
            # STEP 5: Sort by ranking strategy
            # ====================================================
            ranked_df = self._sort_by_strategy(rankings_df)
            
            # ====================================================
            # STEP 6: Add rank column
            # ====================================================
            ranked_df.insert(0, 'Rank', range(1, len(ranked_df) + 1))
            
            # ====================================================
            # STEP 7: Limit to top N if specified
            # ====================================================
            if top_n is not None and len(ranked_df) > top_n:
                ranked_df = ranked_df.head(top_n)
                logger.info(f"Limited results to top {top_n} signals")
            
            logger.info(f"Ranking complete: {len(ranked_df)} stocks ranked")
            return ranked_df
        
        except Exception as e:
            logger.error(f"Error in rank_batch: {e}")
            raise
    
    def rank_symbol(self, symbol: str, df: pd.DataFrame) -> dict:
        """
        Extract ranking information for a single symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE')
            df: DataFrame with crossover columns
        
        Returns:
            dict with keys: Symbol, Close, State, Days_Since, Score, 
                           Last_Crossover_Type, SMA_Fast, SMA_Slow
        
        Raises:
            KeyError: If required columns missing
        """
        # Get latest row
        latest = df.iloc[-1]
        
        # Required columns
        required_cols = [
            'Close', 'Crossover_State', 'Days_Since_Crossover',
            'Crossover_Score', 'Last_Crossover_Type'
        ]
        
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"Missing required column: {col}")
        
        # Extract SMA columns (dynamically detect SMA_Fast and SMA_Slow)
        sma_cols = [col for col in df.columns if col.startswith('SMA_')]
        sma_values = {}
        for col in sma_cols:
            sma_values[col] = latest[col]
        
        # Build ranking dict
        rank_dict = {
            'Symbol': symbol,
            'Close': float(latest['Close']),
            'State': latest['Crossover_State'],
            'Days_Since': float(latest['Days_Since_Crossover']),
            'Score': float(latest['Crossover_Score']),
            'Last_Crossover_Type': latest['Last_Crossover_Type'],
        }
        
        # Add SMA values
        rank_dict.update(sma_values)
        
        return rank_dict
    
    def _sort_by_strategy(self, rankings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort rankings based on configured strategy.
        
        Strategies:
        - 'days_since': Primary sort by Days_Since (ascending), Score (descending)
        - 'score': Primary sort by Score (descending), Days_Since (ascending)
        - 'hybrid': Weighted combination of both
        
        Args:
            rankings_df: DataFrame with ranking data
        
        Returns:
            Sorted DataFrame
        """
        if self.rank_by == 'days_since':
            # Fresher signals first (fewer days = higher priority)
            # Tie-breaker: stronger signals first (higher score)
            sorted_df = rankings_df.sort_values(
                by=['State', 'Days_Since', 'Score'],
                ascending=[False, True, False]  # BULLISH first, then fresher, then stronger
            )
        
        elif self.rank_by == 'score':
            # Stronger signals first (higher score = higher priority)
            # Tie-breaker: fresher signals first (fewer days)
            sorted_df = rankings_df.sort_values(
                by=['State', 'Score', 'Days_Since'],
                ascending=[False, False, True]  # BULLISH first, then stronger, then fresher
            )
        
        elif self.rank_by == 'hybrid':
            # Composite score combining both factors
            rankings_df = rankings_df.copy()
            rankings_df['Composite_Score'] = rankings_df.apply(
                lambda row: self.calculate_composite_score(row), axis=1
            )
            sorted_df = rankings_df.sort_values(
                by=['State', 'Composite_Score'],
                ascending=[False, False]  # BULLISH first, then higher composite
            )
        
        else:
            logger.warning(f"Unknown rank_by={self.rank_by}, using 'days_since'")
            sorted_df = rankings_df.sort_values(
                by=['State', 'Days_Since', 'Score'],
                ascending=[False, True, False]
            )
        
        return sorted_df.reset_index(drop=True)
    
    def calculate_composite_score(
        self,
        row: pd.Series,
        days_weight: float = 0.6,
        score_weight: float = 0.4
    ) -> float:
        """
        Calculate weighted composite score combining recency and strength.
        
        Formula:
            composite = (normalized_days * days_weight) + (score * score_weight)
        
        Args:
            row: Series with Days_Since and Score columns
            days_weight: Weight for recency component (0-1)
            score_weight: Weight for signal strength component (0-1)
        
        Returns:
            float: Composite score (higher = better)
        """
        # Normalize days: convert to 0-100 scale (fewer days = higher score)
        # Use 100 as reference: 0 days = 100, 100 days = 0
        days_component = max(0, 100 - row['Days_Since'])
        
        # Score component is already 0-100
        score_component = row['Score']
        
        # Composite: weighted average
        composite = (days_component * days_weight) + (score_component * score_weight)
        
        return composite
    
    def get_top_signals(
        self,
        ranked_df: pd.DataFrame,
        top_n: int = 10,
        state: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get top N signals, optionally filtered by state.
        
        Args:
            ranked_df: Already ranked DataFrame (from rank_batch)
            top_n: Number of top signals to return
            state: Filter by state ('BULLISH', 'BEARISH', or None)
        
        Returns:
            DataFrame with top N signals
        """
        result = ranked_df.copy()
        
        if state is not None:
            result = result[result['State'] == state]
        
        return result.head(top_n)
    
    def get_summary_stats(self, ranked_df: pd.DataFrame) -> dict:
        """
        Get summary statistics from ranked results.
        
        Args:
            ranked_df: Ranked DataFrame
        
        Returns:
            dict with statistics:
            - total_symbols: Total symbols ranked
            - bullish_count: Number of BULLISH signals
            - bearish_count: Number of BEARISH signals
            - avg_days: Average days since crossover
            - avg_score: Average crossover score
            - top_signal: Best ranking signal info
        """
        if ranked_df.empty:
            return {
                'total_symbols': 0,
                'bullish_count': 0,
                'bearish_count': 0,
                'avg_days': 0,
                'avg_score': 0,
                'top_signal': None
            }
        
        bullish_df = ranked_df[ranked_df['State'] == 'BULLISH']
        bearish_df = ranked_df[ranked_df['State'] == 'BEARISH']
        
        top_signal = None
        if not ranked_df.empty:
            top_row = ranked_df.iloc[0]
            top_signal = {
                'symbol': top_row['Symbol'],
                'state': top_row['State'],
                'days_since': top_row['Days_Since'],
                'score': top_row['Score']
            }
        
        stats = {
            'total_symbols': len(ranked_df),
            'bullish_count': len(bullish_df),
            'bearish_count': len(bearish_df),
            'avg_days': float(ranked_df['Days_Since'].mean()) if not ranked_df.empty else 0,
            'avg_score': float(ranked_df['Score'].mean()) if not ranked_df.empty else 0,
            'top_signal': top_signal
        }
        
        return stats


# =================================================================
# Utility Functions
# =================================================================

def create_ranking_report(
    ranked_df: pd.DataFrame,
    output_cols: Optional[List[str]] = None
) -> str:
    """
    Create a formatted text report of rankings.
    
    Args:
        ranked_df: Ranked DataFrame from rank_batch()
        output_cols: Columns to include (None = defaults)
    
    Returns:
        Formatted string report
    """
    if output_cols is None:
        output_cols = ['Rank', 'Symbol', 'State', 'Days_Since', 'Score', 'Close']
    
    # Filter to available columns
    available_cols = [col for col in output_cols if col in ranked_df.columns]
    
    # Format for display
    display_df = ranked_df[available_cols].copy()
    
    # Format numeric columns
    for col in ['Days_Since', 'Score', 'Close']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")
    
    # Convert to string
    report = display_df.to_string(index=False)
    
    return report
