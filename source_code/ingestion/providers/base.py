# =================================================================
# source_code/ingestion/providers/base.py
# Abstract DataProvider base class
# =================================================================

from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd


class BaseDataProvider(ABC):
    """Abstract base class for all data providers."""

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        period: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a single symbol."""
        pass

    @abstractmethod
    def get_bulk_historical_data(
        self,
        symbols: List[str],
        period: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d"
    ) -> dict:
        """Fetch OHLCV data for multiple symbols. Returns dict: {symbol: DataFrame}"""
        pass

    def validate_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate and clean OHLCV DataFrame."""
        if df.empty:
            return df
        
        # Ensure numeric columns
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Remove rows with NaN prices
        df = df.dropna(subset=["Close"])
        
        return df
