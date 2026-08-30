# =================================================================
# providers/base.py
# Abstract DataProvider — all providers must implement this contract
# Swap yfinance → Zerodha by changing one line in settings.yaml
# =================================================================

from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd


class BaseDataProvider(ABC):
    """
    Abstract base class for all data providers.
    
    Any new provider (yfinance, Zerodha, Upstox etc.) must inherit
    from this class and implement all abstract methods.
    This ensures the rest of the system never needs to change
    when we swap data sources.
    """

    def __init__(self, config: dict):
        self.config = config

    # ----------------------------------------------------------
    # HISTORICAL DATA
    # ----------------------------------------------------------

    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        period: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a single symbol.

        Returns DataFrame with columns:
            Date, Open, High, Low, Close, Volume
        """
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
        """
        Fetch OHLCV data for multiple symbols.

        Returns dict: { symbol: DataFrame }
        """
        pass

    # ----------------------------------------------------------
    # LIVE / QUOTE DATA (Phase 2/3 - yfinance has limited support)
    # ----------------------------------------------------------

    @abstractmethod
    def get_quote(self, symbol: str) -> dict:
        """
        Get current market quote for a symbol.

        Returns dict with keys:
            symbol, last_price, open, high, low, close,
            volume, timestamp
        """
        pass

    @abstractmethod
    def get_bulk_quotes(self, symbols: List[str]) -> dict:
        """
        Get current quotes for multiple symbols.

        Returns dict: { symbol: quote_dict }
        """
        pass

    # ----------------------------------------------------------
    # INSTRUMENT INFO
    # ----------------------------------------------------------

    @abstractmethod
    def get_instrument_info(self, symbol: str) -> dict:
        """
        Get metadata about an instrument.

        Returns dict with keys:
            symbol, name, exchange, sector, industry,
            market_cap, pe_ratio etc.
        """
        pass

    # ----------------------------------------------------------
    # DERIVATIVES (Phase 2)
    # ----------------------------------------------------------

    @abstractmethod
    def get_options_chain(self, symbol: str, expiry: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch options chain for a symbol.
        Phase 2 feature — yfinance has partial support.

        Returns DataFrame with calls and puts.
        """
        pass

    # ----------------------------------------------------------
    # UTILITY
    # ----------------------------------------------------------

    def validate_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Common validation applied to all OHLCV DataFrames.
        Ensures consistent column names and types across providers.
        """
        if df is None or df.empty:
            raise ValueError(f"No data returned for symbol: {symbol}")

        # Standardise column names to title case
        df.columns = [col.capitalize() for col in df.columns]

        # Ensure required columns exist
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for {symbol}: {missing}")

        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Drop rows where Close is NaN
        df = df.dropna(subset=["Close"])

        # Sort by date ascending
        df = df.sort_index(ascending=True)

        return df

    def __repr__(self):
        return f"{self.__class__.__name__}(provider={self.__class__.__name__})"