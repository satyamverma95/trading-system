# =================================================================
# source_code/ingestion/providers/yfinance_provider.py
# yfinance implementation - Free and reliable for testing
# =================================================================

import logging
from typing import List, Optional, Dict
import pandas as pd
import yfinance as yf

from source_code.ingestion.providers.base import BaseDataProvider

logger = logging.getLogger(__name__)


class YFinanceProvider(BaseDataProvider):
    """
    yfinance-based data provider for NSE stocks.
    
    Symbol convention:
        Stocks  → RELIANCE.NS, HDFCBANK.NS
        Indices → ^NSEI (Nifty 50)
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.default_period = config.get("historical", {}).get("default_period", "2y")
        self.default_interval = config.get("historical", {}).get("default_interval", "1d")

    def _normalize_symbol(self, symbol: str) -> str:
        """Convert trading symbol to yfinance format."""
        symbol = symbol.strip().upper()
        
        # If already has .NS or .BO, use as-is
        if symbol.endswith(".NS") or symbol.endswith(".BO"):
            return symbol
        
        # Add .NS for NSE stocks
        if not symbol.endswith((".NS", ".BO", "^")):
            return f"{symbol}.NS"
        
        return symbol

    def get_historical_data(
        self,
        symbol: str,
        period: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a single symbol."""
        
        try:
            yf_symbol = self._normalize_symbol(symbol)
            logger.debug(f"Fetching {symbol} as {yf_symbol} (interval={interval})")
            
            # Handle both start/end and start_date/end_date parameter names
            start = start or start_date
            end = end or end_date
            
            ticker = yf.Ticker(yf_symbol)
            
            if start and end:
                df = ticker.history(start=start, end=end, interval=interval)
            elif period:
                df = ticker.history(period=period or self.default_period, interval=interval)
            else:
                df = ticker.history(period=self.default_period, interval=interval)
            
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Standardize column names
            df = df.rename(columns={
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Volume": "Volume"
            })
            
            # Keep only OHLCV columns
            ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
            df = df[[col for col in ohlcv_cols if col in df.columns]]
            
            # Validate
            df = self.validate_dataframe(df, symbol)
            
            logger.info(f"Fetched {len(df)} rows for {symbol} ({yf_symbol})")
            return df
        
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def get_bulk_historical_data(
        self,
        symbols: List[str],
        period: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV data for multiple symbols."""
        
        results = {}
        for symbol in symbols:
            try:
                df = self.get_historical_data(symbol, period, start, end, interval)
                if not df.empty:
                    results[symbol] = df
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
        
        return results
