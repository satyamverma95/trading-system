# =================================================================
# source_code/ingestion/providers/zerodha_provider.py
# Zerodha Kite Connect implementation
# =================================================================

import logging
from typing import List, Optional, Dict
import pandas as pd

from source_code.ingestion.providers.base import BaseDataProvider

logger = logging.getLogger(__name__)


class ZerodhaProvider(BaseDataProvider):
    """
    Zerodha Kite Connect Data Provider.
    Requires active session with valid API credentials.
    """

    def __init__(self, config: dict, kite_client=None):
        super().__init__(config)
        self._kite = kite_client
        self.default_period = config.get("historical", {}).get("default_period", "2y")
        self.default_interval = config.get("historical", {}).get("default_interval", "1d")
        logger.info("ZerodhaProvider initialized")

    @property
    def kite(self):
        """Lazy authentication for Kite client."""
        if self._kite is None:
            try:
                from source_code.ingestion.auth.session_manager import get_authenticated_kite
                self._kite = get_authenticated_kite()
                logger.info("Authenticated with Kite Connect")
            except Exception as e:
                logger.error(f"Failed to authenticate with Zerodha: {e}")
                raise
        return self._kite

    def get_historical_data(
        self,
        symbol: str,
        period: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a single symbol from Zerodha."""
        
        try:
            logger.debug(f"Fetching {symbol} from Zerodha (interval={interval})")
            
            # Use Kite API to fetch data
            # This is a simplified version - full implementation in Source Code folder
            df = pd.DataFrame()
            logger.info(f"Fetched {len(df)} rows for {symbol} from Zerodha")
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching {symbol} from Zerodha: {e}")
            return pd.DataFrame()

    def get_bulk_historical_data(
        self,
        symbols: List[str],
        period: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV data for multiple symbols from Zerodha."""
        
        results = {}
        for symbol in symbols:
            try:
                df = self.get_historical_data(symbol, period, start, end, interval)
                if not df.empty:
                    results[symbol] = df
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
        
        return results
