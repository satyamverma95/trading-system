# =================================================================
# source_code/ingestion/providers/zerodha_provider.py
# Zerodha Kite Connect implementation
# =================================================================

import logging
from datetime import date, datetime, timedelta
import re
from typing import List, Optional, Dict
import pandas as pd

from source_code.ingestion.providers.base import BaseDataProvider
from source_code.ingestion.instrument_resolver import InstrumentResolver

logger = logging.getLogger(__name__)

KITE_INTERVALS = {
    "1m": "minute",
    "5m": "5minute",
    "15m": "15minute",
    "30m": "30minute",
    "1h": "60minute",
    "60m": "60minute",
    "1d": "day",
    "1wk": "week",
    "1mo": "month",
}


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
        self._token_resolver = None
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
        interval: str = "1d",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a single symbol from Zerodha."""

        try:
            if start is None:
                start = start_date
            if end is None:
                end = end_date

            from_date, to_date = self._resolve_date_range(period, start, end)
            if self._token_resolver is None:
                self._token_resolver = InstrumentResolver(self.config, kite_client=self.kite)
            instrument_token = self._token_resolver.resolve_symbol(symbol, exchange="NSE")
            if instrument_token is None:
                raise ValueError(f"No NSE instrument token found for {symbol}")

            logger.info(
                "Fetching %s from Zerodha (%s to %s, interval=%s)",
                symbol, from_date, to_date, interval,
            )
            kite_interval = KITE_INTERVALS.get(interval.lower(), interval)
            candles = self.kite.historical_data(
                instrument_token,
                from_date,
                to_date,
                kite_interval,
                continuous=False,
                oi=False,
            )
            df = pd.DataFrame(candles)
            if df.empty:
                return df

            df = df.rename(columns={
                "date": "Datetime",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            })
            if "Datetime" in df.columns:
                df["Datetime"] = pd.to_datetime(df["Datetime"])
                df = df.set_index("Datetime")
            df["Symbol"] = symbol
            df["Exchange"] = "NSE"
            logger.info(f"Fetched {len(df)} rows for {symbol} from Zerodha")
            return df
        
        except Exception as e:
            logger.error(f"Error fetching {symbol} from Zerodha: {e}")
            return pd.DataFrame()

    def _resolve_date_range(
        self,
        period: Optional[str],
        start: Optional[str],
        end: Optional[str],
    ) -> tuple[str, str]:
        """Convert relative periods or explicit dates to Kite API date strings."""
        end_date = datetime.strptime(end, "%Y-%m-%d").date() if end else date.today()
        if start:
            start_date = datetime.strptime(start, "%Y-%m-%d").date()
        else:
            match = re.fullmatch(r"(\d+)([dwmy])", (period or self.default_period).lower())
            if not match:
                raise ValueError("period must look like 5d, 2w, 3mo, or 1y")
            amount, unit = int(match.group(1)), match.group(2)
            days = {"d": 1, "w": 7, "m": 30, "y": 365}[unit]
            start_date = end_date - timedelta(days=amount * days)
        if start_date > end_date:
            raise ValueError("start date cannot be after end date")
        return start_date.isoformat(), end_date.isoformat()

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
