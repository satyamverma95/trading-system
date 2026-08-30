# =================================================================
# data/fetcher.py
# Generic Market Data Fetcher & Broker Wrapper
# The single unified entry point for all data queries across the system
# =================================================================

import os
from datetime import datetime, date
from typing import Optional, Union, List, Dict
import pandas as pd

from utils.helpers import load_config
from utils.logger import get_logger
from data.storage import (
    save_market_data,
    load_market_data,
    resample_ohlcv,
    get_storage_path
)

logger = get_logger(__name__)


class MarketDataFetcher:
    """
    Unified market data service that routes requests to the configured provider
    (Zerodha or yfinance), caches to Parquet in data/raw/, and supports on-the-fly resampling.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self._zerodha_provider = None
        self._yfinance_provider = None

    def _get_provider_name(self, override: Optional[str] = None) -> str:
        if override:
            return override.lower()
        return self.config.get("data_provider", "yfinance").lower()

    @property
    def zerodha(self):
        if self._zerodha_provider is None:
            from providers.zerodha_provider import ZerodhaProvider
            self._zerodha_provider = ZerodhaProvider(self.config)
        return self._zerodha_provider

    @property
    def yfinance(self):
        if self._yfinance_provider is None:
            from providers.yfinance_provider import YFinanceProvider
            self._yfinance_provider = YFinanceProvider(self.config)
        return self._yfinance_provider

    def _infer_metadata(self, symbol: str, exchange: Optional[str], instrument_type: Optional[str]) -> Dict[str, str]:
        sym = symbol.strip()
        ex = exchange.upper() if exchange else "NSE"
        it = instrument_type.lower() if instrument_type else "equity"

        # Check for known index patterns
        if sym.startswith("^") or sym.upper() in ["NIFTY", "NIFTY 50", "NIFTY50", "BANKNIFTY", "NIFTY BANK", "SENSEX"]:
            it = "indices"

        if sym.endswith(".BO") or ex == "BSE":
            ex = "BSE"
            sym = sym[:-3] if sym.endswith(".BO") else sym
        elif sym.endswith(".NS"):
            ex = "NSE"
            sym = sym[:-3]

        return {"symbol": sym, "exchange": ex, "instrument_type": it}

    def fetch_historical(
        self,
        symbol: str,
        interval: str = "1d",
        period: Optional[str] = None,
        start: Optional[Union[str, date, datetime]] = None,
        end: Optional[Union[str, date, datetime]] = None,
        exchange: Optional[str] = None,
        instrument_type: Optional[str] = None,
        provider: Optional[str] = None,
        use_cache: bool = True,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch OHLCV DataFrame for a symbol with automatic Parquet caching and resampling.
        """
        provider_name = self._get_provider_name(provider)
        meta = self._infer_metadata(symbol, exchange, instrument_type)
        sym = meta["symbol"]
        ex = meta["exchange"]
        it = meta["instrument_type"]

        clean_interval = interval.lower()
        is_intraday = clean_interval not in ["1d", "day", "daily", "1w", "1wk"]
        storage_tf = "1m" if is_intraday else "1d"

        # 1. Check local Parquet Cache
        if use_cache and not force_refresh:
            cached_df = load_market_data(
                timeframe=storage_tf,
                exchange=ex,
                instrument_type=it,
                symbol=sym,
                start=start,
                end=end
            )
            if cached_df is not None and not cached_df.empty:
                logger.info("Loaded %s [%s] from local parquet cache (%d rows)", sym, interval, len(cached_df))
                if is_intraday and clean_interval not in ["1m", "1min", "minute"]:
                    return resample_ohlcv(cached_df, clean_interval)
                return cached_df

        # 2. Fetch fresh data from provider
        logger.info("Fetching %s from provider '%s' (interval=%s)...", sym, provider_name, clean_interval)

        if provider_name == "zerodha":
            # For Zerodha intraday, fetch base 1m for storage or requested interval
            fetch_intv = "1m" if is_intraday else "1d"
            df = self.zerodha.get_historical_data(
                symbol=sym,
                period=period,
                start=start,
                end=end,
                interval=fetch_intv,
                exchange=ex,
                instrument_type=it
            )
        else:
            # yfinance fallback
            yf_symbol = sym
            if it == "indices":
                yf_symbol = "^NSEI" if "NIFTY" in sym.upper() else ("^BSESN" if "SENSEX" in sym.upper() else sym)
            elif not yf_symbol.endswith(".NS") and not yf_symbol.endswith(".BO"):
                yf_symbol = f"{sym}.NS" if ex == "NSE" else f"{sym}.BO"

            df = self.yfinance.get_historical_data(
                symbol=yf_symbol,
                period=period,
                start=str(start) if start else None,
                end=str(end) if end else None,
                interval=clean_interval if not is_intraday else "1m"
            )

        if df is None or df.empty:
            logger.warning("No data returned for %s", sym)
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        # 3. Save / Upsert to Parquet Storage
        if use_cache:
            save_market_data(
                df=df,
                timeframe=storage_tf,
                exchange=ex,
                instrument_type=it,
                symbol=sym,
                merge=True
            )

        # 4. Resample if requested higher intraday timeframe
        if is_intraday and clean_interval not in ["1m", "1min", "minute"]:
            df = resample_ohlcv(df, clean_interval)

        return df

    def fetch_quote(self, symbol: str, provider: Optional[str] = None) -> dict:
        """Fetch live quote for a symbol."""
        provider_name = self._get_provider_name(provider)
        if provider_name == "zerodha":
            return self.zerodha.get_quote(symbol)
        else:
            return self.yfinance.get_quote(symbol)


# =================================================================
# CONVENIENCE TOP-LEVEL FUNCTIONS
# =================================================================

_default_fetcher = None

def _get_fetcher() -> MarketDataFetcher:
    global _default_fetcher
    if _default_fetcher is None:
        _default_fetcher = MarketDataFetcher()
    return _default_fetcher


def get_data(
    symbol: str,
    interval: str = "1d",
    period: Optional[str] = None,
    start: Optional[Union[str, date, datetime]] = None,
    end: Optional[Union[str, date, datetime]] = None,
    exchange: Optional[str] = None,
    instrument_type: Optional[str] = None,
    provider: Optional[str] = None,
    use_cache: bool = True,
    force_refresh: bool = False,
    quote: bool = False
) -> Union[pd.DataFrame, dict]:
    """
    Universal data fetching function.
    
    Examples:
        # 1. Fetch Daily Candles with automatic parquet caching
        df = get_data("RELIANCE", interval="1d", period="1y")

        # 2. Fetch 5-Minute Intraday Candles
        df = get_data("TCS", interval="5m", start="2024-01-01", end="2024-03-01")

        # 3. Fetch Index Data
        df = get_data("NIFTY 50", interval="1d", period="6mo")

        # 4. Fetch Live Quote
        q = get_data("INFY", quote=True)
    """
    fetcher = _get_fetcher()
    if quote:
        return fetcher.fetch_quote(symbol, provider=provider)
    return fetcher.fetch_historical(
        symbol=symbol,
        interval=interval,
        period=period,
        start=start,
        end=end,
        exchange=exchange,
        instrument_type=instrument_type,
        provider=provider,
        use_cache=use_cache,
        force_refresh=force_refresh
    )


def get_bulk_data(
    symbols: List[str],
    interval: str = "1d",
    period: Optional[str] = None,
    start: Optional[Union[str, date, datetime]] = None,
    end: Optional[Union[str, date, datetime]] = None,
    provider: Optional[str] = None,
    use_cache: bool = True
) -> Dict[str, pd.DataFrame]:
    """Fetch historical data for a list of symbols."""
    results = {}
    for sym in symbols:
        try:
            results[sym] = get_data(
                symbol=sym,
                interval=interval,
                period=period,
                start=start,
                end=end,
                provider=provider,
                use_cache=use_cache
            )
        except Exception as e:
            logger.error(f"Error fetching {sym}: {e}")
    return results