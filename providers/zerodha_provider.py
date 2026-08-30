# =================================================================
# providers/zerodha_provider.py
# Zerodha Kite Connect implementation of BaseDataProvider
# =================================================================

import os
import time
from datetime import datetime, date, timedelta
import pytz
from typing import List, Optional, Dict, Tuple, Union
import pandas as pd
from kiteconnect import KiteConnect

from providers.base import BaseDataProvider
from auth.session_manager import get_authenticated_kite, load_session
from utils.logger import get_logger

logger = get_logger(__name__)

CACHE_DIR = "data/instruments_cache"
IST = pytz.timezone("Asia/Kolkata")

# Timeframe mapping to Kite interval strings and max chunk size in days
INTERVAL_CONFIG = {
    "1m": {"kite_interval": "minute", "chunk_days": 55},
    "minute": {"kite_interval": "minute", "chunk_days": 55},
    "2m": {"kite_interval": "minute", "chunk_days": 55},       # fetched as 1m then resampled
    "3m": {"kite_interval": "3minute", "chunk_days": 90},
    "3minute": {"kite_interval": "3minute", "chunk_days": 90},
    "5m": {"kite_interval": "5minute", "chunk_days": 90},
    "5minute": {"kite_interval": "5minute", "chunk_days": 90},
    "10m": {"kite_interval": "5minute", "chunk_days": 90},
    "15m": {"kite_interval": "15minute", "chunk_days": 90},
    "15minute": {"kite_interval": "15minute", "chunk_days": 90},
    "30m": {"kite_interval": "30minute", "chunk_days": 90},
    "30minute": {"kite_interval": "30minute", "chunk_days": 90},
    "60m": {"kite_interval": "60minute", "chunk_days": 365},
    "1h": {"kite_interval": "60minute", "chunk_days": 365},
    "60minute": {"kite_interval": "60minute", "chunk_days": 365},
    "1d": {"kite_interval": "day", "chunk_days": 1800},
    "day": {"kite_interval": "day", "chunk_days": 1800},
}

# Common Index aliases to Zerodha standard names
INDEX_ALIASES = {
    "^NSEI": "NIFTY 50",
    "NIFTY": "NIFTY 50",
    "NIFTY50": "NIFTY 50",
    "NIFTY 50": "NIFTY 50",
    "^NSEBANK": "NIFTY BANK",
    "BANKNIFTY": "NIFTY BANK",
    "NIFTY BANK": "NIFTY BANK",
    "^BSESN": "SENSEX",
    "SENSEX": "SENSEX",
    "BSE SENSEX": "SENSEX",
    "FINNIFTY": "NIFTY FIN SERVICE",
    "NIFTY FIN SERVICE": "NIFTY FIN SERVICE",
    "MIDCPNIFTY": "NIFTY MID SELECT",
    "NIFTY MID SELECT": "NIFTY MID SELECT",
}


class ZerodhaProvider(BaseDataProvider):
    """
    Zerodha Kite Connect Data Provider.
    Implements BaseDataProvider contract for Historical and Live data.
    """

    def __init__(self, config: dict, kite_client: Optional[KiteConnect] = None):
        super().__init__(config)
        self._kite = kite_client
        self._instruments_df: Optional[pd.DataFrame] = None
        self.default_period = config.get("historical", {}).get("default_period", "2y")
        self.default_interval = config.get("historical", {}).get("default_interval", "1d")

    @property
    def kite(self) -> KiteConnect:
        """Lazy authentication for Kite client."""
        if self._kite is None:
            self._kite = get_authenticated_kite()
        return self._kite

    # ----------------------------------------------------------
    # INSTRUMENT RESOLUTION
    # ----------------------------------------------------------

    def get_instruments(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Download and cache Kite instrument master daily.
        """
        if self._instruments_df is not None and not force_refresh:
            return self._instruments_df

        today_str = datetime.now(IST).strftime("%Y-%m-%d")
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(CACHE_DIR, f"instruments_{today_str}.parquet")

        if os.path.exists(cache_path) and not force_refresh:
            try:
                self._instruments_df = pd.read_parquet(cache_path)
                return self._instruments_df
            except Exception as e:
                logger.warning(f"Could not load cached instruments: {e}")

        logger.info("Downloading fresh instrument master from Zerodha...")
        instruments_list = self.kite.instruments()
        df = pd.DataFrame(instruments_list)

        # Save cache
        try:
            df.to_parquet(cache_path, index=False)
            logger.info("Saved %d instruments to %s", len(df), cache_path)
        except Exception as e:
            logger.warning(f"Could not cache instruments to parquet: {e}")

        self._instruments_df = df
        return self._instruments_df

    def resolve_instrument(
        self,
        symbol: str,
        exchange: Optional[str] = None,
        instrument_type: Optional[str] = None
    ) -> Dict:
        """
        Resolve a symbol name into Zerodha instrument metadata (instrument_token, exchange, etc.).
        """
        df = self.get_instruments()
        sym = symbol.strip().upper()

        # Clean .NS / .BO extensions if present
        if sym.endswith(".NS"):
            sym = sym[:-3]
            exchange = exchange or "NSE"
        elif sym.endswith(".BO"):
            sym = sym[:-3]
            exchange = exchange or "BSE"

        # Check Index Aliases
        if sym in INDEX_ALIASES:
            idx_name = INDEX_ALIASES[sym]
            # Try NSE first, then BSE
            matches = df[(df["name"] == idx_name) & (df["segment"] == "INDICES")]
            if matches.empty:
                matches = df[(df["tradingsymbol"] == idx_name) & (df["segment"] == "INDICES")]
            if matches.empty:
                matches = df[df["tradingsymbol"] == idx_name]

            if not matches.empty:
                row = matches.iloc[0]
                return {
                    "instrument_token": int(row["instrument_token"]),
                    "tradingsymbol": str(row["tradingsymbol"]),
                    "exchange": str(row["exchange"]),
                    "segment": str(row.get("segment", "INDICES")),
                    "instrument_type": "indices",
                    "name": str(row.get("name", sym)),
                }

        # Filter by exchange if provided
        filtered = df
        if exchange:
            filtered = filtered[filtered["exchange"] == exchange.upper()]

        # Filter by instrument type (EQ / FUT / CE / PE)
        if instrument_type and instrument_type.lower() == "equity":
            filtered = filtered[filtered["instrument_type"] == "EQ"]

        # Exact match on tradingsymbol
        matches = filtered[filtered["tradingsymbol"] == sym]

        # If empty, try matching on name for equities
        if matches.empty:
            matches = filtered[(filtered["name"] == sym) & (filtered["instrument_type"] == "EQ")]

        # Default fallback to NSE EQ if multiple found
        if not matches.empty:
            if len(matches) > 1:
                nse_matches = matches[matches["exchange"] == "NSE"]
                row = nse_matches.iloc[0] if not nse_matches.empty else matches.iloc[0]
            else:
                row = matches.iloc[0]

            itype = "equity"
            if row.get("segment") == "INDICES" or row.get("instrument_type") == "INDEX":
                itype = "indices"
            elif row.get("segment") in ["NFO-FUT", "NFO-OPT", "BFO-FUT", "BFO-OPT"]:
                itype = "derivatives"

            return {
                "instrument_token": int(row["instrument_token"]),
                "tradingsymbol": str(row["tradingsymbol"]),
                "exchange": str(row["exchange"]),
                "segment": str(row.get("segment", "NSE")),
                "instrument_type": itype,
                "name": str(row.get("name", sym)),
            }

        raise ValueError(f"Could not resolve instrument token for symbol '{symbol}' on exchange '{exchange or 'ANY'}'.")

    # ----------------------------------------------------------
    # HISTORICAL DATA
    # ----------------------------------------------------------

    def get_historical_data(
        self,
        symbol: str,
        period: Optional[str] = None,
        start: Optional[Union[str, date, datetime]] = None,
        end: Optional[Union[str, date, datetime]] = None,
        interval: str = "1d",
        exchange: Optional[str] = None,
        instrument_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV historical data for a symbol from Zerodha Kite.
        Handles date range chunking and rate limits automatically.
        """
        meta = self.resolve_instrument(symbol, exchange=exchange, instrument_type=instrument_type)
        instrument_token = meta["instrument_token"]

        # Parse date range
        now = datetime.now(IST)
        if end is None:
            end_dt = now
        elif isinstance(end, str):
            end_dt = IST.localize(datetime.fromisoformat(end)) if "T" in end else IST.localize(datetime.strptime(end, "%Y-%m-%d").replace(hour=15, minute=30))
        elif isinstance(end, date) and not isinstance(end, datetime):
            end_dt = IST.localize(datetime.combine(end, datetime.min.time()).replace(hour=15, minute=30))
        else:
            end_dt = end if end.tzinfo else IST.localize(end)

        if start is None:
            period_str = period or self.default_period
            start_dt = self._calculate_start_date(end_dt, period_str)
        elif isinstance(start, str):
            start_dt = IST.localize(datetime.fromisoformat(start)) if "T" in start else IST.localize(datetime.strptime(start, "%Y-%m-%d"))
        elif isinstance(start, date) and not isinstance(start, datetime):
            start_dt = IST.localize(datetime.combine(start, datetime.min.time()))
        else:
            start_dt = start if start.tzinfo else IST.localize(start)

        # Determine Kite interval and chunk size
        intv_key = interval.lower()
        if intv_key not in INTERVAL_CONFIG:
            raise ValueError(f"Unsupported interval '{interval}'. Choose from: {list(INTERVAL_CONFIG.keys())}")

        kite_interval = INTERVAL_CONFIG[intv_key]["kite_interval"]
        chunk_days = INTERVAL_CONFIG[intv_key]["chunk_days"]

        # Fetch in chunks
        all_records = []
        cur_from = start_dt

        while cur_from < end_dt:
            cur_to = min(cur_from + timedelta(days=chunk_days), end_dt)
            logger.info("Fetching Kite data for %s (%s to %s, interval=%s)...", meta['tradingsymbol'], cur_from.date(), cur_to.date(), kite_interval)

            try:
                records = self.kite.historical_data(
                    instrument_token=instrument_token,
                    from_date=cur_from.strftime("%Y-%m-%d %H:%M:%S") if kite_interval != "day" else cur_from.strftime("%Y-%m-%d"),
                    to_date=cur_to.strftime("%Y-%m-%d %H:%M:%S") if kite_interval != "day" else cur_to.strftime("%Y-%m-%d"),
                    interval=kite_interval
                )
                if records:
                    all_records.extend(records)
            except Exception as e:
                logger.error(f"Error fetching historical chunk for {symbol}: {e}")
                raise

            cur_from = cur_to + timedelta(seconds=1)
            time.sleep(0.35)  # Rate limit safety (3 req/sec)

        if not all_records:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        # Convert to DataFrame
        df = pd.DataFrame(all_records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        # Standardize columns to Open, High, Low, Close, Volume
        df = df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "oi": "OI"
        })

        df = self.validate_dataframe(df, symbol)
        df["Symbol"] = meta["tradingsymbol"]
        df["Exchange"] = meta["exchange"]

        return df

    def get_bulk_historical_data(
        self,
        symbols: List[str],
        period: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for a list of symbols."""
        results = {}
        for sym in symbols:
            try:
                results[sym] = self.get_historical_data(sym, period=period, start=start, end=end, interval=interval)
            except Exception as e:
                logger.error(f"Failed to fetch {sym}: {e}")
        return results

    # ----------------------------------------------------------
    # QUOTES & LIVE DATA
    # ----------------------------------------------------------

    def get_quote(self, symbol: str) -> dict:
        """Fetch live quote for a single symbol."""
        meta = self.resolve_instrument(symbol)
        exchange_symbol = f"{meta['exchange']}:{meta['tradingsymbol']}"
        raw = self.kite.quote(exchange_symbol)
        q = raw.get(exchange_symbol, {})
        ohlc = q.get("ohlc", {})

        return {
            "symbol": meta["tradingsymbol"],
            "exchange": meta["exchange"],
            "instrument_token": meta["instrument_token"],
            "last_price": q.get("last_price"),
            "open": ohlc.get("open"),
            "high": ohlc.get("high"),
            "low": ohlc.get("low"),
            "close": ohlc.get("close"),
            "volume": q.get("volume"),
            "change_pct": q.get("net_change"),
            "timestamp": q.get("last_trade_time") or datetime.now(IST).isoformat(),
        }

    def get_bulk_quotes(self, symbols: List[str]) -> dict:
        """Fetch live quotes for multiple symbols in a single call."""
        resolved = [self.resolve_instrument(s) for s in symbols]
        exchange_symbols = [f"{m['exchange']}:{m['tradingsymbol']}" for m in resolved]
        raw = self.kite.quote(exchange_symbols)

        results = {}
        for m in resolved:
            k = f"{m['exchange']}:{m['tradingsymbol']}"
            q = raw.get(k, {})
            ohlc = q.get("ohlc", {})
            results[m["tradingsymbol"]] = {
                "symbol": m["tradingsymbol"],
                "exchange": m["exchange"],
                "last_price": q.get("last_price"),
                "open": ohlc.get("open"),
                "high": ohlc.get("high"),
                "low": ohlc.get("low"),
                "close": ohlc.get("close"),
                "volume": q.get("volume"),
                "timestamp": q.get("last_trade_time") or datetime.now(IST).isoformat(),
            }
        return results

    def get_instrument_info(self, symbol: str) -> dict:
        """Get metadata about an instrument."""
        return self.resolve_instrument(symbol)

    def get_options_chain(self, symbol: str, expiry: Optional[str] = None) -> pd.DataFrame:
        """Fetch derivative chain contracts."""
        df = self.get_instruments()
        sym = symbol.upper()
        derivatives = df[(df["name"] == sym) & (df["segment"].isin(["NFO-OPT", "NFO-FUT", "BFO-OPT", "BFO-FUT"]))]
        if expiry:
            derivatives = derivatives[derivatives["expiry"] == expiry]
        return derivatives

    # ----------------------------------------------------------
    # HELPERS
    # ----------------------------------------------------------

    def _calculate_start_date(self, end_dt: datetime, period: str) -> datetime:
        """Convert period strings ('1mo', '3mo', '1y', '2y', '5y') to start datetime."""
        p = period.lower()
        if p.endswith("d"):
            days = int(p[:-1])
            return end_dt - timedelta(days=days)
        elif p.endswith("mo") or p.endswith("m"):
            months = int(p.replace("mo", "").replace("m", ""))
            return end_dt - timedelta(days=months * 30)
        elif p.endswith("y"):
            years = int(p[:-1])
            return end_dt - timedelta(days=years * 365)
        return end_dt - timedelta(days=730)  # default 2y