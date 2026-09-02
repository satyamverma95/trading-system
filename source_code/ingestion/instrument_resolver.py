# =================================================================
# source_code/ingestion/instrument_resolver.py
# LAYER 2: INSTRUMENT MAPPING - Zerodha Token Resolution
# Maps trading symbols to Kite Connect instrument tokens
# =================================================================

import os
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import pandas as pd
import logging
from datetime import datetime
import pytz

from kiteconnect import KiteConnect

from source_code.common.config_loader import load_config
from source_code.common.path_resolver import resolve_path, get_project_root, ensure_dir
from source_code.ingestion.auth.session_manager import get_authenticated_kite

logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")


class InstrumentResolver:
    """
    Maps NSE stock symbols to Zerodha Kite Connect instrument tokens.
    
    Functionality:
    - Fetches instrument master from Kite (on demand or from cache)
    - Caches instrument master daily to avoid repeated API calls
    - Resolves a symbol to its instrument token
    - Batch resolve multiple symbols with error handling
    - Provides instrument details (name, lot_size, exchange, segment)
    
    Example:
        resolver = InstrumentResolver()
        token_map = resolver.resolve_symbols(["RELIANCE", "HDFCBANK", "INFY"])
        # Returns: {"RELIANCE": 738561, "HDFCBANK": 345089, "INFY": 408065}
    """

    def __init__(self, config: Optional[dict] = None, kite_client: Optional[KiteConnect] = None):
        """
        Initialize the resolver.
        
        Args:
            config: Config dict. If None, loads from settings.yaml
            kite_client: KiteConnect instance. If None, authenticates fresh.
        """
        self.config = config or load_config()
        self._kite = kite_client
        self._instruments_df: Optional[pd.DataFrame] = None
        self._cache_path: Optional[Path] = None

    @property
    def kite(self) -> KiteConnect:
        """Lazy authentication for Kite client."""
        if self._kite is None:
            self._kite = get_authenticated_kite()
            logger.info("Authenticated with Kite Connect")
        return self._kite

    def _get_cache_path(self) -> Path:
        """Get path to instrument master cache file."""
        if self._cache_path is None:
            root = get_project_root()
            cache_dir = root / "data" / "cache"
            ensure_dir(cache_dir)
            
            today_str = datetime.now(IST).strftime("%Y-%m-%d")
            self._cache_path = cache_dir / f"instruments_{today_str}.parquet"
        
        return self._cache_path

    def _load_from_cache(self) -> Optional[pd.DataFrame]:
        """
        Load instrument master from daily cache if available.
        Returns None if cache doesn't exist or is stale.
        """
        cache_path = self._get_cache_path()
        
        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                logger.info(f"Loaded {len(df)} instruments from cache: {cache_path}")
                return df
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_path}: {e}")
                return None
        
        return None

    def _fetch_from_kite(self) -> pd.DataFrame:
        """
        Fetch instrument master from Kite Connect API.
        This is slower but always current.
        """
        logger.info("Fetching instrument master from Kite Connect...")
        
        try:
            instruments = self.kite.instruments()
            
            if not instruments:
                raise ValueError("Kite returned empty instrument list")
            
            df = pd.DataFrame(instruments)
            logger.info(f"Fetched {len(df)} instruments from Kite")
            
            # Cache for today
            self._save_to_cache(df)
            
            return df
        
        except Exception as e:
            logger.error(f"Failed to fetch instruments from Kite: {e}")
            raise

    def _save_to_cache(self, df: pd.DataFrame) -> None:
        """Save instrument master to cache."""
        try:
            cache_path = self._get_cache_path()
            # Instrument metadata includes mixed expiry values; normalize before Parquet.
            cache_df = df.copy()
            if "expiry" in cache_df.columns:
                cache_df["expiry"] = cache_df["expiry"].astype(str)
            cache_df.to_parquet(cache_path, index=False)
            logger.debug(f"Saved {len(df)} instruments to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache instruments: {e}")

    def get_instruments(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get instrument master (cache-first strategy).
        
        Args:
            force_refresh: If True, always fetch from Kite API
            
        Returns:
            DataFrame with columns: [tradingsymbol, instrument_token, exchange, ...]
        """
        if self._instruments_df is not None and not force_refresh:
            return self._instruments_df

        # Try cache first
        if not force_refresh:
            df = self._load_from_cache()
            if df is not None:
                self._instruments_df = df
                return df

        # Fetch from Kite
        df = self._fetch_from_kite()
        self._instruments_df = df
        return df

    def resolve_symbol(self, symbol: str, exchange: str = "NSE") -> Optional[int]:
        """
        Resolve a single symbol to instrument token.
        
        Args:
            symbol: Trading symbol (e.g., "RELIANCE", "HDFCBANK")
            exchange: Exchange code (default "NSE")
            
        Returns:
            Instrument token (int) or None if not found
        """
        instruments = self.get_instruments()
        
        # Normalize symbol
        symbol = symbol.strip().upper()
        
        # Filter by symbol and exchange
        matches = instruments[
            (instruments['tradingsymbol'].str.upper() == symbol) &
            (instruments['exchange'] == exchange)
        ]

        # A cache can be incomplete or stale; refresh once before reporting a miss.
        if len(matches) == 0 and self._instruments_df is not None:
            logger.info(f"{symbol} not found in cache; refreshing instrument master")
            instruments = self.get_instruments(force_refresh=True)
            matches = instruments[
                (instruments['tradingsymbol'].str.upper() == symbol) &
                (instruments['exchange'] == exchange)
            ]
        
        if len(matches) == 0:
            logger.warning(f"No instrument found for {symbol} in {exchange}")
            return None
        
        if len(matches) > 1:
            logger.warning(
                f"Multiple instruments found for {symbol} in {exchange}. "
                f"Using first match."
            )
        
        token = int(matches.iloc[0]['instrument_token'])
        return token

    def resolve_symbols(
        self, 
        symbols: List[str], 
        exchange: str = "NSE",
        skip_missing: bool = False
    ) -> Dict[str, int]:
        """
        Resolve multiple symbols to tokens (batch operation).
        
        Args:
            symbols: List of trading symbols
            exchange: Exchange code (default "NSE")
            skip_missing: If True, skip symbols not found. 
                         If False, raise error on missing symbols.
            
        Returns:
            Dict mapping symbol -> token
            
        Raises:
            ValueError: If skip_missing=False and symbols are missing
        """
        token_map = {}
        missing = []
        
        for symbol in symbols:
            token = self.resolve_symbol(symbol, exchange=exchange)
            
            if token is None:
                missing.append(symbol)
                if not skip_missing:
                    continue
            else:
                token_map[symbol] = token
        
        if missing:
            if skip_missing:
                logger.warning(
                    f"Skipped {len(missing)} missing symbols: {missing}"
                )
            else:
                raise ValueError(
                    f"Failed to resolve {len(missing)} symbols: {missing}\n"
                    f"Set skip_missing=True to ignore."
                )
        
        logger.info(
            f"Resolved {len(token_map)}/{len(symbols)} symbols to tokens"
        )
        
        return token_map

    def get_instrument_info(self, symbol: str, exchange: str = "NSE") -> Optional[dict]:
        """
        Get detailed instrument information.
        
        Returns:
            Dict with columns: tradingsymbol, instrument_token, name, exchange, 
                              segment, lot_size, tick_size, expiry, etc.
        """
        instruments = self.get_instruments()
        
        symbol = symbol.strip().upper()
        matches = instruments[
            (instruments['tradingsymbol'].str.upper() == symbol) &
            (instruments['exchange'] == exchange)
        ]
        
        if len(matches) == 0:
            return None
        
        return matches.iloc[0].to_dict()

    def batch_get_info(
        self, 
        symbols: List[str], 
        exchange: str = "NSE"
    ) -> pd.DataFrame:
        """
        Get instrument info for multiple symbols.
        
        Returns:
            DataFrame filtered to requested symbols.
        """
        instruments = self.get_instruments()
        
        symbols_upper = [s.strip().upper() for s in symbols]
        
        filtered = instruments[
            (instruments['tradingsymbol'].str.upper().isin(symbols_upper)) &
            (instruments['exchange'] == exchange)
        ]
        
        return filtered


def resolve_symbols_to_tokens(
    symbols: List[str], 
    config: Optional[dict] = None
) -> Dict[str, int]:
    """
    Convenience function: resolve symbols to tokens in one call.
    
    Usage:
        token_map = resolve_symbols_to_tokens(["RELIANCE", "HDFCBANK", "INFY"])
        # Returns: {"RELIANCE": 738561, "HDFCBANK": 345089, "INFY": 408065}
    """
    resolver = InstrumentResolver(config=config)
    return resolver.resolve_symbols(symbols, skip_missing=False)
