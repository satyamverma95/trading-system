# =================================================================
# source_code/ingestion/batch_fetcher.py
# LAYER 3: INGESTION - Batch Historical Candle Fetcher
# Fetches OHLCV candles for multiple symbols in parallel/sequential mode
# =================================================================

import logging
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import pandas as pd

from source_code.common.config_loader import load_config
from source_code.common.path_resolver import resolve_path, get_project_root, ensure_dir
from source_code.ingestion.providers.zerodha_provider import ZerodhaProvider
from source_code.ingestion.providers.yfinance_provider import YFinanceProvider
from source_code.ingestion.instrument_resolver import InstrumentResolver

logger = logging.getLogger(__name__)


class BatchCandleFetcher:
    """
    Batch fetcher for historical OHLCV candles across multiple symbols.
    
    Features:
    - Fetches multiple symbols sequentially or in parallel
    - Supports multiple time intervals (1d, 1h, 15m, 5m, 1m, etc.)
    - Handles missing symbols gracefully (continues without failing)
    - Caches results to avoid redundant API calls
    - Supports both Zerodha and yfinance providers
    - Returns Dict[symbol] = DataFrame with OHLCV data
    
    Example:
        fetcher = BatchCandleFetcher()
        symbols = ["RELIANCE", "HDFCBANK", "INFY"]
        data = fetcher.fetch_batch(symbols, period="1y", interval="1d")
        # Returns: {"RELIANCE": DataFrame, "HDFCBANK": DataFrame, ...}
    """

    def __init__(self, config: Optional[dict] = None, provider: str = "zerodha"):
        """
        Initialize batch fetcher.
        
        Args:
            config: Config dict. If None, loads from settings.yaml
            provider: "zerodha" (live Kite API) or "yfinance" (fallback)
        """
        self.config = config or load_config()
        self.provider_name = provider.lower()
        self._provider = None
        self._token_resolver: Optional[InstrumentResolver] = None
        self._cache: Dict[str, pd.DataFrame] = {}

    @property
    def provider(self):
        """Lazy initialization of data provider."""
        if self._provider is None:
            if self.provider_name == "zerodha":
                self._provider = ZerodhaProvider(self.config)
                logger.info("Using ZerodhaProvider for batch fetch")
            else:
                self._provider = YFinanceProvider(self.config)
                logger.info("Using YFinanceProvider for batch fetch")
        return self._provider

    @property
    def token_resolver(self) -> InstrumentResolver:
        """Lazy initialization of token resolver (Zerodha only)."""
        if self._token_resolver is None and self.provider_name == "zerodha":
            self._token_resolver = InstrumentResolver(config=self.config)
        return self._token_resolver

    def fetch_symbol(
        self,
        symbol: str,
        period: Optional[str] = None,
        interval: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical candles for a single symbol.
        
        Args:
            symbol: Trading symbol (e.g., "RELIANCE")
            period: Relative period ("1d", "7d", "1mo", "1y", etc.)
                   If None, uses config default
            interval: Candle interval ("1m", "5m", "15m", "1h", "1d", etc.)
                     If None, uses config default
            start_date: Absolute start date (format: "YYYY-MM-DD")
                       Overrides period if provided
            end_date: Absolute end date (format: "YYYY-MM-DD")
            use_cache: If True, use cached result if available
            
        Returns:
            DataFrame with columns [Open, High, Low, Close, Volume, Symbol, Exchange]
            or None if fetch failed
        """
        # Check cache
        cache_key = f"{symbol}:{period}:{interval}"
        if use_cache and cache_key in self._cache:
            logger.debug(f"Using cached data for {cache_key}")
            return self._cache[cache_key]

        try:
            logger.info(f"Fetching {symbol} (period={period}, interval={interval})")
            
            df = self.provider.get_historical_data(
                symbol=symbol,
                period=period,
                interval=interval,
                start_date=start_date,
                end_date=end_date
            )
            
            if df is not None and len(df) > 0:
                self._cache[cache_key] = df
                logger.info(
                    f"Successfully fetched {len(df)} candles for {symbol}"
                )
                return df
            else:
                logger.warning(f"No data returned for {symbol}")
                return None
        
        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            return None

    def fetch_batch(
        self,
        symbols: List[str],
        period: Optional[str] = None,
        interval: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        skip_missing: bool = True,
        max_workers: int = 1,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical candles for multiple symbols.
        
        Args:
            symbols: List of trading symbols
            period: Relative period (default from config)
            interval: Candle interval (default from config)
            start_date: Absolute start date
            end_date: Absolute end date
            skip_missing: If True, continue on fetch failure.
                         If False, raise error on first failure.
            max_workers: Number of parallel threads (1 = sequential)
            use_cache: Use cached results if available
            
        Returns:
            Dict[symbol] = DataFrame (only successfully fetched symbols)
            
        Raises:
            ValueError: If skip_missing=False and any fetch fails
        """
        # Set defaults from config if not provided
        if period is None:
            period = self.config.get("historical", {}).get("default_period", "1y")
        if interval is None:
            interval = self.config.get("historical", {}).get("default_interval", "1d")

        logger.info(
            f"Batch fetching {len(symbols)} symbols "
            f"(period={period}, interval={interval}, workers={max_workers})"
        )

        results = {}
        failed = []

        # Sequential fetch (safer for rate limits)
        if max_workers == 1:
            for symbol in symbols:
                df = self.fetch_symbol(
                    symbol,
                    period=period,
                    interval=interval,
                    start_date=start_date,
                    end_date=end_date,
                    use_cache=use_cache
                )
                
                if df is not None:
                    results[symbol] = df
                else:
                    failed.append(symbol)
                    if not skip_missing:
                        raise ValueError(f"Failed to fetch {symbol}")

        # Parallel fetch (faster but higher rate limit risk)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self.fetch_symbol,
                        symbol,
                        period=period,
                        interval=interval,
                        start_date=start_date,
                        end_date=end_date,
                        use_cache=use_cache
                    ): symbol
                    for symbol in symbols
                }
                
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        df = future.result()
                        if df is not None:
                            results[symbol] = df
                        else:
                            failed.append(symbol)
                            if not skip_missing:
                                raise ValueError(f"Failed to fetch {symbol}")
                    
                    except Exception as e:
                        logger.error(f"Exception fetching {symbol}: {e}")
                        failed.append(symbol)
                        if not skip_missing:
                            raise

        # Summary
        logger.info(
            f"Batch fetch complete: {len(results)} succeeded, "
            f"{len(failed)} failed"
        )
        
        if failed:
            logger.warning(f"Failed symbols: {failed}")

        return results

    def fetch_and_save_batch(
        self,
        symbols: List[str],
        output_dir: Optional[str] = None,
        period: Optional[str] = None,
        interval: Optional[str] = None,
        format: str = "parquet",
        skip_missing: bool = True
    ) -> Dict[str, str]:
        """
        Fetch batch and save each symbol's data to disk.
        
        Args:
            symbols: List of symbols to fetch
            output_dir: Directory to save files (default: data/raw/)
            period: Relative period
            interval: Candle interval
            format: "parquet" (default) or "csv"
            skip_missing: Skip failed symbols
            
        Returns:
            Dict[symbol] = file_path for successfully saved symbols
        """
        # Fetch data
        data = self.fetch_batch(
            symbols,
            period=period,
            interval=interval,
            skip_missing=skip_missing,
            max_workers=1
        )

        # Determine output directory
        if output_dir is None:
            output_dir = resolve_path("data/raw")
        else:
            output_dir = resolve_path(output_dir)

        ensure_dir(output_dir)

        # Save each symbol
        saved_files = {}
        
        for symbol, df in data.items():
            try:
                if format.lower() == "parquet":
                    filename = f"{symbol}_{interval}_{period}.parquet"
                    filepath = output_dir / filename
                    df.to_parquet(filepath, index=True)
                
                elif format.lower() == "csv":
                    filename = f"{symbol}_{interval}_{period}.csv"
                    filepath = output_dir / filename
                    df.to_csv(filepath, index=True)
                
                else:
                    raise ValueError(f"Unsupported format: {format}")
                
                saved_files[symbol] = str(filepath)
                logger.info(f"Saved {symbol} to {filepath}")
            
            except Exception as e:
                logger.error(f"Failed to save {symbol}: {e}")

        logger.info(f"Saved {len(saved_files)}/{len(data)} symbol files")
        return saved_files

    def clear_cache(self) -> None:
        """Clear in-memory cache."""
        self._cache.clear()
        logger.info("Cache cleared")

    def get_cache_info(self) -> dict:
        """Get cache statistics."""
        return {
            "cached_symbols": len(self._cache),
            "cached_keys": list(self._cache.keys()),
            "total_rows": sum(len(df) for df in self._cache.values())
        }


def fetch_nifty100_batch(
    symbols: List[str],
    period: str = "1y",
    interval: str = "1d",
    provider: str = "zerodha",
    config: Optional[dict] = None
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function: fetch Nifty 100 symbols in one call.
    
    Usage:
        symbols = ["RELIANCE", "HDFCBANK", "INFY"]
        data = fetch_nifty100_batch(symbols, period="1y", interval="1d")
        # Returns: {"RELIANCE": DataFrame, "HDFCBANK": DataFrame, ...}
    """
    fetcher = BatchCandleFetcher(config=config, provider=provider)
    return fetcher.fetch_batch(
        symbols,
        period=period,
        interval=interval,
        skip_missing=True,
        max_workers=1
    )
