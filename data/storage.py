# =================================================================
# data/storage.py
# Parquet Time-Series Storage Engine for Market Data
# Structure: data/raw/{timeframe}/{exchange}/{instrument_type}/{symbol}.parquet
# =================================================================

import os
import pandas as pd
from typing import Optional, Union
from datetime import datetime, date
from utils.logger import get_logger

logger = get_logger(__name__)

RAW_DATA_ROOT = "data/raw"


def sanitize_filename(symbol: str) -> str:
    """Sanitize symbol string for filesystem compatibility."""
    return symbol.strip().replace(" ", "_").replace("^", "idx_").replace(".", "_").upper()


def get_storage_path(
    timeframe: str,
    exchange: str,
    instrument_type: str,
    symbol: str,
    root: str = RAW_DATA_ROOT
) -> str:
    """
    Generate standard parquet file path.
    Example: data/raw/1d/NSE/equity/RELIANCE.parquet
    """
    clean_tf = "1d" if timeframe.lower() in ["1d", "day", "daily"] else "1m"
    clean_ex = exchange.upper()
    clean_it = instrument_type.lower()
    clean_sym = sanitize_filename(symbol)

    return os.path.join(root, clean_tf, clean_ex, clean_it, f"{clean_sym}.parquet")


def save_market_data(
    df: pd.DataFrame,
    timeframe: str,
    exchange: str,
    instrument_type: str,
    symbol: str,
    merge: bool = True
) -> str:
    """
    Save OHLCV DataFrame to partitioned Parquet file.
    If merge=True and file exists, performs strict deduplication & sorting.
    """
    if df is None or df.empty:
        return ""

    filepath = get_storage_path(timeframe, exchange, instrument_type, symbol)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Ensure index is datetime and sorted
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    if merge and os.path.exists(filepath):
        try:
            existing = pd.read_parquet(filepath)
            if not isinstance(existing.index, pd.DatetimeIndex):
                existing.index = pd.to_datetime(existing.index)

            # Combine existing with new data
            combined = pd.concat([existing, df])
            # Deduplicate keeping newest data
            combined = combined[~combined.index.duplicated(keep="last")]
            # Chronological sort
            combined = combined.sort_index(ascending=True)
            combined.to_parquet(filepath, index=True)
            logger.info("Updated parquet storage: %s (%d total rows)", filepath, len(combined))
            return filepath
        except Exception as e:
            logger.warning(f"Merge failed for {filepath}, overwriting: {e}")

    df = df[~df.index.duplicated(keep="last")].sort_index(ascending=True)
    df.to_parquet(filepath, index=True)
    logger.info("Saved parquet storage: %s (%d rows)", filepath, len(df))
    return filepath


def load_market_data(
    timeframe: str,
    exchange: str,
    instrument_type: str,
    symbol: str,
    start: Optional[Union[str, datetime, date]] = None,
    end: Optional[Union[str, datetime, date]] = None
) -> Optional[pd.DataFrame]:
    """
    Load cached OHLCV data from Parquet file, optionally filtered by date range.
    """
    filepath = get_storage_path(timeframe, exchange, instrument_type, symbol)
    if not os.path.exists(filepath):
        return None

    try:
        df = pd.read_parquet(filepath)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        df = df.sort_index(ascending=True)

        # Slice date range if requested
        if start:
            start_dt = pd.to_datetime(start)
            if start_dt.tzinfo is not None and df.index.tz is None:
                start_dt = start_dt.tz_localize(None)
            elif start_dt.tzinfo is None and df.index.tz is not None:
                start_dt = start_dt.tz_localize(df.index.tz)
            df = df[df.index >= start_dt]

        if end:
            end_dt = pd.to_datetime(end)
            if end_dt.tzinfo is not None and df.index.tz is None:
                end_dt = end_dt.tz_localize(None)
            elif end_dt.tzinfo is None and df.index.tz is not None:
                end_dt = end_dt.tz_localize(df.index.tz)
            df = df[df.index <= end_dt]

        return df if not df.empty else None
    except Exception as e:
        logger.warning(f"Failed to load cached parquet at {filepath}: {e}")
        return None


def resample_ohlcv(df: pd.DataFrame, target_interval: str) -> pd.DataFrame:
    """
    Resample 1-minute base OHLCV candles to any higher timeframe.
    Supported: 2m, 3m, 5m, 10m, 15m, 30m, 60m/1h, 1d, 1w
    """
    if df is None or df.empty:
        return df

    rule_map = {
        "2m": "2min", "2min": "2min",
        "3m": "3min", "3minute": "3min",
        "5m": "5min", "5min": "5min", "5minute": "5min",
        "10m": "10min", "10min": "10min",
        "15m": "15min", "15min": "15min", "15minute": "15min",
        "30m": "30min", "30min": "30min", "30minute": "30min",
        "60m": "60min", "1h": "60min", "1hour": "60min", "60minute": "60min",
        "4h": "4h",
        "1d": "1D", "day": "1D", "daily": "1D",
        "1w": "1W", "1wk": "1W", "weekly": "1W",
    }

    target = target_interval.lower()
    if target in ["1m", "1min", "minute"]:
        return df

    rule = rule_map.get(target)
    if not rule:
        raise ValueError(f"Unsupported resampling interval '{target_interval}'.")

    resampled = df.resample(rule, label="left", closed="left").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
        **({col: "last" for col in df.columns if col not in ["Open", "High", "Low", "Close", "Volume"]})
    }).dropna(subset=["Close"])

    return resampled