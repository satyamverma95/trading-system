"""
advisory_agent/analysis/composite.py
Assembles all five analysis dimensions into a single IndicatorSnapshot dict.

This is the main entry point for the analysis layer. Callers pass in a
Zerodha OHLCV DataFrame (uppercase columns) and receive a fully structured
snapshot ready for the strategy classifier and FastAPI response.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from advisory_agent.analysis.trend      import analyze_trend
from advisory_agent.analysis.momentum   import analyze_momentum
from advisory_agent.analysis.volatility import analyze_volatility
from advisory_agent.analysis.volume     import analyze_volume
from advisory_agent.analysis.structure  import analyze_structure

logger = logging.getLogger(__name__)

# Minimum candles required for a meaningful analysis
MIN_CANDLES_DAILY    = 60    # ~3 months of daily data
MIN_CANDLES_INTRADAY = 100   # ~100 bars for intraday timeframes


def build_snapshot(
    df: pd.DataFrame,
    symbol: str,
    interval: str,
) -> dict:
    """
    Run all five analysis dimensions and assemble the full IndicatorSnapshot.

    Args:
        df:       OHLCV DataFrame from Zerodha (uppercase columns: Open/High/Low/Close/Volume).
                  Must have a DatetimeIndex.
        symbol:   NSE trading symbol (e.g. "RELIANCE").
        interval: Human-readable interval (e.g. "day", "15m", "1h").

    Returns:
        dict with keys: symbol, interval, ltp, candle_count, computed_at,
                        trend, momentum, volatility, volume, structure

    Raises:
        ValueError: If the DataFrame is too short for meaningful analysis.
    """
    # ── Normalize columns to lowercase ────────────────────────────────────────
    df_lower = _normalize(df)

    min_required = MIN_CANDLES_DAILY if interval in ("day", "1d", "week") else MIN_CANDLES_INTRADAY
    if len(df_lower) < min_required:
        raise ValueError(
            f"{symbol}: Only {len(df_lower)} candles available; "
            f"need at least {min_required} for interval '{interval}'."
        )

    ltp = float(df_lower["close"].iloc[-1])
    logger.info("Building snapshot for %s [%s] | LTP=%.2f | %d candles",
                symbol, interval, ltp, len(df_lower))

    # ── Run all five dimensions (any failures are non-fatal) ──────────────────
    trend      = _safe("trend",      analyze_trend,      df_lower)
    momentum   = _safe("momentum",   analyze_momentum,   df_lower)
    volatility = _safe("volatility", analyze_volatility, df_lower)
    volume     = _safe("volume",     analyze_volume,     df_lower)
    structure  = _safe("structure",  analyze_structure,  df_lower)

    return {
        "symbol":       symbol,
        "interval":     interval,
        "ltp":          round(ltp, 2),
        "candle_count": len(df_lower),
        "computed_at":  datetime.now(timezone.utc).isoformat(),
        "trend":        trend,
        "momentum":     momentum,
        "volatility":   volatility,
        "volume":       volume,
        "structure":    structure,
    }


# ── Private helpers ────────────────────────────────────────────────────────────

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lowercase column names and ensure numeric types.
    Handles both Zerodha uppercase (Open/High/Low/Close/Volume) and
    already-lowercase input.
    """
    col_map = {
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
        "Datetime": "date",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)

    return df


def _safe(dimension: str, fn, df: pd.DataFrame) -> Optional[dict]:
    """Call analysis function, returning None on any error instead of crashing."""
    try:
        return fn(df)
    except Exception as exc:
        logger.error("Failed to compute %s dimension: %s", dimension, exc)
        return {"error": str(exc)}
