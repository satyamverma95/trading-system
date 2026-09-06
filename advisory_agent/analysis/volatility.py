"""
advisory_agent/analysis/volatility.py
Dimension 3 of 5 — Volatility Analysis

Answers: How much room does this stock have to move? Is a big move imminent?

Indicators:
  - ATR-14   (absolute volatility → stop-loss sizing, position sizing)
  - Bollinger Bands (20, 2σ)   (relative volatility, squeeze detection)

Output: dict with keys matching the 'volatility' block of the API response.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
try:
    import pandas_ta_classic as ta
except ImportError:
    import pandas_ta as ta

from advisory_agent.core.schemas import VolatilityState

logger = logging.getLogger(__name__)

# BB width percentile thresholds (vs rolling 252-bar / 1-year history)
SQUEEZE_PERCENTILE    = 20    # BB width in bottom 20% of history → squeeze
EXPANSION_PERCENTILE  = 80    # BB width in top 20% → expansion underway


def analyze_volatility(df: pd.DataFrame) -> dict:
    """
    Compute ATR-14, Bollinger Bands, and volatility state.

    Args:
        df: OHLCV DataFrame with lowercase columns (close, high, low).

    Returns:
        dict with keys:
            atr, atr_pct, bb_upper, bb_middle, bb_lower,
            bb_width, bb_width_percentile, state, description
    """
    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    # ── ATR ──────────────────────────────────────────────────────────────────
    atr_series = ta.atr(high, low, close, length=14)
    curr_atr   = _last(atr_series)
    curr_close = float(close.iloc[-1])
    atr_pct    = round((curr_atr / curr_close) * 100, 3) if curr_atr and curr_close else None

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_df = ta.bbands(close, length=20, std=2.0)

    bb_lower_col  = _find_col(bb_df, "BBL")
    bb_middle_col = _find_col(bb_df, "BBM")
    bb_upper_col  = _find_col(bb_df, "BBU")
    bb_width_col  = _find_col(bb_df, "BBW")   # (upper - lower) / middle × 100

    curr_bb_lower  = _last(bb_df[bb_lower_col])  if bb_lower_col  else None
    curr_bb_middle = _last(bb_df[bb_middle_col]) if bb_middle_col else None
    curr_bb_upper  = _last(bb_df[bb_upper_col])  if bb_upper_col  else None
    curr_bb_width  = _last(bb_df[bb_width_col])  if bb_width_col  else None

    # BB width percentile vs rolling 252-bar window
    bb_width_series = bb_df[bb_width_col] if bb_width_col else None
    bb_percentile   = _width_percentile(bb_width_series)

    state = _classify_volatility(bb_percentile, curr_atr, atr_series)
    description = _describe(state, curr_atr, atr_pct, bb_percentile)

    return {
        "atr":                  round(curr_atr, 2)         if curr_atr         is not None else None,
        "atr_pct":              atr_pct,                   # ATR as % of price
        "bb_upper":             round(curr_bb_upper, 2)    if curr_bb_upper    is not None else None,
        "bb_middle":            round(curr_bb_middle, 2)   if curr_bb_middle   is not None else None,
        "bb_lower":             round(curr_bb_lower, 2)    if curr_bb_lower    is not None else None,
        "bb_width":             round(curr_bb_width, 4)    if curr_bb_width    is not None else None,
        "bb_width_percentile":  round(bb_percentile, 1)    if bb_percentile    is not None else None,
        "state":                state,
        "description":          description,
    }


# ── Private helpers ────────────────────────────────────────────────────────────

def _last(series: Optional[pd.Series]) -> Optional[float]:
    if series is None or series.empty:
        return None
    val = series.iloc[-1]
    return float(val) if pd.notna(val) else None


def _find_col(df: Optional[pd.DataFrame], prefix: str) -> Optional[str]:
    if df is None:
        return None
    for col in df.columns:
        if col.upper().startswith(prefix.upper()):
            return col
    return None


def _width_percentile(bb_width: Optional[pd.Series], window: int = 252) -> Optional[float]:
    """Return percentile rank of current BB width vs rolling history."""
    if bb_width is None or bb_width.dropna().empty:
        return None
    width_clean = bb_width.dropna()
    if len(width_clean) < 20:
        return None
    tail = width_clean.tail(window)
    current = float(tail.iloc[-1])
    pct = float((tail < current).sum() / len(tail) * 100)
    return round(pct, 1)


def _classify_volatility(
    bb_percentile: Optional[float],
    curr_atr: Optional[float],
    atr_series: Optional[pd.Series],
) -> str:
    if bb_percentile is None:
        return VolatilityState.NORMAL

    if bb_percentile <= SQUEEZE_PERCENTILE:
        return VolatilityState.SQUEEZE

    if bb_percentile >= EXPANSION_PERCENTILE:
        # Check if ATR is also rising (confirming expansion underway)
        atr_rising = _is_atr_rising(atr_series)
        return VolatilityState.EXPANSION if atr_rising else VolatilityState.HIGH

    if bb_percentile < 35:
        return VolatilityState.LOW
    if bb_percentile > 65:
        return VolatilityState.HIGH
    return VolatilityState.NORMAL


def _is_atr_rising(atr_series: Optional[pd.Series], lookback: int = 5) -> bool:
    if atr_series is None or len(atr_series.dropna()) < lookback:
        return False
    tail = atr_series.dropna().tail(lookback)
    return float(tail.iloc[-1]) > float(tail.iloc[0])


def _describe(
    state: str,
    atr: Optional[float],
    atr_pct: Optional[float],
    bb_pct: Optional[float],
) -> str:
    atr_str  = f"ATR ₹{atr:.2f} ({atr_pct:.2f}% of price)" if atr and atr_pct else ""
    pct_str  = f"BB width at {bb_pct:.0f}th percentile." if bb_pct is not None else ""

    if state == VolatilityState.SQUEEZE:
        return (f"Volatility SQUEEZE — {pct_str} Bollinger Bands are at historically "
                f"tight levels. A sharp directional move is building. {atr_str}.")
    if state == VolatilityState.EXPANSION:
        return (f"Volatility EXPANSION in progress — {pct_str} A breakout move is "
                f"already underway. {atr_str}.")
    if state == VolatilityState.LOW:
        return f"Below-average volatility. {pct_str} {atr_str}. Favours tight stops."
    if state == VolatilityState.HIGH:
        return f"Above-average volatility. {pct_str} {atr_str}. Widen stops or reduce size."
    return f"Normal volatility. {pct_str} {atr_str}."
