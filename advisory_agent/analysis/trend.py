"""
advisory_agent/analysis/trend.py
Dimension 1 of 5 — Trend Analysis

Answers: Which direction is this stock trending? How strong is that trend?

Indicators:
  - EMA-20 / EMA-50 / EMA-200  (direction and structure)
  - ADX-14 with DI+/DI-         (trend strength, not direction)

Output: dict with keys matching the 'trend' block of the API response.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
try:
    import pandas_ta_classic as ta
except ImportError:
    import pandas_ta as ta

from advisory_agent.core.schemas import TrendState

logger = logging.getLogger(__name__)

# ADX thresholds
ADX_TRENDING = 20      # Below this → ranging market, most setups fail
ADX_STRONG = 25        # Strong trend — ideal swing entry territory
ADX_VERY_STRONG = 40   # Late stage, momentum plays work but reversals dangerous


def analyze_trend(df: pd.DataFrame) -> dict:
    """
    Compute EMA stack and ADX-based trend classification.

    Args:
        df: OHLCV DataFrame with lowercase columns (close, high, low).
            Must have DatetimeIndex. Min 200 rows recommended for EMA-200.

    Returns:
        dict with keys:
            ema_20, ema_50, ema_200, adx, di_plus, di_minus,
            state (TrendState str), adx_state (str), description (str)
    """
    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    # ── EMA calculations ────────────────────────────────────────────────────
    ema_20 = ta.ema(close, length=20)
    ema_50 = ta.ema(close, length=50)
    ema_200 = ta.ema(close, length=200) if len(df) >= 200 else None

    curr_ema_20  = _last(ema_20)
    curr_ema_50  = _last(ema_50)
    curr_ema_200 = _last(ema_200) if ema_200 is not None else None
    curr_close   = float(close.iloc[-1])

    # ── ADX calculation ──────────────────────────────────────────────────────
    adx_df = ta.adx(high, low, close, length=14)

    # pandas_ta column names: ADX_14, DMP_14 (DI+), DMN_14 (DI-)
    adx_col   = _find_col(adx_df, "ADX")
    dmp_col   = _find_col(adx_df, "DMP")
    dmn_col   = _find_col(adx_df, "DMN")

    curr_adx = _last(adx_df[adx_col]) if adx_col else None
    curr_dip  = _last(adx_df[dmp_col]) if dmp_col else None
    curr_dim  = _last(adx_df[dmn_col]) if dmn_col else None

    # ── State classification ─────────────────────────────────────────────────
    state = _classify_trend(
        curr_close, curr_ema_20, curr_ema_50, curr_ema_200
    )
    adx_state = _classify_adx(curr_adx)

    description = _describe(state, adx_state, curr_ema_20, curr_ema_50)

    result = {
        "ema_20":     round(curr_ema_20, 2) if curr_ema_20 else None,
        "ema_50":     round(curr_ema_50, 2) if curr_ema_50 else None,
        "ema_200":    round(curr_ema_200, 2) if curr_ema_200 else None,
        "adx":        round(curr_adx, 2) if curr_adx else None,
        "di_plus":    round(curr_dip, 2) if curr_dip else None,
        "di_minus":   round(curr_dim, 2) if curr_dim else None,
        "state":      state,
        "adx_state":  adx_state,
        "description": description,
    }

    logger.debug("Trend: %s | ADX: %s | EMA20: %.2f EMA50: %.2f",
                 state, adx_state, curr_ema_20 or 0, curr_ema_50 or 0)
    return result


# ── Private helpers ────────────────────────────────────────────────────────────

def _last(series: Optional[pd.Series]) -> Optional[float]:
    if series is None or series.empty:
        return None
    val = series.iloc[-1]
    return float(val) if pd.notna(val) else None


def _find_col(df: Optional[pd.DataFrame], prefix: str) -> Optional[str]:
    """Find the first column whose name starts with prefix."""
    if df is None:
        return None
    for col in df.columns:
        if col.upper().startswith(prefix.upper()):
            return col
    return None


def _classify_trend(
    price: float,
    ema_20: Optional[float],
    ema_50: Optional[float],
    ema_200: Optional[float],
) -> str:
    if None in (ema_20, ema_50):
        return TrendState.NEUTRAL

    # Strong bull: all EMAs aligned, price above all of them
    if ema_200 and price > ema_200 and ema_20 > ema_50 and price > ema_20:
        return TrendState.STRONG_BULL

    # Bull: 20 > 50, price above 50 EMA
    if ema_20 > ema_50 and price > ema_50:
        return TrendState.BULL

    # Strong bear: price below 200 EMA and 20 < 50
    if ema_200 and price < ema_200 and ema_20 < ema_50:
        return TrendState.STRONG_BEAR

    # Bear: price below 50 EMA
    if price < ema_50:
        return TrendState.BEAR

    return TrendState.NEUTRAL


def _classify_adx(adx: Optional[float]) -> str:
    if adx is None:
        return "UNKNOWN"
    if adx < ADX_TRENDING:
        return "RANGING"          # Most breakouts/pullbacks fail in this zone
    if adx < ADX_STRONG:
        return "DEVELOPING"       # Trend forming — early entry window
    if adx < ADX_VERY_STRONG:
        return "STRONG"           # Best swing territory
    return "VERY_STRONG"          # Late stage — momentum plays, not pullbacks


def _describe(state: str, adx_state: str, ema_20: Optional[float], ema_50: Optional[float]) -> str:
    msgs = {
        TrendState.STRONG_BULL: "Price is above all three EMAs in a textbook bull alignment.",
        TrendState.BULL:        "EMA-20 above EMA-50 confirms intermediate uptrend.",
        TrendState.NEUTRAL:     "Mixed EMA structure — no clear directional bias.",
        TrendState.BEAR:        "Price below EMA-50 — bearish structure dominates.",
        TrendState.STRONG_BEAR: "Price below EMA-200 with EMA-20 < EMA-50 — strong downtrend.",
    }
    base = msgs.get(state, "")
    if adx_state == "RANGING":
        base += " ADX below 20 indicates a ranging market — setups have lower success rates."
    elif adx_state == "STRONG":
        base += " ADX above 25 confirms trend strength — ideal swing entry territory."
    return base
