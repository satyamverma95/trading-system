"""
advisory_agent/analysis/volume.py
Dimension 4 of 5 — Volume Analysis

Answers: Is smart money (institutions, FIIs) participating in this move?
Volume is the "lie detector" — it validates or invalidates every price move.

Indicators:
  - Volume ratio vs 20-day average  (participation level)
  - OBV (On-Balance Volume)          (accumulation vs distribution trend)

Output: dict with keys matching the 'volume' block of the API response.
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

from advisory_agent.core.schemas import VolumeState, OBVTrend

logger = logging.getLogger(__name__)

# Volume ratio thresholds (current / 20d average)
VOL_SURGE         = 2.0    # > 2× → major institutional move
VOL_ABOVE_AVG     = 1.3    # > 1.3× → above average participation
VOL_NORMAL_HI     = 1.3
VOL_NORMAL_LO     = 0.8
VOL_CONTRACTING   = 0.5    # < 0.5× → very thin, no conviction

# OBV trend detection: compare slope over N bars
OBV_TREND_WINDOW = 10


def analyze_volume(df: pd.DataFrame) -> dict:
    """
    Compute volume ratio and OBV trend.

    Args:
        df: OHLCV DataFrame with lowercase columns (close, volume).

    Returns:
        dict with keys:
            current_volume, avg_volume_20d, volume_ratio,
            obv_trend, state, description
    """
    close  = df["close"]
    volume = df["volume"]

    # ── Volume ratio ──────────────────────────────────────────────────────────
    curr_vol  = float(volume.iloc[-1])
    avg_20d   = float(volume.tail(21).iloc[:-1].mean())   # exclude today
    vol_ratio = round(curr_vol / avg_20d, 3) if avg_20d > 0 else None

    state = _classify_volume(vol_ratio)

    # ── OBV ───────────────────────────────────────────────────────────────────
    obv_series = ta.obv(close, volume)
    obv_trend  = _classify_obv_trend(obv_series)

    description = _describe(state, obv_trend, vol_ratio)

    return {
        "current_volume":  int(curr_vol),
        "avg_volume_20d":  int(avg_20d),
        "volume_ratio":    vol_ratio,
        "obv_trend":       obv_trend,
        "state":           state,
        "description":     description,
    }


# ── Private helpers ────────────────────────────────────────────────────────────

def _classify_volume(ratio: Optional[float]) -> str:
    if ratio is None:
        return "UNKNOWN"
    if ratio >= VOL_SURGE:
        return VolumeState.SURGING
    if ratio >= VOL_ABOVE_AVG:
        return VolumeState.ABOVE_AVERAGE
    if ratio >= VOL_NORMAL_LO:
        return VolumeState.NORMAL
    if ratio >= VOL_CONTRACTING:
        return VolumeState.CONTRACTING
    return VolumeState.VERY_LOW


def _classify_obv_trend(obv: Optional[pd.Series]) -> str:
    """
    Classify OBV as uptrend, downtrend, or flat based on slope.
    Uses linear regression on the recent window to avoid noise.
    """
    if obv is None or len(obv.dropna()) < OBV_TREND_WINDOW:
        return OBVTrend.FLAT

    tail = obv.dropna().tail(OBV_TREND_WINDOW)
    x    = np.arange(len(tail))
    slope, _ = np.polyfit(x, tail.values, 1)

    # Normalize slope by OBV magnitude to get relative direction
    scale = abs(tail.mean()) if abs(tail.mean()) > 0 else 1
    rel_slope = slope / scale

    if rel_slope > 0.003:
        return OBVTrend.UPTREND
    if rel_slope < -0.003:
        return OBVTrend.DOWNTREND
    return OBVTrend.FLAT


def _describe(state: str, obv_trend: str, ratio: Optional[float]) -> str:
    ratio_str = f"{ratio:.2f}× average" if ratio else ""

    lines = []

    if state == VolumeState.SURGING:
        lines.append(f"Volume SURGING at {ratio_str} — major institutional participation.")
    elif state == VolumeState.ABOVE_AVERAGE:
        lines.append(f"Above-average volume ({ratio_str}) confirms institutional interest.")
    elif state == VolumeState.CONTRACTING:
        lines.append(f"Volume contracting to {ratio_str} — sellers exhausted, healthy pullback.")
    elif state == VolumeState.VERY_LOW:
        lines.append(f"Very low volume ({ratio_str}) — move lacks conviction.")
    else:
        lines.append(f"Normal volume ({ratio_str}).")

    if obv_trend == OBVTrend.UPTREND:
        lines.append("OBV trending up — smart money accumulating.")
    elif obv_trend == OBVTrend.DOWNTREND:
        lines.append("OBV trending down — distribution by institutions. Caution.")
    else:
        lines.append("OBV flat — no clear institutional bias.")

    return " ".join(lines)
