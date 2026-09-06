"""
advisory_agent/analysis/structure.py
Dimension 5 of 5 — Price Structure Analysis

Answers: Where will price react? What are the mathematically derived
support and resistance zones?

Indicators:
  - Fibonacci retracements (38.2%, 50%, 61.8% of most recent swing)
  - Weekly pivot points (standard floor pivots from prior week's OHLC)

Output: dict with keys matching the 'structure' block of the API response.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Fibonacci ratios (retracement of upswing: high → levels below)
FIB_LEVELS = {
    "0.0%":   0.000,
    "23.6%":  0.236,
    "38.2%":  0.382,
    "50.0%":  0.500,
    "61.8%":  0.618,
    "78.6%":  0.786,
    "100.0%": 1.000,
}

# How close to a Fib level counts as "at" the level (± %)
FIB_PROXIMITY_PCT = 1.5


def analyze_structure(df: pd.DataFrame) -> dict:
    """
    Compute Fibonacci retracement levels and weekly pivot points.

    Args:
        df: OHLCV DataFrame with lowercase columns (open, high, low, close).
            Must have a DatetimeIndex for weekly resampling.

    Returns:
        dict with keys:
            swing_high, swing_low, fib_236, fib_382, fib_500, fib_618, fib_786,
            nearest_fib_level, nearest_fib_price, nearest_fib_distance_pct,
            weekly_pp, weekly_r1, weekly_r2, weekly_s1, weekly_s2,
            price_vs_pivot, description
    """
    curr_close = float(df["close"].iloc[-1])

    # ── Fibonacci ─────────────────────────────────────────────────────────────
    fib = _calculate_fibonacci(df, lookback=60)

    nearest_name, nearest_price, nearest_dist = _nearest_fib_level(curr_close, fib)

    # ── Weekly Pivots ─────────────────────────────────────────────────────────
    pivots = _calculate_weekly_pivots(df)

    price_vs_pivot = None
    if pivots and pivots.get("pp"):
        price_vs_pivot = "ABOVE_PP" if curr_close >= pivots["pp"] else "BELOW_PP"

    description = _describe(
        fib, nearest_name, nearest_dist, pivots, price_vs_pivot, curr_close
    )

    return {
        "swing_high":               round(fib["swing_high"], 2) if fib else None,
        "swing_low":                round(fib["swing_low"],  2) if fib else None,
        "fib_236":                  round(fib["fib_236"],    2) if fib else None,
        "fib_382":                  round(fib["fib_382"],    2) if fib else None,
        "fib_500":                  round(fib["fib_500"],    2) if fib else None,
        "fib_618":                  round(fib["fib_618"],    2) if fib else None,
        "fib_786":                  round(fib["fib_786"],    2) if fib else None,
        "nearest_fib_level":        nearest_name,             # e.g. "50.0%"
        "nearest_fib_price":        round(nearest_price, 2) if nearest_price else None,
        "nearest_fib_distance_pct": round(nearest_dist, 3)  if nearest_dist  is not None else None,
        "weekly_pp":                round(pivots["pp"],  2)  if pivots and pivots.get("pp")  else None,
        "weekly_r1":                round(pivots["r1"],  2)  if pivots and pivots.get("r1")  else None,
        "weekly_r2":                round(pivots["r2"],  2)  if pivots and pivots.get("r2")  else None,
        "weekly_s1":                round(pivots["s1"],  2)  if pivots and pivots.get("s1")  else None,
        "weekly_s2":                round(pivots["s2"],  2)  if pivots and pivots.get("s2")  else None,
        "price_vs_pivot":           price_vs_pivot,
        "description":              description,
    }


# ── Private helpers ────────────────────────────────────────────────────────────

def _calculate_fibonacci(df: pd.DataFrame, lookback: int = 60) -> Optional[dict]:
    """
    Auto-detect swing high and low from the last `lookback` candles,
    then compute retracement levels assuming an upswing (low → high).

    For a downtrend setup, levels would be measured differently, but for
    swing longs (our primary use case) we always measure from low to high.
    """
    if len(df) < 20:
        return None

    recent = df.tail(lookback)
    swing_high = float(recent["high"].max())
    swing_low  = float(recent["low"].min())
    diff       = swing_high - swing_low

    if diff <= 0:
        return None

    return {
        "swing_high": swing_high,
        "swing_low":  swing_low,
        "fib_236":    swing_high - 0.236 * diff,
        "fib_382":    swing_high - 0.382 * diff,
        "fib_500":    swing_high - 0.500 * diff,
        "fib_618":    swing_high - 0.618 * diff,
        "fib_786":    swing_high - 0.786 * diff,
    }


def _nearest_fib_level(
    price: float, fib: Optional[dict]
) -> tuple[Optional[str], Optional[float], Optional[float]]:
    """Return (level name, level price, distance %) of nearest Fibonacci level."""
    if not fib:
        return None, None, None

    candidates = {
        "23.6%": fib["fib_236"],
        "38.2%": fib["fib_382"],
        "50.0%": fib["fib_500"],
        "61.8%": fib["fib_618"],
        "78.6%": fib["fib_786"],
    }

    nearest_name  = None
    nearest_price = None
    nearest_dist  = float("inf")

    for name, level_price in candidates.items():
        dist = abs(price - level_price) / price * 100
        if dist < nearest_dist:
            nearest_dist  = dist
            nearest_name  = name
            nearest_price = level_price

    return nearest_name, nearest_price, nearest_dist


def _calculate_weekly_pivots(df: pd.DataFrame) -> Optional[dict]:
    """
    Compute standard floor pivot points from the previous week's OHLC.
    Requires a DatetimeIndex.
    """
    try:
        if not isinstance(df.index, pd.DatetimeIndex):
            return None

        weekly = df[["high", "low", "close"]].resample("W").agg(
            {"high": "max", "low": "min", "close": "last"}
        )

        if len(weekly) < 2:
            return None

        prev = weekly.iloc[-2]
        H, L, C = float(prev["high"]), float(prev["low"]), float(prev["close"])

        pp = (H + L + C) / 3
        return {
            "pp": pp,
            "r1": 2 * pp - L,
            "r2": pp + (H - L),
            "s1": 2 * pp - H,
            "s2": pp - (H - L),
        }
    except Exception as exc:
        logger.warning("Weekly pivot calculation failed: %s", exc)
        return None


def _describe(
    fib: Optional[dict],
    nearest_name: Optional[str],
    nearest_dist: Optional[float],
    pivots: Optional[dict],
    price_vs_pivot: Optional[str],
    curr_close: float,
) -> str:
    parts = []

    if fib and nearest_name and nearest_dist is not None:
        if nearest_dist <= FIB_PROXIMITY_PCT:
            parts.append(
                f"Price is {nearest_dist:.1f}% from the {nearest_name} Fibonacci "
                f"retracement level — a key structural support/resistance zone."
            )
        else:
            parts.append(
                f"Nearest Fibonacci level is {nearest_name} "
                f"({nearest_dist:.1f}% away)."
            )

    if pivots and pivots.get("pp"):
        if price_vs_pivot == "ABOVE_PP":
            parts.append(
                f"Price is above weekly pivot (PP ₹{pivots['pp']:.2f}) — bullish weekly bias."
            )
        else:
            parts.append(
                f"Price is below weekly pivot (PP ₹{pivots['pp']:.2f}) — bearish weekly bias."
            )

    return " ".join(parts) if parts else "Structural levels computed."
