"""
advisory_agent/analysis/momentum.py
Dimension 2 of 5 — Momentum Analysis

Answers: Is the price move accelerating or exhausting?

Indicators:
  - RSI-14  (overbought/oversold, pullback zones, divergence)
  - MACD (12, 26, 9)  (trend momentum, signal-line crossovers)

Output: dict with keys matching the 'momentum' block of the API response.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
try:
    import pandas_ta_classic as ta
except ImportError:
    import pandas_ta as ta

from advisory_agent.core.schemas import RSIState, MACDState

logger = logging.getLogger(__name__)

# RSI thresholds
RSI_OVERBOUGHT   = 70
RSI_MOMENTUM     = 55   # Above this in uptrend = momentum running
RSI_PULLBACK_HI  = 55   # Pullback entry zone: 40–55
RSI_PULLBACK_LO  = 40
RSI_WEAK         = 30
RSI_OVERSOLD     = 30

# Look back N bars for crossover detection
CROSSOVER_LOOKBACK = 5


def analyze_momentum(df: pd.DataFrame) -> dict:
    """
    Compute RSI-14 and MACD momentum indicators.

    Args:
        df: OHLCV DataFrame with lowercase column 'close'. DatetimeIndex.

    Returns:
        dict with keys:
            rsi, rsi_state, rsi_description,
            macd_line, macd_signal, macd_hist,
            macd_state, macd_crossover_bars_ago,
            description (str)
    """
    close = df["close"]

    # ── RSI ──────────────────────────────────────────────────────────────────
    rsi_series = ta.rsi(close, length=14)
    curr_rsi   = _last(rsi_series)

    rsi_state       = _classify_rsi(curr_rsi)
    rsi_description = _describe_rsi(curr_rsi, rsi_state)

    # RSI divergence: compare last 2 price lows/highs with corresponding RSI
    rsi_divergence = _detect_rsi_divergence(close, rsi_series)

    # ── MACD ─────────────────────────────────────────────────────────────────
    macd_df = ta.macd(close, fast=12, slow=26, signal=9)

    macd_col = _find_col(macd_df, "MACD_")
    hist_col = _find_col(macd_df, "MACDh_")
    sig_col  = _find_col(macd_df, "MACDs_")

    curr_macd = _last(macd_df[macd_col]) if macd_col else None
    curr_hist = _last(macd_df[hist_col]) if hist_col else None
    curr_sig  = _last(macd_df[sig_col])  if sig_col  else None

    macd_state, crossover_bars = _classify_macd(
        macd_df[macd_col] if macd_col else None,
        macd_df[sig_col]  if sig_col  else None,
    )

    description = _describe_macd(macd_state, crossover_bars, curr_hist)

    return {
        "rsi":                     round(curr_rsi, 2) if curr_rsi is not None else None,
        "rsi_state":               rsi_state,
        "rsi_divergence":          rsi_divergence,        # "BULLISH", "BEARISH", or None
        "rsi_description":         rsi_description,
        "macd_line":               round(curr_macd, 4) if curr_macd is not None else None,
        "macd_signal":             round(curr_sig,  4) if curr_sig  is not None else None,
        "macd_hist":               round(curr_hist, 4) if curr_hist is not None else None,
        "macd_state":              macd_state,
        "macd_crossover_bars_ago": crossover_bars,
        "description":             description,
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


def _classify_rsi(rsi: Optional[float]) -> str:
    if rsi is None:
        return "UNKNOWN"
    if rsi > RSI_OVERBOUGHT:
        return RSIState.OVERBOUGHT
    if rsi > RSI_MOMENTUM:
        return RSIState.MOMENTUM_ZONE
    if rsi >= RSI_PULLBACK_LO:
        return RSIState.PULLBACK_ZONE
    if rsi >= RSI_WEAK:
        return RSIState.WEAK
    return RSIState.OVERSOLD


def _describe_rsi(rsi: Optional[float], state: str) -> str:
    if rsi is None:
        return "RSI unavailable."
    msgs = {
        RSIState.OVERBOUGHT:   f"RSI at {rsi:.1f} — overbought territory. Avoid new longs; watch for bearish divergence.",
        RSIState.MOMENTUM_ZONE:f"RSI at {rsi:.1f} — momentum zone. Trend running strong; chasing risk is elevated.",
        RSIState.PULLBACK_ZONE:f"RSI at {rsi:.1f} — ideal pullback zone (40–55). Momentum cooling without becoming oversold.",
        RSIState.WEAK:         f"RSI at {rsi:.1f} — weak momentum. Approaching oversold; watch for reversal signals.",
        RSIState.OVERSOLD:     f"RSI at {rsi:.1f} — oversold. Potential reversal zone; confirm with price action.",
    }
    return msgs.get(state, f"RSI at {rsi:.1f}.")


def _detect_rsi_divergence(
    close: pd.Series,
    rsi: Optional[pd.Series],
    lookback: int = 20,
) -> Optional[str]:
    """
    Detect simple bullish or bearish RSI divergence.
    Compares the last 2 significant highs/lows on price vs RSI.
    Returns 'BULLISH', 'BEARISH', or None.
    """
    if rsi is None or len(close) < lookback or len(rsi) < lookback:
        return None

    price_tail = close.tail(lookback)
    rsi_tail   = rsi.tail(lookback).dropna()

    if len(rsi_tail) < 10:
        return None

    # Bearish: price new high, RSI lower high
    if (close.iloc[-1] >= price_tail.max() * 0.99 and
            rsi_tail.iloc[-1] < rsi_tail.max() * 0.95):
        return "BEARISH"

    # Bullish: price new low, RSI higher low
    if (close.iloc[-1] <= price_tail.min() * 1.01 and
            rsi_tail.iloc[-1] > rsi_tail.min() * 1.05):
        return "BULLISH"

    return None


def _classify_macd(
    macd_line: Optional[pd.Series],
    signal_line: Optional[pd.Series],
) -> tuple[str, Optional[int]]:
    """Returns (state_str, bars_since_crossover or None)."""
    if macd_line is None or signal_line is None:
        return "UNKNOWN", None

    # Scan recent bars for crossover
    for i in range(1, CROSSOVER_LOOKBACK + 1):
        try:
            prev_diff = macd_line.iloc[-i-1] - signal_line.iloc[-i-1]
            curr_diff = macd_line.iloc[-i]   - signal_line.iloc[-i]
        except IndexError:
            break

        if pd.isna(prev_diff) or pd.isna(curr_diff):
            continue

        if prev_diff < 0 and curr_diff >= 0:
            return MACDState.CROSSOVER_BULLISH, i
        if prev_diff > 0 and curr_diff <= 0:
            return MACDState.CROSSOVER_BEARISH, i

    # No recent crossover — use current position
    curr = _last(macd_line - signal_line)
    if curr is None:
        return "UNKNOWN", None
    return (MACDState.BULLISH if curr > 0 else MACDState.BEARISH), None


def _describe_macd(state: str, bars_ago: Optional[int], hist: Optional[float]) -> str:
    if state == MACDState.CROSSOVER_BULLISH:
        return f"MACD crossed above signal line {bars_ago} bar(s) ago — bullish momentum shift."
    if state == MACDState.CROSSOVER_BEARISH:
        return f"MACD crossed below signal line {bars_ago} bar(s) ago — bearish momentum shift."
    if state == MACDState.BULLISH:
        expanding = hist is not None and hist > 0
        return "MACD above signal line — bullish momentum." + (
            " Histogram expanding, momentum accelerating." if expanding else "")
    if state == MACDState.BEARISH:
        return "MACD below signal line — bearish momentum."
    return "MACD state unavailable."
