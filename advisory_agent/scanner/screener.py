"""
advisory_agent/scanner/screener.py
Market-Wide Swing Screener & Multi-Tier Bucketing Engine.

Processes batch OHLCV candle data across an equity universe (e.g. Nifty 100),
applies the 5-dimension mathematical engine, categorizes each symbol into one of
three actionable buckets, and ranks symbols within each bucket.

Buckets:
  1. PRIME_SETUPS  — High Conviction / Actionable Now (ready for entry, defined risk)
  2. DEVELOPING    — On Radar / Watchlist (good structure, waiting for pullback/trigger)
  3. AVOID         — Stay Away / Broken Structure (bearish, choppy, or no edge)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import pandas as pd

from advisory_agent.analysis.composite import build_snapshot
from advisory_agent.strategies.classifier import classify
from advisory_agent.core.schemas import Signal, TrendState, RSIState, VolumeState

logger = logging.getLogger(__name__)


class BucketName:
    PRIME_SETUPS = "PRIME_SETUPS"   # Actionable Now
    DEVELOPING   = "DEVELOPING"     # On Radar / Watchlist
    AVOID        = "AVOID"          # Stay Away


def screen_batch(
    batch_candles: Dict[str, pd.DataFrame],
    interval: str = "day",
) -> Dict[str, Any]:
    """
    Run 5-dimension analysis, bucketing, and ranking across all symbols in the batch.

    Args:
        batch_candles: Dict of symbol -> OHLCV DataFrame
        interval: Candle interval (default "day")

    Returns:
        dict containing:
          - scanned_count: int
          - success_count: int
          - market_breadth: dict (bullish_pct, bearish_pct, etc.)
          - buckets: dict with lists of ranked items:
              - prime_setups: list of dicts
              - developing: list of dicts
              - avoid: list of dicts
    """
    prime_setups: List[Dict[str, Any]] = []
    developing: List[Dict[str, Any]] = []
    avoid: List[Dict[str, Any]] = []

    bullish_trends = 0
    bearish_trends = 0
    neutral_trends = 0

    scanned = len(batch_candles)
    success = 0

    for symbol, df in batch_candles.items():
        if df is None or len(df) < 55:
            continue

        try:
            snapshot = build_snapshot(df, symbol, interval)
            classification = classify(snapshot)
            success += 1

            trend = snapshot.get("trend") or {}
            momentum = snapshot.get("momentum") or {}
            volatility = snapshot.get("volatility") or {}
            volume = snapshot.get("volume") or {}
            structure = snapshot.get("structure") or {}

            t_state = trend.get("state", TrendState.NEUTRAL)
            if t_state in (TrendState.STRONG_BULL, TrendState.BULL):
                bullish_trends += 1
            elif t_state in (TrendState.STRONG_BEAR, TrendState.BEAR):
                bearish_trends += 1
            else:
                neutral_trends += 1

            item = _build_screener_item(symbol, snapshot, classification)

            # ── Bucketing Decision ─────────────────────────────────────────
            bucket = _assign_bucket(snapshot, classification)
            item["bucket"] = bucket

            if bucket == BucketName.PRIME_SETUPS:
                item["rank_score"] = _compute_prime_rank_score(item)
                prime_setups.append(item)
            elif bucket == BucketName.DEVELOPING:
                item["trigger_note"] = _build_developing_trigger_note(snapshot)
                item["proximity_pct"] = _compute_ema_distance(snapshot)
                developing.append(item)
            else:
                item["avoid_reason"] = _build_avoid_reason(snapshot, classification)
                avoid.append(item)

        except Exception as exc:
            logger.warning("Screener failed for %s (non-fatal): %s", symbol, exc)
            continue

    # ── Sort each bucket ───────────────────────────────────────────────────────
    # Prime: Highest rank score first
    prime_setups.sort(key=lambda x: x.get("rank_score", 0), reverse=True)
    for idx, item in enumerate(prime_setups, 1):
        item["rank"] = idx

    # Developing: Lowest proximity percentage (closest to entry trigger) first
    developing.sort(key=lambda x: abs(x.get("proximity_pct", 999.0)))
    for idx, item in enumerate(developing, 1):
        item["rank"] = idx

    # Avoid: Weakest trend / lowest RSI first
    avoid.sort(key=lambda x: x.get("indicators", {}).get("rsi") or 50.0)
    for idx, item in enumerate(avoid, 1):
        item["rank"] = idx

    total_valid = bullish_trends + bearish_trends + neutral_trends
    bullish_pct = round((bullish_trends / total_valid * 100), 1) if total_valid > 0 else 0.0
    bearish_pct = round((bearish_trends / total_valid * 100), 1) if total_valid > 0 else 0.0

    return {
        "scanned_count": scanned,
        "success_count": success,
        "market_breadth": {
            "total_analyzed": total_valid,
            "bullish_count": bullish_trends,
            "bearish_count": bearish_trends,
            "neutral_count": neutral_trends,
            "bullish_pct": bullish_pct,
            "bearish_pct": bearish_pct,
            "regime": "BULLISH_DOMINANT" if bullish_pct > 55 else ("BEARISH_DOMINANT" if bearish_pct > 55 else "MIXED_CHOPPY"),
        },
        "summary": {
            "prime_count": len(prime_setups),
            "developing_count": len(developing),
            "avoid_count": len(avoid),
        },
        "buckets": {
            "prime_setups": prime_setups,
            "developing": developing,
            "avoid": avoid,
        },
    }


# ── Private Bucketing & Scoring Helpers ───────────────────────────────────────

def _assign_bucket(snapshot: dict, classification: dict) -> str:
    """Classify into PRIME_SETUPS, DEVELOPING, or AVOID."""
    signal = classification.get("signal")
    confluence = classification.get("confluence", 0)
    trend = snapshot.get("trend") or {}
    t_state = trend.get("state", "")

    # BUCKET 1: Actionable Prime Setup
    # Buy signal, at least 4/8 confluence, defined risk levels, and not in bear structure
    if signal == Signal.BUY and confluence >= 4 and classification.get("risk_levels"):
        if t_state not in (TrendState.BEAR, TrendState.STRONG_BEAR):
            return BucketName.PRIME_SETUPS

    # BUCKET 2: Developing Setup (On Radar / Watchlist)
    # The macro trend is bullish, but it hasn't pulled back to support or RSI is still hot
    if t_state in (TrendState.STRONG_BULL, TrendState.BULL):
        return BucketName.DEVELOPING

    ltp = snapshot.get("ltp", 0)
    ema_50 = trend.get("ema_50")
    if ema_50 and ltp > ema_50 and confluence >= 3:
        return BucketName.DEVELOPING

    # BUCKET 3: Avoid
    return BucketName.AVOID


def _build_screener_item(symbol: str, snapshot: dict, classification: dict) -> dict:
    """Build standardized summary record for table/card view."""
    trend = snapshot.get("trend") or {}
    momentum = snapshot.get("momentum") or {}
    volatility = snapshot.get("volatility") or {}
    volume = snapshot.get("volume") or {}
    structure = snapshot.get("structure") or {}

    return {
        "symbol":           symbol,
        "ltp":              snapshot.get("ltp"),
        "interval":         snapshot.get("interval", "day"),
        "signal":           classification.get("signal"),
        "setup_type":       classification.get("setup_type"),
        "confluence":       classification.get("confluence"),
        "max_confluence":   classification.get("max_confluence"),
        "confluence_label": classification.get("confluence_label"),
        "risk_levels":      classification.get("risk_levels"),
        "bullets":          classification.get("bullets", [])[:3],
        "indicators": {
            "trend_state":    trend.get("state"),
            "ema_20":         trend.get("ema_20"),
            "ema_50":         trend.get("ema_50"),
            "adx":            trend.get("adx"),
            "rsi":            momentum.get("rsi"),
            "rsi_state":      momentum.get("rsi_state"),
            "macd_state":     momentum.get("macd_state"),
            "atr":            volatility.get("atr"),
            "atr_pct":        volatility.get("atr_pct"),
            "bb_state":       volatility.get("state"),
            "volume_ratio":   volume.get("volume_ratio"),
            "obv_trend":      volume.get("obv_trend"),
            "nearest_fib":    structure.get("nearest_fib_level"),
            "weekly_pp":      structure.get("weekly_pp"),
            "price_vs_pivot": structure.get("price_vs_pivot"),
        },
    }


def _compute_prime_rank_score(item: dict) -> float:
    """
    Composite Multi-Factor Score (0–100) for ranking Bucket 1 candidates:
      - 35% Confluence Score
      - 25% Proximity to EMA/Support (closer = higher score)
      - 20% Risk/Reward Ratio potential
      - 20% Volume Confirmation
    """
    conf = item.get("confluence", 0)
    max_conf = item.get("max_confluence", 8)
    conf_score = (conf / max_conf) * 100 if max_conf else 50.0

    # Proximity: ideal is within 0.5% of support
    prox_pct = abs(item.get("proximity_pct", 1.0))
    prox_score = max(0.0, 100.0 - (prox_pct * 25.0))

    # Risk-Reward score
    risk = item.get("risk_levels") or {}
    rr_t2 = risk.get("rr_t2", 2.0)
    rr_score = min(100.0, (rr_t2 / 2.5) * 100.0)

    # Volume score
    vol_ratio = item.get("indicators", {}).get("volume_ratio") or 1.0
    vol_score = min(100.0, max(20.0, vol_ratio * 50.0))

    total = (0.35 * conf_score) + (0.25 * prox_score) + (0.20 * rr_score) + (0.20 * vol_score)
    return round(total, 1)


def _compute_ema_distance(snapshot: dict) -> float:
    """Distance from current price to EMA-20 in percentage."""
    ltp = snapshot.get("ltp", 0)
    trend = snapshot.get("trend") or {}
    ema_20 = trend.get("ema_20")
    if ltp and ema_20:
        return round(((ltp - ema_20) / ema_20) * 100, 2)
    return 0.0


def _build_developing_trigger_note(snapshot: dict) -> str:
    """Explain what conditions need to occur before this stock triggers an entry."""
    ltp = snapshot.get("ltp", 0)
    trend = snapshot.get("trend") or {}
    momentum = snapshot.get("momentum") or {}
    volatility = snapshot.get("volatility") or {}

    ema_20 = trend.get("ema_20")
    rsi = momentum.get("rsi")
    bb_state = volatility.get("state")

    notes = []
    if ema_20 and ltp > ema_20 * 1.015:
        dist = round(((ltp - ema_20) / ema_20) * 100, 1)
        notes.append(f"Extended +{dist}% above EMA-20. Wait for retrace to ₹{ema_20:,.1f}.")

    if rsi and rsi > 58:
        notes.append(f"RSI at {rsi:.1f}. Wait for momentum to cool below 58.")

    if bb_state == "SQUEEZE":
        notes.append("Volatility Squeeze active. Watch for expansion breakout with volume.")

    if not notes:
        notes.append("Structure is constructive. Awaiting pullback candle confirmation.")

    return " ".join(notes)


def _build_avoid_reason(snapshot: dict, classification: dict) -> str:
    """Briefly state why this stock is disqualified from long swing trades."""
    trend = snapshot.get("trend") or {}
    t_state = trend.get("state", "")
    adx = trend.get("adx")
    momentum = snapshot.get("momentum") or {}
    rsi = momentum.get("rsi")

    reasons = []
    if t_state in (TrendState.STRONG_BEAR, TrendState.BEAR):
        reasons.append("Bearish EMA alignment (price below 50/200 EMA).")
    elif adx and adx < 18:
        reasons.append(f"No active trend (ADX {adx:.1f} < 18). Range-bound chop.")
    elif rsi and rsi < 35:
        reasons.append(f"Severe downward momentum (RSI {rsi:.1f}). No reversal confirmation.")
    else:
        reasons.append("Insufficient technical confluence for swing trading.")

    return " ".join(reasons)
