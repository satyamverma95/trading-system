"""
advisory_agent/strategies/classifier.py
Master strategy classifier.

Takes the full IndicatorSnapshot dict, evaluates all three setup types,
picks the best one (or NO_SETUP), computes risk levels, and returns
a complete AdvisoryResult dict ready for FastAPI serialization.
"""

from __future__ import annotations

import logging
from typing import Optional

from advisory_agent.core.schemas import Signal, Setup, confluence_label
from advisory_agent.strategies.pullback  import check_pullback
from advisory_agent.strategies.breakout  import check_breakout
from advisory_agent.strategies.reversal  import check_reversal

logger = logging.getLogger(__name__)

# Stop-loss multiplier (× ATR) per setup type
ATR_STOP_MULTIPLIER = {
    Setup.MOMENTUM_PULLBACK:  2.0,   # Wider — give pullback room to breathe
    Setup.BREAKOUT:           1.5,   # Tighter — stop below breakout level
    Setup.OVERSOLD_REVERSAL:  1.5,   # Tight — below reversal candle low
}

# Minimum confluence to show a non-NEUTRAL signal
MIN_CONFLUENCE_FOR_SIGNAL = 3


def classify(snapshot: dict) -> dict:
    """
    Evaluate all setup types and return the best classification.

    Args:
        snapshot: Output of composite.build_snapshot()

    Returns:
        dict with keys:
            signal, setup_type, confluence, max_confluence, confluence_label,
            risk_levels (or None), bullets, all_setups
    """
    # Run all three evaluators
    pullback_result  = check_pullback(snapshot)
    breakout_result  = check_breakout(snapshot)
    reversal_result  = check_reversal(snapshot)

    all_setups = [pullback_result, breakout_result, reversal_result]

    # Rank by qualification then by confluence score
    qualifying = [s for s in all_setups if s["qualifies"]]
    qualifying.sort(key=lambda s: s["confluence"], reverse=True)

    if qualifying:
        best = qualifying[0]
    else:
        # No setup qualifies — pick the one with highest confluence for "WATCH"
        best = max(all_setups, key=lambda s: s["confluence"])

    setup_type  = best["setup_type"]
    signal      = best["signal"]
    confluence  = best["confluence"]
    max_conf    = best["max_confluence"]
    bullets     = best["bullets"]

    # Override signal for NO_SETUP / low confluence
    if not qualifying and confluence < MIN_CONFLUENCE_FOR_SIGNAL:
        signal     = Signal.NEUTRAL
        setup_type = Setup.NO_SETUP

    # Sell signal: if trend is strongly bearish regardless of setup
    trend = snapshot.get("trend") or {}
    trend_state = trend.get("state", "")
    if trend_state in ("STRONG_BEAR", "BEAR") and not qualifying:
        signal = Signal.SELL_EXIT
        bullets.insert(0, "❌ Price is in a bearish trend structure. Avoid new long positions.")

    # Risk levels
    risk_levels = _calculate_risk_levels(snapshot, setup_type) if signal == Signal.BUY else None

    cl = confluence_label(confluence, max_conf)

    logger.info(
        "Classified %s as %s [%s] | Confluence: %d/%d [%s]",
        snapshot.get("symbol"), setup_type, signal, confluence, max_conf, cl
    )

    return {
        "signal":           signal,
        "setup_type":       setup_type,
        "confluence":       confluence,
        "max_confluence":   max_conf,
        "confluence_label": cl,
        "risk_levels":      risk_levels,
        "bullets":          bullets,
        "all_setups": [
            {
                "type":       s["setup_type"],
                "qualifies":  s["qualifies"],
                "confluence": s["confluence"],
            }
            for s in all_setups
        ],
    }


# ── Risk level calculator ──────────────────────────────────────────────────────

def _calculate_risk_levels(snapshot: dict, setup_type: str) -> Optional[dict]:
    """
    Compute ATR-based stop-loss and targets.
    Returns None if ATR is not available.
    """
    ltp = snapshot.get("ltp")
    volatility = snapshot.get("volatility") or {}
    atr = volatility.get("atr")

    if not ltp or not atr:
        return None

    sl_mult = ATR_STOP_MULTIPLIER.get(setup_type, 2.0)
    stop_loss = round(ltp - sl_mult * atr, 2)
    risk = ltp - stop_loss

    if risk <= 0:
        return None

    target_1 = round(ltp + 1.5 * risk, 2)
    target_2 = round(ltp + 2.5 * risk, 2)
    target_3 = round(ltp + 4.0 * risk, 2)

    return {
        "entry_low":      round(ltp * 0.998, 2),
        "entry_high":     round(ltp * 1.002, 2),
        "stop_loss":      stop_loss,
        "atr_used":       round(atr, 2),
        "sl_multiplier":  sl_mult,
        "target_1":       target_1,
        "target_2":       target_2,
        "target_3":       target_3,
        "risk_per_share": round(risk, 2),
        "rr_t1":          1.5,
        "rr_t2":          2.5,
        "rr_t3":          4.0,
    }
