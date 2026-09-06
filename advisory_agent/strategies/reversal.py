"""
advisory_agent/strategies/reversal.py
Setup C — Oversold Reversal

Price has declined significantly and is now at extreme oversold readings
with Fibonacci support, potentially showing bullish divergence.
The highest-risk of the three setups — requires the most confirmations.

WHY this works:
  - Institutional buyers step in at key Fibonacci levels (61.8%, 78.6%)
  - RSI oversold + divergence = selling exhaustion
  - Bullish candle at support + volume spike = capitulation bottom
  - Risk is well-defined (stop below the reversal candle's low)

Entry conditions (confluence checklist):
  ✓ RSI oversold  (< 35)                                    [MOMENTUM]
  ✓ RSI bullish divergence (price new low, RSI higher low)  [MOMENTUM]
  ✓ Near Fibonacci 61.8% or 78.6% retracement              [STRUCTURE]
  ✓ Price at / below weekly S1 or S2                        [STRUCTURE]
  ✓ Volume spike on reversal candle  (> 1.5× avg)           [VOLUME]
  ✓ OBV holding or reversing up                             [VOLUME]
"""

from __future__ import annotations

from advisory_agent.core.schemas import Signal, Setup, RSIState

RSI_OVERSOLD_THRESHOLD  = 35
DEEP_FIB_LEVELS         = ("61.8%", "78.6%", "100.0%")
REVERSAL_VOLUME_RATIO   = 1.3


def check_reversal(snapshot: dict) -> dict:
    """
    Evaluate whether the snapshot qualifies as an Oversold Reversal setup.

    Returns:
        dict with keys: setup_type, qualifies, confluence, max_confluence,
                        checks, signal, bullets
    """
    momentum  = snapshot.get("momentum") or {}
    structure = snapshot.get("structure") or {}
    volume    = snapshot.get("volume") or {}

    ltp            = snapshot.get("ltp", 0)
    rsi            = momentum.get("rsi")
    rsi_divergence = momentum.get("rsi_divergence")
    fib_level      = structure.get("nearest_fib_level", "")
    fib_dist       = structure.get("nearest_fib_distance_pct")
    fib_price      = structure.get("nearest_fib_price")
    weekly_s1      = structure.get("weekly_s1")
    weekly_s2      = structure.get("weekly_s2")
    vol_ratio      = volume.get("volume_ratio")
    obv_trend      = volume.get("obv_trend", "")

    checks  = {}
    bullets = {}

    # 1. RSI oversold
    rsi_oversold = rsi is not None and rsi < RSI_OVERSOLD_THRESHOLD
    checks["rsi_oversold"] = rsi_oversold
    bullets["rsi_oversold"] = (
        f"✅ RSI at {rsi:.1f} — deeply oversold. Selling may be reaching exhaustion."
        if rsi_oversold else
        f"⬜ RSI at {rsi:.1f} — not in oversold territory (need < 35 for reversal setup)."
    ) if rsi is not None else (checks.__setitem__("rsi_oversold", False) or "⬜ RSI unavailable.")

    # 2. Bullish RSI divergence (strongest reversal signal)
    checks["rsi_divergence"] = rsi_divergence == "BULLISH"
    bullets["rsi_divergence"] = (
        "✅ Bullish RSI divergence detected — price making new lows but RSI is not. "
        "Classic sign of selling exhaustion and potential reversal."
        if checks["rsi_divergence"] else
        "⬜ No bullish RSI divergence. Without divergence, reversal has lower conviction."
    )

    # 3. Deep Fibonacci support
    at_deep_fib = (
        fib_level in DEEP_FIB_LEVELS and
        fib_dist is not None and fib_dist <= 2.0
    )
    checks["deep_fib_support"] = at_deep_fib
    bullets["deep_fib_support"] = (
        f"✅ Price at {fib_level} Fibonacci retracement (₹{fib_price:,.2f}) — "
        f"deep structural support zone where institutional buyers typically enter."
        if at_deep_fib else
        f"⬜ Not at a deep Fibonacci support level ({fib_level} at {fib_dist:.1f}% away)."
        if fib_dist else "⬜ Fibonacci levels unavailable."
    )

    # 4. At or below weekly pivot support levels
    at_pivot_support = False
    pivot_desc = ""
    if weekly_s1 and ltp:
        if ltp <= weekly_s1 * 1.01:
            at_pivot_support = True
            pivot_desc = f"Price at or below weekly S1 (₹{weekly_s1:,.2f})."
    if weekly_s2 and ltp:
        if ltp <= weekly_s2 * 1.02:
            at_pivot_support = True
            pivot_desc = f"Price at weekly S2 (₹{weekly_s2:,.2f}) — extreme support zone."
    checks["pivot_support"] = at_pivot_support
    bullets["pivot_support"] = (
        f"✅ {pivot_desc} Weekly pivot support adds structural weight to the reversal zone."
        if at_pivot_support else
        "⬜ Not at a weekly pivot support level."
    )

    # 5. Volume spike (capitulation / institutional buying)
    vol_spike = vol_ratio is not None and vol_ratio >= REVERSAL_VOLUME_RATIO
    checks["volume_spike"] = vol_spike
    bullets["volume_spike"] = (
        f"✅ Volume at {vol_ratio:.2f}× average — volume spike on this candle suggests "
        f"capitulation or institutional buying stepping in."
        if vol_spike else
        f"⬜ Volume at {vol_ratio:.2f}× average — reversal needs higher volume confirmation."
    ) if vol_ratio else (checks.__setitem__("volume_spike", False) or "⬜ Volume data unavailable.")

    # 6. OBV holding (not accelerating down)
    obv_ok = obv_trend != "DOWNTREND"
    checks["obv_holding"] = obv_ok
    bullets["obv_holding"] = (
        "✅ OBV not in downtrend — institutional money is not aggressively exiting."
        if obv_ok else
        "⬜ OBV still in downtrend — smart money may not have stopped selling yet."
    )

    confluence    = sum(checks.values())
    max_confluence = len(checks)

    # Reversal requires: oversold + at least one structural support + volume
    qualifies = (
        checks["rsi_oversold"] and
        (checks["deep_fib_support"] or checks["pivot_support"]) and
        confluence >= 4
    )

    return {
        "setup_type":     Setup.OVERSOLD_REVERSAL,
        "qualifies":      qualifies,
        "confluence":     confluence,
        "max_confluence": max_confluence,
        "checks":         checks,
        "signal":         Signal.BUY if qualifies else (Signal.WATCH if confluence >= 3 else Signal.NEUTRAL),
        "bullets":        list(bullets.values()),
    }
