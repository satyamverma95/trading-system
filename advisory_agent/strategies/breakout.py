"""
advisory_agent/strategies/breakout.py
Setup B — Volume-Confirmed Breakout

Price has been consolidating (Bollinger Squeeze) and breaks above a key
resistance level with above-average volume. The compression releases stored
energy into a directional move.

WHY this works:
  - Tight consolidation = supply and demand approaching equilibrium
  - A breakout on high volume = institutional buying stepped in at resistance
  - ADX rising from low level = a new trend is beginning (not a fakeout)
  - OBV confirming = money was flowing in during the quiet consolidation

Entry conditions (confluence checklist):
  ✓ Bollinger Band squeeze (BB width < 30th percentile)   [VOLATILITY]
  ✓ Volume surging  (> 1.5× average)                      [VOLUME]
  ✓ OBV breaking out / uptrend                            [VOLUME]
  ✓ ADX rising or starting to rise                        [TREND - ADX]
  ✓ RSI not overbought  (< 70)                            [MOMENTUM]
  ✓ Price above weekly pivot                              [STRUCTURE]
"""

from __future__ import annotations

from advisory_agent.core.schemas import Signal, Setup, VolatilityState, MACDState

VOLUME_BREAKOUT_RATIO   = 1.5   # Volume must be > 1.5× average
ADX_BREAKOUT_DEVELOPING = 18    # ADX doesn't need to be > 20 yet, just rising
RSI_NOT_OVERBOUGHT      = 70


def check_breakout(snapshot: dict) -> dict:
    """
    Evaluate whether the snapshot qualifies as a Breakout setup.

    Returns:
        dict with keys: setup_type, qualifies, confluence, max_confluence,
                        checks, signal, bullets
    """
    trend     = snapshot.get("trend") or {}
    momentum  = snapshot.get("momentum") or {}
    vol       = snapshot.get("volatility") or {}
    volume    = snapshot.get("volume") or {}
    structure = snapshot.get("structure") or {}

    ltp             = snapshot.get("ltp", 0)
    bb_state        = vol.get("state", "")
    bb_pct          = vol.get("bb_width_percentile")
    vol_ratio       = volume.get("volume_ratio")
    obv_trend       = volume.get("obv_trend", "")
    adx             = trend.get("adx")
    rsi             = momentum.get("rsi")
    price_vs_pivot  = structure.get("price_vs_pivot", "")

    checks  = {}
    bullets = {}

    # 1. Volatility squeeze (energy compression before breakout)
    squeeze = bb_state == VolatilityState.SQUEEZE or (bb_pct is not None and bb_pct < 30)
    checks["bb_squeeze"] = squeeze
    pct_str = f"({bb_pct:.0f}th percentile)" if bb_pct is not None else ""
    bullets["bb_squeeze"] = (
        f"✅ Bollinger Band SQUEEZE {pct_str} — volatility compressed to historical lows. "
        f"A sharp directional move is building."
        if squeeze else
        f"⬜ No Bollinger squeeze detected {pct_str}. Breakout conditions not yet set."
    )

    # 2. Volume surge (institutional participation)
    vol_surge = vol_ratio is not None and vol_ratio >= VOLUME_BREAKOUT_RATIO
    checks["volume_surge"] = vol_surge
    bullets["volume_surge"] = (
        f"✅ Volume at {vol_ratio:.2f}× average — institutional participation confirmed on the breakout."
        if vol_surge else
        f"⬜ Volume at {vol_ratio:.2f}× average — insufficient for a valid breakout. "
        f"High-volume breakouts need > 1.5× average."
    ) if vol_ratio else (checks.__setitem__("volume_surge", False) or "⬜ Volume data unavailable.")

    # 3. OBV confirming (smart money was accumulating during consolidation)
    obv_ok = obv_trend in ("UPTREND",)
    checks["obv_confirm"] = obv_ok
    bullets["obv_confirm"] = (
        "✅ OBV uptrend intact — smart money was accumulating during the consolidation."
        if obv_ok else
        "⬜ OBV not in uptrend. Without OBV confirmation, breakout risk of being a fakeout."
    )

    # 4. ADX not too high (new trend beginning, not exhausted)
    adx_ok = adx is not None and adx < 45  # Too high = late stage
    checks["adx_suitable"] = adx_ok
    bullets["adx_suitable"] = (
        f"✅ ADX at {adx:.1f} — suitable for breakout entry (trend developing)."
        if adx_ok and adx else
        f"⬜ ADX at {adx:.1f} — elevated ADX may indicate a late-stage move."
    ) if adx else (checks.__setitem__("adx_suitable", False) or "⬜ ADX unavailable.")

    # 5. RSI not overbought
    rsi_ok = rsi is not None and rsi < RSI_NOT_OVERBOUGHT
    checks["rsi_not_overbought"] = rsi_ok
    bullets["rsi_not_overbought"] = (
        f"✅ RSI at {rsi:.1f} — not overbought. Room for price to run."
        if rsi_ok else
        f"⬜ RSI at {rsi:.1f} — overbought. Chasing a breakout here carries reversal risk."
    ) if rsi is not None else (checks.__setitem__("rsi_not_overbought", False) or "⬜ RSI unavailable.")

    # 6. Above weekly pivot
    checks["above_pivot"] = price_vs_pivot == "ABOVE_PP"
    pivot_pp = structure.get("weekly_pp")
    bullets["above_pivot"] = (
        f"✅ Price above weekly pivot (₹{pivot_pp:,.2f}) — bullish weekly context."
        if checks["above_pivot"] else
        f"⬜ Price below weekly pivot — weekly context is not supportive."
    )

    confluence    = sum(checks.values())
    max_confluence = len(checks)

    # Breakout needs at minimum: squeeze + volume surge + OBV
    qualifies = checks["bb_squeeze"] and checks["volume_surge"] and checks["obv_confirm"] and confluence >= 4

    return {
        "setup_type":     Setup.BREAKOUT,
        "qualifies":      qualifies,
        "confluence":     confluence,
        "max_confluence": max_confluence,
        "checks":         checks,
        "signal":         Signal.BUY if qualifies else (Signal.WATCH if confluence >= 3 else Signal.NEUTRAL),
        "bullets":        list(bullets.values()),
    }
