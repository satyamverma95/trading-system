"""
advisory_agent/strategies/pullback.py
Setup A — Momentum Pullback

The highest-probability swing trade setup. Price is in a confirmed uptrend
and pulls back to a key support zone (EMA-20 or EMA-50) with cooling momentum
and contracting volume, then resumes the uptrend.

WHY this works:
  - The uptrend is already established (big money is already long)
  - The pullback shakes out weak hands (volume contracts)
  - Buyers step in at the EMA/Fibonacci zone (institutional support)
  - Entry on resumption gives a defined stop below the pullback low

Entry conditions (confluence checklist):
  ✓ EMA-20 > EMA-50  (uptrend confirmed)       [TREND]
  ✓ Price within ±1.5% of EMA-20 or EMA-50     [TREND - pullback to support]
  ✓ ADX > 20  (trend has enough strength)       [TREND - ADX]
  ✓ RSI in 40–55 zone                           [MOMENTUM]
  ✓ Volume contracting  (< 0.85× avg)           [VOLUME]
  ✓ OBV uptrend intact                          [VOLUME - OBV]
  ✓ MACD positive or bullish crossover          [MOMENTUM - MACD]
  ✓ Near Fibonacci support (38.2–61.8%)         [STRUCTURE]
  ✓ Price above weekly pivot                    [STRUCTURE - pivot]
"""

from __future__ import annotations

from advisory_agent.core.schemas import Signal, Setup, RSIState, MACDState, VolumeState, OBVTrend

# Tolerance for "near EMA" (price within ±X% of the EMA value)
EMA_PROXIMITY_PCT = 1.5
FIB_PROXIMITY_PCT = 2.0

# RSI zone for pullback entry
RSI_PULLBACK_LOW  = 38
RSI_PULLBACK_HIGH = 58


def check_pullback(snapshot: dict) -> dict:
    """
    Evaluate whether the snapshot qualifies as a Momentum Pullback setup.

    Returns:
        dict with keys:
            qualifies (bool), confluence (int), max_confluence (int),
            checks (dict[str, bool]), signal (str),
            bullets (list[str]) — human-readable explanation of each check
    """
    trend     = snapshot.get("trend") or {}
    momentum  = snapshot.get("momentum") or {}
    vol       = snapshot.get("volume") or {}
    structure = snapshot.get("structure") or {}

    ltp       = snapshot.get("ltp", 0)
    ema_20    = trend.get("ema_20")
    ema_50    = trend.get("ema_50")
    adx       = trend.get("adx")
    rsi       = momentum.get("rsi")
    macd_state = momentum.get("macd_state", "")
    vol_ratio  = vol.get("volume_ratio")
    obv_trend  = vol.get("obv_trend", "")
    fib_dist   = structure.get("nearest_fib_distance_pct")
    fib_level  = structure.get("nearest_fib_level", "")
    near_fib   = fib_level in ("38.2%", "50.0%", "61.8%")
    price_vs_pivot = structure.get("price_vs_pivot", "")

    checks = {}
    bullets = {}

    # 1. Uptrend confirmed
    if ema_20 and ema_50 and ema_20 > ema_50:
        near_ema = (
            ltp and
            (abs(ltp - ema_20) / ema_20 * 100 <= EMA_PROXIMITY_PCT or
             abs(ltp - ema_50) / ema_50 * 100 <= EMA_PROXIMITY_PCT)
        )
        if near_ema:
            checks["ema_pullback"] = True
            bullets["ema_pullback"] = (
                f"✅ EMA-20 (₹{ema_20:,.2f}) > EMA-50 (₹{ema_50:,.2f}) — uptrend confirmed. "
                f"Price pulling back to EMA support zone."
            )
        else:
            checks["ema_pullback"] = False
            bullets["ema_pullback"] = (
                f"⬜ EMAs bullish (20 > 50) but price is not yet near EMA-20 (₹{ema_20:,.2f}). "
                f"Wait for a deeper pullback."
            )
    else:
        checks["ema_pullback"] = False
        bullets["ema_pullback"] = "❌ EMA-20 not above EMA-50 — no uptrend structure."

    # 2. ADX trend strength
    if adx and adx >= 20:
        checks["adx_strength"] = True
        bullets["adx_strength"] = f"✅ ADX at {adx:.1f} — trend has sufficient strength for a pullback entry."
    else:
        checks["adx_strength"] = False
        bullets["adx_strength"] = f"⬜ ADX at {adx:.1f} — below 20 indicates a ranging market. Pullback setup less reliable."

    # 3. RSI in pullback zone
    if rsi and RSI_PULLBACK_LOW <= rsi <= RSI_PULLBACK_HIGH:
        checks["rsi_zone"] = True
        bullets["rsi_zone"] = (
            f"✅ RSI at {rsi:.1f} — in the ideal pullback entry zone (38–58). "
            f"Momentum cooling without becoming oversold."
        )
    elif rsi:
        checks["rsi_zone"] = False
        if rsi > RSI_PULLBACK_HIGH:
            bullets["rsi_zone"] = f"⬜ RSI at {rsi:.1f} — still elevated. Wait for further cooling below 58."
        else:
            bullets["rsi_zone"] = f"⬜ RSI at {rsi:.1f} — too oversold for a pullback entry. May need time to stabilize."
    else:
        checks["rsi_zone"] = False
        bullets["rsi_zone"] = "⬜ RSI unavailable."

    # 4. MACD positive (bullish or crossover)
    macd_bullish = macd_state in (MACDState.BULLISH, MACDState.CROSSOVER_BULLISH)
    checks["macd_bullish"] = macd_bullish
    if macd_bullish:
        if macd_state == MACDState.CROSSOVER_BULLISH:
            bars_ago = momentum.get("macd_crossover_bars_ago")
            bullets["macd_bullish"] = f"✅ MACD bullish crossover {bars_ago} bar(s) ago — momentum resuming."
        else:
            bullets["macd_bullish"] = "✅ MACD above signal line — bullish momentum intact."
    else:
        bullets["macd_bullish"] = "⬜ MACD below signal line — momentum not yet confirmed."

    # 5. Volume contracting (healthy pullback)
    contracting = vol_ratio is not None and vol_ratio < 0.90
    checks["volume_contracting"] = contracting
    if contracting:
        bullets["volume_contracting"] = (
            f"✅ Volume at {vol_ratio:.2f}× average — contracting on pullback. "
            f"Sellers are not aggressively distributing."
        )
    elif vol_ratio:
        bullets["volume_contracting"] = (
            f"⬜ Volume at {vol_ratio:.2f}× average — not clearly contracting. "
            f"Watch for further volume decline."
        )
    else:
        checks["volume_contracting"] = False
        bullets["volume_contracting"] = "⬜ Volume data unavailable."

    # 6. OBV uptrend (smart money still holding)
    obv_ok = obv_trend == OBVTrend.UPTREND
    checks["obv_uptrend"] = obv_ok
    bullets["obv_uptrend"] = (
        "✅ OBV in uptrend — institutional accumulation intact. Smart money has not sold."
        if obv_ok else
        "⬜ OBV not in uptrend — watch for OBV confirmation before entering."
    )

    # 7. Near Fibonacci support
    if near_fib and fib_dist is not None and fib_dist <= FIB_PROXIMITY_PCT:
        checks["fibonacci_support"] = True
        fib_price = structure.get("nearest_fib_price", "")
        bullets["fibonacci_support"] = (
            f"✅ Price at {fib_level} Fibonacci retracement (₹{fib_price:,.2f}) — "
            f"structural support zone with {fib_dist:.1f}% proximity."
        )
    else:
        checks["fibonacci_support"] = False
        bullets["fibonacci_support"] = (
            f"⬜ Not near a key Fibonacci level ({fib_level} at {fib_dist:.1f}% away)."
            if fib_dist else "⬜ Fibonacci levels unavailable."
        )

    # 8. Above weekly pivot
    checks["above_pivot"] = price_vs_pivot == "ABOVE_PP"
    pivot_pp = structure.get("weekly_pp")
    bullets["above_pivot"] = (
        f"✅ Price above weekly pivot (₹{pivot_pp:,.2f}) — bullish weekly bias."
        if checks["above_pivot"] else
        f"⬜ Price below weekly pivot (₹{pivot_pp:,.2f}) — bullish weekly bias absent."
    )

    confluence = sum(checks.values())
    max_confluence = len(checks)
    qualifies = checks["ema_pullback"] and checks["rsi_zone"] and confluence >= 4

    return {
        "setup_type":    Setup.MOMENTUM_PULLBACK,
        "qualifies":     qualifies,
        "confluence":    confluence,
        "max_confluence": max_confluence,
        "checks":        checks,
        "signal":        Signal.BUY if qualifies else (Signal.WATCH if confluence >= 3 else Signal.NEUTRAL),
        "bullets":       list(bullets.values()),
    }
