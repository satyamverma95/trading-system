"""
advisory_agent/core/schemas.py
Shared constants and type definitions for the advisory agent.

These are intentionally lightweight — the main data structures flowing
through the system are plain Python dicts (directly JSON-serializable by
FastAPI). Type constants live here so every module speaks the same vocabulary.
"""

from __future__ import annotations


# ── Signal Types ──────────────────────────────────────────────────────────────

class Signal:
    BUY = "BUY"
    SELL_EXIT = "SELL_EXIT"
    NEUTRAL = "NEUTRAL"
    WATCH = "WATCH"           # Interesting but no clear edge yet


# ── Setup Types ───────────────────────────────────────────────────────────────

class Setup:
    MOMENTUM_PULLBACK = "MOMENTUM_PULLBACK"   # Trend continuation pullback
    BREAKOUT = "BREAKOUT"                      # Volume-confirmed range breakout
    OVERSOLD_REVERSAL = "OVERSOLD_REVERSAL"    # Oversold bounce at Fibonacci
    NO_SETUP = "NO_SETUP"                      # No tradeable edge identified


# ── Trend States ──────────────────────────────────────────────────────────────

class TrendState:
    STRONG_BULL = "STRONG_BULL"   # All EMAs aligned, price above 200 EMA
    BULL = "BULL"                 # EMA-20 > EMA-50, price above 50 EMA
    NEUTRAL = "NEUTRAL"           # Mixed EMA structure or ranging
    BEAR = "BEAR"                 # Price below 50 EMA
    STRONG_BEAR = "STRONG_BEAR"   # Price below 200 EMA, 20 < 50


# ── Momentum States ───────────────────────────────────────────────────────────

class RSIState:
    OVERBOUGHT = "OVERBOUGHT"         # RSI > 70
    MOMENTUM_ZONE = "MOMENTUM_ZONE"   # RSI 55–70 (trend is running)
    PULLBACK_ZONE = "PULLBACK_ZONE"   # RSI 40–55 (ideal pullback entry)
    WEAK = "WEAK"                     # RSI 30–40
    OVERSOLD = "OVERSOLD"             # RSI < 30


class MACDState:
    BULLISH = "BULLISH"                       # MACD above signal line
    BEARISH = "BEARISH"                       # MACD below signal line
    CROSSOVER_BULLISH = "CROSSOVER_BULLISH"   # Just crossed above signal
    CROSSOVER_BEARISH = "CROSSOVER_BEARISH"   # Just crossed below signal


# ── Volatility States ─────────────────────────────────────────────────────────

class VolatilityState:
    SQUEEZE = "SQUEEZE"         # BB width at historical low — breakout imminent
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXPANSION = "EXPANSION"     # BB expanding after squeeze — move underway


# ── Volume States ─────────────────────────────────────────────────────────────

class VolumeState:
    SURGING = "SURGING"           # > 2× average — major institutional move
    ABOVE_AVERAGE = "ABOVE_AVG"   # 1.3–2× average
    NORMAL = "NORMAL"             # 0.8–1.3× average
    CONTRACTING = "CONTRACTING"   # 0.5–0.8× average (healthy pullback)
    VERY_LOW = "VERY_LOW"         # < 0.5× average (no conviction)


class OBVTrend:
    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    FLAT = "FLAT"


# ── VIX States ────────────────────────────────────────────────────────────────

class VIXState:
    LOW = "LOW"           # VIX < 12 — complacency / calm market
    NORMAL = "NORMAL"     # VIX 12–18 — healthy market
    ELEVATED = "ELEVATED" # VIX 18–25 — uncertainty, tighten stops
    HIGH = "HIGH"         # VIX 25–35 — fear, reduce position sizes
    EXTREME = "EXTREME"   # VIX > 35 — panic, avoid new longs


# ── Institutional Flow States ─────────────────────────────────────────────────

class InstFlow:
    STRONG_FII_BUY = "STRONG_FII_BUYING"   # FII net > +2000 Cr over 5 days
    FII_BUY = "FII_BUYING"                  # FII net positive
    MIXED = "MIXED"                          # FII and DII diverging
    FII_SELL = "FII_SELLING"                # FII net negative
    STRONG_FII_SELL = "STRONG_FII_SELLING"  # FII net < -2000 Cr over 5 days


# ── Confluence scoring ────────────────────────────────────────────────────────

CONFLUENCE_LABELS = {
    (0, 2): "VERY_LOW",
    (2, 3): "LOW",
    (3, 4): "MEDIUM",
    (4, 5): "HIGH",
    (5, 7): "VERY_HIGH",
}


def confluence_label(score: int, max_score: int = 6) -> str:
    pct = score / max_score
    if pct < 0.33:
        return "VERY_LOW"
    if pct < 0.50:
        return "LOW"
    if pct < 0.67:
        return "MEDIUM"
    if pct < 0.84:
        return "HIGH"
    return "VERY_HIGH"
