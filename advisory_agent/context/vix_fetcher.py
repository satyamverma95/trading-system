"""
advisory_agent/context/vix_fetcher.py
Fetch India VIX — the market fear gauge.

India VIX is available as a Zerodha Kite instrument (NSE exchange).
We use the existing authenticated Kite session to pull the last few days
of VIX data and compute the current level and its trend.

VIX interpretation for swing trading:
  < 12   → LOW       — complacency, calm market, good for directional bets
  12–18  → NORMAL    — healthy market
  18–25  → ELEVATED  — uncertainty; tighten stops
  25–35  → HIGH      — fear; reduce position sizes
  > 35   → EXTREME   — panic; avoid new longs entirely
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# Zerodha instrument token for India VIX (NSE)
# Token: 264969 — stable across sessions
INDIA_VIX_TOKEN   = 264969
INDIA_VIX_SYMBOL  = "INDIA VIX"
VIX_LOOKBACK_DAYS = 10


def fetch_vix(kite) -> Optional[dict]:
    """
    Fetch India VIX current value and 5-day trend via Zerodha Kite.

    Args:
        kite: Authenticated KiteConnect instance.

    Returns:
        dict with keys: current_vix, vix_5d_ago, vix_trend, state, description
        Returns None on failure (graceful degradation).
    """
    try:
        from_date = date.today() - timedelta(days=VIX_LOOKBACK_DAYS + 5)
        to_date   = date.today()

        candles = kite.historical_data(
            instrument_token=INDIA_VIX_TOKEN,
            from_date=str(from_date),
            to_date=str(to_date),
            interval="day",
        )

        if not candles:
            logger.warning("India VIX: no data returned from Kite.")
            return None

        closes = [float(c["close"]) for c in candles if c.get("close")]
        if not closes:
            return None

        current_vix = closes[-1]
        vix_5d_ago  = closes[-6] if len(closes) >= 6 else closes[0]
        vix_trend   = "RISING" if current_vix > vix_5d_ago else "FALLING"
        state       = _classify_vix(current_vix)

        return {
            "current_vix": round(current_vix, 2),
            "vix_5d_ago":  round(vix_5d_ago, 2),
            "vix_trend":   vix_trend,
            "state":       state,
            "description": _describe(current_vix, state, vix_trend),
        }

    except Exception as exc:
        logger.warning("India VIX fetch failed (non-fatal): %s", exc)
        return None


def _classify_vix(vix: float) -> str:
    if vix < 12:
        return "LOW"
    if vix < 18:
        return "NORMAL"
    if vix < 25:
        return "ELEVATED"
    if vix < 35:
        return "HIGH"
    return "EXTREME"


def _describe(vix: float, state: str, trend: str) -> str:
    base = {
        "LOW":      f"VIX at {vix:.1f} — low fear environment. Markets are calm; "
                    "directional trades have favourable risk conditions.",
        "NORMAL":   f"VIX at {vix:.1f} — normal market volatility. "
                    "Standard risk management applies.",
        "ELEVATED": f"VIX at {vix:.1f} — elevated uncertainty. "
                    "Consider tightening stop-losses or reducing position sizes.",
        "HIGH":     f"VIX at {vix:.1f} — high market fear. "
                    "Reduce position sizes significantly; avoid aggressive entries.",
        "EXTREME":  f"VIX at {vix:.1f} — PANIC conditions. "
                    "Avoid new long positions. Wait for VIX to peak and turn.",
    }.get(state, f"VIX at {vix:.1f}.")

    if trend == "RISING":
        base += " VIX is rising — fear is increasing."
    else:
        base += " VIX is falling — fear is declining, conditions improving."
    return base
