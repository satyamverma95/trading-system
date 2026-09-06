"""
advisory_agent/intelligence/prompt_builder.py
Assembles the structured LLM prompt from the deterministic analysis output.

CRITICAL DESIGN RULE: The LLM receives ALL numbers pre-computed.
It must never recalculate, round, or modify any price level, indicator
value, stop-loss, or target. Its ONLY job is to:
  1. Narrate WHY the technical setup is or isn't compelling
  2. Contextualise with external factors (VIX, FII, news)
  3. Surface any risks the chart cannot show
  4. Give a 1-sentence conviction summary
"""

from __future__ import annotations

from typing import Optional


SYSTEM_PROMPT = """\
You are a senior technical analyst specializing in Indian equity swing trading (3–10 day holds).
You provide structured, objective, actionable trade analysis.

STRICT RULES:
1. ALL numerical data (prices, stop-losses, targets, indicators) is computed by a deterministic engine. 
   You must NEVER recalculate, modify, or contradict any number in the provided data.
2. Your role is ONLY to explain, contextualise, and identify external risks.
3. Write for an experienced retail trader who understands technical analysis.
4. Be direct and concise — 4 to 6 sentences maximum.
5. If the setup is weak or the external context is negative, say so clearly.
6. Do NOT use hollow phrases like "it is important to note" or "please be aware".
7. End with one sentence on the most important thing to watch during the trade.
"""


def build_prompt(
    snapshot: dict,
    classification: dict,
    context: Optional[dict],
) -> str:
    """
    Build the full LLM prompt combining technical analysis + market context.

    Args:
        snapshot:       Output of composite.build_snapshot()
        classification: Output of classifier.classify()
        context:        Merged market context dict (VIX + FII + news) or None

    Returns:
        Complete prompt string ready for Gemini.
    """
    symbol   = snapshot.get("symbol", "UNKNOWN")
    interval = snapshot.get("interval", "day")
    ltp      = snapshot.get("ltp", 0)

    signal     = classification.get("signal", "NEUTRAL")
    setup_type = classification.get("setup_type", "NO_SETUP")
    conf       = classification.get("confluence", 0)
    max_conf   = classification.get("max_confluence", 6)
    conf_label = classification.get("confluence_label", "LOW")
    risk       = classification.get("risk_levels")
    bullets    = classification.get("bullets", [])

    trend     = snapshot.get("trend")     or {}
    momentum  = snapshot.get("momentum")  or {}
    volatility = snapshot.get("volatility") or {}
    volume    = snapshot.get("volume")    or {}
    structure = snapshot.get("structure") or {}

    lines = [
        f"## Trade Analysis Request: {symbol} [{interval.upper()}]",
        f"Signal: {signal} | Setup: {_fmt_setup(setup_type)} | "
        f"Confluence: {conf}/{max_conf} [{conf_label}]",
        "",
        "### Computed Technical Data (DO NOT MODIFY THESE NUMBERS)",
        f"LTP: {_fmt_val(ltp, '₹')}",
        "",
        "**Trend (Dimension 1):**",
        f"  EMA-20: {_fmt_val(trend.get('ema_20'), '₹')}  |  "
        f"EMA-50: {_fmt_val(trend.get('ema_50'), '₹')}  |  "
        f"EMA-200: {_fmt_val(trend.get('ema_200'), '₹')}",
        f"  ADX: {_fmt_val(trend.get('adx'))} [{trend.get('adx_state', '')}]  |  "
        f"Trend State: {trend.get('state', 'N/A')}",
        "",
        "**Momentum (Dimension 2):**",
        f"  RSI-14: {_fmt_val(momentum.get('rsi'))} [{momentum.get('rsi_state', '')}]  |  "
        f"RSI Divergence: {momentum.get('rsi_divergence') or 'None'}",
        f"  MACD: {_fmt_val(momentum.get('macd_line'))} | Signal: {_fmt_val(momentum.get('macd_signal'))} | "
        f"Hist: {_fmt_val(momentum.get('macd_hist'))} | State: {momentum.get('macd_state', '')}",
        "",
        "**Volatility (Dimension 3):**",
        f"  ATR-14: {_fmt_val(volatility.get('atr'), '₹')} ({_fmt_val(volatility.get('atr_pct'))}% of price)  |  "
        f"BB Width Percentile: {_fmt_val(volatility.get('bb_width_percentile'))}th  |  "
        f"State: {volatility.get('state', 'N/A')}",
        "",
        "**Volume (Dimension 4):**",
        f"  Volume Ratio: {_fmt_val(volume.get('volume_ratio'))}× average  |  "
        f"OBV: {volume.get('obv_trend', 'N/A')}  |  "
        f"State: {volume.get('state', 'N/A')}",
        "",
        "**Structure (Dimension 5):**",
        f"  Nearest Fibonacci: {structure.get('nearest_fib_level', 'N/A')} "
        f"@ {_fmt_val(structure.get('nearest_fib_price'), '₹')} "
        f"({_fmt_val(structure.get('nearest_fib_distance_pct'))}% away)",
        f"  Weekly Pivot (PP): {_fmt_val(structure.get('weekly_pp'), '₹')}  |  "
        f"Price vs Pivot: {structure.get('price_vs_pivot', 'N/A')}",
    ]

    if risk:
        lines += [
            "",
            "**Risk Levels (ATR-computed, DO NOT MODIFY):**",
            f"  Entry Zone: {_fmt_val(risk.get('entry_low'), '₹')} – {_fmt_val(risk.get('entry_high'), '₹')}",
            f"  Stop-Loss:  {_fmt_val(risk.get('stop_loss'), '₹')}  (ATR × {risk.get('sl_multiplier', 'N/A')})",
            f"  Target 1:   {_fmt_val(risk.get('target_1'), '₹')}  (R:R = 1:{risk.get('rr_t1', 'N/A')})",
            f"  Target 2:   {_fmt_val(risk.get('target_2'), '₹')}  (R:R = 1:{risk.get('rr_t2', 'N/A')})",
        ]

    if bullets:
        lines += ["", "**Setup Checks:**"]
        lines += [f"  {b}" for b in bullets[:6]]  # Top 6 most relevant

    if context:
        lines += ["", "### Market Context"]
        vix = context.get("vix")
        if vix:
            lines.append(
                f"India VIX: {_fmt_val(vix.get('current_vix'))} [{vix.get('state', '')}] | "
                f"Trend: {vix.get('vix_trend', '')}"
            )

        fii = context.get("fii_dii")
        if fii:
            lines.append(
                f"FII 5-day net: {_fmt_val(fii.get('fii_net_5d_cr'), '₹')} Cr | "
                f"DII 5-day net: {_fmt_val(fii.get('dii_net_5d_cr'), '₹')} Cr | "
                f"Flow: {fii.get('institutional_flow', 'N/A')}"
            )

        news = context.get("news")
        if news and news.get("headlines"):
            lines.append("Recent Headlines:")
            for h in news["headlines"][:3]:
                lines.append(f"  • {h}")

    lines += [
        "",
        "### Your Task",
        "Write 4–6 sentences of actionable trade commentary:",
        "1. Explain WHY the technical setup is compelling (or why it is not).",
        "2. Contextualise with VIX, FII flow, and any relevant news.",
        "3. Identify the 1–2 most important risks to this trade.",
        "4. End with the single most important thing to monitor during the trade.",
    ]

    return "\n".join(lines)


def _fmt_setup(setup_type: str) -> str:
    return {
        "MOMENTUM_PULLBACK":  "Momentum Pullback",
        "BREAKOUT":           "Volume Breakout",
        "OVERSOLD_REVERSAL":  "Oversold Reversal",
        "NO_SETUP":           "No Clear Setup",
    }.get(setup_type, setup_type)


def _fmt_val(val, prefix: str = "", decimals: int = 2) -> str:
    """Format numeric values safely with optional prefix and comma separation."""
    if val is None or val == "N/A":
        return "N/A"
    try:
        num = float(val)
        if decimals == 0:
            formatted = f"{int(round(num)):,}"
        else:
            formatted = f"{num:,.{decimals}f}"
        return f"{prefix}{formatted}" if prefix else formatted
    except (TypeError, ValueError):
        return str(val)

