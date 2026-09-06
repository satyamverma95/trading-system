"""
advisory_agent/intelligence/advisor.py
Gemini-powered advisory narrative generator.

This module is the ONLY place where an LLM is called.
The LLM receives pre-computed deterministic data and generates
a natural language explanation. It never sees raw price data
and never produces trade signals or price levels.

Setup: Add to config/secrets.yaml:
    gemini:
      api_key: "AIza..."

Or set environment variable: GEMINI_API_KEY
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from advisory_agent.intelligence.prompt_builder import SYSTEM_PROMPT, build_prompt

logger = logging.getLogger(__name__)

GEMINI_MODEL   = "gemini-1.5-flash"   # Fast, cost-effective for structured responses
MAX_OUTPUT_TOKENS = 400


def generate_advisory(
    snapshot: dict,
    classification: dict,
    context: Optional[dict] = None,
) -> dict:
    """
    Generate an LLM-powered advisory narrative via Gemini.

    Args:
        snapshot:       Output of composite.build_snapshot()
        classification: Output of classifier.classify()
        context:        Market context dict (VIX + FII + news) or None

    Returns:
        dict with keys:
            advisory_text (str) — LLM narrative
            source (str) — "gemini" or "rule_based" (fallback)
            model (str)
    """
    prompt = build_prompt(snapshot, classification, context)

    # Try Gemini first
    gemini_response = _call_gemini(prompt)
    if gemini_response:
        return {
            "advisory_text": gemini_response,
            "source":        "gemini",
            "model":         GEMINI_MODEL,
        }

    # Fallback: rule-based narrative (always available, no external API)
    logger.info("Falling back to rule-based advisory for %s", snapshot.get("symbol"))
    return {
        "advisory_text": _rule_based_fallback(snapshot, classification, context),
        "source":        "rule_based",
        "model":         "deterministic",
    }


def _call_gemini(prompt: str) -> Optional[str]:
    """Call Gemini API and return text response. Returns None on any failure."""
    api_key = _get_api_key()
    if not api_key:
        logger.debug("GEMINI_API_KEY not set — using rule-based fallback.")
        return None

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=SYSTEM_PROMPT,
            generation_config=genai.GenerationConfig(
                max_output_tokens=MAX_OUTPUT_TOKENS,
                temperature=0.3,       # Low temperature = consistent, factual
            ),
        )
        response = model.generate_content(prompt)
        text = response.text.strip()
        logger.debug("Gemini advisory generated (%d chars)", len(text))
        return text

    except ImportError:
        logger.warning("google-generativeai package not installed. pip install google-generativeai")
        return None
    except Exception as exc:
        logger.warning("Gemini API call failed (non-fatal): %s", exc)
        return None


def _get_api_key() -> Optional[str]:
    """
    Get Gemini API key from environment or secrets.yaml.
    Priority: env var > secrets.yaml > None
    """
    # 1. Environment variable
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key

    # 2. secrets.yaml
    try:
        from advisory_agent.config.settings import _load_secrets
        secrets = _load_secrets()
        key = str(secrets.get("gemini", {}).get("api_key", "")).strip()
        if key and key != "None":
            return key
    except Exception:
        pass

    return None


def _rule_based_fallback(
    snapshot: dict,
    classification: dict,
    context: Optional[dict],
) -> str:
    """
    Deterministic rule-based advisory when Gemini is not available.
    Assembles a narrative from pre-computed indicator descriptions.
    """
    symbol     = snapshot.get("symbol", "")
    setup_type = classification.get("setup_type", "NO_SETUP")
    signal     = classification.get("signal", "NEUTRAL")
    conf       = classification.get("confluence", 0)
    max_conf   = classification.get("max_confluence", 6)
    trend      = snapshot.get("trend")     or {}
    momentum   = snapshot.get("momentum")  or {}
    volatility = snapshot.get("volatility") or {}
    volume     = snapshot.get("volume")    or {}

    parts = []

    # Opening with setup context
    setup_labels = {
        "MOMENTUM_PULLBACK": "momentum pullback within an established uptrend",
        "BREAKOUT":          "volume-confirmed breakout from consolidation",
        "OVERSOLD_REVERSAL": "oversold reversal at a key structural support",
        "NO_SETUP":          "unclear setup with insufficient confluence",
    }
    setup_desc = setup_labels.get(setup_type, setup_type)
    parts.append(
        f"{symbol} is presenting a {setup_desc} with {conf}/{max_conf} confluence factors aligned."
    )

    # Trend
    t_desc = trend.get("description", "")
    if t_desc:
        parts.append(t_desc)

    # Momentum
    rsi_desc = momentum.get("rsi_description", "")
    if rsi_desc:
        parts.append(rsi_desc)

    # Volume
    vol_desc = volume.get("description", "")
    if vol_desc:
        parts.append(vol_desc)

    # Volatility
    atv_desc = volatility.get("description", "")
    if atv_desc:
        parts.append(atv_desc)

    # Market context
    if context:
        vix = context.get("vix")
        if vix:
            parts.append(vix.get("description", ""))

        fii = context.get("fii_dii")
        if fii:
            parts.append(fii.get("description", ""))

    # Closing
    risk = classification.get("risk_levels")
    if risk and signal == "BUY":
        parts.append(
            f"Stop-loss is set at ₹{risk['stop_loss']:,.2f} — a breach of this level "
            f"invalidates the setup and requires immediate exit."
        )

    return " ".join(p for p in parts if p)
