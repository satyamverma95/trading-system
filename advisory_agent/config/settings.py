"""
advisory_agent/config/settings.py
Credential and configuration loader.

Priority (highest to lowest):
  1. Environment variables (TELEGRAM_BOT_TOKEN, TELEGRAM_ALLOWED_USER_IDS)
  2. config/secrets.yaml  →  telegram: section
  3. Defaults / empty
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# advisory_agent/config/ → advisory_agent/ → trading-system/ (repo root)
_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
_SECRETS_PATH: Path = _REPO_ROOT / "config" / "secrets.yaml"


def _load_secrets() -> dict:
    """Read config/secrets.yaml — returns empty dict if missing."""
    if not _SECRETS_PATH.exists():
        logger.warning("secrets.yaml not found at %s", _SECRETS_PATH)
        return {}
    with open(_SECRETS_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ── Zerodha ──────────────────────────────────────────────────────────────────

def get_zerodha_config() -> dict:
    """Return the zerodha block from secrets.yaml."""
    return _load_secrets().get("zerodha", {})


# ── Telegram ─────────────────────────────────────────────────────────────────

def get_telegram_config() -> dict:
    """
    Merge environment variables and secrets.yaml telegram block.

    Environment variables:
        TELEGRAM_BOT_TOKEN            — overrides secrets.yaml
        TELEGRAM_ALLOWED_USER_IDS     — comma-separated ints, e.g. "123456,789012"
    """
    secrets = _load_secrets()
    tg = secrets.get("telegram", {})

    # Bot token: env > yaml
    bot_token: str = (
        os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        or str(tg.get("bot_token", "")).strip()
    )

    # Allowed user IDs: env > yaml
    raw_env = os.environ.get("TELEGRAM_ALLOWED_USER_IDS", "").strip()
    if raw_env:
        allowed_ids: list[int] = [
            int(x.strip()) for x in raw_env.split(",") if x.strip().isdigit()
        ]
    else:
        raw_yaml = tg.get("allowed_user_ids", [])
        allowed_ids = [int(uid) for uid in raw_yaml] if raw_yaml else []

    return {
        "bot_token": bot_token,
        "allowed_user_ids": allowed_ids,
    }


def get_bot_token() -> str:
    """
    Return Telegram bot token, raising RuntimeError if not configured.
    Create a bot via @BotFather on Telegram to get a token.
    """
    token = get_telegram_config().get("bot_token", "")
    if not token:
        raise RuntimeError(
            "Telegram bot token is not configured.\n"
            "Add 'telegram.bot_token' to config/secrets.yaml, "
            "or set the TELEGRAM_BOT_TOKEN environment variable.\n"
            "Create a bot at https://t.me/BotFather to get a token."
        )
    return token


def get_allowed_user_ids() -> list[int]:
    """
    Return whitelisted Telegram user IDs.
    An empty list means the whitelist is disabled (all users allowed) — only
    use this during development. Always set at least your own ID in production.
    """
    return get_telegram_config().get("allowed_user_ids", [])
