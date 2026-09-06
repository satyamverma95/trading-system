"""
advisory_agent/core/security.py
Telegram user-ID whitelist enforcer.

Wrap every command handler with @require_authorized to silently drop
messages from users not in config/secrets.yaml → telegram.allowed_user_ids.
"""

from __future__ import annotations

import logging
from functools import wraps
from typing import Callable

from telegram import Update
from telegram.ext import ContextTypes

from advisory_agent.config.settings import get_allowed_user_ids

logger = logging.getLogger(__name__)


def require_authorized(handler: Callable) -> Callable:
    """
    Decorator that silently drops Telegram messages from non-whitelisted users.

    - If the whitelist is empty, all users are allowed (development mode).
    - If a user is not in the whitelist, the message is dropped with no reply.
      This prevents the bot from leaking its existence to unknown callers.

    Usage::

        @require_authorized
        async def cmd_scan(update, context):
            ...
    """

    @wraps(handler)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if user is None:
            logger.warning("Received update with no user object — dropping silently.")
            return

        allowed = get_allowed_user_ids()
        if allowed and user.id not in allowed:
            logger.warning(
                "Unauthorized access attempt — user_id=%s username=%s first_name=%s",
                user.id,
                user.username,
                user.first_name,
            )
            return  # silent drop — never reply to unauthorized callers

        return await handler(update, context)

    return wrapper
