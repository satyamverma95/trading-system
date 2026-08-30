# =================================================================
# auth/session_manager.py
# Zerodha Kite Connect Authentication & Daily Session Manager
# =================================================================

import os
import json
from datetime import datetime, time, timedelta
import pytz
from typing import Optional, Tuple
from kiteconnect import KiteConnect
from utils.helpers import load_config
from utils.logger import get_logger

logger = get_logger(__name__)

SESSION_FILE = "config/.session.json"


def is_session_valid(session_data: dict) -> bool:
    """
    Check if a stored session is valid for the current trading day.
    Zerodha tokens expire every day at 06:00 AM IST.
    """
    if not session_data or "access_token" not in session_data:
        return False

    created_at_str = session_data.get("created_at")
    if not created_at_str:
        return False

    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)

    try:
        created_at = datetime.fromisoformat(created_at_str)
        if created_at.tzinfo is None:
            created_at = ist.localize(created_at)
    except Exception:
        return False

    cutoff_today = ist.localize(datetime.combine(now.date(), time(6, 0)))

    if now < cutoff_today:
        cutoff_yesterday = cutoff_today - timedelta(days=1)
        return created_at >= cutoff_yesterday
    else:
        return created_at >= cutoff_today


def load_session() -> Optional[dict]:
    """Load stored session from config/.session.json if valid."""
    if not os.path.exists(SESSION_FILE):
        return None

    try:
        with open(SESSION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if is_session_valid(data):
            return data
        else:
            logger.info("Stored Kite session has expired.")
            return None
    except Exception as e:
        logger.warning(f"Failed to load session file: {e}")
        return None


def save_session(access_token: str, public_token: Optional[str] = None, user_id: Optional[str] = None):
    """Save active access token and metadata to config/.session.json."""
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)

    data = {
        "access_token": access_token,
        "public_token": public_token,
        "user_id": user_id,
        "created_at": now.isoformat()
    }

    os.makedirs(os.path.dirname(SESSION_FILE), exist_ok=True)
    with open(SESSION_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Kite session successfully saved to %s", SESSION_FILE)


def get_login_url() -> Tuple[str, str, str]:
    """
    Returns (login_url, api_key, api_secret) using secrets.yaml.
    """
    config = load_config()
    zerodha_cfg = config.get("zerodha", {})
    api_key = zerodha_cfg.get("api_key", "").strip()
    api_secret = zerodha_cfg.get("api_secret", "").strip()

    if not api_key or api_key == "PASTE_YOUR_API_KEY_HERE":
        raise ValueError("Please configure 'api_key' in config/secrets.yaml first.")
    if not api_secret or api_secret == "PASTE_YOUR_API_SECRET_HERE":
        raise ValueError("Please configure 'api_secret' in config/secrets.yaml first.")

    kite = KiteConnect(api_key=api_key)
    return kite.login_url(), api_key, api_secret


def generate_session_from_request_token(request_token: str) -> KiteConnect:
    """
    Exchange request_token for access_token, save it, and return authenticated KiteConnect client.
    """
    _, api_key, api_secret = get_login_url()
    kite = KiteConnect(api_key=api_key)

    data = kite.generate_session(request_token=request_token.strip(), api_secret=api_secret)
    access_token = data["access_token"]
    public_token = data.get("public_token")
    user_id = data.get("user_id")

    save_session(access_token, public_token, user_id)
    kite.set_access_token(access_token)
    return kite


def get_authenticated_kite() -> KiteConnect:
    """
    Returns an authenticated KiteConnect client using stored session.
    Raises ValueError if no valid session is found.
    """
    _, api_key, _ = get_login_url()
    session = load_session()

    if not session:
        raise ValueError(
            "No valid Zerodha session found!\n"
            "Please run `python generate_session.py` to authenticate for today."
        )

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(session["access_token"])
    return kite