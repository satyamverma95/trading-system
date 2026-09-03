# =================================================================
# source_code/ingestion/auth/session_manager.py
# Zerodha Kite Connect Authentication
# =================================================================

import logging
from pathlib import Path
from typing import Optional
from kiteconnect import KiteConnect
import yaml

from source_code.common.config_loader import load_config

logger = logging.getLogger(__name__)


def get_authenticated_kite() -> Optional[KiteConnect]:
    """
    Get an authenticated KiteConnect instance.
    
    Requires valid API credentials in config/secrets.yaml:
        zerodha:
            api_key: your_api_key
            api_secret: your_api_secret
            access_token: your_access_token (optional - for quick auth)
    
    Returns:
        KiteConnect instance if authenticated, None if credentials missing
    
    Raises:
        ValueError: If required credentials not found
    """
    try:
        config = load_config()
        zerodha_cfg = config.get("zerodha", {})
        
        api_key = zerodha_cfg.get("api_key", "").strip()
        api_secret = zerodha_cfg.get("api_secret", "").strip()
        access_token = zerodha_cfg.get("access_token", "").strip()
        
        if not api_key:
            raise ValueError(
                "Zerodha API Key not configured. "
                "Please add 'zerodha.api_key' to config/secrets.yaml"
            )
        
        if not api_secret:
            raise ValueError(
                "Zerodha API Secret not configured. "
                "Please add 'zerodha.api_secret' to config/secrets.yaml"
            )
        
        # Initialize Kite client
        kite = KiteConnect(api_key=api_key)
        
        # If access token provided, use it directly
        if access_token:
            kite.set_access_token(access_token)
            logger.info("Authenticated with Zerodha using access token")
            return kite

        raise ValueError(
            "Zerodha access token not configured. Generate today's access token "
            "through Kite login and add 'zerodha.access_token' to config/secrets.yaml"
        )
    
    except Exception as e:
        logger.error(f"Failed to authenticate with Zerodha: {e}")
        raise


def load_session() -> Optional[dict]:
    """Load stored Zerodha session."""
    logger.debug("load_session called")
    return None


def save_session(access_token: str, public_token: Optional[str] = None):
    """Save Zerodha session."""
    if not access_token or not access_token.strip():
        raise ValueError("Cannot save an empty Zerodha access token")

    secrets_path = Path(__file__).resolve().parents[3] / "config" / "secrets.yaml"
    with secrets_path.open("r", encoding="utf-8") as file_handle:
        secrets = yaml.safe_load(file_handle) or {}
    secrets.setdefault("zerodha", {})["access_token"] = access_token.strip()
    with secrets_path.open("w", encoding="utf-8") as file_handle:
        yaml.safe_dump(secrets, file_handle, default_flow_style=False, sort_keys=False)
    logger.info("Saved Zerodha access token")
