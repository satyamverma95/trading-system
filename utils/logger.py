# =================================================================
# utils/logger.py
# Centralised logging for the trading system
# =================================================================

import logging
import os
from datetime import datetime


def get_logger(name: str, log_to_file: bool = True) -> logging.Logger:
    """
    Returns a logger with console + optional file output.
    
    Usage:
        from utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Fetching data...")
        logger.warning("Missing symbol")
        logger.error("API failed")
    """

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on re-import
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler — INFO and above
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler — DEBUG and above
    if log_to_file:
        os.makedirs("logs", exist_ok=True)
        today = datetime.now().strftime("%Y-%m-%d")
        file_handler = logging.FileHandler(f"logs/trading_{today}.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger