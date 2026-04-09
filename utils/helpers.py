# =================================================================
# utils/helpers.py
# Common utility functions used across the system
# =================================================================

import yaml
import os
import pandas as pd
from datetime import datetime
from typing import List


def load_config(path: str = "config/settings.yaml") -> dict:
    """Load YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_all_symbols(config: dict) -> List[str]:
    """
    Returns flat list of all symbols from config universe.
    
    Usage:
        symbols = get_all_symbols(config)
        # ['RELIANCE.NS', 'TCS.NS', ...]
    """
    universe = config.get("universe", {})
    symbols = []
    for key, val in universe.items():
        if isinstance(val, list):
            symbols.extend(val)
    return list(set(symbols))  # deduplicate


def save_to_parquet(df: pd.DataFrame, symbol: str, folder: str) -> str:
    """
    Save a DataFrame to parquet format.
    Parquet is faster and smaller than CSV for time-series data.
    
    Returns the saved file path.
    """
    os.makedirs(folder, exist_ok=True)
    clean_symbol = symbol.replace(".", "_").replace("^", "idx_")
    filename = f"{clean_symbol}.parquet"
    filepath = os.path.join(folder, filename)
    df.to_parquet(filepath, index=True)
    return filepath


def load_from_parquet(symbol: str, folder: str) -> pd.DataFrame:
    """Load a previously saved symbol DataFrame from parquet."""
    clean_symbol = symbol.replace(".", "_").replace("^", "idx_")
    filepath = os.path.join(folder, f"{clean_symbol}.parquet")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No cached data found for {symbol} at {filepath}")
    return pd.read_parquet(filepath)


def is_market_open() -> bool:
    """
    Simple check if NSE market is currently open.
    NSE hours: Mon–Fri, 09:15 to 15:30 IST
    """
    import pytz
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)

    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False

    market_open  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

    return market_open <= now <= market_close


def format_currency(value: float) -> str:
    """Format a number as Indian currency string."""
    if value is None:
        return "N/A"
    if value >= 1e7:
        return f"₹{value/1e7:.2f} Cr"
    elif value >= 1e5:
        return f"₹{value/1e5:.2f} L"
    else:
        return f"₹{value:,.2f}"