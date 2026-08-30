# =================================================================
# source_code/ingestion/nifty_loader.py
# LAYER 1: DATA EXTRACTION - Nifty Universe Loader
# Reads official Nifty 100 CSV and extracts trading symbols
# =================================================================

import csv
from pathlib import Path
from typing import List, Optional
import logging

from source_code.common.path_resolver import resolve_path, get_project_root
from source_code.common.config_loader import load_config

logger = logging.getLogger(__name__)


class NiftyUniverseLoader:
    """
    Loads Nifty Index constituents from an official CSV file.
    
    Typical CSV format (as provided by NSE):
    - Column 1: Symbol (e.g., "RELIANCE", "HDFCBANK")
    - Column 2: Company Name
    - Column 3: Industry
    - Additional columns: Weight, etc.
    
    Handles:
    - File not found errors
    - Missing or malformed CSV rows
    - Symbol normalization (removes spaces, handles aliases)
    - Caching of loaded universe
    """

    def __init__(self, csv_path: Optional[str] = None, config: Optional[dict] = None):
        """
        Initialize the universe loader.
        
        Args:
            csv_path: Path to Nifty 100 CSV file. 
                      If None, looks for 'nifty100.csv' in data/input/
            config: Config dict. If None, loads from settings.yaml
        """
        self.config = config or load_config()
        self._csv_path = csv_path
        self._symbols_cache: Optional[List[str]] = None

    def _get_csv_path(self) -> Path:
        """Resolve CSV file path."""
        if self._csv_path:
            return resolve_path(self._csv_path)
        
        # Default: look in data/input/nifty100.csv
        root = get_project_root()
        default_path = root / "data" / "input" / "nifty100.csv"
        return default_path

    def load(self, force_reload: bool = False) -> List[str]:
        """
        Load and return list of Nifty 100 symbols.
        
        Args:
            force_reload: If True, reload from disk even if cached
            
        Returns:
            List of stock symbols (e.g., ["RELIANCE", "HDFCBANK", "INFY", ...])
            
        Raises:
            FileNotFoundError: If CSV file not found
            ValueError: If CSV is malformed or no symbols extracted
        """
        if self._symbols_cache is not None and not force_reload:
            return self._symbols_cache

        csv_path = self._get_csv_path()
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Nifty 100 CSV not found at {csv_path}\n"
                f"Please provide a CSV with stock symbols in the first column."
            )

        symbols = self._parse_csv(csv_path)
        
        if not symbols:
            raise ValueError(f"No valid symbols extracted from {csv_path}")

        self._symbols_cache = symbols
        logger.info(f"Loaded {len(symbols)} symbols from {csv_path}")
        return symbols

    def _parse_csv(self, csv_path: Path) -> List[str]:
        """
        Parse CSV and extract symbols from first column.
        
        Handles:
        - Header rows (skips them)
        - Whitespace and normalization
        - Empty or invalid rows
        """
        symbols = []
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                
                if header is None:
                    logger.warning(f"CSV is empty: {csv_path}")
                    return []
                
                # Skip header if it looks like a header row
                first_row = header
                if self._looks_like_header(first_row):
                    logger.debug(f"Detected header row: {first_row}")
                else:
                    # Not a header, process it as data
                    symbol = self._extract_symbol(first_row[0])
                    if symbol:
                        symbols.append(symbol)
                
                # Process remaining rows
                for row in reader:
                    if not row or not row[0].strip():
                        continue
                    
                    symbol = self._extract_symbol(row[0])
                    if symbol:
                        symbols.append(symbol)
        
        except csv.Error as e:
            logger.error(f"CSV parsing error in {csv_path}: {e}")
            raise ValueError(f"Failed to parse CSV: {e}")

        return symbols

    @staticmethod
    def _looks_like_header(row: List[str]) -> bool:
        """
        Heuristic to detect if a row is a header.
        Headers typically contain words like "Symbol", "Name", "Company", etc.
        """
        if not row:
            return False
        
        first_cell = row[0].lower()
        header_keywords = ["symbol", "ticker", "code", "name", "company", "industry"]
        
        return any(keyword in first_cell for keyword in header_keywords)

    @staticmethod
    def _extract_symbol(cell: str) -> Optional[str]:
        """
        Extract and normalize a symbol from a cell.
        
        - Strips whitespace
        - Converts to uppercase
        - Filters out invalid/non-tradable entries
        - Removes common suffixes like (.NS) if present
        
        Returns:
            Normalized symbol or None if invalid
        """
        if not isinstance(cell, str):
            return None
        
        symbol = cell.strip().upper()
        
        # Skip empty or too-short symbols
        if not symbol or len(symbol) < 2:
            return None
        
        # Remove common exchange suffixes (e.g., ".NS", ".BO")
        for suffix in [".NS", ".BO", ".BSE", ".NSE"]:
            if symbol.endswith(suffix):
                symbol = symbol[:-len(suffix)]
        
        # Skip if symbol contains invalid characters
        if not all(c.isalnum() or c in ['-', '_', '&'] for c in symbol):
            return None
        
        return symbol

    def get_symbols(self) -> List[str]:
        """
        Convenience method: get symbols (loads if not already cached).
        Same as load() but uses cache if available.
        """
        return self.load()

    def validate_symbols(self, symbols: List[str]) -> tuple[List[str], List[str]]:
        """
        Validate a list of symbols.
        Returns (valid_symbols, invalid_symbols)
        
        Useful for debugging malformed CSV entries.
        """
        valid, invalid = [], []
        for sym in symbols:
            normalized = self._extract_symbol(sym)
            if normalized:
                valid.append(normalized)
            else:
                invalid.append(sym)
        return valid, invalid


def load_nifty100_universe(csv_path: Optional[str] = None) -> List[str]:
    """
    Convenience function to load Nifty 100 universe in one call.
    
    Usage:
        symbols = load_nifty100_universe()
        # Returns: ["RELIANCE", "HDFCBANK", "INFY", ...]
    """
    loader = NiftyUniverseLoader(csv_path=csv_path)
    return loader.load()
