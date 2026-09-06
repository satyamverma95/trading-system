"""
advisory_agent/strategies/base.py
Abstract base class for all strategy implementations.

Every strategy must accept raw OHLCV data (lowercase columns) and
return a frozen TradeSetup dataclass — never a raw dict, never a string.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from advisory_agent.core.schemas import TradeSetup


class BaseStrategy(ABC):
    """
    Contract for all technical strategies in this agent.

    Implementors receive a clean OHLCV DataFrame and must return
    a fully populated TradeSetup. All numeric calculations must be
    deterministic and reproducible — no randomness, no LLM involvement.
    """

    @abstractmethod
    def evaluate(self, symbol: str, df: pd.DataFrame, interval: str) -> TradeSetup:
        """
        Run the strategy against historical OHLCV data.

        Args:
            symbol:   NSE trading symbol (e.g. "RELIANCE").
            df:       OHLCV DataFrame with lowercase columns:
                        open, high, low, close, volume
                      Index must be a DatetimeIndex (ascending, no gaps).
            interval: Human-readable label forwarded to TradeSetup
                        (e.g. "day", "15m", "1h").

        Returns:
            Frozen TradeSetup with signal, risk levels, indicators, rationale.

        Raises:
            ValueError: If df has insufficient rows or missing required columns.
        """
        ...
