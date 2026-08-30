# =================================================================
# analysis/screener.py
# Composite scored screener across all symbols
# Loads saved parquet files and ranks by combined signal score
# python run_screener.py
# =================================================================

import os
import pandas as pd
import numpy as np
from utils.helpers import load_config, load_from_parquet, get_all_symbols
from utils.logger import get_logger

logger = get_logger(__name__)


class Screener:
    """
    Loads processed parquet files and scores each symbol
    on a composite of RSI, MACD, trend, volume signals.

    Scoring logic (each component 0-100, weighted average):
        RSI Score       : 30% — oversold = bullish, overbought = bearish
        MACD Score      : 25% — crossover direction and distance
        Trend Score     : 25% — price vs SMA/EMA alignment
        Volume Score    : 20% — volume ratio vs average
    """

    WEIGHTS = {
        "rsi_score"    : 0.30,
        "macd_score"   : 0.25,
        "trend_score"  : 0.25,
        "volume_score" : 0.20,
    }

    def __init__(self, config: dict):
        self.config   = config
        self.data_dir = config["paths"]["processed_data"]

    # ----------------------------------------------------------
    # LOAD DATA
    # ----------------------------------------------------------

    def load_symbol(self, symbol: str, interval: str) -> pd.DataFrame:
        """Load processed parquet for a symbol."""
        return load_from_parquet(symbol, self.data_dir, interval)

    # ----------------------------------------------------------
    # INDIVIDUAL SCORES (0-100)
    # ----------------------------------------------------------

    def _rsi_score(self, latest: pd.Series) -> float:
        """
        RSI Score:
            RSI < 30  → score 90-100 (strong buy zone)
            RSI 30-40 → score 70-90  (approaching oversold)
            RSI 40-50 → score 50-70  (neutral-bullish)
            RSI 50-60 → score 30-50  (neutral-bearish)
            RSI 60-70 → score 10-30  (approaching overbought)
            RSI > 70  → score 0-10   (overbought, risky)
        """
        rsi = latest.get("RSI")
        if rsi is None or pd.isna(rsi):
            return 50.0
        # Invert RSI for buy scoring — lower RSI = higher score
        score = 100 - rsi
        return float(np.clip(score, 0, 100))

    def _macd_score(self, df: pd.DataFrame) -> float:
        """
        MACD Score based on:
            - MACD vs Signal direction (bullish/bearish)
            - Recent crossover (extra points)
            - MACD histogram direction
        """
        latest = df.iloc[-1]
        macd     = latest.get("MACD")
        signal   = latest.get("MACD_Signal")
        hist     = latest.get("MACD_Hist")
        crossover  = latest.get("MACD_Crossover", False)
        crossunder = latest.get("MACD_Crossunder", False)

        if any(pd.isna(v) for v in [macd, signal] if v is not None):
            return 50.0

        score = 50.0  # neutral base

        try:
            # Direction
            if macd > signal:
                score += 20
            else:
                score -= 20

            # Recent crossover bonus
            if crossover:
                score += 20
            elif crossunder:
                score -= 20

            # Histogram trending up
            if hist is not None and not pd.isna(hist):
                prev_hist = df.iloc[-2].get("MACD_Hist")
                if prev_hist is not None and not pd.isna(prev_hist):
                    if hist > prev_hist:
                        score += 10
                    else:
                        score -= 10
        except:
            pass

        return float(np.clip(score, 0, 100))

    def _trend_score(self, latest: pd.Series) -> float:
        """
        Trend Score based on price vs moving averages.
        Each MA above = +points, below = -points
        """
        close = latest.get("Close")
        if close is None or pd.isna(close):
            return 50.0

        score  = 50.0
        checks = [
            ("EMA_9",   8),
            ("EMA_21",  10),
            ("EMA_55",  12),
            ("SMA_20",  8),
            ("SMA_50",  12),
            ("SMA_100", 10),
        ]

        total_weight = sum(w for _, w in checks)
        earned       = 0

        for col, weight in checks:
            val = latest.get(col)
            if val is not None and not pd.isna(val):
                try:
                    if close > val:
                        earned += weight
                except:
                    pass

        # Normalize to 0-100
        score = (earned / total_weight) * 100
        return float(np.clip(score, 0, 100))

    def _volume_score(self, latest: pd.Series) -> float:
        """
        Volume Score based on Vol_Ratio (current vol / avg vol).
            > 2.0x  → 90-100 (very high volume)
            1.5-2x  → 70-90
            1.0-1.5 → 50-70
            0.5-1.0 → 30-50
            < 0.5   → 0-30  (very low volume)
        """
        vol_ratio = latest.get("Vol_Ratio")
        if vol_ratio is None or pd.isna(vol_ratio):
            return 50.0
        try:
            score = float(np.clip(vol_ratio * 40, 0, 100))
        except:
            score = 50.0
        return score

    # ----------------------------------------------------------
    # COMPOSITE SCORE
    # ----------------------------------------------------------

    def score_symbol(self, symbol: str, interval: str) -> dict:
        """Compute composite score for a single symbol."""
        try:
            df     = self.load_symbol(symbol, interval)
            latest = df.iloc[-1]
            prev   = df.iloc[-2]

            rsi_s    = self._rsi_score(latest)
            macd_s   = self._macd_score(df)
            trend_s  = self._trend_score(latest)
            volume_s = self._volume_score(latest)

            composite = (
                rsi_s    * self.WEIGHTS["rsi_score"]    +
                macd_s   * self.WEIGHTS["macd_score"]   +
                trend_s  * self.WEIGHTS["trend_score"]  +
                volume_s * self.WEIGHTS["volume_score"]
            )

            change_pct = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100

            return {
                "Symbol"       : symbol,
                "Interval"     : interval,
                "Close"        : round(latest["Close"], 2),
                "Change%"      : round(change_pct, 2),
                "RSI"          : round(latest.get("RSI", float("nan")), 1),
                "RSI_Score"    : round(rsi_s, 1),
                "MACD_Score"   : round(macd_s, 1),
                "Trend_Score"  : round(trend_s, 1),
                "Volume_Score" : round(volume_s, 1),
                "Composite"    : round(composite, 1),
                "Signal"       : self._label(composite),
            }

        except FileNotFoundError:
            logger.warning(f"No parquet found for {symbol} ({interval}) — run run_analysis.py first")
            return None
        except Exception as e:
            logger.error(f"Scoring failed for {symbol}: {e}")
            return None

    def _label(self, score: float) -> str:
        """Convert composite score to human-readable signal."""
        if score >= 75:   return "🟢 Strong Buy"
        elif score >= 60: return "🟩 Buy"
        elif score >= 45: return "⚪ Neutral"
        elif score >= 30: return "🟥 Sell"
        else:             return "🔴 Strong Sell"

    # ----------------------------------------------------------
    # SCREEN ALL
    # ----------------------------------------------------------

    def screen(self, symbols: list, interval: str) -> pd.DataFrame:
        """Score all symbols and return ranked DataFrame."""
        results = []
        for symbol in symbols:
            row = self.score_symbol(symbol, interval)
            if row:
                results.append(row)

        if not results:
            logger.warning("No results to screen")
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df = df.sort_values("Composite", ascending=False).reset_index(drop=True)
        df.index += 1  # rank starts at 1
        return df