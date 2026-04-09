# =================================================================
# analysis/indicators.py
# Technical indicators built on pandas-ta
# Every indicator returns the same DataFrame with new columns added
# =================================================================

import pandas as pd
import pandas_ta as ta
from utils.logger import get_logger

logger = get_logger(__name__)


class Indicators:
    """
    Adds technical indicators to OHLCV DataFrames.

    Usage:
        ind = Indicators(config)
        df  = ind.add_all(df)          # Add everything at once
        df  = ind.add_rsi(df)          # Or one at a time
    """

    def __init__(self, config: dict):
        cfg = config.get("indicators", {})
        self.rsi_period   = cfg.get("rsi_period", 14)
        self.macd_fast    = cfg.get("macd_fast", 12)
        self.macd_slow    = cfg.get("macd_slow", 26)
        self.macd_signal  = cfg.get("macd_signal", 9)
        self.bb_period    = cfg.get("bb_period", 20)
        self.bb_std       = cfg.get("bb_std", 2)
        self.atr_period   = cfg.get("atr_period", 14)
        self.sma_periods  = cfg.get("sma_periods", [20, 50, 200])
        self.ema_periods  = cfg.get("ema_periods", [9, 21, 55])

    # ----------------------------------------------------------
    # TREND INDICATORS
    # ----------------------------------------------------------

    def add_sma(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simple Moving Averages."""
        for period in self.sma_periods:
            df[f"SMA_{period}"] = ta.sma(df["Close"], length=period)
        logger.debug(f"Added SMA: {self.sma_periods}")
        return df

    def add_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Exponential Moving Averages."""
        for period in self.ema_periods:
            df[f"EMA_{period}"] = ta.ema(df["Close"], length=period)
        logger.debug(f"Added EMA: {self.ema_periods}")
        return df

    def add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        MACD — Moving Average Convergence Divergence.
        Adds: MACD, MACD_Signal, MACD_Hist
        """
        macd = ta.macd(
            df["Close"],
            fast=self.macd_fast,
            slow=self.macd_slow,
            signal=self.macd_signal
        )
        if macd is not None:
            df["MACD"]        = macd.iloc[:, 0]
            df["MACD_Signal"] = macd.iloc[:, 1]
            df["MACD_Hist"]   = macd.iloc[:, 2]
        logger.debug("Added MACD")
        return df

    def add_supertrend(self, df: pd.DataFrame, period: int = 7, multiplier: float = 3.0) -> pd.DataFrame:
        """
        Supertrend — excellent trend direction indicator.
        Adds: Supertrend, Supertrend_Direction (1=bullish, -1=bearish)
        """
        st = ta.supertrend(df["High"], df["Low"], df["Close"], length=period, multiplier=multiplier)
        if st is not None:
            df["Supertrend"]           = st.iloc[:, 0]
            df["Supertrend_Direction"] = st.iloc[:, 1]
        logger.debug("Added Supertrend")
        return df

    # ----------------------------------------------------------
    # MOMENTUM INDICATORS
    # ----------------------------------------------------------

    def add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        RSI — Relative Strength Index.
        >70 = overbought, <30 = oversold
        """
        df["RSI"] = ta.rsi(df["Close"], length=self.rsi_period)
        logger.debug(f"Added RSI({self.rsi_period})")
        return df

    def add_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stochastic Oscillator.
        Adds: Stoch_K, Stoch_D
        """
        stoch = ta.stoch(df["High"], df["Low"], df["Close"])
        if stoch is not None:
            df["Stoch_K"] = stoch.iloc[:, 0]
            df["Stoch_D"] = stoch.iloc[:, 1]
        logger.debug("Added Stochastic")
        return df

    def add_roc(self, df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """Rate of Change — momentum in percentage terms."""
        df[f"ROC_{period}"] = ta.roc(df["Close"], length=period)
        logger.debug(f"Added ROC({period})")
        return df

    # ----------------------------------------------------------
    # VOLATILITY INDICATORS
    # ----------------------------------------------------------

    def add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bollinger Bands.
        Adds: BB_Upper, BB_Mid, BB_Lower, BB_Width, BB_Percent
        """
        bb = ta.bbands(df["Close"], length=self.bb_period, std=self.bb_std)
        if bb is not None:
            df["BB_Lower"]   = bb.iloc[:, 0]
            df["BB_Mid"]     = bb.iloc[:, 1]
            df["BB_Upper"]   = bb.iloc[:, 2]
            df["BB_Width"]   = bb.iloc[:, 3]
            df["BB_Percent"] = bb.iloc[:, 4]
        logger.debug("Added Bollinger Bands")
        return df

    def add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ATR — Average True Range.
        Measures volatility. Essential for position sizing.
        """
        df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=self.atr_period)
        logger.debug(f"Added ATR({self.atr_period})")
        return df

    # ----------------------------------------------------------
    # VOLUME INDICATORS
    # ----------------------------------------------------------

    def add_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        VWAP — Volume Weighted Average Price.
        Key institutional reference price.
        """
        df["VWAP"] = ta.vwap(df["High"], df["Low"], df["Close"], df["Volume"])
        logger.debug("Added VWAP")
        return df

    def add_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """OBV — On Balance Volume. Tracks buying/selling pressure."""
        df["OBV"] = ta.obv(df["Close"], df["Volume"])
        logger.debug("Added OBV")
        return df

    def add_volume_sma(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Volume SMA — helps identify unusual volume spikes."""
        df[f"Vol_SMA_{period}"] = ta.sma(df["Volume"], length=period)
        df["Vol_Ratio"]         = df["Volume"] / df[f"Vol_SMA_{period}"]
        logger.debug(f"Added Volume SMA({period})")
        return df

    # ----------------------------------------------------------
    # SIGNAL FLAGS
    # ----------------------------------------------------------

    def add_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive simple buy/sell signal flags from indicators.
        These are starting points — not trading advice.
        
        Adds boolean columns:
            RSI_Oversold, RSI_Overbought,
            MACD_Crossover, MACD_Crossunder,
            Price_Above_SMA200, Golden_Cross, Death_Cross
        """
        if "RSI" in df.columns:
            df["RSI_Oversold"]   = df["RSI"] < 30
            df["RSI_Overbought"] = df["RSI"] > 70

        if "MACD" in df.columns and "MACD_Signal" in df.columns:
            df["MACD_Crossover"]  = (df["MACD"] > df["MACD_Signal"]) & \
                                    (df["MACD"].shift(1) <= df["MACD_Signal"].shift(1))
            df["MACD_Crossunder"] = (df["MACD"] < df["MACD_Signal"]) & \
                                    (df["MACD"].shift(1) >= df["MACD_Signal"].shift(1))

        if "SMA_50" in df.columns and "SMA_200" in df.columns:
            df["Price_Above_SMA200"] = df["Close"] > df["SMA_200"]
            df["Golden_Cross"] = (df["SMA_50"] > df["SMA_200"]) & \
                                 (df["SMA_50"].shift(1) <= df["SMA_200"].shift(1))
            df["Death_Cross"]  = (df["SMA_50"] < df["SMA_200"]) & \
                                 (df["SMA_50"].shift(1) >= df["SMA_200"].shift(1))

        logger.debug("Added signal flags")
        return df

    # ----------------------------------------------------------
    # ADD ALL AT ONCE
    # ----------------------------------------------------------

    def add_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all indicators in one shot.
        Use this for full analysis runs.
        """
        df = self.add_sma(df)
        df = self.add_ema(df)
        df = self.add_macd(df)
        df = self.add_rsi(df)
        df = self.add_bollinger_bands(df)
        df = self.add_atr(df)
        df = self.add_stochastic(df)
        df = self.add_supertrend(df)
        df = self.add_vwap(df)
        df = self.add_obv(df)
        df = self.add_volume_sma(df)
        df = self.add_roc(df)
        df = self.add_signals(df)
        logger.info(f"All indicators added. DataFrame shape: {df.shape}")
        return df