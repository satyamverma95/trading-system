# =================================================================
# analysis/indicators.py
# Technical indicators built on pandas-ta
# Fully NaN-safe, compatible with pandas 2.x
# =================================================================

import pandas as pd
import pandas_ta as ta
from utils.logger import get_logger

logger = get_logger(__name__)


class Indicators:

    def __init__(self, config: dict):
        cfg = config.get("indicators", {})
        self.rsi_period   = cfg.get("rsi_period", 14)
        self.macd_fast    = cfg.get("macd_fast", 12)
        self.macd_slow    = cfg.get("macd_slow", 26)
        self.macd_signal  = cfg.get("macd_signal", 9)
        self.bb_period    = cfg.get("bb_period", 20)
        self.bb_std       = cfg.get("bb_std", 2)
        self.atr_period   = cfg.get("atr_period", 14)
        self.sma_periods  = cfg.get("sma_periods", [20, 50, 100])
        self.ema_periods  = cfg.get("ema_periods", [9, 21, 55])

    def _safe_assign(self, df, col, series):
        if series is None:
            df[col] = float("nan")
        else:
            df[col] = pd.to_numeric(series, errors="coerce")
        return df

    def _safe_gt(self, a, b):
        try:
            a = pd.to_numeric(a, errors="coerce")
            b = pd.to_numeric(b, errors="coerce")
            return (a > b).fillna(False).astype(bool)
        except:
            return pd.Series([False] * len(a), index=a.index)

    def _safe_lt(self, a, b):
        try:
            a = pd.to_numeric(a, errors="coerce")
            b = pd.to_numeric(b, errors="coerce")
            return (a < b).fillna(False).astype(bool)
        except:
            return pd.Series([False] * len(a), index=a.index)

    def add_sma(self, df):
        for period in self.sma_periods:
            if len(df) >= period:
                df = self._safe_assign(df, f"SMA_{period}", ta.sma(df["Close"], length=period))
            else:
                df[f"SMA_{period}"] = float("nan")
                logger.warning(f"Not enough bars for SMA_{period} (need {period}, got {len(df)})")
        return df

    def add_ema(self, df):
        for period in self.ema_periods:
            if len(df) >= period:
                df = self._safe_assign(df, f"EMA_{period}", ta.ema(df["Close"], length=period))
            else:
                df[f"EMA_{period}"] = float("nan")
        return df

    def add_macd(self, df):
        try:
            macd = ta.macd(df["Close"], fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
            if macd is not None and not macd.empty:
                df["MACD"]        = pd.to_numeric(macd.iloc[:, 0], errors="coerce")
                df["MACD_Signal"] = pd.to_numeric(macd.iloc[:, 1], errors="coerce")
                df["MACD_Hist"]   = pd.to_numeric(macd.iloc[:, 2], errors="coerce")
            else:
                df["MACD"] = df["MACD_Signal"] = df["MACD_Hist"] = float("nan")
        except Exception as e:
            logger.warning(f"MACD failed: {e}")
            df["MACD"] = df["MACD_Signal"] = df["MACD_Hist"] = float("nan")
        return df

    def add_rsi(self, df):
        try:
            df = self._safe_assign(df, "RSI", ta.rsi(df["Close"], length=self.rsi_period))
        except Exception as e:
            logger.warning(f"RSI failed: {e}")
            df["RSI"] = float("nan")
        return df

    def add_bollinger_bands(self, df):
        try:
            bb = ta.bbands(df["Close"], length=self.bb_period, std=self.bb_std)
            if bb is not None and not bb.empty:
                df["BB_Lower"]   = pd.to_numeric(bb.iloc[:, 0], errors="coerce")
                df["BB_Mid"]     = pd.to_numeric(bb.iloc[:, 1], errors="coerce")
                df["BB_Upper"]   = pd.to_numeric(bb.iloc[:, 2], errors="coerce")
                df["BB_Width"]   = pd.to_numeric(bb.iloc[:, 3], errors="coerce")
                df["BB_Percent"] = pd.to_numeric(bb.iloc[:, 4], errors="coerce")
            else:
                for col in ["BB_Lower","BB_Mid","BB_Upper","BB_Width","BB_Percent"]:
                    df[col] = float("nan")
        except Exception as e:
            logger.warning(f"BB failed: {e}")
            for col in ["BB_Lower","BB_Mid","BB_Upper","BB_Width","BB_Percent"]:
                df[col] = float("nan")
        return df

    def add_atr(self, df):
        try:
            df = self._safe_assign(df, "ATR", ta.atr(df["High"], df["Low"], df["Close"], length=self.atr_period))
        except Exception as e:
            logger.warning(f"ATR failed: {e}")
            df["ATR"] = float("nan")
        return df

    def add_stochastic(self, df):
        try:
            stoch = ta.stoch(df["High"], df["Low"], df["Close"])
            if stoch is not None and not stoch.empty:
                df["Stoch_K"] = pd.to_numeric(stoch.iloc[:, 0], errors="coerce")
                df["Stoch_D"] = pd.to_numeric(stoch.iloc[:, 1], errors="coerce")
            else:
                df["Stoch_K"] = df["Stoch_D"] = float("nan")
        except Exception as e:
            logger.warning(f"Stochastic failed: {e}")
            df["Stoch_K"] = df["Stoch_D"] = float("nan")
        return df

    def add_supertrend(self, df, period=7, multiplier=3.0):
        try:
            st = ta.supertrend(df["High"], df["Low"], df["Close"], length=period, multiplier=multiplier)
            if st is not None and not st.empty:
                df["Supertrend"]           = pd.to_numeric(st.iloc[:, 0], errors="coerce")
                df["Supertrend_Direction"] = pd.to_numeric(st.iloc[:, 1], errors="coerce")
            else:
                df["Supertrend"] = df["Supertrend_Direction"] = float("nan")
        except Exception as e:
            logger.warning(f"Supertrend failed: {e}")
            df["Supertrend"] = df["Supertrend_Direction"] = float("nan")
        return df

    def add_vwap(self, df):
        try:
            df = self._safe_assign(df, "VWAP", ta.vwap(df["High"], df["Low"], df["Close"], df["Volume"]))
        except Exception as e:
            logger.warning(f"VWAP failed: {e}")
            df["VWAP"] = float("nan")
        return df

    def add_obv(self, df):
        try:
            df = self._safe_assign(df, "OBV", ta.obv(df["Close"], df["Volume"]))
        except Exception as e:
            logger.warning(f"OBV failed: {e}")
            df["OBV"] = float("nan")
        return df

    def add_volume_sma(self, df, period=20):
        try:
            vol_sma = ta.sma(df["Volume"], length=period)
            df = self._safe_assign(df, f"Vol_SMA_{period}", vol_sma)
            df["Vol_Ratio"] = pd.to_numeric(df["Volume"], errors="coerce") / pd.to_numeric(df[f"Vol_SMA_{period}"], errors="coerce")
        except Exception as e:
            logger.warning(f"Volume SMA failed: {e}")
            df[f"Vol_SMA_{period}"] = df["Vol_Ratio"] = float("nan")
        return df

    def add_roc(self, df, period=10):
        try:
            df = self._safe_assign(df, f"ROC_{period}", ta.roc(df["Close"], length=period))
        except Exception as e:
            logger.warning(f"ROC failed: {e}")
            df[f"ROC_{period}"] = float("nan")
        return df

    def add_signals(self, df):
        # RSI signals
        if "RSI" in df.columns:
            rsi = pd.to_numeric(df["RSI"], errors="coerce")
            df["RSI_Oversold"]   = rsi.apply(lambda x: bool(x < 30) if pd.notna(x) else False)
            df["RSI_Overbought"] = rsi.apply(lambda x: bool(x > 70) if pd.notna(x) else False)
        else:
            df["RSI_Oversold"] = df["RSI_Overbought"] = False

        # MACD signals
        if "MACD" in df.columns and "MACD_Signal" in df.columns:
            macd   = pd.to_numeric(df["MACD"],        errors="coerce").ffill()
            signal = pd.to_numeric(df["MACD_Signal"], errors="coerce").ffill()
            df["MACD_Crossover"]  = (self._safe_gt(macd, signal) & self._safe_lt(macd.shift(1), signal.shift(1)))
            df["MACD_Crossunder"] = (self._safe_lt(macd, signal) & self._safe_gt(macd.shift(1), signal.shift(1)))
        else:
            df["MACD_Crossover"] = df["MACD_Crossunder"] = False

        # SMA signals
        sma50_col  = "SMA_50"  if "SMA_50"  in df.columns else None
        sma100_col = "SMA_100" if "SMA_100" in df.columns else None
        sma_long   = sma100_col or "SMA_200" if "SMA_200" in df.columns else None

        if sma50_col and sma_long:
            close  = pd.to_numeric(df["Close"],        errors="coerce")
            sma50  = pd.to_numeric(df[sma50_col],      errors="coerce").ffill()
            sma200 = pd.to_numeric(df[sma_long],       errors="coerce").ffill()
            df["Price_Above_SMA_Long"] = self._safe_gt(close, sma200)
            df["Golden_Cross"]         = self._safe_gt(sma50, sma200) & self._safe_lt(sma50.shift(1), sma200.shift(1))
            df["Death_Cross"]          = self._safe_lt(sma50, sma200) & self._safe_gt(sma50.shift(1), sma200.shift(1))
        else:
            df["Price_Above_SMA_Long"] = df["Golden_Cross"] = df["Death_Cross"] = False

        logger.debug("Added signal flags")
        return df

    def add_all(self, df):
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
        logger.info(f"All indicators added. Shape: {df.shape}")
        return df
