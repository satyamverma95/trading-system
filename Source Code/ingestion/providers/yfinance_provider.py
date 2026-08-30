# =================================================================
# providers/yfinance_provider.py
# yfinance implementation of BaseDataProvider
# Free, reliable for EOD data — Phase 1 workhorse
# =================================================================

import yfinance as yf
import pandas as pd
from typing import List, Optional
from providers.base import BaseDataProvider


class YFinanceProvider(BaseDataProvider):
    """
    yfinance-based data provider for NSE stocks.
    
    Symbol convention for NSE:
        Stocks  → RELIANCE.NS
        Indices → ^NSEI (Nifty 50), ^NSEBANK (Bank Nifty)
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.default_period   = config.get("historical", {}).get("default_period", "2y")
        self.default_interval = config.get("historical", {}).get("default_interval", "1d")

    # ----------------------------------------------------------
    # HISTORICAL DATA
    # ----------------------------------------------------------

    def get_historical_data(
        self,
        symbol: str,
        period: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a single NSE symbol.

        Usage:
            provider.get_historical_data("RELIANCE.NS", period="1y")
            provider.get_historical_data("TCS.NS", start="2023-01-01", end="2024-01-01")
        """
        try:
            ticker = yf.Ticker(symbol)

            if start:
                df = ticker.history(start=start, end=end, interval=interval)
            else:
                df = ticker.history(period=period or self.default_period, interval=interval)

            # Drop yfinance-specific columns we don't need
            df = df.drop(columns=["Dividends", "Stock Splits"], errors="ignore")

            # Validate and standardise
            df = self.validate_dataframe(df, symbol)
            df["Symbol"] = symbol

            return df

        except Exception as e:
            raise RuntimeError(f"Failed to fetch data for {symbol}: {e}")

    def get_bulk_historical_data(
        self,
        symbols: List[str],
        period: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d"
    ) -> dict:
        """
        Fetch OHLCV data for multiple symbols efficiently.
        Uses yfinance bulk download for speed.

        Returns:
            { "RELIANCE.NS": DataFrame, "TCS.NS": DataFrame, ... }
        """
        try:
            if start:
                raw = yf.download(
                    symbols,
                    start=start,
                    end=end,
                    interval=interval,
                    group_by="ticker",
                    auto_adjust=True,
                    progress=False
                )
            else:
                raw = yf.download(
                    symbols,
                    period=period or self.default_period,
                    interval=interval,
                    group_by="ticker",
                    auto_adjust=True,
                    progress=False
                )

            result = {}

            # Single symbol returns flat DataFrame — handle separately
            if len(symbols) == 1:
                df = raw.copy()
                df = df.drop(columns=["Dividends", "Stock Splits"], errors="ignore")
                df = self.validate_dataframe(df, symbols[0])
                df["Symbol"] = symbols[0]
                result[symbols[0]] = df
            else:
                for symbol in symbols:
                    try:
                        df = raw[symbol].copy()
                        df = df.dropna(how="all")
                        df = self.validate_dataframe(df, symbol)
                        df["Symbol"] = symbol
                        result[symbol] = df
                    except Exception as inner_e:
                        print(f"⚠️  Skipping {symbol}: {inner_e}")

            print(f"✅ Fetched data for {len(result)}/{len(symbols)} symbols")
            return result

        except Exception as e:
            raise RuntimeError(f"Bulk download failed: {e}")

    # ----------------------------------------------------------
    # LIVE QUOTES
    # ----------------------------------------------------------

    def get_quote(self, symbol: str) -> dict:
        """
        Get latest available price info for a symbol.
        Note: yfinance is not a true real-time feed.
        Use Zerodha for live ticks in Phase 3.
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                "symbol"      : symbol,
                "last_price"  : info.get("currentPrice") or info.get("regularMarketPrice"),
                "open"        : info.get("open") or info.get("regularMarketOpen"),
                "high"        : info.get("dayHigh") or info.get("regularMarketDayHigh"),
                "low"         : info.get("dayLow") or info.get("regularMarketDayLow"),
                "prev_close"  : info.get("previousClose"),
                "volume"      : info.get("volume") or info.get("regularMarketVolume"),
                "market_cap"  : info.get("marketCap"),
                "pe_ratio"    : info.get("trailingPE"),
                "52w_high"    : info.get("fiftyTwoWeekHigh"),
                "52w_low"     : info.get("fiftyTwoWeekLow"),
            }

        except Exception as e:
            raise RuntimeError(f"Failed to fetch quote for {symbol}: {e}")

    def get_bulk_quotes(self, symbols: List[str]) -> dict:
        """Get quotes for multiple symbols."""
        result = {}
        for symbol in symbols:
            try:
                result[symbol] = self.get_quote(symbol)
            except Exception as e:
                print(f"⚠️  Quote failed for {symbol}: {e}")
        return result

    # ----------------------------------------------------------
    # INSTRUMENT INFO
    # ----------------------------------------------------------

    def get_instrument_info(self, symbol: str) -> dict:
        """Get company/instrument metadata."""
        try:
            info = yf.Ticker(symbol).info
            return {
                "symbol"     : symbol,
                "name"       : info.get("longName") or info.get("shortName"),
                "exchange"   : info.get("exchange"),
                "sector"     : info.get("sector"),
                "industry"   : info.get("industry"),
                "market_cap" : info.get("marketCap"),
                "pe_ratio"   : info.get("trailingPE"),
                "pb_ratio"   : info.get("priceToBook"),
                "dividend"   : info.get("dividendYield"),
                "beta"       : info.get("beta"),
                "description": info.get("longBusinessSummary"),
            }
        except Exception as e:
            raise RuntimeError(f"Failed to fetch info for {symbol}: {e}")

    # ----------------------------------------------------------
    # DERIVATIVES (Phase 2 — partial yfinance support)
    # ----------------------------------------------------------

    def get_options_chain(
        self,
        symbol: str,
        expiry: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch options chain for a symbol.
        yfinance supports this for NSE — limited but usable for Phase 2 exploration.
        """
        try:
            ticker = yf.Ticker(symbol)
            expiries = ticker.options

            if not expiries:
                raise ValueError(f"No options data available for {symbol}")

            # Use provided expiry or nearest one
            target_expiry = expiry if expiry in expiries else expiries[0]
            chain = ticker.option_chain(target_expiry)

            calls = chain.calls.copy()
            puts  = chain.puts.copy()

            calls["option_type"] = "CE"
            puts["option_type"]  = "PE"

            combined = pd.concat([calls, puts], ignore_index=True)
            combined["expiry"] = target_expiry
            combined["symbol"] = symbol

            return combined

        except Exception as e:
            raise RuntimeError(f"Failed to fetch options chain for {symbol}: {e}")