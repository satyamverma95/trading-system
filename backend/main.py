"""FastAPI backend for Zerodha authentication, profile, and signals."""

import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from kiteconnect import KiteConnect

from source_code.common.config_loader import load_config
from source_code.ingestion.auth.session_manager import save_session
from source_code.ingestion.batch_fetcher import BatchCandleFetcher
from source_code.ingestion.nifty_loader import load_nifty100_universe
from source_code.processing.analysis.crossover_detector import CrossoverDetector
from source_code.processing.analysis.sma_calculator import SMACalculator

app = FastAPI(title="Nifty Signal Desk API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoginRequest(BaseModel):
    api_key: str = Field(min_length=1)
    api_secret: str = Field(min_length=1)
    request_token: str = Field(min_length=1)


class SignalRequest(BaseModel):
    short_sma: int = Field(default=6, ge=1, le=200)
    long_sma: int = Field(default=30, ge=2, le=500)
    lookback_days: int = Field(default=365, ge=30, le=3650)
    max_stocks: int = Field(default=20, ge=1, le=100)


class AnalyzeRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=30, description="NSE trading symbol e.g. RELIANCE")
    interval: str = Field(default="day", description="Candle interval: day, 1h, 15m, 5m")
    lookback_days: Optional[int] = Field(default=None, description="Override default lookback")


class ScreenerRequest(BaseModel):
    universe: str = Field(default="nifty100", description="Universe: nifty100 or nifty50")
    interval: str = Field(default="day", description="Candle interval: day")
    max_stocks: int = Field(default=100, ge=10, le=200)
    lookback_days: int = Field(default=180, ge=60, le=365)


def _config() -> dict:
    return load_config().get("zerodha", {})


def _kite() -> KiteConnect:
    config = _config()
    api_key = config.get("api_key", "").strip()
    access_token = config.get("access_token", "").strip()
    if not api_key or not access_token:
        raise HTTPException(status_code=401, detail="Zerodha is not connected")
    client = KiteConnect(api_key=api_key)
    client.set_access_token(access_token)
    return client


def _safe(value):
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value.item() if hasattr(value, "item") else value


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/auth/login-url")
def login_url():
    config = _config()
    if not config.get("api_key"):
        raise HTTPException(status_code=400, detail="API key is not configured")
    return {"url": KiteConnect(api_key=config["api_key"]).login_url()}


@app.get("/api/auth/callback")
def auth_callback(request: Request):
    """Receive Zerodha's browser redirect and return the user to React."""
    request_token = request.query_params.get("request_token")
    status = request.query_params.get("status")
    frontend_url = "http://127.0.0.1:5173/"
    if status != "success" or not request_token:
        return RedirectResponse(f"{frontend_url}?auth=failed&reason=zerodha_login_cancelled")
    return _finish_login(request_token, frontend_url)


@app.get("/")
def legacy_auth_callback(request: Request):
    """Compatibility callback for Kite apps still configured to use port 5000."""
    request_token = request.query_params.get("request_token")
    status = request.query_params.get("status")
    frontend_url = "http://127.0.0.1:5173/"
    if status != "success" or not request_token:
        return RedirectResponse(f"{frontend_url}?auth=failed&reason=zerodha_login_cancelled")
    return _finish_login(request_token, frontend_url)


def _finish_login(request_token: str, frontend_url: str):
    try:
        config = _config()
        client = KiteConnect(api_key=config["api_key"])
        session = client.generate_session(request_token, api_secret=config["api_secret"])
        save_session(session["access_token"])
        return RedirectResponse(f"{frontend_url}?auth=success")
    except Exception:
        return RedirectResponse(f"{frontend_url}?auth=failed&reason=token_exchange_failed")


@app.post("/api/auth/login")
def login(payload: LoginRequest):
    try:
        client = KiteConnect(api_key=payload.api_key.strip())
        session = client.generate_session(payload.request_token.strip(), api_secret=payload.api_secret.strip())
        save_session(session["access_token"])
        return {"connected": True, "message": "Zerodha connected"}
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Zerodha login failed: {exc}") from exc


@app.get("/api/auth/status")
def auth_status():
    try:
        profile = _kite().profile()
        return {"connected": True, "message": f"Connected as {profile.get('user_name', profile.get('user_id', 'Zerodha user'))}"}
    except Exception as exc:
        message = str(exc).lower()
        return {"connected": False, "expired": "token" in message or "access" in message, "message": "Token expired" if "token" in message or "access" in message else "Not connected"}


@app.get("/api/profile")
def profile():
    try:
        data = _kite().profile()
        return {"user_name": data.get("user_name"), "user_id": data.get("user_id"), "products": data.get("products", []), "exchanges": data.get("exchanges", [])}
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Unable to load profile: {exc}") from exc


@app.post("/api/signals")
def signals(payload: SignalRequest):
    if payload.short_sma >= payload.long_sma:
        raise HTTPException(status_code=400, detail="Short SMA must be smaller than Long SMA")
    try:
        config = load_config()
        symbols = load_nifty100_universe()
        fetcher = BatchCandleFetcher(config, provider="zerodha")
        data = fetcher.fetch_batch(symbols, period=f"{payload.lookback_days}d", interval="1d", skip_missing=True)
        if not data:
            raise HTTPException(status_code=502, detail="No market data was fetched")
        sma_data = SMACalculator(config).process_batch(data, windows=[payload.short_sma, payload.long_sma])
        crossover_data = CrossoverDetector(config, payload.short_sma, payload.long_sma).process_batch(sma_data)
        rows = []
        cutoff = datetime.now() - timedelta(days=payload.lookback_days)
        for symbol, dataframe in crossover_data.items():
            for index in reversed(dataframe.index):
                row = dataframe.loc[index]
                signal = row.get("Crossover_Signal")
                if signal not in ("BULLISH", "BEARISH"):
                    continue
                signal_date = pd.Timestamp(index).to_pydatetime().replace(tzinfo=None)
                if signal_date < cutoff:
                    break
                rows.append({
                    "ticker": symbol,
                    "company": symbol,
                    "crossover_type": signal,
                    "crossover_date": signal_date.strftime("%Y-%m-%d"),
                    "close": _safe(row.get("Close")),
                    "short_sma": _safe(row.get(f"SMA_{payload.short_sma}")),
                    "long_sma": _safe(row.get(f"SMA_{payload.long_sma}")),
                    "_date": signal_date,
                })
        rows.sort(key=lambda row: row["_date"], reverse=True)
        for rank, row in enumerate(rows[:payload.max_stocks], 1):
            row["rank"] = rank
            row.pop("_date")
        return {"requested": len(symbols), "fetched": len(data), "results": rows[:payload.max_stocks]}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Signal generation failed: {exc}") from exc


# ── Interval → Zerodha interval string + lookback days ───────────────────────

_INTERVAL_MAP = {
    "day":   ("day",       180),
    "1d":    ("day",       180),
    "1h":    ("60minute",   30),
    "60m":   ("60minute",   30),
    "30m":   ("30minute",   20),
    "15m":   ("15minute",   10),
    "5m":    ("5minute",    10),
}


@app.post("/api/analyze")
def analyze(payload: AnalyzeRequest):
    """
    Full technical advisory for a single NSE symbol.

    Runs all 5 analysis dimensions (trend, momentum, volatility, volume,
    structure), classifies the swing trade setup, fetches market context
    (VIX, FII/DII, news), and generates a Gemini-powered advisory narrative.

    Returns the complete advisory card ready for the React dashboard.
    """
    import asyncio
    from datetime import date, timedelta

    symbol   = payload.symbol.upper().strip()
    interval = payload.interval.lower().strip()

    kite_interval, default_lookback = _INTERVAL_MAP.get(interval, ("day", 180))
    lookback_days = payload.lookback_days or default_lookback

    try:
        kite = _kite()

        # ── 1. Resolve instrument token ───────────────────────────────────────
        try:
            instruments = kite.instruments("NSE")
            token_map   = {
                inst["tradingsymbol"]: inst["instrument_token"]
                for inst in instruments
            }
            token = token_map.get(symbol)
            if not token:
                raise HTTPException(
                    status_code=404,
                    detail=f"Symbol '{symbol}' not found on NSE. Check spelling."
                )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Instrument lookup failed: {exc}") from exc

        # ── 2. Fetch OHLCV candles ────────────────────────────────────────────
        from_date = date.today() - timedelta(days=lookback_days)
        to_date   = date.today()

        try:
            raw_candles = kite.historical_data(
                instrument_token=token,
                from_date=str(from_date),
                to_date=str(to_date),
                interval=kite_interval,
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Candle data fetch failed: {exc}") from exc

        if not raw_candles:
            raise HTTPException(status_code=502, detail=f"No candle data returned for {symbol}")

        # Build DataFrame (Zerodha returns lowercase keys)
        df = pd.DataFrame(raw_candles)
        df = df.rename(columns={
            "date":   "Datetime",
            "open":   "Open",
            "high":   "High",
            "low":    "Low",
            "close":  "Close",
            "volume": "Volume",
        })
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df.set_index("Datetime").sort_index()

        # ── 3. Run composite analysis ─────────────────────────────────────────
        from advisory_agent.analysis.composite import build_snapshot
        from advisory_agent.strategies.classifier import classify

        try:
            snapshot = build_snapshot(df, symbol, interval)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        classification = classify(snapshot)

        # ── 4. Fetch market context (non-blocking, best-effort) ───────────────
        from advisory_agent.context.vix_fetcher     import fetch_vix
        from advisory_agent.context.fii_dii_fetcher import fetch_fii_dii
        from advisory_agent.context.news_fetcher    import fetch_news

        context = {
            "vix":     fetch_vix(kite),
            "fii_dii": fetch_fii_dii(),
            "news":    fetch_news(symbol),
        }

        # ── 5. Generate advisory narrative (Gemini or rule-based) ────────────
        from advisory_agent.intelligence.advisor import generate_advisory
        advisory = generate_advisory(snapshot, classification, context)

        # ── 6. Assemble final response ────────────────────────────────────────
        return {
            "symbol":           snapshot["symbol"],
            "interval":         snapshot["interval"],
            "ltp":              snapshot["ltp"],
            "candle_count":     snapshot["candle_count"],
            "computed_at":      snapshot["computed_at"],

            "signal":           classification["signal"],
            "setup_type":       classification["setup_type"],
            "confluence":       classification["confluence"],
            "max_confluence":   classification["max_confluence"],
            "confluence_label": classification["confluence_label"],
            "all_setups":       classification["all_setups"],

            "risk_levels":      classification["risk_levels"],

            "indicators": {
                "trend":      snapshot["trend"],
                "momentum":   snapshot["momentum"],
                "volatility": snapshot["volatility"],
                "volume":     snapshot["volume"],
                "structure":  snapshot["structure"],
            },

            "context":          context,

            "rationale": {
                "advisory_text":     advisory["advisory_text"],
                "source":            advisory["source"],
                "model":             advisory["model"],
                "rule_based_bullets": classification["bullets"],
            },
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc


@app.post("/api/screener")
def screener(payload: ScreenerRequest):
    """
    Market-wide swing screener with automated 3-tier bucketing & ranking.

    Scans Nifty 100 or Nifty 50 constituents, runs the 5-dimension mathematical
    diagnostic on each stock, and categorizes into:
      1. PRIME_SETUPS — High Conviction / Actionable Now (ready for entry, defined risk)
      2. DEVELOPING   — On Radar / Watchlist (bullish structure, waiting for pullback/trigger)
      3. AVOID        — Stay Away / Broken Structure (bearish, choppy, or high risk)

    Ranks candidates inside each bucket and computes market breadth indicators.
    """
    try:
        # 1. Verify Kite connection
        _kite()

        # 2. Load universe symbols
        config = load_config()
        symbols = load_nifty100_universe()
        if payload.universe.lower() == "nifty50":
            symbols = symbols[:50]
        elif payload.max_stocks:
            symbols = symbols[:payload.max_stocks]

        # 3. Fetch batch candles in parallel
        fetcher = BatchCandleFetcher(config, provider="zerodha")
        batch_data = fetcher.fetch_batch(
            symbols,
            period=f"{payload.lookback_days}d",
            interval="1d",
            skip_missing=True,
        )

        if not batch_data:
            raise HTTPException(status_code=502, detail="No market data was fetched from Zerodha")

        # 4. Run screener engine
        from advisory_agent.scanner.screener import screen_batch
        results = screen_batch(batch_data, interval=payload.interval)
        results["universe"] = payload.universe
        results["requested_count"] = len(symbols)

        return results

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Screener failed: {exc}") from exc


