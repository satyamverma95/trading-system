"""FastAPI backend for Zerodha authentication, profile, and signals."""

import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
