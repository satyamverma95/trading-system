# Trading System Handoff

## Project

Workspace: `trading-system`

Purpose: Nifty 100 SMA crossover scanner using Zerodha Kite Connect data, with a React/Vite dashboard and FastAPI backend.

## Current Architecture

```text
React/Vite frontend :5173
        |
        | /api proxy
        v
FastAPI backend :8000
        |
        +--> Zerodha Kite Connect
        +--> Nifty universe loader
        +--> Instrument resolver/cache
        +--> Batch candle fetcher
        +--> SMA calculator
        +--> Crossover detector
        +--> Ranking engine
```

A compatibility FastAPI listener may also run on port `5000` because the Kite Console was previously configured with that redirect URL.

## Implemented Work

### Existing pipeline modules

1. Nifty universe loader
   - File: `source_code/ingestion/nifty_loader.py`
   - Loads symbols from a CSV file.

2. Instrument resolver
   - File: `source_code/ingestion/instrument_resolver.py`
   - Resolves NSE symbols to Kite instrument tokens.
   - Uses a daily Parquet cache.
   - Refreshes the instrument master when a symbol is missing from cache.

3. Batch candle fetcher
   - File: `source_code/ingestion/batch_fetcher.py`
   - Supports yfinance and Zerodha providers.

4. SMA calculator
   - File: `source_code/processing/analysis/sma_calculator.py`
   - Adds configurable SMA columns such as `SMA_6`, `SMA_20`, `SMA_30`, `SMA_50`.

5. Crossover detector
   - File: `source_code/processing/analysis/crossover_detector.py`
   - Detects bullish and bearish SMA crossovers.
   - Tracks crossover state, signal date, days since crossover, and score.
   - Insufficient SMA history should be represented as `UNKNOWN`, not a trading signal.

6. Ranking engine
   - File: `source_code/processing/analysis/ranking_engine.py`
   - Ranks bullish, bearish, and unknown states.
   - Supports `days_since`, `score`, and `hybrid` strategies.

7. Pipeline orchestrator
   - File: `source_code/orchestration/nifty_pipeline.py`
   - CLI flow: fetch -> SMA -> crossover -> ranking -> export.

8. Result writer
   - File: `source_code/ingestion/data/result_writer.py`
   - Supports CSV, JSON, Parquet, and standalone HTML dashboard output.

### Zerodha integration

- Provider: `source_code/ingestion/providers/zerodha_provider.py`
- Authentication: `source_code/ingestion/auth/session_manager.py`
- Zerodha interval aliases are mapped correctly:

```text
1m  -> minute
5m  -> 5minute
15m -> 15minute
30m -> 30minute
1h  -> 60minute
1d  -> day
1wk -> week
1mo -> month
```

- Access tokens are saved to `config/secrets.yaml` by the backend.
- Do not store or collect Zerodha password, PIN, or TOTP values.
- Access tokens expire and require a new Zerodha login flow.

## Current React + FastAPI Application

### Backend

- File: `backend/main.py`
- FastAPI endpoints:

```text
GET  /api/health
GET  /api/auth/login-url
GET  /api/auth/status
GET  /api/auth/callback
GET  /api/profile
POST /api/auth/login
POST /api/signals
```

- FastAPI CORS allows the Vite development origin:

```text
http://127.0.0.1:5173
http://localhost:5173
```

- `/api/auth/callback` exchanges the Zerodha `request_token`, saves the access token, and redirects to the React app.
- `/` also acts as a compatibility callback for the old Kite Console redirect on port `5000`.

### Frontend

- Directory: `frontend/`
- Entry: `frontend/src/main.jsx`
- Styles: `frontend/src/styles.css`
- Vite config: `frontend/vite.config.js`
- Production build validated with `npm run build`.

The React app contains:

- Login screen
- Open Zerodha login action
- Manual API key/API secret/request-token form
- Dark dashboard shell
- User tab with profile information:
  - User Name
  - User ID
  - Products
  - Exchanges
- Signals tab with:
  - Short SMA, default 6
  - Long SMA, default 30
  - Lookback Days
  - Max Stocks
  - Generate Signals button
  - Ranked results table
  - Ticker, company, crossover type/date, close, SMA values
- Signal results are persisted in `sessionStorage` so switching between User and Signals tabs does not clear the table.

## Nifty Signals API Behavior

`POST /api/signals` accepts:

```json
{
  "short_sma": 6,
  "long_sma": 30,
  "lookback_days": 365,
  "max_stocks": 20
}
```

The backend:

1. Loads `data/input/nifty100.csv`.
2. Resolves symbols to NSE instrument tokens.
3. Fetches daily Zerodha candles.
4. Calculates SMA short and SMA long.
5. Detects bullish/bearish crossovers.
6. Filters signals by lookback period.
7. Sorts by latest crossover date descending.
8. Returns ranked JSON results.

## Universe Input

File:

```text
data/input/nifty100.csv
```

It currently contains a local 100-symbol seed list for testing. It should be refreshed against the official NSE/Nifty Indices constituent CSV before production use.

## Run Commands

Run from the repository root.

### Activate environment

```powershell
Set-Location C:\Users\satya\OneDrive\Documents\GitHub\trading-system
.\.venv\Scripts\Activate.ps1
```

### Start FastAPI backend

Terminal 1:

```powershell
Set-Location C:\Users\satya\OneDrive\Documents\GitHub\trading-system
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --reload --port 8000
```

API docs:

```text
http://127.0.0.1:8000/docs
```

### Start React frontend

Terminal 2:

```powershell
Set-Location C:\Users\satya\OneDrive\Documents\GitHub\trading-system\frontend
npm run dev -- --host 127.0.0.1
```

Dashboard:

```text
http://127.0.0.1:5173/
```

### Compatibility callback server

If Kite Console still redirects to port 5000, start this third process:

```powershell
Set-Location C:\Users\satya\OneDrive\Documents\GitHub\trading-system
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --host 127.0.0.1 --port 5000
```

Preferred Kite Console Redirect URL:

```text
http://127.0.0.1:8000/api/auth/callback
```

Legacy-compatible Redirect URL:

```text
http://127.0.0.1:5000/
```

Use only one configured redirect URL at a time and make sure the matching server is running.

## Authentication Flow

Preferred flow:

```text
React dashboard
  -> /api/auth/login-url
  -> Zerodha browser login + 2FA
  -> /api/auth/callback
  -> generate_session(request_token)
  -> save access_token backend-side
  -> redirect to React with auth=success
```

Manual fallback:

1. Open Zerodha login URL.
2. Complete login and 2FA.
3. Copy the `request_token` from the redirect URL.
4. Submit it through the React login form.
5. FastAPI exchanges and saves the access token.

Never commit credentials, request tokens, or access tokens.

## Validation Already Completed

- Existing SMA tests run against real yfinance data.
- Existing crossover tests passed across daily, weekly, and intraday scenarios.
- Zerodha provider was tested with a mocked Kite client.
- Real Zerodha two-symbol pipeline succeeded for RELIANCE and HDFCBANK.
- Real values observed in one run:
  - RELIANCE: BULLISH, score approximately 85
  - HDFCBANK: BEARISH, score approximately 14.5
- FastAPI import and route checks passed.
- React/Vite production build passed.
- Frontend and backend health endpoints responded successfully.
- JSON serialization was fixed so `NaN` values become JSON `null`.

## Known Issues / Follow-up

1. The React callback flow and the legacy port-5000 callback must use a matching Kite Console Redirect URL.
2. The official Nifty Indices CSV should replace the local seed universe.
3. Company names currently default to ticker symbols. Add a company-name mapping from the official universe CSV.
4. The instrument cache can contain mixed `expiry` values and may emit a Parquet warning; symbol resolution still works, but cache normalization should be hardened.
5. The crossover detector still emits a pandas `FutureWarning` around boolean `fillna`.
6. Zerodha access tokens expire; daily browser login/2FA is expected.
7. The current backend uses the local `config/secrets.yaml` file for token persistence. A production deployment should use an encrypted secret store.
8. The current frontend is a local development app, not a production deployment.

## Important Security Note

Credentials and tokens were exposed during development conversation history. Rotate the Zerodha API key, API secret, and any compromised access token before production use. Keep `config/secrets.yaml` excluded from Git.

## Recommended Next Step

1. Set the Kite Console Redirect URL to `http://127.0.0.1:8000/api/auth/callback`.
2. Start FastAPI and Vite.
3. Open `http://127.0.0.1:5173/`.
4. Complete Zerodha login once.
5. Open the Signals tab and run SMA 6/30 scan.
6. Compare a few exact candle rows with TradingView using the same exchange, daily timeframe, source price, and completed candle.
