# trading-system

## React + FastAPI dashboard

The current dashboard has a Vite React frontend and a FastAPI backend. Start
the backend and frontend in separate terminals from the repository root.

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

In a second terminal:

```powershell
Set-Location frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Use **Open Zerodha login**, complete the normal
Zerodha login and 2FA, then exchange the returned request token in the login
form. The backend stores the resulting access token; passwords, PINs, and TOTP
secrets are never collected by this application.

The **User** tab loads the Zerodha profile. The **Signals** tab runs the Nifty
100 daily SMA crossover scan with configurable short SMA, long SMA, lookback,
and result count.