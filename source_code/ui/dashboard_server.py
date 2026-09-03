"""Local Zerodha connection and signal-refresh UI."""

import os
import threading
import time
from datetime import datetime
import math

from flask import Flask, jsonify, redirect, render_template_string, request, url_for
from kiteconnect import KiteConnect

from source_code.common.config_loader import load_config
from source_code.ingestion.auth.session_manager import get_authenticated_kite, save_session
from source_code.orchestration.nifty_pipeline import NiftyPipeline

app = Flask(__name__)
_last_refresh = {"status": "never", "message": "Connect Zerodha to begin."}


def _json_safe(value):
    """Convert pandas/numpy values into standards-compliant JSON values."""
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if value != value:
            return None
    except (TypeError, ValueError):
        pass
    return value.item() if hasattr(value, "item") else value


def _records_for_json(dataframe):
    return [
        {key: _json_safe(value) for key, value in record.items()}
        for record in dataframe.to_dict(orient="records")
    ]

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nifty Signal Desk</title><style>
:root{--ink:#17212b;--muted:#66737d;--paper:#f4f1ea;--panel:#fffdf8;--line:#ddd8cd;--green:#087f5b;--red:#b42318;--accent:#d97706}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 90% 0%,#f9dfb3 0,transparent 28%),var(--paper);font:15px/1.5 Georgia,serif}main{max-width:1180px;margin:auto;padding:42px 22px 64px}.eyebrow{color:var(--accent);font:700 12px ui-sans-serif,sans-serif;letter-spacing:2px;text-transform:uppercase}h1{margin:8px 0 4px;font-size:clamp(34px,5vw,58px);line-height:1;font-weight:500}.sub{color:var(--muted);font-family:ui-sans-serif,sans-serif;margin:0 0 25px}.actions{display:flex;gap:10px;align-items:center;margin-bottom:25px}button{border:1px solid var(--ink);background:var(--ink);color:#fff;padding:11px 15px;font:700 13px ui-sans-serif,sans-serif;cursor:pointer}button:disabled{opacity:.45;cursor:not-allowed}.status{font:13px ui-sans-serif,sans-serif;color:var(--muted)}.status.connected{color:var(--green)}.status.expired,.error{color:var(--red)}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:25px}.stat{background:var(--panel);border:1px solid var(--line);padding:18px}.stat strong{display:block;font:700 30px ui-sans-serif,sans-serif}.stat span{color:var(--muted);font:12px ui-sans-serif,sans-serif;text-transform:uppercase;letter-spacing:1px}.table-wrap{overflow:auto;background:var(--panel);border:1px solid var(--line)}table{width:100%;border-collapse:collapse;font:13px ui-sans-serif,sans-serif}th,td{padding:13px 14px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}th{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.7px}.pill{padding:3px 8px;font-weight:700;font-size:11px}.bullish{color:var(--green);background:#d9f4e8}.bearish{color:var(--red);background:#fde1de}.unknown{color:var(--muted);background:#eee}@media(max-width:600px){main{padding:28px 14px}.stats{grid-template-columns:1fr}.actions{align-items:stretch;flex-direction:column}}
</style></head><body><main><div class="eyebrow">Trading system / local control panel</div><h1>Nifty Signal Desk</h1><p class="sub">Connect Zerodha, refresh market data, and review calculated signals.</p>
<div class="actions"><button id="connect">Connect Zerodha</button><button id="refresh" disabled>Refresh signals</button><span id="status" class="status">Checking connection...</span></div>
<section class="stats"><div class="stat"><strong id="total">-</strong><span>Ranked symbols</span></div><div class="stat"><strong id="bullish">-</strong><span>Bullish signals</span></div><div class="stat"><strong id="bearish">-</strong><span>Bearish signals</span></div></section>
<div class="table-wrap"><table><thead><tr><th>Rank</th><th>Symbol</th><th>Close</th><th>State</th><th>Days since</th><th>Score</th><th>Last crossover</th><th>SMA fast</th><th>SMA slow</th></tr></thead><tbody id="rows"><tr><td colspan="9">Connect Zerodha to begin.</td></tr></tbody></table></div></main>
<script>const statusEl=document.getElementById('status'),connect=document.getElementById('connect'),refresh=document.getElementById('refresh');function setStatus(s,c=''){statusEl.textContent=s;statusEl.className='status '+c}function cell(v){return v===null||v===undefined||Number.isNaN(v)?'-':v}function render(r){document.getElementById('total').textContent=r.length;document.getElementById('bullish').textContent=r.filter(x=>x.State==='BULLISH').length;document.getElementById('bearish').textContent=r.filter(x=>x.State==='BEARISH').length;document.getElementById('rows').innerHTML=r.map(x=>`<tr><td>${cell(x.Rank)}</td><td>${cell(x.Symbol)}</td><td>${Number(x.Close).toFixed(2)}</td><td><span class="pill ${(x.State||'unknown').toLowerCase()}">${cell(x.State)}</span></td><td>${cell(x.Days_Since)}</td><td>${Number(x.Score).toFixed(1)}</td><td>${cell(x.Last_Crossover_Type)}</td><td>${cell(x.SMA_20)}</td><td>${cell(x.SMA_50)}</td></tr>`).join('')}async function status(){const r=await fetch('/api/status');const d=await r.json();refresh.disabled=!d.connected;setStatus(d.message,d.connected?'connected':d.expired?'expired':'')}connect.onclick=()=>location.href='/auth/login';refresh.onclick=async()=>{refresh.disabled=true;setStatus('Fetching and calculating...');try{const r=await fetch('/api/refresh',{method:'POST'}),d=await r.json();if(!r.ok)throw Error(d.error);render(d.results);setStatus(`Updated ${d.fetched} symbols at ${d.updated}`,'connected')}catch(e){setStatus(e.message,'error')}finally{refresh.disabled=false}};status();</script></body></html>"""


def _kite() -> KiteConnect:
    config = load_config().get("zerodha", {})
    return KiteConnect(api_key=config.get("api_key", ""))


def _connection_status() -> dict:
    try:
        kite = get_authenticated_kite()
        profile = kite.profile()
        return {"connected": True, "expired": False, "message": f"Connected as {profile.get('user_name', profile.get('user_id', 'Zerodha user'))}"}
    except Exception as exc:
        message = str(exc)
        expired = "token" in message.lower() or "access" in message.lower()
        return {"connected": False, "expired": expired, "message": "Token expired" if expired else "Not connected"}


@app.get("/")
def index():
    request_token = request.args.get("request_token")
    if request_token:
        try:
            config = load_config()["zerodha"]
            session = _kite().generate_session(request_token, api_secret=config["api_secret"])
            save_session(session["access_token"])
            return redirect(url_for("index", auth="success"))
        except Exception as exc:
            return render_template_string(PAGE + f"<script>setStatus('Login failed: {str(exc)}','error')</script>"), 400
    return render_template_string(PAGE)


@app.get("/auth/login")
def login():
    return redirect(_kite().login_url())


@app.get("/api/status")
def api_status():
    return jsonify(_connection_status())


@app.post("/api/refresh")
def api_refresh():
    try:
        result = NiftyPipeline(provider="zerodha").run(universe_csv="data/input/nifty100.csv", period="1y", interval="1d", output_format="csv", output_path="data/gold/latest_zerodha_signals.csv")
        ranked = result["ranked_results"]
        updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _last_refresh.update(status="success", message=updated)
        return jsonify({"fetched": len(result["symbols_fetched"]), "updated": updated, "results": _records_for_json(ranked)})
    except Exception as exc:
        _last_refresh.update(status="error", message=str(exc))
        return jsonify({"error": str(exc)}), 500


def _scheduled_refresh() -> None:
    refresh_time = os.getenv("AUTO_REFRESH_TIME")
    if not refresh_time:
        return
    while True:
        if datetime.now().strftime("%H:%M") == refresh_time and _connection_status()["connected"]:
            with app.test_request_context():
                api_refresh()
            time.sleep(61)
        time.sleep(20)


if __name__ == "__main__":
    threading.Thread(target=_scheduled_refresh, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=False)
