"""Local control panel for Zerodha authentication and signal refresh."""

from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, redirect, render_template_string, request, url_for
from kiteconnect import KiteConnect

from source_code.common.config_loader import load_config
from source_code.ingestion.auth.session_manager import save_session
from source_code.orchestration.nifty_pipeline import NiftyPipeline

app = Flask(__name__)

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nifty Signal Desk</title>
<style>
:root{--ink:#17212b;--muted:#66737d;--paper:#f4f1ea;--panel:#fffdf8;--line:#ddd8cd;--green:#087f5b;--red:#b42318;--accent:#d97706}
*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 90% 0%,#f9dfb3 0,transparent 28%),var(--paper);font:15px/1.5 Georgia,serif}main{max-width:1180px;margin:auto;padding:42px 22px 64px}.eyebrow{color:var(--accent);font:700 12px ui-sans-serif,sans-serif;letter-spacing:2px;text-transform:uppercase}h1{margin:8px 0 4px;font-size:clamp(34px,5vw,58px);line-height:1;font-weight:500}.sub{color:var(--muted);font-family:ui-sans-serif,sans-serif;margin:0 0 25px}.actions{display:flex;gap:10px;align-items:center;margin-bottom:25px}button,a{border:1px solid var(--ink);background:var(--ink);color:white;padding:11px 15px;text-decoration:none;font:700 13px ui-sans-serif,sans-serif;cursor:pointer}a{background:transparent;color:var(--ink)}#status{color:var(--muted);font-family:ui-sans-serif,sans-serif}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:25px}.stat{background:var(--panel);border:1px solid var(--line);padding:18px}.stat strong{display:block;font:700 30px ui-sans-serif,sans-serif}.stat span{color:var(--muted);font:12px ui-sans-serif,sans-serif;text-transform:uppercase;letter-spacing:1px}.table-wrap{overflow:auto;background:var(--panel);border:1px solid var(--line)}table{width:100%;border-collapse:collapse;font:13px ui-sans-serif,sans-serif}th,td{padding:13px 14px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}th{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.7px}.pill{padding:3px 8px;font-weight:700;font-size:11px}.bullish{color:var(--green);background:#d9f4e8}.bearish{color:var(--red);background:#fde1de}.unknown{color:var(--muted);background:#eee}@media(max-width:600px){main{padding:28px 14px}.stats{grid-template-columns:1fr}.actions{align-items:stretch;flex-direction:column}}
</style></head><body><main>
<div class="eyebrow">Trading system / local control panel</div><h1>Nifty Signal Desk</h1>
<p class="sub">Authenticate Zerodha, refresh market data, and review the latest calculations.</p>
<div class="actions"><a href="/auth/login">Login to Zerodha</a><button id="refresh">Refresh signals</button><span id="status">Ready</span></div>
<section class="stats"><div class="stat"><strong id="total">-</strong><span>Ranked symbols</span></div><div class="stat"><strong id="bullish">-</strong><span>Bullish signals</span></div><div class="stat"><strong id="bearish">-</strong><span>Bearish signals</span></div></section>
<div class="table-wrap"><table><thead><tr><th>Rank</th><th>Symbol</th><th>Close</th><th>State</th><th>Days since</th><th>Score</th><th>Last crossover</th><th>SMA fast</th><th>SMA slow</th></tr></thead><tbody id="rows"><tr><td colspan="9">Click Refresh signals to load data.</td></tr></tbody></table></div>
</main><script>
const statusEl=document.getElementById('status');
function cell(value){return value===null||value===undefined||Number.isNaN(value)?'-':value}
function render(result){document.getElementById('total').textContent=result.length;document.getElementById('bullish').textContent=result.filter(x=>x.State==='BULLISH').length;document.getElementById('bearish').textContent=result.filter(x=>x.State==='BEARISH').length;document.getElementById('rows').innerHTML=result.map(x=>`<tr><td>${cell(x.Rank)}</td><td>${cell(x.Symbol)}</td><td>${Number(x.Close).toFixed(2)}</td><td><span class="pill ${(x.State||'unknown').toLowerCase()}">${cell(x.State)}</span></td><td>${cell(x.Days_Since)}</td><td>${Number(x.Score).toFixed(1)}</td><td>${cell(x.Last_Crossover_Type)}</td><td>${cell(x.SMA_fast)}</td><td>${cell(x.SMA_slow)}</td></tr>`).join('')}
document.getElementById('refresh').onclick=async()=>{statusEl.textContent='Fetching and calculating...';try{const response=await fetch('/api/refresh',{method:'POST'});const data=await response.json();if(!response.ok)throw new Error(data.error);render(data.results);statusEl.textContent=`Updated ${data.fetched} symbols at ${data.output_path}`}catch(error){statusEl.textContent=error.message}};
</script></body></html>"""


def _kite_client() -> KiteConnect:
    config = load_config()["zerodha"]
    return KiteConnect(api_key=config["api_key"])


@app.get("/")
def index():
    request_token = request.args.get("request_token")
    if request_token:
        try:
            config = load_config()["zerodha"]
            session = _kite_client().generate_session(request_token, api_secret=config["api_secret"])
            save_session(session["access_token"])
            return redirect(url_for("index", auth="success"))
        except Exception as exc:
            return render_template_string(PAGE + f"<script>document.getElementById('status').textContent='Login failed: {str(exc)}'</script>"), 400
    return render_template_string(PAGE)


@app.get("/auth/login")
def login():
    return redirect(_kite_client().login_url())


@app.post("/api/refresh")
def refresh():
    try:
        config = load_config()
        result = NiftyPipeline(config=config, provider="zerodha").run(
            universe_csv="data/input/nifty100.csv",
            period="1y",
            interval="1d",
            output_format="csv",
            output_path="data/gold/latest_zerodha_signals.csv",
        )
        ranked = result["ranked_results"].copy()
        records = ranked.to_dict(orient="records")
        return jsonify({"fetched": len(result["symbols_fetched"]), "output_path": result["output_path"], "results": records})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
