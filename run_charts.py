# =================================================================
# run_charts.py  —  v4
# One chart per symbol per interval
# JS-based Signals ON/OFF toggle (bulletproof across Plotly versions)
# Usage: python run_charts.py
#        python run_charts.py --symbols RELIANCE.NS --intervals 1d
# =================================================================

import os
import argparse
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.helpers import load_config, load_from_parquet, save_to_parquet
from providers.yfinance_provider import YFinanceProvider
from analysis.indicators import Indicators
from utils.logger import get_logger

logger = get_logger(__name__)

SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS",
    "ICICIBANK.NS", "HINDUNILVR.NS", "SBIN.NS", "BAJFINANCE.NS",
    "BHARTIARTL.NS", "KOTAKBANK.NS", "MARUTI.NS", "WIPRO.NS",
    "AXISBANK.NS"
]

OUTPUT_DIR = "data/charts"
PERIOD_MAP = {"1d": "1y", "1wk": "3y", "1mo": "5y"}

# Keywords that identify a trace as a signal trace
SIGNAL_KEYWORDS = [
    "Oversold", "Overbought", "Crossover", "Crossunder",
    "SMA50", "Golden", "Death"
]


# ----------------------------------------------------------
# DATA LOADER
# ----------------------------------------------------------

def load_data(symbol, interval, config, provider, ind):
    try:
        df = load_from_parquet(symbol, config["paths"]["processed_data"], interval)
        logger.info(f"Loaded parquet: {symbol} ({interval})")
        return df
    except FileNotFoundError:
        logger.info(f"Fetching fresh: {symbol} ({interval})")
        period = PERIOD_MAP.get(interval, "1y")
        df = provider.get_historical_data(symbol, period=period, interval=interval)
        df = ind.add_all(df)
        save_to_parquet(df, symbol, config["paths"]["processed_data"], interval)
        return df


# ----------------------------------------------------------
# SIGNAL DETECTOR
# ----------------------------------------------------------

def get_signals(df):
    signals = {
        "rsi_oversold"          : [],
        "rsi_overbought"        : [],
        "macd_crossover"        : [],
        "macd_crossunder"       : [],
        "price_cross_sma50_up"  : [],
        "price_cross_sma50_down": [],
        "golden_cross"          : [],
        "death_cross"           : [],
    }

    for i in range(1, len(df)):
        row   = df.iloc[i]
        prev  = df.iloc[i - 1]
        date  = df.index[i]
        price = row["Close"]

        # RSI crossings
        rsi      = row.get("RSI")
        prev_rsi = prev.get("RSI")
        if pd.notna(rsi) and pd.notna(prev_rsi):
            if prev_rsi >= 30 and rsi < 30:
                signals["rsi_oversold"].append((date, price, f"RSI={rsi:.1f}"))
            if prev_rsi <= 70 and rsi > 70:
                signals["rsi_overbought"].append((date, price, f"RSI={rsi:.1f}"))

        # MACD crossings
        macd      = row.get("MACD")
        macd_sig  = row.get("MACD_Signal")
        pmacd     = prev.get("MACD")
        pmacd_sig = prev.get("MACD_Signal")
        if all(pd.notna(v) for v in [macd, macd_sig, pmacd, pmacd_sig]):
            if pmacd <= pmacd_sig and macd > macd_sig:
                signals["macd_crossover"].append((date, price, "MACD Up"))
            if pmacd >= pmacd_sig and macd < macd_sig:
                signals["macd_crossunder"].append((date, price, "MACD Dn"))

        # Price vs SMA50
        sma50      = row.get("SMA_50")
        prev_sma50 = prev.get("SMA_50")
        if pd.notna(sma50) and pd.notna(prev_sma50):
            if prev["Close"] <= prev_sma50 and price > sma50:
                signals["price_cross_sma50_up"].append((date, price, "SMA50 Up"))
            if prev["Close"] >= prev_sma50 and price < sma50:
                signals["price_cross_sma50_down"].append((date, price, "SMA50 Dn"))

        # Golden / Death cross
        sma100      = row.get("SMA_100")
        prev_sma100 = prev.get("SMA_100")
        prev_sma50v = prev.get("SMA_50")
        if all(pd.notna(v) for v in [sma50, sma100, prev_sma50v, prev_sma100]):
            if prev_sma50v <= prev_sma100 and sma50 > sma100:
                signals["golden_cross"].append((date, price, "Golden"))
            if prev_sma50v >= prev_sma100 and sma50 < sma100:
                signals["death_cross"].append((date, price, "Death"))

    return signals


# ----------------------------------------------------------
# BUILD SINGLE CHART
# ----------------------------------------------------------

def build_single_chart(symbol, df, interval):
    """
    Builds one clean interactive chart.
    Signal traces are added AFTER base traces.
    JS toggle identifies them by name keywords.
    """

    signals = get_signals(df)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.60, 0.20, 0.20],
        subplot_titles=[
            f"{symbol} — Price & Indicators ({interval.upper()})",
            "RSI",
            "MACD"
        ]
    )

    # ----------------------------------------------------------
    # BASE TRACES — always visible
    # ----------------------------------------------------------

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350"
    ), row=1, col=1)

    # Moving Averages
    for col, color, width, dash in [
        ("SMA_20",  "#f6c90e", 1.0, "dot"),
        ("SMA_50",  "#2979ff", 1.2, "solid"),
        ("SMA_100", "#ff6d00", 1.0, "dash"),
    ]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], name=col,
                line=dict(color=color, width=width, dash=dash)
            ), row=1, col=1)

    # Bollinger Bands
    if "BB_Upper" in df.columns and "BB_Lower" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Upper"], name="BB Upper",
            line=dict(color="rgba(180,180,180,0.4)", width=0.8, dash="dash")
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Lower"], name="BB Lower",
            line=dict(color="rgba(180,180,180,0.4)", width=0.8, dash="dash"),
            fill="tonexty", fillcolor="rgba(180,180,180,0.04)"
        ), row=1, col=1)

    # RSI
    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI"], name="RSI",
            line=dict(color="#ce93d8", width=1.5)
        ), row=2, col=1)

    # MACD
    if "MACD" in df.columns and "MACD_Hist" in df.columns:
        hist_colors = [
            "#26a69a" if v >= 0 else "#ef5350"
            for v in df["MACD_Hist"].fillna(0)
        ]
        fig.add_trace(go.Bar(
            x=df.index, y=df["MACD_Hist"],
            name="Hist", marker_color=hist_colors
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD"], name="MACD",
            line=dict(color="#2979ff", width=1.5)
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD_Signal"], name="Signal",
            line=dict(color="#ff9800", width=1.5)
        ), row=3, col=1)

    # ----------------------------------------------------------
    # SIGNAL TRACES — added AFTER base, toggled via JS by name
    # ----------------------------------------------------------
    signal_configs = [
        ("rsi_oversold",           "#00e676", "triangle-up",   10, "RSI Oversold (Entry)"),
        ("rsi_overbought",         "#ff1744", "triangle-down",  10, "RSI Overbought (Exit)"),
        ("macd_crossover",         "#69f0ae", "triangle-up",    8,  "MACD Crossover (Entry)"),
        ("macd_crossunder",        "#ff6e40", "triangle-down",  8,  "MACD Crossunder (Exit)"),
        ("price_cross_sma50_up",   "#40c4ff", "triangle-up",    7,  "Price above SMA50 (Entry)"),
        ("price_cross_sma50_down", "#ff80ab", "triangle-down",  7,  "Price below SMA50 (Exit)"),
        ("golden_cross",           "#ffd740", "star",           13, "Golden Cross (Strong Entry)"),
        ("death_cross",            "#e040fb", "x",              13, "Death Cross (Strong Exit)"),
    ]

    for sig_key, color, marker, size, label in signal_configs:
        pts = signals.get(sig_key, [])
        if pts:
            dates  = [p[0] for p in pts]
            prices = [p[1] for p in pts]
            texts  = [p[2] for p in pts]
            fig.add_trace(go.Scatter(
                x=dates, y=prices,
                mode="markers+text",
                name=label,
                text=texts,
                textposition="top center",
                textfont=dict(size=8, color=color),
                marker=dict(
                    symbol=marker, size=size, color=color,
                    line=dict(width=1, color="white")
                ),
                visible=True
            ), row=1, col=1)

    # ----------------------------------------------------------
    # REFERENCE LINES
    # ----------------------------------------------------------
    fig.add_hline(y=70, line_dash="dash", line_color="#ef5350",
                  line_width=0.8, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#26a69a",
                  line_width=0.8, row=2, col=1)
    fig.add_hline(y=50, line_dash="dot",  line_color="gray",
                  line_width=0.5, row=2, col=1)
    fig.add_hline(y=0,  line_dash="dot",  line_color="gray",
                  line_width=0.5, row=3, col=1)

    # ----------------------------------------------------------
    # LAYOUT — no updatemenus, JS handles the toggle
    # ----------------------------------------------------------
    fig.update_layout(
        height=900,
        template="plotly_dark",
        title=dict(
            text=f"{symbol} — {interval.upper()} Chart",
            font=dict(size=16)
        ),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="v",
            x=1.01, y=1,
            font=dict(size=9),
            bgcolor="rgba(0,0,0,0)"
        ),
        margin=dict(l=60, r=200, t=80, b=40)
    )

    return fig


# ----------------------------------------------------------
# SUMMARY DASHBOARD
# ----------------------------------------------------------

def build_summary(all_data, interval):
    fig = go.Figure()
    colors = [
        "#2979ff", "#00e676", "#ff6d00", "#ff4081", "#ffd740",
        "#40c4ff", "#e040fb", "#69f0ae", "#ff6e40", "#b2ff59",
        "#ea80fc", "#84ffff", "#ccff90"
    ]
    for i, (symbol, df) in enumerate(all_data.items()):
        if df is None or df.empty:
            continue
        norm = (df["Close"] / df["Close"].iloc[0]) * 100
        fig.add_trace(go.Scatter(
            x=df.index, y=norm,
            name=symbol.replace(".NS", ""),
            line=dict(color=colors[i % len(colors)], width=1.5),
            hovertemplate="%{x}<br>%{y:.1f}<extra>" + symbol + "</extra>"
        ))

    fig.update_layout(
        title=f"Normalised Price Comparison — {interval.upper()} (Base = 100)",
        height=550,
        template="plotly_dark",
        yaxis_title="Indexed Price",
        legend=dict(orientation="h", y=-0.15),
        hovermode="x unified",
        margin=dict(l=60, r=40, t=80, b=100)
    )
    return fig


# ----------------------------------------------------------
# JS TOGGLE SNIPPET — injected into every chart HTML
# ----------------------------------------------------------

JS_INJECT = """
<style>
  #signal-controls {
    position: fixed;
    top: 14px;
    right: 320px;
    z-index: 9999;
    display: flex;
    gap: 8px;
  }
  .sig-btn {
    background: #1e1e2e;
    color: #ccc;
    border: 1px solid #555;
    padding: 7px 18px;
    border-radius: 5px;
    cursor: pointer;
    font-size: 12px;
    font-family: 'Segoe UI', sans-serif;
    transition: all 0.15s;
  }
  .sig-btn:hover  { background: #2979ff; color: #fff; border-color: #2979ff; }
  .sig-btn.active { background: #2979ff; color: #fff; border-color: #2979ff; }
</style>

<div id="signal-controls">
  <button class="sig-btn active" onclick="toggleSignals(true,  this)">Signals ON</button>
  <button class="sig-btn"        onclick="toggleSignals(false, this)">Signals OFF</button>
</div>

<script>
var SIGNAL_KEYWORDS = [
  'Oversold', 'Overbought', 'Crossover', 'Crossunder',
  'SMA50', 'Golden', 'Death'
];

function isSignalTrace(name) {
  if (!name) return false;
  return SIGNAL_KEYWORDS.some(function(kw) {
    return name.indexOf(kw) !== -1;
  });
}

function toggleSignals(show, btn) {
  // Update active button styling
  document.querySelectorAll('.sig-btn').forEach(function(b) {
    b.classList.remove('active');
  });
  btn.classList.add('active');

  // Find the Plotly graph div
  var plotDiv = document.querySelector('.plotly-graph-div');
  if (!plotDiv || !plotDiv.data) {
    console.warn('Plotly graph not found');
    return;
  }

  // Build visibility array — signals toggled, base always true
  var visibilityUpdate = plotDiv.data.map(function(trace) {
    return isSignalTrace(trace.name) ? show : true;
  });

  Plotly.restyle(plotDiv, { visible: visibilityUpdate });
}

// Auto-hide signals on load for a clean default view
window.addEventListener('load', function() {
  // Small delay to ensure Plotly has fully rendered
  setTimeout(function() {
    var offBtn = document.querySelectorAll('.sig-btn')[1];
    // Start with signals ON — comment next line to start with signals OFF
    // toggleSignals(false, offBtn);
  }, 500);
});
</script>
"""


# ----------------------------------------------------------
# SAVE CHART — HTML + PNG
# ----------------------------------------------------------

def save_chart(fig, filename_base, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, f"{filename_base}.html")
    png_path  = os.path.join(output_dir, f"{filename_base}.png")

    # Export to HTML and inject JS toggle
    html_content = fig.to_html(full_html=True, include_plotlyjs="cdn")
    html_content = html_content.replace("<body>", "<body>" + JS_INJECT, 1)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"HTML → {html_path}")

    # PNG export
    try:
        fig.write_image(png_path, width=1400, height=900, scale=1.5)
        logger.info(f"PNG  → {png_path}")
    except Exception as e:
        logger.warning(f"PNG skipped: {e}")

    return html_path


# ----------------------------------------------------------
# INDEX PAGE
# ----------------------------------------------------------

def build_index(symbols, intervals, output_dir):
    cards = ""
    for symbol in symbols:
        clean = symbol.replace(".", "_").replace("^", "idx_")
        links = " ".join([
            f'<a href="{clean}_{iv}_chart.html">{iv.upper()}</a>'
            for iv in intervals
        ])
        cards += f'<div class="card"><h2>{symbol}</h2>{links}</div>\n'

    summary_links = " ".join([
        f'<a href="00_summary_{iv}.html">Summary {iv.upper()}</a>'
        for iv in intervals
    ])

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Trading Dashboard</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: #0d0d1a; color: #fff;
            font-family: 'Segoe UI', sans-serif; padding: 30px; }}
    h1   {{ color: #2979ff; font-size: 26px; margin-bottom: 6px; }}
    .sub {{ color: #666; font-size: 12px; margin-bottom: 24px; }}
    .summary-bar {{ margin-bottom: 28px; }}
    .summary-bar a {{
      display: inline-block; background: #1a1a2e;
      border: 1px solid #2979ff; color: #ffd740;
      padding: 7px 16px; border-radius: 6px;
      text-decoration: none; margin-right: 10px; font-size: 13px;
    }}
    .summary-bar a:hover {{ background: #2979ff; color: #fff; }}
    hr {{ border: none; border-top: 1px solid #1e1e3a; margin: 20px 0; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 14px;
    }}
    .card {{
      background: #1a1a2e; border: 1px solid #2a2a4a;
      border-radius: 10px; padding: 16px;
      transition: border-color 0.2s, transform 0.1s;
    }}
    .card:hover {{ border-color: #2979ff; transform: translateY(-2px); }}
    .card h2 {{ color: #69f0ae; font-size: 14px; margin-bottom: 10px; }}
    .card a {{
      display: inline-block; background: #0d0d1a;
      border: 1px solid #333; color: #ffd740;
      padding: 4px 12px; border-radius: 4px;
      text-decoration: none; margin-right: 6px; font-size: 12px;
    }}
    .card a:hover {{ background: #2979ff; color: #fff; border-color: #2979ff; }}
  </style>
</head>
<body>
  <h1>&#x1F4CA; Trading Dashboard</h1>
  <p class="sub">NSE Large Cap — Technical Analysis | trading-system</p>
  <hr>
  <div class="summary-bar">
    <strong style="color:#aaa;font-size:12px;margin-right:10px">
      &#x1F4C8; Market Summary:
    </strong>
    {summary_links}
  </div>
  <div class="grid">
    {cards}
  </div>
</body>
</html>"""

    index_path = os.path.join(output_dir, "index.html")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"Index → {index_path}")


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

def run(symbols, intervals):
    config   = load_config()
    provider = YFinanceProvider(config)
    ind      = Indicators(config)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n📊 Generating: {len(symbols)} symbols x {len(intervals)} intervals\n")

    for interval in intervals:
        all_data = {}
        print(f"  ── {interval.upper()} ──")

        for symbol in symbols:
            try:
                df = load_data(symbol, interval, config, provider, ind)
                all_data[symbol] = df

                fig   = build_single_chart(symbol, df, interval)
                clean = symbol.replace(".", "_").replace("^", "idx_")
                save_chart(fig, f"{clean}_{interval}_chart", OUTPUT_DIR)
                print(f"    ✅ {symbol}")

            except Exception as e:
                logger.error(f"{symbol} ({interval}): {e}")
                print(f"    ❌ {symbol} — {e}")

        # Summary per interval
        try:
            summary = build_summary(all_data, interval)
            save_chart(summary, f"00_summary_{interval}", OUTPUT_DIR)
            print(f"    ✅ Summary ({interval.upper()})")
        except Exception as e:
            print(f"    ❌ Summary — {e}")

    # Index page
    build_index(symbols, intervals, OUTPUT_DIR)
    print(f"\n✅ Done! Serve with:")
    print(f"   python -m http.server 8080 --directory {OUTPUT_DIR}/\n")


# ----------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading Chart Generator")
    parser.add_argument("--symbols",   nargs="+", default=SYMBOLS,
                        help="Symbols to chart")
    parser.add_argument("--intervals", nargs="+", default=["1d", "1wk", "1mo"],
                        help="Intervals: 1d 1wk 1mo")
    args = parser.parse_args()
    run(args.symbols, args.intervals)