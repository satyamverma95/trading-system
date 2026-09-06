import React, { useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import {
  Activity,
  ArrowRight,
  BarChart2,
  CircleUserRound,
  Compass,
  KeyRound,
  Layers,
  LineChart,
  LogIn,
  Newspaper,
  RefreshCw,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Target,
  TrendingDown,
  TrendingUp,
} from 'lucide-react'
import './styles.css'

const api = async (path, options = {}) => {
  const response = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...options })
  const body = await response.json()
  if (!response.ok) throw new Error(body.detail || 'Request failed')
  return body
}

function Login({ onConnected }) {
  const [form, setForm] = useState({ api_key: '', api_secret: '', request_token: '' })
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const update = (key) => (event) => setForm({ ...form, [key]: event.target.value })
  const submit = async (event) => {
    event.preventDefault(); setBusy(true); setError('')
    try { await api('/api/auth/login', { method: 'POST', body: JSON.stringify(form) }); onConnected() }
    catch (err) { setError(err.message) } finally { setBusy(false) }
  }
  const openLogin = async () => {
    try { const result = await api('/api/auth/login-url'); window.location.href = result.url }
    catch (err) { setError(err.message) }
  }
  return <main className="auth-layout"><section className="intro"><div className="brand-mark"><Activity size={22} /></div><p className="kicker">KITE CONNECT / LOCAL WORKSPACE</p><h1>Trade with a clearer signal.</h1><p className="lede">A focused command center for market context, crossover signals, and disciplined review.</p><div className="trust"><ShieldCheck size={18} /><span>Your token stays on the backend.</span></div></section><section className="auth-panel"><div className="panel-heading"><p className="eyebrow">Welcome back</p><h2>Connect Zerodha</h2><p>Login to your Kite account, then paste the one-time request token.</p></div><button className="secondary full" onClick={openLogin}><LogIn size={17} /> Open Zerodha login <ArrowRight size={16} /></button><div className="divider"><span>or enter token manually</span></div><form onSubmit={submit}><label>API Key<input value={form.api_key} onChange={update('api_key')} required autoComplete="off" /></label><label>API Secret<input type="password" value={form.api_secret} onChange={update('api_secret')} required autoComplete="off" /></label><label>Request Token<input value={form.request_token} onChange={update('request_token')} required autoComplete="off" /></label>{error && <div className="error">{error}</div>}<button className="primary full" disabled={busy}>{busy ? 'Connecting...' : 'Connect securely'} <ArrowRight size={17} /></button></form></section></main>
}

function UserTab() {
  const [profile, setProfile] = useState(null); const [error, setError] = useState('')
  useEffect(() => { api('/api/profile').then(setProfile).catch((err) => setError(err.message)) }, [])
  if (error) return <div className="notice error">{error}</div>
  if (!profile) return <div className="notice">Loading profile...</div>
  return <section className="user-grid"><div className="profile-hero"><div className="avatar"><CircleUserRound size={30} /></div><p className="eyebrow">Authenticated account</p><h2>{profile.user_name || 'Zerodha user'}</h2><p className="muted">User ID: {profile.user_id}</p></div><div className="info-card"><span className="label">Products</span><div className="chips">{profile.products.map((item) => <span key={item}>{item}</span>)}</div></div><div className="info-card"><span className="label">Exchanges</span><div className="chips">{profile.exchanges.map((item) => <span key={item}>{item}</span>)}</div></div></section>
}

function SignalsTab() {
  const [params, setParams] = useState(() => JSON.parse(sessionStorage.getItem('signalParams') || '{"short_sma":6,"long_sma":30,"lookback_days":365,"max_stocks":20}'))
  const [results, setResults] = useState(() => JSON.parse(sessionStorage.getItem('signalResults') || '[]'))
  const [meta, setMeta] = useState(() => JSON.parse(sessionStorage.getItem('signalMeta') || 'null'))
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const update = (key) => (event) => setParams({ ...params, [key]: Number(event.target.value) })
  const generate = async () => {
    setBusy(true); setError('')
    try {
      const data = await api('/api/signals', { method: 'POST', body: JSON.stringify(params) })
      setResults(data.results); setMeta(data)
      sessionStorage.setItem('signalParams', JSON.stringify(params))
      sessionStorage.setItem('signalResults', JSON.stringify(data.results))
      sessionStorage.setItem('signalMeta', JSON.stringify(data))
    } catch (err) { setError(err.message) } finally { setBusy(false) }
  }
  return <section><div className="signal-head"><div><p className="eyebrow">Nifty 100 scanner</p><h2>Fresh crossover signals</h2><p className="muted">Daily candles, ranked by the most recent crossover date.</p></div><button className="primary" onClick={generate} disabled={busy}><RefreshCw size={17} className={busy ? 'spin' : ''} /> {busy ? 'Scanning...' : 'Generate signals'}</button></div><div className="control-bar"><label><SlidersHorizontal size={15} /> Short SMA<input type="number" min="1" value={params.short_sma} onChange={update('short_sma')} /></label><label>Long SMA<input type="number" min="2" value={params.long_sma} onChange={update('long_sma')} /></label><label>Lookback days<input type="number" min="30" value={params.lookback_days} onChange={update('lookback_days')} /></label><label>Max stocks<input type="number" min="1" max="100" value={params.max_stocks} onChange={update('max_stocks')} /></label></div>{error && <div className="notice error">{error}</div>}{meta && <div className="scan-meta"><span>{meta.fetched} of {meta.requested} symbols fetched</span><span>{results.length} signals found</span></div>}<div className="table-shell"><table><thead><tr><th>Rank</th><th>Ticker</th><th>Company</th><th>Crossover</th><th>Date</th><th>Close</th><th>SMA {params.short_sma}</th><th>SMA {params.long_sma}</th></tr></thead><tbody>{results.length ? results.map((row) => <tr key={`${row.ticker}-${row.crossover_date}`}><td className="rank">{row.rank}</td><td className="ticker">{row.ticker}</td><td>{row.company}</td><td><span className={`signal ${row.crossover_type.toLowerCase()}`}>{row.crossover_type}</span></td><td>{row.crossover_date}</td><td>{Number(row.close).toFixed(2)}</td><td>{row.short_sma == null ? '-' : Number(row.short_sma).toFixed(2)}</td><td>{row.long_sma == null ? '-' : Number(row.long_sma).toFixed(2)}</td></tr>) : <tr><td colSpan="8" className="empty">Run the scanner to load live crossover signals.</td></tr>}</tbody></table></div></section>
}

const PRESET_TICKERS = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'TATAMOTORS', 'LT', 'SBIN', 'BHARTIARTL', 'ITC']

function AnalyzeTab() {
  const [symbol, setSymbol] = useState(() => sessionStorage.getItem('advisor_symbol') || 'RELIANCE')
  const [interval, setInterval] = useState(() => sessionStorage.getItem('advisor_interval') || 'day')
  const [advisory, setAdvisory] = useState(() => {
    try { return JSON.parse(sessionStorage.getItem('advisor_data') || 'null') }
    catch { return null }
  })
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const runAnalysis = async (ticker = symbol, timeFrame = interval) => {
    setBusy(true); setError('')
    try {
      const data = await api('/api/analyze', {
        method: 'POST',
        body: JSON.stringify({ symbol: ticker.trim().toUpperCase(), interval: timeFrame }),
      })
      setAdvisory(data)
      sessionStorage.setItem('advisor_symbol', ticker.trim().toUpperCase())
      sessionStorage.setItem('advisor_interval', timeFrame)
      sessionStorage.setItem('advisor_data', JSON.stringify(data))
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const handleChipClick = (t) => {
    setSymbol(t)
    runAnalysis(t, interval)
  }

  return (
    <section className="advisor-layout">
      <div className="advisor-head">
        <div>
          <p className="eyebrow">Institutional-Grade Swing Advisory</p>
          <h2>Technical Advisor Agent</h2>
          <p className="muted">5-Dimension mathematical analysis, automated setup classification, risk levels, and external market context.</p>
        </div>
      </div>

      {/* ── Search & Controls Panel ── */}
      <div className="advisor-search-panel">
        <div className="search-controls">
          <label>
            NSE Symbol
            <input
              type="text"
              placeholder="e.g. RELIANCE, TCS, INFY"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              onKeyDown={(e) => { if (e.key === 'Enter') runAnalysis() }}
            />
          </label>

          <label>
            Time Horizon / Interval
            <select value={interval} onChange={(e) => setInterval(e.target.value)}>
              <option value="day">Daily (Swing 3–10 Days)</option>
              <option value="1h">1 Hour (Short Swing)</option>
              <option value="15m">15 Minutes (Intraday Momentum)</option>
              <option value="5m">5 Minutes (Tactical Entry)</option>
            </select>
          </label>

          <button className="primary" onClick={() => runAnalysis()} disabled={busy}>
            <Compass size={17} className={busy ? 'spin' : ''} />
            {busy ? 'Analyzing...' : 'Run Technical Advisory'}
          </button>
        </div>

        <div className="ticker-chips">
          <span className="chip-label">Quick Scan:</span>
          {PRESET_TICKERS.map((t) => (
            <button
              key={t}
              className={`ticker-chip ${symbol === t ? 'active' : ''}`}
              onClick={() => handleChipClick(t)}
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      {error && <div className="notice error">{error}</div>}

      {busy && (
        <div className="notice" style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <RefreshCw size={18} className="spin" style={{ color: '#d7f36b' }} />
          <span>Computing Trend, Momentum, Volatility, Volume, Structure, and fetching live FII & VIX context for {symbol}...</span>
        </div>
      )}

      {/* ── Advisory Result Card ── */}
      {advisory && !busy && (
        <>
          {/* Top Hero Signal Card */}
          <div className="hero-card">
            <div className="hero-header">
              <div className="hero-symbol-title">
                <h2>{advisory.symbol}</h2>
                <span className="hero-ltp">₹{advisory.ltp?.toLocaleString('en-IN', { minimumFractionDigits: 2 })}</span>
                <span className="muted" style={{ fontSize: '12px' }}>NSE · {advisory.interval?.toUpperCase()} · {advisory.candle_count} CANDLES</span>
              </div>

              <div className="hero-badges">
                <span className={`badge-signal ${advisory.signal?.toLowerCase()}`}>
                  {advisory.signal === 'BUY' && '🟢 '}
                  {advisory.signal === 'SELL_EXIT' && '🔴 '}
                  {advisory.signal === 'WATCH' && '🟡 '}
                  {advisory.signal}
                </span>
                <span className="badge-setup">
                  {advisory.setup_type?.replace(/_/g, ' ')}
                </span>
                <span className="badge-confluence">
                  CONFLUENCE {advisory.confluence}/{advisory.max_confluence} ({advisory.confluence_label})
                </span>
              </div>
            </div>

            {/* Execution Risk Levels Matrix */}
            {advisory.risk_levels && (
              <div className="execution-matrix">
                <div className="exec-item">
                  <span className="exec-label">Entry Zone</span>
                  <span className="exec-value">
                    ₹{advisory.risk_levels.entry_low?.toLocaleString('en-IN')} – ₹{advisory.risk_levels.entry_high?.toLocaleString('en-IN')}
                  </span>
                  <span className="exec-sub">Current price bracket</span>
                </div>

                <div className="exec-item">
                  <span className="exec-label">Stop-Loss</span>
                  <span className="exec-value stop">
                    ₹{advisory.risk_levels.stop_loss?.toLocaleString('en-IN')}
                  </span>
                  <span className="exec-sub">
                    Risk: ₹{advisory.risk_levels.risk_per_share} ({advisory.risk_levels.sl_multiplier}× ATR)
                  </span>
                </div>

                <div className="exec-item">
                  <span className="exec-label">Target 1 (R:R 1:{advisory.risk_levels.rr_t1})</span>
                  <span className="exec-value target">
                    ₹{advisory.risk_levels.target_1?.toLocaleString('en-IN')}
                  </span>
                  <span className="exec-sub">Initial take-profit zone</span>
                </div>

                <div className="exec-item">
                  <span className="exec-label">Target 2 (R:R 1:{advisory.risk_levels.rr_t2})</span>
                  <span className="exec-value target">
                    ₹{advisory.risk_levels.target_2?.toLocaleString('en-IN')}
                  </span>
                  <span className="exec-sub">Runner / trend extension</span>
                </div>
              </div>
            )}

            {/* AI Advisory Rationale Box */}
            {advisory.rationale && (
              <div className="rationale-panel">
                <div className="rationale-top">
                  <div className="rationale-tag">
                    <Sparkles size={16} />
                    <span>Technical Advisory Synthesis</span>
                  </div>
                  <span className="model-pill">
                    {advisory.rationale.source === 'gemini' ? 'Gemini 1.5 Flash' : 'Rule-Based Engine'}
                  </span>
                </div>

                <p className="advisory-text">{advisory.rationale.advisory_text}</p>

                {advisory.rationale.rule_based_bullets?.length > 0 && (
                  <div className="checks-list">
                    {advisory.rationale.rule_based_bullets.map((b, i) => (
                      <div key={i} className="check-bullet">{b}</div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* ── 5 Dimensions Grid ── */}
          <div className="dimensions-section">
            <h3>Five-Dimension Technical Diagnostic</h3>
            <div className="dimensions-grid">
              {/* 1. Trend */}
              <div className="dimension-card">
                <div className="dimension-card-header">
                  <span className="dimension-title"><TrendingUp size={16} /> 1. Trend</span>
                  <span className="dimension-state-pill">{advisory.indicators?.trend?.state}</span>
                </div>
                <div className="dim-metrics-grid">
                  <div className="dim-metric">
                    <span className="dim-metric-label">EMA 20</span>
                    <span className="dim-metric-value">₹{advisory.indicators?.trend?.ema_20?.toLocaleString('en-IN') || '-'}</span>
                  </div>
                  <div className="dim-metric">
                    <span className="dim-metric-label">EMA 50</span>
                    <span className="dim-metric-value">₹{advisory.indicators?.trend?.ema_50?.toLocaleString('en-IN') || '-'}</span>
                  </div>
                  <div className="dim-metric">
                    <span className="dim-metric-label">EMA 200</span>
                    <span className="dim-metric-value">₹{advisory.indicators?.trend?.ema_200?.toLocaleString('en-IN') || '-'}</span>
                  </div>
                  <div className="dim-metric">
                    <span className="dim-metric-label">ADX (14)</span>
                    <span className="dim-metric-value">{advisory.indicators?.trend?.adx} ({advisory.indicators?.trend?.adx_state})</span>
                  </div>
                </div>
                <div className="dimension-card-desc">{advisory.indicators?.trend?.description}</div>
              </div>

              {/* 2. Momentum */}
              <div className="dimension-card">
                <div className="dimension-card-header">
                  <span className="dimension-title"><Activity size={16} /> 2. Momentum</span>
                  <span className="dimension-state-pill">{advisory.indicators?.momentum?.rsi_state}</span>
                </div>
                <div className="dim-metrics-grid">
                  <div className="dim-metric">
                    <span className="dim-metric-label">RSI (14)</span>
                    <span className="dim-metric-value">{advisory.indicators?.momentum?.rsi}</span>
                  </div>
                  <div className="dim-metric">
                    <span className="dim-metric-label">Divergence</span>
                    <span className="dim-metric-value">{advisory.indicators?.momentum?.rsi_divergence || 'None'}</span>
                  </div>
                  <div className="dim-metric">
                    <span className="dim-metric-label">MACD State</span>
                    <span className="dim-metric-value">{advisory.indicators?.momentum?.macd_state}</span>
                  </div>
                  <div className="dim-metric">
                    <span className="dim-metric-label">MACD Hist</span>
                    <span className="dim-metric-value">{advisory.indicators?.momentum?.macd_hist}</span>
                  </div>
                </div>
                <div className="dimension-card-desc">{advisory.indicators?.momentum?.description}</div>
              </div>

              {/* 3. Volatility */}
              <div className="dimension-card">
                <div className="dimension-card-header">
                  <span className="dimension-title"><Layers size={16} /> 3. Volatility</span>
                  <span className="dimension-state-pill">{advisory.indicators?.volatility?.state}</span>
                </div>
                <div className="dim-metrics-grid">
                  <div className="dim-metric">
                    <span className="dim-metric-label">ATR (14)</span>
                    <span className="dim-metric-value">₹{advisory.indicators?.volatility?.atr} ({advisory.indicators?.volatility?.atr_pct}%)</span>
                  </div>
                  <div className="dim-metric">
                    <span className="dim-metric-label">BB Width Pct</span>
                    <span className="dim-metric-value">{advisory.indicators?.volatility?.bb_width_percentile}th %tile</span>
                  </div>
                  <div className="dim-metric">
                    <span className="dim-metric-label">BB Upper</span>
                    <span className="dim-metric-value">₹{advisory.indicators?.volatility?.bb_upper?.toLocaleString('en-IN') || '-'}</span>
                  </div>
                  <div className="dim-metric">
                    <span className="dim-metric-label">BB Lower</span>
                    <span className="dim-metric-value">₹{advisory.indicators?.volatility?.bb_lower?.toLocaleString('en-IN') || '-'}</span>
                  </div>
                </div>
                <div className="dimension-card-desc">{advisory.indicators?.volatility?.description}</div>
              </div>

              {/* 4. Volume */}
              <div className="dimension-card">
                <div className="dimension-card-header">
                  <span className="dimension-title"><BarChart2 size={16} /> 4. Volume</span>
                  <span className="dimension-state-pill">{advisory.indicators?.volume?.state}</span>
                </div>
                <div className="dim-metrics-grid">
                  <div className="dim-metric">
                    <span className="dim-metric-label">Volume Ratio</span>
                    <span className="dim-metric-value">{advisory.indicators?.volume?.volume_ratio}× Avg</span>
                  </div>
                  <div className="dim-metric">
                    <span className="dim-metric-label">OBV Trend</span>
                    <span className="dim-metric-value">{advisory.indicators?.volume?.obv_trend}</span>
                  </div>
                  <div className="dim-metric">
                    <span className="dim-metric-label">Current Vol</span>
                    <span className="dim-metric-value">{advisory.indicators?.volume?.current_volume?.toLocaleString('en-IN')}</span>
                  </div>
                  <div className="dim-metric">
                    <span className="dim-metric-label">20D Avg Vol</span>
                    <span className="dim-metric-value">{advisory.indicators?.volume?.avg_volume_20d?.toLocaleString('en-IN')}</span>
                  </div>
                </div>
                <div className="dimension-card-desc">{advisory.indicators?.volume?.description}</div>
              </div>

              {/* 5. Structure */}
              <div className="dimension-card">
                <div className="dimension-card-header">
                  <span className="dimension-title"><Target size={16} /> 5. Structure</span>
                  <span className="dimension-state-pill">{advisory.indicators?.structure?.price_vs_pivot || 'PP'}</span>
                </div>
                <div className="dim-metrics-grid">
                  <div className="dim-metric">
                    <span className="dim-metric-label">Nearest Fib</span>
                    <span className="dim-metric-value">{advisory.indicators?.structure?.nearest_fib_level} (₹{advisory.indicators?.structure?.nearest_fib_price?.toLocaleString('en-IN')})</span>
                  </div>
                  <div className="dim-metric">
                    <span className="dim-metric-label">Fib Proximity</span>
                    <span className="dim-metric-value">{advisory.indicators?.structure?.nearest_fib_distance_pct}% away</span>
                  </div>
                  <div className="dim-metric">
                    <span className="dim-metric-label">Weekly PP</span>
                    <span className="dim-metric-value">₹{advisory.indicators?.structure?.weekly_pp?.toLocaleString('en-IN') || '-'}</span>
                  </div>
                  <div className="dim-metric">
                    <span className="dim-metric-label">Pivot R1 / S1</span>
                    <span className="dim-metric-value">₹{advisory.indicators?.structure?.weekly_r1?.toFixed(0)} / ₹{advisory.indicators?.structure?.weekly_s1?.toFixed(0)}</span>
                  </div>
                </div>
                <div className="dimension-card-desc">{advisory.indicators?.structure?.description}</div>
              </div>
            </div>
          </div>

          {/* ── Macro & External Factors ── */}
          <div className="context-section">
            <h3>External Context (What The Chart Can't See)</h3>
            <div className="context-grid">
              {/* India VIX */}
              <div className="context-card">
                <span className="label">Market Sentiment / India VIX</span>
                {advisory.context?.vix ? (
                  <>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                      <span style={{ fontSize: '22px', fontWeight: 700 }}>{advisory.context.vix.current_vix}</span>
                      <span className="dimension-state-pill">{advisory.context.vix.state}</span>
                    </div>
                    <span className="muted" style={{ fontSize: '11px' }}>
                      5-day trend: {advisory.context.vix.vix_trend} (from {advisory.context.vix.vix_5d_ago})
                    </span>
                    <p style={{ fontSize: '12px', color: '#8ea1ae', margin: 0, lineHeight: 1.5 }}>
                      {advisory.context.vix.description}
                    </p>
                  </>
                ) : (
                  <span className="muted" style={{ fontSize: '12px' }}>India VIX live data unavailable.</span>
                )}
              </div>

              {/* FII / DII Flows */}
              <div className="context-card">
                <span className="label">Institutional Flow (FII / DII 5-Day)</span>
                {advisory.context?.fii_dii ? (
                  <>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                      <span style={{ fontSize: '18px', fontWeight: 700 }}>
                        FII: ₹{advisory.context.fii_dii.fii_net_5d_cr?.toLocaleString('en-IN')} Cr
                      </span>
                      <span className="dimension-state-pill">{advisory.context.fii_dii.institutional_flow}</span>
                    </div>
                    <span className="muted" style={{ fontSize: '11px' }}>
                      DII: ₹{advisory.context.fii_dii.dii_net_5d_cr?.toLocaleString('en-IN')} Cr
                    </span>
                    <p style={{ fontSize: '12px', color: '#8ea1ae', margin: 0, lineHeight: 1.5 }}>
                      {advisory.context.fii_dii.description}
                    </p>
                  </>
                ) : (
                  <span className="muted" style={{ fontSize: '12px' }}>FII/DII flow feed unavailable.</span>
                )}
              </div>

              {/* News Headlines */}
              <div className="context-card">
                <span className="label"><Newspaper size={13} style={{ display: 'inline', marginRight: '4px' }} /> Recent News Headlines</span>
                {advisory.context?.news?.headlines?.length > 0 ? (
                  <div className="news-list">
                    {advisory.context.news.headlines.slice(0, 3).map((h, i) => (
                      <div key={i} className="news-item">{h}</div>
                    ))}
                  </div>
                ) : (
                  <span className="muted" style={{ fontSize: '12px' }}>No major breaking catalyst detected for {advisory.symbol}.</span>
                )}
              </div>
            </div>
          </div>
        </>
      )}
    </section>
  )
}

function Dashboard({ onLogout }) {
  const [tab, setTab] = useState(() => sessionStorage.getItem('active_tab') || 'analyze')
  const [status, setStatus] = useState('Checking connection...')

  const switchTab = (t) => {
    setTab(t)
    sessionStorage.setItem('active_tab', t)
  }

  useEffect(() => {
    api('/api/auth/status')
      .then((data) => setStatus(data.message))
      .catch(() => setStatus('Not connected'))
  }, [])

  return (
    <main className="dashboard">
      <header className="topbar">
        <div className="wordmark">
          <div className="brand-mark small"><LineChart size={18} /></div>
          <span>Signal Desk · Technical Advisor</span>
        </div>
        <div className="top-actions">
          <span className="connection"><i />{status}</span>
          <button className="ghost" onClick={onLogout}>Disconnect</button>
        </div>
      </header>

      <section className="dashboard-title">
        <p className="kicker">INDIAN EQUITIES / SWING ADVISORY AGENT</p>
        <h1>Good trades start with deep confluence.</h1>
        <p className="lede">Analyze any NSE equity across 5 technical dimensions, evaluate structural risk levels, and read institutional-grade advisory insights.</p>
      </section>

      <nav className="tabs">
        <button className={tab === 'analyze' ? 'active' : ''} onClick={() => switchTab('analyze')}>
          <Compass size={16} /> Technical Advisor
        </button>
        <button className={tab === 'signals' ? 'active' : ''} onClick={() => switchTab('signals')}>
          <Activity size={16} /> Signals Scanner
        </button>
        <button className={tab === 'user' ? 'active' : ''} onClick={() => switchTab('user')}>
          <CircleUserRound size={16} /> User
        </button>
      </nav>

      <section className="content">
        {tab === 'analyze' && <AnalyzeTab />}
        {tab === 'signals' && <SignalsTab />}
        {tab === 'user' && <UserTab />}
      </section>
    </main>
  )
}

function App() {
  const [connected, setConnected] = useState(false)
  const [authMessage, setAuthMessage] = useState('')

  useEffect(() => {
    const query = new URLSearchParams(window.location.search)
    if (query.get('auth') === 'failed') setAuthMessage('Zerodha login was not completed. Please try again.')
    api('/api/auth/status').then((data) => setConnected(data.connected)).catch(() => {})
  }, [])

  return connected ? (
    <Dashboard onLogout={() => setConnected(false)} />
  ) : (
    <>
      {authMessage && <div className="global-error">{authMessage}</div>}
      <Login onConnected={() => setConnected(true)} />
    </>
  )
}

createRoot(document.getElementById('root')).render(<App />)
