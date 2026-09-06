import React, { useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { Activity, ArrowRight, CircleUserRound, KeyRound, LineChart, LogIn, RefreshCw, ShieldCheck, SlidersHorizontal } from 'lucide-react'
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
  const [params, setParams] = useState(() => JSON.parse(sessionStorage.getItem('signalParams') || '{"short_sma":6,"long_sma":30,"lookback_days":365,"max_stocks":20}')); const [results, setResults] = useState(() => JSON.parse(sessionStorage.getItem('signalResults') || '[]')); const [meta, setMeta] = useState(() => JSON.parse(sessionStorage.getItem('signalMeta') || 'null')); const [error, setError] = useState(''); const [busy, setBusy] = useState(false)
  const update = (key) => (event) => setParams({ ...params, [key]: Number(event.target.value) })
  const generate = async () => { setBusy(true); setError(''); try { const data = await api('/api/signals', { method: 'POST', body: JSON.stringify(params) }); setResults(data.results); setMeta(data); sessionStorage.setItem('signalParams', JSON.stringify(params)); sessionStorage.setItem('signalResults', JSON.stringify(data.results)); sessionStorage.setItem('signalMeta', JSON.stringify(data)) } catch (err) { setError(err.message) } finally { setBusy(false) } }
  return <section><div className="signal-head"><div><p className="eyebrow">Nifty 100 scanner</p><h2>Fresh crossover signals</h2><p className="muted">Daily candles, ranked by the most recent crossover date.</p></div><button className="primary" onClick={generate} disabled={busy}><RefreshCw size={17} className={busy ? 'spin' : ''} /> {busy ? 'Scanning...' : 'Generate signals'}</button></div><div className="control-bar"><label><SlidersHorizontal size={15} /> Short SMA<input type="number" min="1" value={params.short_sma} onChange={update('short_sma')} /></label><label>Long SMA<input type="number" min="2" value={params.long_sma} onChange={update('long_sma')} /></label><label>Lookback days<input type="number" min="30" value={params.lookback_days} onChange={update('lookback_days')} /></label><label>Max stocks<input type="number" min="1" max="100" value={params.max_stocks} onChange={update('max_stocks')} /></label></div>{error && <div className="notice error">{error}</div>}{meta && <div className="scan-meta"><span>{meta.fetched} of {meta.requested} symbols fetched</span><span>{results.length} signals found</span></div>}<div className="table-shell"><table><thead><tr><th>Rank</th><th>Ticker</th><th>Company</th><th>Crossover</th><th>Date</th><th>Close</th><th>SMA {params.short_sma}</th><th>SMA {params.long_sma}</th></tr></thead><tbody>{results.length ? results.map((row) => <tr key={`${row.ticker}-${row.crossover_date}`}><td className="rank">{row.rank}</td><td className="ticker">{row.ticker}</td><td>{row.company}</td><td><span className={`signal ${row.crossover_type.toLowerCase()}`}>{row.crossover_type}</span></td><td>{row.crossover_date}</td><td>{Number(row.close).toFixed(2)}</td><td>{row.short_sma == null ? '-' : Number(row.short_sma).toFixed(2)}</td><td>{row.long_sma == null ? '-' : Number(row.long_sma).toFixed(2)}</td></tr>) : <tr><td colSpan="8" className="empty">Run the scanner to load live crossover signals.</td></tr>}</tbody></table></div></section>
}

function Dashboard({ onLogout }) {
  const [tab, setTab] = useState('signals'); const [status, setStatus] = useState('Checking connection...')
  useEffect(() => { api('/api/auth/status').then((data) => setStatus(data.message)).catch(() => setStatus('Not connected')) }, [])
  return <main className="dashboard"><header className="topbar"><div className="wordmark"><div className="brand-mark small"><LineChart size={18} /></div><span>Signal Desk</span></div><div className="top-actions"><span className="connection"><i />{status}</span><button className="ghost" onClick={onLogout}>Disconnect</button></div></header><section className="dashboard-title"><p className="kicker">NIFTY 100 / DAILY ANALYTICS</p><h1>Good signals start with good context.</h1><p className="lede">Review your account and scan the market without leaving your local workspace.</p></section><nav className="tabs"><button className={tab === 'signals' ? 'active' : ''} onClick={() => setTab('signals')}><Activity size={16} /> Signals</button><button className={tab === 'user' ? 'active' : ''} onClick={() => setTab('user')}><CircleUserRound size={16} /> User</button></nav><section className="content">{tab === 'signals' ? <SignalsTab /> : <UserTab />}</section></main>
}

function App() { const [connected, setConnected] = useState(false); const [authMessage, setAuthMessage] = useState(''); useEffect(() => { const query = new URLSearchParams(window.location.search); if (query.get('auth') === 'failed') setAuthMessage('Zerodha login was not completed. Please try again.'); api('/api/auth/status').then((data) => setConnected(data.connected)).catch(() => {}) }, []); return connected ? <Dashboard onLogout={() => setConnected(false)} /> : <>{authMessage && <div className="global-error">{authMessage}</div>}<Login onConnected={() => setConnected(true)} /></> }

createRoot(document.getElementById('root')).render(<App />)
