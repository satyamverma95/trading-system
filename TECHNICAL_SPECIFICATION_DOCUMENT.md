# Technical Specification & Mathematical Reference Manual
**Project:** Indian Equity Technical Advisory Agent & Swing Screener  
**Target Horizon:** Swing Trading (3–10 Day Holding Cycles)  
**Execution Type:** Pure Algorithmic Decision-Support & Risk Governance (Non-Custodial)  
**Document Version:** 2.0.0  
**Date:** September 2026  

---

## Table of Contents
1. [Executive Summary & System Philosophy](#1-executive-summary--system-philosophy)
2. [End-to-End System Architecture](#2-end-to-end-system-architecture)
3. [Dimension 1: Trend Architecture (EMA Stack & ADX)](#3-dimension-1-trend-architecture-ema-stack--adx)
4. [Dimension 2: Momentum & Divergence Dynamics (RSI & MACD)](#4-dimension-2-momentum--divergence-dynamics-rsi--macd)
5. [Dimension 3: Volatility & Volatility Cycles (ATR & Bollinger Squeeze)](#5-dimension-3-volatility--volatility-cycles-atr--bollinger-squeeze)
6. [Dimension 4: Volume & Institutional Participation (Volume Ratio & OBV)](#6-dimension-4-volume--institutional-participation-volume-ratio--obv)
7. [Dimension 5: Structural Geometry & Levels (Fibonacci & Weekly Pivots)](#7-dimension-5-structural-geometry--levels-fibonacci--weekly-pivots)
8. [Strategy Classification & Confluence Engine](#8-strategy-classification--confluence-engine)
9. [Deterministic Risk Management & Target Allocation](#9-deterministic-risk-management--target-allocation)
10. [Market-Wide Screener & Automated 3-Tier Bucketing](#10-market-wide-screener--automated-3-tier-bucketing)
11. [Multi-Factor Intra-Bucket Ranking Algorithm](#11-multi-factor-intra-bucket-ranking-algorithm)
12. [External Market Context Integration (VIX, FII/DII, Catalysts)](#12-external-market-context-integration-vix-fiidii-catalysts)
13. [Artificial Intelligence Synthesis & Strict Guardrails](#13-artificial-intelligence-synthesis--strict-guardrails)
14. [API Endpoints & Integration Contract](#14-api-endpoints--integration-contract)

---

## 1. Executive Summary & System Philosophy

### 1.1 Objective
The purpose of the system is to solve the two biggest failure points of retail swing traders:
1. **The Discovery Problem:** Manually scanning 100–500 stocks every evening is exhausting. Traders miss high-probability setups triggering at exact support zones simply because they did not inspect the stock at 3:15 PM.
2. **The Psychological Trap (FOMO & Undefined Risk):** Traders chase stocks that are already up +5% on the day, enter at overbought extremes, and trade without predefined, mathematical stop-losses and profit targets.

### 1.2 Core Architectural Axioms
- **Zero Order Placement:** The engine is strictly a decision-support and risk-governance system. It reads market data through Zerodha Kite Connect, computes mathematical diagnostics, and provides clear, actionable advice. It never places or routes orders.
- **Strict Role Separation (Deterministic Math vs. Generative AI):**
  - All indicator values, trend states, setup classifications, confluence scores, stop-losses, and profit targets are calculated by **deterministic mathematical functions**.
  - The Large Language Model (Gemini 1.5 Flash) **never computes, modifies, or invents numbers**. Its sole responsibility is natural language synthesis: translating the mathematical confluence and external macro context into an institutional-grade narrative for the trader.
- **The Swing Trading Cadence:** All mathematical models operate on **Daily (EOD) OHLCV candles** with lookbacks tailored for 3-to-10-day holding cycles.

---

## 2. End-to-End System Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION LAYER                            │
│  - Zerodha Kite Connect API (Token Authentication & Candle Fetching)   │
│  - NSE India Universe Loader (Nifty 100 / Nifty 50 Constituents)       │
│  - Daily Historical Parquet Cache (Local Disk Storage)                 │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ Raw OHLCV DataFrames
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│               5-DIMENSION MATHEMATICAL ANALYSIS ENGINE                 │
│  1. Trend:      EMA-20, EMA-50, EMA-200, ADX-14, DI+, DI-             │
│  2. Momentum:   RSI-14, Bullish/Bearish Divergence, MACD (12, 26, 9)   │
│  3. Volatility: ATR-14, Bollinger Bands (20, 2σ), Squeeze Percentile   │
│  4. Volume:     Volume Ratio vs 20d Avg, OBV Regression Slope          │
│  5. Structure:  60-Candle Swing Fibonacci Ratios, Weekly Floor Pivots  │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ IndicatorSnapshot
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│              STRATEGY CLASSIFIER & RISK GOVERNANCE                     │
│  - Setup A: Momentum Pullback (8-point confluence checklist)           │
│  - Setup B: Volume-Confirmed Breakout (6-point checklist)              │
│  - Setup C: Oversold Reversal (6-point checklist)                      │
│  - ATR Execution Matrix: Entry Zone, Stop-Loss, Target 1, 2, 3         │
└─────────────────┬──────────────────────────────────┬───────────────────┘
                  │                                  │
                  ▼ Single Stock                     ▼ Universe Batch
┌───────────────────────────────────┐  ┌─────────────────────────────────┐
│     EXTERNAL CONTEXT & AI         │  │   MARKET SCREENER & BUCKETING   │
│ - India VIX Gauge & Trend         │  │ - Bucket 1: Prime Setups        │
│ - NSE FII/DII 5-Day Net Flow      │  │ - Bucket 2: On Radar Watchlist  │
│ - Google News RSS Headlines       │  │ - Bucket 3: Avoid / Broken      │
│ - Gemini 1.5 Flash Narrative      │  │ - Intra-Bucket Ranking (0-100)  │
└─────────────────┬─────────────────┘  └─────────────────┬───────────────┘
                  │                                      │
                  └───────────────────┬──────────────────┘
                                      ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 REACT DASHBOARD & VISUALIZATION                        │
│  - Technical Advisor Desk (Single Stock Deep-Dive)                     │
│  - Market Screener View (Breadth Bar & 3 Bucket Tabs)                  │
│  - SMA Signal Scanner (Historical Crossovers)                          │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Dimension 1: Trend Architecture (EMA Stack & ADX)

### 3.1 Exponential Moving Averages (EMA)
The Exponential Moving Average applies weighting multipliers that decrease exponentially:

$$\text{EMA}_t = \left( \text{Close}_t \times \alpha \right) + \left( \text{EMA}_{t-1} \times (1 - \alpha) \right)$$

$$\alpha = \frac{2}{N + 1}$$

We compute three distinct EMA lengths:
1. **Fast Trend ($\text{EMA}_{20}$):** $\alpha = \frac{2}{21} \approx 0.0952$ — Represents short-term institutional momentum.
2. **Intermediate Trend ($\text{EMA}_{50}$):** $\alpha = \frac{2}{51} \approx 0.0392$ — The primary benchmark for swing trading direction.
3. **Macro Trend ($\text{EMA}_{200}$):** $\alpha = \frac{2}{201} \approx 0.00995$ — The institutional boundary separating secular bull and bear markets.

### 3.2 Trend Classification State Machine
| State | Mathematical Definition | Interpretation |
| :--- | :--- | :--- |
| **`STRONG_BULL`** | $\text{Price} > \text{EMA}_{20} > \text{EMA}_{50} > \text{EMA}_{200}$ | All institutional timeframes aligned upwards. Prime long territory. |
| **`BULL`** | $\text{Price} > \text{EMA}_{50}$ and $\text{EMA}_{20} > \text{EMA}_{50}$ | Confirmed intermediate uptrend. Pullbacks to $\text{EMA}_{20}$ are buyable. |
| **`NEUTRAL`** | EMAs intertwined, or $\text{Price}$ oscillating between $\text{EMA}_{20}$ and $\text{EMA}_{50}$ | Transition or range-bound phase. No directional edge. |
| **`BEAR`** | $\text{Price} < \text{EMA}_{50}$ | Bearish bias dominating. Long swing positions are disqualified. |
| **`STRONG_BEAR`** | $\text{Price} < \text{EMA}_{200}$ and $\text{EMA}_{20} < \text{EMA}_{50}$ | Macro downtrend. Capital preservation mode; stay away. |

### 3.3 Average Directional Index (ADX-14)
ADX quantifies trend strength regardless of direction. It is computed via True Range ($\text{TR}$) and Directional Movement ($\text{DM}^+, \text{DM}^-$):

$$\text{TR}_t = \max(\text{High}_t - \text{Low}_t, |\text{High}_t - \text{Close}_{t-1}|, |\text{Low}_t - \text{Close}_{t-1}|)$$

$$\text{DM}^+_t = \text{High}_t - \text{High}_{t-1} \quad (\text{if } > 0 \text{ and } > \text{Low}_{t-1} - \text{Low}_t, \text{ else } 0)$$

$$\text{DM}^-_t = \text{Low}_{t-1} - \text{Low}_t \quad (\text{if } > 0 \text{ and } > \text{High}_t - \text{High}_{t-1}, \text{ else } 0)$$

$$\text{DI}^+ = \frac{\text{WilderSmooth}(\text{DM}^+, 14)}{\text{WilderSmooth}(\text{TR}, 14)} \times 100, \quad \text{DI}^- = \frac{\text{WilderSmooth}(\text{DM}^-, 14)}{\text{WilderSmooth}(\text{TR}, 14)} \times 100$$

$$\text{DX} = \frac{|\text{DI}^+ - \text{DI}^-|}{\text{DI}^+ + \text{DI}^-} \times 100, \quad \text{ADX}_{14} = \text{WilderSmooth}(\text{DX}, 14)$$

#### ADX Regimes:
- **$\text{ADX} < 20$ (`RANGING`):** Market is in chop or consolidation. Breakouts and pullbacks have high failure rates.
- **$20 \le \text{ADX} < 25$ (`DEVELOPING`):** Trend is actively forming. Early swing entry window.
- **$25 \le \text{ADX} \le 40$ (`STRONG`):** Ideal swing trading sweet spot. High follow-through probability.
- **$\text{ADX} > 40$ (`VERY_STRONG`):** Mature/late-stage trend. Exhaustion risk is elevated; reversals are dangerous.

---

## 4. Dimension 2: Momentum & Divergence Dynamics (RSI & MACD)

### 4.1 Relative Strength Index (RSI-14)
Measures the velocity and magnitude of directional price movements:

$$\text{RS} = \frac{\text{AvgGain}_{14}}{\text{AvgLoss}_{14}}, \quad \text{RSI} = 100 - \left( \frac{100}{1 + \text{RS}} \right)$$

#### Classification Zones:
- **`OVERBOUGHT` ($\text{RSI} > 70$):** Momentum is stretched. Do not initiate new swing longs; risk of mean reversion is high.
- **`MOMENTUM_ZONE` ($55 < \text{RSI} \le 70$):** Strong bullish thrust. Trend continuation runners.
- **`PULLBACK_ZONE` ($38 \le \text{RSI} \le 58$):** **The Primary Swing Target Zone.** Price has cooled down within an uptrend without breaking structural support.
- **`WEAK` ($30 \le \text{RSI} < 38$):** Below average momentum; requires stabilization before considering longs.
- **`OVERSOLD` ($\text{RSI} < 30$):** Capitulation territory. Valid only for Setup C (Oversold Reversal) at deep support.

### 4.2 Automated RSI Divergence Algorithm
Divergence between price and momentum reveals institutional exhaustion before it reflects on the chart:
- **Bullish Divergence:** Over a 20-bar rolling window, $\text{Price}_{\text{current}} \le \min(\text{Price}_{20}) \times 1.01$ while $\text{RSI}_{\text{current}} > \min(\text{RSI}_{20}) \times 1.05$. Price is testing new lows, but selling velocity has dried up.
- **Bearish Divergence:** $\text{Price}_{\text{current}} \ge \max(\text{Price}_{20}) \times 0.99$ while $\text{RSI}_{\text{current}} < \max(\text{RSI}_{20}) \times 0.95$. Price printed a higher high, but buyer momentum is decaying.

### 4.3 Moving Average Convergence Divergence (MACD 12, 26, 9)
$$\text{MACD Line} = \text{EMA}_{12}(\text{Close}) - \text{EMA}_{26}(\text{Close})$$

$$\text{Signal Line} = \text{EMA}_9(\text{MACD Line})$$

$$\text{Histogram} = \text{MACD Line} - \text{Signal Line}$$

- **`CROSSOVER_BULLISH`:** MACD crossed above Signal line within the last 5 bars.
- **`BULLISH`:** MACD Line is above Signal line with expanding histogram.
- **`BEARISH`:** MACD Line is below Signal line.

---

## 5. Dimension 3: Volatility & Volatility Cycles (ATR & Bollinger Squeeze)

### 5.1 Average True Range (ATR-14)
ATR provides an absolute measure of volatility in Rupees ($\text{₹}$):

$$\text{ATR}_{14} = \text{WilderSmooth}(\text{TR}, 14)$$

$$\text{ATR \%} = \frac{\text{ATR}_{14}}{\text{Price}} \times 100$$

ATR is the mathematical engine behind dynamic stop-loss positioning. Fixed percentage stops (e.g. always 2%) fail because a low-volatility stock like TCS ($\text{ATR} \approx 1.2\%$) behaves completely differently from a high-beta stock like DIXON ($\text{ATR} \approx 3.5\%$).

### 5.2 Bollinger Bands (20, 2σ) & Squeeze Percentile
$$\text{Middle Band} = \text{SMA}_{20}(\text{Close})$$

$$\text{Upper Band} = \text{SMA}_{20} + (2 \times \sigma_{20}), \quad \text{Lower Band} = \text{SMA}_{20} - (2 \times \sigma_{20})$$

$$\text{BandWidth} = \frac{\text{Upper Band} - \text{Lower Band}}{\text{Middle Band}} \times 100$$

#### The Volatility Squeeze Metric:
We calculate the rolling **percentile rank** of the current BandWidth against the preceding 252 trading days (1 year):

$$\text{Squeeze \%tile} = \frac{\sum_{i=1}^{252} \mathbb{I}(\text{BandWidth}_i < \text{BandWidth}_{\text{current}})}{252} \times 100$$

- **`SQUEEZE` ($\text{Percentile} \le 20\%$):** Volatility is compressed to historical extremes. Price energy is coiled; a powerful directional breakout is imminent.
- **`EXPANSION` ($\text{Percentile} \ge 80\%$ with rising ATR):** The squeeze has released; directional trend expansion is actively underway.
- **`NORMAL` ($20\% < \text{Percentile} < 80\%$):** Standard volatility environment.

---

## 6. Dimension 4: Volume & Institutional Participation

### 6.1 Volume Ratio vs. 20-Day Trailing Average
Volume validates or invalidates price moves. We compute the current candle's volume relative to the prior 20-day mean:

$$\text{Volume Ratio} = \frac{\text{Volume}_{\text{today}}}{\frac{1}{20} \sum_{i=1}^{20} \text{Volume}_{t-i}}$$

| Classification | Volume Ratio Threshold | Swing Trading Significance |
| :--- | :--- | :--- |
| **`SURGING`** | $\ge 2.0\times$ | Institutional block accumulation or distribution. Breakout confirmation. |
| **`ABOVE_AVG`** | $1.3\times \text{ to } 2.0\times$ | Healthy institutional interest supporting the candle. |
| **`NORMAL`** | $0.8\times \text{ to } 1.3\times$ | Standard retail and institutional flow. |
| **`CONTRACTING`** | $0.5\times \text{ to } 0.85\times$ | **Ideal for Pullbacks.** Sellers are exhausted; no institutional dumping. |
| **`VERY_LOW`** | $< 0.5\times$ | Lack of liquidity or conviction. |

### 6.2 On-Balance Volume (OBV) Trend Slope
OBV accumulates volume on up-days and subtracts volume on down-days:

$$\text{OBV}_t = \text{OBV}_{t-1} + \begin{cases} \text{Volume}_t & \text{if Close}_t > \text{Close}_{t-1} \\ 0 & \text{if Close}_t = \text{Close}_{t-1} \\ -\text{Volume}_t & \text{if Close}_t < \text{Close}_{t-1} \end{cases}$$

To eliminate noise, we fit a 10-bar linear regression line over the normalized OBV values:

$$\text{Slope} = \frac{N \sum(x y) - \sum x \sum y}{N \sum x^2 - (\sum x)^2}, \quad \text{Normalized Slope} = \frac{\text{Slope}}{\text{Mean}(|\text{OBV}|)}$$

- **`UPTREND` ($\text{Slope} > +0.003$):** Smart money is accumulating shares.
- **`DOWNTREND` ($\text{Slope} < -0.003$):** Institutional distribution. Disqualifies long entries.
- **`FLAT`:** Neutral institutional flow.

---

## 7. Dimension 5: Structural Geometry & Levels

### 7.1 Fibonacci Retracements (60-Candle Swing Detection)
The algorithm inspects the last 60 daily candles (~3 calendar months) to locate the rolling extremes:

$$\text{Swing High} = \max(\text{High}_{t-60 \dots t}), \quad \text{Swing Low} = \min(\text{Low}_{t-60 \dots t})$$

$$\Delta_{\text{Range}} = \text{Swing High} - \text{Swing Low}$$

Retracement levels are projected downward from the high:
- $\text{Fib}_{23.6\%} = \text{High} - (0.236 \times \Delta_{\text{Range}})$
- $\text{Fib}_{38.2\%} = \text{High} - (0.382 \times \Delta_{\text{Range}})$
- $\text{Fib}_{50.0\%} = \text{High} - (0.500 \times \Delta_{\text{Range}})$ (Gann/Dow 50% equilibrium)
- $\text{Fib}_{61.8\%} = \text{High} - (0.618 \times \Delta_{\text{Range}})$ (The Golden Ratio)
- $\text{Fib}_{78.6\%} = \text{High} - (0.786 \times \Delta_{\text{Range}})$ (Deep institutional defense level)

**Proximity Formula:** Price is flagged as "at structural support" if:
$$\frac{|\text{Price} - \text{Fib}_{\text{level}}|}{\text{Price}} \times 100 \le 1.5\%$$

### 7.2 Weekly Floor Pivot Points
Computed by resampling daily candles into weekly periods. For the current week, levels are derived from the completed prior week's High ($H$), Low ($L$), and Close ($C$):

$$\text{PP (Pivot Point)} = \frac{H + L + C}{3}$$

$$\text{R}_1 = (2 \times \text{PP}) - L, \quad \text{R}_2 = \text{PP} + (H - L)$$

$$\text{S}_1 = (2 \times \text{PP}) - H, \quad \text{S}_2 = \text{PP} - (H - L)$$

- **$\text{Price} > \text{PP}$:** Bullish weekly bias.
- **$\text{Price} < \text{PP}$:** Bearish weekly bias.

---

## 8. Strategy Classification & Confluence Engine

The system evaluates three swing trading archetypes:

### 8.1 Setup A: Momentum Pullback (The Core Swing Play)
*Philosophy: Buy strong stocks on brief, low-volume dips back to dynamic institutional support.*

#### 8-Point Confluence Checklist:
1. $\text{EMA}_{20} > \text{EMA}_{50}$ (Uptrend confirmed).
2. Price is within $\pm 1.5\%$ of $\text{EMA}_{20}$ or $\text{EMA}_{50}$.
3. $\text{ADX}_{14} \ge 20$ (Market is trending, not choppy).
4. $\text{RSI}_{14}$ is in the pullback cooling zone ($38 \le \text{RSI} \le 58$).
5. Volume is contracting ($< 0.90\times$ 20-day average).
6. OBV trend is `UPTREND` (Institutions are holding, not dumping).
7. Price is near Fibonacci $38.2\%$, $50.0\%$, or $61.8\%$ retracement ($\le 2.0\%$).
8. Price is above the weekly Pivot Point ($\text{PP}$).

**Qualification Threshold:** Minimum 4 out of 8 criteria satisfied, including mandatory criteria #1 and #4.

---

### 8.2 Setup B: Volume-Confirmed Breakout
*Philosophy: Capitalize on the release of coiled energy after a volatility squeeze.*

#### 6-Point Confluence Checklist:
1. Bollinger Band Squeeze active ($\text{BandWidth Percentile} < 30\%$).
2. Volume surging ($\ge 1.5\times$ 20-day average).
3. OBV trend is `UPTREND`.
4. $\text{ADX}_{14} < 45$ (New trend initiating, not exhausted).
5. $\text{RSI}_{14} < 70$ (Room to run before becoming overbought).
6. Price is above weekly Pivot Point ($\text{PP}$).

**Qualification Threshold:** Minimum 4 out of 6 criteria satisfied, including mandatory criteria #1 and #2.

---

### 8.3 Setup C: Oversold Reversal
*Philosophy: Catch deep capitulation bottoms at structural boundaries.*

#### 6-Point Confluence Checklist:
1. $\text{RSI}_{14} < 35$ (Extreme oversold).
2. Bullish RSI divergence confirmed (Price new low, RSI higher low).
3. Price at deep Fibonacci retracement ($61.8\%$ or $78.6\%$).
4. Price at or below weekly $\text{S}_1$ or $\text{S}_2$ pivot.
5. Volume spike on reversal candle ($\ge 1.3\times$).
6. OBV not in active downtrend.

**Qualification Threshold:** Minimum 4 out of 6 criteria satisfied, including mandatory criteria #1 and #2.

---

## 9. Deterministic Risk Management & Target Allocation

All risk levels are computed mathematically. **No human or LLM opinion can override these levels.**

### 9.1 Stop-Loss Formulation
$$\text{Stop-Loss} = \text{LTP} - (M_{\text{SL}} \times \text{ATR}_{14})$$

| Setup Type | $M_{\text{SL}}$ Multiplier | Rationale |
| :--- | :--- | :--- |
| **Momentum Pullback** | $2.0 \times \text{ATR}$ | Provides sufficient buffer below $\text{EMA}_{20}$ and the pullback swing low. |
| **Volume Breakout** | $1.5 \times \text{ATR}$ | Tight stop below the breakout resistance-turned-support level. |
| **Oversold Reversal** | $1.5 \times \text{ATR}$ | Tight stop just below the capitulation candle low. |

### 9.2 Risk and Asymmetric Targets
$$\text{Risk Per Share} = \text{LTP} - \text{Stop-Loss}$$

$$\text{Entry Zone} = [\text{LTP} \times 0.998, \; \text{LTP} \times 1.002]$$

$$\text{Target 1 (1:1.5 R:R)} = \text{LTP} + (1.5 \times \text{Risk Per Share})$$

$$\text{Target 2 (1:2.5 R:R)} = \text{LTP} + (2.5 \times \text{Risk Per Share})$$

$$\text{Target 3 (1:4.0 R:R)} = \text{LTP} + (4.0 \times \text{Risk Per Share})$$

### 9.3 Professional Position Sizing Formula
To risk exactly $1\%$ of account equity per trade:

$$\text{Position Quantity} = \left\lfloor \frac{\text{Total Account Capital} \times 0.01}{\text{Risk Per Share}} \right\rfloor$$

---

## 10. Market-Wide Screener & Automated 3-Tier Bucketing

To prevent analysis paralysis, the screener scans all constituents of Nifty 100 or Nifty 50 and partitions them into **three mutually exclusive buckets**:

```text
┌────────────────────────────────────────────────────────────────────────┐
│  🟢 BUCKET 1: PRIME SETUPS (Actionable Candidates — 3 to 7 Stocks)     │
│  - Signal == BUY                                                       │
│  - Confluence >= 4 / 8                                                 │
│  - TrendState != BEAR and != STRONG_BEAR                               │
│  - Defined ATR Risk Levels with R:R >= 1:1.5                           │
├────────────────────────────────────────────────────────────────────────┤
│  🟡 BUCKET 2: DEVELOPING / ON RADAR (Watchlist — 10 to 20 Stocks)      │
│  - TrendState == STRONG_BULL or BULL (or Price > EMA_50)               │
│  - Setup NOT yet triggered (Price extended >1.5% above EMA-20 or       │
│    RSI still cooling >58)                                              │
│  - Generates explicit trigger note: "Wait for retrace to ₹X,XXX"       │
├────────────────────────────────────────────────────────────────────────┤
│  ⚪ BUCKET 3: AVOID / STAY AWAY (The Remaining Market)                  │
│  - TrendState == BEAR or STRONG_BEAR (Price < EMA_50 or EMA_200)       │
│  - Choppy consolidation (ADX < 18) or severe downward momentum         │
│  - Capital preservation gate: Prevents buying declining assets         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 11. Multi-Factor Intra-Bucket Ranking Algorithm

Inside **Bucket 1 (Prime Setups)**, candidates are ranked from highest to lowest conviction using a normalized composite score:

$$\text{Rank Score} = (0.35 \times S_{\text{Confluence}}) + (0.25 \times S_{\text{Proximity}}) + (0.20 \times S_{\text{RR}}) + (0.20 \times S_{\text{Volume}})$$

Where each factor is scaled from $0$ to $100$:
1. **$S_{\text{Confluence}} = \left( \frac{\text{Confluence Count}}{\text{Max Confluence}} \right) \times 100$**  
   Measures structural depth. An $7/8$ setup scores $87.5$; a $4/8$ setup scores $50.0$.
2. **$S_{\text{Proximity}} = \max\left(0, \; 100 - (\text{Distance \% to EMA}_{20} \times 25)\right)$**  
   Rewards stocks sitting directly at their support zone. A stock $0.2\%$ from support scores $95$; a stock $2.0\%$ away scores $50$.
3. **$S_{\text{RR}} = \min\left(100, \; \frac{\text{Target 2 R:R Ratio}}{2.5} \times 100\right)$**  
   Measures reward asymmetry.
4. **$S_{\text{Volume}} = \min\left(100, \; \max(20, \; \text{Volume Ratio} \times 50)\right)$**  
   Measures institutional footprint.

### Intra-Bucket Sorting for Buckets 2 and 3:
- **Bucket 2 (Developing):** Sorted by $|\text{Proximity \% to EMA}_{20}|$. The stock closest to hitting its pullback trigger ranks #1.
- **Bucket 3 (Avoid):** Sorted by ascending RSI (most damaged momentum first).

---

## 12. External Market Context Integration

### 12.1 India VIX (The Fear Gauge)
Fetched via Zerodha Kite (Instrument Token: `264969`).

| India VIX Reading | Market Regime | Tactical Adjustment |
| :--- | :--- | :--- |
| **$< 12$** | `LOW` | Complacency / calm market. Directional swing longs have highest follow-through. |
| **$12 \le \text{VIX} \le 18$** | `NORMAL` | Standard healthy trading environment. Normal risk parameters. |
| **$18 < \text{VIX} \le 25$** | `ELEVATED` | Heightened uncertainty. Tighten stop-losses; avoid wider multi-day targets. |
| **$25 < \text{VIX} \le 35$** | `HIGH` | High fear. Reduce position sizes by $50\%$. |
| **$> 35$** | `EXTREME` | Market panic. Avoid all new long swing entries. |

### 12.2 Institutional Flow (FII / DII 5-Day Net Equity Flow)
Fetched via NSE India public API:
- Aggregates net equity purchases of Foreign Institutional Investors (FII) and Domestic Institutional Investors (DII) over the last 5 trading sessions.
- Classified as `STRONG_FII_BUYING` ($>+₹2,000 \text{ Cr}$), `FII_BUYING`, `MIXED`, or `FII_SELLING`.

### 12.3 News Catalyst Pipeline
Queries Google News RSS for the stock's NSE ticker, filters headlines from the preceding 3 days, strips syndicate tags, and surfaces the top 3 corporate/regulatory catalysts.

---

## 13. Artificial Intelligence Synthesis & Strict Guardrails

### 13.1 LLM Role Boundary
The Large Language Model (Gemini 1.5 Flash) operates under **strict system instructions**:
1. It receives a pre-compiled JSON payload containing all pre-computed mathematical figures.
2. It is **explicitly forbidden** from calculating or modifying any price, stop-loss, target, or indicator value.
3. It generates a concise 4–6 sentence analyst narrative that weaves the 5-dimension technical diagnostic with VIX, FII flow, and news catalysts.
4. It highlights the single most critical risk and the key level to monitor during the trade.

### 13.2 Deterministic Rule-Based Fallback
If the Gemini API key is absent, rate-limited, or unavailable, the system automatically engages the deterministic fallback engine. It dynamically constructs structured prose from the indicator state descriptions with zero external API dependencies.

---

## 14. API Endpoints & Integration Contract

The FastAPI backend (`http://127.0.0.1:8000`) exposes the following endpoints:

| Endpoint | Method | Payload | Description |
| :--- | :--- | :--- | :--- |
| **`/api/health`** | `GET` | None | Service liveness probe. |
| **`/api/auth/login-url`** | `GET` | None | Returns Kite OAuth redirect URL. |
| **`/api/auth/callback`** | `GET` | `?request_token=...` | Exchanges token and saves session. |
| **`/api/auth/status`** | `GET` | None | Returns active user name and connection status. |
| **`/api/analyze`** | `POST` | `{"symbol": "RELIANCE", "interval": "day"}` | Complete 5-dimension single-stock diagnostic, risk matrix, and AI narrative. |
| **`/api/screener`** | `POST` | `{"universe": "nifty100", "max_stocks": 100, "interval": "day"}` | Universe scan, 3-tier bucketing, intra-bucket ranking, and market breadth. |
| **`/api/signals`** | `POST` | `{"short_sma": 6, "long_sma": 30, ...}` | Historical SMA crossover scanner. |

---
*End of Technical Specification Document.*
