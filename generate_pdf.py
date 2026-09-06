"""
generate_pdf.py
Generates an institutional-grade, publication-quality PDF manual of the
Indian Equity Technical Advisory Agent & Swing Screener Technical Specification.
Exactly 8 pages, with every page meticulously balanced and thematic.
"""

import os
import sys
import shutil
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, Preformatted
)
from reportlab.pdfgen import canvas

# --- Numbered Canvas for Dynamic Page Numbering and Running Headers ---
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        if self._pageNumber == 1:
            # Suppress header and footer on cover / title block
            return

        self.saveState()
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#64748b"))

        # Running Header
        self.drawString(40, 815, "INDIAN EQUITY ADVISORY AGENT  |  TECHNICAL & ALGORITHM REFERENCE MANUAL")
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.75)
        self.line(40, 808, 555, 808)

        # Running Footer
        self.line(40, 42, 555, 42)
        self.drawString(40, 30, "CONFIDENTIAL & PROPRIETARY  --  FOR DECISION SUPPORT & RISK GOVERNANCE ONLY")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(555, 30, page_str)

        self.restoreState()


def build_pdf(output_path):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=46,
        bottomMargin=48
    )

    styles = getSampleStyleSheet()

    # Color Palette
    C_NAVY    = colors.HexColor("#0f172a")
    C_BLUE    = colors.HexColor("#1e40af")
    C_CYAN    = colors.HexColor("#0284c7")
    C_GREEN   = colors.HexColor("#047857")
    C_AMBER   = colors.HexColor("#b45309")
    C_TEXT    = colors.HexColor("#1e293b")
    C_MUTED   = colors.HexColor("#475569")
    C_BG_CARD = colors.HexColor("#f8fafc")
    C_BORDER  = colors.HexColor("#cbd5e1")

    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=25,
        textColor=C_NAVY,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14.5,
        textColor=C_BLUE,
        spaceAfter=10
    )

    h1_style = ParagraphStyle(
        'SecH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=C_NAVY,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'SecH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13,
        textColor=C_BLUE,
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.2,
        leading=12,
        textColor=C_TEXT,
        spaceAfter=3
    )

    bullet_style = ParagraphStyle(
        'DocBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.2,
        leading=12,
        textColor=C_TEXT,
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=2
    )

    formula_style = ParagraphStyle(
        'FormulaText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.2,
        leading=12,
        textColor=C_NAVY,
        alignment=1
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.8,
        leading=10,
        textColor=colors.white
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.3,
        leading=10,
        textColor=C_TEXT
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.3,
        leading=10,
        textColor=C_NAVY
    )

    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=7.2,
        leading=9.8,
        textColor=colors.HexColor("#0f172a")
    )

    story = []

    # --- Helper Flowables ---
    def add_callout(text, title=None, border_color="#2563eb", bg_color="#f0f9ff"):
        content = []
        if title:
            content.append(Paragraph(f"<b>{title}</b>", ParagraphStyle('CalloutTitle', parent=body_style, fontName='Helvetica-Bold', textColor=colors.HexColor(border_color), spaceAfter=2)))
        content.append(Paragraph(text, body_style))
        t = Table([[content]], colWidths=[515])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(bg_color)),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor(border_color)),
            ('LINELEFT', (0,0), (0,0), 3.5, colors.HexColor(border_color)),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('LEFTPADDING', (0,0), (-1,-1), 9),
            ('RIGHTPADDING', (0,0), (-1,-1), 9),
        ]))
        story.append(t)
        story.append(Spacer(1, 4))

    def add_formula_card(formula_str, label=None):
        content = [Paragraph(f"<b>{formula_str}</b>", formula_style)]
        if label:
            content.append(Spacer(1, 2))
            content.append(Paragraph(f"<font color='#64748b' size=7>{label}</font>", ParagraphStyle('FormSub', parent=body_style, alignment=1, spaceAfter=0)))
        t = Table([[content]], colWidths=[515])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), C_BG_CARD),
            ('BOX', (0,0), (-1,-1), 0.5, C_BORDER),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING', (0,0), (-1,-1), 9),
            ('RIGHTPADDING', (0,0), (-1,-1), 9),
        ]))
        story.append(t)
        story.append(Spacer(1, 4))

    def add_styled_table(header_cols, data_rows, col_widths, header_bg="#0f172a"):
        header_cells = [Paragraph(f"<b>{h}</b>", table_header_style) for h in header_cols]
        table_data = [header_cells]
        for row in data_rows:
            row_cells = []
            for idx, c in enumerate(row):
                if idx == 0:
                    row_cells.append(Paragraph(c, table_cell_bold))
                else:
                    row_cells.append(Paragraph(c, table_cell_style))
            table_data.append(row_cells)

        t = Table(table_data, colWidths=col_widths)
        t_style = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(header_bg)),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 3.5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ]
        for i in range(1, len(table_data)):
            if i % 2 == 0:
                t_style.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor("#f8fafc")))
        t.setStyle(TableStyle(t_style))
        story.append(t)
        story.append(Spacer(1, 4))

    def add_ascii_diagram(diagram_str):
        p = Preformatted(diagram_str, code_style)
        t = Table([[p]], colWidths=[515])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f1f5f9")),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
            ('TOPPADDING', (0,0), (-1,-1), 7),
            ('BOTTOMPADDING', (0,0), (-1,-1), 7),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(t)
        story.append(Spacer(1, 4))

    # =========================================================================
    # PAGE 1: TITLE & EXECUTIVE SUMMARY (COVER PAGE)
    # =========================================================================
    story.append(Spacer(1, 10))
    story.append(Paragraph("Indian Equity Technical Advisory Agent & Swing Screener", title_style))
    story.append(Paragraph("Mathematical Reference Manual, Strategy Specifications & Algorithmic Audit", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=C_NAVY, spaceAfter=10))

    meta_data = [
        [
            Paragraph("<b>Target Horizon:</b> Swing (3-10 Day Hold)", body_style),
            Paragraph("<b>Document Version:</b> 2.0.0 (Production)", body_style),
            Paragraph("<b>Asset Universe:</b> NSE Nifty 100 / Nifty 50", body_style),
        ],
        [
            Paragraph("<b>Execution Type:</b> Non-Custodial Decision Support", body_style),
            Paragraph("<b>Date:</b> September 2026", body_style),
            Paragraph("<b>Status:</b> Validated Live on Kite Connect", body_style),
        ]
    ]
    meta_table = Table(meta_data, colWidths=[171, 171, 173])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), C_BG_CARD),
        ('BOX', (0,0), (-1,-1), 0.5, C_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 18))

    story.append(Paragraph("1. Executive Summary & System Philosophy", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BLUE, spaceAfter=8))

    story.append(Paragraph(
        "The Advisory System is engineered to address the twin vulnerabilities that undermine retail swing traders: "
        "<b>(1) The Discovery Dilemma</b> (missing high-probability setups developing at structural inflection zones due to the impossibility of manually auditing 100+ stock charts daily), and "
        "<b>(2) The Psychological Trap</b> (chasing overbought momentum, entering at extreme extensions, and trading without deterministic, mathematically bounded risk levels).",
        body_style
    ))
    story.append(Spacer(1, 10))

    add_callout(
        "<b>Core Architectural Axioms:</b><br/>"
        "&#8226; <b>Strictly Non-Custodial:</b> Zero automated order placement. The engine operates purely as an institutional-grade intelligence and risk-governance advisory terminal.<br/>"
        "&#8226; <b>Strict Separation of Math vs. Generative AI:</b> All trend states, momentum indicators, confluence tallies, stop-losses, and profit targets are strictly derived via <i>deterministic Python algorithms</i>. The Gemini 1.5 Flash LLM acts solely as a natural-language synthesis layer and is strictly prohibited from recalculating or altering any mathematical metric.<br/>"
        "&#8226; <b>Daily Swing Cadence:</b> Built upon Daily (EOD) OHLCV candles calibrated for 3-to-10-day holding periods, insulating the trader from intraday noise.",
        title="OPERATIONAL MANDATE & RISK GOVERNANCE",
        border_color="#047857",
        bg_color="#ecfdf5"
    )

    # =========================================================================
    # PAGE 2: ARCHITECTURE SPECIFICATION
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("2. End-to-End System Architecture", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BLUE, spaceAfter=6))

    arch_diagram = (
        "+-------------------------------------------------------------------------+\n"
        "|                        DATA INGESTION LAYER                             |\n"
        "|  - Zerodha Kite Connect API (Daily OHLCV Candles via Token Session)     |\n"
        "|  - Universe Constituents (Nifty 100 / Nifty 50 Constituents)            |\n"
        "|  - Local Parquet Cache with Daily Stale Invalidation                    |\n"
        "+-----------------------------------+-------------------------------------+\n"
        "                                    |\n"
        "                                    v\n"
        "+-------------------------------------------------------------------------+\n"
        "|               5-DIMENSION MATHEMATICAL ANALYSIS ENGINE                  |\n"
        "|  1. Trend:      EMA-20, EMA-50, EMA-200, ADX-14, DI+, DI-               |\n"
        "|  2. Momentum:   RSI-14, Bull/Bear Divergence, MACD (12, 26, 9)          |\n"
        "|  3. Volatility: ATR-14, Bollinger Bands (20, 2s), Squeeze Percentile    |\n"
        "|  4. Volume:     Volume Ratio vs 20d Mean, OBV Linear Regression Slope   |\n"
        "|  5. Structure:  60-Candle Swing Fibonacci Levels, Weekly Floor Pivots   |\n"
        "+-----------------------------------+-------------------------------------+\n"
        "                                    |\n"
        "                                    v\n"
        "+-------------------------------------------------------------------------+\n"
        "|              STRATEGY CLASSIFIER & RISK GOVERNANCE                      |\n"
        "|  - Setup A: Momentum Pullback (8-point Confluence Checklist)            |\n"
        "|  - Setup B: Volume-Confirmed Breakout (6-point Confluence Checklist)    |\n"
        "|  - Setup C: Oversold Reversal (6-point Confluence Checklist)            |\n"
        "|  - ATR Risk Matrix: Entry Zone, ATR Stop-Loss, Targets 1 / 2 / 3        |\n"
        "+------------------+----------------------------------+-------------------+\n"
        "                   |                                  |\n"
        "                   v [Single Stock Deep-Dive]         v [Universe Batch]\n"
        "+------------------------------------+ +----------------------------------+\n"
        "|      EXTERNAL CONTEXT & AI         | |   MARKET SCREENER & BUCKETING    |\n"
        "|  - India VIX Fear Regime & Trend   | |  - Bucket 1: Prime Setups (>=4/8)|\n"
        "|  - NSE FII/DII 5-Day Net Cash Flow | |  - Bucket 2: Developing / Radar  |\n"
        "|  - Google News RSS Catalyst Radar  | |  - Bucket 3: Avoid / Broken      |\n"
        "|  - Gemini 1.5 Flash Synthesis      | |  - Intra-Bucket Ranking (0-100)  |\n"
        "+------------------+-----------------+ +------------------+---------------+\n"
        "                   |                                      |\n"
        "                   +------------------+-------------------+\n"
        "                                      |\n"
        "                                      v\n"
        "+-------------------------------------------------------------------------+\n"
        "|                  REACT / VITE DASHBOARD FRONTEND                        |\n"
        "|  - Market Screener View (Breadth Ratio, 3 Bucket Tabs, 1-Click Drilldown)|\n"
        "|  - Single Stock Advisor Desk (5 Dimension Cards, Risk Box, AI Narrative)|\n"
        "+-------------------------------------------------------------------------+"
    )
    add_ascii_diagram(arch_diagram)
    story.append(Spacer(1, 8))

    add_callout(
        "<b>Architectural Pipeline Summary:</b><br/>"
        "1. <b>Data Layer:</b> Ingests historical daily candles directly from Zerodha Kite Connect with intelligent disk caching.<br/>"
        "2. <b>Analytics Engine:</b> Computes 5 mathematical dimensions with zero forward-looking bias.<br/>"
        "3. <b>Strategy & Risk Layer:</b> Evaluates 8-point and 6-point confluence criteria and produces strict ATR price envelopes.<br/>"
        "4. <b>Distribution Layer:</b> Feeds both the single-stock deep diagnostic and the market-wide 3-tier screener simultaneously.",
        title="SYSTEM PIPELINE & DATA FLOW",
        border_color="#1e40af",
        bg_color="#f8fafc"
    )

    # =========================================================================
    # PAGE 3: DIMENSION 1 (TREND) & DIMENSION 2 (MOMENTUM)
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("3. Dimension 1: Trend Architecture (EMA Stack & ADX-14)", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BLUE, spaceAfter=5))

    story.append(Paragraph("<b>3.1 Exponential Moving Averages (EMA):</b>", h2_style))
    add_formula_card(
        "EMA<sub>t</sub> = (Close<sub>t</sub> &times; &alpha;) + (EMA<sub>t-1</sub> &times; (1 - &alpha;))",
        "Where &alpha;<sub>20</sub> = 2/21 &approx; 0.0952 (Fast Trend) | &alpha;<sub>50</sub> = 2/51 &approx; 0.0392 (Intermediate Trend) | &alpha;<sub>200</sub> = 2/201 &approx; 0.00995 (Macro Trend)"
    )

    story.append(Paragraph("<b>3.2 Trend Classification State Machine:</b>", h2_style))
    trend_headers = ["Trend State", "Mathematical Criterion", "Swing Trading Actionability"]
    trend_rows = [
        ["STRONG_BULL", "Price > EMA<sub>20</sub> > EMA<sub>50</sub> > EMA<sub>200</sub>", "All institutional horizons bullish. Prime candidate for Momentum Pullback."],
        ["BULL", "Price > EMA<sub>50</sub> and EMA<sub>20</sub> > EMA<sub>50</sub>", "Confirmed intermediate uptrend. Dips toward EMA<sub>20</sub> offer high R:R."],
        ["NEUTRAL", "Price between EMA<sub>20</sub> and EMA<sub>50</sub>, or EMAs entwined", "Consolidation or transition. Wait for clear expansion before allocating capital."],
        ["BEAR", "Price < EMA<sub>50</sub>", "Downward momentum dominant. Disqualified from long swing positions."],
        ["STRONG_BEAR", "Price < EMA<sub>200</sub> and EMA<sub>20</sub> < EMA<sub>50</sub>", "Macro institutional distribution. Strictly classified into Bucket 3 (Avoid)."]
    ]
    add_styled_table(trend_headers, trend_rows, [95, 195, 225])

    story.append(Paragraph("<b>3.3 Average Directional Index (ADX-14):</b>", h2_style))
    add_formula_card(
        "DX = (|DI<sup>+</sup> - DI<sup>-</sup>| / (DI<sup>+</sup> + DI<sup>-</sup>)) &times; 100    ==&gt;    ADX<sub>14</sub> = WilderSmooth(DX, 14)",
        "Where TR = max(H - L, |H - C<sub>prev</sub>|, |L - C<sub>prev</sub>|) and Directional Movement DM+, DM- are 14-period smoothed."
    )

    adx_headers = ["ADX-14 Range", "Regime Identifier", "Implication for Swing Traders"]
    adx_rows = [
        ["ADX < 20", "RANGING / CHOPPY", "Trendless consolidation. High false-breakout rate. Disqualifies trend strategies."],
        ["20 <= ADX < 25", "DEVELOPING", "Trend initiation phase. Early swing entry window."],
        ["25 <= ADX <= 40", "STRONG TREND", "Optimal swing trading environment. Highest statistical follow-through."],
        ["ADX > 40", "VERY STRONG / CLIMACTIC", "Mature trend extension. Trailing stops must be tightened; reversal risk elevated."]
    ]
    add_styled_table(adx_headers, adx_rows, [95, 140, 280])

    story.append(Paragraph("4. Dimension 2: Momentum & Divergence Dynamics (RSI & MACD)", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BLUE, spaceAfter=5))

    story.append(Paragraph("<b>4.1 Relative Strength Index (RSI-14):</b>", h2_style))
    add_formula_card(
        "RSI = 100 - (100 / (1 + RS)),    where RS = AvgGain<sub>14</sub> / AvgLoss<sub>14</sub>",
        "14-Period Exponential Smoothed Momentum Oscillator (Range: 0 to 100)"
    )

    rsi_headers = ["RSI Range", "Zone Classification", "Strategic Utility"]
    rsi_rows = [
        ["RSI > 70", "OVERBOUGHT", "Momentum overextended. New swing longs strictly prohibited. Wait for cooling."],
        ["55 < RSI <= 70", "MOMENTUM_ZONE", "Bullish thrust. Favorable for trailing runners; not optimal for fresh entry."],
        ["38 <= RSI <= 58", "PULLBACK_ZONE", "<b>The Primary Swing Entry Sweet-Spot.</b> Momentum has reset to healthy equilibrium."],
        ["30 <= RSI < 38", "WEAK", "Selling pressure dominant. Requires structural stabilization prior to entry."],
        ["RSI < 30", "OVERSOLD", "Capitulation extreme. Valid exclusively for Setup C (Oversold Reversal) at deep support."]
    ]
    add_styled_table(rsi_headers, rsi_rows, [95, 130, 290])

    story.append(Paragraph("<b>4.2 Automated Divergence & MACD (12, 26, 9):</b>", h2_style))
    story.append(Paragraph(
        "&#8226; <b>Bullish Divergence:</b> Price<sub>current</sub> &le; min(Price<sub>20</sub>)&times;1.01 AND RSI<sub>current</sub> &gt; min(RSI<sub>20</sub>)&times;1.05. Selling velocity dried up.<br/>"
        "&#8226; <b>MACD Crossover:</b> MACD Line = EMA<sub>12</sub>(Close) - EMA<sub>26</sub>(Close). Crossover above Signal line within 5 bars = CROSSOVER_BULLISH.",
        bullet_style
    ))

    # =========================================================================
    # PAGE 4: DIMENSION 3 (VOLATILITY) & DIMENSION 4 (VOLUME)
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("5. Dimension 3: Volatility Cycles (ATR & Bollinger Squeeze)", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BLUE, spaceAfter=5))

    story.append(Paragraph("<b>5.1 Average True Range (ATR-14):</b>", h2_style))
    add_formula_card(
        "ATR<sub>14</sub> = WilderSmooth(TR, 14)    |    ATR % = (ATR<sub>14</sub> / Price) &times; 100",
        "Dynamic volatility yardstick for risk management. Accounts for asset-specific price dispersion."
    )

    story.append(Paragraph("<b>5.2 Bollinger Bands (20, 2&sigma;) & Volatility Squeeze Percentile:</b>", h2_style))
    add_formula_card(
        "BandWidth % = ((UpperBand - LowerBand) / MiddleBand) &times; 100    [Upper/Lower = SMA<sub>20</sub> &plusmn; 2&sigma;]",
        "Squeeze Percentile = Percentage of past 252 trading sessions with BandWidth narrower than current"
    )

    bb_headers = ["Squeeze Percentile", "Volatility State", "Execution Significance"]
    bb_rows = [
        ["<= 20th Percentile", "SQUEEZE (Coiled)", "Extreme compression. Energy is stored; high-probability precursor to explosive breakout."],
        [">= 80th Percentile + Rising ATR", "EXPANSION", "Breakout released. Trend acceleration in progress; risk of late-stage exhaustion."],
        ["20th to 80th Percentile", "NORMAL", "Standard volatility behavior. Favorable for structured swing pullbacks."]
    ]
    add_styled_table(bb_headers, bb_rows, [110, 130, 275])

    story.append(Paragraph("6. Dimension 4: Volume & Institutional Participation", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BLUE, spaceAfter=5))

    story.append(Paragraph("<b>6.1 Volume Ratio vs. 20-Day Trailing Average:</b>", h2_style))
    add_formula_card(
        "Volume Ratio = Volume<sub>today</sub> / ( (1 / 20) &times; &sum;<sub>i=1</sub><sup>20</sup> Volume<sub>t-i</sub> )",
        "Volume validates price conviction. Low volume on pullbacks indicates an absence of institutional selling."
    )

    vol_headers = ["Volume Ratio", "Classification", "Swing Trading Interpretation"]
    vol_rows = [
        [">= 2.0x", "SURGING", "Institutional block accumulation. Mandatory for Setup B (Breakout) confirmation."],
        ["1.3x to 2.0x", "ABOVE_AVERAGE", "Healthy institutional sponsorship supporting the session."],
        ["0.8x to 1.3x", "NORMAL", "Standard baseline liquidity."],
        ["0.5x to 0.85x", "CONTRACTING", "<b>Optimal for Pullbacks (Setup A).</b> Confirms sellers are exhausted; no dumping."],
        ["< 0.5x", "VERY_LOW", "Lack of market participation. Reversals from this level lack follow-through."]
    ]
    add_styled_table(vol_headers, vol_rows, [95, 115, 305])

    story.append(Paragraph("<b>6.2 On-Balance Volume (OBV) Linear Regression Slope:</b>", h2_style))
    add_formula_card(
        "Normalized Slope = Slope<sub>10</sub>(OBV) / Mean(|OBV|)    [Thresholds: > +0.003 = UPTREND, < -0.003 = DOWNTREND]",
        "Disqualifies long setups when OBV is in an active DOWNTREND (indicating institutional distribution)."
    )

    # =========================================================================
    # PAGE 5: DIMENSION 5 (STRUCTURE) & STRATEGY CLASSIFIER
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("7. Dimension 5: Structural Geometry & Levels", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BLUE, spaceAfter=5))

    story.append(Paragraph("<b>7.1 60-Candle Swing Fibonacci Retracements:</b>", h2_style))
    story.append(Paragraph(
        "The engine establishes rolling swing boundaries over the trailing 60 trading sessions (~3 months):<br/>"
        "&Delta;(Range) = Swing High<sub>60</sub> - Swing Low<sub>60</sub>. Levels projected downward: "
        "<b>23.6%</b>, <b>38.2%</b>, <b>50.0%</b> (Gann/Dow equilibrium), <b>61.8%</b> (Golden Ratio), and <b>78.6%</b> (Deep defense).<br/>"
        "<i>Proximity Trigger:</i> Price is flagged at structural support when |Price - Fib<sub>level</sub>| / Price &le; 1.5%.",
        body_style
    ))

    story.append(Paragraph("<b>7.2 Weekly Floor Pivot Points:</b>", h2_style))
    add_formula_card(
        "PP = (H + L + C) / 3  |  R<sub>1</sub> = 2&times;PP - L  |  S<sub>1</sub> = 2&times;PP - H  |  R<sub>2</sub> = PP + (H - L)  |  S<sub>2</sub> = PP - (H - L)",
        "Derived from the completed prior weekly candle. Price > PP denotes a bullish weekly institutional bias."
    )

    story.append(Paragraph("8. Strategy Classification & Confluence Engine", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BLUE, spaceAfter=5))

    story.append(Paragraph("The system scores every stock against three institutional swing trading archetypes:", body_style))

    strat_headers = ["Strategy Archetype", "Key Confluence Checklist", "Qualification Bar"]
    strat_rows = [
        [
            "Setup A:<br/>Momentum Pullback<br/>(Core Swing Play)",
            "1. EMA<sub>20</sub> > EMA<sub>50</sub> (Mandatory)<br/>"
            "2. Price within &plusmn;1.5% of EMA<sub>20</sub> or EMA<sub>50</sub><br/>"
            "3. ADX<sub>14</sub> &ge; 20 (Trending market)<br/>"
            "4. RSI in cooling zone: 38 &le; RSI &le; 58 (Mandatory)<br/>"
            "5. Volume contracting (&lt; 0.90x 20d avg)<br/>"
            "6. OBV trend is UPTREND (Smart money accumulating)<br/>"
            "7. Price at Fibonacci 38.2%, 50%, or 61.8% level (&le; 2%)<br/>"
            "8. Price above Weekly Pivot Point (PP)",
            "<b>Min. 4 of 8 Criteria</b><br/>(Must satisfy #1 and #4)"
        ],
        [
            "Setup B:<br/>Volume-Confirmed Breakout",
            "1. Bollinger Squeeze active (Percentile &lt; 30%) (Mandatory)<br/>"
            "2. Volume surging &ge; 1.5x 20d avg (Mandatory)<br/>"
            "3. OBV trend is UPTREND<br/>"
            "4. ADX &lt; 45 (Emerging trend, not exhausted)<br/>"
            "5. RSI &lt; 70 (Room to expand)<br/>"
            "6. Price above Weekly Pivot Point (PP)",
            "<b>Min. 4 of 6 Criteria</b><br/>(Must satisfy #1 and #2)"
        ],
        [
            "Setup C:<br/>Oversold Reversal",
            "1. RSI<sub>14</sub> &lt; 35 (Extreme oversold) (Mandatory)<br/>"
            "2. Bullish RSI Divergence confirmed (Mandatory)<br/>"
            "3. Price at deep Fib (61.8% or 78.6%)<br/>"
            "4. Price at/below weekly S<sub>1</sub> or S<sub>2</sub> pivot<br/>"
            "5. Volume expansion &ge; 1.3x on turnaround candle<br/>"
            "6. OBV not in downward breakdown",
            "<b>Min. 4 of 6 Criteria</b><br/>(Must satisfy #1 and #2)"
        ]
    ]
    add_styled_table(strat_headers, strat_rows, [115, 290, 110])

    # =========================================================================
    # PAGE 6: RISK MANAGEMENT & MARKET SCREENER BUCKETING
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("9. Deterministic Risk Management & Asymmetric Targets", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BLUE, spaceAfter=5))

    story.append(Paragraph(
        "<b>No human or LLM opinion can override risk calculations.</b> Stop-losses and targets are strictly bounded by volatility:",
        body_style
    ))
    add_formula_card(
        "Stop-Loss = LTP - (M<sub>SL</sub> &times; ATR<sub>14</sub>)    |    Risk Per Share = LTP - Stop-Loss",
        "Where M<sub>SL</sub> = 2.0 for Pullback (Setup A) | M<sub>SL</sub> = 1.5 for Breakout (Setup B) | M<sub>SL</sub> = 1.5 for Reversal (Setup C)"
    )

    risk_headers = ["Execution Parameter", "Mathematical Formulation", "Risk-to-Reward Ratio (R:R)"]
    risk_rows = [
        ["Entry Zone", "[LTP &times; 0.998,  LTP &times; 1.002]", "Current baseline price buffer (&plusmn;0.2%)"],
        ["Stop-Loss", "LTP - (M<sub>SL</sub> &times; ATR<sub>14</sub>)", "1.0R (Risk Unit)"],
        ["Target 1 (Conservative)", "LTP + (1.5 &times; Risk Per Share)", "<b>1 : 1.5</b> (Take 50% profit; move SL to breakeven)"],
        ["Target 2 (Base Case)", "LTP + (2.5 &times; Risk Per Share)", "<b>1 : 2.5</b> (Take 30% profit; trail remaining)"],
        ["Target 3 (Runner)", "LTP + (4.0 &times; Risk Per Share)", "<b>1 : 4.0</b> (Full exit or trailing EMA<sub>20</sub>)"]
    ]
    add_styled_table(risk_headers, risk_rows, [125, 230, 160])

    add_callout(
        "<b>Institutional 1% Risk Position Sizing Rule:</b><br/>"
        "Position Quantity = floor( (Total Capital &times; 0.01) / Risk Per Share )<br/>"
        "<i>Example:</i> Account = Rs. 5,00,000 | 1% Risk = Rs. 5,000 | LTP = Rs. 2,500 | Stop-Loss = Rs. 2,420 (Risk = Rs. 80).<br/>"
        "Position Size = floor( 5,000 / 80 ) = <b>62 Shares</b> (Total Allocation: Rs. 1,55,000). Max loss on stop hit is exactly Rs. 4,960 (&le; 1%).",
        title="CAPITAL PRESERVATION GUARANTEE",
        border_color="#b45309",
        bg_color="#fffbeb"
    )

    story.append(Paragraph("10. Market-Wide Screener & Automated 3-Tier Bucketing", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BLUE, spaceAfter=5))

    story.append(Paragraph(
        "To eliminate information overload, the screener parses the full universe and automatically partitions stocks into <b>three mutually exclusive buckets</b>:",
        body_style
    ))

    bucket_headers = ["Bucket", "Strict Qualification Criteria", "User Interface & Action Protocol"]
    bucket_rows = [
        [
            "Bucket 1:<br/>PRIME SETUPS<br/>(Actionable Now)",
            "&#8226; Signal == BUY<br/>"
            "&#8226; Confluence Count &ge; 4 / 8<br/>"
            "&#8226; TrendState != BEAR and != STRONG_BEAR<br/>"
            "&#8226; Defined ATR Risk Matrix with R:R &ge; 1:1.5",
            "<b>Actionable candidates (3 to 7 stocks).</b><br/>"
            "Displays Entry Zone, SL, T1/T2/T3, Rank Score, and 1-Click Deep Diagnostic drilldown."
        ],
        [
            "Bucket 2:<br/>DEVELOPING<br/>(On Radar / Watchlist)",
            "&#8226; TrendState == STRONG_BULL or BULL (Price &gt; EMA<sub>50</sub>)<br/>"
            "&#8226; Setup NOT yet triggered (Extended &gt;1.5% above EMA<sub>20</sub> or RSI cooling &gt;58)<br/>"
            "&#8226; Structure is sound, but entry is premature",
            "<b>Watchlist candidates (10 to 20 stocks).</b><br/>"
            "Generates explicit tactical advice:<br/>"
            "<i>'Extended +4.8% above EMA-20. Wait for retrace to Rs. 2,450.'</i>"
        ],
        [
            "Bucket 3:<br/>AVOID / STAY AWAY<br/>(Capital Preservation)",
            "&#8226; TrendState == BEAR or STRONG_BEAR (Price &lt; EMA<sub>50</sub> or &lt; EMA<sub>200</sub>)<br/>"
            "&#8226; Choppy consolidation (ADX &lt; 18) or broken momentum<br/>"
            "&#8226; Institutional distribution (OBV downtrend)",
            "<b>The remaining market (Disqualified).</b><br/>"
            "Protects the trader from catching falling knives or holding dead money."
        ]
    ]
    add_styled_table(bucket_headers, bucket_rows, [110, 215, 190])

    # =========================================================================
    # PAGE 7: INTRA-BUCKET RANKING & EXTERNAL CONTEXT
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("11. Multi-Factor Intra-Bucket Ranking Algorithm", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BLUE, spaceAfter=5))

    story.append(Paragraph(
        "Inside <b>Bucket 1 (Prime Setups)</b>, candidates are sorted from highest to lowest conviction using a multi-factor composite formula yielding a score from 0 to 100:",
        body_style
    ))
    add_formula_card(
        "Rank Score = (0.35 &times; S<sub>Confluence</sub>) + (0.25 &times; S<sub>Proximity</sub>) + (0.20 &times; S<sub>RR</sub>) + (0.20 &times; S<sub>Volume</sub>)",
        "Composite intra-bucket score weighting structural strength, support proximity, payoff asymmetry, and volume conviction."
    )

    factor_headers = ["Score Component", "Normalized Mathematical Definition", "Strategic Objective"]
    factor_rows = [
        ["S_Confluence (35%)", "S<sub>Confluence</sub> = (Confluence Count / Max Possible) &times; 100", "Rewards setups satisfying 6/8 or 7/8 institutional conditions."],
        ["S_Proximity (25%)", "S<sub>Proximity</sub> = max( 0, 100 - (Distance % to EMA<sub>20</sub> &times; 25) )", "Rewards stocks sitting directly at support (0.2% away = 95 pts)."],
        ["S_RR (20%)", "S<sub>RR</sub> = min( 100, (Target 2 R:R / 2.5) &times; 100 )", "Rewards asymmetric payoff structures."],
        ["S_Volume (20%)", "S<sub>Volume</sub> = min( 100, max( 20, Volume Ratio &times; 50 ) )", "Rewards institutional volume confirmation."]
    ]
    add_styled_table(factor_headers, factor_rows, [110, 225, 180])

    story.append(Paragraph(
        "<b>Intra-Bucket Sorting for Watchlist and Avoid:</b><br/>"
        "&#8226; <b>Bucket 2 (Developing):</b> Sorted by |Proximity % to EMA<sub>20</sub>| ascending. The stock closest to triggering its pullback ranks #1.<br/>"
        "&#8226; <b>Bucket 3 (Avoid):</b> Sorted by ascending RSI (most technically broken stocks appear first).",
        bullet_style
    ))

    story.append(Paragraph("12. External Market Context Integration", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BLUE, spaceAfter=5))

    story.append(Paragraph("<b>12.1 India VIX (The Fear Gauge):</b>", h2_style))
    story.append(Paragraph(
        "Live India VIX is fetched via Zerodha Kite Connect (Instrument Token: <b>264969</b>) and tracked across 5 days to determine market fragility:",
        body_style
    ))

    vix_headers = ["India VIX Reading", "Regime Identifier", "Tactical Risk Protocol"]
    vix_rows = [
        ["VIX < 12", "LOW (Complacent)", "Calm environment. Directional swing longs have highest statistical follow-through."],
        ["12 <= VIX <= 18", "NORMAL (Healthy)", "Standard trading conditions. Full position sizing and normal targets."],
        ["18 < VIX <= 25", "ELEVATED (Cautious)", "Heightened volatility. Tighten stops; favor Target 1 exits over multi-day runners."],
        ["25 < VIX <= 35", "HIGH FEAR", "Risk-off environment. Reduce position sizing by 50%."],
        ["VIX > 35", "EXTREME PANIC", "Market turbulence. Avoid all fresh swing entries; preserve cash."]
    ]
    add_styled_table(vix_headers, vix_rows, [105, 130, 280])

    story.append(Paragraph("<b>12.2 Institutional Flow (FII / DII) & News Pipeline:</b>", h2_style))
    story.append(Paragraph(
        "&#8226; <b>NSE FII/DII Net Flow:</b> Parses official NSE cash market participant data across 5 days (e.g. STRONG_FII_BUYING &gt; +Rs. 2,000 Cr).<br/>"
        "&#8226; <b>Google News RSS Pipeline:</b> Queries RSS feeds within 72 hours, strips syndicated noise, and surfaces the top 3 corporate catalysts.",
        bullet_style
    ))

    # =========================================================================
    # PAGE 8: AI SYNTHESIS & API INTEGRATION CONTRACT
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("13. Artificial Intelligence Synthesis & Strict Guardrails", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BLUE, spaceAfter=5))

    add_callout(
        "<b>LLM Operational Boundary & Guardrails:</b><br/>"
        "1. <b>Zero Mathematical Authority:</b> The Gemini 1.5 Flash model receives a complete JSON dictionary containing pre-computed indicators, states, and risk levels. It is <i>strictly prohibited</i> from altering or computing numbers.<br/>"
        "2. <b>Institutional Prose Synthesis:</b> Translates the confluence findings, VIX state, FII flows, and news catalysts into a concise 4-6 sentence executive summary for the trader.<br/>"
        "3. <b>Deterministic Fallback:</b> If the Gemini API key is missing or rate-limited, the system automatically activates a local rule-based template engine that compiles structured commentary with zero external API dependencies.",
        title="AI ARCHITECTURAL SAFETY PROTOCOL",
        border_color="#0284c7",
        bg_color="#f0f9ff"
    )

    story.append(Paragraph("14. API Endpoints & Integration Contract", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BLUE, spaceAfter=5))

    story.append(Paragraph("The FastAPI backend (running on <b>http://127.0.0.1:8000</b>) exposes the following REST interface:", body_style))

    api_headers = ["Endpoint", "HTTP", "Payload / Query", "Operational Function"]
    api_rows = [
        ["/api/health", "GET", "None", "Liveness probe and system status."],
        ["/api/auth/login-url", "GET", "None", "Generates Zerodha Kite OAuth redirect URL."],
        ["/api/auth/callback", "GET", "?request_token=...", "Exchanges request token for daily access session."],
        ["/api/auth/status", "GET", "None", "Returns active user name ('Satyam Verma') and connection flag."],
        ["/api/analyze", "POST", '{"symbol": "TCS", "interval": "day"}', "Runs complete 5-dimension deep-dive, risk matrix, and AI narrative."],
        ["/api/screener", "POST", '{"universe": "nifty100", "max_stocks": 100}', "Scans universe, executes 3-tier bucketing, intra-bucket ranking, and breadth."],
        ["/api/signals", "POST", '{"short_sma": 6, "long_sma": 30}', "Executes historical SMA crossover scanner."]
    ]
    add_styled_table(api_headers, api_rows, [95, 45, 175, 200])

    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1, color=C_NAVY, spaceAfter=6))
    story.append(Paragraph(
        "<font color='#64748b' size=7.2><b>Document Sign-off:</b> Indian Equity Technical Advisory Agent Specification v2.0.0. "
        "All algorithms, mathematical formulas, and risk matrices are verified and operational against live Zerodha Kite Connect data feeds.</font>",
        body_style
    ))

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF successfully built at: {output_path}")


if __name__ == "__main__":
    output_pdf = r"C:\Users\satya\OneDrive\Documents\GitHub\trading-system\Trading_System_Technical_Specification.pdf"
    desktop_pdf = r"C:\Users\satya\OneDrive\Desktop\Trading_System_Technical_Specification.pdf"

    build_pdf(output_pdf)

    # Also copy to Desktop for convenient access
    try:
        shutil.copyfile(output_pdf, desktop_pdf)
        print(f"PDF successfully copied to Desktop at: {desktop_pdf}")
    except Exception as e:
        print(f"Could not copy to Desktop: {e}")
