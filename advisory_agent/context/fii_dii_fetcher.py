"""
advisory_agent/context/fii_dii_fetcher.py
Fetch FII / DII net institutional flow data from NSE India.

FII (Foreign Institutional Investors) and DII (Domestic Institutional
Investors) activity is a critical context factor for Indian swing trading:
  - Consistent FII net buying + bullish setup = higher conviction
  - FII net selling + bullish setup = be cautious (smart money leaving)
  - DII buying while FII selling = temporary support, but trend may continue

Data source: NSE India public FII/DII report (requires browser-like session).
Degrades gracefully to None if NSE blocks the request.
"""

from __future__ import annotations

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

NSE_BASE      = "https://www.nseindia.com"
NSE_FIIDII_EP = "/api/fiidii-dg-data?type=AllCat"

# Number of trading days to aggregate for net flow
NET_FLOW_DAYS = 5

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.nseindia.com/reports-indices-historical-vix",
}


def fetch_fii_dii() -> Optional[dict]:
    """
    Fetch last 5 days of FII and DII net equity flow from NSE.

    Returns:
        dict with keys:
            fii_net_5d_cr (float), dii_net_5d_cr (float),
            daily_rows (list of dicts),
            institutional_flow (str), description (str)
        Returns None on failure.
    """
    session = requests.Session()
    session.headers.update(_HEADERS)

    try:
        # Step 1: Hit homepage to get NSE session cookies
        session.get(NSE_BASE, timeout=10)

        # Step 2: Fetch FII/DII data
        resp = session.get(NSE_BASE + NSE_FIIDII_EP, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            logger.warning("NSE FII/DII: empty response.")
            return None

        return _parse_fii_dii(data)

    except requests.RequestException as exc:
        logger.warning("FII/DII fetch failed (non-fatal): %s", exc)
        return None
    except Exception as exc:
        logger.warning("FII/DII parse error (non-fatal): %s", exc)
        return None


def _parse_fii_dii(data: list) -> Optional[dict]:
    """
    Parse NSE FII/DII JSON response.
    NSE returns rows with category, buy value, sell value for each date.
    We aggregate equity purchases only.
    """
    try:
        # NSE structure: list of records with date, category (FII/DII), buy, sell, net
        # Filter for equity category only, last N days
        fii_rows = []
        dii_rows = []

        for row in data[:NET_FLOW_DAYS * 4]:  # NSE often returns mixed categories
            cat = str(row.get("category", "")).upper()
            seg = str(row.get("segment", "")).upper()
            # Only look at equity segment
            if "EQUIT" not in seg:
                continue

            net = _safe_float(row.get("netVal") or row.get("net") or row.get("Net"))
            if net is None:
                continue

            if "FII" in cat or "FPI" in cat:
                fii_rows.append(net)
            elif "DII" in cat:
                dii_rows.append(net)

        if not fii_rows and not dii_rows:
            logger.warning("Could not extract FII/DII rows from NSE response.")
            return None

        fii_net = sum(fii_rows[:NET_FLOW_DAYS])
        dii_net = sum(dii_rows[:NET_FLOW_DAYS])

        flow = _classify_flow(fii_net, dii_net)

        return {
            "fii_net_5d_cr":    round(fii_net, 2),
            "dii_net_5d_cr":    round(dii_net, 2),
            "institutional_flow": flow,
            "description":      _describe(fii_net, dii_net, flow),
        }

    except Exception as exc:
        logger.warning("FII/DII row parsing error: %s", exc)
        return None


def _safe_float(val) -> Optional[float]:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _classify_flow(fii_net: float, dii_net: float) -> str:
    if fii_net > 2000:
        return "STRONG_FII_BUYING"
    if fii_net > 500:
        return "FII_BUYING"
    if fii_net < -2000:
        return "STRONG_FII_SELLING"
    if fii_net < -500:
        return "FII_SELLING"
    if dii_net > 1000:
        return "DII_BUYING"
    return "MIXED"


def _describe(fii_net: float, dii_net: float, flow: str) -> str:
    fii_str = f"₹{abs(fii_net):,.0f} Cr net {'buying' if fii_net > 0 else 'selling'}"
    dii_str = f"₹{abs(dii_net):,.0f} Cr net {'buying' if dii_net > 0 else 'selling'}"

    base = f"FII 5-day flow: {fii_str}. DII 5-day flow: {dii_str}."

    context = {
        "STRONG_FII_BUYING":  " FIIs are aggressively buying — strong institutional tailwind.",
        "FII_BUYING":         " FIIs are net buyers — positive institutional flow supports longs.",
        "FII_SELLING":        " FIIs are net sellers — institutional headwind. Use tighter stops.",
        "STRONG_FII_SELLING": " FIIs aggressively selling — significant institutional headwind. High caution.",
        "DII_BUYING":         " DIIs providing support while FIIs are mixed — temporary cushion.",
        "MIXED":              " No clear institutional direction. Rely on technical setup quality.",
    }.get(flow, "")

    return base + context
