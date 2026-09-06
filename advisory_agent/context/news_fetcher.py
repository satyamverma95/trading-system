"""
advisory_agent/context/news_fetcher.py
Fetch recent news headlines for a stock symbol using Google News RSS.

The news layer adds context the chart cannot show: regulatory actions,
management changes, quarterly result commentary, sector tailwinds/headwinds,
and global macro events that affect the specific company.

Source: Google News RSS (free, no API key required, rate-limit friendly).
Returns at most 5 headlines from the last 2 days.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from typing import Optional
from urllib.parse import quote_plus
from datetime import datetime, timedelta, timezone

import requests

logger = logging.getLogger(__name__)

NEWS_MAX_HEADLINES  = 5
NEWS_LOOKBACK_DAYS  = 3
REQUEST_TIMEOUT     = 8

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}


def fetch_news(symbol: str) -> Optional[dict]:
    """
    Fetch recent news headlines for the given NSE symbol.

    Args:
        symbol: NSE ticker symbol (e.g. "RELIANCE", "HDFCBANK").

    Returns:
        dict with keys: headlines (list[str]), sources (list[str]), description (str)
        Returns None on failure.
    """
    try:
        # Build search query: symbol + NSE / India context
        query = f"{symbol} NSE India stock"
        rss_url = (
            f"https://news.google.com/rss/search?"
            f"q={quote_plus(query)}&hl=en-IN&gl=IN&ceid=IN:en"
        )

        resp = requests.get(rss_url, headers=_HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()

        headlines, sources = _parse_rss(resp.text)

        if not headlines:
            return {"headlines": [], "sources": [], "description": f"No recent news found for {symbol}."}

        description = _describe(symbol, headlines)

        return {
            "headlines":   headlines,
            "sources":     sources,
            "description": description,
        }

    except Exception as exc:
        logger.warning("News fetch failed for %s (non-fatal): %s", symbol, exc)
        return None


def _parse_rss(xml_text: str) -> tuple[list[str], list[str]]:
    """Parse Google News RSS XML and return (headlines, sources)."""
    headlines = []
    sources   = []

    try:
        root = ET.fromstring(xml_text)
        channel = root.find("channel")
        if channel is None:
            return [], []

        cutoff = datetime.now(timezone.utc) - timedelta(days=NEWS_LOOKBACK_DAYS)

        for item in channel.findall("item"):
            if len(headlines) >= NEWS_MAX_HEADLINES:
                break

            title_el = item.find("title")
            if title_el is None or not title_el.text:
                continue

            title = _clean_title(title_el.text)
            if not title:
                continue

            # Try to parse pubDate and filter by recency
            pubdate_el = item.find("pubDate")
            if pubdate_el is not None and pubdate_el.text:
                try:
                    import email.utils
                    pub_ts = email.utils.parsedate_to_datetime(pubdate_el.text)
                    if pub_ts.tzinfo is None:
                        pub_ts = pub_ts.replace(tzinfo=timezone.utc)
                    if pub_ts < cutoff:
                        continue
                except Exception:
                    pass  # If we can't parse date, include it anyway

            source_el = item.find("source")
            source    = source_el.text if source_el is not None and source_el.text else "Unknown"

            headlines.append(title)
            sources.append(source)

    except ET.ParseError as exc:
        logger.warning("RSS parse error: %s", exc)

    return headlines, sources


def _clean_title(title: str) -> str:
    """Remove source attribution appended by Google News (e.g. ' - The Hindu')."""
    # Google appends " - Source Name" at the end
    title = re.sub(r"\s+-\s+[^-]+$", "", title).strip()
    return title


def _describe(symbol: str, headlines: list[str]) -> str:
    if not headlines:
        return f"No recent news for {symbol}."
    return (
        f"{len(headlines)} recent headline(s) found for {symbol}. "
        "Review for any fundamental catalysts or risk events that may affect the trade."
    )
