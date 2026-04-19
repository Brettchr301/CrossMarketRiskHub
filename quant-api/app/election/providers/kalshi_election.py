"""Kalshi election market provider.

Fetches election prediction market quotes from Kalshi's public API.
"""
from __future__ import annotations
import logging
from datetime import UTC, datetime
from typing import Any

import requests

logger = logging.getLogger(__name__)

BASE = "https://api.elections.kalshi.com/trade-api/v2"
TIMEOUT = 20


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


# Known Kalshi election event tickers (discovered via API exploration)
ELECTION_TICKERS = {
    "pres_2028": "PRES-2028",
    "senate_2026_control": "KXSENATE2026",
    "house_2026_control": "KXHOUSE2026",
    "senate_pa_2026": "KXSENPA26",
    "senate_az_2026": "KXSENAZ26",
    "senate_ga_2026": "KXSENGA26",
    "senate_mi_2026": "KXSENMI26",
    "senate_wi_2026": "KXSENWI26",
    "senate_nv_2026": "KXSENNV26",
    "senate_nc_2026": "KXSENNC26",
    "senate_mn_2026": "KXSENMN26",
    "senate_nh_2026": "KXSENNH26",
    "senate_me_2026": "KXSENME26",
    "senate_co_2026": "KXSENCO26",
    "gov_2026_control": "KXGOV2026",
}


def fetch_by_event_ticker(event_ticker: str) -> list[dict[str, Any]]:
    """Fetch all markets for a Kalshi event ticker."""
    try:
        params = {"event_ticker": event_ticker, "limit": 200}
        r = requests.get(f"{BASE}/markets", params=params, timeout=TIMEOUT)
        r.raise_for_status()
        markets = r.json().get("markets", [])

        quotes = []
        for m in markets:
            yes_bid = float(m.get("yes_bid", 0) or 0) / 100.0
            yes_ask = float(m.get("yes_ask", 0) or 0) / 100.0
            if yes_bid <= 0 and yes_ask <= 0:
                last = float(m.get("last_price", 0) or 0) / 100.0
                yes_bid, yes_ask = max(0.0, last - 0.01), min(1.0, last + 0.01)

            volume = float(m.get("volume", 0) or 0)
            oi = float(m.get("open_interest", 0) or 0)
            spread = abs(yes_ask - yes_bid)
            liq = max(0.01, min(1.0, 1.0 - spread * 5.0))

            quotes.append({
                "platform": "kalshi",
                "platform_market_id": m.get("ticker", ""),
                "question": m.get("title", m.get("subtitle", "")),
                "yes_bid": _clamp(yes_bid),
                "yes_ask": _clamp(yes_ask),
                "last_price": _clamp(float(m.get("last_price", 0) or 0) / 100.0),
                "volume_24h": volume,
                "open_interest": oi,
                "liquidity_score": liq,
                "as_of": datetime.now(UTC).replace(tzinfo=None),
            })
        return quotes
    except Exception as exc:
        logger.warning("Kalshi fetch failed for %s: %s", event_ticker, exc)
        return []


def fetch_all_election_markets() -> dict[str, list[dict[str, Any]]]:
    """Fetch quotes for all known election tickers."""
    results = {}
    for event_id, ticker in ELECTION_TICKERS.items():
        quotes = fetch_by_event_ticker(ticker)
        if quotes:
            results[event_id] = quotes
    return results


def search_election_markets(keyword: str = "election") -> list[dict[str, Any]]:
    """Search Kalshi markets by keyword."""
    try:
        params = {"limit": 200, "status": "open"}
        r = requests.get(f"{BASE}/markets", params=params, timeout=TIMEOUT)
        r.raise_for_status()
        markets = r.json().get("markets", [])

        kw_lower = keyword.lower()
        election_markets = [
            m for m in markets
            if kw_lower in str(m.get("title", "")).lower()
            or kw_lower in str(m.get("subtitle", "")).lower()
            or kw_lower in str(m.get("event_ticker", "")).lower()
        ]

        quotes = []
        for m in election_markets:
            yes_bid = float(m.get("yes_bid", 0) or 0) / 100.0
            yes_ask = float(m.get("yes_ask", 0) or 0) / 100.0
            spread = abs(yes_ask - yes_bid)
            quotes.append({
                "platform": "kalshi",
                "platform_market_id": m.get("ticker", ""),
                "question": m.get("title", ""),
                "yes_bid": _clamp(yes_bid),
                "yes_ask": _clamp(yes_ask),
                "last_price": _clamp(float(m.get("last_price", 0) or 0) / 100.0),
                "volume_24h": float(m.get("volume", 0) or 0),
                "open_interest": float(m.get("open_interest", 0) or 0),
                "liquidity_score": max(0.01, min(1.0, 1.0 - spread * 5.0)),
                "as_of": datetime.now(UTC).replace(tzinfo=None),
            })
        return quotes
    except Exception as exc:
        logger.warning("Kalshi search failed: %s", exc)
        return []
