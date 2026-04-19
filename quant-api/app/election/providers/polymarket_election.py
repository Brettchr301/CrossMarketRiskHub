"""Polymarket election market provider.

Fetches election prediction market quotes from Polymarket's Gamma + CLOB APIs.
Reuses the same API patterns as the existing RealPolymarketProvider.
"""
from __future__ import annotations
import json
import logging
import re
from datetime import UTC, datetime, timedelta
from typing import Any

import requests

logger = logging.getLogger(__name__)

BASE_GAMMA = "https://gamma-api.polymarket.com/markets"
BASE_CLOB = "https://clob.polymarket.com/prices-history"
TIMEOUT = 25


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def fetch_election_markets(search_terms: list[str] | None = None) -> list[dict[str, Any]]:
    """Search Polymarket for election-related markets.

    Returns list of market dicts with: id, question, bestBid, bestAsk, volume, etc.
    """
    if search_terms is None:
        search_terms = [
            "senate", "house", "governor", "president", "election",
            "congress", "midterm", "republican", "democrat",
        ]

    all_markets = []
    seen_ids = set()

    for term in search_terms:
        try:
            params = {"limit": 100, "active": "true", "closed": "false"}
            r = requests.get(BASE_GAMMA, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            markets = r.json() if isinstance(r.json(), list) else []

            pattern = re.compile(term, re.I)
            for m in markets:
                q = str(m.get("question", ""))
                mid = str(m.get("id", ""))
                if pattern.search(q) and mid not in seen_ids:
                    seen_ids.add(mid)
                    all_markets.append(m)
        except Exception as exc:
            logger.warning("Polymarket search failed for '%s': %s", term, exc)

    return all_markets


def fetch_quote(market_id: str) -> dict[str, Any] | None:
    """Fetch a single market quote by ID."""
    try:
        r = requests.get(BASE_GAMMA, params={"id": market_id}, timeout=TIMEOUT)
        r.raise_for_status()
        arr = r.json()
        if isinstance(arr, list) and arr:
            m = arr[0]
            bid = float(m.get("bestBid", 0.0) or 0.0)
            ask = float(m.get("bestAsk", 0.0) or 0.0)
            if bid <= 0 and ask <= 0:
                last = float(m.get("lastTradePrice", 0.0) or 0.0)
                bid, ask = max(0.0, last - 0.01), min(1.0, last + 0.01)
            volume = float(m.get("volume", 0.0) or 0.0)
            spread = abs(ask - bid)
            liq = max(0.01, min(1.0, 1.0 - spread * 4.0))
            return {
                "platform": "polymarket",
                "platform_market_id": market_id,
                "question": m.get("question", ""),
                "yes_bid": _clamp(bid),
                "yes_ask": _clamp(ask),
                "last_price": _clamp(float(m.get("lastTradePrice", 0.0) or 0.0)),
                "volume_24h": max(0.0, volume),
                "open_interest": 0.0,
                "liquidity_score": liq,
                "as_of": datetime.now(UTC).replace(tzinfo=None),
                "raw": m,
            }
    except Exception as exc:
        logger.warning("Polymarket quote fetch failed for %s: %s", market_id, exc)
    return None


def fetch_by_regex(pattern: str) -> list[dict[str, Any]]:
    """Search all active Polymarket markets by regex pattern.
    Returns normalized quotes.
    """
    regex = re.compile(pattern, re.I)
    quotes = []
    max_pages = 12

    for page in range(max_pages):
        try:
            params = {"limit": 500, "offset": page * 500, "active": "true", "closed": "false"}
            r = requests.get(BASE_GAMMA, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            arr = r.json()
            if not arr:
                break
            for m in arr:
                if regex.search(str(m.get("question", ""))):
                    bid = float(m.get("bestBid", 0.0) or 0.0)
                    ask = float(m.get("bestAsk", 0.0) or 0.0)
                    if bid <= 0 and ask <= 0:
                        last = float(m.get("lastTradePrice", 0.0) or 0.0)
                        bid, ask = max(0.0, last - 0.01), min(1.0, last + 0.01)
                    spread = abs(ask - bid)
                    quotes.append({
                        "platform": "polymarket",
                        "platform_market_id": str(m.get("id", "")),
                        "question": m.get("question", ""),
                        "yes_bid": _clamp(bid),
                        "yes_ask": _clamp(ask),
                        "last_price": _clamp(float(m.get("lastTradePrice", 0.0) or 0.0)),
                        "volume_24h": float(m.get("volume", 0.0) or 0.0),
                        "open_interest": 0.0,
                        "liquidity_score": max(0.01, min(1.0, 1.0 - spread * 4.0)),
                        "as_of": datetime.now(UTC).replace(tzinfo=None),
                    })
        except Exception as exc:
            logger.warning("Polymarket page %d search failed: %s", page, exc)
            break

    return quotes
