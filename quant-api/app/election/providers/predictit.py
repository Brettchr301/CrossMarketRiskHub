"""PredictIt election market provider.

Fetches from PredictIt's public JSON API.
Rate limit: ~1 request per minute.
"""
from __future__ import annotations
import logging
import re
from datetime import UTC, datetime
from typing import Any

import requests

logger = logging.getLogger(__name__)

API_URL = "https://www.predictit.org/api/marketdata/all/"
TIMEOUT = 30


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def fetch_all_markets() -> list[dict[str, Any]]:
    """Fetch all PredictIt markets and filter for elections."""
    try:
        r = requests.get(API_URL, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        markets = data.get("markets", [])

        election_pattern = re.compile(
            r"senate|house|governor|president|election|congress|"
            r"midterm|republican|democrat|electoral|nominee|primary",
            re.I,
        )

        quotes = []
        for market in markets:
            name = market.get("name", "")
            if not election_pattern.search(name):
                continue

            for contract in market.get("contracts", []):
                last = float(contract.get("lastTradePrice", 0) or 0)
                bid = float(contract.get("bestBuyYesCost", 0) or 0)
                ask = float(contract.get("bestSellYesCost", 0) or 0)

                # PredictIt prices are already 0-1
                if bid <= 0 and ask <= 0:
                    bid = max(0.0, last - 0.01)
                    ask = min(1.0, last + 0.01)

                spread = abs(ask - bid) if ask > bid else 0.05
                quotes.append({
                    "platform": "predictit",
                    "platform_market_id": str(contract.get("id", "")),
                    "question": f"{name}: {contract.get('name', '')}",
                    "yes_bid": _clamp(bid),
                    "yes_ask": _clamp(ask),
                    "last_price": _clamp(last),
                    "volume_24h": 0.0,  # PredictIt doesn't expose volume
                    "open_interest": 0.0,
                    "liquidity_score": max(0.01, min(1.0, 1.0 - spread * 3.0)),
                    "as_of": datetime.now(UTC).replace(tzinfo=None),
                    "market_name": name,
                    "contract_name": contract.get("name", ""),
                })

        logger.info("PredictIt: found %d election contracts", len(quotes))
        return quotes
    except Exception as exc:
        logger.error("PredictIt fetch failed: %s", exc)
        return []
