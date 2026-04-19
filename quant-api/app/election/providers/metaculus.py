"""Metaculus election forecast provider.

Fetches community forecasts from Metaculus API.
"""
from __future__ import annotations
import logging
import re
from datetime import UTC, datetime
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Metaculus API v1 (v2 now returns 403 without auth)
BASE_URL = "https://www.metaculus.com/api/questions/"
TIMEOUT = 25


def fetch_election_questions(
    search: str = "US election",
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Fetch election-related questions from Metaculus."""
    try:
        params = {
            "search": search,
            "limit": limit,
            "status": "open",
            "order_by": "-activity",
            "type": "forecast",
        }
        r = requests.get(BASE_URL, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        questions = data.get("results", [])

        election_pattern = re.compile(
            r"senate|house|governor|president|election|congress|"
            r"midterm|republican|democrat|nominee|primary|2026|2028",
            re.I,
        )

        quotes = []
        for q in questions:
            title = q.get("title", "")
            if not election_pattern.search(title):
                continue

            # Extract community prediction
            prediction = q.get("community_prediction", {})
            if not prediction:
                continue

            # For binary questions
            full = prediction.get("full", {})
            q2 = full.get("q2")  # median
            q1 = full.get("q1")  # 25th percentile
            q3 = full.get("q3")  # 75th percentile

            if q2 is None:
                continue

            prob = float(q2)
            ci_low = float(q1) if q1 is not None else max(0, prob - 0.1)
            ci_high = float(q3) if q3 is not None else min(1, prob + 0.1)

            spread = ci_high - ci_low
            quotes.append({
                "platform": "metaculus",
                "platform_market_id": str(q.get("id", "")),
                "question": title,
                "yes_bid": max(0.0, prob - spread / 4),
                "yes_ask": min(1.0, prob + spread / 4),
                "last_price": prob,
                "volume_24h": float(q.get("number_of_predictions", 0) or 0),
                "open_interest": 0.0,
                "liquidity_score": min(1.0, float(q.get("number_of_predictions", 0) or 0) / 200.0),
                "as_of": datetime.now(UTC).replace(tzinfo=None),
                "ci_low": ci_low,
                "ci_high": ci_high,
            })

        logger.info("Metaculus: found %d election questions", len(quotes))
        return quotes
    except Exception as exc:
        logger.error("Metaculus fetch failed: %s", exc)
        return []
