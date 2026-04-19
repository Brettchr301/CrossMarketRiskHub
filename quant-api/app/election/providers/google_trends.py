"""Google Trends provider.

Uses pytrends library with exponential backoff for rate limiting.
"""
from __future__ import annotations
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

try:
    from pytrends.request import TrendReq
    HAS_PYTRENDS = True
except ImportError:
    HAS_PYTRENDS = False
    logger.warning("pytrends not installed; Google Trends provider disabled")


def fetch_candidate_trends(
    candidates: list[str],
    timeframe: str = "today 3-m",
    geo: str = "US",
    state: str | None = None,
) -> dict[str, float] | None:
    """Fetch Google Trends interest for candidates.

    Returns {candidate_name: relative_interest_score}.
    """
    if not HAS_PYTRENDS:
        return None

    if not candidates:
        return None

    # Limit to 5 terms per request (pytrends limit)
    candidates = candidates[:5]

    geo_code = f"US-{state}" if state else geo

    for attempt in range(3):
        try:
            gt = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
            gt.build_payload(candidates, timeframe=timeframe, geo=geo_code)
            df = gt.interest_over_time()

            if df.empty:
                return None

            # Return average interest over the period
            result = {}
            for candidate in candidates:
                if candidate in df.columns:
                    result[candidate] = float(df[candidate].mean())

            return result
        except Exception as exc:
            wait = 2 ** (attempt + 1)
            logger.warning("Google Trends attempt %d failed: %s. Retrying in %ds", attempt + 1, exc, wait)
            time.sleep(wait)

    logger.error("Google Trends failed after 3 attempts")
    return None


def fetch_election_interest(
    terms: list[str] | None = None,
    geo: str = "US",
) -> dict[str, float] | None:
    """Fetch Google Trends for election-related terms."""
    if terms is None:
        terms = ["how to vote", "voter registration", "election day", "ballot"]
    return fetch_candidate_trends(terms, geo=geo)
