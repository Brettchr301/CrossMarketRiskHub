"""Wikipedia page traffic provider.

Uses the Wikimedia REST API for page view statistics.
Free, no auth, ~100 requests/sec.
"""
from __future__ import annotations
import logging
from datetime import datetime, timedelta
from typing import Any

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
TIMEOUT = 15


def fetch_page_views(
    article: str,
    days_back: int = 30,
    granularity: str = "daily",
) -> list[dict[str, Any]]:
    """Fetch Wikipedia page view counts for an article.

    Args:
        article: Wikipedia article title (e.g., "Donald_Trump")
        days_back: Number of days of history to fetch
        granularity: "daily" or "hourly"
    """
    end = datetime.utcnow()
    start = end - timedelta(days=days_back)
    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")

    url = (
        f"{BASE_URL}/en.wikipedia/all-access/all-agents/"
        f"{article}/{granularity}/{start_str}/{end_str}"
    )

    try:
        headers = {"User-Agent": "ElectionAlpha/1.0 (brett@example.com)"}
        r = requests.get(url, headers=headers, timeout=TIMEOUT)
        r.raise_for_status()
        items = r.json().get("items", [])

        return [
            {
                "date": item.get("timestamp", "")[:8],
                "views": int(item.get("views", 0)),
            }
            for item in items
        ]
    except Exception as exc:
        logger.warning("Wikipedia traffic failed for '%s': %s", article, exc)
        return []


def fetch_candidate_traffic(
    candidates: dict[str, str],
    days_back: int = 30,
) -> dict[str, list[dict[str, Any]]]:
    """Fetch page views for multiple candidates.

    Args:
        candidates: {label: wikipedia_article_title}
            e.g., {"Trump": "Donald_Trump", "DeSantis": "Ron_DeSantis"}
    """
    results = {}
    for label, article in candidates.items():
        views = fetch_page_views(article, days_back=days_back)
        if views:
            results[label] = views
            logger.info("Wikipedia: %s has %d days of data", label, len(views))
    return results


def traffic_ratio(
    candidate_a_views: list[dict],
    candidate_b_views: list[dict],
) -> float:
    """Calculate traffic ratio between two candidates.

    Returns A / (A + B), so 0.5 = equal interest, >0.5 = more interest in A.
    """
    total_a = sum(v.get("views", 0) for v in candidate_a_views)
    total_b = sum(v.get("views", 0) for v in candidate_b_views)

    if total_a + total_b == 0:
        return 0.5

    return total_a / (total_a + total_b)
