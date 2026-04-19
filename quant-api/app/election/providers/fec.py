"""FEC campaign finance provider.

Fetches candidate financial data from the FEC API.
Free tier with DEMO_KEY.
"""
from __future__ import annotations
import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.open.fec.gov/v1"
API_KEY = "DEMO_KEY"  # Free tier, ~1000 requests/hr
TIMEOUT = 20


def fetch_candidate_financials(
    candidate_id: str,
    cycle: int = 2026,
) -> dict[str, Any] | None:
    """Fetch financial summary for a candidate by FEC ID."""
    try:
        url = f"{BASE_URL}/candidate/{candidate_id}/totals/"
        params = {
            "api_key": API_KEY,
            "cycle": cycle,
            "per_page": 1,
            "sort": "-cycle",
        }
        r = requests.get(url, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return None

        d = results[0]
        return {
            "candidate_id": candidate_id,
            "cycle": cycle,
            "receipts": float(d.get("receipts", 0) or 0),
            "disbursements": float(d.get("disbursements", 0) or 0),
            "cash_on_hand": float(d.get("cash_on_hand_end_period", 0) or 0),
            "individual_contributions": float(d.get("individual_contributions", 0) or 0),
            "pac_contributions": float(d.get("other_political_committee_contributions", 0) or 0),
            "coverage_start": d.get("coverage_start_date"),
            "coverage_end": d.get("coverage_end_date"),
        }
    except Exception as exc:
        logger.error("FEC fetch failed for %s: %s", candidate_id, exc)
        return None


def search_candidates(
    name: str,
    office: str = "S",  # S=Senate, H=House, P=President
    cycle: int = 2026,
) -> list[dict[str, Any]]:
    """Search FEC candidates by name."""
    try:
        params = {
            "api_key": API_KEY,
            "q": name,
            "office": office,
            "cycle": cycle,
            "per_page": 10,
            "sort": "-receipts",
        }
        r = requests.get(f"{BASE_URL}/candidates/search/", params=params, timeout=TIMEOUT)
        r.raise_for_status()
        results = r.json().get("results", [])
        return [
            {
                "fec_id": c.get("candidate_id", ""),
                "name": c.get("name", ""),
                "party": c.get("party", ""),
                "state": c.get("state", ""),
                "office": c.get("office", ""),
                "incumbent": c.get("incumbent_challenge", "") == "I",
            }
            for c in results
        ]
    except Exception as exc:
        logger.error("FEC candidate search failed: %s", exc)
        return []
