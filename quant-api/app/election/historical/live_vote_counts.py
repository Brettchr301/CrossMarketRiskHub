"""Historical live vote count archive.

For backtesting how prediction markets respond to vote counts during
election night. Pulls from NYT archive CSVs and Wayback Machine snapshots
of state SOS pages.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

TIMEOUT = 30

# NYT 2024 results archive (public, CDN-served)
NYT_2024_PRESIDENT = "https://static01.nyt.com/elections-assets/pages/data/2024-11-05/results-president.csv"
NYT_2024_SENATE = "https://static01.nyt.com/elections-assets/pages/data/2024-11-05/results-senate.csv"
NYT_2024_HOUSE = "https://static01.nyt.com/elections-assets/pages/data/2024-11-05/results-house.csv"

# Wayback CDX for historical state SOS pages
WAYBACK_CDX = "http://web.archive.org/cdx/search/cdx"


def fetch_nyt_results_archive(race_type: str, cycle: int = 2024) -> pd.DataFrame:
    """Fetch NYT election results archive CSV."""
    url_map = {
        2024: {
            "president": NYT_2024_PRESIDENT,
            "senate": NYT_2024_SENATE,
            "house": NYT_2024_HOUSE,
        },
    }
    url = url_map.get(cycle, {}).get(race_type)
    if not url:
        logger.info("No NYT archive for %s %d", race_type, cycle)
        return pd.DataFrame()

    try:
        df = pd.read_csv(url)
        logger.info("NYT %s %d results: %d rows", race_type, cycle, len(df))
        return df
    except Exception as exc:
        logger.warning("NYT results fetch failed: %s", exc)
        return pd.DataFrame()


def list_sos_snapshots(state: str, election_date: date, max_snapshots: int = 50) -> list[dict[str, Any]]:
    """List Wayback Machine snapshots of a state SOS election results page."""
    from app.election.providers.state_sos import STATE_RESULT_URLS

    url = STATE_RESULT_URLS.get(state.upper())
    if not url:
        return []

    start = (election_date - timedelta(days=1)).strftime("%Y%m%d")
    end = (election_date + timedelta(days=2)).strftime("%Y%m%d")

    try:
        params = {
            "url": url,
            "output": "json",
            "from": start,
            "to": end,
            "limit": max_snapshots,
            "filter": "statuscode:200",
        }
        r = requests.get(WAYBACK_CDX, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        rows = r.json()
        if len(rows) < 2:
            return []
        header = rows[0]
        ts_idx = header.index("timestamp")
        return [
            {
                **dict(zip(header, row)),
                "archive_url": f"http://web.archive.org/web/{row[ts_idx]}/{url}",
            }
            for row in rows[1:]
        ]
    except Exception as exc:
        logger.warning("Wayback SOS snapshots failed for %s: %s", state, exc)
        return []


def extract_timestamp(snapshot: dict[str, Any]) -> datetime | None:
    """Parse a Wayback snapshot timestamp into datetime."""
    ts = snapshot.get("timestamp", "")
    if not ts:
        return None
    try:
        return datetime.strptime(ts, "%Y%m%d%H%M%S")
    except ValueError:
        return None
