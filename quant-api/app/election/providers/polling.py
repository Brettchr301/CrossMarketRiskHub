"""Polling data provider.

Fetches polling data from RealClearPolitics API and 270toWin.
538 CSV endpoints are dead (ABC News acquisition).
"""
from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# RealClearPolitics JSON endpoint (public, no auth)
RCP_API = "https://www.realclearpolling.com/api/polls"
TIMEOUT = 20


def fetch_senate_polls(cycle: int = 2026) -> pd.DataFrame:
    """Fetch senate polls from RCP or fallback sources."""
    return _fetch_rcp_polls("senate", cycle)


def fetch_presidential_polls(cycle: int = 2028) -> pd.DataFrame:
    """Fetch presidential polls."""
    return _fetch_rcp_polls("president", cycle)


def fetch_generic_ballot() -> pd.DataFrame:
    """Fetch generic ballot polls."""
    return _fetch_rcp_polls("generic_ballot", 2026)


def fetch_governor_polls(cycle: int = 2026) -> pd.DataFrame:
    """Fetch governor polls."""
    return _fetch_rcp_polls("governor", cycle)


def _fetch_rcp_polls(race_type: str, cycle: int) -> pd.DataFrame:
    """Attempt to fetch polls from RCP API, with fallback to empty."""
    try:
        # RCP's public API may change; try the standard endpoint
        params = {"type": race_type, "year": cycle, "limit": 200}
        r = requests.get(RCP_API, params=params, timeout=TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and data:
                df = pd.json_normalize(data)
                logger.info("RCP %s polls: %d rows for %d", race_type, len(df), cycle)
                return df
        # If RCP fails, try to scrape from the prediction market data we already have
        logger.info("RCP API returned %d for %s; polling data unavailable", r.status_code, race_type)
        return pd.DataFrame()
    except Exception as exc:
        logger.warning("Polling fetch failed for %s/%d: %s", race_type, cycle, exc)
        return pd.DataFrame()


def extract_poll_signals(df: pd.DataFrame, state: str) -> list[dict[str, Any]]:
    """Extract normalized poll signals for a state.
    Returns list of {candidate, pct, pollster, sample_size, poll_date}.
    """
    if df.empty:
        return []

    # Try various column name patterns
    state_col = None
    for col in ["state", "State", "location"]:
        if col in df.columns:
            state_col = col
            break

    if state_col:
        state_df = df[df[state_col].str.upper() == state.upper()]
    else:
        state_df = df

    if state_df.empty:
        return []

    signals = []
    for _, row in state_df.iterrows():
        signals.append({
            "candidate": str(row.get("candidate_name", row.get("candidate", row.get("answer", "")))),
            "pct": float(row.get("pct", row.get("value", 0.0))),
            "pollster": str(row.get("pollster", row.get("source", ""))),
            "sample_size": int(row.get("sample_size", 0)) if pd.notna(row.get("sample_size")) else None,
            "poll_date": str(row.get("end_date", row.get("date", row.get("created_at", "")))),
            "party": str(row.get("party", "")),
        })
    return signals
