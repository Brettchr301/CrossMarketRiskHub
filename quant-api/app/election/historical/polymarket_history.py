"""Historical Polymarket data provider.

Fetches past election market prices from Polymarket CLOB API.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_GAMMA = "https://gamma-api.polymarket.com/markets"
BASE_CLOB = "https://clob.polymarket.com/prices-history"
TIMEOUT = 30


def fetch_closed_election_markets(cycle: int, max_pages: int = 40) -> list[dict[str, Any]]:
    """Find all closed Polymarket election markets for a given cycle."""
    patterns = [
        f"{cycle}.*president",
        f"{cycle}.*senate",
        f"{cycle}.*house",
        f"{cycle}.*governor",
        f"{cycle}.*election",
        f"{cycle}.*midterm",
    ]
    regex = re.compile("|".join(patterns), re.I)

    all_markets = []
    seen_ids = set()

    for page in range(max_pages):
        try:
            params = {"limit": 500, "offset": page * 500, "active": "false", "closed": "true"}
            r = requests.get(BASE_GAMMA, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            arr = r.json()
            if not arr:
                break
            for m in arr:
                q = str(m.get("question", ""))
                mid = str(m.get("id", ""))
                if regex.search(q) and mid not in seen_ids:
                    seen_ids.add(mid)
                    all_markets.append(m)
        except Exception as exc:
            logger.warning("Page %d failed: %s", page, exc)
            break

    logger.info("Found %d closed Polymarket markets for cycle %d", len(all_markets), cycle)
    return all_markets


def fetch_price_history(token_id: str, fidelity: int = 1440) -> pd.Series:
    """Fetch full price history for a CLOB token.

    fidelity=1440 = daily bars (1440 minutes).
    Returns a pandas Series indexed by timestamp.
    """
    try:
        r = requests.get(
            BASE_CLOB,
            params={"market": token_id, "interval": "max", "fidelity": fidelity},
            timeout=TIMEOUT,
        )
        if r.status_code != 200:
            return pd.Series(dtype=float)
        history = r.json().get("history", [])
        if not history:
            return pd.Series(dtype=float)

        dates = [pd.to_datetime(int(x["t"]), unit="s", utc=True).tz_convert(None) for x in history]
        values = [float(x["p"]) for x in history]
        s = pd.Series(values, index=dates).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        return s.clip(0.0, 1.0)
    except Exception as exc:
        logger.warning("History fetch failed for token %s: %s", token_id[:16], exc)
        return pd.Series(dtype=float)


def get_yes_token(market: dict[str, Any]) -> str | None:
    """Extract the YES token ID from a market dict."""
    raw_tokens = market.get("clobTokenIds")
    raw_outcomes = market.get("outcomes")
    if raw_tokens is None or raw_outcomes is None:
        return None
    try:
        token_ids = json.loads(raw_tokens) if isinstance(raw_tokens, str) else list(raw_tokens)
        outcomes = json.loads(raw_outcomes) if isinstance(raw_outcomes, str) else list(raw_outcomes)
    except Exception:
        return None
    for i, outcome in enumerate(outcomes):
        if str(outcome).strip().lower() == "yes" and i < len(token_ids):
            return str(token_ids[i])
    return str(token_ids[0]) if token_ids else None


def backfill_cycle(cycle: int) -> dict[str, pd.Series]:
    """Backfill all closed markets for a cycle.
    Returns {question: price_history_series}.
    """
    markets = fetch_closed_election_markets(cycle)
    results = {}
    for m in markets:
        token = get_yes_token(m)
        if not token:
            continue
        series = fetch_price_history(token)
        if not series.empty:
            question = m.get("question", f"market_{m.get('id')}")
            results[question] = series
            logger.info("Backfilled '%s': %d data points", question[:60], len(series))
    return results
