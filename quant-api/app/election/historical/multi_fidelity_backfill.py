"""Multi-fidelity backfill for prediction market history.

Fetches daily prices far from election, hourly prices in the final 2 weeks,
and minute-level prices during election week. Gives precise timing for
alignment with live vote counts, weather, and party registration.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import requests

from app.election.historical.polymarket_history import (
    BASE_CLOB,
    TIMEOUT,
    fetch_closed_election_markets,
    fetch_price_history,
    get_yes_token,
)
from app.election.historical.kalshi_history import (
    fetch_candlesticks,
    fetch_markets_for_event,
)

logger = logging.getLogger(__name__)


def fetch_polymarket_window(
    token_id: str,
    start_ts: int,
    end_ts: int,
    fidelity: int = 60,
) -> pd.Series:
    """Fetch Polymarket price history for a specific time window at a given fidelity.

    fidelity in minutes: 1=minute, 60=hourly, 1440=daily.
    """
    try:
        r = requests.get(
            BASE_CLOB,
            params={
                "market": token_id,
                "startTs": int(start_ts),
                "endTs": int(end_ts),
                "fidelity": fidelity,
            },
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
        logger.warning("Polymarket window fetch failed: %s", exc)
        return pd.Series(dtype=float)


def polymarket_multi_fidelity_backfill(
    cycle: int,
    election_date: date,
) -> dict[str, pd.Series]:
    """Backfill closed Polymarket markets at escalating fidelity near election day.

    - 6 months out → election: daily
    - 14 days out → election+3d: hourly (overlay)
    - 3 days out → 3 days after: minute (overlay)
    """
    markets = fetch_closed_election_markets(cycle)

    six_months = datetime.combine(election_date - timedelta(days=180), datetime.min.time())
    two_weeks = datetime.combine(election_date - timedelta(days=14), datetime.min.time())
    three_days = datetime.combine(election_date - timedelta(days=3), datetime.min.time())
    three_after = datetime.combine(election_date + timedelta(days=3), datetime.min.time())

    results: dict[str, pd.Series] = {}

    for m in markets:
        token = get_yes_token(m)
        if not token:
            continue
        question = m.get("question", f"market_{m.get('id')}")

        # Layer 1: daily baseline
        daily = fetch_price_history(token, fidelity=1440)
        if daily.empty:
            continue

        # Layer 2: hourly for final 2 weeks
        hourly = fetch_polymarket_window(
            token, int(two_weeks.timestamp()), int(three_after.timestamp()), fidelity=60
        )

        # Layer 3: minute for election week
        minute = fetch_polymarket_window(
            token, int(three_days.timestamp()), int(three_after.timestamp()), fidelity=1
        )

        merged = daily.copy()
        if not hourly.empty:
            merged = merged.combine_first(hourly)
            merged.update(hourly)
        if not minute.empty:
            merged = merged.combine_first(minute)
            merged.update(minute)

        results[question] = merged.sort_index()
        logger.info(
            "Multi-fidelity '%s': %d daily + %d hourly + %d minute = %d merged",
            question[:60], len(daily), len(hourly), len(minute), len(merged),
        )

    return results


def kalshi_multi_fidelity_backfill(
    event_ticker: str,
    election_date: date,
) -> dict[str, pd.Series]:
    """Backfill a Kalshi event at escalating fidelity near election day."""
    markets = fetch_markets_for_event(event_ticker)
    if not markets:
        return {}

    six_months = datetime.combine(election_date - timedelta(days=180), datetime.min.time())
    two_weeks = datetime.combine(election_date - timedelta(days=14), datetime.min.time())
    three_days = datetime.combine(election_date - timedelta(days=3), datetime.min.time())
    three_after = datetime.combine(election_date + timedelta(days=3), datetime.min.time())

    results: dict[str, pd.Series] = {}
    for m in markets:
        ticker = m.get("ticker")
        if not ticker:
            continue

        daily = fetch_candlesticks(
            ticker, int(six_months.timestamp()), int(three_after.timestamp()), period_interval=1440
        )
        if daily.empty:
            continue

        hourly = fetch_candlesticks(
            ticker, int(two_weeks.timestamp()), int(three_after.timestamp()), period_interval=60
        )

        minute = fetch_candlesticks(
            ticker, int(three_days.timestamp()), int(three_after.timestamp()), period_interval=1
        )

        merged = daily.copy()
        if not hourly.empty:
            merged = merged.combine_first(hourly)
            merged.update(hourly)
        if not minute.empty:
            merged = merged.combine_first(minute)
            merged.update(minute)

        title = m.get("title", ticker)
        results[title] = merged.sort_index()
        logger.info(
            "Kalshi multi-fidelity '%s': %d + %d + %d",
            title[:60], len(daily), len(hourly), len(minute),
        )

    return results
