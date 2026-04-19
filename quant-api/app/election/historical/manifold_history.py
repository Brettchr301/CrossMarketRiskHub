"""Manifold Markets historical provider.

Manifold has a public API with FULL bet-level history (every trade timestamped).
This gives us tick-by-tick precision for election markets, better than
Polymarket's CLOB candlesticks.

API: https://api.manifold.markets/v0/
No auth required for read. Rate limit: generous (10 req/s soft).
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

import pandas as pd
import requests
from sqlalchemy import func, select
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

BASE = "https://api.manifold.markets/v0"
TIMEOUT = 30


def search_election_markets(
    term: str = "2024 election",
    limit: int = 100,
    sort: str = "most-popular",
) -> list[dict[str, Any]]:
    """Search Manifold markets by term. Returns list of market dicts."""
    try:
        r = requests.get(
            f"{BASE}/search-markets",
            params={"term": term, "limit": limit, "sort": sort},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.warning("Manifold search failed for '%s': %s", term, exc)
        return []


def fetch_bet_history(contract_id: str, max_bets: int = 10000) -> pd.DataFrame:
    """Fetch all bet history for a market, paginated.

    Returns DataFrame: ts, prob_before, prob_after, amount, outcome.
    """
    all_bets: list[dict] = []
    before: str | None = None

    while len(all_bets) < max_bets:
        params: dict[str, Any] = {"contractId": contract_id, "limit": 1000}
        if before:
            params["before"] = before
        try:
            r = requests.get(f"{BASE}/bets", params=params, timeout=TIMEOUT)
            if r.status_code != 200:
                break
            batch = r.json()
            if not batch:
                break
            all_bets.extend(batch)
            if len(batch) < 1000:
                break
            before = batch[-1].get("id")
            time.sleep(0.15)  # polite
        except Exception as exc:
            logger.warning("Manifold bet fetch failed: %s", exc)
            break

    if not all_bets:
        return pd.DataFrame()

    df = pd.DataFrame([
        {
            "ts": pd.to_datetime(b.get("createdTime", 0), unit="ms"),
            "prob_before": b.get("probBefore"),
            "prob_after": b.get("probAfter"),
            "amount": b.get("amount"),
            "outcome": b.get("outcome"),
            "shares": b.get("shares"),
            "user_id": b.get("userId"),
        }
        for b in all_bets
    ]).sort_values("ts").reset_index(drop=True)
    return df


def bet_history_to_price_series(df: pd.DataFrame) -> pd.Series:
    """Convert bet history to a time-indexed price series."""
    if df.empty:
        return pd.Series(dtype=float)
    s = pd.Series(df["prob_after"].values, index=df["ts"]).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.astype(float).clip(0.0, 1.0)


def backfill_election_markets(
    search_terms: list[str] | None = None,
    markets_per_term: int = 20,
    db: Session | None = None,
) -> dict[str, pd.Series]:
    """Backfill Manifold election market bet histories.

    If a DB session is provided, skips markets whose platform_market_id
    already exists in the historical_quotes table (cross-run dedup).
    """
    if search_terms is None:
        search_terms = [
            "2024 presidential",
            "2024 senate",
            "2024 house",
            "2022 midterm",
            "2022 senate",
            "2020 presidential",
            "2018 midterm",
            "2026 senate",
            "2028 presidential",
        ]

    # Build set of already-ingested market IDs from DB
    existing_ids: set[str] = set()
    if db is not None:
        try:
            from app.election.db.historical_models import HistoricalQuote
            existing_ids = set(
                db.execute(
                    select(HistoricalQuote.platform_market_id)
                    .where(HistoricalQuote.platform == "manifold")
                    .distinct()
                ).scalars().all()
            )
            logger.info("Manifold dedup: %d markets already in DB", len(existing_ids))
        except Exception as exc:
            logger.warning("Failed to query existing Manifold markets: %s", exc)

    seen: set[str] = set()
    results: dict[str, pd.Series] = {}
    skipped = 0

    for term in search_terms:
        markets = search_election_markets(term, limit=markets_per_term, sort="most-popular")
        for m in markets:
            mid = m.get("id")
            question = m.get("question", "")
            if not mid or mid in seen:
                continue
            seen.add(mid)

            # Skip if already ingested in DB
            if mid in existing_ids:
                logger.info("Skipping '%s', already ingested", question[:60])
                skipped += 1
                continue

            df = fetch_bet_history(mid, max_bets=5000)
            series = bet_history_to_price_series(df)
            if series.empty:
                continue

            results[question] = series
            logger.info(
                "Manifold '%s': %d bets spanning %s to %s",
                question[:60],
                len(series),
                series.index.min(),
                series.index.max(),
            )

    if skipped:
        logger.info("Manifold backfill: skipped %d already-ingested markets", skipped)
    return results


def deduplicate_manifold_quotes(db: Session) -> int:
    """Remove duplicate Manifold quotes from historical_quotes table.

    Duplicates defined as: same platform_market_id + same as_of timestamp
    where platform = 'manifold'. Keeps the row with the lowest id.

    Returns count of deleted rows.
    """
    from app.election.db.historical_models import HistoricalQuote

    # Find duplicate groups
    subq = (
        db.execute(
            select(
                HistoricalQuote.platform_market_id,
                HistoricalQuote.as_of,
                func.min(HistoricalQuote.id).label("keep_id"),
                func.count(HistoricalQuote.id).label("cnt"),
            )
            .where(HistoricalQuote.platform == "manifold")
            .group_by(HistoricalQuote.platform_market_id, HistoricalQuote.as_of)
            .having(func.count(HistoricalQuote.id) > 1)
        ).all()
    )

    if not subq:
        logger.info("No Manifold duplicates found")
        return 0

    deleted = 0
    for row in subq:
        market_id, as_of, keep_id, cnt = row
        # Delete all but the lowest-id row
        dupes = db.execute(
            select(HistoricalQuote)
            .where(
                HistoricalQuote.platform == "manifold",
                HistoricalQuote.platform_market_id == market_id,
                HistoricalQuote.as_of == as_of,
                HistoricalQuote.id != keep_id,
            )
        ).scalars().all()
        for dupe in dupes:
            db.delete(dupe)
            deleted += 1

    db.commit()
    logger.info("Manifold dedup: deleted %d duplicate rows", deleted)
    return deleted
