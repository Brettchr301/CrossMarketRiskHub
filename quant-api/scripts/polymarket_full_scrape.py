"""Full Polymarket closed markets scrape across all election cycles.

Paginates all active=false/closed=true markets, filters with broad election
regex, pulls YES-token CLOB hourly history, links via race_linker, inserts
into HistoricalQuote.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import UTC, datetime
from typing import Any

import requests

from app.election.db.session import get_session_factory, init_election_db, _get_engine
from app.election.db.models import ElectionBase
from app.election.db.historical_models import HistoricalQuote
from app.election.historical.polymarket_history import (
    BASE_GAMMA,
    TIMEOUT,
    fetch_price_history,
    get_yes_token,
)
from app.election.mappings.race_linker import link_contract_to_race

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

BROAD_ELECTION_RE = re.compile(
    r"\belection\b|senate|\bhouse\b|governor|president|midterm|"
    r"primary|caucus|nominat|congress|gubernator|mayor|recall|"
    r"ballot|referendum|legislature|democrat|republican|"
    r"\bvote|electoral|incumbent|\bvp\b|vice.president",
    re.I,
)

YEAR_RE = re.compile(r"\b(20\d{2})\b")
MAX_PAGES = 100
PAGE_SIZE = 500


def extract_cycle(question: str) -> int | None:
    """Best-effort cycle extraction from the question text."""
    matches = YEAR_RE.findall(question)
    if not matches:
        return None
    # Prefer the most recent election-relevant year
    candidates = [int(m) for m in matches if 2016 <= int(m) <= 2030]
    if not candidates:
        return None
    # Heuristic: election years tend to be even; off-years are odd
    return max(candidates)


def scrape_all_closed_markets() -> list[dict[str, Any]]:
    """Paginate through all closed markets; return election-matching subset."""
    all_markets: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for page in range(MAX_PAGES):
        try:
            params = {
                "limit": PAGE_SIZE,
                "offset": page * PAGE_SIZE,
                "active": "false",
                "closed": "true",
            }
            r = requests.get(BASE_GAMMA, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            arr = r.json()
        except Exception as exc:
            logger.warning("Page %d failed: %s", page, exc)
            break

        if not arr:
            logger.info("Page %d empty, stopping", page)
            break

        matched = 0
        for m in arr:
            q = str(m.get("question", ""))
            mid = str(m.get("id", ""))
            if mid in seen_ids:
                continue
            if BROAD_ELECTION_RE.search(q):
                seen_ids.add(mid)
                all_markets.append(m)
                matched += 1

        logger.info("Page %d: %d markets, %d election-matched (total: %d)",
                    page, len(arr), matched, len(all_markets))

        # Polite pacing
        time.sleep(0.25)

    return all_markets


def ingest_markets(markets: list[dict[str, Any]]) -> dict[str, int]:
    """Fetch history for each and insert into DB."""
    init_election_db()
    ElectionBase.metadata.create_all(_get_engine())
    db = get_session_factory()()

    stats = {
        "markets_processed": 0,
        "markets_with_history": 0,
        "quotes_inserted": 0,
        "linked": 0,
        "unlinked": 0,
    }
    cycle_counts: dict[int, int] = {}

    try:
        for i, m in enumerate(markets):
            stats["markets_processed"] += 1
            token = get_yes_token(m)
            if not token:
                continue

            series = fetch_price_history(token, fidelity=60)  # hourly
            if series.empty:
                continue
            stats["markets_with_history"] += 1

            question = str(m.get("question", ""))
            mid = str(m.get("id", ""))
            cycle = extract_cycle(question) or 0
            link = link_contract_to_race(question)

            if link.race_id:
                stats["linked"] += 1
            else:
                stats["unlinked"] += 1

            for ts, price in series.items():
                db.add(HistoricalQuote(
                    race_id=link.race_id,
                    platform="polymarket",
                    platform_market_id=f"poly_full_{mid}",
                    question=question,
                    cycle=cycle,
                    price=float(price),
                    as_of=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                ))
                stats["quotes_inserted"] += 1
                cycle_counts[cycle] = cycle_counts.get(cycle, 0) + 1

            # Commit every 25 markets
            if (i + 1) % 25 == 0:
                db.commit()
                logger.info(
                    "Progress: %d/%d markets, %d quotes inserted",
                    i + 1, len(markets), stats["quotes_inserted"],
                )

        db.commit()
    finally:
        db.close()

    stats["by_cycle"] = cycle_counts
    return stats


if __name__ == "__main__":
    logger.info("=== Polymarket Full Scrape ===")
    markets = scrape_all_closed_markets()
    logger.info("Found %d election-matched closed markets", len(markets))
    stats = ingest_markets(markets)
    logger.info("Final stats: %s", stats)
    print(f"\nDone. Processed {stats['markets_processed']} markets, "
          f"inserted {stats['quotes_inserted']:,} quotes.")
    print(f"Linked: {stats['linked']}, Unlinked: {stats['unlinked']}")
    print(f"By cycle: {stats['by_cycle']}")
