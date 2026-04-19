"""Polymarket CLOB minute-level scrape for closing hours of past elections.

For each election cycle with Polymarket markets already in DB, hit the CLOB
prices-history endpoint with fidelity=1 (minute) bounded to the 48-hour
closing window.

Complements the HF trades.parquet extraction by filling in minute-bar data
for markets whose trades may be sparse in the raw tick dump.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Any

import requests

from app.election.db.session import get_session_factory, init_election_db, _get_engine
from app.election.db.models import ElectionBase
from app.election.db.historical_models import HistoricalQuote
from app.election.historical.polymarket_history import (
    BASE_GAMMA,
    TIMEOUT,
    get_yes_token,
)
from app.election.historical.multi_fidelity_backfill import fetch_polymarket_window
from app.election.mappings.race_linker import link_contract_to_race

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# Polymarket launched mid-2020, so 2020+ only.
ELECTION_DATES = {
    2020: datetime(2020, 11, 3),
    2021: datetime(2021, 11, 2),
    2022: datetime(2022, 11, 8),
    2023: datetime(2023, 11, 7),
    2024: datetime(2024, 11, 5),
}

# Poll close ~11pm ET. Window: 24h before to 48h after.
WINDOW_BEFORE_HOURS = 24
WINDOW_AFTER_HOURS = 48


def find_election_markets(cycle: int) -> list[dict[str, Any]]:
    """Find closed Polymarket markets matching this cycle's election patterns."""
    patterns = [
        rf"{cycle}.*president", rf"president.*{cycle}",
        rf"{cycle}.*senate", rf"senate.*{cycle}",
        rf"{cycle}.*house", rf"house.*{cycle}",
        rf"{cycle}.*governor", rf"governor.*{cycle}",
        rf"{cycle}.*election", rf"election.*{cycle}",
        rf"{cycle}.*midterm", rf"midterm.*{cycle}",
    ]
    regex = re.compile("|".join(patterns), re.I)

    markets: list[dict[str, Any]] = []
    seen: set[str] = set()

    for page in range(40):
        params = {"limit": 500, "offset": page * 500, "active": "false", "closed": "true"}
        try:
            r = requests.get(BASE_GAMMA, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            arr = r.json()
        except Exception as exc:
            logger.warning("Page %d failed: %s", page, exc)
            break
        if not arr:
            break
        for m in arr:
            mid = str(m.get("id", ""))
            if mid in seen:
                continue
            q = str(m.get("question", ""))
            if regex.search(q):
                seen.add(mid)
                markets.append(m)
    return markets


def run():
    init_election_db()
    ElectionBase.metadata.create_all(_get_engine())
    db = get_session_factory()()

    total_inserted = 0

    try:
        for cycle, election_dt in ELECTION_DATES.items():
            logger.info("=== Cycle %d ===", cycle)
            win_end = election_dt.replace(hour=23, minute=59)
            win_start = win_end - timedelta(hours=WINDOW_BEFORE_HOURS)
            win_end_after = win_end + timedelta(hours=WINDOW_AFTER_HOURS)

            start_ts = int(win_start.timestamp())
            end_ts = int(win_end_after.timestamp())

            markets = find_election_markets(cycle)
            logger.info("Cycle %d: %d markets found", cycle, len(markets))

            cycle_total = 0
            for m in markets:
                token = get_yes_token(m)
                if not token:
                    continue

                # Minute fidelity for closing window
                series = fetch_polymarket_window(token, start_ts, end_ts, fidelity=1)
                if series.empty:
                    continue

                question = str(m.get("question", ""))
                mid = str(m.get("id", ""))
                link = link_contract_to_race(question)

                for ts, price in series.items():
                    db.add(HistoricalQuote(
                        race_id=link.race_id,
                        platform="polymarket_minute",
                        platform_market_id=f"polym_{mid}",
                        question=question,
                        cycle=cycle,
                        price=float(price),
                        as_of=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                    ))
                    cycle_total += 1
                    total_inserted += 1

                db.commit()

            logger.info("Cycle %d: %d minute-level quotes inserted", cycle, cycle_total)

    finally:
        db.close()

    logger.info("=== Total inserted: %d ===", total_inserted)
    return total_inserted


if __name__ == "__main__":
    n = run()
    print(f"\nDone. {n:,} minute-level closing-hours quotes inserted.")
