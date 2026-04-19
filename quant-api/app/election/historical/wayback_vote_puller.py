"""Wayback Machine vote count puller.

For each historical election, fetches Wayback snapshots of state SOS result pages,
parses them for vote counts (pct reporting, leader, margin), and stores in
live_vote_counts table with timestamps.
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta
from typing import Any

import requests
from bs4 import BeautifulSoup
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.election.db.historical_models import RaceOutcome
from app.election.db.session import get_session_factory, init_election_db, _get_engine
from app.election.db.models import ElectionBase
from app.election.db.timing_models import LiveVoteCount
from app.election.historical.live_vote_counts import list_sos_snapshots, extract_timestamp

logger = logging.getLogger(__name__)
TIMEOUT = 30

# Election dates per cycle
ELECTION_DATES = {
    2018: date(2018, 11, 6),
    2020: date(2020, 11, 3),
    2022: date(2022, 11, 8),
    2024: date(2024, 11, 5),
}

# Priority states for each cycle (presidential + swing states)
CYCLE_STATES = {
    2018: ["FL", "TX", "AZ", "NV", "MO", "MT", "ND", "WI", "GA"],
    2020: ["PA", "MI", "WI", "AZ", "GA", "NV", "NC", "FL"],
    2022: ["PA", "GA", "AZ", "NV", "NH", "OH", "NC", "WI"],
    2024: ["PA", "MI", "WI", "AZ", "GA", "NV", "NC", "OH"],
}


def parse_snapshot_html(html: str) -> dict[str, Any] | None:
    """Extract pct reporting and leader from raw HTML.

    This is a best-effort heuristic parser - state SOS pages vary wildly.
    Returns None if we can't extract meaningful data.
    """
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True).lower()

    # Try to extract pct reporting
    pct_reporting = None
    pct_patterns = [
        r"(\d{1,3}(?:\.\d+)?)\s*%\s*(?:of\s*)?(?:precincts?|districts?|counties)\s*report",
        r"precincts?\s*report(?:ing|ed)\s*:?\s*(\d{1,3}(?:\.\d+)?)",
        r"(\d{1,3}(?:\.\d+)?)\s*%\s*report",
        r"reporting\s*:?\s*(\d{1,3}(?:\.\d+)?)\s*%",
    ]
    for p in pct_patterns:
        m = re.search(p, text)
        if m:
            try:
                pct_reporting = float(m.group(1))
                if 0 <= pct_reporting <= 100:
                    break
            except ValueError:
                continue

    # Extract leader party based on candidate mentions + vote totals
    # (This is very rough - real state sites have structured tables)
    dem_total = 0
    rep_total = 0
    num_pattern = re.compile(r"[\d,]+")

    # Look for Trump/Harris/Biden mentions with nearby numbers
    candidates = {
        "R": ["trump", "republican"],
        "D": ["biden", "harris", "democrat"],
    }
    for party, names in candidates.items():
        for name in names:
            for m in re.finditer(name, text):
                # Search for a vote-like number within 100 chars
                window = text[max(0, m.start() - 50):m.end() + 100]
                for num_match in num_pattern.finditer(window):
                    num_str = num_match.group(0).replace(",", "")
                    try:
                        n = int(num_str)
                        if 1000 <= n <= 10_000_000:
                            if party == "R":
                                rep_total = max(rep_total, n)
                            else:
                                dem_total = max(dem_total, n)
                            break
                    except ValueError:
                        continue

    if pct_reporting is None and dem_total == 0 and rep_total == 0:
        return None

    leader_party = "D" if dem_total > rep_total else "R"
    total = dem_total + rep_total
    margin = abs(dem_total - rep_total) / total if total > 0 else 0

    return {
        "pct_reporting": pct_reporting or 0.0,
        "leader_party": leader_party,
        "leader_margin_pct": margin,
        "total_votes": total if total > 0 else None,
    }


def fetch_snapshot_html(archive_url: str) -> str | None:
    """Fetch a single Wayback snapshot's HTML."""
    try:
        r = requests.get(archive_url, timeout=TIMEOUT)
        r.raise_for_status()
        return r.text
    except Exception as exc:
        logger.debug("Snapshot fetch failed: %s", exc)
        return None


def pull_state_cycle(db: Session, state: str, cycle: int, max_snapshots: int = 20) -> int:
    """Pull Wayback snapshots of a state SOS page for a cycle and parse them."""
    election_date = ELECTION_DATES.get(cycle)
    if not election_date:
        return 0

    snapshots = list_sos_snapshots(state, election_date, max_snapshots=max_snapshots)
    if not snapshots:
        logger.info("No Wayback snapshots for %s %d", state, cycle)
        return 0

    # Find matching race(s) for state + cycle (may be multiple: senate, governor, pres)
    outcomes = db.execute(
        select(RaceOutcome).where(RaceOutcome.state == state, RaceOutcome.cycle == cycle)
    ).scalars().all()

    if not outcomes:
        logger.info("No race outcomes for %s %d", state, cycle)
        return 0

    # Process snapshots
    n = 0
    for snap in snapshots:
        ts = extract_timestamp(snap)
        if ts is None:
            continue

        html = fetch_snapshot_html(snap["archive_url"])
        if not html:
            continue

        parsed = parse_snapshot_html(html)
        if parsed is None:
            continue

        # Insert one row per matched race (state SOS pages cover all state races)
        for outcome in outcomes:
            db.add(LiveVoteCount(
                race_id=outcome.race_id,
                state=state,
                cycle=cycle,
                timestamp=ts,
                pct_reporting=parsed["pct_reporting"],
                leader_party=parsed["leader_party"],
                leader_margin_pct=parsed["leader_margin_pct"],
                total_votes=parsed["total_votes"],
                source="wayback_sos",
            ))
            n += 1

    db.commit()
    logger.info("Pulled %d vote count rows for %s %d (%d snapshots)", n, state, cycle, len(snapshots))
    return n


def pull_all_cycles(cycles: list[int] | None = None, max_snapshots_per: int = 15) -> dict[int, int]:
    """Run the Wayback pull across all cycles and states."""
    if cycles is None:
        cycles = [2018, 2020, 2022, 2024]

    init_election_db()
    ElectionBase.metadata.create_all(_get_engine())

    db = get_session_factory()()
    results: dict[int, int] = {}
    try:
        for cycle in cycles:
            states = CYCLE_STATES.get(cycle, [])
            total = 0
            for state in states:
                try:
                    n = pull_state_cycle(db, state, cycle, max_snapshots=max_snapshots_per)
                    total += n
                except Exception as exc:
                    logger.warning("Failed %s %d: %s", state, cycle, exc)
            results[cycle] = total
            logger.info("===== Cycle %d total: %d vote count rows =====", cycle, total)
    finally:
        db.close()
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = pull_all_cycles(max_snapshots_per=10)
    print("Results:", results)
