"""Backfill historical alternative data (weather, vote counts, party registration).

Runs alongside the prediction market history backfill.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.election.db.historical_models import RaceOutcome
from app.election.db.timing_models import HourlyWeather, LiveVoteCount, PartyRegistration
from app.election.historical.weather_history import (
    SWING_STATE_COORDS,
    fetch_hourly_weather,
    weather_turnout_score_hourly,
)
from app.election.historical.live_vote_counts import fetch_nyt_results_archive

logger = logging.getLogger(__name__)


def backfill_hourly_weather_for_cycle(db: Session, cycle: int) -> int:
    """Backfill hourly weather for all swing states around the cycle's election date."""
    election_date = None
    if cycle in (2018, 2022, 2026):
        election_date = date(cycle, 11, 8) if cycle == 2022 else date(cycle, 11, 6 if cycle == 2018 else 3)
    elif cycle in (2020, 2024, 2028):
        election_date = date(cycle, 11, 3) if cycle == 2020 else date(cycle, 11, 5 if cycle == 2024 else 7)
    else:
        logger.warning("Unknown cycle %d", cycle)
        return 0

    start = election_date - timedelta(days=3)
    end = election_date + timedelta(days=3)

    n = 0
    for state, (lat, lon) in SWING_STATE_COORDS.items():
        df = fetch_hourly_weather(lat, lon, start, end)
        if df.empty:
            continue

        for ts, row in df.iterrows():
            score = weather_turnout_score_hourly(row)
            db.add(HourlyWeather(
                state=state,
                timestamp=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                temperature=float(row.get("temperature", 0) or 0) if pd.notna(row.get("temperature")) else None,
                precipitation=float(row.get("precipitation", 0) or 0) if pd.notna(row.get("precipitation")) else None,
                wind_speed=float(row.get("wind_speed", 0) or 0) if pd.notna(row.get("wind_speed")) else None,
                cloud_cover=float(row.get("cloud_cover", 0) or 0) if pd.notna(row.get("cloud_cover")) else None,
                snowfall=float(row.get("snowfall", 0) or 0) if pd.notna(row.get("snowfall")) else None,
                turnout_score=score,
            ))
            n += 1
        logger.info("Backfilled weather for %s cycle %d: %d hourly points", state, cycle, len(df))

    db.commit()
    logger.info("Total hourly weather rows for cycle %d: %d", cycle, n)
    return n


def backfill_nyt_vote_counts_2024(db: Session) -> int:
    """Backfill 2024 election final results from NYT archive.

    NYT CSV gives final state-level results. Live progression would need
    Wayback Machine scraping which is slower. This gives us the "settlement"
    anchor for each race.
    """
    n = 0
    # NYT archive URLs often 404 due to redirects; this may need adjustment
    for race_type in ["president", "senate"]:
        try:
            df = fetch_nyt_results_archive(race_type, cycle=2024)
            if df.empty:
                continue

            # Columns vary; try common patterns
            state_col = next((c for c in df.columns if c.lower() in ("state", "state_name", "fips_state")), None)
            if not state_col:
                continue

            # Map race_type+state to race_id
            for _, row in df.iterrows():
                state = str(row[state_col]).strip().upper()
                if len(state) > 2:
                    # Full name to abbrev
                    from app.election.mappings.race_linker import STATE_NAMES
                    state = STATE_NAMES.get(state.lower(), state[:2])

                # Find matching race
                outcome = db.execute(
                    select(RaceOutcome).where(
                        RaceOutcome.cycle == 2024,
                        RaceOutcome.state == state,
                        RaceOutcome.race_type == race_type,
                    )
                ).scalar_one_or_none()

                if not outcome:
                    continue

                # Record a "final" vote count entry
                election_dt = datetime.combine(outcome.election_date, datetime.min.time()) + timedelta(days=1)
                db.add(LiveVoteCount(
                    race_id=outcome.race_id,
                    state=state,
                    cycle=2024,
                    timestamp=election_dt,
                    pct_reporting=100.0,
                    leader_party=outcome.winner_party,
                    leader_margin_pct=0.0,  # unknown without more columns
                    total_votes=None,
                    source="nyt_archive",
                ))
                n += 1
        except Exception as exc:
            logger.warning("NYT backfill failed for %s: %s", race_type, exc)

    db.commit()
    logger.info("NYT vote count backfill: %d rows", n)
    return n


def run_alt_data_backfill(cycles: list[int] = None) -> dict[int, dict[str, int]]:
    """Full alt-data backfill for historical cycles."""
    from app.election.db.session import get_session_factory, init_election_db, _get_engine
    from app.election.db.models import ElectionBase

    if cycles is None:
        cycles = [2018, 2020, 2022, 2024]

    init_election_db()
    ElectionBase.metadata.create_all(_get_engine())

    db = get_session_factory()()
    results = {}
    try:
        for cycle in cycles:
            logger.info("===== Alt-data backfill for cycle %d =====", cycle)
            weather_n = backfill_hourly_weather_for_cycle(db, cycle)
            vote_n = 0
            if cycle == 2024:
                vote_n = backfill_nyt_vote_counts_2024(db)
            results[cycle] = {"weather_rows": weather_n, "vote_count_rows": vote_n}
    finally:
        db.close()
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_alt_data_backfill([2024])
    print(results)
