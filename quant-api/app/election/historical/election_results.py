"""Election results ground truth provider.

Sources:
- MIT Election Data + Science Lab (GitHub CSVs)
- Wikipedia election pages (via infobox parsing)
- Static data from race_registry_historical.py
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# MIT Election Lab GitHub data
MIT_SENATE_URL = "https://raw.githubusercontent.com/MEDSL/returns-by-congress/master/senate.csv"
MIT_HOUSE_URL = "https://raw.githubusercontent.com/MEDSL/returns-by-congress/master/house.csv"
MIT_PRESIDENT_URL = "https://raw.githubusercontent.com/MEDSL/returns-by-congress/master/president.csv"
TIMEOUT = 30


def fetch_mit_senate_results() -> pd.DataFrame:
    """Fetch historical senate election results from MIT Election Lab."""
    try:
        df = pd.read_csv(MIT_SENATE_URL)
        logger.info("MIT senate results: %d rows", len(df))
        return df
    except Exception as exc:
        logger.warning("MIT senate fetch failed: %s", exc)
        return pd.DataFrame()


def fetch_mit_house_results() -> pd.DataFrame:
    """Fetch historical house election results from MIT Election Lab."""
    try:
        df = pd.read_csv(MIT_HOUSE_URL)
        logger.info("MIT house results: %d rows", len(df))
        return df
    except Exception as exc:
        logger.warning("MIT house fetch failed: %s", exc)
        return pd.DataFrame()


def get_race_outcome(race_type: str, state: str, cycle: int) -> dict[str, Any] | None:
    """Get the canonical outcome for a race from static registry.

    Returns {winner_party, winner_name, cycle} or None if not found.
    """
    from app.election.mappings.race_registry_historical import ALL_RACES_HISTORICAL

    for spec in ALL_RACES_HISTORICAL:
        if spec.race_type == race_type and spec.state == state and spec.cycle == cycle:
            return {
                "winner_party": spec.winner,
                "winner_name": spec.winner_name,
                "cycle": spec.cycle,
                "election_date": str(spec.election_date),
            }
    return None


def bulk_outcomes_for_cycle(cycle: int) -> list[dict[str, Any]]:
    """Return all known race outcomes for a cycle."""
    from app.election.mappings.race_registry_historical import ALL_RACES_HISTORICAL

    results = []
    for spec in ALL_RACES_HISTORICAL:
        if spec.cycle == cycle and spec.winner is not None:
            results.append({
                "race_type": spec.race_type,
                "state": spec.state,
                "cycle": spec.cycle,
                "winner_party": spec.winner,
                "winner_name": spec.winner_name,
            })
    return results
