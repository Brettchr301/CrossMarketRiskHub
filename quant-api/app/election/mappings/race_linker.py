"""Fuzzy matcher for mapping platform contracts to canonical races.

Given a contract question like "Will the Democratic Party control the Senate
after the 2026 Midterm election?", finds the matching Race in the registry.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from app.election.mappings.race_registry_historical import ALL_RACES_HISTORICAL, RaceSpec

logger = logging.getLogger(__name__)

STATE_NAMES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY",
}

RACE_TYPE_KEYWORDS = {
    "presidential": ["president", "presidential", "white house", "oval office"],
    "senate": ["senate", "senator", "u.s. senate", "us senate"],
    "senate_control": ["control.*senate", "senate.*majority", "senate.*control", "senate.*chamber"],
    "house": ["house of representatives", "u.s. house", "us house", "congressional"],
    "house_control": ["control.*house", "house.*majority", "house.*control", "house.*chamber"],
    "governor": ["governor", "gubernatorial"],
}


@dataclass
class LinkResult:
    race_id: int | None
    score: float
    matched_state: str | None
    matched_race_type: str | None
    matched_cycle: int | None


def extract_state(question: str) -> str | None:
    """Extract state from question text (full name or 2-letter abbrev)."""
    q_lower = question.lower()
    # Try full state names first (longer matches win)
    for full, abbrev in sorted(STATE_NAMES.items(), key=lambda x: -len(x[0])):
        if full in q_lower:
            return abbrev
    # Try 2-letter abbreviations (bounded)
    for abbrev in STATE_NAMES.values():
        pattern = r"\b" + abbrev + r"\b"
        if re.search(pattern, question):
            return abbrev
    return None


def extract_cycle(question: str) -> int | None:
    """Extract election cycle year from question text."""
    # Look for 4-digit years from 2016-2030
    for year in [2028, 2026, 2024, 2022, 2020, 2018, 2016, 2030]:
        if str(year) in question:
            return year
    # "midterm" with no year => probably the upcoming one
    if "midterm" in question.lower():
        return 2026
    return None


def extract_race_type(question: str) -> str | None:
    """Extract race type from question text.

    Returns one of: presidential, senate, senate_control, house, house_control, governor
    """
    q_lower = question.lower()

    # Control markets take precedence (more specific)
    for rt in ["senate_control", "house_control"]:
        for kw in RACE_TYPE_KEYWORDS[rt]:
            if re.search(kw, q_lower):
                return rt

    for rt in ["presidential", "governor", "senate", "house"]:
        for kw in RACE_TYPE_KEYWORDS[rt]:
            if kw in q_lower:
                return rt

    return None


def score_match(spec: RaceSpec, state: str | None, race_type: str | None, cycle: int | None) -> float:
    """Score how well a race spec matches the extracted features."""
    score = 0.0

    # Cycle match is critical
    if cycle is not None and spec.cycle == cycle:
        score += 3.0
    elif cycle is not None and spec.cycle != cycle:
        return 0.0  # wrong cycle = no match

    # Race type match
    if race_type is not None:
        if spec.race_type == race_type:
            score += 3.0
        # Presidential/senate_control/house_control are unique, so mismatch = 0
        elif race_type in ("presidential", "senate_control", "house_control"):
            return 0.0
        # Senate != senate_control (user may have asked either)
        elif race_type == "senate" and spec.race_type == "senate_control":
            score += 1.0
        elif race_type == "house" and spec.race_type == "house_control":
            score += 1.0

    # State match
    if state is not None:
        if spec.state == state:
            score += 2.0
        elif spec.state != "US":  # state-specific mismatch
            return 0.0

    return score


def link_contract_to_race(question: str, min_score: float = 5.0) -> LinkResult:
    """Find the canonical race that best matches a contract question."""
    state = extract_state(question)
    race_type = extract_race_type(question)
    cycle = extract_cycle(question)

    best_score = 0.0
    best_spec_idx = -1

    for idx, spec in enumerate(ALL_RACES_HISTORICAL):
        s = score_match(spec, state, race_type, cycle)
        if s > best_score:
            best_score = s
            best_spec_idx = idx

    if best_score < min_score or best_spec_idx < 0:
        return LinkResult(None, best_score, state, race_type, cycle)

    return LinkResult(
        race_id=best_spec_idx + 1,  # 1-indexed to match DB
        score=best_score,
        matched_state=state,
        matched_race_type=race_type,
        matched_cycle=cycle,
    )


def bulk_link_contracts(questions: list[str], min_score: float = 5.0) -> dict[int, list[int]]:
    """Link a batch of contract questions to race_ids.

    Returns {race_id: [contract_indices]} for all matches above threshold.
    """
    links: dict[int, list[int]] = {}
    for i, q in enumerate(questions):
        result = link_contract_to_race(q, min_score)
        if result.race_id is not None:
            links.setdefault(result.race_id, []).append(i)
    return links
