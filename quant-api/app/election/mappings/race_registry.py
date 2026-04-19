"""Race and candidate registry for 2026 midterms + 2028 presidential cycle.

Provides the canonical list of races and candidates to track.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(slots=True)
class RaceSpec:
    race_type: str  # presidential, senate, house, governor
    state: str  # 2-letter code; "US" for presidential
    district: str | None  # only for house races
    cycle: int
    election_date: date
    candidates: list[CandidateSpec] = field(default_factory=list)


@dataclass(slots=True)
class CandidateSpec:
    name: str
    party: str  # D, R, I, L
    incumbent: bool = False
    fec_candidate_id: str | None = None
    # Wikipedia article for traffic tracking
    wikipedia_article: str | None = None


# ── 2026 Midterm Senate Races (competitive) ──────────────────────────────────

SENATE_2026: list[RaceSpec] = [
    RaceSpec("senate", "PA", None, 2026, date(2026, 11, 3)),
    RaceSpec("senate", "MI", None, 2026, date(2026, 11, 3)),
    RaceSpec("senate", "WI", None, 2026, date(2026, 11, 3)),
    RaceSpec("senate", "AZ", None, 2026, date(2026, 11, 3)),
    RaceSpec("senate", "GA", None, 2026, date(2026, 11, 3)),
    RaceSpec("senate", "NV", None, 2026, date(2026, 11, 3)),
    RaceSpec("senate", "NC", None, 2026, date(2026, 11, 3)),
    RaceSpec("senate", "MN", None, 2026, date(2026, 11, 3)),
    RaceSpec("senate", "NH", None, 2026, date(2026, 11, 3)),
    RaceSpec("senate", "ME", None, 2026, date(2026, 11, 3)),
    RaceSpec("senate", "CO", None, 2026, date(2026, 11, 3)),
    RaceSpec("senate", "OR", None, 2026, date(2026, 11, 3)),
    RaceSpec("senate", "IA", None, 2026, date(2026, 11, 3)),
    RaceSpec("senate", "TX", None, 2026, date(2026, 11, 3)),
]

# ── 2026 Aggregate Control Markets ───────────────────────────────────────────

CONTROL_2026: list[RaceSpec] = [
    RaceSpec("senate_control", "US", None, 2026, date(2026, 11, 3)),
    RaceSpec("house_control", "US", None, 2026, date(2026, 11, 3)),
]

# ── 2026 Governor Races (competitive) ────────────────────────────────────────

GOVERNOR_2026: list[RaceSpec] = [
    RaceSpec("governor", "PA", None, 2026, date(2026, 11, 3)),
    RaceSpec("governor", "MI", None, 2026, date(2026, 11, 3)),
    RaceSpec("governor", "WI", None, 2026, date(2026, 11, 3)),
    RaceSpec("governor", "GA", None, 2026, date(2026, 11, 3)),
    RaceSpec("governor", "NV", None, 2026, date(2026, 11, 3)),
]

# ── 2028 Presidential ────────────────────────────────────────────────────────

PRESIDENTIAL_2028: list[RaceSpec] = [
    RaceSpec("presidential", "US", None, 2028, date(2028, 11, 7)),
]

# ── Combined Registry ────────────────────────────────────────────────────────

ALL_RACES: list[RaceSpec] = SENATE_2026 + CONTROL_2026 + GOVERNOR_2026 + PRESIDENTIAL_2028
