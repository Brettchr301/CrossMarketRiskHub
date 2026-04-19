"""Analog Matcher — maps 2026 races to historical analogs.

Uses state, race type, partisan lean (Cook PVI), and 2022 mispricing data
to find the best historical comparison for each 2026 race.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.election.db.historical_models import RaceOutcome

logger = logging.getLogger(__name__)

# Cook PVI approximations (positive = D lean, negative = R lean)
STATE_PVI: dict[str, float] = {
    "PA": 0.0, "MI": +1.0, "WI": 0.0, "AZ": -2.0, "GA": -3.0,
    "NV": -1.0, "NC": -3.0, "MN": +2.0, "NH": 0.0, "ME": +3.0,
    "CO": +4.0, "OR": +5.0, "IA": -6.0, "TX": -8.0,
    "OH": -6.0, "FL": -3.0, "VA": +2.0, "NJ": +3.0, "US": 0.0,
}

# The 9 races where Polymarket underpriced Dems by 51-74pp in 2022
MISPRICED_2022: dict[str, dict[str, Any]] = {
    "AZ_governor_2022": {"state": "AZ", "type": "governor", "winner": "Hobbs", "party": "D", "market_prob": 0.26, "error_pp": 74},
    "MI_governor_2022": {"state": "MI", "type": "governor", "winner": "Whitmer", "party": "D", "market_prob": 0.28, "error_pp": 72},
    "PA_senate_2022":   {"state": "PA", "type": "senate", "winner": "Fetterman", "party": "D", "market_prob": 0.38, "error_pp": 62},
    "NH_senate_2022":   {"state": "NH", "type": "senate", "winner": "Hassan", "party": "D", "market_prob": 0.38, "error_pp": 62},
    "NV_senate_2022":   {"state": "NV", "type": "senate", "winner": "Cortez Masto", "party": "D", "market_prob": 0.40, "error_pp": 60},
    "US_senate_control_2022": {"state": "US", "type": "senate", "winner": "D", "party": "D", "market_prob": 0.41, "error_pp": 59},
    "WI_governor_2022": {"state": "WI", "type": "governor", "winner": "Evers", "party": "D", "market_prob": 0.43, "error_pp": 57},
    "GA_senate_2022":   {"state": "GA", "type": "senate", "winner": "Warnock", "party": "D", "market_prob": 0.48, "error_pp": 52},
    "AZ_senate_2022":   {"state": "AZ", "type": "senate", "winner": "Kelly", "party": "D", "market_prob": 0.49, "error_pp": 51},
}

# 2026 races to match
RACES_2026: list[dict[str, str]] = [
    {"state": "PA", "type": "senate"}, {"state": "MI", "type": "senate"},
    {"state": "WI", "type": "senate"}, {"state": "AZ", "type": "senate"},
    {"state": "GA", "type": "senate"}, {"state": "NV", "type": "senate"},
    {"state": "NC", "type": "senate"}, {"state": "MN", "type": "senate"},
    {"state": "NH", "type": "senate"}, {"state": "ME", "type": "senate"},
    {"state": "CO", "type": "senate"}, {"state": "OR", "type": "senate"},
    {"state": "IA", "type": "senate"}, {"state": "TX", "type": "senate"},
    {"state": "PA", "type": "governor"}, {"state": "MI", "type": "governor"},
    {"state": "WI", "type": "governor"}, {"state": "GA", "type": "governor"},
    {"state": "NV", "type": "governor"},
]


@dataclass
class AnalogMatch:
    """A 2026 race matched to its historical analog."""
    race_2026_state: str
    race_2026_type: str
    analog_cycle: int
    analog_state: str
    analog_type: str
    analog_winner_party: str
    analog_market_prob_24h: float | None
    analog_error_pp: float | None
    similarity_score: float
    was_mispriced_2022: bool
    pvi_delta: float
    same_state: bool


def _get_historical_races(
    db: Session | None,
    cycles: list[int],
) -> list[dict[str, Any]]:
    """Get historical race outcomes from DB or MISPRICED_2022 data."""
    races: list[dict[str, Any]] = []

    # Always include the 2022 mispriced races (our primary signal)
    for key, data in MISPRICED_2022.items():
        races.append({
            "state": data["state"],
            "type": data["type"],
            "cycle": 2022,
            "winner_party": data["party"],
            "market_prob_24h": data["market_prob"],
            "error_pp": data["error_pp"],
            "mispriced_key": key,
        })

    # Add DB outcomes if available
    if db is not None:
        try:
            outcomes = db.execute(
                select(RaceOutcome).where(RaceOutcome.cycle.in_(cycles))
            ).scalars().all()
            seen = {(r["state"], r["type"], r["cycle"]) for r in races}
            for o in outcomes:
                key = (o.state, o.race_type, o.cycle)
                if key not in seen:
                    races.append({
                        "state": o.state,
                        "type": o.race_type,
                        "cycle": o.cycle,
                        "winner_party": o.winner_party,
                        "market_prob_24h": None,
                        "error_pp": None,
                        "mispriced_key": None,
                    })
                    seen.add(key)
        except Exception as exc:
            logger.warning("Failed to query historical outcomes: %s", exc)

    return races


def _compute_similarity(
    target_state: str,
    target_type: str,
    analog: dict[str, Any],
) -> float:
    """Compute similarity score between a 2026 race and a historical analog."""
    score = 0.0

    # Same state: +0.40
    if target_state == analog["state"]:
        score += 0.40

    # Same race type: +0.20
    if target_type == analog["type"]:
        score += 0.20

    # PVI within ±2: +0.15
    target_pvi = STATE_PVI.get(target_state, 0.0)
    analog_pvi = STATE_PVI.get(analog["state"], 0.0)
    pvi_diff = abs(target_pvi - analog_pvi)
    if pvi_diff <= 2.0:
        score += 0.15
    elif pvi_diff <= 5.0:
        score += 0.08

    # 2022 cycle (most recent midterm): +0.10
    if analog["cycle"] == 2022:
        score += 0.10

    # Was mispriced in 2022: +0.05
    if analog.get("mispriced_key"):
        score += 0.05

    # Same incumbent party defending: +0.10
    # (simplified: D incumbents in 2026 defending seats won in 2020)
    if analog["winner_party"] == "D":
        score += 0.10

    return min(1.0, score)


def match_2026_to_analogs(
    db: Session | None = None,
    target_races: list[dict[str, str]] | None = None,
    analog_cycles: list[int] | None = None,
    top_n: int = 3,
) -> dict[str, list[AnalogMatch]]:
    """For each 2026 race, find the top N historical analogs.

    Returns dict mapping "STATE_TYPE" -> list of AnalogMatch sorted by similarity.
    """
    if target_races is None:
        target_races = RACES_2026
    if analog_cycles is None:
        analog_cycles = [2022, 2020, 2018]

    historical = _get_historical_races(db, analog_cycles)
    results: dict[str, list[AnalogMatch]] = {}

    for target in target_races:
        t_state = target["state"]
        t_type = target["type"]
        key = f"{t_state}_{t_type}"

        matches: list[AnalogMatch] = []
        for analog in historical:
            sim = _compute_similarity(t_state, t_type, analog)
            if sim < 0.1:
                continue

            pvi_delta = STATE_PVI.get(t_state, 0.0) - STATE_PVI.get(analog["state"], 0.0)

            matches.append(AnalogMatch(
                race_2026_state=t_state,
                race_2026_type=t_type,
                analog_cycle=analog["cycle"],
                analog_state=analog["state"],
                analog_type=analog["type"],
                analog_winner_party=analog["winner_party"],
                analog_market_prob_24h=analog.get("market_prob_24h"),
                analog_error_pp=analog.get("error_pp"),
                similarity_score=round(sim, 3),
                was_mispriced_2022=analog.get("mispriced_key") is not None,
                pvi_delta=pvi_delta,
                same_state=(t_state == analog["state"]),
            ))

        matches.sort(key=lambda m: m.similarity_score, reverse=True)
        results[key] = matches[:top_n]

    return results


def get_mispricing_forecast(
    matches: dict[str, list[AnalogMatch]],
) -> list[dict[str, Any]]:
    """Given analog matches, forecast expected mispricing for 2026.

    Weights analog errors by similarity score to compute expected error.
    """
    forecasts: list[dict[str, Any]] = []

    for race_key, analogs in matches.items():
        if not analogs:
            continue

        # Weight analog errors by similarity
        weighted_error = 0.0
        total_weight = 0.0
        top_analog = analogs[0]

        for a in analogs:
            if a.analog_error_pp is not None:
                weighted_error += a.analog_error_pp * a.similarity_score
                total_weight += a.similarity_score

        expected_error = weighted_error / total_weight if total_weight > 0 else 0.0

        # Classify confidence
        if expected_error > 40:
            confidence = "high"
        elif expected_error > 20:
            confidence = "medium"
        elif expected_error > 10:
            confidence = "low"
        else:
            confidence = "none"

        recommendation = ""
        if confidence in ("high", "medium"):
            recommendation = f"BUY Dem at market_prob < 0.50 (analog error: {expected_error:.0f}pp)"

        state, rtype = race_key.split("_", 1)
        forecasts.append({
            "race": f"{state}_{rtype}_2026",
            "state": state,
            "race_type": rtype,
            "top_analog": f"{top_analog.analog_state}_{top_analog.analog_type}_{top_analog.analog_cycle}",
            "top_analog_error_pp": top_analog.analog_error_pp,
            "expected_error_pp": round(expected_error, 1),
            "confidence": confidence,
            "recommendation": recommendation,
            "was_mispriced_2022": top_analog.was_mispriced_2022,
            "similarity": top_analog.similarity_score,
        })

    forecasts.sort(key=lambda f: f["expected_error_pp"], reverse=True)
    return forecasts
