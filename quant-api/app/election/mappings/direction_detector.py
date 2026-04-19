"""Direction detector for prediction market contracts.

Determines which party (D/R) the YES side of a contract represents,
so cross-platform comparisons use same-direction quotes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Direction = Literal["D", "R", "I", "unknown"]


@dataclass
class DirectionResult:
    yes_party: Direction  # which party wins = YES settles at 1.0
    confidence: float  # 0-1
    reason: str


# Known candidate names mapped to party
CANDIDATES_D = {
    # 2024
    "harris", "biden", "walz", "kaine", "slotkin", "baldwin", "gallego", "rosen",
    "casey", "brown", "tester", "manchin", "stein", "whitmer", "evers",
    # 2022
    "fetterman", "warnock", "kelly", "cortez masto", "hassan", "ryan", "barnes",
    "beasley", "demings", "shapiro", "hobbs", "hochul",
    # 2020
    "ossoff", "hickenlooper", "peters", "warner", "coons", "markey", "booker",
    # 2018
    "sinema", "rosen", "manchin", "tester", "donnelly", "heitkamp", "mccaskill",
    "whitmer", "evers", "pritzker", "stacey abrams", "gillum",
    # Generic
    "democrat", "democratic", "democrats",
}

CANDIDATES_R = {
    # 2024
    "trump", "vance", "mccormick", "moreno", "sheehy", "justice", "hovde",
    "lake", "brown", "cameron", "bailey", "hogan", "banks",
    # 2022
    "vance", "walker", "masters", "laxalt", "bolduc", "vance", "oz", "johnson",
    "budd", "rubio", "desantis", "kemp", "abbott", "youngkin",
    # 2020
    "perdue", "loeffler", "mcsally", "tillis", "ernst", "collins", "daines",
    "cornyn", "mcconnell",
    # 2018
    "scott", "cruz", "mcsally", "hawley", "braun", "cramer", "desantis", "kemp",
    # Generic
    "republican", "republicans", "gop",
}


def detect_direction(question: str) -> DirectionResult:
    """Determine which party's victory = YES settlement."""
    if not question:
        return DirectionResult("unknown", 0.0, "empty_question")

    q = question.lower()

    # Explicit "control" questions
    if re.search(r"democrat.*control|dems.*control|democratic.*majority", q):
        return DirectionResult("D", 0.95, "explicit_dem_control")
    if re.search(r"republican.*control|gop.*control|republican.*majority", q):
        return DirectionResult("R", 0.95, "explicit_rep_control")

    # Party-specific win questions
    if re.search(r"\b(democrat|democratic|democrats)\b.*\bwin\b", q) or re.search(r"\bwin\b.*\b(democrat|democratic)\b", q):
        return DirectionResult("D", 0.9, "party_dem_win")
    if re.search(r"\b(republican|republicans|gop)\b.*\bwin\b", q) or re.search(r"\bwin\b.*\b(republican|gop)\b", q):
        return DirectionResult("R", 0.9, "party_rep_win")

    # "Which party wins..." — neutral, treat as D by convention for normalization
    if re.search(r"which party", q):
        return DirectionResult("D", 0.5, "which_party_neutral")

    # Candidate-specific: scan for known candidate surnames
    dem_hits = [c for c in CANDIDATES_D if c in q]
    rep_hits = [c for c in CANDIDATES_R if c in q]

    if dem_hits and not rep_hits:
        return DirectionResult("D", 0.85, f"candidate_dem:{dem_hits[0]}")
    if rep_hits and not dem_hits:
        return DirectionResult("R", 0.85, f"candidate_rep:{rep_hits[0]}")
    if dem_hits and rep_hits:
        # Head-to-head: question is usually framed "Will X beat Y?" — the first candidate = YES
        # This is imperfect but better than nothing
        # Prefer whichever name comes first in the question
        first_dem_pos = min(q.find(c) for c in dem_hits)
        first_rep_pos = min(q.find(c) for c in rep_hits)
        if first_dem_pos < first_rep_pos:
            return DirectionResult("D", 0.7, f"first_candidate_dem:{dem_hits[0]}")
        else:
            return DirectionResult("R", 0.7, f"first_candidate_rep:{rep_hits[0]}")

    # Incumbent defaults: if "re-elect" or "hold seat"
    if re.search(r"re.?elect|hold.*seat|keep.*seat", q):
        return DirectionResult("D", 0.4, "reelect_unknown_party")

    return DirectionResult("unknown", 0.0, "no_signal")


def normalize_price(price: float, yes_party: Direction) -> float:
    """Normalize a price to represent 'probability Democrat wins'.

    For comparison across platforms, we standardize on P(Dem wins).
    If a market's YES = Rep, we invert it.
    """
    if yes_party == "D":
        return price
    if yes_party == "R":
        return 1.0 - price
    # Unknown - return as-is; caller should filter these
    return price
