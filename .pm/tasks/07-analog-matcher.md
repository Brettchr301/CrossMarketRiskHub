# Task: Analog Matcher — Map 2026 Races to Historical Analogs

## Classification
| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Security | 0 | Internal analytics |
| Complexity | 2 | Similarity scoring across multiple dimensions |
| Novelty | 1 | Standard similarity matching |
| Blast Radius | 1 | Additive module |
| Existing Code | 1 | Uses Race, RaceOutcome, HistoricalQuote |
| **Total** | **5** | |

## Objective

Map each 2026 race to its closest 2022 analog so we can apply the "red wave miscalibration" pattern. For example, PA Senate 2026 maps to PA Senate 2022 (Fetterman, underpriced by 62pp).

The matcher scores similarity on: state match, race type, partisan lean (Cook PVI), incumbent party, market liquidity, and whether the 2022 analog was one of the 9 mispriced races.

## Deliverables

- [ ] `app/election/signals/analog_matcher.py` — matching engine
- [ ] `tests/test_analog_matcher.py` — tests

## Exact Interface

```python
from dataclasses import dataclass


# Cook PVI approximations (positive = D lean, negative = R lean)
STATE_PVI: dict[str, float] = {
    "PA": 0.0, "MI": +1.0, "WI": 0.0, "AZ": -2.0, "GA": -3.0,
    "NV": -1.0, "NC": -3.0, "MN": +2.0, "NH": 0.0, "ME": +3.0,
    "CO": +4.0, "OR": +5.0, "IA": -6.0, "TX": -8.0,
}

# The 9 races where Polymarket underpriced Dems by 51-74pp in 2022
MISPRICED_2022: dict[str, dict] = {
    "AZ_governor_2022": {"state": "AZ", "type": "governor", "winner": "Hobbs", "party": "D", "market_prob": 0.26, "error_pp": 74},
    "MI_governor_2022": {"state": "MI", "type": "governor", "winner": "Whitmer", "party": "D", "market_prob": 0.28, "error_pp": 72},
    "PA_senate_2022":   {"state": "PA", "type": "senate", "winner": "Fetterman", "party": "D", "market_prob": 0.38, "error_pp": 62},
    "NH_senate_2022":   {"state": "NH", "type": "senate", "winner": "Hassan", "party": "D", "market_prob": 0.38, "error_pp": 62},
    "NV_senate_2022":   {"state": "NV", "type": "senate", "winner": "Cortez Masto", "party": "D", "market_prob": 0.40, "error_pp": 60},
    "US_senate_control": {"state": "US", "type": "senate_control", "winner": "D", "party": "D", "market_prob": 0.41, "error_pp": 59},
    "WI_governor_2022": {"state": "WI", "type": "governor", "winner": "Evers", "party": "D", "market_prob": 0.43, "error_pp": 57},
    "GA_senate_2022":   {"state": "GA", "type": "senate", "winner": "Warnock", "party": "D", "market_prob": 0.48, "error_pp": 52},
    "AZ_senate_2022":   {"state": "AZ", "type": "senate", "winner": "Kelly", "party": "D", "market_prob": 0.49, "error_pp": 51},
}


@dataclass
class AnalogMatch:
    """A 2026 race matched to its historical analog."""
    race_2026_state: str
    race_2026_type: str
    analog_cycle: int
    analog_state: str
    analog_type: str
    analog_winner_party: str
    analog_market_prob_24h: float | None  # market P(Dem) at T-24h
    analog_error_pp: float | None         # how much market underpriced winner
    similarity_score: float               # 0-1 composite similarity
    was_mispriced_2022: bool              # True if analog is one of the 9 mispriced races
    pvi_delta: float                       # PVI difference between 2026 race and analog
    same_state: bool


def match_2026_to_analogs(
    db: Session,
    target_races: list[dict] | None = None,
    analog_cycles: list[int] = [2022, 2020, 2018],
    top_n: int = 3,
) -> dict[str, list[AnalogMatch]]:
    """For each 2026 race, find the top N historical analogs.

    Similarity scoring (weights):
    - Same state: +0.40
    - Same race type: +0.20
    - PVI within ±2: +0.15
    - Same incumbent party defending: +0.10
    - 2022 cycle (most recent midterm): +0.10
    - Was mispriced in 2022: +0.05

    Returns: dict mapping "STATE_TYPE" -> list of AnalogMatch, sorted by similarity desc.
    """


def get_mispricing_forecast(
    matches: dict[str, list[AnalogMatch]],
) -> list[dict]:
    """Given analog matches, forecast expected mispricing for 2026.

    For each 2026 race:
    - Weight analog errors by similarity score
    - Compute expected_error_pp = weighted average of analog errors
    - Flag races with expected_error > 30pp as high-confidence targets

    Returns list of:
    {
        "race": "PA_senate_2026",
        "top_analog": "PA_senate_2022",
        "analog_error_pp": 62,
        "expected_error_pp": 45.2,
        "confidence": "high",
        "recommendation": "BUY Dem at market_prob < 0.50",
    }
    """
```

## Constraints

- DO NOT modify any existing files except creating new ones
- Use existing `RaceOutcome` and `Race` models for historical data
- Handle missing PVI values gracefully (default to 0.0)
- All similarity scores must be in [0.0, 1.0]

## Tests to Write

1. **test_same_state_same_type_highest**: PA Senate 2026 should match PA Senate 2022 as top analog.
2. **test_similarity_score_range**: All scores should be in [0, 1].
3. **test_mispricing_forecast_pa**: PA Senate 2026 forecast should show expected_error > 30pp (based on 2022 analog).
4. **test_all_2026_races_matched**: All 14 Senate + 5 Governor races should have at least 1 analog.
5. **test_mispriced_2022_flag**: Analogs from the 9 mispriced races should have was_mispriced_2022=True.
6. **test_pvi_delta_computed**: Verify PVI difference is correctly calculated.

## Files to Touch
- `app/election/signals/analog_matcher.py` — create
- `tests/test_analog_matcher.py` — create

## Success Criteria
1. All 6 tests pass
2. PA Senate 2026 → PA Senate 2022 is top analog with similarity > 0.8
3. Mispricing forecast flags PA, MI, WI, AZ, GA, NV as high-confidence targets
