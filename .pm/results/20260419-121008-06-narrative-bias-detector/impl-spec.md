# Task: Narrative Bias Detector

## Classification
| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Security | 0 | Internal analytics, no user input |
| Complexity | 3 | Population-level statistics across races, rolling windows, signal generation |
| Novelty | 2 | Novel detection pattern not in existing codebase |
| Blast Radius | 1 | Additive — new module, doesn't change existing code |
| Existing Code | 1 | Builds on existing HistoricalQuote, RaceOutcome, BlendedProbability |
| **Total** | **7** | |

## Objective

Build a module that detects when prediction markets are systematically underpricing one party across multiple competitive races — the exact pattern that produced 51-74pp alpha in 2022.

The 2022 signal was: at T-24h before election, Polymarket had Democrats below 50% on 9 competitive races they all won. This wasn't individual race noise — it was a correlated, narrative-driven bias ("red wave") that infected the entire market.

The detector needs to:
1. Identify "competitive" races (market prob between 0.25-0.75)
2. Measure directional skew: are most competitive races tilted the same way?
3. Compare market consensus to polling/fundamentals for systematic divergence
4. Fire a signal when skew exceeds historical norms

## Deliverables

- [ ] `app/election/signals/narrative_bias.py` — core detection engine
- [ ] `app/election/signals/__init__.py` — package init
- [ ] `tests/test_narrative_bias.py` — comprehensive tests

## Exact Interface

```python
"""Narrative bias detector for prediction markets.

Detects systematic directional mispricing across multiple competitive races.
The 2022 "red wave" signal: Polymarket underpriced Dems by 51-74pp on 9/9 races.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session


@dataclass
class RaceMispricing:
    """Individual race mispricing assessment."""
    race_id: int
    state: str
    race_type: str
    cycle: int
    platform: str
    market_prob_dem: float      # Market-implied P(Dem wins), normalized
    polling_avg_dem: float | None  # Polling average for Dem candidate (0-1)
    fundamental_lean: float | None  # PVI or fundamentals-based lean (-1 R to +1 D)
    market_vs_polls: float | None   # market_prob - polling_avg (negative = market underprices Dem)
    confidence: float           # 0-1, based on data quality


@dataclass
class NarrativeBiasSignal:
    """Population-level bias signal across races."""
    as_of: datetime
    cycle: int
    platform: str
    n_competitive_races: int    # races with prob in [0.25, 0.75]
    n_dem_underpriced: int      # competitive races where market < polls for Dem
    n_rep_underpriced: int      # competitive races where market < polls for Rep
    skew_ratio: float           # n_dem_underpriced / n_competitive (0-1, >0.6 = Dem underpriced)
    avg_mispricing_pp: float    # average market-vs-polls gap in percentage points
    max_mispricing_pp: float    # worst single race gap
    historical_percentile: float  # where this skew falls vs 2018-2024 distribution
    signal_strength: str        # "none", "weak", "moderate", "strong"
    races: list[RaceMispricing] = field(default_factory=list)
    analog_2022_similarity: float = 0.0  # 0-1 how similar to 2022 red wave pattern


def detect_narrative_bias(
    db: Session,
    cycle: int,
    platform: str = "polymarket",
    as_of: datetime | None = None,
    competitive_range: tuple[float, float] = (0.25, 0.75),
) -> NarrativeBiasSignal:
    """Detect systematic directional bias across competitive races.

    Steps:
    1. Get latest market quotes for all races in cycle on platform
    2. Normalize to P(Dem wins) using direction_detector
    3. Filter to competitive races (prob in competitive_range)
    4. Get polling averages for same races
    5. Compare market vs polls: positive = market overprices Dem, negative = underprices
    6. Count directional skew across races
    7. Compare to historical distribution of skew ratios
    8. Classify signal strength
    """


def compute_historical_skew_distribution(
    db: Session,
    cycles: list[int] = [2018, 2020, 2022, 2024],
    platform: str = "polymarket",
    days_before_election: list[int] = [1, 3, 7, 14, 30],
) -> pd.DataFrame:
    """Compute historical skew ratios at various time horizons.

    Returns DataFrame: cycle, days_before, skew_ratio, n_competitive, avg_mispricing_pp
    Used to calibrate what's "normal" vs "extreme" skew.
    """


def backtest_narrative_signal(
    db: Session,
    cycles: list[int] = [2018, 2020, 2022, 2024],
    platform: str = "polymarket",
    entry_days_before: int = 7,
    signal_threshold: str = "moderate",
) -> dict[str, Any]:
    """Backtest: if we'd bought the underpriced party when signal fired, what was PnL?

    Returns:
    {
        "cycles_tested": [2018, 2020, 2022, 2024],
        "signals_fired": [
            {"cycle": 2022, "signal_strength": "strong", "skew_ratio": 0.89,
             "avg_mispricing_pp": 62.3, "entry_day": -7},
        ],
        "trades": [
            {"cycle": 2022, "race": "PA Senate", "entry_price": 0.38,
             "settlement": 1.0, "pnl": 0.62, "won": True},
            ...
        ],
        "total_pnl": 5.42,
        "win_rate": 1.0,
        "avg_edge_pp": 58.7,
        "sharpe": 4.2,
    }
    """


def score_2022_similarity(
    current: NarrativeBiasSignal,
) -> float:
    """Score how similar the current signal is to the 2022 red wave pattern.

    Key features of 2022:
    - 9/9 competitive races underpriced Dems
    - Average underpricing was 59pp
    - Skew ratio was ~1.0 (all races same direction)
    - Happened in a midterm with strong media narrative

    Returns 0-1 similarity score.
    """
```

## Signal Strength Classification

| Skew Ratio | Avg Mispricing | Classification |
|-----------|---------------|----------------|
| < 0.55 | any | none |
| 0.55-0.65 | < 10pp | weak |
| 0.65-0.75 | 10-25pp | moderate |
| 0.75-0.85 | 25-40pp | strong |
| > 0.85 | > 40pp | extreme (2022 was here) |

## Constraints

- DO NOT modify any existing files
- Use existing `detect_direction()` + `normalize_price()` for price normalization
- Use existing `PollingData` table for polling comparison
- All functions must work with the existing `election_arb.db` schema
- Must handle missing polling data gracefully (return None for market_vs_polls)
- Historical skew computation must work across all 4 cycles even with sparse data

## Tests to Write

1. **test_competitive_race_filtering**: Create 5 races with probs [0.1, 0.3, 0.5, 0.7, 0.9]. Verify only 3 middle ones pass competitive filter.

2. **test_skew_ratio_all_dem_underpriced**: 4 competitive races, all with market_prob < polling_avg. Verify skew_ratio = 1.0.

3. **test_skew_ratio_balanced**: 4 races, 2 Dem underpriced, 2 Rep underpriced. Verify skew_ratio ≈ 0.5.

4. **test_signal_strength_classification**: Test all 5 thresholds map to correct labels.

5. **test_2022_similarity_score**: Create a signal matching 2022 pattern (skew=0.9, avg_mispricing=60pp). Verify similarity > 0.8.

6. **test_2022_similarity_low_for_balanced**: Create balanced signal (skew=0.5). Verify similarity < 0.3.

7. **test_backtest_2022_profitable**: Mock historical data matching 2022 pattern. Verify backtest shows positive PnL and win_rate=1.0.

8. **test_historical_skew_distribution**: Verify returns DataFrame with expected columns and one row per cycle×days_before combination.

## Files to Touch
- `app/election/signals/__init__.py` — create
- `app/election/signals/narrative_bias.py` — create
- `tests/test_narrative_bias.py` — create

## Success Criteria
1. All 8 tests pass
2. `detect_narrative_bias(db, 2022, "polymarket")` returns signal_strength="strong" or "extreme"
3. `backtest_narrative_signal(db)` shows 2022 was profitable, other cycles had no/weak signals
4. `score_2022_similarity()` returns > 0.8 for 2022 data, < 0.3 for 2020/2024



## Pre-Generated Tests (your code MUST pass these)
These tests were generated independently. Include them EXACTLY as shown.

```json
{"content": "```python\n--- FILE: tests/test_narrative_bias.py ---\n\"\"\"Tests for the Narrative Bias Detector.\n\nVerifies the detection of systematic directional mispricing across competitive races.\n\"\"\"\nimport pytest\nfrom unittest.mock import MagicMock, patch\nfrom datetime import datetime\nimport pandas as pd\n\nfrom app.election.signals.narrative_bias import (\n    detect_narrative_bias,\n    compute_historical_skew_distribution,\n    backtest_narrative_signal,\n    score_2022_similarity,\n    NarrativeBiasSignal,\n    RaceMispricing\n)\n\n\n@pytest.fixture\ndef mock_db():\n    \"\"\"Provides a generic mock SQLAlchemy session.\"\"\"\n    session = MagicMock()\n    query_mock = MagicMock()\n    session.query.return_value = query_mock\n    query_mock.filter.return_value = query_mock\n    query_mock.join.return_value = query_mock\n    query_mock.all.return_value = []\n    return session\n\n\n@patch('app.election.signals.narrative_bias.normalize_price')\n@patch('app.election.signals.narrative_bias.detect_direction')\ndef test_competitive_race_filtering(mock_direction, mock_normalize, mock_db):\n    \"\"\"1. Create 5 races with probs [0.1, 0.3, 0.5, 0.7, 0.9]. Verify only 3 middle ones pass.\"\"\"\n    mock_direction.return_value = \"Democrat\"\n    mock_normalize.side_effect = [0.1, 0.3, 0.5, 0.7, 0.9]\n\n    # Mock DB returning 5 quotes, then 5 polls\n    quotes = [MagicMock(race_id=i, price=p) for i, p in enumerate([0.1, 0.3, 0.5, 0.7, 0.9])]\n    polls = [MagicMock(race_id=i, polling_avg=0.5) for i in range(5)]\n    \n    mock_db.query.return_value.filter.return_value.all.side_effect = [quotes, polls]\n\n    signal = detect_narrative_bias(mock_db, cycle=2024, competitive_range=(0.25, 0.75))\n\n    assert signal.n_competitive_races == 3\n\n\n@patch('app.election.signals.narrative_bias.normalize_price')\n@patch('app.election.signals.narrative_bias.detect_direction')\ndef test_skew_ratio_all_dem_underpriced(mock_direction, mock_normalize, mock_db):\n    \"\"\"2. 4 competitive races, all with market_prob < polling_avg. Verify skew_ratio = 1.0.\"\"\"\n    mock_direction.return_value = \"Democrat\"\n    mock_normalize.side_effect = [0.4, 0.4, 0.4, 0.4]\n\n    quotes = [MagicMock(race_id=i, price=0.4) for i in range(4)]\n    polls = [MagicMock(race_id=i, polling_avg=0.6) for i in range(4)]\n    \n    mock_db.query.return_value.filter.return_value.all.side_effect = [quotes, polls]\n\n    signal = detect_narrative_bias(mock_db, cycle=2024)\n\n    assert signal.n_competitive_races == 4\n    assert signal.n_dem_underpriced == 4\n    assert signal.n_rep_underpriced == 0\n    assert signal.skew_ratio == 1.0\n\n\n@patch('app.election.signals.narrative_bias.normalize_price')\n@patch('app.election.signals.narrative_bias.detect_direction')\ndef test_skew_ratio_balanced(mock_direction, mock_normalize, mock_db):\n    \"\"\"3. 4 races, 2 Dem underpriced, 2 Rep underpriced. Verify skew_ratio \u2248 0.5.\"\"\"\n    mock_direction.return_value = \"Democrat\"\n    mock_normalize.side_effect = [0.4, 0.4, 0.6, 0.6]\n\n    quotes = [MagicMock(race_id=i, price=p) for i, p in enumerate([0.4, 0.4, 0.6, 0.6])]\n    polls = [MagicMock(race_id=i, polling_avg=p) for i, p in enumerate([0.6, 0.6, 0.4, 0.4])]\n    \n    mock_db.query.return_value.filter.return_value.all.side_effect = [quotes, polls]\n\n    signal = detect_narrative_bias(mock_db, cycle=2024)\n\n    assert signal.n_competitive_races == 4\n    assert signal.n_dem_underpriced == 2\n    assert signal.n_rep_underpriced == 2\n    assert signal.skew_ratio == 0.5\n\n\n@patch('app.election.signals.narrative_bias.normalize_price')\n@patch('app.election.signals.narrative_bias.detect_direction')\ndef test_signal_strength_classification(mock_direction, mock_normalize, mock_db):\n    \"\"\"4. Test all 5 thresholds map to correct labels.\"\"\"\n    mock_direction.return_value = \"Democrat\"\n\n    # Case 1: none (Skew < 0.55)\n    quotes_none = [MagicMock(race_id=i, price=0.5) for i in range(10)]\n    polls_none = [MagicMock(race_id=i, polling_avg=0.6 if i < 5 else 0.4) for i in range(10)]\n    mock_normalize.side_effect = [0.5] * 10\n    mock_db.query.return_value.filter.return_value.all.side_effect = [quotes_none, polls_none]\n    assert detect_narrative_bias(mock_db, cycle=2024).signal_strength == \"none\"\n\n    # Case 2: weak (Skew 0.60, Avg Mispricing 5pp)\n    quotes_weak = [MagicMock(race_id=i, price=0.45 if i < 6 else 0.55) for i in range(10)]\n    polls_weak = [MagicMock(race_id=i, polling_avg=0.50) for i in range(10)]\n    mock_normalize.side_effect = [0.45]*6 + [0.55]*4\n    mock_db.query.return_value.filter.return_value.all.side_effect = [quotes_weak, polls_weak]\n    assert detect_narrative_bias(mock_db, cycle=2024).signal_strength == \"weak\"\n\n    # Case 3: moderate (Skew 0.70, Avg Mispricing 15pp)\n    quotes_mod = [MagicMock(race_id=i, price=0.35 if i < 7 else 0.65) for i in range(10)]\n    polls_mod = [MagicMock(race_id=i, polling_avg=0.50) for i in range(10)]\n    mock_normalize.side_effect = [0.35]*7 + [0.65]*3\n    mock_db.query.return_value.filter.return_value.all.side_effect = [quotes_mod, polls_mod]\n    assert detect_narrative_bias(mock_db, cycle=2024).signal_strength == \"moderate\"\n\n    # Case 4: strong (Skew 0.80, Avg Mispricing 30pp)\n    quotes_strong = [MagicMock(race_id=i, price=0.20 if i < 8 else 0.80) for i in range(10)]\n    polls_strong = [MagicMock(race_id=i, polling_avg=0.50) for i in range(10)]\n    mock_normalize.side_effect = [0.20]*8 + [0.80]*2\n    mock_db.query.return_value.filter.return_value.all.side_effect = [quotes_strong, polls_strong]\n    assert detect_narrative_bias(mock_db, cycle=2024).signal_strength == \"strong\"\n\n    # Case 5: extreme (Skew 0.90, Avg Mispricing 45pp)\n    quotes_ext = [MagicMock(race_id=i, price=0.05 if i < 9 else 0.95) for i in range(10)]\n    polls_ext = [MagicMock(race_id=i, polling_avg=0.50) for i in range(10)]\n    mock_normalize.side_effect = [0.05]*9 + [0.95]*1\n    mock_db.query.return_value.filter.return_value.all.side_effect = [quotes_ext, polls_ext]\n    assert detect_narrative_bias(mock_db, cycle=2024).signal_strength == \"extreme\"\n\n\ndef test_2022_similarity_score():\n    \"\"\"5. Create a signal matching 2022 pattern. Verify similarity > 0.8.\"\"\"\n    signal = NarrativeBiasSignal(\n        as_of=datetime.now(),\n        cycle=2022,\n        platform=\"polymarket\",\n        n_competitive_races=9,\n        n_dem_underpriced=9,\n        n_rep_underpriced=0,\n        skew_ratio=1.0,\n        avg_mispricing_pp=59.0,\n        max_mispricing_pp=74.0,\n        historical_percentile=0.99,\n        signal_strength=\"extreme\"\n    )\n    score = score_2022_similarity(signal)\n    assert score > 0.8\n\n\ndef test_2022_similarity_low_for_balanced():\n    \"\"\"6. Create balanced signal (skew=0.5). Verify similarity < 0.3.\"\"\"\n    signal = NarrativeBiasSignal(\n        as_of=datetime.now(),\n        cycle=2024,\n        platform=\"polymarket\",\n        n_competitive_races=10,\n        n_dem_underpriced=5,\n        n_rep_underpriced=5,\n        skew_ratio=0.5,\n        avg_mispricing_pp=5.0,\n        max_mispricing_pp=10.0,\n        historical_percentile=0.5,\n        signal_strength=\"none\"\n    )\n    score = score_2022_similarity(signal)\n    assert score < 0.3\n\n\n@patch('app.election.signals.narrative_bias.detect_narrative_bias')\ndef test_backtest_2022_profitable(mock_detect, mock_db):\n    \"\"\"7. Mock historical data matching 2022 pattern. Verify backtest shows positive PnL and win_rate=1.0.\"\"\"\n    mock_detect.return_value = NarrativeBiasSignal(\n        as_of=datetime(2022, 11, 1),\n        cycle=2022,\n        platform=\"polymarket\",\n        n_competitive_races=9,\n        n_dem_underpriced=9,\n        n_rep_underpriced=0,\n        skew_ratio=1.0,\n        avg_mispricing_pp=59.0,\n        max_mispricing_pp=74.0,\n        historical_percentile=0.99,\n        signal_strength=\"extreme\"\n    )\n\n    # Mock DB to return winning outcomes for Democrats\n    mock_outcomes = [MagicMock(race_id=i, winner=\"Democrat\") for i in range(9)]\n    mock_db.query.return_value.filter.return_value.all.return_value = mock_outcomes\n\n    result = backtest_narrative_signal(mock_db, cycles=[2022])\n\n    assert result[\"total_pnl\"] > 0\n    assert result[\"win_rate\"] == 1.0\n    assert len(result[\"signals_fired\"]) >= 1\n    assert result[\"signals_fired\"][0][\"signal_strength\"] == \"extreme\"\n\n\n@patch('app.election.signals.narrative_bias.detect_narrative_bias')\ndef test_historical_skew_distribution(mock_detect, mock_db):\n    \"\"\"8. Verify returns DataFrame with expected columns and one row per cycle\u00d7days_before combination.\"\"\"\n    mock_detect.return_value = NarrativeBiasSignal(\n        as_of=datetime.now(), cycle=2020, platform=\"polymarket\",\n        n_competitive_races=10, n_dem_underpriced=5, n_rep_underpriced=5,\n        skew_ratio=0.5, avg_mispricing_pp=5.0, max_mispricing_pp=10.0,\n        historical_percentile=0.5, signal_strength=\"none\"\n    )\n\n    cycles = [2018, 2020, 2022, 2024]\n    days = [1, 7]\n    df = compute_historical_skew_distribution(mock_db, cycles=cycles, days_before_election=days)\n\n    assert isinstance(df, pd.DataFrame)\n    \n    expected_columns = ['cycle', 'days_before', 'skew_ratio', 'n_competitive', 'avg_mispricing_pp']\n    for col in expected_columns:\n        assert col in df.columns\n\n    # 4 cycles * 2 days = 8 rows\n    assert len(df) == 8\n--- END FILE ---\n```", "model": "gemini-3.1-pro-preview", "usage": {"prompt_tokens": 2706, "completion_tokens": 3286, "total_tokens": 5992}, "cost_usd": 0.044844}
```
