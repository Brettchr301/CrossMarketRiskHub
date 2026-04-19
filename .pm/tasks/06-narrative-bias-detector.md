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
