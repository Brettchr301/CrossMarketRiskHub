"""Tests for the Narrative Bias Detector.

Verifies the detection of systematic directional mispricing across competitive races.
"""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
import pandas as pd

from app.election.signals.narrative_bias import (
    detect_narrative_bias,
    compute_historical_skew_distribution,
    backtest_narrative_signal,
    score_2022_similarity,
    NarrativeBiasSignal,
    RaceMispricing
)


@pytest.fixture
def mock_db():
    """Provides a generic mock SQLAlchemy session."""
    session = MagicMock()
    query_mock = MagicMock()
    session.query.return_value = query_mock
    query_mock.filter.return_value = query_mock
    query_mock.join.return_value = query_mock
    query_mock.all.return_value = []
    return session


@patch('app.election.signals.narrative_bias.normalize_price')
@patch('app.election.signals.narrative_bias.detect_direction')
def test_competitive_race_filtering(mock_direction, mock_normalize, mock_db):
    """1. Create 5 races with probs [0.1, 0.3, 0.5, 0.7, 0.9]. Verify only 3 middle ones pass."""
    mock_direction.return_value = "Democrat"
    mock_normalize.side_effect = [0.1, 0.3, 0.5, 0.7, 0.9]

    # Mock DB returning 5 quotes, then 5 polls
    quotes = [MagicMock(race_id=i, price=p) for i, p in enumerate([0.1, 0.3, 0.5, 0.7, 0.9])]
    polls = [MagicMock(race_id=i, polling_avg=0.5) for i in range(5)]
    
    mock_db.query.return_value.filter.return_value.all.side_effect = [quotes, polls]

    signal = detect_narrative_bias(mock_db, cycle=2024, competitive_range=(0.25, 0.75))

    assert signal.n_competitive_races == 3


@patch('app.election.signals.narrative_bias.normalize_price')
@patch('app.election.signals.narrative_bias.detect_direction')
def test_skew_ratio_all_dem_underpriced(mock_direction, mock_normalize, mock_db):
    """2. 4 competitive races, all with market_prob < polling_avg. Verify skew_ratio = 1.0."""
    mock_direction.return_value = "Democrat"
    mock_normalize.side_effect = [0.4, 0.4, 0.4, 0.4]

    quotes = [MagicMock(race_id=i, price=0.4) for i in range(4)]
    polls = [MagicMock(race_id=i, polling_avg=0.6) for i in range(4)]
    
    mock_db.query.return_value.filter.return_value.all.side_effect = [quotes, polls]

    signal = detect_narrative_bias(mock_db, cycle=2024)

    assert signal.n_competitive_races == 4
    assert signal.n_dem_underpriced == 4
    assert signal.n_rep_underpriced == 0
    assert signal.skew_ratio == 1.0


@patch('app.election.signals.narrative_bias.normalize_price')
@patch('app.election.signals.narrative_bias.detect_direction')
def test_skew_ratio_balanced(mock_direction, mock_normalize, mock_db):
    """3. 4 races, 2 Dem underpriced, 2 Rep underpriced. Verify skew_ratio ≈ 0.5."""
    mock_direction.return_value = "Democrat"
    mock_normalize.side_effect = [0.4, 0.4, 0.6, 0.6]

    quotes = [MagicMock(race_id=i, price=p) for i, p in enumerate([0.4, 0.4, 0.6, 0.6])]
    polls = [MagicMock(race_id=i, polling_avg=p) for i, p in enumerate([0.6, 0.6, 0.4, 0.4])]
    
    mock_db.query.return_value.filter.return_value.all.side_effect = [quotes, polls]

    signal = detect_narrative_bias(mock_db, cycle=2024)

    assert signal.n_competitive_races == 4
    assert signal.n_dem_underpriced == 2
    assert signal.n_rep_underpriced == 2
    assert signal.skew_ratio == 0.5


@patch('app.election.signals.narrative_bias.normalize_price')
@patch('app.election.signals.narrative_bias.detect_direction')
def test_signal_strength_classification(mock_direction, mock_normalize, mock_db):
    """4. Test all 5 thresholds map to correct labels."""
    mock_direction.return_value = "Democrat"

    # Case 1: none (Skew < 0.55)
    quotes_none = [MagicMock(race_id=i, price=0.5) for i in range(10)]
    polls_none = [MagicMock(race_id=i, polling_avg=0.6 if i < 5 else 0.4) for i in range(10)]
    mock_normalize.side_effect = [0.5] * 10
    mock_db.query.return_value.filter.return_value.all.side_effect = [quotes_none, polls_none]
    assert detect_narrative_bias(mock_db, cycle=2024).signal_strength == "none"

    # Case 2: weak (Skew 0.60, Avg Mispricing 5pp)
    quotes_weak = [MagicMock(race_id=i, price=0.45 if i < 6 else 0.55) for i in range(10)]
    polls_weak = [MagicMock(race_id=i, polling_avg=0.50) for i in range(10)]
    mock_normalize.side_effect = [0.45]*6 + [0.55]*4
    mock_db.query.return_value.filter.return_value.all.side_effect = [quotes_weak, polls_weak]
    assert detect_narrative_bias(mock_db, cycle=2024).signal_strength == "weak"

    # Case 3: moderate (Skew 0.70, Avg Mispricing 15pp)
    quotes_mod = [MagicMock(race_id=i, price=0.35 if i < 7 else 0.65) for i in range(10)]
    polls_mod = [MagicMock(race_id=i, polling_avg=0.50) for i in range(10)]
    mock_normalize.side_effect = [0.35]*7 + [0.65]*3
    mock_db.query.return_value.filter.return_value.all.side_effect = [quotes_mod, polls_mod]
    assert detect_narrative_bias(mock_db, cycle=2024).signal_strength == "moderate"

    # Case 4: strong (Skew 0.80, Avg Mispricing 30pp)
    quotes_strong = [MagicMock(race_id=i, price=0.20 if i < 8 else 0.80) for i in range(10)]
    polls_strong = [MagicMock(race_id=i, polling_avg=0.50) for i in range(10)]
    mock_normalize.side_effect = [0.20]*8 + [0.80]*2
    mock_db.query.return_value.filter.return_value.all.side_effect = [quotes_strong, polls_strong]
    assert detect_narrative_bias(mock_db, cycle=2024).signal_strength == "strong"

    # Case 5: extreme (Skew 0.90, Avg Mispricing 45pp)
    quotes_ext = [MagicMock(race_id=i, price=0.05 if i < 9 else 0.95) for i in range(10)]
    polls_ext = [MagicMock(race_id=i, polling_avg=0.50) for i in range(10)]
    mock_normalize.side_effect = [0.05]*9 + [0.95]*1
    mock_db.query.return_value.filter.return_value.all.side_effect = [quotes_ext, polls_ext]
    assert detect_narrative_bias(mock_db, cycle=2024).signal_strength == "extreme"


def test_2022_similarity_score():
    """5. Create a signal matching 2022 pattern. Verify similarity > 0.8."""
    signal = NarrativeBiasSignal(
        as_of=datetime.now(),
        cycle=2022,
        platform="polymarket",
        n_competitive_races=9,
        n_dem_underpriced=9,
        n_rep_underpriced=0,
        skew_ratio=1.0,
        avg_mispricing_pp=59.0,
        max_mispricing_pp=74.0,
        historical_percentile=0.99,
        signal_strength="extreme"
    )
    score = score_2022_similarity(signal)
    assert score > 0.8


def test_2022_similarity_low_for_balanced():
    """6. Create balanced signal (skew=0.5). Verify similarity < 0.3."""
    signal = NarrativeBiasSignal(
        as_of=datetime.now(),
        cycle=2024,
        platform="polymarket",
        n_competitive_races=10,
        n_dem_underpriced=5,
        n_rep_underpriced=5,
        skew_ratio=0.5,
        avg_mispricing_pp=5.0,
        max_mispricing_pp=10.0,
        historical_percentile=0.5,
        signal_strength="none"
    )
    score = score_2022_similarity(signal)
    assert score < 0.3


@patch('app.election.signals.narrative_bias.detect_narrative_bias')
def test_backtest_2022_profitable(mock_detect, mock_db):
    """7. Mock historical data matching 2022 pattern. Verify backtest shows positive PnL and win_rate=1.0."""
    mock_detect.return_value = NarrativeBiasSignal(
        as_of=datetime(2022, 11, 1),
        cycle=2022,
        platform="polymarket",
        n_competitive_races=9,
        n_dem_underpriced=9,
        n_rep_underpriced=0,
        skew_ratio=1.0,
        avg_mispricing_pp=59.0,
        max_mispricing_pp=74.0,
        historical_percentile=0.99,
        signal_strength="extreme"
    )

    # Mock DB to return winning outcomes for Democrats
    mock_outcomes = [MagicMock(race_id=i, winner="Democrat") for i in range(9)]
    mock_db.query.return_value.filter.return_value.all.return_value = mock_outcomes

    result = backtest_narrative_signal(mock_db, cycles=[2022])

    assert result["total_pnl"] > 0
    assert result["win_rate"] == 1.0
    assert len(result["signals_fired"]) >= 1
    assert result["signals_fired"][0]["signal_strength"] == "extreme"


@patch('app.election.signals.narrative_bias.detect_narrative_bias')
def test_historical_skew_distribution(mock_detect, mock_db):
    """8. Verify returns DataFrame with expected columns and one row per cycle×days_before combination."""
    mock_detect.return_value = NarrativeBiasSignal(
        as_of=datetime.now(), cycle=2020, platform="polymarket",
        n_competitive_races=10, n_dem_underpriced=5, n_rep_underpriced=5,
        skew_ratio=0.5, avg_mispricing_pp=5.0, max_mispricing_pp=10.0,
        historical_percentile=0.5, signal_strength="none"
    )

    cycles = [2018, 2020, 2022, 2024]
    days = [1, 7]
    df = compute_historical_skew_distribution(mock_db, cycles=cycles, days_before_election=days)

    assert isinstance(df, pd.DataFrame)
    
    expected_columns = ['cycle', 'days_before', 'skew_ratio', 'n_competitive', 'avg_mispricing_pp']
    for col in expected_columns:
        assert col in df.columns

    # 4 cycles * 2 days = 8 rows
    assert len(df) == 8
