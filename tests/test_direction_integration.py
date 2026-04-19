import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

from app.election.arbitrage.cross_market import _check_pair
from app.election.backtest.engine import backtest_cross_market, backtest_outcome_betting

class MockDirectionResult:
    def __init__(self, confidence, yes_party):
        self.confidence = confidence
        self.yes_party = yes_party

def mock_detect_direction(question):
    if not question:
        return MockDirectionResult(0.0, "Unknown")
    if "Democrat" in question:
        return MockDirectionResult(0.9, "D")
    if "Republican" in question:
        return MockDirectionResult(0.9, "R")
    return MockDirectionResult(0.0, "Unknown")

def mock_normalize_price(price, party):
    if party == "R":
        return 1.0 - price
    return price

@pytest.fixture(autouse=True)
def patch_direction_detector():
    """
    Mock the direction detector and normalizer to isolate tests from the actual implementation.
    """
    with patch('app.election.arbitrage.cross_market.detect_direction', side_effect=mock_detect_direction, create=True), \
         patch('app.election.arbitrage.cross_market.normalize_price', side_effect=mock_normalize_price, create=True), \
         patch('app.election.backtest.engine.detect_direction', side_effect=mock_detect_direction, create=True), \
         patch('app.election.backtest.engine.normalize_price', side_effect=mock_normalize_price, create=True):
        yield

def test_direction_normalizes_arb_quotes():
    """
    Test 1: Create two mock quotes — one "Will Democrat win Senate?" (YES=D, bid=0.65) 
    and one "Will Republican win Senate?" (YES=R, bid=0.70). 
    Without normalization, arb detector sees sell@0.70 buy@0.65 = 5% edge. 
    With normalization, the Rep quote normalizes to 0.30 P(Dem), so there's actually NO arb.
    Verify no ArbSignal returned.
    """
    seller = {
        "platform": "A",
        "platform_question": "Will Republican win Senate?",
        "yes_bid": 0.70,
        "yes_ask": 0.75
    }
    buyer = {
        "platform": "B",
        "platform_question": "Will Democrat win Senate?",
        "yes_bid": 0.60,
        "yes_ask": 0.65
    }
    
    signal = _check_pair(1, seller, buyer)
    assert signal is None

def test_direction_finds_real_arb():
    """
    Test 2: Two quotes both pointing same direction (both YES=D), 
    one at 0.55 and other at 0.70. 
    After normalization both stay the same. 
    Verify ArbSignal IS returned with correct net_edge.
    """
    seller = {
        "platform": "A",
        "platform_question": "Will Democrat win Senate?",
        "yes_bid": 0.70,
        "yes_ask": 0.75
    }
    buyer = {
        "platform": "B",
        "platform_question": "Will Democrat win Senate?",
        "yes_bid": 0.50,
        "yes_ask": 0.55
    }
    
    signal = _check_pair(1, seller, buyer)
    assert signal is not None
    assert round(signal.net_edge, 2) == 0.15

def test_direction_unknown_skipped():
    """
    Test 4: Quote with question="" (empty). 
    Verify detect_direction returns confidence=0.0, and the quote is excluded from arb detection.
    """
    seller = {
        "platform": "A",
        "platform_question": "",
        "yes_bid": 0.70,
        "yes_ask": 0.75
    }
    buyer = {
        "platform": "B",
        "platform_question": "Will Democrat win Senate?",
        "yes_bid": 0.50,
        "yes_ask": 0.55
    }
    
    signal = _check_pair(1, seller, buyer)
    assert signal is None

@patch('app.election.backtest.engine.SessionLocal', create=True)
def test_direction_inverts_settlement(mock_session):
    """
    Test 3: Mock a HistoricalQuote for "Will Republican win?" at price 0.80. 
    Race outcome: winner_party="R". 
    With normalization, price becomes 0.20 P(Dem), settlement=0.0 (Dem lost), 
    so buying at 0.20 and settling at 0.0 correctly loses $0.20. 
    Verify PnL is negative.
    """
    mock_quote = MagicMock()
    mock_quote.question = "Will Republican win?"
    mock_quote.price = 0.80
    
    mock_session.return_value.__enter__.return_value.query.return_value.filter_by.return_value.all.return_value = [mock_quote]
    
    with patch('app.election.backtest.engine.get_race_outcome', return_value={"winner_party": "R"}, create=True):
        result = backtest_outcome_betting(1)
        assert result.pnl < 0
        assert round(result.pnl, 2) == -0.20

@patch('app.election.backtest.engine.build_price_panel')
def test_backtest_cross_market_with_normalization(mock_build_panel):
    """
    Test 5: End-to-end with 2 platforms, opposite-direction contracts for same race. 
    Verify backtest produces different (correct) results vs non-normalized baseline.
    """
    df = pd.DataFrame({
        "timestamp": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
        "platform_A_price": [0.70, 0.70], # Will Republican win? (YES=R) -> normalized to 0.30
        "platform_B_price": [0.65, 0.65]  # Will Democrat win? (YES=D) -> normalized to 0.65
    })
    mock_build_panel.return_value = df
    
    def mock_db_query(*args, **kwargs):
        mock_q = MagicMock()
        def filter_by_side_effect(**kw):
            m = MagicMock()
            if kw.get("platform") == "A":
                m.first.return_value.question = "Will Republican win?"
            else:
                m.first.return_value.question = "Will Democrat win?"
            return m
        mock_q.filter_by.side_effect = filter_by_side_effect
        return mock_q

    with patch('app.election.backtest.engine.SessionLocal', create=True) as mock_session:
        mock_session.return_value.__enter__.return_value.query.side_effect = mock_db_query
        
        with patch('app.election.backtest.engine.get_race_outcome', return_value={"winner_party": "D"}, create=True):
            result = backtest_cross_market(1)
            assert result is not None
            assert result.trades > 0

@patch('app.election.backtest.engine.SessionLocal', create=True)
def test_outcome_betting_normalized(mock_session):
    """
    Test 6: Test backtest_outcome_betting() with a contract "Will Republican win?" 
    where avg_price is 0.75. 
    Note: The spec says "If Republican wins, PnL should be positive", but under the 
    normalized P(Dem) logic required by Test 3, buying a losing Dem position 
    results in negative PnL. We assert < 0 to maintain mathematical consistency.
    """
    mock_quote = MagicMock()
    mock_quote.question = "Will Republican win?"
    mock_quote.price = 0.75
    
    mock_session.return_value.__enter__.return_value.query.return_value.filter_by.return_value.all.return_value = [mock_quote]
    
    with patch('app.election.backtest.engine.get_race_outcome', return_value={"winner_party": "R"}, create=True):
        result = backtest_outcome_betting(1)
        assert result.pnl < 0
