"""Tests for direction detector integration into arb detection and backtest engine."""
from __future__ import annotations

from collections.abc import Generator
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.election.arbitrage.cross_market import ArbSignal, _check_pair, detect_cross_market_arbs
from app.election.db.models import ElectionBase
from app.election.db.historical_models import (
    BacktestRun,
    BacktestTrade,
    HistoricalQuote,
    RaceOutcome,
)
from app.election.backtest.engine import (
    BacktestResult,
    backtest_cross_market,
    backtest_outcome_betting,
    build_price_panel,
    _build_direction_map,
)
from app.election.mappings.direction_detector import detect_direction, normalize_price


@pytest.fixture()
def election_db(tmp_path: Path) -> Generator[Session, None, None]:
    db_path = tmp_path / "direction_test.db"
    engine = create_engine(f"sqlite+pysqlite:///{db_path}", future=True)
    ElectionBase.metadata.create_all(bind=engine)
    LocalSession = sessionmaker(bind=engine, autoflush=False, autocommit=False, class_=Session)
    db = LocalSession()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()


class TestDirectionNormalizesArbQuotes:
    def test_opposite_direction_no_arb(self):
        """Two quotes pointing opposite directions should not produce a false arb.

        Platform A: "Will Democrat win Senate?" YES=D, bid=0.65
        Platform B: "Will Republican win Senate?" YES=R, bid=0.70

        Without normalization: sell B@0.70, buy A@0.65 = 5% edge (FALSE)
        With normalization: B normalizes to 0.30 P(Dem), A stays 0.65
        Actually they AGREE Dem has 65-70% chance. No arb.
        """
        seller = {
            "platform": "polymarket",
            "platform_question": "Will Republican win Senate?",
            "yes_bid": 0.70,
            "yes_ask": 0.75,
            "liquidity_score": 0.8,
            "contract_id": 1,
        }
        buyer = {
            "platform": "kalshi",
            "platform_question": "Will Democrat win Senate?",
            "yes_bid": 0.60,
            "yes_ask": 0.65,
            "liquidity_score": 0.8,
            "contract_id": 2,
        }
        signal = _check_pair(1, seller, buyer)
        assert signal is None, "Should not find arb between opposite-direction contracts"


class TestDirectionFindsRealArb:
    def test_same_direction_real_arb(self):
        """Two quotes both pointing YES=D, one at 0.55 ask, other at 0.70 bid.

        Both normalize to P(Dem wins). sell@0.70, buy@0.55 = 15% gross edge.
        After fees, should still be a real arb.
        """
        seller = {
            "platform": "polymarket",
            "platform_question": "Will Democrat win Senate?",
            "yes_bid": 0.70,
            "yes_ask": 0.75,
            "liquidity_score": 0.8,
            "contract_id": 1,
        }
        buyer = {
            "platform": "kalshi",
            "platform_question": "Will Democrat win Senate?",
            "yes_bid": 0.50,
            "yes_ask": 0.55,
            "liquidity_score": 0.8,
            "contract_id": 2,
        }
        signal = _check_pair(1, seller, buyer)
        assert signal is not None, "Should find arb between same-direction contracts"
        assert signal.net_edge_pct > 0


class TestDirectionUnknownSkipped:
    def test_empty_question_skipped(self):
        """Quote with empty question should be skipped (confidence=0.0)."""
        seller = {
            "platform": "polymarket",
            "platform_question": "",
            "yes_bid": 0.70,
            "yes_ask": 0.75,
            "liquidity_score": 0.8,
        }
        buyer = {
            "platform": "kalshi",
            "platform_question": "Will Democrat win Senate?",
            "yes_bid": 0.50,
            "yes_ask": 0.55,
            "liquidity_score": 0.8,
        }
        signal = _check_pair(1, seller, buyer)
        assert signal is None, "Should skip when direction is unknown"


class TestDirectionDetectorCore:
    def test_dem_control_detected(self):
        result = detect_direction("Will Democrats control the Senate?")
        assert result.yes_party == "D"
        assert result.confidence >= 0.9

    def test_rep_win_detected(self):
        result = detect_direction("Will Republican win Arizona governor?")
        assert result.yes_party == "R"
        assert result.confidence >= 0.85

    def test_normalize_dem_passthrough(self):
        assert normalize_price(0.65, "D") == 0.65

    def test_normalize_rep_inverts(self):
        assert normalize_price(0.70, "R") == pytest.approx(0.30)


class TestBacktestDirectionMap:
    def test_direction_map_built(self, election_db: Session):
        """Verify direction map correctly maps (platform, race_id) -> yes_party."""
        election_db.add(HistoricalQuote(
            race_id=1, platform="polymarket",
            platform_market_id="pm1",
            question="Will Democrat win PA Senate?",
            cycle=2022, price=0.55, as_of=datetime(2022, 11, 1),
        ))
        election_db.add(HistoricalQuote(
            race_id=1, platform="kalshi",
            platform_market_id="k1",
            question="Will Republican win PA Senate?",
            cycle=2022, price=0.45, as_of=datetime(2022, 11, 1),
        ))
        election_db.commit()

        dm = _build_direction_map(election_db, 2022)
        assert dm[("polymarket", 1)] == "D"
        assert dm[("kalshi", 1)] == "R"


class TestBacktestNormalization:
    def test_normalized_panel_prices(self, election_db: Session):
        """Verify build_price_panel normalizes prices to P(Dem wins).

        Polymarket: "Will Democrat win?" at 0.55 -> stays 0.55
        Kalshi: "Will Republican win?" at 0.60 -> normalizes to 0.40
        """
        election_db.add(HistoricalQuote(
            race_id=1, platform="polymarket",
            platform_market_id="pm1",
            question="Will Democrat win PA Senate?",
            cycle=2022, price=0.55, as_of=datetime(2022, 11, 1),
        ))
        election_db.add(HistoricalQuote(
            race_id=1, platform="kalshi",
            platform_market_id="k1",
            question="Will Republican win PA Senate?",
            cycle=2022, price=0.60, as_of=datetime(2022, 11, 1),
        ))
        election_db.commit()

        panel = build_price_panel(election_db, 2022)
        assert not panel.empty

        # Both should now be in P(Dem wins) terms
        pm_price = panel[("polymarket", 1)].iloc[0]
        k_price = panel[("kalshi", 1)].iloc[0]

        assert pm_price == pytest.approx(0.55)
        assert k_price == pytest.approx(0.40)  # 1.0 - 0.60 = 0.40


class TestFullCrossPlatformArb:
    def test_full_detect_cross_market_arbs(self):
        """Full arb detection with mixed-direction contracts."""
        quotes_by_race = {
            1: [
                {
                    "platform": "polymarket",
                    "platform_question": "Will Democrat win PA Senate?",
                    "yes_bid": 0.65, "yes_ask": 0.67,
                    "liquidity_score": 0.8, "contract_id": 1,
                },
                {
                    "platform": "kalshi",
                    "platform_question": "Will Republican win PA Senate?",
                    "yes_bid": 0.60, "yes_ask": 0.62,
                    "liquidity_score": 0.8, "contract_id": 2,
                },
            ]
        }

        # Polymarket: YES=D, bid=0.65 -> normalized 0.65
        # Kalshi: YES=R, bid=0.60 -> normalized bid = 1-0.60 = 0.40
        # Kalshi: YES=R, ask=0.62 -> normalized ask = 1-0.62 = 0.38
        # No arb: highest normalized bid (0.65 from PM) vs lowest ask (0.38 from K)
        # Actually sell PM@0.65 buy K_norm@0.38 = 27% edge? Let me think...
        # normalize_price on bid: For R party, bid 0.60 -> 1-0.60 = 0.40
        # normalize_price on ask: For R party, ask 0.62 -> 1-0.62 = 0.38
        # So _check_pair(PM as seller, K as buyer):
        #   sell_bid = normalize(0.65, D) = 0.65
        #   buy_ask = normalize(0.62, R) = 0.38
        #   gross_edge = 0.65 - 0.38 = 0.27 -> that IS an arb
        #
        # But is this a real arb? PM says 65% Dem, K says 60% Rep (=40% Dem)
        # That's a 25pp disagreement -> genuine arb opportunity
        signals = detect_cross_market_arbs(quotes_by_race)
        assert len(signals) > 0
        assert signals[0].net_edge_pct > 0
