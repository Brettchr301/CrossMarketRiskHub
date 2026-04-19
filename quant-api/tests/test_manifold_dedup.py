"""Tests for Manifold market deduplication."""
from __future__ import annotations

from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.election.db.models import ElectionBase
from app.election.db.historical_models import HistoricalQuote
from app.election.historical.manifold_history import (
    backfill_election_markets,
    deduplicate_manifold_quotes,
)


@pytest.fixture()
def election_db(tmp_path: Path) -> Generator[Session, None, None]:
    db_path = tmp_path / "election_test.db"
    engine = create_engine(f"sqlite+pysqlite:///{db_path}", future=True)
    ElectionBase.metadata.create_all(bind=engine)
    LocalSession = sessionmaker(bind=engine, autoflush=False, autocommit=False, class_=Session)
    db = LocalSession()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()


def _make_quote(market_id: str, as_of: datetime, price: float = 0.5) -> HistoricalQuote:
    return HistoricalQuote(
        platform="manifold",
        platform_market_id=market_id,
        question=f"Test market {market_id}",
        cycle=2024,
        price=price,
        as_of=as_of,
    )


class TestSkipExistingMarket:
    def test_skip_existing_market(self, election_db: Session):
        """Markets already in DB should be skipped during backfill."""
        # Insert an existing market
        election_db.add(_make_quote("abc123", datetime(2024, 1, 1)))
        election_db.commit()

        # Mock API to return a market with same ID
        mock_markets = [{"id": "abc123", "question": "Already ingested market"}]
        with patch("app.election.historical.manifold_history.search_election_markets", return_value=mock_markets):
            with patch("app.election.historical.manifold_history.fetch_bet_history") as mock_fetch:
                results = backfill_election_markets(
                    search_terms=["test"],
                    markets_per_term=10,
                    db=election_db,
                )
                # fetch_bet_history should NOT be called — market was skipped
                mock_fetch.assert_not_called()
                assert len(results) == 0


class TestNewMarketIngested:
    def test_new_market_ingested(self, election_db: Session):
        """New markets not in DB should be fetched and returned."""
        import pandas as pd

        mock_markets = [{"id": "new123", "question": "New test market"}]
        mock_bets = pd.DataFrame({
            "ts": pd.date_range("2024-01-01", periods=5, freq="h"),
            "prob_before": [0.4, 0.45, 0.5, 0.55, 0.6],
            "prob_after": [0.45, 0.5, 0.55, 0.6, 0.65],
            "amount": [10, 20, 15, 25, 30],
            "outcome": ["YES"] * 5,
            "shares": [10, 20, 15, 25, 30],
            "user_id": ["u1"] * 5,
        })

        with patch("app.election.historical.manifold_history.search_election_markets", return_value=mock_markets):
            with patch("app.election.historical.manifold_history.fetch_bet_history", return_value=mock_bets):
                results = backfill_election_markets(
                    search_terms=["test"],
                    markets_per_term=10,
                    db=election_db,
                )
                assert len(results) == 1
                assert "New test market" in results


class TestDeduplicateRemovesExtras:
    def test_deduplicate_removes_extras(self, election_db: Session):
        """Duplicate rows (same market_id + as_of) should be reduced to one."""
        ts = datetime(2024, 6, 1, 12, 0, 0)
        for _ in range(3):
            election_db.add(_make_quote("x", ts, price=0.55))
        election_db.commit()

        count_before = election_db.execute(
            select(HistoricalQuote).where(HistoricalQuote.platform_market_id == "x")
        ).scalars().all()
        assert len(count_before) == 3

        deleted = deduplicate_manifold_quotes(election_db)
        assert deleted == 2

        count_after = election_db.execute(
            select(HistoricalQuote).where(HistoricalQuote.platform_market_id == "x")
        ).scalars().all()
        assert len(count_after) == 1


class TestDeduplicateIdempotent:
    def test_deduplicate_idempotent(self, election_db: Session):
        """Running dedup twice should delete 0 on second run."""
        ts = datetime(2024, 6, 1, 12, 0, 0)
        for _ in range(3):
            election_db.add(_make_quote("y", ts))
        election_db.commit()

        first_run = deduplicate_manifold_quotes(election_db)
        assert first_run == 2

        second_run = deduplicate_manifold_quotes(election_db)
        assert second_run == 0


class TestNoFalseDedup:
    def test_no_false_dedup(self, election_db: Session):
        """Rows with same market_id but different timestamps are NOT duplicates."""
        for i in range(3):
            election_db.add(_make_quote("z", datetime(2024, 6, i + 1)))
        election_db.commit()

        deleted = deduplicate_manifold_quotes(election_db)
        assert deleted == 0

        remaining = election_db.execute(
            select(HistoricalQuote).where(HistoricalQuote.platform_market_id == "z")
        ).scalars().all()
        assert len(remaining) == 3


class TestBackwardCompatible:
    def test_backward_compatible(self):
        """Calling without db=None still works (old behavior)."""
        import pandas as pd

        mock_markets = [{"id": "compat1", "question": "Compat test"}]
        mock_bets = pd.DataFrame({
            "ts": pd.date_range("2024-01-01", periods=3, freq="h"),
            "prob_before": [0.4, 0.45, 0.5],
            "prob_after": [0.45, 0.5, 0.55],
            "amount": [10, 20, 15],
            "outcome": ["YES"] * 3,
            "shares": [10, 20, 15],
            "user_id": ["u1"] * 3,
        })

        with patch("app.election.historical.manifold_history.search_election_markets", return_value=mock_markets):
            with patch("app.election.historical.manifold_history.fetch_bet_history", return_value=mock_bets):
                # db=None (default) — should work without errors
                results = backfill_election_markets(
                    search_terms=["test"],
                    markets_per_term=10,
                    db=None,
                )
                assert len(results) == 1
