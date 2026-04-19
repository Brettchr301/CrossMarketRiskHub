"""Arbitrage detection engine.

Orchestrates all arbitrage detection strategies and writes results to DB.
"""
from __future__ import annotations
import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from app.election.arbitrage.cross_market import ArbSignal, detect_cross_market_arbs
from app.election.arbitrage.dutch_book import detect_dutch_books
from app.election.arbitrage.parlay import detect_parlay_arbs
from app.election.arbitrage.correlation_arb import detect_correlation_arbs
from app.election.db.models import ArbitrageOpportunity

logger = logging.getLogger(__name__)


class ArbitrageEngine:
    """Orchestrates all arbitrage detection strategies."""

    def run_all(
        self,
        db: Session,
        quotes_by_race: dict[int, list[dict[str, Any]]],
        quotes_by_race_platform: dict[tuple[int, str], list[dict[str, Any]]],
        aggregate_quotes: dict[str, dict[str, Any]],
        component_probs: dict[str, list[float]],
        market_probs: dict[str, dict[str, Any]],
    ) -> list[ArbSignal]:
        """Run all arbitrage detection and persist results."""
        all_signals: list[ArbSignal] = []

        # 1. Cross-market
        try:
            cross = detect_cross_market_arbs(quotes_by_race)
            all_signals.extend(cross)
            logger.info("Cross-market arbs found: %d", len(cross))
        except Exception as exc:
            logger.error("Cross-market detection failed: %s", exc)

        # 2. Dutch books
        try:
            dutch = detect_dutch_books(quotes_by_race_platform)
            all_signals.extend(dutch)
            logger.info("Dutch book arbs found: %d", len(dutch))
        except Exception as exc:
            logger.error("Dutch book detection failed: %s", exc)

        # 3. Parlay decomposition
        try:
            parlay = detect_parlay_arbs(aggregate_quotes, component_probs)
            all_signals.extend(parlay)
            logger.info("Parlay arbs found: %d", len(parlay))
        except Exception as exc:
            logger.error("Parlay detection failed: %s", exc)

        # 4. Correlation arbitrage
        try:
            corr = detect_correlation_arbs(market_probs)
            all_signals.extend(corr)
            logger.info("Correlation arbs found: %d", len(corr))
        except Exception as exc:
            logger.error("Correlation detection failed: %s", exc)

        # Persist to DB
        now = datetime.now(UTC).replace(tzinfo=None)
        for sig in all_signals:
            row = ArbitrageOpportunity(
                arb_type=sig.arb_type,
                race_id=sig.race_id,
                description=sig.description,
                gross_edge_pct=sig.gross_edge_pct,
                net_edge_pct=sig.net_edge_pct,
                buy_platform=sig.buy_platform,
                buy_contract_id=sig.buy_contract_id,
                buy_price=sig.buy_price,
                sell_platform=sig.sell_platform,
                sell_contract_id=sig.sell_contract_id,
                sell_price=sig.sell_price,
                confidence=sig.confidence,
                detected_at=now,
                status="active",
            )
            db.add(row)

        if all_signals:
            db.commit()
            logger.info("Persisted %d arbitrage opportunities", len(all_signals))

        return all_signals
