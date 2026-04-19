"""Backtest runner.

Orchestrates: backfill historical data → link contracts → seed outcomes → run backtests.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from app.election.backtest.engine import (
    BacktestResult,
    backtest_cross_market,
    backtest_outcome_betting,
)
from app.election.db.historical_models import HistoricalQuote, RaceOutcome
from app.election.db.session import get_session_factory, init_election_db
from app.election.historical import kalshi_history, polymarket_history, predictit_history
from app.election.mappings.race_linker import link_contract_to_race
from app.election.mappings.race_registry_historical import ALL_RACES_HISTORICAL

logger = logging.getLogger(__name__)


def seed_race_outcomes(db: Session) -> int:
    """Populate race_outcomes table from the static registry."""
    from sqlalchemy import select

    existing = db.execute(select(RaceOutcome)).scalars().all()
    if existing:
        logger.info("race_outcomes already has %d rows, skipping seed", len(existing))
        return 0

    n = 0
    for idx, spec in enumerate(ALL_RACES_HISTORICAL):
        winner = getattr(spec, "winner", None)
        if winner is None:
            continue
        db.add(RaceOutcome(
            race_id=idx + 1,
            race_type=spec.race_type,
            state=spec.state,
            cycle=spec.cycle,
            winner_party=winner,
            winner_name=getattr(spec, "winner_name", None),
            election_date=spec.election_date,
            source="registry",
        ))
        n += 1

    db.commit()
    logger.info("Seeded %d race outcomes", n)
    return n


def backfill_polymarket_cycle(db: Session, cycle: int) -> int:
    """Backfill Polymarket historical data for a cycle."""
    results = polymarket_history.backfill_cycle(cycle)
    n = 0
    for question, series in results.items():
        link = link_contract_to_race(question)
        for ts, price in series.items():
            db.add(HistoricalQuote(
                race_id=link.race_id,
                platform="polymarket",
                platform_market_id=f"poly_{abs(hash(question)) % 10**10}",
                question=question,
                cycle=cycle,
                price=float(price),
                as_of=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
            ))
            n += 1
    db.commit()
    logger.info("Polymarket %d: backfilled %d historical quotes", cycle, n)
    return n


def backfill_predictit_wayback(db: Session, cycle: int, from_date: str, to_date: str) -> int:
    """Backfill PredictIt historical from Wayback Machine snapshots."""
    results = predictit_history.backfill_from_wayback(from_date, to_date, sample_n=30)
    n = 0
    for contract_key, series in results.items():
        link = link_contract_to_race(contract_key)
        for ts, price in series.items():
            if price <= 0:
                continue
            db.add(HistoricalQuote(
                race_id=link.race_id,
                platform="predictit",
                platform_market_id=f"pi_{abs(hash(contract_key)) % 10**10}",
                question=contract_key,
                cycle=cycle,
                price=float(price),
                as_of=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
            ))
            n += 1
    db.commit()
    logger.info("PredictIt %d: backfilled %d historical quotes from Wayback", cycle, n)
    return n


def run_full_backfill_and_backtest(cycles: list[int] = None) -> dict[int, dict[str, Any]]:
    """Full pipeline: backfill → seed outcomes → run both backtests per cycle."""
    if cycles is None:
        cycles = [2018, 2020, 2022, 2024]

    init_election_db()
    # Ensure historical tables exist too
    from app.election.db.historical_models import HistoricalQuote as _, RaceOutcome as __
    from app.election.db.session import _get_engine
    from app.election.db.models import ElectionBase
    ElectionBase.metadata.create_all(_get_engine())

    factory = get_session_factory()
    db = factory()

    try:
        # Seed outcomes once
        seed_race_outcomes(db)

        results: dict[int, dict[str, Any]] = {}
        for cycle in cycles:
            logger.info("===== Cycle %d =====", cycle)
            # Polymarket backfill (only meaningful for 2020+)
            if cycle >= 2020:
                try:
                    backfill_polymarket_cycle(db, cycle)
                except Exception as exc:
                    logger.warning("Polymarket backfill failed for %d: %s", cycle, exc)

            # PredictIt Wayback backfill
            if cycle >= 2018:
                try:
                    from_date = f"{cycle}0101"
                    to_date = f"{cycle}1231"
                    backfill_predictit_wayback(db, cycle, from_date, to_date)
                except Exception as exc:
                    logger.warning("PredictIt Wayback backfill failed for %d: %s", cycle, exc)

            # Run backtests
            try:
                cm = backtest_cross_market(db, cycle, run_name=f"cm_{cycle}")
            except Exception as exc:
                logger.warning("Cross-market backtest failed for %d: %s", cycle, exc)
                cm = None

            try:
                ob = backtest_outcome_betting(db, cycle, run_name=f"ob_{cycle}")
            except Exception as exc:
                logger.warning("Outcome-betting backtest failed for %d: %s", cycle, exc)
                ob = None

            results[cycle] = {
                "cross_market": cm.__dict__ if cm else None,
                "outcome_betting": ob.__dict__ if ob else None,
            }
    finally:
        db.close()

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_full_backfill_and_backtest([2022, 2024])
    for cycle, res in results.items():
        print(f"\n=== Cycle {cycle} ===")
        for strategy, metrics in res.items():
            if metrics:
                print(f"  {strategy}: trades={metrics.get('n_trades')}, pnl=${metrics.get('total_pnl'):.2f}, win_rate={metrics.get('win_rate'):.1%}, sharpe={metrics.get('sharpe'):.2f}")
