from __future__ import annotations

from datetime import datetime
from typing import Iterable, Sequence, TypeVar

from sqlalchemy import Select, desc, func, select
from sqlalchemy.orm import Session

from app.db import models

T = TypeVar("T")


def save_all(db: Session, rows: Iterable[T]) -> None:
    for row in rows:
        db.add(row)
    db.commit()


def latest_event_probabilities(db: Session) -> Sequence[models.EventProbabilityModel]:
    subq = (
        select(
            models.EventProbabilityModel.event_id,
            func.max(models.EventProbabilityModel.as_of).label("max_as_of"),
        )
        .group_by(models.EventProbabilityModel.event_id)
        .subquery()
    )
    stmt: Select[tuple[models.EventProbabilityModel]] = (
        select(models.EventProbabilityModel)
        .join(
            subq,
            (models.EventProbabilityModel.event_id == subq.c.event_id)
            & (models.EventProbabilityModel.as_of == subq.c.max_as_of),
        )
        .order_by(models.EventProbabilityModel.event_id.asc())
    )
    return db.scalars(stmt).all()


def latest_commodity_distributions(db: Session) -> Sequence[models.CommodityDistributionModel]:
    subq = (
        select(
            models.CommodityDistributionModel.symbol,
            func.max(models.CommodityDistributionModel.as_of).label("max_as_of"),
        )
        .group_by(models.CommodityDistributionModel.symbol)
        .subquery()
    )
    stmt = (
        select(models.CommodityDistributionModel)
        .join(
            subq,
            (models.CommodityDistributionModel.symbol == subq.c.symbol)
            & (models.CommodityDistributionModel.as_of == subq.c.max_as_of),
        )
        .order_by(models.CommodityDistributionModel.symbol.asc())
    )
    return db.scalars(stmt).all()


def latest_fundamental_state(db: Session, ticker: str) -> models.FundamentalStateModel | None:
    stmt = (
        select(models.FundamentalStateModel)
        .where(models.FundamentalStateModel.ticker == ticker.upper())
        .order_by(desc(models.FundamentalStateModel.as_of))
        .limit(1)
    )
    return db.scalar(stmt)


def latest_valuation(db: Session, ticker: str) -> models.ValuationSnapshotModel | None:
    stmt = (
        select(models.ValuationSnapshotModel)
        .where(models.ValuationSnapshotModel.ticker == ticker.upper())
        .order_by(desc(models.ValuationSnapshotModel.as_of))
        .limit(1)
    )
    return db.scalar(stmt)


def latest_options_distribution(db: Session, ticker: str) -> models.OptionsImpliedDistributionModel | None:
    stmt = (
        select(models.OptionsImpliedDistributionModel)
        .where(models.OptionsImpliedDistributionModel.ticker == ticker.upper())
        .order_by(desc(models.OptionsImpliedDistributionModel.as_of))
        .limit(1)
    )
    return db.scalar(stmt)


def latest_signals(db: Session, limit: int = 25) -> Sequence[models.SignalModel]:
    subq = (
        select(
            models.SignalModel.ticker,
            func.max(models.SignalModel.as_of).label("max_as_of"),
        )
        .group_by(models.SignalModel.ticker)
        .subquery()
    )
    stmt = (
        select(models.SignalModel)
        .join(
            subq,
            (models.SignalModel.ticker == subq.c.ticker) & (models.SignalModel.as_of == subq.c.max_as_of),
        )
        .order_by(desc(models.SignalModel.score))
        .limit(limit)
    )
    return db.scalars(stmt).all()


def latest_backtest_metrics(db: Session) -> models.BacktestMetricModel | None:
    stmt = select(models.BacktestMetricModel).order_by(desc(models.BacktestMetricModel.as_of)).limit(1)
    return db.scalar(stmt)


def latest_equity_price(db: Session, ticker: str) -> float | None:
    stmt = (
        select(models.EquityQuote.close_price)
        .where(models.EquityQuote.ticker == ticker.upper())
        .order_by(desc(models.EquityQuote.as_of))
        .limit(1)
    )
    return db.scalar(stmt)


def event_probability_deltas(db: Session, as_of: datetime) -> dict[str, float]:
    deltas: dict[str, float] = {}
    latest = latest_event_probabilities(db)
    for row in latest:
        prev_stmt = (
            select(models.EventProbabilityModel)
            .where(
                models.EventProbabilityModel.event_id == row.event_id,
                models.EventProbabilityModel.as_of < as_of,
            )
            .order_by(desc(models.EventProbabilityModel.as_of))
            .limit(1)
        )
        prev = db.scalar(prev_stmt)
        if prev is None:
            deltas[row.event_id] = 0.0
        else:
            deltas[row.event_id] = row.prob - prev.prob
    return deltas
