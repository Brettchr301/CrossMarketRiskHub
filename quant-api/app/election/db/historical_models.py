"""Historical election database models.

Adds tables for backtesting: historical quotes, race outcomes, backtest results.
Uses the same ElectionBase as models.py so they share the same DB.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    Boolean, Date, DateTime, Float, ForeignKey, Index,
    Integer, JSON, String, Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.election.db.models import ElectionBase


class HistoricalQuote(ElectionBase):
    """Daily-resolution historical price points for closed markets."""
    __tablename__ = "historical_quotes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    platform: Mapped[str] = mapped_column(String(32), index=True)
    platform_market_id: Mapped[str] = mapped_column(String(256), index=True)
    question: Mapped[str] = mapped_column(Text)
    cycle: Mapped[int] = mapped_column(Integer, index=True)
    price: Mapped[float] = mapped_column(Float)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True)


class RaceOutcome(ElectionBase):
    """Actual election results (ground truth)."""
    __tablename__ = "race_outcomes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[int] = mapped_column(Integer, index=True)
    race_type: Mapped[str] = mapped_column(String(32))
    state: Mapped[str] = mapped_column(String(4))
    cycle: Mapped[int] = mapped_column(Integer, index=True)
    winner_party: Mapped[str] = mapped_column(String(4))
    winner_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    election_date: Mapped[date] = mapped_column(Date)
    source: Mapped[str] = mapped_column(String(64), default="registry")
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=datetime.utcnow)


class BacktestRun(ElectionBase):
    """Metadata for a backtest run."""
    __tablename__ = "backtest_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_name: Mapped[str] = mapped_column(String(128))
    strategy: Mapped[str] = mapped_column(String(64))  # cross_market, dutch_book, parlay, correlation, alpha
    cycle: Mapped[int] = mapped_column(Integer)
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date] = mapped_column(Date)
    n_trades: Mapped[int] = mapped_column(Integer, default=0)
    total_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    win_rate: Mapped[float] = mapped_column(Float, default=0.0)
    sharpe: Mapped[float] = mapped_column(Float, default=0.0)
    max_drawdown: Mapped[float] = mapped_column(Float, default=0.0)
    config: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=datetime.utcnow)


class BacktestTrade(ElectionBase):
    """Individual simulated trade from a backtest run."""
    __tablename__ = "election_backtest_trades"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(Integer, ForeignKey("backtest_runs.id"), index=True)
    race_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    trade_type: Mapped[str] = mapped_column(String(32))  # arb_cross, arb_dutch, alpha_long, etc.
    entry_date: Mapped[date] = mapped_column(Date)
    entry_price: Mapped[float] = mapped_column(Float)
    exit_date: Mapped[date] = mapped_column(Date)
    exit_price: Mapped[float] = mapped_column(Float)
    settlement_price: Mapped[float] = mapped_column(Float)  # 0 or 1 at resolution
    gross_pnl: Mapped[float] = mapped_column(Float)
    fees: Mapped[float] = mapped_column(Float, default=0.0)
    net_pnl: Mapped[float] = mapped_column(Float)
    won: Mapped[bool] = mapped_column(Boolean, default=False)
    description: Mapped[str] = mapped_column(Text)


Index("ix_hq_race_asof", HistoricalQuote.race_id, HistoricalQuote.as_of.desc())
Index("ix_hq_cycle_platform", HistoricalQuote.cycle, HistoricalQuote.platform)
Index("ix_ro_cycle_type", RaceOutcome.cycle, RaceOutcome.race_type)
Index("ix_bt_trade_run", BacktestTrade.run_id, BacktestTrade.entry_date)
