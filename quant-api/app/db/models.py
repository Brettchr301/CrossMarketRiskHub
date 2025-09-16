from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import JSON, Date, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class PredictionQuote(Base):
    __tablename__ = "prediction_quotes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(64), index=True)
    event_id: Mapped[str] = mapped_column(String(128), index=True)
    mid_price: Mapped[float] = mapped_column(Float)
    bid: Mapped[float] = mapped_column(Float)
    ask: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)
    liquidity_score: Mapped[float] = mapped_column(Float)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=datetime.utcnow)


class CommodityQuote(Base):
    __tablename__ = "commodity_quotes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    price: Mapped[float] = mapped_column(Float)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True)


class ShippingIndexQuote(Base):
    __tablename__ = "shipping_indices"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    index_name: Mapped[str] = mapped_column(String(64), index=True)
    value: Mapped[float] = mapped_column(Float)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True)


class EquityQuote(Base):
    __tablename__ = "equity_quotes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    close_price: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True)


class OptionChainQuote(Base):
    __tablename__ = "options_chain_eod"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    expiration: Mapped[date] = mapped_column(Date, index=True)
    strike: Mapped[float] = mapped_column(Float, index=True)
    option_type: Mapped[str] = mapped_column(String(8), index=True)
    bid: Mapped[float] = mapped_column(Float)
    ask: Mapped[float] = mapped_column(Float)
    implied_vol: Mapped[float] = mapped_column(Float)
    open_interest: Mapped[float] = mapped_column(Float)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True)


class EventProbabilityModel(Base):
    __tablename__ = "event_probabilities"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(128), index=True)
    prob: Mapped[float] = mapped_column(Float)
    ci_low: Mapped[float] = mapped_column(Float)
    ci_high: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(32), default="blended")
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True)


class ScenarioPricePath(Base):
    __tablename__ = "scenario_price_paths"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    simulation_tag: Mapped[str] = mapped_column(String(64), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    step: Mapped[int] = mapped_column(Integer)
    price: Mapped[float] = mapped_column(Float)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True)


class CommodityDistributionModel(Base):
    __tablename__ = "commodity_distributions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    horizon_days: Mapped[int] = mapped_column(Integer, index=True)
    p05: Mapped[float] = mapped_column(Float)
    p50: Mapped[float] = mapped_column(Float)
    p95: Mapped[float] = mapped_column(Float)
    simulation_tag: Mapped[str] = mapped_column(String(64), index=True)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True)


class CompanyFactorExposure(Base):
    __tablename__ = "company_factor_exposure"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    factor: Mapped[str] = mapped_column(String(64), index=True)
    beta: Mapped[float] = mapped_column(Float)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True)


class FundamentalStateModel(Base):
    __tablename__ = "fundamental_states"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    guidance_period: Mapped[str] = mapped_column(String(32))
    sector_type: Mapped[str] = mapped_column(String(32), index=True)
    production: Mapped[float] = mapped_column(Float)
    cost_per_unit: Mapped[float] = mapped_column(Float)
    transport_cost: Mapped[float] = mapped_column(Float)
    sga: Mapped[float] = mapped_column(Float)
    capex: Mapped[float] = mapped_column(Float)
    debt: Mapped[float] = mapped_column(Float)
    interest_rate: Mapped[float] = mapped_column(Float)
    hedge_ratio: Mapped[float] = mapped_column(Float)
    utilization: Mapped[float] = mapped_column(Float)
    share_count: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    meta_payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True)


class ValuationSnapshotModel(Base):
    __tablename__ = "valuation_snapshots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    horizon_days: Mapped[int] = mapped_column(Integer)
    ev_p05: Mapped[float] = mapped_column(Float)
    ev_p50: Mapped[float] = mapped_column(Float)
    ev_p95: Mapped[float] = mapped_column(Float)
    equity_ps_p05: Mapped[float] = mapped_column(Float)
    equity_ps_p50: Mapped[float] = mapped_column(Float)
    equity_ps_p95: Mapped[float] = mapped_column(Float)
    expected_return_net_cost: Mapped[float] = mapped_column(Float)
    downside_p05: Mapped[float] = mapped_column(Float)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True)


class OptionsImpliedDistributionModel(Base):
    __tablename__ = "options_implied_distributions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    horizon_days: Mapped[int] = mapped_column(Integer)
    mean_return: Mapped[float] = mapped_column(Float)
    std_return: Mapped[float] = mapped_column(Float)
    downside_p05: Mapped[float] = mapped_column(Float)
    upside_p95: Mapped[float] = mapped_column(Float)
    meta_payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True)


class SignalModel(Base):
    __tablename__ = "signal_book"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    score: Mapped[float] = mapped_column(Float)
    direction: Mapped[str] = mapped_column(String(8))
    holding_period_days: Mapped[int] = mapped_column(Integer)
    expected_return_net_cost: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    risk_flags: Mapped[str] = mapped_column(Text, default="")
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True)


class BacktestTradeModel(Base):
    __tablename__ = "backtest_trades"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    entry_time: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True)
    exit_time: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True)
    direction: Mapped[str] = mapped_column(String(8))
    gross_return: Mapped[float] = mapped_column(Float)
    net_return: Mapped[float] = mapped_column(Float)
    cost_bps: Mapped[float] = mapped_column(Float)
    signal_score: Mapped[float] = mapped_column(Float)


class BacktestMetricModel(Base):
    __tablename__ = "backtest_metrics"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True)
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True)
    sharpe: Mapped[float] = mapped_column(Float)
    hit_rate: Mapped[float] = mapped_column(Float)
    average_alpha: Mapped[float] = mapped_column(Float)
    max_drawdown: Mapped[float] = mapped_column(Float)
    turnover: Mapped[float] = mapped_column(Float)
    capacity: Mapped[float] = mapped_column(Float)
    irr: Mapped[float] = mapped_column(Float)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True)


Index("ix_prediction_quote_event_asof", PredictionQuote.event_id, PredictionQuote.as_of.desc())
Index("ix_equity_quote_ticker_asof", EquityQuote.ticker, EquityQuote.as_of.desc())
Index("ix_signal_ticker_asof", SignalModel.ticker, SignalModel.as_of.desc())
