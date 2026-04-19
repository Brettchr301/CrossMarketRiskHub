from __future__ import annotations
from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    Boolean, Date, DateTime, Float, ForeignKey, Index,
    Integer, JSON, String, Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class ElectionBase(DeclarativeBase):
    pass


class Race(ElectionBase):
    __tablename__ = "races"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_type: Mapped[str] = mapped_column(String(32), index=True)  # presidential, senate, house, governor
    state: Mapped[str] = mapped_column(String(4), index=True)  # PA, AZ, US for presidential
    district: Mapped[str | None] = mapped_column(String(8), nullable=True)  # NULL for senate/pres
    cycle: Mapped[int] = mapped_column(Integer, index=True)  # 2026, 2028
    election_date: Mapped[date] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=datetime.utcnow)


class Candidate(ElectionBase):
    __tablename__ = "candidates"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[int] = mapped_column(Integer, ForeignKey("races.id"), index=True)
    name: Mapped[str] = mapped_column(String(128))
    party: Mapped[str] = mapped_column(String(32), index=True)  # D, R, I, L
    incumbent: Mapped[bool] = mapped_column(Boolean, default=False)
    fec_candidate_id: Mapped[str | None] = mapped_column(String(16), nullable=True)


class MarketContract(ElectionBase):
    __tablename__ = "market_contracts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[int] = mapped_column(Integer, ForeignKey("races.id"), index=True)
    candidate_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("candidates.id"), nullable=True)
    platform: Mapped[str] = mapped_column(String(32), index=True)  # polymarket, kalshi, predictit, metaculus
    platform_market_id: Mapped[str] = mapped_column(String(256))
    platform_question: Mapped[str | None] = mapped_column(Text, nullable=True)
    contract_type: Mapped[str] = mapped_column(String(16), default="binary")  # binary, multi_outcome
    is_inverted: Mapped[bool] = mapped_column(Boolean, default=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    discovered_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=datetime.utcnow)


class MarketQuote(ElectionBase):
    __tablename__ = "market_quotes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    contract_id: Mapped[int] = mapped_column(Integer, ForeignKey("market_contracts.id"), index=True)
    yes_bid: Mapped[float] = mapped_column(Float)
    yes_ask: Mapped[float] = mapped_column(Float)
    last_price: Mapped[float] = mapped_column(Float)
    volume_24h: Mapped[float] = mapped_column(Float, default=0.0)
    open_interest: Mapped[float] = mapped_column(Float, default=0.0)
    liquidity_score: Mapped[float] = mapped_column(Float, default=0.5)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True)


class BlendedProbability(ElectionBase):
    __tablename__ = "blended_probabilities"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[int] = mapped_column(Integer, ForeignKey("races.id"), index=True)
    candidate_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("candidates.id"), nullable=True)
    prob: Mapped[float] = mapped_column(Float)
    ci_low: Mapped[float] = mapped_column(Float)
    ci_high: Mapped[float] = mapped_column(Float)
    n_platforms: Mapped[int] = mapped_column(Integer)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True)


class ArbitrageOpportunity(ElectionBase):
    __tablename__ = "arbitrage_opportunities"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    arb_type: Mapped[str] = mapped_column(String(32), index=True)
    race_id: Mapped[int] = mapped_column(Integer, ForeignKey("races.id"), index=True)
    description: Mapped[str] = mapped_column(Text)
    gross_edge_pct: Mapped[float] = mapped_column(Float)
    net_edge_pct: Mapped[float] = mapped_column(Float)
    buy_platform: Mapped[str] = mapped_column(String(32))
    buy_contract_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    buy_price: Mapped[float] = mapped_column(Float)
    sell_platform: Mapped[str] = mapped_column(String(32))
    sell_contract_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sell_price: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True)
    status: Mapped[str] = mapped_column(String(16), default="active")


class PollingData(ElectionBase):
    __tablename__ = "polling_data"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[int] = mapped_column(Integer, ForeignKey("races.id"), index=True)
    pollster: Mapped[str] = mapped_column(String(128))
    sample_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    margin_of_error: Mapped[float | None] = mapped_column(Float, nullable=True)
    candidate_id: Mapped[int] = mapped_column(Integer, ForeignKey("candidates.id"), index=True)
    pct: Mapped[float] = mapped_column(Float)
    poll_date: Mapped[date] = mapped_column(Date, index=True)
    source: Mapped[str] = mapped_column(String(32))  # 538, rcp
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=datetime.utcnow)


class CampaignFinance(ElectionBase):
    __tablename__ = "campaign_finance"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    candidate_id: Mapped[int] = mapped_column(Integer, ForeignKey("candidates.id"), index=True)
    period_start: Mapped[date] = mapped_column(Date)
    period_end: Mapped[date] = mapped_column(Date)
    receipts: Mapped[float] = mapped_column(Float, default=0.0)
    disbursements: Mapped[float] = mapped_column(Float, default=0.0)
    cash_on_hand: Mapped[float] = mapped_column(Float, default=0.0)
    individual_contributions: Mapped[float] = mapped_column(Float, default=0.0)
    pac_contributions: Mapped[float] = mapped_column(Float, default=0.0)
    source_filing_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=datetime.utcnow)


class AltDataSignal(ElectionBase):
    __tablename__ = "alt_data_signals"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    signal_type: Mapped[str] = mapped_column(String(32), index=True)  # weather, google_trends, wikipedia, early_vote
    race_id: Mapped[int] = mapped_column(Integer, ForeignKey("races.id"), index=True)
    state: Mapped[str] = mapped_column(String(4), index=True)
    value: Mapped[float] = mapped_column(Float)
    raw_value: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True)


class CorrelationFeature(ElectionBase):
    __tablename__ = "correlation_features"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[int] = mapped_column(Integer, ForeignKey("races.id"), index=True)
    feature_name: Mapped[str] = mapped_column(String(128))
    feature_value: Mapped[float] = mapped_column(Float)
    target_prob_change: Mapped[float | None] = mapped_column(Float, nullable=True)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True)


class AlphaModelPrediction(ElectionBase):
    __tablename__ = "alpha_model_predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[int] = mapped_column(Integer, ForeignKey("races.id"), index=True)
    predicted_prob_change: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    top_features: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    model_version: Mapped[str] = mapped_column(String(32))
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True)


# Composite indexes for time-series queries
Index("ix_mq_contract_asof", MarketQuote.contract_id, MarketQuote.as_of.desc())
Index("ix_bp_race_asof", BlendedProbability.race_id, BlendedProbability.as_of.desc())
Index("ix_arb_type_detected", ArbitrageOpportunity.arb_type, ArbitrageOpportunity.detected_at.desc())
Index("ix_poll_race_date", PollingData.race_id, PollingData.poll_date.desc())
Index("ix_alt_type_asof", AltDataSignal.signal_type, AltDataSignal.as_of.desc())
Index("ix_cf_race_asof", CorrelationFeature.race_id, CorrelationFeature.as_of.desc())
Index("ix_amp_race_asof", AlphaModelPrediction.race_id, AlphaModelPrediction.as_of.desc())
