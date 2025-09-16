from __future__ import annotations

from datetime import datetime
from typing import List

from pydantic import BaseModel, Field


class EventProbability(BaseModel):
    event_id: str
    prob: float = Field(ge=0.0, le=1.0)
    ci_low: float = Field(ge=0.0, le=1.0)
    ci_high: float = Field(ge=0.0, le=1.0)
    as_of: datetime


class CommodityDistribution(BaseModel):
    symbol: str
    horizon_days: int
    p05: float
    p50: float
    p95: float
    as_of: datetime


class FundamentalState(BaseModel):
    ticker: str
    guidance_period: str
    production: float
    costs: float
    capex: float
    debt: float
    shares: float
    confidence: float = Field(ge=0.0, le=1.0)
    as_of: datetime


class ValuationDistribution(BaseModel):
    ticker: str
    ev_p50: float
    equity_ps_p50: float
    expected_return_net_cost: float
    downside_p05: float
    ev_p05: float
    ev_p95: float
    equity_ps_p05: float
    equity_ps_p95: float
    as_of: datetime


class ImpliedDistribution(BaseModel):
    ticker: str
    horizon_days: int
    mean_return: float
    std_return: float
    downside_p05: float
    upside_p95: float
    as_of: datetime


class Signal(BaseModel):
    ticker: str
    score: float
    direction: str
    holding_period_days: int
    expected_return_net_cost: float
    risk_flags: List[str]
    as_of: datetime


class BacktestMetrics(BaseModel):
    sharpe: float
    hit_rate: float
    average_alpha: float
    max_drawdown: float
    turnover: float
    capacity: float
    irr: float
    as_of: datetime


class CorrelationDriver(BaseModel):
    name: str
    source: str
    correlation: float
    lag_days: int = 0


class TickerCorrelationStats(BaseModel):
    ticker: str
    sample_size: int
    corr_brent: float | None = None
    corr_wti: float | None = None
    corr_shipping: float | None = None
    top_drivers: List[CorrelationDriver]


class CorrelationSnapshot(BaseModel):
    as_of: datetime
    lookback_days: int
    tickers: List[TickerCorrelationStats]


class PredictiveContract(BaseModel):
    market_id: str
    question: str
    category: str
    best_target: str
    lead_days: int
    correlation: float
    liquidity_score: float
    staleness_days: int
    predictive_score: float


class PredictiveContractsSnapshot(BaseModel):
    as_of: datetime
    lookback_days: int
    contracts: List[PredictiveContract]


class ResearchSeriesPoint(BaseModel):
    date: str
    stock: float
    brent: float
    wti: float
    shipping_spot: float
    shipping_fwd: float
    event_hormuz: float
    event_red_sea: float
    event_oil_100: float


class ModelValidationStats(BaseModel):
    baseline_hit_rate: float
    enriched_hit_rate: float
    baseline_mae: float
    enriched_mae: float
    enriched_expected_return_20d: float
    fair_value_price: float
    spot_price: float


class ShippingHedgeStats(BaseModel):
    spot_proxy: str
    forward_proxy: str
    current_basis_pct: float
    one_month_expected_basis_pct: float
    hedge_beta_to_forward: float


class TickerResearchView(BaseModel):
    ticker: str
    as_of: datetime
    series: List[ResearchSeriesPoint]
    top_predictive_contracts: List[PredictiveContract]
    validation: ModelValidationStats
    hedge: ShippingHedgeStats


class CommodityTypeEffectiveness(BaseModel):
    commodity_type: str
    modeled_count: int
    avg_hit_rate: float
    avg_expected_return_net_cost: float
    avg_score: float
    top_bucket_avg_net_return: float
    contract_coverage_pct: float = 0.0
    avg_commodity_beta: float = 0.0
    best_ticker: str = ""
    best_hit_rate: float = 0.0


class GlobalOpportunity(BaseModel):
    ticker: str
    commodity_type: str
    country: str
    sector: str
    direction: str
    score: float
    spot_price: float
    fair_value_price: float
    expected_return_gross: float
    expected_return_net_cost: float
    cost_bps: float
    hit_rate: float
    mae: float
    confidence: float
    predicted_margin_next: float
    predicted_margin_change: float
    production_growth_assumption: float
    oil_beta: float
    oil_gamma: float
    shipping_beta: float
    shipping_gamma: float
    event_beta: float
    event_gamma: float
    commodity_beta: float = 0.0
    market_cap: float
    avg_daily_volume: float
    risk_flags: List[str]
    top_predictive_contracts: List[str]
    top_features: List[str] = []


class GlobalOpportunitiesSnapshot(BaseModel):
    as_of: datetime
    lookback_days: int
    universe_size: int
    modeled_count: int
    spot_proxy: str
    forward_proxy: str
    commodity_type_stats: List[CommodityTypeEffectiveness]
    opportunities: List[GlobalOpportunity]
