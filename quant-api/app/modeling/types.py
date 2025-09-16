from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


@dataclass(slots=True)
class EventProbabilityPoint:
    event_id: str
    prob: float
    ci_low: float
    ci_high: float
    as_of: datetime


@dataclass(slots=True)
class DistributionSummary:
    symbol: str
    horizon_days: int
    p05: float
    p50: float
    p95: float
    as_of: datetime
    simulation_tag: str
    samples: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))


@dataclass(slots=True)
class FundamentalStatePoint:
    ticker: str
    guidance_period: str
    sector_type: str
    production: float
    cost_per_unit: float
    transport_cost: float
    sga: float
    capex: float
    debt: float
    interest_rate: float
    hedge_ratio: float
    utilization: float
    share_count: float
    confidence: float
    meta_payload: dict[str, Any]
    as_of: datetime


@dataclass(slots=True)
class ValuationPoint:
    ticker: str
    horizon_days: int
    ev_p05: float
    ev_p50: float
    ev_p95: float
    equity_ps_p05: float
    equity_ps_p50: float
    equity_ps_p95: float
    expected_return_net_cost: float
    downside_p05: float
    as_of: datetime
    ev_samples: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    equity_ps_samples: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))


@dataclass(slots=True)
class OptionsImpliedPoint:
    ticker: str
    horizon_days: int
    mean_return: float
    std_return: float
    downside_p05: float
    upside_p95: float
    as_of: datetime
    meta_payload: dict[str, Any]


@dataclass(slots=True)
class SignalPoint:
    ticker: str
    score: float
    direction: str
    holding_period_days: int
    expected_return_net_cost: float
    confidence: float
    risk_flags: list[str]
    as_of: datetime

