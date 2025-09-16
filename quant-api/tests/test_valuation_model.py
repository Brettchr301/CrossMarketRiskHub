from __future__ import annotations

from datetime import datetime, UTC

import numpy as np

from app.modeling.types import FundamentalStatePoint
from app.modeling.valuation import ScenarioValuationModel


def test_scenario_valuation_produces_distribution_for_producer():
    state = FundamentalStatePoint(
        ticker="CIVI",
        guidance_period="2026Q1",
        sector_type="producer",
        production=300000.0,
        cost_per_unit=28.0,
        transport_cost=7.0,
        sga=300_000_000.0,
        capex=1_000_000_000.0,
        debt=3_000_000_000.0,
        interest_rate=0.065,
        hedge_ratio=0.4,
        utilization=0.98,
        share_count=100_000_000.0,
        confidence=0.8,
        meta_payload={},
        as_of=datetime.now(UTC).replace(tzinfo=None),
    )
    oil = np.linspace(70.0, 105.0, 600)
    model = ScenarioValuationModel(horizon_days=60)
    out = model.value_company(state, {"BRENT": oil}, spot_price=40.0)
    assert out.ev_p95 >= out.ev_p50 >= out.ev_p05
    assert out.equity_ps_p95 >= out.equity_ps_p50 >= out.equity_ps_p05


def test_scenario_valuation_uses_convexity_terms():
    state = FundamentalStatePoint(
        ticker="AR",
        guidance_period="2026Q1",
        sector_type="producer",
        production=180000.0,
        cost_per_unit=24.0,
        transport_cost=6.0,
        sga=180_000_000.0,
        capex=800_000_000.0,
        debt=2_000_000_000.0,
        interest_rate=0.06,
        hedge_ratio=0.25,
        utilization=0.96,
        share_count=120_000_000.0,
        confidence=0.8,
        meta_payload={
            "production_growth_assumption": 0.04,
            "realized_price_beta_oil": 1.2,
            "realized_price_gamma_oil": 1.0,
            "unit_cost_beta_oil": 0.2,
            "transport_beta_oil": 0.2,
        },
        as_of=datetime.now(UTC).replace(tzinfo=None),
    )
    model = ScenarioValuationModel(horizon_days=60)
    base = np.full(1000, 80.0)
    low_var = base + np.linspace(-5.0, 5.0, 1000)
    high_var = base + np.linspace(-20.0, 20.0, 1000)
    out_low = model.value_company(state, {"BRENT": low_var}, spot_price=35.0)
    out_high = model.value_company(state, {"BRENT": high_var}, spot_price=35.0)
    assert out_high.ev_p50 >= out_low.ev_p50
