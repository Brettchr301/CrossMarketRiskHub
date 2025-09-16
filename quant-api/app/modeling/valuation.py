from __future__ import annotations

from datetime import datetime, UTC
from typing import Mapping

import numpy as np

from app.modeling.cost_model import estimate_total_cost_bps, net_return_after_cost
from app.modeling.types import FundamentalStatePoint, ValuationPoint


class ScenarioValuationModel:
    def __init__(
        self,
        horizon_days: int = 60,
        discount_rate: float = 0.11,
        terminal_growth: float = 0.02,
    ) -> None:
        self.horizon_days = horizon_days
        self.discount_rate = discount_rate
        self.terminal_growth = terminal_growth

    def value_company(
        self,
        state: FundamentalStatePoint,
        market_paths: Mapping[str, np.ndarray],
        spot_price: float,
        event_probabilities: Mapping[str, float] | None = None,
    ) -> ValuationPoint:
        event_probs = event_probabilities or {}
        if state.sector_type == "producer":
            ev_samples, equity_ps_samples = self._value_producer(state, market_paths, event_probs)
        else:
            ev_samples, equity_ps_samples = self._value_shipping(state, market_paths, event_probs)

        # Anchor intrinsic estimates to observed market levels to avoid extreme
        # overreaction in sparse/free-data environments.
        equity_ps_samples = 0.86 * spot_price + 0.14 * equity_ps_samples
        equity_ps_samples = np.clip(equity_ps_samples, 0.5 * spot_price, 1.9 * spot_price)

        ev_p05, ev_p50, ev_p95 = np.quantile(ev_samples, [0.05, 0.5, 0.95])
        eq_p05, eq_p50, eq_p95 = np.quantile(equity_ps_samples, [0.05, 0.5, 0.95])
        gross_ret = (eq_p50 - spot_price) / max(spot_price, 0.01)
        cost_bps = estimate_total_cost_bps(hold_days=self.horizon_days)
        expected_return_net_cost = net_return_after_cost(gross_ret, cost_bps)
        downside_p05 = (eq_p05 - spot_price) / max(spot_price, 0.01)
        return ValuationPoint(
            ticker=state.ticker,
            horizon_days=self.horizon_days,
            ev_p05=float(ev_p05),
            ev_p50=float(ev_p50),
            ev_p95=float(ev_p95),
            equity_ps_p05=float(eq_p05),
            equity_ps_p50=float(eq_p50),
            equity_ps_p95=float(eq_p95),
            expected_return_net_cost=float(expected_return_net_cost),
            downside_p05=float(downside_p05),
            as_of=datetime.now(UTC).replace(tzinfo=None),
            ev_samples=ev_samples,
            equity_ps_samples=equity_ps_samples,
        )

    def _value_producer(
        self,
        state: FundamentalStatePoint,
        market_paths: Mapping[str, np.ndarray],
        event_probs: Mapping[str, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        oil = market_paths.get("BRENT")
        if oil is None:
            oil = market_paths.get("WTI")
        if oil is None:
            oil = np.full(2500, 80.0)

        base_oil = float(np.median(oil))
        oil_ret = oil / max(base_oil, 1e-6) - 1.0
        oil_var = float(np.var(oil_ret))

        base_growth = float(state.meta_payload.get("production_growth_assumption", 0.03))
        growth_sensitivity = float(state.meta_payload.get("growth_beta_oil", 0.25))
        realized_beta = float(state.meta_payload.get("realized_price_beta_oil", 1.15))
        realized_gamma = float(state.meta_payload.get("realized_price_gamma_oil", 0.5))
        unit_cost_beta = float(state.meta_payload.get("unit_cost_beta_oil", 0.28))
        transport_beta = float(state.meta_payload.get("transport_beta_oil", 0.2))

        geo_prob = (
            float(event_probs.get("hormuz_closure", 0.0))
            + float(event_probs.get("sanctions_escalation", 0.0))
            + 0.5 * float(event_probs.get("red_sea_disruption", 0.0))
        )
        geo_shift = max(-0.2, min(0.45, geo_prob - 0.5))

        production_growth = np.clip(base_growth + growth_sensitivity * (np.mean(oil_ret) + geo_shift * 0.2), -0.22, 0.4)
        annual_volume = state.production * (1.0 + production_growth) * 365.0

        convex_component = oil_ret * oil_ret + oil_var
        realized_price = oil * (
            1.0
            + (1.0 - state.hedge_ratio) * (realized_beta * oil_ret + 0.5 * realized_gamma * convex_component)
            + 0.12 * geo_shift
        )
        realized_price = np.maximum(1.0, realized_price)
        unit_cost = state.cost_per_unit * (1.0 + unit_cost_beta * np.maximum(oil_ret, -0.35))
        transport = state.transport_cost * (1.0 + transport_beta * np.maximum(oil_ret, -0.35))
        ebitda = annual_volume * (realized_price - unit_cost - transport)
        ebitda = ebitda - state.sga
        financing = state.debt * state.interest_rate
        fcf = ebitda - state.capex - financing
        terminal_multiple = 6.5
        ev_dcf = np.maximum(
            0.0,
            fcf * (1.0 + self.terminal_growth) / max(1e-4, (self.discount_rate - self.terminal_growth)),
        )
        ev_mult = np.maximum(0.0, ebitda * terminal_multiple)
        ev = 0.7 * ev_dcf + 0.3 * ev_mult
        equity_value = np.maximum(0.0, ev - state.debt)
        equity_ps = equity_value / max(state.share_count, 1.0)
        return ev, equity_ps

    def _value_shipping(
        self,
        state: FundamentalStatePoint,
        market_paths: Mapping[str, np.ndarray],
        event_probs: Mapping[str, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        freight = market_paths.get("TD3")
        if freight is None:
            freight = market_paths.get("BDI", np.full(2500, 1500.0))

        fwd = market_paths.get("BDI")
        if fwd is None:
            fwd = freight
        base_freight = float(np.median(freight))
        base_fwd = float(np.median(fwd))
        freight_ret = freight / max(base_freight, 1e-6) - 1.0
        fwd_ret = fwd / max(base_fwd, 1e-6) - 1.0

        if freight.mean() > 300:
            tce_base = freight / 20.0
        else:
            tce_base = freight

        route_disruption = (
            0.8 * float(event_probs.get("red_sea_disruption", 0.0))
            + 0.5 * float(event_probs.get("hormuz_closure", 0.0))
            + 0.4 * float(event_probs.get("sanctions_escalation", 0.0))
        )
        route_shift = max(-0.2, min(0.55, route_disruption - 0.45))

        tce_beta = float(state.meta_payload.get("tce_beta_freight", 1.3))
        tce_gamma = float(state.meta_payload.get("tce_gamma_freight", 0.6))
        utilization_beta = float(state.meta_payload.get("utilization_beta_freight", 0.16))
        opex_beta = float(state.meta_payload.get("opex_beta_freight", 0.1))
        bunker_beta = float(state.meta_payload.get("bunker_beta_freight", 0.32))
        fleet_growth = float(state.meta_payload.get("fleet_growth_assumption", 0.02))

        convex = freight_ret * freight_ret + 0.6 * (fwd_ret * fwd_ret)
        tce = tce_base * (
            1.0
            + tce_beta * freight_ret
            + 0.5 * tce_gamma * convex
            + 0.22 * route_shift
        )
        fleet_days = state.production * (1.0 + fleet_growth + 0.2 * np.mean(freight_ret))
        utilization = np.clip(state.utilization + utilization_beta * freight_ret + 0.08 * route_shift, 0.6, 0.995)
        revenue = tce * fleet_days * utilization
        opex = state.cost_per_unit * fleet_days * (1.0 + opex_beta * np.maximum(freight_ret, -0.4))
        bunkers = state.transport_cost * fleet_days * (1.0 + bunker_beta * np.maximum(freight_ret, -0.4))
        financing = state.debt * state.interest_rate
        ebitda = revenue - opex - bunkers - state.sga
        fcf = ebitda - state.capex - financing
        terminal_multiple = 5.8
        ev_dcf = np.maximum(
            0.0,
            fcf * (1.0 + self.terminal_growth) / max(1e-4, (self.discount_rate - self.terminal_growth)),
        )
        ev_mult = np.maximum(0.0, ebitda * terminal_multiple)
        ev = 0.65 * ev_dcf + 0.35 * ev_mult
        equity_value = np.maximum(0.0, ev - state.debt)
        equity_ps = equity_value / max(state.share_count, 1.0)
        return ev, equity_ps
