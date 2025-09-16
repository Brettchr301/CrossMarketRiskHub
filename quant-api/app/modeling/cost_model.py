from __future__ import annotations


def estimate_total_cost_bps(
    commission_bps: float = 2.0,
    spread_bps: float = 10.0,
    slippage_bps: float = 6.0,
    impact_bps: float = 8.0,
    borrow_bps_annual: float = 300.0,
    hold_days: int = 60,
) -> float:
    borrow_bps = borrow_bps_annual * (hold_days / 365.0)
    return commission_bps + spread_bps + slippage_bps + impact_bps + borrow_bps


def net_return_after_cost(gross_return: float, cost_bps: float) -> float:
    return gross_return - (cost_bps / 10_000.0)

