from __future__ import annotations

from app.modeling.cost_model import estimate_total_cost_bps, net_return_after_cost


def test_cost_model_reduces_gross_return():
    cost = estimate_total_cost_bps(hold_days=60)
    gross = 0.10
    net = net_return_after_cost(gross, cost)
    assert cost > 0
    assert net < gross

