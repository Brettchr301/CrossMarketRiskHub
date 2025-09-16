from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RiskLimits:
    max_single_name_weight: float = 0.12
    max_sector_weight: float = 0.35
    max_event_cluster: float = 0.45
    max_drawdown_threshold: float = 0.2


def apply_signal_risk_overrides(
    score: float,
    expected_return_net_cost: float,
    downside_p05: float,
    options_mismatch: float,
    limits: RiskLimits | None = None,
) -> tuple[float, list[str]]:
    limits = limits or RiskLimits()
    flags: list[str] = []
    adjusted_score = score

    if expected_return_net_cost < 0.03:
        flags.append("edge_below_threshold")
        adjusted_score -= 10.0
    if downside_p05 < -limits.max_drawdown_threshold:
        flags.append("tail_drawdown_risk")
        adjusted_score -= 8.0
    if options_mismatch > 0.12:
        flags.append("options_model_divergence")
        adjusted_score -= 5.0
    if adjusted_score < 0:
        flags.append("no_trade_mode")
    return adjusted_score, flags

