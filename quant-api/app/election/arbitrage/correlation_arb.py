"""Correlation arbitrage detection.

Exploits markets that price correlated events as if independent.
"""
from __future__ import annotations
import logging
from typing import Any

from app.election.arbitrage.cross_market import ArbSignal
from app.election.arbitrage.fee_model import total_arb_fee

logger = logging.getLogger(__name__)

# Known correlation pairs: (event_a, event_b, estimated_rho)
CORRELATION_PAIRS = [
    ("senate_2026_dem", "house_2026_dem", 0.6),
    ("senate_pa_2026", "senate_wi_2026", 0.45),
    ("senate_pa_2026", "senate_mi_2026", 0.50),
    ("senate_ga_2026", "senate_nc_2026", 0.55),
    ("senate_az_2026", "senate_nv_2026", 0.50),
    ("pres_2028_democrat", "senate_2026_dem", 0.35),
]


def detect_correlation_arbs(
    market_probs: dict[str, dict[str, Any]],
    min_edge_pct: float = 2.0,
) -> list[ArbSignal]:
    """Detect correlation arbitrage between related markets.

    market_probs: {event_id: {prob, platform, race_id, ...}}
    """
    signals = []
    for event_a, event_b, rho in CORRELATION_PAIRS:
        prob_a_data = market_probs.get(event_a)
        prob_b_data = market_probs.get(event_b)
        if not prob_a_data or not prob_b_data:
            continue

        p_a = prob_a_data.get("prob", 0.5)
        p_b = prob_b_data.get("prob", 0.5)

        # Independent: P(A∩B) = P(A)*P(B)
        p_joint_independent = p_a * p_b

        # Correlated: P(A∩B) = P(A)*P(B) + rho*sqrt(P(A)*(1-P(A))*P(B)*(1-P(B)))
        import math
        cov_term = rho * math.sqrt(p_a * (1 - p_a) * p_b * (1 - p_b))
        p_joint_correlated = p_joint_independent + cov_term
        p_joint_correlated = max(0.01, min(0.99, p_joint_correlated))

        edge = abs(p_joint_correlated - p_joint_independent)
        if edge * 100 < min_edge_pct:
            continue

        platform = prob_a_data.get("platform", "unknown")
        fee = total_arb_fee(platform, p_a, platform, p_b)
        net_edge = edge - fee

        if net_edge <= 0:
            continue

        signals.append(ArbSignal(
            arb_type="correlation",
            race_id=prob_a_data.get("race_id", 0),
            description=(
                f"Correlation arb: {event_a}×{event_b} "
                f"independent={p_joint_independent:.3f}, "
                f"correlated={p_joint_correlated:.3f}, rho={rho}"
            ),
            gross_edge_pct=round(edge * 100, 3),
            net_edge_pct=round(net_edge * 100, 3),
            buy_platform=platform,
            buy_contract_id=None,
            buy_price=p_joint_independent,
            sell_platform="model",
            sell_contract_id=None,
            sell_price=p_joint_correlated,
            confidence=min(0.7, (prob_a_data.get("liquidity_score", 0.5) + prob_b_data.get("liquidity_score", 0.5)) / 2),
        ))

    signals.sort(key=lambda s: s.net_edge_pct, reverse=True)
    return signals
