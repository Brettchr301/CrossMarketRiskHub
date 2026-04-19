"""Parlay decomposition arbitrage.

Compare aggregate market (e.g. "Dems win Senate") to implied probability
from individual race markets, accounting for correlation.
"""
from __future__ import annotations
import logging
from typing import Any

import numpy as np

from app.election.arbitrage.cross_market import ArbSignal
from app.election.arbitrage.fee_model import total_arb_fee

logger = logging.getLogger(__name__)


def detect_parlay_arbs(
    aggregate_quotes: dict[str, dict[str, Any]],
    component_probs: dict[str, list[float]],
    correlation_adj: float = 0.15,
    min_edge_pct: float = 2.0,
) -> list[ArbSignal]:
    """Detect parlay decomposition arbitrage.

    aggregate_quotes: {"senate_2026_dem": {platform, yes_bid, yes_ask, ...}}
    component_probs: {"senate_2026_dem": [prob_seat_1, prob_seat_2, ...]}
    correlation_adj: positive correlation adjustment (seats move together)
    """
    signals = []
    for event_id, agg_quote in aggregate_quotes.items():
        component = component_probs.get(event_id)
        if not component or len(component) < 3:
            continue

        # Independent probability (product of individual seat wins)
        independent_prob = float(np.prod(component))

        # Adjust for positive correlation (seats in same party move together)
        # Simple adjustment: blend independent with average
        avg_prob = float(np.mean(component))
        correlated_prob = (1.0 - correlation_adj) * independent_prob + correlation_adj * avg_prob
        correlated_prob = max(0.01, min(0.99, correlated_prob))

        market_mid = (agg_quote.get("yes_bid", 0.5) + agg_quote.get("yes_ask", 0.5)) / 2
        edge = abs(correlated_prob - market_mid)

        if edge * 100 < min_edge_pct:
            continue

        # Direction: if model says higher prob than market, buy; else sell
        if correlated_prob > market_mid:
            direction = "buy"
            buy_price = agg_quote.get("yes_ask", market_mid)
            sell_price = correlated_prob
        else:
            direction = "sell"
            buy_price = 1.0 - agg_quote.get("yes_bid", market_mid)
            sell_price = 1.0 - correlated_prob

        platform = agg_quote.get("platform", "unknown")
        fee = total_arb_fee(platform, buy_price, platform, 1.0 - buy_price)
        net_edge = edge - fee

        if net_edge <= 0:
            continue

        signals.append(ArbSignal(
            arb_type="parlay",
            race_id=agg_quote.get("race_id", 0),
            description=(
                f"Parlay decomposition: {event_id} market={market_mid:.3f}, "
                f"model={correlated_prob:.3f}, direction={direction}"
            ),
            gross_edge_pct=round(edge * 100, 3),
            net_edge_pct=round(net_edge * 100, 3),
            buy_platform=platform,
            buy_contract_id=agg_quote.get("contract_id"),
            buy_price=buy_price,
            sell_platform="model",
            sell_contract_id=None,
            sell_price=sell_price,
            confidence=min(0.8, agg_quote.get("liquidity_score", 0.5)),
        ))

    signals.sort(key=lambda s: s.net_edge_pct, reverse=True)
    return signals
