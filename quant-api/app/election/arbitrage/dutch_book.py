"""Dutch book arbitrage detection.

Buy all outcomes in a multi-candidate race when total ask < 1.0.
"""
from __future__ import annotations
import logging
from typing import Any

from app.election.arbitrage.cross_market import ArbSignal
from app.election.arbitrage.fee_model import polymarket_election_fee, kalshi_taker_fee

logger = logging.getLogger(__name__)

FEE_FUNCS = {
    "polymarket": polymarket_election_fee,
    "kalshi": kalshi_taker_fee,
    "predictit": lambda p: p * 0.15,
}


def detect_dutch_books(
    quotes_by_race_platform: dict[tuple[int, str], list[dict[str, Any]]],
) -> list[ArbSignal]:
    """Detect Dutch book opportunities.

    quotes_by_race_platform: {(race_id, platform): [quotes for each candidate/outcome]}
    Only works for multi-outcome markets (3+ candidates).
    """
    signals = []
    for (race_id, platform), quotes in quotes_by_race_platform.items():
        if len(quotes) < 2:
            continue

        total_ask = sum(q.get("yes_ask", 1.0) for q in quotes)
        fee_func = FEE_FUNCS.get(platform, lambda p: 0.01)
        total_fees = sum(fee_func(q.get("yes_ask", 0.5)) for q in quotes)

        gross_edge = 1.0 - total_ask
        net_edge = gross_edge - total_fees

        if net_edge <= 0:
            continue

        confidence = min(q.get("liquidity_score", 0.5) for q in quotes)

        signals.append(ArbSignal(
            arb_type="dutch_book",
            race_id=race_id,
            description=(
                f"Dutch book on {platform}: buy all {len(quotes)} outcomes "
                f"for total {total_ask:.3f}, guaranteed payout 1.0"
            ),
            gross_edge_pct=round(gross_edge * 100, 3),
            net_edge_pct=round(net_edge * 100, 3),
            buy_platform=platform,
            buy_contract_id=None,
            buy_price=total_ask,
            sell_platform=platform,
            sell_contract_id=None,
            sell_price=1.0,
            confidence=confidence,
        ))

    signals.sort(key=lambda s: s.net_edge_pct, reverse=True)
    return signals
