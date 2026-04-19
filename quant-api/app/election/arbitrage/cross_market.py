"""Cross-market arbitrage detection.

For each race+candidate, compare best bid on one platform vs best ask on another.
If bid_A > ask_B (after fees), there's an arb.

Direction-aware: normalizes all quotes to P(Dem wins) before comparing
across platforms, so "Will Dem win?" at 0.70 and "Will Rep win?" at 0.70
are correctly interpreted as opposite positions (0.70 vs 0.30 P(Dem)).
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Any

from app.election.arbitrage.fee_model import total_arb_fee
from app.election.mappings.direction_detector import detect_direction, normalize_price

logger = logging.getLogger(__name__)


@dataclass
class ArbSignal:
    arb_type: str
    race_id: int
    description: str
    gross_edge_pct: float
    net_edge_pct: float
    buy_platform: str
    buy_contract_id: int | None
    buy_price: float
    sell_platform: str
    sell_contract_id: int | None
    sell_price: float
    confidence: float


def detect_cross_market_arbs(
    quotes_by_race: dict[int, list[dict[str, Any]]],
) -> list[ArbSignal]:
    """Detect cross-market arbitrage opportunities.

    quotes_by_race: {race_id: [{platform, contract_id, yes_bid, yes_ask, ...}, ...]}
    """
    signals = []
    for race_id, quotes in quotes_by_race.items():
        if len(quotes) < 2:
            continue
        # Compare every pair of platforms
        for i, q_a in enumerate(quotes):
            for q_b in quotes[i + 1:]:
                # Check A.bid vs B.ask
                signal = _check_pair(race_id, q_a, q_b)
                if signal:
                    signals.append(signal)
                # Check B.bid vs A.ask
                signal = _check_pair(race_id, q_b, q_a)
                if signal:
                    signals.append(signal)
    signals.sort(key=lambda s: s.net_edge_pct, reverse=True)
    return signals


def _check_pair(race_id: int, seller: dict, buyer: dict) -> ArbSignal | None:
    """Check if we can sell on seller's platform and buy on buyer's platform.

    Direction-aware: normalizes both quotes to P(Dem wins) before comparing.
    Skips quotes where direction confidence < 0.5 (ambiguous YES/NO semantics).
    """
    # Detect direction for both contracts
    seller_dir = detect_direction(seller.get("platform_question", ""))
    buyer_dir = detect_direction(buyer.get("platform_question", ""))

    # Skip if either direction is unknown or low confidence
    if seller_dir.confidence < 0.5 or buyer_dir.confidence < 0.5:
        return None

    # Normalize to P(Dem wins) before comparing
    sell_bid = normalize_price(seller.get("yes_bid", 0.0), seller_dir.yes_party)
    buy_ask = normalize_price(buyer.get("yes_ask", 0.0), buyer_dir.yes_party)

    if sell_bid <= 0 or buy_ask <= 0:
        return None

    gross_edge = sell_bid - buy_ask
    if gross_edge <= 0:
        return None

    fees = total_arb_fee(
        buy_platform=buyer["platform"], buy_price=buy_ask,
        sell_platform=seller["platform"], sell_price=1.0 - sell_bid,
    )
    net_edge = gross_edge - fees

    if net_edge <= 0:
        return None

    confidence = min(1.0, (
        min(seller.get("liquidity_score", 0.5), buyer.get("liquidity_score", 0.5))
    ))

    return ArbSignal(
        arb_type="cross_market",
        race_id=race_id,
        description=(
            f"Buy (norm) on {buyer['platform']} at {buy_ask:.3f}, "
            f"sell (norm) on {seller['platform']} at {sell_bid:.3f} "
            f"[{seller_dir.yes_party}/{buyer_dir.yes_party}]"
        ),
        gross_edge_pct=round(gross_edge * 100, 3),
        net_edge_pct=round(net_edge * 100, 3),
        buy_platform=buyer["platform"],
        buy_contract_id=buyer.get("contract_id"),
        buy_price=buy_ask,
        sell_platform=seller["platform"],
        sell_contract_id=seller.get("contract_id"),
        sell_price=sell_bid,
        confidence=confidence,
    )
