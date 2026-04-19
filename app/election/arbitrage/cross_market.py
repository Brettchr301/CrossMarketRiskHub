from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from app.election.mappings.direction_detector import detect_direction, normalize_price


@dataclass
class ArbSignal:
    race_id: int
    seller_platform: str
    buyer_platform: str
    net_edge: float
    buy_price: float
    sell_price: float


def _check_pair(race_id: int, seller: dict, buyer: dict) -> Optional[ArbSignal]:
    # Detect direction for both contracts
    seller_dir = detect_direction(seller.get("platform_question", ""))
    buyer_dir = detect_direction(buyer.get("platform_question", ""))
    
    # Skip if either direction is unknown or low confidence
    if seller_dir.confidence < 0.5 or buyer_dir.confidence < 0.5:
        return None
    
    # Normalize both to P(Dem wins) before comparing
    sell_bid = normalize_price(seller.get("yes_bid", 0.0), seller_dir.yes_party)
    buy_ask = normalize_price(buyer.get("yes_ask", 0.0), buyer_dir.yes_party)
    
    # Check if we can sell on seller platform (higher normalized bid)
    # and buy on buyer platform (lower normalized ask)
    if sell_bid > buy_ask and sell_bid > 0 and buy_ask < 1:
        net_edge = sell_bid - buy_ask
        return ArbSignal(
            race_id=race_id,
            seller_platform=seller.get("platform", ""),
            buyer_platform=buyer.get("platform", ""),
            net_edge=net_edge,
            buy_price=buy_ask,
            sell_price=sell_bid
        )
    
    return None


def detect_cross_market_arbs(quotes_by_race: Dict[int, List[Dict[str, Any]]]) -> List[ArbSignal]:
    """
    Detect cross-market arbitrage opportunities across platforms.
    
    Args:
        quotes_by_race: dict mapping race_id -> list of quote dicts
                        Each quote dict must have keys:
                        - platform
                        - platform_question
                        - yes_bid
                        - yes_ask
    
    Returns:
        List of ArbSignal objects sorted by descending net_edge
    """
    all_signals = []
    
    for race_id, quotes in quotes_by_race.items():
        # Sort quotes by platform name for consistent ordering
        quotes = sorted(quotes, key=lambda x: x.get("platform", ""))
        
        # Compare all pairs (naive O(n^2) but n is small)
        for i in range(len(quotes)):
            for j in range(i + 1, len(quotes)):
                # Try both directions
                signal = _check_pair(race_id, quotes[i], quotes[j])
                if signal:
                    all_signals.append(signal)
                
                signal = _check_pair(race_id, quotes[j], quotes[i])
                if signal:
                    all_signals.append(signal)
    
    # Sort by best opportunity first
    return sorted(all_signals, key=lambda x: x.net_edge, reverse=True)
