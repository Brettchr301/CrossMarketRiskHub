"""Fee models for prediction market platforms.

Each function returns the fee as a fraction of $1 notional.
"""
from __future__ import annotations


def polymarket_taker_fee(price: float) -> float:
    """Polymarket taker fee: p * (1-p) * 0.02, capped at 1.75%.
    For US election markets, flat 0.30% taker (as of 2026).
    """
    return min(0.0175, price * (1.0 - price) * 0.02)


def polymarket_election_fee(price: float) -> float:
    """Polymarket US DCM election markets: flat 0.30% taker."""
    return 0.003


def kalshi_taker_fee(price: float) -> float:
    """Kalshi taker fee: 0.07 * price * (1 - price), capped.
    Price should be in 0-1 range.
    """
    return min(0.07 * price * (1.0 - price), 0.0175)


def predictit_fee(profit: float) -> float:
    """PredictIt: 10% withdrawal fee on all revenue + 5% on profit.
    Only charged on winning side.
    """
    if profit <= 0:
        return 0.0
    return profit * 0.10 + profit * 0.05


def predictit_roundtrip_fee(buy_price: float) -> float:
    """Estimated PredictIt round-trip fee assuming win.
    Profit = 1.0 - buy_price. Fee = 15% of profit.
    """
    profit = max(0.0, 1.0 - buy_price)
    return profit * 0.15


def total_arb_fee(
    buy_platform: str, buy_price: float,
    sell_platform: str, sell_price: float,
) -> float:
    """Calculate total fees for a cross-market arbitrage trade.
    buy_price: price to buy YES on buy_platform
    sell_price: price to buy NO on sell_platform (= 1 - YES price there)
    """
    fee_funcs = {
        "polymarket": polymarket_election_fee,
        "kalshi": kalshi_taker_fee,
        "predictit": predictit_roundtrip_fee,
        "metaculus": lambda p: 0.0,  # no money
    }
    buy_fee = fee_funcs.get(buy_platform, lambda p: 0.01)(buy_price)
    sell_fee = fee_funcs.get(sell_platform, lambda p: 0.01)(sell_price)
    return buy_fee + sell_fee
