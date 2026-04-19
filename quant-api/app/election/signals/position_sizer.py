"""Position Sizer — Kelly criterion + portfolio construction.

Computes optimal position sizes for election prediction market trades
using fractional Kelly criterion with correlation adjustments.
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PositionRecommendation:
    """Single race position recommendation."""
    race_id: int | None
    state: str
    race_type: str
    platform: str
    direction: str              # "BUY_DEM" or "BUY_REP"
    current_market_prob: float
    estimated_true_prob: float
    edge_pp: float
    kelly_fraction: float
    adjusted_fraction: float
    notional_usd: float
    expected_pnl: float
    max_loss: float


@dataclass
class PortfolioRecommendation:
    """Portfolio-level recommendation across races."""
    as_of: datetime
    bankroll: float
    signal_strength: str
    n_positions: int
    total_notional: float
    total_expected_pnl: float
    total_max_loss: float
    expected_sharpe: float
    positions: list[PositionRecommendation] = field(default_factory=list)


def compute_kelly_fraction(
    estimated_prob: float,
    market_prob: float,
    fee_pct: float = 0.003,
    max_kelly: float = 0.25,
) -> float:
    """Compute Kelly criterion fraction for a binary bet.

    Kelly = (p * b - q) / b
    where p = estimated win prob, q = 1-p, b = payout odds

    Capped at max_kelly (default quarter-Kelly) for safety.
    """
    if estimated_prob <= 0 or estimated_prob >= 1:
        return 0.0
    if market_prob <= 0 or market_prob >= 1:
        return 0.0

    # Adjust for fees
    effective_cost = market_prob * (1 + fee_pct)
    if effective_cost >= 1.0:
        return 0.0

    # Payout odds: if you buy at market_prob, you get 1.0 on win
    # Net profit on win = 1.0 - effective_cost
    # Loss on lose = effective_cost
    b = (1.0 - effective_cost) / effective_cost  # odds ratio

    if b <= 0:
        return 0.0

    p = estimated_prob
    q = 1.0 - p

    kelly = (p * b - q) / b

    if kelly <= 0:
        return 0.0

    return min(kelly, max_kelly)


def build_portfolio(
    bias_signal: dict[str, Any],
    analog_forecasts: list[dict[str, Any]],
    bankroll: float = 10000.0,
    max_position_pct: float = 0.15,
    max_portfolio_pct: float = 0.60,
    correlation_factor: float = 0.7,
    kelly_divisor: float = 4.0,
    min_edge_pp: float = 10.0,
    platform: str = "polymarket",
) -> PortfolioRecommendation:
    """Build a portfolio of positions from bias signal + analog forecasts.

    Args:
        bias_signal: NarrativeBiasSignal-like dict with race-level data
        analog_forecasts: from get_mispricing_forecast()
        bankroll: total capital available
        max_position_pct: max fraction per race
        max_portfolio_pct: max total exposure fraction
        correlation_factor: estimated cross-race correlation (0-1)
        kelly_divisor: divide Kelly by this for safety (4 = quarter-Kelly)
        min_edge_pp: minimum edge in percentage points to take position
        platform: trading platform
    """
    signal_strength = bias_signal.get("signal_strength", "none")
    if signal_strength in ("none", "weak"):
        return PortfolioRecommendation(
            as_of=datetime.utcnow(),
            bankroll=bankroll,
            signal_strength=signal_strength,
            n_positions=0,
            total_notional=0.0,
            total_expected_pnl=0.0,
            total_max_loss=0.0,
            expected_sharpe=0.0,
        )

    # Build race-level data from bias_signal races + analog forecasts
    race_data: list[dict[str, Any]] = []
    forecast_map = {f["state"]: f for f in analog_forecasts}

    races = bias_signal.get("races", [])
    for race in races:
        state = race.get("state", "")
        market_prob = race.get("market_prob_dem", 0.5)
        polling_avg = race.get("polling_avg_dem")

        # Get analog forecast for this race
        forecast = forecast_map.get(state, {})
        analog_error = forecast.get("expected_error_pp", 0.0)

        # Estimate true probability
        if polling_avg is not None and polling_avg > 0:
            estimated_true = polling_avg
        elif analog_error > 0:
            # If no polling, use market + half the analog error
            estimated_true = min(0.95, market_prob + analog_error / 200.0)
        else:
            continue

        edge_pp = (estimated_true - market_prob) * 100

        if abs(edge_pp) < min_edge_pp:
            continue

        direction = "BUY_DEM" if edge_pp > 0 else "BUY_REP"
        buy_prob = market_prob if direction == "BUY_DEM" else (1.0 - market_prob)
        est_prob = estimated_true if direction == "BUY_DEM" else (1.0 - estimated_true)

        race_data.append({
            "state": state,
            "race_type": race.get("race_type", "senate"),
            "race_id": race.get("race_id"),
            "direction": direction,
            "market_prob": buy_prob,
            "estimated_true": est_prob,
            "edge_pp": abs(edge_pp),
        })

    if not race_data:
        return PortfolioRecommendation(
            as_of=datetime.utcnow(),
            bankroll=bankroll,
            signal_strength=signal_strength,
            n_positions=0,
            total_notional=0.0,
            total_expected_pnl=0.0,
            total_max_loss=0.0,
            expected_sharpe=0.0,
        )

    # Sort by edge descending
    race_data.sort(key=lambda r: r["edge_pp"], reverse=True)
    n_correlated = len(race_data)

    positions: list[PositionRecommendation] = []
    total_notional = 0.0
    max_total = max_portfolio_pct * bankroll

    for r in race_data:
        kelly = compute_kelly_fraction(r["estimated_true"], r["market_prob"])

        # Correlation adjustment: divide by sqrt(n_correlated) * correlation_factor
        corr_adj = math.sqrt(n_correlated * correlation_factor) if n_correlated > 1 else 1.0
        adjusted = kelly / kelly_divisor / corr_adj

        # Cap per position
        adjusted = min(adjusted, max_position_pct)

        notional = adjusted * bankroll

        # Check portfolio cap
        if total_notional + notional > max_total:
            notional = max(0.0, max_total - total_notional)
            adjusted = notional / bankroll if bankroll > 0 else 0.0

        if notional <= 0:
            break

        expected_pnl = (r["estimated_true"] - r["market_prob"]) * notional
        max_loss = r["market_prob"] * notional

        positions.append(PositionRecommendation(
            race_id=r.get("race_id"),
            state=r["state"],
            race_type=r["race_type"],
            platform=platform,
            direction=r["direction"],
            current_market_prob=r["market_prob"],
            estimated_true_prob=r["estimated_true"],
            edge_pp=r["edge_pp"],
            kelly_fraction=round(kelly, 4),
            adjusted_fraction=round(adjusted, 4),
            notional_usd=round(notional, 2),
            expected_pnl=round(expected_pnl, 2),
            max_loss=round(max_loss, 2),
        ))
        total_notional += notional

    total_expected = sum(p.expected_pnl for p in positions)
    total_max_loss = sum(p.max_loss for p in positions)
    # Rough Sharpe estimate: expected / std, assuming ~50% correlation
    std_estimate = math.sqrt(sum(p.max_loss**2 for p in positions) * correlation_factor) if positions else 1.0
    expected_sharpe = total_expected / std_estimate if std_estimate > 0 else 0.0

    return PortfolioRecommendation(
        as_of=datetime.utcnow(),
        bankroll=bankroll,
        signal_strength=signal_strength,
        n_positions=len(positions),
        total_notional=round(total_notional, 2),
        total_expected_pnl=round(total_expected, 2),
        total_max_loss=round(total_max_loss, 2),
        expected_sharpe=round(expected_sharpe, 3),
        positions=positions,
    )


def format_trade_plan(portfolio: PortfolioRecommendation) -> str:
    """Format portfolio as human-readable trade plan."""
    if not portfolio.positions:
        return f"No positions recommended (signal: {portfolio.signal_strength})"

    lines = [
        "=== ELECTION ALPHA TRADE PLAN ===",
        f"Signal: {portfolio.signal_strength.upper()} narrative bias",
        f"Bankroll: ${portfolio.bankroll:,.0f} | "
        f"Total exposure: ${portfolio.total_notional:,.0f} "
        f"({portfolio.total_notional / portfolio.bankroll * 100:.0f}%)",
        f"Expected PnL: ${portfolio.total_expected_pnl:,.0f} | "
        f"Max loss: ${portfolio.total_max_loss:,.0f} | "
        f"Sharpe: {portfolio.expected_sharpe:.2f}",
        "",
    ]

    for i, pos in enumerate(portfolio.positions, 1):
        lines.append(
            f"#{i} {pos.state} {pos.race_type} — {pos.direction} @ ${pos.current_market_prob:.2f} "
            f"(est true: ${pos.estimated_true_prob:.2f})"
        )
        lines.append(
            f"   Edge: {pos.edge_pp:.0f}pp | "
            f"Size: ${pos.notional_usd:,.0f} ({pos.adjusted_fraction * 100:.1f}%) | "
            f"E[PnL]: ${pos.expected_pnl:+,.0f}"
        )
        lines.append("")

    return "\n".join(lines)
