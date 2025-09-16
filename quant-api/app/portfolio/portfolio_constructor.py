"""Portfolio constructor — builds optimal portfolios for sub-$100K accounts.

Takes backtest signals + robustness scores + quality filters and constructs
a concrete portfolio with:
  - Position weights respecting all constraints
  - Rebalancing only on signal changes (not calendar-based)
  - Minimum 20-day holding period enforced
  - Expected Value as the ranking criterion
  - Small-portfolio advantages maximized
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.portfolio.risk_manager import PortfolioConstraints, compute_position_size, PositionSizeResult
from app.portfolio.robustness import RobustnessResult
from app.portfolio.quality_filter import ROICProfile
from app.portfolio.translation_edge import TranslationEdge

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CandidatePosition:
    """A candidate position that has passed all quality gates."""
    ticker: str
    commodity_type: str
    country: str
    market_cap: float
    avg_price: float
    avg_daily_volume: float

    # Signal quality
    predicted_return_pct: float        # model's OOS predicted 20d return
    robustness: RobustnessResult | None
    quality: ROICProfile | None
    translation: TranslationEdge | None

    # Composite scores
    ev_per_trade_pct: float            # statistical expected value
    conviction_score: float            # 0-100 combined score
    information_advantage_score: float # translation + coverage gap edge

    # Position sizing
    sizing: PositionSizeResult | None


@dataclass(slots=True)
class PortfolioSnapshot:
    """A complete portfolio recommendation."""
    timestamp: str
    total_capital: float
    invested_capital: float
    cash_reserve: float
    positions: list[CandidatePosition]
    num_positions: int
    # Portfolio-level metrics
    portfolio_ev_pct: float            # weighted avg EV per rebalance
    portfolio_robustness_score: float  # weighted avg robustness
    sector_exposures: dict[str, float] # commodity_type -> % of portfolio
    country_exposures: dict[str, float]
    avg_hold_target_days: int
    # Small portfolio edge
    pct_micro_cap: float               # % in micro/nano caps
    pct_foreign_listed: float          # % in non-US exchanges
    information_edge_score: float      # portfolio-level info asymmetry
    # Warnings
    warnings: list[str]


@dataclass(slots=True)
class RebalanceSignal:
    """Signal to add, close, or hold a position."""
    action: str                    # "BUY", "HOLD", "CLOSE"
    ticker: str
    reason: str
    position_dollars: float
    shares: int
    days_held: int
    target_hold_days: int
    # Optional enrichment fields for execution layer
    avg_price: float = 0.0
    conviction_score: float = 50.0
    ev_per_trade_pct: float = 0.0
    kelly_fraction: float = 0.05
    commodity_type: str = ""
    country: str = "US"
    avg_daily_volume: float = 50_000.0


# ────────────────────────────────────────────────────────────────────────────
# CONVICTION RANKING
# ────────────────────────────────────────────────────────────────────────────

def compute_conviction_score(
    robustness: RobustnessResult | None,
    quality: ROICProfile | None,
    translation: TranslationEdge | None,
    predicted_return_pct: float,
) -> float:
    """Composite conviction score (0-100) combining all signal dimensions.

    Weighting:
      40% — Statistical robustness (is the alpha real?)
      25% — ROIC quality (is this a good company at a good price?)
      15% — Predicted return magnitude (how much alpha?)
      20% — Information advantage (language/accounting/coverage edge)

    The philosophy: a statistically robust signal in a quality company
    with an information edge is worth MORE than a huge predicted return
    in a low-quality name with no edge.
    """
    score = 0.0

    # Robustness: 0-40 pts
    if robustness:
        score += robustness.robustness_score * 0.40
    else:
        score += 20.0  # neutral if no robustness data

    # Quality: 0-25 pts
    if quality:
        score += quality.quality_score * 0.25
    else:
        score += 12.5  # neutral

    # Predicted return: 0-15 pts (capped at 5% predicted return)
    return_score = min(15.0, max(0.0, predicted_return_pct / 5.0 * 15.0))
    score += return_score

    # Information advantage: 0-20 pts
    if translation:
        score += translation.total_asymmetry_score * 100.0 * 0.20
    else:
        score += 0.0  # no edge from translation

    return round(min(100.0, max(0.0, score)), 1)


# ────────────────────────────────────────────────────────────────────────────
# PORTFOLIO CONSTRUCTION
# ────────────────────────────────────────────────────────────────────────────

def construct_portfolio(
    candidates: list[CandidatePosition],
    constraints: PortfolioConstraints,
    existing_positions: list[str] | None = None,
    pairwise_correlations: dict[str, dict[str, float]] | None = None,
) -> PortfolioSnapshot:
    """Construct optimal portfolio from ranked candidates.

    Algorithm:
      1. Filter: only candidates that pass all quality gates
      2. Rank by conviction score (EV-weighted)
      3. Greedily add positions in rank order, respecting constraints
      4. Size each position using quarter-Kelly with adjustments
      5. Check portfolio-level risk limits

    This is NOT mean-variance optimization (which requires stable
    covariance estimation, unreliable for commodity micro-caps).
    Instead: greedy rank-based allocation with constraint enforcement.
    This is more robust for small universes and short histories.
    """
    from datetime import datetime, timezone

    existing = set(existing_positions or [])
    corr_map = pairwise_correlations or {}

    # Step 1: Filter to tradeable candidates
    tradeable = []
    for c in candidates:
        reasons = []
        if c.robustness and not c.robustness.overall_robust:
            reasons.append(f"robustness fail: {c.robustness.rejection_reasons}")
        if c.quality and not c.quality.overall_quality_pass:
            reasons.append("ROIC quality fail")
        if c.ev_per_trade_pct <= 0:
            reasons.append(f"negative EV ({c.ev_per_trade_pct:.3f}%)")
        if c.avg_daily_volume < constraints.min_avg_daily_volume:
            reasons.append(f"thin volume ({c.avg_daily_volume:.0f})")

        if not reasons:
            tradeable.append(c)
        else:
            logger.debug("Rejected %s: %s", c.ticker, "; ".join(reasons))

    # Step 2: Rank by conviction score descending
    tradeable.sort(key=lambda c: c.conviction_score, reverse=True)

    # Step 3: Greedy allocation
    selected: list[CandidatePosition] = []
    sector_exposure: dict[str, float] = {}
    country_exposure: dict[str, float] = {}
    selected_tickers: list[str] = list(existing)
    invested = 0.0
    warnings: list[str] = []

    for candidate in tradeable:
        if len(selected) >= constraints.max_positions:
            break

        # Position sizing
        ticker_corr = corr_map.get(candidate.ticker, {})
        sect_exp = sector_exposure.get(candidate.commodity_type, 0.0)
        ctry_exp = country_exposure.get(candidate.country, 0.0)

        sizing = compute_position_size(
            ticker=candidate.ticker,
            kelly_fraction=candidate.robustness.kelly_fraction if candidate.robustness else 0.05,
            robustness_score=candidate.robustness.robustness_score if candidate.robustness else 50.0,
            avg_daily_volume=candidate.avg_daily_volume,
            avg_price=candidate.avg_price,
            market_cap=candidate.market_cap,
            existing_sector_exposure_pct=sect_exp,
            existing_country_exposure_pct=ctry_exp,
            existing_position_tickers=selected_tickers,
            pairwise_correlations=ticker_corr,
            commodity_type=candidate.commodity_type,
            country=candidate.country,
            constraints=constraints,
        )

        if sizing.rejection_reason:
            warnings.append(f"{candidate.ticker}: {sizing.rejection_reason}")
            continue

        if invested + sizing.position_dollars > constraints.investable_capital:
            # Try a smaller position
            remaining = constraints.investable_capital - invested
            if remaining >= constraints.min_position_size:
                sizing = PositionSizeResult(
                    ticker=sizing.ticker,
                    raw_kelly_fraction=sizing.raw_kelly_fraction,
                    adjusted_kelly_fraction=sizing.adjusted_kelly_fraction,
                    position_dollars=remaining,
                    position_pct=round(remaining / constraints.investable_capital * 100, 2),
                    shares=int(remaining / candidate.avg_price) if candidate.avg_price > 0 else 0,
                    conviction_multiplier=sizing.conviction_multiplier,
                    liquidity_discount=sizing.liquidity_discount,
                    correlation_penalty=sizing.correlation_penalty,
                    small_cap_bonus=sizing.small_cap_bonus,
                    passes_min_size=True,
                    passes_max_size=True,
                    passes_volume_check=sizing.passes_volume_check,
                    rejection_reason=None,
                )
            else:
                break

        candidate.sizing = sizing
        selected.append(candidate)

        # Update tracking
        invested += sizing.position_dollars
        selected_tickers.append(candidate.ticker)
        pct = sizing.position_dollars / constraints.investable_capital
        sector_exposure[candidate.commodity_type] = sect_exp + pct
        country_exposure[candidate.country] = ctry_exp + pct

    # Step 4: Compute portfolio-level metrics
    total_ev = 0.0
    total_rob = 0.0
    weight_sum = 0.0
    micro_cap_dollars = 0.0
    foreign_dollars = 0.0
    info_edge_sum = 0.0

    for p in selected:
        w = p.sizing.position_dollars if p.sizing else 0.0
        weight_sum += w
        total_ev += p.ev_per_trade_pct * w
        total_rob += (p.robustness.robustness_score if p.robustness else 50.0) * w
        if p.market_cap < 500_000_000:
            micro_cap_dollars += w
        if p.translation and p.translation.total_asymmetry_score > 0.05:
            foreign_dollars += w
        info_edge_sum += p.information_advantage_score * w

    wtd_ev = total_ev / weight_sum if weight_sum > 0 else 0.0
    wtd_rob = total_rob / weight_sum if weight_sum > 0 else 0.0
    pct_micro = micro_cap_dollars / weight_sum * 100 if weight_sum > 0 else 0.0
    pct_foreign = foreign_dollars / weight_sum * 100 if weight_sum > 0 else 0.0
    info_edge = info_edge_sum / weight_sum if weight_sum > 0 else 0.0

    return PortfolioSnapshot(
        timestamp=datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
        total_capital=constraints.total_capital,
        invested_capital=round(invested, 2),
        cash_reserve=round(constraints.total_capital - invested, 2),
        positions=selected,
        num_positions=len(selected),
        portfolio_ev_pct=round(wtd_ev, 4),
        portfolio_robustness_score=round(wtd_rob, 1),
        sector_exposures={k: round(v * 100, 1) for k, v in sector_exposure.items()},
        country_exposures={k: round(v * 100, 1) for k, v in country_exposure.items()},
        avg_hold_target_days=constraints.target_hold_days,
        pct_micro_cap=round(pct_micro, 1),
        pct_foreign_listed=round(pct_foreign, 1),
        information_edge_score=round(info_edge, 3),
        warnings=warnings,
    )


# ────────────────────────────────────────────────────────────────────────────
# REBALANCING LOGIC — signal-driven, NOT calendar-driven
# ────────────────────────────────────────────────────────────────────────────

def generate_rebalance_signals(
    current_positions: dict[str, dict[str, Any]],
    new_candidates: list[CandidatePosition],
    constraints: PortfolioConstraints,
) -> list[RebalanceSignal]:
    """Generate rebalance signals based on signal changes, NOT calendar.

    Rules:
      1. HOLD: if position is < min_hold_days old, ALWAYS hold (no daytrading)
      2. CLOSE: if signal has degraded (turned negative) AND past min hold
      3. CLOSE: if position > max_hold_days and signal is weak
      4. BUY: if new high-conviction candidate and portfolio has room
      5. HOLD: everything else (fewer trades = lower cost drag)

    The goal: ~3-5 trades per month (each held 20-120 days)
    This is deep cyclical positioning, not tactical trading.
    """
    signals: list[RebalanceSignal] = []
    new_ticker_map = {c.ticker: c for c in new_candidates}

    for ticker, pos_info in current_positions.items():
        days_held = pos_info.get("days_held", 0)
        entry_dollars = pos_info.get("position_dollars", 0)
        entry_shares = pos_info.get("shares", 0)

        # RULE 1: Never close before min_hold_days
        if days_held < constraints.min_hold_days:
            signals.append(RebalanceSignal(
                action="HOLD",
                ticker=ticker,
                reason=f"Min hold period ({days_held}/{constraints.min_hold_days} days)",
                position_dollars=entry_dollars,
                shares=entry_shares,
                days_held=days_held,
                target_hold_days=constraints.target_hold_days,
            ))
            continue

        # Check if we still have a positive signal for this ticker
        new_cand = new_ticker_map.get(ticker)

        # RULE 2: Close if signal turned negative
        if new_cand is None or new_cand.ev_per_trade_pct <= 0:
            signals.append(RebalanceSignal(
                action="CLOSE",
                ticker=ticker,
                reason="Signal degraded (EV <= 0 or no longer in candidates)",
                position_dollars=entry_dollars,
                shares=entry_shares,
                days_held=days_held,
                target_hold_days=constraints.target_hold_days,
            ))
            continue

        # RULE 3: Close stale positions with weak signal
        if days_held > constraints.max_hold_days and new_cand.conviction_score < 50:
            signals.append(RebalanceSignal(
                action="CLOSE",
                ticker=ticker,
                reason=f"Stale position ({days_held}d) with weak conviction ({new_cand.conviction_score:.0f})",
                position_dollars=entry_dollars,
                shares=entry_shares,
                days_held=days_held,
                target_hold_days=constraints.target_hold_days,
            ))
            continue

        # RULE 5: Hold everything else
        signals.append(RebalanceSignal(
            action="HOLD",
            ticker=ticker,
            reason=f"Active signal (EV={new_cand.ev_per_trade_pct:.3f}%, conv={new_cand.conviction_score:.0f})",
            position_dollars=entry_dollars,
            shares=entry_shares,
            days_held=days_held,
            target_hold_days=constraints.target_hold_days,
        ))

    # RULE 4: New buys for candidates not yet in portfolio
    current_tickers = set(current_positions.keys())
    for cand in sorted(new_candidates, key=lambda c: c.conviction_score, reverse=True):
        if cand.ticker in current_tickers:
            continue
        if cand.ev_per_trade_pct <= 0:
            continue
        if cand.sizing and cand.sizing.rejection_reason:
            continue

        signals.append(RebalanceSignal(
            action="BUY",
            ticker=cand.ticker,
            reason=f"New high-conviction (score={cand.conviction_score:.0f}, EV={cand.ev_per_trade_pct:.3f}%)",
            position_dollars=cand.sizing.position_dollars if cand.sizing else 0,
            shares=cand.sizing.shares if cand.sizing else 0,
            days_held=0,
            target_hold_days=constraints.target_hold_days,
            avg_price=cand.avg_price,
            conviction_score=cand.conviction_score,
            ev_per_trade_pct=cand.ev_per_trade_pct,
            kelly_fraction=cand.sizing.adjusted_kelly_fraction if cand.sizing else 0.05,
            commodity_type=cand.commodity_type,
            country=cand.country,
            avg_daily_volume=cand.avg_daily_volume,
        ))

    return signals
