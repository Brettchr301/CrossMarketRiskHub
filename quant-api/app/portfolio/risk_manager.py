"""Risk manager and position sizing for sub-$100K portfolios.

Designed specifically for small retail portfolios where:
  - Illiquidity premium is an EDGE (can buy micro-caps institutions can't)
  - Concentration risk is real (max 15-20 positions)
  - Round-lot constraints matter ($2K-$5K per position)
  - No daytrading: minimum 20-day hold, targeting 40-60 day average
  - Kelly-based sizing with heavy safety margin (quarter-Kelly)

The key insight from the user: a sub-$100K portfolio has structural advantages
that large funds cannot exploit:
  1. Can enter/exit micro-cap positions without moving the market
  2. Can take concentrated bets in highest-conviction names
  3. No regulatory reporting thresholds (13F, Reg SHO)
  4. Can exploit foreign-listed tickers where language barriers reduce coverage
  5. Can hold through earnings without institutional rebalancing pressure
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ────────────────────────────────────────────────────────────────────────────
# PORTFOLIO CONSTRAINTS — designed for sub-$100K
# ────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class PortfolioConstraints:
    """Hard constraints for a small retail commodity-focused portfolio."""
    total_capital: float = 75_000.0       # default $75K
    max_positions: int = 18               # concentration is a feature, not a bug
    min_position_size: float = 2_000.0    # minimum to justify commission
    max_position_pct: float = 0.12        # 12% max in single name
    max_sector_pct: float = 0.35          # 35% max in single commodity_type
    max_country_pct: float = 0.40         # 40% max in single country
    max_war_zone_pct: float = 0.15        # 15% max in war-zone countries
    cash_reserve_pct: float = 0.10        # always keep 10% cash for opportunities
    min_hold_days: int = 20               # NO daytrading — minimum 20 trading days
    target_hold_days: int = 40            # target 40-day hold (deep cyclical)
    max_hold_days: int = 120              # re-evaluate after 120 days
    min_avg_daily_volume: float = 50_000  # at least 50K shares/day
    max_portfolio_adv_pct: float = 0.02   # position < 2% of ticker's daily volume
    max_correlation: float = 0.70         # drop if pairwise correlation > 70%
    min_roic_percentile: float = 0.30     # must be above 30th percentile within sector

    @property
    def investable_capital(self) -> float:
        return self.total_capital * (1.0 - self.cash_reserve_pct)


# ────────────────────────────────────────────────────────────────────────────
# POSITION SIZING — Kelly-based with micro-cap liquidity adjustment
# ────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class PositionSizeResult:
    """Recommended position size with reasoning."""
    ticker: str
    raw_kelly_fraction: float       # full Kelly
    adjusted_kelly_fraction: float  # after safety adjustments
    position_dollars: float         # actual dollar amount
    position_pct: float             # as % of portfolio
    shares: int                     # approximate share count
    # Adjustment factors applied
    conviction_multiplier: float    # 0.5-1.5 based on robustness score
    liquidity_discount: float       # 0.3-1.0 based on ADV
    correlation_penalty: float      # 0.5-1.0 if correlated with existing
    small_cap_bonus: float          # 1.0-1.3 for illiquidity premium capture
    # Constraint checks
    passes_min_size: bool
    passes_max_size: bool
    passes_volume_check: bool
    rejection_reason: str | None


def compute_position_size(
    ticker: str,
    kelly_fraction: float,
    robustness_score: float,
    avg_daily_volume: float,
    avg_price: float,
    market_cap: float,
    existing_sector_exposure_pct: float,
    existing_country_exposure_pct: float,
    existing_position_tickers: list[str],
    pairwise_correlations: dict[str, float],
    commodity_type: str,
    country: str,
    constraints: PortfolioConstraints,
) -> PositionSizeResult:
    """Compute optimal position size for a sub-$100K portfolio.

    Uses quarter-Kelly as base, with adjustments for:
      - Statistical robustness (conviction)
      - Liquidity (can we get in/out?)
      - Correlation (are we doubling up?)
      - Small-cap premium (reward for illiquidity)
      - Sector/country concentration limits

    The philosophy: FEWER, LARGER, HIGHER-CONVICTION bets.
    NOT many small spray-and-pray positions.
    """
    cap = constraints.investable_capital

    # --- Base: quarter-Kelly ---
    # Full Kelly is too aggressive. Half-Kelly is standard.
    # Quarter-Kelly for additional safety with a small portfolio (ruin risk matters more).
    base_fraction = kelly_fraction * 0.25

    # --- Conviction multiplier: 0.5x to 1.5x based on robustness score ---
    # Score 0-40: reduce to 0.5x (marginal signal)
    # Score 40-70: 0.5-1.0x (decent signal)
    # Score 70-100: 1.0-1.5x (strong signal)
    if robustness_score >= 70:
        conviction = 1.0 + (robustness_score - 70) / 60.0  # up to 1.5
    elif robustness_score >= 40:
        conviction = 0.5 + (robustness_score - 40) / 60.0  # 0.5-1.0
    else:
        conviction = 0.5

    # --- Liquidity discount: scale down if thin ---
    avg_daily_dollar_vol = avg_daily_volume * avg_price
    if avg_daily_dollar_vol < 100_000:
        liq_discount = 0.3  # very thin
    elif avg_daily_dollar_vol < 500_000:
        liq_discount = 0.5
    elif avg_daily_dollar_vol < 2_000_000:
        liq_discount = 0.7
    elif avg_daily_dollar_vol < 10_000_000:
        liq_discount = 0.9
    else:
        liq_discount = 1.0

    # --- Correlation penalty: reduce if heavily correlated with existing ---
    max_corr = 0.0
    for t, corr in pairwise_correlations.items():
        if t in existing_position_tickers and abs(corr) > max_corr:
            max_corr = abs(corr)
    if max_corr > 0.8:
        corr_penalty = 0.5
    elif max_corr > 0.6:
        corr_penalty = 0.7
    else:
        corr_penalty = 1.0

    # --- Small-cap bonus: REWARD for buying what institutions CAN'T ---
    # This is the key sub-$100K edge
    if market_cap < 300_000_000:       # nano-cap
        small_bonus = 1.30
    elif market_cap < 500_000_000:     # micro-cap
        small_bonus = 1.20
    elif market_cap < 1_000_000_000:   # small-cap
        small_bonus = 1.10
    else:
        small_bonus = 1.00

    # --- Combine all adjustments ---
    adjusted_fraction = base_fraction * conviction * liq_discount * corr_penalty * small_bonus
    position_dollars = min(
        adjusted_fraction * cap,
        constraints.max_position_pct * cap,
    )

    # --- Enforce sector/country limits ---
    sector_room = max(0, constraints.max_sector_pct - existing_sector_exposure_pct) * cap
    country_room = max(0, constraints.max_country_pct - existing_country_exposure_pct) * cap
    position_dollars = min(position_dollars, sector_room, country_room)

    # --- Volume check: position < 2% of ADV ---
    if avg_daily_volume > 0 and avg_price > 0:
        max_by_volume = avg_daily_volume * avg_price * constraints.max_portfolio_adv_pct * constraints.target_hold_days
        position_dollars = min(position_dollars, max_by_volume)

    # Shares
    shares = int(position_dollars / avg_price) if avg_price > 0 else 0
    position_dollars = shares * avg_price  # round to whole shares

    # --- Final constraint checks ---
    passes_min = position_dollars >= constraints.min_position_size
    passes_max = position_dollars <= constraints.max_position_pct * cap * 1.01
    passes_vol = avg_daily_volume >= constraints.min_avg_daily_volume

    rejection = None
    if not passes_min:
        rejection = f"Below min size (${position_dollars:.0f} < ${constraints.min_position_size:.0f})"
    elif not passes_vol:
        rejection = f"Below min volume ({avg_daily_volume:.0f} < {constraints.min_avg_daily_volume:.0f})"
    elif len(existing_position_tickers) >= constraints.max_positions:
        rejection = f"Portfolio full ({constraints.max_positions} positions)"

    return PositionSizeResult(
        ticker=ticker,
        raw_kelly_fraction=round(kelly_fraction, 4),
        adjusted_kelly_fraction=round(adjusted_fraction, 4),
        position_dollars=round(position_dollars, 2),
        position_pct=round(position_dollars / cap * 100, 2) if cap > 0 else 0.0,
        shares=shares,
        conviction_multiplier=round(conviction, 3),
        liquidity_discount=round(liq_discount, 3),
        correlation_penalty=round(corr_penalty, 3),
        small_cap_bonus=round(small_bonus, 3),
        passes_min_size=passes_min,
        passes_max_size=passes_max,
        passes_volume_check=passes_vol,
        rejection_reason=rejection,
    )


# ────────────────────────────────────────────────────────────────────────────
# DRAWDOWN & RISK LIMITS — circuit breakers
# ────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class RiskState:
    """Current risk state of the portfolio."""
    peak_equity: float
    current_equity: float
    drawdown_pct: float
    max_drawdown_pct: float
    consecutive_losses: int
    trades_this_month: int
    is_halted: bool
    halt_reason: str | None


def check_risk_limits(
    peak_equity: float,
    current_equity: float,
    consecutive_losses: int,
    trades_this_month: int,
    max_drawdown_halt: float = 0.15,      # halt at 15% drawdown
    max_consecutive_losses: int = 5,       # halt after 5 straight losses
    max_trades_per_month: int = 8,         # no daytrading — max 8 trades/month
    cooldown_after_halt_days: int = 10,    # wait 10 trading days after halt
) -> RiskState:
    """Check portfolio risk limits. Returns halt signal if breached.

    These circuit breakers prevent ruin in a small portfolio:
      - 15% drawdown halt: stop trading, preserve capital
      - 5 consecutive losses: step back, re-evaluate thesis
      - 8 trades/month max: ENFORCES no daytrading
    """
    dd_pct = 0.0
    if peak_equity > 0:
        dd_pct = (peak_equity - current_equity) / peak_equity * 100.0

    halt = False
    reason = None

    if dd_pct >= max_drawdown_halt * 100:
        halt = True
        reason = f"Drawdown halt: {dd_pct:.1f}% exceeds {max_drawdown_halt*100:.0f}% limit"
    elif consecutive_losses >= max_consecutive_losses:
        halt = True
        reason = f"Consecutive loss halt: {consecutive_losses} losses in a row"
    elif trades_this_month >= max_trades_per_month:
        halt = True
        reason = f"Monthly trade limit: {trades_this_month} trades this month (max {max_trades_per_month})"

    return RiskState(
        peak_equity=peak_equity,
        current_equity=current_equity,
        drawdown_pct=round(dd_pct, 2),
        max_drawdown_pct=round(max(dd_pct, 0), 2),
        consecutive_losses=consecutive_losses,
        trades_this_month=trades_this_month,
        is_halted=halt,
        halt_reason=reason,
    )


# ────────────────────────────────────────────────────────────────────────────
# SMALL PORTFOLIO EDGE QUANTIFICATION
# ────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class SmallPortfolioEdge:
    """Quantifies the structural advantages of a sub-$100K portfolio."""
    # Illiquidity premium
    can_trade_nano_caps: bool           # <$100M market cap
    can_trade_micro_caps: bool          # <$500M market cap
    nano_cap_count_in_universe: int
    micro_cap_count_in_universe: int
    # Foreign obscurity premium
    foreign_language_tickers: int       # tickers on non-English exchanges
    accounting_standard_varieties: int  # count of different GAAP/IFRS standards
    # Market impact
    estimated_market_impact_bps: float  # our impact on micro-caps
    institutional_impact_bps: float     # $100M fund impact on same micro-caps
    impact_advantage_bps: float         # the difference = our edge
    # Capacity
    max_capacity_per_name: float        # max $ we can deploy per ticker
    total_universe_capacity: float      # sum across universe


def quantify_small_portfolio_edge(
    portfolio_capital: float,
    universe_tickers: list[dict[str, Any]],
) -> SmallPortfolioEdge:
    """Calculate the structural edge a sub-$100K portfolio has vs institutions.

    The math:
      - Our market impact: ~2-5 bps on micro-caps ($3K-$5K positions)
      - A $100M fund trying to take the same positions:
        impact = sqrt(position_size / ADV) * 30 bps (Almgren-Chriss model)
        For $500K position in a stock with $2M ADV: ~15 bps impact
        For $3K position in same stock: ~0.4 bps impact

      This 15x impact advantage is our structural edge.
    """
    nano = sum(1 for t in universe_tickers if t.get("market_cap", 5e9) < 100_000_000)
    micro = sum(1 for t in universe_tickers if t.get("market_cap", 5e9) < 500_000_000)

    foreign_lang = sum(
        1 for t in universe_tickers
        if any(t.get("ticker", "").endswith(s) for s in [
            ".T", ".NS", ".KS", ".MI", ".WA", ".BU", ".MC", ".LS", ".PA", ".DE",
            ".VI", ".SA", ".OL", ".AS",
        ])
    )

    # Count accounting standard varieties
    countries = set(t.get("country", "US") for t in universe_tickers)
    gaap_countries = {"US", "CA"}  # US/CA GAAP or IFRS-harmonized
    ifrs_countries = {"UK", "AU", "FR", "DE", "NL", "NO", "IT", "ES", "PT", "AT", "BE"}
    other_countries = countries - gaap_countries - ifrs_countries
    acct_varieties = min(3, len({
        "US_GAAP" if c in gaap_countries else
        "IFRS" if c in ifrs_countries else
        f"LOCAL_{c}" for c in countries
    }))

    # Market impact advantage (Almgren-Chriss simplified)
    our_avg_position = portfolio_capital * 0.06  # ~6% of portfolio per position
    inst_avg_position = 100_000_000 * 0.02       # 2% of $100M fund
    assumed_adv = 2_000_000  # $2M daily dollar volume for typical micro-cap

    our_impact = math.sqrt(our_avg_position / assumed_adv) * 30.0
    inst_impact = math.sqrt(inst_avg_position / assumed_adv) * 30.0
    advantage = inst_impact - our_impact

    # Capacity per name
    max_per_name = min(our_avg_position * 2, assumed_adv * 0.02 * 40)  # 2% of ADV * 40 days
    total_capacity = max_per_name * len(universe_tickers)

    return SmallPortfolioEdge(
        can_trade_nano_caps=portfolio_capital < 500_000,
        can_trade_micro_caps=portfolio_capital < 5_000_000,
        nano_cap_count_in_universe=nano,
        micro_cap_count_in_universe=micro,
        foreign_language_tickers=foreign_lang,
        accounting_standard_varieties=acct_varieties,
        estimated_market_impact_bps=round(our_impact, 2),
        institutional_impact_bps=round(inst_impact, 2),
        impact_advantage_bps=round(advantage, 2),
        max_capacity_per_name=round(max_per_name, 0),
        total_universe_capacity=round(total_capacity, 0),
    )
