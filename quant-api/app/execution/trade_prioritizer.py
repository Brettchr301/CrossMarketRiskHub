"""Trade Prioritization Engine — EV-ranked trade allocation.

Takes raw RebalanceSignal objects from the decision engine and:
  1. Calculates Expected Value per trade (win_prob × win - loss_prob × loss)
  2. Ranks by risk-adjusted EV (correlation penalty, liquidity factor)
  3. Allocates cash via greedy knapsack (SELLS FIRST, then BUYs)
  4. Handles partial fills and minimum sizes
  5. Checks share affordability (raw share price × qty vs. available cash)
  6. Produces a prioritized TradePlan with full constraint headroom data

Critical Rules:
  - SELLS/CLOSES before BUYs (frees cash)
  - T+1 settlement: cash from sells not available for buys today
  - Never exceed ANY constraint even if approved
  - Never go below 5% emergency cash buffer
  - Check that raw share cost is affordable (no fractional shares for most tickers)
"""
from __future__ import annotations

import dataclasses
import json
import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

from app.execution.ib_sync import IBPortfolioState, IBPosition
from app.execution.db import get_connection, audit_log
from app.portfolio.risk_manager import PortfolioConstraints

logger = logging.getLogger(__name__)

# Tickers known to support fractional shares on IB
# (most micro-caps do NOT support fractional shares)
FRACTIONAL_ELIGIBLE = {
    "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "NVDA",
    "JPM", "V", "JNJ", "WMT", "PG", "MA", "HD", "DIS", "BAC",
    # Most commodity micro-caps do NOT support fractional shares at IB
}


# ────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class PlannedTrade:
    """A single trade in the prioritized plan."""
    rank: int
    ticker: str
    action: str                     # BUY, CLOSE, REDUCE
    shares: int
    estimated_price: float
    estimated_cost: float           # total dollar cost (shares × price)
    expected_value_pct: float
    conviction_score: float
    kelly_fraction: float
    risk_flags: list[str] = field(default_factory=list)
    constraint_headroom: dict[str, Any] = field(default_factory=dict)
    depends_on_trade_id: str | None = None   # T+1 dependency
    status: str = "PENDING"
    trade_id: str = ""

    def __post_init__(self):
        if not self.trade_id:
            self.trade_id = f"trd_{uuid.uuid4().hex[:12]}"


@dataclass(slots=True)
class RejectedTrade:
    """A trade that was considered but rejected."""
    ticker: str
    action: str
    reason: str
    would_need: float               # how much cash it would need


@dataclass(slots=True)
class TradePlan:
    """Complete daily trade plan."""
    plan_id: str
    timestamp: str
    portfolio_value: float
    cash_available: float           # settled cash minus buffer
    cash_buffer_target: float       # 10% reserve
    cash_after_trades: float        # projected cash after all trades
    is_below_buffer: bool
    trades: list[PlannedTrade] = field(default_factory=list)
    rejected_trades: list[RejectedTrade] = field(default_factory=list)
    num_current_positions: int = 0
    expires_at: str = ""

    def __post_init__(self):
        if not self.plan_id:
            self.plan_id = f"plan_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        if not self.expires_at:
            self.expires_at = (datetime.now(timezone.utc) + timedelta(hours=16)).isoformat()


# ────────────────────────────────────────────────────────────────────────────
# EXPECTED VALUE CALCULATION
# ────────────────────────────────────────────────────────────────────────────

def _compute_expected_value(
    expected_return_pct: float,
    downside_p05_pct: float | None,
    conviction_score: float,
    kelly_fraction: float,
) -> float:
    """Compute risk-adjusted expected value per trade.

    EV = (prob_win × avg_win) - (prob_loss × avg_loss)

    We estimate:
      - prob_win from conviction_score (transformed)
      - avg_win from expected_return_pct
      - avg_loss from downside_p05_pct (5th percentile scenario)
      - Adjusted by Kelly fraction (higher Kelly = better edge/variance ratio)
    """
    # Convert conviction score (0-100) to win probability estimate
    # Logistic transform: 0→30%, 50→55%, 70→65%, 90→78%, 100→82%
    prob_win = 0.30 + 0.52 / (1.0 + math.exp(-0.06 * (conviction_score - 50)))
    prob_loss = 1.0 - prob_win

    avg_win = max(0.0, expected_return_pct)
    avg_loss = abs(downside_p05_pct) if downside_p05_pct else avg_win * 1.5

    ev = (prob_win * avg_win) - (prob_loss * avg_loss)

    # Kelly adjustment: higher Kelly = EV is more reliable
    kelly_mult = min(1.5, max(0.5, 1.0 + kelly_fraction * 2))
    return ev * kelly_mult


def _compute_priority_score(
    ev: float,
    conviction_score: float,
    correlation_with_existing: float,
    avg_daily_volume: float,
    avg_daily_dollar_volume: float,
) -> float:
    """Risk-adjusted priority score for ranking.

    Priority = EV × conviction_factor × (1 - corr_penalty) × liquidity_factor
    """
    # Conviction factor: normalize to 0.5-1.5
    conv_factor = 0.5 + (conviction_score / 100.0)

    # Correlation penalty: higher correlation with existing positions = lower priority
    corr_penalty = min(0.5, max(0.0, correlation_with_existing - 0.3))

    # Liquidity factor: penalize very illiquid names
    if avg_daily_dollar_volume < 50_000:
        liq_factor = 0.3
    elif avg_daily_dollar_volume < 200_000:
        liq_factor = 0.5
    elif avg_daily_dollar_volume < 500_000:
        liq_factor = 0.7
    elif avg_daily_dollar_volume < 2_000_000:
        liq_factor = 0.9
    else:
        liq_factor = 1.0

    return ev * conv_factor * (1.0 - corr_penalty) * liq_factor


# ────────────────────────────────────────────────────────────────────────────
# TRADE PRIORITIZER
# ────────────────────────────────────────────────────────────────────────────

class TradePrioritizer:
    """Converts RebalanceSignals into a ranked, cash-constrained TradePlan."""

    def __init__(
        self,
        constraints: PortfolioConstraints | None = None,
        emergency_buffer_pct: float = 0.05,  # 5% absolute minimum cash
    ):
        self.constraints = constraints or PortfolioConstraints()
        self.emergency_buffer_pct = emergency_buffer_pct

    def build_trade_plan(
        self,
        signals: list[dict[str, Any]],
        portfolio_state: IBPortfolioState,
        candidate_data: list[dict[str, Any]] | None = None,
    ) -> TradePlan:
        """Build a prioritized trade plan from rebalance signals.

        Args:
            signals: list of signal dicts with keys:
                ticker, action, shares, position_dollars, reason,
                conviction_score, expected_return_pct, downside_p05_pct,
                kelly_fraction, commodity_type, country, avg_daily_volume, avg_price
            portfolio_state: current IB portfolio state
            candidate_data: optional extra candidate analysis data

        Returns:
            TradePlan with ranked trades and rejections
        """
        nlv = portfolio_state.net_liquidation
        settled = portfolio_state.settled_cash
        buffer_target = nlv * self.constraints.cash_reserve_pct
        emergency_min = nlv * self.emergency_buffer_pct
        available = max(0.0, settled - buffer_target)

        # Build current exposure maps
        sector_exposure = self._compute_sector_exposure(portfolio_state)
        country_exposure = self._compute_country_exposure(portfolio_state)
        current_tickers = {p.ticker for p in portfolio_state.positions}
        num_positions = len(portfolio_state.positions)

        # Step 1: Separate SELL/CLOSE signals from BUY signals
        close_signals = [s for s in signals if s.get("action") in ("CLOSE", "REDUCE")]
        buy_signals = [s for s in signals if s.get("action") == "BUY"]

        plan_trades: list[PlannedTrade] = []
        rejected: list[RejectedTrade] = []
        cash_freed_unsettled = 0.0  # cash from sells (available T+1)
        close_trade_ids: dict[str, str] = {}  # ticker -> trade_id of the CLOSE

        rank = 0

        # ── STEP 2: Process CLOSE/REDUCE signals first (they free cash) ──
        for sig in close_signals:
            ticker = sig["ticker"]
            action = sig["action"]
            shares = sig.get("shares", 0)
            price = sig.get("avg_price", sig.get("estimated_price", 0))

            # Find existing position
            existing = next((p for p in portfolio_state.positions if p.ticker == ticker), None)
            if not existing:
                rejected.append(RejectedTrade(
                    ticker=ticker, action=action,
                    reason="No existing position found to close",
                    would_need=0.0,
                ))
                continue

            if action == "CLOSE":
                sell_shares = existing.shares
            else:  # REDUCE
                sell_shares = min(shares, existing.shares)

            sell_price = existing.market_price if existing.market_price > 0 else price
            sell_value = sell_shares * sell_price

            rank += 1
            trade = PlannedTrade(
                rank=rank,
                ticker=ticker,
                action=action,
                shares=sell_shares,
                estimated_price=round(sell_price, 4),
                estimated_cost=round(sell_value, 2),
                expected_value_pct=sig.get("expected_return_pct", 0),
                conviction_score=sig.get("conviction_score", 0),
                kelly_fraction=sig.get("kelly_fraction", 0),
                risk_flags=[],
                constraint_headroom={"frees_cash": round(sell_value, 2)},
            )
            plan_trades.append(trade)
            close_trade_ids[ticker] = trade.trade_id
            cash_freed_unsettled += sell_value

            # Update tracking: position count decreases
            if action == "CLOSE":
                num_positions -= 1
                if ticker in current_tickers:
                    current_tickers.discard(ticker)

        # ── STEP 3: Score and rank BUY signals by EV ──
        scored_buys: list[tuple[float, dict[str, Any]]] = []
        for sig in buy_signals:
            ev = _compute_expected_value(
                expected_return_pct=sig.get("expected_return_pct", 0),
                downside_p05_pct=sig.get("downside_p05_pct", None),
                conviction_score=sig.get("conviction_score", 50),
                kelly_fraction=sig.get("kelly_fraction", 0.05),
            )
            price = sig.get("avg_price", sig.get("estimated_price", 0))
            adv = sig.get("avg_daily_volume", 50_000)
            adv_dollar = adv * price if price > 0 else 100_000

            priority = _compute_priority_score(
                ev=ev,
                conviction_score=sig.get("conviction_score", 50),
                correlation_with_existing=sig.get("correlation_with_existing", 0.3),
                avg_daily_volume=adv,
                avg_daily_dollar_volume=adv_dollar,
            )
            sig["_ev"] = ev
            sig["_priority"] = priority
            scored_buys.append((priority, sig))

        # Sort by priority descending
        scored_buys.sort(key=lambda x: x[0], reverse=True)

        # ── STEP 4: Greedy knapsack allocation for BUYs ──
        remaining_cash = available
        cash_available_t1 = available + cash_freed_unsettled  # available tomorrow

        for _, sig in scored_buys:
            ticker = sig["ticker"]
            action = "BUY"
            price = sig.get("avg_price", sig.get("estimated_price", 0))
            kelly = sig.get("kelly_fraction", 0.05)
            conviction = sig.get("conviction_score", 50)
            commodity_type = sig.get("commodity_type", "")
            country = sig.get("country", "US")
            adv = sig.get("avg_daily_volume", 50_000)

            if price <= 0:
                rejected.append(RejectedTrade(
                    ticker=ticker, action=action,
                    reason="No price data available",
                    would_need=0.0,
                ))
                continue

            # Position constraints check
            if num_positions >= self.constraints.max_positions:
                rejected.append(RejectedTrade(
                    ticker=ticker, action=action,
                    reason=f"Max positions reached ({self.constraints.max_positions})",
                    would_need=0.0,
                ))
                continue

            if ticker in current_tickers:
                rejected.append(RejectedTrade(
                    ticker=ticker, action=action,
                    reason="Already in portfolio",
                    would_need=0.0,
                ))
                continue

            # ── SIZE THE POSITION ──
            # Prefer the pre-computed size from the decision engine/portfolio
            # constructor (which already accounts for conviction, liquidity, etc.)
            # Fall back to quarter-Kelly if no pre-computed size available.
            signal_dollars = sig.get("position_dollars", 0)
            investable = nlv * (1.0 - self.constraints.cash_reserve_pct)
            if signal_dollars > 0:
                raw_dollars = signal_dollars
            else:
                raw_dollars = kelly * 0.25 * investable

            # Cap at max position pct
            max_single = self.constraints.max_position_pct * nlv
            target_dollars = min(raw_dollars, max_single)

            # Cap at sector limit
            sect_current = sector_exposure.get(commodity_type, 0.0)
            sect_room = max(0, (self.constraints.max_sector_pct * nlv) - sect_current)
            target_dollars = min(target_dollars, sect_room)
            if target_dollars < self.constraints.min_position_size:
                rejected.append(RejectedTrade(
                    ticker=ticker, action=action,
                    reason=f"Exceeds sector limit ({commodity_type}: "
                           f"${sect_current:,.0f}/{self.constraints.max_sector_pct*nlv:,.0f})",
                    would_need=target_dollars,
                ))
                continue

            # Cap at country limit
            ctry_current = country_exposure.get(country, 0.0)
            ctry_room = max(0, (self.constraints.max_country_pct * nlv) - ctry_current)
            target_dollars = min(target_dollars, ctry_room)
            if target_dollars < self.constraints.min_position_size:
                rejected.append(RejectedTrade(
                    ticker=ticker, action=action,
                    reason=f"Exceeds country limit ({country}: "
                           f"${ctry_current:,.0f}/{self.constraints.max_country_pct*nlv:,.0f})",
                    would_need=target_dollars,
                ))
                continue

            # Cap at 2% of ADV × target hold days
            if adv > 0 and price > 0:
                max_by_vol = adv * price * self.constraints.max_portfolio_adv_pct * self.constraints.target_hold_days
                target_dollars = min(target_dollars, max_by_vol)

            # ── SHARE AFFORDABILITY CHECK ──
            # Critical: if one share costs more than our target, we can't buy it
            # (unless fractional shares are supported)
            shares_needed = int(target_dollars / price)
            if shares_needed < 1:
                supports_fractional = ticker.upper() in FRACTIONAL_ELIGIBLE
                if supports_fractional:
                    shares_needed = 1  # allow fractional
                    target_dollars = price  # one share
                else:
                    rejected.append(RejectedTrade(
                        ticker=ticker, action=action,
                        reason=f"Share price ${price:,.2f} exceeds target position "
                               f"${target_dollars:,.2f} — cannot afford 1 share "
                               f"(no fractional shares for {ticker})",
                        would_need=price,
                    ))
                    continue

            actual_cost = shares_needed * price

            # Enforce minimum position size
            if actual_cost < self.constraints.min_position_size:
                # Try to buy more shares to meet minimum
                min_shares = math.ceil(self.constraints.min_position_size / price)
                actual_cost_min = min_shares * price
                if actual_cost_min <= remaining_cash or actual_cost_min <= cash_available_t1:
                    shares_needed = min_shares
                    actual_cost = actual_cost_min
                else:
                    rejected.append(RejectedTrade(
                        ticker=ticker, action=action,
                        reason=f"Below min position (${actual_cost:,.0f} < "
                               f"${self.constraints.min_position_size:,.0f})",
                        would_need=actual_cost_min,
                    ))
                    continue

            # ── CASH CHECK ──
            risk_flags: list[str] = []
            depends_on: str | None = None

            if actual_cost <= remaining_cash:
                # Can buy today with settled cash
                remaining_cash -= actual_cost
            elif actual_cost <= cash_available_t1:
                # Need to wait for CLOSE to settle (T+1)
                risk_flags.append("REQUIRES_T1_SETTLEMENT")
                depends_on = next(
                    (tid for tid in close_trade_ids.values()), None
                )
                # Don't deduct from remaining_cash (not available today)
                cash_available_t1 -= actual_cost
            else:
                # Try a smaller position
                smaller_cost = remaining_cash
                smaller_shares = int(smaller_cost / price)
                if smaller_shares > 0 and smaller_shares * price >= self.constraints.min_position_size:
                    shares_needed = smaller_shares
                    actual_cost = smaller_shares * price
                    remaining_cash -= actual_cost
                    risk_flags.append("REDUCED_SIZE")
                else:
                    rejected.append(RejectedTrade(
                        ticker=ticker, action=action,
                        reason=f"Insufficient cash (need ${actual_cost:,.0f}, "
                               f"have ${remaining_cash:,.0f} settled / "
                               f"${cash_available_t1:,.0f} T+1)",
                        would_need=actual_cost,
                    ))
                    continue

            # Emergency buffer check
            projected_cash = (nlv - portfolio_state.gross_position_value
                              + cash_freed_unsettled - actual_cost)
            if projected_cash < emergency_min and actual_cost > remaining_cash * 0.5:
                risk_flags.append("NEAR_EMERGENCY_BUFFER")

            # Build constraint headroom dict
            headroom = {
                "single_name_pct": round((nlv * self.constraints.max_position_pct - actual_cost) / nlv * 100, 1),
                "sector_room": round(sect_room - actual_cost, 0),
                "country_room": round(ctry_room - actual_cost, 0),
                "positions_remaining": self.constraints.max_positions - num_positions - 1,
                "cash_remaining": round(remaining_cash, 0),
            }

            rank += 1
            trade = PlannedTrade(
                rank=rank,
                ticker=ticker,
                action=action,
                shares=shares_needed,
                estimated_price=round(price, 4),
                estimated_cost=round(actual_cost, 2),
                expected_value_pct=round(sig.get("_ev", 0), 4),
                conviction_score=conviction,
                kelly_fraction=kelly,
                risk_flags=risk_flags,
                constraint_headroom=headroom,
                depends_on_trade_id=depends_on,
            )
            plan_trades.append(trade)
            num_positions += 1
            current_tickers.add(ticker)

            # Update exposure maps
            sector_exposure[commodity_type] = sect_current + actual_cost
            country_exposure[country] = ctry_current + actual_cost

        # ── STEP 5: Compute final cash projection ──
        total_buys = sum(t.estimated_cost for t in plan_trades if t.action == "BUY")
        total_sells = sum(t.estimated_cost for t in plan_trades if t.action in ("CLOSE", "REDUCE"))
        cash_after = settled - total_buys + total_sells  # approximate (sells settle T+1)

        plan = TradePlan(
            plan_id=f"plan_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            portfolio_value=round(nlv, 2),
            cash_available=round(available, 2),
            cash_buffer_target=round(buffer_target, 2),
            cash_after_trades=round(cash_after, 2),
            is_below_buffer=cash_after < buffer_target,
            trades=plan_trades,
            rejected_trades=rejected,
            num_current_positions=len(portfolio_state.positions),
        )

        audit_log("PLAN_CREATED", "plan", plan.plan_id,
                  details={"trades": len(plan_trades), "rejected": len(rejected),
                           "total_buys": round(total_buys, 2),
                           "total_sells": round(total_sells, 2)})

        return plan

    def _compute_sector_exposure(self, state: IBPortfolioState) -> dict[str, float]:
        """Compute current dollar exposure by commodity_type."""
        exposure: dict[str, float] = {}
        for pos in state.positions:
            ct = pos.commodity_type or "unknown"
            exposure[ct] = exposure.get(ct, 0.0) + (pos.market_value or 0.0)
        return exposure

    def _compute_country_exposure(self, state: IBPortfolioState) -> dict[str, float]:
        """Compute current dollar exposure by country."""
        exposure: dict[str, float] = {}
        for pos in state.positions:
            c = pos.country or "US"
            exposure[c] = exposure.get(c, 0.0) + (pos.market_value or 0.0)
        return exposure


# ────────────────────────────────────────────────────────────────────────────
# CONVENIENCE
# ────────────────────────────────────────────────────────────────────────────

def signals_to_trade_plan(
    rebalance_signals: list,
    portfolio_state: IBPortfolioState,
    constraints: PortfolioConstraints | None = None,
) -> TradePlan:
    """Convert RebalanceSignal objects (from decision_engine) into a TradePlan.

    Bridges the existing decision_engine output format into the execution layer.
    """
    signal_dicts: list[dict[str, Any]] = []

    for sig in rebalance_signals:
        d: dict[str, Any] = {}
        if isinstance(sig, dict):
            d = sig
        elif dataclasses.is_dataclass(sig) and not isinstance(sig, type):
            d = dataclasses.asdict(sig)
        elif hasattr(sig, "__dict__"):
            d = {k: v for k, v in sig.__dict__.items() if not k.startswith("_")}
        else:
            continue

        # Ensure required fields
        d.setdefault("ticker", d.get("ticker", ""))
        d.setdefault("action", d.get("action", "BUY"))
        d.setdefault("shares", d.get("shares", 0))
        d.setdefault("conviction_score", d.get("conviction_score", 50))
        d.setdefault("expected_return_pct", d.get("ev_per_trade_pct", d.get("expected_return_pct", 0)))
        d.setdefault("kelly_fraction", d.get("kelly_fraction", 0.05))
        # Compute avg_price from position_dollars / shares if missing
        raw_price = d.get("avg_price", d.get("estimated_price", 0))
        if (raw_price is None or raw_price <= 0) and d.get("shares", 0) > 0 and d.get("position_dollars", 0) > 0:
            raw_price = d["position_dollars"] / d["shares"]
        d.setdefault("avg_price", raw_price)
        d["avg_price"] = d["avg_price"] if d["avg_price"] and d["avg_price"] > 0 else raw_price
        d.setdefault("avg_daily_volume", d.get("avg_daily_volume", 50_000))
        d.setdefault("commodity_type", d.get("commodity_type", ""))
        d.setdefault("country", d.get("country", "US"))

        signal_dicts.append(d)

    prioritizer = TradePrioritizer(constraints=constraints)
    return prioritizer.build_trade_plan(signal_dicts, portfolio_state)
