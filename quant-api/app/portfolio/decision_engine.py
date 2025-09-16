"""Investment Decision Engine — the master module.

Combines ALL signals into a final GO / NO-GO investment decision.
This is the production-grade component that turns backtested alpha
into actual trade recommendations for a sub-$100K portfolio.

Decision Pipeline:
  1. Run robustness tests on segment-level backtest results
  2. Screen each candidate through ROIC quality gates
  3. Score translation/accounting information edges
  4. Compute statistical Expected Value per trade
  5. Size positions using quarter-Kelly with risk limits
  6. Construct portfolio respecting all constraints
  7. Generate rebalance signals (BUY / HOLD / CLOSE)
  8. Apply circuit breakers (drawdown, consecutive losses, trade count)

The output is a concrete trade list suitable for IB/Schwab order entry.
NO daytrading. Target: 3-5 trades/month, 20-120 day holds, deep cyclical.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from app.portfolio.robustness import compute_robustness, RobustnessResult
from app.portfolio.quality_filter import assess_roic_quality, ROICProfile
from app.portfolio.translation_edge import score_translation_edge, TranslationEdge
from app.portfolio.risk_manager import (
    PortfolioConstraints, check_risk_limits, RiskState,
    quantify_small_portfolio_edge, SmallPortfolioEdge,
)
from app.portfolio.portfolio_constructor import (
    CandidatePosition,
    PortfolioSnapshot,
    RebalanceSignal,
    compute_conviction_score,
    construct_portfolio,
    generate_rebalance_signals,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# DECISION RESULT
# ────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class InvestmentDecision:
    """Complete investment decision with full reasoning chain."""
    timestamp: str
    # Portfolio state
    portfolio: PortfolioSnapshot | None
    risk_state: RiskState | None
    small_edge: SmallPortfolioEdge | None
    # Actions to take
    rebalance_signals: list[RebalanceSignal]
    # Per-candidate analysis
    candidate_analyses: list[dict[str, Any]]
    # Summary
    total_candidates_evaluated: int
    candidates_passed_robustness: int
    candidates_passed_quality: int
    candidates_passed_all: int
    # Top-level recommendation
    recommendation: str              # "INVEST", "HOLD", "REDUCE", "HALT"
    recommendation_reason: str
    # Performance context
    backtest_alpha_pct: float
    backtest_sharpe: float
    backtest_ev_per_trade_pct: float
    robustness_summary: dict[str, Any]


# ────────────────────────────────────────────────────────────────────────────
# BACKTEST RESULT -> CANDIDATE CONVERSION
# ────────────────────────────────────────────────────────────────────────────

def backtest_results_to_candidates(
    ticker_results: list[dict[str, Any]],
    benchmark_returns_by_ticker: dict[str, np.ndarray],
    constraints: PortfolioConstraints,
    skip_data_fetch: bool = True,
) -> list[CandidatePosition]:
    """Convert raw backtest results into screened CandidatePositions.

    This is the main entry point: takes TickerBacktestResult data
    (from alpha_attribution.py) and runs through the full pipeline.
    """
    candidates: list[CandidatePosition] = []

    for tr in ticker_results:
        ticker = tr["ticker"]
        net_returns = np.array(tr["net_returns"], dtype=float)
        bench = benchmark_returns_by_ticker.get(ticker, np.zeros(len(net_returns)))
        bench = np.array(bench[:len(net_returns)], dtype=float)

        # Step 1: Robustness test
        robustness = compute_robustness(
            strategy_returns=net_returns,
            benchmark_returns=bench,
            vix_at_entry=None,  # TODO: pass VIX data when available
            n_strategies_tested=50,
            holding_days=20,
        )

        # Step 2: ROIC quality check
        quality = assess_roic_quality(
            ticker=ticker,
            commodity_type=tr.get("commodity_type", "oil_gas_upstream"),
            country=tr.get("country", "US"),
            market_cap=tr.get("market_cap", 2e9),
            financials=None if skip_data_fetch else tr.get("financials"),
        )

        # Step 3: Translation edge
        translation = score_translation_edge(ticker, tr.get("market_cap", 2e9))

        # Step 4: Compute EV
        ev = robustness.expected_value_per_trade_pct

        # Step 5: Conviction score
        pred_return = float(np.mean(tr.get("predictions", [0.0]))) * 100.0
        conviction = compute_conviction_score(robustness, quality, translation, pred_return)

        candidate = CandidatePosition(
            ticker=ticker,
            commodity_type=tr.get("commodity_type", "oil_gas_upstream"),
            country=tr.get("country", "US"),
            market_cap=tr.get("market_cap", 2e9),
            avg_price=tr.get("avg_price", 50.0),
            avg_daily_volume=tr.get("avg_daily_volume", 100_000),
            predicted_return_pct=pred_return,
            robustness=robustness,
            quality=quality,
            translation=translation,
            ev_per_trade_pct=ev,
            conviction_score=conviction,
            information_advantage_score=translation.total_asymmetry_score if translation else 0.0,
            sizing=None,
        )
        candidates.append(candidate)

    return candidates


# ────────────────────────────────────────────────────────────────────────────
# MAIN DECISION ENGINE
# ────────────────────────────────────────────────────────────────────────────

def make_investment_decision(
    ticker_results: list[dict[str, Any]],
    benchmark_returns_by_ticker: dict[str, np.ndarray],
    current_positions: dict[str, dict[str, Any]] | None = None,
    peak_equity: float | None = None,
    current_equity: float | None = None,
    consecutive_losses: int = 0,
    trades_this_month: int = 0,
    constraints: PortfolioConstraints | None = None,
    skip_data_fetch: bool = True,
) -> InvestmentDecision:
    """Master investment decision function.

    This is the SINGLE FUNCTION a user calls to go from
    backtest results -> actionable trade recommendations.

    Args:
        ticker_results: list of per-ticker backtest results (from alpha_attribution)
        benchmark_returns_by_ticker: benchmark returns keyed by ticker
        current_positions: existing portfolio positions (None if starting fresh)
        peak_equity: historical peak portfolio equity (for drawdown)
        current_equity: current portfolio equity
        consecutive_losses: count of consecutive losing trades
        trades_this_month: trades executed this month
        constraints: portfolio constraints (defaults for $75K portfolio)
        skip_data_fetch: if True, skip yfinance fundamental data fetch

    Returns:
        InvestmentDecision with full reasoning chain and concrete actions
    """
    if constraints is None:
        constraints = PortfolioConstraints()

    if peak_equity is None:
        peak_equity = constraints.total_capital
    if current_equity is None:
        current_equity = constraints.total_capital

    ts = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

    # Step 1: Check risk limits FIRST (circuit breakers)
    risk_state = check_risk_limits(
        peak_equity=peak_equity,
        current_equity=current_equity,
        consecutive_losses=consecutive_losses,
        trades_this_month=trades_this_month,
    )

    if risk_state.is_halted:
        return InvestmentDecision(
            timestamp=ts,
            portfolio=None,
            risk_state=risk_state,
            small_edge=None,
            rebalance_signals=[],
            candidate_analyses=[],
            total_candidates_evaluated=len(ticker_results),
            candidates_passed_robustness=0,
            candidates_passed_quality=0,
            candidates_passed_all=0,
            recommendation="HALT",
            recommendation_reason=risk_state.halt_reason or "Risk limit breached",
            backtest_alpha_pct=0.0,
            backtest_sharpe=0.0,
            backtest_ev_per_trade_pct=0.0,
            robustness_summary={"halted": True, "reason": risk_state.halt_reason},
        )

    # Step 2: Convert backtest results to candidates
    candidates = backtest_results_to_candidates(
        ticker_results=ticker_results,
        benchmark_returns_by_ticker=benchmark_returns_by_ticker,
        constraints=constraints,
        skip_data_fetch=skip_data_fetch,
    )

    # Step 3: Count how many pass each gate
    passed_robust = sum(1 for c in candidates if c.robustness and c.robustness.overall_robust)
    passed_quality = sum(1 for c in candidates if c.quality and c.quality.overall_quality_pass)
    passed_all = sum(
        1 for c in candidates
        if (c.robustness and c.robustness.overall_robust)
        and (c.quality and c.quality.overall_quality_pass)
        and c.ev_per_trade_pct > 0
    )

    # Step 4: Build candidate analysis summaries
    analyses = []
    for c in sorted(candidates, key=lambda x: x.conviction_score, reverse=True):
        analysis = {
            "ticker": c.ticker,
            "commodity_type": c.commodity_type,
            "country": c.country,
            "conviction_score": c.conviction_score,
            "ev_per_trade_pct": c.ev_per_trade_pct,
            "predicted_return_pct": c.predicted_return_pct,
            "robustness_pass": c.robustness.overall_robust if c.robustness else None,
            "robustness_score": c.robustness.robustness_score if c.robustness else None,
            "quality_pass": c.quality.overall_quality_pass if c.quality else None,
            "quality_score": c.quality.quality_score if c.quality else None,
            "treadmill_score": c.quality.treadmill_score if c.quality else None,
            "roic_vs_expectations_pct": c.quality.roic_vs_expectations if c.quality else None,
            "translation_edge": c.translation.total_asymmetry_score if c.translation else 0.0,
            "market_cap": c.market_cap,
        }
        if c.robustness and c.robustness.rejection_reasons:
            analysis["rejection_reasons"] = c.robustness.rejection_reasons
        analyses.append(analysis)

    # Step 5: Construct portfolio
    portfolio = construct_portfolio(
        candidates=candidates,
        constraints=constraints,
        existing_positions=list(current_positions.keys()) if current_positions else None,
    )

    # Step 6: Generate rebalance signals
    # Always generate signals — even with no current positions, we need
    # BUY signals for new portfolio candidates (fresh start scenario).
    signals = generate_rebalance_signals(
        current_positions=current_positions or {},
        new_candidates=portfolio.positions,
        constraints=constraints,
    )

    # Step 7: Small portfolio edge quantification
    universe_meta = [
        {"ticker": c.ticker, "market_cap": c.market_cap, "country": c.country}
        for c in candidates
    ]
    small_edge = quantify_small_portfolio_edge(constraints.total_capital, universe_meta)

    # Step 8: Overall backtest-level metrics
    all_net = []
    all_bench = []
    for tr in ticker_results:
        all_net.extend(tr.get("net_returns", []))
        bench = benchmark_returns_by_ticker.get(tr["ticker"], [])
        all_bench.extend(bench[:len(tr.get("net_returns", []))])

    net_arr = np.array(all_net, dtype=float) if all_net else np.array([0.0])
    bench_arr = np.array(all_bench[:len(net_arr)], dtype=float) if all_bench else np.zeros(len(net_arr))
    alpha_arr = net_arr - bench_arr

    trades_per_year = 252.0 / 20.0
    bt_alpha = float(np.mean(alpha_arr)) * trades_per_year * 100 if len(alpha_arr) > 0 else 0.0
    bt_sharpe = (
        float(np.mean(net_arr) / np.std(net_arr, ddof=1) * math.sqrt(trades_per_year))
        if len(net_arr) > 1 and np.std(net_arr, ddof=1) > 0 else 0.0
    )

    # Portfolio-level robustness
    overall_robustness = compute_robustness(net_arr, bench_arr, holding_days=20)

    # Step 9: Top-level recommendation
    if risk_state.is_halted:
        recommendation = "HALT"
        rec_reason = risk_state.halt_reason or "Risk limit breached"
    elif passed_all == 0:
        recommendation = "HOLD"
        rec_reason = "No candidates pass all quality gates. Stay in cash."
    elif not overall_robustness.overall_robust:
        recommendation = "REDUCE"
        rec_reason = (
            f"Overall alpha may not be real: {overall_robustness.rejection_reasons}. "
            "Reduce position sizes by 50%."
        )
    elif bt_alpha < 2.0:
        recommendation = "REDUCE"
        rec_reason = f"Overall alpha ({bt_alpha:.1f}%) below 2% minimum. Reduce exposure."
    else:
        recommendation = "INVEST"
        rec_reason = (
            f"Alpha {bt_alpha:.1f}%, Sharpe {bt_sharpe:.2f}, "
            f"{passed_all}/{len(candidates)} candidates pass all gates, "
            f"portfolio EV {portfolio.portfolio_ev_pct:.3f}%/trade"
        )

    return InvestmentDecision(
        timestamp=ts,
        portfolio=portfolio,
        risk_state=risk_state,
        small_edge=small_edge,
        rebalance_signals=signals,
        candidate_analyses=analyses[:50],  # top 50
        total_candidates_evaluated=len(candidates),
        candidates_passed_robustness=passed_robust,
        candidates_passed_quality=passed_quality,
        candidates_passed_all=passed_all,
        recommendation=recommendation,
        recommendation_reason=rec_reason,
        backtest_alpha_pct=round(bt_alpha, 2),
        backtest_sharpe=round(bt_sharpe, 3),
        backtest_ev_per_trade_pct=round(overall_robustness.expected_value_per_trade_pct, 4),
        robustness_summary={
            "overall_robust": overall_robustness.overall_robust,
            "robustness_score": overall_robustness.robustness_score,
            "t_stat": overall_robustness.alpha_t_stat,
            "p_value": overall_robustness.alpha_p_value,
            "bootstrap_ci": [overall_robustness.bootstrap_ci_low_pct, overall_robustness.bootstrap_ci_high_pct],
            "deflated_sharpe": overall_robustness.deflated_sharpe,
            "ev_per_trade_pct": overall_robustness.expected_value_per_trade_pct,
            "kelly_fraction": overall_robustness.kelly_fraction,
            "min_backtest_length": overall_robustness.min_backtest_length_trades,
            "actual_trades": overall_robustness.actual_trades,
            "rejection_reasons": overall_robustness.rejection_reasons,
        },
    )


# ────────────────────────────────────────────────────────────────────────────
# FORMATTED OUTPUT
# ────────────────────────────────────────────────────────────────────────────

def format_decision_report(decision: InvestmentDecision) -> str:
    """Format investment decision as human-readable report."""
    lines: list[str] = []
    lines.append("=" * 90)
    lines.append("  CROSSMARKETRISKHUB -- INVESTMENT DECISION REPORT")
    lines.append("=" * 90)
    lines.append(f"  Timestamp:      {decision.timestamp}")
    lines.append(f"  Recommendation: {decision.recommendation}")
    lines.append(f"  Reason:         {decision.recommendation_reason}")
    lines.append("")

    # Risk state
    if decision.risk_state:
        rs = decision.risk_state
        lines.append(f"  Risk State:")
        lines.append(f"    Drawdown:          {rs.drawdown_pct:.1f}%")
        lines.append(f"    Consecutive Losses: {rs.consecutive_losses}")
        lines.append(f"    Trades This Month: {rs.trades_this_month}")
        lines.append(f"    Halted:            {rs.is_halted}" + (f" ({rs.halt_reason})" if rs.halt_reason else ""))

    # Backtest context
    lines.append("")
    lines.append(f"  Backtest Alpha:    {decision.backtest_alpha_pct:+.2f}% annualized")
    lines.append(f"  Backtest Sharpe:   {decision.backtest_sharpe:.3f}")
    lines.append(f"  EV per Trade:      {decision.backtest_ev_per_trade_pct:.4f}%")

    # Robustness
    rob = decision.robustness_summary
    lines.append("")
    lines.append(f"  Robustness:")
    lines.append(f"    Overall Robust:    {rob.get('overall_robust', False)}")
    lines.append(f"    Score:             {rob.get('robustness_score', 0)}/100")
    lines.append(f"    T-stat:            {rob.get('t_stat', 0):.3f} (p={rob.get('p_value', 1):.4f})")
    ci = rob.get("bootstrap_ci", [0, 0])
    lines.append(f"    95% CI:            [{ci[0]:+.2f}%, {ci[1]:+.2f}%]")
    lines.append(f"    Deflated Sharpe:   {rob.get('deflated_sharpe', 0):.3f}")
    lines.append(f"    Kelly Fraction:    {rob.get('kelly_fraction', 0):.4f}")
    if rob.get("rejection_reasons"):
        for reason in rob["rejection_reasons"]:
            lines.append(f"    WARNING: {reason}")

    # Candidate funnel
    lines.append("")
    lines.append(f"  Candidate Funnel:")
    lines.append(f"    Evaluated:         {decision.total_candidates_evaluated}")
    lines.append(f"    Passed Robustness: {decision.candidates_passed_robustness}")
    lines.append(f"    Passed Quality:    {decision.candidates_passed_quality}")
    lines.append(f"    Passed All Gates:  {decision.candidates_passed_all}")

    # Portfolio
    if decision.portfolio:
        port = decision.portfolio
        lines.append("")
        lines.append(f"  Portfolio:")
        lines.append(f"    Positions:        {port.num_positions}")
        lines.append(f"    Invested:         ${port.invested_capital:,.0f} / ${port.total_capital:,.0f}")
        lines.append(f"    Cash Reserve:     ${port.cash_reserve:,.0f}")
        lines.append(f"    Portfolio EV:     {port.portfolio_ev_pct:.4f}%/trade")
        lines.append(f"    Robustness Score: {port.portfolio_robustness_score:.0f}/100")
        lines.append(f"    % Micro-cap:      {port.pct_micro_cap:.1f}%")
        lines.append(f"    % Foreign-listed: {port.pct_foreign_listed:.1f}%")
        lines.append(f"    Info Edge Score:  {port.information_edge_score:.3f}")

        # Sector exposure
        if port.sector_exposures:
            lines.append(f"    Sector Exposure:")
            for sector, pct in sorted(port.sector_exposures.items(), key=lambda x: -x[1]):
                lines.append(f"      {sector:<25} {pct:.1f}%")

        # Positions list
        if port.positions:
            lines.append("")
            lines.append(f"  {'Ticker':<12} {'Type':<18} {'Ctry':>4} {'$Size':>8} {'%Port':>6} "
                        f"{'Shares':>6} {'Conv':>5} {'EV%':>7} {'Rob':>4} {'Qual':>4}")
            lines.append(f"  {'-'*10}  {'-'*16}  {'-'*2}  {'-'*6}  {'-'*4}  "
                        f"{'-'*4}  {'-'*3}  {'-'*5}  {'-'*2}  {'-'*2}")
            for p in port.positions:
                s = p.sizing
                lines.append(
                    f"  {p.ticker:<12} {p.commodity_type:<18} {p.country:>4} "
                    f"${s.position_dollars:>7,.0f} {s.position_pct:>5.1f}% "
                    f"{s.shares:>6} {p.conviction_score:>5.0f} "
                    f"{p.ev_per_trade_pct:>6.3f}% "
                    f"{(p.robustness.robustness_score if p.robustness else 0):>4.0f} "
                    f"{(p.quality.quality_score if p.quality else 0):>4.0f}"
                )

    # Rebalance signals
    if decision.rebalance_signals:
        lines.append("")
        lines.append(f"  Rebalance Signals:")
        for sig in decision.rebalance_signals:
            emoji = {"BUY": ">>", "HOLD": "--", "CLOSE": "XX"}
            lines.append(f"    {emoji.get(sig.action, '??')} {sig.action:<6} {sig.ticker:<12} "
                        f"${sig.position_dollars:>7,.0f}  {sig.reason}")

    # Small portfolio edge
    if decision.small_edge:
        se = decision.small_edge
        lines.append("")
        lines.append(f"  Small Portfolio Edge ($<100K):")
        lines.append(f"    Our market impact:     {se.estimated_market_impact_bps:.1f} bps")
        lines.append(f"    $100M fund impact:     {se.institutional_impact_bps:.1f} bps")
        lines.append(f"    Impact advantage:      {se.impact_advantage_bps:.1f} bps")
        lines.append(f"    Micro-caps available:  {se.micro_cap_count_in_universe}")
        lines.append(f"    Foreign tickers:       {se.foreign_language_tickers}")

    lines.append("")
    lines.append("=" * 90)
    return "\n".join(lines)
