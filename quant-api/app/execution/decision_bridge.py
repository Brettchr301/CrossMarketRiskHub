"""Bridge between execution layer and the decision engine.

The decision engine (decision_engine.py) requires full backtest results
(ticker_results + benchmark_returns) to produce InvestmentDecision objects.
Running the full backtest takes minutes, so we:

  1. Cache the decision's rebalance_signals to JSON after each backtest run.
  2. The execution layer loads cached signals for daily plan generation.
  3. Optionally, the generate-plan route can force a fresh backtest.

Usage:
    # After running backtest:
    save_decision_cache(decision)

    # In execution layer:
    signals = load_cached_signals()
    plan = signals_to_trade_plan(signals, ...)
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.portfolio.portfolio_constructor import RebalanceSignal

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DECISION_CACHE = CACHE_DIR / "decision_signals_cache.json"


def save_decision_cache(decision: Any) -> Path:
    """Persist decision's rebalance_signals to JSON for the execution layer.

    Call this after running the backtest + decision engine (e.g. in
    run_alpha_backtest.py or a periodic refresh job).

    Args:
        decision: InvestmentDecision from make_investment_decision()

    Returns:
        Path to the saved cache file.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    signals_data = []
    for sig in decision.rebalance_signals:
        signals_data.append({
            "action": sig.action,
            "ticker": sig.ticker,
            "reason": sig.reason,
            "position_dollars": sig.position_dollars,
            "shares": sig.shares,
            "days_held": sig.days_held,
            "target_hold_days": sig.target_hold_days,
            "avg_price": getattr(sig, "avg_price", 0.0),
            "conviction_score": getattr(sig, "conviction_score", 50.0),
            "ev_per_trade_pct": getattr(sig, "ev_per_trade_pct", 0.0),
            "kelly_fraction": getattr(sig, "kelly_fraction", 0.05),
            "commodity_type": getattr(sig, "commodity_type", ""),
            "country": getattr(sig, "country", "US"),
            "avg_daily_volume": getattr(sig, "avg_daily_volume", 50_000.0),
        })

    cache = {
        "timestamp": decision.timestamp,
        "recommendation": decision.recommendation,
        "recommendation_reason": decision.recommendation_reason,
        "backtest_alpha_pct": decision.backtest_alpha_pct,
        "backtest_sharpe": decision.backtest_sharpe,
        "backtest_ev_per_trade_pct": decision.backtest_ev_per_trade_pct,
        "total_candidates_evaluated": decision.total_candidates_evaluated,
        "candidates_passed_all": decision.candidates_passed_all,
        "rebalance_signals": signals_data,
        "cached_at": datetime.now(timezone.utc).isoformat(),
    }

    DECISION_CACHE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    logger.info("Decision cache saved: %d signals, recommendation=%s",
                len(signals_data), decision.recommendation)
    return DECISION_CACHE


def load_cached_signals() -> list[RebalanceSignal]:
    """Load rebalance signals from the most recent decision cache.

    Returns:
        List of RebalanceSignal objects. Empty list if no cache or stale.
    """
    if not DECISION_CACHE.exists():
        logger.warning("No decision cache found at %s", DECISION_CACHE)
        return []

    try:
        data = json.loads(DECISION_CACHE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to read decision cache: %s", exc)
        return []

    # Check recommendation — if HOLD or HALT, no signals to act on
    rec = data.get("recommendation", "HOLD")
    if rec in ("HALT",):
        logger.info("Decision cache recommendation is %s — no trades", rec)
        return []

    signals = []
    for s in data.get("rebalance_signals", []):
        try:
            pos_dollars = s.get("position_dollars", 0.0)
            shares = s.get("shares", 0)
            avg_price = s.get("avg_price", 0.0)
            # Fallback: compute avg_price from position_dollars / shares
            if avg_price <= 0 and shares > 0 and pos_dollars > 0:
                avg_price = pos_dollars / shares
            signals.append(RebalanceSignal(
                action=s["action"],
                ticker=s["ticker"],
                reason=s.get("reason", ""),
                position_dollars=pos_dollars,
                shares=shares,
                days_held=s.get("days_held", 0),
                target_hold_days=s.get("target_hold_days", 20),
                avg_price=avg_price,
                conviction_score=s.get("conviction_score", 50.0),
                ev_per_trade_pct=s.get("ev_per_trade_pct", 0.0),
                kelly_fraction=s.get("kelly_fraction", 0.05),
                commodity_type=s.get("commodity_type", ""),
                country=s.get("country", "US"),
                avg_daily_volume=s.get("avg_daily_volume", 50_000.0),
            ))
        except (KeyError, TypeError) as exc:
            logger.warning("Skipping malformed signal: %s — %s", s, exc)

    logger.info("Loaded %d cached signals (recommendation=%s, cached_at=%s)",
                len(signals), rec, data.get("cached_at", "unknown"))
    return signals


def load_decision_metadata() -> dict[str, Any]:
    """Load just the metadata from the decision cache (no signals)."""
    if not DECISION_CACHE.exists():
        return {}
    try:
        data = json.loads(DECISION_CACHE.read_text(encoding="utf-8"))
        return {k: v for k, v in data.items() if k != "rebalance_signals"}
    except (json.JSONDecodeError, OSError):
        return {}


def run_fresh_backtest_and_cache() -> list[RebalanceSignal]:
    """Run a fresh backtest, generate decision, cache it, return signals.

    This is EXPENSIVE (takes several minutes) — only use when explicitly
    requested or when cached signals are too stale.
    """
    try:
        from app.backtest.alpha_attribution import SegmentedAlphaBacktester
        from app.portfolio.decision_engine import make_investment_decision
        import numpy as np

        logger.info("Starting fresh backtest for execution layer...")

        backtester = SegmentedAlphaBacktester(
            holding_days=20,
            min_predictions=12,
            lookback_days=780,
            walk_forward_lookback=150,
            min_trades_per_segment=10,
            signal_threshold=0.001,
            verbose=False,
        )

        result = backtester.run()

        # Convert ticker results to decision engine format
        ticker_dicts = []
        bench_by_ticker: dict[str, Any] = {}

        for tr in result.ticker_results:
            ticker_dicts.append({
                "ticker": tr.ticker,
                "commodity_type": tr.commodity_type,
                "country": tr.country,
                "market_cap": tr.market_cap,
                "avg_price": 50.0,
                "avg_daily_volume": 100_000,
                "predictions": tr.predictions,
                "net_returns": tr.net_returns,
            })
            spy_rets = tr.benchmark_returns.get("SPY", [0.0] * len(tr.net_returns))
            bench_by_ticker[tr.ticker] = np.array(
                spy_rets[:len(tr.net_returns)], dtype=float
            )

        decision = make_investment_decision(
            ticker_results=ticker_dicts,
            benchmark_returns_by_ticker=bench_by_ticker,
            skip_data_fetch=True,
        )

        save_decision_cache(decision)
        return list(decision.rebalance_signals)

    except Exception as exc:
        logger.error("Fresh backtest failed: %s", exc, exc_info=True)
        raise
