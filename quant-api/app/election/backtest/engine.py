"""Backtest engine for election prediction market strategies.

Replays historical price data through arbitrage detectors and the alpha model,
then computes PnL against actual race outcomes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.election.arbitrage.fee_model import total_arb_fee
from app.election.db.historical_models import (
    BacktestRun,
    BacktestTrade,
    HistoricalQuote,
    RaceOutcome,
)
from app.election.mappings.direction_detector import detect_direction, normalize_price

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    run_id: int
    n_trades: int
    total_pnl: float
    win_rate: float
    sharpe: float
    max_drawdown: float
    trades_by_day: pd.Series


def _build_direction_map(db: Session, cycle: int) -> dict[tuple[str, int], str]:
    """Build a map of (platform, race_id) -> yes_party direction.

    Uses the question text from HistoricalQuote to detect direction.
    Returns only entries where confidence >= 0.5.
    """
    quotes = db.execute(
        select(
            HistoricalQuote.platform,
            HistoricalQuote.race_id,
            HistoricalQuote.question,
        )
        .where(HistoricalQuote.cycle == cycle)
        .distinct()
    ).all()

    direction_map: dict[tuple[str, int], str] = {}
    for platform, race_id, question in quotes:
        if not race_id or not question:
            continue
        result = detect_direction(question)
        if result.confidence >= 0.5:
            direction_map[(platform, race_id)] = result.yes_party
    return direction_map


def build_price_panel(db: Session, cycle: int) -> pd.DataFrame:
    """Build a wide-format price panel for a cycle.

    Returns DataFrame with index=date, columns=(platform, race_id).
    Prices are normalized to P(Dem wins) using direction detection.
    """
    quotes = db.execute(
        select(HistoricalQuote).where(HistoricalQuote.cycle == cycle)
    ).scalars().all()

    if not quotes:
        return pd.DataFrame()

    # Build direction map for normalization
    direction_map = _build_direction_map(db, cycle)

    rows = []
    for q in quotes:
        race_id = q.race_id or -1
        price = q.price
        # Normalize to P(Dem wins) if direction is known
        yes_party = direction_map.get((q.platform, race_id))
        if yes_party:
            price = normalize_price(price, yes_party)

        rows.append({
            "date": pd.to_datetime(q.as_of).normalize(),
            "platform": q.platform,
            "race_id": race_id,
            "price": price,
        })

    df = pd.DataFrame(rows)
    panel = df.pivot_table(
        index="date",
        columns=["platform", "race_id"],
        values="price",
        aggfunc="mean",
    ).sort_index()
    return panel.ffill()


def get_outcome(db: Session, race_id: int) -> dict[str, Any] | None:
    """Look up the actual outcome for a race."""
    outcome = db.execute(
        select(RaceOutcome).where(RaceOutcome.race_id == race_id)
    ).scalar_one_or_none()
    if outcome is None:
        return None
    return {
        "winner_party": outcome.winner_party,
        "election_date": outcome.election_date,
        "settlement": 1.0,  # YES contract settles at $1 if winner
    }


def backtest_cross_market(
    db: Session,
    cycle: int,
    run_name: str = "cross_market_backtest",
    min_net_edge_pct: float = 1.0,
) -> BacktestResult:
    """Replay historical prices and detect cross-market arbitrage opportunities.

    For each day, for each race, compare bid/ask across platforms. If net edge > threshold,
    open a pair trade. Settle at actual outcome.
    """
    panel = build_price_panel(db, cycle)
    if panel.empty:
        logger.warning("No historical data for cycle %d", cycle)
        return BacktestResult(0, 0, 0.0, 0.0, 0.0, 0.0, pd.Series(dtype=float))

    # Create run
    race_ids = {rid for _, rid in panel.columns if rid != -1}

    run = BacktestRun(
        run_name=run_name,
        strategy="cross_market",
        cycle=cycle,
        start_date=panel.index.min().date(),
        end_date=panel.index.max().date(),
        config={"min_net_edge_pct": min_net_edge_pct},
    )
    db.add(run)
    db.flush()
    run_id = run.id

    trades: list[dict[str, Any]] = []

    for race_id in race_ids:
        outcome = get_outcome(db, race_id)
        if outcome is None:
            continue

        # Get platform columns for this race
        race_cols = [(p, r) for p, r in panel.columns if r == race_id]
        if len(race_cols) < 2:
            continue  # Need at least 2 platforms

        race_prices = panel[race_cols].copy()

        # All prices are now normalized to P(Dem wins) by build_price_panel(),
        # so settlement = 1.0 if Dem wins, 0.0 if Rep wins is correct.
        settlement = 1.0 if outcome["winner_party"] == "D" else 0.0

        # Walk forward day by day — only allow trades BEFORE election (real-world constraint)
        election_ts = pd.Timestamp(outcome["election_date"])
        pre_election = race_prices[race_prices.index < election_ts]

        for ts in pre_election.index:
            row = pre_election.loc[ts].dropna()
            if len(row) < 2:
                continue

            # Best bid and best ask proxies (we only have mid prices; use ±0.01)
            platforms_prices = [(p, float(px)) for (p, _), px in row.items()]
            # Find max price (sell at) and min price (buy at)
            platforms_prices.sort(key=lambda x: x[1])
            buy_platform, buy_price = platforms_prices[0]
            sell_platform, sell_price = platforms_prices[-1]

            gross_edge = sell_price - buy_price
            if gross_edge <= 0:
                continue

            fees = total_arb_fee(buy_platform, buy_price, sell_platform, 1.0 - sell_price)
            net_edge = gross_edge - fees
            net_edge_pct = net_edge * 100

            if net_edge_pct < min_net_edge_pct:
                continue

            # Simulate the trade: buy YES on buy_platform, short YES on sell_platform
            # PnL at settlement: (settlement - buy_price) + (sell_price - settlement) = sell_price - buy_price
            # Minus fees
            gross_pnl = gross_edge
            net_pnl = net_edge

            trades.append({
                "run_id": run_id,
                "race_id": race_id,
                "trade_type": "arb_cross",
                "entry_date": ts.date(),
                "entry_price": buy_price,
                "exit_date": outcome["election_date"],
                "exit_price": sell_price,
                "settlement_price": settlement,
                "gross_pnl": gross_pnl,
                "fees": fees,
                "net_pnl": net_pnl,
                "won": net_pnl > 0,
                "description": (
                    f"Buy {buy_platform}@{buy_price:.3f}, Sell {sell_platform}@{sell_price:.3f}, "
                    f"net edge {net_edge_pct:.2f}%"
                ),
            })

    # Persist trades
    for t in trades:
        db.add(BacktestTrade(**t))

    # Compute metrics
    if not trades:
        db.commit()
        return BacktestResult(run_id, 0, 0.0, 0.0, 0.0, 0.0, pd.Series(dtype=float))

    pnls = np.array([t["net_pnl"] for t in trades])
    wins = sum(1 for p in pnls if p > 0)
    total_pnl = float(pnls.sum())
    win_rate = wins / len(pnls)

    # Sharpe (daily)
    df = pd.DataFrame(trades)
    daily_pnl = df.groupby("entry_date")["net_pnl"].sum()
    daily_std = daily_pnl.std()
    sharpe = float((daily_pnl.mean() / daily_std) * np.sqrt(252)) if daily_std > 0 else 0.0

    # Max drawdown
    cum = daily_pnl.cumsum()
    running_max = cum.cummax()
    drawdown = (cum - running_max).min()
    max_dd = float(drawdown)

    # Update run record
    run.n_trades = len(trades)
    run.total_pnl = total_pnl
    run.win_rate = win_rate
    run.sharpe = sharpe
    run.max_drawdown = max_dd
    db.commit()

    logger.info(
        "Backtest %s: %d trades, $%.2f PnL, %.1f%% win rate, Sharpe %.2f",
        run_name, len(trades), total_pnl, win_rate * 100, sharpe,
    )

    return BacktestResult(
        run_id=run_id,
        n_trades=len(trades),
        total_pnl=total_pnl,
        win_rate=win_rate,
        sharpe=sharpe,
        max_drawdown=max_dd,
        trades_by_day=daily_pnl,
    )


def backtest_outcome_betting(
    db: Session,
    cycle: int,
    run_name: str = "outcome_betting_backtest",
    confidence_threshold: float = 0.70,
    days_before_election: int = 30,
) -> BacktestResult:
    """Simpler backtest: at N days before election, bet on whichever side the market favors.

    Tests whether the market's confidence > threshold predicts the actual winner.
    """
    panel = build_price_panel(db, cycle)
    if panel.empty:
        return BacktestResult(0, 0, 0.0, 0.0, 0.0, 0.0, pd.Series(dtype=float))

    race_ids = {rid for _, rid in panel.columns if rid != -1}

    run = BacktestRun(
        run_name=run_name,
        strategy="outcome_betting",
        cycle=cycle,
        start_date=panel.index.min().date(),
        end_date=panel.index.max().date(),
        config={"confidence_threshold": confidence_threshold, "days_before": days_before_election},
    )
    db.add(run)
    db.flush()
    run_id = run.id

    trades = []
    for race_id in race_ids:
        outcome = get_outcome(db, race_id)
        if outcome is None:
            continue

        election_date = outcome["election_date"]
        bet_date = election_date - timedelta(days=days_before_election)
        bet_ts = pd.Timestamp(bet_date)

        race_cols = [(p, r) for p, r in panel.columns if r == race_id]
        if not race_cols:
            continue

        # Find nearest data point
        race_prices = panel[race_cols].dropna(how="all")
        if race_prices.empty:
            continue

        nearest_idx = race_prices.index.get_indexer([bet_ts], method="nearest")[0]
        if nearest_idx < 0:
            continue

        row = race_prices.iloc[nearest_idx].dropna()
        if row.empty:
            continue

        avg_price = float(row.mean())
        if max(avg_price, 1.0 - avg_price) < confidence_threshold:
            continue

        # Bet the favorite
        buy_side = "YES" if avg_price > 0.5 else "NO"
        buy_price = avg_price if buy_side == "YES" else 1.0 - avg_price
        settlement = 1.0 if outcome["winner_party"] == "D" else 0.0
        payout = settlement if buy_side == "YES" else (1.0 - settlement)
        gross_pnl = payout - buy_price

        # Approximate fees (avg across platforms)
        fees = buy_price * (1.0 - buy_price) * 0.05
        net_pnl = gross_pnl - fees

        trades.append({
            "run_id": run_id,
            "race_id": race_id,
            "trade_type": f"outcome_{buy_side.lower()}",
            "entry_date": bet_date,
            "entry_price": buy_price,
            "exit_date": election_date,
            "exit_price": payout,
            "settlement_price": settlement,
            "gross_pnl": gross_pnl,
            "fees": fees,
            "net_pnl": net_pnl,
            "won": net_pnl > 0,
            "description": f"{buy_side} at {buy_price:.3f} confidence, outcome={outcome['winner_party']}",
        })

    for t in trades:
        db.add(BacktestTrade(**t))

    if not trades:
        db.commit()
        return BacktestResult(run_id, 0, 0.0, 0.0, 0.0, 0.0, pd.Series(dtype=float))

    pnls = np.array([t["net_pnl"] for t in trades])
    total_pnl = float(pnls.sum())
    win_rate = sum(1 for p in pnls if p > 0) / len(pnls)

    df = pd.DataFrame(trades)
    daily_pnl = df.groupby("entry_date")["net_pnl"].sum()
    daily_std = daily_pnl.std()
    sharpe = float((daily_pnl.mean() / daily_std) * np.sqrt(252)) if daily_std > 0 else 0.0

    cum = daily_pnl.cumsum()
    max_dd = float((cum - cum.cummax()).min())

    run.n_trades = len(trades)
    run.total_pnl = total_pnl
    run.win_rate = win_rate
    run.sharpe = sharpe
    run.max_drawdown = max_dd
    db.commit()

    logger.info(
        "Outcome-betting backtest %s: %d trades, $%.2f PnL, %.1f%% win rate",
        run_name, len(trades), total_pnl, win_rate * 100,
    )

    return BacktestResult(
        run_id=run_id,
        n_trades=len(trades),
        total_pnl=total_pnl,
        win_rate=win_rate,
        sharpe=sharpe,
        max_drawdown=max_dd,
        trades_by_day=daily_pnl,
    )
