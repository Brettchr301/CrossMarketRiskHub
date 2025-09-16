from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

import numpy as np
import pandas as pd

from app.backtest.metrics import hit_rate, irr_from_periodic_returns, max_drawdown, sharpe_ratio
from app.modeling.cost_model import estimate_total_cost_bps, net_return_after_cost


@dataclass(slots=True)
class BacktestResult:
    sharpe: float
    hit_rate: float
    average_alpha: float
    max_drawdown: float
    turnover: float
    capacity: float
    irr: float
    trade_count: int


def validate_no_lookahead(signal_time: datetime, entry_time: datetime, exit_time: datetime) -> bool:
    return signal_time <= entry_time < exit_time


class WalkForwardBacktester:
    def __init__(self, holding_days: int = 60):
        self.holding_days = holding_days

    def run(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        benchmark_returns: Iterable[float] | None = None,
    ) -> BacktestResult:
        # signals columns: ticker, as_of, direction, score
        # prices columns: ticker, as_of, close
        if signals.empty or prices.empty:
            return BacktestResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

        px = prices.copy()
        px["as_of"] = pd.to_datetime(px["as_of"])
        px = px.sort_values(["ticker", "as_of"])

        trade_returns: list[float] = []
        for _, row in signals.iterrows():
            ticker = str(row["ticker"]).upper()
            signal_time = pd.to_datetime(row["as_of"]).to_pydatetime()
            entry_time = signal_time + timedelta(days=1)
            exit_time = signal_time + timedelta(days=self.holding_days)
            if not validate_no_lookahead(signal_time, entry_time, exit_time):
                continue
            ticker_px = px[px["ticker"].str.upper() == ticker]
            entry_row = ticker_px[ticker_px["as_of"] >= entry_time].head(1)
            exit_row = ticker_px[ticker_px["as_of"] >= exit_time].head(1)
            if entry_row.empty or exit_row.empty:
                continue
            p0 = float(entry_row.iloc[0]["close"])
            p1 = float(exit_row.iloc[0]["close"])
            direction = str(row.get("direction", "LONG")).upper()
            gross = (p1 - p0) / max(p0, 0.01)
            if direction == "SHORT":
                gross = -gross
            cost_bps = estimate_total_cost_bps(hold_days=self.holding_days)
            net = net_return_after_cost(gross, cost_bps)
            trade_returns.append(net)

        if not trade_returns:
            return BacktestResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

        returns = np.asarray(trade_returns, dtype=float)
        benchmark = np.asarray(list(benchmark_returns), dtype=float) if benchmark_returns is not None else np.zeros_like(returns)
        alpha = returns - benchmark[: len(returns)]
        equity_curve = np.cumprod(1.0 + returns)

        turnover = min(3.0, len(returns) / 40.0)
        capacity = max(1_000_000.0, 15_000_000.0 / (1.0 + turnover))
        return BacktestResult(
            sharpe=float(sharpe_ratio(returns)),
            hit_rate=float(hit_rate(returns)),
            average_alpha=float(np.mean(alpha)),
            max_drawdown=float(max_drawdown(equity_curve)),
            turnover=float(turnover),
            capacity=float(capacity),
            irr=float(irr_from_periodic_returns(returns)),
            trade_count=len(returns),
        )

