from __future__ import annotations

from datetime import datetime, UTC, timedelta

import pandas as pd

from app.backtest.engine import WalkForwardBacktester, validate_no_lookahead


def test_no_lookahead_validation():
    now = datetime.now(UTC).replace(tzinfo=None)
    assert validate_no_lookahead(now, now + timedelta(days=1), now + timedelta(days=10))
    assert not validate_no_lookahead(now, now - timedelta(days=1), now + timedelta(days=10))


def test_walk_forward_backtester_runs_without_lookahead():
    base = datetime(2025, 1, 1)
    signals = pd.DataFrame(
        [
            {"ticker": "TNK", "as_of": base + timedelta(days=3), "direction": "LONG", "score": 12.0},
            {"ticker": "TNK", "as_of": base + timedelta(days=10), "direction": "LONG", "score": 14.0},
        ]
    )
    prices = pd.DataFrame(
        [
            {"ticker": "TNK", "as_of": base + timedelta(days=i), "close": 20.0 + 0.15 * i}
            for i in range(1, 110)
        ]
    )
    backtester = WalkForwardBacktester(holding_days=30)
    result = backtester.run(signals=signals, prices=prices)
    assert result.trade_count > 0
    assert result.hit_rate >= 0.0

