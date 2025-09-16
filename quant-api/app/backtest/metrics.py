from __future__ import annotations

import numpy as np


def sharpe_ratio(returns: np.ndarray, annualization_factor: float = 252.0) -> float:
    if len(returns) < 2:
        return 0.0
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    if std <= 0:
        return 0.0
    return (mean / std) * (annualization_factor**0.5)


def hit_rate(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    return float(np.mean(returns > 0))


def max_drawdown(equity_curve: np.ndarray) -> float:
    if len(equity_curve) == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peaks) / np.maximum(peaks, 1e-9)
    return float(np.min(dd))


def irr_from_periodic_returns(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    equity = float(np.prod(1.0 + returns))
    years = max(1e-6, len(returns) / 252.0)
    return float(equity ** (1.0 / years) - 1.0)

