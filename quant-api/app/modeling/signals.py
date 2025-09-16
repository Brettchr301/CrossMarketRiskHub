from __future__ import annotations

from datetime import datetime, UTC

from app.modeling.risk import apply_signal_risk_overrides
from app.modeling.types import OptionsImpliedPoint, SignalPoint, ValuationPoint


def estimate_distribution_mismatch(
    valuation: ValuationPoint, options_implied: OptionsImpliedPoint, spot_price: float
) -> float:
    valuation_std_proxy = (valuation.equity_ps_p95 - valuation.equity_ps_p05) / max(spot_price, 0.01)
    options_std_proxy = 2.0 * options_implied.std_return
    return abs(valuation_std_proxy - options_std_proxy)


class SignalEngine:
    def __init__(self, min_holding_days: int = 30, max_holding_days: int = 90):
        self.min_holding_days = min_holding_days
        self.max_holding_days = max_holding_days

    def build(
        self,
        valuation: ValuationPoint,
        options_implied: OptionsImpliedPoint,
        spot_price: float,
        confidence: float,
    ) -> SignalPoint:
        mismatch = estimate_distribution_mismatch(valuation, options_implied, spot_price)
        raw_score = (
            100.0 * valuation.expected_return_net_cost
            + 15.0 * confidence
            - 20.0 * mismatch
            - 20.0 * abs(min(0.0, valuation.downside_p05))
        )
        score, flags = apply_signal_risk_overrides(
            score=raw_score,
            expected_return_net_cost=valuation.expected_return_net_cost,
            downside_p05=valuation.downside_p05,
            options_mismatch=mismatch,
        )
        direction = "LONG" if valuation.expected_return_net_cost >= 0 else "SHORT"
        hold = max(self.min_holding_days, min(self.max_holding_days, valuation.horizon_days))
        return SignalPoint(
            ticker=valuation.ticker,
            score=float(score),
            direction=direction,
            holding_period_days=hold,
            expected_return_net_cost=float(valuation.expected_return_net_cost),
            confidence=float(confidence),
            risk_flags=flags,
            as_of=datetime.now(UTC).replace(tzinfo=None),
        )

