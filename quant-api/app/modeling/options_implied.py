from __future__ import annotations

from datetime import datetime, UTC
from math import exp
from statistics import fmean
from typing import Sequence

import numpy as np

from app.modeling.types import OptionsImpliedPoint
from app.providers.base import OptionQuoteRow


class OptionsImpliedDistributionModel:
    """Extracts a compact risk-neutral distribution summary from option chains."""

    def __init__(self, horizon_days: int = 60):
        self.horizon_days = horizon_days

    def infer(self, ticker: str, chain: Sequence[OptionQuoteRow], spot_price: float) -> OptionsImpliedPoint:
        if not chain:
            return OptionsImpliedPoint(
                ticker=ticker.upper(),
                horizon_days=self.horizon_days,
                mean_return=0.0,
                std_return=0.2,
                downside_p05=-0.3,
                upside_p95=0.35,
                as_of=datetime.now(UTC).replace(tzinfo=None),
                meta_payload={"fallback": True},
            )

        call_rows = [x for x in chain if x.option_type.lower() == "call"]
        if not call_rows:
            call_rows = list(chain)
        atm = min(call_rows, key=lambda x: abs(x.strike - spot_price))
        iv_values = [x.implied_vol for x in call_rows if x.implied_vol > 0]
        iv_atm = float(atm.implied_vol if atm.implied_vol > 0 else fmean(iv_values))
        iv_mean = float(fmean(iv_values)) if iv_values else iv_atm

        t = self.horizon_days / 365.0
        # Simple lognormal approximation for risk-neutral return distribution.
        mu = -0.5 * iv_mean * iv_mean * t
        sigma = max(1e-6, iv_atm * (t ** 0.5))
        rng = np.random.default_rng(42)
        log_returns = rng.normal(mu, sigma, 5000)
        simple_returns = np.exp(log_returns) - 1.0

        mean_return = float(np.mean(simple_returns))
        std_return = float(np.std(simple_returns))
        downside_p05 = float(np.quantile(simple_returns, 0.05))
        upside_p95 = float(np.quantile(simple_returns, 0.95))
        return OptionsImpliedPoint(
            ticker=ticker.upper(),
            horizon_days=self.horizon_days,
            mean_return=mean_return,
            std_return=std_return,
            downside_p05=downside_p05,
            upside_p95=upside_p95,
            as_of=datetime.now(UTC).replace(tzinfo=None),
            meta_payload={"atm_iv": iv_atm, "mean_iv": iv_mean, "proxy_model": "lognormal"},
        )


def approximate_terminal_price(spot_price: float, implied: OptionsImpliedPoint, q: float) -> float:
    if q <= 0.05:
        return spot_price * exp(implied.downside_p05)
    if q >= 0.95:
        return spot_price * exp(implied.upside_p95)
    return spot_price * exp(implied.mean_return)

