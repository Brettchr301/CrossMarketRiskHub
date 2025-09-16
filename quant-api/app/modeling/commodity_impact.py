from __future__ import annotations

from datetime import datetime, UTC
from hashlib import sha256
from typing import Mapping

import numpy as np

from app.modeling.types import DistributionSummary, EventProbabilityPoint


class CommodityImpactModel:
    def __init__(self, horizon_days: int = 60, n_sims: int = 3000):
        self.horizon_days = horizon_days
        self.n_sims = n_sims
        self.event_shocks = {
            "hormuz_closure": {"BRENT": (30.0, 50.0), "WTI": (20.0, 35.0), "TD3": (15.0, 40.0)},
            "red_sea_disruption": {"BRENT": (6.0, 16.0), "WTI": (5.0, 12.0), "BDI": (120.0, 400.0)},
            "sanctions_escalation": {"BRENT": (4.0, 12.0), "WTI": (3.0, 10.0), "BCTI": (30.0, 140.0)},
            "oil_above_100": {"BRENT": (5.0, 18.0), "WTI": (4.0, 16.0)},
        }
        self.base_vol = {"BRENT": 0.12, "WTI": 0.14, "BDI": 0.22, "TD3": 0.3, "BCTI": 0.28}

    def generate(
        self,
        event_probs: list[EventProbabilityPoint],
        base_prices: Mapping[str, float],
    ) -> tuple[list[DistributionSummary], dict[str, np.ndarray], str]:
        event_map = {x.event_id: x.prob for x in event_probs}
        tag = self._simulation_tag(event_map, base_prices)
        out: list[DistributionSummary] = []
        raw_paths: dict[str, np.ndarray] = {}
        now = datetime.now(UTC).replace(tzinfo=None)

        for symbol, start in base_prices.items():
            rng = np.random.default_rng(self._seed(tag, symbol))
            levels = np.full(self.n_sims, float(start))

            for event_id, prob in event_map.items():
                shocks = self.event_shocks.get(event_id, {})
                if symbol not in shocks:
                    continue
                low, high = shocks[symbol]
                event_realized = rng.random(self.n_sims) < prob
                magnitude = rng.uniform(low, high, self.n_sims)
                levels = levels + event_realized * magnitude

            # Regime-aware volatility proxy: higher mean event risk => wider distribution.
            regime_risk = min(1.0, sum(event_map.values()) / max(len(event_map), 1))
            sigma = self.base_vol.get(symbol, 0.15) * (1.0 + 0.8 * regime_risk)
            levels = levels * np.exp(rng.normal(loc=0.0, scale=sigma, size=self.n_sims))
            levels = np.clip(levels, 0.01, None)

            raw_paths[symbol] = levels
            out.append(
                DistributionSummary(
                    symbol=symbol,
                    horizon_days=self.horizon_days,
                    p05=float(np.quantile(levels, 0.05)),
                    p50=float(np.quantile(levels, 0.5)),
                    p95=float(np.quantile(levels, 0.95)),
                    as_of=now,
                    simulation_tag=tag,
                    samples=levels,
                )
            )
        out.sort(key=lambda x: x.symbol)
        return out, raw_paths, tag

    @staticmethod
    def _seed(tag: str, symbol: str) -> int:
        return int(sha256(f"{tag}:{symbol}".encode("utf-8")).hexdigest()[:8], 16)

    @staticmethod
    def _simulation_tag(event_map: Mapping[str, float], base_prices: Mapping[str, float]) -> str:
        raw = "|".join(f"{k}:{event_map[k]:.4f}" for k in sorted(event_map))
        raw += "|" + "|".join(f"{k}:{base_prices[k]:.2f}" for k in sorted(base_prices))
        return sha256(raw.encode("utf-8")).hexdigest()[:16]

