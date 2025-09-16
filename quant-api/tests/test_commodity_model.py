from __future__ import annotations

from datetime import datetime, UTC

from app.modeling.commodity_impact import CommodityImpactModel
from app.modeling.types import EventProbabilityPoint


def test_commodity_impact_generates_distributions_and_paths():
    model = CommodityImpactModel(horizon_days=60, n_sims=500)
    event_probs = [
        EventProbabilityPoint(
            event_id="hormuz_closure",
            prob=0.2,
            ci_low=0.15,
            ci_high=0.25,
            as_of=datetime.now(UTC).replace(tzinfo=None),
        )
    ]
    base = {"BRENT": 82.0, "WTI": 78.0, "TD3": 85.0}
    distributions, paths, tag = model.generate(event_probs, base)
    assert tag
    assert "BRENT" in paths
    assert len(paths["BRENT"]) == 500
    brent = [x for x in distributions if x.symbol == "BRENT"][0]
    assert brent.p95 >= brent.p50 >= brent.p05

