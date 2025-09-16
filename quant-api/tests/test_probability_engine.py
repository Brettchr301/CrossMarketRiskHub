from __future__ import annotations

from datetime import datetime, UTC

from app.modeling.probability import EventProbabilityEngine
from app.providers.base import PredictionQuoteRow


def test_probability_engine_blends_liquid_quotes():
    engine = EventProbabilityEngine(min_liquidity=0.05)
    now = datetime.now(UTC).replace(tzinfo=None)
    quotes = [
        PredictionQuoteRow(
            provider="a",
            event_id="hormuz_closure",
            bid=0.30,
            ask=0.34,
            volume=3000,
            liquidity_score=0.8,
            as_of=now,
        ),
        PredictionQuoteRow(
            provider="b",
            event_id="hormuz_closure",
            bid=0.36,
            ask=0.42,
            volume=1000,
            liquidity_score=0.5,
            as_of=now,
        ),
    ]
    out = engine.compute(quotes)
    assert len(out) == 1
    assert 0.3 <= out[0].prob <= 0.42
    assert out[0].ci_low <= out[0].prob <= out[0].ci_high

