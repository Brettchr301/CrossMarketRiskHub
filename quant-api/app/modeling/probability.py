from __future__ import annotations

from collections import defaultdict
from datetime import datetime, UTC, timedelta
from math import sqrt
from typing import Iterable

from app.modeling.types import EventProbabilityPoint
from app.providers.base import PredictionQuoteRow


class EventProbabilityEngine:
    def __init__(self, min_liquidity: float = 0.05, stale_minutes: int = 180):
        self.min_liquidity = min_liquidity
        self.stale_minutes = stale_minutes

    def compute(
        self,
        quotes: Iterable[PredictionQuoteRow],
        linked_events: dict[str, list[str]] | None = None,
        as_of: datetime | None = None,
    ) -> list[EventProbabilityPoint]:
        now = as_of or datetime.now(UTC).replace(tzinfo=None)
        grouped: dict[str, list[PredictionQuoteRow]] = defaultdict(list)
        stale_cutoff = now - timedelta(minutes=self.stale_minutes)
        for quote in quotes:
            if quote.liquidity_score < self.min_liquidity:
                continue
            if quote.as_of < stale_cutoff:
                continue
            grouped[quote.event_id].append(quote)

        baseline: dict[str, EventProbabilityPoint] = {}
        for event_id, rows in grouped.items():
            weighted_mid = 0.0
            total_w = 0.0
            sample_size = 0.0
            for row in rows:
                w = max(0.01, row.liquidity_score) * max(1.0, row.volume)
                weighted_mid += row.mid_price * w
                total_w += w
                sample_size += max(5.0, row.volume * row.liquidity_score)
            if total_w <= 0:
                continue
            p = max(0.0, min(1.0, weighted_mid / total_w))
            n_eff = max(10.0, sample_size / 100.0)
            variance = (p * (1.0 - p)) / (n_eff + 3.0)
            half_width = 1.96 * sqrt(max(variance, 1e-8))
            baseline[event_id] = EventProbabilityPoint(
                event_id=event_id,
                prob=p,
                ci_low=max(0.0, p - half_width),
                ci_high=min(1.0, p + half_width),
                as_of=now,
            )

        if not linked_events:
            return sorted(baseline.values(), key=lambda x: x.event_id)

        adjusted: dict[str, EventProbabilityPoint] = {}
        for event_id, point in baseline.items():
            related = linked_events.get(event_id, [])
            related_probs = [baseline[r].prob for r in related if r in baseline]
            if not related_probs:
                adjusted[event_id] = point
                continue
            blended = 0.85 * point.prob + 0.15 * (sum(related_probs) / len(related_probs))
            ci_half = max(0.01, 0.5 * (point.ci_high - point.ci_low))
            adjusted[event_id] = EventProbabilityPoint(
                event_id=event_id,
                prob=max(0.0, min(1.0, blended)),
                ci_low=max(0.0, blended - ci_half),
                ci_high=min(1.0, blended + ci_half),
                as_of=point.as_of,
            )
        return sorted(adjusted.values(), key=lambda x: x.event_id)

