from __future__ import annotations

from datetime import datetime, UTC
from hashlib import sha256
from typing import Sequence

from app.providers.base import PredictionQuoteRow


class KalshiProvider:
    """Kalshi adapter with deterministic fallback data."""

    def fetch_event_quotes(self, events: Sequence[str]) -> Sequence[PredictionQuoteRow]:
        now = datetime.now(UTC).replace(tzinfo=None)
        rows: list[PredictionQuoteRow] = []
        for event in events:
            seed = int(sha256(f"kalshi:{event}".encode("utf-8")).hexdigest()[:8], 16)
            center = 0.15 + (seed % 6500) / 10000.0
            spread = 0.03 + (seed % 4) * 0.01
            bid = max(0.01, center - spread / 2)
            ask = min(0.99, center + spread / 2)
            volume = 1200 + (seed % 3500)
            rows.append(
                PredictionQuoteRow(
                    provider="kalshi",
                    event_id=event,
                    bid=bid,
                    ask=ask,
                    volume=float(volume),
                    liquidity_score=min(1.0, volume / 7000.0),
                    as_of=now,
                )
            )
        return rows

