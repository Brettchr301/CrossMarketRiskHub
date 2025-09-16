from __future__ import annotations

from datetime import datetime, UTC, timezone
from hashlib import sha256
from typing import Sequence

import requests

from app.providers.base import PredictionQuoteRow


class PolymarketProvider:
    """Free-first adapter.

    If a compatible API endpoint is not configured or fails, deterministic mock quotes
    are produced so the pipeline remains testable end-to-end.
    """

    def __init__(self, api_url: str | None = None, timeout_seconds: int = 8):
        self.api_url = api_url
        self.timeout_seconds = timeout_seconds

    def fetch_event_quotes(self, events: Sequence[str]) -> Sequence[PredictionQuoteRow]:
        if self.api_url:
            try:
                response = requests.get(self.api_url, timeout=self.timeout_seconds)
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, list):
                    mapped: list[PredictionQuoteRow] = []
                    now = datetime.now(timezone.utc).replace(tzinfo=None)
                    for row in payload:
                        event_id = str(row.get("event_id") or row.get("slug") or "")
                        if event_id not in events:
                            continue
                        bid = float(row.get("bid", row.get("yes_bid", 0.4)))
                        ask = float(row.get("ask", row.get("yes_ask", 0.6)))
                        volume = float(row.get("volume", 1000.0))
                        liquidity = float(row.get("liquidity_score", min(1.0, volume / 5000.0)))
                        mapped.append(
                            PredictionQuoteRow(
                                provider="polymarket",
                                event_id=event_id,
                                bid=max(0.0, min(1.0, bid)),
                                ask=max(0.0, min(1.0, ask)),
                                volume=max(1.0, volume),
                                liquidity_score=max(0.01, min(1.0, liquidity)),
                                as_of=now,
                            )
                        )
                    if mapped:
                        return mapped
            except requests.RequestException:
                pass
        return self._mock_quotes(events)

    def _mock_quotes(self, events: Sequence[str]) -> Sequence[PredictionQuoteRow]:
        now = datetime.now(UTC).replace(tzinfo=None)
        rows: list[PredictionQuoteRow] = []
        for event in events:
            seed = int(sha256(f"polymarket:{event}".encode("utf-8")).hexdigest()[:8], 16)
            center = 0.2 + (seed % 5500) / 10000.0
            spread = 0.04 + (seed % 3) * 0.01
            bid = max(0.01, center - spread / 2)
            ask = min(0.99, center + spread / 2)
            volume = 1500 + (seed % 3000)
            rows.append(
                PredictionQuoteRow(
                    provider="polymarket",
                    event_id=event,
                    bid=bid,
                    ask=ask,
                    volume=float(volume),
                    liquidity_score=min(1.0, volume / 6000.0),
                    as_of=now,
                )
            )
        return rows

