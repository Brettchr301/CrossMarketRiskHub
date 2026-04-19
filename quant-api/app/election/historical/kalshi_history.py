"""Historical Kalshi data provider via candlesticks API."""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE = "https://api.elections.kalshi.com/trade-api/v2"
TIMEOUT = 20


def fetch_markets_for_event(event_ticker: str) -> list[dict[str, Any]]:
    """Fetch all markets under an event ticker (including settled)."""
    try:
        params = {"event_ticker": event_ticker, "limit": 200}
        r = requests.get(f"{BASE}/markets", params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json().get("markets", [])
    except Exception as exc:
        logger.warning("Kalshi markets fetch failed for %s: %s", event_ticker, exc)
        return []


def fetch_candlesticks(market_ticker: str, start_ts: int, end_ts: int, period_interval: int = 1440) -> pd.Series:
    """Fetch candlestick history for a single Kalshi market.

    period_interval: 1 (minute), 60 (hour), 1440 (day)
    """
    try:
        params = {
            "market_tickers": market_ticker,
            "start_ts": int(start_ts),
            "end_ts": int(end_ts),
            "period_interval": period_interval,
        }
        r = requests.get(f"{BASE}/markets/candlesticks", params=params, timeout=TIMEOUT)
        if r.status_code >= 400:
            # Try shorter window
            one_year_ago = int(datetime.now(UTC).timestamp()) - 365 * 24 * 3600
            params["start_ts"] = max(one_year_ago, int(start_ts))
            r = requests.get(f"{BASE}/markets/candlesticks", params=params, timeout=TIMEOUT)
            if r.status_code >= 400:
                return pd.Series(dtype=float)

        markets = r.json().get("markets", [])
        if not markets:
            return pd.Series(dtype=float)

        rows = []
        for m in markets:
            for c in m.get("candlesticks", []):
                close = c.get("price", {}).get("close")
                if close is None:
                    continue
                ts = pd.to_datetime(int(c.get("end_period_ts", 0)), unit="s", utc=True).tz_convert(None)
                rows.append((ts, float(close) / 100.0))

        if not rows:
            return pd.Series(dtype=float)

        s = pd.Series([p for _, p in rows], index=[t for t, _ in rows]).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        return s.clip(0.0, 1.0)
    except Exception as exc:
        logger.warning("Candlestick fetch failed for %s: %s", market_ticker, exc)
        return pd.Series(dtype=float)


def backfill_event(event_ticker: str, days_back: int = 730) -> dict[str, pd.Series]:
    """Backfill all markets under an event ticker."""
    end = datetime.now(UTC)
    start = end - timedelta(days=days_back)
    start_ts = int(start.timestamp())
    end_ts = int(end.timestamp())

    markets = fetch_markets_for_event(event_ticker)
    results = {}
    for m in markets:
        ticker = m.get("ticker")
        if not ticker:
            continue
        series = fetch_candlesticks(ticker, start_ts, end_ts)
        if not series.empty:
            title = m.get("title", ticker)
            results[title] = series
            logger.info("Kalshi backfilled '%s': %d points", title[:60], len(series))
    return results
