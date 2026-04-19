"""Historical PredictIt data provider.

PredictIt does not publish a historical price API. Options:
1. Internet Archive (archive.org) snapshots of their marketdata endpoint
2. Research-access CSV exports (academic only)
3. Third-party aggregators (e.g., pmxt archive)

This module provides helpers for querying cached snapshots and pmxt archive.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Wayback Machine CDX API for PredictIt snapshots
WAYBACK_CDX = "http://web.archive.org/cdx/search/cdx"
PREDICTIT_URL = "https://www.predictit.org/api/marketdata/all/"
TIMEOUT = 30


def list_wayback_snapshots(from_date: str, to_date: str, limit: int = 500) -> list[dict[str, Any]]:
    """List Wayback Machine snapshots of PredictIt's market data endpoint.

    Dates in YYYYMMDD format. Returns list of snapshots with timestamps.
    """
    try:
        params = {
            "url": PREDICTIT_URL,
            "output": "json",
            "from": from_date,
            "to": to_date,
            "limit": limit,
            "filter": "statuscode:200",
        }
        r = requests.get(WAYBACK_CDX, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        rows = r.json()
        if len(rows) < 2:
            return []
        header = rows[0]
        snapshots = []
        for row in rows[1:]:
            snap = dict(zip(header, row))
            snap["archive_url"] = f"http://web.archive.org/web/{snap['timestamp']}/{PREDICTIT_URL}"
            snapshots.append(snap)
        return snapshots
    except Exception as exc:
        logger.warning("Wayback CDX fetch failed: %s", exc)
        return []


def fetch_snapshot(archive_url: str) -> dict[str, Any] | None:
    """Fetch a single Wayback snapshot of PredictIt market data."""
    try:
        r = requests.get(archive_url, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.warning("Snapshot fetch failed: %s", exc)
        return None


def backfill_from_wayback(from_date: str, to_date: str, sample_n: int = 50) -> dict[str, pd.Series]:
    """Reconstruct PredictIt price history from Wayback snapshots.

    Dates in YYYYMMDD format. Samples up to sample_n snapshots evenly.
    Returns {contract_key: price_series}.
    """
    snapshots = list_wayback_snapshots(from_date, to_date, limit=sample_n * 2)
    if not snapshots:
        return {}

    # Evenly sample snapshots
    step = max(1, len(snapshots) // sample_n)
    sampled = snapshots[::step][:sample_n]

    contract_series: dict[str, list[tuple[pd.Timestamp, float]]] = {}
    for snap in sampled:
        data = fetch_snapshot(snap["archive_url"])
        if not data:
            continue
        ts = pd.to_datetime(snap["timestamp"], format="%Y%m%d%H%M%S")
        for market in data.get("markets", []):
            for contract in market.get("contracts", []):
                key = f"{market.get('name', '')} :: {contract.get('name', '')}"
                last = float(contract.get("lastTradePrice", 0) or 0)
                contract_series.setdefault(key, []).append((ts, last))

    # Convert to pandas Series
    result = {}
    for key, points in contract_series.items():
        if len(points) < 2:
            continue
        points.sort()
        s = pd.Series([p for _, p in points], index=[t for t, _ in points])
        result[key] = s[~s.index.duplicated(keep="last")]

    logger.info("PredictIt Wayback: reconstructed %d contracts from %d snapshots", len(result), len(sampled))
    return result
