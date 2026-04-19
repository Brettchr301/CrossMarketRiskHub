"""Historical weather data for backtesting.

Uses Open-Meteo archive API (free, no auth) for hourly historical weather
going back decades. Precision-aligned with prediction market timestamps.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

OPEN_METEO_API = "https://archive-api.open-meteo.com/v1/archive"
TIMEOUT = 30

SWING_STATE_COORDS = {
    "PA": (40.59, -77.21), "MI": (43.33, -84.54), "WI": (44.27, -89.99),
    "AZ": (34.17, -111.93), "GA": (32.68, -83.22), "NV": (39.88, -117.22),
    "NC": (35.63, -79.81), "MN": (46.39, -94.64), "NH": (43.45, -71.56),
    "ME": (45.37, -68.97), "CO": (39.06, -105.31), "FL": (28.55, -81.80),
    "OH": (40.29, -82.79), "TX": (31.05, -97.56), "IA": (41.88, -93.10),
    "OR": (43.80, -120.55), "MO": (38.46, -92.29), "IN": (39.85, -86.28),
    "ND": (47.55, -100.48), "MT": (46.88, -110.36), "WV": (38.64, -80.62),
}


def fetch_hourly_weather(
    lat: float,
    lon: float,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Fetch hourly historical weather for a location via Open-Meteo archive."""
    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "hourly": "temperature_2m,precipitation,wind_speed_10m,cloud_cover,visibility,rain,snowfall",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": "America/New_York",
        }
        r = requests.get(OPEN_METEO_API, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        if not times:
            return pd.DataFrame()

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(times),
            "temperature": hourly.get("temperature_2m", []),
            "precipitation": hourly.get("precipitation", []),
            "wind_speed": hourly.get("wind_speed_10m", []),
            "cloud_cover": hourly.get("cloud_cover", []),
            "visibility": hourly.get("visibility", []),
            "rain": hourly.get("rain", []),
            "snowfall": hourly.get("snowfall", []),
        }).set_index("timestamp")
        return df
    except Exception as exc:
        logger.warning("Open-Meteo fetch failed for (%s,%s): %s", lat, lon, exc)
        return pd.DataFrame()


def fetch_election_day_weather(
    election_date: date,
    days_around: int = 3,
) -> dict[str, pd.DataFrame]:
    """Fetch hourly weather for all swing states around an election date."""
    start = election_date - timedelta(days=days_around)
    end = election_date + timedelta(days=days_around)

    results = {}
    for state, (lat, lon) in SWING_STATE_COORDS.items():
        df = fetch_hourly_weather(lat, lon, start, end)
        if not df.empty:
            results[state] = df
            logger.info("Historical weather %s: %d hourly points", state, len(df))
    return results


def weather_turnout_score_hourly(row: pd.Series) -> float:
    """Compute turnout impact from hourly weather (voting hours 7am-8pm)."""
    score = 0.0
    temp = row.get("temperature")
    precip = row.get("precipitation", 0) or 0
    wind = row.get("wind_speed", 0) or 0
    snow = row.get("snowfall", 0) or 0

    if temp is not None and pd.notna(temp):
        if temp < 32: score -= 0.25
        elif temp < 45: score -= 0.10
        elif 60 <= temp <= 80: score += 0.10
        elif temp > 95: score -= 0.15

    if precip > 0.1: score -= 0.20
    if precip > 0.3: score -= 0.30
    if snow > 0.1: score -= 0.30
    if wind > 25: score -= 0.15

    return max(-1.0, min(1.0, score))
