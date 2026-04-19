"""NOAA weather provider.

Fetches county-level weather for election day turnout correlation.
Free API, no auth required, ~300 requests/min.
"""
from __future__ import annotations
import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.weather.gov"
TIMEOUT = 15

# Key swing state county centroids (lat, lon)
SWING_STATE_COORDS = {
    "PA": (40.59, -77.21),
    "MI": (43.33, -84.54),
    "WI": (44.27, -89.99),
    "AZ": (34.17, -111.93),
    "GA": (32.68, -83.22),
    "NV": (39.88, -117.22),
    "NC": (35.63, -79.81),
    "MN": (46.39, -94.64),
    "NH": (43.45, -71.56),
    "ME": (45.37, -68.97),
    "CO": (39.06, -105.31),
}


def fetch_forecast(lat: float, lon: float) -> dict[str, Any] | None:
    """Fetch weather forecast for a lat/lon point."""
    try:
        headers = {"User-Agent": "ElectionAlpha/1.0 (brett@example.com)"}

        # Step 1: Get grid point
        r = requests.get(f"{BASE_URL}/points/{lat},{lon}", headers=headers, timeout=TIMEOUT)
        r.raise_for_status()
        props = r.json().get("properties", {})
        forecast_url = props.get("forecast")

        if not forecast_url:
            return None

        # Step 2: Get forecast
        r = requests.get(forecast_url, headers=headers, timeout=TIMEOUT)
        r.raise_for_status()
        periods = r.json().get("properties", {}).get("periods", [])

        if not periods:
            return None

        # Return first period (current/today)
        p = periods[0]
        return {
            "temperature": p.get("temperature"),
            "temperature_unit": p.get("temperatureUnit"),
            "wind_speed": p.get("windSpeed", ""),
            "precipitation_pct": _extract_precip_pct(p),
            "short_forecast": p.get("shortForecast", ""),
            "detailed_forecast": p.get("detailedForecast", ""),
            "is_daytime": p.get("isDaytime", True),
        }
    except Exception as exc:
        logger.warning("NOAA forecast failed for (%s, %s): %s", lat, lon, exc)
        return None


def fetch_swing_state_weather() -> dict[str, dict[str, Any]]:
    """Fetch weather for all swing state centroids."""
    results = {}
    for state, (lat, lon) in SWING_STATE_COORDS.items():
        forecast = fetch_forecast(lat, lon)
        if forecast:
            results[state] = forecast
    return results


def weather_turnout_signal(forecast: dict[str, Any]) -> float:
    """Convert weather forecast to turnout impact signal.

    Returns a value from -1.0 (very negative for turnout) to 1.0 (very positive).
    Based on political science research on weather-turnout correlations.
    """
    score = 0.0

    # Precipitation
    precip = forecast.get("precipitation_pct", 0)
    if precip > 50:
        score -= 0.3
    elif precip > 30:
        score -= 0.15

    # Temperature extremes
    temp = forecast.get("temperature")
    if temp is not None:
        if forecast.get("temperature_unit") == "F":
            if temp < 32:
                score -= 0.2
            elif temp < 45:
                score -= 0.1
            elif 60 <= temp <= 80:
                score += 0.1
            elif temp > 95:
                score -= 0.15

    # Wind
    wind_str = str(forecast.get("wind_speed", ""))
    try:
        wind_mph = int("".join(c for c in wind_str.split()[0] if c.isdigit()) or "0")
        if wind_mph > 30:
            score -= 0.15
        elif wind_mph > 20:
            score -= 0.05
    except (ValueError, IndexError):
        pass

    return max(-1.0, min(1.0, score))


def _extract_precip_pct(period: dict) -> int:
    """Extract precipitation probability from forecast period."""
    prob = period.get("probabilityOfPrecipitation", {})
    if isinstance(prob, dict):
        return int(prob.get("value", 0) or 0)
    return 0
