"""Universe segmentation definitions for alpha attribution backtest.

Segments the ~300 ticker universe across 5 dimensions:
  1. Cap size: micro / small / mid / large
  2. Geography: US / Canada / Australia / Europe_Core / Europe_Periphery / EM / War_Zone
  3. Commodity type: all 17 types
  4. Exchange type: US_Listed / CA / AU / European / Asian / EM
  5. War proximity: war-adjacent vs non-war

Each segment gets an appropriate benchmark ETF for alpha calculation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SegmentDimension = Literal[
    "cap_size", "geography", "commodity_type", "exchange_type", "war_proximity"
]


@dataclass(slots=True)
class SegmentDefinition:
    dimension: SegmentDimension
    label: str
    benchmark_ticker: str  # yfinance symbol for benchmark
    benchmark_name: str


# ── Cap size buckets ──────────────────────────────────────────────────
CAP_SEGMENTS = {
    "micro": SegmentDefinition("cap_size", "Micro (<$500M)", "XME", "SPDR S&P Metals & Mining"),
    "small": SegmentDefinition("cap_size", "Small ($500M-$2B)", "XME", "SPDR S&P Metals & Mining"),
    "mid": SegmentDefinition("cap_size", "Mid ($2B-$10B)", "XLE", "Energy Select Sector"),
    "large": SegmentDefinition("cap_size", "Large (>$10B)", "XLE", "Energy Select Sector"),
}


def classify_cap(market_cap: float) -> str:
    if market_cap < 500_000_000:
        return "micro"
    elif market_cap < 2_000_000_000:
        return "small"
    elif market_cap < 10_000_000_000:
        return "mid"
    else:
        return "large"


# ── Geography buckets ─────────────────────────────────────────────────
# War-zone countries: active conflict or imminent risk
WAR_ZONE_COUNTRIES = {"IL", "BM", "TW", "KR", "PL", "HU", "UA", "NO"}

# Near-war: elevated geopolitical risk, sanctions exposure
NEAR_WAR_COUNTRIES = {"ZA", "GR", "TR", "BR", "CO", "AR", "CL"}

GEOGRAPHY_MAP = {
    "US": {"US"},
    "Canada": {"CA"},
    "Australia": {"AU"},
    "Europe_Core": {"UK", "FR", "DE", "NL", "BE", "AT", "LU", "DK"},
    "Europe_Periphery": {"ES", "PT", "IT", "GR", "PL", "HU", "NO"},
    "Emerging": {"BR", "ZA", "KR", "TW", "HK", "CO", "AR", "CL", "MH"},
    "War_Zone": WAR_ZONE_COUNTRIES,
}

GEO_BENCHMARKS = {
    "US": SegmentDefinition("geography", "United States", "XLE", "Energy Select Sector"),
    "Canada": SegmentDefinition("geography", "Canada", "EWC", "iShares MSCI Canada"),
    "Australia": SegmentDefinition("geography", "Australia", "EWA", "iShares MSCI Australia"),
    "Europe_Core": SegmentDefinition("geography", "Europe Core", "VGK", "Vanguard FTSE Europe"),
    "Europe_Periphery": SegmentDefinition("geography", "Europe Periphery", "VGK", "Vanguard FTSE Europe"),
    "Emerging": SegmentDefinition("geography", "Emerging Markets", "EEM", "iShares MSCI Emerging Mkts"),
    "War_Zone": SegmentDefinition("geography", "War Zone / Near-War", "EEM", "iShares MSCI Emerging Mkts"),
}


def classify_geography(country: str) -> str:
    for geo, countries in GEOGRAPHY_MAP.items():
        if geo == "War_Zone":
            continue  # handled separately
        if country in countries:
            return geo
    return "Emerging"  # default fallback


def classify_war_proximity(country: str) -> str:
    if country in WAR_ZONE_COUNTRIES:
        return "war_zone"
    elif country in NEAR_WAR_COUNTRIES:
        return "near_war"
    else:
        return "non_war"


# ── Exchange type ─────────────────────────────────────────────────────
# Determines IB accessibility and foreign-language market characteristics
EXCHANGE_SUFFIXES = {
    "US_Listed": {""},  # no suffix = US
    "CA_Listed": {".TO", ".V"},
    "AU_Listed": {".AX"},
    "European": {".L", ".OL", ".WA", ".BD", ".VI", ".MC", ".LS", ".MI", ".PA", ".DE", ".AS", ".BU"},
    "Asian": {".HK", ".TW", ".KS", ".T", ".NS"},
    "EM_Listed": {".SA", ".JO"},
}

EXCHANGE_BENCHMARKS = {
    "US_Listed": SegmentDefinition("exchange_type", "US Listed", "XLE", "Energy Select"),
    "CA_Listed": SegmentDefinition("exchange_type", "Canada Listed", "EWC", "iShares Canada"),
    "AU_Listed": SegmentDefinition("exchange_type", "Australia Listed", "EWA", "iShares Australia"),
    "European": SegmentDefinition("exchange_type", "European Listed", "VGK", "Vanguard Europe"),
    "Asian": SegmentDefinition("exchange_type", "Asian Listed", "EEM", "iShares EM"),
    "EM_Listed": SegmentDefinition("exchange_type", "EM Listed", "EEM", "iShares EM"),
}


def classify_exchange(ticker: str) -> str:
    for exchange_type, suffixes in EXCHANGE_SUFFIXES.items():
        for suffix in suffixes:
            if suffix == "" and "." not in ticker:
                return "US_Listed"
            elif suffix and ticker.endswith(suffix):
                return exchange_type
    return "US_Listed"  # fallback


# ── Commodity type benchmarks ─────────────────────────────────────────
COMMODITY_TYPE_BENCHMARKS = {
    "oil_gas_upstream": SegmentDefinition("commodity_type", "Oil & Gas Upstream", "XOP", "SPDR S&P Oil & Gas E&P"),
    "oil_refining": SegmentDefinition("commodity_type", "Oil Refining", "CRAK", "VanEck Oil Refiners"),
    "oil_services": SegmentDefinition("commodity_type", "Oil Services", "OIH", "VanEck Oil Services"),
    "midstream": SegmentDefinition("commodity_type", "Midstream", "AMLP", "Alerian MLP"),
    "precious_metals": SegmentDefinition("commodity_type", "Precious Metals", "GDX", "VanEck Gold Miners"),
    "uranium": SegmentDefinition("commodity_type", "Uranium", "URA", "Global X Uranium"),
    "lithium": SegmentDefinition("commodity_type", "Lithium", "LIT", "Global X Lithium"),
    "rare_earths": SegmentDefinition("commodity_type", "Rare Earths", "REMX", "VanEck Rare Earth"),
}

# All benchmark tickers needed for downloads
ALL_BENCHMARK_TICKERS = sorted(set(
    [s.benchmark_ticker for s in CAP_SEGMENTS.values()]
    + [s.benchmark_ticker for s in GEO_BENCHMARKS.values()]
    + [s.benchmark_ticker for s in EXCHANGE_BENCHMARKS.values()]
    + [s.benchmark_ticker for s in COMMODITY_TYPE_BENCHMARKS.values()]
    + ["SPY"]  # universal benchmark
))


def classify_ticker(ticker: str, country: str, commodity_type: str, market_cap: float) -> dict[str, str]:
    """Return all segment classifications for a single ticker."""
    return {
        "cap_size": classify_cap(market_cap),
        "geography": classify_geography(country),
        "war_proximity": classify_war_proximity(country),
        "exchange_type": classify_exchange(ticker),
        "commodity_type": commodity_type,
    }


def benchmark_for_segment(dimension: str, label: str) -> SegmentDefinition | None:
    """Look up the benchmark definition for a given segment."""
    if dimension == "cap_size":
        return CAP_SEGMENTS.get(label)
    elif dimension == "geography":
        return GEO_BENCHMARKS.get(label)
    elif dimension == "exchange_type":
        return EXCHANGE_BENCHMARKS.get(label)
    elif dimension == "commodity_type":
        return COMMODITY_TYPE_BENCHMARKS.get(label)
    elif dimension == "war_proximity":
        if label == "war_zone":
            return SegmentDefinition("war_proximity", "War Zone", "EEM", "iShares EM")
        elif label == "near_war":
            return SegmentDefinition("war_proximity", "Near War", "EEM", "iShares EM")
        else:
            return SegmentDefinition("war_proximity", "Non-War", "SPY", "S&P 500")
    return None
