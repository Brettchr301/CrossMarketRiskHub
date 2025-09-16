"""Translation & Accounting Edge Module.

CORE THESIS:
  When commodity stocks trade on non-English exchanges or report under
  non-US-GAAP accounting standards, there is a structural INFORMATION ASYMMETRY.
  Fewer English-speaking analysts cover these companies, leading to:

  1. LANGUAGE BARRIER EDGE: Earnings calls, filings, management commentary
     in Korean/Japanese/Polish/etc. are underanalyzed by global capital.
     This creates delayed price discovery that our model can exploit.

  2. ACCOUNTING STANDARD EDGE: IFRS vs US GAAP vs local GAAP creates
     confusion in comparing companies cross-border. Key differences:
     - IFRS allows revaluation of PP&E (inflates book values)
     - US GAAP requires LIFO option (affects COGS in commodity stocks)
     - Local standards may capitalize exploration costs differently
     - Impairment testing differs (IAS 36 vs ASC 350/360)

  3. ANALYST COVERAGE GAP: Micro/small-cap commodity stocks on foreign
     exchanges often have 0-2 analysts vs 15+ for US large-caps.
     Price discovery is slower, mispricings persist longer.

  4. TIME ZONE & MARKET HOURS: Overnight information from prediction
     markets and futures may not be fully priced into non-US markets
     until the next local open.

This module scores each ticker's "information asymmetry advantage" and
feeds it into position sizing and conviction signals.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# EXCHANGE -> LANGUAGE / ACCOUNTING STANDARD MAPPINGS
# ────────────────────────────────────────────────────────────────────────────

# Exchange suffix -> (country, primary_language, accounting_standard, time_zone_offset_from_ny)
EXCHANGE_META: dict[str, dict[str, Any]] = {
    # English-speaking (low language edge, high analyst coverage)
    "": {"country": "US", "lang": "English", "acct_std": "US_GAAP", "tz_offset": 0, "lang_score": 0.0},
    ".TO": {"country": "Canada", "lang": "English", "acct_std": "IFRS", "tz_offset": 0, "lang_score": 0.05},
    ".V": {"country": "Canada", "lang": "English", "acct_std": "IFRS", "tz_offset": 0, "lang_score": 0.05},
    ".L": {"country": "UK", "lang": "English", "acct_std": "IFRS", "tz_offset": 5, "lang_score": 0.0},
    ".AX": {"country": "Australia", "lang": "English", "acct_std": "IFRS", "tz_offset": 15, "lang_score": 0.05},

    # Major non-English (moderate language edge)
    ".PA": {"country": "France", "lang": "French", "acct_std": "IFRS", "tz_offset": 6, "lang_score": 0.15},
    ".DE": {"country": "Germany", "lang": "German", "acct_std": "IFRS", "tz_offset": 6, "lang_score": 0.15},
    ".MI": {"country": "Italy", "lang": "Italian", "acct_std": "IFRS", "tz_offset": 6, "lang_score": 0.18},
    ".MC": {"country": "Spain", "lang": "Spanish", "acct_std": "IFRS", "tz_offset": 6, "lang_score": 0.12},
    ".OL": {"country": "Norway", "lang": "Norwegian", "acct_std": "IFRS", "tz_offset": 6, "lang_score": 0.20},
    ".ST": {"country": "Sweden", "lang": "Swedish", "acct_std": "IFRS", "tz_offset": 6, "lang_score": 0.18},
    ".HE": {"country": "Finland", "lang": "Finnish", "acct_std": "IFRS", "tz_offset": 7, "lang_score": 0.22},

    # High language edge (fewer English-speaking analysts)
    ".T": {"country": "Japan", "lang": "Japanese", "acct_std": "J_GAAP", "tz_offset": 14, "lang_score": 0.35},
    ".KS": {"country": "South Korea", "lang": "Korean", "acct_std": "K_IFRS", "tz_offset": 14, "lang_score": 0.35},
    ".NS": {"country": "India", "lang": "Hindi/English", "acct_std": "IND_AS", "tz_offset": 10.5, "lang_score": 0.15},
    ".SI": {"country": "Singapore", "lang": "English", "acct_std": "IFRS", "tz_offset": 13, "lang_score": 0.05},
    ".HK": {"country": "Hong Kong", "lang": "Cantonese/English", "acct_std": "HKFRS", "tz_offset": 13, "lang_score": 0.20},

    # Very high language edge (minimal English coverage)
    ".WA": {"country": "Poland", "lang": "Polish", "acct_std": "IFRS", "tz_offset": 6, "lang_score": 0.40},
    ".BU": {"country": "Romania", "lang": "Romanian", "acct_std": "IFRS", "tz_offset": 7, "lang_score": 0.45},
    ".SA": {"country": "Brazil", "lang": "Portuguese", "acct_std": "BR_GAAP", "tz_offset": 2, "lang_score": 0.25},
    ".MX": {"country": "Mexico", "lang": "Spanish", "acct_std": "IFRS", "tz_offset": 1, "lang_score": 0.18},
    ".IS": {"country": "Turkey", "lang": "Turkish", "acct_std": "TFRS", "tz_offset": 8, "lang_score": 0.40},
    ".JK": {"country": "Indonesia", "lang": "Indonesian", "acct_std": "IFRS", "tz_offset": 12, "lang_score": 0.42},
    ".KL": {"country": "Malaysia", "lang": "Malay", "acct_std": "MFRS", "tz_offset": 13, "lang_score": 0.30},
}

# Accounting standard confusion matrix (higher = more confusion vs US GAAP)
ACCOUNTING_CONFUSION: dict[str, float] = {
    "US_GAAP": 0.0,      # no confusion for US companies
    "IFRS": 0.10,         # well-understood, but key differences exist
    "J_GAAP": 0.35,       # Japanese GAAP is materially different (consolidation, goodwill)
    "K_IFRS": 0.15,       # Korean IFRS plus local overlay
    "IND_AS": 0.20,       # Indian IFRS convergence with local modifications
    "HKFRS": 0.12,        # HK mirrors IFRS closely
    "BR_GAAP": 0.25,      # Brazilian standards have unique aspects
    "TFRS": 0.30,         # Turkish standards complex, inflation accounting
    "MFRS": 0.15,         # Malaysian IFRS-converged
}


@dataclass(slots=True)
class TranslationEdge:
    """Information asymmetry score for a single ticker."""
    ticker: str
    exchange_suffix: str
    country: str
    primary_language: str
    accounting_standard: str

    # Component scores (0 to 1)
    language_barrier_score: float   # higher = more information asymmetry from language
    accounting_confusion_score: float  # higher = more accounting standard differences
    timezone_edge_score: float      # higher = more overnight info gap
    coverage_gap_score: float       # higher = fewer analysts

    # Composite
    total_asymmetry_score: float    # 0 to 1, weighted composite
    position_bonus: float           # multiplier for position sizing (1.0 to 1.4)

    # Key differences to watch
    accounting_notes: list[str]


# ────────────────────────────────────────────────────────────────────────────
# ACCOUNTING DIFFERENCES THAT MATTER FOR COMMODITY STOCKS
# ────────────────────────────────────────────────────────────────────────────

COMMODITY_ACCOUNTING_NOTES: dict[str, list[str]] = {
    "J_GAAP": [
        "Japan: goodwill amortized (20yr max) vs US GAAP impairment-only -> inflates ROIC",
        "Japan: exploration costs may be expensed vs capitalized differently",
        "Japan: cross-holdings at cost, not fair value -> hidden value",
    ],
    "IFRS": [
        "IFRS: PP&E revaluation allowed -> book values may be higher",
        "IFRS: no LIFO -> COGS differs in rising commodity environment",
        "IFRS: IAS 36 impairment reversal allowed (US GAAP does not allow)",
    ],
    "K_IFRS": [
        "Korea: IFRS base but local large-group (chaebol) consolidation complexity",
        "Korea: operating lease treatment may differ from US ASC 842",
    ],
    "BR_GAAP": [
        "Brazil: inflation-adjusted financials under IFRS since 2010",
        "Brazil: different tax incentive accounting (SUDENE/SUDAM)",
        "Brazil: complex Petrobras-style state-enterprise structures",
    ],
    "IND_AS": [
        "India: different related party disclosure levels",
        "India: forex gain/loss treatment differences for import-heavy companies",
    ],
    "TFRS": [
        "Turkey: IAS 29 hyperinflation accounting applied",
        "Turkey: real vs nominal financials diverge significantly",
    ],
}

# Analyst coverage by country (approximate number of stocks with >5 covering analysts)
# Lower coverage = bigger price discovery delay = bigger edge
COUNTRY_COVERAGE_DEPTH: dict[str, float] = {
    "US": 0.0,         # highest coverage, no gap
    "UK": 0.05,
    "Canada": 0.08,
    "Australia": 0.10,
    "Japan": 0.25,
    "South Korea": 0.30,
    "France": 0.10,
    "Germany": 0.10,
    "Norway": 0.20,
    "Sweden": 0.15,
    "Brazil": 0.20,
    "India": 0.18,
    "Singapore": 0.12,
    "Hong Kong": 0.15,
    "Poland": 0.35,
    "Romania": 0.42,
    "Turkey": 0.35,
    "Indonesia": 0.35,
    "Malaysia": 0.25,
    "Mexico": 0.22,
    "Italy": 0.15,
    "Spain": 0.12,
    "Finland": 0.22,
}


# ────────────────────────────────────────────────────────────────────────────
# MAIN SCORING
# ────────────────────────────────────────────────────────────────────────────

def score_translation_edge(ticker: str, market_cap: float = 2e9) -> TranslationEdge:
    """Score the information asymmetry for a given ticker.

    Higher asymmetry = slower price discovery = bigger opportunity for
    systematic models that can process cross-language/cross-standard data.

    The position_bonus feeds into risk_manager.py's position sizing:
    a higher asymmetry score means the mispricing persists longer,
    so we can hold with more conviction.
    """
    suffix = _extract_suffix(ticker)
    meta = EXCHANGE_META.get(suffix, EXCHANGE_META[""])

    country = meta["country"]
    lang = meta["lang"]
    acct_std = meta["acct_std"]

    # Language barrier
    lang_score = float(meta["lang_score"])

    # Accounting confusion
    acct_score = ACCOUNTING_CONFUSION.get(acct_std, 0.15)

    # Timezone edge (overnight info gap)
    tz_offset = abs(float(meta["tz_offset"]))
    # Normalize: 0 hours = 0, 14 hours = max edge ~0.35
    tz_score = min(0.35, tz_offset / 40.0)

    # Coverage gap
    coverage_gap = COUNTRY_COVERAGE_DEPTH.get(country, 0.20)
    # Small-cap bonus: micro-caps have even less coverage everywhere
    if market_cap < 500_000_000:
        coverage_gap = min(1.0, coverage_gap + 0.15)
    elif market_cap < 2_000_000_000:
        coverage_gap = min(1.0, coverage_gap + 0.05)

    # Composite: weighted sum
    total = (
        0.35 * lang_score       # language is the biggest barrier
        + 0.25 * acct_score     # accounting differences create confusion
        + 0.15 * tz_score       # timezone gap allows overnight positioning
        + 0.25 * coverage_gap   # fewer analysts = slower price discovery
    )
    total = min(1.0, total)

    # Position sizing bonus: 1.0 (US large-cap) to 1.4 (foreign micro-cap)
    pos_bonus = 1.0 + total * 0.4

    # Accounting notes
    notes = COMMODITY_ACCOUNTING_NOTES.get(acct_std, [])
    if not notes and acct_std != "US_GAAP":
        notes = [f"{acct_std}: review local standard differences vs US GAAP for commodity accounting"]

    return TranslationEdge(
        ticker=ticker,
        exchange_suffix=suffix,
        country=country,
        primary_language=lang,
        accounting_standard=acct_std,
        language_barrier_score=round(lang_score, 3),
        accounting_confusion_score=round(acct_score, 3),
        timezone_edge_score=round(tz_score, 3),
        coverage_gap_score=round(coverage_gap, 3),
        total_asymmetry_score=round(total, 3),
        position_bonus=round(pos_bonus, 3),
        accounting_notes=notes,
    )


def _extract_suffix(ticker: str) -> str:
    """Extract exchange suffix from ticker: 'NTPC.NS' -> '.NS', 'XOM' -> ''."""
    if "." in ticker:
        parts = ticker.rsplit(".", 1)
        if len(parts) == 2 and parts[1].isalpha() and len(parts[1]) <= 3:
            return "." + parts[1]
    return ""


def batch_score_translation_edges(
    tickers: list[str],
    market_caps: dict[str, float] | None = None,
) -> dict[str, TranslationEdge]:
    """Score translation edge for a batch of tickers."""
    results: dict[str, TranslationEdge] = {}
    for t in tickers:
        mc = (market_caps or {}).get(t, 2_000_000_000)
        results[t] = score_translation_edge(t, mc)
    return results


# ────────────────────────────────────────────────────────────────────────────
# PREDICTION MARKET / FUTURES / OPTIONS INFORMATION SOURCES
# ────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class InformationSourceMeta:
    """Metadata about external information sources that provide edge.

    These are NOT directly traded — they INFORM our equity positions.
    The key insight: prediction markets, futures curves, and options skew
    contain forward-looking information that may not yet be priced into
    individual commodity equities, especially foreign-listed ones where
    information travels slower.
    """
    source_type: str           # "prediction_market", "futures_curve", "options_spread", "commodity_spot"
    description: str
    edge_mechanism: str        # HOW this creates alpha
    applicable_sectors: list[str]
    data_refresh_hours: int
    example_signals: list[str]


# Catalog of information sources that feed the decision engine
INFORMATION_SOURCES: list[InformationSourceMeta] = [
    InformationSourceMeta(
        source_type="futures_curve",
        description="Commodity futures term structure (contango/backwardation)",
        edge_mechanism=(
            "Futures curves predict spot price direction better than analyst estimates. "
            "Backwardation signals tightening supply -> bullish for producers. "
            "Contango signals oversupply -> bearish. Already captured in EVENT_MAPPINGS "
            "via CL=F, GC=F, NG=F, BZ=F, SI=F etc. The model uses 10/20/60-day "
            "momentum of these futures to predict equity returns."
        ),
        applicable_sectors=["oil_gas_upstream", "oil_refining", "oil_services", "midstream",
                           "precious_metals", "uranium", "lithium", "rare_earths"],
        data_refresh_hours=1,
        example_signals=[
            "WTI front-month vs 6-month spread narrowing -> anticipate E&P rally",
            "Gold futures curve steep contango -> miners may lag spot price",
            "Lithium hydroxide futures in backwardation -> lithium miners bullish",
        ],
    ),
    InformationSourceMeta(
        source_type="options_spread",
        description="Options market implied volatility and skew",
        edge_mechanism=(
            "Options skew (put/call IV ratio) reveals institutional hedging activity. "
            "Heavy put buying on commodity ETFs (XOP, GDX) before earnings/events = bearish signal. "
            "VIX term structure inversion = risk-off, commodity stocks hit hardest. "
            "Already captured via ^VIX in EVENT_MAPPINGS. Can extend to individual "
            "stock put/call ratios for conviction scoring."
        ),
        applicable_sectors=["oil_gas_upstream", "oil_services", "precious_metals"],
        data_refresh_hours=4,
        example_signals=[
            "XOP put/call ratio spikes above 1.5 -> sector-wide bearish positioning",
            "GDX IV30 > IV60 -> short-term fear in gold miners, potential reversal setup",
            "VIX above 25 + contango -> risk-off regime, reduce all positions",
        ],
    ),
    InformationSourceMeta(
        source_type="prediction_market",
        description="Geopolitical and policy prediction markets (Polymarket, Kalshi, Metaculus)",
        edge_mechanism=(
            "Prediction markets aggregate crowd intelligence on events that move commodity prices: "
            "sanctions probability, OPEC decisions, elections affecting energy policy, "
            "trade war escalation. These resolve BEFORE the event, providing early signal. "
            "Key: the equity market prices events with a lag vs prediction markets, "
            "especially for non-US stocks where information travels slower (our translation edge)."
        ),
        applicable_sectors=["oil_gas_upstream", "oil_refining", "oil_services", "midstream",
                           "uranium", "rare_earths"],
        data_refresh_hours=6,
        example_signals=[
            "Polymarket: 'OPEC production cut by Dec 2025' > 70% -> bullish oil upstream",
            "Kalshi: 'US bans Russian uranium by 2026' > 60% -> bullish UUUU, CCJ",
            "Metaculus: 'US-China rare earth restrictions' > 50% -> bullish MP, REMX",
        ],
    ),
    InformationSourceMeta(
        source_type="commodity_spot",
        description="Physical commodity spot prices and production data",
        edge_mechanism=(
            "Production/consumption contracts in EVENT_MAPPINGS: HG=F (copper demand proxy), "
            "ALI=F (aluminum), ZN=F (zinc), ZS=F (soybeans for biofuel), "
            "KC=F (coffee as EM economy indicator), CT=F (cotton for trade flow). "
            "Physical market tightness signals that equities haven't fully priced. "
            "Especially powerful for non-US producers where local exchange prices lag."
        ),
        applicable_sectors=["oil_gas_upstream", "oil_refining", "midstream",
                           "lithium", "rare_earths", "precious_metals"],
        data_refresh_hours=1,
        example_signals=[
            "Copper spot premium over LME + backwardation -> supply crunch for base metals",
            "Lithium carbonate spot down 40% but futures curve flattening -> cycle bottom",
            "Gold/silver ratio above 80 -> silver underpriced vs gold, bullish silver miners",
        ],
    ),
]


def get_information_source_signals(commodity_type: str) -> list[InformationSourceMeta]:
    """Get relevant information sources for a given commodity type."""
    return [
        src for src in INFORMATION_SOURCES
        if commodity_type in src.applicable_sectors
    ]
