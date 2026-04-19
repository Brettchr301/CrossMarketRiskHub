"""Narrative Bias Detector for prediction markets.

Detects systematic directional mispricing across multiple competitive races.
The 2022 "red wave" signal: Polymarket underpriced Dems by 51-74pp on 9/9 races.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.election.db.models import (
    BlendedProbability,
    MarketContract,
    MarketQuote,
    PollingData,
    Race,
)
from app.election.db.historical_models import HistoricalQuote, RaceOutcome
from app.election.mappings.direction_detector import detect_direction, normalize_price

logger = logging.getLogger(__name__)

STRENGTH_THRESHOLDS = [
    # (min_skew, min_avg_mispricing_pp, label)
    (0.85, 40, "extreme"),
    (0.75, 25, "strong"),
    (0.65, 10, "moderate"),
    (0.55, 0, "weak"),
]


@dataclass
class RaceMispricing:
    """Individual race mispricing assessment."""
    race_id: int
    state: str
    race_type: str
    cycle: int
    platform: str
    market_prob_dem: float
    polling_avg_dem: float | None
    fundamental_lean: float | None
    market_vs_polls: float | None
    confidence: float


@dataclass
class NarrativeBiasSignal:
    """Population-level bias signal across races."""
    as_of: datetime
    cycle: int
    platform: str
    n_competitive_races: int
    n_dem_underpriced: int
    n_rep_underpriced: int
    skew_ratio: float
    avg_mispricing_pp: float
    max_mispricing_pp: float
    historical_percentile: float
    signal_strength: str
    races: list[RaceMispricing] = field(default_factory=list)
    analog_2022_similarity: float = 0.0


def _classify_strength(skew_ratio: float, avg_mispricing_pp: float) -> str:
    """Classify signal strength from skew ratio and average mispricing."""
    for min_skew, min_mispricing, label in STRENGTH_THRESHOLDS:
        if skew_ratio >= min_skew and avg_mispricing_pp >= min_mispricing:
            return label
    return "none"


def detect_narrative_bias(
    db: Session,
    cycle: int,
    platform: str = "polymarket",
    as_of: datetime | None = None,
    competitive_range: tuple[float, float] = (0.25, 0.75),
) -> NarrativeBiasSignal:
    """Detect systematic directional bias across competitive races.

    Steps:
    1. Get latest market quotes for all races in cycle on platform
    2. Normalize to P(Dem wins) using direction_detector
    3. Filter to competitive races (prob in competitive_range)
    4. Get polling averages for same races
    5. Compare market vs polls: negative = market underprices Dem
    6. Count directional skew across races
    7. Classify signal strength
    """
    now = as_of or datetime.utcnow()

    # Get all races for this cycle
    races = db.execute(
        select(Race).where(Race.cycle == cycle)
    ).scalars().all()

    if not races:
        return NarrativeBiasSignal(
            as_of=now, cycle=cycle, platform=platform,
            n_competitive_races=0, n_dem_underpriced=0, n_rep_underpriced=0,
            skew_ratio=0.0, avg_mispricing_pp=0.0, max_mispricing_pp=0.0,
            historical_percentile=0.0, signal_strength="none",
        )

    race_mispricings: list[RaceMispricing] = []

    for race in races:
        # Get latest quote for this race on this platform
        contract = db.execute(
            select(MarketContract)
            .where(MarketContract.race_id == race.id, MarketContract.platform == platform)
            .limit(1)
        ).scalar_one_or_none()

        if contract is None:
            continue

        latest_quote = db.execute(
            select(MarketQuote)
            .where(MarketQuote.contract_id == contract.id)
            .order_by(MarketQuote.as_of.desc())
            .limit(1)
        ).scalar_one_or_none()

        if latest_quote is None:
            continue

        # Normalize to P(Dem wins)
        direction = detect_direction(contract.platform_question or "")
        if direction.confidence < 0.5:
            market_prob_dem = latest_quote.last_price  # can't normalize, use raw
        else:
            market_prob_dem = normalize_price(latest_quote.last_price, direction.yes_party)

        # Filter to competitive
        if not (competitive_range[0] <= market_prob_dem <= competitive_range[1]):
            continue

        # Get polling average for this race
        polls = db.execute(
            select(PollingData)
            .where(PollingData.race_id == race.id)
            .order_by(PollingData.poll_date.desc())
            .limit(10)
        ).scalars().all()

        polling_avg_dem = None
        if polls:
            # Average the most recent polls (simplified — in production would weight by recency)
            polling_avg_dem = sum(p.pct for p in polls) / len(polls) / 100.0

        market_vs_polls = None
        if polling_avg_dem is not None:
            market_vs_polls = (market_prob_dem - polling_avg_dem) * 100  # in pp

        race_mispricings.append(RaceMispricing(
            race_id=race.id,
            state=race.state,
            race_type=race.race_type,
            cycle=cycle,
            platform=platform,
            market_prob_dem=market_prob_dem,
            polling_avg_dem=polling_avg_dem,
            fundamental_lean=None,
            market_vs_polls=market_vs_polls,
            confidence=direction.confidence if direction.confidence >= 0.5 else 0.3,
        ))

    # Compute population-level metrics
    n_competitive = len(race_mispricings)
    if n_competitive == 0:
        return NarrativeBiasSignal(
            as_of=now, cycle=cycle, platform=platform,
            n_competitive_races=0, n_dem_underpriced=0, n_rep_underpriced=0,
            skew_ratio=0.0, avg_mispricing_pp=0.0, max_mispricing_pp=0.0,
            historical_percentile=0.0, signal_strength="none",
        )

    # Count directional skew (races where market < polls for Dem)
    races_with_polls = [r for r in race_mispricings if r.market_vs_polls is not None]
    n_dem_underpriced = sum(1 for r in races_with_polls if r.market_vs_polls < 0)
    n_rep_underpriced = sum(1 for r in races_with_polls if r.market_vs_polls > 0)

    n_with_polls = len(races_with_polls)
    skew_ratio = n_dem_underpriced / n_with_polls if n_with_polls > 0 else 0.0

    gaps = [abs(r.market_vs_polls) for r in races_with_polls if r.market_vs_polls is not None]
    avg_mispricing = float(np.mean(gaps)) if gaps else 0.0
    max_mispricing = float(np.max(gaps)) if gaps else 0.0

    strength = _classify_strength(skew_ratio, avg_mispricing)
    similarity = score_2022_similarity_from_metrics(skew_ratio, avg_mispricing, n_competitive)

    return NarrativeBiasSignal(
        as_of=now,
        cycle=cycle,
        platform=platform,
        n_competitive_races=n_competitive,
        n_dem_underpriced=n_dem_underpriced,
        n_rep_underpriced=n_rep_underpriced,
        skew_ratio=round(skew_ratio, 3),
        avg_mispricing_pp=round(avg_mispricing, 1),
        max_mispricing_pp=round(max_mispricing, 1),
        historical_percentile=0.0,  # needs historical distribution
        signal_strength=strength,
        races=race_mispricings,
        analog_2022_similarity=round(similarity, 3),
    )


def detect_narrative_bias_historical(
    db: Session,
    cycle: int,
    platform: str = "polymarket",
    days_before_election: int = 1,
    competitive_range: tuple[float, float] = (0.25, 0.75),
) -> NarrativeBiasSignal:
    """Detect narrative bias from historical data (for backtesting).

    Uses HistoricalQuote + RaceOutcome instead of live MarketQuote.
    """
    # Get outcomes for this cycle to know election dates
    outcomes = db.execute(
        select(RaceOutcome).where(RaceOutcome.cycle == cycle)
    ).scalars().all()

    if not outcomes:
        return NarrativeBiasSignal(
            as_of=datetime.utcnow(), cycle=cycle, platform=platform,
            n_competitive_races=0, n_dem_underpriced=0, n_rep_underpriced=0,
            skew_ratio=0.0, avg_mispricing_pp=0.0, max_mispricing_pp=0.0,
            historical_percentile=0.0, signal_strength="none",
        )

    race_mispricings: list[RaceMispricing] = []

    for outcome in outcomes:
        election_dt = datetime.combine(outcome.election_date, datetime.min.time())
        target_dt = election_dt - timedelta(days=days_before_election)

        # Get historical quote closest to target date
        quote = db.execute(
            select(HistoricalQuote)
            .where(
                HistoricalQuote.race_id == outcome.race_id,
                HistoricalQuote.platform == platform,
                HistoricalQuote.as_of <= election_dt,
                HistoricalQuote.as_of >= target_dt - timedelta(days=1),
            )
            .order_by(HistoricalQuote.as_of.desc())
            .limit(1)
        ).scalar_one_or_none()

        if quote is None:
            continue

        # Normalize price
        direction = detect_direction(quote.question)
        if direction.confidence >= 0.5:
            market_prob_dem = normalize_price(quote.price, direction.yes_party)
        else:
            market_prob_dem = quote.price

        if not (competitive_range[0] <= market_prob_dem <= competitive_range[1]):
            continue

        # Actual outcome as "true probability" (1.0 or 0.0)
        actual = 1.0 if outcome.winner_party == "D" else 0.0
        error_pp = (actual - market_prob_dem) * 100  # positive = market underpriced Dem

        race_mispricings.append(RaceMispricing(
            race_id=outcome.race_id or 0,
            state=outcome.state,
            race_type=outcome.race_type,
            cycle=cycle,
            platform=platform,
            market_prob_dem=market_prob_dem,
            polling_avg_dem=None,
            fundamental_lean=None,
            market_vs_polls=error_pp,  # repurpose: market_vs_actual in pp
            confidence=direction.confidence,
        ))

    n_competitive = len(race_mispricings)
    if n_competitive == 0:
        return NarrativeBiasSignal(
            as_of=datetime.utcnow(), cycle=cycle, platform=platform,
            n_competitive_races=0, n_dem_underpriced=0, n_rep_underpriced=0,
            skew_ratio=0.0, avg_mispricing_pp=0.0, max_mispricing_pp=0.0,
            historical_percentile=0.0, signal_strength="none",
        )

    n_dem_underpriced = sum(1 for r in race_mispricings if (r.market_vs_polls or 0) > 0)
    n_rep_underpriced = sum(1 for r in race_mispricings if (r.market_vs_polls or 0) < 0)

    skew_ratio = n_dem_underpriced / n_competitive if n_competitive > 0 else 0.0
    gaps = [abs(r.market_vs_polls) for r in race_mispricings if r.market_vs_polls is not None]
    avg_mispricing = float(np.mean(gaps)) if gaps else 0.0
    max_mispricing = float(np.max(gaps)) if gaps else 0.0

    strength = _classify_strength(skew_ratio, avg_mispricing)
    similarity = score_2022_similarity_from_metrics(skew_ratio, avg_mispricing, n_competitive)

    return NarrativeBiasSignal(
        as_of=datetime.utcnow(),
        cycle=cycle,
        platform=platform,
        n_competitive_races=n_competitive,
        n_dem_underpriced=n_dem_underpriced,
        n_rep_underpriced=n_rep_underpriced,
        skew_ratio=round(skew_ratio, 3),
        avg_mispricing_pp=round(avg_mispricing, 1),
        max_mispricing_pp=round(max_mispricing, 1),
        historical_percentile=0.0,
        signal_strength=strength,
        races=race_mispricings,
        analog_2022_similarity=round(similarity, 3),
    )


def score_2022_similarity_from_metrics(
    skew_ratio: float,
    avg_mispricing_pp: float,
    n_competitive: int,
) -> float:
    """Score how similar a signal is to the 2022 red wave pattern.

    2022 benchmarks:
    - skew_ratio ≈ 1.0 (9/9 races)
    - avg_mispricing ≈ 59pp
    - n_competitive ≈ 9

    Returns 0-1 similarity.
    """
    # Skew similarity (weight: 0.4)
    skew_sim = min(1.0, skew_ratio / 1.0) * 0.4

    # Mispricing magnitude similarity (weight: 0.35)
    mispricing_sim = min(1.0, avg_mispricing_pp / 59.0) * 0.35

    # Race count similarity (weight: 0.25)
    count_sim = min(1.0, n_competitive / 9.0) * 0.25

    return round(skew_sim + mispricing_sim + count_sim, 3)


def backtest_narrative_signal(
    db: Session,
    cycles: list[int] | None = None,
    platform: str = "polymarket",
    entry_days_before: int = 7,
) -> dict[str, Any]:
    """Backtest: if we'd bought the underpriced party when signal fired, what was PnL?"""
    if cycles is None:
        cycles = [2018, 2020, 2022, 2024]

    all_signals: list[dict[str, Any]] = []
    all_trades: list[dict[str, Any]] = []

    for cycle in cycles:
        signal = detect_narrative_bias_historical(db, cycle, platform, entry_days_before)

        if signal.signal_strength in ("none", "weak"):
            continue

        signal_info = {
            "cycle": cycle,
            "signal_strength": signal.signal_strength,
            "skew_ratio": signal.skew_ratio,
            "avg_mispricing_pp": signal.avg_mispricing_pp,
            "n_competitive": signal.n_competitive_races,
            "entry_day": -entry_days_before,
        }
        all_signals.append(signal_info)

        # Each underpriced race is a trade
        for race in signal.races:
            if race.market_vs_polls is not None and race.market_vs_polls > 0:
                # Market underpriced Dem → buy Dem
                entry_price = race.market_prob_dem
                settlement = 1.0  # Dem won (since error > 0)
                pnl = settlement - entry_price

                all_trades.append({
                    "cycle": cycle,
                    "race": f"{race.state} {race.race_type}",
                    "entry_price": round(entry_price, 3),
                    "settlement": settlement,
                    "pnl": round(pnl, 3),
                    "won": pnl > 0,
                })

    total_pnl = sum(t["pnl"] for t in all_trades)
    wins = sum(1 for t in all_trades if t["won"])
    win_rate = wins / len(all_trades) if all_trades else 0.0
    avg_edge = float(np.mean([t["pnl"] for t in all_trades]) * 100) if all_trades else 0.0

    return {
        "cycles_tested": cycles,
        "signals_fired": all_signals,
        "trades": all_trades,
        "total_pnl": round(total_pnl, 3),
        "n_trades": len(all_trades),
        "win_rate": round(win_rate, 3),
        "avg_edge_pp": round(avg_edge, 1),
    }
