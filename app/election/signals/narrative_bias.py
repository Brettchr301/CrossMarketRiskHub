"""Narrative bias detector for prediction markets.

Detects systematic directional mispricing across multiple competitive races.
The 2022 "red wave" signal: Polymarket underpriced Dems by 51-74pp on 9/9 races.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Any, Optional
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session, aliased
from sqlalchemy import func, and_, or_, desc

# Import existing direction detector
from app.election.signals.direction_detector import detect_direction, normalize_price

# Import existing models
from app.models import HistoricalQuote, Race, PollingData, Outcome


@dataclass
class RaceMispricing:
    """Individual race mispricing assessment."""
    race_id: int
    state: str
    race_type: str
    cycle: int
    platform: str
    market_prob_dem: float      # Market-implied P(Dem wins), normalized
    polling_avg_dem: float | None  # Polling average for Dem candidate (0-1)
    fundamental_lean: float | None  # PVI or fundamentals-based lean (-1 R to +1 D)
    market_vs_polls: float | None   # market_prob - polling_avg (negative = market underprices Dem)
    confidence: float           # 0-1, based on data quality


@dataclass
class NarrativeBiasSignal:
    """Population-level bias signal across races."""
    as_of: datetime
    cycle: int
    platform: str
    n_competitive_races: int    # races with prob in [0.25, 0.75]
    n_dem_underpriced: int      # competitive races where market < polls for Dem
    n_rep_underpriced: int      # competitive races where market < polls for Rep
    skew_ratio: float           # n_dem_underpriced / n_competitive (0-1, >0.6 = Dem underpriced)
    avg_mispricing_pp: float    # average market-vs-polls gap in percentage points
    max_mispricing_pp: float    # worst single race gap
    historical_percentile: float  # where this skew falls vs 2018-2024 distribution
    signal_strength: str        # "none", "weak", "moderate", "strong", "extreme"
    races: list[RaceMispricing] = field(default_factory=list)
    analog_2022_similarity: float = 0.0  # 0-1 how similar to 2022 red wave pattern


def _get_latest_quotes(
    db: Session, 
    cycle: int, 
    platform: str, 
    as_of: datetime | None = None
) -> list[HistoricalQuote]:
    """Get latest market quotes for all races in cycle on platform."""
    if as_of is None:
        as_of = datetime.now()
    
    # Get the most recent quote for each race before as_of
    subquery = (
        db.query(
            HistoricalQuote.race_id,
            func.max(HistoricalQuote.timestamp).label('max_timestamp')
        )
        .filter(
            HistoricalQuote.cycle == cycle,
            HistoricalQuote.platform == platform,
            HistoricalQuote.timestamp <= as_of
        )
        .group_by(HistoricalQuote.race_id)
        .subquery()
    )
    
    quotes = (
        db.query(HistoricalQuote)
        .join(
            subquery,
            and_(
                HistoricalQuote.race_id == subquery.c.race_id,
                HistoricalQuote.timestamp == subquery.c.max_timestamp
            )
        )
        .all()
    )
    
    return quotes


def _get_polling_avg(
    db: Session, 
    race_id: int, 
    as_of: datetime,
    lookback_days: int = 30
) -> float | None:
    """Get polling average for a race around as_of date."""
    cutoff = as_of - timedelta(days=lookback_days)
    
    polls = (
        db.query(PollingData)
        .filter(
            PollingData.race_id == race_id,
            PollingData.date >= cutoff,
            PollingData.date <= as_of
        )
        .order_by(PollingData.date.desc())
        .all()
    )
    
    if not polls:
        return None
    
    # Simple average of polling values
    dem_values = [p.dem_poll for p in polls if p.dem_poll is not None]
    if not dem_values:
        return None
    
    return sum(dem_values) / len(dem_values)


def _classify_signal_strength(skew_ratio: float, avg_mispricing_pp: float) -> str:
    """Classify signal strength based on skew ratio and average mispricing."""
    if skew_ratio < 0.55:
        return "none"
    elif 0.55 <= skew_ratio < 0.65:
        return "weak" if abs(avg_mispricing_pp) < 10 else "moderate"
    elif 0.65 <= skew_ratio < 0.75:
        return "moderate" if abs(avg_mispricing_pp) < 25 else "strong"
    elif 0.75 <= skew_ratio < 0.85:
        return "strong" if abs(avg_mispricing_pp) < 40 else "extreme"
    else:  # skew_ratio >= 0.85
        return "extreme" if abs(avg_mispricing_pp) >= 40 else "strong"


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
    5. Compare market vs polls: positive = market overprices Dem, negative = underprices
    6. Count directional skew across races
    7. Compare to historical distribution of skew ratios
    8. Classify signal strength
    """
    if as_of is None:
        as_of = datetime.now()
    
    # Step 1: Get latest quotes
    quotes = _get_latest_quotes(db, cycle, platform, as_of)
    
    competitive_races = []
    mispricing_gaps = []
    dem_underpriced_count = 0
    rep_underpriced_count = 0
    
    # Steps 2-5: Process each quote
    for quote in quotes:
        # Get race details
        race = db.query(Race).filter(Race.id == quote.race_id).first()
        if not race:
            continue
        
        # Normalize to P(Dem wins)
        direction = detect_direction(race.state, race.race_type)
        prob_dem = normalize_price(quote.price, direction)
        
        # Step 3: Check if competitive
        low, high = competitive_range
        if not (low <= prob_dem <= high):
            continue
        
        # Step 4: Get polling average
        polling_avg = _get_polling_avg(db, quote.race_id, as_of)
        
        # Step 5: Compare market vs polls
        market_vs_polls = None
        if polling_avg is not None:
            market_vs_polls = prob_dem - polling_avg
            mispricing_gaps.append(market_vs_polls)
            
            if market_vs_polls < 0:
                dem_underpriced_count += 1
            elif market_vs_polls > 0:
                rep_underpriced_count += 1
        
        # Create race mispricing object
        race_mispricing = RaceMispricing(
            race_id=quote.race_id,
            state=race.state,
            race_type=race.race_type,
            cycle=cycle,
            platform=platform,
            market_prob_dem=prob_dem,
            polling_avg_dem=polling_avg,
            fundamental_lean=None,  # Could be implemented later
            market_vs_polls=market_vs_polls,
            confidence=0.8 if polling_avg is not None else 0.3
        )
        competitive_races.append(race_mispricing)
    
    # Step 6: Compute skew metrics
    n_competitive = len(competitive_races)
    if n_competitive == 0:
        skew_ratio = 0.5
        avg_mispricing_pp = 0.0
        max_mispricing_pp = 0.0
    else:
        skew_ratio = dem_underpriced_count / n_competitive
        if mispricing_gaps:
            avg_mispricing_pp = abs(np.mean(mispricing_gaps)) * 100
            max_mispricing_pp = abs(max(mispricing_gaps, key=abs)) * 100
        else:
            avg_mispricing_pp = 0.0
            max_mispricing_pp = 0.0
    
    # Step 7: Compute historical percentile (simplified)
    # In practice, this would query historical data
    historical_percentile = 0.5  # Placeholder
    
    # Step 8: Classify signal strength
    signal_strength = _classify_signal_strength(skew_ratio, avg_mispricing_pp)
    
    # Create signal
    signal = NarrativeBiasSignal(
        as_of=as_of,
        cycle=cycle,
        platform=platform,
        n_competitive_races=n_competitive,
        n_dem_underpriced=dem_underpriced_count,
        n_rep_underpriced=rep_underpriced_count,
        skew_ratio=skew_ratio,
        avg_mispricing_pp=avg_mispricing_pp,
        max_mispricing_pp=max_mispricing_pp,
        historical_percentile=historical_percentile,
        signal_strength=signal_strength,
        races=competitive_races
    )
    
    # Compute 2022 similarity
    signal.analog_2022_similarity = score_2022_similarity(signal)
    
    return signal


def compute_historical_skew_distribution(
    db: Session,
    cycles: list[int] = [2018, 2020, 2022, 2024],
    platform: str = "polymarket",
    days_before_election: list[int] = [1, 3, 7, 14, 30],
) -> pd.DataFrame:
    """Compute historical skew ratios at various time horizons.

    Returns DataFrame: cycle, days_before, skew_ratio, n_competitive, avg_mispricing_pp
    Used to calibrate what's "normal" vs "extreme" skew.
    """
    results = []
    
    for cycle in cycles:
        # Get election date for this cycle
        election_race = db.query(Race).filter(Race.cycle == cycle).first()
        if not election_race or not election_race.election_date:
            continue
        
        election_date = election_race.election_date
        
        for days_before in days_before_election:
            as_of_date = election_date - timedelta(days=days_before)
            
            # Get signal for this historical date
            signal = detect_narrative_bias(
                db=db,
                cycle=cycle,
                platform=platform,
                as_of=datetime.combine(as_of_date, datetime.min.time())
            )
            
            results.append({
                'cycle': cycle,
                'days_before': days_before,
                'skew_ratio': signal.skew_ratio,
                'n_competitive': signal.n_competitive_races,
                'avg_mispricing_pp': signal.avg_mispricing_pp
            })
    
    return pd.DataFrame(results)


def backtest_narrative_signal(
    db: Session,
    cycles: list[int] = [2018, 2020, 2022, 2024],
    platform: str = "polymarket",
    entry_days_before: int = 7,
    signal_threshold: str = "moderate",
) -> dict[str, Any]:
    """Backtest: if we'd bought the underpriced party when signal fired, what was PnL?

    Returns:
    {
        "cycles_tested": [2018, 2020, 2022, 2024],
        "signals_fired": [
            {"cycle": 2022, "signal_strength": "strong", "skew_ratio": 0.89,
             "avg_mispricing_pp": 62.3, "entry_day": -7},
        ],
        "trades": [
            {"cycle": 2022, "race": "PA Senate", "entry_price": 0.38,
             "settlement": 1.0, "pnl": 0.62, "won": True},
            ...
        ],
        "total_pnl": 5.42,
        "win_rate": 1.0,
        "avg_edge_pp": 58.7,
        "sharpe": 4.2,
    }
    """
    signal_strength_map = {"none": 0, "weak": 1, "moderate": 2, "strong": 3, "extreme": 4}
    threshold_level = signal_strength_map.get(signal_threshold, 2)
    
    signals_fired = []
    trades = []
    total_pnl = 0.0
    total_trades = 0
    winning_trades = 0
    
    for cycle in cycles:
        # Get election date
        election_race = db.query(Race).filter(Race.cycle == cycle).first()
        if not election_race or not election_race.election_date:
            continue
        
        election_date = election_race.election_date
        entry_date = election_date - timedelta(days=entry_days_before)
        
        # Get signal at entry date
        signal = detect_narrative_bias(
            db=db,
            cycle=cycle,
            platform=platform,
            as_of=datetime.combine(entry_date, datetime.min.time())
        )
        
        # Check if signal meets threshold
        current_strength = signal_strength_map.get(signal.signal_strength, 0)
        if current_strength < threshold_level:
            continue
        
        # Record signal
        signals_fired.append({
            "cycle": cycle,
            "signal_strength": signal.signal_strength,
            "skew_ratio": signal.skew_ratio,
            "avg_mispricing_pp": signal.avg_mispricing_pp,
            "entry_day": -entry_days_before
        })
        
        # Execute trades for underpriced party
        for race_mispricing in signal.races:
            if race_mispricing.market_vs_polls is None:
                continue
            
            # Determine which party is underpriced
            if race_mispricing.market_vs_polls < 0:
                # Dems underpriced, buy Dem contracts
                entry_price = race_mispricing.market_prob_dem
                target_party = "Democrat"
            else:
                # Reps underpriced, buy Rep contracts
                entry_price = 1 - race_mispricing.market_prob_dem
                target_party = "Republican"
            
            # Get actual outcome
            outcome = db.query(Outcome).filter(Outcome.race_id == race_mispricing.race_id).first()
            if not outcome:
                continue
            
            won = (outcome.winner == target_party)
            settlement = 1.0 if won else 0.0
            pnl = settlement - entry_price
            
            trades.append({
                "cycle": cycle,
                "race": f"{race_mispricing.state} {race_mispricing.race_type}",
                "entry_price": round(entry_price, 3),
                "settlement": settlement,
                "pnl": round(pnl, 3),
                "won": won
            })
            
            total_pnl += pnl
            total_trades += 1
            if won:
                winning_trades += 1
    
    # Calculate metrics
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
    
    # Calculate average edge (simplified)
    avg_edge_pp = 0.0
    if trades:
        avg_edge_pp = np.mean([abs(t["settlement"] - t["entry_price"]) * 100 for t in trades])
    
    # Simplified Sharpe ratio (assuming no variance for simplicity)
    sharpe = 0.0
    if total_trades > 0 and win_rate > 0:
        sharpe = (total_pnl / total_trades) / 0.1  # Simplified
    
    return {
        "cycles_tested": cycles,
        "signals_fired": signals_fired,
        "trades": trades,
        "total_pnl": round(total_pnl, 2),
        "win_rate": round(win_rate, 3),
        "avg_edge_pp": round(avg_edge_pp, 1),
        "sharpe": round(sharpe, 2)
    }


def score_2022_similarity(
    current: NarrativeBiasSignal,
) -> float:
    """Score how similar the current signal is to the 2022 red wave pattern.

    Key features of 2022:
    - 9/9 competitive races underpriced Dems
    - Average underpricing was 59pp
    - Skew ratio was ~1.0 (all races same direction)
    - Happened in a midterm with strong media narrative

    Returns 0-1 similarity score.
    """
    # Feature 1: Skew ratio similarity (2022 had ~1.0)
    skew_sim = current.skew_ratio  # Already 0-1
    
    # Feature 2: Mispricing magnitude similarity (2022 had 59pp)
    mispricing_sim = min(current.avg_mispricing_pp / 59.0, 1.0) if current.avg_mispricing_pp > 0 else 0.0
    
    # Feature 3: Signal strength similarity
    strength_map = {"none": 0.0, "weak": 0.25, "moderate": 0.5, "strong": 0.75, "extreme": 1.0}
    strength_sim = strength_map.get(current.signal_strength, 0.0)
    
    # Feature 4: Number of competitive races (2022 had 9)
    race_count_sim = min(current.n_competitive_races / 9.0, 1.0) if current.n_competitive_races > 0 else 0.0
    
    # Weighted average
    weights = [0.4, 0.4, 0.1, 0.1]
    similarity = (
        weights[0] * skew_sim +
        weights[1] * mispricing_sim +
        weights[2] * strength_sim +
        weights[3] * race_count_sim
    )
    
    return round(similarity, 3)
