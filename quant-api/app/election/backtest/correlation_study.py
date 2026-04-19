"""Event-study correlation analysis.

Computes:
- Weather turnout_score vs market price correlation per race
- Price volatility pre/post election across cycles
- Price accuracy: how close was the pre-election price to the actual outcome?
- Best/worst markets by accuracy
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from app.election.db.historical_models import HistoricalQuote, RaceOutcome
from app.election.db.timing_models import HourlyWeather, LiveVoteCount

logger = logging.getLogger(__name__)


def compute_price_accuracy(db: Session, cycle: int, hours_before_election: int = 24) -> pd.DataFrame:
    """For each race, compute: market's implied prob (N hours before election) vs actual outcome (0 or 1).

    Returns DataFrame with columns: race_id, state, race_type, final_prob, actual, error, brier_score.
    """
    outcomes = db.execute(select(RaceOutcome).where(RaceOutcome.cycle == cycle)).scalars().all()
    if not outcomes:
        return pd.DataFrame()

    rows = []
    for o in outcomes:
        election_ts = datetime.combine(o.election_date, datetime.min.time()) + timedelta(hours=18)  # 6pm election day
        cutoff = election_ts - timedelta(hours=hours_before_election)

        # Fetch all quotes for this race within 24h before cutoff
        quotes = db.execute(
            select(HistoricalQuote.as_of, HistoricalQuote.price, HistoricalQuote.question)
            .where(
                HistoricalQuote.race_id == o.race_id,
                HistoricalQuote.as_of <= cutoff,
                HistoricalQuote.as_of >= cutoff - timedelta(hours=48),
            )
        ).all()

        if not quotes:
            continue

        # Group by question and average
        df_q = pd.DataFrame([{"ts": q[0], "price": q[1], "q": q[2]} for q in quotes])
        avg_by_q = df_q.groupby("q")["price"].mean()

        # Final prob = max volume market OR average
        final_prob = float(avg_by_q.mean())

        # Ground truth: 1 if winner is D, 0 if R (this is a simplification - YES semantics vary per market)
        # For YES-on-winning-party markets: actual = 1 if winner matches
        # We use a default "D wins" framing
        actual = 1.0 if o.winner_party == "D" else 0.0

        # Brier score (calibration)
        brier = (final_prob - actual) ** 2

        rows.append({
            "race_id": o.race_id,
            "state": o.state,
            "race_type": o.race_type,
            "cycle": o.cycle,
            "winner": o.winner_party,
            "n_markets": len(avg_by_q),
            "final_prob": final_prob,
            "actual": actual,
            "error": final_prob - actual,
            "brier_score": brier,
        })

    return pd.DataFrame(rows)


def weather_price_correlation_per_race(db: Session, cycle: int) -> pd.DataFrame:
    """Compute correlation between hourly weather turnout_score and market mid-price per race.

    Returns DataFrame: race_id, state, race_type, correlation, n_hours.
    """
    outcomes = db.execute(select(RaceOutcome).where(RaceOutcome.cycle == cycle)).scalars().all()
    rows = []

    for o in outcomes:
        election_date = o.election_date
        start = datetime.combine(election_date - timedelta(days=2), datetime.min.time())
        end = datetime.combine(election_date + timedelta(days=2), datetime.min.time())

        # Weather for this state
        weather_q = db.execute(
            select(HourlyWeather.timestamp, HourlyWeather.turnout_score)
            .where(
                HourlyWeather.state == o.state,
                HourlyWeather.timestamp >= start,
                HourlyWeather.timestamp <= end,
            ).order_by(HourlyWeather.timestamp)
        ).all()

        if not weather_q:
            continue

        w_df = pd.DataFrame(weather_q, columns=["ts", "score"]).set_index("ts")
        w_df.index = pd.to_datetime(w_df.index)

        # Market prices for this race
        price_q = db.execute(
            select(HistoricalQuote.as_of, HistoricalQuote.price)
            .where(
                HistoricalQuote.race_id == o.race_id,
                HistoricalQuote.as_of >= start,
                HistoricalQuote.as_of <= end,
            ).order_by(HistoricalQuote.as_of)
        ).all()

        if not price_q:
            continue

        p_df = pd.DataFrame(price_q, columns=["ts", "price"]).set_index("ts")
        p_df.index = pd.to_datetime(p_df.index)
        p_hourly = p_df["price"].resample("1h").mean()

        # Align
        aligned = pd.concat([p_hourly.rename("price"), w_df["score"].rename("weather")], axis=1).dropna()
        if len(aligned) < 5:
            continue

        corr = aligned.corr().loc["price", "weather"]
        rows.append({
            "race_id": o.race_id,
            "state": o.state,
            "race_type": o.race_type,
            "cycle": cycle,
            "correlation": float(corr) if pd.notna(corr) else 0.0,
            "n_hours": len(aligned),
        })

    return pd.DataFrame(rows)


def price_volatility_stats(db: Session, cycle: int) -> pd.DataFrame:
    """Compute pre/post-election price volatility per race.

    Returns DataFrame: race_id, pre_vol, post_vol, max_move, election_night_move.
    """
    outcomes = db.execute(select(RaceOutcome).where(RaceOutcome.cycle == cycle)).scalars().all()
    rows = []

    for o in outcomes:
        election_dt = datetime.combine(o.election_date, datetime.min.time()) + timedelta(hours=18)
        pre_start = election_dt - timedelta(days=30)
        post_end = election_dt + timedelta(days=3)

        quotes = db.execute(
            select(HistoricalQuote.as_of, HistoricalQuote.price)
            .where(
                HistoricalQuote.race_id == o.race_id,
                HistoricalQuote.as_of >= pre_start,
                HistoricalQuote.as_of <= post_end,
            ).order_by(HistoricalQuote.as_of)
        ).all()

        if len(quotes) < 10:
            continue

        df = pd.DataFrame(quotes, columns=["ts", "price"]).set_index("ts")
        df.index = pd.to_datetime(df.index)
        df = df.groupby(level=0).last()

        pre = df.loc[:election_dt]
        post = df.loc[election_dt:]

        pre_vol = float(pre["price"].diff().std()) if len(pre) > 1 else 0.0
        post_vol = float(post["price"].diff().std()) if len(post) > 1 else 0.0

        # Max 1-hour move during election night
        election_night = df.loc[election_dt:election_dt + timedelta(hours=12)]
        if len(election_night) > 1:
            hourly = election_night["price"].resample("1h").mean().dropna()
            max_move = float(hourly.diff().abs().max()) if len(hourly) > 1 else 0.0
        else:
            max_move = 0.0

        # Election night total move (8pm to 6am)
        start_8pm = election_dt + timedelta(hours=2)  # 8pm
        end_6am = election_dt + timedelta(hours=12)  # 6am next day
        night = df.loc[start_8pm:end_6am]
        total_night_move = float(night["price"].iloc[-1] - night["price"].iloc[0]) if len(night) > 1 else 0.0

        rows.append({
            "race_id": o.race_id,
            "state": o.state,
            "race_type": o.race_type,
            "cycle": cycle,
            "pre_vol": pre_vol,
            "post_vol": post_vol,
            "max_1h_move": max_move,
            "election_night_total_move": total_night_move,
            "n_quotes": len(df),
        })

    return pd.DataFrame(rows)


def cross_cycle_summary(db: Session, cycles: list[int] = None) -> dict[str, Any]:
    """Aggregate stats across all cycles."""
    if cycles is None:
        cycles = [2018, 2020, 2022, 2024]

    all_accuracy = []
    all_weather_corr = []
    all_volatility = []

    for cycle in cycles:
        acc = compute_price_accuracy(db, cycle)
        if not acc.empty:
            all_accuracy.append(acc)
        corr = weather_price_correlation_per_race(db, cycle)
        if not corr.empty:
            all_weather_corr.append(corr)
        vol = price_volatility_stats(db, cycle)
        if not vol.empty:
            all_volatility.append(vol)

    summary: dict[str, Any] = {"cycles": cycles}

    if all_accuracy:
        acc_df = pd.concat(all_accuracy, ignore_index=True)
        summary["accuracy"] = {
            "n_races": int(len(acc_df)),
            "mean_brier": float(acc_df["brier_score"].mean()),
            "by_cycle": acc_df.groupby("cycle")["brier_score"].agg(["mean", "count"]).to_dict(),
        }

    if all_weather_corr:
        corr_df = pd.concat(all_weather_corr, ignore_index=True)
        summary["weather_correlation"] = {
            "n_races": int(len(corr_df)),
            "mean_correlation": float(corr_df["correlation"].mean()),
            "median_correlation": float(corr_df["correlation"].median()),
            "by_cycle": corr_df.groupby("cycle")["correlation"].agg(["mean", "count"]).to_dict(),
        }

    if all_volatility:
        vol_df = pd.concat(all_volatility, ignore_index=True)
        summary["volatility"] = {
            "n_races": int(len(vol_df)),
            "mean_pre_vol": float(vol_df["pre_vol"].mean()),
            "mean_post_vol": float(vol_df["post_vol"].mean()),
            "mean_election_night_move": float(vol_df["election_night_total_move"].abs().mean()),
        }

    return summary
