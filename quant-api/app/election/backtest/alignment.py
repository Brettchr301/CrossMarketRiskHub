"""Event alignment layer.

Joins prediction market quotes with weather, vote counts, and party registration
on precise timestamps for event-study analysis.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.election.db.historical_models import HistoricalQuote, RaceOutcome
from app.election.db.timing_models import HourlyWeather, LiveVoteCount, PartyRegistration

logger = logging.getLogger(__name__)


@dataclass
class AlignedEventStudy:
    race_id: int
    cycle: int
    election_date: date
    market_series: pd.DataFrame  # platform x time
    weather_series: pd.DataFrame  # hourly weather + turnout score
    vote_count_series: pd.DataFrame | None
    party_reg: dict[str, Any] | None


def build_event_study(
    db: Session,
    race_id: int,
    hours_before: int = 48,
    hours_after: int = 48,
) -> AlignedEventStudy | None:
    """Build a complete event study for a race around election day.

    Fetches all prediction market quotes, hourly weather, vote counts, and
    party registration data within the window, aligned on UTC timestamps.
    """
    outcome = db.execute(
        select(RaceOutcome).where(RaceOutcome.race_id == race_id)
    ).scalar_one_or_none()
    if outcome is None:
        return None

    election_dt = datetime.combine(outcome.election_date, datetime.min.time())
    start_dt = election_dt - timedelta(hours=hours_before)
    end_dt = election_dt + timedelta(hours=hours_after)

    # 1. Market quotes
    quotes = db.execute(
        select(HistoricalQuote).where(
            HistoricalQuote.race_id == race_id,
            HistoricalQuote.as_of >= start_dt,
            HistoricalQuote.as_of <= end_dt,
        ).order_by(HistoricalQuote.as_of)
    ).scalars().all()

    if not quotes:
        return None

    market_df = pd.DataFrame([
        {"timestamp": q.as_of, "platform": q.platform, "price": q.price}
        for q in quotes
    ])
    market_pivot = market_df.pivot_table(
        index="timestamp", columns="platform", values="price", aggfunc="mean"
    ).sort_index()

    # 2. Hourly weather
    weather = db.execute(
        select(HourlyWeather).where(
            HourlyWeather.state == outcome.state,
            HourlyWeather.timestamp >= start_dt,
            HourlyWeather.timestamp <= end_dt,
        ).order_by(HourlyWeather.timestamp)
    ).scalars().all()

    weather_df = pd.DataFrame([
        {
            "timestamp": w.timestamp,
            "temperature": w.temperature,
            "precipitation": w.precipitation,
            "wind_speed": w.wind_speed,
            "turnout_score": w.turnout_score,
        }
        for w in weather
    ])
    if not weather_df.empty:
        weather_df = weather_df.set_index("timestamp").sort_index()

    # 3. Vote counts
    votes = db.execute(
        select(LiveVoteCount).where(
            LiveVoteCount.race_id == race_id,
            LiveVoteCount.timestamp >= start_dt,
            LiveVoteCount.timestamp <= end_dt,
        ).order_by(LiveVoteCount.timestamp)
    ).scalars().all()

    vote_df = None
    if votes:
        vote_df = pd.DataFrame([
            {
                "timestamp": v.timestamp,
                "pct_reporting": v.pct_reporting,
                "leader_party": v.leader_party,
                "leader_margin_pct": v.leader_margin_pct,
            }
            for v in votes
        ]).set_index("timestamp").sort_index()

    # 4. Party registration (latest before election)
    reg = db.execute(
        select(PartyRegistration).where(
            PartyRegistration.state == outcome.state,
            PartyRegistration.as_of_date <= outcome.election_date,
        ).order_by(PartyRegistration.as_of_date.desc()).limit(1)
    ).scalar_one_or_none()

    reg_dict = None
    if reg:
        reg_dict = {
            "as_of_date": str(reg.as_of_date),
            "dem_registered": reg.dem_registered,
            "rep_registered": reg.rep_registered,
            "dem_advantage_pct": reg.dem_advantage_pct,
        }

    return AlignedEventStudy(
        race_id=race_id,
        cycle=outcome.cycle,
        election_date=outcome.election_date,
        market_series=market_pivot,
        weather_series=weather_df,
        vote_count_series=vote_df,
        party_reg=reg_dict,
    )


def price_response_to_vote_reporting(
    study: AlignedEventStudy,
    lag_minutes: int = 5,
) -> pd.DataFrame:
    """Measure market price response to each vote count update.

    For each vote count timestamp, measure the price change in the following `lag_minutes`.
    Returns DataFrame with: vote_ts, pct_reporting, leader_margin, price_change_post.
    """
    if study.vote_count_series is None or study.market_series.empty:
        return pd.DataFrame()

    mid_price = study.market_series.mean(axis=1)  # avg across platforms

    rows = []
    for vote_ts, vote_row in study.vote_count_series.iterrows():
        # Price just before the vote update
        pre_mask = mid_price.index <= vote_ts
        if not pre_mask.any():
            continue
        pre_price = mid_price[pre_mask].iloc[-1]

        # Price `lag_minutes` after
        post_ts = vote_ts + timedelta(minutes=lag_minutes)
        post_mask = mid_price.index <= post_ts
        if not post_mask.any():
            continue
        post_price = mid_price[post_mask].iloc[-1]

        rows.append({
            "vote_ts": vote_ts,
            "pct_reporting": vote_row["pct_reporting"],
            "leader_party": vote_row["leader_party"],
            "leader_margin_pct": vote_row["leader_margin_pct"],
            "pre_price": pre_price,
            "post_price": post_price,
            "price_change": post_price - pre_price,
        })

    return pd.DataFrame(rows)


def weather_price_correlation(
    study: AlignedEventStudy,
) -> dict[str, float] | None:
    """Compute correlation between weather turnout_score and market prices.

    Both series are resampled to hourly and correlated.
    """
    if study.weather_series.empty or study.market_series.empty:
        return None

    mid_price = study.market_series.mean(axis=1).resample("1h").mean()
    turnout = study.weather_series["turnout_score"].resample("1h").mean()

    aligned = pd.concat([mid_price.rename("price"), turnout.rename("turnout")], axis=1).dropna()
    if len(aligned) < 3:
        return None

    corr = aligned.corr().loc["price", "turnout"]
    return {
        "correlation": float(corr),
        "n_points": int(len(aligned)),
        "price_range": float(aligned["price"].max() - aligned["price"].min()),
        "turnout_range": float(aligned["turnout"].max() - aligned["turnout"].min()),
    }
