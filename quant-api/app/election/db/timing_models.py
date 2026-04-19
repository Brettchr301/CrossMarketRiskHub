"""Timing-aligned data tables.

Stores high-fidelity weather, vote counts, and party registration with
minute-level timestamps for alignment with prediction market quotes.
"""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.election.db.models import ElectionBase


class HourlyWeather(ElectionBase):
    """Hourly historical weather for swing states."""
    __tablename__ = "hourly_weather"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    state: Mapped[str] = mapped_column(String(4), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True)
    temperature: Mapped[float | None] = mapped_column(Float, nullable=True)
    precipitation: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_speed: Mapped[float | None] = mapped_column(Float, nullable=True)
    cloud_cover: Mapped[float | None] = mapped_column(Float, nullable=True)
    snowfall: Mapped[float | None] = mapped_column(Float, nullable=True)
    turnout_score: Mapped[float | None] = mapped_column(Float, nullable=True)


class LiveVoteCount(ElectionBase):
    """Live vote count progression during election night."""
    __tablename__ = "live_vote_counts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[int] = mapped_column(Integer, index=True)
    state: Mapped[str] = mapped_column(String(4), index=True)
    cycle: Mapped[int] = mapped_column(Integer, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True)
    pct_reporting: Mapped[float] = mapped_column(Float)
    leader_party: Mapped[str] = mapped_column(String(4))
    leader_margin_pct: Mapped[float] = mapped_column(Float)
    total_votes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source: Mapped[str] = mapped_column(String(64))


class PartyRegistration(ElectionBase):
    """State party registration snapshots."""
    __tablename__ = "party_registration"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    state: Mapped[str] = mapped_column(String(4), index=True)
    as_of_date: Mapped[date] = mapped_column(Date, index=True)
    dem_registered: Mapped[int] = mapped_column(Integer, default=0)
    rep_registered: Mapped[int] = mapped_column(Integer, default=0)
    ind_registered: Mapped[int] = mapped_column(Integer, default=0)
    total_registered: Mapped[int] = mapped_column(Integer, default=0)
    dem_advantage_pct: Mapped[float] = mapped_column(Float, default=0.0)
    source: Mapped[str] = mapped_column(String(64))


Index("ix_hw_state_ts", HourlyWeather.state, HourlyWeather.timestamp.desc())
Index("ix_lvc_race_ts", LiveVoteCount.race_id, LiveVoteCount.timestamp.desc())
Index("ix_pr_state_date", PartyRegistration.state, PartyRegistration.as_of_date.desc())
