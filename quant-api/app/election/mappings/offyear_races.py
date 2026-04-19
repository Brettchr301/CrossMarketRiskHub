"""Off-year election races (odd years 2019/2021/2023/2025).

US has regular off-year elections in:
- NJ, VA (gubernatorial in years ending in odd)
- KY, LA, MS (odd-year gubernatorial)
- Special elections for US Senate/House vacancies
- State legislature elections in odd years
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(slots=True)
class RaceSpec:
    race_type: str
    state: str
    district: str | None
    cycle: int
    election_date: date
    winner: str | None = None
    winner_name: str | None = None


# ── 2019 Off-Year (3 governor races) ──────────────────────────────────
RACES_2019 = [
    RaceSpec("governor", "KY", None, 2019, date(2019, 11, 5), winner="D", winner_name="Andy Beshear"),
    RaceSpec("governor", "LA", None, 2019, date(2019, 11, 16), winner="D", winner_name="John Bel Edwards"),
    RaceSpec("governor", "MS", None, 2019, date(2019, 11, 5), winner="R", winner_name="Tate Reeves"),
    RaceSpec("governor", "VA", None, 2019, date(2019, 11, 5), winner="D"),  # legislature
    RaceSpec("governor", "NJ", None, 2019, date(2019, 11, 5), winner="D"),  # legislature
    # Special US House elections in 2019
    RaceSpec("house_special", "NC", "03", 2019, date(2019, 9, 10), winner="R", winner_name="Greg Murphy"),
    RaceSpec("house_special", "NC", "09", 2019, date(2019, 9, 10), winner="R", winner_name="Dan Bishop"),
    RaceSpec("house_special", "PA", "12", 2019, date(2019, 5, 21), winner="R", winner_name="Fred Keller"),
]

# ── 2021 Off-Year ──────────────────────────────────────────────────────
RACES_2021 = [
    RaceSpec("governor", "NJ", None, 2021, date(2021, 11, 2), winner="D", winner_name="Phil Murphy"),
    RaceSpec("governor", "VA", None, 2021, date(2021, 11, 2), winner="R", winner_name="Glenn Youngkin"),
    # VA Attorney General & Lt Gov flipped R
    RaceSpec("state_ag", "VA", None, 2021, date(2021, 11, 2), winner="R", winner_name="Jason Miyares"),
    RaceSpec("state_lt_gov", "VA", None, 2021, date(2021, 11, 2), winner="R", winner_name="Winsome Sears"),
    # NYC Mayor
    RaceSpec("mayor", "NY", "NYC", 2021, date(2021, 11, 2), winner="D", winner_name="Eric Adams"),
    # Special US House
    RaceSpec("house_special", "OH", "11", 2021, date(2021, 11, 2), winner="D", winner_name="Shontel Brown"),
    RaceSpec("house_special", "OH", "15", 2021, date(2021, 11, 2), winner="R", winner_name="Mike Carey"),
    RaceSpec("house_special", "TX", "06", 2021, date(2021, 7, 27), winner="R", winner_name="Jake Ellzey"),
    RaceSpec("house_special", "LA", "02", 2021, date(2021, 4, 24), winner="D", winner_name="Troy Carter"),
    RaceSpec("house_special", "LA", "05", 2021, date(2021, 3, 20), winner="R", winner_name="Julia Letlow"),
    RaceSpec("house_special", "NM", "01", 2021, date(2021, 6, 1), winner="D", winner_name="Melanie Stansbury"),
    # California recall (Newsom survived)
    RaceSpec("recall", "CA", None, 2021, date(2021, 9, 14), winner="D", winner_name="Gavin Newsom (survived)"),
]

# ── 2023 Off-Year ──────────────────────────────────────────────────────
RACES_2023 = [
    RaceSpec("governor", "KY", None, 2023, date(2023, 11, 7), winner="D", winner_name="Andy Beshear"),
    RaceSpec("governor", "LA", None, 2023, date(2023, 10, 14), winner="R", winner_name="Jeff Landry"),
    RaceSpec("governor", "MS", None, 2023, date(2023, 11, 7), winner="R", winner_name="Tate Reeves"),
    # VA legislature (no gov race) - Dems held both chambers
    RaceSpec("va_legislature", "VA", None, 2023, date(2023, 11, 7), winner="D"),
    # Special US House
    RaceSpec("house_special", "RI", "01", 2023, date(2023, 11, 7), winner="D", winner_name="Gabe Amo"),
    RaceSpec("house_special", "UT", "02", 2023, date(2023, 11, 21), winner="R", winner_name="Celeste Maloy"),
    # Ballot measures
    RaceSpec("abortion_referendum", "OH", None, 2023, date(2023, 11, 7), winner="YES", winner_name="Issue 1 passed"),
]

# ── 2025 Off-Year (future/just past) ───────────────────────────────────
RACES_2025 = [
    RaceSpec("governor", "NJ", None, 2025, date(2025, 11, 4)),
    RaceSpec("governor", "VA", None, 2025, date(2025, 11, 4)),
    # NYC Mayor
    RaceSpec("mayor", "NY", "NYC", 2025, date(2025, 11, 4)),
    # Special elections will be added as they happen
]

# ── Combined Off-Year Registry ─────────────────────────────────────────
ALL_OFFYEAR_RACES: list[RaceSpec] = RACES_2019 + RACES_2021 + RACES_2023 + RACES_2025
