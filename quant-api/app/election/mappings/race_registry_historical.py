"""Extended race registry with historical cycles for backtesting.

Includes 2018, 2020, 2022, 2024, 2026, 2028 elections.
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
    winner: str | None = None  # actual winner party, None if future
    winner_name: str | None = None


# -- 2018 Midterms (Dems won House, GOP kept Senate) -----------------------
RACES_2018 = [
    # Control markets
    RaceSpec("house_control", "US", None, 2018, date(2018, 11, 6), winner="D"),
    RaceSpec("senate_control", "US", None, 2018, date(2018, 11, 6), winner="R"),
    # Key Senate races
    RaceSpec("senate", "FL", None, 2018, date(2018, 11, 6), winner="R", winner_name="Rick Scott"),
    RaceSpec("senate", "TX", None, 2018, date(2018, 11, 6), winner="R", winner_name="Ted Cruz"),
    RaceSpec("senate", "AZ", None, 2018, date(2018, 11, 6), winner="D", winner_name="Kyrsten Sinema"),
    RaceSpec("senate", "NV", None, 2018, date(2018, 11, 6), winner="D", winner_name="Jacky Rosen"),
    RaceSpec("senate", "MO", None, 2018, date(2018, 11, 6), winner="R", winner_name="Josh Hawley"),
    RaceSpec("senate", "IN", None, 2018, date(2018, 11, 6), winner="R", winner_name="Mike Braun"),
    RaceSpec("senate", "ND", None, 2018, date(2018, 11, 6), winner="R", winner_name="Kevin Cramer"),
    RaceSpec("senate", "MT", None, 2018, date(2018, 11, 6), winner="D", winner_name="Jon Tester"),
    RaceSpec("senate", "WV", None, 2018, date(2018, 11, 6), winner="D", winner_name="Joe Manchin"),
    # Key Governor races
    RaceSpec("governor", "FL", None, 2018, date(2018, 11, 6), winner="R", winner_name="Ron DeSantis"),
    RaceSpec("governor", "GA", None, 2018, date(2018, 11, 6), winner="R", winner_name="Brian Kemp"),
    RaceSpec("governor", "WI", None, 2018, date(2018, 11, 6), winner="D", winner_name="Tony Evers"),
    RaceSpec("governor", "MI", None, 2018, date(2018, 11, 6), winner="D", winner_name="Gretchen Whitmer"),
    RaceSpec("governor", "OH", None, 2018, date(2018, 11, 6), winner="R", winner_name="Mike DeWine"),
]

# -- 2020 Presidential + Senate --------------------------------------------
RACES_2020 = [
    RaceSpec("presidential", "US", None, 2020, date(2020, 11, 3), winner="D", winner_name="Joe Biden"),
    RaceSpec("senate_control", "US", None, 2020, date(2020, 11, 3), winner="D"),  # 50-50 + Harris
    # Key Senate races
    RaceSpec("senate", "GA", None, 2020, date(2020, 11, 3), winner="D", winner_name="Jon Ossoff"),
    RaceSpec("senate", "AZ", None, 2020, date(2020, 11, 3), winner="D", winner_name="Mark Kelly"),
    RaceSpec("senate", "NC", None, 2020, date(2020, 11, 3), winner="R", winner_name="Thom Tillis"),
    RaceSpec("senate", "ME", None, 2020, date(2020, 11, 3), winner="R", winner_name="Susan Collins"),
    RaceSpec("senate", "IA", None, 2020, date(2020, 11, 3), winner="R", winner_name="Joni Ernst"),
    RaceSpec("senate", "MT", None, 2020, date(2020, 11, 3), winner="R", winner_name="Steve Daines"),
    RaceSpec("senate", "CO", None, 2020, date(2020, 11, 3), winner="D", winner_name="John Hickenlooper"),
    RaceSpec("senate", "MI", None, 2020, date(2020, 11, 3), winner="D", winner_name="Gary Peters"),
]

# -- 2022 Midterms (Dems held Senate, GOP took House narrowly) -------------
RACES_2022 = [
    RaceSpec("house_control", "US", None, 2022, date(2022, 11, 8), winner="R"),
    RaceSpec("senate_control", "US", None, 2022, date(2022, 11, 8), winner="D"),
    # Key Senate races
    RaceSpec("senate", "PA", None, 2022, date(2022, 11, 8), winner="D", winner_name="John Fetterman"),
    RaceSpec("senate", "GA", None, 2022, date(2022, 11, 8), winner="D", winner_name="Raphael Warnock"),
    RaceSpec("senate", "AZ", None, 2022, date(2022, 11, 8), winner="D", winner_name="Mark Kelly"),
    RaceSpec("senate", "NV", None, 2022, date(2022, 11, 8), winner="D", winner_name="Catherine Cortez Masto"),
    RaceSpec("senate", "NH", None, 2022, date(2022, 11, 8), winner="D", winner_name="Maggie Hassan"),
    RaceSpec("senate", "OH", None, 2022, date(2022, 11, 8), winner="R", winner_name="J.D. Vance"),
    RaceSpec("senate", "NC", None, 2022, date(2022, 11, 8), winner="R", winner_name="Ted Budd"),
    RaceSpec("senate", "WI", None, 2022, date(2022, 11, 8), winner="R", winner_name="Ron Johnson"),
    RaceSpec("senate", "FL", None, 2022, date(2022, 11, 8), winner="R", winner_name="Marco Rubio"),
    # Key Governor races
    RaceSpec("governor", "PA", None, 2022, date(2022, 11, 8), winner="D", winner_name="Josh Shapiro"),
    RaceSpec("governor", "MI", None, 2022, date(2022, 11, 8), winner="D", winner_name="Gretchen Whitmer"),
    RaceSpec("governor", "WI", None, 2022, date(2022, 11, 8), winner="D", winner_name="Tony Evers"),
    RaceSpec("governor", "AZ", None, 2022, date(2022, 11, 8), winner="D", winner_name="Katie Hobbs"),
    RaceSpec("governor", "NV", None, 2022, date(2022, 11, 8), winner="R", winner_name="Joe Lombardo"),
    RaceSpec("governor", "GA", None, 2022, date(2022, 11, 8), winner="R", winner_name="Brian Kemp"),
    RaceSpec("governor", "FL", None, 2022, date(2022, 11, 8), winner="R", winner_name="Ron DeSantis"),
]

# -- 2024 Presidential + Senate --------------------------------------------
RACES_2024 = [
    RaceSpec("presidential", "US", None, 2024, date(2024, 11, 5), winner="R", winner_name="Donald Trump"),
    RaceSpec("senate_control", "US", None, 2024, date(2024, 11, 5), winner="R"),
    RaceSpec("house_control", "US", None, 2024, date(2024, 11, 5), winner="R"),
    # Key Senate races
    RaceSpec("senate", "PA", None, 2024, date(2024, 11, 5), winner="R", winner_name="Dave McCormick"),
    RaceSpec("senate", "OH", None, 2024, date(2024, 11, 5), winner="R", winner_name="Bernie Moreno"),
    RaceSpec("senate", "MT", None, 2024, date(2024, 11, 5), winner="R", winner_name="Tim Sheehy"),
    RaceSpec("senate", "WV", None, 2024, date(2024, 11, 5), winner="R", winner_name="Jim Justice"),
    RaceSpec("senate", "WI", None, 2024, date(2024, 11, 5), winner="D", winner_name="Tammy Baldwin"),
    RaceSpec("senate", "MI", None, 2024, date(2024, 11, 5), winner="D", winner_name="Elissa Slotkin"),
    RaceSpec("senate", "AZ", None, 2024, date(2024, 11, 5), winner="D", winner_name="Ruben Gallego"),
    RaceSpec("senate", "NV", None, 2024, date(2024, 11, 5), winner="D", winner_name="Jacky Rosen"),
    RaceSpec("senate", "NC", None, 2024, date(2024, 11, 5), winner="R", winner_name="Ted Budd"),  # Burr replacement
    RaceSpec("senate", "FL", None, 2024, date(2024, 11, 5), winner="R", winner_name="Rick Scott"),
    RaceSpec("senate", "TX", None, 2024, date(2024, 11, 5), winner="R", winner_name="Ted Cruz"),
    # Governor races
    RaceSpec("governor", "NC", None, 2024, date(2024, 11, 5), winner="D", winner_name="Josh Stein"),
    RaceSpec("governor", "NH", None, 2024, date(2024, 11, 5), winner="R", winner_name="Kelly Ayotte"),
]

# -- 2026 Midterms (from existing registry) --------------------------------
from app.election.mappings.race_registry import SENATE_2026, CONTROL_2026, GOVERNOR_2026, PRESIDENTIAL_2028
from app.election.mappings.offyear_races import ALL_OFFYEAR_RACES

RACES_2026 = SENATE_2026 + CONTROL_2026 + GOVERNOR_2026

# -- Combined Historical Registry ------------------------------------------
ALL_RACES_HISTORICAL: list[RaceSpec] = (
    RACES_2018 + RACES_2020 + RACES_2022 + RACES_2024
    + ALL_OFFYEAR_RACES  # 2019/2021/2023/2025 off-year races
    + RACES_2026 + PRESIDENTIAL_2028
)
