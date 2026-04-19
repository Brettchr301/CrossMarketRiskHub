"""Party registration data provider.

Public data sources for party registration by state (pre-election snapshots).
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

TIMEOUT = 20

STATE_REGISTRATION_URLS = {
    "PA": "https://www.dos.pa.gov/VotingElections/OtherServicesEvents/VotingElectionStatistics/Pages/VotingElectionStatistics.aspx",
    "FL": "https://dos.myflorida.com/elections/data-statistics/voter-registration-statistics/",
    "NC": "https://www.ncsbe.gov/results-data/voter-registration-data",
    "AZ": "https://azsos.gov/elections/voter-registration-statistics",
    "NV": "https://www.nvsos.gov/sos/elections/voters/voter-registration-statistics",
    "CO": "https://www.sos.state.co.us/pubs/elections/VoterRegNumbers/VoterRegNumbers.html",
    "NH": "https://sos.nh.gov/elections/voters/voter-information-look-up/",
    "ME": "https://www.maine.gov/sos/cec/elec/data/index.html",
}


def get_registration_url(state: str) -> str | None:
    """Return the public registration data URL for a state."""
    return STATE_REGISTRATION_URLS.get(state.upper())


def registration_advantage(dem_count: int, rep_count: int) -> float:
    """Calculate party registration advantage as (Dem - Rep) / total.

    Positive = Dem advantage, negative = Rep advantage.
    """
    total = dem_count + rep_count
    if total == 0:
        return 0.0
    return (dem_count - rep_count) / total
