"""Election event mappings for prediction market providers.

Maps canonical race identifiers to platform-specific market IDs / tickers / regex.
Follows the same EventMapping pattern from providers/real_prediction.py.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ElectionEventMapping:
    """Maps a canonical election event to platform-specific identifiers."""
    event_id: str
    # Polymarket
    poly_regex: str | None = None
    poly_market_ids: tuple[str, ...] = ()
    poly_invert: bool = False
    # Kalshi
    kalshi_event_ticker: str | None = None
    kalshi_invert: bool = False
    # PredictIt — matched by question text regex
    predictit_regex: str | None = None
    # Metaculus — matched by question text regex
    metaculus_regex: str | None = None


ELECTION_EVENT_MAPPINGS: dict[str, ElectionEventMapping] = {
    # ── 2028 Presidential ─────────────────────────────────────────────
    "pres_2028_winner": ElectionEventMapping(
        event_id="pres_2028_winner",
        poly_regex=r"president.*2028|win.*2028.*presiden|2028.*presiden.*election|next.*president",
        kalshi_event_ticker="PRES-2028",
        predictit_regex=r"2028.*president|president.*2028",
        metaculus_regex=r"2028.*president|president.*2028",
    ),
    "pres_2028_democrat": ElectionEventMapping(
        event_id="pres_2028_democrat",
        poly_regex=r"democrat.*win.*2028.*presiden|democrat.*president.*2028",
        kalshi_event_ticker="PRES-2028-DEM",
        predictit_regex=r"democrat.*president.*2028",
    ),
    "pres_2028_republican": ElectionEventMapping(
        event_id="pres_2028_republican",
        poly_regex=r"republican.*win.*2028.*presiden|republican.*president.*2028",
        kalshi_event_ticker="PRES-2028-REP",
        predictit_regex=r"republican.*president.*2028",
    ),

    # ── 2026 Senate Control ───────────────────────────────────────────
    "senate_2026_dem": ElectionEventMapping(
        event_id="senate_2026_dem",
        poly_regex=r"democrat.*control.*senate.*2026|democrat.*win.*senate.*2026|senate.*majority.*democrat.*2026",
        kalshi_event_ticker="KXSENATE2026",
        predictit_regex=r"democrat.*senate.*2026|senate.*control.*2026",
        metaculus_regex=r"democrat.*senate.*2026",
    ),
    "house_2026_dem": ElectionEventMapping(
        event_id="house_2026_dem",
        poly_regex=r"democrat.*control.*house.*2026|democrat.*win.*house.*2026|house.*majority.*democrat",
        kalshi_event_ticker="KXHOUSE2026",
        predictit_regex=r"democrat.*house.*2026|house.*control",
    ),

    # ── 2026 Individual Senate Races ──────────────────────────────────
    "senate_pa_2026": ElectionEventMapping(
        event_id="senate_pa_2026",
        poly_regex=r"pennsylvania.*senate.*2026|pa.*senate.*2026",
        kalshi_event_ticker="KXSENPA26",
        predictit_regex=r"pennsylvania.*senate",
    ),
    "senate_mi_2026": ElectionEventMapping(
        event_id="senate_mi_2026",
        poly_regex=r"michigan.*senate.*2026|mi.*senate.*2026",
        kalshi_event_ticker="KXSENMI26",
        predictit_regex=r"michigan.*senate",
    ),
    "senate_wi_2026": ElectionEventMapping(
        event_id="senate_wi_2026",
        poly_regex=r"wisconsin.*senate.*2026|wi.*senate.*2026",
        kalshi_event_ticker="KXSENWI26",
        predictit_regex=r"wisconsin.*senate",
    ),
    "senate_az_2026": ElectionEventMapping(
        event_id="senate_az_2026",
        poly_regex=r"arizona.*senate.*2026|az.*senate.*2026",
        kalshi_event_ticker="KXSENAZ26",
        predictit_regex=r"arizona.*senate",
    ),
    "senate_ga_2026": ElectionEventMapping(
        event_id="senate_ga_2026",
        poly_regex=r"georgia.*senate.*2026|ga.*senate.*2026",
        kalshi_event_ticker="KXSENGA26",
        predictit_regex=r"georgia.*senate",
    ),
    "senate_nv_2026": ElectionEventMapping(
        event_id="senate_nv_2026",
        poly_regex=r"nevada.*senate.*2026|nv.*senate.*2026",
        kalshi_event_ticker="KXSENNV26",
        predictit_regex=r"nevada.*senate",
    ),
    "senate_nc_2026": ElectionEventMapping(
        event_id="senate_nc_2026",
        poly_regex=r"north carolina.*senate.*2026|nc.*senate.*2026",
        kalshi_event_ticker="KXSENNC26",
        predictit_regex=r"north carolina.*senate",
    ),
    "senate_mn_2026": ElectionEventMapping(
        event_id="senate_mn_2026",
        poly_regex=r"minnesota.*senate.*2026|mn.*senate.*2026",
        kalshi_event_ticker="KXSENMN26",
        predictit_regex=r"minnesota.*senate",
    ),
    "senate_nh_2026": ElectionEventMapping(
        event_id="senate_nh_2026",
        poly_regex=r"new hampshire.*senate.*2026|nh.*senate.*2026",
        kalshi_event_ticker="KXSENNH26",
        predictit_regex=r"new hampshire.*senate",
    ),
    "senate_me_2026": ElectionEventMapping(
        event_id="senate_me_2026",
        poly_regex=r"maine.*senate.*2026|me.*senate.*2026",
        kalshi_event_ticker="KXSENME26",
        predictit_regex=r"maine.*senate",
    ),
    "senate_co_2026": ElectionEventMapping(
        event_id="senate_co_2026",
        poly_regex=r"colorado.*senate.*2026|co.*senate.*2026",
        kalshi_event_ticker="KXSENCO26",
        predictit_regex=r"colorado.*senate",
    ),
    "senate_or_2026": ElectionEventMapping(
        event_id="senate_or_2026",
        poly_regex=r"oregon.*senate.*2026|or.*senate.*2026",
        kalshi_event_ticker="KXSENOR26",
        predictit_regex=r"oregon.*senate",
    ),
    "senate_ia_2026": ElectionEventMapping(
        event_id="senate_ia_2026",
        poly_regex=r"iowa.*senate.*2026|ia.*senate.*2026",
        kalshi_event_ticker="KXSENIA26",
        predictit_regex=r"iowa.*senate",
    ),
    "senate_tx_2026": ElectionEventMapping(
        event_id="senate_tx_2026",
        poly_regex=r"texas.*senate.*2026|tx.*senate.*2026",
        kalshi_event_ticker="KXSENTX26",
        predictit_regex=r"texas.*senate",
    ),
}
