from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Sequence

import pandas as pd
import requests

from app.providers.base import PredictionQuoteRow


@dataclass(slots=True)
class EventMapping:
    event_id: str
    poly_regex: str | None = None
    poly_market_ids: tuple[str, ...] = ()
    poly_invert: bool = False
    kalshi_event_ticker: str | None = None
    kalshi_invert: bool = False


EVENT_MAPPINGS: dict[str, EventMapping] = {
    "hormuz_closure": EventMapping(
        event_id="hormuz_closure",
        poly_regex=r"close the strait of hormuz",
        poly_market_ids=("665307", "1227361", "1227362"),
        poly_invert=False,
        kalshi_event_ticker="KXCLOSEHORMUZ-27JAN",
        kalshi_invert=False,
    ),
    "red_sea_disruption": EventMapping(
        event_id="red_sea_disruption",
        poly_regex=r"suez canal.*transits|red sea",
        poly_market_ids=("704239", "704240"),
        poly_invert=True,  # transit probability -> disruption inverse
        kalshi_event_ticker=None,
    ),
    "sanctions_escalation": EventMapping(
        event_id="sanctions_escalation",
        poly_regex=r"us-iran nuclear deal",
        poly_market_ids=("665325", "957019", "1402792"),
        poly_invert=True,  # nuclear deal probability -> sanctions escalation inverse
        kalshi_event_ticker="KXUSAIRANAGREEMENT-27",
        kalshi_invert=True,
    ),
    "oil_above_100": EventMapping(
        event_id="oil_above_100",
        poly_regex=r"crude oil .*\\$100|wti.*100|barrel of crude oil be \\$100",
        poly_market_ids=("1467766",),
        poly_invert=False,
        kalshi_event_ticker="KXWTIMAX-26DEC31",
        kalshi_invert=False,
    ),
    "opec_production_cut": EventMapping(
        event_id="opec_production_cut",
        poly_regex=r"opec.*cut|opec.*reduc|opec.*extend.*cut",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker="KXOPECCUT",
        kalshi_invert=False,
    ),
    "panama_canal_disruption": EventMapping(
        event_id="panama_canal_disruption",
        poly_regex=r"panama canal.*transits|panama canal.*restric|panama canal.*drought",
        poly_market_ids=(),
        poly_invert=True,  # transit probability -> disruption inverse
        kalshi_event_ticker=None,
    ),
    "china_stimulus": EventMapping(
        event_id="china_stimulus",
        poly_regex=r"china.*stimulus|pboc.*rate cut|china.*gdp.*target",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    "us_spr_release": EventMapping(
        event_id="us_spr_release",
        poly_regex=r"strategic petroleum reserve.*release|spr.*release",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    "us_refinery_utilization_low": EventMapping(
        event_id="us_refinery_utilization_low",
        poly_regex=r"refinery utilization|refinery capacity.*low",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    # ── Precious metals events ──
    "gold_above_3000": EventMapping(
        event_id="gold_above_3000",
        poly_regex=r"gold.*\$?3[,.]?000|price of gold.*3000|gold hit 3000",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker="KXGOLDMAX",
        kalshi_invert=False,
    ),
    "silver_above_40": EventMapping(
        event_id="silver_above_40",
        poly_regex=r"silver.*\$?40|price of silver.*40|silver hit 40",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    # ── Base metals / mining events ──
    "copper_above_5": EventMapping(
        event_id="copper_above_5",
        poly_regex=r"copper.*\$?5|price of copper|copper hit",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    "iron_ore_above_150": EventMapping(
        event_id="iron_ore_above_150",
        poly_regex=r"iron ore.*\$?150|iron ore.*price|price of iron ore",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    # ── Trade & geopolitical events ──
    "us_tariff_escalation": EventMapping(
        event_id="us_tariff_escalation",
        poly_regex=r"tariff.*increase|tariff.*rais|new tariff|trade war|import dut",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker="KXTARIFF",
        kalshi_invert=False,
    ),
    "china_property_crisis": EventMapping(
        event_id="china_property_crisis",
        poly_regex=r"china.*property|china.*real estate|evergrande|country garden.*default",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    "rare_earth_export_ban": EventMapping(
        event_id="rare_earth_export_ban",
        poly_regex=r"rare earth.*ban|rare earth.*restrict|china.*rare earth.*export",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    # ── Energy transition events ──
    "nuclear_renaissance": EventMapping(
        event_id="nuclear_renaissance",
        poly_regex=r"nuclear.*plant.*approv|new nuclear|nuclear.*restart|smr.*approv|nuclear.*energy",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    "ev_adoption_milestone": EventMapping(
        event_id="ev_adoption_milestone",
        poly_regex=r"electric vehicle.*sale|ev.*sale.*million|ev.*market share|ev.*adoption",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    "lithium_oversupply": EventMapping(
        event_id="lithium_oversupply",
        poly_regex=r"lithium.*price|lithium.*surplus|lithium.*oversupply|lithium.*glut",
        poly_market_ids=(),
        poly_invert=True,  # oversupply inverts = bearish signal
        kalshi_event_ticker=None,
    ),
    # ── Agricultural / fertilizer events ──
    "potash_sanctions": EventMapping(
        event_id="potash_sanctions",
        poly_regex=r"potash.*sanction|belarus.*sanction|fertilizer.*sanction|russia.*fertilizer",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    "food_crisis": EventMapping(
        event_id="food_crisis",
        poly_regex=r"food.*crisis|famine|grain.*export.*ban|wheat.*price|food.*shortage",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    # ── Coal / carbon events ──
    "carbon_price_above_100": EventMapping(
        event_id="carbon_price_above_100",
        poly_regex=r"carbon.*price|carbon.*tax|emission.*trading|carbon.*credit.*\$?100",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    # ── India / EM infrastructure ──
    "india_infrastructure_boom": EventMapping(
        event_id="india_infrastructure_boom",
        poly_regex=r"india.*infrastructure|india.*gdp.*grow|india.*steel.*demand|make in india",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    # ────────────────────────────────────────────────────────────────────
    # HIGH-ALPHA TAIL-RISK EVENTS — binary geopolitical / policy events
    # that prediction markets uniquely price but equity markets underprice
    # ────────────────────────────────────────────────────────────────────
    "taiwan_strait_crisis": EventMapping(
        event_id="taiwan_strait_crisis",
        poly_regex=r"taiwan.*invade|china.*taiwan.*military|taiwan.*strait.*conflict|taiwan.*blockade|china.*invade.*taiwan",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker="KXTAIWAN",
        kalshi_invert=False,
    ),
    "russia_ukraine_ceasefire": EventMapping(
        event_id="russia_ukraine_ceasefire",
        poly_regex=r"russia.*ukraine.*ceasefire|ukraine.*peace|russia.*ukraine.*deal|ukraine.*war.*end|ukraine.*negotiat",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker="KXRUSUKR",
        kalshi_invert=False,
    ),
    "south_africa_grid_crisis": EventMapping(
        event_id="south_africa_grid_crisis",
        poly_regex=r"south africa.*power|eskom|load.?shedding|south africa.*electric|south africa.*grid|south africa.*blackout",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    "chile_lithium_nationalization": EventMapping(
        event_id="chile_lithium_nationalization",
        poly_regex=r"chile.*lithium.*national|chile.*lithium.*state|chile.*mining.*reform|chile.*nationali",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    "indonesia_nickel_ban": EventMapping(
        event_id="indonesia_nickel_ban",
        poly_regex=r"indonesia.*nickel.*ban|indonesia.*nickel.*export|indonesia.*mineral.*ban|indonesia.*ore.*export",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    "us_permitting_reform": EventMapping(
        event_id="us_permitting_reform",
        poly_regex=r"permitting reform|nepa.*reform|mining.*permit|energy.*permitting|blm.*lease",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    "eu_cbam_implementation": EventMapping(
        event_id="eu_cbam_implementation",
        poly_regex=r"cbam|carbon border|eu.*carbon.*import|carbon.*adjustment|eu.*import.*tariff.*carbon",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    "australia_china_trade_thaw": EventMapping(
        event_id="australia_china_trade_thaw",
        poly_regex=r"australia.*china.*trade|australia.*china.*tariff|australia.*china.*ban.*lift|australia.*china.*relat",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    "us_recession": EventMapping(
        event_id="us_recession",
        poly_regex=r"us.*recession|recession.*202[5-9]|recession.*united states|america.*recession|gdp.*contract|nber.*recession",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker="KXRECESSION",
        kalshi_invert=False,
    ),
    "middle_east_war_escalation": EventMapping(
        event_id="middle_east_war_escalation",
        poly_regex=r"middle east.*war|israel.*iran.*war|iran.*strike|israel.*hezbollah|iran.*israel|israel.*war|iran.*attack",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker="KXMIDEAST",
        kalshi_invert=False,
    ),
    # ── Monetary policy / central bank events ──
    "fed_rate_cut": EventMapping(
        event_id="fed_rate_cut",
        poly_regex=r"fed.*rate.*cut|federal reserve.*cut|fomc.*rate.*lower|fed.*lower.*rate|fed fund.*rate.*cut|interest rate.*cut",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker="KXFED",
        kalshi_invert=False,
    ),
    "fed_rate_hike": EventMapping(
        event_id="fed_rate_hike",
        poly_regex=r"fed.*rate.*hike|federal reserve.*hike|fed.*raise.*rate|fomc.*rate.*higher|interest rate.*rais",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    "ecb_rate_cut": EventMapping(
        event_id="ecb_rate_cut",
        poly_regex=r"ecb.*rate.*cut|european central bank.*cut|ecb.*lower|ecb.*eas",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    "boj_rate_hike": EventMapping(
        event_id="boj_rate_hike",
        poly_regex=r"bank of japan.*rate|boj.*rate.*hike|japan.*rate.*rais|boj.*tighten|japan.*yield curve",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    "us_government_shutdown": EventMapping(
        event_id="us_government_shutdown",
        poly_regex=r"government shutdown|congress.*shutdown|government.*fund.*lapse|us.*shutdown",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker="KXSHUTDOWN",
        kalshi_invert=False,
    ),
    "us_debt_ceiling": EventMapping(
        event_id="us_debt_ceiling",
        poly_regex=r"debt ceiling|debt limit|us.*default|treasury.*exhaust|extraordinary measure",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    "pboc_easing": EventMapping(
        event_id="pboc_easing",
        poly_regex=r"pboc.*cut|pboc.*eas|china.*rate.*cut|china.*rrr.*cut|people.*bank.*china.*cut",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    "dollar_strength_extreme": EventMapping(
        event_id="dollar_strength_extreme",
        poly_regex=r"strong dollar|dollar.*rally|dxy.*above|dollar.*index.*high|greenback.*surge",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    "us_inflation_above_3": EventMapping(
        event_id="us_inflation_above_3",
        poly_regex=r"inflation.*above.*3|cpi.*above.*3|inflation.*3.*percent|us.*inflation.*high|core.*inflation.*3",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker="KXCPI",
        kalshi_invert=False,
    ),
    "us_unemployment_above_5": EventMapping(
        event_id="us_unemployment_above_5",
        poly_regex=r"unemployment.*above.*5|unemployment.*rate.*5|jobless.*5|us.*unemployment.*rise",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker="KXUNEMPLOY",
        kalshi_invert=False,
    ),
    "natural_gas_above_5": EventMapping(
        event_id="natural_gas_above_5",
        poly_regex=r"natural gas.*\$?5|henry hub.*5|nat.*gas.*price.*5|lng.*price.*spike",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    "oil_below_50": EventMapping(
        event_id="oil_below_50",
        poly_regex=r"oil.*below.*\$?50|crude.*below.*50|oil.*crash|wti.*below.*50|barrel.*\$?50",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),

    # ════════════════════════════════════════════════════════════════════
    # PRODUCTION / CONSUMPTION / SUPPLY-CHAIN CONTRACTS
    # These track the micro+macro fundamentals that drive commodity stocks
    # ════════════════════════════════════════════════════════════════════

    # ── Oil & Gas Production/Consumption ──
    "us_oil_production_record": EventMapping(
        event_id="us_oil_production_record",
        poly_regex=r"us.*oil.*production.*record|us.*crude.*production.*high|eia.*production.*record|us.*output.*13",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    "opec_compliance_below_80": EventMapping(
        event_id="opec_compliance_below_80",
        poly_regex=r"opec.*compliance|opec.*cheat|opec.*over.*produc|opec.*output.*exceed|opec.*quota",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    "china_oil_demand_slowdown": EventMapping(
        event_id="china_oil_demand_slowdown",
        poly_regex=r"china.*oil.*demand.*slow|china.*oil.*import.*declin|china.*refinery.*throughput|china.*crude.*import.*drop",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    "us_gasoline_demand_peak": EventMapping(
        event_id="us_gasoline_demand_peak",
        poly_regex=r"gasoline.*demand.*peak|peak.*gasoline|us.*gasoline.*consumption|ev.*displace.*gasoline",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    "global_lng_oversupply": EventMapping(
        event_id="global_lng_oversupply",
        poly_regex=r"lng.*oversupply|lng.*glut|lng.*capacity.*excess|qatar.*lng.*expansion|us.*lng.*export.*record",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    "permian_basin_peak": EventMapping(
        event_id="permian_basin_peak",
        poly_regex=r"permian.*peak|permian.*plateau|permian.*decline|shale.*peak.*oil|permian.*output",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),

    # ── Refining Margin / Crack Spread ──
    "crack_spread_above_30": EventMapping(
        event_id="crack_spread_above_30",
        poly_regex=r"crack spread.*30|refining margin.*high|gasoline.*crack.*30|refin.*margin.*record",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),

    # ── Lithium / Battery Supply Chain ──
    "lithium_price_rebound": EventMapping(
        event_id="lithium_price_rebound",
        poly_regex=r"lithium.*price.*rebound|lithium.*price.*recover|lithium carbonate.*rise|spodumene.*price.*up",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    "china_ev_sales_record": EventMapping(
        event_id="china_ev_sales_record",
        poly_regex=r"china.*ev.*sales.*record|china.*electric.*vehicle.*million|byd.*sales.*record|china.*nev.*penetration",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    "battery_cathode_shift": EventMapping(
        event_id="battery_cathode_shift",
        poly_regex=r"lfp.*cathode|sodium.*ion.*battery|solid.?state.*battery|cathode.*chemistry.*shift|battery.*technology",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    "australia_lithium_mine_closure": EventMapping(
        event_id="australia_lithium_mine_closure",
        poly_regex=r"lithium.*mine.*clos|lithium.*mine.*suspend|spodumene.*mine.*shut|australia.*lithium.*curtail",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),

    # ── Precious Metals Supply/Demand ──
    "central_bank_gold_buying": EventMapping(
        event_id="central_bank_gold_buying",
        poly_regex=r"central bank.*gold.*buy|central bank.*gold.*reserve|pboc.*gold|reserve.*gold.*purchas",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    "silver_industrial_demand_surge": EventMapping(
        event_id="silver_industrial_demand_surge",
        poly_regex=r"silver.*industrial.*demand|silver.*solar.*panel|silver.*deficit|silver.*supply.*short",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),

    # ── Midstream / Pipeline ──
    "us_pipeline_capacity_constraint": EventMapping(
        event_id="us_pipeline_capacity_constraint",
        poly_regex=r"pipeline.*capacity|permian.*pipeline|pipeline.*bottleneck|midstream.*capacity|takeaway.*capacity",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    "lng_export_terminal_approval": EventMapping(
        event_id="lng_export_terminal_approval",
        poly_regex=r"lng.*terminal.*approv|lng.*export.*permit|lng.*facility.*sanction|ferc.*lng|doe.*lng.*export",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),

    # ── Rare Earths / Critical Minerals Supply Chain ──
    "china_rare_earth_processing_dominance": EventMapping(
        event_id="china_rare_earth_processing_dominance",
        poly_regex=r"china.*rare earth.*process|china.*critical mineral.*dominan|china.*gallium|china.*germanium|china.*antimony",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    "us_critical_minerals_act": EventMapping(
        event_id="us_critical_minerals_act",
        poly_regex=r"critical mineral.*act|defense production act.*mineral|ira.*critical mineral|dpa.*mineral|mineral.*security",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),

    # ── Fertilizer / Agriculture Production ──
    "global_fertilizer_shortage": EventMapping(
        event_id="global_fertilizer_shortage",
        poly_regex=r"fertilizer.*short|urea.*price.*spike|ammonia.*price.*surge|nutrient.*supply.*crisis|potash.*shortage",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    "el_nino_la_nina_shift": EventMapping(
        event_id="el_nino_la_nina_shift",
        poly_regex=r"el nino|la nina|enso.*shift|pacific.*oscillation|el ni.o.*develop",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),

    # ── Macro Demand Indicators ──
    "china_pmi_below_50": EventMapping(
        event_id="china_pmi_below_50",
        poly_regex=r"china.*pmi.*below.*50|china.*pmi.*contract|china.*manufactur.*contract|caixin.*pmi.*below",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    "global_shipping_congestion": EventMapping(
        event_id="global_shipping_congestion",
        poly_regex=r"port.*congestion|shipping.*congestion|container.*backlog|port.*delay|freight.*bottleneck",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    "eu_energy_crisis": EventMapping(
        event_id="eu_energy_crisis",
        poly_regex=r"eu.*energy.*crisis|europe.*gas.*price|ttf.*natural gas|europe.*energy.*ration|gas.*storage.*low",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    "india_refinery_expansion": EventMapping(
        event_id="india_refinery_expansion",
        poly_regex=r"india.*refiner.*expan|india.*oil.*demand.*grow|india.*crude.*import.*record|india.*refin.*capacity",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),

    # ── Uranium Supply/Demand ──
    "uranium_supply_deficit": EventMapping(
        event_id="uranium_supply_deficit",
        poly_regex=r"uranium.*supply.*deficit|uranium.*shortage|yellowcake.*price|kazatomprom.*produc|uranium.*enrichment",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
    "smr_deployment_milestone": EventMapping(
        event_id="smr_deployment_milestone",
        poly_regex=r"smr.*deploy|small modular reactor.*order|nuscale.*order|smr.*construct|smr.*license",
        poly_market_ids=(),
        poly_invert=False,
        kalshi_event_ticker=None,
    ),
}


def _clamp_prob(x: float) -> float:
    return max(0.0, min(1.0, x))


class RealKalshiProvider:
    BASE = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(self, timeout_seconds: int = 20):
        self.timeout_seconds = timeout_seconds

    def fetch_current_quote(self, mapping: EventMapping) -> PredictionQuoteRow | None:
        if not mapping.kalshi_event_ticker:
            return None
        params = {"event_ticker": mapping.kalshi_event_ticker, "limit": 200}
        r = requests.get(f"{self.BASE}/markets", params=params, timeout=self.timeout_seconds)
        r.raise_for_status()
        markets = r.json().get("markets", [])
        if not markets:
            return None
        best = sorted(
            markets,
            key=lambda m: float(m.get("volume", 0.0) or 0.0) + float(m.get("open_interest", 0.0) or 0.0),
            reverse=True,
        )[0]
        yes_bid = float(best.get("yes_bid", 0.0)) / 100.0
        yes_ask = float(best.get("yes_ask", 0.0)) / 100.0
        if yes_bid <= 0 and yes_ask <= 0:
            last = float(best.get("last_price", 0.0)) / 100.0
            yes_bid, yes_ask = max(0.0, last - 0.01), min(1.0, last + 0.01)
        if mapping.kalshi_invert:
            yes_bid, yes_ask = 1.0 - yes_ask, 1.0 - yes_bid
        now = datetime.now(UTC).replace(tzinfo=None)
        volume = float(best.get("volume", 1.0) or 1.0)
        spread = abs(yes_ask - yes_bid)
        liq = max(0.01, min(1.0, 1.0 - spread * 5.0))
        return PredictionQuoteRow(
            provider="kalshi",
            event_id=mapping.event_id,
            bid=_clamp_prob(yes_bid),
            ask=_clamp_prob(yes_ask),
            volume=max(1.0, volume),
            liquidity_score=liq,
            as_of=now,
        )

    def fetch_event_history(
        self, mapping: EventMapping, start_ts: int, end_ts: int, period_interval: int = 1440
    ) -> pd.Series:
        if not mapping.kalshi_event_ticker:
            return pd.Series(dtype=float)
        markets = requests.get(
            f"{self.BASE}/markets",
            params={"event_ticker": mapping.kalshi_event_ticker, "limit": 200},
            timeout=self.timeout_seconds,
        ).json().get("markets", [])
        if not markets:
            return pd.Series(dtype=float)
        candidate_tickers = [m["ticker"] for m in markets[:8]]
        payload: list[dict[str, Any]] = []
        chunk_size = 4
        for i in range(0, len(candidate_tickers), chunk_size):
            tickers = candidate_tickers[i : i + chunk_size]
            params = {
                "market_tickers": ",".join(tickers),
                "start_ts": int(start_ts),
                "end_ts": int(end_ts),
                "period_interval": period_interval,
            }
            r = requests.get(f"{self.BASE}/markets/candlesticks", params=params, timeout=self.timeout_seconds)
            if r.status_code >= 400:
                # Some market families only allow shorter windows.
                one_year_ago = int(datetime.now(UTC).timestamp()) - 365 * 24 * 3600
                params["start_ts"] = max(one_year_ago, int(start_ts))
                r = requests.get(f"{self.BASE}/markets/candlesticks", params=params, timeout=self.timeout_seconds)
            if r.status_code >= 400:
                continue
            payload.extend(r.json().get("markets", []))
        rows: list[tuple[pd.Timestamp, float, float]] = []
        for market in payload:
            candles = market.get("candlesticks", [])
            for c in candles:
                close_raw = c.get("price", {}).get("close", None)
                if close_raw is None:
                    continue
                ts = pd.to_datetime(int(c.get("end_period_ts", 0)), unit="s", utc=True).tz_convert(None).normalize()
                price = float(close_raw) / 100.0
                oi = float(c.get("open_interest", 1.0) or 1.0)
                rows.append((ts, price, oi))
        if not rows:
            return pd.Series(dtype=float)
        frame = pd.DataFrame(rows, columns=["date", "price", "weight"])
        frame["weighted_price"] = frame["price"] * frame["weight"]
        grouped = frame.groupby("date", as_index=True)[["weighted_price", "weight"]].sum()
        grouped = grouped["weighted_price"] / grouped["weight"]
        series = grouped.sort_index()
        if mapping.kalshi_invert:
            series = 1.0 - series
        return series.clip(0.0, 1.0)


class RealPolymarketProvider:
    BASE_GAMMA = "https://gamma-api.polymarket.com/markets"
    BASE_CLOB = "https://clob.polymarket.com/prices-history"

    def __init__(self, timeout_seconds: int = 25):
        self.timeout_seconds = timeout_seconds
        self._cache: dict[str, tuple[datetime, dict[str, Any]]] = {}

    def fetch_current_quote(self, mapping: EventMapping) -> PredictionQuoteRow | None:
        market = self._select_market(mapping)
        if market is None:
            return None
        best_bid = float(market.get("bestBid", 0.0) or 0.0)
        best_ask = float(market.get("bestAsk", 0.0) or 0.0)
        if best_bid <= 0 and best_ask <= 0:
            last = float(market.get("lastTradePrice", 0.0) or 0.0)
            best_bid = max(0.0, last - 0.01)
            best_ask = min(1.0, last + 0.01)
        if mapping.poly_invert:
            best_bid, best_ask = 1.0 - best_ask, 1.0 - best_bid
        volume = float(market.get("volume", 1.0) or 1.0)
        spread = abs(best_ask - best_bid)
        liq = max(0.01, min(1.0, 1.0 - spread * 4.0))
        return PredictionQuoteRow(
            provider="polymarket",
            event_id=mapping.event_id,
            bid=_clamp_prob(best_bid),
            ask=_clamp_prob(best_ask),
            volume=max(1.0, volume),
            liquidity_score=liq,
            as_of=datetime.now(UTC).replace(tzinfo=None),
        )

    def fetch_event_history(self, mapping: EventMapping) -> pd.Series:
        markets: list[dict[str, Any]] = []
        if mapping.poly_market_ids:
            for market_id in mapping.poly_market_ids:
                r = requests.get(self.BASE_GAMMA, params={"id": market_id}, timeout=self.timeout_seconds)
                if r.status_code != 200:
                    continue
                arr = r.json()
                if isinstance(arr, list) and arr:
                    markets.extend(arr)
        if not markets:
            markets = self._find_markets(mapping, include_closed=True)
        if not markets:
            return pd.Series(dtype=float)
        rows: list[pd.Series] = []
        for market in markets[:8]:
            yes_token = self._yes_token_id(market)
            if not yes_token:
                continue
            series = self._history_for_token(yes_token)
            if series.empty:
                continue
            if mapping.poly_invert:
                series = 1.0 - series
            volume = float(market.get("volume", 1.0) or 1.0)
            rows.append(series.rename(str(volume)))
        if not rows:
            return pd.Series(dtype=float)
        frame = pd.concat(rows, axis=1).sort_index().ffill()
        weights = pd.Series([float(c) for c in frame.columns], index=frame.columns).replace(0.0, 1.0)
        weighted = (frame * weights).sum(axis=1) / weights.sum()
        return weighted.clip(0.0, 1.0)

    def _select_market(self, mapping: EventMapping) -> dict[str, Any] | None:
        cached = self._cache.get(mapping.event_id)
        now = datetime.now(UTC)
        if cached and now - cached[0] < timedelta(minutes=30):
            return cached[1]
        if mapping.poly_market_ids:
            for market_id in mapping.poly_market_ids:
                r = requests.get(self.BASE_GAMMA, params={"id": market_id}, timeout=self.timeout_seconds)
                if r.status_code != 200:
                    continue
                arr = r.json()
                if isinstance(arr, list) and arr:
                    best = arr[0]
                    self._cache[mapping.event_id] = (now, best)
                    return best
        markets = self._find_markets(mapping, include_closed=False)
        if not markets:
            return None
        best = sorted(markets, key=lambda m: float(m.get("volume", 0.0) or 0.0), reverse=True)[0]
        self._cache[mapping.event_id] = (now, best)
        return best

    def _find_markets(self, mapping: EventMapping, include_closed: bool) -> list[dict[str, Any]]:
        if not mapping.poly_regex:
            return []
        pattern = re.compile(mapping.poly_regex, re.I)
        matches: list[dict[str, Any]] = []
        limit = 500
        max_pages = 40 if include_closed else 12
        for page in range(max_pages):
            params = {"limit": limit, "offset": page * limit, "active": "true"}
            if not include_closed:
                params["closed"] = "false"
            r = requests.get(self.BASE_GAMMA, params=params, timeout=self.timeout_seconds)
            r.raise_for_status()
            arr = r.json()
            if not arr:
                break
            for market in arr:
                question = str(market.get("question", ""))
                if not pattern.search(question):
                    continue
                if include_closed is False and market.get("closed"):
                    continue
                matches.append(market)
        matches.sort(key=lambda m: float(m.get("volume", 0.0) or 0.0), reverse=True)
        return matches

    def _history_for_token(self, token_id: str) -> pd.Series:
        r = requests.get(
            self.BASE_CLOB,
            params={"market": token_id, "interval": "max", "fidelity": 1440},
            timeout=self.timeout_seconds,
        )
        if r.status_code != 200:
            return pd.Series(dtype=float)
        payload = r.json()
        history = payload.get("history", [])
        if not history:
            return pd.Series(dtype=float)
        dates = [pd.to_datetime(int(x["t"]), unit="s", utc=True).tz_convert(None).normalize() for x in history]
        values = [_clamp_prob(float(x["p"])) for x in history]
        s = pd.Series(values, index=dates).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        return s

    @staticmethod
    def _yes_token_id(market: dict[str, Any]) -> str | None:
        raw_tokens = market.get("clobTokenIds")
        raw_outcomes = market.get("outcomes")
        if raw_tokens is None or raw_outcomes is None:
            return None
        try:
            token_ids = json.loads(raw_tokens) if isinstance(raw_tokens, str) else list(raw_tokens)
            outcomes = json.loads(raw_outcomes) if isinstance(raw_outcomes, str) else list(raw_outcomes)
        except Exception:
            return None
        for i, outcome in enumerate(outcomes):
            if str(outcome).strip().lower() == "yes" and i < len(token_ids):
                return str(token_ids[i])
        return str(token_ids[0]) if token_ids else None


class RealPredictionProvider:
    def __init__(self) -> None:
        self.kalshi = RealKalshiProvider()
        self.polymarket = RealPolymarketProvider()

    def fetch_event_quotes(self, events: Sequence[str]) -> Sequence[PredictionQuoteRow]:
        rows: list[PredictionQuoteRow] = []
        for event in events:
            mapping = EVENT_MAPPINGS.get(event)
            if mapping is None:
                continue
            poly_row = self.polymarket.fetch_current_quote(mapping)
            kalshi_row = self.kalshi.fetch_current_quote(mapping)
            if poly_row is not None:
                rows.append(poly_row)
            if kalshi_row is not None:
                rows.append(kalshi_row)
        return rows
