"""
Signal Narrative Generator — T1
Returns plain-English explanation of why the model is bullish/bearish on a ticker.
Template-based (no LLM API call required); uses feature importances from Ridge coef.
"""
from __future__ import annotations

from typing import Any

_FEATURE_LABELS: dict[str, str] = {
    "brent_ret": "Brent crude daily return",
    "brent_sq": "Brent crude squared return (volatility)",
    "brent_accel": "Brent crude price acceleration",
    "wti_ret": "WTI crude daily return",
    "wti_sq": "WTI crude squared return",
    "wti_accel": "WTI crude acceleration",
    "ship_spot_ret": "Spot freight rate (BOAT/SEA)",
    "ship_spot_sq": "Spot freight rate volatility",
    "ship_fwd_ret": "Forward freight rate (BDI)",
    "ship_fwd_sq": "Forward freight rate volatility",
    "brent_ship_cross": "Brent x freight rate interaction",
    "wti_brent_spread_ret": "WTI-Brent spread",
    "high_vol_regime": "High oil volatility regime",
    "event_freight_cross": "Hormuz event x freight rate cross",
    "brent_contango_ret": "Brent term structure (contango/backwardation)",
    "brent_contango_d1": "Brent term structure momentum",
    "hormuz_closure_d1": "Strait of Hormuz closure probability change",
    "hormuz_closure_sq": "Hormuz event volatility",
    "red_sea_disruption_d1": "Red Sea disruption probability change",
    "sanctions_escalation_d1": "Iran sanctions escalation probability change",
    "oil_above_100_d1": "Oil >$100 probability change",
    "opec_production_cut_d1": "OPEC+ production cut probability change",
    "panama_canal_disruption_d1": "Panama Canal disruption probability change",
    "china_stimulus_d1": "China stimulus probability change",
    "us_spr_release_d1": "US SPR release probability change",
    "us_refinery_utilization_low_d1": "US refinery utilization (low) probability change",
    # Commodity events
    "gold_above_3000_d1": "Gold >$3000 probability change",
    "silver_above_40_d1": "Silver >$40 probability change",
    "copper_above_5_d1": "Copper >$5/lb probability change",
    "iron_ore_above_150_d1": "Iron ore >$150 probability change",
    "us_tariff_escalation_d1": "US tariff escalation probability change",
    "china_property_crisis_d1": "China property crisis probability change",
    "rare_earth_export_ban_d1": "China rare earth export ban probability change",
    "nuclear_renaissance_d1": "Nuclear renaissance probability change",
    "ev_adoption_milestone_d1": "EV adoption milestone probability change",
    "lithium_oversupply_d1": "Lithium oversupply probability change",
    "potash_sanctions_d1": "Potash/fertilizer sanctions probability change",
    "food_crisis_d1": "Food crisis probability change",
    "carbon_price_above_100_d1": "Carbon price above $100 probability change",
    "india_infrastructure_boom_d1": "India infrastructure boom probability change",
    # Commodity factor features
    "cmd_gold_ret": "Gold futures daily return",
    "cmd_silver_ret": "Silver futures daily return",
    "cmd_copper_ret": "Copper futures daily return",
    "cmd_platinum_ret": "Platinum futures daily return",
    "cmd_palladium_ret": "Palladium futures daily return",
    "cmd_uranium_etf_ret": "Uranium ETF (URA) daily return",
    "cmd_lithium_etf_ret": "Lithium ETF (LIT) daily return",
    "cmd_copper_miners_ret": "Copper miners ETF (COPX) daily return",
    "cmd_rare_earth_etf_ret": "Rare earth ETF (REMX) daily return",
    "cmd_aluminum_ret": "Aluminum futures daily return",
    "cmd_wheat_etf_ret": "Wheat ETF (WEAT) daily return",
    # Cross-features
    "gold_dxy_cross": "Gold × DXY interaction",
    "gold_vix_cross": "Gold × VIX interaction (safe-haven)",
    "copper_gold_spread_ret": "Copper-Gold spread (growth vs safety)",
    "copper_oil_cross": "Copper × Brent interaction",
    "tariff_copper_cross": "Tariff event × copper price interaction",
    "china_copper_cross": "China stimulus × copper price interaction",
    "nuclear_uranium_cross": "Nuclear event × uranium ETF interaction",
    # Tail-risk geopolitical / policy events (high alpha)
    "taiwan_strait_crisis_d1": "Taiwan Strait crisis probability change",
    "russia_ukraine_ceasefire_d1": "Russia-Ukraine ceasefire probability change",
    "south_africa_grid_crisis_d1": "South Africa grid crisis (Eskom) probability change",
    "chile_lithium_nationalization_d1": "Chile lithium nationalization probability change",
    "indonesia_nickel_ban_d1": "Indonesia nickel export ban probability change",
    "us_permitting_reform_d1": "US permitting reform probability change",
    "eu_cbam_implementation_d1": "EU carbon border (CBAM) implementation probability change",
    "australia_china_trade_thaw_d1": "Australia-China trade thaw probability change",
    "us_recession_d1": "US recession probability change",
    "middle_east_war_escalation_d1": "Middle East war escalation probability change",
    # Tail-risk cross-features (company-specific alpha)
    "mideast_war_oil_cross": "Middle East war × oil price (supply disruption alpha)",
    "mideast_war_gold_cross": "Middle East war × gold price (safe-haven alpha)",
    "taiwan_rare_earth_cross": "Taiwan crisis × rare earth price (supply chain alpha)",
    "taiwan_shipping_cross": "Taiwan crisis × shipping rates (strait blockade alpha)",
    "ceasefire_wheat_cross": "Russia-Ukraine ceasefire × wheat price (grain corridor alpha)",
    "ceasefire_palladium_cross": "Russia-Ukraine ceasefire × palladium price (supply alpha)",
    "recession_copper_cross": "US recession × copper price (demand destruction alpha)",
    "cbam_aluminum_cross": "EU CBAM × aluminum price (carbon border alpha)",
    "chile_lithium_cross": "Chile nationalization × lithium price (supply alpha)",
    "sa_grid_gold_cross": "SA grid crisis × gold price (mining disruption alpha)",
    "indo_nickel_base_cross": "Indonesia nickel ban × base metals (export ban alpha)",
}


def _label(feature: str) -> str:
    return _FEATURE_LABELS.get(feature, feature.replace("_", " "))


def _confidence_word(c: float) -> str:
    if c >= 0.7:
        return "high-confidence"
    if c >= 0.5:
        return "moderate-confidence"
    return "low-confidence"


def _return_word(r: float) -> str:
    if r >= 0.15:
        return "strong upside"
    if r >= 0.07:
        return "moderate upside"
    if r >= 0.01:
        return "modest upside"
    if r >= -0.01:
        return "roughly flat"
    if r >= -0.07:
        return "modest downside"
    return "significant downside"


def build_narrative(opp: dict[str, Any]) -> dict[str, Any]:
    """Build a narrative dict from a GlobalOpportunity row (as dict)."""
    ticker = opp["ticker"]
    direction = opp.get("direction", "LONG")
    gross = opp.get("expected_return_gross", 0.0)
    net = opp.get("expected_return_net_cost", 0.0)
    conf = opp.get("confidence", 0.0)
    hit = opp.get("hit_rate", 0.0)
    score = opp.get("score", 0.0)
    oil_beta = opp.get("oil_beta", 0.0)
    ship_beta = opp.get("shipping_beta", 0.0)
    event_beta = opp.get("event_beta", 0.0)
    risk_flags = opp.get("risk_flags", [])
    top_features = opp.get("top_features", [])
    top_contracts = opp.get("top_predictive_contracts", [])
    spot = opp.get("spot_price", 0.0)
    fair = opp.get("fair_value_price", 0.0)
    commodity_type = opp.get("commodity_type", "")

    commodity_beta = opp.get("commodity_beta", 0.0)

    # Dominant driver
    betas = {"oil": abs(oil_beta), "shipping": abs(ship_beta), "event": abs(event_beta), "commodity": abs(commodity_beta)}
    dominant = max(betas, key=betas.get)  # type: ignore[arg-type]
    dominant_labels = {
        "oil": "oil price movements (Brent/WTI)",
        "shipping": "freight rate dynamics (spot and forward)",
        "event": "geopolitical and macro event probabilities",
        "commodity": "commodity factor returns (gold/copper/uranium/lithium etc.)",
    }

    # Build feature attribution list
    feature_bullets = [f"- {_label(f)}" for f in top_features[:5]]

    # Summary sentence
    price_gap_pct = (fair / spot - 1.0) * 100 if spot > 0 else 0.0
    price_gap_dir = "undervalued" if fair > spot else "overvalued"

    summary = (
        f"{ticker} ({commodity_type}) is a {_confidence_word(conf)} {direction} signal "
        f"with {_return_word(gross)} expected ({gross*100:+.1f}% gross, {net*100:+.1f}% net of costs). "
        f"The model sees {ticker} as {price_gap_dir} by {abs(price_gap_pct):.1f}% "
        f"(spot ${spot:.2f} vs fair value ${fair:.2f}). "
        f"Primary return driver: {dominant_labels[dominant]}. "
        f"Walk-forward hit rate: {hit*100:.0f}% on {opp.get('hit_rate', 0)*100:.0f}% of predictions. "
        f"Composite score: {score:.1f}."
    )

    # Risk note
    risk_note = ""
    if risk_flags:
        risk_note = "Risk flags: " + "; ".join(f.replace("_", " ") for f in risk_flags) + "."

    # Contract context
    contract_note = ""
    if top_contracts:
        contract_note = "Aligned prediction markets: " + "; ".join(top_contracts[:3]) + "."

    return {
        "ticker": ticker,
        "direction": direction,
        "summary": summary,
        "top_drivers": feature_bullets,
        "risk_note": risk_note,
        "contract_context": contract_note,
        "metrics": {
            "expected_return_gross_pct": round(gross * 100, 2),
            "expected_return_net_pct": round(net * 100, 2),
            "confidence": round(conf, 3),
            "hit_rate": round(hit, 3),
            "score": round(score, 2),
            "oil_beta": round(oil_beta, 3),
            "shipping_beta": round(ship_beta, 3),
            "event_beta": round(event_beta, 3),
            "commodity_beta": round(commodity_beta, 3),
        },
    }
