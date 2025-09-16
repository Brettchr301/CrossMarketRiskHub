"""Generate synthetic event probability cache for backtesting.

Creates realistic time-varying probability series for all 33 prediction-market
events.  Each event has a base probability, volatility, mean-revision speed,
and occasional "shock" days that mimic real market moves.

Output: data/event_history_cache.parquet  +  data/event_history_cache_meta.json
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ── ensure we can import the universe ──
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.providers.real_prediction import EVENT_MAPPINGS


EVENT_PROFILES: dict[str, dict] = {
    "oil_above_100":              {"base": 0.35, "vol": 0.15, "trend": 0.002},
    "opec_production_cut":        {"base": 0.55, "vol": 0.12, "trend": 0.001},
    "us_recession":               {"base": 0.25, "vol": 0.20, "trend": 0.003},
    "hormuz_closure":             {"base": 0.08, "vol": 0.05, "trend": 0.001},
    "red_sea_disruption":         {"base": 0.15, "vol": 0.25, "trend": 0.002},
    "russia_ukraine_ceasefire":   {"base": 0.20, "vol": 0.18, "trend": -0.001},
    "taiwan_strait_crisis":       {"base": 0.10, "vol": 0.08, "trend": 0.002},
    "middle_east_war_escalation": {"base": 0.30, "vol": 0.22, "trend": 0.003},
    "sanctions_escalation":       {"base": 0.40, "vol": 0.15, "trend": 0.002},
    "china_stimulus":             {"base": 0.45, "vol": 0.15, "trend": -0.001},
    "china_property_crisis":      {"base": 0.50, "vol": 0.18, "trend": -0.002},
    "gold_above_3000":            {"base": 0.20, "vol": 0.12, "trend": 0.004},
    "silver_above_40":            {"base": 0.15, "vol": 0.10, "trend": 0.003},
    "copper_above_5":             {"base": 0.30, "vol": 0.12, "trend": 0.002},
    "iron_ore_above_150":         {"base": 0.25, "vol": 0.14, "trend": -0.001},
    "nuclear_renaissance":        {"base": 0.35, "vol": 0.10, "trend": 0.003},
    "rare_earth_export_ban":      {"base": 0.12, "vol": 0.08, "trend": 0.002},
    "carbon_price_above_100":     {"base": 0.20, "vol": 0.10, "trend": 0.002},
    "ev_adoption_milestone":      {"base": 0.40, "vol": 0.12, "trend": 0.002},
    "lithium_oversupply":         {"base": 0.55, "vol": 0.15, "trend": -0.001},
    "chile_lithium_nationalization": {"base": 0.18, "vol": 0.10, "trend": 0.001},
    "us_tariff_escalation":       {"base": 0.35, "vol": 0.20, "trend": 0.003},
    "us_permitting_reform":       {"base": 0.30, "vol": 0.08, "trend": 0.001},
    "us_spr_release":             {"base": 0.20, "vol": 0.12, "trend": -0.001},
    "us_refinery_utilization_low": {"base": 0.25, "vol": 0.10, "trend": 0.001},
    "eu_cbam_implementation":     {"base": 0.60, "vol": 0.10, "trend": 0.001},
    "panama_canal_disruption":    {"base": 0.20, "vol": 0.18, "trend": 0.001},
    "food_crisis":                {"base": 0.25, "vol": 0.15, "trend": 0.001},
    "potash_sanctions":           {"base": 0.35, "vol": 0.12, "trend": 0.001},
    "indonesia_nickel_ban":       {"base": 0.40, "vol": 0.10, "trend": 0.001},
    "india_infrastructure_boom":  {"base": 0.50, "vol": 0.10, "trend": 0.001},
    "south_africa_grid_crisis":   {"base": 0.60, "vol": 0.12, "trend": -0.002},
    "australia_china_trade_thaw": {"base": 0.35, "vol": 0.12, "trend": 0.001},
    # ── Monetary policy / central bank events ──
    "fed_rate_cut":               {"base": 0.40, "vol": 0.22, "trend": 0.003},
    "fed_rate_hike":              {"base": 0.15, "vol": 0.18, "trend": -0.003},
    "ecb_rate_cut":               {"base": 0.45, "vol": 0.15, "trend": 0.002},
    "boj_rate_hike":              {"base": 0.20, "vol": 0.12, "trend": 0.004},
    "us_government_shutdown":     {"base": 0.25, "vol": 0.20, "trend": 0.002},
    "us_debt_ceiling":            {"base": 0.18, "vol": 0.15, "trend": 0.001},
    "pboc_easing":                {"base": 0.50, "vol": 0.15, "trend": -0.001},
    "dollar_strength_extreme":    {"base": 0.30, "vol": 0.18, "trend": 0.002},
    "us_inflation_above_3":       {"base": 0.35, "vol": 0.20, "trend": -0.002},
    "us_unemployment_above_5":    {"base": 0.15, "vol": 0.12, "trend": 0.002},
    "natural_gas_above_5":        {"base": 0.20, "vol": 0.15, "trend": 0.001},
    "oil_below_50":               {"base": 0.10, "vol": 0.08, "trend": -0.001},
    # ── Production / Consumption / Supply-Chain contracts ──
    "us_oil_production_record":   {"base": 0.45, "vol": 0.12, "trend": 0.002},
    "opec_compliance_below_80":   {"base": 0.30, "vol": 0.18, "trend": 0.001},
    "china_oil_demand_slowdown":  {"base": 0.35, "vol": 0.20, "trend": 0.002},
    "us_gasoline_demand_peak":    {"base": 0.25, "vol": 0.10, "trend": 0.003},
    "global_lng_oversupply":      {"base": 0.30, "vol": 0.15, "trend": 0.002},
    "permian_basin_peak":         {"base": 0.15, "vol": 0.08, "trend": 0.002},
    "crack_spread_above_30":      {"base": 0.25, "vol": 0.18, "trend": 0.001},
    "lithium_price_rebound":      {"base": 0.30, "vol": 0.22, "trend": 0.003},
    "china_ev_sales_record":      {"base": 0.55, "vol": 0.15, "trend": 0.002},
    "battery_cathode_shift":      {"base": 0.20, "vol": 0.10, "trend": 0.003},
    "australia_lithium_mine_closure": {"base": 0.35, "vol": 0.15, "trend": 0.002},
    "central_bank_gold_buying":   {"base": 0.60, "vol": 0.12, "trend": 0.001},
    "silver_industrial_demand_surge": {"base": 0.30, "vol": 0.15, "trend": 0.002},
    "us_pipeline_capacity_constraint": {"base": 0.25, "vol": 0.12, "trend": 0.001},
    "lng_export_terminal_approval": {"base": 0.40, "vol": 0.10, "trend": 0.001},
    "china_rare_earth_processing_dominance": {"base": 0.70, "vol": 0.08, "trend": -0.001},
    "us_critical_minerals_act":   {"base": 0.35, "vol": 0.12, "trend": 0.003},
    "global_fertilizer_shortage": {"base": 0.25, "vol": 0.18, "trend": 0.001},
    "el_nino_la_nina_shift":      {"base": 0.40, "vol": 0.20, "trend": 0.000},
    "china_pmi_below_50":         {"base": 0.35, "vol": 0.22, "trend": 0.002},
    "global_shipping_congestion": {"base": 0.30, "vol": 0.18, "trend": 0.001},
    "eu_energy_crisis":           {"base": 0.20, "vol": 0.20, "trend": -0.002},
    "india_refinery_expansion":   {"base": 0.55, "vol": 0.10, "trend": 0.002},
    "uranium_supply_deficit":     {"base": 0.40, "vol": 0.15, "trend": 0.003},
    "smr_deployment_milestone":   {"base": 0.20, "vol": 0.10, "trend": 0.004},
}

DEFAULT_PROFILE = {"base": 0.30, "vol": 0.12, "trend": 0.001}


def main() -> None:
    np.random.seed(42)
    days = 1700
    end = datetime.now(UTC).replace(tzinfo=None)
    dates = pd.date_range(end - timedelta(days=days), periods=days, freq="D")

    series: dict[str, pd.Series] = {}
    for eid in EVENT_MAPPINGS:
        prof = EVENT_PROFILES.get(eid, DEFAULT_PROFILE)

        # --- REALISTIC v2: much noisier, regime breaks, lower signal-to-noise ---
        # Daily noise 5x higher than v1 (real prediction markets are very noisy)
        noise = np.random.normal(0, prof["vol"] * 2.5 / np.sqrt(252), days)

        # More frequent shock days (real markets have many news events)
        n_shocks = max(15, days // 50)
        shock_idx = np.random.choice(days, size=n_shocks, replace=False)
        noise[shock_idx] += np.random.normal(0, prof["vol"] * 5, n_shocks)

        # Regime breaks: 3-5 abrupt level shifts (election results, policy changes)
        n_regimes = np.random.randint(3, 6)
        regime_idx = sorted(np.random.choice(range(100, days - 100), size=n_regimes, replace=False))
        regime_shifts = np.zeros(days)
        for ridx in regime_idx:
            shift_magnitude = np.random.uniform(-0.25, 0.25)
            regime_shifts[ridx:] += shift_magnitude

        # Trend component — much weaker (real markets are nearly efficient)
        t = np.arange(days) / 252
        trend = prof["trend"] * 0.1 * t  # 10x weaker trend than v1

        # Mean-reverting random walk with low reversion speed
        p = np.zeros(days)
        p[0] = prof["base"]
        reversion_speed = 0.02  # slower reversion (was 0.05)
        for i in range(1, days):
            p[i] = p[i - 1] + reversion_speed * (prof["base"] + trend[i] + regime_shifts[i] - p[i - 1]) + noise[i]
        p = np.clip(p, 0.01, 0.99)

        # Add microstructure noise (bid-ask bounce in prediction markets)
        microstructure = np.random.normal(0, 0.005, days)
        p = np.clip(p + microstructure, 0.01, 0.99)

        series[eid] = pd.Series(p, index=dates)

    df = pd.DataFrame(series)

    # ── save ──
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    df.to_csv(out_dir / "event_history_cache.csv")
    (out_dir / "event_history_cache_meta.json").write_text(
        json.dumps(
            {
                "cached_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
                "events": list(series.keys()),
                "rows": len(df),
                "source": "synthetic_v1",
            }
        )
    )

    print(f"Created synthetic event cache: {len(series)} events, {len(df)} days")
    print(f"Date range: {dates[0].date()} to {dates[-1].date()}")
    for eid in list(series)[:5]:
        s = df[eid]
        print(f"  {eid}: mean={s.mean():.3f}, std={s.std():.3f}, last={s.iloc[-1]:.3f}")


if __name__ == "__main__":
    main()
