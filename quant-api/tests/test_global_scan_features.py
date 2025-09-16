from __future__ import annotations

import numpy as np
import pandas as pd

from app.modeling.global_scan import GlobalOpportunityService


def test_build_feature_frame_keeps_varying_xplat_and_drops_constant_columns():
    svc = GlobalOpportunityService()
    idx = pd.date_range("2023-01-01", periods=320, freq="D")
    ticker = "TEST"

    equity_prices = pd.DataFrame(
        {ticker: 20.0 + np.cumsum(np.linspace(-0.02, 0.03, len(idx)))},
        index=idx,
    )
    equity_volume = pd.DataFrame(
        {ticker: 500_000.0 + 10_000.0 * np.sin(np.linspace(0, 10, len(idx)))},
        index=idx,
    )
    factors = pd.DataFrame(
        {
            "BZ=F": 80.0 + np.cumsum(np.linspace(-0.01, 0.02, len(idx))),
            "CL=F": 75.0 + np.cumsum(np.linspace(-0.008, 0.018, len(idx))),
            "BOAT": 25.0 + np.cumsum(np.linspace(-0.002, 0.003, len(idx))),
            "BDRY": 16.0 + np.cumsum(np.linspace(-0.001, 0.002, len(idx))),
            "^VIX": 18.0 + 3.0 * np.sin(np.linspace(0, 12, len(idx))),
            "DX-Y.NYB": 103.0 + np.cumsum(np.linspace(-0.005, 0.005, len(idx))),
            "^TNX": 4.0 + np.cumsum(np.linspace(-0.001, 0.001, len(idx))),
            "^IRX": 5.0 + np.cumsum(np.linspace(-0.0008, 0.0008, len(idx))),
            "GC=F": 1900.0 + np.cumsum(np.linspace(-0.1, 0.1, len(idx))),
            "^GSPC": 4200.0 + np.cumsum(np.linspace(-0.5, 0.7, len(idx))),
        },
        index=idx,
    )

    events = pd.DataFrame(
        {
            "hormuz_closure": 0.2 + 0.1 * np.sin(np.linspace(0, 8, len(idx))),
            "red_sea_disruption": 0.3 + 0.1 * np.cos(np.linspace(0, 9, len(idx))),
            "sanctions_escalation": 0.25 + 0.05 * np.sin(np.linspace(0, 7, len(idx))),
            "oil_above_100": 0.15 + 0.1 * np.sin(np.linspace(0, 11, len(idx))),
            "opec_production_cut": 0.35 + 0.05 * np.cos(np.linspace(0, 6, len(idx))),
            "panama_canal_disruption": 0.18 + 0.06 * np.sin(np.linspace(0, 5, len(idx))),
            "china_stimulus": 0.4 + 0.07 * np.cos(np.linspace(0, 7, len(idx))),
            "us_spr_release": 0.3 + 0.04 * np.sin(np.linspace(0, 9, len(idx))),
            "us_refinery_utilization_low": 0.22 + 0.05 * np.cos(np.linspace(0, 10, len(idx))),
        },
        index=idx,
    )

    xplat_spreads = pd.DataFrame(
        {
            "pm_xplat_spread_hormuz_closure": np.sin(np.linspace(0, 7, len(idx))) * 0.1,
            "pm_xplat_spread_sanctions_escalation": np.zeros(len(idx)),
        },
        index=idx,
    )

    frame = svc._build_feature_frame(
        ticker=ticker,
        equity_prices=equity_prices,
        equity_volume=equity_volume,
        factors=factors,
        events=events,
        xplat_spreads=xplat_spreads,
        contracts=[],
        contract_delta_map={},
        lookback_days=260,
        spot_proxy="BOAT",
        fwd_proxy="BDRY",
    )
    assert frame is not None
    assert len(frame) >= 140
    assert "pm_xplat_spread_hormuz_closure" in frame.columns
    assert "pm_xplat_spread_hormuz_closure_d1" in frame.columns
    assert "pm_xplat_spread_sanctions_escalation" not in frame.columns
