"""Tests for actionable alpha modules: analog matcher, position sizer, signal monitor."""
from __future__ import annotations

import json
from datetime import datetime

import pytest

from app.election.signals.analog_matcher import (
    MISPRICED_2022,
    AnalogMatch,
    match_2026_to_analogs,
    get_mispricing_forecast,
)
from app.election.signals.position_sizer import (
    PortfolioRecommendation,
    compute_kelly_fraction,
    build_portfolio,
    format_trade_plan,
)
from app.election.signals.monitor import SignalMonitor, SignalState


# --- Analog Matcher Tests ---

class TestAnalogMatcher:
    def test_same_state_same_type_highest(self):
        """PA Senate 2026 should match PA Senate 2022 as top analog."""
        matches = match_2026_to_analogs(
            target_races=[{"state": "PA", "type": "senate"}],
        )
        top = matches["PA_senate"][0]
        assert top.analog_state == "PA"
        assert top.analog_type == "senate"
        assert top.analog_cycle == 2022

    def test_similarity_score_range(self):
        """All similarity scores should be in [0, 1]."""
        matches = match_2026_to_analogs()
        for key, analogs in matches.items():
            for a in analogs:
                assert 0.0 <= a.similarity_score <= 1.0, f"{key}: {a.similarity_score}"

    def test_mispricing_forecast_pa(self):
        """PA Senate 2026 forecast should show expected_error > 30pp."""
        matches = match_2026_to_analogs(
            target_races=[{"state": "PA", "type": "senate"}],
        )
        forecasts = get_mispricing_forecast(matches)
        pa = next(f for f in forecasts if f["state"] == "PA")
        assert pa["expected_error_pp"] > 30
        assert pa["confidence"] in ("high", "medium")

    def test_all_2026_races_matched(self):
        """All 19 races (14 Senate + 5 Governor) should have at least 1 analog."""
        matches = match_2026_to_analogs()
        assert len(matches) == 19
        for key, analogs in matches.items():
            assert len(analogs) >= 1, f"{key} has no analogs"

    def test_mispriced_2022_flag(self):
        """Analogs from 2022 mispriced races should have flag set."""
        matches = match_2026_to_analogs(
            target_races=[{"state": "PA", "type": "senate"}],
        )
        pa_top = matches["PA_senate"][0]
        assert pa_top.was_mispriced_2022 is True

    def test_pvi_delta_computed(self):
        """PVI difference should be calculated."""
        matches = match_2026_to_analogs(
            target_races=[{"state": "PA", "type": "senate"}],
        )
        pa_top = matches["PA_senate"][0]
        assert isinstance(pa_top.pvi_delta, float)


# --- Position Sizer Tests ---

class TestPositionSizer:
    def test_kelly_fair_bet(self):
        """No edge → Kelly = 0."""
        assert compute_kelly_fraction(0.5, 0.5) == 0.0

    def test_kelly_edge_exists(self):
        """Positive edge → Kelly > 0."""
        kelly = compute_kelly_fraction(0.65, 0.38)
        assert kelly > 0.0

    def test_kelly_capped(self):
        """Even with huge edge, Kelly ≤ 0.25."""
        kelly = compute_kelly_fraction(0.99, 0.10)
        assert kelly <= 0.25

    def test_kelly_no_edge(self):
        """Estimated prob below market → Kelly = 0."""
        kelly = compute_kelly_fraction(0.30, 0.50)
        assert kelly == 0.0

    def test_portfolio_max_exposure(self):
        """Total notional ≤ max_portfolio_pct × bankroll."""
        signal = {
            "signal_strength": "strong",
            "races": [
                {"state": s, "race_type": "senate", "market_prob_dem": 0.35, "polling_avg_dem": 0.60}
                for s in ["PA", "MI", "WI", "AZ", "GA", "NV", "NC", "NH", "ME"]
            ],
        }
        portfolio = build_portfolio(signal, [], bankroll=10000, max_portfolio_pct=0.60)
        assert portfolio.total_notional <= 6000.01  # allow float rounding

    def test_correlation_reduces_size(self):
        """With many correlated races, each position is smaller."""
        signal_1 = {
            "signal_strength": "strong",
            "races": [{"state": "PA", "race_type": "senate", "market_prob_dem": 0.35, "polling_avg_dem": 0.60}],
        }
        signal_9 = {
            "signal_strength": "strong",
            "races": [
                {"state": s, "race_type": "senate", "market_prob_dem": 0.35, "polling_avg_dem": 0.60}
                for s in ["PA", "MI", "WI", "AZ", "GA", "NV", "NC", "NH", "ME"]
            ],
        }
        p1 = build_portfolio(signal_1, [], bankroll=10000)
        p9 = build_portfolio(signal_9, [], bankroll=10000)
        if p1.positions and p9.positions:
            assert p9.positions[0].adjusted_fraction < p1.positions[0].adjusted_fraction

    def test_trade_plan_format(self):
        """Trade plan should be non-empty with key fields."""
        signal = {
            "signal_strength": "strong",
            "races": [{"state": "PA", "race_type": "senate", "market_prob_dem": 0.35, "polling_avg_dem": 0.60}],
        }
        portfolio = build_portfolio(signal, [], bankroll=10000)
        plan = format_trade_plan(portfolio)
        assert "TRADE PLAN" in plan
        assert "PA" in plan
        assert "$" in plan

    def test_no_positions_when_no_signal(self):
        """Weak signal → 0 positions."""
        signal = {"signal_strength": "weak", "races": []}
        portfolio = build_portfolio(signal, [])
        assert portfolio.n_positions == 0


# --- Signal Monitor Tests ---

class TestSignalMonitor:
    def test_poll_updates_state(self):
        """Consecutive polls with different strengths should set strength_changed."""
        mon = SignalMonitor()
        mon.poll({"signal_strength": "weak", "skew_ratio": 0.55})
        state = mon.poll({"signal_strength": "moderate", "skew_ratio": 0.70})
        assert state.strength_changed is True
        assert state.current_strength == "moderate"
        assert state.previous_strength == "weak"

    def test_consecutive_polls_counter(self):
        """Same strength repeatedly should increment counter."""
        mon = SignalMonitor()
        for _ in range(5):
            mon.poll({"signal_strength": "moderate", "skew_ratio": 0.70})
        assert mon.state.consecutive_polls == 5

    def test_no_alert_on_same_strength(self):
        """Same strength twice → strength_changed=False on second."""
        mon = SignalMonitor()
        mon.poll({"signal_strength": "moderate", "skew_ratio": 0.70})
        state = mon.poll({"signal_strength": "moderate", "skew_ratio": 0.71})
        assert state.strength_changed is False

    def test_skew_history_capped(self):
        """After 25 polls, skew_history should have exactly 20 entries."""
        mon = SignalMonitor()
        for i in range(25):
            mon.poll({"signal_strength": "none", "skew_ratio": i * 0.04})
        assert len(mon.state.skew_history) == 20

    def test_status_json_serializable(self):
        """get_status() output must be JSON-serializable."""
        mon = SignalMonitor()
        mon.poll({"signal_strength": "strong", "skew_ratio": 0.85, "analog_2022_similarity": 0.75,
                  "n_competitive_races": 11, "n_dem_underpriced": 8, "avg_mispricing_pp": 45.0})
        status = mon.get_status()
        json_str = json.dumps(status)
        assert len(json_str) > 10

    def test_trade_plan_generated_for_strong(self):
        """Strong signal should have trade_plan if provided."""
        mon = SignalMonitor()
        mon.poll({"signal_strength": "strong", "skew_ratio": 0.85,
                  "trade_plan": "BUY PA DEM"})
        assert mon.state.trade_plan == "BUY PA DEM"

    def test_trade_plan_none_for_weak(self):
        """Weak signal should NOT have trade_plan."""
        mon = SignalMonitor()
        mon.poll({"signal_strength": "weak", "skew_ratio": 0.55})
        assert mon.state.trade_plan is None

    def test_first_detected_at_set(self):
        """first_detected_at set on first non-none signal."""
        mon = SignalMonitor()
        mon.poll({"signal_strength": "none", "skew_ratio": 0.0})
        assert mon.state.first_detected_at is None
        mon.poll({"signal_strength": "weak", "skew_ratio": 0.55})
        assert mon.state.first_detected_at is not None
