from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(slots=True)
class TickerConfig:
    ticker: str
    sector: str
    beta_brent: float
    beta_td3: float
    beta_bdi: float
    idio_vol: float
    lag_strength: float


TICKERS = [
    TickerConfig("TNK", "shipping", 0.11, 0.70, 0.18, 0.018, 0.45),
    TickerConfig("INSW", "shipping", 0.09, 0.62, 0.15, 0.017, 0.48),
    TickerConfig("STNG", "shipping", 0.12, 0.58, 0.16, 0.019, 0.52),
    TickerConfig("SBLK", "shipping", 0.04, 0.14, 0.58, 0.022, 0.46),
    TickerConfig("GOGL", "shipping", 0.03, 0.12, 0.54, 0.023, 0.49),
    TickerConfig("CIVI", "producer", 0.62, 0.08, 0.02, 0.021, 0.55),
    TickerConfig("SM", "producer", 0.58, 0.07, 0.01, 0.023, 0.58),
    TickerConfig("VTLE", "producer", 0.55, 0.06, 0.02, 0.024, 0.60),
]


def simulate_event_probabilities(n: int, rng: np.random.Generator) -> pd.DataFrame:
    p_hormuz = np.zeros(n)
    p_red = np.zeros(n)
    p_sanctions = np.zeros(n)
    p_oil100 = np.zeros(n)

    p_hormuz[0] = 0.12
    p_red[0] = 0.18
    p_sanctions[0] = 0.22
    p_oil100[0] = 0.25

    for t in range(1, n):
        jump_h = 0.0
        jump_r = 0.0
        jump_s = 0.0
        if rng.random() < 0.015:
            jump_h = rng.uniform(0.08, 0.25)
        if rng.random() < 0.018:
            jump_r = rng.uniform(0.06, 0.20)
        if rng.random() < 0.012:
            jump_s = rng.uniform(0.05, 0.18)

        p_hormuz[t] = np.clip(0.94 * p_hormuz[t - 1] + 0.006 + rng.normal(0, 0.01) + jump_h, 0.01, 0.90)
        p_red[t] = np.clip(
            0.92 * p_red[t - 1] + 0.008 + 0.18 * p_hormuz[t] + rng.normal(0, 0.012) + jump_r, 0.01, 0.95
        )
        p_sanctions[t] = np.clip(
            0.93 * p_sanctions[t - 1] + 0.005 + 0.10 * p_hormuz[t] + rng.normal(0, 0.011) + jump_s,
            0.01,
            0.95,
        )
        p_oil100[t] = np.clip(
            0.90 * p_oil100[t - 1]
            + 0.006
            + 0.33 * p_hormuz[t]
            + 0.18 * p_sanctions[t]
            + rng.normal(0, 0.014),
            0.01,
            0.98,
        )

    return pd.DataFrame(
        {
            "p_hormuz": p_hormuz,
            "p_redsea": p_red,
            "p_sanctions": p_sanctions,
            "p_oil100": p_oil100,
        }
    )


def simulate_market_returns(prob_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(prob_df)
    p_h = prob_df["p_hormuz"].values
    p_r = prob_df["p_redsea"].values
    p_s = prob_df["p_sanctions"].values
    p_o = prob_df["p_oil100"].values

    realized_h = rng.binomial(1, p_h)
    realized_r = rng.binomial(1, p_r)
    realized_s = rng.binomial(1, p_s)

    surprise_h = realized_h - p_h
    surprise_r = realized_r - p_r
    surprise_s = realized_s - p_s

    mu_brent = 0.0001 + 0.0038 * p_h + 0.0015 * p_r + 0.0012 * p_s + 0.0014 * p_o
    mu_td3 = 0.0001 + 0.0043 * p_h + 0.0031 * p_r + 0.0006 * p_s
    mu_bdi = 0.0001 + 0.0030 * p_r + 0.0011 * p_h

    brent = mu_brent + 0.012 * surprise_h + 0.005 * surprise_r + 0.003 * surprise_s + rng.normal(0.0, 0.012, n)
    td3 = mu_td3 + 0.014 * surprise_h + 0.009 * surprise_r + 0.002 * surprise_s + rng.normal(0.0, 0.014, n)
    bdi = mu_bdi + 0.011 * surprise_r + 0.003 * surprise_h + rng.normal(0.0, 0.013, n)

    brent = np.clip(brent, -0.13, 0.13)
    td3 = np.clip(td3, -0.16, 0.16)
    bdi = np.clip(bdi, -0.15, 0.15)

    return pd.DataFrame(
        {
            "ret_brent": brent,
            "ret_td3": td3,
            "ret_bdi": bdi,
            "realized_hormuz": realized_h,
            "realized_redsea": realized_r,
            "realized_sanctions": realized_s,
        }
    )


def expected_factor_returns(prob_df: pd.DataFrame) -> pd.DataFrame:
    exp_brent = (
        0.0001
        + 0.0038 * prob_df["p_hormuz"]
        + 0.0015 * prob_df["p_redsea"]
        + 0.0012 * prob_df["p_sanctions"]
        + 0.0014 * prob_df["p_oil100"]
    )
    exp_td3 = 0.0001 + 0.0043 * prob_df["p_hormuz"] + 0.0031 * prob_df["p_redsea"] + 0.0006 * prob_df["p_sanctions"]
    exp_bdi = 0.0001 + 0.0030 * prob_df["p_redsea"] + 0.0011 * prob_df["p_hormuz"]
    return pd.DataFrame({"exp_brent": exp_brent, "exp_td3": exp_td3, "exp_bdi": exp_bdi})


def build_signal_dataset(
    dates: pd.DatetimeIndex,
    prob_df: pd.DataFrame,
    market_df: pd.DataFrame,
    exp_df: pd.DataFrame,
    hold_days: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    n = len(dates)
    trade_cost = 0.012  # 120 bps round-trip + borrow/impact proxy for 1-3m holds

    for cfg in TICKERS:
        realized = (
            cfg.beta_brent * market_df["ret_brent"].values
            + cfg.beta_td3 * market_df["ret_td3"].values
            + cfg.beta_bdi * market_df["ret_bdi"].values
            + rng.normal(0, cfg.idio_vol, n)
        )
        realized = np.clip(realized, -0.25, 0.25)
        exp_fundamental = (
            cfg.beta_brent * exp_df["exp_brent"].values
            + cfg.beta_td3 * exp_df["exp_td3"].values
            + cfg.beta_bdi * exp_df["exp_bdi"].values
        )

        market_implied = 0.58 * exp_fundamental + rng.normal(0, 0.0008, n)

        # Lag means market has only partially incorporated expected repricing.
        model_edge_daily = exp_fundamental - cfg.lag_strength * market_implied
        signal_strength = (model_edge_daily * hold_days) - trade_cost

        for t in range(n - hold_days):
            if signal_strength[t] <= 0.005:
                continue
            future_path = realized[t + 1 : t + 1 + hold_days]
            if len(future_path) < hold_days:
                continue
            gross = float(np.prod(1.0 + future_path) - 1.0)
            net = gross - trade_cost
            rows.append(
                {
                    "date": dates[t],
                    "ticker": cfg.ticker,
                    "sector": cfg.sector,
                    "p_hormuz": float(prob_df.iloc[t]["p_hormuz"]),
                    "p_redsea": float(prob_df.iloc[t]["p_redsea"]),
                    "p_sanctions": float(prob_df.iloc[t]["p_sanctions"]),
                    "p_oil100": float(prob_df.iloc[t]["p_oil100"]),
                    "signal_strength": float(signal_strength[t]),
                    "gross_return": gross,
                    "net_return": net,
                    "success": int(net > 0),
                }
            )
    return pd.DataFrame(rows)


def summarize_opportunities(signals: pd.DataFrame) -> dict[str, object]:
    if signals.empty:
        return {
            "trade_count": 0,
            "message": "No signals generated",
        }

    def label_regime(row: pd.Series) -> str:
        if row["p_hormuz"] > 0.35:
            return "High Hormuz Risk"
        if row["p_redsea"] > 0.45:
            return "High Red Sea Disruption"
        if row["p_sanctions"] > 0.40:
            return "High Sanctions Risk"
        if row["p_oil100"] > 0.55:
            return "High Oil>100 Probability"
        return "Baseline/Low Geopolitical"

    out = signals.copy()
    out["regime"] = out.apply(label_regime, axis=1)
    out["month"] = out["date"].dt.month
    out["quarter"] = out["date"].dt.to_period("Q").astype(str)

    overall = {
        "trade_count": int(len(out)),
        "hit_rate": float(out["success"].mean()),
        "avg_net_return": float(out["net_return"].mean()),
        "median_net_return": float(out["net_return"].median()),
        "p10_net_return": float(out["net_return"].quantile(0.10)),
        "p90_net_return": float(out["net_return"].quantile(0.90)),
        "top_decile_avg_net_return": float(out[out["signal_strength"] >= out["signal_strength"].quantile(0.9)]["net_return"].mean()),
    }

    by_regime = (
        out.groupby("regime")
        .agg(
            trades=("net_return", "size"),
            hit_rate=("success", "mean"),
            avg_net_return=("net_return", "mean"),
            median_net_return=("net_return", "median"),
        )
        .sort_values("avg_net_return", ascending=False)
        .reset_index()
    )

    by_ticker = (
        out.groupby(["ticker", "sector"])
        .agg(
            trades=("net_return", "size"),
            hit_rate=("success", "mean"),
            avg_net_return=("net_return", "mean"),
            avg_signal=("signal_strength", "mean"),
        )
        .sort_values("avg_net_return", ascending=False)
        .reset_index()
    )

    by_month = (
        out.groupby("month")
        .agg(
            trades=("net_return", "size"),
            hit_rate=("success", "mean"),
            avg_net_return=("net_return", "mean"),
        )
        .sort_values("avg_net_return", ascending=False)
        .reset_index()
    )

    return {
        "overall": overall,
        "best_regimes": by_regime.head(5).to_dict(orient="records"),
        "best_tickers": by_ticker.head(8).to_dict(orient="records"),
        "best_months": by_month.head(5).to_dict(orient="records"),
    }


def main() -> None:
    rng = np.random.default_rng(20260308)
    dates = pd.bdate_range(start="2020-01-02", end="2025-12-31")
    prob_df = simulate_event_probabilities(len(dates), rng)
    market_df = simulate_market_returns(prob_df, rng)
    exp_df = expected_factor_returns(prob_df)
    signals = build_signal_dataset(
        dates=dates,
        prob_df=prob_df,
        market_df=market_df,
        exp_df=exp_df,
        hold_days=40,
        rng=rng,
    )
    summary = summarize_opportunities(signals)
    output = {
        "generated_at": datetime.utcnow().isoformat(),
        "sample_start": str(dates.min().date()),
        "sample_end": str(dates.max().date()),
        "hold_days": 40,
        "summary": summary,
    }

    out_path = Path(__file__).resolve().parent.parent / "analysis_output"
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / "opportunity_regime_backtest.json"
    csv_path = out_path / "opportunity_signals.csv"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    signals.to_csv(csv_path, index=False)

    print(json.dumps(output, indent=2))
    print(f"\nWrote:\n- {json_path}\n- {csv_path}")


if __name__ == "__main__":
    main()
