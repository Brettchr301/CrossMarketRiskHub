from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable
import sys

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.providers.real_prediction import EVENT_MAPPINGS, RealPredictionProvider


UNIVERSE = ["TNK", "INSW", "STNG", "SBLK", "DHT", "FRO", "NAT", "SM", "MTDR", "RRC", "MUR", "AR"]
FACTORS = {
    "ret_brent": "BZ=F",
    "ret_wti": "CL=F",
    "ret_bdry": "BDRY",
    "ret_boat": "BOAT",
    "ret_sea": "SEA",
}


@dataclass(slots=True)
class BacktestConfig:
    hold_days: int = 20
    min_history_days: int = 90
    min_signal_edge: float = 0.02
    trade_cost: float = 0.008
    max_positions_per_day: int = 3


def _ols_predict_one_step(x_train: np.ndarray, y_train: np.ndarray, x_now: np.ndarray) -> float:
    x_aug = np.c_[np.ones(len(x_train)), x_train]
    beta, *_ = np.linalg.lstsq(x_aug, y_train, rcond=None)
    x_now_aug = np.r_[1.0, x_now]
    return float(x_now_aug @ beta)


def _download_close_frame(tickers: Iterable[str], period: str = "5y") -> pd.DataFrame:
    tickers = list(tickers)
    raw = yf.download(
        tickers,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
        group_by="ticker",
    )
    if raw.empty:
        return pd.DataFrame()
    close = pd.DataFrame(index=raw.index)
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            if (t, "Close") in raw.columns:
                close[t] = raw[(t, "Close")]
    else:
        if len(tickers) == 1 and "Close" in raw.columns:
            close[tickers[0]] = raw["Close"]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.sort_index().dropna(how="all")


def _fetch_prediction_probabilities() -> pd.DataFrame:
    rp = RealPredictionProvider()
    now = datetime.now(UTC)
    start_ts = int(datetime(2021, 1, 1, tzinfo=UTC).timestamp())
    end_ts = int(now.timestamp())

    series_map: dict[str, pd.Series] = {}
    for event_id, mapping in EVENT_MAPPINGS.items():
        poly = rp.polymarket.fetch_event_history(mapping)
        kalshi = rp.kalshi.fetch_event_history(mapping, start_ts=start_ts, end_ts=end_ts, period_interval=1440)
        cols = [s for s in [poly, kalshi] if s is not None and not s.empty]
        if not cols:
            continue
        merged = pd.concat(cols, axis=1).sort_index().ffill()
        combined = merged.mean(axis=1).clip(0.0, 1.0)
        series_map[event_id] = combined.rename(event_id)
    if not series_map:
        raise RuntimeError("No real prediction-market histories found for configured events.")
    frame = pd.concat(series_map.values(), axis=1).sort_index()
    full_index = pd.date_range(frame.index.min(), frame.index.max(), freq="D")
    frame = frame.reindex(full_index).sort_index().ffill().fillna(0.5)
    return frame


def _build_feature_frame(prob_df: pd.DataFrame) -> pd.DataFrame:
    out = prob_df.copy()
    for col in prob_df.columns:
        out[f"d_{col}"] = prob_df[col].diff()
    out = out.dropna(how="any")
    return out


def run_real_backtest(config: BacktestConfig) -> dict[str, object]:
    prob = _fetch_prediction_probabilities()

    factor_close = _download_close_frame(FACTORS.values(), period="5y")
    eq_close = _download_close_frame(UNIVERSE, period="5y")
    if factor_close.empty or eq_close.empty:
        raise RuntimeError("Unable to download required real market data from yfinance.")

    data = prob.join(factor_close, how="inner").join(eq_close, how="inner")
    data = data.dropna(how="any")
    overlap_warning = None
    if len(data) < config.min_history_days:
        if len(data) < 60:
            raise RuntimeError(
                f"Insufficient overlapping real data after joins ({len(data)} rows). "
                "This depends on currently available prediction-market history."
            )
        overlap_warning = (
            f"Reduced effective history to {len(data)} rows (below configured minimum "
            f"{config.min_history_days}); interpret metrics with caution."
        )

    price_cols = list(FACTORS.values()) + [c for c in UNIVERSE if c in data.columns]
    ret = data[price_cols].pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="any")
    feature_base = _build_feature_frame(data[list(prob.columns)])
    ret = ret.join(feature_base, how="inner").dropna(how="any")

    feature_cols = [c for c in ret.columns if c.startswith("d_") or c in prob.columns]
    factor_cols = list(FACTORS.values())
    eq_cols = [c for c in UNIVERSE if c in ret.columns]

    # Step 1: infer next-day factor returns from event probabilities.
    factor_preds = pd.DataFrame(index=ret.index, columns=factor_cols, dtype=float)
    train_start = max(40, len(ret) // 4)
    for t in range(train_start, len(ret) - 1):
        x_train = ret.iloc[:t][feature_cols].values
        x_now = ret.iloc[t][feature_cols].values
        for fc in factor_cols:
            y_train = ret.iloc[1 : t + 1][fc].values
            if len(y_train) != len(x_train):
                continue
            factor_preds.iloc[t + 1, factor_preds.columns.get_loc(fc)] = _ols_predict_one_step(
                x_train, y_train, x_now
            )
    factor_preds = factor_preds.dropna(how="all")

    # Step 2: infer ticker expected return from predicted factors and rolling betas.
    ticker_pred = pd.DataFrame(index=factor_preds.index, columns=eq_cols, dtype=float)
    roll = 60
    for ticker in eq_cols:
        for i, dt in enumerate(factor_preds.index):
            loc = ret.index.get_loc(dt)
            if loc < roll:
                continue
            train_slice = ret.iloc[loc - roll : loc]
            x_train = train_slice[factor_cols].values
            y_train = train_slice[ticker].values
            x_now = factor_preds.loc[dt, factor_cols].values
            if np.isnan(x_now).any():
                continue
            ticker_pred.loc[dt, ticker] = _ols_predict_one_step(x_train, y_train, x_now)

    # Step 3: trade top expected opportunities and evaluate realized horizon returns.
    trades: list[dict[str, object]] = []
    pred = ticker_pred.dropna(how="all")
    for dt in pred.index:
        loc = ret.index.get_loc(dt)
        if loc + config.hold_days >= len(ret):
            continue
        day_preds = pred.loc[dt].dropna().sort_values(ascending=False)
        if day_preds.empty:
            continue
        candidates = day_preds.head(config.max_positions_per_day)
        for ticker, pred_daily in candidates.items():
            horizon_pred = float(pred_daily) * config.hold_days
            if horizon_pred < config.min_signal_edge:
                continue
            future_rets = ret.iloc[loc + 1 : loc + 1 + config.hold_days][ticker].values
            if len(future_rets) < config.hold_days:
                continue
            gross = float(np.prod(1.0 + future_rets) - 1.0)
            net = gross - config.trade_cost
            row = {
                "date": dt,
                "ticker": ticker,
                "pred_horizon_return": horizon_pred,
                "gross_return": gross,
                "net_return": net,
                "success": int(net > 0),
            }
            for evt in prob.columns:
                row[evt] = float(data.loc[dt, evt])
            trades.append(row)
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        raise RuntimeError("No trades passed the minimum signal threshold in real-data backtest.")

    def regime(row: pd.Series) -> str:
        if row.get("hormuz_closure", 0.0) >= 0.45:
            return "High Hormuz"
        if row.get("red_sea_disruption", 0.0) >= 0.50:
            return "High Red Sea"
        if row.get("sanctions_escalation", 0.0) >= 0.55:
            return "High Sanctions"
        if row.get("oil_above_100", 0.0) >= 0.50:
            return "High Oil>100"
        return "Baseline"

    trades_df["regime"] = trades_df.apply(regime, axis=1)
    trades_df["month"] = pd.to_datetime(trades_df["date"]).dt.month

    overall = {
        "trade_count": int(len(trades_df)),
        "hit_rate": float(trades_df["success"].mean()),
        "avg_net_return": float(trades_df["net_return"].mean()),
        "median_net_return": float(trades_df["net_return"].median()),
        "p10_net_return": float(trades_df["net_return"].quantile(0.10)),
        "p90_net_return": float(trades_df["net_return"].quantile(0.90)),
    }
    by_regime = (
        trades_df.groupby("regime")
        .agg(trades=("net_return", "size"), hit_rate=("success", "mean"), avg_net_return=("net_return", "mean"))
        .sort_values("avg_net_return", ascending=False)
        .reset_index()
    )
    by_ticker = (
        trades_df.groupby("ticker")
        .agg(trades=("net_return", "size"), hit_rate=("success", "mean"), avg_net_return=("net_return", "mean"))
        .sort_values("avg_net_return", ascending=False)
        .reset_index()
    )
    by_month = (
        trades_df.groupby("month")
        .agg(trades=("net_return", "size"), hit_rate=("success", "mean"), avg_net_return=("net_return", "mean"))
        .sort_values("avg_net_return", ascending=False)
        .reset_index()
    )

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "date_range": [str(trades_df["date"].min().date()), str(trades_df["date"].max().date())],
        "hold_days": config.hold_days,
        "overlap_warning": overlap_warning,
        "overall": overall,
        "best_regimes": by_regime.head(5).to_dict(orient="records"),
        "best_tickers": by_ticker.head(8).to_dict(orient="records"),
        "best_months": by_month.head(5).to_dict(orient="records"),
    }
    out_dir = Path(__file__).resolve().parent.parent / "analysis_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "real_data_backtest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    trades_df.to_csv(out_dir / "real_data_backtest_trades.csv", index=False)
    return summary


def main() -> None:
    cfg = BacktestConfig()
    summary = run_real_backtest(cfg)
    print(json.dumps(summary, indent=2))
    print("\nWrote analysis_output/real_data_backtest_summary.json and analysis_output/real_data_backtest_trades.csv")


if __name__ == "__main__":
    main()
