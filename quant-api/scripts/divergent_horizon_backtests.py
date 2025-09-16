from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.modeling.cost_model import estimate_total_cost_bps, net_return_after_cost
from app.modeling.global_scan import GlobalOpportunityService
from app.modeling.global_universe import GLOBAL_COMMODITY_UNIVERSE, global_universe_tickers
from app.modeling.research_hub import ResearchHubService
from app.providers.real_prediction import EVENT_MAPPINGS, RealPredictionProvider


DEFAULT_HORIZONS = [1, 5, 10, 21, 42, 63, 126, 252]
FACTOR_TICKERS = ["BZ=F", "CL=F", "BOAT", "SEA", "BDRY", "ACWI"]
DEFAULT_EVAL_DAYS = 504
MIN_TICKERS_PER_DATE = 14
TOP_QUANTILE = 0.2

MODEL_VARIANTS: dict[str, dict[str, list[str]]] = {
    "market_only": {
        "must": ["brent_ret", "wti_ret"],
        "candidates": ["ship_spot_ret", "ship_fwd_ret", "stock_ret_1d", "mom_5", "mom_21"],
    },
    "market_event": {
        "must": ["brent_ret", "wti_ret", "hormuz_closure_d1", "red_sea_disruption_d1"],
        "candidates": ["ship_spot_ret", "ship_fwd_ret", "sanctions_escalation_d1", "oil_above_100_d1", "mom_21"],
    },
    "convexity": {
        "must": ["brent_ret", "wti_ret", "brent_sq", "wti_sq", "hormuz_closure_d1", "hormuz_closure_d2"],
        "candidates": [
            "ship_spot_ret",
            "ship_fwd_ret",
            "ship_spot_sq",
            "ship_fwd_sq",
            "sanctions_escalation_d2",
            "oil_above_100_d2",
            "mom_5",
            "mom_21",
        ],
    },
    "full_cross_market": {
        "must": [
            "brent_ret",
            "wti_ret",
            "brent_sq",
            "wti_sq",
            "hormuz_closure_d1",
            "red_sea_disruption_d1",
            "sanctions_escalation_d1",
            "oil_above_100_d1",
        ],
        "candidates": [
            "ship_spot_ret",
            "ship_fwd_ret",
            "ship_spot_sq",
            "ship_fwd_sq",
            "hormuz_closure_d2",
            "red_sea_disruption_d2",
            "sanctions_escalation_d2",
            "oil_above_100_d2",
            "mom_5",
            "mom_21",
            "mom_63",
        ],
    },
}


@dataclass(slots=True)
class PredictionRecord:
    date: pd.Timestamp
    ticker: str
    commodity_type: str
    pred: float
    actual: float
    horizon_days: int
    variant: str


def _event_history(provider: RealPredictionProvider, event_ids: list[str]) -> pd.DataFrame:
    series_map: dict[str, pd.Series] = {}
    now = datetime.now(UTC)
    start_ts = int((now - timedelta(days=1900)).timestamp())
    end_ts = int(now.timestamp())
    for event_id in event_ids:
        mapping = EVENT_MAPPINGS.get(event_id)
        if mapping is None:
            continue
        poly = provider.polymarket.fetch_event_history(mapping)
        kalshi = provider.kalshi.fetch_event_history(
            mapping=mapping,
            start_ts=start_ts,
            end_ts=end_ts,
            period_interval=1440,
        )
        cols = [x for x in [poly, kalshi] if x is not None and not x.empty]
        if not cols:
            continue
        merged = pd.concat(cols, axis=1).sort_index().ffill()
        series_map[event_id] = merged.mean(axis=1).clip(0.0, 1.0)
    if not series_map:
        return pd.DataFrame()
    out = pd.concat(series_map.values(), axis=1).sort_index()
    out.columns = list(series_map)
    full_idx = pd.date_range(out.index.min(), out.index.max(), freq="D")
    return out.reindex(full_idx).ffill().fillna(0.5)


def _contract_features(hub: ResearchHubService, lookback_days: int = 780) -> pd.DataFrame:
    cache = hub._ensure_predictive_cache(lookback_days=lookback_days, max_markets=24)
    rows = sorted(cache.contracts, key=lambda x: abs(float(x["predictive_score"])), reverse=True)[:12]
    feature_cols: list[pd.Series] = []
    for row in rows:
        cid = row["market_id"]
        lag = int(row.get("lead_days", 0))
        delta = cache.contract_delta_map.get(cid)
        if delta is None or delta.empty:
            continue
        shifted = delta.shift(lag)
        feature_cols.append(shifted.rename(f"pm_{cid}"))
        feature_cols.append(shifted.diff().rename(f"pm2_{cid}"))
    if not feature_cols:
        return pd.DataFrame()
    out = pd.concat(feature_cols, axis=1).sort_index()
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _build_ticker_frames(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    contract_features: pd.DataFrame,
    spot_proxy: str,
    fwd_proxy: str,
    ticker_to_type: dict[str, str],
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for ticker, commodity_type in ticker_to_type.items():
        if ticker not in prices.columns:
            continue
        stock = prices[ticker].dropna()
        if len(stock) < 480:
            continue
        frame = pd.DataFrame(index=stock.index)
        frame["stock_px"] = stock
        frame["stock_ret_1d"] = stock.pct_change(fill_method=None)
        frame["mom_5"] = stock.pct_change(5, fill_method=None)
        frame["mom_21"] = stock.pct_change(21, fill_method=None)
        frame["mom_63"] = stock.pct_change(63, fill_method=None)
        frame["commodity_type"] = commodity_type

        if "BZ=F" in prices.columns:
            brent = prices["BZ=F"].reindex(frame.index).ffill()
            frame["brent_ret"] = brent.pct_change(fill_method=None)
            frame["brent_sq"] = frame["brent_ret"] * frame["brent_ret"]
            frame["brent_accel"] = frame["brent_ret"].diff()
        if "CL=F" in prices.columns:
            wti = prices["CL=F"].reindex(frame.index).ffill()
            frame["wti_ret"] = wti.pct_change(fill_method=None)
            frame["wti_sq"] = frame["wti_ret"] * frame["wti_ret"]
            frame["wti_accel"] = frame["wti_ret"].diff()
        if spot_proxy in prices.columns:
            sp = prices[spot_proxy].reindex(frame.index).ffill()
            frame["ship_spot_ret"] = sp.pct_change(fill_method=None)
            frame["ship_spot_sq"] = frame["ship_spot_ret"] * frame["ship_spot_ret"]
            frame["ship_spot_accel"] = frame["ship_spot_ret"].diff()
        if fwd_proxy in prices.columns:
            fw = prices[fwd_proxy].reindex(frame.index).ffill()
            frame["ship_fwd_ret"] = fw.pct_change(fill_method=None)
            frame["ship_fwd_sq"] = frame["ship_fwd_ret"] * frame["ship_fwd_ret"]
            frame["ship_fwd_accel"] = frame["ship_fwd_ret"].diff()

        ev = events.reindex(frame.index).ffill()
        for col in ["hormuz_closure", "red_sea_disruption", "sanctions_escalation", "oil_above_100"]:
            if col not in ev.columns:
                ev[col] = 0.5
            d1 = ev[col].diff()
            frame[f"{col}_d1"] = d1
            frame[f"{col}_d2"] = d1.diff()
            frame[f"{col}_sq"] = d1 * d1

        if not contract_features.empty:
            frame = frame.join(contract_features.reindex(frame.index).fillna(0.0), how="left")

        frame = frame.replace([np.inf, -np.inf], np.nan)
        out[ticker] = frame
    return out


def _feature_cols_for_variant(frame: pd.DataFrame, variant: str) -> list[str]:
    spec = MODEL_VARIANTS[variant]
    must = [c for c in spec["must"] if c in frame.columns]
    cand = [c for c in spec["candidates"] if c in frame.columns]
    if variant == "full_cross_market":
        pm = [c for c in frame.columns if c.startswith("pm_") or c.startswith("pm2_")]
        cand = cand + pm[:14]
    return list(dict.fromkeys(must + cand))


def _walkforward_predictions(
    frame: pd.DataFrame,
    feature_cols: list[str],
    horizon_days: int,
    ticker: str,
    commodity_type: str,
    eval_days: int,
) -> list[PredictionRecord]:
    if len(feature_cols) < 4:
        return []
    y = frame["stock_px"].shift(-horizon_days) / frame["stock_px"] - 1.0
    df = frame[feature_cols].copy()
    df["target"] = y
    df[feature_cols] = df[feature_cols].fillna(0.0)
    df = df.dropna(subset=["target"])
    if len(df) < 420:
        return []

    eval_start = max(252, len(df) - max(60, int(eval_days)))
    lookback = max(220, min(420, len(df) - 30))
    out: list[PredictionRecord] = []
    for i in range(eval_start, len(df)):
        if i - lookback < 0:
            continue
        train = df.iloc[i - lookback : i]
        x = train[feature_cols].values
        y_train = train["target"].values
        x_aug = np.c_[np.ones(len(x)), x]
        beta, *_ = np.linalg.lstsq(x_aug, y_train, rcond=None)
        row = df.iloc[i]
        pred = float(np.r_[1.0, row[feature_cols].values] @ beta)
        actual = float(row["target"])
        out.append(
            PredictionRecord(
                date=pd.Timestamp(df.index[i]),
                ticker=ticker,
                commodity_type=commodity_type,
                pred=pred,
                actual=actual,
                horizon_days=horizon_days,
                variant="",
            )
        )
    return out


def _strategy_from_predictions(
    rows: list[PredictionRecord],
    horizon_days: int,
    benchmark_forward: pd.Series,
    variant: str,
) -> tuple[dict[str, float], pd.Series]:
    if not rows:
        return {}, pd.Series(dtype=float)
    frame = pd.DataFrame(
        [
            {
                "date": x.date,
                "ticker": x.ticker,
                "commodity_type": x.commodity_type,
                "pred": x.pred,
                "actual": x.actual,
            }
            for x in rows
        ]
    )
    frame = frame.sort_values(["date", "pred"])
    pnl: list[tuple[pd.Timestamp, float, float, float]] = []
    for date, grp in frame.groupby("date"):
        grp = grp.sort_values("pred")
        if len(grp) < MIN_TICKERS_PER_DATE:
            continue
        n_leg = max(1, int(np.floor(TOP_QUANTILE * len(grp))))
        short_leg = grp.head(n_leg)
        long_leg = grp.tail(n_leg)
        gross = float(long_leg["actual"].mean() - short_leg["actual"].mean())
        cost_bps = estimate_total_cost_bps(
            commission_bps=2.0,
            spread_bps=12.0,
            slippage_bps=8.0,
            impact_bps=10.0,
            borrow_bps_annual=250.0,
            hold_days=horizon_days,
        )
        net = net_return_after_cost(gross, cost_bps)
        bench = float(benchmark_forward.get(date, 0.0))
        alpha = net - bench
        pnl.append((pd.Timestamp(date), gross, net, alpha))

    if not pnl:
        return {}, pd.Series(dtype=float)
    pnl_df = pd.DataFrame(pnl, columns=["date", "gross", "net", "alpha"]).set_index("date").sort_index()
    scale = 252.0 / max(1, horizon_days)
    avg_net = float(pnl_df["net"].mean())
    vol_net = float(pnl_df["net"].std(ddof=0))
    avg_alpha = float(pnl_df["alpha"].mean())
    vol_alpha = float(pnl_df["alpha"].std(ddof=0))
    sharpe = avg_net / max(vol_net, 1e-9) * np.sqrt(scale)
    ir = avg_alpha / max(vol_alpha, 1e-9) * np.sqrt(scale)
    hit = float((pnl_df["alpha"] > 0).mean())
    ann_alpha = avg_alpha * scale
    ann_return = avg_net * scale
    cagr = float((1.0 + pnl_df["net"]).prod() ** (scale / max(len(pnl_df), 1)) - 1.0)
    max_dd = float((pnl_df["net"].cumsum() - pnl_df["net"].cumsum().cummax()).min())
    metrics = {
        "variant": variant,
        "horizon_days": int(horizon_days),
        "n_periods": int(len(pnl_df)),
        "annualized_alpha": float(ann_alpha),
        "annualized_return": float(ann_return),
        "cagr": cagr,
        "information_ratio": float(ir),
        "sharpe": float(sharpe),
        "hit_rate": hit,
        "avg_alpha": avg_alpha,
        "max_drawdown": max_dd,
    }
    return metrics, pnl_df["alpha"]


def _sector_metrics(
    rows: list[PredictionRecord],
    horizon_days: int,
    benchmark_forward: pd.Series,
    variant: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not rows:
        return out
    by_type: dict[str, list[PredictionRecord]] = {}
    for row in rows:
        by_type.setdefault(row.commodity_type, []).append(row)
    for commodity_type, subset in by_type.items():
        tickers = {x.ticker for x in subset}
        if len(tickers) < 6:
            continue
        metrics, _ = _strategy_from_predictions(subset, horizon_days, benchmark_forward, variant)
        if not metrics:
            continue
        metrics["commodity_type"] = commodity_type
        metrics["n_tickers"] = len(tickers)
        out.append(metrics)
    return out


def run(
    horizons: list[int] | None = None,
    max_tickers: int | None = None,
    eval_days: int = DEFAULT_EVAL_DAYS,
    variants: list[str] | None = None,
) -> dict[str, Any]:
    now = datetime.now(UTC).replace(tzinfo=None)
    ticker_map = {row.ticker: row.commodity_type for row in GLOBAL_COMMODITY_UNIVERSE}
    tickers = global_universe_tickers()
    if max_tickers is not None and max_tickers > 0:
        tickers = tickers[:max_tickers]
        ticker_map = {t: ticker_map[t] for t in tickers if t in ticker_map}
    horizons = horizons or list(DEFAULT_HORIZONS)
    variants_to_run = variants or list(MODEL_VARIANTS)

    scan_service = GlobalOpportunityService()
    prices, _ = scan_service._download_price_volume_frames(tickers + FACTOR_TICKERS, period="6y", batch_size=52)
    prices = prices.loc[:, ~prices.columns.duplicated()].sort_index()
    if prices.empty:
        raise RuntimeError("Failed to fetch price history.")

    spot_proxy = "BOAT" if "BOAT" in prices.columns else ("SEA" if "SEA" in prices.columns else "BDRY")
    fwd_proxy = "BDRY" if "BDRY" in prices.columns else spot_proxy

    provider = RealPredictionProvider()
    events = _event_history(provider, ["hormuz_closure", "red_sea_disruption", "sanctions_escalation", "oil_above_100"])
    if events.empty:
        raise RuntimeError("Event history unavailable.")

    hub = ResearchHubService()
    contracts = _contract_features(hub, lookback_days=780)

    ticker_frames = _build_ticker_frames(
        prices=prices,
        events=events,
        contract_features=contracts,
        spot_proxy=spot_proxy,
        fwd_proxy=fwd_proxy,
        ticker_to_type=ticker_map,
    )
    if not ticker_frames:
        raise RuntimeError("No ticker feature frames built.")

    benchmark_px = prices["ACWI"].dropna()
    if benchmark_px.empty:
        raise RuntimeError("ACWI benchmark history unavailable.")

    grid_rows: list[dict[str, Any]] = []
    sector_rows: list[dict[str, Any]] = []
    for horizon in horizons:
        benchmark_forward = benchmark_px.shift(-horizon) / benchmark_px - 1.0
        for variant in variants_to_run:
            records: list[PredictionRecord] = []
            for ticker, frame in ticker_frames.items():
                cols = _feature_cols_for_variant(frame, variant)
                per_ticker = _walkforward_predictions(
                    frame=frame,
                    feature_cols=cols,
                    horizon_days=horizon,
                    ticker=ticker,
                    commodity_type=ticker_map[ticker],
                    eval_days=eval_days,
                )
                for row in per_ticker:
                    row.variant = variant
                records.extend(per_ticker)

            metrics, _ = _strategy_from_predictions(
                rows=records,
                horizon_days=horizon,
                benchmark_forward=benchmark_forward,
                variant=variant,
            )
            if metrics:
                metrics["modeled_tickers"] = len({x.ticker for x in records})
                metrics["prediction_count"] = len(records)
                grid_rows.append(metrics)
            sector_rows.extend(
                _sector_metrics(
                    rows=records,
                    horizon_days=horizon,
                    benchmark_forward=benchmark_forward,
                    variant=variant,
                )
            )

    if not grid_rows:
        raise RuntimeError("No backtest combinations produced usable metrics.")

    grid_df = pd.DataFrame(grid_rows).sort_values(
        ["annualized_alpha", "information_ratio", "annualized_return"],
        ascending=False,
    )
    sector_df = pd.DataFrame(sector_rows).sort_values(
        ["annualized_alpha", "information_ratio", "annualized_return"],
        ascending=False,
    )
    best = grid_df.iloc[0].to_dict()

    out_dir = Path("analysis_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_df.to_csv(out_dir / "divergent_backtest_grid.csv", index=False)
    sector_df.to_csv(out_dir / "divergent_backtest_sector.csv", index=False)

    summary = {
        "generated_at": now.isoformat(),
        "modeled_universe_size": len(ticker_frames),
        "horizons_tested": horizons,
        "variants_tested": variants_to_run,
        "best_overall": best,
        "top_10": grid_df.head(10).to_dict(orient="records"),
        "top_sector_setups": sector_df.head(30).to_dict(orient="records"),
    }
    (out_dir / "divergent_backtest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Divergent horizon cross-market backtests.")
    parser.add_argument(
        "--horizons",
        default="1,5,10,21,42,63,126,252",
        help="Comma-separated horizon days, e.g. 5,21,63",
    )
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=0,
        help="Optional cap on modeled tickers for faster runs (0 = all).",
    )
    parser.add_argument(
        "--eval-days",
        type=int,
        default=DEFAULT_EVAL_DAYS,
        help="Walk-forward evaluation window length in days.",
    )
    parser.add_argument(
        "--variants",
        default="all",
        help="Comma-separated variant names or 'all'.",
    )
    args = parser.parse_args()

    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    if not horizons:
        horizons = list(DEFAULT_HORIZONS)

    if args.variants.strip().lower() == "all":
        variants = list(MODEL_VARIANTS)
    else:
        variants = [x.strip() for x in args.variants.split(",") if x.strip()]
        invalid = [v for v in variants if v not in MODEL_VARIANTS]
        if invalid:
            raise SystemExit(f"Unknown variants: {', '.join(invalid)}")

    result = run(
        horizons=horizons,
        max_tickers=args.max_tickers if args.max_tickers > 0 else None,
        eval_days=args.eval_days,
        variants=variants,
    )
    print(json.dumps(result["best_overall"], indent=2))
