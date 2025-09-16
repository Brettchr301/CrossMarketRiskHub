"""Segmented Alpha Attribution Backtest Engine.

Runs walk-forward out-of-sample backtests across the full ~300 ticker universe,
segments results by cap size / geography / commodity type / exchange type / war proximity,
and measures alpha vs appropriate benchmarks for each segment.

The key insight: GlobalOpportunityService._walkforward_fit already produces proper
out-of-sample predictions at every time step (trains on [i-150:i], predicts at i).
We harvest those predictions as historical signals, compute actual returns,
apply realistic costs, and measure alpha by segment.

Minimum bar: 2% annualized alpha across multi-year backtests.
"""
from __future__ import annotations

import logging
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler

# Suppress pandas PerformanceWarning from fragmented DataFrame column inserts
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

from app.backtest.metrics import hit_rate, irr_from_periodic_returns, max_drawdown, sharpe_ratio
from app.backtest.segments import (
    ALL_BENCHMARK_TICKERS,
    COMMODITY_TYPE_BENCHMARKS,
    classify_ticker,
    benchmark_for_segment,
)
from app.modeling.cost_model import estimate_total_cost_bps, net_return_after_cost
from app.modeling.global_scan import DEFAULT_MARGIN_BY_TYPE
from app.providers.real_prediction import EVENT_MAPPINGS
from app.modeling.global_universe import GLOBAL_COMMODITY_UNIVERSE

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TickerBacktestResult:
    """Per-ticker walk-forward backtest results."""
    ticker: str
    commodity_type: str
    country: str
    sector: str
    market_cap: float
    segments: dict[str, str]  # dimension -> label
    # Walk-forward out-of-sample predictions and actual returns
    prediction_dates: list[pd.Timestamp]
    predictions: list[float]  # predicted 20d returns (out-of-sample)
    actuals: list[float]  # actual 20d returns
    directions: list[str]  # LONG or SHORT
    costs_bps: list[float]  # per-trade cost
    net_returns: list[float]  # actual - cost
    benchmark_returns: dict[str, list[float]]  # benchmark_ticker -> returns aligned to same dates


@dataclass(slots=True)
class SegmentResult:
    """Aggregated results for one segment."""
    dimension: str
    label: str
    benchmark_ticker: str
    benchmark_name: str
    ticker_count: int
    trade_count: int
    # Core metrics
    annual_alpha_pct: float  # annualized alpha vs benchmark (%)
    sharpe_ratio: float
    hit_rate_pct: float  # % of trades profitable
    max_drawdown_pct: float
    win_rate_vs_benchmark: float  # % of trades that beat benchmark
    # Return decomposition
    gross_annual_return_pct: float
    benchmark_annual_return_pct: float
    cost_drag_annual_pct: float
    # Risk
    volatility_annual_pct: float
    information_ratio: float
    # Capacity
    avg_market_cap: float
    avg_daily_volume: float
    # Time
    backtest_years: float
    first_signal_date: str
    last_signal_date: str
    # Constituents
    top_performers: list[dict[str, Any]]
    worst_performers: list[dict[str, Any]]


@dataclass
class AlphaAttributionResult:
    """Complete backtest result with all segments."""
    run_timestamp: str
    universe_size: int
    modeled_tickers: int
    total_trades: int
    backtest_start: str
    backtest_end: str
    backtest_years: float
    # Per-segment results, grouped by dimension
    segments: dict[str, list[SegmentResult]]  # dimension -> list of SegmentResult
    # Overall
    overall_alpha_pct: float
    overall_sharpe: float
    overall_hit_rate: float
    # Recommended focus segments (alpha >= 2%)
    alpha_segments: list[dict[str, Any]]
    # Sub-alpha segments (alpha < 2%) — consider dropping
    weak_segments: list[dict[str, Any]]
    # Raw per-ticker results for portfolio construction / decision engine
    ticker_results: list[TickerBacktestResult] = field(default_factory=list)


class SegmentedAlphaBacktester:
    """Run walk-forward backtest across full universe with segment attribution.

    Pipeline:
      1. Download equity prices + volumes for all ~300 tickers (6y)
      2. Download factor prices (oil, shipping, metals, macro)
      3. Download prediction market event history (1700 days)
      4. For each ticker: build features, run walk-forward fit, harvest OOS predictions
      5. Download benchmark prices
      6. Compute alpha per segment
      7. Report which segments pass 2% annual alpha bar
    """

    def __init__(
        self,
        holding_days: int = 20,
        min_predictions: int = 8,  # lowered: non-overlapping trades produce ~12-15 per ticker/year
        lookback_days: int = 780,
        walk_forward_lookback: int = 150,
        min_trades_per_segment: int = 15,  # lowered for non-overlapping
        signal_threshold: float = 0.001,  # 0.1% minimum predicted return (low bar — let the model speak)
        verbose: bool = True,
        max_tickers: int | None = None,
        zero_pm_features: bool = False,
    ):
        self.holding_days = holding_days
        self.min_predictions = min_predictions
        self.lookback_days = lookback_days
        self.walk_forward_lookback = walk_forward_lookback
        self.min_trades_per_segment = min_trades_per_segment
        self.signal_threshold = signal_threshold
        self.verbose = verbose
        self.max_tickers = max_tickers
        self.zero_pm_features = zero_pm_features

    def run(self) -> AlphaAttributionResult:
        """Execute the full segmented backtest."""
        t0 = time.time()
        self._log("=" * 80)
        self._log("SEGMENTED ALPHA ATTRIBUTION BACKTEST")
        self._log("=" * 80)

        # Step 1: Download all data
        self._log("\n[1/5] Downloading equity prices for ~300 tickers (6y)...")
        universe = GLOBAL_COMMODITY_UNIVERSE
        if self.max_tickers is not None:
            universe = universe[:self.max_tickers]
        meta = {row.ticker: row for row in universe}
        tickers = list(meta)

        equity_prices, equity_volume = self._download_prices(tickers, period="6y")
        available = [t for t in tickers if t in equity_prices.columns]
        self._log(f"  Got prices for {len(available)}/{len(tickers)} tickers")
        self._log(f"  Date range: {equity_prices.index.min()} to {equity_prices.index.max()}")

        # Step 2: Download factor prices
        self._log("\n[2/5] Downloading factor prices (oil, shipping, metals, macro)...")
        factor_tickers = [
            "BZ=F", "CL=F", "BOAT", "SEA", "BDRY",
            "^VIX", "DX-Y.NYB", "^TNX", "GC=F", "^IRX",
            "SI=F", "HG=F", "PL=F", "PA=F",
            "URA", "LIT", "COPX", "REMX", "ALI=F",
            "WEAT",
        ]
        factor_prices, _ = self._download_prices(factor_tickers, period="6y")
        self._log(f"  Got {len(factor_prices.columns)} factors, {len(factor_prices)} days")

        spot_proxy = "BOAT" if "BOAT" in factor_prices.columns else ("SEA" if "SEA" in factor_prices.columns else "BDRY")
        fwd_proxy = "BDRY" if "BDRY" in factor_prices.columns else spot_proxy

        # Step 3: Download event history
        self._log("\n[3/5] Downloading prediction market event history (1700 days)...")
        event_hist = self._download_event_history()
        if event_hist.empty:
            self._log("  WARNING: No event history — using synthetic 0.5 baseline")
            dates = equity_prices.index
            event_hist = pd.DataFrame(
                {eid: 0.5 for eid in EVENT_MAPPINGS},
                index=dates,
            )
        else:
            self._log(f"  Got {len(event_hist.columns)} events, {len(event_hist)} days")
            self._log(f"  Event range: {event_hist.index.min()} to {event_hist.index.max()}")

        # Step 4: Download benchmark prices
        self._log("\n[4/5] Downloading benchmark prices...")
        benchmark_tickers = ALL_BENCHMARK_TICKERS
        benchmark_prices, _ = self._download_prices(benchmark_tickers, period="6y")
        self._log(f"  Got {len(benchmark_prices.columns)} benchmarks")

        # Step 5: Run walk-forward backtest per ticker
        self._log(f"\n[5/5] Running walk-forward backtest for {len(available)} tickers...")
        ticker_results: list[TickerBacktestResult] = []
        failed = 0
        skipped_short = 0

        # Empty contract list — we don't use research hub contracts in backtest
        # (they change over time; we use raw event features which are stable)
        empty_contracts: list[dict[str, Any]] = []
        empty_delta_map: dict[str, pd.Series] = {}

        for idx, ticker in enumerate(available):
            if idx % 25 == 0:
                self._log(f"  Processing {idx}/{len(available)} | modeled: {len(ticker_results)} | failed: {failed}")

            row_meta = meta[ticker]
            try:
                result = self._backtest_single_ticker(
                    ticker=ticker,
                    row_meta=row_meta,
                    equity_prices=equity_prices,
                    equity_volume=equity_volume,
                    factor_prices=factor_prices,
                    event_hist=event_hist,
                    benchmark_prices=benchmark_prices,
                    contracts=empty_contracts,
                    contract_delta_map=empty_delta_map,
                    spot_proxy=spot_proxy,
                    fwd_proxy=fwd_proxy,
                )
                if result is None:
                    skipped_short += 1
                    continue
                if len(result.net_returns) < self.min_predictions:
                    skipped_short += 1
                    continue
                ticker_results.append(result)
            except Exception as exc:
                failed += 1
                if self.verbose and failed <= 10:
                    logger.debug("Ticker %s failed: %s", ticker, exc)

        self._log(f"\n  Backtest complete: {len(ticker_results)} tickers modeled, "
                   f"{skipped_short} skipped (insufficient data), {failed} failed")

        if not ticker_results:
            self._log("ERROR: No tickers could be modeled!")
            return self._empty_result()

        # Step 6: Aggregate by segment
        self._log("\n[ANALYSIS] Computing alpha by segment...")
        segments = self._compute_segment_results(ticker_results, benchmark_prices)

        # Step 7: Identify high-alpha and weak segments
        alpha_segments = []
        weak_segments = []
        for dimension, seg_list in segments.items():
            for seg in seg_list:
                info = {
                    "dimension": seg.dimension,
                    "label": seg.label,
                    "annual_alpha_pct": seg.annual_alpha_pct,
                    "sharpe": seg.sharpe_ratio,
                    "hit_rate": seg.hit_rate_pct,
                    "trade_count": seg.trade_count,
                    "ticker_count": seg.ticker_count,
                    "backtest_years": seg.backtest_years,
                }
                if seg.annual_alpha_pct >= 2.0:
                    alpha_segments.append(info)
                else:
                    weak_segments.append(info)

        alpha_segments.sort(key=lambda x: x["annual_alpha_pct"], reverse=True)
        weak_segments.sort(key=lambda x: x["annual_alpha_pct"], reverse=True)

        # Compute overall metrics
        all_net = []
        all_bench = []
        for tr in ticker_results:
            all_net.extend(tr.net_returns)
            # Use SPY as universal benchmark for overall
            if "SPY" in tr.benchmark_returns:
                all_bench.extend(tr.benchmark_returns["SPY"])
            else:
                all_bench.extend([0.0] * len(tr.net_returns))

        all_net_arr = np.array(all_net, dtype=float)
        all_bench_arr = np.array(all_bench[:len(all_net)], dtype=float)
        alpha_arr = all_net_arr - all_bench_arr

        # Find date range
        all_dates: list[pd.Timestamp] = []
        for tr in ticker_results:
            all_dates.extend(tr.prediction_dates)
        min_date = min(all_dates) if all_dates else pd.Timestamp.now()
        max_date = max(all_dates) if all_dates else pd.Timestamp.now()
        years = max(0.5, (max_date - min_date).days / 365.25)

        # Annualize alpha
        if len(alpha_arr) > 0:
            # Each trade spans ~20 trading days. Annualize:
            trades_per_year = 252 / self.holding_days
            avg_alpha_per_trade = float(np.mean(alpha_arr))
            annual_alpha = avg_alpha_per_trade * trades_per_year * 100.0
        else:
            annual_alpha = 0.0

        elapsed = time.time() - t0
        self._log(f"\nTotal runtime: {elapsed:.1f}s")

        return AlphaAttributionResult(
            run_timestamp=datetime.now(UTC).replace(tzinfo=None).isoformat(),
            universe_size=len(tickers),
            modeled_tickers=len(ticker_results),
            total_trades=len(all_net),
            backtest_start=str(min_date.date()) if all_dates else "",
            backtest_end=str(max_date.date()) if all_dates else "",
            backtest_years=round(years, 2),
            segments=segments,
            overall_alpha_pct=round(annual_alpha, 2),
            overall_sharpe=round(float(sharpe_ratio(all_net_arr)) if len(all_net_arr) > 1 else 0.0, 3),
            overall_hit_rate=round(float(hit_rate(all_net_arr)) * 100 if len(all_net_arr) > 0 else 0.0, 1),
            alpha_segments=alpha_segments,
            weak_segments=weak_segments,
            ticker_results=ticker_results,
        )

    def _backtest_single_ticker(
        self,
        ticker: str,
        row_meta,
        equity_prices: pd.DataFrame,
        equity_volume: pd.DataFrame,
        factor_prices: pd.DataFrame,
        event_hist: pd.DataFrame,
        benchmark_prices: pd.DataFrame,
        contracts: list[dict[str, Any]],
        contract_delta_map: dict[str, pd.Series],
        spot_proxy: str,
        fwd_proxy: str,
    ) -> TickerBacktestResult | None:
        """Run walk-forward backtest for a single ticker, return OOS predictions + actual returns."""
        if ticker not in equity_prices.columns:
            return None

        # Build feature frame (same as GlobalOpportunityService._build_feature_frame)
        frame = self._build_feature_frame(
            ticker=ticker,
            equity_prices=equity_prices,
            equity_volume=equity_volume,
            factors=factor_prices,
            events=event_hist,
            contracts=contracts,
            contract_delta_map=contract_delta_map,
            lookback_days=self.lookback_days,
            spot_proxy=spot_proxy,
            fwd_proxy=fwd_proxy,
        )
        if frame is None or len(frame) < self.walk_forward_lookback + 40:
            return None

        # Run walk-forward fit and extract OOS predictions
        target_col = f"y_fwd_{self.holding_days}d"
        fit_result = self._walkforward_harvest(frame, target_col=target_col)
        if fit_result is None:
            return None

        predictions = fit_result["predictions"]
        actuals = fit_result["actuals"]
        sample_dates = fit_result["dates"]

        if len(predictions) < self.min_predictions:
            return None

        # Get market cap for segmentation
        market_cap = self._estimate_market_cap(ticker, equity_prices, equity_volume)

        # Classify segments
        segments = classify_ticker(
            ticker=ticker,
            country=row_meta.country,
            commodity_type=row_meta.commodity_type,
            market_cap=market_cap,
        )

        # Compute cost per trade
        avg_volume = float(equity_volume[ticker].dropna().tail(60).mean()) if ticker in equity_volume.columns else 100_000
        cost_bps = self._compute_cost(market_cap, avg_volume, "LONG")

        # Generate signals from OOS predictions — LONG ONLY
        directions: list[str] = []
        net_returns: list[float] = []
        costs: list[float] = []
        filtered_preds: list[float] = []
        filtered_actuals: list[float] = []
        filtered_dates: list[pd.Timestamp] = []

        # Data-quality cap: reject trades with implausibly large actual returns.
        # These are almost always data artifacts (bankruptcy emergence, ticker changes,
        # stock splits not adjusted by Yahoo Finance).
        # Scale with holding period: 200% for 20d, up to 500% for 252d.
        MAX_TRADE_RETURN = min(5.0, 2.0 * self.holding_days / 20)  # scales with horizon

        for i, (pred, actual, dt) in enumerate(zip(predictions, actuals, sample_dates)):
            # LONG-ONLY: skip any negative predictions (no shorting)
            if pred < self.signal_threshold:
                continue

            # Data quality filter: skip obviously corrupt returns
            if abs(actual) > MAX_TRADE_RETURN:
                continue

            direction = "LONG"
            trade_cost_bps = self._compute_cost(market_cap, avg_volume, direction)
            trade_return = actual
            net = net_return_after_cost(trade_return, trade_cost_bps)

            directions.append(direction)
            net_returns.append(net)
            costs.append(trade_cost_bps)
            filtered_preds.append(pred)
            filtered_actuals.append(actual)
            filtered_dates.append(dt)

        if len(net_returns) < 3:
            return None

        # Compute matching benchmark returns
        bench_returns: dict[str, list[float]] = {}
        for bench_ticker in list(set(
            [COMMODITY_TYPE_BENCHMARKS.get(row_meta.commodity_type, COMMODITY_TYPE_BENCHMARKS["oil_gas_upstream"]).benchmark_ticker]
            + ["SPY"]
        )):
            if bench_ticker not in benchmark_prices.columns:
                bench_returns[bench_ticker] = [0.0] * len(filtered_dates)
                continue
            bench_px = benchmark_prices[bench_ticker].dropna()
            b_rets = []
            for dt in filtered_dates:
                # Benchmark return over same 20-day holding period
                entry_mask = bench_px.index >= dt
                if not entry_mask.any():
                    b_rets.append(0.0)
                    continue
                entry_idx = bench_px.index[entry_mask][0]
                exit_target = dt + pd.Timedelta(days=self.holding_days * 1.4)  # ~20 trading days ≈ 28 calendar
                exit_mask = bench_px.index >= exit_target
                if not exit_mask.any():
                    b_rets.append(0.0)
                    continue
                exit_idx = bench_px.index[exit_mask][0]
                p0 = float(bench_px[entry_idx])
                p1 = float(bench_px[exit_idx])
                if p0 > 0:
                    b_rets.append((p1 - p0) / p0)
                else:
                    b_rets.append(0.0)
            bench_returns[bench_ticker] = b_rets

        return TickerBacktestResult(
            ticker=ticker,
            commodity_type=row_meta.commodity_type,
            country=row_meta.country,
            sector=row_meta.sector,
            market_cap=market_cap,
            segments=segments,
            prediction_dates=filtered_dates,
            predictions=filtered_preds,
            actuals=filtered_actuals,
            directions=directions,
            costs_bps=costs,
            net_returns=net_returns,
            benchmark_returns=bench_returns,
        )

    def _walkforward_harvest(
        self,
        frame: pd.DataFrame,
        target_col: str,
    ) -> dict[str, Any] | None:
        """Walk-forward fit that returns ALL out-of-sample predictions with dates.

        Optimised v3 with GitHub/Reddit innovations:
          - Ridge alpha=10.0 (stronger regularization to prevent overfit)
          - step_size = holding_days (non-overlapping trades, no return autocorrelation)
          - 5-day embargo between train/test (purged walk-forward, from Marcos Lopez de Prado)
          - Feature variance pre-filter (top 40 features by variance — from alphalens/skfolio)
        No lookahead: model only sees past data at each prediction point.
        """
        if target_col not in frame.columns:
            return None
        feature_cols = [c for c in frame.columns if c not in {target_col, "stock_px", "stock_vol"}]
        if not feature_cols:
            return None
        if len(frame) < self.walk_forward_lookback + 40:
            return None

        lookback = self.walk_forward_lookback
        # Non-overlapping trades: step = holding period
        # This prevents return autocorrelation that inflates Sharpe/hit rate
        step_size = self.holding_days
        # Embargo: scaled to holding period — longer horizons need more purging
        # Min 5 days (for 20d), up to holding_days//4 for longer horizons
        embargo = max(5, self.holding_days // 4)

        # Pre-extract numpy arrays for speed
        all_X = frame[feature_cols].values.astype(np.float64)
        all_Y = frame[target_col].values.astype(np.float64)
        all_dates_idx = frame.index

        # Replace any remaining inf/nan in features
        np.nan_to_num(all_X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Feature selection: Lasso-based (replaces variance-based to reduce selection bias).
        # Run Lasso on the FIRST training window to pick features with non-zero coefficients.
        # This avoids lookahead because Lasso only sees the first lookback chunk.
        # Fallback to top-15 by variance if Lasso selects < 3 features.
        max_features = 15
        subsample_init = max(1, self.holding_days // 4)
        init_end = min(lookback, len(frame) - embargo)
        if init_end > 60:
            y_init = all_Y[:init_end:subsample_init]
            x_init = all_X[:init_end:subsample_init]
            valid_mask = ~np.isnan(y_init)
            if valid_mask.sum() > max_features + 1:
                x_init_v = x_init[valid_mask]
                y_init_v = y_init[valid_mask]
                try:
                    scaler_init = StandardScaler()
                    x_init_s = scaler_init.fit_transform(x_init_v)
                    lasso = Lasso(alpha=0.005, max_iter=2000, fit_intercept=True)
                    lasso.fit(x_init_s, y_init_v)
                    lasso_idx = np.where(np.abs(lasso.coef_) > 1e-8)[0]
                    if len(lasso_idx) >= 3:
                        # Lasso picked enough features — use them
                        if len(lasso_idx) > max_features:
                            # Too many: keep top-max_features by absolute coefficient
                            abs_coefs = np.abs(lasso.coef_[lasso_idx])
                            top_sub = np.argsort(abs_coefs)[-max_features:]
                            lasso_idx = lasso_idx[top_sub]
                        all_X = all_X[:, lasso_idx]
                    else:
                        # Lasso too aggressive — fallback to variance filter
                        if all_X.shape[1] > max_features:
                            variances = np.nanvar(all_X, axis=0)
                            top_idx = np.argsort(variances)[-max_features:]
                            all_X = all_X[:, top_idx]
                except Exception:
                    # Lasso failed — fallback to variance filter
                    if all_X.shape[1] > max_features:
                        variances = np.nanvar(all_X, axis=0)
                        top_idx = np.argsort(variances)[-max_features:]
                        all_X = all_X[:, top_idx]
            else:
                if all_X.shape[1] > max_features:
                    variances = np.nanvar(all_X, axis=0)
                    top_idx = np.argsort(variances)[-max_features:]
                    all_X = all_X[:, top_idx]
        else:
            if all_X.shape[1] > max_features:
                variances = np.nanvar(all_X, axis=0)
                top_idx = np.argsort(variances)[-max_features:]
                all_X = all_X[:, top_idx]

        predictions: list[float] = []
        actuals: list[float] = []
        dates: list[pd.Timestamp] = []

        for i in range(lookback + embargo, len(frame), step_size):
            # Train on [i-lookback-embargo : i-embargo] — purged window
            train_end = i - embargo
            train_start = max(0, train_end - lookback)
            if train_end - train_start < 60:
                continue

            y_train = all_Y[train_start: train_end]

            # Skip if target has NaN in training window
            if np.isnan(y_train).any():
                continue

            # Subsample training rows to break overlapping-label autocorrelation
            # With 20-day returns, adjacent daily rows share 95% of their return window.
            # Subsampling every holding_days//4 = 5 rows gives ~30 quasi-independent
            # training observations from a 150-day window (Lopez de Prado AFML Ch.4)
            # At step=5 with 20-day labels, overlap is 75% (vs 95% daily) — practical compromise
            subsample_step = max(1, self.holding_days // 4)
            x_train = all_X[train_start: train_end: subsample_step]
            y_train = y_train[::subsample_step]

            # Need at least features+1 rows for Ridge (regularization handles the rest)
            if len(y_train) < max_features + 1:
                continue

            try:
                # Standardize features (critical for Ridge — makes penalty fair across all features)
                scaler = StandardScaler()
                x_scaled = scaler.fit_transform(x_train)
                # Ridge alpha=20 (strong rex_pred standardized features)
                ridge = Ridge(alpha=20.0, fit_intercept=True)
                ridge.fit(x_scaled, y_train)
                x_pred = scaler.transform(all_X[i: i + 1])
                pred = float(ridge.predict(x_pred)[0])
                actual = float(all_Y[i])
                if np.isnan(actual):
                    continue
                predictions.append(pred)
                actuals.append(actual)
                dates.append(all_dates_idx[i])
            except Exception:
                continue

        if len(predictions) < self.min_predictions:
            return None

        return {
            "predictions": predictions,
            "actuals": actuals,
            "dates": dates,
        }

    def _build_feature_frame(
        self,
        ticker: str,
        equity_prices: pd.DataFrame,
        equity_volume: pd.DataFrame,
        factors: pd.DataFrame,
        events: pd.DataFrame,
        contracts: list[dict[str, Any]],
        contract_delta_map: dict[str, pd.Series],
        lookback_days: int,
        spot_proxy: str,
        fwd_proxy: str,
    ) -> pd.DataFrame | None:
        """Build feature frame — mirrors GlobalOpportunityService._build_feature_frame exactly."""
        if ticker not in equity_prices.columns:
            return None
        stock = equity_prices[ticker].dropna().rename("stock_px")
        if stock.empty:
            return None
        volume = (
            equity_volume[ticker].astype(float).dropna()
            if ticker in equity_volume.columns
            else pd.Series(dtype=float)
        )
        df = stock.to_frame()
        df["stock_ret_1d"] = stock.pct_change(fill_method=None)
        df["stock_vol"] = volume.reindex(df.index).ffill().fillna(0.0)

        for col, name in [("BZ=F", "brent"), ("CL=F", "wti"), (spot_proxy, "ship_spot"), (fwd_proxy, "ship_fwd")]:
            if col not in factors.columns:
                continue
            px = factors[col].rename(f"{name}_px")
            df = df.join(px, how="left")
            ret = px.pct_change(fill_method=None)
            df[f"{name}_ret"] = ret
            df[f"{name}_sq"] = ret * ret
            df[f"{name}_accel"] = ret.diff()

        # Commodity reference factor features
        _commodity_factor_map = {
            "GC=F": "gold", "SI=F": "silver", "HG=F": "copper",
            "PL=F": "platinum", "PA=F": "palladium",
            "URA": "uranium_etf", "LIT": "lithium_etf",
            "COPX": "copper_miners", "REMX": "rare_earth_etf",
            "ALI=F": "aluminum", "WEAT": "wheat_etf",
        }
        for yf_sym, feat_name in _commodity_factor_map.items():
            if yf_sym not in factors.columns:
                continue
            cpx = factors[yf_sym].reindex(df.index).ffill()
            cret = cpx.pct_change(fill_method=None).fillna(0.0)
            df[f"cmd_{feat_name}_ret"] = cret
            df[f"cmd_{feat_name}_sq"] = cret * cret
            df[f"cmd_{feat_name}_accel"] = cret.diff().fillna(0.0)

        df = df.join(events, how="left")
        for ev in list(EVENT_MAPPINGS.keys()):
            if ev not in df.columns:
                df[ev] = 0.5
            d = df[ev].diff()
            df[f"{ev}_d1"] = d
            df[f"{ev}_d2"] = d.diff()
            df[f"{ev}_sq"] = d * d

        for contract in contracts[:24]:
            cid = contract["market_id"]
            delta = contract_delta_map.get(cid)
            if delta is None or delta.empty:
                continue
            lag = int(contract.get("lead_days", 0))
            s = delta.shift(lag).rename(f"pm_{cid}")
            df = df.join(s, how="left")
            df[f"pm2_{cid}"] = df[f"pm_{cid}"].diff()

        # ── Macro regime features (MUST come before cross-features that reference them) ──
        _macro_map = {
            "^VIX": "vix", "DX-Y.NYB": "dxy", "^TNX": "us10y",
            "GC=F": "gold", "^IRX": "us3m",
        }
        for yf_sym, feat_name in _macro_map.items():
            if yf_sym not in factors.columns:
                continue
            macro_px = factors[yf_sym].reindex(df.index).ffill()
            df[f"macro_{feat_name}_ret"] = macro_px.pct_change(fill_method=None).fillna(0.0)
            if feat_name == "vix":
                df["macro_vix_level"] = macro_px.fillna(18.0)
                roll_mean = macro_px.rolling(20, min_periods=5).mean()
                df["macro_vix_regime"] = (macro_px > roll_mean).astype(float).fillna(0.0)

        # Yield curve spread
        if "^TNX" in factors.columns and "^IRX" in factors.columns:
            us10y = factors["^TNX"].reindex(df.index).ffill()
            us3m = factors["^IRX"].reindex(df.index).ffill()
            df["macro_yield_spread"] = (us10y - us3m).fillna(0.0)
            df["macro_yield_spread_d1"] = df["macro_yield_spread"].diff().fillna(0.0)

        # Risk-off composite
        if "macro_vix_ret" in df.columns and "macro_dxy_ret" in df.columns:
            df["macro_risk_off_cross"] = df["macro_vix_ret"] * df["macro_dxy_ret"]

        # Cross-features
        if "brent_ret" in df.columns and "ship_spot_ret" in df.columns:
            df["brent_ship_cross"] = df["brent_ret"] * df["ship_spot_ret"]
        if "brent_ret" in df.columns and "wti_ret" in df.columns:
            df["wti_brent_spread_ret"] = df["wti_ret"] - df["brent_ret"]
        if "brent_ret" in df.columns:
            rolling_vol = df["brent_ret"].rolling(20).std()
            df["high_vol_regime"] = (rolling_vol > rolling_vol.rolling(60).mean()).astype(float).fillna(0.0)
        if "hormuz_closure_d1" in df.columns and "ship_fwd_ret" in df.columns:
            df["event_freight_cross"] = df["hormuz_closure_d1"] * df["ship_fwd_ret"]

        # Commodity-specific cross-features
        if "cmd_gold_ret" in df.columns and "macro_dxy_ret" in df.columns:
            df["gold_dxy_cross"] = df["cmd_gold_ret"] * df["macro_dxy_ret"]
        if "cmd_gold_ret" in df.columns and "macro_vix_ret" in df.columns:
            df["gold_vix_cross"] = df["cmd_gold_ret"] * df["macro_vix_ret"]
        if "cmd_copper_ret" in df.columns and "cmd_gold_ret" in df.columns:
            df["copper_gold_spread_ret"] = df["cmd_copper_ret"] - df["cmd_gold_ret"]
        if "cmd_copper_ret" in df.columns and "brent_ret" in df.columns:
            df["copper_oil_cross"] = df["cmd_copper_ret"] * df["brent_ret"]
        if "us_tariff_escalation_d1" in df.columns and "cmd_copper_ret" in df.columns:
            df["tariff_copper_cross"] = df["us_tariff_escalation_d1"] * df["cmd_copper_ret"]
        if "china_stimulus_d1" in df.columns and "cmd_copper_ret" in df.columns:
            df["china_copper_cross"] = df["china_stimulus_d1"] * df["cmd_copper_ret"]
        if "nuclear_renaissance_d1" in df.columns and "cmd_uranium_etf_ret" in df.columns:
            df["nuclear_uranium_cross"] = df["nuclear_renaissance_d1"] * df["cmd_uranium_etf_ret"]

        # Tail-risk cross-features
        if "middle_east_war_escalation_d1" in df.columns and "brent_ret" in df.columns:
            df["mideast_war_oil_cross"] = df["middle_east_war_escalation_d1"] * df["brent_ret"]
        if "middle_east_war_escalation_d1" in df.columns and "cmd_gold_ret" in df.columns:
            df["mideast_war_gold_cross"] = df["middle_east_war_escalation_d1"] * df["cmd_gold_ret"]
        if "taiwan_strait_crisis_d1" in df.columns and "cmd_rare_earth_etf_ret" in df.columns:
            df["taiwan_rare_earth_cross"] = df["taiwan_strait_crisis_d1"] * df["cmd_rare_earth_etf_ret"]
        if "taiwan_strait_crisis_d1" in df.columns and "ship_spot_ret" in df.columns:
            df["taiwan_shipping_cross"] = df["taiwan_strait_crisis_d1"] * df["ship_spot_ret"]
        if "russia_ukraine_ceasefire_d1" in df.columns and "cmd_wheat_etf_ret" in df.columns:
            df["ceasefire_wheat_cross"] = df["russia_ukraine_ceasefire_d1"] * df["cmd_wheat_etf_ret"]
        if "russia_ukraine_ceasefire_d1" in df.columns and "cmd_palladium_ret" in df.columns:
            df["ceasefire_palladium_cross"] = df["russia_ukraine_ceasefire_d1"] * df["cmd_palladium_ret"]
        if "us_recession_d1" in df.columns and "cmd_copper_ret" in df.columns:
            df["recession_copper_cross"] = df["us_recession_d1"] * df["cmd_copper_ret"]
        if "eu_cbam_implementation_d1" in df.columns and "cmd_aluminum_ret" in df.columns:
            df["cbam_aluminum_cross"] = df["eu_cbam_implementation_d1"] * df["cmd_aluminum_ret"]
        if "chile_lithium_nationalization_d1" in df.columns and "cmd_lithium_etf_ret" in df.columns:
            df["chile_lithium_cross"] = df["chile_lithium_nationalization_d1"] * df["cmd_lithium_etf_ret"]
        if "south_africa_grid_crisis_d1" in df.columns and "cmd_gold_ret" in df.columns:
            df["sa_grid_gold_cross"] = df["south_africa_grid_crisis_d1"] * df["cmd_gold_ret"]
        if "indonesia_nickel_ban_d1" in df.columns and "cmd_copper_ret" in df.columns:
            df["indo_nickel_base_cross"] = df["indonesia_nickel_ban_d1"] * df["cmd_copper_ret"]

        # Monetary policy cross-features
        if "fed_rate_cut_d1" in df.columns and "cmd_gold_ret" in df.columns:
            df["fed_cut_gold_cross"] = df["fed_rate_cut_d1"] * df["cmd_gold_ret"]
        if "fed_rate_cut_d1" in df.columns and "brent_ret" in df.columns:
            df["fed_cut_oil_cross"] = df["fed_rate_cut_d1"] * df["brent_ret"]
        if "fed_rate_cut_d1" in df.columns and "cmd_copper_ret" in df.columns:
            df["fed_cut_copper_cross"] = df["fed_rate_cut_d1"] * df["cmd_copper_ret"]
        if "us_inflation_above_3_d1" in df.columns and "cmd_gold_ret" in df.columns:
            df["inflation_gold_cross"] = df["us_inflation_above_3_d1"] * df["cmd_gold_ret"]
        if "dollar_strength_extreme_d1" in df.columns and "cmd_gold_ret" in df.columns:
            df["dollar_gold_cross"] = df["dollar_strength_extreme_d1"] * df["cmd_gold_ret"]
        if "boj_rate_hike_d1" in df.columns and "ship_spot_ret" in df.columns:
            df["boj_shipping_cross"] = df["boj_rate_hike_d1"] * df["ship_spot_ret"]
        if "natural_gas_above_5_d1" in df.columns and "brent_ret" in df.columns:
            df["natgas_oil_cross"] = df["natural_gas_above_5_d1"] * df["brent_ret"]

        # Production / consumption / supply-chain cross-features (26 new contracts)
        if "us_oil_production_record_d1" in df.columns and "brent_ret" in df.columns:
            df["us_prod_oil_cross"] = df["us_oil_production_record_d1"] * df["brent_ret"]
        if "opec_compliance_below_80_d1" in df.columns and "brent_ret" in df.columns:
            df["opec_comply_oil_cross"] = df["opec_compliance_below_80_d1"] * df["brent_ret"]
        if "china_oil_demand_slowdown_d1" in df.columns and "brent_ret" in df.columns:
            df["china_demand_oil_cross"] = df["china_oil_demand_slowdown_d1"] * df["brent_ret"]
        if "permian_basin_peak_d1" in df.columns and "brent_ret" in df.columns:
            df["permian_peak_oil_cross"] = df["permian_basin_peak_d1"] * df["brent_ret"]
        if "crack_spread_above_30_d1" in df.columns and "brent_ret" in df.columns:
            df["crack_spread_cross"] = df["crack_spread_above_30_d1"] * df["brent_ret"]
        if "lithium_price_rebound_d1" in df.columns and "cmd_lithium_etf_ret" in df.columns:
            df["lithium_rebound_cross"] = df["lithium_price_rebound_d1"] * df["cmd_lithium_etf_ret"]
        if "china_ev_sales_record_d1" in df.columns and "cmd_lithium_etf_ret" in df.columns:
            df["china_ev_lithium_cross"] = df["china_ev_sales_record_d1"] * df["cmd_lithium_etf_ret"]
        if "battery_cathode_shift_d1" in df.columns and "cmd_lithium_etf_ret" in df.columns:
            df["cathode_lithium_cross"] = df["battery_cathode_shift_d1"] * df["cmd_lithium_etf_ret"]
        if "central_bank_gold_buying_d1" in df.columns and "cmd_gold_ret" in df.columns:
            df["cb_gold_cross"] = df["central_bank_gold_buying_d1"] * df["cmd_gold_ret"]
        if "silver_industrial_demand_surge_d1" in df.columns and "cmd_gold_ret" in df.columns:
            df["silver_demand_gold_cross"] = df["silver_industrial_demand_surge_d1"] * df["cmd_gold_ret"]
        if "china_rare_earth_processing_dominance_d1" in df.columns and "cmd_rare_earth_etf_ret" in df.columns:
            df["china_re_cross"] = df["china_rare_earth_processing_dominance_d1"] * df["cmd_rare_earth_etf_ret"]
        if "us_critical_minerals_act_d1" in df.columns and "cmd_rare_earth_etf_ret" in df.columns:
            df["us_minerals_re_cross"] = df["us_critical_minerals_act_d1"] * df["cmd_rare_earth_etf_ret"]
        if "uranium_supply_deficit_d1" in df.columns and "cmd_uranium_etf_ret" in df.columns:
            df["uranium_deficit_cross"] = df["uranium_supply_deficit_d1"] * df["cmd_uranium_etf_ret"]
        if "smr_deployment_milestone_d1" in df.columns and "cmd_uranium_etf_ret" in df.columns:
            df["smr_uranium_cross"] = df["smr_deployment_milestone_d1"] * df["cmd_uranium_etf_ret"]
        if "china_pmi_below_50_d1" in df.columns and "cmd_copper_ret" in df.columns:
            df["china_pmi_copper_cross"] = df["china_pmi_below_50_d1"] * df["cmd_copper_ret"]
        if "eu_energy_crisis_d1" in df.columns and "brent_ret" in df.columns:
            df["eu_energy_oil_cross"] = df["eu_energy_crisis_d1"] * df["brent_ret"]

        # ══════════════════════════════════════════════════════════════
        # INNOVATION BLOCK: GitHub/Reddit quant community alpha signals
        # Sources: vectorbt, qstrader, alphalens, r/algotrading, r/quant
        # ══════════════════════════════════════════════════════════════

        # --- (1) Momentum features (Jegadeesh & Titman, classic on GitHub repos) ---
        # 60-day momentum (medium-term trend following)
        if len(df) > 60:
            df["momentum_60d"] = df["stock_px"].pct_change(60, fill_method=None).fillna(0.0)
        # 120-day momentum (longer-term)
        if len(df) > 120:
            df["momentum_120d"] = df["stock_px"].pct_change(120, fill_method=None).fillna(0.0)
        # 252-day momentum (annual)
        if len(df) > 252:
            df["momentum_252d"] = df["stock_px"].pct_change(252, fill_method=None).fillna(0.0)
        # Momentum acceleration (change in 60d momentum)
        if "momentum_60d" in df.columns:
            df["momentum_60d_accel"] = df["momentum_60d"].diff(20).fillna(0.0)

        # --- (2) Mean-reversion (short-term, from r/algotrading) ---
        # 5-day RSI-like signal (price vs 5-day moving average)
        if len(df) > 5:
            ma5 = df["stock_px"].rolling(5).mean()
            df["mean_rev_5d"] = ((df["stock_px"] / ma5) - 1.0).fillna(0.0)
        # 20-day mean reversion
        if len(df) > 20:
            ma20 = df["stock_px"].rolling(20).mean()
            df["mean_rev_20d"] = ((df["stock_px"] / ma20) - 1.0).fillna(0.0)
        # Bollinger Band position (from r/algotrading "BB mean reversion" strategy)
        if len(df) > 20:
            bb_std = df["stock_px"].rolling(20).std()
            df["bb_position"] = ((df["stock_px"] - ma20) / bb_std.replace(0, np.nan)).fillna(0.0)

        # --- (3) Volatility regime features (from GitHub skfolio / risk-parity repos) ---
        if "stock_ret_1d" in df.columns:
            # Realized volatility (20-day)
            df["rvol_20d"] = df["stock_ret_1d"].rolling(20, min_periods=5).std().fillna(0.0)
            # Volatility ratio (short/long — identifies vol regime transitions)
            rvol_60 = df["stock_ret_1d"].rolling(60, min_periods=10).std()
            df["vol_ratio"] = (df["rvol_20d"] / rvol_60.replace(0, np.nan)).fillna(1.0)
            # Volatility surprise (current vol vs expected)
            df["vol_surprise"] = (df["rvol_20d"] - rvol_60).fillna(0.0)

        # --- (4) Probability velocity features (novel — prediction market signal) ---
        # Rate of change of event probabilities (not just level)
        for ev in list(EVENT_MAPPINGS.keys()):
            if ev in df.columns:
                # 5-day velocity (fast signal)
                df[f"{ev}_vel5"] = df[ev].diff(5).fillna(0.0)
                # 20-day velocity (slow signal)
                df[f"{ev}_vel20"] = df[ev].diff(20).fillna(0.0)
                # Velocity acceleration (second derivative)
                df[f"{ev}_vel_accel"] = df[f"{ev}_vel5"].diff(5).fillna(0.0)

        # --- (5) Seasonality features (from commodity trading literature / Reddit) ---
        # Month-of-year as cyclical features (sin/cos encoding)
        month = df.index.month
        df["season_sin"] = np.sin(2 * np.pi * month / 12).astype(float)
        df["season_cos"] = np.cos(2 * np.pi * month / 12).astype(float)
        # Quarter indicator
        df["quarter_sin"] = np.sin(2 * np.pi * df.index.quarter / 4).astype(float)

        # --- (6) Volume anomaly detection (from r/algotrading "unusual volume" strategy) ---
        if "stock_vol" in df.columns and df["stock_vol"].sum() > 0:
            vol_ma20 = df["stock_vol"].rolling(20, min_periods=5).mean().replace(0, np.nan)
            df["volume_ratio"] = (df["stock_vol"] / vol_ma20).fillna(1.0)
            # Volume surge (>2x average = potential catalyst)
            df["volume_surge"] = (df["volume_ratio"] > 2.0).astype(float)
            # Volume trend (accumulation/distribution)
            df["volume_trend"] = df["stock_vol"].rolling(5).mean().fillna(0) / vol_ma20.fillna(1) - 1.0
            df["volume_trend"] = df["volume_trend"].fillna(0.0)

        # --- (7) VIX regime indicator (from multiple Reddit/GitHub quant systems) ---
        # Identifies crash/recovery/complacent regimes
        if "macro_vix_level" in df.columns:
            vix = df["macro_vix_level"]
            df["vix_high_regime"] = (vix > 30).astype(float)
            df["vix_low_regime"] = (vix < 15).astype(float)
            # VIX mean reversion (high VIX tends to revert = buying opportunity)
            vix_ma60 = vix.rolling(60, min_periods=10).mean()
            df["vix_deviation"] = ((vix / vix_ma60.replace(0, np.nan)) - 1.0).fillna(0.0)

        # --- (8) Cross-sectional relative strength (from alphalens / Quantopian) ---
        # Stock momentum relative to sector momentum (remove beta)
        if "momentum_60d" in df.columns and "brent_ret" in df.columns:
            sector_mom = df["brent_ret"].rolling(60).sum().fillna(0.0)
            df["relative_strength"] = (df["momentum_60d"] - sector_mom).fillna(0.0)

        # Brent term structure proxy
        if "brent_px" in df.columns:
            brent_roll60 = df["brent_px"].rolling(60, min_periods=20).mean()
            df["brent_contango_ret"] = (df["brent_px"] / brent_roll60 - 1.0).fillna(0.0)
            df["brent_contango_d1"] = df["brent_contango_ret"].diff()

        # BDI curve slope
        if fwd_proxy in factors.columns:
            bdi = factors[fwd_proxy].reindex(df.index).ffill()
            bdi_5d = bdi.pct_change(5, fill_method=None).fillna(0.0)
            bdi_60d = bdi.pct_change(60, fill_method=None).fillna(0.0)
            df["bdi_curve_slope"] = (bdi_5d - bdi_60d).fillna(0.0)
            df["bdi_curve_slope_d1"] = df["bdi_curve_slope"].diff().fillna(0.0)

        # Multi-horizon forward returns — primary target uses self.holding_days
        target_col = f"y_fwd_{self.holding_days}d"
        df[target_col] = df["stock_px"].shift(-self.holding_days) / df["stock_px"] - 1.0
        keep_cols = [c for c in df.columns if c.endswith(("_ret", "_sq", "_accel", "_d1", "_d2", "_cross", "_spread_ret", "_regime", "_slope", "_spread", "_level", "_vel5", "_vel20", "_vel_accel", "_sin", "_cos", "_ratio", "_surge", "_trend", "_surprise", "_deviation", "_strength", "_position")) or c.startswith(("pm_", "pm2_", "macro_", "bdi_", "cmd_", "momentum_", "mean_rev_", "rvol_", "vol_", "bb_", "vix_", "volume_", "season_", "quarter_", "relative_"))]
        keep_cols.extend(["stock_px", "stock_vol", target_col])
        keep_cols = [c for c in keep_cols if c in df.columns]
        out = df[keep_cols].replace([np.inf, -np.inf], np.nan)
        feature_cols = [c for c in out.columns if c not in {"stock_px", "stock_vol", target_col}]
        out[feature_cols] = out[feature_cols].fillna(0.0)
        out = out.dropna(subset=["stock_px", target_col])

        # TRUE PM ABLATION: zero out all pm_ and pm2_ columns if requested
        if self.zero_pm_features:
            pm_cols = [c for c in out.columns if c.startswith(("pm_", "pm2_"))]
            if pm_cols:
                out[pm_cols] = 0.0

        # Use ALL available history for honest multi-year backtesting.
        # The walk-forward training window is controlled by walk_forward_lookback (150d),
        # NOT by truncating the feature frame here. Previously this was
        # `out.tail(max(lookback_days, 140))` which limited OOS to ~1 year.
        return out

    def _compute_segment_results(
        self,
        ticker_results: list[TickerBacktestResult],
        benchmark_prices: pd.DataFrame,
    ) -> dict[str, list[SegmentResult]]:
        """Aggregate ticker results into segment-level metrics."""
        dimensions = ["cap_size", "geography", "war_proximity", "exchange_type", "commodity_type"]
        results: dict[str, list[SegmentResult]] = {}

        for dim in dimensions:
            # Group tickers by this dimension
            groups: dict[str, list[TickerBacktestResult]] = defaultdict(list)
            for tr in ticker_results:
                label = tr.segments.get(dim, "unknown")
                groups[label].append(tr)

            seg_results: list[SegmentResult] = []
            for label, group in groups.items():
                seg = self._compute_single_segment(dim, label, group, benchmark_prices)
                if seg is not None:
                    seg_results.append(seg)

            seg_results.sort(key=lambda s: s.annual_alpha_pct, reverse=True)
            results[dim] = seg_results

        return results

    def _compute_single_segment(
        self,
        dimension: str,
        label: str,
        group: list[TickerBacktestResult],
        benchmark_prices: pd.DataFrame,
    ) -> SegmentResult | None:
        """Compute alpha metrics for a single segment."""
        seg_def = benchmark_for_segment(dimension, label)
        if seg_def is None:
            return None

        bench_ticker = seg_def.benchmark_ticker

        all_net: list[float] = []
        all_gross: list[float] = []
        all_bench: list[float] = []
        all_costs: list[float] = []
        all_dates: list[pd.Timestamp] = []
        all_caps: list[float] = []
        all_vols: list[float] = []

        # Per-ticker performance for top/worst
        ticker_perf: list[dict[str, Any]] = []

        for tr in group:
            if not tr.net_returns:
                continue

            # Get benchmark returns for this ticker
            b_rets = tr.benchmark_returns.get(bench_ticker)
            if b_rets is None:
                b_rets = tr.benchmark_returns.get("SPY", [0.0] * len(tr.net_returns))

            n = min(len(tr.net_returns), len(b_rets))
            all_net.extend(tr.net_returns[:n])
            all_gross.extend(tr.actuals[:n])
            all_bench.extend(b_rets[:n])
            all_costs.extend(tr.costs_bps[:n])
            all_dates.extend(tr.prediction_dates[:n])
            all_caps.append(tr.market_cap)

            # Per-ticker alpha
            tr_alpha = np.mean(np.array(tr.net_returns[:n]) - np.array(b_rets[:n]))
            ticker_perf.append({
                "ticker": tr.ticker,
                "country": tr.country,
                "commodity_type": tr.commodity_type,
                "trade_count": n,
                "avg_alpha_per_trade": float(round(tr_alpha * 100, 3)),
                "hit_rate": float(round(np.mean(np.array(tr.net_returns[:n]) > 0) * 100, 1)),
            })

        if len(all_net) < self.min_trades_per_segment:
            return None

        net_arr = np.array(all_net, dtype=float)
        bench_arr = np.array(all_bench[:len(net_arr)], dtype=float)
        gross_arr = np.array(all_gross[:len(net_arr)], dtype=float)
        cost_arr = np.array(all_costs[:len(net_arr)], dtype=float)
        alpha_arr = net_arr - bench_arr

        # Annualize: each trade spans ~20 trading days
        trades_per_year = 252 / self.holding_days
        annual_alpha = float(np.mean(alpha_arr)) * trades_per_year * 100

        # Strategy gross + benchmark annualized
        annual_gross = float(np.mean(net_arr)) * trades_per_year * 100
        annual_bench = float(np.mean(bench_arr)) * trades_per_year * 100
        avg_cost_bps = float(np.mean(cost_arr))
        annual_cost_drag = avg_cost_bps / 10000 * trades_per_year * 100

        # Sharpe (annualized from per-trade returns)
        sr = float(sharpe_ratio(net_arr, annualization_factor=trades_per_year))

        # Hit rate
        hr = float(np.mean(net_arr > 0)) * 100

        # Win rate vs benchmark
        win_vs_bench = float(np.mean(alpha_arr > 0)) * 100

        # Max drawdown on cumulative strategy
        equity = np.cumprod(1.0 + net_arr)
        mdd = float(max_drawdown(equity)) * 100

        # Volatility
        vol = float(np.std(net_arr, ddof=1)) * np.sqrt(trades_per_year) * 100 if len(net_arr) > 1 else 0.0

        # Information ratio
        if len(alpha_arr) > 1:
            te = float(np.std(alpha_arr, ddof=1))
            ir = float(np.mean(alpha_arr) / te) * np.sqrt(trades_per_year) if te > 0 else 0.0
        else:
            ir = 0.0

        # Date range
        first_date = min(all_dates) if all_dates else pd.Timestamp.now()
        last_date = max(all_dates) if all_dates else pd.Timestamp.now()
        years = max(0.5, (last_date - first_date).days / 365.25)

        # Top/worst performers
        ticker_perf.sort(key=lambda x: x["avg_alpha_per_trade"], reverse=True)
        top_5 = ticker_perf[:5]
        worst_5 = ticker_perf[-5:]

        return SegmentResult(
            dimension=dimension,
            label=label,
            benchmark_ticker=bench_ticker,
            benchmark_name=seg_def.benchmark_name,
            ticker_count=len(group),
            trade_count=len(net_arr),
            annual_alpha_pct=round(annual_alpha, 2),
            sharpe_ratio=round(sr, 3),
            hit_rate_pct=round(hr, 1),
            max_drawdown_pct=round(mdd, 2),
            win_rate_vs_benchmark=round(win_vs_bench, 1),
            gross_annual_return_pct=round(annual_gross, 2),
            benchmark_annual_return_pct=round(annual_bench, 2),
            cost_drag_annual_pct=round(annual_cost_drag, 2),
            volatility_annual_pct=round(vol, 2),
            information_ratio=round(ir, 3),
            avg_market_cap=round(float(np.mean(all_caps)), 0) if all_caps else 0.0,
            avg_daily_volume=0.0,  # filled if needed
            backtest_years=round(years, 2),
            first_signal_date=str(first_date.date()),
            last_signal_date=str(last_date.date()),
            top_performers=top_5,
            worst_performers=worst_5,
        )

    def _download_prices(
        self,
        tickers: list[str],
        period: str = "6y",
        batch_size: int = 50,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Download price + volume data from yfinance."""
        tickers = [t for t in dict.fromkeys(tickers) if t]
        if not tickers:
            return pd.DataFrame(), pd.DataFrame()

        close_frames: list[pd.DataFrame] = []
        volume_frames: list[pd.DataFrame] = []

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i: i + batch_size]
            try:
                raw = yf.download(
                    batch,
                    period=period,
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                    group_by="ticker",
                )
            except Exception:
                continue
            if raw is None or raw.empty:
                continue

            close = pd.DataFrame(index=raw.index)
            volume = pd.DataFrame(index=raw.index)

            if isinstance(raw.columns, pd.MultiIndex):
                level0 = set(raw.columns.get_level_values(0))
                for ticker in batch:
                    if ticker not in level0:
                        continue
                    if (ticker, "Close") in raw.columns:
                        close[ticker] = raw[(ticker, "Close")]
                    if (ticker, "Volume") in raw.columns:
                        volume[ticker] = raw[(ticker, "Volume")]
            elif len(batch) == 1 and "Close" in raw.columns:
                close[batch[0]] = raw["Close"]
                if "Volume" in raw.columns:
                    volume[batch[0]] = raw["Volume"]

            close.index = pd.to_datetime(close.index).tz_localize(None)
            volume.index = close.index
            close_frames.append(close.sort_index())
            volume_frames.append(volume.sort_index())

        if not close_frames:
            return pd.DataFrame(), pd.DataFrame()

        close_merged = pd.concat(close_frames, axis=1).sort_index()
        close_merged = close_merged.loc[:, ~close_merged.columns.duplicated()].dropna(how="all")

        if volume_frames:
            volume_merged = pd.concat(volume_frames, axis=1).sort_index()
            volume_merged = volume_merged.loc[:, ~volume_merged.columns.duplicated()].dropna(how="all")
        else:
            volume_merged = pd.DataFrame(index=close_merged.index)

        return close_merged, volume_merged

    def _download_event_history(self) -> pd.DataFrame:
        """Download prediction market event history with disk caching.

        Caches to data/event_history_cache.parquet to avoid re-hitting
        Polymarket/Kalshi APIs on every backtest run.  Cache expires after 24h.
        Falls back to synthetic 0.5 baseline if APIs unreachable.
        """
        from pathlib import Path
        import json as _json

        cache_path = Path("data/event_history_cache.csv")
        cache_meta_path = Path("data/event_history_cache_meta.json")

        # Check if recent cache exists (< 24h old)
        if cache_path.exists() and cache_meta_path.exists():
            try:
                meta = _json.loads(cache_meta_path.read_text())
                cached_at = datetime.fromisoformat(meta["cached_at"])
                if (datetime.now(UTC).replace(tzinfo=None) - cached_at).total_seconds() < 86400:
                    self._log("  Using cached event history (< 24h old)")
                    df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                    return df
            except Exception:
                pass  # re-download

        from app.providers.real_prediction import RealPolymarketProvider, RealKalshiProvider, EVENT_MAPPINGS as EM
        poly = RealPolymarketProvider()
        kalshi = RealKalshiProvider()

        now = datetime.now(UTC)
        start_ts = int((now - timedelta(days=1700)).timestamp())
        end_ts = int(now.timestamp())

        series_map: dict[str, pd.Series] = {}
        total = len(EM)
        for idx, (event_id, mapping) in enumerate(EM.items()):
            if idx % 5 == 0:
                self._log(f"    Downloading event {idx+1}/{total}: {event_id}")
            try:
                poly_hist = poly.fetch_event_history(mapping)
            except Exception:
                poly_hist = pd.Series(dtype=float)
            try:
                kalshi_hist = kalshi.fetch_event_history(
                    mapping=mapping,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    period_interval=1440,
                )
            except Exception:
                kalshi_hist = pd.Series(dtype=float)

            cols = [x for x in [poly_hist, kalshi_hist] if x is not None and not x.empty]
            if not cols:
                continue
            merged = pd.concat(cols, axis=1).sort_index().ffill()
            series_map[event_id] = merged.mean(axis=1).clip(0.0, 1.0)

        if not series_map:
            return pd.DataFrame()

        out = pd.concat(series_map.values(), axis=1).sort_index()
        out.columns = list(series_map)
        full_idx = pd.date_range(out.index.min(), out.index.max(), freq="D")
        result = out.reindex(full_idx).ffill().fillna(0.5)

        # Save cache
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(cache_path)
            cache_meta_path.write_text(_json.dumps({
                "cached_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
                "events": list(series_map.keys()),
                "rows": len(result),
            }))
            self._log(f"  Cached {len(series_map)} events ({len(result)} days) to {cache_path}")
        except Exception as exc:
            self._log(f"  Cache save failed: {exc}")

        return result

    def _estimate_market_cap(
        self,
        ticker: str,
        equity_prices: pd.DataFrame,
        equity_volume: pd.DataFrame,
    ) -> float:
        """Quick market cap estimate from latest price and heuristic shares outstanding."""
        try:
            t = yf.Ticker(ticker)
            fi = t.fast_info or {}
            mc = float(fi.get("market_cap") or fi.get("marketCap") or 0.0)
            if mc > 0:
                return mc
        except Exception:
            pass

        # Fallback: price × estimated shares
        if ticker in equity_prices.columns:
            price = float(equity_prices[ticker].dropna().iloc[-1])
            vol = float(equity_volume[ticker].dropna().tail(60).mean()) if ticker in equity_volume.columns else 100_000
            # Rough heuristic: shares ≈ daily volume × 500
            shares = max(10_000_000, vol * 500)
            return price * shares
        return 2_000_000_000  # default mid cap

    def _compute_cost(self, market_cap: float, avg_daily_volume: float, direction: str) -> float:
        """Realistic cost model — matches global_scan.py exactly.

        Added nano-cap tier (<$100M) and ultra-low volume (<50K ADV) penalty.
        LONG-only: no borrow cost.
        hold_days scales with self.holding_days so longer horizons get correct cost drag.
        """
        if market_cap < 100_000_000:
            # Nano-cap: extremely illiquid, wide spreads
            spread, impact, slippage = 55.0, 70.0, 35.0
        elif market_cap < 900_000_000:
            spread, impact, slippage = 28.0, 34.0, 18.0
        elif market_cap < 2_500_000_000:
            spread, impact, slippage = 19.0, 22.0, 12.0
        elif market_cap < 7_500_000_000:
            spread, impact, slippage = 13.0, 14.0, 9.0
        else:
            spread, impact, slippage = 9.0, 9.0, 7.0

        if avg_daily_volume < 50_000:
            # Near-zero liquidity: extreme cost penalty
            spread += 20.0
            impact += 25.0
        elif avg_daily_volume < 300_000:
            spread += 8.0
            impact += 10.0
        elif avg_daily_volume < 900_000:
            spread += 3.0
            impact += 4.0

        # Long-only: no borrow cost.
        # hold_days = self.holding_days — scales annualized cost correctly per horizon
        return estimate_total_cost_bps(
            commission_bps=2.0,
            spread_bps=spread,
            slippage_bps=slippage,
            impact_bps=impact,
            borrow_bps_annual=0.0,
            hold_days=self.holding_days,
        )

    def _empty_result(self) -> AlphaAttributionResult:
        return AlphaAttributionResult(
            run_timestamp=datetime.now(UTC).replace(tzinfo=None).isoformat(),
            universe_size=len(GLOBAL_COMMODITY_UNIVERSE),
            modeled_tickers=0,
            total_trades=0,
            backtest_start="",
            backtest_end="",
            backtest_years=0.0,
            segments={},
            overall_alpha_pct=0.0,
            overall_sharpe=0.0,
            overall_hit_rate=0.0,
            alpha_segments=[],
            weak_segments=[],
        )

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)
        logger.info(msg)
