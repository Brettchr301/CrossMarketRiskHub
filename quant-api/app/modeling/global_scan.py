from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

from app.modeling.commodity_impact import CommodityImpactModel
from app.modeling.cost_model import estimate_total_cost_bps, net_return_after_cost
from app.modeling.global_universe import GLOBAL_COMMODITY_UNIVERSE
from app.modeling.probability import EventProbabilityEngine
from app.modeling.research_hub import ResearchHubService
from app.providers.filing_provider import SECFilingProvider
from app.providers.real_prediction import EVENT_MAPPINGS, RealPredictionProvider


@dataclass(slots=True)
class GlobalScanCache:
    created_at: datetime
    lookback_days: int
    snapshot: dict[str, Any]


DEFAULT_MARGIN_BY_TYPE: dict[str, float] = {
    "oil_gas_upstream": 0.31,
    "oil_refining": 0.14,
    "oil_services": 0.16,
    "shipping_tanker": 0.27,
    "shipping_drybulk": 0.22,
    "shipping_container": 0.2,
    "shipping_services": 0.17,
    "lng_shipping": 0.24,
    "coal": 0.26,
    "base_metals": 0.19,
    "precious_metals": 0.29,
    "lithium": 0.24,
    "uranium": 0.18,
    "agri_inputs": 0.21,
    "midstream": 0.34,
    "rare_earths": 0.2,
}


class GlobalOpportunityService:
    def __init__(self) -> None:
        self.prediction = RealPredictionProvider()
        self.probability_engine = EventProbabilityEngine(min_liquidity=0.05, stale_minutes=240)
        self.impact = CommodityImpactModel(horizon_days=60, n_sims=3200)
        self.research_hub = ResearchHubService()
        self._sec_provider = SECFilingProvider()
        self._cache: dict[tuple[int, int, int], GlobalScanCache] = {}
        self._fundamental_cache: dict[str, dict[str, float]] = {}
        self._lock = threading.Lock()
        self._computing: set[tuple[int, int, int]] = set()
        self._cache_file = Path("analysis_output/global_opportunity_scan_cache.json")

    def scan_async(
        self,
        lookback_days: int = 780,
        min_modeled_count: int = 200,
        max_rows: int = 220,
    ) -> dict[str, Any] | None:
        """Return cached snapshot immediately, or None if still computing.
        Spawns a background thread to compute if cache is stale."""
        key = (lookback_days, min_modeled_count, max_rows)
        now = datetime.now(UTC).replace(tzinfo=None)
        with self._lock:
            cached = self._cache.get(key)
            if cached and now - cached.created_at <= timedelta(minutes=45):
                return cached.snapshot
            if key in self._computing:
                return None  # still computing
            self._computing.add(key)

        def _run() -> None:
            try:
                result = self.scan(lookback_days=lookback_days, min_modeled_count=min_modeled_count, max_rows=max_rows)
                with self._lock:
                    self._cache[key] = GlobalScanCache(created_at=datetime.now(UTC).replace(tzinfo=None), lookback_days=lookback_days, snapshot=result)
            except Exception as exc:
                logger.warning("Background global scan failed: %s", exc)
            finally:
                with self._lock:
                    self._computing.discard(key)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return None  # computing, check back later

    def scan(
        self,
        lookback_days: int = 780,
        min_modeled_count: int = 200,
        max_rows: int = 220,
    ) -> dict[str, Any]:
        key = (lookback_days, min_modeled_count, max_rows)
        now = datetime.now(UTC).replace(tzinfo=None)
        cached = self._cache.get(key)
        if cached and now - cached.created_at <= timedelta(minutes=45):
            return cached.snapshot
        disk_cached = self._load_disk_cache(key=key, now=now)
        if disk_cached is not None:
            self._cache[key] = GlobalScanCache(created_at=now, lookback_days=lookback_days, snapshot=disk_cached)
            return disk_cached

        meta = {row.ticker: row for row in GLOBAL_COMMODITY_UNIVERSE}
        tickers = list(meta)
        factor_prices = self._download_close_frame(
            [
                "BZ=F",
                "CL=F",
                "BOAT",
                "SEA",
                "BDRY",
                "^VIX",
                "DX-Y.NYB",
                "^TNX",
                "^IRX",
                "GC=F",
                "^GSPC",
            ],
            period="6y",
        )
        if factor_prices.empty or ("BZ=F" not in factor_prices.columns and "CL=F" not in factor_prices.columns):
            raise RuntimeError("Missing factor history for global scan.")

        spot_proxy = "BOAT" if "BOAT" in factor_prices.columns else ("SEA" if "SEA" in factor_prices.columns else "BDRY")
        fwd_proxy = "BDRY" if "BDRY" in factor_prices.columns else spot_proxy

        event_hist = self._event_history(
            [
                "hormuz_closure", "red_sea_disruption", "sanctions_escalation", "oil_above_100",
                "opec_production_cut", "panama_canal_disruption", "china_stimulus",
                "us_spr_release", "us_refinery_utilization_low",
            ]
        )
        xplat_spreads = self._cross_platform_event_spreads(
            ["hormuz_closure", "oil_above_100", "sanctions_escalation"]
        )
        if event_hist.empty:
            raise RuntimeError("Prediction-event history unavailable for global scan.")

        probs = self._current_event_probabilities(now=now)
        base_prices = {
            "BRENT": float(factor_prices["BZ=F"].dropna().iloc[-1]) if "BZ=F" in factor_prices.columns else 80.0,
            "WTI": float(factor_prices["CL=F"].dropna().iloc[-1]) if "CL=F" in factor_prices.columns else 75.0,
            "TD3": float(factor_prices[spot_proxy].dropna().iloc[-1]),
            "BDI": float(factor_prices[fwd_proxy].dropna().iloc[-1]),
            "BCTI": float(factor_prices[spot_proxy].dropna().iloc[-1]),
        }
        _, market_paths, _ = self.impact.generate(event_probs=probs, base_prices=base_prices)
        path_expectations = self._path_expectations(base_prices=base_prices, market_paths=market_paths)

        predictive_cache = self.research_hub._ensure_predictive_cache(
            lookback_days=max(420, lookback_days),
            max_markets=28,
        )
        top_contracts = predictive_cache.contracts[:120]
        contract_delta_map = predictive_cache.contract_delta_map

        equity_prices, equity_volume = self._download_price_volume_frames(tickers, period="6y", batch_size=50)
        if equity_prices.empty:
            raise RuntimeError("No equity history downloaded for global universe.")

        modeled_rows: list[dict[str, Any]] = []
        for ticker in tickers:
            row_meta = meta[ticker]
            if ticker not in equity_prices.columns:
                continue
            frame = self._build_feature_frame(
                ticker=ticker,
                equity_prices=equity_prices,
                factors=factor_prices,
                events=event_hist,
                xplat_spreads=xplat_spreads,
                contracts=top_contracts,
                contract_delta_map=contract_delta_map,
                lookback_days=lookback_days,
                spot_proxy=spot_proxy,
                fwd_proxy=fwd_proxy,
                equity_volume=equity_volume,
            )
            if frame is None or len(frame) < 120:
                continue

            fit = self._walkforward_fit(frame=frame, target_col="y_fwd_20d")
            if fit is None:
                continue

            spot = float(frame["stock_px"].iloc[-1])
            fund = self._fundamental_proxy(
                ticker=ticker,
                commodity_type=row_meta.commodity_type,
                spot_price=spot,
                volume_hint=float(frame["stock_vol"].tail(20).mean()) if "stock_vol" in frame.columns else 0.0,
            )
            valuation = self._valuation_from_fit(
                fit=fit,
                fund=fund,
                commodity_type=row_meta.commodity_type,
                spot=spot,
                path_expectations=path_expectations,
                latest_event_frame=frame,
            )
            if valuation is None:
                continue

            direction = "LONG" if valuation["expected_return_gross"] >= 0 else "SHORT"
            cost_bps = self._dynamic_cost_bps(
                market_cap=fund["market_cap"],
                avg_daily_volume=fund["avg_daily_volume"],
                direction=direction,
            )
            expected_net = net_return_after_cost(valuation["expected_return_gross"], cost_bps)
            score = 100.0 * expected_net * (0.5 + 0.5 * fit["confidence"]) * valuation["liquidity_quality"]

            risk_flags: list[str] = []
            if fund["market_cap"] < 1_000_000_000.0:
                risk_flags.append("micro_small_cap")
            if fund["avg_daily_volume"] < 300_000.0:
                risk_flags.append("low_liquidity")
            if fit["hit_rate"] < 0.5:
                risk_flags.append("weak_predictive_fit")
            if abs(valuation["predicted_margin_change"]) > 0.2:
                risk_flags.append("high_margin_volatility")

            matching_contracts = [
                c["question"]
                for c in top_contracts
                if c["best_target"] in {ticker, "BZ=F", "CL=F", spot_proxy, fwd_proxy}
            ][:6]

            # Top 5 model features by absolute coefficient (for narrative endpoint)
            coef = fit.get("coef", {})
            top_features = sorted(
                [(abs(v), k) for k, v in coef.items() if k != "intercept" and v != 0.0],
                reverse=True,
            )[:5]
            top_feature_names = [k for _, k in top_features]

            modeled_rows.append(
                {
                    "ticker": ticker,
                    "commodity_type": row_meta.commodity_type,
                    "country": row_meta.country,
                    "sector": row_meta.sector,
                    "direction": direction,
                    "score": float(score),
                    "spot_price": float(spot),
                    "fair_value_price": float(valuation["fair_value_price"]),
                    "expected_return_gross": float(valuation["expected_return_gross"]),
                    "expected_return_net_cost": float(expected_net),
                    "cost_bps": float(cost_bps),
                    "hit_rate": float(fit["hit_rate"]),
                    "mae": float(fit["mae"]),
                    "confidence": float(fit["confidence"]),
                    "predicted_margin_next": float(valuation["predicted_margin_next"]),
                    "predicted_margin_change": float(valuation["predicted_margin_change"]),
                    "production_growth_assumption": float(valuation["production_growth_assumption"]),
                    "oil_beta": float(valuation["oil_beta"]),
                    "oil_gamma": float(valuation["oil_gamma"]),
                    "shipping_beta": float(valuation["shipping_beta"]),
                    "shipping_gamma": float(valuation["shipping_gamma"]),
                    "event_beta": float(valuation["event_beta"]),
                    "event_gamma": float(valuation["event_gamma"]),
                    "market_cap": float(fund["market_cap"]),
                    "avg_daily_volume": float(fund["avg_daily_volume"]),
                    "risk_flags": risk_flags,
                    "top_predictive_contracts": matching_contracts,
                    "top_features": top_feature_names,
                }
            )

        modeled_rows.sort(key=lambda x: x["score"], reverse=True)
        if len(modeled_rows) < min_modeled_count:
            raise RuntimeError(
                f"Modeled {len(modeled_rows)} names, below requested minimum {min_modeled_count}. "
                "Increase lookback or reduce minimum."
            )

        commodity_stats = self._commodity_type_stats(modeled_rows)
        snapshot = {
            "as_of": now,
            "lookback_days": lookback_days,
            "universe_size": len(tickers),
            "modeled_count": len(modeled_rows),
            "spot_proxy": spot_proxy,
            "forward_proxy": fwd_proxy,
            "commodity_type_stats": commodity_stats,
            "opportunities": modeled_rows[:max_rows],
        }
        self._cache[key] = GlobalScanCache(created_at=now, lookback_days=lookback_days, snapshot=snapshot)
        self._write_disk_cache(key=key, snapshot=snapshot)
        return snapshot

    def _load_disk_cache(self, key: tuple[int, int, int], now: datetime) -> dict[str, Any] | None:
        if not self._cache_file.exists():
            return None
        try:
            payload = json.loads(self._cache_file.read_text(encoding="utf-8"))
            if payload.get("key") != list(key):
                return None
            generated_at = datetime.fromisoformat(payload["generated_at"])
            if now - generated_at > timedelta(hours=24):
                return None
            snapshot = payload["snapshot"]
            if isinstance(snapshot.get("as_of"), str):
                snapshot["as_of"] = datetime.fromisoformat(snapshot["as_of"])
            return snapshot
        except Exception:
            return None

    def _write_disk_cache(self, key: tuple[int, int, int], snapshot: dict[str, Any]) -> None:
        try:
            serializable = dict(snapshot)
            serializable["as_of"] = snapshot["as_of"].isoformat()
            payload = {
                "generated_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
                "key": list(key),
                "snapshot": serializable,
            }
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            self._cache_file.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            return

    def _path_expectations(self, base_prices: dict[str, float], market_paths: dict[str, np.ndarray]) -> dict[str, float]:
        def _moment(symbol: str, base: float) -> tuple[float, float]:
            samples = market_paths.get(symbol)
            if samples is None or len(samples) == 0 or base <= 0:
                return 0.0, 0.0
            ret = np.asarray(samples, dtype=float) / base - 1.0
            return float(np.mean(ret)), float(np.var(ret))

        oil_mu, oil_var = _moment("BRENT", base_prices["BRENT"])
        ship_mu, ship_var = _moment("TD3", base_prices["TD3"])
        fwd_mu, fwd_var = _moment("BDI", base_prices["BDI"])
        return {
            "oil_mu": oil_mu,
            "oil_var": oil_var,
            "shipping_mu": ship_mu,
            "shipping_var": ship_var,
            "shipping_fwd_mu": fwd_mu,
            "shipping_fwd_var": fwd_var,
        }

    def _current_event_probabilities(self, now: datetime) -> list[Any]:
        quotes = self.prediction.fetch_event_quotes(list(EVENT_MAPPINGS))
        return self.probability_engine.compute(quotes=quotes, linked_events=None, as_of=now)

    def _event_history(self, event_ids: list[str]) -> pd.DataFrame:
        series_map: dict[str, pd.Series] = {}
        now = datetime.now(UTC)
        start_ts = int((now - timedelta(days=1700)).timestamp())
        end_ts = int(now.timestamp())
        for event_id in event_ids:
            mapping = EVENT_MAPPINGS.get(event_id)
            if mapping is None:
                continue
            poly = self.prediction.polymarket.fetch_event_history(mapping)
            kalshi = self.prediction.kalshi.fetch_event_history(
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

    def _cross_platform_event_spreads(self, event_ids: list[str]) -> pd.DataFrame:
        """
        Historical Polymarket-Kalshi spread series per event.
        Prevents live-snapshot leakage in walk-forward modeling.
        """
        spread_map: dict[str, pd.Series] = {}
        now = datetime.now(UTC)
        start_ts = int((now - timedelta(days=1700)).timestamp())
        end_ts = int(now.timestamp())
        for event_id in event_ids:
            mapping = EVENT_MAPPINGS.get(event_id)
            if mapping is None:
                continue
            poly = self.prediction.polymarket.fetch_event_history(mapping)
            kalshi = self.prediction.kalshi.fetch_event_history(
                mapping=mapping,
                start_ts=start_ts,
                end_ts=end_ts,
                period_interval=1440,
            )
            if poly.empty or kalshi.empty:
                continue
            merged = pd.concat([poly.rename("poly"), kalshi.rename("kalshi")], axis=1).sort_index().ffill()
            spread = (merged["poly"] - merged["kalshi"]).clip(-1.0, 1.0)
            spread_map[f"pm_xplat_spread_{event_id}"] = spread
        if not spread_map:
            return pd.DataFrame()
        out = pd.concat(spread_map.values(), axis=1).sort_index()
        out.columns = list(spread_map)
        full_idx = pd.date_range(out.index.min(), out.index.max(), freq="D")
        return out.reindex(full_idx).ffill().fillna(0.0)

    @staticmethod
    def _download_close_frame(tickers: list[str], period: str = "6y", batch_size: int = 80) -> pd.DataFrame:
        close, _ = GlobalOpportunityService._download_price_volume_frames(
            tickers=tickers,
            period=period,
            batch_size=batch_size,
        )
        return close

    @staticmethod
    def _download_price_volume_frames(
        tickers: list[str],
        period: str = "6y",
        batch_size: int = 80,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        tickers = [t for t in dict.fromkeys(tickers) if t]
        if not tickers:
            return pd.DataFrame(), pd.DataFrame()
        close_frames: list[pd.DataFrame] = []
        volume_frames: list[pd.DataFrame] = []
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i + batch_size]
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

    def _build_feature_frame(
        self,
        ticker: str,
        equity_prices: pd.DataFrame,
        equity_volume: pd.DataFrame,
        factors: pd.DataFrame,
        events: pd.DataFrame,
        xplat_spreads: pd.DataFrame,
        contracts: list[dict[str, Any]],
        contract_delta_map: dict[str, pd.Series],
        lookback_days: int,
        spot_proxy: str,
        fwd_proxy: str,
    ) -> pd.DataFrame | None:
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

        for col, name in [
            ("^VIX", "macro_vix"),
            ("DX-Y.NYB", "macro_dxy"),
            ("^TNX", "macro_us10y"),
            ("^IRX", "macro_us3m"),
            ("GC=F", "macro_gold"),
            ("^GSPC", "macro_sp500"),
        ]:
            if col not in factors.columns:
                continue
            px = factors[col].rename(f"{name}_px")
            df = df.join(px, how="left")
            ret = px.pct_change(fill_method=None)
            df[f"{name}_ret"] = ret
            df[f"{name}_sq"] = ret * ret

        if "macro_us10y_px" in df.columns and "macro_us3m_px" in df.columns:
            curve = (df["macro_us10y_px"] - df["macro_us3m_px"]) / 100.0
            df["macro_yield_curve_spread"] = curve
            df["macro_yield_curve_spread_d1"] = curve.diff()
        if "macro_vix_px" in df.columns:
            df["macro_risk_off_regime"] = (df["macro_vix_px"] > 22.0).astype(float)

        df = df.join(events, how="left")
        for ev in [
            "hormuz_closure", "red_sea_disruption", "sanctions_escalation", "oil_above_100",
            "opec_production_cut", "panama_canal_disruption", "china_stimulus",
            "us_spr_release", "us_refinery_utilization_low",
        ]:
            if ev not in df.columns:
                df[ev] = 0.5
            d = df[ev].diff()
            df[f"{ev}_d1"] = d
            df[f"{ev}_d2"] = d.diff()
            df[f"{ev}_sq"] = d * d

        if not xplat_spreads.empty:
            df = df.join(xplat_spreads, how="left")
            for col in xplat_spreads.columns:
                if col not in df.columns:
                    continue
                df[f"{col}_d1"] = df[col].diff()
                df[f"{col}_sq"] = df[col] * df[col]

        for contract in contracts[:24]:
            cid = contract["market_id"]
            delta = contract_delta_map.get(cid)
            if delta is None or delta.empty:
                continue
            lag = int(contract.get("lead_days", 0))
            s = delta.shift(lag).rename(f"pm_{cid}")
            df = df.join(s, how="left")
            df[f"pm2_{cid}"] = df[f"pm_{cid}"].diff()

        # Cross-features recommended by DeepSeek: brent×shipping interaction, WTI-Brent spread
        if "brent_ret" in df.columns and "ship_spot_ret" in df.columns:
            df["brent_ship_cross"] = df["brent_ret"] * df["ship_spot_ret"]
        if "brent_ret" in df.columns and "wti_ret" in df.columns:
            df["wti_brent_spread_ret"] = df["wti_ret"] - df["brent_ret"]
        if "brent_ret" in df.columns:
            # Rolling 20d vol regime: 1 = high vol, 0 = low vol (proxy for VIX regime)
            rolling_vol = df["brent_ret"].rolling(20).std()
            df["high_vol_regime"] = (rolling_vol > rolling_vol.rolling(60).mean()).astype(float).fillna(0.0)
        if "hormuz_closure_d1" in df.columns and "ship_fwd_ret" in df.columns:
            df["event_freight_cross"] = df["hormuz_closure_d1"] * df["ship_fwd_ret"]

        # T4: Brent term structure proxy — contango (>0) vs backwardation (<0) signal
        # Uses price vs 60-day rolling mean as a proxy for 1M-3M curve shape
        if "brent_px" in df.columns:
            brent_roll60 = df["brent_px"].rolling(60, min_periods=20).mean()
            df["brent_contango_ret"] = (df["brent_px"] / brent_roll60 - 1.0).fillna(0.0)
            df["brent_contango_d1"] = df["brent_contango_ret"].diff()

        df["y_fwd_20d"] = df["stock_px"].shift(-20) / df["stock_px"] - 1.0
        keep_cols = [c for c in df.columns if c.endswith(("_ret", "_sq", "_accel", "_d1", "_d2", "_cross", "_spread_ret", "_regime", "_spread", "_boost")) or c.startswith(("pm_", "pm2_", "pm_xplat_", "opt_", "macro_"))]
        keep_cols.extend(["stock_px", "stock_vol", "y_fwd_20d"])
        out = df[keep_cols].replace([np.inf, -np.inf], np.nan)
        feature_cols = [c for c in out.columns if c not in {"stock_px", "stock_vol", "y_fwd_20d"}]
        out[feature_cols] = out[feature_cols].fillna(0.0)
        constant_cols = [c for c in feature_cols if out[c].nunique(dropna=True) <= 1]
        if constant_cols:
            out = out.drop(columns=constant_cols)
        out = out.dropna(subset=["stock_px", "y_fwd_20d"])
        return out.tail(max(lookback_days, 140))

    def _walkforward_fit(
        self,
        frame: pd.DataFrame,
        target_col: str,
        lookback: int = 150,
    ) -> dict[str, Any] | None:
        if target_col not in frame.columns:
            return None
        feature_cols = [c for c in frame.columns if c not in {target_col, "stock_px", "stock_vol"}]
        if not feature_cols:
            return None
        if len(frame) < max(lookback + 20, 120):
            return None

        preds: list[float] = []
        actuals: list[float] = []
        last_model: Ridge | None = None
        sample_idx: list[int] = []
        # RidgeCV with TimeSeriesSplit(5): auto-selects optimal alpha per walk-forward window
        # Outperforms fixed alpha=0.8 by adapting regularization to feature correlation regime
        _alphas = [0.01, 0.1, 0.5, 1.0, 5.0, 20.0]
        _tscv = TimeSeriesSplit(n_splits=5)
        for i in range(lookback, len(frame)):
            train = frame.iloc[i - lookback : i]
            x = train[feature_cols].values
            y = train[target_col].values
            ridge = RidgeCV(alphas=_alphas, cv=_tscv, fit_intercept=True)
            ridge.fit(x, y)
            row = frame.iloc[i]
            pred = float(ridge.predict(row[feature_cols].values.reshape(1, -1))[0])
            preds.append(pred)
            actuals.append(float(row[target_col]))
            sample_idx.append(i)
            last_model = Ridge(alpha=float(ridge.alpha_), fit_intercept=True)
            last_model.intercept_ = ridge.intercept_
            last_model.coef_ = ridge.coef_.copy()
        if not preds or last_model is None:
            return None

        pred_arr = np.array(preds, dtype=float)
        act_arr = np.array(actuals, dtype=float)
        mae = float(np.mean(np.abs(pred_arr - act_arr)))
        hit = float(np.mean(np.sign(pred_arr) == np.sign(act_arr)))
        corr = float(np.corrcoef(pred_arr, act_arr)[0, 1]) if len(pred_arr) > 3 else 0.0
        if np.isnan(corr):
            corr = 0.0
        confidence = float(max(0.0, min(1.0, 0.45 * hit + 0.35 * max(corr, 0.0) + 0.2 * (1.0 - min(mae, 0.2) / 0.2))))

        coef_map = {"intercept": float(last_model.intercept_)}
        for idx, col in enumerate(feature_cols):
            coef_map[col] = float(last_model.coef_[idx])
        return {
            "latest_pred": float(pred_arr[-1]),
            "hit_rate": hit,
            "mae": mae,
            "correlation": corr,
            "confidence": confidence,
            "coef": coef_map,
            "n_samples": len(pred_arr),
            "feature_cols": feature_cols,
            "sample_index": sample_idx,
            "ridge_alpha": float(last_model.alpha),
        }

    def _fundamental_proxy(
        self,
        ticker: str,
        commodity_type: str,
        spot_price: float,
        volume_hint: float,
    ) -> dict[str, float]:
        cached = self._fundamental_cache.get(ticker)
        if cached:
            return cached

        market_cap = 0.0
        shares = 0.0
        debt = 0.0
        cash = 0.0
        ev_ebitda = 6.0
        base_margin = DEFAULT_MARGIN_BY_TYPE.get(commodity_type, 0.2)
        production_growth = 0.03
        avg_daily_volume = max(0.0, volume_hint)

        # T2: Pull real fundamentals from SEC EDGAR (10-K/10-Q) via SECFilingProvider
        try:
            sec_data = self._sec_provider.get_fundamentals_dict(ticker)
            if sec_data:
                if sec_data.get("gross_margin", 0.0) > 0:
                    base_margin = float(sec_data["gross_margin"])
                if sec_data.get("production_volume", 0.0) > 0:
                    # Use production volume as a proxy for scale; normalize to growth signal
                    production_growth = np.clip(
                        production_growth + 0.01 * np.log1p(sec_data["production_volume"] / 1000.0),
                        -0.2, 0.35
                    )
                logger.debug("SEC fundamentals loaded for %s: margin=%.3f src=%s", ticker, base_margin, sec_data.get("source", ""))
        except Exception as exc:
            logger.debug("SEC filing lookup failed for %s: %s", ticker, exc)

        try:
            t = yf.Ticker(ticker)
            fi = t.fast_info or {}
            market_cap = float(fi.get("market_cap") or fi.get("marketCap") or 0.0)
            shares = float(fi.get("shares") or fi.get("sharesOutstanding") or 0.0)
            if avg_daily_volume <= 0:
                avg_daily_volume = float(fi.get("ten_day_average_volume") or fi.get("tenDayAverageVolume") or 0.0)
        except Exception:
            pass

        if market_cap <= 0:
            shares = shares if shares > 0 else 60_000_000.0
            market_cap = max(150_000_000.0, spot_price * shares)
        if shares <= 0:
            shares = max(10_000_000.0, market_cap / max(spot_price, 0.1))
        if debt <= 0:
            debt = market_cap * (0.5 if "shipping" in commodity_type else 0.35)
        if cash <= 0:
            cash = market_cap * 0.07
        if avg_daily_volume <= 0:
            avg_daily_volume = max(40_000.0, shares * 0.002)
        production_growth = np.clip(production_growth + 0.07 * np.tanh(avg_daily_volume / 2_000_000.0), -0.2, 0.35)
        ev_ebitda = max(2.5, min(15.0, ev_ebitda))
        base_margin = max(-0.1, min(0.7, base_margin))
        production_growth = float(max(-0.2, min(0.35, production_growth)))

        ev_current = market_cap + debt - cash
        base_ebitda = max(20_000_000.0, ev_current / max(ev_ebitda, 1.5))

        out = {
            "market_cap": float(market_cap),
            "shares": float(shares),
            "debt": float(debt),
            "cash": float(cash),
            "ev_to_ebitda": float(ev_ebitda),
            "base_margin": float(base_margin),
            "production_growth": float(production_growth),
            "base_ebitda": float(base_ebitda),
            "avg_daily_volume": float(avg_daily_volume),
        }
        self._fundamental_cache[ticker] = out
        return out

    def _valuation_from_fit(
        self,
        fit: dict[str, Any],
        fund: dict[str, float],
        commodity_type: str,
        spot: float,
        path_expectations: dict[str, float],
        latest_event_frame: pd.DataFrame,
    ) -> dict[str, float] | None:
        coef = fit["coef"]
        oil_beta = float(coef.get("brent_ret", 0.0) + 0.75 * coef.get("wti_ret", 0.0))
        oil_gamma = float(coef.get("brent_sq", 0.0) + 0.6 * coef.get("wti_sq", 0.0) + 0.4 * coef.get("brent_accel", 0.0))
        shipping_beta = float(
            coef.get("ship_spot_ret", 0.0)
            + 0.8 * coef.get("ship_fwd_ret", 0.0)
            + 0.4 * coef.get("ship_fwd_accel", 0.0)
        )
        shipping_gamma = float(coef.get("ship_spot_sq", 0.0) + coef.get("ship_fwd_sq", 0.0))

        event_beta = float(
            coef.get("hormuz_closure_d1", 0.0)
            + coef.get("red_sea_disruption_d1", 0.0)
            + coef.get("sanctions_escalation_d1", 0.0)
            + coef.get("oil_above_100_d1", 0.0)
        )
        event_gamma = float(
            coef.get("hormuz_closure_d2", 0.0)
            + coef.get("red_sea_disruption_d2", 0.0)
            + coef.get("sanctions_escalation_d2", 0.0)
            + coef.get("oil_above_100_d2", 0.0)
        )

        latest = latest_event_frame.iloc[-1]
        event_slope = float(
            latest.get("hormuz_closure_d1", 0.0)
            + latest.get("red_sea_disruption_d1", 0.0)
            + latest.get("sanctions_escalation_d1", 0.0)
            + latest.get("oil_above_100_d1", 0.0)
        )
        event_curv = float(
            latest.get("hormuz_closure_d2", 0.0)
            + latest.get("red_sea_disruption_d2", 0.0)
            + latest.get("sanctions_escalation_d2", 0.0)
            + latest.get("oil_above_100_d2", 0.0)
        )

        oil_component = oil_beta * path_expectations["oil_mu"] + 0.5 * oil_gamma * (
            path_expectations["oil_var"] + path_expectations["oil_mu"] ** 2
        )
        ship_component = shipping_beta * path_expectations["shipping_mu"] + 0.5 * shipping_gamma * (
            path_expectations["shipping_var"] + path_expectations["shipping_mu"] ** 2
        )
        event_component = event_beta * event_slope + 0.5 * event_gamma * (event_curv * event_curv)

        beta_scale = 1.0
        if "shipping" in commodity_type:
            beta_scale = 1.2
        elif commodity_type in {"oil_gas_upstream", "oil_refining", "oil_services"}:
            beta_scale = 1.1
        elif commodity_type in {"coal", "base_metals", "lithium", "uranium"}:
            beta_scale = 0.95

        predicted_margin_change = beta_scale * (oil_component + ship_component + event_component)
        predicted_margin_change = float(max(-0.22, min(0.32, predicted_margin_change)))
        predicted_margin_next = float(max(-0.12, min(0.78, fund["base_margin"] + predicted_margin_change)))
        production_growth = float(fund["production_growth"] + 0.35 * event_component + 0.2 * oil_component)
        production_growth = max(-0.22, min(0.36, production_growth))

        margin_lift = (predicted_margin_next - fund["base_margin"]) / max(abs(fund["base_margin"]), 0.08)
        ebitda_growth = production_growth + margin_lift
        fair_ebitda = fund["base_ebitda"] * (1.0 + ebitda_growth)
        fair_multiple = fund["ev_to_ebitda"] * (0.9 + 0.25 * fit["confidence"] + 0.08 * np.sign(ebitda_growth))
        fair_multiple = max(2.0, min(16.0, fair_multiple))
        ev_fair = max(0.0, fair_ebitda * fair_multiple)
        equity_fair = max(0.0, ev_fair - fund["debt"] + fund["cash"])
        fair_value_price = equity_fair / max(fund["shares"], 1.0)
        expected_return_gross = fair_value_price / max(spot, 0.01) - 1.0
        liquidity_quality = max(0.35, min(1.0, np.log10(max(fund["avg_daily_volume"], 50_000.0)) / 7.0))

        return {
            "fair_value_price": float(fair_value_price),
            "expected_return_gross": float(expected_return_gross),
            "predicted_margin_next": float(predicted_margin_next),
            "predicted_margin_change": float(predicted_margin_change),
            "production_growth_assumption": float(production_growth),
            "oil_beta": oil_beta,
            "oil_gamma": oil_gamma,
            "shipping_beta": shipping_beta,
            "shipping_gamma": shipping_gamma,
            "event_beta": event_beta,
            "event_gamma": event_gamma,
            "liquidity_quality": float(liquidity_quality),
        }

    @staticmethod
    def _dynamic_cost_bps(market_cap: float, avg_daily_volume: float, direction: str) -> float:
        if market_cap < 900_000_000.0:
            spread = 28.0
            impact = 34.0
            slippage = 18.0
        elif market_cap < 2_500_000_000.0:
            spread = 19.0
            impact = 22.0
            slippage = 12.0
        elif market_cap < 7_500_000_000.0:
            spread = 13.0
            impact = 14.0
            slippage = 9.0
        else:
            spread = 9.0
            impact = 9.0
            slippage = 7.0

        if avg_daily_volume < 300_000.0:
            spread += 8.0
            impact += 10.0
        elif avg_daily_volume < 900_000.0:
            spread += 3.0
            impact += 4.0

        borrow = 320.0 if direction == "SHORT" else 90.0
        return estimate_total_cost_bps(
            commission_bps=2.0,
            spread_bps=spread,
            slippage_bps=slippage,
            impact_bps=impact,
            borrow_bps_annual=borrow,
            hold_days=60,
        )

    @staticmethod
    def _commodity_type_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        agg: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            agg[row["commodity_type"]].append(row)
        out: list[dict[str, Any]] = []
        for commodity_type, items in agg.items():
            items_sorted = sorted(items, key=lambda x: x["score"], reverse=True)
            top_n = max(1, int(0.15 * len(items_sorted)))
            top = items_sorted[:top_n]
            out.append(
                {
                    "commodity_type": commodity_type,
                    "modeled_count": len(items),
                    "avg_hit_rate": float(np.mean([x["hit_rate"] for x in items])),
                    "avg_expected_return_net_cost": float(np.mean([x["expected_return_net_cost"] for x in items])),
                    "avg_score": float(np.mean([x["score"] for x in items])),
                    "top_bucket_avg_net_return": float(np.mean([x["expected_return_net_cost"] for x in top])),
                }
            )
        out.sort(
            key=lambda x: (
                x["top_bucket_avg_net_return"] * np.sqrt(max(1, x["modeled_count"])),
                x["avg_score"],
            ),
            reverse=True,
        )
        return out
