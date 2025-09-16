from __future__ import annotations

import json
import math
import re
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from app.config import get_settings
from app.providers.real_prediction import EVENT_MAPPINGS, RealPredictionProvider


@dataclass(slots=True)
class PredictiveCache:
    created_at: datetime
    lookback_days: int
    contracts: list[dict[str, Any]]
    contract_delta_map: dict[str, pd.Series]
    returns: pd.DataFrame


class ResearchHubService:
    BASE_GAMMA = "https://gamma-api.polymarket.com/markets"

    def __init__(self) -> None:
        self.settings = get_settings()
        self.prediction = RealPredictionProvider()
        self._predictive_cache: PredictiveCache | None = None
        self._ticker_cache: dict[tuple[str, int], tuple[datetime, dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._computing_predictive: bool = False
        self._computing_ticker: set[tuple[str, int]] = set()

    def predictive_contracts_snapshot(self, lookback_days: int = 420) -> dict[str, Any]:
        cache = self._ensure_predictive_cache(lookback_days=lookback_days)
        return {
            "as_of": cache.created_at,
            "lookback_days": cache.lookback_days,
            "contracts": cache.contracts[:80],
        }

    def predictive_contracts_snapshot_async(self, lookback_days: int = 420) -> dict[str, Any] | None:
        """Return cached snapshot immediately or None if computing. Spawns background thread if stale."""
        now = datetime.now(UTC).replace(tzinfo=None)
        with self._lock:
            if self._predictive_cache and now - self._predictive_cache.created_at <= timedelta(minutes=25):
                return {
                    "as_of": self._predictive_cache.created_at,
                    "lookback_days": self._predictive_cache.lookback_days,
                    "contracts": self._predictive_cache.contracts[:80],
                }
            if self._computing_predictive:
                return None
            self._computing_predictive = True

        def _run() -> None:
            try:
                self._ensure_predictive_cache(lookback_days=lookback_days)
            except Exception:
                pass
            finally:
                with self._lock:
                    self._computing_predictive = False

        threading.Thread(target=_run, daemon=True).start()
        return None

    def ticker_research_async(self, ticker: str, lookback_days: int = 260) -> dict[str, Any] | None:
        """Return cached ticker research immediately or None if computing."""
        ticker_u = ticker.upper()
        key = (ticker_u, lookback_days)
        now = datetime.now(UTC).replace(tzinfo=None)
        with self._lock:
            cached = self._ticker_cache.get(key)
            if cached and now - cached[0] <= timedelta(minutes=15):
                return cached[1]
            if key in self._computing_ticker:
                return None
            self._computing_ticker.add(key)

        def _run() -> None:
            try:
                self.ticker_research(ticker=ticker, lookback_days=lookback_days)
            except Exception:
                pass
            finally:
                with self._lock:
                    self._computing_ticker.discard(key)

        threading.Thread(target=_run, daemon=True).start()
        return None

    def ticker_research(self, ticker: str, lookback_days: int = 260) -> dict[str, Any]:
        ticker_u = ticker.upper()
        key = (ticker_u, lookback_days)
        now = datetime.now(UTC).replace(tzinfo=None)
        cached = self._ticker_cache.get(key)
        if cached and now - cached[0] <= timedelta(minutes=15):
            return cached[1]

        predictive = self._ensure_predictive_cache(lookback_days=max(420, lookback_days + 120))
        top_contracts = [
            c
            for c in predictive.contracts
            if c["best_target"] in {ticker_u, "BZ=F", "CL=F", "BDRY", "BOAT", "SEA"}
        ][:18]

        base_prices = self._download_close_frame([ticker_u, "BZ=F", "CL=F", "BOAT", "SEA", "BDRY"], period="5y")
        if base_prices.empty or ticker_u not in base_prices.columns:
            raise RuntimeError(f"No price history available for {ticker_u}.")

        spot_col = "BOAT" if "BOAT" in base_prices.columns else ("SEA" if "SEA" in base_prices.columns else ticker_u)
        fwd_col = "BDRY" if "BDRY" in base_prices.columns else spot_col
        event_history = self._event_history(["hormuz_closure", "red_sea_disruption", "oil_above_100"])

        joined = base_prices.join(event_history, how="inner").sort_index().ffill()
        joined = joined.dropna(subset=[ticker_u, "BZ=F", "CL=F", spot_col, fwd_col], how="any")
        if joined.empty:
            raise RuntimeError("No overlapping joined frame for ticker research.")
        recent = joined.tail(lookback_days)

        line_data = self._normalized_series(
            recent,
            ticker=ticker_u,
            spot_col=spot_col,
            fwd_col=fwd_col,
        )

        validation = self._validation_stats(
            ticker=ticker_u,
            frame=joined,
            spot_col=spot_col,
            fwd_col=fwd_col,
            top_contracts=top_contracts,
            contract_delta_map=predictive.contract_delta_map,
        )

        hedge = self._shipping_hedge_stats(
            ticker=ticker_u,
            frame=joined,
            spot_col=spot_col,
            fwd_col=fwd_col,
        )

        out = {
            "ticker": ticker_u,
            "as_of": now,
            "series": line_data,
            "top_predictive_contracts": top_contracts[:10],
            "validation": validation,
            "hedge": hedge,
        }
        self._ticker_cache[key] = (now, out)
        return out

    def _ensure_predictive_cache(self, lookback_days: int, max_markets: int = 80) -> PredictiveCache:
        now = datetime.now(UTC).replace(tzinfo=None)
        if self._predictive_cache and now - self._predictive_cache.created_at <= timedelta(minutes=25):
            return self._predictive_cache

        # Include commodity reference tickers so contracts can find non-oil best_targets
        _commodity_refs = [
            "BZ=F", "CL=F", "BDRY", "BOAT", "SEA",
            "GC=F", "SI=F", "HG=F", "PL=F", "PA=F",  # precious + base metals futures
            "URA", "LIT", "COPX", "REMX",              # commodity ETFs
            "ALI=F",                                      # aluminum futures
        ]
        target_cols = self.settings.universe + _commodity_refs
        prices = self._download_close_frame(target_cols, period="5y")
        returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna(how="all")
        returns = returns.tail(max(lookback_days + 100, 520))
        if returns.empty:
            raise RuntimeError("Unable to build return frame for predictive contracts.")

        markets = self._candidate_markets(max_markets=max_markets)
        if not markets:
            raise RuntimeError("No candidate prediction contracts found.")
        max_volume = max(float(x.get("volume", 1.0) or 1.0) for x in markets)

        contracts: list[dict[str, Any]] = []
        delta_map: dict[str, pd.Series] = {}
        now_date = now.date()

        for market in markets:
            market_id = str(market.get("id"))
            question = str(market.get("question", ""))
            token_id = self._yes_token_id(market)
            if not token_id:
                continue
            hist = self.prediction.polymarket._history_for_token(token_id)
            if hist.empty:
                continue
            prob = hist.reindex(returns.index).ffill().dropna()
            delta = prob.diff().replace([np.inf, -np.inf], np.nan).dropna()
            if len(delta) < 45:
                continue
            overlap_ratio = min(1.0, len(delta) / max(lookback_days, 1))
            staleness_days = int((now_date - prob.index.max().date()).days)
            staleness_penalty = math.exp(-max(staleness_days, 0) / 90.0)
            volume = float(market.get("volume", 0.0) or 0.0)
            liquidity_score = math.log1p(max(1.0, volume)) / math.log1p(max(2.0, max_volume))
            liquidity_score = max(0.05, min(1.0, liquidity_score))

            best_target: str | None = None
            best_corr = 0.0
            best_lag = 0
            for target in returns.columns:
                target_ret = returns[target]
                corr, lag = self._best_lag_corr(delta, target_ret, max_lag=7)
                if corr is None:
                    continue
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_target = target
                    best_lag = lag
            if best_target is None:
                continue

            predictive_score = abs(best_corr) * (0.35 + 0.65 * liquidity_score) * overlap_ratio * staleness_penalty
            category = self._market_category(question)
            record = {
                "market_id": market_id,
                "question": question,
                "category": category,
                "best_target": best_target,
                "lead_days": int(best_lag),
                "correlation": float(best_corr),
                "liquidity_score": float(liquidity_score),
                "staleness_days": staleness_days,
                "predictive_score": float(predictive_score),
            }
            contracts.append(record)
            delta_map[market_id] = delta

        contracts.sort(key=lambda x: abs(x["predictive_score"]), reverse=True)
        self._predictive_cache = PredictiveCache(
            created_at=now,
            lookback_days=lookback_days,
            contracts=contracts,
            contract_delta_map=delta_map,
            returns=returns,
        )
        return self._predictive_cache

    def _validation_stats(
        self,
        ticker: str,
        frame: pd.DataFrame,
        spot_col: str,
        fwd_col: str,
        top_contracts: list[dict[str, Any]],
        contract_delta_map: dict[str, pd.Series],
    ) -> dict[str, float]:
        px = frame[ticker].dropna()
        y = px.shift(-20) / px - 1.0

        base = pd.DataFrame(index=frame.index)
        for col in ["BZ=F", "CL=F", spot_col, fwd_col]:
            if col in frame.columns:
                ret = frame[col].pct_change(fill_method=None)
                base[f"ret_{col}"] = ret
                base[f"ret2_{col}"] = ret.diff()
                base[f"sq_{col}"] = ret * ret
        for ev in ["hormuz_closure", "red_sea_disruption", "oil_above_100"]:
            if ev in frame.columns:
                d = frame[ev].diff()
                base[f"d_{ev}"] = d
                base[f"d2_{ev}"] = d.diff()
                base[f"sq_{ev}"] = d * d
        base = base.replace([np.inf, -np.inf], np.nan)

        enriched = base.copy()
        for contract in top_contracts[:6]:
            cid = contract["market_id"]
            delta = contract_delta_map.get(cid)
            if delta is None or delta.empty:
                continue
            lag = int(contract.get("lead_days", 0))
            shifted = delta.shift(lag)
            enriched[f"pm_{cid}"] = shifted
            enriched[f"pm2_{cid}"] = shifted.diff()
            enriched[f"pm_sq_{cid}"] = shifted * shifted

        baseline_stats = self._walkforward_stats(y, base)
        enriched_stats = self._walkforward_stats(y, enriched)
        spot = float(px.iloc[-1])
        fair = float(spot * (1.0 + enriched_stats["latest_pred"]))

        return {
            "baseline_hit_rate": float(baseline_stats["hit_rate"]),
            "enriched_hit_rate": float(enriched_stats["hit_rate"]),
            "baseline_mae": float(baseline_stats["mae"]),
            "enriched_mae": float(enriched_stats["mae"]),
            "enriched_expected_return_20d": float(enriched_stats["latest_pred"]),
            "fair_value_price": fair,
            "spot_price": spot,
        }

    def _walkforward_stats(self, y: pd.Series, X: pd.DataFrame, lookback: int = 120) -> dict[str, float]:
        df = y.rename("y").to_frame().join(X, how="inner").dropna(how="any")
        min_needed = 50
        if len(df) < min_needed:
            return {"hit_rate": 0.0, "mae": 0.0, "latest_pred": 0.0}
        lookback = max(50, min(lookback, len(df) - 20))

        feat_cols = [c for c in df.columns if c != "y"]
        preds: list[float] = []
        actuals: list[float] = []
        for i in range(lookback, len(df)):
            train = df.iloc[i - lookback : i]
            x_train = train[feat_cols].values
            y_train = train["y"].values
            x_aug = np.c_[np.ones(len(x_train)), x_train]
            beta, *_ = np.linalg.lstsq(x_aug, y_train, rcond=None)
            x_now = df.iloc[i][feat_cols].values
            pred = float(np.r_[1.0, x_now] @ beta)
            preds.append(pred)
            actuals.append(float(df.iloc[i]["y"]))
        if not preds:
            return {"hit_rate": 0.0, "mae": 0.0, "latest_pred": 0.0}
        pred_arr = np.array(preds)
        act_arr = np.array(actuals)
        hit = float(np.mean(np.sign(pred_arr) == np.sign(act_arr)))
        mae = float(np.mean(np.abs(pred_arr - act_arr)))
        return {"hit_rate": hit, "mae": mae, "latest_pred": float(pred_arr[-1])}

    def _shipping_hedge_stats(
        self,
        ticker: str,
        frame: pd.DataFrame,
        spot_col: str,
        fwd_col: str,
    ) -> dict[str, Any]:
        spot = frame[spot_col].dropna()
        fwd = frame[fwd_col].dropna()
        joined = pd.concat([spot, fwd], axis=1).dropna(how="any")
        joined.columns = ["spot", "fwd"]
        if joined.empty:
            return {
                "spot_proxy": spot_col,
                "forward_proxy": fwd_col,
                "current_basis_pct": 0.0,
                "one_month_expected_basis_pct": 0.0,
                "hedge_beta_to_forward": 0.0,
            }

        basis = joined["fwd"] / joined["spot"] - 1.0
        current_basis = float(basis.iloc[-1])
        month_expectation = float(basis.tail(22).mean()) if len(basis) >= 5 else current_basis

        ticker_ret = frame[ticker].pct_change(fill_method=None)
        fwd_ret = frame[fwd_col].pct_change(fill_method=None)
        pair = pd.concat([ticker_ret, fwd_ret], axis=1).dropna(how="any")
        beta = 0.0
        if len(pair) >= 40 and pair.iloc[:, 1].var() > 0:
            beta = float(pair.iloc[:, 0].cov(pair.iloc[:, 1]) / pair.iloc[:, 1].var())

        return {
            "spot_proxy": spot_col,
            "forward_proxy": fwd_col,
            "current_basis_pct": current_basis,
            "one_month_expected_basis_pct": month_expectation,
            "hedge_beta_to_forward": beta,
        }

    def _normalized_series(self, frame: pd.DataFrame, ticker: str, spot_col: str, fwd_col: str) -> list[dict[str, Any]]:
        def _norm(series: pd.Series) -> pd.Series:
            clean = series.dropna()
            if clean.empty:
                return pd.Series(index=series.index, data=np.nan)
            base = float(clean.iloc[0]) if float(clean.iloc[0]) != 0 else 1.0
            return series / base

        stock = _norm(frame[ticker])
        brent = _norm(frame["BZ=F"])
        wti = _norm(frame["CL=F"])
        ship_spot = _norm(frame[spot_col])
        ship_fwd = _norm(frame[fwd_col])

        event_h = 1.0 + (frame.get("hormuz_closure", pd.Series(index=frame.index, data=0.5)) - 0.5)
        event_r = 1.0 + (frame.get("red_sea_disruption", pd.Series(index=frame.index, data=0.5)) - 0.5)
        event_o = 1.0 + (frame.get("oil_above_100", pd.Series(index=frame.index, data=0.5)) - 0.5)

        out: list[dict[str, Any]] = []
        for idx in frame.index:
            vals = [
                stock.get(idx),
                brent.get(idx),
                wti.get(idx),
                ship_spot.get(idx),
                ship_fwd.get(idx),
                event_h.get(idx),
                event_r.get(idx),
                event_o.get(idx),
            ]
            if any(pd.isna(v) for v in vals):
                continue
            out.append(
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "stock": float(vals[0]),
                    "brent": float(vals[1]),
                    "wti": float(vals[2]),
                    "shipping_spot": float(vals[3]),
                    "shipping_fwd": float(vals[4]),
                    "event_hormuz": float(vals[5]),
                    "event_red_sea": float(vals[6]),
                    "event_oil_100": float(vals[7]),
                }
            )
        return out

    def _event_history(self, event_ids: list[str]) -> pd.DataFrame:
        series_map: dict[str, pd.Series] = {}
        now = datetime.now(UTC)
        start_ts = int((now - timedelta(days=1400)).timestamp())
        end_ts = int(now.timestamp())
        for event_id in event_ids:
            mapping = EVENT_MAPPINGS.get(event_id)
            if mapping is None:
                continue
            poly = self.prediction.polymarket.fetch_event_history(mapping)
            kalshi = self.prediction.kalshi.fetch_event_history(
                mapping,
                start_ts=start_ts,
                end_ts=end_ts,
                period_interval=1440,
            )
            cols = [s for s in [poly, kalshi] if s is not None and not s.empty]
            if not cols:
                continue
            merged = pd.concat(cols, axis=1).sort_index().ffill()
            series_map[event_id] = merged.mean(axis=1).clip(0.0, 1.0)
        if not series_map:
            return pd.DataFrame()
        out = pd.concat(series_map.values(), axis=1).sort_index()
        full_idx = pd.date_range(out.index.min(), out.index.max(), freq="D")
        out = out.reindex(full_idx).sort_index().ffill().fillna(0.5)
        return out

    def _candidate_markets(self, max_markets: int = 80) -> list[dict[str, Any]]:
        keywords = re.compile(
            r"oil|crude|wti|brent|hormuz|red sea|suez|iran|sanction|shipping|tanker|freight|opec"
            r"|natural gas|lng|recession|fed|inflation|rate cut|rate hike"
            r"|gold|silver|copper|iron ore|steel|aluminum|aluminium|nickel|zinc|tin|cobalt|mining"
            r"|uranium|nuclear|reactor|smr|enrichment"
            r"|lithium|battery|electric vehicle|\bev\b|tesla|charging"
            r"|fertilizer|potash|wheat|corn|agriculture|food|famine"
            r"|tariff|trade war|protectionism|import duty"
            r"|china gdp|china real estate|china property|evergrande|country garden"
            r"|rare earth|semiconductor|chip|silicon"
            r"|coal|carbon|emission|climate"
            r"|rubber|palm oil|soybean|sugar|cotton|cocoa"
            r"|india.*infrastructure|brics|commodity|supply chain|shortage"
            r"|palladium|platinum|pgm|rhodium"
            r"|\biron\b|\bore\b|bhp|rio tinto|vale"
            # Tail-risk geopolitical / policy events (high alpha)
            r"|taiwan|strait|blockade|invasion|china.*military"
            r"|ceasefire|ukraine.*peace|russia.*deal|minsk"
            r"|eskom|load.?shedding|south africa.*grid|south africa.*power"
            r"|nationali[zs]|chile.*lithium|sqm|albemarle"
            r"|indonesia.*nickel|nickel.*ban|mineral.*export.*ban"
            r"|permitting.*reform|nepa|mining.*permit|energy.*permit"
            r"|cbam|carbon.*border|border.*adjustment|eu.*carbon.*import"
            r"|australia.*china.*trade|australia.*ban.*lift"
            r"|recession.*20[2-3]|gdp.*contract|nber.*recession|economic.*downturn"
            r"|middle.?east.*war|israel.*iran|iran.*strike|hezbollah|escalat",
            re.I,
        )
        markets: list[dict[str, Any]] = []
        limit = 500
        for page in range(0, 35):
            r = requests.get(
                self.BASE_GAMMA,
                params={"limit": limit, "offset": page * limit, "active": "true"},
                timeout=30,
            )
            if r.status_code != 200:
                break
            arr = r.json()
            if not arr:
                break
            for m in arr:
                q = str(m.get("question", ""))
                if not keywords.search(q):
                    continue
                volume = float(m.get("volume", 0.0) or 0.0)
                if volume < 500.0:
                    continue
                if not self._yes_token_id(m):
                    continue
                markets.append(m)
        uniq: dict[str, dict[str, Any]] = {}
        for market in markets:
            mid = str(market.get("id"))
            prev = uniq.get(mid)
            if prev is None or float(market.get("volume", 0.0) or 0.0) > float(prev.get("volume", 0.0) or 0.0):
                uniq[mid] = market
        ranked = sorted(uniq.values(), key=lambda x: float(x.get("volume", 0.0) or 0.0), reverse=True)
        return ranked[:max_markets]

    @staticmethod
    def _market_category(question: str) -> str:
        q = question.lower()
        # Tail-risk geopolitical events first (highest alpha categories)
        if any(x in q for x in ["taiwan", "strait", "blockade", "china invad", "china military"]):
            return "geopolitical"
        if any(x in q for x in ["ceasefire", "ukraine peace", "ukraine deal", "minsk"]):
            return "geopolitical"
        if any(x in q for x in ["middle east war", "israel iran", "iran strike", "hezbollah", "israel war"]):
            return "geopolitical"
        if any(x in q for x in ["eskom", "load shedding", "south africa grid", "south africa power"]):
            return "geopolitical"
        if any(x in q for x in ["nationali", "chile lithium", "indonesia nickel", "nickel ban", "mineral ban"]):
            return "trade_policy"
        if any(x in q for x in ["cbam", "carbon border", "border adjustment"]):
            return "trade_policy"
        if any(x in q for x in ["permitting reform", "nepa", "mining permit", "energy permit"]):
            return "trade_policy"
        if any(x in q for x in ["recession", "gdp contract", "nber", "economic downturn"]):
            return "macro"
        if any(x in q for x in ["oil", "wti", "brent", "crude", "opec", "refinery", "gasoline", "natural gas", "lng"]):
            return "oil"
        if any(x in q for x in ["shipping", "tanker", "freight", "suez", "red sea", "hormuz", "panama canal"]):
            return "shipping"
        if any(x in q for x in ["gold", "silver", "platinum", "palladium", "pgm", "rhodium"]):
            return "precious_metals"
        if any(x in q for x in ["copper", "iron", "ore", "steel", "aluminum", "aluminium", "nickel", "zinc", "tin", "cobalt", "mining", "bhp", "rio tinto", "vale"]):
            return "base_metals"
        if any(x in q for x in ["uranium", "nuclear", "reactor", "smr", "enrichment"]):
            return "uranium"
        if any(x in q for x in ["lithium", "battery", "ev ", "electric vehicle", "charging", "tesla"]):
            return "energy_transition"
        if any(x in q for x in ["fertilizer", "potash", "wheat", "corn", "agriculture", "food", "soybean", "sugar", "cotton", "cocoa", "famine", "palm oil"]):
            return "agriculture"
        if any(x in q for x in ["coal", "carbon", "emission", "climate"]):
            return "coal_carbon"
        if any(x in q for x in ["rare earth", "semiconductor", "chip", "silicon"]):
            return "rare_earths_tech"
        if any(x in q for x in ["tariff", "trade war", "protectionism", "import duty", "sanction"]):
            return "trade_policy"
        if any(x in q for x in ["china", "india", "infrastructure", "brics", "stimulus"]):
            return "emerging_markets"
        if any(x in q for x in ["fed", "rate", "recession", "inflation"]):
            return "macro"
        return "geopolitical"

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

    @staticmethod
    def _download_close_frame(tickers: list[str], period: str = "5y") -> pd.DataFrame:
        tickers = list(dict.fromkeys(tickers))
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
            for ticker in tickers:
                if (ticker, "Close") in raw.columns:
                    close[ticker] = raw[(ticker, "Close")]
        else:
            if len(tickers) == 1 and "Close" in raw.columns:
                close[tickers[0]] = raw["Close"]
        close.index = pd.to_datetime(close.index).tz_localize(None)
        return close.sort_index().dropna(how="all")

    @staticmethod
    def _best_lag_corr(left: pd.Series, right: pd.Series, max_lag: int = 7) -> tuple[float | None, int]:
        best_corr: float | None = None
        best_lag = 0
        for lag in range(max_lag + 1):
            joined = pd.concat([left.shift(lag), right], axis=1).dropna(how="any")
            if len(joined) < 45:
                continue
            corr = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
            if np.isnan(corr):
                continue
            if best_corr is None or abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag
        return best_corr, best_lag
