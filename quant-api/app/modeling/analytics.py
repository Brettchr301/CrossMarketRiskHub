from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from app.config import get_settings
from app.providers.real_prediction import EVENT_MAPPINGS, RealPredictionProvider


@dataclass(slots=True)
class CorrelationAnalyticsSnapshot:
    as_of: datetime
    lookback_days: int
    tickers: list[dict[str, Any]]


class CorrelationAnalyticsService:
    _cache: dict[int, tuple[datetime, CorrelationAnalyticsSnapshot]] = {}

    def __init__(self) -> None:
        self.settings = get_settings()
        self.prediction = RealPredictionProvider()
        self.factor_tickers = ["BZ=F", "CL=F", "BDRY"]
        self.factor_alias = {"BZ=F": "Brent", "CL=F": "WTI", "BDRY": "ShippingETF"}

    def build_snapshot(self, lookback_days: int = 260) -> CorrelationAnalyticsSnapshot:
        now = datetime.now(UTC).replace(tzinfo=None)
        cached = self._cache.get(lookback_days)
        if cached and (now - cached[0]) <= timedelta(minutes=20):
            return cached[1]

        probs = self._event_probability_history()
        closes = self._download_closes(self.settings.universe + self.factor_tickers, period="5y")
        if probs.empty or closes.empty:
            raise RuntimeError("Insufficient data to build correlation snapshot.")

        frame = probs.join(closes, how="inner").sort_index().dropna(how="any")
        if len(frame) < 60:
            raise RuntimeError(f"Insufficient overlap for correlations: {len(frame)} rows.")

        event_cols = list(probs.columns)
        price_cols = [c for c in closes.columns if c in frame.columns]
        returns = frame[price_cols].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        event_delta = frame[event_cols].diff().replace([np.inf, -np.inf], np.nan).fillna(0.0)

        combined = returns.join(event_delta, how="inner").dropna(how="all")
        if combined.empty:
            raise RuntimeError("No aligned return/probability data after processing.")
        recent = combined.tail(lookback_days)

        ticker_stats: list[dict[str, Any]] = []
        for ticker in self.settings.universe:
            if ticker not in recent.columns:
                continue
            ret = recent[ticker].dropna()
            if len(ret) < 40:
                continue

            corr_brent = self._corr(ret, recent.get("BZ=F"))
            corr_wti = self._corr(ret, recent.get("CL=F"))
            corr_ship = self._corr(ret, recent.get("BDRY"))

            drivers: list[dict[str, Any]] = []
            for fac in self.factor_tickers:
                c = self._corr(ret, recent.get(fac))
                if c is None:
                    continue
                drivers.append(
                    {
                        "name": self.factor_alias.get(fac, fac),
                        "source": "factor",
                        "correlation": float(c),
                        "lag_days": 0,
                    }
                )
            for event in event_cols:
                series = event_delta[event]
                best = self._best_lag_corr(ret, series, max_lag=5)
                if best is None:
                    continue
                corr, lag = best
                drivers.append(
                    {
                        "name": event,
                        "source": "event",
                        "correlation": float(corr),
                        "lag_days": int(lag),
                    }
                )

            drivers.sort(key=lambda d: abs(d["correlation"]), reverse=True)
            ticker_stats.append(
                {
                    "ticker": ticker,
                    "sample_size": int(len(ret)),
                    "corr_brent": corr_brent,
                    "corr_wti": corr_wti,
                    "corr_shipping": corr_ship,
                    "top_drivers": drivers[:8],
                }
            )

        snapshot = CorrelationAnalyticsSnapshot(
            as_of=now,
            lookback_days=lookback_days,
            tickers=sorted(ticker_stats, key=lambda x: x["ticker"]),
        )
        self._cache[lookback_days] = (now, snapshot)
        return snapshot

    def _event_probability_history(self) -> pd.DataFrame:
        now = datetime.now(UTC)
        start_ts = int((now - timedelta(days=1400)).timestamp())
        end_ts = int(now.timestamp())
        series_map: dict[str, pd.Series] = {}
        for event_id, mapping in EVENT_MAPPINGS.items():
            poly = self.prediction.polymarket.fetch_event_history(mapping)
            kalshi = self.prediction.kalshi.fetch_event_history(
                mapping, start_ts=start_ts, end_ts=end_ts, period_interval=1440
            )
            cols = [s for s in [poly, kalshi] if s is not None and not s.empty]
            if not cols:
                continue
            merged = pd.concat(cols, axis=1).sort_index().ffill()
            series_map[event_id] = merged.mean(axis=1).clip(0.0, 1.0).rename(event_id)
        if not series_map:
            return pd.DataFrame()
        out = pd.concat(series_map.values(), axis=1).sort_index()
        full_idx = pd.date_range(out.index.min(), out.index.max(), freq="D")
        out = out.reindex(full_idx).ffill().fillna(0.5)
        return out

    @staticmethod
    def _download_closes(tickers: list[str], period: str = "5y") -> pd.DataFrame:
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
    def _corr(left: pd.Series, right: pd.Series | None) -> float | None:
        if right is None:
            return None
        joined = pd.concat([left, right], axis=1).dropna(how="any")
        if len(joined) < 30:
            return None
        val = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
        if np.isnan(val):
            return None
        return val

    def _best_lag_corr(self, left: pd.Series, right: pd.Series, max_lag: int = 5) -> tuple[float, int] | None:
        best: tuple[float, int] | None = None
        for lag in range(max_lag + 1):
            shifted = right.shift(lag)
            c = self._corr(left, shifted)
            if c is None:
                continue
            if best is None or abs(c) > abs(best[0]):
                best = (c, lag)
        return best
