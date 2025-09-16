from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf


UNIVERSE = ["TNK", "INSW", "STNG", "SBLK", "DHT", "FRO", "NAT", "SM", "MTDR", "RRC", "MUR", "AR"]
THEMES = {
    "hormuz_closure": r"hormuz|strait of hormuz",
    "red_sea_suez": r"red sea|suez",
    "iran_military": r"iran military response|iran response",
    "us_action_iran": r"u\.s\. military action against iran|us military action against iran",
    "iran_nuclear_deal": r"us-iran nuclear deal|iran nuclear deal",
    "oil_above_80": r"crude oil.*\$80|wti.*80|brent.*80",
    "oil_above_90": r"crude oil.*\$90|wti.*90|brent.*90",
    "oil_above_100": r"crude oil.*\$100|wti.*100|brent.*100",
    "oil_above_120": r"crude oil.*\$120|wti.*120|brent.*120",
    "us_recession": r"u\.s\. recession",
    "fed_cut": r"fed.*(decrease|cut)|rate cut",
    "russia_ukraine_ceasefire": r"russia-ukraine ceasefire|ukraine ceasefire",
    "china_taiwan_conflict": r"china.*taiwan.*(invasion|war|conflict)",
    "opec_output": r"opec",
    "lng_natgas": r"natural gas|lng",
}
BASE_GAMMA = "https://gamma-api.polymarket.com/markets"
BASE_CLOB = "https://clob.polymarket.com/prices-history"


def load_markets(max_pages: int = 50, limit: int = 500) -> list[dict]:
    out: list[dict] = []
    for page in range(max_pages):
        r = requests.get(BASE_GAMMA, params={"limit": limit, "offset": page * limit}, timeout=45)
        if r.status_code != 200:
            break
        arr = r.json()
        if not arr:
            break
        out.extend(arr)
    return out


def yes_token_id(market: dict) -> str | None:
    try:
        raw_tokens = market.get("clobTokenIds")
        raw_outcomes = market.get("outcomes")
        tokens = json.loads(raw_tokens) if isinstance(raw_tokens, str) else list(raw_tokens or [])
        outcomes = json.loads(raw_outcomes) if isinstance(raw_outcomes, str) else list(raw_outcomes or [])
        for i, outcome in enumerate(outcomes):
            if str(outcome).strip().lower() == "yes" and i < len(tokens):
                return str(tokens[i])
        return str(tokens[0]) if tokens else None
    except Exception:
        return None


def history_for_token(token_id: str) -> pd.Series:
    r = requests.get(BASE_CLOB, params={"market": token_id, "interval": "max", "fidelity": 1440}, timeout=45)
    if r.status_code != 200:
        return pd.Series(dtype=float)
    payload = r.json()
    history = payload.get("history", [])
    if not history:
        return pd.Series(dtype=float)
    idx = [pd.to_datetime(int(x["t"]), unit="s", utc=True).tz_convert(None).normalize() for x in history]
    vals = [max(0.0, min(1.0, float(x["p"]))) for x in history]
    s = pd.Series(vals, index=idx).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s


def download_close_frame(tickers: list[str], period: str = "5y") -> pd.DataFrame:
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
    out = pd.DataFrame(index=raw.index)
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            if (t, "Close") in raw.columns:
                out[t] = raw[(t, "Close")]
    else:
        if len(tickers) == 1 and "Close" in raw.columns:
            out[tickers[0]] = raw["Close"]
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out.dropna(how="all").sort_index()


def run() -> dict[str, object]:
    markets = load_markets()
    selected: dict[str, dict[str, object]] = {}
    series_map: dict[str, pd.Series] = {}
    for theme, pattern in THEMES.items():
        rx = re.compile(pattern, re.I)
        candidates: list[tuple[float, dict, str]] = []
        for market in markets:
            question = str(market.get("question", ""))
            if not rx.search(question):
                continue
            token = yes_token_id(market)
            if not token:
                continue
            volume = float(market.get("volume") or 0.0)
            candidates.append((volume, market, token))
        candidates.sort(key=lambda x: x[0], reverse=True)
        best = None
        best_series = pd.Series(dtype=float)
        for vol, market, token in candidates[:120]:
            s = history_for_token(token)
            if len(s) >= 20:
                best = (vol, market, token)
                best_series = s
                break
        if best is None:
            continue
        vol, market, _ = best
        selected[theme] = {
            "question": market.get("question"),
            "volume": vol,
            "history_days": int(len(best_series)),
        }
        series_map[theme] = best_series.rename(theme)

    if len(series_map) < 10:
        raise RuntimeError(
            f"Only {len(series_map)} qualifying contracts were found. "
            "Broaden theme patterns or lower history requirements."
        )

    probs = pd.concat(series_map.values(), axis=1).sort_index()
    full_idx = pd.date_range(probs.index.min(), probs.index.max(), freq="D")
    probs = probs.reindex(full_idx).ffill().fillna(0.5)

    prices = download_close_frame(UNIVERSE, period="5y")
    joined = probs.join(prices, how="inner").dropna(how="any")

    rets = joined[UNIVERSE].pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="any")
    feature_cols = list(series_map.keys()) + [f"d_{k}" for k in series_map.keys()]
    features = joined[list(series_map.keys())].join(joined[list(series_map.keys())].diff().add_prefix("d_")).dropna(how="any")
    rets = rets.join(features, how="inner").dropna(how="any")

    pred = pd.DataFrame(index=rets.index, columns=UNIVERSE, dtype=float)
    roll = 90
    for ticker in UNIVERSE:
        if ticker not in rets.columns:
            continue
        for i in range(roll, len(rets) - 1):
            x = rets.iloc[i - roll : i][feature_cols].values
            y = rets.iloc[i - roll + 1 : i + 1][ticker].values
            x_aug = np.c_[np.ones(len(x)), x]
            beta, *_ = np.linalg.lstsq(x_aug, y, rcond=None)
            x_now = np.r_[1.0, rets.iloc[i][feature_cols].values]
            pred.iloc[i + 1, pred.columns.get_loc(ticker)] = float(x_now @ beta)

    hold_days = 20
    cost = 0.008
    trades: list[dict[str, object]] = []
    for dt in pred.dropna(how="all").index:
        loc = rets.index.get_loc(dt)
        if loc + hold_days >= len(rets):
            continue
        day = pred.loc[dt].dropna().sort_values(ascending=False).head(3)
        for ticker, value in day.items():
            pred_h = float(value) * hold_days
            if pred_h < 0.02:
                continue
            fwd = rets.iloc[loc + 1 : loc + 1 + hold_days][ticker].values
            if len(fwd) < hold_days:
                continue
            gross = float(np.prod(1.0 + fwd) - 1.0)
            net = gross - cost
            trades.append(
                {
                    "date": str(dt.date()),
                    "ticker": ticker,
                    "pred_horizon_return": pred_h,
                    "gross_return": gross,
                    "net_return": net,
                    "success": int(net > 0),
                }
            )
    if not trades:
        raise RuntimeError("No trades were generated from expanded-contract model.")

    tr = pd.DataFrame(trades)
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "n_contracts": len(series_map),
        "contracts": selected,
        "window": [str(tr["date"].min()), str(tr["date"].max())],
        "trade_count": int(len(tr)),
        "hit_rate": float(tr["success"].mean()),
        "avg_net_return": float(tr["net_return"].mean()),
        "median_net_return": float(tr["net_return"].median()),
        "best_tickers": tr.groupby("ticker")["net_return"].mean().sort_values(ascending=False).head(8).to_dict(),
    }

    out_dir = Path(__file__).resolve().parent.parent / "analysis_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "expanded_contract_backtest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    tr.to_csv(out_dir / "expanded_contract_backtest_trades.csv", index=False)
    return summary


def main() -> None:
    summary = run()
    print(json.dumps(summary, indent=2))
    print("\nWrote analysis_output/expanded_contract_backtest_summary.json and analysis_output/expanded_contract_backtest_trades.csv")


if __name__ == "__main__":
    main()
