from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from html import unescape
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import yfinance as yf


UNIVERSE = ["TNK", "INSW", "STNG", "SBLK", "DHT", "FRO", "NAT", "SM", "MTDR", "RRC", "MUR", "AR"]
USER_AGENT = "Quant Research (research@example.com)"

THEME_PATTERNS = {
    "hormuz_closure": r"hormuz|strait of hormuz",
    "red_sea_suez": r"red sea|suez",
    "iran_military": r"iran military response|iran response",
    "us_action_iran": r"u\.s\. military action against iran|us military action against iran",
    "iran_nuclear_deal": r"us-iran nuclear deal|iran nuclear deal",
    "oil_above_100": r"crude oil.*\$100|wti.*100|brent.*100",
    "oil_above_120": r"crude oil.*\$120|wti.*120|brent.*120",
    "us_recession": r"u\.s\. recession",
    "fed_cut": r"fed.*(decrease|cut)|rate cut",
    "russia_ukraine_ceasefire": r"russia-ukraine ceasefire|ukraine ceasefire",
    "lng_natgas": r"natural gas|lng",
}

THEME_KEYWORDS = {
    "hormuz_closure": [r"hormuz", r"strait"],
    "red_sea_suez": [r"red sea", r"suez", r"shipping route", r"canal"],
    "iran_military": [r"iran", r"middle east", r"military", r"geopolitical"],
    "us_action_iran": [r"u\.s\.", r"military action", r"iran", r"sanction"],
    "iran_nuclear_deal": [r"nuclear", r"iran", r"sanction", r"export restriction"],
    "oil_above_100": [r"oil price", r"crude", r"commodity price", r"realized price"],
    "oil_above_120": [r"oil price", r"crude", r"price volatility", r"inflation"],
    "us_recession": [r"recession", r"demand", r"economic slowdown", r"consumer demand"],
    "fed_cut": [r"interest rate", r"fed", r"monetary", r"capital markets"],
    "russia_ukraine_ceasefire": [r"russia", r"ukraine", r"war", r"sanctions"],
    "lng_natgas": [r"natural gas", r"lng", r"gas price", r"pipeline"],
}

BASE_GAMMA = "https://gamma-api.polymarket.com/markets"
BASE_CLOB = "https://clob.polymarket.com/prices-history"


@dataclass(slots=True)
class TickerRiskInput:
    ticker: str
    cik: str | None
    tenk_url: str | None
    theme_scores: dict[str, int]
    top_themes: list[str]


def _strip_html(raw_html: str) -> str:
    cleaned = re.sub(r"<script[^>]*>.*?</script>", " ", raw_html, flags=re.I | re.S)
    cleaned = re.sub(r"<style[^>]*>.*?</style>", " ", cleaned, flags=re.I | re.S)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _extract_item_1a_risk_factors(text: str) -> str:
    low = text.lower()
    starts = [m.start() for m in re.finditer(r"item\s*1a\.?\s*risk factors", low)]
    ends_1b = [m.start() for m in re.finditer(r"item\s*1b\.?", low)]
    ends_2 = [m.start() for m in re.finditer(r"item\s*2\.?", low)]
    ends = sorted(set(ends_1b + ends_2))
    if not starts:
        return text[:0]
    # Prefer later starts to skip table of contents.
    for start in reversed(starts):
        end = next((x for x in ends if x > start + 1200), None)
        if end is not None and end - start > 2500:
            return text[start:end]
    start = starts[-1]
    return text[start : start + 25000]


def _sec_headers() -> dict[str, str]:
    return {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}


def _load_cik_map() -> dict[str, str]:
    r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=_sec_headers(), timeout=30)
    r.raise_for_status()
    payload = r.json()
    out: dict[str, str] = {}
    for _, row in payload.items():
        t = str(row.get("ticker", "")).upper()
        cik = str(int(row.get("cik_str"))).zfill(10)
        out[t] = cik
    return out


def _latest_10k_url(cik: str) -> str | None:
    r = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=_sec_headers(), timeout=30)
    if r.status_code != 200:
        return None
    payload = r.json()
    recent = payload.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    acc = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])
    for form, accession, doc in zip(forms, acc, docs):
        form_u = str(form).upper()
        if form_u.startswith("10-K") or form_u.startswith("20-F"):
            acc_clean = str(accession).replace("-", "")
            return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{doc}"
    return None


def _theme_scores_from_risk_text(risk_text: str) -> dict[str, int]:
    low = risk_text.lower()
    scores: dict[str, int] = {}
    for theme, keywords in THEME_KEYWORDS.items():
        score = 0
        for kw in keywords:
            score += len(re.findall(kw, low, flags=re.I))
        scores[theme] = score
    return scores


def _infer_sector_defaults(ticker: str) -> list[str]:
    shipping = {"TNK", "INSW", "STNG", "SBLK", "DHT", "FRO", "NAT"}
    if ticker in shipping:
        return ["red_sea_suez", "hormuz_closure", "russia_ukraine_ceasefire", "oil_above_100", "lng_natgas"]
    return ["oil_above_100", "oil_above_120", "us_recession", "fed_cut", "iran_military"]


def _collect_ticker_risk_inputs(tickers: Iterable[str]) -> list[TickerRiskInput]:
    cik_map = _load_cik_map()
    out: list[TickerRiskInput] = []
    for ticker in tickers:
        t = ticker.upper()
        cik = cik_map.get(t)
        tenk_url = _latest_10k_url(cik) if cik else None
        theme_scores = {k: 0 for k in THEME_KEYWORDS.keys()}
        if tenk_url:
            try:
                html = requests.get(tenk_url, headers=_sec_headers(), timeout=45).text
                plain = _strip_html(html)
                risk_section = _extract_item_1a_risk_factors(plain)
                theme_scores = _theme_scores_from_risk_text(risk_section)
            except Exception:
                pass
        ranked = sorted(theme_scores.items(), key=lambda kv: kv[1], reverse=True)
        top_themes = [k for k, v in ranked if v > 0][:5]
        if not top_themes:
            top_themes = _infer_sector_defaults(t)[:5]
        out.append(
            TickerRiskInput(
                ticker=t,
                cik=cik,
                tenk_url=tenk_url,
                theme_scores=theme_scores,
                top_themes=top_themes,
            )
        )
    return out


def _load_polymarket_markets(max_pages: int = 40, limit: int = 500) -> list[dict]:
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


def _yes_token_id(market: dict) -> str | None:
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


def _history_for_token(token_id: str) -> pd.Series:
    r = requests.get(BASE_CLOB, params={"market": token_id, "interval": "max", "fidelity": 1440}, timeout=45)
    if r.status_code != 200:
        return pd.Series(dtype=float)
    history = r.json().get("history", [])
    if not history:
        return pd.Series(dtype=float)
    idx = [pd.to_datetime(int(x["t"]), unit="s", utc=True).tz_convert(None).normalize() for x in history]
    vals = [max(0.0, min(1.0, float(x["p"]))) for x in history]
    s = pd.Series(vals, index=idx).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s


def _build_theme_probability_frame(min_history_days: int = 20) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    markets = _load_polymarket_markets()
    series_map: dict[str, pd.Series] = {}
    meta: dict[str, dict[str, object]] = {}
    for theme, pattern in THEME_PATTERNS.items():
        rx = re.compile(pattern, re.I)
        candidates: list[tuple[float, dict, str]] = []
        for market in markets:
            question = str(market.get("question", ""))
            if not rx.search(question):
                continue
            token = _yes_token_id(market)
            if not token:
                continue
            volume = float(market.get("volume") or 0.0)
            candidates.append((volume, market, token))
        candidates.sort(key=lambda x: x[0], reverse=True)
        chosen = None
        chosen_series = pd.Series(dtype=float)
        for vol, market, token in candidates[:120]:
            s = _history_for_token(token)
            if len(s) >= min_history_days:
                chosen = (vol, market, token)
                chosen_series = s
                break
        if chosen is None:
            continue
        vol, market, token = chosen
        series_map[theme] = chosen_series.rename(theme)
        meta[theme] = {
            "question": market.get("question"),
            "volume": vol,
            "history_days": int(len(chosen_series)),
            "token_id": token,
        }
    if len(series_map) < 6:
        raise RuntimeError(f"Insufficient prediction-market theme coverage: {len(series_map)} themes.")
    df = pd.concat(series_map.values(), axis=1).sort_index()
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_idx).ffill().fillna(0.5)
    return df, meta


def _download_equity_close(tickers: list[str], period: str = "5y") -> pd.DataFrame:
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


def _fit_walkforward_model(
    price_series: pd.Series,
    probs: pd.DataFrame,
    themes: list[str],
    horizon_days: int = 20,
    lookback_days: int = 180,
) -> dict[str, object]:
    p = price_series.dropna().copy()
    y = p.shift(-horizon_days) / p - 1.0

    x_levels = probs[themes].copy()
    roll_mean = x_levels.rolling(90, min_periods=20).mean()
    centered = (x_levels - roll_mean).add_prefix("c_")
    deltas = x_levels.diff().add_prefix("d_")
    X = centered.join(deltas).fillna(0.0)

    df = y.rename("y").to_frame().join(X, how="inner").dropna(how="any")
    if len(df) < max(lookback_days + 20, 180):
        return {"valid": False, "reason": f"insufficient_data_{len(df)}"}

    feature_cols = [c for c in df.columns if c != "y"]
    preds: list[float] = []
    actuals: list[float] = []
    pred_dates: list[pd.Timestamp] = []

    for i in range(lookback_days, len(df)):
        train = df.iloc[i - lookback_days : i]
        x_train = train[feature_cols].values
        y_train = train["y"].values
        x_aug = np.c_[np.ones(len(x_train)), x_train]
        beta, *_ = np.linalg.lstsq(x_aug, y_train, rcond=None)
        x_now = df.iloc[i][feature_cols].values
        pred = float(np.r_[1.0, x_now] @ beta)
        preds.append(pred)
        actuals.append(float(df.iloc[i]["y"]))
        pred_dates.append(df.index[i])

    pred_arr = np.array(preds, dtype=float)
    act_arr = np.array(actuals, dtype=float)
    if len(pred_arr) == 0:
        return {"valid": False, "reason": "no_predictions"}

    hit_rate = float(np.mean(np.sign(pred_arr) == np.sign(act_arr)))
    mae = float(np.mean(np.abs(pred_arr - act_arr)))
    corr = float(np.corrcoef(pred_arr, act_arr)[0, 1]) if len(pred_arr) > 2 else 0.0

    # Current prediction from the latest available feature row.
    latest_features = X.dropna(how="any")
    if latest_features.empty:
        return {"valid": False, "reason": "no_latest_features"}
    last_feat_dt = latest_features.index[-1]
    train = df.iloc[-lookback_days:]
    x_train = train[feature_cols].values
    y_train = train["y"].values
    x_aug = np.c_[np.ones(len(x_train)), x_train]
    beta, *_ = np.linalg.lstsq(x_aug, y_train, rcond=None)
    x_cur = latest_features.loc[last_feat_dt, feature_cols].values
    pred_current = float(np.r_[1.0, x_cur] @ beta)

    spot = float(price_series.dropna().iloc[-1])
    fair_price = float(spot * (1.0 + pred_current))

    contributions: dict[str, float] = {}
    coef_map = dict(zip(feature_cols, beta[1:]))
    for theme in themes:
        contrib = 0.0
        for fcol in (f"c_{theme}", f"d_{theme}"):
            if fcol in coef_map and fcol in latest_features.columns:
                contrib += float(coef_map[fcol] * latest_features.loc[last_feat_dt, fcol])
        contributions[theme] = contrib

    return {
        "valid": True,
        "obs": int(len(pred_arr)),
        "hit_rate": hit_rate,
        "mae": mae,
        "corr": corr,
        "predicted_20d_return": pred_current,
        "spot": spot,
        "fair_price": fair_price,
        "edge_pct": (fair_price - spot) / max(spot, 1e-8),
        "attribution": contributions,
        "last_feature_date": str(last_feat_dt.date()),
    }


def run() -> dict[str, object]:
    probs, contract_meta = _build_theme_probability_frame(min_history_days=20)
    close = _download_equity_close(UNIVERSE, period="5y")
    risk_inputs = _collect_ticker_risk_inputs(UNIVERSE)

    ticker_rows: list[dict[str, object]] = []
    mapped_hits: list[float] = []
    mapped_mae: list[float] = []
    baseline_hits: list[float] = []
    baseline_mae: list[float] = []

    baseline_themes = [t for t in ["hormuz_closure", "red_sea_suez", "oil_above_100", "iran_nuclear_deal"] if t in probs.columns]

    for risk_input in risk_inputs:
        ticker = risk_input.ticker
        if ticker not in close.columns:
            continue
        themes = [t for t in risk_input.top_themes if t in probs.columns]
        if len(themes) < 3:
            themes = [t for t in _infer_sector_defaults(ticker) if t in probs.columns][:5]

        mapped = _fit_walkforward_model(close[ticker], probs, themes)
        baseline = _fit_walkforward_model(close[ticker], probs, baseline_themes) if len(baseline_themes) >= 3 else {"valid": False}

        if mapped.get("valid"):
            mapped_hits.append(float(mapped["hit_rate"]))
            mapped_mae.append(float(mapped["mae"]))
        if baseline.get("valid"):
            baseline_hits.append(float(baseline["hit_rate"]))
            baseline_mae.append(float(baseline["mae"]))

        ticker_rows.append(
            {
                "ticker": ticker,
                "tenk_url": risk_input.tenk_url,
                "top_themes": themes,
                "theme_scores": risk_input.theme_scores,
                "mapped_model": mapped,
                "baseline_model": baseline,
            }
        )

    winners = []
    for row in ticker_rows:
        mm = row.get("mapped_model", {})
        if isinstance(mm, dict) and mm.get("valid"):
            winners.append((row["ticker"], float(mm.get("edge_pct", 0.0)), float(mm.get("hit_rate", 0.0))))
    winners.sort(key=lambda x: abs(x[1]), reverse=True)

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "n_prediction_contract_themes": int(probs.shape[1]),
        "contract_meta": contract_meta,
        "comparison": {
            "tenk_risk_mapped": {
                "mean_hit_rate": float(np.mean(mapped_hits)) if mapped_hits else None,
                "mean_mae": float(np.mean(mapped_mae)) if mapped_mae else None,
                "n_tickers": len(mapped_hits),
            },
            "baseline_4theme": {
                "mean_hit_rate": float(np.mean(baseline_hits)) if baseline_hits else None,
                "mean_mae": float(np.mean(baseline_mae)) if baseline_mae else None,
                "n_tickers": len(baseline_hits),
                "themes": baseline_themes,
            },
        },
        "top_edge_tickers": [
            {"ticker": t, "edge_pct": e, "hit_rate": h}
            for t, e, h in winners[:8]
        ],
        "tickers": ticker_rows,
    }

    out_dir = Path(__file__).resolve().parent.parent / "analysis_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tenk_risk_pricing_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    summary = run()
    print(json.dumps(summary["comparison"], indent=2))
    print(json.dumps(summary["top_edge_tickers"], indent=2))
    print("\nWrote analysis_output/tenk_risk_pricing_summary.json")


if __name__ == "__main__":
    main()
