"""Comprehensive Alpha Diagnostic — Separate Genuine Model Skill from Luck.

This script runs a full backtest (same as run_alpha_backtest.py), then immediately
performs 7 rigorous statistical tests on the trade-level data to determine whether
the reported alpha is real or an artifact of outliers / luck / bad annualization math.

Tests:
  1. ANNUALIZATION CROSS-CHECK — Compare multiplicative formula vs actual equity curve
  2. OUTLIER DEPENDENCE — Remove top N trades and recompute alpha
  3. PERMUTATION TEST — Shuffle predictions 5000 times; what % beats actual?
  4. REGIME STABILITY — Split into time halves; is alpha stable or concentrated?
  5. PER-TICKER CONCENTRATION — Is alpha from 3 tickers or 30?
  6. HIT RATE vs MAGNITUDE — Win rate, avg win, avg loss, profit factor, skew
  7. TEMPORAL AUTOCORRELATION — Are returns clustered (suggesting a regime trade, not skill)?

Usage:
    cd quant-api
    python -m scripts.diagnose_alpha [--fast] [--subset N]
"""
from __future__ import annotations

import json
import sys
import os
import warnings
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# Suppress noisy warnings during yfinance downloads
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(str(Path(__file__).resolve().parent.parent))

from app.backtest.alpha_attribution import (
    AlphaAttributionResult,
    SegmentedAlphaBacktester,
    TickerBacktestResult,
)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: extract flat trade-level arrays from ticker results
# ─────────────────────────────────────────────────────────────────────────────


def extract_trades(ticker_results: list[TickerBacktestResult]) -> pd.DataFrame:
    """Flatten all trade-level data into a single DataFrame for analysis."""
    rows = []
    for tr in ticker_results:
        spy_rets = tr.benchmark_returns.get("SPY", [0.0] * len(tr.net_returns))
        for i in range(len(tr.net_returns)):
            bench = spy_rets[i] if i < len(spy_rets) else 0.0
            rows.append({
                "ticker": tr.ticker,
                "commodity_type": tr.commodity_type,
                "country": tr.country,
                "date": tr.prediction_dates[i] if i < len(tr.prediction_dates) else None,
                "prediction": tr.predictions[i] if i < len(tr.predictions) else 0.0,
                "actual_return": tr.actuals[i] if i < len(tr.actuals) else 0.0,
                "net_return": tr.net_returns[i],
                "benchmark_return": bench,
                "cost_bps": tr.costs_bps[i] if i < len(tr.costs_bps) else 0.0,
                "direction": tr.directions[i] if i < len(tr.directions) else "LONG",
                "market_cap": tr.market_cap,
            })
    df = pd.DataFrame(rows)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    df["alpha"] = df["net_return"] - df["benchmark_return"]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: ANNUALIZATION CROSS-CHECK
# ─────────────────────────────────────────────────────────────────────────────


def test_annualization(trades: pd.DataFrame, holding_days: int = 20) -> dict:
    """Compare the multiplicative formula vs real equity curve annualization.

    The engine uses: annual_alpha = mean(alpha) * (252/holding_days) * 100
    This is ARITHMETIC annualization — correct only if returns are small and IID.

    We also compute:
      - Geometric annualization from cumulative product
      - Time-weighted return using actual calendar days
      - Bootstrap median annualized alpha
    """
    alpha_arr = trades["alpha"].values
    net_arr = trades["net_return"].values
    bench_arr = trades["benchmark_return"].values

    trades_per_year = 252 / holding_days

    # Method 1: Engine formula (arithmetic)
    arithmetic_annual = float(np.mean(alpha_arr)) * trades_per_year * 100

    # Method 2: Geometric annualization (per-trade geometric mean)
    # Use geometric mean of per-trade alpha, NOT sequential compounding
    # (Trades are concurrent across tickers, not sequential)
    n_trades = len(alpha_arr)
    # Geometric mean of (1 + alpha) per trade
    geo_mean_per_trade = float(np.exp(np.mean(np.log(np.maximum(1 + alpha_arr, 1e-10)))))
    geometric_annual = (geo_mean_per_trade ** trades_per_year - 1) * 100

    # Method 3: Portfolio-level time-weighted return
    # Group trades by date, compute equal-weight portfolio return per period
    if "date" in trades.columns and trades["date"].notna().sum() > 1:
        date_range_days = (trades["date"].max() - trades["date"].min()).days
        years_actual = max(0.5, date_range_days / 365.25)
        # Compute PORTFOLIO alpha per 20-day period by averaging concurrent trades
        period_alphas = trades.groupby("date")["alpha"].mean().sort_index()
        if len(period_alphas) > 0:
            cum_portfolio = float(np.prod(1 + period_alphas.values))
            time_weighted_annual = (cum_portfolio ** (1.0 / years_actual) - 1) * 100
        else:
            time_weighted_annual = geometric_annual
    else:
        years_actual = max(0.5, n_trades / trades_per_year)
        time_weighted_annual = geometric_annual

    # Method 4: Strategy vs benchmark (portfolio-level)
    # Average across concurrent positions per period
    if "date" in trades.columns and trades["date"].notna().sum() > 1:
        period_strat = trades.groupby("date")["net_return"].mean().sort_index()
        period_bench = trades.groupby("date")["benchmark_return"].mean().sort_index()
        cum_strat = float(np.prod(1 + period_strat.values))
        cum_bench = float(np.prod(1 + period_bench.values))
        strat_annual = (cum_strat ** (1.0 / years_actual) - 1) * 100
        bench_annual = (cum_bench ** (1.0 / years_actual) - 1) * 100
    else:
        cum_strat = float(np.prod(1 + net_arr))
        cum_bench = float(np.prod(1 + bench_arr))
        strat_annual = (cum_strat ** (1.0 / years_actual) - 1) * 100
        bench_annual = (cum_bench ** (1.0 / years_actual) - 1) * 100
    diff_annual = strat_annual - bench_annual

    # Method 5: Bootstrap median
    rng = np.random.default_rng(42)
    boot_alphas = []
    for _ in range(5000):
        sample = rng.choice(alpha_arr, size=len(alpha_arr), replace=True)
        boot_alphas.append(float(np.mean(sample)) * trades_per_year * 100)
    boot_median = float(np.median(boot_alphas))
    boot_5 = float(np.percentile(boot_alphas, 5))
    boot_95 = float(np.percentile(boot_alphas, 95))

    # Determine inflation factor
    methods = [arithmetic_annual, geometric_annual, time_weighted_annual, diff_annual, boot_median]
    honest_range = [min(methods), max(methods)]
    spread = max(methods) - min(methods)
    median_method = float(np.median(methods))

    if spread < 15:
        verdict = f"CONSISTENT — methods agree within {spread:.1f}pp (median={median_method:.1f}%)"
    elif arithmetic_annual > 2 * median_method:
        verdict = f"INFLATED — arithmetic ({arithmetic_annual:.1f}%) >> median of methods ({median_method:.1f}%)"
    else:
        verdict = f"MIXED — methods diverge by {spread:.1f}pp, use median ({median_method:.1f}%)"

    return {
        "arithmetic_annual_alpha_pct": round(arithmetic_annual, 3),
        "geometric_annual_alpha_pct": round(geometric_annual, 3),
        "time_weighted_annual_alpha_pct": round(time_weighted_annual, 3),
        "strategy_minus_bench_annual_pct": round(diff_annual, 3),
        "strategy_annual_pct": round(strat_annual, 3),
        "benchmark_annual_pct": round(bench_annual, 3),
        "bootstrap_median_alpha_pct": round(boot_median, 3),
        "bootstrap_90ci": [round(boot_5, 3), round(boot_95, 3)],
        "n_trades": n_trades,
        "n_years_actual": round(years_actual, 2),
        "n_periods_portfolio": len(period_alphas) if "date" in trades.columns else n_trades,
        "cum_portfolio_strategy": round(cum_strat, 4),
        "cum_portfolio_benchmark": round(cum_bench, 4),
        "median_of_methods": round(median_method, 3),
        "VERDICT": verdict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: OUTLIER DEPENDENCE
# ─────────────────────────────────────────────────────────────────────────────


def test_outlier_dependence(trades: pd.DataFrame, holding_days: int = 20) -> dict:
    """Remove top N trades by alpha and recompute — see how fragile the result is.

    If removing 3 trades kills the alpha, it's luck, not skill.
    """
    alpha_arr = trades["alpha"].values.copy()
    trades_per_year = 252 / holding_days
    base_alpha = float(np.mean(alpha_arr)) * trades_per_year * 100
    n = len(alpha_arr)

    results = {"base_alpha_pct": round(base_alpha, 3), "removals": []}

    # Remove top 1, 3, 5, 10, and bottom 1, 3, 5
    sorted_idx = np.argsort(alpha_arr)

    for remove_count in [1, 3, 5, 10]:
        if remove_count >= n - 5:
            continue

        # Remove top N (best trades)
        top_mask = np.ones(n, dtype=bool)
        top_mask[sorted_idx[-remove_count:]] = False
        top_removed = alpha_arr[top_mask]
        alpha_no_top = float(np.mean(top_removed)) * trades_per_year * 100

        # Remove bottom N (worst trades)
        bot_mask = np.ones(n, dtype=bool)
        bot_mask[sorted_idx[:remove_count]] = False
        bot_removed = alpha_arr[bot_mask]
        alpha_no_bot = float(np.mean(bot_removed)) * trades_per_year * 100

        # Remove both top and bottom N (winsorized)
        both_mask = np.ones(n, dtype=bool)
        both_mask[sorted_idx[-remove_count:]] = False
        both_mask[sorted_idx[:remove_count]] = False
        both_removed = alpha_arr[both_mask]
        alpha_winsor = float(np.mean(both_removed)) * trades_per_year * 100 if len(both_removed) > 0 else 0.0

        # Removed trade details
        top_trades = trades.iloc[sorted_idx[-remove_count:]].sort_values("alpha", ascending=False)
        top_detail = [
            f"{r['ticker']} {r['date'].strftime('%Y-%m-%d') if pd.notna(r.get('date')) else '?'} alpha={r['alpha']*100:.2f}%"
            for _, r in top_trades.iterrows()
        ]

        results["removals"].append({
            "removed_top": remove_count,
            "alpha_without_top_pct": round(alpha_no_top, 3),
            "alpha_without_bottom_pct": round(alpha_no_bot, 3),
            "alpha_winsorized_pct": round(alpha_winsor, 3),
            "pct_change_from_base": round((alpha_no_top - base_alpha) / abs(base_alpha) * 100 if base_alpha != 0 else 0, 1),
            "top_trades_removed": top_detail[:5],
        })

    # Overall verdict
    if results["removals"]:
        # Check if removing top 3 drops alpha by >50%
        removal_3 = next((r for r in results["removals"] if r["removed_top"] == 3), None)
        if removal_3 and abs(removal_3["pct_change_from_base"]) > 50:
            results["VERDICT"] = "FRAGILE — alpha depends heavily on a few outlier trades"
        elif removal_3 and abs(removal_3["pct_change_from_base"]) > 25:
            results["VERDICT"] = "MODERATE — some outlier dependence, verify trades are repeatable"
        else:
            results["VERDICT"] = "ROBUST — alpha survives outlier removal"
    else:
        results["VERDICT"] = "INSUFFICIENT DATA"

    return results


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: PERMUTATION TEST (Skill vs Luck)
# ─────────────────────────────────────────────────────────────────────────────


def test_permutation(trades: pd.DataFrame, n_perms: int = 5000, holding_days: int = 20) -> dict:
    """Classic permutation test: shuffle which trades the model 'took' vs 'skipped'.

    The model selects LONG trades when pred > 0.001. We test: if we randomly
    selected the same NUMBER of trades from the pool of all 20-day windows
    (including the ones the model skipped), would we get similar alpha?

    Since we don't have the skipped trades, we instead shuffle the SIGNS of alpha
    (randomly assign each trade's alpha to be positive or negative at its magnitude).
    This tests: "Is the model's directional accuracy better than coin flips?"

    Also: shuffle trade-to-ticker assignments (permute which alpha goes with which trade).
    """
    alpha_arr = trades["alpha"].values
    trades_per_year = 252 / holding_days
    actual_alpha = float(np.mean(alpha_arr)) * trades_per_year * 100

    rng = np.random.default_rng(42)

    # Test A: Sign permutation (randomize direction)
    sign_perms = []
    for _ in range(n_perms):
        signs = rng.choice([-1, 1], size=len(alpha_arr))
        perm_alpha = float(np.mean(alpha_arr * signs)) * trades_per_year * 100
        sign_perms.append(perm_alpha)
    sign_perms = np.array(sign_perms)
    sign_p_value = float(np.mean(sign_perms >= actual_alpha))

    # Test B: Row permutation (shuffle order, breaking date clustering)
    row_perms = []
    for _ in range(n_perms):
        shuffled = rng.permutation(alpha_arr)
        perm_alpha = float(np.mean(shuffled)) * trades_per_year * 100
        row_perms.append(perm_alpha)
    # Row permutation of mean always equals original mean — this is a control test
    # (should always be ~same as actual). Real test is the sign permutation.

    # Test C: Cross-ticker shuffle — assign each net_return to a random benchmark_return
    net_arr = trades["net_return"].values
    bench_arr = trades["benchmark_return"].values
    cross_perms = []
    for _ in range(n_perms):
        shuffled_bench = rng.permutation(bench_arr)
        perm_alpha_arr = net_arr - shuffled_bench
        perm_alpha = float(np.mean(perm_alpha_arr)) * trades_per_year * 100
        cross_perms.append(perm_alpha)
    cross_perms = np.array(cross_perms)
    cross_p_value = float(np.mean(cross_perms >= actual_alpha))

    # Summarize
    if sign_p_value < 0.01:
        verdict = f"SIGNIFICANT (p={sign_p_value:.4f}) — model beats random direction at 99% confidence"
    elif sign_p_value < 0.05:
        verdict = f"SIGNIFICANT (p={sign_p_value:.4f}) — model beats random direction at 95% confidence"
    elif sign_p_value < 0.10:
        verdict = f"MARGINAL (p={sign_p_value:.4f}) — weakly significant, could be luck"
    else:
        verdict = f"NOT SIGNIFICANT (p={sign_p_value:.4f}) — CANNOT reject that this is random"

    return {
        "actual_annual_alpha_pct": round(actual_alpha, 3),
        "sign_permutation_p_value": round(sign_p_value, 4),
        "sign_perm_mean_alpha": round(float(np.mean(sign_perms)), 3),
        "sign_perm_95th_pctl": round(float(np.percentile(sign_perms, 95)), 3),
        "cross_ticker_perm_p_value": round(cross_p_value, 4),
        "cross_perm_mean": round(float(np.mean(cross_perms)), 3),
        "n_permutations": n_perms,
        "n_trades": len(alpha_arr),
        "VERDICT": verdict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: REGIME STABILITY
# ─────────────────────────────────────────────────────────────────────────────


def test_regime_stability(trades: pd.DataFrame, holding_days: int = 20) -> dict:
    """Split trades into time halves, thirds, and years. Is alpha stable?

    A model with genuine skill shows alpha across multiple time periods.
    If alpha is concentrated in one period, it may be a spurious regime bet.
    """
    if "date" not in trades.columns or trades["date"].isna().all():
        return {"VERDICT": "CANNOT TEST — no date information"}

    alpha_arr = trades["alpha"].values
    dates = trades["date"].values
    trades_per_year = 252 / holding_days

    # Split into halves
    mid = len(trades) // 2
    first_half_alpha = float(np.mean(alpha_arr[:mid])) * trades_per_year * 100
    second_half_alpha = float(np.mean(alpha_arr[mid:])) * trades_per_year * 100

    # Split into thirds
    t1 = len(trades) // 3
    t2 = 2 * t1
    third_1 = float(np.mean(alpha_arr[:t1])) * trades_per_year * 100
    third_2 = float(np.mean(alpha_arr[t1:t2])) * trades_per_year * 100
    third_3 = float(np.mean(alpha_arr[t2:])) * trades_per_year * 100

    # Annual breakdown
    valid_dates = trades.dropna(subset=["date"])
    annual_data = []
    if len(valid_dates) > 0:
        valid_dates = valid_dates.copy()
        valid_dates["year"] = valid_dates["date"].dt.year
        for year, grp in valid_dates.groupby("year"):
            if len(grp) >= 3:
                year_alpha = float(np.mean(grp["alpha"].values)) * trades_per_year * 100
                annual_data.append({
                    "year": int(year),
                    "trades": len(grp),
                    "annual_alpha_pct": round(year_alpha, 3),
                    "hit_rate_pct": round(float(np.mean(grp["alpha"] > 0)) * 100, 1),
                })

    # Quarterly breakdown
    quarterly_data = []
    if len(valid_dates) > 0:
        valid_dates_q = valid_dates.copy()
        valid_dates_q["quarter"] = valid_dates_q["date"].dt.to_period("Q").astype(str)
        for qtr, grp in valid_dates_q.groupby("quarter"):
            if len(grp) >= 3:
                q_alpha = float(np.mean(grp["alpha"].values)) * trades_per_year * 100
                quarterly_data.append({
                    "quarter": str(qtr),
                    "trades": len(grp),
                    "annual_alpha_pct": round(q_alpha, 3),
                })

    # Count positive vs negative regime quarters
    n_pos_q = sum(1 for q in quarterly_data if q["annual_alpha_pct"] > 0)
    n_neg_q = sum(1 for q in quarterly_data if q["annual_alpha_pct"] <= 0)

    # Check consistency
    halves_consistent = (first_half_alpha > 0 and second_half_alpha > 0)
    thirds_consistent = sum(1 for x in [third_1, third_2, third_3] if x > 0) >= 2

    if halves_consistent and thirds_consistent and n_pos_q > n_neg_q:
        verdict = "STABLE — alpha present in multiple time periods"
    elif halves_consistent:
        verdict = "MODERATE — alpha in both halves but uneven across sub-periods"
    else:
        verdict = "CONCENTRATED — alpha may be a regime artifact, not persistent skill"

    return {
        "first_half_alpha_pct": round(first_half_alpha, 3),
        "first_half_trades": mid,
        "second_half_alpha_pct": round(second_half_alpha, 3),
        "second_half_trades": len(trades) - mid,
        "third_1_pct": round(third_1, 3),
        "third_2_pct": round(third_2, 3),
        "third_3_pct": round(third_3, 3),
        "annual_breakdown": annual_data,
        "quarterly_breakdown": quarterly_data,
        "positive_quarters": n_pos_q,
        "negative_quarters": n_neg_q,
        "VERDICT": verdict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5: PER-TICKER CONCENTRATION
# ─────────────────────────────────────────────────────────────────────────────


def test_ticker_concentration(trades: pd.DataFrame, holding_days: int = 20) -> dict:
    """Check if alpha is concentrated in a few tickers or broadly distributed."""
    trades_per_year = 252 / holding_days
    base_alpha = float(np.mean(trades["alpha"].values)) * trades_per_year * 100

    # Per-ticker alpha contribution
    ticker_stats = []
    for ticker, grp in trades.groupby("ticker"):
        g_alpha = grp["alpha"].values
        avg_alpha_per_trade = float(np.mean(g_alpha))
        # This ticker's contribution to the overall mean alpha
        contribution = avg_alpha_per_trade * len(g_alpha) / len(trades) * trades_per_year * 100
        ticker_stats.append({
            "ticker": ticker,
            "trades": len(grp),
            "avg_alpha_per_trade_pct": round(avg_alpha_per_trade * 100, 3),
            "annual_alpha_pct": round(float(np.mean(g_alpha)) * trades_per_year * 100, 3),
            "contribution_to_total_pct": round(contribution, 3),
            "hit_rate_pct": round(float(np.mean(g_alpha > 0)) * 100, 1),
            "commodity_type": grp["commodity_type"].iloc[0] if "commodity_type" in grp.columns else "",
        })

    ticker_stats.sort(key=lambda x: x["contribution_to_total_pct"], reverse=True)

    # Concentration metrics
    contributions = np.array([t["contribution_to_total_pct"] for t in ticker_stats])
    pos_contributions = contributions[contributions > 0]

    if len(pos_contributions) > 0:
        # What fraction of total alpha comes from top 3/5/10 tickers?
        cum = np.cumsum(sorted(pos_contributions, reverse=True))
        total_pos = cum[-1]
        top3_pct = float(cum[min(2, len(cum)-1)]) / total_pos * 100 if total_pos > 0 else 0
        top5_pct = float(cum[min(4, len(cum)-1)]) / total_pos * 100 if total_pos > 0 else 0
        top10_pct = float(cum[min(9, len(cum)-1)]) / total_pos * 100 if total_pos > 0 else 0
    else:
        top3_pct = top5_pct = top10_pct = 0.0

    n_positive = sum(1 for t in ticker_stats if t["annual_alpha_pct"] > 0)
    n_negative = sum(1 for t in ticker_stats if t["annual_alpha_pct"] <= 0)
    n_above_2pct = sum(1 for t in ticker_stats if t["annual_alpha_pct"] >= 2.0)

    # Leave-one-ticker-out (LOTO) analysis
    loto_results = []
    for i, ts in enumerate(ticker_stats):
        mask = trades["ticker"] != ts["ticker"]
        remaining = trades.loc[mask, "alpha"].values
        if len(remaining) > 0:
            loto_alpha = float(np.mean(remaining)) * trades_per_year * 100
            loto_results.append({
                "ticker_removed": ts["ticker"],
                "alpha_without_pct": round(loto_alpha, 3),
                "change_pct": round(loto_alpha - base_alpha, 3),
            })
    loto_results.sort(key=lambda x: x["change_pct"])

    if top3_pct > 80:
        verdict = "CONCENTRATED — top 3 tickers drive >80% of alpha"
    elif top5_pct > 80:
        verdict = "MODERATE CONCENTRATION — top 5 tickers drive >80%"
    elif n_positive > n_negative * 1.5:
        verdict = "WELL-DISTRIBUTED — alpha comes from many tickers"
    else:
        verdict = "MIXED — nearly equal positive and negative alpha tickers"

    return {
        "total_tickers": len(ticker_stats),
        "positive_alpha_tickers": n_positive,
        "negative_alpha_tickers": n_negative,
        "above_2pct_threshold": n_above_2pct,
        "top3_concentration_pct": round(top3_pct, 1),
        "top5_concentration_pct": round(top5_pct, 1),
        "top10_concentration_pct": round(top10_pct, 1),
        "top_contributors": ticker_stats[:10],
        "worst_detractors": ticker_stats[-5:],
        "most_impactful_loto": loto_results[:5] + loto_results[-3:],
        "VERDICT": verdict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6: HIT RATE vs MAGNITUDE (Profit Quality)
# ─────────────────────────────────────────────────────────────────────────────


def test_profit_quality(trades: pd.DataFrame) -> dict:
    """Decompose profitability into win rate, average win, average loss, and skew.

    A model could have:
    - High hit rate, tiny wins, giant losses = BAD (picking up pennies before steamroller)
    - Low hit rate, giant wins, tiny losses = GOOD (asymmetric risk/reward)
    - High hit rate, balanced wins/losses = BEST (genuine prediction accuracy)
    """
    alpha_arr = trades["alpha"].values
    net_arr = trades["net_return"].values

    # Alpha-based (excess over benchmark)
    alpha_wins = alpha_arr[alpha_arr > 0]
    alpha_losses = alpha_arr[alpha_arr <= 0]

    alpha_hit_rate = len(alpha_wins) / len(alpha_arr) * 100 if len(alpha_arr) > 0 else 0
    avg_win = float(np.mean(alpha_wins)) * 100 if len(alpha_wins) > 0 else 0
    avg_loss = float(np.mean(alpha_losses)) * 100 if len(alpha_losses) > 0 else 0
    median_alpha = float(np.median(alpha_arr)) * 100
    skewness = float(sp_stats.skew(alpha_arr)) if len(alpha_arr) > 2 else 0.0
    kurtosis = float(sp_stats.kurtosis(alpha_arr)) if len(alpha_arr) > 3 else 0.0

    # Profit factor: sum of wins / abs(sum of losses)
    sum_wins = float(np.sum(alpha_wins)) if len(alpha_wins) > 0 else 0
    sum_losses = float(abs(np.sum(alpha_losses))) if len(alpha_losses) > 0 else 0.001
    profit_factor = sum_wins / sum_losses if sum_losses > 0 else float("inf")

    # Kelly criterion (simplified)
    if alpha_hit_rate > 0 and alpha_hit_rate < 100 and avg_loss != 0:
        win_loss_ratio = abs(avg_win / avg_loss)
        p = alpha_hit_rate / 100
        kelly = p - (1 - p) / win_loss_ratio
    else:
        kelly = 0.0

    # Net return-based (absolute)
    net_wins = net_arr[net_arr > 0]
    net_losses = net_arr[net_arr <= 0]
    net_hit_rate = len(net_wins) / len(net_arr) * 100 if len(net_arr) > 0 else 0

    # Largest winners and losers
    sorted_trades = trades.sort_values("alpha", ascending=False)
    top5 = sorted_trades.head(5)[["ticker", "date", "alpha", "net_return", "benchmark_return"]].copy()
    top5["alpha_pct"] = top5["alpha"] * 100
    bot5 = sorted_trades.tail(5)[["ticker", "date", "alpha", "net_return", "benchmark_return"]].copy()
    bot5["alpha_pct"] = bot5["alpha"] * 100

    if profit_factor > 1.5 and alpha_hit_rate > 50:
        verdict = "STRONG — good hit rate with positive profit factor"
    elif profit_factor > 1.0 and alpha_hit_rate > 45:
        verdict = "ADEQUATE — marginal edge"
    elif skewness > 1.0:
        verdict = "LOTTERY — relies on rare big wins (right-tail skew)"
    else:
        verdict = "WEAK — no clear systematic edge"

    return {
        "alpha_hit_rate_pct": round(alpha_hit_rate, 1),
        "avg_alpha_win_pct": round(avg_win, 3),
        "avg_alpha_loss_pct": round(avg_loss, 3),
        "median_alpha_pct": round(median_alpha, 3),
        "skewness": round(skewness, 3),
        "kurtosis": round(kurtosis, 3),
        "profit_factor": round(profit_factor, 3),
        "kelly_fraction": round(kelly, 4),
        "net_hit_rate_pct": round(net_hit_rate, 1),
        "avg_net_win_pct": round(float(np.mean(net_wins)) * 100, 3) if len(net_wins) > 0 else 0,
        "avg_net_loss_pct": round(float(np.mean(net_losses)) * 100, 3) if len(net_losses) > 0 else 0,
        "top_5_winners": [
            {"ticker": r["ticker"],
             "date": r["date"].strftime("%Y-%m-%d") if pd.notna(r["date"]) else "?",
             "alpha_pct": round(r["alpha_pct"], 2)}
            for _, r in top5.iterrows()
        ],
        "bottom_5_losers": [
            {"ticker": r["ticker"],
             "date": r["date"].strftime("%Y-%m-%d") if pd.notna(r["date"]) else "?",
             "alpha_pct": round(r["alpha_pct"], 2)}
            for _, r in bot5.iterrows()
        ],
        "VERDICT": verdict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TEST 7: TEMPORAL AUTOCORRELATION
# ─────────────────────────────────────────────────────────────────────────────


def test_autocorrelation(trades: pd.DataFrame) -> dict:
    """Check if trade alpha is autocorrelated — suggesting regime dependence, not skill.

    If winning streaks / losing streaks cluster, the 'model' may just be riding
    a macro regime (e.g., oil rally 2021-2022) rather than having genuine predictive power.
    """
    alpha_arr = trades["alpha"].values
    if len(alpha_arr) < 10:
        return {"VERDICT": "INSUFFICIENT DATA"}

    # Lag-1 autocorrelation
    lag1_corr = float(np.corrcoef(alpha_arr[:-1], alpha_arr[1:])[0, 1])

    # Runs test: count consecutive runs of positive/negative alpha
    # Under IID, number of runs follows a known distribution
    signs = (alpha_arr > 0).astype(int)
    n_pos = np.sum(signs)
    n_neg = len(signs) - n_pos

    if n_pos == 0 or n_neg == 0:
        return {
            "lag1_autocorrelation": round(lag1_corr, 4),
            "VERDICT": "ALL SAME SIGN — cannot run runs test",
        }

    # Count runs
    runs = 1
    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1]:
            runs += 1

    # Expected runs and variance under null (IID)
    n = len(signs)
    expected_runs = 1 + 2 * n_pos * n_neg / n
    var_runs = 2 * n_pos * n_neg * (2 * n_pos * n_neg - n) / (n**2 * (n - 1))
    if var_runs > 0:
        z_runs = (runs - expected_runs) / np.sqrt(var_runs)
        runs_p_value = 2 * (1 - sp_stats.norm.cdf(abs(z_runs)))  # two-sided
    else:
        z_runs = 0.0
        runs_p_value = 1.0

    # Ljung-Box test for serial correlation (lags 1-5)
    n_obs = len(alpha_arr)
    lb_stat = 0.0
    lb_lags = min(5, n_obs // 4)
    for lag in range(1, lb_lags + 1):
        r_k = float(np.corrcoef(alpha_arr[:-lag], alpha_arr[lag:])[0, 1])
        lb_stat += r_k**2 / (n_obs - lag)
    lb_stat *= n_obs * (n_obs + 2)
    lb_p_value = 1 - sp_stats.chi2.cdf(lb_stat, lb_lags) if lb_lags > 0 else 1.0

    if abs(lag1_corr) < 0.1 and runs_p_value > 0.05:
        verdict = "IID — no significant autocorrelation (good: model is not just regime-riding)"
    elif abs(lag1_corr) > 0.3:
        verdict = "CORRELATED — strong autocorrelation suggests regime dependence"
    else:
        verdict = "WEAK CORRELATION — some clustering but not conclusive"

    return {
        "lag1_autocorrelation": round(lag1_corr, 4),
        "runs_test_statistic": round(float(z_runs), 4),
        "runs_test_p_value": round(float(runs_p_value), 4),
        "observed_runs": runs,
        "expected_runs": round(expected_runs, 1),
        "ljung_box_stat": round(lb_stat, 4),
        "ljung_box_p_value": round(lb_p_value, 4),
        "n_positive_alpha_trades": int(n_pos),
        "n_negative_alpha_trades": int(n_neg),
        "VERDICT": verdict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BONUS TEST 8: t-TEST vs ZERO + DEFLATED SHARPE
# ─────────────────────────────────────────────────────────────────────────────


def test_statistical_significance(trades: pd.DataFrame, holding_days: int = 20) -> dict:
    """Classic t-test: is mean alpha significantly different from zero?

    Also computes deflated Sharpe ratio (Harvey, Liu & Zhu, 2015) to account
    for multiple testing — we tested hundreds of features and many tickers.
    """
    alpha_arr = trades["alpha"].values
    n = len(alpha_arr)
    trades_per_year = 252 / holding_days

    if n < 3:
        return {"VERDICT": "INSUFFICIENT DATA"}

    # Simple t-test: mean alpha vs 0
    t_stat, p_value = sp_stats.ttest_1samp(alpha_arr, 0.0)

    # Autocorrelation-adjusted (Newey-West style) effective sample size
    # If trades are autocorrelated, the effective N is much smaller than actual N
    lag1_corr = float(np.corrcoef(alpha_arr[:-1], alpha_arr[1:])[0, 1]) if n > 2 else 0.0
    # Effective N shrinks with positive autocorrelation
    rho = max(0, lag1_corr)  # only positive autocorrelation inflates
    effective_n = n * (1 - rho) / (1 + rho) if rho < 0.99 else 1.0
    effective_n = max(3, effective_n)

    # Autocorrelation-adjusted t-test
    mean_alpha = float(np.mean(alpha_arr))
    se_alpha_raw = float(np.std(alpha_arr, ddof=1)) / np.sqrt(n)
    se_alpha_adj = float(np.std(alpha_arr, ddof=1)) / np.sqrt(effective_n)
    t_stat_adj = mean_alpha / se_alpha_adj if se_alpha_adj > 0 else 0.0
    p_value_adj = 2 * (1 - sp_stats.t.cdf(abs(t_stat_adj), df=max(1, int(effective_n) - 1)))

    # Confidence intervals for annual alpha (use ADJUSTED standard error)
    se_alpha = se_alpha_adj  # use conservative adjusted SE
    annual_mean = mean_alpha * trades_per_year * 100
    annual_se = se_alpha * trades_per_year * 100

    ci_95_low = annual_mean - 1.96 * annual_se
    ci_95_high = annual_mean + 1.96 * annual_se
    ci_99_low = annual_mean - 2.576 * annual_se
    ci_99_high = annual_mean + 2.576 * annual_se

    # Sharpe ratio (annualized)
    if np.std(alpha_arr, ddof=1) > 0:
        sr = float(np.mean(alpha_arr) / np.std(alpha_arr, ddof=1)) * np.sqrt(trades_per_year)
    else:
        sr = 0.0

    # Deflated Sharpe Ratio (DSR) — accounts for number of trials
    # Honest count: ~15 features × ~10 config combos × ~8 universe cuts = ~1200 implicit trials
    # Using n_trials=1000 as conservative lower bound (Bailey & Lopez de Prado 2014)
    n_trials = 1000
    skew = float(sp_stats.skew(alpha_arr)) if n > 2 else 0.0
    kurt = float(sp_stats.kurtosis(alpha_arr)) if n > 3 else 0.0

    # Bailey & Lopez de Prado (2014) deflated Sharpe
    # Expected maximum Sharpe under n_trials independent tests
    e_max_sr = float(sp_stats.norm.ppf(1 - 1 / n_trials)) * (1 - 0.5772 / np.log(n_trials))

    # DSR test statistic
    sr_std = np.sqrt((1 - skew * sr + (kurt - 1) / 4 * sr**2) / (n - 1))
    if sr_std > 0:
        dsr_stat = (sr - e_max_sr) / sr_std
        dsr_p_value = 1 - sp_stats.norm.cdf(dsr_stat)
    else:
        dsr_stat = 0.0
        dsr_p_value = 1.0

    # Minimum Backtest Length (MinBTL) — Lopez de Prado (2016)
    # How many years of data needed for this SR to be significant at 95%?
    if sr > 0:
        min_btl = (1 + (1 - skew * sr + (kurt - 1) / 4 * sr**2) *
                   (sp_stats.norm.ppf(0.95) / sr) ** 2) / trades_per_year
    else:
        min_btl = float("inf")

    if p_value_adj < 0.01 and dsr_p_value < 0.05:
        verdict = f"SIGNIFICANT at 99% (autocorr-adjusted) and SURVIVES deflated Sharpe"
    elif p_value_adj < 0.05 and dsr_p_value < 0.10:
        verdict = f"SIGNIFICANT at 95% (autocorr-adjusted), marginal after multiple-testing correction"
    elif p_value_adj < 0.05:
        verdict = f"SIGNIFICANT at 95% (autocorr-adjusted) but FAILS deflated Sharpe — may be data-mined"
    elif p_value < 0.05 and p_value_adj >= 0.05:
        verdict = f"INFLATED — significant before autocorr-adjustment (p={p_value:.4f}) but NOT after (p_adj={p_value_adj:.4f})"
    elif p_value_adj < 0.10:
        verdict = f"MARGINAL significance (p_adj={p_value_adj:.4f}) — not reliable for investment"
    else:
        verdict = f"NOT SIGNIFICANT (p_adj={p_value_adj:.4f}) — cannot conclude alpha is real"

    return {
        "t_statistic_raw": round(float(t_stat), 4),
        "p_value_raw": round(float(p_value), 6),
        "lag1_autocorrelation": round(lag1_corr, 4),
        "effective_n": round(float(effective_n), 1),
        "actual_n": n,
        "t_statistic_adjusted": round(float(t_stat_adj), 4),
        "p_value_adjusted": round(float(p_value_adj), 6),
        "annualized_alpha_pct": round(annual_mean, 3),
        "annualized_se_pct": round(annual_se, 3),
        "ci_95_pct": [round(ci_95_low, 3), round(ci_95_high, 3)],
        "ci_99_pct": [round(ci_99_low, 3), round(ci_99_high, 3)],
        "sharpe_ratio": round(sr, 4),
        "deflated_sharpe_ratio_stat": round(float(dsr_stat), 4),
        "deflated_sharpe_p_value": round(float(dsr_p_value), 4),
        "expected_max_sr_from_trials": round(e_max_sr, 4),
        "n_assumed_trials": n_trials,
        "minimum_backtest_length_years": round(float(min_btl), 2),
        "actual_backtest_years": round(n / trades_per_year, 2),
        "VERDICT": verdict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TEST 9: PREDICTION MARKET ABLATION (PM Feature Impact)
# ─────────────────────────────────────────────────────────────────────────────


def test_pm_ablation(trades: pd.DataFrame, holding_days: int = 20) -> dict:
    """Test whether prediction market features actually contribute to alpha.

    TWO-PRONGED TEST:
      A) SECTOR PROXY: Compare alpha in PM-relevant sectors (oil, shipping) vs irrelevant.
         If PM features drive alpha, oil/shipping should massively outperform.
      B) TRUE ABLATION: (if ablation_trades provided) Compare full model vs PM-zeroed model.

    Only 4 events have real Polymarket IDs (hormuz, red_sea, sanctions, oil_above_100).
    The other 52 are constant 0.5 = zero variance = filtered out by variance pre-filter.
    So PM features likely contribute nothing. This test checks by comparing alpha in
    PM-relevant sectors (oil, shipping) vs PM-irrelevant sectors (metals, uranium, etc.).

    If PM features drive alpha, oil/shipping should massively outperform other sectors.
    If they don't, the PM thesis is window dressing and alpha comes from commodity/macro features.
    """
    trades_per_year = 252 / holding_days

    # Sectors where real PM events exist (hormuz, red_sea → shipping/oil)
    pm_sectors = {"oil_gas_upstream", "oil_refining", "oil_services", "midstream",
                  "shipping_tanker", "shipping_drybulk", "shipping_container",
                  "shipping_services", "lng_shipping"}

    pm_trades = trades[trades["commodity_type"].isin(pm_sectors)]
    non_pm_trades = trades[~trades["commodity_type"].isin(pm_sectors)]

    pm_alpha = float(np.mean(pm_trades["alpha"])) * trades_per_year * 100 if len(pm_trades) > 0 else 0.0
    non_pm_alpha = float(np.mean(non_pm_trades["alpha"])) * trades_per_year * 100 if len(non_pm_trades) > 0 else 0.0

    # Per-sector alpha breakdown
    sector_alphas = {}
    for sector, grp in trades.groupby("commodity_type"):
        if len(grp) >= 10:
            sector_alphas[sector] = {
                "trades": len(grp),
                "alpha_pct": round(float(np.mean(grp["alpha"])) * trades_per_year * 100, 2),
                "hit_rate": round(float((grp["alpha"] > 0).mean()) * 100, 1),
            }
    sector_alphas = dict(sorted(sector_alphas.items(), key=lambda x: x[1]["alpha_pct"], reverse=True))

    # Statistical test: are PM-sector and non-PM-sector alphas different?
    if len(pm_trades) >= 20 and len(non_pm_trades) >= 20:
        t_stat, p_val = sp_stats.ttest_ind(pm_trades["alpha"].values, non_pm_trades["alpha"].values)
        t_stat = float(t_stat)
        p_val = float(p_val)
    else:
        t_stat, p_val = 0.0, 1.0

    # Alpha from ALL trades if we assume PM contributed nothing
    # (i.e., the non-PM sectors should show same alpha if PM is irrelevant)
    alpha_diff = pm_alpha - non_pm_alpha

    if abs(alpha_diff) < 2.0 and p_val > 0.05:
        verdict = f"PM IRRELEVANT — PM-sector alpha ({pm_alpha:+.1f}%) ≈ non-PM alpha ({non_pm_alpha:+.1f}%), p={p_val:.3f}. Alpha comes from commodity/macro features, not PM."
    elif pm_alpha > non_pm_alpha + 5.0 and p_val < 0.05:
        verdict = f"PM CONTRIBUTES — PM-sectors ({pm_alpha:+.1f}%) >> non-PM ({non_pm_alpha:+.1f}%), p={p_val:.3f}. But only 4 real events — fragile."
    elif non_pm_alpha > pm_alpha + 2.0:
        verdict = f"PM NEGATIVE — non-PM sectors ({non_pm_alpha:+.1f}%) > PM sectors ({pm_alpha:+.1f}%), p={p_val:.3f}. PM features may be adding noise."
    else:
        verdict = f"INCONCLUSIVE — PM-sector ({pm_alpha:+.1f}%) vs non-PM ({non_pm_alpha:+.1f}%), p={p_val:.3f}. Need more data."

    return {
        "pm_sector_trades": len(pm_trades),
        "non_pm_sector_trades": len(non_pm_trades),
        "pm_sector_alpha_pct": round(pm_alpha, 2),
        "non_pm_sector_alpha_pct": round(non_pm_alpha, 2),
        "alpha_difference": round(alpha_diff, 2),
        "t_statistic": round(t_stat, 3),
        "p_value": round(p_val, 4),
        "sector_breakdown": sector_alphas,
        "real_pm_events": ["hormuz_closure", "red_sea_disruption", "russia_oil_sanctions", "oil_above_100"],
        "note": "Only 4 of 56 events have real Polymarket IDs. Others default to 0.5 (constant) → zero variance → filtered out by top-15 variance pre-filter.",
        "VERDICT": verdict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TEST 10: TRUE PM ABLATION (zero PM columns, re-run backtest)
# ─────────────────────────────────────────────────────────────────────────────


def test_pm_ablation_true(
    full_result: AlphaAttributionResult,
    ablation_result: AlphaAttributionResult,
    holding_days: int = 20,
) -> dict:
    """TRUE ablation: compare full model vs model with PM columns zeroed.

    This is the definitive test — not a sector proxy.
    If the full model and ablated model produce similar alpha, PM adds nothing.
    If the full model is significantly better, PM features matter.
    """
    trades_per_year = 252 / holding_days

    full_trades = extract_trades(full_result.ticker_results)
    ablated_trades = extract_trades(ablation_result.ticker_results)

    full_alpha = float(np.mean(full_trades["alpha"])) * trades_per_year * 100 if len(full_trades) > 0 else 0.0
    ablated_alpha = float(np.mean(ablated_trades["alpha"])) * trades_per_year * 100 if len(ablated_trades) > 0 else 0.0

    full_sharpe = 0.0
    if len(full_trades) > 1 and np.std(full_trades["net_return"]) > 0:
        full_sharpe = float(np.mean(full_trades["net_return"]) / np.std(full_trades["net_return"], ddof=1) * np.sqrt(trades_per_year))

    ablated_sharpe = 0.0
    if len(ablated_trades) > 1 and np.std(ablated_trades["net_return"]) > 0:
        ablated_sharpe = float(np.mean(ablated_trades["net_return"]) / np.std(ablated_trades["net_return"], ddof=1) * np.sqrt(trades_per_year))

    alpha_diff = full_alpha - ablated_alpha
    sharpe_diff = full_sharpe - ablated_sharpe

    if abs(alpha_diff) < 2.0 and abs(sharpe_diff) < 0.1:
        verdict = f"PM IRRELEVANT (TRUE ABLATION) — full ({full_alpha:+.1f}%) ≈ ablated ({ablated_alpha:+.1f}%), ΔSharpe={sharpe_diff:+.3f}. PM features add nothing."
    elif alpha_diff > 5.0:
        verdict = f"PM CONTRIBUTES (TRUE ABLATION) — full ({full_alpha:+.1f}%) >> ablated ({ablated_alpha:+.1f}%), ΔSharpe={sharpe_diff:+.3f}."
    elif alpha_diff < -2.0:
        verdict = f"PM HARMFUL (TRUE ABLATION) — ablated ({ablated_alpha:+.1f}%) > full ({full_alpha:+.1f}%), ΔSharpe={sharpe_diff:+.3f}. PM adds noise."
    else:
        verdict = f"MARGINAL PM EFFECT — full ({full_alpha:+.1f}%) vs ablated ({ablated_alpha:+.1f}%), ΔSharpe={sharpe_diff:+.3f}."

    return {
        "full_model_alpha_pct": round(full_alpha, 2),
        "full_model_sharpe": round(full_sharpe, 3),
        "full_model_trades": len(full_trades),
        "ablated_model_alpha_pct": round(ablated_alpha, 2),
        "ablated_model_sharpe": round(ablated_sharpe, 3),
        "ablated_model_trades": len(ablated_trades),
        "alpha_difference_pct": round(alpha_diff, 2),
        "sharpe_difference": round(sharpe_diff, 3),
        "VERDICT": verdict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MASTER DIAGNOSTIC RUNNER
# ─────────────────────────────────────────────────────────────────────────────


def run_full_diagnostic(result: AlphaAttributionResult, holding_days: int = 20) -> dict:
    """Run ALL diagnostic tests on a backtest result and produce a unified report."""
    ticker_results = result.ticker_results
    if not ticker_results:
        return {"ERROR": "No ticker results — run the backtest first"}

    trades = extract_trades(ticker_results)
    print(f"\n  Extracted {len(trades)} individual trades from {trades['ticker'].nunique()} tickers")
    print(f"  Date range: {trades['date'].min()} to {trades['date'].max()}")
    print(f"  Mean alpha per trade: {trades['alpha'].mean()*100:.3f}%")
    print(f"  Median alpha per trade: {trades['alpha'].median()*100:.3f}%")

    report = {}

    print("\n" + "=" * 80)
    print("  TEST 1: ANNUALIZATION CROSS-CHECK")
    print("=" * 80)
    report["annualization"] = test_annualization(trades, holding_days=holding_days)
    _print_test(report["annualization"])

    print("\n" + "=" * 80)
    print("  TEST 2: OUTLIER DEPENDENCE")
    print("=" * 80)
    report["outlier_dependence"] = test_outlier_dependence(trades, holding_days=holding_days)
    _print_test(report["outlier_dependence"])

    print("\n" + "=" * 80)
    print("  TEST 3: PERMUTATION TEST (Skill vs Luck)")
    print("=" * 80)
    report["permutation"] = test_permutation(trades, holding_days=holding_days)
    _print_test(report["permutation"])

    print("\n" + "=" * 80)
    print("  TEST 4: REGIME STABILITY")
    print("=" * 80)
    report["regime_stability"] = test_regime_stability(trades, holding_days=holding_days)
    _print_test(report["regime_stability"])

    print("\n" + "=" * 80)
    print("  TEST 5: PER-TICKER CONCENTRATION")
    print("=" * 80)
    report["ticker_concentration"] = test_ticker_concentration(trades, holding_days=holding_days)
    _print_test(report["ticker_concentration"])

    print("\n" + "=" * 80)
    print("  TEST 6: HIT RATE vs MAGNITUDE (Profit Quality)")
    print("=" * 80)
    report["profit_quality"] = test_profit_quality(trades)
    _print_test(report["profit_quality"])

    print("\n" + "=" * 80)
    print("  TEST 7: TEMPORAL AUTOCORRELATION")
    print("=" * 80)
    report["autocorrelation"] = test_autocorrelation(trades)
    _print_test(report["autocorrelation"])

    print("\n" + "=" * 80)
    print("  TEST 8: STATISTICAL SIGNIFICANCE + DEFLATED SHARPE")
    print("=" * 80)
    report["significance"] = test_statistical_significance(trades, holding_days=holding_days)
    _print_test(report["significance"])

    print("\n" + "=" * 80)
    print("  TEST 9: PREDICTION MARKET ABLATION (PM Feature Impact)")
    print("=" * 80)
    report["pm_ablation"] = test_pm_ablation(trades, holding_days=holding_days)
    _print_test(report["pm_ablation"])

    # ── FINAL SUMMARY ──────────────────────────────────────────────────────
    print("\n" + "#" * 80)
    print("#  FINAL HONEST ALPHA ASSESSMENT")
    print("#" * 80)

    verdicts = {k: v.get("VERDICT", "?") for k, v in report.items()}
    for test_name, verdict in verdicts.items():
        pass_words = ["CONSISTENT", "SIGNIFICANT", "ROBUST", "STABLE", "STRONG", "WELL-DISTRIBUTED", "IID"]
        warn_words = ["MODERATE", "MARGINAL", "ADEQUATE", "MIXED", "CAUTIOUS", "INFLATED", "CONCENTRATED"]
        fail_words = ["CORRELATED", "FRAGILE", "NOT SIGNIFICANT", "WEAK", "LOTTERY", "FAILS", "DO NOT"]
        if any(w in verdict.upper() for w in fail_words):
            symbol = "FAIL"
        elif any(w in verdict.upper() for w in warn_words):
            symbol = "WARN"
        elif any(w in verdict.upper() for w in pass_words):
            symbol = "PASS"
        else:
            symbol = "????"
        print(f"  [{symbol:4s}] {test_name:<25s} : {verdict}")

    # Determine the most honest alpha number
    ann = report["annualization"]
    # Use the MEDIAN of all methods as the most honest single number
    honest_alpha = ann.get("median_of_methods", ann["geometric_annual_alpha_pct"])
    engine_alpha = ann["arithmetic_annual_alpha_pct"]

    print(f"\n  Engine-reported alpha:     {engine_alpha:+.2f}% annual")
    print(f"  Geometric alpha:          {ann['geometric_annual_alpha_pct']:+.2f}% annual")
    print(f"  Strategy-minus-bench:     {ann['strategy_minus_bench_annual_pct']:+.2f}% annual")
    print(f"  Bootstrap median:         {ann['bootstrap_median_alpha_pct']:+.2f}% annual")
    print(f"  Bootstrap 90% CI:         [{ann['bootstrap_90ci'][0]:+.2f}%, {ann['bootstrap_90ci'][1]:+.2f}%]")
    print(f"\n  >>> MOST HONEST ALPHA:    {honest_alpha:+.2f}% annual <<<")

    sig = report["significance"]
    print(f"  95% CI (adj):             [{sig['ci_95_pct'][0]:+.2f}%, {sig['ci_95_pct'][1]:+.2f}%]")
    print(f"  p-value (raw, no adj):    {sig['p_value_raw']}")
    print(f"  p-value (autocorr-adj):   {sig['p_value_adjusted']}")
    print(f"  Effective N:              {sig['effective_n']:.1f} (actual: {sig['actual_n']})")
    print(f"  Deflated Sharpe p-value:  {sig['deflated_sharpe_p_value']}")

    # Final verdict
    passes_bar = honest_alpha >= 2.0
    is_significant = sig["p_value_adjusted"] < 0.05
    survives_deflation = sig["deflated_sharpe_p_value"] < 0.10
    ci_above_zero = sig["ci_95_pct"][0] > 0

    print(f"\n  2% bar:         {'PASS' if passes_bar else 'FAIL'}")
    print(f"  Statistically significant: {'YES' if is_significant else 'NO'}")
    print(f"  Survives multiple-testing: {'YES' if survives_deflation else 'NO'}")
    print(f"  95% CI above zero: {'YES' if ci_above_zero else 'NO'}")

    if passes_bar and is_significant and ci_above_zero:
        if survives_deflation:
            final = "INVEST — alpha is real, significant, and survives all tests"
        else:
            final = "CAUTIOUS — alpha passes basic tests but may not survive multiple-testing correction"
    elif is_significant and ci_above_zero and honest_alpha > 0:
        final = "HOLD — positive alpha exists but below 2% bar. Keep monitoring."
    else:
        final = "DO NOT INVEST — cannot verify alpha is real with statistical confidence"

    print(f"\n  >>> FINAL VERDICT: {final} <<<")

    report["final_summary"] = {
        "engine_alpha_pct": round(engine_alpha, 3),
        "honest_alpha_pct": round(honest_alpha, 3),
        "passes_2pct_bar": passes_bar,
        "statistically_significant": is_significant,
        "survives_deflated_sharpe": survives_deflation,
        "ci_95_above_zero": ci_above_zero,
        "FINAL_VERDICT": final,
    }

    return report


def _print_test(result: dict, indent: int = 2) -> None:
    """Pretty-print a test result dict."""
    prefix = " " * indent
    for k, v in result.items():
        if k == "VERDICT":
            print(f"\n{prefix}>>> {k}: {v}")
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
            print(f"{prefix}{k}:")
            for item in v[:8]:
                parts = [f"{ik}={iv}" for ik, iv in item.items()]
                print(f"{prefix}  - {', '.join(parts)}")
            if len(v) > 8:
                print(f"{prefix}  ... ({len(v) - 8} more)")
        elif isinstance(v, list):
            print(f"{prefix}{k}: {v}")
        elif isinstance(v, dict):
            print(f"{prefix}{k}:")
            for ik, iv in v.items():
                print(f"{prefix}  {ik}: {iv}")
        else:
            print(f"{prefix}{k}: {v}")


def main() -> None:
    fast = "--fast" in sys.argv
    subset = None
    horizon = 20  # default 20-day holding period
    for i, arg in enumerate(sys.argv):
        if arg == "--subset" and i + 1 < len(sys.argv):
            subset = int(sys.argv[i + 1])
        if arg == "--horizon" and i + 1 < len(sys.argv):
            horizon = int(sys.argv[i + 1])

    lookback = 400 if fast else 780
    min_preds = 8 if fast else 12

    # Scale min_preds down for long horizons (fewer non-overlapping trades available)
    if horizon >= 120:
        min_preds = max(4, min_preds // 2)

    print("=" * 80)
    print("  CROSSMARKETRISKHUB — ALPHA INTEGRITY DIAGNOSTIC")
    print("=" * 80)
    print(f"  Mode: {'FAST' if fast else 'FULL'} | Lookback: {lookback}d | Min preds: {min_preds}")
    print(f"  Horizon: {horizon}d (holding period)")
    if subset:
        print(f"  Subset: first {subset} tickers")
    print()

    # Run the backtest
    print("  Step 1: Running walk-forward backtest...")
    backtester = SegmentedAlphaBacktester(
        holding_days=horizon,
        min_predictions=min_preds,
        lookback_days=lookback,
        walk_forward_lookback=150,
        min_trades_per_segment=10,
        signal_threshold=0.001,
        verbose=True,
        max_tickers=subset,
    )
    result = backtester.run()

    print(f"\n  Backtest complete: {result.modeled_tickers} tickers, {result.total_trades} trades")
    print(f"  Engine-reported alpha: {result.overall_alpha_pct:+.2f}%")
    print(f"  Engine Sharpe: {result.overall_sharpe:.3f}")
    print(f"  Engine Hit Rate: {result.overall_hit_rate:.1f}%")

    # Run diagnostics
    print(f"\n  Step 2: Running 9 diagnostic tests (horizon={horizon}d)...\n")
    report = run_full_diagnostic(result, holding_days=horizon)

    # Step 3: TRUE PM ABLATION — re-run backtest with PM features zeroed
    skip_ablation = "--no-pm-ablation" in sys.argv
    if not skip_ablation:
        print(f"\n  Step 3: Running TRUE PM ablation (PM features zeroed)...\n")
        ablation_backtester = SegmentedAlphaBacktester(
            holding_days=horizon,
            min_predictions=min_preds,
            lookback_days=lookback,
            walk_forward_lookback=150,
            min_trades_per_segment=10,
            signal_threshold=0.001,
            verbose=False,
            max_tickers=subset,
            zero_pm_features=True,
        )
        ablation_result = ablation_backtester.run()
        print(f"  Ablation complete: {ablation_result.modeled_tickers} tickers, {ablation_result.total_trades} trades")
        print(f"  Ablation alpha: {ablation_result.overall_alpha_pct:+.2f}%")

        print("\n" + "=" * 80)
        print("  TEST 10: TRUE PM ABLATION (PM features zeroed)")
        print("=" * 80)
        report["pm_ablation_true"] = test_pm_ablation_true(result, ablation_result, holding_days=horizon)
        _print_test(report["pm_ablation_true"])
    else:
        print("\n  Step 3: Skipping TRUE PM ablation (--no-pm-ablation flag)")

    # Save report
    out_path = Path(f"data/alpha_diagnostic_{horizon}d_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Deep-convert all numpy types for JSON serialization
    def _deep_convert(obj):
        if isinstance(obj, dict):
            return {k: _deep_convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_deep_convert(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp, datetime)):
            return str(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    try:
        out_path.write_text(
            json.dumps(_deep_convert(report), indent=2),
            encoding="utf-8",
        )
        print(f"\n  Full diagnostic report saved to: {out_path}")
    except Exception as exc:
        print(f"\n  Warning: Could not save JSON report: {exc}")


if __name__ == "__main__":
    main()
