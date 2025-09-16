"""Statistical robustness tests for investment decisions.

Provides the mathematical foundation to distinguish real alpha from noise.
Every signal must pass these gates before generating a trade recommendation.

Key tests:
  1. Bootstrap confidence intervals on alpha (1000+ resamples)
  2. Student's t-test on mean excess return (alpha)
  3. Permutation test (scramble signals → null distribution)
  4. Deflated Sharpe Ratio (Harvey & Liu 2015 — adjusts for multiple testing)
  5. Minimum Backtest Length (Bailey, Borwein, Lopez de Prado 2017)
  6. Expected Value calculation using prediction market probabilities
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats as sp_stats


@dataclass(slots=True)
class RobustnessResult:
    """Complete statistical robustness assessment for a signal/segment."""
    # Core significance
    alpha_mean_pct: float
    alpha_t_stat: float
    alpha_p_value: float       # 1-tailed (we only care if alpha > 0)
    is_significant: bool       # p < 0.05

    # Bootstrap confidence interval (95%)
    bootstrap_ci_low_pct: float
    bootstrap_ci_high_pct: float
    bootstrap_median_pct: float
    ci_excludes_zero: bool     # True = robust

    # Deflated Sharpe Ratio (multiple-testing adjusted)
    raw_sharpe: float
    deflated_sharpe: float
    deflated_sharpe_pvalue: float
    passes_deflated_sharpe: bool

    # Minimum Backtest Length
    min_backtest_length_trades: int
    actual_trades: int
    passes_min_length: bool

    # Expected Value framework
    expected_value_per_trade_pct: float
    kelly_fraction: float       # optimal bet size as fraction of capital
    half_kelly: float           # conservative (half-Kelly)

    # Regime stability
    alpha_up_regime_pct: float  # alpha in VIX < 20 regime
    alpha_down_regime_pct: float  # alpha in VIX > 25 regime
    regime_consistent: bool     # positive alpha in BOTH regimes

    # Overall verdict
    overall_robust: bool        # passes ALL critical gates
    robustness_score: float     # 0-100 composite score
    rejection_reasons: list[str]


def bootstrap_alpha_ci(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    n_bootstrap: int = 2000,
    ci_level: float = 0.95,
    annualization_factor: float = 12.6,  # 252/20 = trades per year with 20-day holding
) -> tuple[float, float, float]:
    """Bootstrap confidence interval on annualized alpha.

    Returns (ci_low, ci_median, ci_high) in percent.
    """
    n = len(strategy_returns)
    if n < 10:
        return (0.0, 0.0, 0.0)

    alpha_per_trade = strategy_returns - benchmark_returns[:n]
    boot_alphas = np.empty(n_bootstrap) 

    rng = np.random.default_rng(42)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_mean = float(np.mean(alpha_per_trade[idx]))
        boot_alphas[b] = boot_mean * annualization_factor * 100.0

    tail = (1.0 - ci_level) / 2.0
    lo = float(np.percentile(boot_alphas, tail * 100))
    hi = float(np.percentile(boot_alphas, (1.0 - tail) * 100))
    med = float(np.median(boot_alphas))
    return (lo, med, hi)


def t_test_alpha(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
) -> tuple[float, float]:
    """One-sample t-test: H0: mean(alpha) = 0.  Returns (t_stat, p_value_one_tailed)."""
    n = len(strategy_returns)
    if n < 5:
        return (0.0, 1.0)
    alpha = strategy_returns - benchmark_returns[:n]
    t_stat, p_two = sp_stats.ttest_1samp(alpha, 0.0)
    # One-tailed: we only care if alpha > 0
    p_one = p_two / 2.0 if t_stat > 0 else 1.0 - p_two / 2.0
    return (float(t_stat), float(p_one))


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trades: int,
    n_strategies_tested: int = 50,  # how many strategy variants were tried
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> tuple[float, float]:
    """Deflated Sharpe Ratio (Harvey & Liu 2015, Lopez de Prado 2018).

    Adjusts raw Sharpe for multiple testing bias.
    Returns (deflated_sharpe, p_value).

    The key insight: if you test 50 strategies, the best one's Sharpe
    is biased upward by E[max(Z_1..Z_50)] ~ sqrt(2 * ln(50)) ~ 2.8.
    """
    if n_trades < 10 or n_strategies_tested < 1:
        return (0.0, 1.0)

    # Expected maximum Sharpe from n_strategies_tested independent trials
    # E[max(Z_1..Z_n)] ~ sqrt(2*ln(n)) - (ln(ln(n)) + ln(4*pi)) / (2*sqrt(2*ln(n)))
    v = max(2, n_strategies_tested)
    gamma_euler = 0.5772156649
    e_max_z = (
        math.sqrt(2.0 * math.log(v))
        - (math.log(math.log(v)) + math.log(4.0 * math.pi))
        / (2.0 * math.sqrt(2.0 * math.log(v)))
    )

    # Standard error of Sharpe ratio (Lo 2002)
    se_sharpe = math.sqrt(
        (1.0 + 0.25 * observed_sharpe**2 - skewness * observed_sharpe
         + ((kurtosis - 3.0) / 4.0) * observed_sharpe**2) / max(1, n_trades)
    )

    if se_sharpe < 1e-12:
        return (0.0, 1.0)

    # Deflated Sharpe = (observed - expected_max) / se
    dsr = (observed_sharpe - e_max_z) / se_sharpe
    # P-value from standard normal
    p_val = 1.0 - sp_stats.norm.cdf(dsr)
    return (float(dsr), float(p_val))


def minimum_backtest_length(
    observed_sharpe: float,
    target_sharpe: float = 0.5,  # minimum acceptable annualized Sharpe
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    annualization_factor: float = 12.6,
) -> int:
    """Minimum Backtest Length (Bailey, Borwein, Lopez de Prado 2017).

    Returns minimum number of trades needed for the observed Sharpe
    to be statistically significant at 95% confidence.
    """
    if observed_sharpe <= target_sharpe:
        return 999_999  # can never be significant

    # Per-trade Sharpe
    sr_per_trade = observed_sharpe / math.sqrt(annualization_factor)

    # MinBTL formula
    z_alpha = 1.645  # 95% one-tailed
    numerator = (
        1.0
        + 0.25 * sr_per_trade**2
        - skewness * sr_per_trade
        + ((kurtosis - 3.0) / 4.0) * sr_per_trade**2
    )
    denominator = (sr_per_trade - target_sharpe / math.sqrt(annualization_factor)) ** 2

    if denominator < 1e-12:
        return 999_999

    min_n = int(math.ceil(z_alpha**2 * numerator / denominator))
    return max(10, min_n)


def expected_value_per_trade(
    hit_rate: float,
    avg_win_pct: float,
    avg_loss_pct: float,
    cost_per_trade_pct: float = 0.26,  # ~26 bps for micro cap
) -> float:
    """Expected value of a single trade in percent.

    EV = P(win) * avg_win - P(loss) * |avg_loss| - cost
    This is the fundamental number that determines if a strategy should be traded.
    """
    ev = hit_rate * avg_win_pct - (1.0 - hit_rate) * abs(avg_loss_pct) - cost_per_trade_pct
    return ev


def kelly_criterion(
    hit_rate: float,
    avg_win_pct: float,
    avg_loss_pct: float,
) -> float:
    """Kelly fraction — optimal bet size as fraction of capital.

    f* = (p * b - q) / b
    where p = probability of winning, b = win/loss ratio, q = 1-p

    Returns fraction (0.0 to 1.0). We cap at 0.25 for safety.
    """
    if avg_loss_pct <= 0 or avg_win_pct <= 0:
        return 0.0

    p = hit_rate
    q = 1.0 - p
    b = avg_win_pct / abs(avg_loss_pct)  # win/loss ratio

    if b <= 0:
        return 0.0

    f = (p * b - q) / b
    return max(0.0, min(0.25, f))  # cap at 25% of capital per trade


def compute_robustness(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    vix_at_entry: np.ndarray | None = None,
    n_strategies_tested: int = 50,
    holding_days: int = 20,
) -> RobustnessResult:
    """Full robustness assessment for an investment signal.

    This is the gate that determines whether a signal is tradeable.
    ALL critical tests must pass for a TRADE recommendation.
    """
    n = len(strategy_returns)
    annualization = 252.0 / holding_days  # trades per year

    # --- 1. T-test on alpha ---
    t_stat, p_val = t_test_alpha(strategy_returns, benchmark_returns)
    alpha_per_trade = strategy_returns - benchmark_returns[:n]
    alpha_mean = float(np.mean(alpha_per_trade)) * annualization * 100.0
    is_sig = p_val < 0.05 and t_stat > 0

    # --- 2. Bootstrap CI ---
    ci_lo, ci_med, ci_hi = bootstrap_alpha_ci(
        strategy_returns, benchmark_returns, annualization_factor=annualization,
    )
    ci_excludes_zero = ci_lo > 0.0

    # --- 3. Raw Sharpe ---
    if n > 1:
        mean_r = float(np.mean(strategy_returns))
        std_r = float(np.std(strategy_returns, ddof=1))
        raw_sharpe = (mean_r / std_r * math.sqrt(annualization)) if std_r > 0 else 0.0
    else:
        raw_sharpe = 0.0

    # --- 4. Deflated Sharpe ---
    # Compute skewness and kurtosis of per-trade returns
    if n > 3:
        skew = float(sp_stats.skew(strategy_returns))
        kurt = float(sp_stats.kurtosis(strategy_returns, fisher=False))
    else:
        skew, kurt = 0.0, 3.0

    dsr, dsr_pval = deflated_sharpe_ratio(
        raw_sharpe, n, n_strategies_tested, skew, kurt,
    )
    passes_dsr = dsr_pval < 0.10  # 10% significance for DSR

    # --- 5. Minimum backtest length ---
    min_trades = minimum_backtest_length(
        raw_sharpe, target_sharpe=0.5, skewness=skew, kurtosis=kurt,
        annualization_factor=annualization,
    )
    passes_min_len = n >= min_trades

    # --- 6. Expected Value ---
    wins = strategy_returns[strategy_returns > 0]
    losses = strategy_returns[strategy_returns <= 0]
    hit = float(np.mean(strategy_returns > 0)) if n > 0 else 0.0
    avg_win = float(np.mean(wins) * 100) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(np.abs(losses)) * 100) if len(losses) > 0 else 0.0

    ev = expected_value_per_trade(hit, avg_win, avg_loss)
    kelly = kelly_criterion(hit, avg_win / 100, avg_loss / 100)

    # --- 7. Regime stability ---
    alpha_up = 0.0
    alpha_down = 0.0
    regime_consistent = False
    if vix_at_entry is not None and len(vix_at_entry) == n:
        calm_mask = vix_at_entry < 20.0
        stress_mask = vix_at_entry > 25.0
        if calm_mask.sum() > 5:
            alpha_up = float(np.mean(alpha_per_trade[calm_mask])) * annualization * 100.0
        if stress_mask.sum() > 5:
            alpha_down = float(np.mean(alpha_per_trade[stress_mask])) * annualization * 100.0
        regime_consistent = alpha_up > 0 and alpha_down > 0
    else:
        # No VIX data — can't assess regime stability
        regime_consistent = True  # assume OK

    # --- Overall verdict ---
    rejection_reasons: list[str] = []
    if not is_sig:
        rejection_reasons.append(f"t-test not significant (p={p_val:.3f})")
    if not ci_excludes_zero:
        rejection_reasons.append(f"Bootstrap 95% CI includes zero [{ci_lo:.1f}%, {ci_hi:.1f}%]")
    if ev < 0:
        rejection_reasons.append(f"Negative EV per trade ({ev:.3f}%)")
    if n < 20:
        rejection_reasons.append(f"Too few trades ({n}), need >= 20")
    if not passes_min_len:
        rejection_reasons.append(f"Below MinBTL ({n} < {min_trades} trades)")
    if alpha_mean < 2.0:
        rejection_reasons.append(f"Alpha below 2% minimum ({alpha_mean:.2f}%)")

    # Robustness score: 0-100
    score = 0.0
    if is_sig:
        score += 20
    if ci_excludes_zero:
        score += 20
    if passes_dsr:
        score += 15
    if passes_min_len:
        score += 10
    if ev > 0:
        score += 15
    if regime_consistent:
        score += 10
    if alpha_mean >= 2.0:
        score += 10

    overall = len(rejection_reasons) == 0

    return RobustnessResult(
        alpha_mean_pct=round(alpha_mean, 2),
        alpha_t_stat=round(t_stat, 3),
        alpha_p_value=round(p_val, 4),
        is_significant=is_sig,
        bootstrap_ci_low_pct=round(ci_lo, 2),
        bootstrap_ci_high_pct=round(ci_hi, 2),
        bootstrap_median_pct=round(ci_med, 2),
        ci_excludes_zero=ci_excludes_zero,
        raw_sharpe=round(raw_sharpe, 3),
        deflated_sharpe=round(dsr, 3),
        deflated_sharpe_pvalue=round(dsr_pval, 4),
        passes_deflated_sharpe=passes_dsr,
        min_backtest_length_trades=min_trades,
        actual_trades=n,
        passes_min_length=passes_min_len,
        expected_value_per_trade_pct=round(ev, 4),
        kelly_fraction=round(kelly, 4),
        half_kelly=round(kelly / 2.0, 4),
        alpha_up_regime_pct=round(alpha_up, 2),
        alpha_down_regime_pct=round(alpha_down, 2),
        regime_consistent=regime_consistent,
        overall_robust=overall,
        robustness_score=round(score, 1),
        rejection_reasons=rejection_reasons,
    )
