"""Run the full segmented alpha attribution backtest and print results.

Usage:
    cd quant-api
    python -m scripts.run_alpha_backtest [--fast] [--subset N]

Flags:
    --fast    : Use smaller lookback (400 days instead of 780) - faster but less history
    --subset N: Only test first N tickers (for quick debugging)
"""
from __future__ import annotations

import json
import sys
import os
from datetime import datetime, UTC
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(str(Path(__file__).resolve().parent.parent))

from app.backtest.alpha_attribution import (
    AlphaAttributionResult,
    SegmentedAlphaBacktester,
    SegmentResult,
    TickerBacktestResult,
)
from app.portfolio.decision_engine import (
    make_investment_decision,
    format_decision_report,
)


def print_header(result: AlphaAttributionResult) -> None:
    print("\n" + "=" * 100)
    print("  CROSSMARKETRISKHUB -- SEGMENTED ALPHA ATTRIBUTION BACKTEST REPORT")
    print("=" * 100)
    print(f"  Run:         {result.run_timestamp}")
    print(f"  Universe:    {result.universe_size} tickers -> {result.modeled_tickers} modeled")
    print(f"  Trades:      {result.total_trades:,}")
    print(f"  Period:      {result.backtest_start} -> {result.backtest_end} ({result.backtest_years:.1f} years)")
    print(f"  Overall a:   {result.overall_alpha_pct:+.2f}% annual (vs SPY)")
    print(f"  Sharpe:      {result.overall_sharpe:.3f}")
    print(f"  Hit Rate:    {result.overall_hit_rate:.1f}%")
    print("=" * 100)


def print_segment_table(dimension: str, segments: list[SegmentResult]) -> None:
    print(f"\n{'-' * 100}")
    title_map = {
        "cap_size": "BY MARKET CAP",
        "geography": "BY GEOGRAPHY",
        "war_proximity": "BY WAR PROXIMITY",
        "exchange_type": "BY EXCHANGE / LANGUAGE",
        "commodity_type": "BY COMMODITY TYPE",
    }
    print(f"  {title_map.get(dimension, dimension.upper())}")
    print(f"{'-' * 100}")

    # Header
    print(f"  {'Segment':<22} {'Ann.a%':>8} {'Sharpe':>8} {'Hit%':>6} {'IR':>7} "
          f"{'MaxDD%':>8} {'Trades':>7} {'Tkrs':>5} {'Bench':>5} {'Years':>6}")
    print(f"  {'-' * 20}  {'-' * 6}  {'-' * 6}  {'-' * 4}  {'-' * 5}  "
          f"{'-' * 6}  {'-' * 5}  {'-' * 3}  {'-' * 3}  {'-' * 4}")

    for seg in segments:
        # Color-code alpha (terminal colors)
        alpha_str = f"{seg.annual_alpha_pct:+.2f}"
        passes = seg.annual_alpha_pct >= 2.0

        marker = " *" if passes else "  "
        print(f"  {seg.label:<22} {alpha_str:>8} {seg.sharpe_ratio:>8.3f} "
              f"{seg.hit_rate_pct:>6.1f} {seg.information_ratio:>7.3f} "
              f"{seg.max_drawdown_pct:>8.2f} {seg.trade_count:>7,} {seg.ticker_count:>5} "
              f"{seg.benchmark_ticker:>5} {seg.backtest_years:>6.1f}{marker}")


def print_segment_detail(seg: SegmentResult) -> None:
    """Print detailed breakdown for a single segment."""
    print(f"\n    +- {seg.label} ({seg.dimension})")
    print(f"    |  Gross Return:    {seg.gross_annual_return_pct:+.2f}% annual")
    print(f"    |  Benchmark ({seg.benchmark_ticker}):  {seg.benchmark_annual_return_pct:+.2f}% annual")
    print(f"    |  Cost Drag:       {seg.cost_drag_annual_pct:.2f}% annual")
    print(f"    |  NET ALPHA:       {seg.annual_alpha_pct:+.2f}% annual")
    print(f"    |  Volatility:      {seg.volatility_annual_pct:.2f}%")
    print(f"    |  Win vs Bench:    {seg.win_rate_vs_benchmark:.1f}%")
    print(f"    |  Avg Market Cap:  ${seg.avg_market_cap:,.0f}")
    print(f"    |  Period:          {seg.first_signal_date} -> {seg.last_signal_date}")
    if seg.top_performers:
        print(f"    |  Top Performers:")
        for tp in seg.top_performers[:3]:
            print(f"    |    {tp['ticker']:<12} a={tp['avg_alpha_per_trade']:+.3f}%/trade  "
                  f"hit={tp['hit_rate']:.1f}%  ({tp['country']}, {tp['commodity_type']})")
    if seg.worst_performers:
        print(f"    |  Worst Performers:")
        for wp in seg.worst_performers[-3:]:
            print(f"    |    {wp['ticker']:<12} a={wp['avg_alpha_per_trade']:+.3f}%/trade  "
                  f"hit={wp['hit_rate']:.1f}%  ({wp['country']}, {wp['commodity_type']})")
    print(f"    +-")


def print_recommendations(result: AlphaAttributionResult) -> None:
    print(f"\n{'=' * 100}")
    print("  ALPHA RECOMMENDATION SUMMARY")
    print(f"{'=' * 100}")

    if result.alpha_segments:
        print(f"\n  * SEGMENTS PASSING 2% ALPHA BAR ({len(result.alpha_segments)}):")
        for seg in result.alpha_segments:
            print(f"    PASS {seg['dimension']}/{seg['label']}: "
                  f"{seg['annual_alpha_pct']:+.2f}% a, "
                  f"Sharpe={seg['sharpe']:.3f}, "
                  f"Hit={seg['hit_rate']:.1f}%, "
                  f"{seg['trade_count']} trades over {seg['backtest_years']:.1f} yrs")
    else:
        print("\n  !! NO SEGMENTS PASS 2% ALPHA BAR")
        print("    Consider: tighter signal threshold, longer holding period, or narrower universe")

    if result.weak_segments:
        print(f"\n  FAIL WEAK SEGMENTS (a < 2%) - consider dropping ({len(result.weak_segments)}):")
        for seg in result.weak_segments[:10]:
            print(f"    - {seg['dimension']}/{seg['label']}: "
                  f"{seg['annual_alpha_pct']:+.2f}% a, "
                  f"{seg['trade_count']} trades")

    # Trading recommendations
    print(f"\n{'-' * 100}")
    print("  INTERACTIVE BROKERS TRADING GUIDANCE:")
    print(f"{'-' * 100}")

    # Find best segments by dimension
    for dim in ["cap_size", "geography", "war_proximity", "exchange_type"]:
        if dim in result.segments and result.segments[dim]:
            best = result.segments[dim][0]
            if best.annual_alpha_pct > 0:
                print(f"  Best {dim.replace('_', ' ')}: {best.label} -> "
                      f"{best.annual_alpha_pct:+.2f}% a (Sharpe {best.sharpe_ratio:.3f})")


def save_report(result: AlphaAttributionResult, path: Path) -> None:
    """Save full results as JSON for further analysis."""
    data = {
        "run_timestamp": result.run_timestamp,
        "universe_size": result.universe_size,
        "modeled_tickers": result.modeled_tickers,
        "total_trades": result.total_trades,
        "backtest_start": result.backtest_start,
        "backtest_end": result.backtest_end,
        "backtest_years": result.backtest_years,
        "overall_alpha_pct": result.overall_alpha_pct,
        "overall_sharpe": result.overall_sharpe,
        "overall_hit_rate": result.overall_hit_rate,
        "alpha_segments": result.alpha_segments,
        "weak_segments": result.weak_segments,
        "segments": {},
    }
    for dim, segs in result.segments.items():
        data["segments"][dim] = [
            {
                "label": s.label,
                "benchmark_ticker": s.benchmark_ticker,
                "ticker_count": s.ticker_count,
                "trade_count": s.trade_count,
                "annual_alpha_pct": s.annual_alpha_pct,
                "sharpe_ratio": s.sharpe_ratio,
                "hit_rate_pct": s.hit_rate_pct,
                "max_drawdown_pct": s.max_drawdown_pct,
                "information_ratio": s.information_ratio,
                "gross_annual_return_pct": s.gross_annual_return_pct,
                "benchmark_annual_return_pct": s.benchmark_annual_return_pct,
                "cost_drag_annual_pct": s.cost_drag_annual_pct,
                "volatility_annual_pct": s.volatility_annual_pct,
                "backtest_years": s.backtest_years,
                "top_performers": s.top_performers,
                "worst_performers": s.worst_performers,
            }
            for s in segs
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(f"\n  Report saved to: {path}")


def _convert_ticker_results(
    ticker_results: list[TickerBacktestResult],
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Convert TickerBacktestResult dataclass instances to dicts for decision engine."""
    ticker_dicts = []
    bench_by_ticker: dict[str, np.ndarray] = {}

    for tr in ticker_results:
        # Compute avg price from returns (approximate)
        avg_volume = 100_000  # default
        ticker_dicts.append({
            "ticker": tr.ticker,
            "commodity_type": tr.commodity_type,
            "country": tr.country,
            "market_cap": tr.market_cap,
            "avg_price": 50.0,  # approximate — decision engine uses this only for share count
            "avg_daily_volume": avg_volume,
            "predictions": tr.predictions,
            "net_returns": tr.net_returns,
        })

        # Use SPY benchmark returns (universal) for robustness testing
        spy_rets = tr.benchmark_returns.get("SPY", [0.0] * len(tr.net_returns))
        bench_by_ticker[tr.ticker] = np.array(spy_rets[:len(tr.net_returns)], dtype=float)

    return ticker_dicts, bench_by_ticker


def main() -> None:
    fast = "--fast" in sys.argv
    subset = None
    for i, arg in enumerate(sys.argv):
        if arg == "--subset" and i + 1 < len(sys.argv):
            subset = int(sys.argv[i + 1])

    lookback = 400 if fast else 780
    # With non-overlapping trades (step=20), 400d lookback gives ~12 predictions.
    # Need min_preds low enough to not filter out all tickers.
    min_preds = 8 if fast else 12

    print(f"Mode: {'FAST' if fast else 'FULL'} | Lookback: {lookback}d | Min predictions: {min_preds}")
    if subset:
        print(f"Subset: first {subset} tickers only")

    backtester = SegmentedAlphaBacktester(
        holding_days=20,
        min_predictions=min_preds,
        lookback_days=lookback,
        walk_forward_lookback=150,
        min_trades_per_segment=10,
        signal_threshold=0.001,  # 0.1% minimum (let the regularized model decide)
        verbose=True,
        max_tickers=subset,
    )

    result = backtester.run()

    # Print report
    print_header(result)

    for dim in ["cap_size", "geography", "war_proximity", "exchange_type", "commodity_type"]:
        if dim in result.segments:
            print_segment_table(dim, result.segments[dim])
            # Print detail for top 3 segments in this dimension
            for seg in result.segments[dim][:3]:
                if seg.annual_alpha_pct > 0:
                    print_segment_detail(seg)

    print_recommendations(result)

    # Save JSON report
    report_path = Path("data/alpha_attribution_report.json")
    save_report(result, report_path)

    # ── INVESTMENT DECISION ENGINE ─────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print("  RUNNING INVESTMENT DECISION ENGINE")
    print(f"{'=' * 100}")

    try:
        ticker_dicts, bench_by_ticker = _convert_ticker_results(result.ticker_results)

        decision = make_investment_decision(
            ticker_results=ticker_dicts,
            benchmark_returns_by_ticker=bench_by_ticker,
            current_positions=None,     # fresh portfolio
            skip_data_fetch=True,       # use backtest data, don't re-fetch fundamentals
        )

        report_text = format_decision_report(decision)
        print(report_text)

        # Save decision report
        decision_path = Path("data/investment_decision_report.txt")
        decision_path.parent.mkdir(parents=True, exist_ok=True)
        decision_path.write_text(report_text, encoding="utf-8")
        print(f"\n  Decision report saved to: {decision_path}")

        # Cache signals for execution layer
        try:
            from app.execution.decision_bridge import save_decision_cache
            cache_path = save_decision_cache(decision)
            print(f"  Execution signal cache saved to: {cache_path}")
        except Exception as cache_exc:
            print(f"  Warning: could not cache signals for execution: {cache_exc}")

    except Exception as exc:
        print(f"\n  Decision engine error: {exc}")
        import traceback
        traceback.print_exc()

    # Exit code: 0 if overall alpha >= 2%, 1 otherwise
    if result.overall_alpha_pct >= 2.0:
        print(f"\n  PASS PASS: Overall alpha {result.overall_alpha_pct:+.2f}% exceeds 2% minimum bar")
        sys.exit(0)
    else:
        print(f"\n  FAIL BELOW BAR: Overall alpha {result.overall_alpha_pct:+.2f}% is below 2% minimum")
        print("    -> Focus on high-alpha segments identified above")
        sys.exit(1)


if __name__ == "__main__":
    main()
