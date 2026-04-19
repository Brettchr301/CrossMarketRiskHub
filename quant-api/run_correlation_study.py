"""Run event-study correlation analysis across all historical cycles."""
from app.election.db.session import get_session_factory
from app.election.backtest.correlation_study import (
    compute_price_accuracy,
    weather_price_correlation_per_race,
    price_volatility_stats,
    cross_cycle_summary,
)
import pandas as pd
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)

db = get_session_factory()()

print("=" * 80)
print("ELECTION PREDICTION MARKET EVENT-STUDY REPORT")
print("=" * 80)

for cycle in [2018, 2020, 2022, 2024]:
    print(f"\n--- Cycle {cycle} ---")

    acc = compute_price_accuracy(db, cycle)
    if not acc.empty:
        print(f"\nPrice Accuracy ({len(acc)} races):")
        print(f"  Mean Brier score: {acc['brier_score'].mean():.4f}")
        print(f"  Worst predictions (top 3 by error):")
        worst = acc.nlargest(3, 'brier_score')[['state', 'race_type', 'final_prob', 'winner', 'brier_score']]
        print(worst.to_string(index=False))

    corr = weather_price_correlation_per_race(db, cycle)
    if not corr.empty:
        print(f"\nWeather-Price Correlation ({len(corr)} races):")
        print(f"  Mean correlation: {corr['correlation'].mean():.4f}")
        print(f"  Strongest correlations:")
        strongest = corr.reindex(corr['correlation'].abs().nlargest(3).index)
        print(strongest[['state', 'race_type', 'correlation', 'n_hours']].to_string(index=False))

    vol = price_volatility_stats(db, cycle)
    if not vol.empty:
        print(f"\nVolatility ({len(vol)} races):")
        print(f"  Mean pre-election vol: {vol['pre_vol'].mean():.5f}")
        print(f"  Mean post-election vol: {vol['post_vol'].mean():.5f}")
        print(f"  Mean election-night total move: {vol['election_night_total_move'].abs().mean():.4f}")
        print(f"  Biggest election night moves:")
        biggest = vol.reindex(vol['election_night_total_move'].abs().nlargest(3).index)
        print(biggest[['state', 'race_type', 'election_night_total_move', 'max_1h_move']].to_string(index=False))

print("\n" + "=" * 80)
print("CROSS-CYCLE SUMMARY")
print("=" * 80)
summary = cross_cycle_summary(db, [2018, 2020, 2022, 2024])
import json
print(json.dumps(summary, indent=2, default=str))

db.close()
