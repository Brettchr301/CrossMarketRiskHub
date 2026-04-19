# Task: Integrate Direction Detector into Backtest + Arbitrage Engine

## Classification
| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Security | 0 | No auth, no user input — internal data pipeline |
| Complexity | 2 | Multi-file integration across 3 modules, data flow changes |
| Novelty | 1 | Pattern exists in direction_detector.py, extending to consumers |
| Blast Radius | 2 | Changes backtest PnL calculations and arb detection — correctness-critical |
| Existing Code | 2 | Must modify 3 existing files without breaking 16 API routes |
| **Total** | **7** | |

## Objective

The `direction_detector.py` module already detects which party (D/R) a market's YES side represents and can normalize prices to P(Dem wins). However, neither the backtest engine (`backtest/engine.py`) nor the cross-market arbitrage detector (`arbitrage/cross_market.py`) use it. This causes:

1. **Backtest false PnL**: `backtest_cross_market()` at line 134 hardcodes `settlement = 1.0 if outcome["winner_party"] == "D" else 0.0` without knowing if the contract's YES side represents D or R. If a contract asks "Will Republican win?" and the Republican wins, settlement should be 1.0 but the code would set it to 0.0.

2. **Arb false positives**: `cross_market.py` compares `yes_bid` across platforms without checking if both contracts point the same direction. Platform A might be "Will Dem win?" (YES=D) while Platform B is "Will Rep win?" (YES=R). Comparing their YES prices directly is comparing apples to oranges — a 70% YES on A means 70% Dem, but 70% YES on B means 70% Rep (= 30% Dem).

## Deliverables

- [ ] `app/election/arbitrage/cross_market.py` — modify `detect_cross_market_arbs()` to normalize all quotes to P(Dem wins) before comparing cross-platform. Use `detect_direction()` on `platform_question` field, then `normalize_price()` on yes_bid/yes_ask. Skip quotes where direction confidence < 0.5.
- [ ] `app/election/backtest/engine.py` — modify `backtest_cross_market()` to use direction-aware settlement. Look up `HistoricalQuote.question` for each race/platform, call `detect_direction()`, and use `normalize_price()` on the price column. Fix settlement logic to match direction.
- [ ] `app/election/backtest/engine.py` — modify `backtest_outcome_betting()` similarly: normalize prices before computing avg_price, and fix settlement to account for contract direction.
- [ ] `tests/test_direction_integration.py` — comprehensive tests

## Constraints

- DO NOT modify `direction_detector.py` itself — it's already correct
- DO NOT change the `ArbSignal` dataclass or `BacktestResult` dataclass signatures
- DO NOT change API route signatures in `routes.py`
- All existing imports must still work
- The `HistoricalQuote` model has a `question` field (Text) that contains the market question text — use this for direction detection

## Exact Interface Changes

### cross_market.py

Current `_check_pair()` signature (line 58):
```python
def _check_pair(race_id: int, seller: dict, buyer: dict) -> ArbSignal | None:
```

The `seller` and `buyer` dicts already contain a `platform_question` key (or should — verify in `orchestrator.py` line ~140 where quotes are organized). Add direction normalization:

```python
from app.election.mappings.direction_detector import detect_direction, normalize_price

def _check_pair(race_id: int, seller: dict, buyer: dict) -> ArbSignal | None:
    # Detect direction for both contracts
    seller_dir = detect_direction(seller.get("platform_question", ""))
    buyer_dir = detect_direction(buyer.get("platform_question", ""))

    # Skip if either direction is unknown or low confidence
    if seller_dir.confidence < 0.5 or buyer_dir.confidence < 0.5:
        return None

    # Normalize both to P(Dem wins) before comparing
    sell_bid = normalize_price(seller.get("yes_bid", 0.0), seller_dir.yes_party)
    buy_ask = normalize_price(buyer.get("yes_ask", 0.0), buyer_dir.yes_party)
    # ... rest of logic uses normalized prices
```

### backtest/engine.py

In `build_price_panel()` (line 40): also fetch the `question` field and store direction metadata alongside prices. Or better: normalize prices at panel-build time.

In `backtest_cross_market()` (line 130-134): Replace the hardcoded settlement with:
```python
# For each platform column, detect direction from the question text
# Normalize all prices in the panel to P(Dem wins)
# Then settlement = 1.0 if winner_party == "D" else 0.0 is correct
# because all prices now represent P(Dem wins)
```

## Tests to Write

1. **test_direction_normalizes_arb_quotes**: Create two mock quotes — one "Will Democrat win Senate?" (YES=D, bid=0.65) and one "Will Republican win Senate?" (YES=R, bid=0.70). Without normalization, arb detector sees sell@0.70 buy@0.65 = 5% edge. With normalization, the Rep quote normalizes to 0.30 P(Dem), so there's actually NO arb — they agree Dem has 65% and 70% chance. Verify no ArbSignal returned.

2. **test_direction_finds_real_arb**: Two quotes both pointing same direction (both YES=D), one at 0.55 and other at 0.70. After normalization both stay the same. Verify ArbSignal IS returned with correct net_edge.

3. **test_direction_inverts_settlement**: Mock a HistoricalQuote for "Will Republican win?" at price 0.80. Race outcome: winner_party="R". Without normalization, settlement=0.0 (wrong). With normalization, price becomes 0.20 P(Dem), settlement=0.0 (Dem lost), so buying at 0.20 and settling at 0.0 correctly loses $0.20. Verify PnL is negative.

4. **test_direction_unknown_skipped**: Quote with question="" (empty). Verify detect_direction returns confidence=0.0, and the quote is excluded from arb detection.

5. **test_backtest_cross_market_with_normalization**: End-to-end with 2 platforms, opposite-direction contracts for same race. Verify backtest produces different (correct) results vs non-normalized baseline.

6. **test_outcome_betting_normalized**: Test `backtest_outcome_betting()` with a contract "Will Republican win?" where avg_price is 0.75. The favorite is actually the Republican. If Republican wins, PnL should be positive.

## Files to Touch
- `app/election/arbitrage/cross_market.py` — modify
- `app/election/backtest/engine.py` — modify
- `tests/test_direction_integration.py` — create

## Success Criteria
1. All 6 tests pass
2. `detect_cross_market_arbs()` no longer returns false-positive arbs from opposite-direction contracts
3. `backtest_cross_market()` produces correct PnL when contracts have mixed YES=D/YES=R semantics
4. No regressions on existing API routes (import check: `python -c "from app.election.api.routes import router"`)
