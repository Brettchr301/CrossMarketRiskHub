# Task: Position Sizer — Kelly Criterion + Portfolio Construction

## Classification
| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Security | 0 | Internal analytics |
| Complexity | 2 | Kelly math, correlation adjustment, portfolio optimization |
| Novelty | 1 | Standard quant portfolio construction |
| Blast Radius | 1 | Additive module |
| Existing Code | 1 | Consumes NarrativeBiasSignal, AnalogMatch |
| **Total** | **5** | |

## Objective

Given a narrative bias signal and analog matches, compute optimal position sizes per race using Kelly criterion with correlation adjustments. The 2022 signal had 9 correlated races — naive Kelly would oversize because of the correlation.

## Deliverables

- [ ] `app/election/signals/position_sizer.py` — sizing engine
- [ ] `tests/test_position_sizer.py` — tests

## Exact Interface

```python
@dataclass
class PositionRecommendation:
    race_id: int | None
    state: str
    race_type: str
    platform: str
    direction: str              # "BUY_DEM" or "BUY_REP"
    current_market_prob: float  # current market P(Dem wins)
    estimated_true_prob: float  # our estimate of true P(Dem wins)
    edge_pp: float              # estimated_true_prob - current_market_prob (in pp)
    kelly_fraction: float       # raw Kelly bet fraction (0-1)
    adjusted_fraction: float    # Kelly / n_correlated_races (correlation adjustment)
    notional_usd: float         # adjusted_fraction × bankroll
    expected_pnl: float         # edge × notional
    max_loss: float             # notional × current_market_prob


@dataclass
class PortfolioRecommendation:
    as_of: datetime
    bankroll: float
    signal_strength: str
    n_positions: int
    total_notional: float
    total_expected_pnl: float
    total_max_loss: float
    expected_sharpe: float
    positions: list[PositionRecommendation]


def compute_kelly_fraction(
    estimated_prob: float,
    market_prob: float,
    fee_pct: float = 0.003,
) -> float:
    """Compute Kelly criterion fraction for a binary bet.

    Kelly = (p * b - q) / b
    where p = estimated_prob, q = 1 - p, b = payout odds = (1/market_prob - 1)

    Returns fraction of bankroll to bet. Capped at 0.25 (quarter-Kelly for safety).
    """


def build_portfolio(
    bias_signal: dict,          # NarrativeBiasSignal as dict
    analog_matches: dict,       # from match_2026_to_analogs
    bankroll: float = 10000.0,
    max_position_pct: float = 0.15,   # max 15% per race
    max_portfolio_pct: float = 0.60,  # max 60% total exposure
    correlation_factor: float = 0.7,  # estimated cross-race correlation
    kelly_divisor: float = 4.0,       # quarter-Kelly for safety
) -> PortfolioRecommendation:
    """Build a portfolio of positions from bias signal + analog forecasts.

    Steps:
    1. For each race in bias_signal.races where market_vs_polls < -10pp:
       - estimated_true_prob = polling_avg (or market + analog_error * 0.5)
       - edge = estimated_true_prob - market_prob
       - kelly = compute_kelly_fraction(estimated_true_prob, market_prob)
       - adjusted = kelly / kelly_divisor / sqrt(n_correlated)
       - notional = min(adjusted * bankroll, max_position_pct * bankroll)
    2. Sort by edge descending
    3. Cap total notional at max_portfolio_pct * bankroll
    4. Compute portfolio-level expected PnL and Sharpe
    """


def format_trade_plan(portfolio: PortfolioRecommendation) -> str:
    """Format portfolio as human-readable trade plan.

    Example output:
    === ELECTION ALPHA TRADE PLAN ===
    Signal: STRONG narrative bias (2022 similarity: 0.85)
    Bankroll: $10,000 | Total exposure: $4,200 (42%)

    #1 PA Senate — BUY DEM @ $0.38 (est true: $0.65)
       Edge: 27pp | Size: $1,050 (10.5%) | E[PnL]: +$283

    #2 AZ Governor — BUY DEM @ $0.26 (est true: $0.55)
       Edge: 29pp | Size: $900 (9.0%) | E[PnL]: +$261
    ...
    """
```

## Constraints

- DO NOT modify any existing files
- Kelly fraction must be capped at 0.25 (quarter-Kelly safety)
- No single position may exceed max_position_pct of bankroll
- Total portfolio exposure must not exceed max_portfolio_pct of bankroll
- Must handle edge cases: zero edge, negative edge, estimated_prob outside [0,1]

## Tests to Write

1. **test_kelly_fair_bet**: estimated_prob=0.5, market_prob=0.5 → kelly=0.
2. **test_kelly_edge_exists**: estimated_prob=0.65, market_prob=0.38 → kelly > 0.
3. **test_kelly_capped**: Even with huge edge, kelly ≤ 0.25.
4. **test_portfolio_max_exposure**: Total notional ≤ max_portfolio_pct × bankroll.
5. **test_correlation_reduces_size**: With 9 correlated races, each position smaller than with 1 race.
6. **test_trade_plan_format**: Verify format_trade_plan returns non-empty string with key fields.
7. **test_no_positions_when_no_signal**: Balanced signal (skew=0.5) → 0 positions.

## Files to Touch
- `app/election/signals/position_sizer.py` — create
- `tests/test_position_sizer.py` — create

## Success Criteria
1. All 7 tests pass
2. For 2022-like signal with $10K bankroll, portfolio allocates $3-6K across 5-9 races
3. No single position exceeds 15% of bankroll
4. Trade plan is human-readable and actionable
