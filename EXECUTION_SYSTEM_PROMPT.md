# Build Prompt: IB Execution & Trade Approval System for CrossMarketRiskHub

## Context

I have a quantitative trading system (`CrossMarketRiskHub`) that produces daily trade recommendations for a sub-$100K commodity micro-cap equity portfolio. The backtesting, modeling, and signal generation are already built. What's missing is the **execution layer** — the system that connects the model's output to Interactive Brokers, manages cash, prioritizes trades, and lets me approve/reject before execution.

### Existing Codebase Structure

```
quant-api/
  app/
    modeling/
      global_scan.py          # Runs the daily scan — produces scored trade candidates
      valuation.py             # Scenario-based DCF/EV valuation
      signals.py               # Signal scoring engine
      cost_model.py            # Transaction cost estimation (spread, slippage, impact)
    portfolio/
      decision_engine.py       # Master module — converts backtest results → trade decisions
      risk_manager.py          # Position sizing (quarter-Kelly), portfolio constraints
      portfolio_constructor.py # Greedy allocation, rebalance signals
      robustness.py            # Statistical tests (bootstrap, deflated Sharpe, etc.)
      quality_filter.py        # ROIC quality gates
      translation_edge.py      # Information asymmetry scoring for foreign tickers
    backtest/
      alpha_attribution.py     # Walk-forward backtest engine
    api/
      routes.py                # FastAPI endpoints
    providers/
      real_prediction.py       # Polymarket/Kalshi prediction market data
  scripts/
    diagnose_alpha.py          # Alpha diagnostic suite (9 statistical tests)
```

### Key Existing Config (from `risk_manager.py`)

```python
class PortfolioConstraints:
    total_capital: float = 75_000.0       # default $75K — MUST be replaced with live IB value
    max_positions: int = 18
    min_position_size: float = 2_000.0    # don't buy less than $2K
    max_position_pct: float = 0.12        # 12% max in single name
    max_sector_pct: float = 0.35          # 35% max in single commodity sector
    max_country_pct: float = 0.40         # 40% max in single country
    max_war_zone_pct: float = 0.15
    cash_reserve_pct: float = 0.10        # always keep 10% cash
    min_hold_days: int = 20               # NO daytrading
    target_hold_days: int = 40
    max_hold_days: int = 120
    min_avg_daily_volume: float = 50_000
    max_portfolio_adv_pct: float = 0.02   # position < 2% of ticker's daily volume
    max_correlation: float = 0.70
```

### What the Model Currently Produces

The `decision_engine.py` outputs a list of `RebalanceSignal` objects:

```python
@dataclass
class RebalanceSignal:
    ticker: str
    action: str              # "BUY", "CLOSE", "REDUCE", "HOLD"
    target_shares: int
    target_dollars: float
    reason: str
    conviction_score: float  # 0-100
    expected_return_pct: float
    downside_p05_pct: float
    kelly_fraction: float
```

---

## What You Need to Build

### Module 1: IB Portfolio Sync (`app/execution/ib_sync.py`)

Connect to Interactive Brokers via `ib_insync` library and pull:

- **Current cash balance** (settled + unsettled, track both)
- **Current positions** (ticker, shares, avg cost, current market value, unrealized P&L)
- **Total portfolio value** (net liquidation value — IB provides this)
- **Pending orders** (any orders still open from previous days)

Store this in a local SQLite database (`data/portfolio_state.db`) so the system has state even when IB is disconnected.

Run this sync at **4:30 PM ET daily** (after market close, prices are final).

Must handle:

- IB Gateway reconnection after daily reset (~11:45 PM ET)
- Weekend/holiday detection (don't run on non-trading days)
- Graceful failure if IB is unreachable (use last known state, flag as stale)

### Module 2: Trade Prioritization Engine (`app/execution/trade_prioritizer.py`)

Takes the raw list of `RebalanceSignal` objects from the decision engine and:

1. **Calculates Expected Value per trade:**

   ```
   EV = (probability_of_profit × avg_expected_win) - (probability_of_loss × avg_expected_loss)
   ```

   Use the model's `expected_return_pct` and `downside_p05_pct` to estimate these.
   Also factor in the `conviction_score` and `kelly_fraction`.

2. **Ranks all candidates by risk-adjusted EV:**

   ```
   Priority Score = EV × conviction_score × (1 - correlation_penalty) × liquidity_factor
   ```

   Where:
   - `correlation_penalty` increases if the candidate is correlated with existing positions
   - `liquidity_factor` decreases for very illiquid names (< 100K avg daily volume)

3. **Allocates cash using a greedy knapsack approach:**
   - Start with available cash (settled cash minus 10% buffer)
   - Take the highest-priority trade first
   - Size it using quarter-Kelly from `risk_manager.py`
   - Check ALL constraints (max 12% single name, max 35% sector, max 40% country, etc.)
   - If it fits, allocate. If not, try the next one.
   - Stop when cash is exhausted or no trades fit constraints.
   - CLOSE/REDUCE signals always execute first (they free up cash for BUYs)

4. **Handles partial fills and sizing:**
   - If the ideal position is $6,000 but only $4,200 is available, offer a reduced size
   - Never go below $2,000 minimum position size
   - Never exceed 2% of the ticker's average daily volume

5. **Produces a prioritized trade plan:**

   ```python
   @dataclass
   class TradePlan:
       timestamp: str
       portfolio_value: float
       cash_available: float
       cash_buffer_target: float
       cash_after_trades: float
       is_below_buffer: bool  # warning flag
       trades: list[PlannedTrade]
       rejected_trades: list[RejectedTrade]  # with reasons
       
   @dataclass
   class PlannedTrade:
       rank: int                    # priority order
       ticker: str
       action: str                  # BUY, CLOSE, REDUCE
       shares: int
       estimated_price: float
       estimated_cost: float        # total dollar cost
       expected_value_pct: float    # EV calculation
       conviction_score: float
       kelly_fraction: float
       risk_flags: list[str]        # any warnings
       constraint_headroom: dict    # how close to each limit
       status: str                  # PENDING_APPROVAL
       
   @dataclass
   class RejectedTrade:
       ticker: str
       action: str
       reason: str                  # "insufficient cash", "exceeds sector limit", etc.
       would_need: float            # how much cash it would need
   ```

### Module 3: Approval Queue (`app/execution/approval_queue.py`)

SQLite-backed approval system:

**Table: `trade_plans`**

```sql
CREATE TABLE trade_plans (
    plan_id TEXT PRIMARY KEY,
    created_at TIMESTAMP,
    portfolio_value REAL,
    cash_available REAL,
    status TEXT  -- 'PENDING', 'PARTIALLY_APPROVED', 'EXECUTED', 'EXPIRED'
);
```

**Table: `planned_trades`**

```sql
CREATE TABLE planned_trades (
    trade_id TEXT PRIMARY KEY,
    plan_id TEXT REFERENCES trade_plans,
    rank INTEGER,
    ticker TEXT,
    action TEXT,
    shares INTEGER,
    limit_price REAL,
    estimated_cost REAL,
    expected_value_pct REAL,
    conviction_score REAL,
    status TEXT,  -- 'PENDING', 'APPROVED', 'REJECTED', 'SUBMITTED', 'FILLED', 'CANCELLED', 'EXPIRED'
    approved_at TIMESTAMP,
    submitted_at TIMESTAMP,
    filled_at TIMESTAMP,
    fill_price REAL,
    fill_shares INTEGER,
    rejection_reason TEXT
);
```

**Rules:**

- Plans expire if not approved within 16 hours (overnight is fine, but not next evening)
- If you approve a CLOSE and a BUY, the BUY order should be queued for T+1 if the BUY depends on cash from the CLOSE (T+1 settlement)
- Modified shares/price should recalculate whether constraints still hold
- Keep full audit trail — never delete, only update status

### Module 4: Order Executor (`app/execution/order_executor.py`)

Submits approved trades to IB at **9:31 AM ET** (1 minute after open to let the opening auction settle):

- Use IB's **Adaptive Algo** order type (minimizes market impact for micro-caps)
- Set limit price = yesterday's close ± 1.5% (long: +1.5%, sell: -1.5%)
- Time-in-force = DAY (expires at 4 PM if unfilled)
- Monitor fills in real-time, update the approval queue DB
- Send fill confirmations

**Safety rails (MANDATORY):**

- Max single order: $10,000
- Max daily order count: 5
- Max daily notional: $25,000
- If portfolio is down >5% from high-water mark: halt all BUY orders, notify me
- If any order would put cash below 5% (emergency buffer below the 10% target): reject
- Paper trade mode flag: logs everything but submits nothing to IB

### Module 5: Nightly Notification (`app/execution/notifier.py`)

Send a formatted summary at **4:45 PM ET** via Discord webhook (or email via SMTP):

```
━━━ CROSSMARKETRISKHUB — DAILY TRADE PLAN ━━━
Date: March 10, 2026

PORTFOLIO STATUS
  Total Value:     $78,200
  Cash:            $12,340
  Positions:       14 of 18 max
  Buffer Target:   $7,820 (10%)
  Available Cash:  $4,520

SUGGESTED TRADES (ranked by EV)
  #1 CLOSE STNG    200sh  est $54.30   EV: +2.1%  Conv: 82
     → Frees $10,860 cash (avail T+1)
     [Reply 1 to APPROVE]

  #2 BUY PTEN      400sh  est $8.12    EV: +3.8%  Conv: 71
     Cost: $3,248  │ Kelly: 0.08  │ Sector: oil (22% → 26%)
     [Reply 2 to APPROVE]

  #3 BUY DNORD.OL  150sh  est $27.33   EV: +2.9%  Conv: 65
     Cost: $4,100  │ Kelly: 0.06  │ Sector: shipping (18% → 24%)
     ⚠️ INSUFFICIENT CASH TODAY — needs STNG sell to settle (T+1)
     [Reply 3 to APPROVE (queued for T+1)]

REJECTED (auto)
  ✗ TGS.OL  — exceeds Norway country limit (38% → 44%)
  ✗ ARCH    — below ROIC quality gate

Reply with numbers to approve (e.g., "1 2 3")
Reply "SKIP" to reject all. Expires in 16 hours.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Module 6: FastAPI Endpoints (`app/api/execution_routes.py`)

Add to the existing FastAPI app:

```
GET  /api/execution/portfolio    → current positions, cash, value
GET  /api/execution/trade-plan   → today's prioritized trade plan
POST /api/execution/approve      → approve specific trades by ID
POST /api/execution/reject       → reject specific trades
POST /api/execution/modify       → change shares or price
GET  /api/execution/history      → past trade plans and fills
GET  /api/execution/performance  → realized P&L, win rate, etc.
```

The existing React dashboard should get a new "Execution" tab with:

- Current portfolio table (positions, P&L, days held)
- Today's trade plan with approve/reject buttons
- Cash waterfall chart (available → after trades → buffer)
- Trade history log

---

## Technical Requirements

- **Python 3.11+**
- **`ib_insync`** for IB API (pip install ib_insync)
- **SQLite** for local state (no external DB needed)
- **APScheduler** for scheduling (pip install apscheduler)
- **Discord webhook** for notifications (just HTTP POST, no library needed)
- **Existing FastAPI app** for API endpoints
- Must integrate with existing `PortfolioConstraints`, `compute_position_size`, and `RebalanceSignal` from the codebase
- All dollar amounts in USD
- All times in US Eastern

## Critical Rules

1. **SELLS BEFORE BUYS.** Always execute CLOSE/REDUCE signals first in the priority queue. They free cash.
2. **T+1 SETTLEMENT.** Cash from today's sells isn't available for buys until tomorrow. Track settled vs unsettled cash separately.
3. **NEVER EXCEED CONSTRAINTS.** Even if I approve a trade, reject it silently if it would violate any constraint at execution time (price might have moved overnight).
4. **PAPER TRADE DEFAULT.** The system must start in paper-trade mode. I will explicitly switch to live when ready.
5. **AUDIT EVERYTHING.** Every decision, every approval, every fill, every rejection — logged with timestamp and reason.
6. **EXPECTED VALUE DRIVES PRIORITY.** Don't just sort by conviction score. Calculate EV = (win_prob × expected_win) - (loss_prob × expected_loss) and use that as the primary ranking, adjusted for correlation and liquidity.
