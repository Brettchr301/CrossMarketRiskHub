# Task: Live Signal Monitor — Alerts When Bias Signal Fires

## Classification
| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Security | 0 | Internal alerting |
| Complexity | 2 | Scheduler, state tracking, multi-channel alerts |
| Novelty | 1 | Standard monitoring pattern |
| Blast Radius | 1 | Additive, replaces Discord stub |
| Existing Code | 1 | Integrates with orchestrator.py pipeline |
| **Total** | **5** | |

## Objective

Wire the narrative bias detector into the live pipeline so it runs every poll cycle, tracks signal state over time, and sends actionable alerts when the signal strengthens. Replace the Discord stub with a real notification system.

## Deliverables

- [ ] `app/election/signals/monitor.py` — signal state machine + alerting
- [ ] `app/election/api/routes.py` — add GET /v1/election/signal-status endpoint
- [ ] `tests/test_signal_monitor.py` — tests

## Exact Interface

```python
@dataclass
class SignalState:
    """Tracks signal evolution over time."""
    current_strength: str       # none/weak/moderate/strong/extreme
    previous_strength: str
    strength_changed: bool
    first_detected_at: datetime | None
    last_updated_at: datetime
    consecutive_polls: int      # how many consecutive polls showed this strength
    skew_history: list[float]   # last 20 skew_ratio readings
    similarity_2022: float
    trade_plan: str | None      # formatted trade plan if signal is moderate+


class SignalMonitor:
    """Stateful monitor that tracks narrative bias signal over time."""

    def __init__(self, db_path: str, alert_webhook: str = "", alert_email: str = ""):
        ...

    def poll(self) -> SignalState:
        """Run one detection cycle.

        1. Call detect_narrative_bias() for current cycle
        2. Update signal state
        3. If signal strengthened to moderate+, generate trade plan
        4. If signal_changed and strength >= "moderate", send alert
        5. Return current state
        """

    def send_alert(self, state: SignalState) -> bool:
        """Send alert via configured channels.

        Channels (tried in order):
        1. Discord webhook (if configured)
        2. Log file (always — append to signals.log)

        Alert format:
        🚨 NARRATIVE BIAS SIGNAL: {strength}
        Cycle: 2026 | Platform: Polymarket
        Skew: {skew_ratio:.0%} ({n_underpriced}/{n_competitive} races)
        Avg mispricing: {avg_mispricing_pp:.0f}pp
        2022 similarity: {similarity:.0%}
        {trade_plan}
        """

    def get_status(self) -> dict:
        """Return JSON-serializable status for API endpoint."""


def integrate_with_pipeline(orchestrator_module) -> None:
    """Monkey-patch the orchestrator to include bias detection.

    Adds detect_narrative_bias() call after run_alpha_model()
    in the run_full_pipeline() flow.
    """
```

## API Endpoint

```python
@router.get("/signal-status")
def get_signal_status() -> dict:
    """Current narrative bias signal status.

    Returns:
    {
        "current_strength": "moderate",
        "consecutive_polls": 12,
        "skew_ratio": 0.72,
        "similarity_2022": 0.65,
        "n_competitive_races": 11,
        "n_underpriced": 8,
        "avg_mispricing_pp": 18.5,
        "first_detected_at": "2026-10-15T14:00:00",
        "trade_plan": "=== ELECTION ALPHA TRADE PLAN ===\n...",
    }
    """
```

## Constraints

- DO NOT break existing pipeline — monitor is additive
- Alert deduplication: never send same-strength alert twice in a row
- Skew history capped at 20 entries
- Log file always written regardless of webhook config
- Thread-safe state management

## Tests to Write

1. **test_poll_updates_state**: Two consecutive polls with different strengths. Verify strength_changed=True.
2. **test_consecutive_polls_counter**: 5 polls at "moderate". Verify consecutive_polls=5.
3. **test_alert_fires_on_strengthen**: Signal goes from "weak" to "moderate". Verify alert sent.
4. **test_no_alert_on_same_strength**: Signal stays "moderate". Verify no duplicate alert.
5. **test_skew_history_capped**: After 25 polls, verify skew_history has exactly 20 entries.
6. **test_status_json_serializable**: Verify get_status() output passes json.dumps().
7. **test_trade_plan_generated**: Signal at "strong". Verify trade_plan is not None.

## Files to Touch
- `app/election/signals/monitor.py` — create
- `app/election/api/routes.py` — modify (add 1 endpoint)
- `tests/test_signal_monitor.py` — create

## Success Criteria
1. All 7 tests pass
2. Signal monitor integrates with existing pipeline
3. Alerts fire when signal strengthens
4. GET /signal-status returns well-structured JSON
