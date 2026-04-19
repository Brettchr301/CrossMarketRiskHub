"""Live Signal Monitor — tracks narrative bias signal and sends alerts.

Runs as part of the election pipeline, maintains signal state across
poll cycles, and fires alerts when the signal strengthens.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

SIGNALS_LOG = Path(__file__).parent.parent / "signals.log"
MAX_SKEW_HISTORY = 20

STRENGTH_ORDER = {"none": 0, "weak": 1, "moderate": 2, "strong": 3, "extreme": 4}


@dataclass
class SignalState:
    """Tracks signal evolution over time."""
    current_strength: str = "none"
    previous_strength: str = "none"
    strength_changed: bool = False
    first_detected_at: datetime | None = None
    last_updated_at: datetime = field(default_factory=datetime.utcnow)
    consecutive_polls: int = 0
    skew_history: list[float] = field(default_factory=list)
    similarity_2022: float = 0.0
    trade_plan: str | None = None
    n_competitive: int = 0
    n_underpriced: int = 0
    avg_mispricing_pp: float = 0.0
    skew_ratio: float = 0.0


class SignalMonitor:
    """Stateful monitor that tracks narrative bias signal over time."""

    def __init__(
        self,
        db_path: str = "",
        alert_webhook: str = "",
        cycle: int = 2026,
        platform: str = "polymarket",
    ):
        self.db_path = db_path
        self.alert_webhook = alert_webhook
        self.cycle = cycle
        self.platform = platform
        self._state = SignalState()

    @property
    def state(self) -> SignalState:
        return self._state

    def poll(self, bias_signal: dict[str, Any] | None = None) -> SignalState:
        """Run one detection cycle.

        If bias_signal is provided, uses it directly. Otherwise would call
        detect_narrative_bias() (requires DB session).
        """
        if bias_signal is None:
            # In production, this would call detect_narrative_bias(db, cycle, platform)
            logger.warning("No bias_signal provided, skipping poll")
            return self._state

        new_strength = bias_signal.get("signal_strength", "none")
        old_strength = self._state.current_strength
        skew = bias_signal.get("skew_ratio", 0.0)

        # Update state
        self._state.previous_strength = old_strength
        self._state.current_strength = new_strength
        self._state.strength_changed = (new_strength != old_strength)
        self._state.last_updated_at = datetime.utcnow()
        self._state.similarity_2022 = bias_signal.get("analog_2022_similarity", 0.0)
        self._state.n_competitive = bias_signal.get("n_competitive_races", 0)
        self._state.n_underpriced = bias_signal.get("n_dem_underpriced", 0)
        self._state.avg_mispricing_pp = bias_signal.get("avg_mispricing_pp", 0.0)
        self._state.skew_ratio = skew

        # Consecutive polls at same strength
        if new_strength == old_strength:
            self._state.consecutive_polls += 1
        else:
            self._state.consecutive_polls = 1

        # First detection tracking
        if new_strength != "none" and self._state.first_detected_at is None:
            self._state.first_detected_at = datetime.utcnow()
        elif new_strength == "none":
            self._state.first_detected_at = None

        # Skew history (capped)
        self._state.skew_history.append(skew)
        if len(self._state.skew_history) > MAX_SKEW_HISTORY:
            self._state.skew_history = self._state.skew_history[-MAX_SKEW_HISTORY:]

        # Generate trade plan for moderate+ signals
        if STRENGTH_ORDER.get(new_strength, 0) >= STRENGTH_ORDER["moderate"]:
            self._state.trade_plan = bias_signal.get("trade_plan")
        else:
            self._state.trade_plan = None

        # Alert if signal strengthened to moderate+
        strengthened = (
            STRENGTH_ORDER.get(new_strength, 0) > STRENGTH_ORDER.get(old_strength, 0)
            and STRENGTH_ORDER.get(new_strength, 0) >= STRENGTH_ORDER["moderate"]
        )
        if strengthened:
            self.send_alert(self._state)

        # Always log
        self._log_state()

        return self._state

    def send_alert(self, state: SignalState) -> bool:
        """Send alert via configured channels."""
        message = self._format_alert(state)

        # Always log to file
        try:
            with open(SIGNALS_LOG, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"{datetime.utcnow().isoformat()}\n")
                f.write(message)
                f.write("\n")
        except Exception as exc:
            logger.error("Failed to write signal log: %s", exc)

        # Discord webhook
        if self.alert_webhook:
            try:
                requests.post(
                    self.alert_webhook,
                    json={"content": message},
                    timeout=10,
                )
                logger.info("Discord alert sent")
                return True
            except Exception as exc:
                logger.warning("Discord alert failed: %s", exc)

        return True

    def _format_alert(self, state: SignalState) -> str:
        """Format alert message."""
        lines = [
            f"NARRATIVE BIAS SIGNAL: {state.current_strength.upper()}",
            f"Cycle: {self.cycle} | Platform: {self.platform}",
            f"Skew: {state.skew_ratio:.0%} ({state.n_underpriced}/{state.n_competitive} races)",
            f"Avg mispricing: {state.avg_mispricing_pp:.0f}pp",
            f"2022 similarity: {state.similarity_2022:.0%}",
            f"Consecutive polls: {state.consecutive_polls}",
        ]
        if state.trade_plan:
            lines.append("")
            lines.append(state.trade_plan)
        return "\n".join(lines)

    def _log_state(self) -> None:
        """Log current state for debugging."""
        logger.info(
            "Signal: %s (prev: %s) | skew=%.2f | %d/%d races | %dpp avg | 2022_sim=%.2f",
            self._state.current_strength,
            self._state.previous_strength,
            self._state.skew_ratio,
            self._state.n_underpriced,
            self._state.n_competitive,
            self._state.avg_mispricing_pp,
            self._state.similarity_2022,
        )

    def get_status(self) -> dict[str, Any]:
        """Return JSON-serializable status for API endpoint."""
        return {
            "current_strength": self._state.current_strength,
            "previous_strength": self._state.previous_strength,
            "strength_changed": self._state.strength_changed,
            "consecutive_polls": self._state.consecutive_polls,
            "skew_ratio": self._state.skew_ratio,
            "n_competitive_races": self._state.n_competitive,
            "n_underpriced": self._state.n_underpriced,
            "avg_mispricing_pp": self._state.avg_mispricing_pp,
            "similarity_2022": self._state.similarity_2022,
            "first_detected_at": self._state.first_detected_at.isoformat() if self._state.first_detected_at else None,
            "last_updated_at": self._state.last_updated_at.isoformat(),
            "trade_plan": self._state.trade_plan,
            "skew_history": self._state.skew_history[-10:],  # last 10 for API
        }
