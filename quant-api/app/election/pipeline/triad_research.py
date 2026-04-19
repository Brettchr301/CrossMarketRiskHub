"""Triad PM integration for cheap deep research.

Dispatches research tasks to DeepSeek via MCP Bridge for 28-45x cheaper
data gathering than Claude solo.
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TRIAD_PM_DIR = Path("C:/Users/BrettC/Documents/triad-pm")
MCP_CLI = TRIAD_PM_DIR / "scripts" / "mcp-bridge" / "cli.mjs"
PROJECT_DIR = Path("C:/Users/BrettC/Documents/CrossMarketRiskHub")


def generate_research_spec(
    race_id: int,
    state: str,
    race_type: str,
    candidates: list[str],
) -> str:
    """Generate a research spec markdown for Triad PM dispatch."""
    candidate_list = ", ".join(candidates) if candidates else "TBD"
    return f"""# Election Research: {state} {race_type} (Race #{race_id})

## Task
Gather alternative data signals that correlate with prediction market movements
for the {state} {race_type} race. Candidates: {candidate_list}.

## Data to Collect

### 1. Wikipedia Traffic (REQUIRED)
- Fetch daily page views for each candidate's Wikipedia article for the last 30 days
- Use: https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{{article}}/daily/{{start}}/{{end}}
- Calculate traffic ratio between candidates

### 2. Google Trends (REQUIRED)
- Compare search interest for candidate names in {state}
- Time range: last 3 months
- Note relative interest scores

### 3. Weather Forecast (if near election)
- Check NOAA forecast for {state} on election day
- Use: https://api.weather.gov/points/{{lat}},{{lon}}
- Report precipitation probability and temperature

### 4. Campaign Finance (REQUIRED)
- Check FEC API for latest financials
- Use: https://api.open.fec.gov/v1/candidates/search/?q={{name}}&office=S&api_key=DEMO_KEY
- Report receipts, cash on hand, small donor percentage

## Output Format
Return a JSON object with:
```json
{{
  "race_id": {race_id},
  "state": "{state}",
  "wikipedia_traffic": {{"candidate_a": [...], "candidate_b": [...]}},
  "google_trends": {{"candidate_a": score, "candidate_b": score}},
  "weather": {{"temperature": X, "precipitation_pct": Y}},
  "campaign_finance": {{"candidate_a": {{}}, "candidate_b": {{}}}},
  "signals": [
    {{"name": "...", "value": X, "interpretation": "..."}}
  ]
}}
```
"""


def dispatch_research(
    race_id: int,
    state: str,
    race_type: str,
    candidates: list[str],
    output_path: str | None = None,
) -> dict[str, Any] | None:
    """Dispatch research to DeepSeek via Triad PM MCP Bridge.

    Returns parsed JSON output or None if dispatch fails.
    """
    if not MCP_CLI.exists():
        logger.warning("Triad PM MCP CLI not found at %s", MCP_CLI)
        return None

    spec = generate_research_spec(race_id, state, race_type, candidates)

    # Write spec to temp file
    spec_path = PROJECT_DIR / f"election-research-{race_id}.md"
    spec_path.write_text(spec, encoding="utf-8")

    if output_path is None:
        output_path = str(PROJECT_DIR / f"research-output-{race_id}.json")

    cmd = [
        "node", str(MCP_CLI),
        "--model", "deepseek",
        "--spec", str(spec_path),
        "--project-dir", str(PROJECT_DIR),
        "--output", output_path,
        "--profile", "web",
    ]

    try:
        logger.info("Dispatching Triad research for race %d (%s %s)", race_id, state, race_type)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(TRIAD_PM_DIR),
        )

        if result.returncode != 0:
            logger.error("Triad dispatch failed: %s", result.stderr[:500])
            return None

        # Try to parse output
        output_file = Path(output_path)
        if output_file.exists():
            raw = output_file.read_text(encoding="utf-8")
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Triad output not valid JSON, returning raw text")
                return {"raw_output": raw}

        return {"stdout": result.stdout[:2000]}

    except subprocess.TimeoutExpired:
        logger.error("Triad research timed out for race %d", race_id)
        return None
    except Exception as exc:
        logger.error("Triad dispatch error: %s", exc)
        return None
    finally:
        # Cleanup spec file
        try:
            spec_path.unlink(missing_ok=True)
        except Exception:
            pass
