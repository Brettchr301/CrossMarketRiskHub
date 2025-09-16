from __future__ import annotations

from app.pipeline import PipelineOrchestrator


def test_pipeline_daily_run_populates_tables(db_session):
    orchestrator = PipelineOrchestrator()
    summary = orchestrator.run_daily(db_session)
    assert summary["signals"] > 0
    assert summary["events"] > 0
    assert summary["commodities"] > 0

