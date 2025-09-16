from __future__ import annotations

from app.db.session import SessionLocal
from app.pipeline import PipelineOrchestrator
from app.workers.celery_app import celery_app


orchestrator = PipelineOrchestrator()


@celery_app.task(name="app.workers.tasks.run_daily_pipeline")
def run_daily_pipeline():
    with SessionLocal() as db:
        return orchestrator.run_daily(db)


@celery_app.task(name="app.workers.tasks.run_event_triggered_pipeline")
def run_event_triggered_pipeline():
    with SessionLocal() as db:
        return orchestrator.run_event_triggered(db)


@celery_app.task(name="app.workers.tasks.run_quarterly_fundamentals")
def run_quarterly_fundamentals():
    with SessionLocal() as db:
        return orchestrator.run_quarterly_fundamentals(db)

