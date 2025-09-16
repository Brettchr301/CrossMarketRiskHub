from __future__ import annotations

from celery import Celery

from app.config import get_settings


settings = get_settings()
celery_app = Celery(
    "quant_platform",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    timezone="UTC",
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    beat_schedule={
        "daily-scenario-run": {
            "task": "app.workers.tasks.run_daily_pipeline",
            "schedule": 60 * 60 * 24,
        },
        "event-triggered-run": {
            "task": "app.workers.tasks.run_event_triggered_pipeline",
            "schedule": 60 * 30,
        },
        "quarterly-fundamental-refresh": {
            "task": "app.workers.tasks.run_quarterly_fundamentals",
            "schedule": 60 * 60 * 24 * 14,
        },
    },
)

