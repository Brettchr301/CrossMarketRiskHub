from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import router as api_router
from app.config import get_settings
from app.db.init_db import init_db


settings = get_settings()
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Initializing Cross-Market Valuation API database")
    init_db()

    # Initialize execution DB
    try:
        from app.execution.db import init_execution_db
        init_execution_db()
        logger.info("Execution database initialized")
    except Exception as exc:
        logger.warning("Execution DB init failed: %s", exc)

    # Start background news monitor (T3)
    try:
        from app.providers.news_provider import get_or_start_monitor
        get_or_start_monitor()
        logger.info("News monitor started")
    except Exception as exc:
        logger.warning("News monitor failed to start: %s", exc)

    # Start execution scheduler
    scheduler = None
    try:
        from app.execution.scheduler import start_scheduler
        scheduler = start_scheduler()
        logger.info("Execution scheduler started")
    except Exception as exc:
        logger.warning("Execution scheduler not started: %s", exc)

    yield

    # Shutdown scheduler
    if scheduler:
        try:
            scheduler.shutdown(wait=False)
        except Exception:
            pass
    logger.info("Shutting down Cross-Market Valuation API")
    logger.info("Shutting down Cross-Market Valuation API")


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    lifespan=lifespan,
    description="Standalone probability-driven valuation platform for shipping and commodity equities.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "environment": settings.app_env,
        "live_trading_enabled": settings.live_trading_enabled,
    }
