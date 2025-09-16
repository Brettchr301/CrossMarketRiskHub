from __future__ import annotations

from collections.abc import Generator
import os
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# Force test mode to avoid network dependency in integration tests.
os.environ.setdefault("REAL_DATA_ONLY", "false")

from app.api.routes import router
from app.db.models import Base


@pytest.fixture()
def db_session(tmp_path: Path) -> Generator[Session, None, None]:
    db_path = tmp_path / "quant_test.db"
    engine = create_engine(f"sqlite+pysqlite:///{db_path}", future=True)
    Base.metadata.create_all(bind=engine)
    LocalSession = sessionmaker(bind=engine, autoflush=False, autocommit=False, class_=Session)
    db = LocalSession()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()


@pytest.fixture()
def api_client(db_session: Session) -> Generator[TestClient, None, None]:
    app = FastAPI()
    app.include_router(router)

    from app.db.session import get_db

    def _override_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_db
    with TestClient(app) as client:
        yield client
