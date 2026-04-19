from __future__ import annotations
import threading
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.election.config import get_election_settings
from app.election.db.models import ElectionBase

_lock = threading.Lock()
_engine = None
_SessionLocal = None


def _get_engine():
    global _engine
    if _engine is None:
        settings = get_election_settings()
        url = f"sqlite:///{settings.election_db_path}"
        _engine = create_engine(url, echo=False, pool_pre_ping=True)
        # Enable WAL mode for safe concurrent reads during OneDrive sync
        @event.listens_for(_engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()
    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=_get_engine(), expire_on_commit=False)
    return _SessionLocal


def init_election_db():
    """Create all election tables."""
    engine = _get_engine()
    ElectionBase.metadata.create_all(engine)


def get_election_db():
    """Yield a DB session with write locking for SQLite safety."""
    factory = get_session_factory()
    db = factory()
    try:
        yield db
    finally:
        db.close()
