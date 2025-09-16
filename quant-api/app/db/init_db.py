from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.db.models import Base
from app.db.session import engine


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    # Timescale-specific setup is best effort and only applied on Postgres targets.
    if not engine.url.drivername.startswith("postgresql"):
        return
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
    except SQLAlchemyError:
        # Timescale extension may be unavailable in some local DB images.
        # The API can still run with normal Postgres tables.
        pass

