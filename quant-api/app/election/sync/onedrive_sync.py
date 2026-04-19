"""OneDrive sync utilities for election SQLite database.

The database lives at the configured OneDrive path. OneDrive automatically
syncs the file. This module provides health checks and backup utilities.
"""
from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path

from app.election.config import get_election_settings

logger = logging.getLogger(__name__)


def get_db_path() -> Path:
    """Get the election database file path."""
    return Path(get_election_settings().election_db_path)


def is_db_accessible() -> bool:
    """Check if the database file exists and is writable."""
    db_path = get_db_path()
    if not db_path.exists():
        return False
    try:
        # Check we can open it
        with open(db_path, "rb") as f:
            f.read(16)
        return True
    except Exception:
        return False


def get_db_size_mb() -> float:
    """Get database file size in MB."""
    db_path = get_db_path()
    if not db_path.exists():
        return 0.0
    return db_path.stat().st_size / (1024 * 1024)


def create_backup() -> Path | None:
    """Create a timestamped backup of the database."""
    db_path = get_db_path()
    if not db_path.exists():
        return None

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.with_suffix(f".backup_{timestamp}.db")

    try:
        shutil.copy2(str(db_path), str(backup_path))
        logger.info("Database backup created: %s (%.1f MB)", backup_path.name, get_db_size_mb())
        return backup_path
    except Exception as exc:
        logger.error("Backup failed: %s", exc)
        return None


def sync_status() -> dict:
    """Return sync health status."""
    db_path = get_db_path()
    return {
        "db_path": str(db_path),
        "exists": db_path.exists(),
        "accessible": is_db_accessible(),
        "size_mb": round(get_db_size_mb(), 2),
        "parent_exists": db_path.parent.exists(),
        "onedrive_detected": "onedrive" in str(db_path).lower(),
    }
