from __future__ import annotations

from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import pandas as pd


def write_raw_archive(base_path: Path, source: str, records: list[dict[str, Any]]) -> Path | None:
    if not records:
        return None
    now = datetime.now(UTC).replace(tzinfo=None)
    folder = base_path / source / now.strftime("%Y-%m-%d")
    folder.mkdir(parents=True, exist_ok=True)
    stem = now.strftime("%H%M%S_%f")
    frame = pd.DataFrame.from_records(records)
    parquet_path = folder / f"{stem}.parquet"
    try:
        frame.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception:
        csv_path = folder / f"{stem}.csv"
        frame.to_csv(csv_path, index=False)
        return csv_path

