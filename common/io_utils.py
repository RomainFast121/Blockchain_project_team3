"""File-system utilities and parquet helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_directory(path: Path) -> Path:
    """Create the directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Persist a DataFrame as a parquet file with stable settings."""
    ensure_directory(path.parent)
    df.to_parquet(path, index=False)


def read_parquet(path: Path) -> pd.DataFrame:
    """Load a parquet file."""
    return pd.read_parquet(path)

