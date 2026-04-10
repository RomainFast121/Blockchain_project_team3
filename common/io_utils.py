"""Small filesystem helpers used across modules."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not exist and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to parquet with stable settings."""

    ensure_directory(path.parent)
    df.to_parquet(path, index=False)


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a parquet file from disk."""

    return pd.read_parquet(path)
