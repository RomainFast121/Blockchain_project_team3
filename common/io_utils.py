"""Small filesystem helpers used across modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not exist and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def _fits_signed_int64(value: int) -> bool:
    """Return whether an integer can be represented safely as int64."""

    return -(2**63) <= value <= (2**63 - 1)


def _is_python_int_like(value: Any) -> bool:
    """Detect plain Python integers while excluding booleans."""

    return isinstance(value, int) and not isinstance(value, bool)


def prepare_dataframe_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Convert parquet-unfriendly object columns into stable serializable types.

    PyArrow cannot always store arbitrary-size Python integers as numeric parquet
    columns. Raw blockchain values such as `sqrtPriceX96` or signed int256 event
    amounts can exceed int64, so we store those specific columns as strings when
    needed instead of failing the whole pipeline.
    """

    prepared = df.copy()
    for column_name in prepared.columns:
        series = prepared[column_name]
        if series.dtype != "object":
            continue

        non_null = series.dropna()
        if non_null.empty:
            continue
        if not non_null.map(_is_python_int_like).all():
            continue
        if non_null.map(_fits_signed_int64).all():
            continue

        prepared[column_name] = series.map(lambda value: None if pd.isna(value) else str(value))
    return prepared


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to parquet with stable settings."""

    ensure_directory(path.parent)
    prepare_dataframe_for_parquet(df).to_parquet(path, index=False)


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a parquet file from disk."""

    return pd.read_parquet(path)
