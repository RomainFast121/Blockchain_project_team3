"""UTC-safe date and timestamp helpers."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Iterator


UTC = timezone.utc


def utc_datetime(value: datetime) -> datetime:
    """Return a timezone-aware UTC datetime."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def date_range(start: date, end: date) -> Iterator[date]:
    """Yield each calendar day in the inclusive range."""
    day_count = (end - start).days
    for offset in range(day_count + 1):
        yield start + timedelta(days=offset)


def utc_midnight(day: date) -> datetime:
    """Return the UTC midnight for a calendar day."""
    return datetime.combine(day, time(0, 0, 0), tzinfo=UTC)


def unix_timestamp(value: datetime) -> int:
    """Convert a datetime to a Unix timestamp in seconds."""
    return int(utc_datetime(value).timestamp())

