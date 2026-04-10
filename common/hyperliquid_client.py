"""Minimal Hyperliquid client used by Module 5.

The project only needs two datasets from Hyperliquid:

- hourly candles,
- funding history.

So this client intentionally stays small and focused rather than becoming a
general-purpose API wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import requests


HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
PAGE_LIMIT = 500
PERP_PRICE_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "t", "T", "n", "s", "i"]
FUNDING_HISTORY_COLUMNS = ["timestamp", "coin", "funding_rate", "premium", "time"]


@dataclass
class HyperliquidClient:
    """Small wrapper around the Hyperliquid info endpoint."""

    session: requests.Session | None = None

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = requests.Session()

    def _post(self, payload: dict) -> list[dict]:
        """POST one request and require a list-like response payload."""

        response = self.session.post(HYPERLIQUID_INFO_URL, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list):
            return data
        raise ValueError(f"Unexpected Hyperliquid response payload: {data}")

    def fetch_hourly_candles(self, coin: str, start_time_ms: int, end_time_ms: int) -> pd.DataFrame:
        """Fetch hourly OHLCV candles across the requested time window."""

        cursor = int(start_time_ms)
        rows: list[dict] = []

        while cursor <= end_time_ms:
            payload = {
                "type": "candleSnapshot",
                "req": {
                    "coin": coin,
                    "interval": "1h",
                    "startTime": cursor,
                    "endTime": int(end_time_ms),
                },
            }
            batch = self._post(payload)
            if not batch:
                break

            rows.extend(batch)
            last_end_time = int(batch[-1]["T"])
            if last_end_time >= end_time_ms or len(batch) < PAGE_LIMIT:
                break
            cursor = last_end_time + 1

        frame = pd.DataFrame(rows)
        if frame.empty:
            return pd.DataFrame(columns=PERP_PRICE_COLUMNS)

        frame = frame.drop_duplicates(subset=["t", "T"]).sort_values("t").reset_index(drop=True)
        frame["timestamp"] = pd.to_datetime(frame["T"], unit="ms", utc=True)
        frame["open"] = frame["o"].astype(float)
        frame["high"] = frame["h"].astype(float)
        frame["low"] = frame["l"].astype(float)
        frame["close"] = frame["c"].astype(float)
        frame["volume"] = frame["v"].astype(float)
        return frame[PERP_PRICE_COLUMNS]

    def fetch_funding_history(self, coin: str, start_time_ms: int, end_time_ms: int) -> pd.DataFrame:
        """Fetch funding history over the requested time window."""

        cursor = int(start_time_ms)
        rows: list[dict] = []

        while cursor <= end_time_ms:
            payload = {
                "type": "fundingHistory",
                "coin": coin,
                "startTime": cursor,
                "endTime": int(end_time_ms),
            }
            batch = self._post(payload)
            if not batch:
                break

            rows.extend(batch)
            last_time = int(batch[-1]["time"])
            if last_time >= end_time_ms or len(batch) < PAGE_LIMIT:
                break
            cursor = last_time + 1

        frame = pd.DataFrame(rows)
        if frame.empty:
            return pd.DataFrame(columns=FUNDING_HISTORY_COLUMNS)

        frame = frame.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
        frame["timestamp"] = pd.to_datetime(frame["time"], unit="ms", utc=True)
        frame["funding_rate"] = frame["fundingRate"].astype(float)
        frame["premium"] = pd.to_numeric(frame["premium"], errors="coerce")
        return frame[FUNDING_HISTORY_COLUMNS]
