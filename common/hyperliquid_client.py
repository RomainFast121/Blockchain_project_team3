"""Small Hyperliquid REST client for hourly candles and funding history."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import requests


HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
PAGE_LIMIT = 500


@dataclass
class HyperliquidClient:
    """Minimal client for the Hyperliquid info endpoint."""

    session: requests.Session | None = None

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = requests.Session()

    def _post(self, payload: dict) -> list[dict]:
        response = self.session.post(HYPERLIQUID_INFO_URL, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list):
            return data
        raise ValueError(f"Unexpected Hyperliquid response payload: {data}")

    def fetch_hourly_candles(self, coin: str, start_time_ms: int, end_time_ms: int) -> pd.DataFrame:
        """Fetch hourly OHLCV candles."""
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
            return frame
        frame = frame.drop_duplicates(subset=["t", "T"]).sort_values("t").reset_index(drop=True)
        frame["timestamp"] = pd.to_datetime(frame["T"], unit="ms", utc=True)
        frame["open"] = frame["o"].astype(float)
        frame["high"] = frame["h"].astype(float)
        frame["low"] = frame["l"].astype(float)
        frame["close"] = frame["c"].astype(float)
        frame["volume"] = frame["v"].astype(float)
        return frame[["timestamp", "open", "high", "low", "close", "volume", "t", "T", "n", "s", "i"]]

    def fetch_funding_history(self, coin: str, start_time_ms: int, end_time_ms: int) -> pd.DataFrame:
        """Fetch funding history and paginate across the study window."""
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
            return frame
        frame = frame.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
        frame["timestamp"] = pd.to_datetime(frame["time"], unit="ms", utc=True)
        frame["funding_rate"] = frame["fundingRate"].astype(float)
        frame["premium"] = pd.to_numeric(frame["premium"], errors="coerce")
        return frame[["timestamp", "coin", "funding_rate", "premium", "time"]]
