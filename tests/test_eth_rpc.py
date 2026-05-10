"""Tests for the Ethereum RPC helper layer."""

from __future__ import annotations

from datetime import datetime, timezone
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from requests import HTTPError, RequestException

from common.eth_rpc import EthereumArchiveClient


class EthereumRpcClientAutoSplitTests(unittest.TestCase):
    """Check that log requests split automatically when a provider rejects them."""

    def test_get_logs_with_auto_split_recovers_from_large_range_error(self) -> None:
        client = object.__new__(EthereumArchiveClient)

        def fake_get_logs(event_name: str, from_block: int, to_block: int):  # noqa: ANN001
            if to_block - from_block >= 2:
                raise HTTPError("range too large")
            return [
                {
                    "event": event_name,
                    "block_number": block,
                    "transaction_hash": f"0x{block}",
                    "log_index": 0,
                    "args": {},
                }
                for block in range(from_block, to_block + 1)
            ]

        client.get_logs = fake_get_logs  # type: ignore[method-assign]
        logs = client.get_logs_with_auto_split("Swap", 100, 104)
        self.assertEqual([log["block_number"] for log in logs], [100, 101, 102, 103, 104])

    def test_get_logs_with_retry_recovers_from_rate_limit(self) -> None:
        client = object.__new__(EthereumArchiveClient)
        client.request_spacing_seconds = 0.0
        client.retry_attempts = 3
        client.retry_base_delay_seconds = 0.0

        responses = [
            HTTPError(response=SimpleNamespace(status_code=429)),
            [{"blockNumber": 1, "transactionHash": b"\x01", "logIndex": 0, "args": {}, "event": "Swap"}],
        ]

        class DummyEth:
            def get_logs(self, filter_params):  # noqa: ANN001
                response = responses.pop(0)
                if isinstance(response, Exception):
                    raise response
                return response

        client.web3 = SimpleNamespace(eth=DummyEth())

        with patch("common.eth_rpc.sleep"):
            logs = client._get_logs_with_retry({"fromBlock": 1, "toBlock": 1})

        self.assertEqual(len(logs), 1)

    def test_get_block_timestamp_retries_then_caches(self) -> None:
        client = object.__new__(EthereumArchiveClient)
        client._timestamp_cache = {}
        client.request_spacing_seconds = 0.0
        client.retry_attempts = 2
        client.retry_base_delay_seconds = 0.0

        responses = [
            RequestException("temporary outage"),
            {"timestamp": 1_700_000_000},
        ]

        class DummyEth:
            def get_block(self, block_number):  # noqa: ANN001
                response = responses.pop(0)
                if isinstance(response, Exception):
                    raise response
                return response

        client.web3 = SimpleNamespace(eth=DummyEth())

        with patch("common.eth_rpc.sleep"):
            timestamp = client.get_block_timestamp(123)
            cached_timestamp = client.get_block_timestamp(123)

        self.assertEqual(timestamp, cached_timestamp)
        self.assertEqual(len(responses), 0)

    def test_block_timestamps_frame_batch_size_one_uses_sequential_path(self) -> None:
        client = object.__new__(EthereumArchiveClient)
        client._timestamp_cache = {}
        client.request_spacing_seconds = 0.0
        client.retry_attempts = 1
        client.retry_base_delay_seconds = 0.0

        with patch.object(client, "_fetch_block_timestamps_batch") as batch_mock:
            with patch.object(
                client,
                "get_block_timestamp",
                side_effect=[
                    datetime(2026, 1, 1, tzinfo=timezone.utc),
                    datetime(2026, 1, 2, tzinfo=timezone.utc),
                ],
            ) as sequential_mock:
                frame = client.block_timestamps_frame([1, 2, 1], timestamp_batch_size=1, progress_seconds=0)

        batch_mock.assert_not_called()
        self.assertEqual(sequential_mock.call_count, 2)
        self.assertListEqual(frame["block_number"].tolist(), [1, 2])

    def test_block_timestamps_frame_batch_fetches_same_schema(self) -> None:
        client = object.__new__(EthereumArchiveClient)
        client._timestamp_cache = {}
        client.request_spacing_seconds = 0.0
        client.retry_attempts = 1
        client.retry_base_delay_seconds = 0.0

        timestamps = {
            10: datetime(2026, 1, 1, tzinfo=timezone.utc),
            11: datetime(2026, 1, 2, tzinfo=timezone.utc),
        }
        with patch.object(client, "_fetch_block_timestamps_batch", return_value=timestamps) as batch_mock:
            frame = client.block_timestamps_frame([11, 10, 11], timestamp_batch_size=50, progress_seconds=0)

        batch_mock.assert_called_once_with([10, 11])
        self.assertIsInstance(frame, pd.DataFrame)
        self.assertListEqual(frame.columns.tolist(), ["block_number", "block_timestamp"])
        self.assertListEqual(frame["block_number"].tolist(), [10, 11])
        self.assertEqual(frame.loc[0, "block_timestamp"], timestamps[10])

    def test_block_timestamps_frame_batch_failure_falls_back_to_sequential(self) -> None:
        client = object.__new__(EthereumArchiveClient)
        client._timestamp_cache = {}
        client.request_spacing_seconds = 0.0
        client.retry_attempts = 1
        client.retry_base_delay_seconds = 0.0

        with patch.object(client, "_fetch_block_timestamps_batch", side_effect=RequestException("boom")):
            with patch.object(
                client,
                "get_block_timestamp",
                side_effect=[
                    datetime(2026, 1, 3, tzinfo=timezone.utc),
                    datetime(2026, 1, 4, tzinfo=timezone.utc),
                ],
            ) as sequential_mock:
                frame = client.block_timestamps_frame([12, 13], timestamp_batch_size=2, progress_seconds=0)

        self.assertEqual(sequential_mock.call_count, 2)
        self.assertListEqual(frame["block_number"].tolist(), [12, 13])


if __name__ == "__main__":
    unittest.main()
