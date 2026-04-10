"""Tests for the Ethereum RPC helper layer."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from requests import HTTPError

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


if __name__ == "__main__":
    unittest.main()
