"""Thin Ethereum archive-node client tailored to the project.

The goal of this wrapper is not to hide Web3 entirely. It simply packages the
few blockchain operations used repeatedly in the assignment so the module files
can read more like data workflows:

- fetch event logs,
- find blocks around UTC timestamps,
- query `slot0()` and `ticks()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import sleep
from typing import Any, Iterable

import pandas as pd
from requests import HTTPError
from eth_utils import event_abi_to_log_topic
from web3 import Web3
from web3._utils.events import get_event_data

from common.constants import (
    DEFAULT_RPC_REQUEST_SPACING_SECONDS,
    DEFAULT_RPC_RETRY_ATTEMPTS,
    DEFAULT_RPC_RETRY_BASE_DELAY_SECONDS,
)
from common.uniswap_abi import POOL_ABI


UTC = timezone.utc
DEFAULT_INITIAL_ADAPTIVE_LOG_CHUNK = 128


def _event_abi_map() -> dict[str, dict[str, Any]]:
    """Index the pool ABI by event name for faster lookup."""

    return {item["name"]: item for item in POOL_ABI if item.get("type") == "event"}


@dataclass
class EthereumArchiveClient:
    """Convenience wrapper around Web3 with a simple block-timestamp cache."""

    rpc_url: str
    pool_address: str
    _timestamp_cache: dict[int, datetime] = field(default_factory=dict)
    request_spacing_seconds: float = DEFAULT_RPC_REQUEST_SPACING_SECONDS
    retry_attempts: int = DEFAULT_RPC_RETRY_ATTEMPTS
    retry_base_delay_seconds: float = DEFAULT_RPC_RETRY_BASE_DELAY_SECONDS

    def __post_init__(self) -> None:
        self.web3 = Web3(Web3.HTTPProvider(self.rpc_url))
        if not self.web3.is_connected():
            raise ConnectionError(f"Unable to connect to Ethereum RPC endpoint: {self.rpc_url}")

        self.pool_address = Web3.to_checksum_address(self.pool_address)
        self.pool_contract = self.web3.eth.contract(address=self.pool_address, abi=POOL_ABI)
        self._event_abi_by_name = _event_abi_map()

    @property
    def latest_block(self) -> int:
        """Return the latest block known by the RPC node."""

        return int(self.web3.eth.block_number)

    def get_block_timestamp(self, block_number: int) -> datetime:
        """Return a UTC block timestamp, using a local cache when possible."""

        if block_number not in self._timestamp_cache:
            block = self.web3.eth.get_block(int(block_number))
            self._timestamp_cache[block_number] = datetime.fromtimestamp(int(block["timestamp"]), tz=UTC)
        return self._timestamp_cache[block_number]

    def get_logs(self, event_name: str, from_block: int, to_block: int) -> list[dict[str, Any]]:
        """Fetch and decode logs for one event type over a block interval."""

        event_abi = self._event_abi_by_name[event_name]
        topic0 = event_abi_to_log_topic(event_abi)
        filter_params = {
            "address": self.pool_address,
            "fromBlock": int(from_block),
            "toBlock": int(to_block),
            "topics": [topic0],
        }
        raw_logs = self._get_logs_with_retry(filter_params)

        decoded_logs: list[dict[str, Any]] = []
        for raw_log in raw_logs:
            parsed = get_event_data(self.web3.codec, event_abi, raw_log)
            decoded_logs.append(
                {
                    "event": parsed["event"],
                    "block_number": int(parsed["blockNumber"]),
                    "transaction_hash": parsed["transactionHash"].hex(),
                    "log_index": int(parsed["logIndex"]),
                    "args": dict(parsed["args"]),
                }
            )
        return decoded_logs

    def _get_logs_with_retry(self, filter_params: dict[str, Any]) -> list[dict[str, Any]]:
        """Call `eth_getLogs` with gentle pacing and retry on provider throttling."""

        attempt = 0
        while True:
            if self.request_spacing_seconds > 0:
                sleep(self.request_spacing_seconds)
            try:
                return self.web3.eth.get_logs(filter_params)
            except HTTPError as exc:
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                if status_code != 429 or attempt >= self.retry_attempts:
                    raise

                sleep(self.retry_base_delay_seconds * (2**attempt))
                attempt += 1

    def get_logs_with_auto_split(self, event_name: str, from_block: int, to_block: int) -> list[dict[str, Any]]:
        """Fetch logs and recursively split the block range if the RPC rejects it.

        This matters for busy contracts such as the USDC/WETH 0.05% pool: a
        provider may reject a valid `eth_getLogs` request simply because the
        response would contain too many logs. Splitting ranges automatically makes
        the project much easier to run on free-tier plans.
        """

        start = int(from_block)
        end = int(to_block)
        if start > end:
            return []

        try:
            return self.get_logs(event_name=event_name, from_block=start, to_block=end)
        except HTTPError:
            if start == end:
                raise
        except ValueError:
            if start == end:
                raise

        midpoint = (start + end) // 2
        left_logs = self.get_logs_with_auto_split(event_name=event_name, from_block=start, to_block=midpoint)
        right_logs = self.get_logs_with_auto_split(event_name=event_name, from_block=midpoint + 1, to_block=end)
        return left_logs + right_logs

    def iter_event_logs(
        self,
        event_name: str,
        from_block: int,
        to_block: int,
        chunk_size: int,
    ) -> Iterable[dict[str, Any]]:
        """Yield decoded logs while chunking requests to stay RPC-friendly."""

        start = int(from_block)
        end = int(to_block)
        max_chunk_size = max(1, int(chunk_size))

        # We intentionally start smaller than the user-provided ceiling. For a
        # busy pool, this avoids immediately hammering the provider with ranges
        # that are known to be too large for free-tier plans.
        current_chunk_size = min(max_chunk_size, DEFAULT_INITIAL_ADAPTIVE_LOG_CHUNK)

        while start <= end:
            stop = min(start + current_chunk_size - 1, end)
            attempted_span = stop - start + 1

            try:
                decoded_logs = self.get_logs_with_auto_split(event_name=event_name, from_block=start, to_block=stop)
            except ValueError:
                if attempted_span == 1:
                    raise

                # Learn from the failure and retry the same starting block with a
                # smaller window instead of repeatedly resetting to the original
                # chunk size.
                current_chunk_size = max(1, attempted_span // 2)
                continue

            for decoded_log in decoded_logs:
                yield decoded_log

            start = stop + 1

            # Empty spans can safely grow again, which matters when moving from
            # busy to calm periods. Dense successful spans keep the current size.
            if not decoded_logs and current_chunk_size < max_chunk_size:
                current_chunk_size = min(max_chunk_size, attempted_span * 2)
            elif len(decoded_logs) >= 50 and attempted_span > 1:
                current_chunk_size = max(1, attempted_span // 2)

    def find_block_at_or_after(self, timestamp: datetime) -> int:
        """Binary-search the first block whose timestamp is at or after a target."""

        target = int(timestamp.astimezone(UTC).timestamp())
        low = 0
        high = self.latest_block
        answer = high

        while low <= high:
            mid = (low + high) // 2
            mid_timestamp = int(self.get_block_timestamp(mid).timestamp())
            if mid_timestamp >= target:
                answer = mid
                high = mid - 1
            else:
                low = mid + 1

        return answer

    def find_closest_block(self, timestamp: datetime) -> int:
        """Return the block closest in time to a target timestamp."""

        at_or_after = self.find_block_at_or_after(timestamp)
        if at_or_after == 0:
            return 0

        before = at_or_after - 1
        target = int(timestamp.astimezone(UTC).timestamp())
        before_distance = abs(int(self.get_block_timestamp(before).timestamp()) - target)
        after_distance = abs(int(self.get_block_timestamp(at_or_after).timestamp()) - target)
        return before if before_distance <= after_distance else at_or_after

    def block_timestamps_frame(self, block_numbers: Iterable[int]) -> pd.DataFrame:
        """Return a merge-ready DataFrame of block numbers and timestamps."""

        rows = [
            {
                "block_number": int(block_number),
                "block_timestamp": self.get_block_timestamp(int(block_number)),
            }
            for block_number in sorted({int(value) for value in block_numbers})
        ]
        return pd.DataFrame(rows)

    def call_slot0(self, block_number: int) -> dict[str, Any]:
        """Call `slot0()` at a historical block."""

        sqrt_price_x96, tick, observation_index, _, _, _, unlocked = self.pool_contract.functions.slot0().call(
            block_identifier=int(block_number)
        )
        return {
            "sqrtPriceX96": int(sqrt_price_x96),
            "tick": int(tick),
            "observation_index": int(observation_index),
            "unlocked": bool(unlocked),
        }

    def call_tick_state(self, block_number: int, tick: int) -> dict[str, Any]:
        """Call `ticks(tick)` at a historical block."""

        liquidity_gross, liquidity_net, *_unused, initialized = self.pool_contract.functions.ticks(int(tick)).call(
            block_identifier=int(block_number)
        )
        return {
            "tick": int(tick),
            "liquidityGross": int(liquidity_gross),
            "liquidityNet": int(liquidity_net),
            "initialized": bool(initialized),
        }
