"""Thin Ethereum archive-node client tailored to the project."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd
from eth_utils import event_abi_to_log_topic
from web3 import Web3
from web3._utils.events import get_event_data

from common.uniswap_abi import POOL_ABI


UTC = timezone.utc


def _event_abi_map() -> dict[str, dict[str, Any]]:
    return {
        item["name"]: item
        for item in POOL_ABI
        if item.get("type") == "event"
    }


@dataclass
class EthereumArchiveClient:
    """Utility wrapper around Web3 with a simple block timestamp cache."""

    rpc_url: str
    pool_address: str
    _timestamp_cache: dict[int, datetime] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.web3 = Web3(Web3.HTTPProvider(self.rpc_url))
        if not self.web3.is_connected():
            raise ConnectionError(f"Unable to connect to Ethereum RPC endpoint: {self.rpc_url}")
        self.pool_address = Web3.to_checksum_address(self.pool_address)
        self.pool_contract = self.web3.eth.contract(address=self.pool_address, abi=POOL_ABI)
        self._event_abi_by_name = _event_abi_map()

    @property
    def latest_block(self) -> int:
        return int(self.web3.eth.block_number)

    def get_block_timestamp(self, block_number: int) -> datetime:
        """Return a UTC timestamp for the block, using a local cache."""
        if block_number not in self._timestamp_cache:
            block = self.web3.eth.get_block(int(block_number))
            timestamp = datetime.fromtimestamp(int(block["timestamp"]), tz=UTC)
            self._timestamp_cache[block_number] = timestamp
        return self._timestamp_cache[block_number]

    def get_logs(self, event_name: str, from_block: int, to_block: int) -> list[dict[str, Any]]:
        """Fetch and decode logs for a single event over a block interval."""
        event_abi = self._event_abi_by_name[event_name]
        topic0 = event_abi_to_log_topic(event_abi)
        raw_logs = self.web3.eth.get_logs(
            {
                "address": self.pool_address,
                "fromBlock": int(from_block),
                "toBlock": int(to_block),
                "topics": [topic0],
            }
        )
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

    def iter_event_logs(
        self,
        event_name: str,
        from_block: int,
        to_block: int,
        chunk_size: int,
    ) -> Iterable[dict[str, Any]]:
        """Yield decoded logs while chunking the RPC requests."""
        start = int(from_block)
        end = int(to_block)
        while start <= end:
            stop = min(start + chunk_size - 1, end)
            for item in self.get_logs(event_name=event_name, from_block=start, to_block=stop):
                yield item
            start = stop + 1

    def find_block_at_or_after(self, timestamp: datetime) -> int:
        """Binary-search the first block whose timestamp is at or after the target."""
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
        """Return the block closest in time to the target timestamp."""
        at_or_after = self.find_block_at_or_after(timestamp)
        if at_or_after == 0:
            return 0
        before = at_or_after - 1
        target = int(timestamp.astimezone(UTC).timestamp())
        before_diff = abs(int(self.get_block_timestamp(before).timestamp()) - target)
        after_diff = abs(int(self.get_block_timestamp(at_or_after).timestamp()) - target)
        if before_diff <= after_diff:
            return before
        return at_or_after

    def block_timestamps_frame(self, block_numbers: Iterable[int]) -> pd.DataFrame:
        """Return a DataFrame mapping block numbers to UTC timestamps."""
        rows = [
            {
                "block_number": int(block_number),
                "block_timestamp": self.get_block_timestamp(int(block_number)),
            }
            for block_number in sorted({int(value) for value in block_numbers})
        ]
        return pd.DataFrame(rows)

    def call_slot0(self, block_number: int) -> dict[str, Any]:
        """Return the pool slot0 state at a given block."""
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
        """Return the on-chain state for a specific tick at a given block."""
        liquidity_gross, liquidity_net, *_unused, initialized = self.pool_contract.functions.ticks(int(tick)).call(
            block_identifier=int(block_number)
        )
        return {
            "tick": int(tick),
            "liquidityGross": int(liquidity_gross),
            "liquidityNet": int(liquidity_net),
            "initialized": bool(initialized),
        }

