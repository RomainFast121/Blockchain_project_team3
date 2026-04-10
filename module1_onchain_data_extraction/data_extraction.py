"""Module 1: extract on-chain data and rebuild the daily pool state.

This module answers the first part of the PDF:

1. download the relevant Uniswap V3 events for the pool,
2. convert them into clean parquet datasets,
3. rebuild daily liquidity snapshots from Mint/Burn history,
4. fetch `slot0()` at the same daily timestamps,
5. run a few validation checks so later modules start from trusted inputs.

The code is written in a top-down way on purpose:
`resolve study window -> fetch logs -> decode tables -> rebuild snapshots ->
validate -> save outputs`.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from random import Random
from time import monotonic
from typing import Iterable

import pandas as pd

from common.constants import (
    DEFAULT_RPC_LOG_CHUNK,
    DEFAULT_STUDY_END,
    DEFAULT_STUDY_START,
    POOL_ADDRESS,
    POOL_DEPLOYMENT_BLOCK,
    PROCESSED_DATA_DIR,
    TICK_SPACING,
    TOKEN0_DECIMALS,
    TOKEN1_DECIMALS,
)
from common.dates import date_range, utc_midnight
from common.eth_rpc import EthereumArchiveClient
from common.io_utils import ensure_directory, write_parquet
from common.uniswap_math import sqrt_price_x96_to_price_usdc_per_weth, tick_to_price_usdc_per_weth


@dataclass(frozen=True)
class Module1Paths:
    """All files produced by Module 1."""

    output_dir: Path

    @property
    def swap_events(self) -> Path:
        return self.output_dir / "swap_events.parquet"

    @property
    def mint_burn_events(self) -> Path:
        return self.output_dir / "mint_burn_events.parquet"

    @property
    def collect_events(self) -> Path:
        return self.output_dir / "collect_events.parquet"

    @property
    def liquidity_snapshots(self) -> Path:
        return self.output_dir / "liquidity_snapshots.parquet"

    @property
    def slot0_snapshots(self) -> Path:
        return self.output_dir / "slot0_snapshots.parquet"

    @property
    def validations(self) -> Path:
        return self.output_dir / "module1_validations.json"


@dataclass
class ProgressTracker:
    """Print sparse progress updates during long live runs.

    We keep this intentionally lightweight:
    - one start line,
    - then a periodic summary every N seconds,
    - then one completion line.
    """

    label: str
    start_block: int
    end_block: int
    progress_seconds: float
    rows_seen: int = 0
    chunks_seen: int = 0

    def __post_init__(self) -> None:
        self._started_at = monotonic()
        self._last_print_at = self._started_at
        self._block_span = max(1, self.end_block - self.start_block + 1)
        print(
            f"[{self.label}] starting blocks {self.start_block:,}-{self.end_block:,}",
            flush=True,
        )

    def maybe_print(self, current_end_block: int) -> None:
        """Emit one compact update if enough time has passed."""

        if self.progress_seconds <= 0:
            return

        now = monotonic()
        if now - self._last_print_at < self.progress_seconds:
            return

        completed_blocks = max(0, current_end_block - self.start_block + 1)
        pct_complete = min(100.0, 100 * completed_blocks / self._block_span)
        elapsed = now - self._started_at
        print(
            f"[{self.label}] {pct_complete:5.1f}% "
            f"up to block {current_end_block:,} "
            f"chunks={self.chunks_seen:,} rows={self.rows_seen:,} "
            f"elapsed={elapsed:,.0f}s",
            flush=True,
        )
        self._last_print_at = now

    def finish(self) -> None:
        """Emit one final summary line."""

        elapsed = monotonic() - self._started_at
        print(
            f"[{self.label}] done rows={self.rows_seen:,} chunks={self.chunks_seen:,} "
            f"elapsed={elapsed:,.0f}s",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    """Parse CLI inputs for the extraction step."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rpc-url", required=True, help="Ethereum archive node URL.")
    parser.add_argument("--pool-address", default=POOL_ADDRESS, help="Uniswap V3 pool address.")
    parser.add_argument(
        "--study-start",
        type=date.fromisoformat,
        default=DEFAULT_STUDY_START,
        help="Study window start date in YYYY-MM-DD.",
    )
    parser.add_argument(
        "--study-end",
        type=date.fromisoformat,
        default=DEFAULT_STUDY_END,
        help="Study window end date in YYYY-MM-DD.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Directory where parquet files are written.",
    )
    parser.add_argument(
        "--log-chunk-size",
        type=int,
        default=DEFAULT_RPC_LOG_CHUNK,
        help=(
            "Block range per eth_getLogs request. The default is intentionally "
            "kept free-tier friendly; increase it only if your provider supports it."
        ),
    )
    parser.add_argument(
        "--validation-seed",
        type=int,
        default=17,
        help="Random seed used for the tick validation sample.",
    )
    parser.add_argument(
        "--smoke-test-days",
        type=int,
        default=None,
        help=(
            "Optional small-window run used to confirm the live RPC pipeline works. "
            "If provided, only the first N calendar days starting at --study-start are extracted."
        ),
    )
    parser.add_argument(
        "--progress-seconds",
        type=float,
        default=30.0,
        help=(
            "How often to print one compact progress update during long live runs. "
            "Set to 0 to disable periodic progress messages."
        ),
    )
    return parser.parse_args()


def apply_smoke_test_window(study_start: date, study_end: date, smoke_test_days: int | None) -> tuple[date, date]:
    """Optionally shrink the study window for a quick live end-to-end check."""

    if smoke_test_days is None:
        return study_start, study_end
    if smoke_test_days <= 0:
        raise ValueError("--smoke-test-days must be a positive integer")

    truncated_end = study_start + timedelta(days=smoke_test_days - 1)
    return study_start, min(study_end, truncated_end)


def liquidity_history_start_block(study_start_block: int, smoke_test_days: int | None) -> int:
    """Choose where Mint/Burn/Collect history starts.

    Full runs must scan from pool deployment because existing positions matter for
    later snapshots. Smoke tests have a different purpose: confirm the pipeline
    works end to end on a tiny slice. For that mode we intentionally cap the
    history at the study start block so the check stays lightweight.
    """

    if smoke_test_days is None:
        return POOL_DEPLOYMENT_BLOCK
    return study_start_block


# ---------------------------------------------------------------------------
# Small decoding helpers
# ---------------------------------------------------------------------------

def _signed_decimal(raw_value: int, decimals: int) -> float:
    """Turn a raw token amount into human units."""

    return raw_value / (10 ** decimals)


def _block_timestamps(client: EthereumArchiveClient, block_numbers: Iterable[int]) -> pd.DataFrame:
    """Fetch one timestamp per block and return a merge-ready frame."""

    return client.block_timestamps_frame(block_numbers)


def _attach_block_timestamps(
    client: EthereumArchiveClient,
    frame: pd.DataFrame,
    *,
    timestamp_column: str = "block_timestamp",
) -> pd.DataFrame:
    """Attach UTC block timestamps to a decoded event table."""

    if frame.empty:
        return frame

    timestamps = _block_timestamps(client, frame["block_number"].tolist())
    merged = frame.merge(timestamps, on="block_number", how="left")
    merged[timestamp_column] = pd.to_datetime(merged[timestamp_column], utc=True)
    return merged


def _decode_swap_events(
    client: EthereumArchiveClient,
    start_block: int,
    end_block: int,
    chunk_size: int,
    progress_seconds: float,
) -> pd.DataFrame:
    """Decode raw Swap logs into the clean table used throughout the project."""

    rows: list[dict[str, object]] = []
    tracker = ProgressTracker("Swap", start_block, end_block, progress_seconds)
    for event in client.iter_event_logs("Swap", start_block, end_block, chunk_size):
        args = event["args"]
        amount0_raw = int(args["amount0"])
        amount1_raw = int(args["amount1"])
        sqrt_price_x96 = int(args["sqrtPriceX96"])
        amount0_usdc = _signed_decimal(amount0_raw, TOKEN0_DECIMALS)
        amount1_weth = _signed_decimal(amount1_raw, TOKEN1_DECIMALS)

        # In this pool, paying USDC into the pool means the taker bought WETH.
        trade_direction = "buy_weth" if amount0_raw > 0 else "sell_weth"

        rows.append(
            {
                "block_number": event["block_number"],
                "transaction_hash": event["transaction_hash"],
                "log_index": event["log_index"],
                "amount0_raw": amount0_raw,
                "amount0_usdc": amount0_usdc,
                "amount1_raw": amount1_raw,
                "amount1_weth": amount1_weth,
                "sqrtPriceX96": sqrt_price_x96,
                "price_usdc_per_weth": float(sqrt_price_x96_to_price_usdc_per_weth(sqrt_price_x96)),
                "active_liquidity": int(args["liquidity"]),
                "tick": int(args["tick"]),
                "trade_direction": trade_direction,
                # For this assignment the USDC leg is the cleanest notional proxy.
                "notional_usd": abs(amount0_usdc),
            }
        )
        tracker.rows_seen += 1
        tracker.maybe_print(int(event["block_number"]))
    tracker.chunks_seen = max(1, (end_block - start_block + 1 + chunk_size - 1) // chunk_size)
    tracker.finish()

    swaps = pd.DataFrame(rows).sort_values(["block_number", "log_index"]).reset_index(drop=True)
    swaps = _attach_block_timestamps(client, swaps)
    if swaps.empty:
        return swaps

    return swaps[
        [
            "block_number",
            "block_timestamp",
            "transaction_hash",
            "log_index",
            "amount0_raw",
            "amount0_usdc",
            "amount1_raw",
            "amount1_weth",
            "sqrtPriceX96",
            "price_usdc_per_weth",
            "active_liquidity",
            "tick",
            "trade_direction",
            "notional_usd",
        ]
    ]


def _decode_liquidity_events(
    client: EthereumArchiveClient,
    event_name: str,
    start_block: int,
    end_block: int,
    chunk_size: int,
    progress_seconds: float,
) -> pd.DataFrame:
    """Decode Mint or Burn logs into one uniform liquidity-event table."""

    rows: list[dict[str, object]] = []
    tracker = ProgressTracker(event_name, start_block, end_block, progress_seconds)
    for event in client.iter_event_logs(event_name, start_block, end_block, chunk_size):
        args = event["args"]
        amount0_raw = int(args["amount0"])
        amount1_raw = int(args["amount1"])
        rows.append(
            {
                "block_number": event["block_number"],
                "transaction_hash": event["transaction_hash"],
                "log_index": event["log_index"],
                "event_type": event_name.lower(),
                "owner": args["owner"],
                "tick_lower": int(args["tickLower"]),
                "tick_upper": int(args["tickUpper"]),
                "liquidity_raw": int(args.get("amount", 0)),
                "amount0_raw": amount0_raw,
                "amount0_usdc": _signed_decimal(amount0_raw, TOKEN0_DECIMALS),
                "amount1_raw": amount1_raw,
                "amount1_weth": _signed_decimal(amount1_raw, TOKEN1_DECIMALS),
            }
        )
        tracker.rows_seen += 1
        tracker.maybe_print(int(event["block_number"]))

    tracker.chunks_seen = max(1, (end_block - start_block + 1 + chunk_size - 1) // chunk_size)
    tracker.finish()
    frame = pd.DataFrame(rows).sort_values(["block_number", "log_index"]).reset_index(drop=True)
    return _attach_block_timestamps(client, frame)


def _decode_collect_events(
    client: EthereumArchiveClient,
    start_block: int,
    end_block: int,
    chunk_size: int,
    progress_seconds: float,
) -> pd.DataFrame:
    """Decode Collect logs.

    Collect events are not directly needed to rebuild liquidity, but the PDF
    recommends storing them because they are useful context for LP economics.
    """

    rows: list[dict[str, object]] = []
    tracker = ProgressTracker("Collect", start_block, end_block, progress_seconds)
    for event in client.iter_event_logs("Collect", start_block, end_block, chunk_size):
        args = event["args"]
        amount0_raw = int(args["amount0"])
        amount1_raw = int(args["amount1"])
        rows.append(
            {
                "block_number": event["block_number"],
                "transaction_hash": event["transaction_hash"],
                "log_index": event["log_index"],
                "owner": args["owner"],
                "recipient": args["recipient"],
                "tick_lower": int(args["tickLower"]),
                "tick_upper": int(args["tickUpper"]),
                "amount0_raw": amount0_raw,
                "amount0_usdc": _signed_decimal(amount0_raw, TOKEN0_DECIMALS),
                "amount1_raw": amount1_raw,
                "amount1_weth": _signed_decimal(amount1_raw, TOKEN1_DECIMALS),
            }
        )
        tracker.rows_seen += 1
        tracker.maybe_print(int(event["block_number"]))

    tracker.chunks_seen = max(1, (end_block - start_block + 1 + chunk_size - 1) // chunk_size)
    tracker.finish()
    frame = pd.DataFrame(rows).sort_values(["block_number", "log_index"]).reset_index(drop=True)
    return _attach_block_timestamps(client, frame)


# ---------------------------------------------------------------------------
# Study window and snapshot schedule
# ---------------------------------------------------------------------------

def resolve_study_window_blocks(
    client: EthereumArchiveClient,
    study_start: date,
    study_end: date,
) -> tuple[int, int]:
    """Return the inclusive block range covered by the study window.

    The end block is chosen as the last block strictly before the next UTC
    midnight. This avoids leaking data from the following day.
    """

    start_block = client.find_block_at_or_after(utc_midnight(study_start))
    first_block_after_window = client.find_block_at_or_after(utc_midnight(study_end) + pd.Timedelta(days=1))
    end_block = max(first_block_after_window - 1, start_block)
    return start_block, end_block


def _build_snapshot_schedule(client: EthereumArchiveClient, study_start: date, study_end: date) -> pd.DataFrame:
    """Pick one daily block close to 00:00 UTC for each date in the window."""

    rows: list[dict[str, object]] = []
    for day in date_range(study_start, study_end):
        snapshot_timestamp = utc_midnight(day)
        rows.append(
            {
                "snapshot_date": day.isoformat(),
                "snapshot_timestamp": snapshot_timestamp,
                "snapshot_block": client.find_closest_block(snapshot_timestamp),
            }
        )

    schedule = pd.DataFrame(rows)
    schedule["snapshot_timestamp"] = pd.to_datetime(schedule["snapshot_timestamp"], utc=True)
    return schedule


# ---------------------------------------------------------------------------
# Liquidity reconstruction
# ---------------------------------------------------------------------------

def _tick_liquidity_deltas(mint_burn_events: pd.DataFrame) -> pd.DataFrame:
    """Translate Mint/Burn events into tick-level net and gross changes.

    A mint adds liquidity between `tick_lower` and `tick_upper`:
    - liquidityNet increases at the lower bound,
    - liquidityNet decreases at the upper bound.

    A burn does the opposite. Gross liquidity is tracked separately because the
    PDF asks us to compare our reconstructed tick states against `ticks()`.
    """

    rows: list[dict[str, int]] = []
    for event in mint_burn_events.itertuples(index=False):
        signed_liquidity = int(event.liquidity_raw) if event.event_type == "mint" else -int(event.liquidity_raw)
        gross_delta = int(event.liquidity_raw) if event.event_type == "mint" else -int(event.liquidity_raw)

        rows.append(
            {
                "block_number": int(event.block_number),
                "tick": int(event.tick_lower),
                "liquidity_net_delta": signed_liquidity,
                "liquidity_gross_delta": gross_delta,
            }
        )
        rows.append(
            {
                "block_number": int(event.block_number),
                "tick": int(event.tick_upper),
                "liquidity_net_delta": -signed_liquidity,
                "liquidity_gross_delta": gross_delta,
            }
        )

    return pd.DataFrame(rows)


def build_liquidity_snapshots(mint_burn_events: pd.DataFrame, snapshot_blocks: pd.DataFrame) -> pd.DataFrame:
    """Replay Mint/Burn history and rebuild the initialized ticks at each snapshot."""

    if mint_burn_events.empty or snapshot_blocks.empty:
        return pd.DataFrame(
            columns=[
                "snapshot_block",
                "snapshot_timestamp",
                "tick",
                "liquidityNet",
                "liquidityGross",
                "active_liquidity",
                "price_lower",
                "price_upper",
            ]
        )

    tick_deltas = _tick_liquidity_deltas(mint_burn_events).sort_values(["block_number", "tick"]).reset_index(drop=True)
    snapshot_blocks = snapshot_blocks.sort_values("snapshot_block").reset_index(drop=True)

    # We keep one running map for liquidityNet and liquidityGross. Walking across
    # sorted ticks then gives the active liquidity curve needed by later modules.
    liquidity_net_by_tick: defaultdict[int, int] = defaultdict(int)
    liquidity_gross_by_tick: defaultdict[int, int] = defaultdict(int)
    decoded_events = tick_deltas.to_dict("records")
    next_event_index = 0
    rows: list[dict[str, object]] = []

    for snapshot in snapshot_blocks.itertuples(index=False):
        while next_event_index < len(decoded_events):
            next_event = decoded_events[next_event_index]
            if int(next_event["block_number"]) > int(snapshot.snapshot_block):
                break

            tick = int(next_event["tick"])
            liquidity_net_by_tick[tick] += int(next_event["liquidity_net_delta"])
            liquidity_gross_by_tick[tick] += int(next_event["liquidity_gross_delta"])

            # Once gross liquidity is back to zero, the tick is no longer initialized.
            if liquidity_gross_by_tick[tick] == 0:
                liquidity_gross_by_tick.pop(tick, None)
                liquidity_net_by_tick.pop(tick, None)

            next_event_index += 1

        running_active_liquidity = 0
        for tick in sorted(liquidity_gross_by_tick):
            running_active_liquidity += liquidity_net_by_tick[tick]
            rows.append(
                {
                    "snapshot_block": int(snapshot.snapshot_block),
                    "snapshot_timestamp": snapshot.snapshot_timestamp,
                    "tick": int(tick),
                    "liquidityNet": int(liquidity_net_by_tick[tick]),
                    "liquidityGross": int(liquidity_gross_by_tick[tick]),
                    "active_liquidity": int(running_active_liquidity),
                    "price_lower": float(tick_to_price_usdc_per_weth(tick)),
                    "price_upper": float(tick_to_price_usdc_per_weth(tick + TICK_SPACING)),
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "snapshot_block",
            "snapshot_timestamp",
            "tick",
            "liquidityNet",
            "liquidityGross",
            "active_liquidity",
            "price_lower",
            "price_upper",
        ],
    )


def build_slot0_snapshots(client: EthereumArchiveClient, snapshot_blocks: pd.DataFrame) -> pd.DataFrame:
    """Fetch `slot0()` at every daily snapshot block."""

    rows: list[dict[str, object]] = []
    for snapshot in snapshot_blocks.itertuples(index=False):
        slot0_state = client.call_slot0(int(snapshot.snapshot_block))
        rows.append(
            {
                "snapshot_block": int(snapshot.snapshot_block),
                "snapshot_timestamp": snapshot.snapshot_timestamp,
                "sqrtPriceX96": slot0_state["sqrtPriceX96"],
                "price_usdc_per_weth": float(sqrt_price_x96_to_price_usdc_per_weth(slot0_state["sqrtPriceX96"])),
                "current_tick": slot0_state["tick"],
                "observation_index": slot0_state["observation_index"],
                "unlocked": slot0_state["unlocked"],
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def compare_slot0_to_last_swap(slot0_snapshots: pd.DataFrame, swap_events: pd.DataFrame) -> pd.DataFrame:
    """Check whether snapshot `slot0()` lines up with the last observed swap state."""

    if slot0_snapshots.empty or swap_events.empty:
        return pd.DataFrame()

    swaps = swap_events[["block_number", "price_usdc_per_weth", "tick"]].sort_values("block_number")
    rows: list[dict[str, object]] = []
    for snapshot in slot0_snapshots.itertuples(index=False):
        earlier_swaps = swaps[swaps["block_number"] <= int(snapshot.snapshot_block)]
        if earlier_swaps.empty:
            continue

        last_swap = earlier_swaps.iloc[-1]
        tick_distance = abs(int(snapshot.current_tick) - int(last_swap["tick"]))
        rows.append(
            {
                "snapshot_block": int(snapshot.snapshot_block),
                "slot0_price": float(snapshot.price_usdc_per_weth),
                "last_swap_price": float(last_swap["price_usdc_per_weth"]),
                "tick_distance": tick_distance,
                "within_tick_spacing": tick_distance <= TICK_SPACING,
            }
        )

    return pd.DataFrame(rows)


def validate_snapshot_against_rpc(
    client: EthereumArchiveClient,
    liquidity_snapshots: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """Spot-check one reconstructed snapshot against `pool.ticks()` on chain."""

    if liquidity_snapshots.empty:
        return pd.DataFrame()

    rng = Random(seed)
    middle_snapshot_block = int(liquidity_snapshots["snapshot_block"].iloc[len(liquidity_snapshots) // 2])
    snapshot = liquidity_snapshots[liquidity_snapshots["snapshot_block"] == middle_snapshot_block].reset_index(drop=True)
    sampled_ticks = sorted(rng.sample(snapshot["tick"].tolist(), min(10, len(snapshot))))

    rows: list[dict[str, object]] = []
    for tick in sampled_ticks:
        reconstructed_state = snapshot[snapshot["tick"] == tick].iloc[0]
        rpc_state = client.call_tick_state(middle_snapshot_block, int(tick))
        rows.append(
            {
                "snapshot_block": middle_snapshot_block,
                "tick": int(tick),
                "reconstructed_liquidityGross": int(reconstructed_state["liquidityGross"]),
                "rpc_liquidityGross": int(rpc_state["liquidityGross"]),
                "reconstructed_liquidityNet": int(reconstructed_state["liquidityNet"]),
                "rpc_liquidityNet": int(rpc_state["liquidityNet"]),
                "matches": (
                    int(reconstructed_state["liquidityGross"]) == int(rpc_state["liquidityGross"])
                    and int(reconstructed_state["liquidityNet"]) == int(rpc_state["liquidityNet"])
                ),
            }
        )

    return pd.DataFrame(rows)


def _validation_summary(
    slot0_consistency: pd.DataFrame,
    liquidity_validation: pd.DataFrame,
) -> dict[str, object]:
    """Build the JSON summary saved alongside the parquet outputs."""

    return {
        "assumption": "The default study window follows the example given in the project brief: 2025-10-01 to 2026-03-31.",
        "slot0_consistency_summary": {
            "rows_checked": int(len(slot0_consistency)),
            "all_within_tick_spacing": bool(slot0_consistency["within_tick_spacing"].all()) if not slot0_consistency.empty else None,
            "max_tick_distance": int(slot0_consistency["tick_distance"].max()) if not slot0_consistency.empty else None,
        },
        "liquidity_validation_summary": {
            "rows_checked": int(len(liquidity_validation)),
            "all_exact_matches": bool(liquidity_validation["matches"].all()) if not liquidity_validation.empty else None,
        },
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_module_1(
    rpc_url: str,
    pool_address: str,
    study_start: date,
    study_end: date,
    output_dir: Path,
    log_chunk_size: int,
    validation_seed: int,
    progress_seconds: float,
    smoke_test_days: int | None = None,
) -> Module1Paths:
    """Execute the full Module 1 workflow."""

    paths = Module1Paths(output_dir=ensure_directory(output_dir))
    client = EthereumArchiveClient(rpc_url=rpc_url, pool_address=pool_address)

    # Step 1. Define the historical window in block space.
    start_block, end_block = resolve_study_window_blocks(client=client, study_start=study_start, study_end=study_end)
    liquidity_start_block = liquidity_history_start_block(start_block, smoke_test_days)
    print(
        f"[Module1] study window {study_start.isoformat()} to {study_end.isoformat()} "
        f"maps to blocks {start_block:,}-{end_block:,}",
        flush=True,
    )
    if smoke_test_days is not None:
        print(
            "[Module1] smoke-test mode active: Mint/Burn/Collect history starts at the study window "
            "instead of pool deployment",
            flush=True,
        )

    # Step 2. Download and decode the three event families used by the rest of
    # the project. Full runs start Mint/Burn from pool deployment because open
    # positions created before the study window still matter for the snapshots.
    # Smoke tests intentionally use a much smaller history to keep the sanity
    # check practical on free-tier providers.
    swap_events = _decode_swap_events(
        client,
        start_block=start_block,
        end_block=end_block,
        chunk_size=log_chunk_size,
        progress_seconds=progress_seconds,
    )
    mint_events = _decode_liquidity_events(
        client,
        event_name="Mint",
        start_block=liquidity_start_block,
        end_block=end_block,
        chunk_size=log_chunk_size,
        progress_seconds=progress_seconds,
    )
    burn_events = _decode_liquidity_events(
        client,
        event_name="Burn",
        start_block=liquidity_start_block,
        end_block=end_block,
        chunk_size=log_chunk_size,
        progress_seconds=progress_seconds,
    )
    collect_events = _decode_collect_events(
        client,
        start_block=liquidity_start_block,
        end_block=end_block,
        chunk_size=log_chunk_size,
        progress_seconds=progress_seconds,
    )

    mint_burn_events = pd.concat([mint_events, burn_events], ignore_index=True)
    mint_burn_events = mint_burn_events.sort_values(["block_number", "log_index"]).reset_index(drop=True)

    # Step 3. Rebuild one daily snapshot schedule and use it to reconstruct both
    # the liquidity map and the on-chain slot0 state.
    snapshot_blocks = _build_snapshot_schedule(client, study_start=study_start, study_end=study_end)
    print(f"[Module1] building {len(snapshot_blocks):,} daily liquidity snapshots", flush=True)
    liquidity_snapshots = build_liquidity_snapshots(mint_burn_events, snapshot_blocks)
    print(f"[Module1] fetching {len(snapshot_blocks):,} slot0 snapshots", flush=True)
    slot0_snapshots = build_slot0_snapshots(client, snapshot_blocks)

    # Step 4. Run the two validation checks requested in spirit by the PDF:
    # compare tick states to `ticks()` and compare snapshot slot0 to the last swap.
    slot0_consistency = compare_slot0_to_last_swap(slot0_snapshots=slot0_snapshots, swap_events=swap_events)
    liquidity_validation = validate_snapshot_against_rpc(client=client, liquidity_snapshots=liquidity_snapshots, seed=validation_seed)

    # Step 5. Save clean parquet outputs for downstream modules.
    write_parquet(swap_events, paths.swap_events)
    write_parquet(mint_burn_events, paths.mint_burn_events)
    write_parquet(collect_events, paths.collect_events)
    write_parquet(liquidity_snapshots, paths.liquidity_snapshots)
    write_parquet(slot0_snapshots, paths.slot0_snapshots)
    paths.validations.write_text(json.dumps(_validation_summary(slot0_consistency, liquidity_validation), indent=2))
    print(f"[Module1] outputs written to {paths.output_dir}", flush=True)

    return paths


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    study_start, study_end = apply_smoke_test_window(args.study_start, args.study_end, args.smoke_test_days)
    run_module_1(
        rpc_url=args.rpc_url,
        pool_address=args.pool_address,
        study_start=study_start,
        study_end=study_end,
        output_dir=args.output_dir,
        log_chunk_size=args.log_chunk_size,
        validation_seed=args.validation_seed,
        progress_seconds=max(0.0, args.progress_seconds),
        smoke_test_days=args.smoke_test_days,
    )


if __name__ == "__main__":
    main()
