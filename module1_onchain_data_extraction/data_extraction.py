"""Module 1: extract on-chain data and reconstruct daily liquidity snapshots."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from random import Random
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
from common.dates import UTC, date_range, utc_midnight
from common.eth_rpc import EthereumArchiveClient
from common.io_utils import ensure_directory, write_parquet
from common.uniswap_math import (
    align_tick_to_spacing,
    amounts_for_liquidity,
    raw_amounts_to_decimal,
    sqrt_price_x96_to_price_usdc_per_weth,
    tick_to_price_usdc_per_weth,
)


@dataclass(frozen=True)
class Module1Paths:
    """Output paths for Module 1 artifacts."""

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


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
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
        help="Block range per eth_getLogs request.",
    )
    parser.add_argument(
        "--validation-seed",
        type=int,
        default=17,
        help="Random seed used for the tick validation sample.",
    )
    return parser.parse_args()


def _timestamp_map(client: EthereumArchiveClient, block_numbers: Iterable[int]) -> pd.DataFrame:
    timestamps = client.block_timestamps_frame(block_numbers)
    return timestamps.rename(columns={"block_number": "block_number", "block_timestamp": "block_timestamp"})


def _signed_decimal(raw_value: int, decimals: int) -> float:
    scale = 10 ** decimals
    return raw_value / scale


def _decode_swaps(
    client: EthereumArchiveClient,
    start_block: int,
    end_block: int,
    chunk_size: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in client.iter_event_logs("Swap", start_block, end_block, chunk_size):
        args = item["args"]
        amount0_raw = int(args["amount0"])
        amount1_raw = int(args["amount1"])
        sqrt_price_x96 = int(args["sqrtPriceX96"])
        price_usdc_per_weth = float(sqrt_price_x96_to_price_usdc_per_weth(sqrt_price_x96))
        usdc_decimal = _signed_decimal(amount0_raw, TOKEN0_DECIMALS)
        weth_decimal = _signed_decimal(amount1_raw, TOKEN1_DECIMALS)
        direction = "buy_weth" if amount0_raw > 0 else "sell_weth"
        notional_usd = abs(usdc_decimal)
        rows.append(
            {
                "block_number": item["block_number"],
                "transaction_hash": item["transaction_hash"],
                "log_index": item["log_index"],
                "amount0_raw": amount0_raw,
                "amount0_usdc": usdc_decimal,
                "amount1_raw": amount1_raw,
                "amount1_weth": weth_decimal,
                "sqrtPriceX96": sqrt_price_x96,
                "price_usdc_per_weth": price_usdc_per_weth,
                "active_liquidity": int(args["liquidity"]),
                "tick": int(args["tick"]),
                "trade_direction": direction,
                "notional_usd": notional_usd,
            }
        )
    swaps = pd.DataFrame(rows).sort_values(["block_number", "log_index"]).reset_index(drop=True)
    if swaps.empty:
        return swaps
    timestamps = _timestamp_map(client, swaps["block_number"].tolist())
    swaps = swaps.merge(timestamps, on="block_number", how="left")
    swaps["block_timestamp"] = pd.to_datetime(swaps["block_timestamp"], utc=True)
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
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in client.iter_event_logs(event_name, start_block, end_block, chunk_size):
        args = item["args"]
        amount0_raw = int(args["amount0"])
        amount1_raw = int(args["amount1"])
        rows.append(
            {
                "block_number": item["block_number"],
                "transaction_hash": item["transaction_hash"],
                "log_index": item["log_index"],
                "event_type": event_name.lower(),
                "owner": args["owner"],
                "tick_lower": int(args["tickLower"]),
                "tick_upper": int(args["tickUpper"]),
                "liquidity_raw": int(args.get("amount", 0)),
                "amount0_raw": amount0_raw,
                "amount0_usdc": amount0_raw / (10 ** TOKEN0_DECIMALS),
                "amount1_raw": amount1_raw,
                "amount1_weth": amount1_raw / (10 ** TOKEN1_DECIMALS),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["block_number", "log_index"]).reset_index(drop=True)
    if frame.empty:
        return frame
    frame = frame.merge(_timestamp_map(client, frame["block_number"].tolist()), on="block_number", how="left")
    frame["block_timestamp"] = pd.to_datetime(frame["block_timestamp"], utc=True)
    return frame


def _decode_collect_events(
    client: EthereumArchiveClient,
    start_block: int,
    end_block: int,
    chunk_size: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in client.iter_event_logs("Collect", start_block, end_block, chunk_size):
        args = item["args"]
        amount0_raw = int(args["amount0"])
        amount1_raw = int(args["amount1"])
        rows.append(
            {
                "block_number": item["block_number"],
                "transaction_hash": item["transaction_hash"],
                "log_index": item["log_index"],
                "owner": args["owner"],
                "recipient": args["recipient"],
                "tick_lower": int(args["tickLower"]),
                "tick_upper": int(args["tickUpper"]),
                "amount0_raw": amount0_raw,
                "amount0_usdc": amount0_raw / (10 ** TOKEN0_DECIMALS),
                "amount1_raw": amount1_raw,
                "amount1_weth": amount1_raw / (10 ** TOKEN1_DECIMALS),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["block_number", "log_index"]).reset_index(drop=True)
    if frame.empty:
        return frame
    frame = frame.merge(_timestamp_map(client, frame["block_number"].tolist()), on="block_number", how="left")
    frame["block_timestamp"] = pd.to_datetime(frame["block_timestamp"], utc=True)
    return frame


def _build_snapshot_schedule(client: EthereumArchiveClient, study_start: date, study_end: date) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for day in date_range(study_start, study_end):
        target_time = utc_midnight(day)
        snapshot_block = client.find_closest_block(target_time)
        rows.append(
            {
                "snapshot_date": day.isoformat(),
                "snapshot_timestamp": target_time,
                "snapshot_block": snapshot_block,
            }
        )
    frame = pd.DataFrame(rows)
    frame["snapshot_timestamp"] = pd.to_datetime(frame["snapshot_timestamp"], utc=True)
    return frame


def _tick_liquidity_deltas(events: pd.DataFrame) -> pd.DataFrame:
    deltas: list[dict[str, int]] = []
    for row in events.itertuples(index=False):
        sign = 1 if row.event_type == "mint" else -1
        delta = sign * int(row.liquidity_raw)
        deltas.append(
            {
                "block_number": int(row.block_number),
                "tick": int(row.tick_lower),
                "liquidity_net_delta": delta,
                "liquidity_gross_delta": sign * int(row.liquidity_raw),
            }
        )
        deltas.append(
            {
                "block_number": int(row.block_number),
                "tick": int(row.tick_upper),
                "liquidity_net_delta": -delta,
                "liquidity_gross_delta": sign * int(row.liquidity_raw),
            }
        )
    return pd.DataFrame(deltas)


def build_liquidity_snapshots(mint_burn_events: pd.DataFrame, snapshot_blocks: pd.DataFrame) -> pd.DataFrame:
    """Replay mint and burn events to reconstruct initialized ticks per snapshot."""
    if mint_burn_events.empty or snapshot_blocks.empty:
        return pd.DataFrame()
    liquidity_events = _tick_liquidity_deltas(mint_burn_events)
    liquidity_events = liquidity_events.sort_values(["block_number", "tick"]).reset_index(drop=True)
    snapshot_blocks = snapshot_blocks.sort_values("snapshot_block").reset_index(drop=True)

    tick_net: defaultdict[int, int] = defaultdict(int)
    tick_gross: defaultdict[int, int] = defaultdict(int)
    rows: list[dict[str, object]] = []
    event_index = 0
    events = liquidity_events.to_dict("records")

    for snapshot in snapshot_blocks.itertuples(index=False):
        while event_index < len(events) and int(events[event_index]["block_number"]) <= int(snapshot.snapshot_block):
            tick = int(events[event_index]["tick"])
            tick_net[tick] += int(events[event_index]["liquidity_net_delta"])
            tick_gross[tick] += int(events[event_index]["liquidity_gross_delta"])
            if tick_gross[tick] == 0:
                tick_gross.pop(tick, None)
                tick_net.pop(tick, None)
            event_index += 1

        active_liquidity = 0
        for tick in sorted(tick_gross):
            active_liquidity += tick_net[tick]
            rows.append(
                {
                    "snapshot_block": int(snapshot.snapshot_block),
                    "snapshot_timestamp": snapshot.snapshot_timestamp,
                    "tick": int(tick),
                    "liquidityNet": int(tick_net[tick]),
                    "liquidityGross": int(tick_gross[tick]),
                    "active_liquidity": int(active_liquidity),
                    "price_lower": float(tick_to_price_usdc_per_weth(tick)),
                    "price_upper": float(tick_to_price_usdc_per_weth(tick + TICK_SPACING)),
                }
            )
    return pd.DataFrame(rows)


def build_slot0_snapshots(client: EthereumArchiveClient, snapshot_blocks: pd.DataFrame) -> pd.DataFrame:
    """Fetch slot0 at each daily snapshot block."""
    rows: list[dict[str, object]] = []
    for snapshot in snapshot_blocks.itertuples(index=False):
        slot0 = client.call_slot0(int(snapshot.snapshot_block))
        rows.append(
            {
                "snapshot_block": int(snapshot.snapshot_block),
                "snapshot_timestamp": snapshot.snapshot_timestamp,
                "sqrtPriceX96": slot0["sqrtPriceX96"],
                "price_usdc_per_weth": float(sqrt_price_x96_to_price_usdc_per_weth(slot0["sqrtPriceX96"])),
                "current_tick": slot0["tick"],
                "observation_index": slot0["observation_index"],
                "unlocked": slot0["unlocked"],
            }
        )
    return pd.DataFrame(rows)


def compare_slot0_to_last_swap(slot0_snapshots: pd.DataFrame, swap_events: pd.DataFrame) -> pd.DataFrame:
    """Compare snapshot slot0 price to the last observed swap price before the snapshot block."""
    if slot0_snapshots.empty or swap_events.empty:
        return pd.DataFrame()
    last_swaps = swap_events[["block_number", "price_usdc_per_weth", "tick"]].sort_values("block_number")
    rows: list[dict[str, object]] = []
    for snapshot in slot0_snapshots.itertuples(index=False):
        candidate_swaps = last_swaps[last_swaps["block_number"] <= int(snapshot.snapshot_block)]
        if candidate_swaps.empty:
            continue
        swap = candidate_swaps.iloc[-1]
        tick_distance = abs(int(snapshot.current_tick) - int(swap["tick"]))
        rows.append(
            {
                "snapshot_block": int(snapshot.snapshot_block),
                "slot0_price": float(snapshot.price_usdc_per_weth),
                "last_swap_price": float(swap["price_usdc_per_weth"]),
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
    """Spot-check one reconstructed snapshot against pool.ticks()."""
    if liquidity_snapshots.empty:
        return pd.DataFrame()
    rng = Random(seed)
    snapshot_block = int(liquidity_snapshots["snapshot_block"].iloc[len(liquidity_snapshots) // 2])
    snapshot = liquidity_snapshots[liquidity_snapshots["snapshot_block"] == snapshot_block].reset_index(drop=True)
    sample_size = min(10, len(snapshot))
    sampled_ticks = sorted(rng.sample(snapshot["tick"].tolist(), sample_size))
    rows: list[dict[str, object]] = []
    for tick in sampled_ticks:
        rpc_state = client.call_tick_state(snapshot_block, int(tick))
        local_state = snapshot[snapshot["tick"] == tick].iloc[0]
        rows.append(
            {
                "snapshot_block": snapshot_block,
                "tick": int(tick),
                "reconstructed_liquidityGross": int(local_state["liquidityGross"]),
                "rpc_liquidityGross": int(rpc_state["liquidityGross"]),
                "reconstructed_liquidityNet": int(local_state["liquidityNet"]),
                "rpc_liquidityNet": int(rpc_state["liquidityNet"]),
                "matches": (
                    int(local_state["liquidityGross"]) == int(rpc_state["liquidityGross"])
                    and int(local_state["liquidityNet"]) == int(rpc_state["liquidityNet"])
                ),
            }
        )
    return pd.DataFrame(rows)


def run_module_1(
    rpc_url: str,
    pool_address: str,
    study_start: date,
    study_end: date,
    output_dir: Path,
    log_chunk_size: int,
    validation_seed: int,
) -> Module1Paths:
    """Execute the full Module 1 pipeline."""
    paths = Module1Paths(output_dir=ensure_directory(output_dir))
    client = EthereumArchiveClient(rpc_url=rpc_url, pool_address=pool_address)

    start_block = client.find_block_at_or_after(utc_midnight(study_start))
    end_block = client.find_closest_block(utc_midnight(study_end.replace(day=study_end.day)) + pd.Timedelta(days=1))

    swap_events = _decode_swaps(client, start_block=start_block, end_block=end_block, chunk_size=log_chunk_size)
    mint_events = _decode_liquidity_events(
        client,
        event_name="Mint",
        start_block=POOL_DEPLOYMENT_BLOCK,
        end_block=end_block,
        chunk_size=log_chunk_size,
    )
    burn_events = _decode_liquidity_events(
        client,
        event_name="Burn",
        start_block=POOL_DEPLOYMENT_BLOCK,
        end_block=end_block,
        chunk_size=log_chunk_size,
    )
    collect_events = _decode_collect_events(
        client,
        start_block=POOL_DEPLOYMENT_BLOCK,
        end_block=end_block,
        chunk_size=log_chunk_size,
    )

    mint_burn_events = (
        pd.concat([mint_events, burn_events], ignore_index=True)
        .sort_values(["block_number", "log_index"])
        .reset_index(drop=True)
    )
    snapshot_blocks = _build_snapshot_schedule(client, study_start=study_start, study_end=study_end)
    liquidity_snapshots = build_liquidity_snapshots(mint_burn_events=mint_burn_events, snapshot_blocks=snapshot_blocks)
    slot0_snapshots = build_slot0_snapshots(client, snapshot_blocks)
    slot0_consistency = compare_slot0_to_last_swap(slot0_snapshots=slot0_snapshots, swap_events=swap_events)
    liquidity_validation = validate_snapshot_against_rpc(
        client=client,
        liquidity_snapshots=liquidity_snapshots,
        seed=validation_seed,
    )

    write_parquet(swap_events, paths.swap_events)
    write_parquet(mint_burn_events, paths.mint_burn_events)
    write_parquet(collect_events, paths.collect_events)
    write_parquet(liquidity_snapshots, paths.liquidity_snapshots)
    write_parquet(slot0_snapshots, paths.slot0_snapshots)

    paths.validations.write_text(
        json.dumps(
            {
                "assumption": "The default study window follows the example given in the project brief: 2025-10-01 to 2026-03-31.",
                "slot0_consistency_summary": {
                    "rows_checked": int(len(slot0_consistency)),
                    "all_within_tick_spacing": bool(slot0_consistency["within_tick_spacing"].all())
                    if not slot0_consistency.empty
                    else None,
                    "max_tick_distance": int(slot0_consistency["tick_distance"].max())
                    if not slot0_consistency.empty
                    else None,
                },
                "liquidity_validation_summary": {
                    "rows_checked": int(len(liquidity_validation)),
                    "all_exact_matches": bool(liquidity_validation["matches"].all()) if not liquidity_validation.empty else None,
                },
            },
            indent=2,
        )
    )
    return paths


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    run_module_1(
        rpc_url=args.rpc_url,
        pool_address=args.pool_address,
        study_start=args.study_start,
        study_end=args.study_end,
        output_dir=args.output_dir,
        log_chunk_size=args.log_chunk_size,
        validation_seed=args.validation_seed,
    )


if __name__ == "__main__":
    main()

