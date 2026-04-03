"""Module 3: run the swap simulator grid and produce execution-cost figures."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.constants import FIGURES_DIR, POOL_ADDRESS, PROCESSED_DATA_DIR
from common.eth_rpc import EthereumArchiveClient
from common.io_utils import ensure_directory, read_parquet, write_parquet
from common.plotting import save_figure, set_project_style
from common.uniswap_math import sqrt_price_x96_to_price_usdc_per_weth
from module1_onchain_data_extraction.data_extraction import build_liquidity_snapshots
from module3_slippage_simulation_and_execution_cost.swap_simulator import (
    SnapshotPoolState,
    simulate_exact_input_swap,
)


SIMULATION_SIZES = [1_000, 10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000]


@dataclass(frozen=True)
class Module3Paths:
    """Output paths for Module 3 artifacts."""

    data_dir: Path
    figure_dir: Path

    @property
    def simulated_trades(self) -> Path:
        return self.data_dir / "simulated_trades.parquet"

    @property
    def validation_table(self) -> Path:
        return self.data_dir / "simulator_validation.parquet"

    @property
    def effective_spreads(self) -> Path:
        return self.data_dir / "observed_effective_spreads.parquet"

    @property
    def fig_31(self) -> Path:
        return self.figure_dir / "fig_3_1_simulator_validation.png"

    @property
    def fig_32(self) -> Path:
        return self.figure_dir / "fig_3_2_price_impact_curves.png"

    @property
    def fig_33(self) -> Path:
        return self.figure_dir / "fig_3_3_effective_spread_vs_simulated.png"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--rpc-url", required=True, help="Archive RPC URL used for validation and effective spreads.")
    parser.add_argument("--pool-address", default=None, help="Optional pool address override.")
    return parser.parse_args()


def load_module3_inputs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the Module 1 artifacts needed for Module 3."""
    liquidity_snapshots = read_parquet(data_dir / "liquidity_snapshots.parquet")
    slot0_snapshots = read_parquet(data_dir / "slot0_snapshots.parquet")
    swap_events = read_parquet(data_dir / "swap_events.parquet")
    liquidity_snapshots["snapshot_timestamp"] = pd.to_datetime(liquidity_snapshots["snapshot_timestamp"], utc=True)
    slot0_snapshots["snapshot_timestamp"] = pd.to_datetime(slot0_snapshots["snapshot_timestamp"], utc=True)
    swap_events["block_timestamp"] = pd.to_datetime(swap_events["block_timestamp"], utc=True)
    return liquidity_snapshots, slot0_snapshots, swap_events


def build_snapshot_states(liquidity_snapshots: pd.DataFrame, slot0_snapshots: pd.DataFrame) -> dict[int, SnapshotPoolState]:
    """Materialize a simulation state for each daily snapshot."""
    states: dict[int, SnapshotPoolState] = {}
    for snapshot_block, frame in liquidity_snapshots.groupby("snapshot_block"):
        slot0_row = slot0_snapshots.loc[slot0_snapshots["snapshot_block"] == snapshot_block].iloc[0]
        states[int(snapshot_block)] = SnapshotPoolState.from_snapshot_frames(frame, slot0_row)
    return states


def run_simulation_grid(snapshot_states: dict[int, SnapshotPoolState]) -> pd.DataFrame:
    """Run the full trade-size grid across all daily snapshots."""
    rows: list[dict[str, object]] = []
    for snapshot_block, state in snapshot_states.items():
        for direction in ("buy_weth", "sell_weth"):
            for size in SIMULATION_SIZES:
                result = simulate_exact_input_swap(state=state, direction=direction, notional_usd=size)
                rows.append(
                    {
                        "snapshot_block": snapshot_block,
                        "snapshot_timestamp": state.snapshot_timestamp,
                        "direction": direction,
                        "notional_usd": size,
                        "average_price": result.average_price,
                        "pool_mid_price": result.pool_mid_price,
                        "price_impact_bps": result.price_impact_bps,
                        "slippage_bps": result.slippage_bps,
                        "tick_crosses": result.tick_crosses,
                        "ending_price": result.ending_price,
                    }
                )
    return pd.DataFrame(rows)


def _assign_bucket(notional_usd: float) -> int:
    """Assign a continuous trade notional to the closest simulation bucket."""
    return min(SIMULATION_SIZES, key=lambda bucket: abs(np.log(max(notional_usd, 1.0) / bucket)))


def build_validation_table(
    rpc_url: str,
    pool_address: str,
    swap_events: pd.DataFrame,
    mint_burn_events: pd.DataFrame,
) -> pd.DataFrame:
    """Validate the simulator against a small sample of real observed swaps."""
    sampled_swaps = (
        swap_events.assign(size_bucket=swap_events["notional_usd"].apply(_assign_bucket))
        .sort_values(["trade_direction", "size_bucket", "block_number"])
        .groupby(["trade_direction", "size_bucket"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    if sampled_swaps.empty:
        return pd.DataFrame()

    client = EthereumArchiveClient(rpc_url=rpc_url, pool_address=pool_address)
    validation_blocks = pd.DataFrame(
        {
            "snapshot_block": [max(int(block) - 1, 0) for block in sampled_swaps["block_number"]],
            "snapshot_timestamp": [pd.Timestamp(client.get_block_timestamp(max(int(block) - 1, 0))) for block in sampled_swaps["block_number"]],
        }
    )
    liquidity_before = build_liquidity_snapshots(mint_burn_events, validation_blocks)
    slot0_rows = []
    for block in validation_blocks["snapshot_block"]:
        slot0 = client.call_slot0(int(block))
        slot0_rows.append(
            {
                "snapshot_block": int(block),
                "snapshot_timestamp": pd.Timestamp(client.get_block_timestamp(int(block))),
                "sqrtPriceX96": slot0["sqrtPriceX96"],
                "price_usdc_per_weth": float(sqrt_price_x96_to_price_usdc_per_weth(int(slot0["sqrtPriceX96"]))),
                "current_tick": slot0["tick"],
            }
        )
    slot0_before = pd.DataFrame(slot0_rows)

    rows: list[dict[str, object]] = []
    for swap in sampled_swaps.itertuples(index=False):
        block_before = max(int(swap.block_number) - 1, 0)
        liquidity_frame = liquidity_before[liquidity_before["snapshot_block"] == block_before]
        slot0_row = slot0_before.loc[slot0_before["snapshot_block"] == block_before].iloc[0]
        state = SnapshotPoolState.from_snapshot_frames(liquidity_frame, slot0_row)
        result = simulate_exact_input_swap(state=state, direction=swap.trade_direction, notional_usd=float(swap.notional_usd))
        actual_price = abs(float(swap.amount0_usdc) / float(swap.amount1_weth))
        error_pct = abs(result.average_price - actual_price) / actual_price * 100
        rows.append(
            {
                "direction": swap.trade_direction,
                "size_bucket_usd": int(_assign_bucket(float(swap.notional_usd))),
                "block_number": int(swap.block_number),
                "actual_execution_price": actual_price,
                "simulated_execution_price": result.average_price,
                "percentage_error": error_pct,
            }
        )
    return pd.DataFrame(rows)


def build_effective_spread_dataset(
    rpc_url: str,
    pool_address: str,
    swap_events: pd.DataFrame,
) -> pd.DataFrame:
    """Compute observed effective spreads using slot0 at the previous block."""
    client = EthereumArchiveClient(rpc_url=rpc_url, pool_address=pool_address)
    unique_prior_blocks = sorted({max(int(block) - 1, 0) for block in swap_events["block_number"]})
    prior_prices = {}
    for block in unique_prior_blocks:
        slot0 = client.call_slot0(block)
        prior_prices[block] = float(sqrt_price_x96_to_price_usdc_per_weth(slot0["sqrtPriceX96"]))

    rows: list[dict[str, object]] = []
    for swap in swap_events.itertuples(index=False):
        prior_block = max(int(swap.block_number) - 1, 0)
        mid_price = prior_prices[prior_block]
        execution_price = abs(float(swap.amount0_usdc) / float(swap.amount1_weth))
        direction_sign = 1 if swap.trade_direction == "buy_weth" else -1
        effective_spread_bps = 2 * direction_sign * (execution_price - mid_price) / mid_price * 10_000
        rows.append(
            {
                "block_number": int(swap.block_number),
                "block_timestamp": swap.block_timestamp,
                "direction": swap.trade_direction,
                "notional_usd": float(swap.notional_usd),
                "size_bucket_usd": int(_assign_bucket(float(swap.notional_usd))),
                "execution_price": execution_price,
                "mid_price_prior_block": mid_price,
                "effective_spread_bps": effective_spread_bps,
            }
        )
    return pd.DataFrame(rows)


def render_validation_table(validation: pd.DataFrame, path: Path) -> None:
    """Render the validation table as a figure."""
    set_project_style()
    fig, ax = plt.subplots(figsize=(10, 3 + 0.4 * max(len(validation), 1)))
    ax.axis("off")
    display = validation.copy()
    if not display.empty:
        display["percentage_error"] = display["percentage_error"].map(lambda value: f"{value:.2f}%")
        display["actual_execution_price"] = display["actual_execution_price"].map(lambda value: f"{value:,.2f}")
        display["simulated_execution_price"] = display["simulated_execution_price"].map(lambda value: f"{value:,.2f}")
    table = ax.table(cellText=display.values, colLabels=display.columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    save_figure(fig, path)


def plot_price_impact_curves(simulated_trades: pd.DataFrame, path: Path) -> None:
    """Plot median price impact with percentile bands across snapshots."""
    set_project_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for axis, direction in zip(axes, ("buy_weth", "sell_weth"), strict=True):
        subset = simulated_trades[simulated_trades["direction"] == direction].copy()
        summary = (
            subset.groupby("notional_usd")["price_impact_bps"]
            .agg(
                median="median",
                p10=lambda series: series.quantile(0.10),
                p90=lambda series: series.quantile(0.90),
            )
            .reset_index()
        )
        axis.plot(summary["notional_usd"], summary["median"], linewidth=2.0, label="Median impact")
        axis.fill_between(summary["notional_usd"], summary["p10"], summary["p90"], alpha=0.3, label="10th-90th pct.")
        axis.set_xscale("log")
        axis.set_title(direction.replace("_", " ").title())
        axis.set_xlabel("Trade size (USD)")
    axes[0].set_ylabel("Price impact (bps)")
    axes[0].legend(loc="upper left")
    save_figure(fig, path)


def plot_effective_spread_comparison(simulated_trades: pd.DataFrame, observed_spreads: pd.DataFrame, path: Path) -> None:
    """Compare simulated price impact with observed effective spreads."""
    set_project_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for axis, direction in zip(axes, ("buy_weth", "sell_weth"), strict=True):
        sim = (
            simulated_trades[simulated_trades["direction"] == direction]
            .groupby("notional_usd", as_index=False)["price_impact_bps"]
            .median()
            .rename(columns={"price_impact_bps": "simulated_median_bps"})
        )
        obs = (
            observed_spreads[observed_spreads["direction"] == direction]
            .groupby("size_bucket_usd", as_index=False)["effective_spread_bps"]
            .median()
            .rename(columns={"size_bucket_usd": "notional_usd", "effective_spread_bps": "observed_median_bps"})
        )
        merged = sim.merge(obs, on="notional_usd", how="left")
        axis.plot(merged["notional_usd"], merged["simulated_median_bps"], marker="o", label="Simulated")
        axis.plot(merged["notional_usd"], merged["observed_median_bps"], marker="s", label="Observed")
        axis.set_xscale("log")
        axis.set_title(direction.replace("_", " ").title())
        axis.set_xlabel("Trade size bucket (USD)")
    axes[0].set_ylabel("Execution cost (bps)")
    axes[0].legend(loc="upper left")
    save_figure(fig, path)


def run_module_3(data_dir: Path, figure_dir: Path, rpc_url: str, pool_address: str) -> Module3Paths:
    """Execute the full Module 3 pipeline."""
    paths = Module3Paths(data_dir=ensure_directory(data_dir), figure_dir=ensure_directory(figure_dir))
    liquidity_snapshots, slot0_snapshots, swap_events = load_module3_inputs(data_dir)
    mint_burn_events = read_parquet(data_dir / "mint_burn_events.parquet")
    snapshot_states = build_snapshot_states(liquidity_snapshots, slot0_snapshots)
    simulated_trades = run_simulation_grid(snapshot_states)
    validation = build_validation_table(rpc_url=rpc_url, pool_address=pool_address, swap_events=swap_events, mint_burn_events=mint_burn_events)
    observed_spreads = build_effective_spread_dataset(rpc_url=rpc_url, pool_address=pool_address, swap_events=swap_events)

    write_parquet(simulated_trades, paths.simulated_trades)
    write_parquet(validation, paths.validation_table)
    write_parquet(observed_spreads, paths.effective_spreads)
    render_validation_table(validation, paths.fig_31)
    plot_price_impact_curves(simulated_trades, paths.fig_32)
    plot_effective_spread_comparison(simulated_trades, observed_spreads, paths.fig_33)
    return paths


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    pool_address = args.pool_address or POOL_ADDRESS
    run_module_3(data_dir=args.data_dir, figure_dir=args.figure_dir, rpc_url=args.rpc_url, pool_address=pool_address)


if __name__ == "__main__":
    main()
