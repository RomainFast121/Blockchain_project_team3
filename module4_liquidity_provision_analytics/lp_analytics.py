"""Module 4: analyse the pool from the LP point of view.

This module creates the five representative LP positions described in the PDF
and tracks three quantities through time:

- cumulative fee income,
- impermanent loss relative to a HODL benchmark,
- net P&L defined as fee income minus impermanent loss.

The implementation stays close to the report story:
`define positions -> assign fee income -> mark positions over time -> save
tables -> draw figures`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common.constants import (
    FIGURES_DIR,
    POOL_FEE_RATE,
    PROCESSED_DATA_DIR,
    REPRESENTATIVE_POSITIONS,
    TICK_SPACING,
    UNISWAP_V3_MAX_TICK,
    UNISWAP_V3_MIN_TICK,
)
from common.io_utils import ensure_directory, read_parquet, write_parquet
from common.plotting import save_figure, set_project_style
from common.uniswap_math import (
    align_tick_to_spacing,
    price_usdc_per_weth_to_tick,
    solve_liquidity_for_budget,
    synthetic_lp_value,
    tick_interval_prices,
)


TARGET_POSITION_BUDGET = 100_000


@dataclass(frozen=True)
class Module4Paths:
    """All files produced by Module 4."""

    data_dir: Path
    figure_dir: Path

    @property
    def positions(self) -> Path:
        return self.data_dir / "synthetic_lp_positions.parquet"

    @property
    def fee_accruals(self) -> Path:
        return self.data_dir / "lp_fee_accruals.parquet"

    @property
    def position_timeseries(self) -> Path:
        return self.data_dir / "lp_position_timeseries.parquet"

    @property
    def fig_41(self) -> Path:
        return self.figure_dir / "fig_4_1_cumulative_fee_income.png"

    @property
    def fig_42(self) -> Path:
        return self.figure_dir / "fig_4_2_impermanent_loss.png"

    @property
    def fig_43(self) -> Path:
        return self.figure_dir / "fig_4_3_fee_income_minus_il.png"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIGURES_DIR)
    return parser.parse_args()


def _range_ticks_from_width(entry_price: float, width_pct: float | None) -> tuple[int, int, float, float]:
    """Convert a width definition from the PDF into aligned Uniswap ticks."""

    if width_pct is None:
        lower_tick = align_tick_to_spacing(UNISWAP_V3_MIN_TICK, tick_spacing=TICK_SPACING, mode="up")
        upper_tick = align_tick_to_spacing(UNISWAP_V3_MAX_TICK, tick_spacing=TICK_SPACING, mode="down")
    else:
        target_price_low = entry_price * (1 - width_pct / 100)
        target_price_high = entry_price * (1 + width_pct / 100)
        tick_a = align_tick_to_spacing(price_usdc_per_weth_to_tick(target_price_low), tick_spacing=TICK_SPACING, mode="nearest")
        tick_b = align_tick_to_spacing(price_usdc_per_weth_to_tick(target_price_high), tick_spacing=TICK_SPACING, mode="nearest")
        lower_tick = min(tick_a, tick_b)
        upper_tick = max(tick_a, tick_b)

    price_low, price_high = tick_interval_prices(lower_tick, upper_tick)
    return lower_tick, upper_tick, float(price_low), float(price_high)


def build_representative_positions(slot0_snapshots: pd.DataFrame) -> pd.DataFrame:
    """Build the five synthetic LP positions required by the project brief."""

    ordered_snapshots = slot0_snapshots.sort_values("snapshot_timestamp").reset_index(drop=True)
    entry_snapshot = ordered_snapshots.iloc[0]
    exit_snapshot = ordered_snapshots.iloc[-1]
    entry_price = float(entry_snapshot["price_usdc_per_weth"])
    rows: list[dict[str, object]] = []

    for definition in REPRESENTATIVE_POSITIONS:
        lower_tick, upper_tick, price_low, price_high = _range_ticks_from_width(entry_price, definition.width_pct)
        liquidity_raw, entry_usdc, entry_weth = solve_liquidity_for_budget(
            budget_usd=TARGET_POSITION_BUDGET,
            entry_price=entry_price,
            price_lower=price_low,
            price_upper=price_high,
        )
        rows.append(
            {
                "position_id": definition.name,
                "position_label": definition.label,
                "width_pct": definition.width_pct,
                "entry_snapshot_block": int(entry_snapshot["snapshot_block"]),
                "exit_snapshot_block": int(exit_snapshot["snapshot_block"]),
                "entry_timestamp": entry_snapshot["snapshot_timestamp"],
                "exit_timestamp": exit_snapshot["snapshot_timestamp"],
                "entry_price_usdc_per_weth": entry_price,
                "tick_lower": lower_tick,
                "tick_upper": upper_tick,
                "price_lower": price_low,
                "price_upper": price_high,
                "liquidity_raw": int(liquidity_raw),
                "entry_usdc": float(entry_usdc),
                "entry_weth": float(entry_weth),
                "entry_budget_usd": TARGET_POSITION_BUDGET,
            }
        )

    return pd.DataFrame(rows)


def compute_fee_accruals(positions: pd.DataFrame, swap_events: pd.DataFrame) -> pd.DataFrame:
    """Assign swap fees to each synthetic LP when the swap occurs inside its range.

    We approximate fee share as:

    `position liquidity / active pool liquidity at the swap tick`.

    This is the cleanest simple rule for this project because swap-level fee
    growth inside each tick range is not reconstructed separately.
    """

    swaps = swap_events.sort_values(["block_timestamp", "block_number", "log_index"]).reset_index(drop=True)
    rows: list[dict[str, object]] = []

    for position in positions.itertuples(index=False):
        position_swaps = swaps[
            (swaps["block_timestamp"] >= position.entry_timestamp)
            & (swaps["block_timestamp"] <= position.exit_timestamp)
        ]
        for swap in position_swaps.itertuples(index=False):
            # We use the swap tick itself, not just the post-swap price, because
            # that is closer to the "was the position active during this swap?"
            # question the PDF cares about.
            in_range = int(position.tick_lower) <= int(swap.tick) < int(position.tick_upper)
            if not in_range or float(swap.active_liquidity) <= 0:
                continue

            liquidity_share = float(position.liquidity_raw) / float(swap.active_liquidity)
            fee_usdc = max(float(swap.amount0_usdc), 0.0) * POOL_FEE_RATE * liquidity_share
            fee_weth = max(float(swap.amount1_weth), 0.0) * POOL_FEE_RATE * liquidity_share
            fee_value_usd = fee_usdc + fee_weth * float(swap.price_usdc_per_weth)

            rows.append(
                {
                    "position_id": position.position_id,
                    "block_number": int(swap.block_number),
                    "block_timestamp": swap.block_timestamp,
                    "trade_direction": swap.trade_direction,
                    "swap_price_usdc_per_weth": float(swap.price_usdc_per_weth),
                    "fee_usdc": fee_usdc,
                    "fee_weth": fee_weth,
                    "fee_value_usd": fee_value_usd,
                }
            )

    fee_flows = pd.DataFrame(rows)
    if fee_flows.empty:
        return fee_flows

    fee_flows = fee_flows.sort_values(["position_id", "block_timestamp"]).reset_index(drop=True)
    fee_flows["cumulative_fee_income_usd"] = fee_flows.groupby("position_id")["fee_value_usd"].cumsum()
    return fee_flows


def build_position_timeseries(
    positions: pd.DataFrame,
    slot0_snapshots: pd.DataFrame,
    fee_accruals: pd.DataFrame,
) -> pd.DataFrame:
    """Mark LP value, HODL value, IL, and net P&L through time."""

    snapshots = slot0_snapshots.sort_values("snapshot_timestamp").reset_index(drop=True).copy()
    rows: list[dict[str, object]] = []

    for position in positions.itertuples(index=False):
        position_fee_flows = fee_accruals[fee_accruals["position_id"] == position.position_id].copy()
        for snapshot in snapshots.itertuples(index=False):
            price = float(snapshot.price_usdc_per_weth)
            lp_principal_usd = float(
                synthetic_lp_value(
                    liquidity_raw=position.liquidity_raw,
                    price_usdc_per_weth=price,
                    price_lower=position.price_lower,
                    price_upper=position.price_upper,
                )
            )

            # The HODL benchmark is simply "keep the entry token balances untouched".
            hodl_value_usd = float(position.entry_usdc + position.entry_weth * price)
            impermanent_loss_usd = hodl_value_usd - lp_principal_usd

            cumulative_fee_income_usd = 0.0
            if not position_fee_flows.empty:
                earned_so_far = position_fee_flows[position_fee_flows["block_timestamp"] <= snapshot.snapshot_timestamp]
                if not earned_so_far.empty:
                    cumulative_fee_income_usd = float(earned_so_far.iloc[-1]["cumulative_fee_income_usd"])

            rows.append(
                {
                    "position_id": position.position_id,
                    "position_label": position.position_label,
                    "snapshot_block": int(snapshot.snapshot_block),
                    "snapshot_timestamp": snapshot.snapshot_timestamp,
                    "price_usdc_per_weth": price,
                    "lp_principal_usd": lp_principal_usd,
                    "hodl_value_usd": hodl_value_usd,
                    "impermanent_loss_usd": impermanent_loss_usd,
                    "cumulative_fee_income_usd": cumulative_fee_income_usd,
                    "net_pnl_usd": cumulative_fee_income_usd - impermanent_loss_usd,
                }
            )

    return pd.DataFrame(rows)


def plot_position_lines(timeseries: pd.DataFrame, value_column: str, title: str, y_label: str, path: Path) -> None:
    """Plot one line per representative LP position."""

    set_project_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    for position_id, frame in timeseries.groupby("position_id"):
        ax.plot(frame["snapshot_timestamp"], frame[value_column], linewidth=1.8, label=position_id)
    ax.set_title(title)
    ax.set_xlabel("Snapshot date")
    ax.set_ylabel(y_label)
    ax.legend(loc="upper left", ncol=3)
    save_figure(fig, path)


def run_module_4(data_dir: Path, figure_dir: Path) -> Module4Paths:
    """Execute the full Module 4 workflow."""

    paths = Module4Paths(data_dir=ensure_directory(data_dir), figure_dir=ensure_directory(figure_dir))
    slot0_snapshots = read_parquet(data_dir / "slot0_snapshots.parquet")
    swap_events = read_parquet(data_dir / "swap_events.parquet")
    slot0_snapshots["snapshot_timestamp"] = pd.to_datetime(slot0_snapshots["snapshot_timestamp"], utc=True)
    swap_events["block_timestamp"] = pd.to_datetime(swap_events["block_timestamp"], utc=True)

    positions = build_representative_positions(slot0_snapshots)
    fee_accruals = compute_fee_accruals(positions, swap_events)
    position_timeseries = build_position_timeseries(positions, slot0_snapshots, fee_accruals)

    write_parquet(positions, paths.positions)
    write_parquet(fee_accruals, paths.fee_accruals)
    write_parquet(position_timeseries, paths.position_timeseries)
    plot_position_lines(position_timeseries, "cumulative_fee_income_usd", "Cumulative fee income", "USD", paths.fig_41)
    plot_position_lines(position_timeseries, "impermanent_loss_usd", "Impermanent loss", "USD", paths.fig_42)
    plot_position_lines(position_timeseries, "net_pnl_usd", "Cumulative fee income minus impermanent loss", "USD", paths.fig_43)
    return paths


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    run_module_4(data_dir=args.data_dir, figure_dir=args.figure_dir)


if __name__ == "__main__":
    main()
