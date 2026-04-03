"""Module 2: liquidity distribution analysis and report figures."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.constants import FIGURES_DIR, PROCESSED_DATA_DIR
from common.io_utils import ensure_directory, read_parquet, write_parquet
from common.plotting import save_figure, set_project_style
from common.uniswap_math import (
    amounts_for_liquidity,
    raw_amounts_to_decimal,
    tick_to_price_usdc_per_weth,
    tick_to_sqrt_price_x96,
)


@dataclass(frozen=True)
class Module2Paths:
    """Figure and metric output paths for Module 2."""

    figure_dir: Path
    data_dir: Path

    @property
    def fig_21(self) -> Path:
        return self.figure_dir / "fig_2_1_liquidity_profiles.png"

    @property
    def fig_22(self) -> Path:
        return self.figure_dir / "fig_2_2_tvl_decomposition.png"

    @property
    def fig_23(self) -> Path:
        return self.figure_dir / "fig_2_3_ilr_timeseries.png"

    @property
    def fig_24(self) -> Path:
        return self.figure_dir / "fig_2_4_lhhi_vs_eth_price.png"

    @property
    def tvl_decomposition(self) -> Path:
        return self.data_dir / "tvl_decomposition.parquet"

    @property
    def concentration_metrics(self) -> Path:
        return self.data_dir / "liquidity_concentration_metrics.parquet"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIGURES_DIR)
    return parser.parse_args()


def load_module2_inputs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the parquet files required for Module 2."""
    liquidity_snapshots = read_parquet(data_dir / "liquidity_snapshots.parquet")
    slot0_snapshots = read_parquet(data_dir / "slot0_snapshots.parquet")
    mint_burn_events = read_parquet(data_dir / "mint_burn_events.parquet")
    liquidity_snapshots["snapshot_timestamp"] = pd.to_datetime(liquidity_snapshots["snapshot_timestamp"], utc=True)
    slot0_snapshots["snapshot_timestamp"] = pd.to_datetime(slot0_snapshots["snapshot_timestamp"], utc=True)
    mint_burn_events["block_timestamp"] = pd.to_datetime(mint_burn_events["block_timestamp"], utc=True)
    return liquidity_snapshots, slot0_snapshots, mint_burn_events


def expand_liquidity_profile(snapshot_frame: pd.DataFrame) -> pd.DataFrame:
    """Expand initialized-tick rows into tick-spacing intervals with constant active liquidity."""
    frame = snapshot_frame.sort_values("tick").reset_index(drop=True)
    if frame.empty:
        return frame

    rows: list[dict[str, object]] = []
    for index, row in enumerate(frame.itertuples(index=False)):
        current_tick = int(row.tick)
        next_tick = int(frame.iloc[index + 1]["tick"]) if index + 1 < len(frame) else current_tick + 10
        for interval_tick in range(current_tick, next_tick, 10):
            rows.append(
                {
                    "snapshot_block": int(row.snapshot_block),
                    "snapshot_timestamp": row.snapshot_timestamp,
                    "tick": interval_tick,
                    "active_liquidity": float(row.active_liquidity),
                    "price_lower": float(tick_to_price_usdc_per_weth(interval_tick)),
                    "price_upper": float(tick_to_price_usdc_per_weth(interval_tick + 10)),
                }
            )
    return pd.DataFrame(rows)


def pick_reference_snapshots(slot0_snapshots: pd.DataFrame) -> dict[str, int]:
    """Pick the start, high-volatility, and end-of-window snapshot blocks."""
    ordered = slot0_snapshots.sort_values("snapshot_timestamp").reset_index(drop=True).copy()
    price_series = ordered["price_usdc_per_weth"].astype(float)
    ordered["log_return"] = np.log(price_series / price_series.shift(1)).fillna(0.0)
    ordered["abs_log_return"] = ordered["log_return"].abs()
    high_volatility_block = int(ordered.loc[ordered["abs_log_return"].idxmax(), "snapshot_block"]) if len(ordered) > 1 else int(ordered.iloc[0]["snapshot_block"])
    return {
        "start": int(ordered.iloc[0]["snapshot_block"]),
        "high_volatility": high_volatility_block,
        "end": int(ordered.iloc[-1]["snapshot_block"]),
    }


def build_tvl_decomposition(mint_burn_events: pd.DataFrame, slot0_snapshots: pd.DataFrame) -> pd.DataFrame:
    """Compute daily TVL decomposition by active range category."""
    events = mint_burn_events.sort_values(["block_number", "log_index"]).reset_index(drop=True)
    snapshots = slot0_snapshots.sort_values("snapshot_block").reset_index(drop=True)
    active_ranges: defaultdict[tuple[int, int], int] = defaultdict(int)
    event_index = 0
    rows: list[dict[str, object]] = []
    records = events.to_dict("records")

    for snapshot in snapshots.itertuples(index=False):
        while event_index < len(records) and int(records[event_index]["block_number"]) <= int(snapshot.snapshot_block):
            event = records[event_index]
            key = (int(event["tick_lower"]), int(event["tick_upper"]))
            signed_liquidity = int(event["liquidity_raw"]) if event["event_type"] == "mint" else -int(event["liquidity_raw"])
            active_ranges[key] += signed_liquidity
            if active_ranges[key] == 0:
                active_ranges.pop(key, None)
            event_index += 1

        in_range_value = 0.0
        above_value = 0.0
        below_value = 0.0
        current_price = float(snapshot.price_usdc_per_weth)
        for (tick_lower, tick_upper), liquidity_raw in active_ranges.items():
            if liquidity_raw <= 0:
                continue
            lower_price = float(min(tick_to_price_usdc_per_weth(tick_lower), tick_to_price_usdc_per_weth(tick_upper)))
            upper_price = float(max(tick_to_price_usdc_per_weth(tick_lower), tick_to_price_usdc_per_weth(tick_upper)))
            amount0_raw, amount1_raw = amounts_for_liquidity(
                sqrt_price_x96=int(snapshot.sqrtPriceX96),
                sqrt_price_a_x96=tick_to_sqrt_price_x96(tick_lower),
                sqrt_price_b_x96=tick_to_sqrt_price_x96(tick_upper),
                liquidity=liquidity_raw,
            )
            amount0_usdc, amount1_weth = raw_amounts_to_decimal(amount0_raw, amount1_raw)
            position_value = float(amount0_usdc + amount1_weth * Decimal(str(snapshot.price_usdc_per_weth)))

            if current_price < lower_price:
                above_value += position_value
            elif current_price > upper_price:
                below_value += position_value
            else:
                in_range_value += position_value

        rows.append(
            {
                "snapshot_block": int(snapshot.snapshot_block),
                "snapshot_timestamp": snapshot.snapshot_timestamp,
                "price_usdc_per_weth": current_price,
                "tvl_in_range": in_range_value,
                "tvl_above_range": above_value,
                "tvl_below_range": below_value,
                "tvl_total": in_range_value + above_value + below_value,
            }
        )
    return pd.DataFrame(rows)


def compute_concentration_metrics(liquidity_snapshots: pd.DataFrame, slot0_snapshots: pd.DataFrame) -> pd.DataFrame:
    """Compute ILR(k) and L-HHI at every snapshot."""
    rows: list[dict[str, object]] = []

    for snapshot_block, initialized_frame in liquidity_snapshots.groupby("snapshot_block"):
        expanded_frame = expand_liquidity_profile(initialized_frame)
        expanded_frame = expanded_frame.merge(
            slot0_snapshots[["snapshot_block", "price_usdc_per_weth"]],
            on="snapshot_block",
            how="left",
        )
        expanded_frame["price_midpoint"] = (expanded_frame["price_lower"] * expanded_frame["price_upper"]) ** 0.5
        expanded_frame = expanded_frame[expanded_frame["active_liquidity"] > 0].copy()
        if expanded_frame.empty:
            continue
        total_liquidity = float(expanded_frame["active_liquidity"].sum())
        current_price = float(expanded_frame["price_usdc_per_weth"].iloc[0])
        liquidity_shares = expanded_frame["active_liquidity"] / total_liquidity
        row = {
            "snapshot_block": int(snapshot_block),
            "snapshot_timestamp": expanded_frame["snapshot_timestamp"].iloc[0],
            "price_usdc_per_weth": current_price,
            "l_hhi": float((liquidity_shares ** 2).sum()),
        }
        for bandwidth in (0.1, 0.5, 1.0, 2.0, 5.0):
            half_band = bandwidth / 100
            low = current_price * (1 - half_band)
            high = current_price * (1 + half_band)
            in_band = expanded_frame[
                (expanded_frame["price_midpoint"] >= low) & (expanded_frame["price_midpoint"] <= high)
            ]["active_liquidity"].sum()
            row[f"ilr_{str(bandwidth).replace('.', '_')}pct"] = float(in_band / total_liquidity)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("snapshot_timestamp").reset_index(drop=True)


def plot_liquidity_profiles(
    liquidity_snapshots: pd.DataFrame,
    slot0_snapshots: pd.DataFrame,
    snapshot_lookup: dict[str, int],
    path: Path,
) -> None:
    """Plot side-by-side liquidity profiles for three key snapshots."""
    set_project_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    labels = {
        "start": "Start of window",
        "high_volatility": "High-volatility day",
        "end": "End of window",
    }

    for axis, key in zip(axes, ["start", "high_volatility", "end"], strict=True):
        snapshot_block = snapshot_lookup[key]
        frame = expand_liquidity_profile(liquidity_snapshots[liquidity_snapshots["snapshot_block"] == snapshot_block])
        frame["price_midpoint"] = (frame["price_lower"] * frame["price_upper"]) ** 0.5
        axis.bar(frame["price_midpoint"], frame["active_liquidity"], width=frame["price_upper"] - frame["price_lower"], alpha=0.65)
        current_price = float(slot0_snapshots.loc[slot0_snapshots["snapshot_block"] == snapshot_block, "price_usdc_per_weth"].iloc[0])
        axis.axvline(current_price, color="crimson", linestyle="--", linewidth=1.5, label="Current price")
        axis.set_title(labels[key])
        axis.set_xlabel("USDC per WETH")
    axes[0].set_ylabel("Active liquidity")
    axes[-1].legend(loc="upper right")
    save_figure(fig, path)


def plot_tvl_decomposition(tvl_decomposition: pd.DataFrame, path: Path) -> None:
    """Plot the TVL decomposition stacked area chart."""
    set_project_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(
        tvl_decomposition["snapshot_timestamp"],
        tvl_decomposition["tvl_in_range"],
        tvl_decomposition["tvl_above_range"],
        tvl_decomposition["tvl_below_range"],
        labels=["In range", "Above range", "Below range"],
        alpha=0.85,
    )
    ax.set_title("TVL decomposition over time")
    ax.set_xlabel("Snapshot date")
    ax.set_ylabel("USD value")
    ax.legend(loc="upper left")
    save_figure(fig, path)


def plot_ilr_series(metrics: pd.DataFrame, path: Path) -> None:
    """Plot the ILR(k) time series."""
    set_project_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = {
        "ilr_0_1pct": "ILR +/-0.1%",
        "ilr_0_5pct": "ILR +/-0.5%",
        "ilr_1_0pct": "ILR +/-1%",
        "ilr_2_0pct": "ILR +/-2%",
        "ilr_5_0pct": "ILR +/-5%",
    }
    for column, label in labels.items():
        ax.plot(metrics["snapshot_timestamp"], metrics[column], label=label, linewidth=1.6)
    ax.set_title("In-range liquidity ratio across bandwidths")
    ax.set_xlabel("Snapshot date")
    ax.set_ylabel("Liquidity share")
    ax.legend(loc="upper left", ncol=2)
    save_figure(fig, path)


def plot_lhhi_vs_price(metrics: pd.DataFrame, path: Path) -> None:
    """Plot L-HHI and ETH price on dual axes."""
    set_project_style()
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    ax1.plot(metrics["snapshot_timestamp"], metrics["l_hhi"], color="navy", linewidth=1.7, label="L-HHI")
    ax2.plot(metrics["snapshot_timestamp"], metrics["price_usdc_per_weth"], color="darkorange", linewidth=1.4, label="ETH price")
    ax1.set_title("Liquidity concentration and ETH price")
    ax1.set_xlabel("Snapshot date")
    ax1.set_ylabel("L-HHI", color="navy")
    ax2.set_ylabel("USDC per WETH", color="darkorange")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="upper left")
    save_figure(fig, path)


def run_module_2(data_dir: Path, figure_dir: Path) -> Module2Paths:
    """Execute the full Module 2 pipeline."""
    paths = Module2Paths(figure_dir=ensure_directory(figure_dir), data_dir=ensure_directory(data_dir))
    liquidity_snapshots, slot0_snapshots, mint_burn_events = load_module2_inputs(data_dir)
    reference_snapshots = pick_reference_snapshots(slot0_snapshots)
    tvl_decomposition = build_tvl_decomposition(mint_burn_events, slot0_snapshots)
    concentration_metrics = compute_concentration_metrics(liquidity_snapshots, slot0_snapshots)

    write_parquet(tvl_decomposition, paths.tvl_decomposition)
    write_parquet(concentration_metrics, paths.concentration_metrics)
    plot_liquidity_profiles(liquidity_snapshots, slot0_snapshots, reference_snapshots, paths.fig_21)
    plot_tvl_decomposition(tvl_decomposition, paths.fig_22)
    plot_ilr_series(concentration_metrics, paths.fig_23)
    plot_lhhi_vs_price(concentration_metrics, paths.fig_24)
    return paths


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    run_module_2(data_dir=args.data_dir, figure_dir=args.figure_dir)


if __name__ == "__main__":
    main()
