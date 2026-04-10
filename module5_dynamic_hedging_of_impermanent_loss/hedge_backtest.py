"""Module 5: hedge LP delta with Hyperliquid perpetuals.

This module closes the loop of the project:

- Module 4 gave us synthetic LP positions and their fee income,
- Module 5 adds hourly perp prices and funding rates,
- then tests whether delta hedging reduces impermanent loss.

The backtest is intentionally simple and report-friendly:
`download hourly market data -> align fee income -> run hedging variants ->
save results -> build summary figures`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.constants import FIGURES_DIR, PROCESSED_DATA_DIR
from common.hyperliquid_client import HyperliquidClient
from common.io_utils import ensure_directory, read_parquet, write_parquet
from common.plotting import save_figure, set_project_style
from common.uniswap_math import synthetic_lp_delta, synthetic_lp_value


TRADING_FEE_RATE = 0.00045
REBALANCE_FREQUENCIES = {"1h": 1, "4h": 4, "24h": 24}


@dataclass(frozen=True)
class Module5Paths:
    """All files produced by Module 5."""

    data_dir: Path
    figure_dir: Path

    @property
    def perp_prices(self) -> Path:
        return self.data_dir / "perp_prices.parquet"

    @property
    def funding_rates(self) -> Path:
        return self.data_dir / "funding_rates.parquet"

    @property
    def hedge_results(self) -> Path:
        return self.data_dir / "hedge_results.parquet"

    @property
    def fig_payoffs(self) -> Path:
        return self.figure_dir / "fig_5_0_lp_payoffs.png"

    @property
    def fig_funding(self) -> Path:
        return self.figure_dir / "fig_5_0_funding_environment.png"

    @property
    def fig_results(self) -> Path:
        return self.figure_dir / "fig_5_1_hedging_results.png"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--coin", default="ETH", help="Hyperliquid perpetual symbol.")
    return parser.parse_args()


def load_module5_inputs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Module 4 outputs needed for the hedging exercise."""

    positions = read_parquet(data_dir / "synthetic_lp_positions.parquet")
    fee_accruals = read_parquet(data_dir / "lp_fee_accruals.parquet")
    positions["entry_timestamp"] = pd.to_datetime(positions["entry_timestamp"], utc=True)
    positions["exit_timestamp"] = pd.to_datetime(positions["exit_timestamp"], utc=True)
    if not fee_accruals.empty:
        fee_accruals["block_timestamp"] = pd.to_datetime(fee_accruals["block_timestamp"], utc=True)
    return positions, fee_accruals


def fetch_hyperliquid_data(positions: pd.DataFrame, coin: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download hourly candles and funding history for the study window."""

    start_time_ms = int(positions["entry_timestamp"].min().timestamp() * 1000)
    end_time_ms = int(positions["exit_timestamp"].max().timestamp() * 1000)
    client = HyperliquidClient()
    perp_prices = client.fetch_hourly_candles(coin=coin, start_time_ms=start_time_ms, end_time_ms=end_time_ms)
    funding_rates = client.fetch_funding_history(coin=coin, start_time_ms=start_time_ms, end_time_ms=end_time_ms)
    return perp_prices, funding_rates


def prepare_hourly_market_data(perp_prices: pd.DataFrame, funding_rates: pd.DataFrame) -> pd.DataFrame:
    """Merge hourly closes and hourly funding into one clean market table."""

    prices = perp_prices.copy().sort_values("timestamp").reset_index(drop=True)
    funding = funding_rates.copy().sort_values("timestamp").reset_index(drop=True)

    if funding.empty:
        prices["funding_rate"] = 0.0
        prices["cumulative_funding_rate"] = 0.0
        return prices

    merged = prices.merge(funding[["timestamp", "funding_rate"]], on="timestamp", how="left")
    merged["funding_rate"] = merged["funding_rate"].fillna(0.0)
    merged["cumulative_funding_rate"] = merged["funding_rate"].cumsum()
    return merged


def build_hourly_fee_series(positions: pd.DataFrame, fee_accruals: pd.DataFrame, hourly_timestamps: pd.Series) -> pd.DataFrame:
    """Forward-fill cumulative LP fee income onto the hourly hedging grid."""

    base_hourly_index = pd.DataFrame({"timestamp": hourly_timestamps.sort_values().drop_duplicates()})
    rows: list[dict[str, object]] = []

    for position in positions.itertuples(index=False):
        position_flows = fee_accruals[fee_accruals["position_id"] == position.position_id].copy()
        if position_flows.empty:
            zero_fee_frame = base_hourly_index.copy()
            zero_fee_frame["position_id"] = position.position_id
            zero_fee_frame["lp_fee_income_usd"] = 0.0
            rows.extend(zero_fee_frame.to_dict("records"))
            continue

        position_flows = position_flows.sort_values("block_timestamp")
        position_flows = position_flows[["block_timestamp", "cumulative_fee_income_usd"]].rename(columns={"block_timestamp": "timestamp"})
        merged = pd.merge_asof(
            base_hourly_index.sort_values("timestamp"),
            position_flows.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
        merged["position_id"] = position.position_id
        merged["lp_fee_income_usd"] = merged["cumulative_fee_income_usd"].fillna(0.0)
        rows.extend(merged[["timestamp", "position_id", "lp_fee_income_usd"]].to_dict("records"))

    return pd.DataFrame(rows)


def run_delta_hedge_backtest(
    positions: pd.DataFrame,
    hourly_market: pd.DataFrame,
    hourly_fees: pd.DataFrame,
) -> pd.DataFrame:
    """Run the 15 hedging variants: 5 positions times 3 rebalance frequencies."""

    hourly_market = hourly_market.sort_values("timestamp").reset_index(drop=True)
    results: list[dict[str, object]] = []

    for position in positions.itertuples(index=False):
        position_market = hourly_market[
            (hourly_market["timestamp"] >= position.entry_timestamp)
            & (hourly_market["timestamp"] <= position.exit_timestamp)
        ].copy()
        position_market = position_market.reset_index(drop=True)
        if position_market.empty:
            continue

        position_fee_series = hourly_fees[hourly_fees["position_id"] == position.position_id].copy()
        position_market = position_market.merge(position_fee_series, on="timestamp", how="left")
        position_market["lp_fee_income_usd"] = position_market["lp_fee_income_usd"].fillna(0.0)

        for frequency_label, rebalance_hours in REBALANCE_FREQUENCIES.items():
            first_price = float(position_market.loc[0, "price_usdc_per_weth"])
            hedge_size_eth = float(
                synthetic_lp_delta(
                    liquidity_raw=position.liquidity_raw,
                    price_usdc_per_weth=first_price,
                    price_lower=position.price_lower,
                    price_upper=position.price_upper,
                )
            )

            # We start by putting on the initial hedge, so the first trade fee is
            # paid immediately.
            trading_fees_cumulative = hedge_size_eth * first_price * TRADING_FEE_RATE
            hedge_pnl_cumulative = 0.0
            funding_pnl_cumulative = 0.0

            for row_index, row in position_market.iterrows():
                price = float(row["price_usdc_per_weth"])
                lp_principal_usd = float(
                    synthetic_lp_value(
                        liquidity_raw=position.liquidity_raw,
                        price_usdc_per_weth=price,
                        price_lower=position.price_lower,
                        price_upper=position.price_upper,
                    )
                )
                hodl_value_usd = float(position.entry_usdc + position.entry_weth * price)
                gross_il_usd = hodl_value_usd - lp_principal_usd

                if row_index > 0:
                    previous_price = float(position_market.loc[row_index - 1, "price_usdc_per_weth"])
                    hedge_pnl_cumulative += hedge_size_eth * (previous_price - price)
                    funding_pnl_cumulative += hedge_size_eth * price * float(row["funding_rate"])

                if row_index > 0 and row_index % rebalance_hours == 0:
                    target_delta = float(
                        synthetic_lp_delta(
                            liquidity_raw=position.liquidity_raw,
                            price_usdc_per_weth=price,
                            price_lower=position.price_lower,
                            price_upper=position.price_upper,
                        )
                    )
                    trading_fees_cumulative += abs(target_delta - hedge_size_eth) * price * TRADING_FEE_RATE
                    hedge_size_eth = target_delta

                net_hedge_pnl_usd = hedge_pnl_cumulative + funding_pnl_cumulative - trading_fees_cumulative

                # Residual IL is the part of gross IL that remains after applying
                # the hedge. Net position P&L then adds back the LP fees from
                # Module 4 without double counting the hedge.
                residual_il_usd = gross_il_usd - net_hedge_pnl_usd

                results.append(
                    {
                        "position_id": position.position_id,
                        "frequency": frequency_label,
                        "timestamp": row["timestamp"],
                        "price_usdc_per_weth": price,
                        "lp_principal_usd": lp_principal_usd,
                        "hodl_value_usd": hodl_value_usd,
                        "gross_il_usd": gross_il_usd,
                        "hedge_size_eth": hedge_size_eth,
                        "hedge_pnl_usd": hedge_pnl_cumulative,
                        "funding_pnl_usd": funding_pnl_cumulative,
                        "trading_fees_usd": trading_fees_cumulative,
                        "net_hedge_pnl_usd": net_hedge_pnl_usd,
                        "residual_il_usd": residual_il_usd,
                        "lp_fee_income_usd": float(row["lp_fee_income_usd"]),
                        "net_position_pnl_usd": float(row["lp_fee_income_usd"]) - residual_il_usd,
                    }
                )

    return pd.DataFrame(results)


def plot_lp_payoffs(positions: pd.DataFrame, path: Path) -> None:
    """Plot terminal LP principal value across a range of ETH prices."""

    set_project_style()
    entry_price = float(positions["entry_price_usdc_per_weth"].iloc[0])
    terminal_prices = np.linspace(0.5 * entry_price, 1.5 * entry_price, 150)

    fig, ax = plt.subplots(figsize=(12, 6))
    for position in positions.itertuples(index=False):
        payoff = [
            float(
                synthetic_lp_value(
                    liquidity_raw=position.liquidity_raw,
                    price_usdc_per_weth=float(price),
                    price_lower=position.price_lower,
                    price_upper=position.price_upper,
                )
            )
            for price in terminal_prices
        ]
        ax.plot(terminal_prices, payoff, linewidth=1.8, label=position.position_id)

    ax.set_title("LP principal payoff at terminal ETH prices")
    ax.set_xlabel("Terminal ETH price (USDC/WETH)")
    ax.set_ylabel("LP principal value (USD)")
    ax.legend(loc="upper left", ncol=3)
    save_figure(fig, path)


def plot_funding_environment(hourly_market: pd.DataFrame, path: Path) -> None:
    """Plot ETH price together with hourly and cumulative funding."""

    set_project_style()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(hourly_market["timestamp"], hourly_market["price_usdc_per_weth"], color="black", linewidth=1.6, label="ETH perp close")
    axes[0].set_ylabel("USDC per WETH")
    axes[0].legend(loc="upper left")

    axes[1].plot(hourly_market["timestamp"], hourly_market["funding_rate"], color="steelblue", linewidth=1.0, label="Hourly funding")
    axes[1].plot(hourly_market["timestamp"], hourly_market["cumulative_funding_rate"], color="darkorange", linewidth=1.5, label="Cumulative funding")
    axes[1].set_ylabel("Funding rate")
    axes[1].set_xlabel("Timestamp")
    axes[1].legend(loc="upper left")
    save_figure(fig, path)


def plot_hedging_results(hedge_results: pd.DataFrame, path: Path) -> None:
    """Render heatmaps of final residual IL and final net P&L."""

    set_project_style()
    final_rows = hedge_results.sort_values("timestamp").groupby(["position_id", "frequency"], as_index=False).tail(1)
    residual_matrix = final_rows.pivot(index="position_id", columns="frequency", values="residual_il_usd")
    pnl_matrix = final_rows.pivot(index="position_id", columns="frequency", values="net_position_pnl_usd")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for axis, matrix, title in (
        (axes[0], residual_matrix, "Final residual IL"),
        (axes[1], pnl_matrix, "Final net position P&L"),
    ):
        image = axis.imshow(matrix.values, cmap="coolwarm", aspect="auto")
        axis.set_xticks(range(len(matrix.columns)), matrix.columns)
        axis.set_yticks(range(len(matrix.index)), matrix.index)
        axis.set_title(title)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(column, row, f"{matrix.iloc[row, column]:.0f}", ha="center", va="center", fontsize=9)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    save_figure(fig, path)


def run_module_5(data_dir: Path, figure_dir: Path, coin: str) -> Module5Paths:
    """Execute the full Module 5 workflow."""

    paths = Module5Paths(data_dir=ensure_directory(data_dir), figure_dir=ensure_directory(figure_dir))
    positions, fee_accruals = load_module5_inputs(data_dir)

    perp_prices, funding_rates = fetch_hyperliquid_data(positions, coin=coin)
    hourly_market = prepare_hourly_market_data(perp_prices, funding_rates)
    hourly_market = hourly_market.rename(columns={"close": "price_usdc_per_weth"})
    hourly_fees = build_hourly_fee_series(positions, fee_accruals, hourly_market["timestamp"])
    hedge_results = run_delta_hedge_backtest(positions, hourly_market, hourly_fees)

    write_parquet(perp_prices, paths.perp_prices)
    write_parquet(funding_rates, paths.funding_rates)
    write_parquet(hedge_results, paths.hedge_results)
    plot_lp_payoffs(positions, paths.fig_payoffs)
    plot_funding_environment(hourly_market, paths.fig_funding)
    plot_hedging_results(hedge_results, paths.fig_results)
    return paths


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    run_module_5(data_dir=args.data_dir, figure_dir=args.figure_dir, coin=args.coin)


if __name__ == "__main__":
    main()
