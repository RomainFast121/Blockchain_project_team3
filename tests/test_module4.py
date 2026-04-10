"""Unit tests for Module 4 LP analytics."""

from __future__ import annotations

import unittest

import pandas as pd

from module4_liquidity_provision_analytics.lp_analytics import (
    FEE_ACCRUAL_COLUMNS,
    POSITION_TIMESERIES_COLUMNS,
    build_position_timeseries,
    build_representative_positions,
    compute_fee_accruals,
)


class Module4LPAnalyticsTests(unittest.TestCase):
    """Synthetic checks for the LP analytics pipeline."""

    def setUp(self) -> None:
        self.slot0_snapshots = pd.DataFrame(
            [
                {
                    "snapshot_block": 1,
                    "snapshot_timestamp": pd.Timestamp("2026-01-01", tz="UTC"),
                    "sqrtPriceX96": 1771595571142957102961017161,
                    "price_usdc_per_weth": 2_000.0,
                    "current_tick": 0,
                },
                {
                    "snapshot_block": 2,
                    "snapshot_timestamp": pd.Timestamp("2026-01-02", tz="UTC"),
                    "sqrtPriceX96": 1726371784890069398778435477,
                    "price_usdc_per_weth": 2_100.0,
                    "current_tick": 0,
                },
            ]
        )

    def test_positions_are_budgeted_close_to_target(self) -> None:
        positions = build_representative_positions(self.slot0_snapshots)
        self.assertEqual(len(positions), 5)
        for row in positions.itertuples(index=False):
            self.assertAlmostEqual(float(row.entry_usdc + row.entry_weth * row.entry_price_usdc_per_weth), 100_000, delta=5)

    def test_fee_accruals_only_include_in_range_swaps(self) -> None:
        positions = build_representative_positions(self.slot0_snapshots)
        p4 = positions[positions["position_id"] == "P4"].reset_index(drop=True)
        in_range_tick = int((p4.loc[0, "tick_lower"] + p4.loc[0, "tick_upper"]) // 2)
        out_of_range_tick = int(p4.loc[0, "tick_upper"] + 10)
        swap_events = pd.DataFrame(
            [
                {
                    "block_number": 10,
                    "block_timestamp": pd.Timestamp("2026-01-01 12:00:00", tz="UTC"),
                    "log_index": 0,
                    "trade_direction": "buy_weth",
                    "tick": in_range_tick,
                    "price_usdc_per_weth": 2_000.0,
                    "amount0_usdc": 50_000.0,
                    "amount1_weth": -25.0,
                    "active_liquidity": p4.loc[0, "liquidity_raw"] * 10,
                },
                {
                    "block_number": 11,
                    "block_timestamp": pd.Timestamp("2026-01-01 13:00:00", tz="UTC"),
                    "log_index": 0,
                    "trade_direction": "buy_weth",
                    "tick": out_of_range_tick,
                    "price_usdc_per_weth": 5_000.0,
                    "amount0_usdc": 50_000.0,
                    "amount1_weth": -10.0,
                    "active_liquidity": p4.loc[0, "liquidity_raw"] * 10,
                },
            ]
        )
        fee_accruals = compute_fee_accruals(p4, swap_events)
        self.assertEqual(len(fee_accruals), 1)
        self.assertGreater(fee_accruals.loc[0, "fee_value_usd"], 0.0)

    def test_il_is_zero_at_entry_snapshot(self) -> None:
        positions = build_representative_positions(self.slot0_snapshots)
        fee_accruals = pd.DataFrame(columns=["position_id", "block_timestamp", "cumulative_fee_income_usd"])
        timeseries = build_position_timeseries(positions, self.slot0_snapshots.iloc[[0]], fee_accruals)
        self.assertTrue((timeseries["impermanent_loss_usd"].abs() < 5).all())

    def test_position_timeseries_contains_report_columns(self) -> None:
        positions = build_representative_positions(self.slot0_snapshots)
        fee_accruals = pd.DataFrame(columns=["position_id", "block_timestamp", "cumulative_fee_income_usd"])
        timeseries = build_position_timeseries(positions, self.slot0_snapshots, fee_accruals)
        expected_columns = {
            "position_id",
            "position_label",
            "snapshot_block",
            "snapshot_timestamp",
            "price_usdc_per_weth",
            "lp_principal_usd",
            "hodl_value_usd",
            "impermanent_loss_usd",
            "cumulative_fee_income_usd",
            "net_pnl_usd",
        }
        self.assertTrue(expected_columns.issubset(set(timeseries.columns)))

    def test_empty_fee_accruals_keep_expected_schema(self) -> None:
        positions = build_representative_positions(self.slot0_snapshots)
        empty_swaps = pd.DataFrame(
            columns=[
                "block_number",
                "block_timestamp",
                "log_index",
                "trade_direction",
                "tick",
                "price_usdc_per_weth",
                "amount0_usdc",
                "amount1_weth",
                "active_liquidity",
            ]
        )
        fee_accruals = compute_fee_accruals(positions, empty_swaps)
        self.assertListEqual(list(fee_accruals.columns), FEE_ACCRUAL_COLUMNS)
        self.assertTrue(fee_accruals.empty)

        timeseries = build_position_timeseries(positions, self.slot0_snapshots, fee_accruals)
        self.assertListEqual(list(timeseries.columns), POSITION_TIMESERIES_COLUMNS)
        self.assertFalse(timeseries.empty)


if __name__ == "__main__":
    unittest.main()
