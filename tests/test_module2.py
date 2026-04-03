"""Unit tests for the Module 2 analytics layer."""

from __future__ import annotations

import unittest

import pandas as pd

from module2_liquidity_distribution_analysis.liquidity_analysis import (
    build_tvl_decomposition,
    compute_concentration_metrics,
    expand_liquidity_profile,
)


class Module2AnalyticsTests(unittest.TestCase):
    """Deterministic checks for TVL and concentration metrics."""

    def test_concentration_metrics_have_expected_bounds(self) -> None:
        liquidity_snapshots = pd.DataFrame(
            [
                {
                    "snapshot_block": 1,
                    "snapshot_timestamp": pd.Timestamp("2026-01-01", tz="UTC"),
                    "tick": -10,
                    "liquidityNet": 100,
                    "liquidityGross": 100,
                    "active_liquidity": 100,
                    "price_lower": 1_980.0,
                    "price_upper": 2_000.0,
                },
                {
                    "snapshot_block": 1,
                    "snapshot_timestamp": pd.Timestamp("2026-01-01", tz="UTC"),
                    "tick": 0,
                    "liquidityNet": 100,
                    "liquidityGross": 100,
                    "active_liquidity": 100,
                    "price_lower": 2_000.0,
                    "price_upper": 2_020.0,
                },
            ]
        )
        slot0_snapshots = pd.DataFrame(
            [
                {
                    "snapshot_block": 1,
                    "snapshot_timestamp": pd.Timestamp("2026-01-01", tz="UTC"),
                    "sqrtPriceX96": 0,
                    "price_usdc_per_weth": 2_000.0,
                    "current_tick": 0,
                }
            ]
        )

        metrics = compute_concentration_metrics(liquidity_snapshots, slot0_snapshots)
        self.assertEqual(len(metrics), 1)
        self.assertAlmostEqual(metrics.loc[0, "l_hhi"], 0.5)
        self.assertGreaterEqual(metrics.loc[0, "ilr_5_0pct"], metrics.loc[0, "ilr_0_1pct"])

    def test_tvl_decomposition_sums_components(self) -> None:
        mint_burn_events = pd.DataFrame(
            [
                {
                    "block_number": 10,
                    "log_index": 0,
                    "event_type": "mint",
                    "owner": "0x1",
                    "tick_lower": -100,
                    "tick_upper": 100,
                    "liquidity_raw": 10_000_000_000_000,
                }
            ]
        )
        slot0_snapshots = pd.DataFrame(
            [
                {
                    "snapshot_block": 20,
                    "snapshot_timestamp": pd.Timestamp("2026-01-01", tz="UTC"),
                    "sqrtPriceX96": 1771595571142957102961017161,
                    "price_usdc_per_weth": 2_000.0,
                    "current_tick": 0,
                }
            ]
        )

        decomposition = build_tvl_decomposition(mint_burn_events, slot0_snapshots)
        self.assertEqual(len(decomposition), 1)
        total = decomposition.loc[0, "tvl_in_range"] + decomposition.loc[0, "tvl_above_range"] + decomposition.loc[0, "tvl_below_range"]
        self.assertAlmostEqual(total, decomposition.loc[0, "tvl_total"])

    def test_profile_expansion_fills_missing_intervals(self) -> None:
        initialized = pd.DataFrame(
            [
                {
                    "snapshot_block": 1,
                    "snapshot_timestamp": pd.Timestamp("2026-01-01", tz="UTC"),
                    "tick": -20,
                    "active_liquidity": 100,
                    "price_lower": 0.0,
                    "price_upper": 0.0,
                },
                {
                    "snapshot_block": 1,
                    "snapshot_timestamp": pd.Timestamp("2026-01-01", tz="UTC"),
                    "tick": 10,
                    "active_liquidity": 50,
                    "price_lower": 0.0,
                    "price_upper": 0.0,
                },
            ]
        )
        expanded = expand_liquidity_profile(initialized)
        self.assertEqual(expanded["tick"].tolist(), [-20, -10, 0, 10])
        self.assertEqual(expanded["active_liquidity"].tolist(), [100.0, 100.0, 100.0, 50.0])


if __name__ == "__main__":
    unittest.main()
