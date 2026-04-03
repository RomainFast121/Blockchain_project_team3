"""Deterministic tests for the Module 1 math and liquidity reconstruction."""

from __future__ import annotations

import unittest

import pandas as pd

from common.uniswap_math import (
    price_usdc_per_weth_to_sqrt_price_x96,
    sqrt_price_x96_to_price_usdc_per_weth,
    tick_to_price_usdc_per_weth,
)
from module1_onchain_data_extraction.data_extraction import build_liquidity_snapshots, resolve_study_window_blocks


class Module1MathTests(unittest.TestCase):
    """Math sanity checks used before moving on to later modules."""

    def test_price_round_trip_is_stable(self) -> None:
        input_price = 2_550
        sqrt_price_x96 = price_usdc_per_weth_to_sqrt_price_x96(input_price)
        output_price = float(sqrt_price_x96_to_price_usdc_per_weth(sqrt_price_x96))
        self.assertAlmostEqual(output_price, input_price, places=6)

    def test_tick_prices_are_monotonic(self) -> None:
        self.assertGreater(float(tick_to_price_usdc_per_weth(-200_000)), float(tick_to_price_usdc_per_weth(-199_990)))


class Module1LiquidityReplayTests(unittest.TestCase):
    """Replay tests for the tick-level liquidity map builder."""

    def test_reconstruction_matches_expected_tick_map(self) -> None:
        mint_burn_events = pd.DataFrame(
            [
                {
                    "block_number": 10,
                    "log_index": 0,
                    "event_type": "mint",
                    "owner": "0x1",
                    "tick_lower": -20,
                    "tick_upper": 20,
                    "liquidity_raw": 100,
                    "amount0_raw": 0,
                    "amount0_usdc": 0.0,
                    "amount1_raw": 0,
                    "amount1_weth": 0.0,
                },
                {
                    "block_number": 20,
                    "log_index": 0,
                    "event_type": "mint",
                    "owner": "0x2",
                    "tick_lower": 0,
                    "tick_upper": 30,
                    "liquidity_raw": 50,
                    "amount0_raw": 0,
                    "amount0_usdc": 0.0,
                    "amount1_raw": 0,
                    "amount1_weth": 0.0,
                },
                {
                    "block_number": 30,
                    "log_index": 0,
                    "event_type": "burn",
                    "owner": "0x1",
                    "tick_lower": -20,
                    "tick_upper": 20,
                    "liquidity_raw": 25,
                    "amount0_raw": 0,
                    "amount0_usdc": 0.0,
                    "amount1_raw": 0,
                    "amount1_weth": 0.0,
                },
            ]
        )
        snapshot_blocks = pd.DataFrame(
            [
                {"snapshot_block": 25, "snapshot_timestamp": pd.Timestamp("2026-01-01", tz="UTC")},
                {"snapshot_block": 35, "snapshot_timestamp": pd.Timestamp("2026-01-02", tz="UTC")},
            ]
        )

        snapshots = build_liquidity_snapshots(mint_burn_events, snapshot_blocks)
        final_snapshot = snapshots[snapshots["snapshot_block"] == 35].sort_values("tick").reset_index(drop=True)

        self.assertEqual(final_snapshot["tick"].tolist(), [-20, 0, 20, 30])
        self.assertEqual(final_snapshot["liquidityNet"].tolist(), [75, 50, -75, -50])
        self.assertEqual(final_snapshot["liquidityGross"].tolist(), [75, 50, 75, 50])
        self.assertEqual(final_snapshot["active_liquidity"].tolist(), [75, 125, 50, 0])

    def test_study_window_end_block_stays_before_next_midnight(self) -> None:
        class DummyClient:
            def find_block_at_or_after(self, timestamp):  # noqa: ANN001
                if str(timestamp).startswith("2026-01-01"):
                    return 100
                return 201

        start_block, end_block = resolve_study_window_blocks(
            client=DummyClient(),
            study_start=pd.Timestamp("2026-01-01").date(),
            study_end=pd.Timestamp("2026-01-01").date(),
        )
        self.assertEqual(start_block, 100)
        self.assertEqual(end_block, 200)


if __name__ == "__main__":
    unittest.main()
