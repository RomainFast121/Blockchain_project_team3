"""Unit tests for the swap simulator."""

from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import pandas as pd

from common.uniswap_math import price_usdc_per_weth_to_sqrt_price_x96
from module3_slippage_simulation_and_execution_cost.slippage_analysis import (
    OBSERVED_EFFECTIVE_SPREAD_COLUMNS,
    SIMULATED_TRADES_COLUMNS,
    _observed_execution_price,
    build_effective_spread_dataset,
    run_simulation_grid,
)
from module3_slippage_simulation_and_execution_cost.swap_simulator import (
    SnapshotPoolState,
    simulate_exact_input_swap,
)


class Module3SwapSimulatorTests(unittest.TestCase):
    """Synthetic checks for the Uniswap V3 simulator."""

    def setUp(self) -> None:
        self.state = SnapshotPoolState(
            snapshot_block=1,
            snapshot_timestamp=pd.Timestamp("2026-01-01", tz="UTC"),
            sqrt_price_x96=price_usdc_per_weth_to_sqrt_price_x96(2_000),
            current_tick=0,
            current_price=2_000.0,
            active_liquidity=4_000_000_000_000_000,
            initialized_ticks=(-120, 0, 120),
            liquidity_net_by_tick={-120: 1_000_000_000_000_000, 0: 3_000_000_000_000_000, 120: -4_000_000_000_000_000},
        )

    def test_buy_trade_has_positive_impact(self) -> None:
        result = simulate_exact_input_swap(self.state, direction="buy_weth", notional_usd=100_000)
        self.assertGreater(result.average_price, result.pool_mid_price)
        self.assertGreaterEqual(result.price_impact_bps, 0.0)
        self.assertGreaterEqual(result.slippage_bps, 0.0)
        self.assertAlmostEqual(result.input_filled_fraction, 1.0, places=9)

    def test_sell_trade_has_positive_impact_metric(self) -> None:
        result = simulate_exact_input_swap(self.state, direction="sell_weth", notional_usd=100_000)
        self.assertLess(result.average_price, result.pool_mid_price)
        self.assertGreaterEqual(result.price_impact_bps, 0.0)

    def test_larger_trade_moves_price_more(self) -> None:
        small_trade = simulate_exact_input_swap(self.state, direction="buy_weth", notional_usd=10_000)
        large_trade = simulate_exact_input_swap(self.state, direction="buy_weth", notional_usd=500_000)
        self.assertGreater(large_trade.price_impact_bps, small_trade.price_impact_bps)

    def test_invalid_direction_raises_clear_error(self) -> None:
        with self.assertRaises(ValueError):
            simulate_exact_input_swap(self.state, direction="hold", notional_usd=10_000)

    def test_zero_liquidity_returns_nan_metrics_instead_of_crashing(self) -> None:
        empty_state = SnapshotPoolState(
            snapshot_block=2,
            snapshot_timestamp=pd.Timestamp("2026-01-02", tz="UTC"),
            sqrt_price_x96=price_usdc_per_weth_to_sqrt_price_x96(2_000),
            current_tick=0,
            current_price=2_000.0,
            active_liquidity=0,
            initialized_ticks=(),
            liquidity_net_by_tick={},
        )

        result = simulate_exact_input_swap(empty_state, direction="buy_weth", notional_usd=10_000)
        self.assertTrue(math.isnan(result.average_price))
        self.assertTrue(math.isnan(result.price_impact_bps))
        self.assertTrue(math.isnan(result.slippage_bps))
        self.assertEqual(result.input_filled_fraction, 0.0)

    def test_partial_fill_reports_fraction_below_one(self) -> None:
        sparse_state = SnapshotPoolState(
            snapshot_block=3,
            snapshot_timestamp=pd.Timestamp("2026-01-03", tz="UTC"),
            sqrt_price_x96=price_usdc_per_weth_to_sqrt_price_x96(2_000),
            current_tick=0,
            current_price=2_000.0,
            active_liquidity=1,
            initialized_ticks=(),
            liquidity_net_by_tick={},
        )
        result = simulate_exact_input_swap(sparse_state, direction="sell_weth", notional_usd=1_000_000)
        self.assertLess(result.input_filled_fraction, 1.0)

    def test_empty_simulation_grid_keeps_expected_schema(self) -> None:
        result = run_simulation_grid({})
        self.assertListEqual(list(result.columns), SIMULATED_TRADES_COLUMNS)
        self.assertTrue(result.empty)

    def test_simulation_grid_drops_partial_fills(self) -> None:
        sparse_state = SnapshotPoolState(
            snapshot_block=4,
            snapshot_timestamp=pd.Timestamp("2026-01-04", tz="UTC"),
            sqrt_price_x96=price_usdc_per_weth_to_sqrt_price_x96(2_000),
            current_tick=0,
            current_price=2_000.0,
            active_liquidity=1,
            initialized_ticks=(),
            liquidity_net_by_tick={},
        )
        result = run_simulation_grid({4: sparse_state})
        self.assertTrue((result["input_filled_fraction"] >= 0.999999).all())

    def test_observed_execution_price_returns_nan_for_zero_weth_leg(self) -> None:
        price = _observed_execution_price(amount0_usdc=1_000.0, amount1_weth=0.0)
        self.assertTrue(math.isnan(price))

    def test_effective_spread_dataset_skips_degenerate_swaps(self) -> None:
        swap_events = pd.DataFrame(
            [
                {
                    "block_number": 10,
                    "block_timestamp": pd.Timestamp("2026-01-01 00:00:00", tz="UTC"),
                    "trade_direction": "buy_weth",
                    "notional_usd": 1_000.0,
                    "amount0_usdc": 1_000.0,
                    "amount1_weth": 0.0,
                }
            ]
        )

        class DummyClient:
            def __init__(self, rpc_url: str, pool_address: str) -> None:  # noqa: D401
                self.rpc_url = rpc_url
                self.pool_address = pool_address

            def call_slot0(self, block_number: int) -> dict[str, int]:
                return {"sqrtPriceX96": price_usdc_per_weth_to_sqrt_price_x96(2_000)}

        with patch(
            "module3_slippage_simulation_and_execution_cost.slippage_analysis.EthereumArchiveClient",
            DummyClient,
        ):
            result = build_effective_spread_dataset(
                rpc_url="https://example.invalid",
                pool_address="0x0000000000000000000000000000000000000000",
                swap_events=swap_events,
                progress_seconds=0,
            )

        self.assertListEqual(list(result.columns), OBSERVED_EFFECTIVE_SPREAD_COLUMNS)
        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
