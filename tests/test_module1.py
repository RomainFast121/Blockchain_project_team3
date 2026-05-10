"""Deterministic tests for the Module 1 math and liquidity reconstruction."""

from __future__ import annotations

import unittest
from decimal import Decimal

import pandas as pd

from common.uniswap_math import (
    synthetic_lp_amounts,
    synthetic_lp_delta,
    synthetic_lp_gamma,
    price_usdc_per_weth_to_sqrt_price_x96,
    sqrt_price_x96_to_price_usdc_per_weth,
    tick_to_price_usdc_per_weth,
)
from module1_onchain_data_extraction.data_extraction import (
    COLLECT_EVENT_COLUMNS,
    LIQUIDITY_EVENT_COLUMNS,
    SWAP_EVENT_COLUMNS,
    _decode_collect_events,
    _decode_liquidity_events,
    _decode_swap_events,
    _validation_summary,
    apply_smoke_test_window,
    build_liquidity_snapshots,
    resolve_study_window_blocks,
)


class Module1MathTests(unittest.TestCase):
    """Math sanity checks used before moving on to later modules."""

    def test_price_round_trip_is_stable(self) -> None:
        input_price = 2_550
        sqrt_price_x96 = price_usdc_per_weth_to_sqrt_price_x96(input_price)
        output_price = float(sqrt_price_x96_to_price_usdc_per_weth(sqrt_price_x96))
        self.assertAlmostEqual(output_price, input_price, places=6)

    def test_tick_prices_are_monotonic(self) -> None:
        self.assertGreater(float(tick_to_price_usdc_per_weth(-200_000)), float(tick_to_price_usdc_per_weth(-199_990)))

    def test_synthetic_lp_amounts_below_range_are_all_weth(self) -> None:
        usdc_amount, weth_amount = synthetic_lp_amounts(
            liquidity_raw=10**12,
            price_usdc_per_weth=1_900,
            price_lower=2_000,
            price_upper=2_400,
        )
        self.assertEqual(usdc_amount, Decimal(0))
        self.assertGreater(weth_amount, Decimal(0))

    def test_synthetic_lp_amounts_above_range_are_all_usdc(self) -> None:
        usdc_amount, weth_amount = synthetic_lp_amounts(
            liquidity_raw=10**12,
            price_usdc_per_weth=2_500,
            price_lower=2_000,
            price_upper=2_400,
        )
        self.assertGreater(usdc_amount, Decimal(0))
        self.assertEqual(weth_amount, Decimal(0))

    def test_synthetic_lp_delta_matches_constant_weth_below_range(self) -> None:
        usdc_amount, weth_amount = synthetic_lp_amounts(
            liquidity_raw=10**12,
            price_usdc_per_weth=1_900,
            price_lower=2_000,
            price_upper=2_400,
        )
        delta = synthetic_lp_delta(
            liquidity_raw=10**12,
            price_usdc_per_weth=1_900,
            price_lower=2_000,
            price_upper=2_400,
        )
        self.assertEqual(usdc_amount, Decimal(0))
        self.assertGreater(delta, Decimal(0))
        self.assertEqual(delta, weth_amount)

    def test_synthetic_lp_delta_above_range_is_zero(self) -> None:
        delta = synthetic_lp_delta(
            liquidity_raw=10**12,
            price_usdc_per_weth=2_500,
            price_lower=2_000,
            price_upper=2_400,
        )
        self.assertEqual(delta, Decimal(0))

    def test_synthetic_lp_delta_inside_range_is_positive_but_below_constant_weth(self) -> None:
        below_range_delta = synthetic_lp_delta(
            liquidity_raw=10**12,
            price_usdc_per_weth=1_900,
            price_lower=2_000,
            price_upper=2_400,
        )
        in_range_delta = synthetic_lp_delta(
            liquidity_raw=10**12,
            price_usdc_per_weth=2_200,
            price_lower=2_000,
            price_upper=2_400,
        )
        self.assertGreater(in_range_delta, Decimal(0))
        self.assertLess(in_range_delta, below_range_delta)

    def test_synthetic_lp_gamma_is_negative_only_inside_range(self) -> None:
        below = synthetic_lp_gamma(10**12, 1_900, 2_000, 2_400)
        inside = synthetic_lp_gamma(10**12, 2_200, 2_000, 2_400)
        above = synthetic_lp_gamma(10**12, 2_500, 2_000, 2_400)
        self.assertEqual(below, Decimal(0))
        self.assertLess(inside, Decimal(0))
        self.assertEqual(above, Decimal(0))


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

    def test_empty_liquidity_snapshot_keeps_expected_schema(self) -> None:
        snapshots = build_liquidity_snapshots(
            mint_burn_events=pd.DataFrame(),
            snapshot_blocks=pd.DataFrame(
                [{"snapshot_block": 1, "snapshot_timestamp": pd.Timestamp("2026-01-01", tz="UTC")}]
            ),
        )
        self.assertEqual(
            snapshots.columns.tolist(),
            [
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

    def test_liquidity_snapshot_prices_are_ordered(self) -> None:
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
                }
            ]
        )
        snapshot_blocks = pd.DataFrame(
            [{"snapshot_block": 10, "snapshot_timestamp": pd.Timestamp("2026-01-01", tz="UTC")}]
        )
        snapshots = build_liquidity_snapshots(mint_burn_events, snapshot_blocks)
        self.assertTrue((snapshots["price_lower"] <= snapshots["price_upper"]).all())

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

    def test_smoke_test_window_truncates_end_date(self) -> None:
        start, end = apply_smoke_test_window(
            study_start=pd.Timestamp("2026-01-01").date(),
            study_end=pd.Timestamp("2026-01-31").date(),
            smoke_test_days=3,
        )
        self.assertEqual(start, pd.Timestamp("2026-01-01").date())
        self.assertEqual(end, pd.Timestamp("2026-01-03").date())


class _EmptyLogClient:
    """Tiny fake client whose log iterators never yield events."""

    def iter_event_logs(self, event_name, start_block, end_block, chunk_size):  # noqa: ANN001
        return iter(())


class Module1DecoderTests(unittest.TestCase):
    """Decoder edge cases for empty smoke-test windows."""

    def test_decode_swap_events_empty_keeps_expected_schema(self) -> None:
        frame = _decode_swap_events(
            client=_EmptyLogClient(),
            start_block=1,
            end_block=1,
            chunk_size=10,
            progress_seconds=0,
            timestamp_batch_size=1,
        )
        self.assertListEqual(frame.columns.tolist(), SWAP_EVENT_COLUMNS)
        self.assertTrue(frame.empty)

    def test_decode_liquidity_events_empty_keeps_expected_schema(self) -> None:
        frame = _decode_liquidity_events(
            client=_EmptyLogClient(),
            event_name="Mint",
            start_block=1,
            end_block=1,
            chunk_size=10,
            progress_seconds=0,
            timestamp_batch_size=1,
        )
        self.assertListEqual(frame.columns.tolist(), LIQUIDITY_EVENT_COLUMNS)
        self.assertTrue(frame.empty)

    def test_decode_collect_events_empty_keeps_expected_schema(self) -> None:
        frame = _decode_collect_events(
            client=_EmptyLogClient(),
            start_block=1,
            end_block=1,
            chunk_size=10,
            progress_seconds=0,
            timestamp_batch_size=1,
        )
        self.assertListEqual(frame.columns.tolist(), COLLECT_EVENT_COLUMNS)
        self.assertTrue(frame.empty)

    def test_validation_summary_records_failed_status_without_crashing(self) -> None:
        swap_events = pd.DataFrame(
            [
                {
                    "block_number": 100,
                    "amount0_usdc": 12.5,
                    "amount1_weth": -0.005,
                    "notional_usd": 12.5,
                }
            ]
        )
        summary = _validation_summary(
            slot0_consistency=None,
            liquidity_validation=None,
            swap_events=swap_events,
            study_start=pd.Timestamp("2026-01-01").date(),
            study_end=pd.Timestamp("2026-01-02").date(),
            validation_status="failed",
            error_type="HTTPError",
            error_message="rate limited",
        )
        self.assertEqual(summary["validation_status"], "failed")
        self.assertEqual(summary["error_type"], "HTTPError")
        self.assertEqual(summary["volume_cross_check"]["swap_count"], 1)


if __name__ == "__main__":
    unittest.main()
