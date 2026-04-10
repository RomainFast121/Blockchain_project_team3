"""Unit tests for the Module 5 hedging utilities."""

from __future__ import annotations

import unittest
from unittest.mock import Mock

import pandas as pd

from common.hyperliquid_client import HyperliquidClient
from module5_dynamic_hedging_of_impermanent_loss.hedge_backtest import (
    HEDGE_RESULTS_COLUMNS,
    HOURLY_MARKET_COLUMNS,
    build_hourly_fee_series,
    prepare_hourly_market_data,
    run_delta_hedge_backtest,
)


class HyperliquidClientTests(unittest.TestCase):
    """Pagination and parsing checks for the Hyperliquid client."""

    def test_funding_history_parsing(self) -> None:
        client = HyperliquidClient(session=Mock())
        client._post = Mock(
            return_value=[
                {"coin": "ETH", "fundingRate": "0.0001", "premium": "0.0003", "time": 1_700_000_000_000},
                {"coin": "ETH", "fundingRate": "-0.0002", "premium": "-0.0001", "time": 1_700_000_360_000},
            ]
        )
        frame = client.fetch_funding_history("ETH", 1_700_000_000_000, 1_700_000_360_000)
        self.assertEqual(len(frame), 2)
        self.assertAlmostEqual(frame.loc[0, "funding_rate"], 0.0001)


class Module5BacktestTests(unittest.TestCase):
    """Synthetic checks for the hedging backtest."""

    def setUp(self) -> None:
        self.positions = pd.DataFrame(
            [
                {
                    "position_id": "P3",
                    "position_label": "Medium",
                    "entry_timestamp": pd.Timestamp("2026-01-01 00:00:00", tz="UTC"),
                    "exit_timestamp": pd.Timestamp("2026-01-01 03:00:00", tz="UTC"),
                    "entry_price_usdc_per_weth": 2_000.0,
                    "price_lower": 1_960.0,
                    "price_upper": 2_040.0,
                    "liquidity_raw": 2_020_000_000_000_000,
                    "entry_usdc": 50_000.0,
                    "entry_weth": 25.0,
                }
            ]
        )

    def test_prepare_hourly_market_data_merges_funding(self) -> None:
        candles = pd.DataFrame(
            [
                {"timestamp": pd.Timestamp("2026-01-01 00:00:00", tz="UTC"), "close": 2_000.0},
                {"timestamp": pd.Timestamp("2026-01-01 01:00:00", tz="UTC"), "close": 2_010.0},
            ]
        )
        funding = pd.DataFrame(
            [{"timestamp": pd.Timestamp("2026-01-01 01:00:00", tz="UTC"), "funding_rate": 0.0001}]
        )
        prepared = prepare_hourly_market_data(candles, funding)
        self.assertEqual(prepared.loc[0, "funding_rate"], 0.0)
        self.assertEqual(prepared.loc[1, "funding_rate"], 0.0001)

    def test_prepare_hourly_market_data_handles_empty_funding(self) -> None:
        candles = pd.DataFrame(
            [{"timestamp": pd.Timestamp("2026-01-01 00:00:00", tz="UTC"), "close": 2_000.0}]
        )
        prepared = prepare_hourly_market_data(candles, pd.DataFrame())
        self.assertListEqual(list(prepared.columns), HOURLY_MARKET_COLUMNS)
        self.assertEqual(prepared.loc[0, "funding_rate"], 0.0)
        self.assertEqual(prepared.loc[0, "cumulative_funding_rate"], 0.0)

    def test_backtest_outputs_expected_columns(self) -> None:
        hourly_market = pd.DataFrame(
            [
                {"timestamp": pd.Timestamp("2026-01-01 00:00:00", tz="UTC"), "price_usdc_per_weth": 2_000.0, "funding_rate": 0.0},
                {"timestamp": pd.Timestamp("2026-01-01 01:00:00", tz="UTC"), "price_usdc_per_weth": 2_050.0, "funding_rate": 0.0001},
                {"timestamp": pd.Timestamp("2026-01-01 02:00:00", tz="UTC"), "price_usdc_per_weth": 2_030.0, "funding_rate": -0.0001},
                {"timestamp": pd.Timestamp("2026-01-01 03:00:00", tz="UTC"), "price_usdc_per_weth": 2_010.0, "funding_rate": 0.0002},
            ]
        )
        hourly_fees = build_hourly_fee_series(
            self.positions,
            pd.DataFrame(
                [
                    {
                        "position_id": "P3",
                        "block_timestamp": pd.Timestamp("2026-01-01 02:00:00", tz="UTC"),
                        "cumulative_fee_income_usd": 120.0,
                    }
                ]
            ),
            hourly_market["timestamp"],
        )
        results = run_delta_hedge_backtest(self.positions, hourly_market, hourly_fees)
        self.assertIn("net_position_pnl_usd", results.columns)
        self.assertEqual(set(results["frequency"]), {"1h", "4h", "24h"})

    def test_empty_backtest_keeps_expected_schema(self) -> None:
        hourly_market = pd.DataFrame(columns=["timestamp", "price_usdc_per_weth", "funding_rate"])
        hourly_fees = pd.DataFrame(columns=["timestamp", "position_id", "lp_fee_income_usd"])
        results = run_delta_hedge_backtest(self.positions, hourly_market, hourly_fees)
        self.assertListEqual(list(results.columns), HEDGE_RESULTS_COLUMNS)
        self.assertTrue(results.empty)

    def test_hourly_fee_series_forward_fills_cumulative_income(self) -> None:
        hourly_market = pd.DataFrame(
            [
                {"timestamp": pd.Timestamp("2026-01-01 00:00:00", tz="UTC")},
                {"timestamp": pd.Timestamp("2026-01-01 01:00:00", tz="UTC")},
                {"timestamp": pd.Timestamp("2026-01-01 02:00:00", tz="UTC")},
            ]
        )
        fees = build_hourly_fee_series(
            self.positions,
            pd.DataFrame(
                [
                    {
                        "position_id": "P3",
                        "block_timestamp": pd.Timestamp("2026-01-01 01:00:00", tz="UTC"),
                        "cumulative_fee_income_usd": 50.0,
                    }
                ]
            ),
            hourly_market["timestamp"],
        )
        self.assertEqual(fees["lp_fee_income_usd"].tolist(), [0.0, 50.0, 50.0])

    def test_hourly_fee_series_handles_timestamp_precision_mismatch(self) -> None:
        hourly_timestamps = pd.Series(
            pd.to_datetime(
                ["2026-01-01 00:00:00", "2026-01-01 01:00:00"],
                utc=True,
            ).astype("datetime64[ms, UTC]")
        )
        fees = build_hourly_fee_series(
            self.positions,
            pd.DataFrame(
                [
                    {
                        "position_id": "P3",
                        "block_timestamp": pd.Timestamp("2026-01-01 00:30:00", tz="UTC"),
                        "cumulative_fee_income_usd": 25.0,
                    }
                ]
            ),
            hourly_timestamps,
        )
        self.assertEqual(fees["lp_fee_income_usd"].tolist(), [0.0, 25.0])


if __name__ == "__main__":
    unittest.main()
