"""Regression tests for the minimal Uniswap ABI fragments."""

from __future__ import annotations

import unittest

from common.uniswap_abi import POOL_ABI


class UniswapAbiTests(unittest.TestCase):
    """Check that event fragments match the topic structure seen on chain."""

    def test_mint_event_has_three_indexed_inputs(self) -> None:
        mint_event = next(item for item in POOL_ABI if item.get("type") == "event" and item["name"] == "Mint")
        indexed_inputs = [item for item in mint_event["inputs"] if item["indexed"]]
        self.assertEqual(len(indexed_inputs), 3)
        self.assertEqual([item["name"] for item in indexed_inputs], ["owner", "tickLower", "tickUpper"])


if __name__ == "__main__":
    unittest.main()
