"""Smoke tests for the module CLI entry points.

These tests do not run the full pipelines. They simply check that each module can
be invoked from the project root and expose a working `--help` entry point.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    "module1_onchain_data_extraction.data_extraction",
    "module2_liquidity_distribution_analysis.liquidity_analysis",
    "module3_slippage_simulation_and_execution_cost.slippage_analysis",
    "module4_liquidity_provision_analytics.lp_analytics",
    "module5_dynamic_hedging_of_impermanent_loss.hedge_backtest",
)


class ModuleCliSmokeTests(unittest.TestCase):
    """Basic checks that the documented entry points are usable."""

    def test_help_commands_return_success(self) -> None:
        env = os.environ.copy()
        env.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
        env.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

        for module_name in MODULES:
            with self.subTest(module=module_name):
                completed = subprocess.run(
                    [sys.executable, "-m", module_name, "--help"],
                    cwd=PROJECT_ROOT,
                    env=env,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 0, msg=completed.stderr)
                self.assertIn("usage", completed.stdout.lower())


if __name__ == "__main__":
    unittest.main()
