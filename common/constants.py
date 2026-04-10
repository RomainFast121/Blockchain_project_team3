"""Project-wide constants and reproducible defaults.

Keeping the study parameters in one place makes the rest of the code easier to
read because the module files can focus on the workflow rather than repeating the
same literals.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
FIGURES_DIR = PROJECT_ROOT / "figures"

# The PDF uses this six-month window as the running example, so the project
# adopts it as the default unless the user explicitly overrides it.
DEFAULT_STUDY_START = date(2025, 10, 1)
DEFAULT_STUDY_END = date(2026, 3, 31)

# Pool under study: Uniswap V3 USDC/WETH 0.05%.
POOL_DEPLOYMENT_BLOCK = 12_376_729
POOL_ADDRESS = "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"
POOL_FEE_RATE = 0.0005
POOL_FEE_PIPS = 500
TICK_SPACING = 10

# Token order matters in Uniswap math.
TOKEN0_SYMBOL = "USDC"
TOKEN1_SYMBOL = "WETH"
TOKEN0_DECIMALS = 6
TOKEN1_DECIMALS = 18

UNISWAP_V3_MIN_TICK = -887_272
UNISWAP_V3_MAX_TICK = 887_272

# Alchemy free tier currently allows `eth_getLogs` on at most a 10-block range.
# Users on stronger/archive plans can still override this from the CLI.
DEFAULT_RPC_LOG_CHUNK = 10
DEFAULT_RPC_RETRY_ATTEMPTS = 6
DEFAULT_RPC_RETRY_BASE_DELAY_SECONDS = 0.75
DEFAULT_RPC_REQUEST_SPACING_SECONDS = 0.10
DEFAULT_HYPERLIQUID_PAGE_HOURS = 4_800


@dataclass(frozen=True)
class PositionDefinition:
    """One synthetic LP profile requested in Modules 4 and 5."""

    name: str
    label: str
    width_pct: float | None


REPRESENTATIVE_POSITIONS = (
    PositionDefinition("P1", "Ultra-narrow", 0.1),
    PositionDefinition("P2", "Narrow", 0.5),
    PositionDefinition("P3", "Medium", 2.0),
    PositionDefinition("P4", "Wide", 10.0),
    PositionDefinition("P5", "Full range", None),
)
