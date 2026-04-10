"""Readable Uniswap V3 math helpers used across the project.

This file is intentionally more explanatory than a typical utility module because
these formulas sit at the heart of the whole assignment:

- Module 1 converts on-chain state into human prices.
- Module 2 values liquidity across price ranges.
- Module 3 simulates swaps through the reconstructed liquidity map.
- Module 4 values synthetic LP positions and computes IL.
- Module 5 computes LP delta for the hedging exercise.

The pool under study is the USDC/WETH 0.05% Uniswap V3 pool. On chain, prices are
stored in terms of token1 per token0 and in fixed-point `sqrtPriceX96` format.
For the report, we almost always want the more intuitive `USDC per WETH` view.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_FLOOR, getcontext
from math import log

from common.constants import (
    TICK_SPACING,
    TOKEN0_DECIMALS,
    TOKEN1_DECIMALS,
    UNISWAP_V3_MAX_TICK,
    UNISWAP_V3_MIN_TICK,
)


# We rely on Decimal because repeated square roots and tick conversions can lose
# too much precision when performed in plain float arithmetic.
getcontext().prec = 80

# `sqrtPriceX96` means "square-root price scaled by 2^96".
Q96 = Decimal(2) ** 96

# Token0 is USDC (6 decimals) and token1 is WETH (18 decimals). This scale factor
# converts raw token ratios into the human `USDC per WETH` convention.
DECIMAL_SCALE_RATIO = Decimal(10) ** (TOKEN1_DECIMALS - TOKEN0_DECIMALS)

# One Uniswap tick corresponds to a price move by a factor of 1.0001.
ONE_0001 = Decimal("1.0001")


def decimal_sqrt(value: Decimal) -> Decimal:
    """Return a Decimal square root.

    Keeping this as a helper makes the intent explicit whenever we move between
    price and square-root-price representations.
    """

    return value.sqrt()


# ---------------------------------------------------------------------------
# Price and tick conversion helpers
# ---------------------------------------------------------------------------

def sqrt_price_x96_to_price_usdc_per_weth(sqrt_price_x96: int) -> Decimal:
    """Convert on-chain `sqrtPriceX96` into the human price `USDC per WETH`."""

    sqrt_ratio_token1_per_token0 = Decimal(sqrt_price_x96) / Q96
    raw_price_token1_per_token0 = sqrt_ratio_token1_per_token0 * sqrt_ratio_token1_per_token0
    return DECIMAL_SCALE_RATIO / raw_price_token1_per_token0


def price_usdc_per_weth_to_sqrt_price_x96(price_usdc_per_weth: Decimal | float | int) -> int:
    """Convert a human price into the on-chain `sqrtPriceX96` format."""

    price = Decimal(price_usdc_per_weth)
    raw_price_token1_per_token0 = DECIMAL_SCALE_RATIO / price
    sqrt_ratio_token1_per_token0 = decimal_sqrt(raw_price_token1_per_token0)
    return int((sqrt_ratio_token1_per_token0 * Q96).to_integral_value(rounding=ROUND_FLOOR))


def tick_to_price_usdc_per_weth(tick: int) -> Decimal:
    """Convert a Uniswap tick into the human price `USDC per WETH`."""

    raw_price_token1_per_token0 = ONE_0001 ** Decimal(tick)
    return DECIMAL_SCALE_RATIO / raw_price_token1_per_token0


def tick_to_sqrt_price_x96(tick: int) -> int:
    """Convert a Uniswap tick directly into `sqrtPriceX96`."""

    raw_price_token1_per_token0 = ONE_0001 ** Decimal(tick)
    sqrt_ratio_token1_per_token0 = decimal_sqrt(raw_price_token1_per_token0)
    return int((sqrt_ratio_token1_per_token0 * Q96).to_integral_value(rounding=ROUND_FLOOR))


def price_usdc_per_weth_to_tick(price_usdc_per_weth: Decimal | float | int) -> int:
    """Map a human price to the nearest Uniswap tick."""

    price = float(price_usdc_per_weth)
    return int(round(log(float(DECIMAL_SCALE_RATIO) / price, 1.0001)))


def align_tick_to_spacing(tick: int, tick_spacing: int = TICK_SPACING, mode: str = "nearest") -> int:
    """Align a tick to the pool spacing.

    Uniswap V3 only allows positions on ticks that are multiples of the pool's
    tick spacing. This helper makes that rounding choice explicit.
    """

    if mode == "down":
        return (tick // tick_spacing) * tick_spacing
    if mode == "up":
        return ((tick + tick_spacing - 1) // tick_spacing) * tick_spacing

    lower = align_tick_to_spacing(tick, tick_spacing=tick_spacing, mode="down")
    upper = align_tick_to_spacing(tick, tick_spacing=tick_spacing, mode="up")
    return lower if abs(tick - lower) <= abs(upper - tick) else upper


def clamp_tick_to_uniswap_range(tick: int) -> int:
    """Keep a tick inside Uniswap V3's supported global bounds."""

    return min(max(tick, UNISWAP_V3_MIN_TICK), UNISWAP_V3_MAX_TICK)


def tick_interval_prices(tick_lower: int, tick_upper: int) -> tuple[Decimal, Decimal]:
    """Return the price interval implied by two ticks in sorted order."""

    lower_price = tick_to_price_usdc_per_weth(tick_lower)
    upper_price = tick_to_price_usdc_per_weth(tick_upper)
    return min(lower_price, upper_price), max(lower_price, upper_price)


# ---------------------------------------------------------------------------
# Liquidity-to-token formulas
# ---------------------------------------------------------------------------

def _ordered_sqrt_bounds(a: int, b: int) -> tuple[Decimal, Decimal]:
    """Return two sqrt-price bounds in ascending order."""

    left = Decimal(min(a, b))
    right = Decimal(max(a, b))
    return left, right


def amount0_for_liquidity(
    sqrt_price_a_x96: int,
    sqrt_price_b_x96: int,
    liquidity: Decimal | int | float,
) -> Decimal:
    """Return token0 units implied by `liquidity` over a price interval.

    In this pool token0 is USDC, so this quantity is in raw USDC units.
    """

    sqrt_a, sqrt_b = _ordered_sqrt_bounds(sqrt_price_a_x96, sqrt_price_b_x96)
    liquidity_value = Decimal(liquidity)
    return liquidity_value * Q96 * (sqrt_b - sqrt_a) / (sqrt_b * sqrt_a)


def amount1_for_liquidity(
    sqrt_price_a_x96: int,
    sqrt_price_b_x96: int,
    liquidity: Decimal | int | float,
) -> Decimal:
    """Return token1 units implied by `liquidity` over a price interval.

    In this pool token1 is WETH, so this quantity is in raw WETH units.
    """

    sqrt_a, sqrt_b = _ordered_sqrt_bounds(sqrt_price_a_x96, sqrt_price_b_x96)
    liquidity_value = Decimal(liquidity)
    return liquidity_value * (sqrt_b - sqrt_a) / Q96


def amounts_for_liquidity(
    sqrt_price_x96: int,
    sqrt_price_a_x96: int,
    sqrt_price_b_x96: int,
    liquidity: Decimal | int | float,
) -> tuple[Decimal, Decimal]:
    """Return raw token0/token1 balances for a position at the current price.

    This is the standard Uniswap V3 piecewise rule:

    - if current price is below the range, the position is all token0,
    - if current price is above the range, the position is all token1,
    - if current price is inside the range, the position holds both tokens.
    """

    sqrt_lower, sqrt_upper = _ordered_sqrt_bounds(sqrt_price_a_x96, sqrt_price_b_x96)
    current = Decimal(sqrt_price_x96)

    if current <= sqrt_lower:
        return amount0_for_liquidity(int(sqrt_lower), int(sqrt_upper), liquidity), Decimal(0)
    if current >= sqrt_upper:
        return Decimal(0), amount1_for_liquidity(int(sqrt_lower), int(sqrt_upper), liquidity)

    return (
        amount0_for_liquidity(int(current), int(sqrt_upper), liquidity),
        amount1_for_liquidity(int(sqrt_lower), int(current), liquidity),
    )


def raw_amounts_to_decimal(
    amount0_raw: Decimal | int | float,
    amount1_raw: Decimal | int | float,
) -> tuple[Decimal, Decimal]:
    """Convert raw token balances into human units."""

    amount0 = Decimal(amount0_raw) / (Decimal(10) ** TOKEN0_DECIMALS)
    amount1 = Decimal(amount1_raw) / (Decimal(10) ** TOKEN1_DECIMALS)
    return amount0, amount1


# ---------------------------------------------------------------------------
# Synthetic LP analytics helpers
# ---------------------------------------------------------------------------

def liquidity_to_human_scale(liquidity_raw: Decimal | int | float) -> Decimal:
    """Convert raw liquidity into the human scale used in LP derivations."""

    return Decimal(liquidity_raw) / DECIMAL_SCALE_RATIO


def synthetic_lp_amounts(
    liquidity_raw: Decimal | int | float,
    price_usdc_per_weth: Decimal | int | float,
    price_lower: Decimal | int | float,
    price_upper: Decimal | int | float,
) -> tuple[Decimal, Decimal]:
    """Return the LP's token balances in human units.

    This helper is deliberately written in `USDC per WETH` terms because that is
    how the report interprets the positions.
    """

    liquidity_human = liquidity_to_human_scale(liquidity_raw)
    price = Decimal(price_usdc_per_weth)
    lower = Decimal(price_lower)
    upper = Decimal(price_upper)
    sqrt_price = decimal_sqrt(price)
    sqrt_lower = decimal_sqrt(lower)
    sqrt_upper = decimal_sqrt(upper)

    if price <= lower:
        return liquidity_human * (sqrt_upper - sqrt_lower), Decimal(0)
    if price >= upper:
        weth_amount = liquidity_human * ((Decimal(1) / sqrt_lower) - (Decimal(1) / sqrt_upper))
        return Decimal(0), weth_amount

    usdc_amount = liquidity_human * (sqrt_price - sqrt_lower)
    weth_amount = liquidity_human * ((Decimal(1) / sqrt_price) - (Decimal(1) / sqrt_upper))
    return usdc_amount, weth_amount


def synthetic_lp_value(
    liquidity_raw: Decimal | int | float,
    price_usdc_per_weth: Decimal | int | float,
    price_lower: Decimal | int | float,
    price_upper: Decimal | int | float,
) -> Decimal:
    """Return the USD value of a synthetic LP position."""

    usdc_amount, weth_amount = synthetic_lp_amounts(liquidity_raw, price_usdc_per_weth, price_lower, price_upper)
    price = Decimal(price_usdc_per_weth)
    return usdc_amount + weth_amount * price


def synthetic_lp_delta(
    liquidity_raw: Decimal | int | float,
    price_usdc_per_weth: Decimal | int | float,
    price_lower: Decimal | int | float,
    price_upper: Decimal | int | float,
) -> Decimal:
    """Return LP delta in ETH units with respect to the ETH/USD price."""

    liquidity_human = liquidity_to_human_scale(liquidity_raw)
    price = Decimal(price_usdc_per_weth)
    lower = Decimal(price_lower)
    upper = Decimal(price_upper)
    sqrt_lower = decimal_sqrt(lower)
    sqrt_upper = decimal_sqrt(upper)

    if price <= lower:
        return Decimal(0)
    if price >= upper:
        return (Decimal(1) / sqrt_lower - Decimal(1) / sqrt_upper) * liquidity_human

    return (Decimal(1) / decimal_sqrt(price) - Decimal(1) / sqrt_upper) * liquidity_human


def synthetic_lp_gamma(
    liquidity_raw: Decimal | int | float,
    price_usdc_per_weth: Decimal | int | float,
    price_lower: Decimal | int | float,
    price_upper: Decimal | int | float,
) -> Decimal:
    """Return LP gamma with respect to the ETH/USD price."""

    price = Decimal(price_usdc_per_weth)
    lower = Decimal(price_lower)
    upper = Decimal(price_upper)
    if price <= lower or price >= upper:
        return Decimal(0)

    liquidity_human = liquidity_to_human_scale(liquidity_raw)
    return -liquidity_human / (Decimal(2) * (decimal_sqrt(price) ** 3))


def solve_liquidity_for_budget(
    budget_usd: Decimal | int | float,
    entry_price: Decimal | int | float,
    price_lower: Decimal | int | float,
    price_upper: Decimal | int | float,
) -> tuple[int, Decimal, Decimal]:
    """Solve the liquidity amount that matches a target USD budget.

    The simplest way to size a synthetic LP is:

    1. value one unit of raw liquidity at the entry price,
    2. scale that unit position up to the requested budget,
    3. return both the liquidity and the implied entry token balances.
    """

    unit_liquidity_raw = int(DECIMAL_SCALE_RATIO)
    unit_usdc, unit_weth = synthetic_lp_amounts(
        liquidity_raw=unit_liquidity_raw,
        price_usdc_per_weth=entry_price,
        price_lower=price_lower,
        price_upper=price_upper,
    )
    unit_value = unit_usdc + unit_weth * Decimal(entry_price)
    liquidity_human = Decimal(budget_usd) / unit_value
    liquidity_raw = int((liquidity_human * DECIMAL_SCALE_RATIO).to_integral_value(rounding=ROUND_FLOOR))
    usdc_amount, weth_amount = synthetic_lp_amounts(
        liquidity_raw=liquidity_raw,
        price_usdc_per_weth=entry_price,
        price_lower=price_lower,
        price_upper=price_upper,
    )
    return liquidity_raw, usdc_amount, weth_amount
