"""Uniswap V3 math helpers used across modules."""

from __future__ import annotations

from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, getcontext
from math import floor, log

from common.constants import (
    TICK_SPACING,
    TOKEN0_DECIMALS,
    TOKEN1_DECIMALS,
    UNISWAP_V3_MAX_TICK,
    UNISWAP_V3_MIN_TICK,
)


getcontext().prec = 80

Q96 = Decimal(2) ** 96
DECIMAL_SCALE_RATIO = Decimal(10) ** (TOKEN1_DECIMALS - TOKEN0_DECIMALS)
ONE_0001 = Decimal("1.0001")


def decimal_sqrt(value: Decimal) -> Decimal:
    """Return the square root of a Decimal."""
    return value.sqrt()


def sqrt_price_x96_to_price_usdc_per_weth(sqrt_price_x96: int) -> Decimal:
    """Convert an on-chain sqrt price to a human USDC/WETH price."""
    sqrt_ratio = Decimal(sqrt_price_x96) / Q96
    raw_price_token1_per_token0 = sqrt_ratio * sqrt_ratio
    return DECIMAL_SCALE_RATIO / raw_price_token1_per_token0


def price_usdc_per_weth_to_sqrt_price_x96(price_usdc_per_weth: Decimal | float | int) -> int:
    """Convert a human USDC/WETH price to the on-chain sqrtPriceX96 format."""
    price = Decimal(price_usdc_per_weth)
    raw_price_token1_per_token0 = DECIMAL_SCALE_RATIO / price
    sqrt_ratio = decimal_sqrt(raw_price_token1_per_token0)
    return int((sqrt_ratio * Q96).to_integral_value(rounding=ROUND_FLOOR))


def tick_to_price_usdc_per_weth(tick: int) -> Decimal:
    """Convert a Uniswap tick to a human USDC/WETH price."""
    raw_price_token1_per_token0 = ONE_0001 ** Decimal(tick)
    return DECIMAL_SCALE_RATIO / raw_price_token1_per_token0


def tick_to_sqrt_price_x96(tick: int) -> int:
    """Convert a Uniswap tick to an on-chain sqrtPriceX96."""
    raw_price_token1_per_token0 = ONE_0001 ** Decimal(tick)
    sqrt_ratio = decimal_sqrt(raw_price_token1_per_token0)
    return int((sqrt_ratio * Q96).to_integral_value(rounding=ROUND_FLOOR))


def price_usdc_per_weth_to_tick(price_usdc_per_weth: Decimal | float | int) -> int:
    """Convert a human USDC/WETH price to the nearest Uniswap tick."""
    price = float(price_usdc_per_weth)
    return int(round(log(float(DECIMAL_SCALE_RATIO) / price, 1.0001)))


def align_tick_to_spacing(tick: int, tick_spacing: int = TICK_SPACING, mode: str = "nearest") -> int:
    """Align a tick to the pool tick spacing."""
    if mode == "down":
        return (tick // tick_spacing) * tick_spacing
    if mode == "up":
        return ((tick + tick_spacing - 1) // tick_spacing) * tick_spacing
    lower = align_tick_to_spacing(tick, tick_spacing=tick_spacing, mode="down")
    upper = align_tick_to_spacing(tick, tick_spacing=tick_spacing, mode="up")
    if abs(tick - lower) <= abs(upper - tick):
        return lower
    return upper


def clamp_tick_to_uniswap_range(tick: int) -> int:
    """Clamp a tick to the supported Uniswap V3 range."""
    return min(max(tick, UNISWAP_V3_MIN_TICK), UNISWAP_V3_MAX_TICK)


def _ordered_sqrt_bounds(a: int, b: int) -> tuple[Decimal, Decimal]:
    left = Decimal(min(a, b))
    right = Decimal(max(a, b))
    return left, right


def amount0_for_liquidity(
    sqrt_price_a_x96: int,
    sqrt_price_b_x96: int,
    liquidity: Decimal | int | float,
) -> Decimal:
    """Return token0 amount implied by liquidity over a price interval."""
    sqrt_a, sqrt_b = _ordered_sqrt_bounds(sqrt_price_a_x96, sqrt_price_b_x96)
    liquidity_value = Decimal(liquidity)
    return liquidity_value * Q96 * (sqrt_b - sqrt_a) / (sqrt_b * sqrt_a)


def amount1_for_liquidity(
    sqrt_price_a_x96: int,
    sqrt_price_b_x96: int,
    liquidity: Decimal | int | float,
) -> Decimal:
    """Return token1 amount implied by liquidity over a price interval."""
    sqrt_a, sqrt_b = _ordered_sqrt_bounds(sqrt_price_a_x96, sqrt_price_b_x96)
    liquidity_value = Decimal(liquidity)
    return liquidity_value * (sqrt_b - sqrt_a) / Q96


def amounts_for_liquidity(
    sqrt_price_x96: int,
    sqrt_price_a_x96: int,
    sqrt_price_b_x96: int,
    liquidity: Decimal | int | float,
) -> tuple[Decimal, Decimal]:
    """Return token0 and token1 raw amounts for a given liquidity position."""
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


def raw_amounts_to_decimal(amount0_raw: Decimal | int | float, amount1_raw: Decimal | int | float) -> tuple[Decimal, Decimal]:
    """Convert raw token amounts into human token amounts."""
    amount0 = Decimal(amount0_raw) / (Decimal(10) ** TOKEN0_DECIMALS)
    amount1 = Decimal(amount1_raw) / (Decimal(10) ** TOKEN1_DECIMALS)
    return amount0, amount1


def liquidity_to_human_scale(liquidity_raw: Decimal | int | float) -> Decimal:
    """Convert raw liquidity to the human-scale liquidity used in analytics derivations."""
    return Decimal(liquidity_raw) / DECIMAL_SCALE_RATIO


def synthetic_lp_amounts(
    liquidity_raw: Decimal | int | float,
    price_usdc_per_weth: Decimal | int | float,
    price_lower: Decimal | int | float,
    price_upper: Decimal | int | float,
) -> tuple[Decimal, Decimal]:
    """Return human token balances for a synthetic LP position."""
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
    """Return the LP delta in ETH units with respect to the ETH/USD price."""
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
    """Return the LP gamma with respect to the ETH/USD price."""
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
    """Solve the synthetic position liquidity from a USD notional budget."""
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

