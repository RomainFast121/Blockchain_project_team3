"""Standalone Uniswap V3 swap simulator for exact-input trades."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, getcontext
from typing import Iterable

import pandas as pd

from common.constants import (
    POOL_FEE_RATE,
    TOKEN0_DECIMALS,
    TOKEN1_DECIMALS,
    UNISWAP_V3_MAX_TICK,
    UNISWAP_V3_MIN_TICK,
)
from common.uniswap_math import Q96, sqrt_price_x96_to_price_usdc_per_weth, tick_to_sqrt_price_x96


getcontext().prec = 90

FEE_RATE = Decimal(str(POOL_FEE_RATE))


@dataclass(frozen=True)
class SnapshotPoolState:
    """Reconstructed pool state at a specific snapshot block."""

    snapshot_block: int
    snapshot_timestamp: pd.Timestamp
    sqrt_price_x96: int
    current_tick: int
    current_price: float
    active_liquidity: int
    initialized_ticks: tuple[int, ...]
    liquidity_net_by_tick: dict[int, int]

    @classmethod
    def from_snapshot_frames(cls, liquidity_snapshot: pd.DataFrame, slot0_snapshot: pd.Series) -> "SnapshotPoolState":
        """Build a simulation state from Module 1 snapshot outputs."""
        frame = liquidity_snapshot.sort_values("tick").reset_index(drop=True)
        current_tick = int(slot0_snapshot["current_tick"])
        eligible_ticks = frame[frame["tick"] <= current_tick]
        active_liquidity = int(eligible_ticks.iloc[-1]["active_liquidity"]) if not eligible_ticks.empty else 0
        return cls(
            snapshot_block=int(slot0_snapshot["snapshot_block"]),
            snapshot_timestamp=pd.Timestamp(slot0_snapshot["snapshot_timestamp"]),
            sqrt_price_x96=int(slot0_snapshot["sqrtPriceX96"]),
            current_tick=current_tick,
            current_price=float(slot0_snapshot["price_usdc_per_weth"]),
            active_liquidity=active_liquidity,
            initialized_ticks=tuple(int(value) for value in frame["tick"].tolist()),
            liquidity_net_by_tick={int(row["tick"]): int(row["liquidityNet"]) for _, row in frame.iterrows()},
        )


@dataclass(frozen=True)
class SwapSimulationResult:
    """Simulation output for one exact-input trade."""

    direction: str
    notional_usd: float
    amount_in_raw: int
    amount_out_raw: int
    amount_in_decimal: float
    amount_out_decimal: float
    average_price: float
    pool_mid_price: float
    price_impact_bps: float
    slippage_bps: float
    tick_crosses: int
    ending_price: float


def _amount0_delta(liquidity: Decimal, sqrt_a: Decimal, sqrt_b: Decimal) -> Decimal:
    lower = min(sqrt_a, sqrt_b)
    upper = max(sqrt_a, sqrt_b)
    return liquidity * Q96 * (upper - lower) / (upper * lower)


def _amount1_delta(liquidity: Decimal, sqrt_a: Decimal, sqrt_b: Decimal) -> Decimal:
    lower = min(sqrt_a, sqrt_b)
    upper = max(sqrt_a, sqrt_b)
    return liquidity * (upper - lower) / Q96


def _next_sqrt_from_token0_input(sqrt_current: Decimal, liquidity: Decimal, amount0_in: Decimal) -> Decimal:
    numerator = liquidity * Q96 * sqrt_current
    denominator = liquidity * Q96 + amount0_in * sqrt_current
    return numerator / denominator


def _next_sqrt_from_token1_input(sqrt_current: Decimal, liquidity: Decimal, amount1_in: Decimal) -> Decimal:
    return sqrt_current + (amount1_in * Q96 / liquidity)


def _compute_swap_step(
    sqrt_current_x96: Decimal,
    sqrt_target_x96: Decimal,
    liquidity: Decimal,
    amount_remaining: Decimal,
    zero_for_one: bool,
) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    """Compute one constant-liquidity swap step."""
    amount_remaining_less_fee = amount_remaining * (Decimal(1) - FEE_RATE)

    if zero_for_one:
        max_input = _amount0_delta(liquidity, sqrt_target_x96, sqrt_current_x96)
        if amount_remaining_less_fee >= max_input:
            sqrt_next = sqrt_target_x96
            amount_in = max_input
            fee_amount = amount_in * FEE_RATE / (Decimal(1) - FEE_RATE)
        else:
            sqrt_next = _next_sqrt_from_token0_input(sqrt_current_x96, liquidity, amount_remaining_less_fee)
            amount_in = _amount0_delta(liquidity, sqrt_next, sqrt_current_x96)
            fee_amount = amount_remaining - amount_in
        amount_out = _amount1_delta(liquidity, sqrt_next, sqrt_current_x96)
    else:
        max_input = _amount1_delta(liquidity, sqrt_current_x96, sqrt_target_x96)
        if amount_remaining_less_fee >= max_input:
            sqrt_next = sqrt_target_x96
            amount_in = max_input
            fee_amount = amount_in * FEE_RATE / (Decimal(1) - FEE_RATE)
        else:
            sqrt_next = _next_sqrt_from_token1_input(sqrt_current_x96, liquidity, amount_remaining_less_fee)
            amount_in = _amount1_delta(liquidity, sqrt_current_x96, sqrt_next)
            fee_amount = amount_remaining - amount_in
        amount_out = _amount0_delta(liquidity, sqrt_current_x96, sqrt_next)
    return sqrt_next, amount_in, amount_out, fee_amount


def _find_next_initialized_tick(initialized_ticks: Iterable[int], current_tick: int, zero_for_one: bool) -> int | None:
    ticks = list(initialized_ticks)
    if zero_for_one:
        lower_ticks = [tick for tick in ticks if tick < current_tick]
        return max(lower_ticks) if lower_ticks else None
    upper_ticks = [tick for tick in ticks if tick > current_tick]
    return min(upper_ticks) if upper_ticks else None


def simulate_exact_input_swap(
    state: SnapshotPoolState,
    direction: str,
    notional_usd: float,
) -> SwapSimulationResult:
    """Simulate an exact-input trade through the reconstructed liquidity map."""
    if direction not in {"buy_weth", "sell_weth"}:
        raise ValueError("direction must be 'buy_weth' or 'sell_weth'")

    zero_for_one = direction == "buy_weth"
    input_raw = (
        Decimal(str(notional_usd)) * (Decimal(10) ** TOKEN0_DECIMALS)
        if zero_for_one
        else (Decimal(str(notional_usd)) / Decimal(str(state.current_price))) * (Decimal(10) ** TOKEN1_DECIMALS)
    )
    amount_remaining = input_raw
    total_input = Decimal(0)
    total_output = Decimal(0)
    tick_crosses = 0

    sqrt_price = Decimal(state.sqrt_price_x96)
    current_tick = int(state.current_tick)
    active_liquidity = Decimal(state.active_liquidity)

    while amount_remaining > Decimal("0.0001") and active_liquidity > 0:
        next_tick = _find_next_initialized_tick(state.initialized_ticks, current_tick, zero_for_one)
        if next_tick is None:
            boundary_tick = UNISWAP_V3_MIN_TICK if zero_for_one else UNISWAP_V3_MAX_TICK
            target_sqrt = Decimal(tick_to_sqrt_price_x96(boundary_tick))
        else:
            target_sqrt = Decimal(tick_to_sqrt_price_x96(next_tick))

        sqrt_next, amount_in, amount_out, fee_amount = _compute_swap_step(
            sqrt_current_x96=sqrt_price,
            sqrt_target_x96=target_sqrt,
            liquidity=active_liquidity,
            amount_remaining=amount_remaining,
            zero_for_one=zero_for_one,
        )
        spent = amount_in + fee_amount
        total_input += spent
        total_output += amount_out
        amount_remaining -= spent
        reached_target = abs(sqrt_next - target_sqrt) <= Decimal("1")
        sqrt_price = sqrt_next

        if not reached_target:
            break

        if next_tick is None:
            break

        liquidity_net = Decimal(state.liquidity_net_by_tick.get(next_tick, 0))
        active_liquidity = active_liquidity - liquidity_net if zero_for_one else active_liquidity + liquidity_net
        current_tick = next_tick - 1 if zero_for_one else next_tick
        tick_crosses += 1

    if zero_for_one:
        amount_in_decimal = float(total_input / (Decimal(10) ** TOKEN0_DECIMALS))
        amount_out_decimal = float(total_output / (Decimal(10) ** TOKEN1_DECIMALS))
        average_price = amount_in_decimal / amount_out_decimal
        price_impact_bps = ((average_price / state.current_price) - 1.0) * 10_000
    else:
        amount_in_decimal = float(total_input / (Decimal(10) ** TOKEN1_DECIMALS))
        amount_out_decimal = float(total_output / (Decimal(10) ** TOKEN0_DECIMALS))
        average_price = amount_out_decimal / amount_in_decimal
        price_impact_bps = ((state.current_price / average_price) - 1.0) * 10_000

    slippage_bps = max(price_impact_bps - float(FEE_RATE * Decimal(10_000)), 0.0)
    ending_price = float(sqrt_price_x96_to_price_usdc_per_weth(int(sqrt_price)))
    return SwapSimulationResult(
        direction=direction,
        notional_usd=float(notional_usd),
        amount_in_raw=int(total_input),
        amount_out_raw=int(total_output),
        amount_in_decimal=amount_in_decimal,
        amount_out_decimal=amount_out_decimal,
        average_price=average_price,
        pool_mid_price=state.current_price,
        price_impact_bps=price_impact_bps,
        slippage_bps=slippage_bps,
        tick_crosses=tick_crosses,
        ending_price=ending_price,
    )

