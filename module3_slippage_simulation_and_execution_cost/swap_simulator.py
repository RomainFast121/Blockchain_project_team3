"""Standalone Uniswap V3 exact-input swap simulator.

This file is the engine used by Module 3. The surrounding analysis script loops
over days, directions, and trade sizes, but all of the actual execution logic
lives here.

The implementation follows the standard Uniswap V3 idea:

- work within one constant-liquidity interval at a time,
- move the square-root price until either the trade is consumed or the next
  initialized tick is reached,
- if a tick is crossed, update active liquidity and continue.
"""

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
    """Pool state reconstructed for one snapshot block."""

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
        """Build a simulator state from Module 1 daily outputs."""

        initialized_ticks = liquidity_snapshot.sort_values("tick").reset_index(drop=True)
        current_tick = int(slot0_snapshot["current_tick"])

        # The active liquidity at the current tick is the last cumulative level
        # observed at or below the active pool tick.
        eligible_rows = initialized_ticks[initialized_ticks["tick"] <= current_tick]
        if eligible_rows.empty:
            active_liquidity = 0
        else:
            active_liquidity = int(eligible_rows.iloc[-1]["active_liquidity"])

        return cls(
            snapshot_block=int(slot0_snapshot["snapshot_block"]),
            snapshot_timestamp=pd.Timestamp(slot0_snapshot["snapshot_timestamp"]),
            sqrt_price_x96=int(slot0_snapshot["sqrtPriceX96"]),
            current_tick=current_tick,
            current_price=float(slot0_snapshot["price_usdc_per_weth"]),
            active_liquidity=active_liquidity,
            initialized_ticks=tuple(int(tick) for tick in initialized_ticks["tick"].tolist()),
            liquidity_net_by_tick={int(row["tick"]): int(row["liquidityNet"]) for _, row in initialized_ticks.iterrows()},
        )


@dataclass(frozen=True)
class SwapSimulationResult:
    """Summary of one simulated trade."""

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
    """Token0 change implied by moving between two sqrt-price levels."""

    lower = min(sqrt_a, sqrt_b)
    upper = max(sqrt_a, sqrt_b)
    return liquidity * Q96 * (upper - lower) / (upper * lower)


def _amount1_delta(liquidity: Decimal, sqrt_a: Decimal, sqrt_b: Decimal) -> Decimal:
    """Token1 change implied by moving between two sqrt-price levels."""

    lower = min(sqrt_a, sqrt_b)
    upper = max(sqrt_a, sqrt_b)
    return liquidity * (upper - lower) / Q96


def _next_sqrt_from_token0_input(sqrt_current: Decimal, liquidity: Decimal, amount0_in: Decimal) -> Decimal:
    """New sqrt price after token0 is added to the pool."""

    numerator = liquidity * Q96 * sqrt_current
    denominator = liquidity * Q96 + amount0_in * sqrt_current
    return numerator / denominator


def _next_sqrt_from_token1_input(sqrt_current: Decimal, liquidity: Decimal, amount1_in: Decimal) -> Decimal:
    """New sqrt price after token1 is added to the pool."""

    return sqrt_current + (amount1_in * Q96 / liquidity)


def _compute_swap_step(
    sqrt_current_x96: Decimal,
    sqrt_target_x96: Decimal,
    liquidity: Decimal,
    amount_remaining: Decimal,
    zero_for_one: bool,
) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    """Simulate one step within one constant-liquidity segment.

    Returns:
    - next sqrt price,
    - net input consumed before fee,
    - output produced,
    - fee paid in input token units.
    """

    amount_remaining_less_fee = amount_remaining * (Decimal(1) - FEE_RATE)

    if zero_for_one:
        max_input_before_tick = _amount0_delta(liquidity, sqrt_target_x96, sqrt_current_x96)
        if amount_remaining_less_fee >= max_input_before_tick:
            sqrt_next = sqrt_target_x96
            amount_in = max_input_before_tick
            fee_amount = amount_in * FEE_RATE / (Decimal(1) - FEE_RATE)
        else:
            sqrt_next = _next_sqrt_from_token0_input(sqrt_current_x96, liquidity, amount_remaining_less_fee)
            amount_in = _amount0_delta(liquidity, sqrt_next, sqrt_current_x96)
            fee_amount = amount_remaining - amount_in
        amount_out = _amount1_delta(liquidity, sqrt_next, sqrt_current_x96)
    else:
        max_input_before_tick = _amount1_delta(liquidity, sqrt_current_x96, sqrt_target_x96)
        if amount_remaining_less_fee >= max_input_before_tick:
            sqrt_next = sqrt_target_x96
            amount_in = max_input_before_tick
            fee_amount = amount_in * FEE_RATE / (Decimal(1) - FEE_RATE)
        else:
            sqrt_next = _next_sqrt_from_token1_input(sqrt_current_x96, liquidity, amount_remaining_less_fee)
            amount_in = _amount1_delta(liquidity, sqrt_current_x96, sqrt_next)
            fee_amount = amount_remaining - amount_in
        amount_out = _amount0_delta(liquidity, sqrt_current_x96, sqrt_next)

    return sqrt_next, amount_in, amount_out, fee_amount


def _find_next_initialized_tick(initialized_ticks: Iterable[int], current_tick: int, zero_for_one: bool) -> int | None:
    """Return the next initialized tick in the trade direction."""

    ticks = list(initialized_ticks)
    if zero_for_one:
        candidate_ticks = [tick for tick in ticks if tick < current_tick]
        return max(candidate_ticks) if candidate_ticks else None

    candidate_ticks = [tick for tick in ticks if tick > current_tick]
    return min(candidate_ticks) if candidate_ticks else None


def _input_notional_to_raw_units(state: SnapshotPoolState, direction: str, notional_usd: float) -> Decimal:
    """Turn the requested trade notional into raw token units."""

    if direction == "buy_weth":
        return Decimal(str(notional_usd)) * (Decimal(10) ** TOKEN0_DECIMALS)

    weth_amount = Decimal(str(notional_usd)) / Decimal(str(state.current_price))
    return weth_amount * (Decimal(10) ** TOKEN1_DECIMALS)


def _build_result(
    state: SnapshotPoolState,
    direction: str,
    notional_usd: float,
    total_input_raw: Decimal,
    total_output_raw: Decimal,
    ending_sqrt_price_x96: Decimal,
    tick_crosses: int,
) -> SwapSimulationResult:
    """Convert raw simulator bookkeeping into report-friendly execution metrics."""

    zero_for_one = direction == "buy_weth"
    if zero_for_one:
        amount_in_decimal = float(total_input_raw / (Decimal(10) ** TOKEN0_DECIMALS))
        amount_out_decimal = float(total_output_raw / (Decimal(10) ** TOKEN1_DECIMALS))
        average_price = amount_in_decimal / amount_out_decimal
        price_impact_bps = ((average_price / state.current_price) - 1.0) * 10_000
    else:
        amount_in_decimal = float(total_input_raw / (Decimal(10) ** TOKEN1_DECIMALS))
        amount_out_decimal = float(total_output_raw / (Decimal(10) ** TOKEN0_DECIMALS))
        average_price = amount_out_decimal / amount_in_decimal
        price_impact_bps = ((state.current_price / average_price) - 1.0) * 10_000

    slippage_bps = max(price_impact_bps - float(FEE_RATE * Decimal(10_000)), 0.0)
    ending_price = float(sqrt_price_x96_to_price_usdc_per_weth(int(ending_sqrt_price_x96)))

    return SwapSimulationResult(
        direction=direction,
        notional_usd=float(notional_usd),
        amount_in_raw=int(total_input_raw),
        amount_out_raw=int(total_output_raw),
        amount_in_decimal=amount_in_decimal,
        amount_out_decimal=amount_out_decimal,
        average_price=average_price,
        pool_mid_price=state.current_price,
        price_impact_bps=price_impact_bps,
        slippage_bps=slippage_bps,
        tick_crosses=tick_crosses,
        ending_price=ending_price,
    )


def simulate_exact_input_swap(
    state: SnapshotPoolState,
    direction: str,
    notional_usd: float,
) -> SwapSimulationResult:
    """Simulate one exact-input trade through the reconstructed pool state."""

    if direction not in {"buy_weth", "sell_weth"}:
        raise ValueError("direction must be 'buy_weth' or 'sell_weth'")

    zero_for_one = direction == "buy_weth"
    amount_remaining = _input_notional_to_raw_units(state, direction, notional_usd)
    total_input_raw = Decimal(0)
    total_output_raw = Decimal(0)
    tick_crosses = 0

    current_sqrt_price_x96 = Decimal(state.sqrt_price_x96)
    current_tick = int(state.current_tick)
    active_liquidity = Decimal(state.active_liquidity)

    # Keep walking through initialized ticks until the trade is fully consumed or
    # there is no liquidity left to trade against.
    while amount_remaining > Decimal("0.0001") and active_liquidity > 0:
        next_initialized_tick = _find_next_initialized_tick(state.initialized_ticks, current_tick, zero_for_one)
        if next_initialized_tick is None:
            boundary_tick = UNISWAP_V3_MIN_TICK if zero_for_one else UNISWAP_V3_MAX_TICK
            target_sqrt_price_x96 = Decimal(tick_to_sqrt_price_x96(boundary_tick))
        else:
            target_sqrt_price_x96 = Decimal(tick_to_sqrt_price_x96(next_initialized_tick))

        sqrt_next, amount_in, amount_out, fee_amount = _compute_swap_step(
            sqrt_current_x96=current_sqrt_price_x96,
            sqrt_target_x96=target_sqrt_price_x96,
            liquidity=active_liquidity,
            amount_remaining=amount_remaining,
            zero_for_one=zero_for_one,
        )

        trade_input_this_step = amount_in + fee_amount
        total_input_raw += trade_input_this_step
        total_output_raw += amount_out
        amount_remaining -= trade_input_this_step
        current_sqrt_price_x96 = sqrt_next

        reached_tick_boundary = abs(sqrt_next - target_sqrt_price_x96) <= Decimal("1")
        if not reached_tick_boundary or next_initialized_tick is None:
            break

        # Crossing a tick changes active liquidity by the tick's liquidityNet.
        liquidity_net_at_tick = Decimal(state.liquidity_net_by_tick.get(next_initialized_tick, 0))
        if zero_for_one:
            active_liquidity = active_liquidity - liquidity_net_at_tick
            current_tick = next_initialized_tick - 1
        else:
            active_liquidity = active_liquidity + liquidity_net_at_tick
            current_tick = next_initialized_tick
        tick_crosses += 1

    return _build_result(
        state=state,
        direction=direction,
        notional_usd=notional_usd,
        total_input_raw=total_input_raw,
        total_output_raw=total_output_raw,
        ending_sqrt_price_x96=current_sqrt_price_x96,
        tick_crosses=tick_crosses,
    )
