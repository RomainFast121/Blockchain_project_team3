# EPFL Blockchain Final Project

This repository implements the five required modules from the EPFL final project on the Uniswap V3 USDC/WETH 0.05% pool at `0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640`.

The repository is written as a sequence of Python scripts rather than notebooks so it can be:

- read from top to bottom like a report appendix,
- rerun cleanly from the command line,
- pushed to GitHub without hidden notebook state,
- compared later with another team implementation.

The intended reading order is the same as the PDF:

1. Module 1 builds the on-chain datasets.
2. Module 2 studies liquidity distribution.
3. Module 3 studies execution costs and slippage.
4. Module 4 studies LP fee income and impermanent loss.
5. Module 5 studies delta hedging with Hyperliquid perps.

## Repository structure

- `common/`: shared protocol/math helpers, RPC utilities, plotting helpers, and the Hyperliquid client.
- `module1_onchain_data_extraction/`: event download, daily snapshot reconstruction, and validation against on-chain state.
- `module2_liquidity_distribution_analysis/`: liquidity profiles, TVL decomposition, ILR, and L-HHI.
- `module3_slippage_simulation_and_execution_cost/`: the swap simulator plus the analysis/validation workflow.
- `module4_liquidity_provision_analytics/`: synthetic LP construction, fee accrual, IL, and net P&L.
- `module5_dynamic_hedging_of_impermanent_loss/`: market-data download and delta-hedging backtest.
- `tests/`: deterministic tests for math, analytics logic, and CLI smoke checks.

## Environment

Tested locally with Python `3.13.7`.

All commands below assume you are inside the project root:

```bash
cd /Users/romain/Documents/Epfl/MA2/Blockchain/project
```

Create and populate the environment with:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Execution order

Run the modules in the exact order below. Each module expects the outputs of the
previous ones to already exist.

1. Module 1

```bash
.venv/bin/python -m module1_onchain_data_extraction.data_extraction \
  --rpc-url "YOUR_ARCHIVE_RPC_URL"
```

2. Module 2

```bash
.venv/bin/python -m module2_liquidity_distribution_analysis.liquidity_analysis
```

3. Module 3

```bash
.venv/bin/python -m module3_slippage_simulation_and_execution_cost.slippage_analysis \
  --rpc-url "YOUR_ARCHIVE_RPC_URL"
```

4. Module 4

```bash
.venv/bin/python -m module4_liquidity_provision_analytics.lp_analytics
```

5. Module 5

```bash
.venv/bin/python -m module5_dynamic_hedging_of_impermanent_loss.hedge_backtest
```

Generated parquet files are written under `data/processed/` and generated figures under `figures/`.

## Verification

Run the test suite from the project root with explicit writable cache paths:

```bash
XDG_CACHE_HOME=.cache MPLCONFIGDIR=.cache/matplotlib .venv/bin/python -m unittest discover -s tests
```

Check the documented command-line entry points:

```bash
.venv/bin/python -m module1_onchain_data_extraction.data_extraction --help
.venv/bin/python -m module2_liquidity_distribution_analysis.liquidity_analysis --help
.venv/bin/python -m module3_slippage_simulation_and_execution_cost.slippage_analysis --help
.venv/bin/python -m module4_liquidity_provision_analytics.lp_analytics --help
.venv/bin/python -m module5_dynamic_hedging_of_impermanent_loss.hedge_backtest --help
```

## Assumptions and explicit choices

- The default study window is `2025-10-01` to `2026-03-31`, matching the example in the project brief. Module 1 exposes overrides, but the rest of the repository assumes those defaults unless upstream outputs are regenerated.
- Module 1 saves `collect_events.parquet` as an auxiliary file because the brief explicitly recommends downloading Collect events during the historical scan used for Mint and Burn events.
- Module 2 expands initialized ticks into tick-spacing intervals before computing liquidity profiles and concentration metrics. This is important because initialized ticks are only liquidity change points, not the full profile.
- Module 2 computes TVL decomposition from active liquidity aggregated by `(tick_lower, tick_upper)`. This is sufficient for valuation because Uniswap V3 token amount formulas are linear in liquidity.
- Module 3 validates observed swaps against pool state reconstructed at `block_number - 1`. This is exact at the block level, but not at the intra-block event-order level. That limitation should be mentioned in the report when discussing validation accuracy.
- Module 4 assigns fees using the swap tick and the LP's share of active liquidity at that tick. This is a transparent approximation suitable for the project, but it is still a simplification of Uniswap's internal fee-growth bookkeeping.
- Module 5 defines net position P&L as `LP fee income - residual IL`, where `residual IL = gross IL - net hedge P&L`. This avoids double-counting the hedge contribution.

## Output inventory

- Module 1
  `swap_events.parquet`
  `mint_burn_events.parquet`
  `collect_events.parquet`
  `liquidity_snapshots.parquet`
  `slot0_snapshots.parquet`
  `module1_validations.json`
- Module 2
  `tvl_decomposition.parquet`
  `liquidity_concentration_metrics.parquet`
  `fig_2_1_liquidity_profiles.png`
  `fig_2_2_tvl_decomposition.png`
  `fig_2_3_ilr_timeseries.png`
  `fig_2_4_lhhi_vs_eth_price.png`
- Module 3
  `simulated_trades.parquet`
  `simulator_validation.parquet`
  `observed_effective_spreads.parquet`
  `fig_3_1_simulator_validation.png`
  `fig_3_2_price_impact_curves.png`
  `fig_3_3_effective_spread_vs_simulated.png`
- Module 4
  `synthetic_lp_positions.parquet`
  `lp_fee_accruals.parquet`
  `lp_position_timeseries.parquet`
  `fig_4_1_cumulative_fee_income.png`
  `fig_4_2_impermanent_loss.png`
  `fig_4_3_fee_income_minus_il.png`
- Module 5
  `perp_prices.parquet`
  `funding_rates.parquet`
  `hedge_results.parquet`
  `fig_5_0_lp_payoffs.png`
  `fig_5_0_funding_environment.png`
  `fig_5_1_hedging_results.png`

## Data dictionary

### `swap_events.parquet`

| Column | Type | Unit | Description |
| --- | --- | --- | --- |
| `block_number` | int | block | Ethereum block number of the swap. |
| `block_timestamp` | datetime64[ns, UTC] | UTC timestamp | Timestamp of the block containing the swap. |
| `transaction_hash` | string | hex hash | Transaction hash that emitted the swap event. |
| `log_index` | int | log index | Position of the swap log inside the block. |
| `amount0_raw` | int | token0 raw units | Signed USDC amount from the pool event. Positive means USDC entered the pool. |
| `amount0_usdc` | float | USDC | Decimal-adjusted signed USDC amount. |
| `amount1_raw` | int | token1 raw units | Signed WETH amount from the pool event. Positive means WETH entered the pool. |
| `amount1_weth` | float | WETH | Decimal-adjusted signed WETH amount. |
| `sqrtPriceX96` | int | Q64.96 | Pool sqrt price after the swap. |
| `price_usdc_per_weth` | float | USDC/WETH | Human-readable pool price after the swap. |
| `active_liquidity` | int | liquidity units | Active in-range liquidity after the swap. |
| `tick` | int | tick | Pool tick after the swap. |
| `trade_direction` | string | label | `buy_weth` or `sell_weth` from the taker perspective. |
| `notional_usd` | float | USD | Absolute USDC leg used as the trade notional proxy. |

### `mint_burn_events.parquet`

| Column | Type | Unit | Description |
| --- | --- | --- | --- |
| `block_number` | int | block | Ethereum block number of the event. |
| `transaction_hash` | string | hex hash | Transaction hash that emitted the Mint or Burn event. |
| `log_index` | int | log index | Position of the log inside the block. |
| `event_type` | string | label | `mint` or `burn`. |
| `owner` | string | address | LP wallet address. |
| `tick_lower` | int | tick | Lower tick of the LP range. |
| `tick_upper` | int | tick | Upper tick of the LP range. |
| `liquidity_raw` | int | liquidity units | Liquidity amount added or removed. |
| `amount0_raw` | int | token0 raw units | Raw USDC amount associated with the event. |
| `amount0_usdc` | float | USDC | Decimal-adjusted USDC amount. |
| `amount1_raw` | int | token1 raw units | Raw WETH amount associated with the event. |
| `amount1_weth` | float | WETH | Decimal-adjusted WETH amount. |
| `block_timestamp` | datetime64[ns, UTC] | UTC timestamp | Timestamp of the block containing the event. |

### `collect_events.parquet`

| Column | Type | Unit | Description |
| --- | --- | --- | --- |
| `block_number` | int | block | Ethereum block number of the collect event. |
| `transaction_hash` | string | hex hash | Transaction hash that emitted the collect event. |
| `log_index` | int | log index | Position of the log inside the block. |
| `owner` | string | address | LP wallet address. |
| `recipient` | string | address | Recipient of the collected fees. |
| `tick_lower` | int | tick | Lower tick of the LP range. |
| `tick_upper` | int | tick | Upper tick of the LP range. |
| `amount0_raw` | int | token0 raw units | Raw collected USDC fees. |
| `amount0_usdc` | float | USDC | Decimal-adjusted collected USDC fees. |
| `amount1_raw` | int | token1 raw units | Raw collected WETH fees. |
| `amount1_weth` | float | WETH | Decimal-adjusted collected WETH fees. |
| `block_timestamp` | datetime64[ns, UTC] | UTC timestamp | Timestamp of the collect event. |

### `liquidity_snapshots.parquet`

| Column | Type | Unit | Description |
| --- | --- | --- | --- |
| `snapshot_block` | int | block | Snapshot block closest to 00:00 UTC on the day. |
| `snapshot_timestamp` | datetime64[ns, UTC] | UTC timestamp | Snapshot reference timestamp. |
| `tick` | int | tick | Initialised tick index. |
| `liquidityNet` | int | liquidity units | Signed net liquidity change at the tick. |
| `liquidityGross` | int | liquidity units | Total gross liquidity attached to the tick. |
| `active_liquidity` | int | liquidity units | Cumulative active liquidity after crossing the tick from low to high. |
| `price_lower` | float | USDC/WETH | Lower edge of the tick interval in human price terms. |
| `price_upper` | float | USDC/WETH | Upper edge of the tick interval in human price terms. |

### `slot0_snapshots.parquet`

| Column | Type | Unit | Description |
| --- | --- | --- | --- |
| `snapshot_block` | int | block | Snapshot block closest to 00:00 UTC on the day. |
| `snapshot_timestamp` | datetime64[ns, UTC] | UTC timestamp | Snapshot reference timestamp. |
| `sqrtPriceX96` | int | Q64.96 | Pool sqrt price from `slot0()`. |
| `price_usdc_per_weth` | float | USDC/WETH | Human-readable price from `sqrtPriceX96`. |
| `current_tick` | int | tick | Active pool tick from `slot0()`. |
| `observation_index` | int | index | Oracle observation index from `slot0()`. |
| `unlocked` | bool | flag | Pool lock flag from `slot0()`. |

### `tvl_decomposition.parquet`

| Column | Type | Unit | Description |
| --- | --- | --- | --- |
| `snapshot_block` | int | block | Daily snapshot block. |
| `snapshot_timestamp` | datetime64[ns, UTC] | UTC timestamp | Snapshot timestamp. |
| `price_usdc_per_weth` | float | USDC/WETH | Daily ETH price used for valuation. |
| `tvl_in_range` | float | USD | In-range TVL. |
| `tvl_above_range` | float | USD | TVL in ranges fully above the current price. |
| `tvl_below_range` | float | USD | TVL in ranges fully below the current price. |
| `tvl_total` | float | USD | Total TVL across all active ranges. |

### `liquidity_concentration_metrics.parquet`

| Column | Type | Unit | Description |
| --- | --- | --- | --- |
| `snapshot_block` | int | block | Daily snapshot block. |
| `snapshot_timestamp` | datetime64[ns, UTC] | UTC timestamp | Snapshot timestamp. |
| `price_usdc_per_weth` | float | USDC/WETH | Daily ETH price used for the overlay. |
| `l_hhi` | float | index | Liquidity HHI at the snapshot. |
| `ilr_0_1pct` | float | share | ILR within +/-0.1%. |
| `ilr_0_5pct` | float | share | ILR within +/-0.5%. |
| `ilr_1_0pct` | float | share | ILR within +/-1%. |
| `ilr_2_0pct` | float | share | ILR within +/-2%. |
| `ilr_5_0pct` | float | share | ILR within +/-5%. |

### `simulated_trades.parquet`

| Column | Type | Unit | Description |
| --- | --- | --- | --- |
| `snapshot_block` | int | block | Daily snapshot block used for the simulation. |
| `snapshot_timestamp` | datetime64[ns, UTC] | UTC timestamp | Snapshot timestamp. |
| `direction` | string | label | `buy_weth` or `sell_weth`. |
| `notional_usd` | int | USD | Input trade size. |
| `average_price` | float | USDC/WETH | Average execution price. |
| `pool_mid_price` | float | USDC/WETH | Pre-trade mid-price from `slot0()`. |
| `price_impact_bps` | float | bps | Direction-adjusted execution impact. |
| `slippage_bps` | float | bps | Price impact net of the 5 bps trading fee. |
| `tick_crosses` | int | count | Number of initialized tick boundaries crossed. |
| `ending_price` | float | USDC/WETH | Pool price after the simulated trade. |

### `simulator_validation.parquet`

| Column | Type | Unit | Description |
| --- | --- | --- | --- |
| `direction` | string | label | Observed trade direction. |
| `size_bucket_usd` | int | USD | Closest simulation bucket. |
| `block_number` | int | block | Block number of the observed swap. |
| `actual_execution_price` | float | USDC/WETH | Execution price implied by the observed swap. |
| `simulated_execution_price` | float | USDC/WETH | Execution price from the simulator. |
| `percentage_error` | float | % | Absolute percentage error. |

### `observed_effective_spreads.parquet`

| Column | Type | Unit | Description |
| --- | --- | --- | --- |
| `block_number` | int | block | Block number of the observed swap. |
| `block_timestamp` | datetime64[ns, UTC] | UTC timestamp | Timestamp of the observed swap. |
| `direction` | string | label | `buy_weth` or `sell_weth`. |
| `notional_usd` | float | USD | Observed swap notional. |
| `size_bucket_usd` | int | USD | Closest simulation bucket. |
| `execution_price` | float | USDC/WETH | Observed execution price. |
| `mid_price_prior_block` | float | USDC/WETH | Mid-price from `slot0()` at the previous block. |
| `effective_spread_bps` | float | bps | Observed effective spread. |

### `synthetic_lp_positions.parquet`

| Column | Type | Unit | Description |
| --- | --- | --- | --- |
| `position_id` | string | label | Position identifier `P1` to `P5`. |
| `position_label` | string | label | Human-readable width label. |
| `width_pct` | float | % | Target half-width around the entry price. `NaN` for full range. |
| `entry_snapshot_block` | int | block | Entry snapshot block. |
| `exit_snapshot_block` | int | block | Exit snapshot block. |
| `entry_timestamp` | datetime64[ns, UTC] | UTC timestamp | Entry timestamp. |
| `exit_timestamp` | datetime64[ns, UTC] | UTC timestamp | Exit timestamp. |
| `entry_price_usdc_per_weth` | float | USDC/WETH | Entry ETH price. |
| `tick_lower` | int | tick | Lower tick of the synthetic position. |
| `tick_upper` | int | tick | Upper tick of the synthetic position. |
| `price_lower` | float | USDC/WETH | Lower price bound in human units. |
| `price_upper` | float | USDC/WETH | Upper price bound in human units. |
| `liquidity_raw` | int | liquidity units | Synthetic LP liquidity sized to approximately USD 100,000 at entry. |
| `entry_usdc` | float | USDC | Entry USDC inventory. |
| `entry_weth` | float | WETH | Entry WETH inventory. |
| `entry_budget_usd` | int | USD | Entry notional budget. |

### `lp_fee_accruals.parquet`

| Column | Type | Unit | Description |
| --- | --- | --- | --- |
| `position_id` | string | label | Synthetic LP identifier. |
| `block_number` | int | block | Swap block number generating the fee. |
| `block_timestamp` | datetime64[ns, UTC] | UTC timestamp | Swap timestamp. |
| `trade_direction` | string | label | Taker direction of the underlying swap. |
| `swap_price_usdc_per_weth` | float | USDC/WETH | ETH price at the swap. |
| `fee_usdc` | float | USDC | Synthetic LP fee earned in USDC. |
| `fee_weth` | float | WETH | Synthetic LP fee earned in WETH. |
| `fee_value_usd` | float | USD | USD value of the fee earned on the swap. |
| `cumulative_fee_income_usd` | float | USD | Position-level cumulative fee income. |

### `lp_position_timeseries.parquet`

| Column | Type | Unit | Description |
| --- | --- | --- | --- |
| `position_id` | string | label | Synthetic LP identifier. |
| `position_label` | string | label | Human-readable width label. |
| `snapshot_block` | int | block | Daily snapshot block. |
| `snapshot_timestamp` | datetime64[ns, UTC] | UTC timestamp | Snapshot timestamp. |
| `price_usdc_per_weth` | float | USDC/WETH | Daily ETH price. |
| `lp_principal_usd` | float | USD | Mark-to-market LP principal value. |
| `hodl_value_usd` | float | USD | HODL benchmark value using the entry inventory. |
| `impermanent_loss_usd` | float | USD | `V_HODL - V_LP`. |
| `cumulative_fee_income_usd` | float | USD | Cumulative fee income up to the snapshot. |
| `net_pnl_usd` | float | USD | `cumulative_fee_income_usd - impermanent_loss_usd`. |

### `perp_prices.parquet`

| Column | Type | Unit | Description |
| --- | --- | --- | --- |
| `timestamp` | datetime64[ns, UTC] | UTC timestamp | Candle close timestamp. |
| `open` | float | USDC/WETH | Candle open price. |
| `high` | float | USDC/WETH | Candle high price. |
| `low` | float | USDC/WETH | Candle low price. |
| `close` | float | USDC/WETH | Candle close price used as the mark price. |
| `volume` | float | contract units | Reported candle volume from Hyperliquid. |
| `t` | int | ms | Candle start time in milliseconds. |
| `T` | int | ms | Candle end time in milliseconds. |
| `n` | int | count | Number of trades in the candle. |
| `s` | string | symbol | Hyperliquid instrument symbol. |
| `i` | string | interval | Candle interval, expected to be `1h`. |

### `funding_rates.parquet`

| Column | Type | Unit | Description |
| --- | --- | --- | --- |
| `timestamp` | datetime64[ns, UTC] | UTC timestamp | Funding observation timestamp. |
| `coin` | string | symbol | Hyperliquid perpetual symbol. |
| `funding_rate` | float | hourly rate | Hourly funding rate. |
| `premium` | float | rate | Premium series returned by the API. |
| `time` | int | ms | Original API timestamp in milliseconds. |

### `hedge_results.parquet`

| Column | Type | Unit | Description |
| --- | --- | --- | --- |
| `position_id` | string | label | Synthetic LP identifier. |
| `frequency` | string | label | Rebalancing frequency: `1h`, `4h`, or `24h`. |
| `timestamp` | datetime64[ns, UTC] | UTC timestamp | Hourly backtest timestamp. |
| `price_usdc_per_weth` | float | USDC/WETH | Hourly mark price. |
| `lp_principal_usd` | float | USD | LP principal value at the hourly mark. |
| `hodl_value_usd` | float | USD | HODL benchmark value at the hourly mark. |
| `gross_il_usd` | float | USD | Unhedged impermanent loss. |
| `hedge_size_eth` | float | ETH | Short perpetual hedge size. |
| `hedge_pnl_usd` | float | USD | Cumulative mark-to-market hedge P&L. |
| `funding_pnl_usd` | float | USD | Cumulative funding P&L. |
| `trading_fees_usd` | float | USD | Cumulative perpetual trading fees. |
| `net_hedge_pnl_usd` | float | USD | `hedge_pnl_usd + funding_pnl_usd - trading_fees_usd`. |
| `residual_il_usd` | float | USD | `gross_il_usd - net_hedge_pnl_usd`. |
| `lp_fee_income_usd` | float | USD | Cumulative LP fee income mapped to the hourly grid. |
| `net_position_pnl_usd` | float | USD | `lp_fee_income_usd - residual_il_usd`. |

## Local verification

Run the full deterministic test suite with:

```bash
MPLCONFIGDIR="$(pwd)/.cache/matplotlib" .venv/bin/python -m unittest discover -s tests
```
