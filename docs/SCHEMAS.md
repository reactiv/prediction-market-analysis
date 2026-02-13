# Data Schemas

Data is stored as Parquet files in `data/{kalshi,polymarket}/`.

## Kalshi Markets

Each row represents a prediction market contract.

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | string | Unique market identifier (e.g., `PRES-2024-DJT`) |
| `event_ticker` | string | Parent event identifier, used for categorization |
| `market_type` | string | Market type (typically `binary`) |
| `title` | string | Human-readable market title |
| `yes_sub_title` | string | Label for the "Yes" outcome |
| `no_sub_title` | string | Label for the "No" outcome |
| `status` | string | Market status: `open`, `closed`, `finalized` |
| `yes_bid` | int (nullable) | Best bid price for Yes contracts (cents, 1-99) |
| `yes_ask` | int (nullable) | Best ask price for Yes contracts (cents, 1-99) |
| `no_bid` | int (nullable) | Best bid price for No contracts (cents, 1-99) |
| `no_ask` | int (nullable) | Best ask price for No contracts (cents, 1-99) |
| `last_price` | int (nullable) | Last traded price (cents, 1-99) |
| `volume` | int | Total contracts traded |
| `volume_24h` | int | Contracts traded in last 24 hours |
| `open_interest` | int | Outstanding contracts |
| `result` | string | Market outcome: `yes`, `no`, or empty if unresolved |
| `created_time` | datetime | When the market was created |
| `open_time` | datetime (nullable) | When trading opened |
| `close_time` | datetime (nullable) | When trading closed |
| `_fetched_at` | datetime | When this record was fetched |

## Kalshi Trades

Each row represents a single trade execution.

| Column | Type | Description |
|--------|------|-------------|
| `trade_id` | string | Unique trade identifier |
| `ticker` | string | Market ticker this trade belongs to |
| `count` | int | Number of contracts traded |
| `yes_price` | int | Yes contract price (cents, 1-99) |
| `no_price` | int | No contract price (cents, 1-99), always `100 - yes_price` |
| `taker_side` | string | Which side the taker bought: `yes` or `no` |
| `created_time` | datetime | When the trade occurred |
| `_fetched_at` | datetime | When this record was fetched |

**Note on Kalshi prices:** Prices are in cents. A `yes_price` of 65 means the contract costs $0.65 and pays $1.00 if the outcome is "Yes" (implied probability: 65%). The `no_price` is always `100 - yes_price`.

## Polymarket Markets

Each row represents a prediction market.

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Market ID |
| `condition_id` | string | Condition ID (hex hash) |
| `question` | string | Market question |
| `slug` | string | URL slug |
| `outcomes` | string | JSON string of outcome names |
| `outcome_prices` | string | JSON string of outcome prices |
| `volume` | float | Total volume in USD |
| `liquidity` | float | Current liquidity in USD |
| `active` | bool | Is market active |
| `closed` | bool | Is market closed |
| `end_date` | datetime (nullable) | When market ends |
| `created_at` | datetime (nullable) | When market was created |
| `_fetched_at` | datetime | When this record was fetched |

## Polymarket Trades

Each row represents an `OrderFilled` event from the Polygon blockchain.

| Column | Type | Description |
|--------|------|-------------|
| `block_number` | int | Polygon block number |
| `transaction_hash` | string | Blockchain transaction hash |
| `log_index` | int | Log index within transaction |
| `order_hash` | string | Unique order identifier |
| `maker` | string | Address of limit order placer |
| `taker` | string | Address that filled the order |
| `maker_asset_id` | int | Asset ID maker provided (0=USDC) |
| `taker_asset_id` | int | Asset ID taker provided |
| `maker_amount` | int | Amount maker gave (6 decimals) |
| `taker_amount` | int | Amount taker gave (6 decimals) |
| `fee` | int | Trading fee (6 decimals) |
| `_fetched_at` | datetime | When this record was fetched |
| `_contract` | string | Contract name (CTF Exchange or NegRisk) |

**Note on Polymarket prices:** Prices are decimals between 0 and 1. A price of 0.65 means the contract costs $0.65 and pays $1.00 if the outcome wins (implied probability: 65%).

## Polymarket Legacy Trades (FPMM)

Each row represents an `FPMMBuy` or `FPMMSell` event from the legacy Fixed Product Market Maker contracts on Polygon. These are trades from before Polymarket migrated to the CTF Exchange (roughly 2020-2022).

| Column | Type | Description |
|--------|------|-------------|
| `block_number` | int | Polygon block number |
| `transaction_hash` | string | Blockchain transaction hash |
| `log_index` | int | Log index within transaction |
| `fpmm_address` | string | FPMM contract (market) address |
| `trader` | string | Buyer or seller address |
| `amount` | string | Investment amount (buy) or return amount (sell) in collateral units (6 decimals for USDC) |
| `fee_amount` | string | Trading fee in collateral units |
| `outcome_index` | int | Index of the outcome traded (0 or 1) |
| `outcome_tokens` | string | Outcome tokens bought or sold (18 decimals) |
| `is_buy` | bool | True for buy, False for sell |
| `timestamp` | int (nullable) | Unix timestamp (if enriched) |
| `_fetched_at` | datetime | When this record was fetched |

**Note on legacy trade amounts:** The `amount`, `fee_amount`, and `outcome_tokens` fields are stored as strings to avoid integer overflow. Collateral amounts use 6 decimals (for USDC markets), while outcome tokens use 18 decimals.

## Polymarket FPMM Collateral Lookup

Located at `data/polymarket/fpmm_collateral_lookup.json`, this file maps FPMM contract addresses to their collateral token information. Used to filter legacy trades to only include USDC-collateralized markets.

```json
{
  "0x1234...": {
    "collateral_address": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
    "collateral_symbol": "USDC"
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `collateral_address` | string | ERC-20 token address used as collateral |
| `collateral_symbol` | string | Token symbol (e.g., `USDC`, `USDT`) |

## Polymarket Blocks

Mapping from Polygon block numbers to timestamps.

| Column | Type | Description |
|--------|------|-------------|
| `block_number` | int | Polygon block number |
| `timestamp` | string | ISO 8601 timestamp (e.g., `2024-01-15T12:30:00Z`) |

---

# Derived Data (Transforms)

Transform outputs are stored in `data/transforms/<name>/`. Each transform writes a `manifest.json` on completion; re-running skips transforms whose manifest already exists (use `--force` to override).

### Status

| Transform | Status | Notes |
|-----------|--------|-------|
| T1A | **Done** | 72M Kalshi + 405M Polymarket trades enriched |
| T1B | Not started | Depends on T1A |
| T2 | Not started | Depends on T1A, T1B |
| T3 | Not started | No dependencies |
| T4 | Partial | 19,621 implied surfaces written; temporal evolution + summary not completed |
| T5 | Not started | Depends on T1B |
| T6 | Not started | No dependencies |
| T7 | Not started | Depends on T3, T1B |
| T8 | Not started | Depends on T1A |

## T1A — Trade-Level Enrichment

**Path:** `data/transforms/t1a/{kalshi,polymarket}/*.parquet`

Enriches raw trades with sequential context using DuckDB window functions.

### Kalshi Output

| Column | Type | Description |
|--------|------|-------------|
| `trade_id` | string | Original trade ID |
| `ticker` | string | Market ticker |
| `event_ticker` | string | Parent event ticker (joined from markets) |
| `count` | int | Number of contracts |
| `yes_price` | int | Yes price (cents) |
| `no_price` | int | No price (cents) |
| `taker_side` | string | `yes` or `no` |
| `created_time` | datetime | Trade timestamp |
| `norm_price` | int | Normalized price (same as `yes_price` for Kalshi) |
| `signed_flow` | int | Positive = buy, negative = sell (`count` or `-count`) |
| `trade_sequence_num` | int | Sequential trade number per ticker |
| `delta_price` | int (nullable) | Price change from previous trade |
| `time_since_prev` | float (nullable) | Seconds since previous trade |
| `cumulative_volume` | int | Running total volume per ticker |
| `cumulative_net_flow` | int | Running net signed flow per ticker |
| `cumulative_trade_count` | int | Running trade count per ticker |
| `time_to_expiry_seconds` | float (nullable) | Seconds until market close |

### Polymarket Output

| Column | Type | Description |
|--------|------|-------------|
| `block_number` | int | Polygon block number |
| `transaction_hash` | string | Transaction hash |
| `log_index` | int | Log index |
| `order_hash` | string | Order hash |
| `maker` | string | Maker address |
| `taker` | string | Taker address |
| `maker_asset_id` | string | Maker asset ID |
| `taker_asset_id` | string | Taker asset ID |
| `maker_amount` | int | Maker amount |
| `taker_amount` | int | Taker amount |
| `fee` | int | Fee |
| `timestamp` | datetime | Trade timestamp (from blocks join) |
| `_contract` | string | Contract name |
| `token_id` | string | Conditional token asset ID |
| `norm_price` | float | Normalized price (0-1 scale) |
| `volume` | float | USDC volume of the trade |
| `signed_flow` | float | Signed order flow |
| `market_id` | string (nullable) | Polymarket market ID (from token map) |
| `trade_sequence_num` | int | Sequential trade number per token |
| `delta_price` | float (nullable) | Price change from previous trade |
| `time_since_prev` | float (nullable) | Seconds since previous trade |
| `cumulative_volume` | float | Running total volume per token |
| `cumulative_net_flow` | float | Running net signed flow per token |
| `cumulative_trade_count` | int | Running trade count per token |
| `time_to_expiry_seconds` | float (nullable) | Seconds until market end |

## T1B — Conditional OHLCV Bars

**Paths:**
- `data/transforms/t1b/qualifying_markets.parquet`
- `data/transforms/t1b/{kalshi,polymarket}/bars_{5min,1h,1d}.parquet/*.parquet`

Builds gap-filled OHLCV bars at 5-minute, 1-hour, and 1-day intervals for qualifying markets (trades_per_hour > 5 OR total_trades > 100).

### qualifying_markets.parquet

| Column | Type | Description |
|--------|------|-------------|
| `platform` | string | `kalshi` or `polymarket` |
| `market_id` | string | Market/token identifier |
| `total_trades` | int | Total trade count |
| `time_span_hours` | float | Duration of trading activity in hours |
| `trades_per_hour` | float | Average trades per hour |

### bars_{5min,1h,1d}.parquet

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | string | Market/token identifier |
| `bar_start` | datetime | Bar period start |
| `open` | float | Opening price |
| `high` | float | High price |
| `low` | float | Low price |
| `close` | float | Closing price |
| `volume` | float | Total volume in bar |
| `net_flow` | float | Net signed flow |
| `trade_count` | int | Number of trades |
| `vwap` | float (nullable) | Volume-weighted average price |
| `bar_return` | float (nullable) | Close-to-close return |
| `bar_range` | float | High minus low |

## T2 — Order Flow Imbalance & Price Impact

**Paths:**
- `data/transforms/t2/impact_curves.parquet`
- `data/transforms/t2/kalshi_ofi/*.parquet`
- `data/transforms/t2/kalshi_vpin/*.parquet`

### impact_curves.parquet

Category-pooled price impact regression results (Kalshi).

| Column | Type | Description |
|--------|------|-------------|
| `category` | string | Market category |
| `size_bucket` | string | Trade size bucket (`1`, `2-10`, `11-100`, `101-1K`, `1K+`) |
| `expiry_bucket` | string | Time-to-expiry bucket (`<1h`, `1-6h`, `6-24h`, `1-7d`, `7d+`) |
| `impact_coeff` | float | Regression slope of delta_price on signed_flow |
| `r_squared` | float | Regression R-squared |
| `sample_size` | int | Number of observations |
| `avg_abs_delta_price` | float | Average absolute price change |
| `avg_abs_flow` | float | Average absolute flow |

### kalshi_ofi (Rolling Order Flow Imbalance)

All T1A Kalshi columns plus:

| Column | Type | Description |
|--------|------|-------------|
| `ofi_20` | float | Rolling 20-trade order flow imbalance |
| `ofi_50` | float | Rolling 50-trade order flow imbalance |
| `ofi_100` | float | Rolling 100-trade order flow imbalance |

### kalshi_vpin (Volume-Synchronized PIN)

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | string | Market ticker |
| `volume_bucket` | int | Volume bucket index (each 50 contracts) |
| `buy_flow` | float | Buy-side flow in bucket |
| `sell_flow` | float | Sell-side flow in bucket |
| `bucket_volume` | int | Total volume in bucket |
| `vpin_20` | float | VPIN over trailing 20 buckets |

## T3 — Position Ledger (Polymarket)

**Paths:**
- `data/transforms/t3/position_ledger/` (Hive-partitioned by `address_prefix`)
- `data/transforms/t3/address_summary.parquet`

### position_ledger

| Column | Type | Description |
|--------|------|-------------|
| `address` | string | Wallet address |
| `token_id` | string | Conditional token ID |
| `direction` | string | `BUY` or `SELL` |
| `qty` | float | Signed token quantity |
| `cost_usdc` | float | Signed USDC cost |
| `block_number` | int | Block number |
| `log_index` | int | Log index |
| `trade_timestamp` | datetime | Trade timestamp |
| `cumulative_position` | float | Running token position |
| `cumulative_cost` | float | Running USDC cost |
| `cumulative_buy_qty` | float | Running buy-side quantity |
| `cumulative_buy_cost` | float | Running buy-side cost |
| `address_prefix` | string | First 2 hex chars (partition key) |

### address_summary.parquet

| Column | Type | Description |
|--------|------|-------------|
| `address` | string | Wallet address |
| `distinct_markets` | int | Number of distinct markets traded |
| `total_volume` | float | Total absolute USDC volume |
| `total_trades` | int | Total trade count |
| `realized_pnl` | float | Realized P&L on resolved markets |
| `win_rate` | float (nullable) | Fraction of profitable positions |
| `first_trade` | datetime | Earliest trade timestamp |
| `last_trade` | datetime | Latest trade timestamp |
| `active_days` | int | Approximate active trading days |
| `herfindahl` | float | Market concentration index |
| `avg_return_per_trade` | float | P&L per trade |
| `volume_per_market` | float | Average volume per market |

## T4 — Implied Probability Surfaces (Kalshi)

**Paths:**
- `data/transforms/t4/implied_surfaces/{event_ticker}.parquet`
- `data/transforms/t4/surface_evolution/{event_ticker}.parquet`
- `data/transforms/t4/summary_stats.parquet`

### implied_surfaces (per event)

| Column | Type | Description |
|--------|------|-------------|
| `event_ticker` | string | Event ticker |
| `ticker` | string | Contract ticker |
| `threshold` | float | Threshold value extracted from ticker |
| `last_price` | int | Last traded price (cents) |
| `cdf_raw` | float | Raw P(X > threshold), `last_price / 100` |
| `cdf_monotonic` | float | Monotonicity-enforced CDF |
| `pdf` | float | Implied PDF (differenced CDF) |
| `fitted_normal_cdf` | float | Normal distribution fit |
| `fitted_lognorm_cdf` | float | Lognormal distribution fit |
| `normal_mean` | float | Fitted normal mean |
| `normal_std` | float | Fitted normal standard deviation |
| `lognorm_s` | float | Fitted lognormal shape parameter |
| `lognorm_scale` | float | Fitted lognormal scale parameter |
| `ks_normal` | float | KS statistic for normal fit |
| `ks_lognorm` | float | KS statistic for lognormal fit |
| `residual_normal` | float | Residual from normal fit |
| `residual_lognorm` | float | Residual from lognormal fit |

### surface_evolution (per event)

| Column | Type | Description |
|--------|------|-------------|
| `event_ticker` | string | Event ticker |
| `trade_time` | datetime | Timestamp of triggering trade |
| `implied_mean` | float | Normal-fit implied mean |
| `implied_std` | float | Normal-fit implied std |
| `implied_skew` | float | Empirical skewness from discrete PDF |
| `implied_kurtosis` | float | Excess kurtosis from discrete PDF |
| `n_active_contracts` | int | Number of contracts in the family |

### summary_stats.parquet

| Column | Type | Description |
|--------|------|-------------|
| `event_ticker` | string | Event ticker |
| `n_contracts` | int | Number of threshold contracts |
| `threshold_range` | string | `min-max` threshold range |
| `best_fit_model` | string | `normal`, `lognormal`, or `none` |
| `best_ks` | float | Best-fit KS statistic |
| `implied_mean` | float | Normal-fit implied mean |
| `implied_std` | float | Normal-fit implied std |
| `is_resolved` | bool | Whether the event has resolved |
| `actual_outcome` | float (nullable) | Resolved outcome value |

## T5 — Cross-Market Lead-Lag Network

**Path:** `data/transforms/t5/lead_lag_summary.parquet`

Cross-correlation and Granger causality analysis between market pairs using 5-minute bars.

| Column | Type | Description |
|--------|------|-------------|
| `platform_a` | string | Platform of market A |
| `market_id_a` | string | Market A identifier |
| `platform_b` | string | Platform of market B |
| `market_id_b` | string | Market B identifier |
| `pair_type` | string | `intra_family`, `cross_platform`, or `intra_platform` |
| `peak_lag_bars` | int | Lag at peak cross-correlation (positive = A leads B) |
| `peak_lag_minutes` | int | Peak lag in minutes |
| `peak_correlation` | float | Cross-correlation at peak lag |
| `granger_a_to_b_f` | float (nullable) | Granger F-stat: A causes B |
| `granger_a_to_b_p` | float (nullable) | Granger p-value: A causes B |
| `granger_b_to_a_f` | float (nullable) | Granger F-stat: B causes A |
| `granger_b_to_a_p` | float (nullable) | Granger p-value: B causes A |
| `median_lag` | float | Median rolling window peak lag |
| `lag_std` | float | Standard deviation of rolling lags |
| `lag_consistency_pct` | float | Fraction of windows with consistent lag sign |
| `n_aligned_bars` | int | Number of aligned bar observations |
| `pct_overlap` | float | Minimum overlap percentage |

## T6 — Rolling Calibration Scores (Kalshi)

**Paths:**
- `data/transforms/t6/daily_features.parquet`
- `data/transforms/t6/regime_flags.parquet`
- `data/transforms/t6/resolution_lag.parquet`

### daily_features.parquet

| Column | Type | Description |
|--------|------|-------------|
| `trade_date` | datetime | Trading date |
| `category` | string | Market category |
| `price_bucket` | int | Price bucket (5-cent increments) |
| `tte_bucket` | string | Time-to-expiry bucket |
| `total_trades` | int | Trade count |
| `total_wins` | int | Winning trade count |
| `win_rate` | float | Empirical win rate |
| `implied_prob` | float | Implied probability from price |
| `calibration_error` | float | Absolute difference: win_rate - implied_prob |
| `yes_taker_ratio` | float | Fraction of yes-side taker trades |
| `volume` | int | Total volume |
| `mae_7d` | float | 7-day rolling weighted MAE |
| `mae_30d` | float | 30-day rolling weighted MAE |
| `mae_90d` | float | 90-day rolling weighted MAE |
| `opportunity_score` | float | MAE * log(volume) * directional bias |
| `yes_ratio_7d` | float | 7-day rolling yes-taker ratio |

### regime_flags.parquet

| Column | Type | Description |
|--------|------|-------------|
| `trade_date` | datetime | Trading date |
| `category` | string | Market category |
| `mae_7d` | float | 7-day rolling MAE |
| `mae_90d` | float | 90-day rolling MAE |
| `regime_flag` | int | `1` = short-term miscalibration up, `-1` = down, `0` = neutral |

### resolution_lag.parquet

| Column | Type | Description |
|--------|------|-------------|
| `category` | string | Market category |
| `median_resolution_lag_hours` | float | Median hours from trade to close |
| `p25_lag` | float | 25th percentile lag |
| `p75_lag` | float | 75th percentile lag |
| `total_resolved` | int | Total resolved trades in category |

## T7 — Address Interaction Graph (Polymarket)

**Paths:**
- `data/transforms/t7/structural/edge_list.parquet/*.parquet`
- `data/transforms/t7/structural/address_projection.parquet/*.parquet`
- `data/transforms/t7/structural/cluster_assignments.parquet`
- `data/transforms/t7/structural/cluster_performance.parquet`
- `data/transforms/t7/temporal/sybil_candidates.parquet`
- `data/transforms/t7/temporal/cascade_table.parquet`

### edge_list (bipartite)

| Column | Type | Description |
|--------|------|-------------|
| `address` | string | Wallet address |
| `token_id` | string | Conditional token ID |
| `direction` | string | `buy` or `sell` |
| `block_number` | int | Block number |
| `size` | float | USDC trade size |

### address_projection

| Column | Type | Description |
|--------|------|-------------|
| `address_a` | string | First address |
| `address_b` | string | Second address |
| `shared_markets` | int | Number of shared token markets |
| `interaction_count` | int | Total co-occurrences |

### cluster_assignments.parquet

| Column | Type | Description |
|--------|------|-------------|
| `address` | string | Wallet address |
| `cluster_id` | int | Louvain community ID |

### cluster_performance.parquet

| Column | Type | Description |
|--------|------|-------------|
| `cluster_id` | int | Cluster ID |
| `member_count` | int | Number of addresses in cluster |
| `avg_win_rate` | float | Average win rate of members |
| `total_volume` | float | Total USDC volume |
| `avg_realized_pnl` | float | Average realized P&L |

### sybil_candidates.parquet

| Column | Type | Description |
|--------|------|-------------|
| `cluster_id` | int | Cluster ID |
| `token_id` | string | Token market ID |
| `direction` | string | Trade direction |
| `distinct_addresses` | int | Number of coordinated addresses |
| `earliest_block` | int | First block in the burst |
| `latest_block` | int | Last block in the burst |
| `pair_count` | int | Number of address pairs detected |

### cascade_table.parquet

| Column | Type | Description |
|--------|------|-------------|
| `cluster_id` | int | Cluster ID |
| `token_id` | string | Token market ID |
| `leader_address` | string | Address that traded first |
| `leader_direction` | string | Leader trade direction |
| `leader_block` | int | Leader block number |
| `follower_address` | string | Address that followed |
| `follower_direction` | string | Follower trade direction |
| `follower_block` | int | Follower block number |
| `block_lag` | int | Blocks between leader and follower |
| `follower_size` | float | USDC size of follower trade |

## T8 — Market Lifecycle State Machine

**Paths:**
- `data/transforms/t8/kalshi/state_timeline.parquet/*.parquet`
- `data/transforms/t8/per_state_stats.parquet/*.parquet`
- `data/transforms/t8/anomalous_transitions.parquet/*.parquet`
- `data/transforms/t8/state_duration_distributions.parquet/*.parquet`

### state_timeline

Each trade is assigned a lifecycle state based on activity patterns.

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | string | Market ticker |
| `created_time` | datetime | Trade timestamp |
| `trade_sequence_num` | int | Sequential trade number |
| `norm_price` | int | Normalized price |
| `state` | string | Lifecycle state (see below) |
| `prev_state` | string (nullable) | Previous trade's state |
| `is_transition` | bool | Whether state changed from previous trade |

**Lifecycle states:** `NEWLY_LISTED`, `EARLY_TRADING`, `ACTIVE`, `HIGH_ACTIVITY`, `APPROACHING_EXPIRY`, `RESOLVING`, `DORMANT`, `SETTLED`

### per_state_stats

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | string | Market ticker |
| `state` | string | Lifecycle state |
| `trade_count` | int | Trades in this state |
| `avg_price` | float | Average price |
| `price_std` | float | Price standard deviation |
| `avg_signed_flow` | float | Average signed flow |
| `total_volume` | int | Total volume |
| `duration_seconds` | float | Time spent in state |
| `first_entry` | datetime | First entry into state |
| `last_entry` | datetime | Last entry into state |

### anomalous_transitions

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | string | Market ticker |
| `created_time` | datetime | Transition timestamp |
| `from_state` | string | Previous state |
| `to_state` | string | New state |
| `norm_price` | int | Price at transition |
| `trade_sequence_num` | int | Trade sequence number |
| `transition_type` | string | `information_event`, `bot_or_manipulation`, `sudden_resolution`, or `impossible_regression` |

### state_duration_distributions

| Column | Type | Description |
|--------|------|-------------|
| `state` | string | Lifecycle state |
| `p25_duration_hours` | float | 25th percentile duration |
| `median_duration_hours` | float | Median duration |
| `p75_duration_hours` | float | 75th percentile duration |
| `mean_duration_hours` | float | Mean duration |
| `count` | int | Number of state stints |
