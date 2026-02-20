# §8.0: Data Transformations Plan

Derived datasets and feature engineering to unlock more sophisticated analysis beyond the cross-sectional and aggregate approaches used in §1-7. Each transformation introduces **sequencing**, **state**, or **relationships** into the raw trade/market data.

Source data: Kalshi (7.3M markets, 72M trades) and Polymarket (409K markets, ~388M resolved trades).

## Data Completeness & Sparsity

**Trade data is complete on both platforms.** Kalshi trades are fetched via paginated API with cursor-based pagination that loops until exhaustion, deduplicated by `trade_id`. Polymarket trades are fetched directly from on-chain `OrderFilled` event logs on Polygon — immutable and complete by definition.

**Market data is point-in-time snapshots, not time series.** Both Kalshi and Polymarket market files capture the current state (bid/ask, volume, status) at the moment the indexer ran. There is no historical order book or continuous price record beyond what can be reconstructed from trades.

**Most markets are extremely sparse.** From §3.4: the median Kalshi market has ~16 lifetime trades. The distribution is:

| Regime | Markets | Trades |
|--------|---------|--------|
| Low (1 trade) | 202,173 | 202,173 |
| Medium (2-16 trades) | 167,681 | 1,065,569 |
| High (>16 trades) | 184,369 | 66,493,664 |

The top tercile accounts for 98.1% of all trades. Polymarket has similar concentration: Bot/HFT addresses (1% of addresses) produce 71.7% of volume. **Any transformation that requires dense per-market activity is only meaningful for a subset of ~5-10K liquid markets per platform.** Each transform below notes its sparsity sensitivity and mitigation.

---

## T1: Trade-Level Enrichment + Conditional OHLCV

**Status:** `[x]` Complete — Layer A completed 2026-02-13 (476.7M enriched trades), Layer B completed 2026-02-16 (1.6B gap-filled bars across 556K qualifying markets)

**What:** Two-layer transform. **Layer A** enriches every raw trade with derived sequential fields (works at any density). **Layer B** resamples into OHLCV bars only for markets exceeding a trade-density threshold.

**Sparsity sensitivity:** Layer A: **none** (per-trade fields are universal). Layer B: **high** (only meaningful for markets with sufficient trade density, roughly >5 trades/hour average).

**Input data:**
- Kalshi: `data/kalshi/trades/*.parquet` — fields: `ticker`, `yes_price`, `no_price`, `taker_side`, `count`, `created_time`
- Kalshi: `data/kalshi/markets/*.parquet` — fields: `ticker`, `event_ticker`, `close_time` (for time-to-expiry)
- Polymarket: `data/polymarket/trades/*.parquet` — fields: `maker_asset_id`, `taker_asset_id`, `maker_amount`, `taker_amount`, `timestamp`, `block_number`

**Transform steps — Layer A (all trades):**
- [x] Sort trades by timestamp per market ticker / token ID
- [x] Compute normalized trade price (Kalshi: `yes_price`; Polymarket: amount ratio based on `maker_asset_id == '0'` logic)
- [x] Compute signed order flow per trade (+contracts for YES-taker, -contracts for NO-taker)
- [x] Compute trade-to-trade fields: `delta_price` (price change from previous trade in same market), `time_since_prev` (seconds since previous trade), `trade_sequence_num` (1, 2, 3... within market)
- [x] Compute running cumulative fields: `cumulative_volume`, `cumulative_net_flow`, `cumulative_trade_count`
- [x] Compute time-to-expiry at trade time (join with market `close_time`)
- [x] Output: enriched trade parquet with all original fields + derived fields, one per platform

**Transform steps — Layer B (liquid markets only):**
- [x] Filter to markets with average trade density > threshold (>5 trades/hour or >100 total trades) — 556,278 qualifying markets
- [x] Resample into 5-min / 1-hour / daily bars with OHLCV + net flow + trade count aggregation
- [x] Handle gaps: forward-fill close price for empty bars, mark volume and trade count as 0
- [x] Compute per-bar derived fields: bar return (close-to-close), bar range (high-low), VWAP
- [x] Output: OHLCV parquet per platform per interval granularity, with a qualifying market manifest listing which tickers are included

**Unlocks:**
- Layer A: sequential context for every trade (foundation for T2), time-between-trades as a liquidity proxy, trade numbering for lifecycle analysis
- Layer B: realized volatility per market (Parkinson, Garman-Klass), momentum vs. mean-reversion detection, intraday seasonality per market (market-specific §3.1), VWAP execution benchmarks

**Depends on:** Nothing (foundational)

---

## T2: Order Flow Imbalance & Price Impact

**Status:** `[ ]` Not started

**What:** Build on T1 Layer A trade enrichment to compute flow imbalance metrics and empirical price impact curves. Per-trade fields are universal; rolling aggregations and VPIN are computed at two grains — per-market (liquid subset only) and pooled across category (all trades).

**Sparsity sensitivity:** **Moderate.** Per-trade fields (signed flow, delta_price) work universally. Rolling per-market metrics (OFI windows, VPIN) require trade density and are restricted to liquid markets. Category-pooled impact curves sidestep per-market sparsity entirely.

**Input data:**
- T1 Layer A output (enriched trades with `delta_price`, `signed_flow`, `time_since_prev`)
- Kalshi: `data/kalshi/markets/*.parquet` — fields: `event_ticker` (for category classification)

**Transform steps — per-trade (all trades):**
- [ ] Use `delta_price` and `signed_flow` from T1 Layer A (already computed)
- [ ] Tag each trade with category (reuse §1.3 event_ticker prefix mapping)

**Transform steps — category-pooled impact curves (all trades):**
- [ ] Pool all trades within each category
- [ ] Regress `delta_price` on `signed_flow`, stratified by trade size bucket (1, 2-10, 11-100, 101-1K, 1K+), category, and time-to-expiry bucket
- [ ] Output: impact curve coefficients per (category × size_bucket × expiry_bucket)

**Transform steps — per-market rolling metrics (liquid markets only):**
- [ ] Filter to markets in T1 Layer B qualifying manifest
- [ ] Build rolling order flow imbalance (OFI): cumulative signed flow over trailing N trades (e.g., 20, 50, 100)
- [ ] Compute VPIN: partition volume into fixed-size buckets, classify each bucket as buy/sell-initiated, compute rolling |buy_volume - sell_volume| / total_volume
- [ ] Output: per-trade enrichment with rolling_OFI and VPIN columns (for qualifying markets only)

**Unlocks:**
- Category-pooled impact curves: how much does a 100-contract Sports trade move price? Essential for execution modeling and position sizing across the full market universe
- Per-market VPIN: real-time toxicity metric to dynamically adjust the §7.1 maker strategy (widen spread when VPIN is high) — liquid markets only
- Per-market OFI: herding vs. mean-reversion detection — does positive flow predict further positive flow or a reversal?
- Distinguishing informed flow from noise in real time (upgrades §5.2 and §6.1)

**Depends on:** T1 Layer A

---

## T3: Per-Address Position Ledger & P&L Curves (Polymarket)

**Status:** `[x]` Complete — completed 2026-02-17 (809M ledger entries across 1.96M addresses, 591K tokens, $70B total volume)

**What:** For each Polymarket address, reconstruct running token holdings per market over time by accumulating buys and subtracting sells. Compute realized P&L on resolved markets and derive risk-adjusted performance metrics.

**Sparsity sensitivity:** **Low.** This is address-centric, not market-centric. An active address trades across many sparse markets, and the ledger works perfectly even with 1 trade per market. Realized P&L is binary (won or lost) and doesn't depend on intermediate price density. Mark-to-market P&L between trades uses stale prices on sparse markets, but this is a secondary output — realized P&L on resolved markets is the primary use case.

**Input data:**
- Polymarket: `data/polymarket/trades/*.parquet` — fields: `maker`, `taker`, `maker_asset_id`, `taker_asset_id`, `maker_amount`, `taker_amount`, `block_number`, `fee`
- Polymarket: `data/polymarket/markets/*.parquet` — fields: `clob_token_ids`, `outcome_prices`, `closed`
- Polymarket: `data/polymarket/blocks/*.parquet` — fields: `block_number`, `timestamp`

**Transform steps:**
- [x] For each trade, determine buyer/seller address and token acquired/sold (using `maker_asset_id == '0'` logic)
- [x] Build cumulative position per (address, token_id) ordered by block_number
- [x] Compute cost basis per position (average price paid)
- [x] Join with block timestamps to create time-indexed position snapshots
- [x] Compute realized P&L on resolved markets (position × (outcome - avg_cost))
- [ ] Compute mark-to-market P&L using last trade price (noting staleness caveat for sparse markets)
- [x] Derive per-address metrics: Sharpe ratio, max drawdown, win/loss streak, holding period distribution, concentration (Herfindahl index across markets)
- [x] Output: position ledger parquet (address, token_id, block_number, position, cost_basis, realized_pnl) + address summary stats

**Unlocks:**
- Accumulation/distribution patterns — is smart money building or exiting? (dynamic version of §5.1)
- Risk-adjusted address ranking (Sharpe, drawdown) instead of raw excess return — improves smart money identification and reduces the 85% regression-to-mean decay found in §5.1 temporal analysis
- Position sizing and concentration analysis — feasibility of copy-trading
- Holding period analysis per archetype — maps to different strategy horizons
- Entry/exit timing relative to market lifecycle (combines with §3.3)

**Depends on:** Nothing (can be built from raw trades)

---

## T4: Implied Probability Surfaces for Threshold Families

**Status:** `[ ]` Not started

**What:** For each threshold event family (KXBTCD, KXNASDAQ100U, KXETHD, KXINXU), extract the full set of threshold-level prices and fit a cumulative distribution function (CDF). Cross-sectional fits use market snapshot data. Temporal surface evolution is only attempted for the top 4 families where per-contract trade density is sufficient.

**Sparsity sensitivity:** **Moderate, but from a different angle.** The constraint is not trade frequency but the **market snapshot limitation** — market data is point-in-time, not a time series. Cross-sectional CDF fits at snapshot time work well because threshold families are among the most liquid on Kalshi (KXBTCD alone: 55,154 markets, 5,420 events). Reconstructing surface evolution over time from trades requires individual threshold contracts within an event to trade frequently enough; this is feasible for the top ~4 families (crypto/equity-index daily closes) but unreliable for less liquid families where the reconstructed surface would mix current and stale prices.

**Input data:**
- Kalshi: `data/kalshi/markets/*.parquet` — fields: `ticker`, `event_ticker`, `last_price`, `yes_bid`, `yes_ask`, `volume`, `close_time`
- Kalshi: `data/kalshi/trades/*.parquet` — for trade-time surface reconstruction (top 4 families only)

**Transform steps — cross-sectional (all threshold families):**
- [ ] Parse threshold family and numeric threshold from `ticker` / `event_ticker` (regex on the existing prefix extraction from §4.3)
- [ ] For each event (family + date), sort contracts by threshold value ascending
- [ ] Extract the implied CDF: price at each threshold = P(underlying > threshold)
- [ ] Enforce monotonicity via isotonic regression or pool-adjacent-violators algorithm to produce a "cleaned" CDF
- [ ] Compute the implied PDF by differencing adjacent CDF points
- [ ] Fit parametric distributions (normal, log-normal, mixture) and compute goodness-of-fit
- [ ] Compare implied vs. realized distributions (using actual settlement values) to identify systematic biases (tail weight, skew, kurtosis)
- [ ] Compute model-derived fair values for each individual contract — deviation from fitted CDF = mispricing signal with confidence intervals
- [ ] Output: per-event implied surface parquet (family, date, threshold, raw_price, fitted_price, residual, std_err) + summary distribution stats

**Transform steps — temporal surface evolution (KXBTCD, KXNASDAQ100U, KXETHD, KXINXU only):**
- [ ] For each event in the top 4 families, reconstruct the CDF at each trade time by combining the latest trade price for each threshold level
- [ ] Track implied mean, std, skew, kurtosis of the fitted distribution over time
- [ ] Measure surface convergence rate: how quickly do implied moments stabilize as expiry approaches?
- [ ] Output: time-series of implied distribution moments per event

**Unlocks:**
- Proper fair-value pricing for individual threshold contracts (upgrades crude monotonicity check in §4.3)
- Implied vs. realized distribution comparison — systematic biases in how the market models tails and skew
- Greeks-style sensitivities — how do threshold prices respond to underlying moves (essential for hedging)
- Volatility smile/smirk analysis — do Kalshi threshold markets exhibit the same patterns as options markets?
- Confidence-weighted mispricing signals (residual from fitted surface with standard errors)

**Depends on:** Nothing (snapshot fits are standalone; temporal evolution optionally uses T1 Layer A for trade timestamps)

---

## T5: Cross-Market Lead-Lag Network

**Status:** `[ ]` Not started

**What:** For the small subset of simultaneously highly-liquid markets, compute rolling cross-correlations at various lags to identify which markets update first when common information arrives.

**Sparsity sensitivity:** **Severe.** Lead-lag analysis requires two markets to be simultaneously densely traded. The sparsity problem compounds: if market A has 20 trades/day and market B has 15, the number of overlapping windows where both have fresh prices is much smaller than either individually. Cross-platform lead-lag is doubly constrained — §4.1 found only 178 matched pairs with r=0.988 agreement, and the liquid ones where lag is potentially measurable are the ones where the lag is probably too small to trade. **Scope this as a focused case study on ~20 liquid market pairs, not a broad network build.**

**Input data:**
- T1 Layer B output (OHLCV for liquid markets) for both platforms
- Kalshi: `data/kalshi/markets/*.parquet` — fields: `event_ticker`, `title`, `ticker`
- Polymarket: `data/polymarket/markets/*.parquet` — fields: `question`, `clob_token_ids`, `slug`
- Cross-platform matched pairs from §4.1

**Transform steps:**
- [ ] Curate a focused list of ~20 market pairs: (a) highest-volume cross-platform pairs from §4.1, (b) intra-Kalshi pairs within the same event family (e.g., related threshold contracts), (c) thematically linked markets identified by keyword overlap
- [ ] Align time series from T1 Layer B output to common clock (resample to 5-min bars; discard pairs where either side has >50% empty bars)
- [ ] Compute rolling cross-correlation at lags from -60 min to +60 min for each pair
- [ ] Identify statistically significant lead-lag relationships (Granger causality test)
- [ ] For pairs with significant lead-lag: measure lag stability over time and estimate tradable window (lag minus execution time)
- [ ] Output: lead-lag summary table (leader, follower, median_lag_minutes, correlation, stability_score) for the curated pairs

**Unlocks (limited scope):**
- Case-study evidence on whether cross-platform price discovery is simultaneous or has exploitable timing gaps
- Intra-family lead-lag within threshold events (does the at-the-money contract lead the wings?)
- Foundation for deciding whether to invest in real-time cross-market monitoring infrastructure

**Depends on:** T1 Layer B

---

## T6: Rolling Calibration Scores as Real-Time Features

**Status:** `[x]` Complete — completed 2026-02-19 (1,216,883 daily feature rows, regime flags, resolution lag stats)

**What:** Compute rolling N-day calibration MAE per `(category, price_bucket, time-to-expiry bucket, taker_side)` cell. Feature dates are indexed by market resolution day (`close_time`) to keep labels causal. Store as time-indexed features suitable for use as inputs to a trading model.

**Sparsity sensitivity:** **Moderate.** The transform runs at cell-level granularity (not just category-level), so sparse cells can be noisy. The real structural constraint remains **resolution lag**: calibration can only be computed from trades on already-resolved markets. Resolution-day indexing avoids look-ahead but still overweights shorter-duration markets that resolve quickly.

**Input data:**
- Kalshi: `data/kalshi/trades/*.parquet` joined with `data/kalshi/markets/*.parquet` (same join as §1.1–1.4)

**Transform steps:**
- [x] For each resolved trade, record (trade_timestamp, resolution_timestamp, category, price_bucket, taker_side, won)
- [x] Compute rolling 7-day, 30-day, 90-day calibration MAE per `(category, price_bucket, tte_bucket, taker_side)` cell, indexed by resolution day (`close_time`) to avoid look-ahead bias
- [x] Compute rolling MAE and opportunity signals at full cell granularity (not category-only pooling)
- [x] Compute rolling YES-share signal (`yes_ratio_7d`) within each cell
- [ ] Compute rolling maker/taker excess return gap per category (from §2.1)
- [x] Build composite "opportunity score" = `mae_7d * log1p(volume_7d) * (1 + abs(yes_ratio_7d - 0.5))` per cell
- [x] Flag regime transitions when `mae_7d` crosses `mae_90d` per cell
- [x] Document resolution lag bias: report median resolution delay per category and the effective "freshness" of each rolling window
- [x] Output: daily feature table (`trade_date`, `category`, `price_bucket`, `tte_bucket`, `taker_side`, `mae_7d`, `mae_30d`, `mae_90d`, `opportunity_score`) plus per-cell regime flags and per-category resolution-lag summary

**Unlocks:**
- Real-time regime detection (addresses the stationarity limitations flagged in nearly every report)
- Dynamic category rotation — re-weight which categories to trade as their calibration quality shifts (§1.3 temporal showed rankings change year-to-year)
- Model features for any ML-based trading strategy
- Automated scaling: increase position sizes when MAE is elevated, reduce when markets are efficient
- Early warning for calibration shocks (like the 2024 election event in §1.4)

**Depends on:** Nothing (core rolling MAE uses only trades + resolved markets). Volume regime classification optionally benefits from T1 Layer A cumulative volume fields.

---

## T7: Address Interaction Graph (Polymarket)

**Status:** `[ ]` Not started

**What:** Build a bipartite graph of addresses and markets, project onto an address-address co-trading graph, and apply community detection. Structural analysis (who trades what) uses all data. Temporal analysis (Sybil detection, information cascades) is restricted to the dense-market subset.

**Sparsity sensitivity:** **Mixed.** Graph structure (market co-occurrence between addresses) is robust — it's based on which markets addresses share, not timing. Community detection on the projected graph works regardless of per-market trade density. Temporal patterns degrade: Sybil detection (same direction within seconds) and cascade analysis (A trades then B follows within minutes) need dense timestamps to distinguish signal from coincidence. On a market with 3 trades/day, two addresses trading the same day is noise. Run temporal analyses only on markets in the T1 Layer B qualifying set.

**Input data:**
- Polymarket: `data/polymarket/trades/*.parquet` — fields: `maker`, `taker`, `maker_asset_id`, `taker_asset_id`, `block_number`
- Polymarket: `data/polymarket/markets/*.parquet` — fields: `clob_token_ids` (to map token IDs to markets)
- T3 output (position ledger) for address-level features
- T1 Layer B qualifying market manifest (for temporal analysis subset)

**Transform steps — structural graph (all markets):**
- [ ] Build bipartite edge list: (address, market_id, direction, block_number, size)
- [ ] Project onto address-address graph: two addresses share an edge if they co-traded N+ markets, weighted by co-occurrence count (no timing constraint)
- [ ] Apply community detection (Louvain, Label Propagation, or spectral clustering) to find address clusters
- [ ] For each cluster, compute aggregate performance metrics using T3 output (cluster-level excess return, win rate, volume, Sharpe)
- [ ] Cross-reference clusters with §5.3 archetypes (are members of a single cluster classified as different archetypes individually?)
- [ ] Output: address cluster assignments (address, cluster_id, cluster_size, cluster_excess_return) + graph edge list

**Transform steps — temporal patterns (liquid markets only):**
- [ ] Filter trades to markets in T1 Layer B qualifying manifest
- [ ] Identify likely Sybil sets: clusters where addresses always trade the same markets in the same direction within a tight block window (e.g., <10 blocks / ~20 seconds)
- [ ] Build information cascade sequences: when address A trades market X, do addresses in the same cluster follow within N blocks? Compute "follower" scores and median follow-lag
- [ ] Output: Sybil candidate list (cluster_id, confidence_score, co-occurrence_rate) + cascade sequence table

**Unlocks:**
- Structural: entity-level smart money scoring (cluster performance may be more stable than individual, addressing 85% decay in §5.1), improved archetype classification, network-based copy-trading signals
- Temporal: Sybil detection, information cascade monitoring as a leading indicator

**Depends on:** T3 (for cluster performance enrichment). T1 Layer B (for temporal analysis market filter). Core structural graph can be built from raw trades alone.

---

## T8: Market Lifecycle State Machine

**Status:** `[x]` Complete — completed 2026-02-17 (72.1M timeline rows, 611K per-state stats, 587 anomalous transitions, 8 state duration distributions)

**What:** For each market, define lifecycle states based on **observable** features (not future-looking duration), with transition rules that can be evaluated in real time. Only meaningful for markets with enough trade density to distinguish states — sparse markets will collapse to `NEWLY_LISTED → SETTLED` with no intermediate transitions, which is itself useful as a filter (markets that never achieve `ACTIVE` state are not tradable).

**Sparsity sensitivity:** **Moderate, but self-selecting.** State transitions require observable features (trade arrival rate, price level, price volatility) that are noise on sparse markets. A market with 5 lifetime trades cannot sustain state classification beyond `NEWLY_LISTED` and `SETTLED`. However, this is feature not bug: the state machine doubles as a **tradability filter** — any market that can't trigger an `ACTIVE` state transition is by definition too illiquid for the §7.1 maker strategy. The ~184K high-volume markets are the actionable universe.

**Input data:**
- T1 Layer A output (enriched trades with `trade_sequence_num`, `time_since_prev`, `cumulative_volume`)
- Kalshi: `data/kalshi/markets/*.parquet` — fields: `open_time`, `close_time`, `status`

**Transform steps:**
- [x] Define state labels and transition triggers based on observable features:
  - `NEWLY_LISTED`: < N trades AND < T hours since `open_time`
  - `EARLY_TRADING`: cumulative volume below threshold, `time_since_prev` irregular (long gaps)
  - `ACTIVE`: sustained trade arrival rate above threshold (e.g., >1 trade/hour over trailing 6h), cumulative volume above minimum
  - `HIGH_ACTIVITY`: trade arrival rate > 5x the `ACTIVE` threshold (volume spike regime)
  - `APPROACHING_EXPIRY`: < 24h to `close_time`, increasing trade velocity
  - `RESOLVING`: price moved to extreme (>90c or <10c) with accelerating volume
  - `SETTLED`: `status = 'finalized'`
  - `DORMANT`: no trades for > 24h after previously reaching `EARLY_TRADING` or `ACTIVE` (market went quiet — not tradable)
- [x] Compute state at each trade using T1 Layer A fields + time-to-close
- [x] Record state transitions with timestamps and associated features at transition moment
- [ ] Compute per-state calibration metrics (MAE, excess return) — real-time version of §3.3 lifecycle analysis using only observable state, not future-looking percentage
- [x] Identify anomalous transitions (skip from `EARLY_TRADING` to `RESOLVING` = information event)
- [x] Compute state duration distributions: how long does a typical market spend in each state? Stratify by category.
- [x] Output: per-market state timeline (ticker, trade_sequence_num, timestamp, state, features_at_transition) + per-state aggregate stats + anomalous transition log

**Unlocks:**
- Real-time, tradable version of §3.3 (lifecycle inefficiency) without look-ahead bias
- State-conditional strategy parameters — spread width, directional tilt, and position sizing per state for §7.1 market making
- Tradability filter: markets that never reach `ACTIVE` are automatically excluded from the trading universe
- Information event detection via anomalous state transitions
- Market screening: filter for markets in the most profitable state for each strategy type
- Backtest-safe features (no future information leakage)

**Depends on:** T1 Layer A (enriched trades with sequential fields)

---

## Dependency Graph

```
T1 Layer A (trade enrichment) ──┬──> T2 (Order Flow / Price Impact)
                                ├──> T8 (Lifecycle State Machine)
                                │
T1 Layer B (OHLCV, liquid only) ┼──> T5 (Lead-Lag, ~20 pairs)
                                └──> T7 temporal analysis filter

T3 (Position Ledger) ──> T7 (Address Graph, structural enrichment)

T4 (Implied Surfaces) ──> standalone (snapshot fits)
                      ──> temporal evolution uses T1 Layer A optionally

T6 (Rolling Calibration) ──> standalone (core uses only trades + resolved markets)
```

## Sparsity Impact Summary

| Transform | Sparsity Impact | Mitigation |
|-----------|----------------|------------|
| **T1** Trade Enrichment + OHLCV | Layer A: none. Layer B: high | Split into universal per-trade layer + conditional OHLCV for liquid subset |
| **T2** Order Flow / Price Impact | Moderate | Per-trade fields universal; pool impact curves across category, not per-market |
| **T3** Position Ledger | **Low** | Address-centric aggregation across many markets; binary resolution sidesteps intermediate price staleness |
| **T4** Implied Surfaces | Moderate | Cross-sectional fits work well; temporal evolution restricted to top 4 liquid families |
| **T5** Lead-Lag Network | **Severe** | Scoped to ~20 curated liquid pairs as a case study, not a broad network |
| **T6** Rolling Calibration | **Moderate** | Cell-level rolling (category × price × tte × side) is richer but noisier; resolution lag remains the key structural constraint |
| **T7** Address Graph | Mixed | Structural graph (co-occurrence) robust; temporal patterns (Sybil, cascades) restricted to liquid markets |
| **T8** Lifecycle States | Moderate, self-selecting | Sparse markets collapse to trivial state paths, which doubles as a tradability filter |

## Priority Order

| Priority | Transform | Status | Rationale |
|----------|-----------|--------|-----------|
| 1 | **T1** Trade Enrichment + OHLCV | ✅ Done | Foundational — Layer A feeds T2 and T8; Layer B feeds T5 and T7 temporal |
| 2 | **T3** Position Ledger | ✅ Done | Upgrades the entire Family 5 smart money analysis from static to dynamic; naturally sparsity-robust |
| 3 | **T6** Rolling Calibration | ✅ Done | Immediately actionable feature surface with causal indexing; strongest when paired with minimum-cell-liquidity filters |
| 4 | **T2** Order Flow / Price Impact | ❌ Not started | Highest value for live trading — enables real-time toxicity and execution modeling |
| 5 | **T4** Implied Surfaces | ❌ Not started | Largest analytical upgrade for threshold arbitrage in §4.3 and §7.3 |
| 6 | **T8** Lifecycle State Machine | ✅ Done | Makes §3.3 findings tradable without look-ahead; doubles as tradability filter |
| 7 | **T7** Address Graph | ❌ Not started | Most novel analysis; addresses smart money decay from §5.1; structural graph is cheap, temporal is expensive |
| 8 | **T5** Lead-Lag Network | ❌ Not started | Highest ceiling but severely sparsity-constrained; scoped as focused case study |
