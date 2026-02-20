# Roan Parity: Closing the Gap to Institutional-Grade Prediction Market Trading

Reference: [Roan's post](https://x.com/RohOnChain) — "How To Use Prediction Market Data Like Hedge Funds"

This document maps our current state against the three institutional methods Roan describes, identifies every gap, and defines the work required to reach parity.

---

## Current State Summary

### What We Have

| Asset | Status | Scale |
|---|---|---|
| T1A trade enrichment | Done | 476.7M rows (72.1M Kalshi + 404.5M Polymarket) |
| T1B OHLCV bars | Done | 1.6B bars across 556K qualifying markets |
| T3 position ledger | Done | 809M ledger entries, 1.96M address summaries |
| T4 implied surfaces | Partial | 19,621 per-event files (stale layout, needs re-run) |
| T6 rolling calibration | Done | 1.22M daily feature rows with regime flags |
| T8 lifecycle states | Done | 72.1M timeline rows, 587 anomalous transitions |
| T2 order flow | Code complete, not run | Impact curves, OFI, VPIN |
| T5 lead-lag | Code complete, not run | Cross-correlation, Granger causality |
| T7 address graph | Code complete, not run | Louvain communities, sybil/cascade detection |
| 27 strategy reports | Complete | Quantified edges across 7 families |
| Signal analyses (T1A, T1B) | Run | OFI, trade bursts, VWAP-close, volume surge |

### What We Don't Have

- No backtest framework
- No Monte Carlo simulation engine
- No Kelly criterion / position sizing
- No drawdown or risk management code
- No fee-adjusted strategy evaluation
- No execution modeling (fill rates, slippage, market impact)
- No portfolio construction across strategies
- Three transforms unexecuted (T2, T5, T7)
- T4 needs re-run for consolidated output

---

## Method 1: Empirical Kelly with Monte Carlo Uncertainty Quantification

### What Roan Describes

Five-phase pipeline: (1) extract historical trades matching a strategy pattern, (2) build empirical return distribution from resolved outcomes, (3) Monte Carlo resample 10K paths, (4) analyze drawdown distribution at percentiles, (5) size positions using `f_empirical = f_kelly * (1 - CV_edge)`.

### What We Have

| Roan Phase | Our Coverage | Gap |
|---|---|---|
| Phase 1: Historical trade extraction | T1A enrichment (476.7M trades with resolution outcomes) | **Covered** — raw data exists |
| Phase 1: Strategy pattern filtering | T6 daily features bucketed by (category, price_bucket, tte_bucket, taker_side), keyed on resolution date | **Covered** — can filter to any strategy definition |
| Phase 2: Return distribution | Win rate by price (reports 1.1-1.4), excess return per bucket | **Partial** — we have aggregate win rates, not per-trade return sequences |
| Phase 3: Monte Carlo resampling | Nothing | **Missing entirely** |
| Phase 4: Drawdown distribution | Nothing | **Missing entirely** |
| Phase 5: Uncertainty-adjusted sizing | Nothing | **Missing entirely** |

### What Needs to Be Built

#### TODO 1.1: Per-Strategy Return Sequence Extraction
- [ ] Define strategy filter criteria as composable SQL predicates (e.g., `price < 15 AND tte_bucket = '1-7d' AND category IN ('Sports', 'Crypto')`)
- [ ] For each strategy, extract the ordered sequence of per-trade returns: `(entry_price, resolution_outcome, realized_return, timestamp)`
- [ ] Output: one Parquet file per strategy definition containing the full return time series
- [ ] Source data: T1A enriched trades joined with market resolutions
- [ ] Must preserve temporal ordering (critical for path dependency analysis)

#### TODO 1.2: Empirical Return Distribution Builder
- [ ] For each strategy's return sequence, compute the full empirical distribution
- [ ] Measure: mean, median, std, skewness, kurtosis, percentiles (1st, 5th, 25th, 50th, 75th, 95th, 99th)
- [ ] Test for normality (Shapiro-Wilk, Jarque-Bera) — Roan emphasizes these are almost never normal
- [ ] Fit candidate distributions (empirical, Student-t, stable) for comparison
- [ ] Visualize: histogram with fitted overlays, QQ plots
- [ ] Output: distribution summary Parquet + diagnostic figures

#### TODO 1.3: Monte Carlo Resampling Engine
- [ ] Implement bootstrap resampling of the empirical return sequence
- [ ] Generate 10,000 alternative equity paths by randomly reordering returns
- [ ] For each path, compute: cumulative return, max drawdown (peak-to-trough), max run-up, Sharpe ratio, time-to-recovery
- [ ] Preserve the same statistical properties (mean, variance) while varying path dependency
- [ ] Output: `mc_paths.parquet` (10K rows x path metrics) per strategy
- [ ] Consider block bootstrap (preserve short-range autocorrelation) as sensitivity check

#### TODO 1.4: Drawdown Distribution Analysis
- [ ] From the 10K simulated paths, extract the distribution of max drawdowns
- [ ] Report: median, 75th, 90th, 95th, 99th percentile drawdowns
- [ ] Compare to the single historical backtest drawdown (which is just one path realization)
- [ ] Visualize: drawdown CDF, fan chart of equity curves (5th-95th percentile band)
- [ ] Compute: probability of exceeding N% drawdown for N in {10, 20, 30, 40, 50}

#### TODO 1.5: Empirical Kelly Position Sizer
- [ ] Implement standard Kelly: `f* = (p * b - q) / b` using empirical win rate and payoff ratio
- [ ] Implement uncertainty-adjusted Kelly: `f_empirical = f_kelly * (1 - CV_edge)`
  - `CV_edge = std(edge_estimates) / mean(edge_estimates)` across Monte Carlo simulations
- [ ] Implement fractional Kelly variants: half-Kelly, quarter-Kelly as conservative alternatives
- [ ] For each strategy, output: recommended position size at 50th/75th/95th percentile confidence
- [ ] Constraint: 95th percentile drawdown must stay under configurable limit (default 20%)
- [ ] Output: sizing table per strategy with confidence intervals

#### TODO 1.6: Multi-Strategy Portfolio Kelly
- [ ] Extend single-strategy Kelly to portfolio of concurrent strategies
- [ ] Account for correlation between strategy returns (covariance matrix from Monte Carlo)
- [ ] Implement capital allocation across strategies that maximizes growth rate subject to portfolio-level drawdown constraints
- [ ] This is the "Kelly criterion for portfolios" (Thorp/Vince extension)

---

## Method 2: Calibration Surface Analysis (Price x Time)

### What Roan Describes

Build a 2D calibration function C(p, t) mapping (contract price, time-to-resolution) to empirical win probability. Identify systematic mispricing M(p, t) = C(p, t) - p/100. Trade when |M(p,t)| exceeds a threshold calibrated to transaction costs.

### What We Have

| Roan Component | Our Coverage | Gap |
|---|---|---|
| 1D calibration (price only) | Reports 1.1-1.4: longshot bias -1.19pp (contract-weighted), fav-longshot asymmetry, by-category | **Done thoroughly** |
| Time dimension: calibration vs TTE | Report 3.2: MAE drops 3.2x (4.46pp at 7d+ to 1.41pp final hour) | **Done** |
| Early market inefficiency | Report 3.3: peak at 15-30% lifecycle (3.8pp MAE) | **Done** |
| Full 2D surface C(p, t) + side | T6: daily features by (category, price_bucket, tte_bucket, taker_side) with mae_7d/30d/90d | **Done — this is the live surface input** |
| Lifecycle state conditioning | T8: 8 lifecycle states with per-state statistics | **Done** |
| Opportunity scoring | T6: `opportunity_score = mae_7d * log1p(volume_7d) * (1 + abs(yes_ratio_7d - 0.5))` | **Done** |
| Regime detection | T6: regime flags when 7d MAE crosses 90d MAE | **Done** |
| Mispricing function M(p,t) | T6 calibration_error column = `abs(win_rate - implied_prob)` per bucket | **Done** |
| Threshold-based entry rules | Not implemented | **Missing** |
| Fee-adjusted profitability | Not implemented | **Missing** |
| Entry/exit rule backtesting | Not implemented | **Missing** |

### What Needs to Be Built

#### TODO 2.1: Fee Model Implementation
- [ ] Implement Kalshi fee schedule: taker fees by price level, maker rebates
- [ ] Implement Polymarket fee schedule: per-market fee rates from the CLOB API
- [ ] For each (price_bucket, tte_bucket) cell in the calibration surface, compute the net edge after fees
- [ ] Determine the minimum raw edge required to be profitable after fees at each price level
- [ ] Output: fee-adjusted calibration surface with cells flagged as tradeable/untradeable

#### TODO 2.2: Threshold Calibration for Entry Rules
- [ ] Using the T6 mispricing data, define entry rules: enter when |calibration_error| > threshold
- [ ] Sweep thresholds from 0.5pp to 10pp in 0.5pp increments
- [ ] For each threshold, compute: number of qualifying trades, win rate, avg return, Sharpe, max drawdown
- [ ] Find optimal threshold that maximizes risk-adjusted return after fees
- [ ] Segment by (category, tte_bucket) — the optimal threshold likely varies
- [ ] Output: threshold optimization results per segment

#### TODO 2.3: Time-Varying Filter Rules Backtest
- [ ] Implement Roan's three-regime strategy:
  - Far from resolution (7d+): fade longshots aggressively (strongest bias per our 3.2 data)
  - Mid-range (1-7d): reduce activity or switch to microstructure edge
  - Near resolution (<24h): exploit remaining mispricing with smaller size
- [ ] Backtest against historical data: simulate entering/exiting per these rules
- [ ] Compare to static (non-time-varying) calibration strategy
- [ ] Measure improvement from adding the time dimension
- [ ] Output: backtest equity curve, trade log, performance comparison

#### TODO 2.4: Regime-Conditional Strategy Adjustment
- [ ] Use T6 regime flags (mae_7d crossing mae_90d) to scale position sizes
- [ ] When regime_flag = +1 (deteriorating calibration): increase allocation
- [ ] When regime_flag = -1 (improving calibration): reduce allocation
- [ ] Backtest regime-conditional sizing vs static sizing
- [ ] Validate that regime transitions have predictive power for forward returns

---

## Method 3: Order Flow Decomposition and Maker vs Taker

### What Roan Describes

Separate every trade into maker/taker. Analyze profitability by role. Build a market-making framework: quote two-sided, allow small retail fills (biased), flag large fills (potentially informed), monitor inventory, hedge.

### What We Have

| Roan Component | Our Coverage | Gap |
|---|---|---|
| Maker vs taker profitability | Report 2.1: +1.12pp maker, -1.12pp taker, 80/99 price levels | **Done** |
| Maker YES vs NO | Report 2.3: NO +1.28pp vs YES +0.77pp | **Done** |
| Structural vs informational proof | Report 2.5: edge exists against all trade sizes, Cohen's d ~ 0.02 | **Done** |
| Taker behavioral bias | Report 6.3: YES bias 2.23:1, Report 1.2: taker-side asymmetry | **Done** |
| By-category maker edge | Report 2.2: Entertainment +2.06pp, Politics +0.11pp | **Done** |
| OFI (Order Flow Imbalance) | T2 code complete: rolling OFI at 20/50/100 windows | **Code done, not run** |
| VPIN (Volume-sync. Prob. Informed Trading) | T2 code complete: 50-contract buckets, 20-bucket window | **Code done, not run** |
| Price impact curves | T2 code complete: regression by category/size/expiry | **Code done, not run** |
| Adverse selection detection | Report 5.2 whale detection: whales lose -2.75pp on longshots | **Measured, not operationalized** |
| Market-making framework | Report 7.1 filtered market making: +2.0-2.5pp estimated | **Strategy designed, not backtested** |
| Inventory risk modeling | Nothing | **Missing** |
| Fill rate / execution modeling | Nothing | **Missing** |

### What Needs to Be Built

#### TODO 3.1: Execute T2 Order Flow Transform
- [ ] Run `make transform t2` (dependencies T1A + T1B both satisfied)
- [ ] Validate impact_curves.parquet output: check coefficient signs, R-squared values, sample sizes
- [ ] Validate OFI output: spot-check rolling windows, ensure no look-ahead bias
- [ ] Validate VPIN output: check distribution of VPIN values, sensibility at extremes
- [ ] Estimated runtime: unknown (first run); budget 2-4 hours based on T1B precedent

#### TODO 3.2: Adverse Selection Signal from T2
- [ ] Using impact curves: identify which (category, size_bucket, expiry_bucket) combinations show statistically significant price impact (informed flow)
- [ ] Using VPIN: define VPIN threshold above which to reduce or withdraw quotes (high toxicity)
- [ ] Using OFI: test whether OFI extremes predict adverse price moves
- [ ] Combine into a composite "toxicity score" per market-moment
- [ ] Output: toxicity_score timeseries per market, threshold recommendations

#### TODO 3.3: Market-Making Backtest
- [ ] Implement a basic market-making simulator:
  - Quote both sides at configurable spread around midpoint
  - Fill simulation: use T1B volume data to estimate fill probability
  - Track inventory accumulation per market
  - Apply T6 regime flags and T8 lifecycle states as filters
- [ ] Apply the 7.1 filtered market-making strategy:
  - Category gate: Sports, Crypto, Entertainment, Weather, Science/Tech only
  - Directional tilt: tighter quotes on NO side
  - Time window: 12:00-19:00 ET only
  - Price range: 30-60c focus
- [ ] Measure: gross edge, net edge (after fees), fill rate, inventory turnover, max inventory exposure, drawdown
- [ ] Output: backtest equity curve, fill log, inventory timeseries

#### TODO 3.4: Inventory Risk Model
- [ ] For the market-making strategy, model inventory risk:
  - Track net directional exposure across all open positions
  - Compute inventory half-life (how quickly positions resolve)
  - Measure correlation of inventory across markets (concentration risk)
- [ ] Define inventory limits: max exposure per market, per category, portfolio-wide
- [ ] Implement hedging rules: when inventory exceeds threshold, widen spread or cross to flatten
- [ ] Test inventory-constrained market making vs unconstrained

---

## Prerequisite: Complete Unfinished Transforms

Three transforms are fully implemented but have never been executed. They provide foundational data for the methods above.

### TODO 0.1: Execute T4 Implied Surfaces (re-run)
- [ ] Run `make transform t4 --force` to regenerate consolidated output
- [ ] Old per-event file layout exists but is stale; force flag will clean up
- [ ] Validate: `surfaces.parquet`, `evolution.parquet`, `summary_stats.parquet` all produced
- [ ] Verify manifest.json written
- [ ] No dependencies (standalone)

### TODO 0.2: Execute T2 Order Flow
- [ ] Run `make transform t2`
- [ ] Dependencies satisfied: T1A (done) + T1B (done)
- [ ] Validate three outputs: `impact_curves.parquet`, `kalshi_ofi/`, `kalshi_vpin/`
- [ ] Verify manifest.json written
- [ ] Highest priority unrun transform — directly feeds Method 3

### TODO 0.3: Execute T5 Lead-Lag
- [ ] Run `make transform t5`
- [ ] Dependency satisfied: T1B (done)
- [ ] Validate outputs: pair-level cross-correlation and Granger causality results
- [ ] Lower priority for Roan parity but valuable for cross-market strategies (4.1, 4.3)

### TODO 0.4: Execute T7 Address Graph
- [ ] Run `make transform t7`
- [ ] Dependencies satisfied: T3 (done) + T1B (done)
- [ ] Validate: `edge_list.parquet`, `cluster_assignments.parquet`, `cluster_performance.parquet`, `sybil_candidates.parquet`, `cascade_table.parquet`
- [ ] Feeds smart money following (5.1) and address clustering (5.3) strategies
- [ ] Requires `python-igraph` (already in pyproject.toml)

### TODO 0.5: Update SCHEMAS.md
- [ ] SCHEMAS.md (line 147-158) is out of date — lists T1B, T8 as "Not started" when both are complete
- [ ] Add T2, T5, T7 output schemas once they've been run
- [ ] Update T4 schema to reflect consolidated output format

---

## New Infrastructure: Backtest Framework

None of the three methods can be validated without a backtest engine. This is the single largest gap.

### TODO 4.1: Backtest Engine Core
- [ ] Design a generic event-driven backtester that:
  - Replays T1A enriched trades in chronological order
  - Maintains simulated portfolio state (positions, cash, PnL)
  - Accepts pluggable strategy objects with a standard interface
  - Tracks fills, slippage, and fees per trade
- [ ] Strategy interface: `on_tick(event) -> list[Order]`
- [ ] Order types: market (cross spread), limit (post and wait)
- [ ] Fill simulation: use T1B volume data for fill probability, T2 impact curves for slippage
- [ ] Output per run: trade log, equity curve, position timeseries, performance summary

### TODO 4.2: Strategy Implementations for Backtest
- [ ] **Fade-Longshot Strategy** (Method 2 baseline): sell contracts priced <15c, use T6 calibration surface for entry timing
- [ ] **Filtered Market-Making Strategy** (Method 3): implement 7.1 with category/direction/time/price filters
- [ ] **Calibration Surface Strategy** (Method 2 advanced): enter when |M(p,t)| > threshold, size by confidence
- [ ] **Composite Strategy**: combine calibration entry with maker execution and Kelly sizing

### TODO 4.3: Performance Analytics
- [ ] Sharpe ratio (annualized), Sortino ratio
- [ ] Max drawdown, max drawdown duration, time to recovery
- [ ] Win rate, profit factor, average win/loss ratio
- [ ] Trade frequency, avg holding period
- [ ] Rolling 30d/90d performance windows
- [ ] Comparison table across strategies

---

## Execution Priority

### Phase 1: Foundation (unblock everything else)
1. **TODO 0.2** — Run T2 (order flow) — blocks Method 3 signals
2. **TODO 0.1** — Re-run T4 (implied surfaces) — blocks surface validation
3. **TODO 4.1** — Build backtest engine core — blocks all strategy validation

### Phase 2: Calibration Surface Strategy (Method 2 — strongest overlap with existing work)
4. **TODO 2.1** — Fee model
5. **TODO 2.2** — Threshold calibration
6. **TODO 2.3** — Time-varying filter backtest
7. **TODO 2.4** — Regime-conditional sizing

### Phase 3: Monte Carlo & Kelly (Method 1 — risk management layer)
8. **TODO 1.1** — Per-strategy return sequences
9. **TODO 1.2** — Empirical distribution builder
10. **TODO 1.3** — Monte Carlo resampling engine
11. **TODO 1.4** — Drawdown distribution analysis
12. **TODO 1.5** — Empirical Kelly sizer

### Phase 4: Market Making (Method 3 — structural edge)
13. **TODO 3.1** — Validate T2 outputs
14. **TODO 3.2** — Adverse selection / toxicity signal
15. **TODO 3.3** — Market-making backtest
16. **TODO 3.4** — Inventory risk model

### Phase 5: Portfolio Integration
17. **TODO 1.6** — Multi-strategy portfolio Kelly
18. **TODO 0.3** — Run T5 (lead-lag)
19. **TODO 0.4** — Run T7 (address graph)
20. **TODO 0.5** — Update SCHEMAS.md

---

## What We Already Have That Roan Doesn't Mention

Our research goes beyond what Roan covers in several dimensions:

| Our Asset | Relevance |
|---|---|
| Address-level profiling (T3): 1.96M addresses with PnL, win rate, Herfindahl | Smart money signal Roan doesn't address — top 1% earns +35pp |
| Lifecycle state machine (T8): 8 states with anomaly detection | More granular than Roan's "early/mid/late" — we can condition on exact state |
| Category-specific calibration (1.3): 19x variation in MAE across categories | Roan treats all markets equally; we know Politics is 8x worse than Crypto |
| YES bias quantification (6.3): 2.23:1 ratio, growing over time | Directional taker bias signal Roan doesn't surface separately |
| Intra-event parity violations (4.2): 84% of multi-outcome events deviate >5c | Structural arb Roan doesn't cover at all |
| Monotonicity violations (4.3): 48.4% of threshold events violate | Same — pure structural alpha orthogonal to Roan's methods |
| Implied distribution surfaces (T4): fitted normal/lognormal with KS tests | Deeper than Roan's calibration surface — full distributional analysis |
| Composite strategy designs (7.1-7.3): multi-signal combinations | Roan presents methods in isolation; we've mapped the interaction effects |

These are additive to Roan's framework, not redundant. The priority is building the infrastructure (backtest + Monte Carlo + Kelly) that lets us operationalize both Roan's methods and our own unique signals.

---

## Success Criteria

Roan parity is achieved when we can:

1. For any defined strategy, extract its historical return sequence and construct the empirical distribution
2. Run 10K Monte Carlo simulations to produce drawdown percentiles
3. Compute uncertainty-adjusted Kelly sizing that keeps 95th percentile drawdown under 20%
4. Backtest the calibration surface strategy with fee-adjusted entry rules across (price, time, category) dimensions
5. Backtest the filtered market-making strategy with inventory constraints and adverse selection detection
6. Compare all strategies on a standardized performance table (Sharpe, drawdown, win rate, net edge after fees)
7. Allocate capital across strategies using portfolio-level Kelly optimization
