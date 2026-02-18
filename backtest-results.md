# Backtest Results: Kalshi Hold-to-Resolution Strategies

Platform: Kalshi | Period: July 2021 -- November 2025 | Engine: DuckDB-native SQL

---

## 1. Fade-Longshot Strategy Family

The core strategy: buy NO (fade the YES taker) on low-priced contracts, hold to resolution. The longshot bias --- the tendency for cheap YES contracts to resolve NO far more often than their price implies --- is the dominant exploitable edge in the Kalshi dataset.

### 1.1 Performance by Price Threshold

| Strategy | Trades | Win Rate | Avg Return | Sharpe | Sortino | Profit Factor | Max Drawdown | Total PnL |
|----------|-------:|--------:|-----------:|-------:|--------:|--------------:|-------------:|----------:|
| **< 5c** | 2.33M | 98.63% | +0.83c | 0.071 | 0.897 | 1.893 | -61.5M | +1,108M |
| **< 10c** | 4.99M | 96.72% | +1.21c | 0.068 | 0.604 | 1.606 | -229.9M | +2,730M |
| **< 15c** | 7.45M | 94.46% | +1.13c | 0.050 | 0.053 | 1.503 | -359.1M | +3,732M |
| **< 20c** | 9.93M | 92.08% | +1.04c | 0.039 | 0.253 | 1.268 | -564.2M | +4,408M |

*All values in cents. Sharpe and Sortino are per-trade (not annualized). Fees at 7% Kalshi taker rate, capped at 3c.*

### 1.2 Key Observations

**Best risk-adjusted return: < 10c.** Highest avg return (1.21c), best Sharpe (0.068), and strong profit factor (1.61). The < 5c variant has higher win rate (98.6%) but smaller absolute returns per trade.

**Diminishing returns above 15c.** The edge weakens as price increases --- at 20c, the larger loss per losing trade (-87.6c) erodes the advantage despite 92% win rate. The strategy becomes unviable above ~30c.

**Fee impact.** Comparing the < 15c strategy with and without fees:

| | With Fees | No Fees | Difference |
|--|----------:|--------:|-----------:|
| Avg return | 1.13c | 1.64c | -0.50c (31% of gross edge) |
| Total PnL | 3,732M | 4,808M | -1,076M in fees |
| Max drawdown | -359M | -342M | Fees worsen drawdown by 5% |

Fees consume ~31% of the gross edge. At < 5c, fees are proportionally smaller (0.07--0.35c per contract), making the cheapest contracts most fee-efficient.

### 1.3 Payoff Asymmetry

The strategy has a characteristic "picking up pennies" profile:

| Price Range | Avg Win | Avg Loss | Win:Loss Ratio |
|-------------|--------:|---------:|---------------:|
| < 5c | +2.19c | -97.16c | 1:44 |
| < 10c | +4.44c | -93.96c | 1:21 |
| < 15c | +6.53c | -90.74c | 1:14 |
| < 20c | +8.66c | -87.58c | 1:10 |

Small frequent wins vs rare catastrophic losses. The strategy works because the win rate is high enough that total wins overwhelm total losses --- but the distribution is extremely non-normal.

---

## 2. Calibration Surface Strategy

Uses T6 rolling calibration features (7-day rolling MAE by category) to filter trades. Hypothesis: selecting trades in categories with high historical mispricing should improve returns.

### 2.1 Threshold Sweep (< 15c, NO side, 7-day lag)

| MAE Threshold | Trades | Win Rate | Avg Return | Sharpe | Profit Factor |
|--------------:|-------:|--------:|-----------:|-------:|--------------:|
| No filter | 7.45M | 94.46% | **+1.13c** | **0.050** | **1.503** |
| 1pp | 5.25M | 93.89% | +0.67c | 0.029 | 1.214 |
| 3pp | 5.18M | 93.84% | +0.67c | 0.028 | 1.203 |
| 5pp | 5.01M | 93.72% | +0.62c | 0.026 | 1.184 |
| 8pp | 4.53M | 93.53% | +0.50c | 0.021 | 1.148 |
| 10pp | 3.95M | 93.30% | +0.34c | 0.014 | 1.107 |
| 15pp | 1.92M | 92.94% | +0.03c | 0.001 | 1.037 |
| 20pp | 0.76M | 92.31% | -0.60c | -0.023 | 0.932 |

### 2.2 Finding: MAE Filtering is Counter-Productive

**Higher MAE thresholds monotonically decrease performance.** The unfiltered fade-longshot outperforms every calibration-filtered variant. Two factors:

1. **The INNER JOIN loses ~30% of trades** (7.45M to 5.25M at 1pp). These are trades in (category, price, tte) cells without sufficient T6 history --- and they're profitable trades being excluded.

2. **High MAE selects unusual categories**, not persistently mispriced ones. Categories with high 7-day MAE tend to be those experiencing one-off events (elections, breaking news) where past mispricing doesn't predict future returns.

### 2.3 Broader Price Range (< 50c)

| MAE Threshold | Trades | Win Rate | Avg Return | Sharpe |
|--------------:|-------:|--------:|-----------:|-------:|
| 3pp | 19.3M | 75.3% | -0.47c | -0.011 |
| 5pp | 19.0M | 75.1% | -0.49c | -0.012 |
| 10pp | 16.2M | 74.4% | -0.57c | -0.014 |

**Buying NO above ~20c is unprofitable.** At mid-range prices (30--50c), the NO side costs 50--70c but only wins ~60--70% of the time. After the large losses on the 30--40% losing trades and fees, the net return is negative. The longshot bias is a tail phenomenon, not a broad market inefficiency.

### 2.4 Implication

The calibration surface is better suited for **position sizing** (scale into the highest-edge cells) rather than binary entry/exit filtering. The edge is broad-based across all longshot trades.

---

## 3. Return Distribution Analysis (< 15c Fade-Longshot)

Based on 7,448,000 trades.

### 3.1 Distribution Shape

| Statistic | Value |
|-----------|------:|
| Mean | +1.13c |
| Std Dev | 22.57c |
| Skewness | -3.72 |
| Kurtosis | 12.33 |
| Median | +5.58c |

The distribution is **heavily left-skewed** (long left tail from losing trades at -85c to -99c) and **extremely leptokurtic** (fat tails, kurtosis 12.3 vs 0 for normal).

### 3.2 Percentiles

| Percentile | Value |
|-----------:|------:|
| 1st | -93.49c |
| 5th | -86.98c |
| 10th | +0.93c |
| 25th | +2.79c |
| 50th (median) | +5.58c |
| 75th | +9.30c |
| 90th | +12.09c |
| 95th | +13.02c |
| 99th | +13.02c |

~90% of trades are profitable, but the 5--10% of losers are 10--15x larger than the average winner.

### 3.3 Normality Tests

| Test | Statistic | p-value | Normal? |
|------|----------:|--------:|---------|
| Shapiro-Wilk | 0.392 | ~0 | No |
| Jarque-Bera | 64,386,012 | 0 | No |

**Definitively non-normal.** The Shapiro-Wilk statistic (0.39) is far from 1.0, and the Jarque-Bera statistic is astronomical. Standard Gaussian-based risk models are inappropriate.

### 3.4 Student-t Fit

| Parameter | Value |
|-----------|------:|
| Degrees of freedom | 1.53 |
| Location | 6.34 |
| Scale | 3.36 |
| KS statistic | 0.111 |
| KS p-value | ~0 |

The Student-t fit has df = 1.53, indicating extremely heavy tails (heavier than a Cauchy distribution at df=1 would be lighter-tailed). However, the KS test rejects even the Student-t fit --- the return distribution does not match any standard parametric family well due to its bimodal nature (cluster of small wins + cluster of large losses).

---

## 4. Monte Carlo Simulation (< 15c Fade-Longshot)

10,000 bootstrap-resampled equity paths, 100,000 trades per path.

### 4.1 PnL Distribution Across Paths

| Percentile | PnL (cents) |
|-----------:|------------:|
| 5th | +34.2M |
| 25th | +45.0M |
| 50th (median) | +50.9M |
| 75th | +56.4M |
| 95th | +63.4M |
| Std dev | 9.0M |

All 10,000 paths are profitable. The median PnL per 100K trades is +50.9M cents at 1 contract per trade. The 5th percentile path still returns +34.2M cents --- the edge is robust to path variation.

### 4.2 Drawdown Distribution

| Percentile | Max Drawdown (cents) |
|-----------:|---------------------:|
| 50th | -4.1M |
| 75th | -5.6M |
| 90th | -8.4M |
| 95th | -18.1M |
| 99th | -20.1M |

The median max drawdown over 100K trades is -4.1M cents. The tail risk is significant: the 95th percentile drawdown (-18.1M) is 4.4x the median and represents 36% of the median path PnL.

### 4.3 Edge Stability

| Metric | Median | Std Dev | CV |
|--------|-------:|--------:|---:|
| Avg return/trade | 1.13c | 0.071c | 6.3% |
| Win rate | 94.46% | 0.072% | 0.08% |
| Sharpe (per-trade) | 0.050 | 0.003 | 5.7% |

The edge is remarkably stable: CV of average return is only 6.3%, meaning the uncertainty in the edge estimate is small relative to the edge itself. The win rate is nearly constant across paths (std = 0.07%).

---

## 5. Kelly Position Sizing

### 5.1 Kelly Criterion Results

| Parameter | Value |
|-----------|------:|
| Win probability | 94.46% |
| Avg win | +6.53c |
| Avg loss | -90.74c |
| Payoff ratio (b) | 0.072 |
| Raw Kelly fraction | 17.35% |
| CV of edge | 6.28% |
| **Adjusted Kelly** | **16.26%** |
| Half-Kelly | 8.13% |
| Quarter-Kelly | 4.07% |

### 5.2 Interpretation

The adjusted Kelly says to allocate ~16% of bankroll per trade. Given the average contract cost of 92.82c (buying NO at 85--99c), this means:

- **Bankroll $10,000**: ~$1,626 per trade = ~17.5 contracts
- **Bankroll $100,000**: ~$16,260 per trade = ~175 contracts

However, **Kelly fractions are upper bounds** for growth-optimal sizing. In practice:

- **Half-Kelly (8.1%)** is the standard conservative choice, reducing growth rate by only 25% while halving drawdown risk.
- **Quarter-Kelly (4.1%)** is appropriate given the extreme payoff asymmetry (1:14 win:loss ratio) and the bimodal return distribution which violates Kelly's Gaussian assumptions.

### 5.3 Drawdown Constraint

At 1 contract per trade over 100K trades, the 95th percentile max drawdown is 18.1M cents. Position size should be scaled so that:

```
position_size * 18.1M / bankroll < 20%
```

For a $100,000 bankroll (= 10M cents): max position multiplier = 0.20 * 10M / 18.1M = 0.11, or ~11 contracts per trade.

---

## 6. Summary of Actionable Findings

1. **The fade-longshot strategy has a real, quantified edge.** Buy NO on Kalshi taker-YES trades priced < 15c. Win rate 94.5%, net return 1.13c/contract after 7% fees. Edge is stable across 10K Monte Carlo paths (CV = 6.3%).

2. **Best risk-adjusted returns at < 10c.** Sharpe 0.068, profit factor 1.61, and the smallest drawdowns relative to PnL. The < 5c range has the highest win rate (98.6%) and best profit factor (1.89) but fewer available trades.

3. **Fees consume 31% of gross edge.** Any execution improvement (maker rebates, reduced fees) has outsized impact on net returns.

4. **MAE-based trade filtering does not improve performance.** The longshot bias is broad-based; T6 calibration surface is better used for position sizing than entry/exit decisions.

5. **Position sizing: use quarter-Kelly or half-Kelly.** The payoff asymmetry and fat-tailed distribution make full Kelly too aggressive. Quarter-Kelly (4% of bankroll per trade) balances growth with drawdown risk.

6. **The return distribution is bimodal, not normal.** Standard risk metrics (VaR, Gaussian Sharpe) are inappropriate. Use empirical bootstrap quantiles for risk management.

---

## Appendix: Methodology

- **Engine**: DuckDB-native SQL over Parquet files. No Python-level row iteration.
- **Data**: Kalshi raw trades joined to finalized markets with known YES/NO resolution.
- **Fees**: `min(min(price, 100-price) * 0.07, 3.0)` per contract, subtracted from gross return.
- **Sharpe/Sortino**: Per-trade (not annualized). Sortino uses proper downside deviation: `sqrt(mean(min(r, 0)^2))` over all trades.
- **Profit factor**: Gross profit / gross loss (pre-fee returns), correctly representing the strategy's raw edge.
- **Max drawdown**: Computed via cumulative PnL with deterministic ordering (timestamp, trade_id).
- **Monte Carlo**: 10,000 paths, 100K trades each, bootstrap with replacement from empirical return distribution.
- **Kelly**: Binary Kelly `f* = (pb - q) / b`, uncertainty-adjusted by `(1 - CV_edge)` from Monte Carlo.
- **Look-ahead caveat**: The calibration surface strategy uses T6 features with a 7-day lag. T6 win_rates incorporate eventual resolution outcomes, so some residual forward-looking bias may remain for markets resolving >7 days after trade date. This does not affect the fade-longshot strategies which use no calibration signal.
