# §3.4: Volume Regime Switching

## Summary
High-volume markets (top tercile by trade count) are substantially better calibrated than both medium- and low-volume markets, with a mean absolute error (MAE) of just 1.59 pp compared to 4.96 pp for medium and 2.99 pp for low. The relationship is non-monotonic: medium-volume markets are the *worst* calibrated, exhibiting systematic overpricing of low-probability events and underpricing near the midrange, while low-volume markets appear better-calibrated than medium but suffer from sparse coverage across price buckets.

## Methodology
Markets were classified into volume regimes using the distribution of per-market trade counts across all markets in the dataset. The 33rd and 67th percentile thresholds of total trade count per market were used as cutoffs:

- **Low volume:** markets with 1 trade (bottom 33rd percentile, threshold = 1 trade)
- **Medium volume:** markets with 2-16 trades (33rd to 67th percentile)
- **High volume:** markets with >16 trades (top 33rd percentile)

For each regime, trades were grouped into 5-cent price buckets. Within each bucket, the empirical win rate was compared to the expected win rate (the bucket midpoint / 100). Only buckets with at least 200 trades were retained. MAE was computed as the trade-weighted mean of |empirical win rate - expected win rate| across all price buckets within a regime.

**Sample sizes per regime:**

| Regime | Markets | Trades | Contracts |
|--------|---------|--------|-----------|
| Low    | 202,173 | 202,173 | 133.7M |
| Medium | 167,681 | 1,065,569 | 208.0M |
| High   | 184,369 | 66,493,664 | 16.9B |

## Results

### Calibration by Volume Regime
![Calibration](figures/3_4_calibration_by_volume_regime.png)

The high-volume calibration curve (dark blue) hugs the 45-degree perfect-calibration line closely across the full price spectrum, deviating by at most ~3 pp in the 40-50 cent range. The medium-volume curve (medium blue) shows pronounced underpricing of actual outcomes for contracts priced between 10 and 45 cents -- a contract priced at 32.5 cents wins only 21.4% of the time, a 11.1 pp shortfall. In the upper half of the price range (65-85 cents), medium-volume markets flip to overpricing outcomes relative to price. The low-volume curve (light blue) follows a qualitatively similar S-shape but with less extreme deviations; however, it has poor coverage above 60 cents due to sparse data.

### Mean Absolute Error by Regime
![MAE](figures/3_4_mae_by_regime.png)

High-volume markets achieve the best calibration at 1.59 pp MAE, roughly three times better than medium-volume markets (4.96 pp) and about half the error of low-volume markets (2.99 pp). The non-monotonic pattern -- medium being worst -- likely reflects that medium-volume markets have enough activity for prices to form but not enough for prices to converge to true probabilities.

### Summary Table
| Volume Regime | Markets | Trades | Contracts | MAE (pp) | Avg Excess Return (pp) |
|---|---|---|---|---|---|
| Low    | 202,173 | 202,173    | 133.7M  | 2.99 | -2.81 |
| Medium | 167,681 | 1,065,569  | 208.0M  | 4.96 | -2.37 |
| High   | 184,369 | 66,493,664 | 16.9B   | 1.59 | -1.59 |

All regimes show negative average excess returns, indicating the taker side systematically underperforms across all volume levels. However, the magnitude of this underperformance is smallest in high-volume markets (-1.59 pp) and largest in low-volume markets (-2.81 pp).

## Key Findings
- **High-volume markets are 3x better calibrated** than medium-volume markets (1.59 pp vs 4.96 pp MAE), confirming that deeper liquidity drives prices toward true probabilities.
- **The relationship is non-monotonic:** medium-volume markets are the worst calibrated, not low-volume ones. This suggests that markets with intermediate activity develop price signals that are misleadingly precise but still systematically biased.
- **Medium-volume markets exhibit a strong favourite-longshot bias:** contracts priced at 15-35 cents win far less than their price implies (shortfall of 6-11 pp), while contracts priced at 70-85 cents win more than expected (surplus of 5-7 pp). This pattern is consistent with uninformed speculators overweighting longshots in markets without sufficient sophisticated participation to correct prices.
- **Low-volume markets appear relatively well-calibrated** (2.99 pp MAE) but this is partly an artifact -- most low-volume markets have only 1 trade, concentrating prices at extremes (2.5 and 97.5 cents) where calibration error is mechanically low.
- **Taker side loses money everywhere**, but the loss is smallest in high-volume markets (-1.59 pp) and largest in low-volume markets (-2.81 pp), consistent with wider effective spreads in thin markets.

## Strategy Implication
The data supports two complementary strategies:

1. **Fade medium-volume longshots.** In medium-volume markets, contracts priced at 15-35 cents are systematically overpriced by 6-11 pp. Selling (taking the "no" side) on these contracts offers the highest expected edge. These markets have enough liquidity to actually execute trades but not enough to eliminate the bias.

2. **Prefer high-volume markets for directional trading.** If the goal is to express a directional view, high-volume markets offer the fairest prices (lowest MAE) and smallest taker-side disadvantage. Any edge from proprietary information is more likely to be real and less likely to be consumed by bid-ask spread and miscalibration.

Low-volume markets should generally be avoided despite their apparent calibration -- the single-trade nature of most of these markets means fill sizes are tiny and the calibration metric is misleadingly favorable.

## Limitations
- **Static classification:** Markets are classified by their lifetime total trade count, which is only known ex-post. A live strategy would need to estimate expected final volume, perhaps using early trading velocity or market category as a proxy.
- **Survivorship in low-volume bucket:** Markets with exactly 1 trade (the bulk of the "low" regime) are a special population -- they may be markets that were created but never attracted interest. Their calibration properties may not generalize.
- **Market lifecycle effects:** A market that ends up "high volume" spends its early life as "low volume." Classifying by final volume conflates the effect of volume with the effect of being closer to resolution (when more information is available).
- **Percentile thresholds are data-dependent:** The p33 threshold of 1 trade and p67 threshold of 16 trades reflect the extreme skewness of Kalshi's market volume distribution. Different threshold choices (e.g., fixed trade count cutoffs) could yield different conclusions.
- **Taker-side only:** The analysis considers only the taker side of each trade. Maker-side calibration could differ, and the combined picture would require order-book data.
