# §1.1: Longshot Bias

## Summary

Kalshi markets exhibit a clear and consistent longshot bias: contracts priced below 15 cents win less often than their price implies, delivering a trade-weighted excess return of -0.74 percentage points to longshot buyers. Symmetrically, favorites priced above 85 cents overperform by +0.75 pp. A strategy that systematically fades longshots -- selling YES on contracts priced below 15 cents (or equivalently buying the opposing NO side at 86-99 cents) -- earns 0.74 cents per contract on 19 million trades, a small but statistically robust edge given the massive sample size.

## Methodology

- **Data source:** All finalized Kalshi markets with a `yes`/`no` result, joined with their complete trade history. Total sample: 135.5 million position-sides across 99 price points (1-99 cents).
- **Position construction:** Each trade generates two positions -- one for the taker at the taker's price, and one for the counterparty (maker) at the complementary price (100 - taker_price). This symmetric treatment ensures every trade is counted from both sides.
- **Excess return:** For each price point, excess return = actual win rate - price (in percentage points). A positive value means buyers earn more than the price implies; negative means the market overprices those contracts.
- **Fade-longshot strategy:** When a contract is priced at `p < 15`, the opposing side costs `100 - p` cents and wins with probability `100 - win_rate(p)`. The edge from fading is the opposing side's excess return: `(100 - win_rate) - (100 - p) = p - win_rate`.
- **Bucketing:** For the calibration chart, prices are grouped into 5-cent bins (midpoints at 2.5, 7.5, ..., 97.5) with a minimum of 100 observations per bucket.
- **Minimum sample:** All 99 individual price points have at least 100 observations (most have over 1 million).

## Results

### Excess Returns by Price

![Excess returns](figures/1_1_longshot_excess_returns.png)

The bar chart shows excess return (win rate minus price) at each cent from 1 to 99. Several patterns are immediately visible:

- **The longshot zone (1-14 cents, red bars) is uniformly negative.** Every single price point below 15 cents has a negative excess return, meaning longshot buyers systematically lose money. The worst point is at 7 cents (-1.21 pp) and 9 cents (-1.22 pp).
- **The favorite zone (86-99 cents, orange bars) is uniformly positive.** Favorite buyers earn more than their price implies at every price point above 85 cents, with the largest edge at 91 cents (+1.22 pp) and 93 cents (+1.21 pp).
- **The mid-range (15-85 cents) is noisy but roughly mean-zero**, oscillating between positive and negative with no clear directional bias. This suggests the longshot bias is a tail phenomenon, not a market-wide miscalibration.
- The bias is symmetric: the excess return curve is approximately antisymmetric around 50 cents, consistent with the zero-sum nature of binary contracts.

### Calibration in Longshot Zone

![Calibration](figures/1_1_longshot_calibration.png)

The calibration curve shows actual win rates versus contract prices in 5-cent buckets. The curve lies below the perfect-calibration diagonal in the longshot region and above it in the favorite region -- the classic signature of longshot bias. The deviation is small in absolute terms but consistent.

| Price Range | Avg Win Rate | Implied Prob | Excess Return (pp) | Edge (c/contract) | N trades |
|-------------|-------------|-------------|-------------------|-------------------|---------|
| 0-5c        | 2.06%       | 2.5%        | -0.44             | -0.44             | 6,643,756 |
| 5-10c       | 5.87%       | 7.5%        | -1.63             | -1.63             | 6,447,186 |
| 10-15c      | 11.00%      | 12.5%       | -1.50             | -1.50             | 5,923,608 |
| 85-90c      | 87.93%      | 87.5%       | +0.43             | +0.43             | 5,864,687 |
| 90-95c      | 93.14%      | 92.5%       | +0.64             | +0.64             | 6,341,324 |
| 95-100c     | 97.58%      | 97.5%       | +0.08             | +0.08             | 8,078,847 |

The longshot bias is strongest in the 5-10 cent bucket (-1.63 pp) and its mirror the 90-95 cent bucket (+0.64 pp). The asymmetry in magnitude between these mirrors reflects the fact that a 1.63 pp error at 7.5 cents is a 22% relative overpricing, while a 0.64 pp error at 92.5 cents is only a 0.7% relative error -- the same absolute bias manifests more dramatically at low prices.

Notably, the very cheapest contracts (0-5 cents) show a smaller bias (-0.44 pp) than the 5-15 cent range. This may reflect that 1-2 cent contracts are priced so cheaply that even biased bettors are reluctant to bid them higher, or that market makers provide tighter quotes at the extremes.

## Key Findings

- **Longshot bias is real and consistent on Kalshi.** All 14 individual price points from 1 to 14 cents show negative excess returns for buyers, with a trade-weighted average of -0.74 pp across 19.0 million trades. This is not a small-sample artifact.
- **The fade-longshot strategy earns 0.74 cents per contract.** Systematically selling YES (or buying NO) on contracts priced below 15 cents yields a 93.87% win rate on a position costing an average of 93.13 cents, for a net edge of 0.74 cents per contract.
- **The bias is concentrated in the 5-14 cent range.** The peak mispricing occurs around 7-9 cents, where contracts are overpriced by 1.0-1.2 pp. The very cheapest contracts (1-2 cents) are more accurately priced, suggesting that extreme longshots are priced low enough to partially compensate for the bias.
- **Fading favorites (buying NO on >85c contracts) is unprofitable.** The mirror strategy -- betting against favorites -- yields -0.75 cents per contract, because favorites are underpriced relative to their true win probability. This confirms the bias is directional: longshots are overpriced and favorites are underpriced.

## Strategy Implication

The longshot bias provides a small but reliable edge for a disciplined strategy:

1. **Core trade:** Sell YES (or buy NO) on any contract priced below 15 cents. The expected edge is 0.74 cents per contract, with a 93.87% win rate. At the peak mispricing (7-9 cents), the edge rises to approximately 1.1-1.2 cents per contract.
2. **Sizing and bankroll:** Because the opposing side costs 86-99 cents per contract and wins ~94% of the time, the strategy ties up significant capital for small returns. The return on capital is roughly 0.74/93.13 = 0.79% per trade -- meaningful at scale but requiring substantial volume or leverage.
3. **Combine with other signals.** The raw longshot bias is a blunt instrument. Combining it with category-specific miscalibration (see §1.3, where Politics shows 6.21 pp MAE) or time-decay signals can amplify the edge. Political longshots are likely more overpriced than sports longshots.
4. **Avoid fading favorites.** The data clearly shows that buying NO on contracts above 85 cents is a losing strategy. The bias is one-directional: exploit overpriced longshots, do not bet against underpriced favorites.

## Temporal Analysis

![Longshot bias by quarter](figures/1_1_longshot_bias_temporal.png)

The aggregate -0.74 pp longshot excess return masks a dramatic regime shift. When stratified by quarter, the longshot bias was **not present in the early market** and has **grown substantially as Kalshi matured**.

### Quarterly Longshot Excess Returns

| Quarter    | Excess Return (pp) | N Trades    | Avg Price (c) | Win Rate (%) |
|------------|-------------------:|------------:|--------------:|-------------:|
| 2021-Q3    | +1.72              | 7,146       | 6.92          | 8.65         |
| 2021-Q4    | -0.51              | 11,469      | 6.51          | 6.00         |
| 2022-Q1    | +0.03              | 13,745      | 6.79          | 6.82         |
| 2022-Q2    | +1.32              | 36,235      | 6.93          | 8.24         |
| 2022-Q3    | +1.55              | 73,222      | 6.76          | 8.31         |
| 2022-Q4    | +0.86              | 42,429      | 7.44          | 8.30         |
| 2023-Q1    | +1.03              | 43,954      | 7.54          | 8.57         |
| 2023-Q2    | -0.39              | 85,909      | 7.00          | 6.62         |
| 2023-Q3    | -0.42              | 66,609      | 6.82          | 6.40         |
| 2023-Q4    | +0.04              | 60,259      | 6.15          | 6.19         |
| 2024-Q1    | +1.21              | 90,563      | 6.44          | 7.65         |
| 2024-Q2    | -0.33              | 66,910      | 6.62          | 6.29         |
| 2024-Q3    | -1.44              | 50,502      | 6.65          | 5.21         |
| 2024-Q4    | **-3.02**          | 854,191     | 6.32          | 3.30         |
| 2025-Q1    | -1.92              | 888,312     | 6.82          | 4.89         |
| 2025-Q2    | -0.55              | 1,289,842   | 6.83          | 6.27         |
| 2025-Q3    | -1.88              | 2,349,679   | 7.03          | 5.15         |
| 2025-Q4    | -1.71              | 4,159,125   | 6.96          | 5.25         |

### Narrative

The data reveals two distinct eras:

1. **Early Kalshi (2021-Q3 through 2024-Q1):** The longshot bias was weak, noisy, and frequently reversed sign. Seven of the eleven quarters in this period showed *positive* excess returns for longshot buyers, meaning longshots were actually slightly underpriced on average. Trade volumes were modest (7k-91k trades per quarter), and the market was thin enough that persistent mispricing in either direction did not survive.

2. **Mature Kalshi (2024-Q3 onward):** The bias turned sharply and persistently negative. Every quarter from 2024-Q3 through 2025-Q4 shows negative longshot excess returns, ranging from -0.55 pp (2025-Q2) to -3.02 pp (2024-Q4). This coincides with an explosion in volume -- from tens of thousands of trades per quarter to millions. The worst quarter (2024-Q4, -3.02 pp on 854k trades) aligns with the 2024 U.S. election cycle, when speculative interest in political longshots surged.

The favourite zone mirrors this pattern in reverse: early quarters showed modest positive excess returns (+0.2 to +1.2 pp), consistent with favourite underpricing. But from 2024-Q2 onward, favourite excess returns have turned slightly negative or near-zero, suggesting that the classic longshot-favourite asymmetry has weakened at the favourite end even as it deepened at the longshot end.

The mid-range zone shows a separate, unexpected trend: after being roughly mean-zero through 2023, it has turned consistently negative (-1.0 to -1.5 pp) from 2024-Q4 onward, which may reflect systematic taker disadvantage at scale rather than a calibration issue.

### Key Takeaway

The longshot bias is **not a stable market feature** -- it is a phenomenon of the mature, high-volume Kalshi market. A fade-longshot strategy would have been unprofitable or flat in 2021-2023 and only became reliably profitable starting mid-2024. This has two implications: (1) the bias is likely driven by the influx of retail/speculative capital into political and event markets, and (2) a strategy that assumes a constant -0.74 pp edge is underestimating the opportunity in high-volume quarters (where the edge has been -1.5 to -3.0 pp) and overestimating it in quieter periods. Any live implementation should dynamically condition on volume regime and market type.

## Limitations

- **Edge is small relative to transaction costs.** At 0.74 cents per contract, even a 1-cent spread or fee erodes the entire edge. Profitability depends on execution quality, fee structure, and the ability to trade at or near the quoted price.
- **No time segmentation.** The longshot bias may vary over market lifecycle (e.g., stronger far from expiry, weaker near resolution). This aggregate analysis does not capture time-varying dynamics.
- **Survivorship in resolution.** Only finalized markets with a definitive yes/no result are included. Markets that were cancelled, delisted, or never resolved are excluded, which could introduce selection bias.
- **Symmetric position counting may overcount.** Each trade generates two position-sides. While this is necessary for unbiased calibration, it means the 19 million "trades" in the longshot zone represent roughly 9.5 million actual trades, each counted once for each side.
- **No risk adjustment.** The fade-longshot strategy has a ~6% loss rate, but losses are large relative to wins (losing 86-99 cents vs. winning 1-14 cents). The strategy has negative skew and occasional drawdowns, which are not captured by the average edge metric alone.
