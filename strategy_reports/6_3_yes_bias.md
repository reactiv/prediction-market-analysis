# §6.3: Yes Bias

## Summary

There is a strong and systematic YES bias on Kalshi: YES takers account for 46.8M trades versus 21.0M for NO takers, a 2.23:1 overall ratio. This bias is most extreme at mid-low prices (20-35 cents), where YES takers outnumber NO takers by nearly 4:1, and only inverts at very high prices (above 95 cents). YES takers also overpay slightly more than NO takers on average, earning -1.77 pp excess return versus -1.25 pp for NO, a gap of -0.52 pp that represents a persistent cost of the YES-side demand imbalance.

## Methodology

Trades from `data/kalshi/trades/*.parquet` were joined with finalized markets (`result IN ('yes', 'no')`) from `data/kalshi/markets/*.parquet`. For each price point (1-99 cents), we computed the count of YES-taker and NO-taker trades, the volume ratio, and the win rate for each side. Excess return is defined as `actual_win_rate - price/100`. Results were aggregated into 5-cent buckets with a minimum of 100 observations per bucket. Total sample: 46,755,669 YES-taker trades and 20,968,702 NO-taker trades across 99 price points, producing 20 buckets per side.

## Results

### Volume Ratio by Price

![Volume ratio](figures/6_3_yes_no_volume_ratio.png)

YES buying is disproportionate across nearly the entire price spectrum. The ratio peaks at 3.87:1 around 23 cents and remains above 3:1 for prices between 10 and 45 cents. It declines steadily above 50 cents, crossing the 1.0 balanced threshold only at the very highest prices (above 95 cents, ratio 0.71). The median volume ratio across all price points is 2.76:1. This indicates that traders have a strong behavioral preference for buying YES contracts, particularly at lower prices where the contracts represent long-shot-style bets.

### Calibration Asymmetry

![Calibration](figures/6_3_yes_bias_calibration.png)

Both YES and NO takers earn negative excess returns on average, consistent with market maker spreads extracting value. However, YES takers show larger negative excess returns at several key price regions. The YES taker curve is more volatile and dips to -3.5 pp at the 42.5-cent bucket and -3.7 pp at 67.5 cents. NO takers have a smoother excess return profile, with their worst bucket at -2.8 pp (17.5 cents). At high prices (77.5 cents), YES takers briefly reach near-zero excess return (+0.14 pp), the only bucket where either side approaches fair pricing. NO takers consistently outperform YES takers at prices above 40 cents, where the demand imbalance is absorbed by market makers as adverse pricing against YES buyers.

### Summary Table

| Price Range | YES Trades | NO Trades | Ratio | YES Excess (pp) | NO Excess (pp) | Gap (pp) |
|---|---|---|---|---|---|---|
| 1-10 | 5,485,648 | 2,239,555 | 2.45 | -1.88 | -1.87 | -0.01 |
| 11-20 | 4,915,662 | 1,378,857 | 3.57 | -2.40 | -2.65 | +0.24 |
| 21-30 | 5,179,933 | 1,414,514 | 3.66 | -1.77 | -2.26 | +0.49 |
| 31-40 | 5,599,783 | 1,622,124 | 3.45 | -0.79 | -1.51 | +0.72 |
| 41-50 | 5,916,765 | 1,931,784 | 3.06 | -3.20 | -1.82 | -1.38 |
| 51-60 | 5,717,695 | 2,130,095 | 2.68 | -0.68 | -0.76 | +0.08 |
| 61-70 | 4,765,803 | 2,032,629 | 2.34 | -3.06 | -1.57 | -1.48 |
| 71-80 | 3,680,127 | 2,182,401 | 1.69 | -0.97 | -0.80 | -0.17 |
| 81-90 | 2,974,699 | 2,493,219 | 1.19 | -1.20 | -0.49 | -0.72 |
| 91-99 | 2,519,554 | 3,543,524 | 0.71 | -1.02 | -0.48 | -0.55 |

## Key Findings

- **Pervasive YES bias**: YES takers outnumber NO takers at a 2.23:1 overall ratio, with the imbalance peaking at 3.87:1 near 23 cents. Only at the very highest prices (95+ cents) do NO takers dominate — because buying NO at 5 cents is structurally equivalent to the YES-buying impulse at low prices.
- **YES takers overpay more**: YES takers earn -1.77 pp weighted average excess return versus -1.25 pp for NO takers, a 0.52 pp gap. This asymmetry is especially pronounced at mid-range prices (41-50 cents: -1.38 pp gap) and high-mid prices (61-70 cents: -1.48 pp gap), where excess YES demand lets market makers widen spreads against YES buyers.
- **Market makers exploit the bias**: The calibration asymmetry pattern suggests that market makers systematically price YES contracts slightly above fair value to capture the excess demand. NO takers benefit from being on the less crowded side, paying smaller implicit spreads.

## Strategy Implication

The YES bias creates a structural edge for contrarian NO-side positioning. A strategy that systematically takes NO positions — particularly at mid-range prices (40-70 cents) where the YES-NO excess return gap is widest (1-1.5 pp) — would capture the pricing asymmetry created by retail YES-buying pressure. This is most actionable in liquid markets with tight spreads where the 0.5-1.5 pp edge exceeds transaction costs. Combined with the longshot bias findings (§1.1), an optimal approach would favor NO positions at mid-prices while avoiding extreme prices (below 10 or above 90) where both sides face similar calibration errors.

## Temporal Analysis

![Temporal YES bias](figures/6_3_yes_bias_temporal.png)

The YES bias is not a static feature of the Kalshi marketplace -- it emerged gradually and accelerated sharply in the most recent quarters as the platform's trading volume exploded. Quarterly analysis of 67.8M trades across 18 quarters (2021-Q3 through 2025-Q4) reveals two distinct regimes.

### Quarterly Table

| Quarter | YES Trades | NO Trades | YES/NO Ratio | YES Excess (pp) | NO Excess (pp) | Gap (pp) |
|---|---|---|---|---|---|---|
| 2021-Q3 | 28,536 | 29,779 | 0.96 | -3.78 | +11.25 | -15.02 |
| 2021-Q4 | 47,380 | 47,715 | 0.99 | +1.68 | +4.69 | -3.00 |
| 2022-Q1 | 63,800 | 61,346 | 1.04 | +2.92 | +5.38 | -2.47 |
| 2022-Q2 | 112,734 | 117,124 | 0.96 | +2.64 | +3.43 | -0.80 |
| 2022-Q3 | 196,974 | 204,882 | 0.96 | +0.97 | +2.31 | -1.34 |
| 2022-Q4 | 137,661 | 126,117 | 1.09 | -0.67 | +2.56 | -3.22 |
| 2023-Q1 | 146,978 | 151,225 | 0.97 | +0.79 | +1.05 | -0.26 |
| 2023-Q2 | 266,141 | 322,215 | 0.83 | +0.74 | +1.08 | -0.34 |
| 2023-Q3 | 220,750 | 233,554 | 0.95 | -0.41 | +2.38 | -2.79 |
| 2023-Q4 | 178,973 | 190,014 | 0.94 | +0.25 | -0.22 | +0.47 |
| 2024-Q1 | 269,220 | 288,443 | 0.93 | +2.26 | -1.54 | +3.80 |
| 2024-Q2 | 191,264 | 220,400 | 0.87 | -2.89 | +1.11 | -4.00 |
| 2024-Q3 | 179,945 | 168,162 | 1.07 | -1.32 | -0.02 | -1.29 |
| 2024-Q4 | 2,322,317 | 1,746,251 | 1.33 | -1.02 | -1.97 | +0.95 |
| 2025-Q1 | 3,571,327 | 2,245,652 | 1.59 | -2.14 | +0.38 | -2.52 |
| 2025-Q2 | 5,075,777 | 2,600,429 | 1.95 | -1.29 | -0.76 | -0.53 |
| 2025-Q3 | 11,827,608 | 4,470,059 | 2.65 | -1.04 | -1.43 | +0.39 |
| 2025-Q4 | 21,955,310 | 7,745,343 | 2.83 | -1.44 | -1.03 | -0.41 |

### Two-Regime Pattern

**Regime 1 -- Early platform (2021-Q3 to 2024-Q3):** The YES/NO ratio fluctuated in a narrow band between 0.83 and 1.09, with no systematic YES-side dominance. In several quarters (2023-Q2 at 0.83, 2024-Q2 at 0.87), NO takers actually outnumbered YES takers. Total quarterly volume ranged from 58K to 558K trades. During this period, both sides frequently earned positive excess returns, suggesting thinner markets with less efficient pricing and potentially more informed early adopters on both sides.

**Regime 2 -- Mass adoption (2024-Q4 onward):** The YES/NO ratio broke above 1.0 decisively in 2024-Q4 (1.33) and climbed steeply through 2025-Q4 (2.83). This coincided with a 50x increase in quarterly volume, from ~350K trades per quarter in mid-2024 to over 29M by 2025-Q4. The influx of new retail participants brought a strong YES-buying preference that intensified as the platform scaled. By the final two quarters, YES takers outnumbered NO takers by nearly 3:1.

### Excess Return Dynamics

The excess return gap between YES and NO takers is volatile and does not show a clean monotonic trend. In the early period, NO takers consistently earned higher excess returns than YES takers, with the gap occasionally exceeding -3 pp (2021-Q3: -15.0 pp, 2022-Q4: -3.2 pp). In the high-volume late period, both sides earn modestly negative excess returns (typically -1 to -2 pp), and the gap narrows to under 1 pp in absolute terms. This suggests that as volume and competition among market makers increased, pricing efficiency improved and the spread tightened for both sides, even as the YES-side volume imbalance grew.

### Key Finding

The YES bias is an emergent phenomenon of platform scale, not a constant feature of the market. It was essentially absent during Kalshi's first three years of operation (2021-2024) and only materialized with the wave of retail adoption beginning in late 2024. The aggregate 2.23:1 ratio reported in the cross-sectional analysis is dominated by the most recent quarters, which account for over 90% of all trades in the dataset. The strategy implication is that the YES bias is likely to persist or intensify as long as the platform continues to attract retail participants, but it could attenuate if the user base matures or if market makers adjust their pricing more aggressively to capture the imbalance.

## Limitations

- The analysis counts trades, not unique traders — a single participant placing many orders inflates the count without representing broader market sentiment.
- Excess return is computed on unweighted trade counts; contract-size-weighted results could differ if large trades exhibit different bias patterns.
- The YES/NO label is defined by `taker_side` in the trade data, which reflects the aggressor side. The maker on the opposite side may have different motivations (hedging, market-making) that are not captured.
- Markets that did not finalize (still active, or resolved with ambiguous outcomes) are excluded, which may introduce survivorship bias if certain market types are more likely to remain unresolved.
- Time-of-trade effects are not controlled for — the YES bias may vary by time of day, day of week, or market lifecycle phase.
