# §1.2: Favourite-Longshot Asymmetry

## Summary

YES takers are systematically overpriced relative to NO takers, with an overall excess-return gap of -0.52 percentage points (pp). Both sides of the market lose money on average (YES takers: -1.77 pp, NO takers: -1.25 pp), but YES takers fare worse, particularly in the mid-to-high price range (41-70 cents) where their excess losses exceed NO takers by 0.4-1.5 pp. The asymmetry reverses slightly in the low-price longshot zone (below 20 cents), where both sides are similarly mispriced.

## Methodology

- **Data source:** All finalized Kalshi markets with a "yes" or "no" result, joined to trades where the taker side is identified.
- **YES takers:** Trades where `taker_side = 'yes'`, priced at `yes_price` between 1 and 99 cents. Win when market result is "yes".
- **NO takers:** Trades where `taker_side = 'no'`, priced at `no_price` between 1 and 99 cents. Win when market result is "no".
- **Metrics:** For each 5-cent bucket (minimum 100 observations): actual win rate vs. implied probability (price/100), yielding excess return = win_rate - price/100.
- **Sample:** 46.8M YES-taker trades and 21.0M NO-taker trades across 99 price points (20 valid 5-cent buckets per side).

## Results

### Calibration by Taker Side
![Calibration curves for YES vs NO takers](figures/1_2_calibration_yes_vs_no.png)

Both calibration curves sit slightly below the 45-degree perfect-calibration line, confirming that takers on both sides systematically overpay. However, the YES curve deviates further from the diagonal in the 40-70 cent range, where YES takers' actual win rates fall materially below the implied probability. The NO curve hugs the diagonal more closely, especially above 75 cents, indicating that NO takers at high prices (i.e., taking the less popular side of a favourite) are closer to fair value.

### Excess Returns by Side
![Excess returns for YES vs NO takers](figures/1_2_excess_return_yes_vs_no.png)

The excess-return plot reveals the asymmetry in detail. YES takers exhibit sharper negative spikes, particularly around 42.5 cents (-3.51 pp) and 67.5 cents (-3.68 pp), while NO takers show a smoother, more moderate loss profile. Above 80 cents, NO takers' losses shrink to -0.3 to -0.6 pp, while YES takers still lose -0.6 to -1.6 pp. The only price region where YES takers' excess return is near zero is the 77.5-cent bucket (+0.14 pp). Both sides are uniformly negative in the longshot zone below 20 cents.

### Summary Table

| Price Range | YES Excess Return (pp) | NO Excess Return (pp) | Asymmetry (pp) | YES Trades | NO Trades |
|---|---|---|---|---|---|
| 1-10 | -1.88 | -1.87 | -0.01 | 4,957,150 | 2,070,598 |
| 11-20 | -2.40 | -2.65 | +0.25 | 4,931,816 | 1,393,270 |
| 21-30 | -1.77 | -2.26 | +0.49 | 5,140,659 | 1,400,672 |
| 31-40 | -0.79 | -1.51 | +0.72 | 5,567,215 | 1,595,914 |
| 41-50 | -3.20 | -1.82 | -1.38 | 5,880,285 | 1,858,604 |
| 51-60 | -0.68 | -0.76 | +0.08 | 5,803,527 | 2,168,774 |
| 61-70 | -3.06 | -1.57 | -1.49 | 4,867,963 | 2,032,604 |
| 71-80 | -0.97 | -0.80 | -0.17 | 3,731,056 | 2,152,531 |
| 81-90 | -1.20 | -0.49 | -0.71 | 3,059,250 | 2,451,588 |
| 91-99 | -1.02 | -0.48 | -0.54 | 2,816,748 | 3,844,147 |

## Key Findings

- **YES takers overpay more overall.** The weighted-average excess return for YES takers is -1.77 pp vs. -1.25 pp for NO takers, a gap of 0.52 pp in favour of taking the NO side. This confirms the favourite-longshot bias manifests asymmetrically by taker side.
- **The asymmetry is concentrated in mid-to-high prices.** The largest asymmetry appears in the 41-50 cent (-1.38 pp gap) and 61-70 cent (-1.49 pp gap) ranges, where YES takers dramatically underperform relative to NO takers. In the high-price favourite zone (81-99 cents), NO takers consistently lose less than YES takers.
- **Low prices show near-symmetric mispricing.** Below 20 cents, both sides are similarly overpriced (YES: -2.14 pp, NO: -2.19 pp), suggesting the longshot bias at the extremes is a market-wide phenomenon rather than a side-specific one. The slight advantage for YES takers in the 21-40 cent range (+0.49 to +0.72 pp) is the only region where YES outperforms NO.

## Strategy Implication

The data supports a **fade-YES strategy** in the 40-70 cent and 80-99 cent price ranges. In practice, this means preferring NO positions when the implied probability is in these ranges. Specifically:

1. **At 41-50 cents and 61-70 cents**, YES takers lose 1.4-1.5 pp more than NO takers. A trader placing NO at these price levels captures the asymmetry.
2. **At 81-99 cents (favourites)**, NO takers lose only 0.5 pp on average vs. 1.1 pp for YES takers. When a market trades at a high implied probability, the NO side is systematically closer to fair value, making it the better risk-adjusted position.
3. **Avoid both sides below 20 cents.** The longshot zone is equally overpriced for both taker sides (-2.1 to -2.2 pp), so there is no asymmetry to exploit there.

The core trade: when you observe a market in the 50-70 cent range and want to take a position, default to the NO side rather than YES, capturing approximately 1-1.5 pp of edge from the asymmetric overpricing of YES contracts.

## Limitations

- **Taker-side only.** This analysis uses the taker side of each trade. Maker behaviour and order-book dynamics are not captured; makers may be systematically profiting from both sides.
- **No fee adjustment.** Kalshi's fee structure is not deducted. Fees would further erode the already-negative excess returns on both sides.
- **Unweighted by contract count.** Win rates are computed per trade, not weighted by contract size. Large institutional NO trades may have different calibration profiles.
- **Temporal effects ignored.** The asymmetry may be time-varying (e.g., stronger during high-volume events). No time-series decomposition is performed.
- **No spread adjustment.** Taker trades cross the spread, and the bid-ask spread varies by price level. The apparent asymmetry may partly reflect differences in typical spreads faced by YES vs. NO takers.
