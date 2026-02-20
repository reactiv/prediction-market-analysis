# §1.3: Category-Specific Miscalibration

## Summary

After correcting to exact-price, contract-weighted calibration error, category dispersion is larger than previously reported.

- Most miscalibrated: **Politics (13.11 pp MAE)**
- Least miscalibrated: **Crypto (0.72 pp MAE)**
- Overall (top groups, contract-weighted): **2.37 pp MAE**

## Methodology

- Universe: finalized Kalshi yes/no markets.
- Each trade contributes taker and maker position-sides.
- Raw `event_ticker` prefixes are mapped into 8 category groups.
- **MAE is computed at exact price points** (not bucket midpoints), then contract-weighted within each group.
- 5-cent buckets are retained only for plotting.

## Results

| Category | MAE (pp) | Total Trades | Total Contracts |
|---|---:|---:|---:|
| Politics | 13.11 | 6,881,609 | 2,480,916,338 |
| Entertainment | 5.34 | 618,440 | 65,945,718 |
| Science/Tech | 3.69 | 292,292 | 30,619,506 |
| Finance | 3.07 | 8,421,934 | 1,816,434,788 |
| Sports | 1.50 | 71,054,938 | 21,760,754,551 |
| Other | 1.39 | 34,460,534 | 6,670,856,998 |
| Weather | 1.37 | 348,074 | 27,909,096 |
| Crypto | 0.72 | 13,407,950 | 1,558,106,312 |

Worst price regions (contract-weighted mean absolute error):

- 57.5c bucket: 4.84 pp
- 42.5c bucket: 4.28 pp
- 62.5c bucket: 3.38 pp
- 67.5c bucket: 3.17 pp
- 37.5c bucket: 3.02 pp

## Interpretation

- Category choice is a first-order driver of calibration quality.
- Politics error is now much larger under corrected weighting.
- Mid-price regions still dominate miscalibration across groups.

## Figures

- ![Calibration by category](figures/1_3_calibration_by_category.png)
- ![MAE by category](figures/1_3_mae_by_category.png)
- ![Category miscalibration temporal](figures/1_3_category_miscalibration_temporal.png)

## Limitations

- Category mapping remains heuristic (`event_ticker` prefix rules).
- Aggregates are gross of fees/slippage and can hide temporal regime shifts.
