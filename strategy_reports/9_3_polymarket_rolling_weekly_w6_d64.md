# §9.3: Polymarket Rolling Weekly Holdouts (n_dims=64)

Anchor max timestamp (Kalshi-aligned): `2025-11-23 18:59:38.943171`

Weekly folds: `6`

Samples per fold:
- train: 1,500,000
- val: 250,000
- test: 750,000

## Fold Results

| Test Start | Test End | LogLoss Improvement | AUC (Model) | Top-Decile Edge |
|---|---|---:|---:|---:|
| 2025-10-12 18:59:38.943171 | 2025-10-19 18:59:38.943171 | +0.0038 | 0.8787 | +2.354 pp |
| 2025-10-19 18:59:38.943171 | 2025-10-26 18:59:38.943171 | +0.0032 | 0.8911 | +2.959 pp |
| 2025-10-26 18:59:38.943171 | 2025-11-02 18:59:38.943171 | +0.0012 | 0.8817 | +1.912 pp |
| 2025-11-02 18:59:38.943171 | 2025-11-09 18:59:38.943171 | +0.0007 | 0.8591 | +0.813 pp |
| 2025-11-09 18:59:38.943171 | 2025-11-16 18:59:38.943171 | -0.0009 | 0.8710 | +0.007 pp |
| 2025-11-16 18:59:38.943171 | 2025-11-23 18:59:38.943171 | -0.0028 | 0.8670 | +0.745 pp |


## Summary

- Top-decile edge positive in `6/6` weeks
- LogLoss improvement positive in `4/6` weeks
- Mean top-decile edge: `+1.465 pp`
- Std top-decile edge: `1.122 pp`
- Mean logloss improvement: `+0.0009`

## Artifacts

- JSON: `output/polymarket_rolling_weekly_metrics_w6_d64.json`
- CSV: `output/polymarket_rolling_weekly_summary_w6_d64.csv`
- Figure: `strategy_reports/figures/9_3_polymarket_rolling_top_decile_w6_d64.png`
