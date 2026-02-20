# §9.2: Cross-Platform Taker Edge + Transfer (CatBoost)

## Setup

- Same temporal split boundary as §9.1 (from Kalshi max resolved timestamp):
  - Validation start: `2025-11-09 18:59:38.943171`
  - Holdout start: `2025-11-16 18:59:38.943171`
  - Max timestamp reference: `2025-11-23 18:59:38.943171`
- Data:
  - Kalshi + Polymarket resolved trades (taker perspective)
  - Text embeddings disabled (`--no-text` ablation)
- Sampling per platform:
  - Train: 1,500,000
  - Val: 250,000
  - Test: 750,000

## Split Counts

- Kalshi: train=1,500,000, val=250,000, test=750,000
- Polymarket: train=1,500,000, val=250,000, test=750,000

## Results (Held-Out Final Period)

| Model / Eval | N | Baseline LogLoss | Model LogLoss | Improvement | Top-Decile Realized Edge |
|---|---:|---:|---:|---:|---:|
| Combined -> Kalshi | 750,000 | 0.5024 | 0.5058 | -0.0035 | +0.157 pp |
| Combined -> Polymarket | 750,000 | 0.4727 | 0.4732 | -0.0005 | +0.571 pp |
| Kalshi-only -> Kalshi | 750,000 | 0.5024 | 0.5039 | -0.0016 | +0.608 pp |
| Kalshi-only -> Polymarket (transfer) | 750,000 | 0.4727 | 0.8901 | -0.4174 | -0.304 pp |
| Polymarket-only -> Kalshi (transfer) | 750,000 | 0.5024 | 0.5050 | -0.0027 | -1.307 pp |
| Polymarket-only -> Polymarket | 750,000 | 0.4727 | 0.4729 | -0.0003 | +0.554 pp |

## Transfer Takeaways

- Kalshi -> Polymarket transfer logloss improvement: `-0.4174`
- Polymarket -> Kalshi transfer logloss improvement: `-0.0027`
- Combined model improvement on Kalshi: `-0.0035`
- Combined model improvement on Polymarket: `-0.0005`

## Artifacts

- Metrics JSON: `output/cross_platform_transfer_metrics_no_text.json`
- Figure: `strategy_reports/figures/9_2_transfer_logloss_improvement_no_text.png`
- Embedding dims used: `[]`
