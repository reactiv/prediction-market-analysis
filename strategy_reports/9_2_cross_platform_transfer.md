# §9.2: Cross-Platform Taker Edge + Transfer (CatBoost + T9a Embeddings)

## Setup

- Same temporal split boundary as §9.1 (from Kalshi max resolved timestamp):
  - Validation start: `2025-11-09 18:59:38.943171`
  - Holdout start: `2025-11-16 18:59:38.943171`
  - Max timestamp reference: `2025-11-23 18:59:38.943171`
- Data:
  - Kalshi + Polymarket resolved trades (taker perspective)
  - Existing T9a OpenAI embeddings (`embedding_index.parquet` + `embeddings.npy`)
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
| Combined -> Kalshi | 750,000 | 0.5035 | 0.5000 | +0.0035 | +5.076 pp |
| Combined -> Polymarket | 750,000 | 0.4723 | 0.4725 | -0.0001 | +0.946 pp |
| Kalshi-only -> Kalshi | 750,000 | 0.5035 | 0.4998 | +0.0037 | +4.181 pp |
| Kalshi-only -> Polymarket (transfer) | 750,000 | 0.4723 | 0.5392 | -0.0668 | +0.446 pp |
| Polymarket-only -> Kalshi (transfer) | 750,000 | 0.5035 | 0.5037 | -0.0002 | -1.058 pp |
| Polymarket-only -> Polymarket | 750,000 | 0.4723 | 0.4724 | -0.0001 | +1.015 pp |

## Transfer Takeaways

- Kalshi -> Polymarket transfer logloss improvement: `-0.0668`
- Polymarket -> Kalshi transfer logloss improvement: `-0.0002`
- Combined model improvement on Kalshi: `+0.0035`
- Combined model improvement on Polymarket: `-0.0001`

## Artifacts

- Metrics JSON: `output/cross_platform_transfer_metrics.json`
- Figure: `strategy_reports/figures/9_2_transfer_logloss_improvement.png`
- Embedding dims used: `[349, 137, 109, 116, 152, 269, 39, 23, 57, 276, 340, 72, 507, 689, 495, 60, 490, 0, 359, 310, 267, 92, 508, 265, 556, 443, 198, 364, 1037, 24, 83, 147]`
