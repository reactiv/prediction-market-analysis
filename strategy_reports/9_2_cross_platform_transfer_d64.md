# §9.2: Cross-Platform Taker Edge + Transfer (CatBoost)

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
| Combined -> Kalshi | 750,000 | 0.5027 | 0.5037 | -0.0011 | -2.570 pp |
| Combined -> Polymarket | 750,000 | 0.4717 | 0.4718 | -0.0000 | +1.317 pp |
| Kalshi-only -> Kalshi | 750,000 | 0.5027 | 0.4989 | +0.0037 | +2.424 pp |
| Kalshi-only -> Polymarket (transfer) | 750,000 | 0.4717 | 0.4864 | -0.0147 | +0.876 pp |
| Polymarket-only -> Kalshi (transfer) | 750,000 | 0.5027 | 0.5013 | +0.0014 | +3.680 pp |
| Polymarket-only -> Polymarket | 750,000 | 0.4717 | 0.4716 | +0.0001 | +0.935 pp |

## Transfer Takeaways

- Kalshi -> Polymarket transfer logloss improvement: `-0.0147`
- Polymarket -> Kalshi transfer logloss improvement: `+0.0014`
- Combined model improvement on Kalshi: `-0.0011`
- Combined model improvement on Polymarket: `-0.0000`

## Artifacts

- Metrics JSON: `output/cross_platform_transfer_metrics_d64.json`
- Figure: `strategy_reports/figures/9_2_transfer_logloss_improvement_d64.png`
- Embedding dims used (n=64): `[349, 137, 109, 116, 152, 269, 39, 23, 276, 57, 340, 72, 507, 495, 490, 689, 60, 359, 0, 310, 267, 92, 508, 265, 556, 443, 198, 1037, 364, 83, 24, 454, 147, 518, 465, 610, 249, 274, 206, 311, 439, 451, 245, 384, 140, 386, 456, 322, 624, 315, 341, 1264, 5, 18, 407, 328, 433, 343, 354, 229, 1019, 225, 183, 643]`
