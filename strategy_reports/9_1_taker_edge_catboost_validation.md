# §9.1: Simple Taker Edge Detection (CatBoost + T9a OpenAI Embeddings)

## Verdict: SUCCESS

This validation used a **strict held-out final time period** and only features observable at trade time.

## Experimental Setup

- Platform: Kalshi resolved trades (from `t1a` + latest resolved market outcomes).
- Time split:
  - Train: trades before `2025-11-09 18:59:38.943171`
  - Validation: `2025-11-09 18:59:38.943171` to `2025-11-16 18:59:38.943171`
  - Held-out test: `2025-11-16 18:59:38.943171` to `2025-11-23 18:59:38.943171`
- Samples:
  - Train sampled with reservoir: 3,000,000
  - Validation sampled with reservoir: 500,000
  - Test: full final period (no sampling)
- Model: CatBoostClassifier (`iterations=600`, `depth=7`).
- Text signal: Existing **OpenAI embeddings from T9a** (`embeddings.npy`) keyed by `event_ticker`.
  - Kept 48 dimensions selected by **train-event variance only** (no holdout leakage).
- Baseline: implied probability from trade price (`taker_price / 100`).

## Leakage Controls

- Holdout is chronological and fully out-of-time.
- Labels (`won`) come from final outcomes and are never used as features.
- Features are trade-time-only:
  - taker side, prices, size, sequential T1A state, time-to-expiry, hour/day-of-week
  - T9a title embeddings by `event_ticker`
- Embedding dimension selection is fit on train events only.

## Held-Out Results

### Probability Metrics (lower is better for LogLoss/Brier)

| Metric | Baseline (Implied Prob) | CatBoost | Delta |
|---|---:|---:|---:|
| AUC | 0.8306 | 0.8325 | +0.0018 |
| LogLoss | 0.5025 | 0.5001 | +0.0023 (improvement) |
| Brier | 0.1681 | 0.1670 | +0.0011 (improvement) |

### Edge Metrics on Held-Out Period

- Overall taker excess return: -1.610 pp
- Predicted-edge > 0 subset:
  - n = 1,239,683
  - realized excess = +1.668 pp
- Top decile by predicted edge:
  - n = 417,372
  - realized excess = +5.369 pp
- Bottom decile by predicted edge:
  - n = 417,109
  - realized excess = -6.251 pp

## Success Criteria

- LogLoss improvement vs implied-probability baseline: `True`
- Brier improvement vs implied-probability baseline: `True`
- Positive realized edge in top predicted-edge decile: `True`

**Final verdict:** `SUCCESS`.

## Artifacts

- Metrics JSON: `output/taker_edge_catboost_metrics.json`
- Decile table: `output/taker_edge_catboost_deciles.csv`
- Feature importance: `output/taker_edge_catboost_feature_importance.csv`
- Decile chart: `strategy_reports/figures/9_1_taker_edge_deciles.png`
- Embedding dims used (T9a): `[349, 23, 72, 0, 39, 340, 507, 310, 265, 137, 456, 152, 359, 465, 109, 556, 454, 116, 518, 386, 8, 7, 4, 41, 286, 1419, 219, 1264, 5, 472, 254, 382, 443, 76, 337, 229, 1037, 92, 490, 83, 225, 587, 402, 250, 369, 269, 135, 151]`
