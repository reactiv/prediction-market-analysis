#!/usr/bin/env python3
"""
Strategy Report 9.1: Simple taker-edge detection with CatBoost.

Validation design:
- Target: taker_won (1 if taker side matches final market result).
- Holdout: final 7 days of available resolved-trade history.
- Validation: the 7 days before holdout.
- Training: all earlier periods, downsampled (reservoir sample) for speed.

Key requirement from user request:
- Use existing OpenAI market-name embeddings from T9a transform.
- Use only features available at trade time (no future leakage).
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
T1A_GLOB = str(ROOT / "data" / "transforms" / "t1a" / "kalshi" / "*.parquet")
MARKETS_GLOB = str(ROOT / "data" / "kalshi" / "markets" / "*.parquet")
T9A_INDEX = ROOT / "data" / "transforms" / "t9a" / "embedding_index.parquet"
T9A_EMBEDDINGS = ROOT / "data" / "transforms" / "t9a" / "embeddings.npy"

OUT_DIR = ROOT / "output"
FIG_DIR = ROOT / "strategy_reports" / "figures"
REPORT_PATH = ROOT / "strategy_reports" / "9_1_taker_edge_catboost_validation.md"
METRICS_JSON = OUT_DIR / "taker_edge_catboost_metrics.json"
DECILES_CSV = OUT_DIR / "taker_edge_catboost_deciles.csv"
FEATURE_IMPORTANCE_CSV = OUT_DIR / "taker_edge_catboost_feature_importance.csv"
DECILE_FIG = FIG_DIR / "9_1_taker_edge_deciles.png"

# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------
SEED = 42
HOLDOUT_DAYS = 7
VAL_DAYS = 7
TRAIN_SAMPLE_ROWS = 3_000_000
VAL_SAMPLE_ROWS = 500_000

EMBED_DIMS = 48
EMBED_VAR_SAMPLE_EVENTS = 50_000

CATBOOST_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "iterations": 600,
    "learning_rate": 0.08,
    "depth": 7,
    "l2_leaf_reg": 6.0,
    "random_seed": SEED,
    "od_type": "Iter",
    "od_wait": 80,
    "use_best_model": True,
    "thread_count": 8,
    "verbose": 100,
}


def stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def get_time_boundaries(con: duckdb.DuckDBPyConnection) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    q = f"""
        WITH latest_markets AS (
            SELECT
                ticker,
                status,
                result,
                ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY _fetched_at DESC) AS rn
            FROM read_parquet('{MARKETS_GLOB}')
        ),
        resolved AS (
            SELECT ticker
            FROM latest_markets
            WHERE rn = 1
              AND status = 'finalized'
              AND result IN ('yes', 'no')
        )
        SELECT MAX(t.created_time::TIMESTAMP) AS max_t
        FROM read_parquet('{T1A_GLOB}') t
        INNER JOIN resolved r USING (ticker)
    """
    max_t = con.execute(q).fetchone()[0]
    if max_t is None:
        raise RuntimeError("Could not find max timestamp in resolved trades.")

    max_t = pd.Timestamp(max_t)
    test_start = max_t - pd.Timedelta(days=HOLDOUT_DAYS)
    val_start = test_start - pd.Timedelta(days=VAL_DAYS)
    return max_t, val_start, test_start


def load_dataset(
    con: duckdb.DuckDBPyConnection,
    val_start: pd.Timestamp,
    test_start: pd.Timestamp,
) -> pd.DataFrame:
    stderr("[info] Building leakage-safe split dataset from T1A + resolved outcomes...")

    v = val_start.strftime("%Y-%m-%d %H:%M:%S")
    t = test_start.strftime("%Y-%m-%d %H:%M:%S")

    q = f"""
        WITH latest_markets AS (
            SELECT
                ticker,
                status,
                result,
                ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY _fetched_at DESC) AS rn
            FROM read_parquet('{MARKETS_GLOB}')
        ),
        resolved AS (
            SELECT ticker, result
            FROM latest_markets
            WHERE rn = 1
              AND status = 'finalized'
              AND result IN ('yes', 'no')
        ),
        base AS (
            SELECT
                t.event_ticker,
                t.taker_side,
                t.count::DOUBLE AS contracts,
                t.yes_price::DOUBLE AS yes_price,
                t.no_price::DOUBLE AS no_price,
                COALESCE(t.delta_price::DOUBLE, 0.0) AS delta_price,
                COALESCE(t.time_since_prev::DOUBLE, 0.0) AS time_since_prev,
                t.cumulative_volume::DOUBLE AS cumulative_volume,
                t.cumulative_net_flow::DOUBLE AS cumulative_net_flow,
                t.cumulative_trade_count::DOUBLE AS cumulative_trade_count,
                t.trade_sequence_num::DOUBLE AS trade_sequence_num,
                t.time_to_expiry_seconds::DOUBLE AS time_to_expiry_seconds,
                EXTRACT('hour' FROM t.created_time::TIMESTAMP) AS hour_utc,
                EXTRACT('dow' FROM t.created_time::TIMESTAMP) AS dow_utc,
                CASE
                    WHEN t.taker_side = 'yes' THEN t.yes_price / 100.0
                    ELSE t.no_price / 100.0
                END AS implied_prob,
                CASE
                    WHEN t.taker_side = r.result THEN 1
                    ELSE 0
                END AS won,
                t.created_time::TIMESTAMP AS created_ts
            FROM read_parquet('{T1A_GLOB}') t
            INNER JOIN resolved r USING (ticker)
            WHERE t.event_ticker IS NOT NULL
              AND t.created_time IS NOT NULL
              AND t.count > 0
              AND t.yes_price BETWEEN 1 AND 99
              AND t.no_price BETWEEN 1 AND 99
        )
        SELECT *, 'train' AS split
        FROM (
            SELECT
                event_ticker,
                taker_side,
                contracts,
                yes_price,
                no_price,
                delta_price,
                time_since_prev,
                cumulative_volume,
                cumulative_net_flow,
                cumulative_trade_count,
                trade_sequence_num,
                time_to_expiry_seconds,
                hour_utc,
                dow_utc,
                implied_prob,
                won
            FROM base
            WHERE created_ts < TIMESTAMP '{v}'
        ) train_pool
        USING SAMPLE reservoir({TRAIN_SAMPLE_ROWS} ROWS) REPEATABLE ({SEED})

        UNION ALL

        SELECT *, 'val' AS split
        FROM (
            SELECT
                event_ticker,
                taker_side,
                contracts,
                yes_price,
                no_price,
                delta_price,
                time_since_prev,
                cumulative_volume,
                cumulative_net_flow,
                cumulative_trade_count,
                trade_sequence_num,
                time_to_expiry_seconds,
                hour_utc,
                dow_utc,
                implied_prob,
                won
            FROM base
            WHERE created_ts >= TIMESTAMP '{v}'
              AND created_ts < TIMESTAMP '{t}'
        ) val_pool
        USING SAMPLE reservoir({VAL_SAMPLE_ROWS} ROWS) REPEATABLE ({SEED + 1})

        UNION ALL

        SELECT *, 'test' AS split
        FROM (
            SELECT
                event_ticker,
                taker_side,
                contracts,
                yes_price,
                no_price,
                delta_price,
                time_since_prev,
                cumulative_volume,
                cumulative_net_flow,
                cumulative_trade_count,
                trade_sequence_num,
                time_to_expiry_seconds,
                hour_utc,
                dow_utc,
                implied_prob,
                won
            FROM base
            WHERE created_ts >= TIMESTAMP '{t}'
        ) test_pool
    """

    df = con.execute(q).fetchdf()
    if df.empty:
        raise RuntimeError("No rows loaded for train/val/test.")

    split_counts = df["split"].value_counts().to_dict()
    stderr(f"[info] Loaded rows by split: {split_counts}")
    return df


def add_trade_time_features(df: pd.DataFrame) -> None:
    # Fill potential negative/invalid values defensively
    df["time_since_prev"] = np.clip(df["time_since_prev"].astype(np.float64), 0.0, None)
    df["time_to_expiry_seconds"] = np.clip(df["time_to_expiry_seconds"].astype(np.float64), 0.0, None)

    df["side_is_yes"] = (df["taker_side"] == "yes").astype(np.float32)
    df["log_contracts"] = np.log1p(df["contracts"].astype(np.float64)).astype(np.float32)
    df["log_time_since_prev"] = np.log1p(df["time_since_prev"].astype(np.float64)).astype(np.float32)
    df["log_cumulative_volume"] = np.log1p(df["cumulative_volume"].astype(np.float64)).astype(np.float32)
    df["log_cumulative_trades"] = np.log1p(df["cumulative_trade_count"].astype(np.float64)).astype(np.float32)
    df["tte_hours"] = (df["time_to_expiry_seconds"] / 3600.0).astype(np.float32)
    df["trade_notional_usd"] = (df["contracts"] * df["implied_prob"]).astype(np.float32)

    denom = np.maximum(df["cumulative_volume"].astype(np.float64), 1.0)
    df["ofi_ratio"] = (df["cumulative_net_flow"].astype(np.float64) / denom).astype(np.float32)

    hour = df["hour_utc"].astype(np.float64)
    dow = df["dow_utc"].astype(np.float64)
    df["hour_sin"] = np.sin(2.0 * math.pi * hour / 24.0).astype(np.float32)
    df["hour_cos"] = np.cos(2.0 * math.pi * hour / 24.0).astype(np.float32)
    df["dow_sin"] = np.sin(2.0 * math.pi * dow / 7.0).astype(np.float32)
    df["dow_cos"] = np.cos(2.0 * math.pi * dow / 7.0).astype(np.float32)

    df.drop(columns=["taker_side"], inplace=True)


def attach_t9a_embeddings(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[int]]:
    stderr("[info] Loading T9a embedding index and selecting embedding dimensions...")
    index_df = pd.read_parquet(T9A_INDEX, columns=["market_id", "platform"])
    kalshi_index = index_df[index_df["platform"] == "kalshi"].copy().reset_index()
    kalshi_index = kalshi_index.rename(columns={"index": "emb_row", "market_id": "event_ticker"})
    event_to_row = dict(zip(kalshi_index["event_ticker"], kalshi_index["emb_row"]))

    df["emb_row"] = df["event_ticker"].map(event_to_row)

    emb = np.load(T9A_EMBEDDINGS, mmap_mode="r")

    train_rows = (
        df.loc[(df["split"] == "train") & df["emb_row"].notna(), "emb_row"]
        .drop_duplicates()
        .astype(np.int64)
        .to_numpy()
    )
    if len(train_rows) == 0:
        raise RuntimeError("No train events mapped to T9a embeddings.")

    rng = np.random.default_rng(SEED)
    if len(train_rows) > EMBED_VAR_SAMPLE_EVENTS:
        sample_rows = rng.choice(train_rows, size=EMBED_VAR_SAMPLE_EVENTS, replace=False)
    else:
        sample_rows = train_rows

    sample_matrix = np.asarray(emb[sample_rows], dtype=np.float32)
    variances = sample_matrix.var(axis=0)
    top_idx = np.argpartition(variances, -EMBED_DIMS)[-EMBED_DIMS:]
    top_idx = top_idx[np.argsort(variances[top_idx])[::-1]]
    top_dims = top_idx.tolist()
    stderr(f"[info] Selected top-{EMBED_DIMS} embedding dims by train-event variance.")

    all_rows = df["emb_row"].dropna().astype(np.int64).drop_duplicates().to_numpy()
    emb_small = np.empty((len(all_rows), EMBED_DIMS), dtype=np.float32)
    for j, d in enumerate(top_dims):
        emb_small[:, j] = np.asarray(emb[all_rows, d], dtype=np.float32)

    emb_df = pd.DataFrame({"emb_row": all_rows})
    emb_cols = []
    for j in range(EMBED_DIMS):
        c = f"emb_{j:02d}"
        emb_df[c] = emb_small[:, j]
        emb_cols.append(c)

    df = df.merge(emb_df, on="emb_row", how="left")
    for c in emb_cols:
        df[c] = df[c].fillna(0.0).astype(np.float32)

    coverage = float(df["emb_row"].notna().mean())
    stderr(f"[info] Embedding coverage on sampled rows: {coverage:.2%}")

    df.drop(columns=["emb_row"], inplace=True)
    return df, emb_cols, top_dims


def score_metrics(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "auc": float(roc_auc_score(y_true, p)),
        "logloss": float(log_loss(y_true, p)),
        "brier": float(brier_score_loss(y_true, p)),
    }


def edge_stats(x: np.ndarray) -> dict[str, float]:
    n = int(len(x))
    if n == 0:
        return {"n": 0, "mean": float("nan"), "se": float("nan"), "z": float("nan")}
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    z = mean / se if n > 1 and se > 0 else float("nan")
    return {"n": n, "mean": mean, "se": se, "z": z}


def make_decile_table(test_df: pd.DataFrame) -> pd.DataFrame:
    test_df = test_df.copy()
    test_df["decile"] = pd.qcut(
        test_df["pred_edge"],
        q=10,
        labels=list(range(1, 11)),
        duplicates="drop",
    )
    out = (
        test_df.groupby("decile", observed=False)
        .agg(
            n=("won", "size"),
            mean_pred_edge=("pred_edge", "mean"),
            mean_realized_edge=("realized_edge", "mean"),
            mean_pred_prob=("pred_prob", "mean"),
            mean_implied_prob=("implied_prob", "mean"),
            win_rate=("won", "mean"),
        )
        .reset_index()
    )
    out["decile"] = out["decile"].astype(int)
    return out


def plot_deciles(deciles: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    x = deciles["decile"].to_numpy()
    ax.plot(x, deciles["mean_realized_edge"] * 100, marker="o", color="#1f77b4", lw=2, label="Realized edge")
    ax.plot(x, deciles["mean_pred_edge"] * 100, marker="o", color="#ff7f0e", lw=2, label="Predicted edge")
    ax.axhline(0.0, color="#444444", ls="--", lw=1)
    ax.set_xlabel("Predicted-Edge Decile (1=lowest, 10=highest)")
    ax.set_ylabel("Excess Return (pp)")
    ax.set_title("Held-Out Final Period: Predicted vs Realized Taker Edge by Decile")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(DECILE_FIG, dpi=150, facecolor="white")
    plt.close(fig)


def write_report(
    metrics: dict,
    top_dims: list[int],
    val_start: pd.Timestamp,
    test_start: pd.Timestamp,
    max_t: pd.Timestamp,
) -> None:
    verdict = metrics["verdict"]
    success_text = "SUCCESS" if verdict == "success" else "FAILURE"
    criteria = metrics["criteria"]

    md = f"""# §9.1: Simple Taker Edge Detection (CatBoost + T9a OpenAI Embeddings)

## Verdict: {success_text}

This validation used a **strict held-out final time period** and only features observable at trade time.

## Experimental Setup

- Platform: Kalshi resolved trades (from `t1a` + latest resolved market outcomes).
- Time split:
  - Train: trades before `{val_start}`
  - Validation: `{val_start}` to `{test_start}`
  - Held-out test: `{test_start}` to `{max_t}`
- Samples:
  - Train sampled with reservoir: {TRAIN_SAMPLE_ROWS:,}
  - Validation sampled with reservoir: {VAL_SAMPLE_ROWS:,}
  - Test: full final period (no sampling)
- Model: CatBoostClassifier (`iterations={CATBOOST_PARAMS["iterations"]}`, `depth={CATBOOST_PARAMS["depth"]}`).
- Text signal: Existing **OpenAI embeddings from T9a** (`embeddings.npy`) keyed by `event_ticker`.
  - Kept {EMBED_DIMS} dimensions selected by **train-event variance only** (no holdout leakage).
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
| AUC | {metrics["baseline"]["auc"]:.4f} | {metrics["model"]["auc"]:.4f} | {metrics["model"]["auc"] - metrics["baseline"]["auc"]:+.4f} |
| LogLoss | {metrics["baseline"]["logloss"]:.4f} | {metrics["model"]["logloss"]:.4f} | {metrics["baseline"]["logloss"] - metrics["model"]["logloss"]:+.4f} (improvement) |
| Brier | {metrics["baseline"]["brier"]:.4f} | {metrics["model"]["brier"]:.4f} | {metrics["baseline"]["brier"] - metrics["model"]["brier"]:+.4f} (improvement) |

### Edge Metrics on Held-Out Period

- Overall taker excess return: {metrics["overall_edge"]["mean"] * 100:+.3f} pp
- Predicted-edge > 0 subset:
  - n = {metrics["long_signal"]["n"]:,}
  - realized excess = {metrics["long_signal"]["mean"] * 100:+.3f} pp
- Top decile by predicted edge:
  - n = {metrics["top_decile"]["n"]:,}
  - realized excess = {metrics["top_decile"]["mean"] * 100:+.3f} pp
- Bottom decile by predicted edge:
  - n = {metrics["bottom_decile"]["n"]:,}
  - realized excess = {metrics["bottom_decile"]["mean"] * 100:+.3f} pp

## Success Criteria

- LogLoss improvement vs implied-probability baseline: `{criteria["logloss_improved"]}`
- Brier improvement vs implied-probability baseline: `{criteria["brier_improved"]}`
- Positive realized edge in top predicted-edge decile: `{criteria["top_decile_positive"]}`

**Final verdict:** `{success_text}`.

## Artifacts

- Metrics JSON: `output/taker_edge_catboost_metrics.json`
- Decile table: `output/taker_edge_catboost_deciles.csv`
- Feature importance: `output/taker_edge_catboost_feature_importance.csv`
- Decile chart: `strategy_reports/figures/9_1_taker_edge_deciles.png`
- Embedding dims used (T9a): `{top_dims}`
"""
    REPORT_PATH.write_text(md)


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("SET threads = 8")

    max_t, val_start, test_start = get_time_boundaries(con)
    stderr(f"[info] Max resolved trade timestamp: {max_t}")
    stderr(f"[info] Validation start: {val_start}")
    stderr(f"[info] Holdout start: {test_start}")

    df = load_dataset(con, val_start=val_start, test_start=test_start)
    con.close()

    add_trade_time_features(df)
    df, emb_cols, top_dims = attach_t9a_embeddings(df)

    base_feature_cols = [
        "implied_prob",
        "side_is_yes",
        "contracts",
        "log_contracts",
        "yes_price",
        "no_price",
        "delta_price",
        "log_time_since_prev",
        "cumulative_net_flow",
        "log_cumulative_volume",
        "log_cumulative_trades",
        "trade_sequence_num",
        "tte_hours",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "trade_notional_usd",
        "ofi_ratio",
    ]
    feature_cols = base_feature_cols + emb_cols

    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32)
    df["won"] = pd.to_numeric(df["won"], errors="coerce").fillna(0).astype(np.int8)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError("One or more required splits are empty after preprocessing.")

    stderr(
        "[info] Final split sizes: "
        f"train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}"
    )

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    y_train = train_df["won"].to_numpy(dtype=np.int8, copy=False)
    X_val = val_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    y_val = val_df["won"].to_numpy(dtype=np.int8, copy=False)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    y_test = test_df["won"].to_numpy(dtype=np.int8, copy=False)

    stderr("[info] Training CatBoost model...")
    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(X_train, y_train, eval_set=(X_val, y_val))

    stderr("[info] Scoring holdout...")
    p_test = model.predict_proba(X_test)[:, 1]
    p_base = test_df["implied_prob"].to_numpy(dtype=np.float64, copy=False)

    model_metrics = score_metrics(y_test, p_test)
    baseline_metrics = score_metrics(y_test, p_base)

    test_df["pred_prob"] = p_test
    test_df["pred_edge"] = test_df["pred_prob"] - test_df["implied_prob"]
    test_df["realized_edge"] = test_df["won"] - test_df["implied_prob"]

    q10 = float(test_df["pred_edge"].quantile(0.10))
    q90 = float(test_df["pred_edge"].quantile(0.90))
    top_decile = test_df[test_df["pred_edge"] >= q90]["realized_edge"].to_numpy(dtype=np.float64)
    bottom_decile = test_df[test_df["pred_edge"] <= q10]["realized_edge"].to_numpy(dtype=np.float64)
    long_signal = test_df[test_df["pred_edge"] > 0.0]["realized_edge"].to_numpy(dtype=np.float64)
    overall = test_df["realized_edge"].to_numpy(dtype=np.float64)

    deciles = make_decile_table(test_df)
    deciles.to_csv(DECILES_CSV, index=False)
    plot_deciles(deciles)

    fi = model.get_feature_importance(prettified=True)
    fi.to_csv(FEATURE_IMPORTANCE_CSV, index=False)

    criteria = {
        "logloss_improved": model_metrics["logloss"] < baseline_metrics["logloss"],
        "brier_improved": model_metrics["brier"] < baseline_metrics["brier"],
        "top_decile_positive": float(np.mean(top_decile)) > 0.0 if len(top_decile) else False,
    }
    verdict = "success" if all(criteria.values()) else "failure"

    metrics = {
        "config": {
            "seed": SEED,
            "holdout_days": HOLDOUT_DAYS,
            "val_days": VAL_DAYS,
            "train_sample_rows": TRAIN_SAMPLE_ROWS,
            "val_sample_rows": VAL_SAMPLE_ROWS,
            "embed_dims": EMBED_DIMS,
            "embed_dims_selected": top_dims,
            "catboost_params": CATBOOST_PARAMS,
            "val_start": str(val_start),
            "test_start": str(test_start),
            "max_timestamp": str(max_t),
        },
        "counts": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
        },
        "baseline": baseline_metrics,
        "model": model_metrics,
        "overall_edge": edge_stats(overall),
        "top_decile": edge_stats(top_decile),
        "bottom_decile": edge_stats(bottom_decile),
        "long_signal": edge_stats(long_signal),
        "criteria": criteria,
        "verdict": verdict,
    }

    METRICS_JSON.write_text(json.dumps(metrics, indent=2))
    write_report(
        metrics=metrics,
        top_dims=top_dims,
        val_start=val_start,
        test_start=test_start,
        max_t=max_t,
    )

    elapsed = time.time() - t0
    stderr(f"[done] Wrote {METRICS_JSON}")
    stderr(f"[done] Wrote {DECILES_CSV}")
    stderr(f"[done] Wrote {FEATURE_IMPORTANCE_CSV}")
    stderr(f"[done] Wrote {DECILE_FIG}")
    stderr(f"[done] Wrote {REPORT_PATH}")
    stderr(f"[done] Total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
