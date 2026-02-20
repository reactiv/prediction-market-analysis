#!/usr/bin/env python3
"""
Strategy Report 9.2: Cross-platform taker-edge model + transfer assessment.

Builds a shared model over Kalshi + Polymarket using the same temporal split
boundary as §9.1 (derived from Kalshi max resolved-trade timestamp), then
measures transfer:
- train on Kalshi -> test on Polymarket
- train on Polymarket -> test on Kalshi
"""

from __future__ import annotations

import argparse
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

KALSHI_T1A_GLOB = str(ROOT / "data" / "transforms" / "t1a" / "kalshi" / "*.parquet")
POLY_T1A_GLOB = str(ROOT / "data" / "transforms" / "t1a" / "polymarket" / "**" / "*.parquet")
KALSHI_MARKETS_GLOB = str(ROOT / "data" / "kalshi" / "markets" / "*.parquet")
POLY_MARKETS_GLOB = str(ROOT / "data" / "polymarket" / "markets" / "*.parquet")

T9A_INDEX = ROOT / "data" / "transforms" / "t9a" / "embedding_index.parquet"
T9A_EMBEDDINGS = ROOT / "data" / "transforms" / "t9a" / "embeddings.npy"

OUT_DIR = ROOT / "output"
FIG_DIR = ROOT / "strategy_reports" / "figures"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
HOLDOUT_DAYS = 7
VAL_DAYS = 7

TRAIN_SAMPLE_PER_PLATFORM = 1_500_000
VAL_SAMPLE_PER_PLATFORM = 250_000
TEST_SAMPLE_PER_PLATFORM = 750_000

EMBED_DIMS = 32
EMBED_VAR_SAMPLE_EVENTS = 80_000

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
    "allow_writing_files": False,
}


def stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-platform transfer experiment.")
    p.add_argument(
        "--full-text-no-selection",
        action="store_true",
        help="Use all T9a embedding dimensions (no variance-based selection).",
    )
    p.add_argument(
        "--no-text",
        action="store_true",
        help="Disable T9a embedding features (ablation).",
    )
    p.add_argument("--train-sample-per-platform", type=int, default=None)
    p.add_argument("--val-sample-per-platform", type=int, default=None)
    p.add_argument("--test-sample-per-platform", type=int, default=None)
    p.add_argument(
        "--n-dims",
        type=int,
        default=EMBED_DIMS,
        help=f"Number of selected embedding dims when using selected-text mode (default: {EMBED_DIMS}).",
    )
    return p.parse_args()


def artifact_paths(mode: str, n_dims: int) -> tuple[Path, Path, Path]:
    suffix = {
        "selected_text": "",
        "no_text": "_no_text",
        "full_text_all_dims": "_full_text_all_dims",
    }[mode]
    if mode == "selected_text" and n_dims != EMBED_DIMS:
        suffix += f"_d{n_dims}"
    report_path = ROOT / "strategy_reports" / f"9_2_cross_platform_transfer{suffix}.md"
    metrics_path = OUT_DIR / f"cross_platform_transfer_metrics{suffix}.json"
    fig_path = FIG_DIR / f"9_2_transfer_logloss_improvement{suffix}.png"
    return report_path, metrics_path, fig_path


def get_kalshi_time_boundaries(con: duckdb.DuckDBPyConnection) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    q = f"""
        WITH latest_markets AS (
            SELECT
                ticker,
                status,
                result,
                ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY _fetched_at DESC) AS rn
            FROM read_parquet('{KALSHI_MARKETS_GLOB}')
        ),
        resolved AS (
            SELECT ticker
            FROM latest_markets
            WHERE rn = 1
              AND status = 'finalized'
              AND result IN ('yes', 'no')
        )
        SELECT MAX(t.created_time::TIMESTAMP) AS max_t
        FROM read_parquet('{KALSHI_T1A_GLOB}') t
        INNER JOIN resolved r USING (ticker)
    """
    max_t = con.execute(q).fetchone()[0]
    if max_t is None:
        raise RuntimeError("Failed to compute Kalshi max resolved timestamp.")

    max_t = pd.Timestamp(max_t)
    test_start = max_t - pd.Timedelta(days=HOLDOUT_DAYS)
    val_start = test_start - pd.Timedelta(days=VAL_DAYS)
    return max_t, val_start, test_start


def build_poly_resolution_table(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    stderr("[info] Building Polymarket token resolution map...")
    mk = con.execute(
        f"""
        SELECT id, clob_token_ids, outcome_prices
        FROM read_parquet('{POLY_MARKETS_GLOB}')
        WHERE closed = true
        """
    ).fetchdf()

    rows: list[tuple[str, int]] = []
    for _, r in mk.iterrows():
        try:
            token_ids = json.loads(r["clob_token_ids"]) if r["clob_token_ids"] else None
            prices = json.loads(r["outcome_prices"]) if r["outcome_prices"] else None
            if not token_ids or not prices or len(token_ids) != len(prices):
                continue
            probs = [float(x) for x in prices]
            if max(probs) < 0.99 or min(probs) > 0.01:
                continue
            for token_id, p in zip(token_ids, probs):
                rows.append((str(token_id), 1 if p > 0.99 else 0))
        except Exception:
            continue

    res_df = pd.DataFrame(rows, columns=["token_id", "token_won"]).drop_duplicates(subset=["token_id"])
    if res_df.empty:
        raise RuntimeError("No resolved Polymarket tokens found.")

    stderr(
        "[info] Polymarket resolved tokens: "
        f"{len(res_df):,} rows ({res_df['token_id'].nunique():,} unique)"
    )
    return res_df


def load_kalshi_data(
    con: duckdb.DuckDBPyConnection,
    val_start: pd.Timestamp,
    test_start: pd.Timestamp,
    train_sample_per_platform: int,
    val_sample_per_platform: int,
    test_sample_per_platform: int,
) -> pd.DataFrame:
    stderr("[info] Loading Kalshi sampled splits...")
    v = val_start.strftime("%Y-%m-%d %H:%M:%S")
    t = test_start.strftime("%Y-%m-%d %H:%M:%S")

    q = f"""
        WITH latest_markets AS (
            SELECT
                ticker,
                status,
                result,
                ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY _fetched_at DESC) AS rn
            FROM read_parquet('{KALSHI_MARKETS_GLOB}')
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
                'kalshi'::VARCHAR AS platform,
                t.event_ticker::VARCHAR AS event_id,
                t.created_time::TIMESTAMP AS created_ts,
                CASE
                    WHEN t.taker_side = 'yes' THEN t.yes_price / 100.0
                    ELSE t.no_price / 100.0
                END AS implied_prob,
                CASE
                    WHEN t.taker_side = r.result THEN 1
                    ELSE 0
                END AS won,
                t.count::DOUBLE * (
                    CASE WHEN t.taker_side = 'yes' THEN t.yes_price / 100.0 ELSE t.no_price / 100.0 END
                ) AS trade_size_usd,
                COALESCE(t.delta_price::DOUBLE / 100.0, 0.0) AS delta_prob,
                COALESCE(t.time_since_prev::DOUBLE, 0.0) AS time_since_prev,
                t.cumulative_volume::DOUBLE AS cumulative_volume,
                t.cumulative_net_flow::DOUBLE AS cumulative_net_flow,
                t.cumulative_trade_count::DOUBLE AS cumulative_trade_count,
                t.trade_sequence_num::DOUBLE AS trade_sequence_num,
                t.time_to_expiry_seconds::DOUBLE AS time_to_expiry_seconds,
                EXTRACT('hour' FROM t.created_time::TIMESTAMP) AS hour_utc,
                EXTRACT('dow' FROM t.created_time::TIMESTAMP) AS dow_utc
            FROM read_parquet('{KALSHI_T1A_GLOB}') t
            INNER JOIN resolved r USING (ticker)
            WHERE t.event_ticker IS NOT NULL
              AND t.created_time IS NOT NULL
              AND t.count > 0
              AND t.yes_price BETWEEN 1 AND 99
              AND t.no_price BETWEEN 1 AND 99
        )
        SELECT *, 'train' AS split
        FROM (
            SELECT * FROM base WHERE created_ts < TIMESTAMP '{v}'
        ) train_pool
        USING SAMPLE reservoir({train_sample_per_platform} ROWS) REPEATABLE ({SEED + 1})

        UNION ALL

        SELECT *, 'val' AS split
        FROM (
            SELECT * FROM base
            WHERE created_ts >= TIMESTAMP '{v}' AND created_ts < TIMESTAMP '{t}'
        ) val_pool
        USING SAMPLE reservoir({val_sample_per_platform} ROWS) REPEATABLE ({SEED + 2})

        UNION ALL

        SELECT *, 'test' AS split
        FROM (
            SELECT * FROM base WHERE created_ts >= TIMESTAMP '{t}'
        ) test_pool
        USING SAMPLE reservoir({test_sample_per_platform} ROWS) REPEATABLE ({SEED + 3})
    """
    return con.execute(q).fetchdf()


def load_polymarket_data(
    con: duckdb.DuckDBPyConnection,
    val_start: pd.Timestamp,
    test_start: pd.Timestamp,
    token_resolution_df: pd.DataFrame,
    train_sample_per_platform: int,
    val_sample_per_platform: int,
    test_sample_per_platform: int,
) -> pd.DataFrame:
    stderr("[info] Loading Polymarket sampled splits...")
    con.register("token_resolution_df", token_resolution_df)
    con.execute("CREATE TEMP TABLE token_resolution AS SELECT * FROM token_resolution_df")

    v = val_start.strftime("%Y-%m-%d %H:%M:%S")
    t = test_start.strftime("%Y-%m-%d %H:%M:%S")

    q = f"""
        WITH base AS (
            SELECT
                'polymarket'::VARCHAR AS platform,
                t.market_id::VARCHAR AS event_id,
                t.timestamp::TIMESTAMP AS created_ts,
                CASE
                    WHEN t.signed_flow < 0 THEN t.norm_price
                    ELSE 1.0 - t.norm_price
                END AS implied_prob,
                CASE
                    WHEN t.signed_flow < 0 THEN tr.token_won
                    ELSE 1 - tr.token_won
                END AS won,
                t.volume::DOUBLE / 1e6 AS trade_size_usd,
                COALESCE(t.delta_price::DOUBLE, 0.0) AS delta_prob,
                COALESCE(t.time_since_prev::DOUBLE, 0.0) AS time_since_prev,
                t.cumulative_volume::DOUBLE AS cumulative_volume,
                t.cumulative_net_flow::DOUBLE AS cumulative_net_flow,
                t.cumulative_trade_count::DOUBLE AS cumulative_trade_count,
                t.trade_sequence_num::DOUBLE AS trade_sequence_num,
                t.time_to_expiry_seconds::DOUBLE AS time_to_expiry_seconds,
                EXTRACT('hour' FROM t.timestamp::TIMESTAMP) AS hour_utc,
                EXTRACT('dow' FROM t.timestamp::TIMESTAMP) AS dow_utc
            FROM read_parquet('{POLY_T1A_GLOB}') t
            INNER JOIN token_resolution tr ON t.token_id = tr.token_id
            WHERE t.market_id IS NOT NULL
              AND t.timestamp IS NOT NULL
              AND t.norm_price > 0.0
              AND t.norm_price < 1.0
              AND t.volume > 0
        )
        SELECT *, 'train' AS split
        FROM (
            SELECT * FROM base WHERE created_ts < TIMESTAMP '{v}'
        ) train_pool
        USING SAMPLE reservoir({train_sample_per_platform} ROWS) REPEATABLE ({SEED + 11})

        UNION ALL

        SELECT *, 'val' AS split
        FROM (
            SELECT * FROM base
            WHERE created_ts >= TIMESTAMP '{v}' AND created_ts < TIMESTAMP '{t}'
        ) val_pool
        USING SAMPLE reservoir({val_sample_per_platform} ROWS) REPEATABLE ({SEED + 12})

        UNION ALL

        SELECT *, 'test' AS split
        FROM (
            SELECT * FROM base WHERE created_ts >= TIMESTAMP '{t}'
        ) test_pool
        USING SAMPLE reservoir({test_sample_per_platform} ROWS) REPEATABLE ({SEED + 13})
    """
    return con.execute(q).fetchdf()


def add_engineered_features(df: pd.DataFrame) -> None:
    df["implied_prob"] = pd.to_numeric(df["implied_prob"], errors="coerce").fillna(0.5).clip(0.001, 0.999)
    df["trade_size_usd"] = pd.to_numeric(df["trade_size_usd"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["delta_prob"] = pd.to_numeric(df["delta_prob"], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    df["time_since_prev"] = pd.to_numeric(df["time_since_prev"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["cumulative_volume"] = pd.to_numeric(df["cumulative_volume"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["cumulative_net_flow"] = pd.to_numeric(df["cumulative_net_flow"], errors="coerce").fillna(0.0)
    df["cumulative_trade_count"] = pd.to_numeric(df["cumulative_trade_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["trade_sequence_num"] = pd.to_numeric(df["trade_sequence_num"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["time_to_expiry_seconds"] = pd.to_numeric(df["time_to_expiry_seconds"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["hour_utc"] = pd.to_numeric(df["hour_utc"], errors="coerce").fillna(0.0)
    df["dow_utc"] = pd.to_numeric(df["dow_utc"], errors="coerce").fillna(0.0)

    df["is_polymarket"] = (df["platform"] == "polymarket").astype(np.float32)
    df["log_trade_size_usd"] = np.log1p(df["trade_size_usd"]).astype(np.float32)
    df["log_time_since_prev"] = np.log1p(df["time_since_prev"]).astype(np.float32)
    df["log_cumulative_volume"] = np.log1p(df["cumulative_volume"]).astype(np.float32)
    df["log_cumulative_trades"] = np.log1p(df["cumulative_trade_count"]).astype(np.float32)
    df["tte_hours"] = (df["time_to_expiry_seconds"] / 3600.0).astype(np.float32)

    denom = np.maximum(df["cumulative_volume"].to_numpy(dtype=np.float64), 1.0)
    df["ofi_ratio"] = (df["cumulative_net_flow"].to_numpy(dtype=np.float64) / denom).astype(np.float32)

    hour = df["hour_utc"].to_numpy(dtype=np.float64)
    dow = df["dow_utc"].to_numpy(dtype=np.float64)
    df["hour_sin"] = np.sin(2.0 * math.pi * hour / 24.0).astype(np.float32)
    df["hour_cos"] = np.cos(2.0 * math.pi * hour / 24.0).astype(np.float32)
    df["dow_sin"] = np.sin(2.0 * math.pi * dow / 7.0).astype(np.float32)
    df["dow_cos"] = np.cos(2.0 * math.pi * dow / 7.0).astype(np.float32)

    df["won"] = pd.to_numeric(df["won"], errors="coerce").fillna(0).astype(np.int8)


def attach_t9a_embeddings(
    df: pd.DataFrame,
    use_all_dims: bool = False,
    n_dims: int = EMBED_DIMS,
) -> tuple[pd.DataFrame, list[str], list[int]]:
    stderr("[info] Attaching T9a embeddings (Kalshi + Polymarket)...")
    index_df = pd.read_parquet(T9A_INDEX, columns=["market_id", "platform"]).reset_index()
    index_df = index_df.rename(columns={"index": "emb_row"})
    index_df["key"] = index_df["platform"] + "::" + index_df["market_id"].astype(str)
    key_to_row = index_df.set_index("key")["emb_row"]

    df["event_id"] = df["event_id"].astype(str)
    df["key"] = df["platform"] + "::" + df["event_id"]
    df["emb_row"] = df["key"].map(key_to_row)
    df.drop(columns=["key"], inplace=True)

    emb = np.load(T9A_EMBEDDINGS, mmap_mode="r")
    if use_all_dims:
        top_dims = list(range(int(emb.shape[1])))
        stderr(f"[info] Using all embedding dims (no selection): {len(top_dims)}")
    else:
        train_rows = (
            df.loc[(df["split"] == "train") & df["emb_row"].notna(), "emb_row"]
            .drop_duplicates()
            .astype(np.int64)
            .to_numpy()
        )
        if len(train_rows) == 0:
            raise RuntimeError("No train events mapped to embeddings.")

        rng = np.random.default_rng(SEED)
        if len(train_rows) > EMBED_VAR_SAMPLE_EVENTS:
            sample_rows = rng.choice(train_rows, size=EMBED_VAR_SAMPLE_EVENTS, replace=False)
        else:
            sample_rows = train_rows

        sample_matrix = np.asarray(emb[sample_rows], dtype=np.float32)
        variances = sample_matrix.var(axis=0)
        if n_dims <= 0:
            raise ValueError("n_dims must be positive.")
        if n_dims > variances.shape[0]:
            raise ValueError(f"n_dims={n_dims} exceeds embedding dimension {variances.shape[0]}")
        top_idx = np.argpartition(variances, -n_dims)[-n_dims:]
        top_idx = top_idx[np.argsort(variances[top_idx])[::-1]]
        top_dims = top_idx.tolist()

    all_rows = df["emb_row"].dropna().astype(np.int64).drop_duplicates().to_numpy()
    emb_small = np.empty((len(all_rows), len(top_dims)), dtype=np.float32)
    for j, d in enumerate(top_dims):
        emb_small[:, j] = np.asarray(emb[all_rows, d], dtype=np.float32)

    emb_df = pd.DataFrame({"emb_row": all_rows})
    emb_cols: list[str] = []
    for j in range(len(top_dims)):
        c = f"emb_{j:02d}"
        emb_df[c] = emb_small[:, j]
        emb_cols.append(c)

    df = df.merge(emb_df, on="emb_row", how="left")
    for c in emb_cols:
        df[c] = df[c].fillna(0.0).astype(np.float32)

    coverage = float(df["emb_row"].notna().mean())
    stderr(f"[info] Embedding coverage: {coverage:.2%}")

    df.drop(columns=["emb_row"], inplace=True)
    return df, emb_cols, top_dims


def score_probs(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "auc": float(roc_auc_score(y_true, p)),
        "logloss": float(log_loss(y_true, p)),
        "brier": float(brier_score_loss(y_true, p)),
    }


def edge_stats(arr: np.ndarray) -> dict[str, float]:
    n = int(len(arr))
    if n == 0:
        return {"n": 0, "mean": float("nan"), "se": float("nan")}
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    return {"n": n, "mean": mean, "se": se}


def train_catboost(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str]) -> CatBoostClassifier:
    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(
        train_df[feature_cols].to_numpy(dtype=np.float32, copy=False),
        train_df["won"].to_numpy(dtype=np.int8, copy=False),
        eval_set=(
            val_df[feature_cols].to_numpy(dtype=np.float32, copy=False),
            val_df["won"].to_numpy(dtype=np.int8, copy=False),
        ),
    )
    return model


def evaluate_model(model: CatBoostClassifier, test_df: pd.DataFrame, feature_cols: list[str]) -> dict:
    X = test_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    y = test_df["won"].to_numpy(dtype=np.int8, copy=False)
    implied = test_df["implied_prob"].to_numpy(dtype=np.float64, copy=False)

    p = model.predict_proba(X)[:, 1]

    model_metrics = score_probs(y, p)
    base_metrics = score_probs(y, implied)

    pred_edge = p - implied
    realized_edge = y - implied

    q10 = float(np.quantile(pred_edge, 0.10))
    q90 = float(np.quantile(pred_edge, 0.90))

    top = realized_edge[pred_edge >= q90]
    bottom = realized_edge[pred_edge <= q10]
    long_sig = realized_edge[pred_edge > 0.0]

    return {
        "n": int(len(test_df)),
        "baseline": base_metrics,
        "model": model_metrics,
        "logloss_improvement": float(base_metrics["logloss"] - model_metrics["logloss"]),
        "brier_improvement": float(base_metrics["brier"] - model_metrics["brier"]),
        "overall_edge": edge_stats(realized_edge),
        "top_decile_edge": edge_stats(top),
        "bottom_decile_edge": edge_stats(bottom),
        "long_signal_edge": edge_stats(long_sig),
    }


def make_transfer_figure(results: dict, fig_path: Path, use_text: bool) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    labels = ["on Kalshi", "on Polymarket"]
    x = np.arange(len(labels))
    w = 0.25

    v_combined = [
        results["combined_on_kalshi"]["logloss_improvement"],
        results["combined_on_polymarket"]["logloss_improvement"],
    ]
    v_k = [
        results["kalshi_only_on_kalshi"]["logloss_improvement"],
        results["kalshi_only_on_polymarket"]["logloss_improvement"],
    ]
    v_p = [
        results["poly_only_on_kalshi"]["logloss_improvement"],
        results["poly_only_on_polymarket"]["logloss_improvement"],
    ]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    ax.bar(x - w, v_combined, width=w, label="Train: Combined")
    ax.bar(x, v_k, width=w, label="Train: Kalshi only")
    ax.bar(x + w, v_p, width=w, label="Train: Polymarket only")
    ax.axhline(0.0, color="#444444", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("LogLoss Improvement vs Implied Baseline")
    title = "Cross-Platform Transfer: Improvement on Held-Out Final Period"
    if not use_text:
        title += " (No Text Features)"
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, facecolor="white")
    plt.close(fig)


def write_report(
    boundaries: dict,
    counts: dict,
    results: dict,
    top_dims: list[int],
    train_sample_per_platform: int,
    val_sample_per_platform: int,
    test_sample_per_platform: int,
    report_path: Path,
    metrics_path: Path,
    fig_path: Path,
    use_text: bool,
) -> None:
    def row(name: str, key: str) -> str:
        r = results[key]
        return (
            f"| {name} | {r['n']:,} | "
            f"{r['baseline']['logloss']:.4f} | {r['model']['logloss']:.4f} | "
            f"{r['logloss_improvement']:+.4f} | "
            f"{r['top_decile_edge']['mean'] * 100:+.3f} pp |"
        )

    text_line = (
        "- Existing T9a OpenAI embeddings (`embedding_index.parquet` + `embeddings.npy`)"
        if use_text
        else "- Text embeddings disabled (`--no-text` ablation)"
    )
    embed_line = (
        f"- Embedding dims used (n={len(top_dims)}): `{top_dims}`" if use_text else "- Embedding dims used: `[]`"
    )

    md = f"""# §9.2: Cross-Platform Taker Edge + Transfer (CatBoost)

## Setup

- Same temporal split boundary as §9.1 (from Kalshi max resolved timestamp):
  - Validation start: `{boundaries['val_start']}`
  - Holdout start: `{boundaries['test_start']}`
  - Max timestamp reference: `{boundaries['max_t']}`
- Data:
  - Kalshi + Polymarket resolved trades (taker perspective)
  {text_line}
- Sampling per platform:
  - Train: {train_sample_per_platform:,}
  - Val: {val_sample_per_platform:,}
  - Test: {test_sample_per_platform:,}

## Split Counts

- Kalshi: train={counts['kalshi_train']:,}, val={counts['kalshi_val']:,}, test={counts['kalshi_test']:,}
- Polymarket: train={counts['poly_train']:,}, val={counts['poly_val']:,}, test={counts['poly_test']:,}

## Results (Held-Out Final Period)

| Model / Eval | N | Baseline LogLoss | Model LogLoss | Improvement | Top-Decile Realized Edge |
|---|---:|---:|---:|---:|---:|
{row("Combined -> Kalshi", "combined_on_kalshi")}
{row("Combined -> Polymarket", "combined_on_polymarket")}
{row("Kalshi-only -> Kalshi", "kalshi_only_on_kalshi")}
{row("Kalshi-only -> Polymarket (transfer)", "kalshi_only_on_polymarket")}
{row("Polymarket-only -> Kalshi (transfer)", "poly_only_on_kalshi")}
{row("Polymarket-only -> Polymarket", "poly_only_on_polymarket")}

## Transfer Takeaways

- Kalshi -> Polymarket transfer logloss improvement: `{results['kalshi_only_on_polymarket']['logloss_improvement']:+.4f}`
- Polymarket -> Kalshi transfer logloss improvement: `{results['poly_only_on_kalshi']['logloss_improvement']:+.4f}`
- Combined model improvement on Kalshi: `{results['combined_on_kalshi']['logloss_improvement']:+.4f}`
- Combined model improvement on Polymarket: `{results['combined_on_polymarket']['logloss_improvement']:+.4f}`

## Artifacts

- Metrics JSON: `{metrics_path.relative_to(ROOT)}`
- Figure: `{fig_path.relative_to(ROOT)}`
{embed_line}
"""
    report_path.write_text(md)


def main() -> None:
    args = parse_args()
    if args.no_text and args.full_text_no_selection:
        raise ValueError("Use either --no-text or --full-text-no-selection, not both.")

    use_text = not args.no_text
    use_all_text_dims = args.full_text_no_selection
    if args.n_dims <= 0:
        raise ValueError("--n-dims must be positive.")
    n_dims = int(args.n_dims)

    if not use_text:
        mode = "no_text"
    elif use_all_text_dims:
        mode = "full_text_all_dims"
    else:
        mode = "selected_text"

    if mode == "full_text_all_dims":
        # Full 3072-dim embeddings are memory heavy; use smaller default samples
        # unless explicitly overridden.
        train_sample_per_platform = args.train_sample_per_platform or 60_000
        val_sample_per_platform = args.val_sample_per_platform or 20_000
        test_sample_per_platform = args.test_sample_per_platform or 60_000
    else:
        train_sample_per_platform = args.train_sample_per_platform or TRAIN_SAMPLE_PER_PLATFORM
        val_sample_per_platform = args.val_sample_per_platform or VAL_SAMPLE_PER_PLATFORM
        test_sample_per_platform = args.test_sample_per_platform or TEST_SAMPLE_PER_PLATFORM

    if min(train_sample_per_platform, val_sample_per_platform, test_sample_per_platform) <= 0:
        raise ValueError("All sample sizes must be positive.")

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    report_path, metrics_path, fig_path = artifact_paths(mode, n_dims=n_dims)

    con = duckdb.connect()
    con.execute("SET threads = 8")

    max_t, val_start, test_start = get_kalshi_time_boundaries(con)
    stderr(f"[info] Boundaries from Kalshi max resolved timestamp:")
    stderr(f"       max_t={max_t}")
    stderr(f"       val_start={val_start}")
    stderr(f"       test_start={test_start}")

    token_resolution_df = build_poly_resolution_table(con)
    kalshi_df = load_kalshi_data(
        con,
        val_start=val_start,
        test_start=test_start,
        train_sample_per_platform=train_sample_per_platform,
        val_sample_per_platform=val_sample_per_platform,
        test_sample_per_platform=test_sample_per_platform,
    )
    poly_df = load_polymarket_data(
        con,
        val_start=val_start,
        test_start=test_start,
        token_resolution_df=token_resolution_df,
        train_sample_per_platform=train_sample_per_platform,
        val_sample_per_platform=val_sample_per_platform,
        test_sample_per_platform=test_sample_per_platform,
    )
    con.close()

    if kalshi_df.empty or poly_df.empty:
        raise RuntimeError("One of the platform datasets is empty.")

    df = pd.concat([kalshi_df, poly_df], ignore_index=True)
    add_engineered_features(df)
    if use_text:
        df, emb_cols, top_dims = attach_t9a_embeddings(
            df,
            use_all_dims=use_all_text_dims,
            n_dims=n_dims,
        )
    else:
        stderr("[info] Running ablation with no text embeddings.")
        emb_cols = []
        top_dims = []

    feature_cols = [
        "implied_prob",
        "is_polymarket",
        "log_trade_size_usd",
        "delta_prob",
        "log_time_since_prev",
        "log_cumulative_volume",
        "log_cumulative_trades",
        "trade_sequence_num",
        "tte_hours",
        "ofi_ratio",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ] + emb_cols

    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32)

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()

    train_k = train[train["platform"] == "kalshi"].copy()
    train_p = train[train["platform"] == "polymarket"].copy()
    val_k = val[val["platform"] == "kalshi"].copy()
    val_p = val[val["platform"] == "polymarket"].copy()
    test_k = test[test["platform"] == "kalshi"].copy()
    test_p = test[test["platform"] == "polymarket"].copy()

    counts = {
        "kalshi_train": int(len(train_k)),
        "kalshi_val": int(len(val_k)),
        "kalshi_test": int(len(test_k)),
        "poly_train": int(len(train_p)),
        "poly_val": int(len(val_p)),
        "poly_test": int(len(test_p)),
    }
    stderr(f"[info] Counts: {counts}")

    if min(counts.values()) <= 0:
        raise RuntimeError("At least one required split/platform bucket is empty.")

    stderr("[info] Training combined model...")
    model_combined = train_catboost(train, val, feature_cols)

    stderr("[info] Training Kalshi-only model...")
    model_k = train_catboost(train_k, val_k, feature_cols)

    stderr("[info] Training Polymarket-only model...")
    model_p = train_catboost(train_p, val_p, feature_cols)

    stderr("[info] Evaluating holdout/transfer...")
    results = {
        "combined_on_kalshi": evaluate_model(model_combined, test_k, feature_cols),
        "combined_on_polymarket": evaluate_model(model_combined, test_p, feature_cols),
        "kalshi_only_on_kalshi": evaluate_model(model_k, test_k, feature_cols),
        "kalshi_only_on_polymarket": evaluate_model(model_k, test_p, feature_cols),
        "poly_only_on_kalshi": evaluate_model(model_p, test_k, feature_cols),
        "poly_only_on_polymarket": evaluate_model(model_p, test_p, feature_cols),
    }

    payload = {
        "config": {
            "seed": SEED,
            "holdout_days": HOLDOUT_DAYS,
            "val_days": VAL_DAYS,
            "train_sample_per_platform": train_sample_per_platform,
            "val_sample_per_platform": val_sample_per_platform,
            "test_sample_per_platform": test_sample_per_platform,
            "embed_dims": len(top_dims) if use_text else 0,
            "requested_n_dims": n_dims,
            "embed_dims_selected": top_dims,
            "use_text_features": use_text,
            "use_all_text_dims": use_all_text_dims,
            "catboost_params": CATBOOST_PARAMS,
            "boundaries": {
                "max_t": str(max_t),
                "val_start": str(val_start),
                "test_start": str(test_start),
            },
        },
        "counts": counts,
        "results": results,
    }
    metrics_path.write_text(json.dumps(payload, indent=2))

    make_transfer_figure(results, fig_path=fig_path, use_text=use_text)
    write_report(
        boundaries={"max_t": str(max_t), "val_start": str(val_start), "test_start": str(test_start)},
        counts=counts,
        results=results,
        top_dims=top_dims,
        train_sample_per_platform=train_sample_per_platform,
        val_sample_per_platform=val_sample_per_platform,
        test_sample_per_platform=test_sample_per_platform,
        report_path=report_path,
        metrics_path=metrics_path,
        fig_path=fig_path,
        use_text=use_text,
    )

    elapsed = time.time() - t0
    stderr(f"[done] Wrote {metrics_path}")
    stderr(f"[done] Wrote {fig_path}")
    stderr(f"[done] Wrote {report_path}")
    stderr(f"[done] Total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
