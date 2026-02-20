#!/usr/bin/env python3
"""
Strategy Report 9.3: Polymarket-only rolling weekly holdout validation.

Motivation:
- Check whether low top-decile performance in a single week is regime noise
  or persistent weakness.

Design:
- Polymarket-only taker model.
- Consecutive weekly test windows.
- For each week:
  - train: all history before validation week
  - val: week before test week
  - test: target week
- Uses T9a OpenAI embeddings for polymarket market_id titles.
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

ROOT = Path(__file__).resolve().parents[2]

POLY_T1A_GLOB = str(ROOT / "data" / "transforms" / "t1a" / "polymarket" / "**" / "*.parquet")
POLY_MARKETS_GLOB = str(ROOT / "data" / "polymarket" / "markets" / "*.parquet")
KALSHI_T1A_GLOB = str(ROOT / "data" / "transforms" / "t1a" / "kalshi" / "*.parquet")
KALSHI_MARKETS_GLOB = str(ROOT / "data" / "kalshi" / "markets" / "*.parquet")

T9A_INDEX = ROOT / "data" / "transforms" / "t9a" / "embedding_index.parquet"
T9A_EMBEDDINGS = ROOT / "data" / "transforms" / "t9a" / "embeddings.npy"

OUT_DIR = ROOT / "output"
FIG_DIR = ROOT / "strategy_reports" / "figures"

CATBOOST_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "iterations": 600,
    "learning_rate": 0.08,
    "depth": 7,
    "l2_leaf_reg": 6.0,
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
    p = argparse.ArgumentParser(description="Polymarket rolling weekly validation.")
    p.add_argument("--weeks", type=int, default=6, help="Number of consecutive weekly holdouts.")
    p.add_argument("--n-dims", type=int, default=64, help="Number of selected embedding dimensions.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-sample", type=int, default=1_500_000)
    p.add_argument("--val-sample", type=int, default=250_000)
    p.add_argument("--test-sample", type=int, default=750_000)
    return p.parse_args()


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


def get_kalshi_anchor_max_t(con: duckdb.DuckDBPyConnection) -> pd.Timestamp:
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
        raise RuntimeError("Could not compute Kalshi anchor max timestamp.")
    return pd.Timestamp(max_t)


def build_poly_resolution_table(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    stderr("[info] Building Polymarket token resolution table...")
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

    df = pd.DataFrame(rows, columns=["token_id", "token_won"]).drop_duplicates(subset=["token_id"])
    if df.empty:
        raise RuntimeError("No resolved tokens in Polymarket resolution table.")
    stderr(f"[info] Resolved tokens: {len(df):,}")
    return df


def load_week_split(
    con: duckdb.DuckDBPyConnection,
    token_resolution_df: pd.DataFrame,
    val_start: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    seed: int,
    train_sample: int,
    val_sample: int,
    test_sample: int,
) -> pd.DataFrame:
    con.register("token_resolution_df", token_resolution_df)
    con.execute("CREATE TEMP TABLE token_resolution AS SELECT * FROM token_resolution_df")

    v = val_start.strftime("%Y-%m-%d %H:%M:%S")
    t = test_start.strftime("%Y-%m-%d %H:%M:%S")
    e = test_end.strftime("%Y-%m-%d %H:%M:%S")

    q = f"""
        WITH base AS (
            SELECT
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
              AND t.timestamp < TIMESTAMP '{e}'
        )
        SELECT *, 'train' AS split
        FROM (
            SELECT * FROM base WHERE created_ts < TIMESTAMP '{v}'
        ) train_pool
        USING SAMPLE reservoir({train_sample} ROWS) REPEATABLE ({seed + 11})

        UNION ALL

        SELECT *, 'val' AS split
        FROM (
            SELECT * FROM base
            WHERE created_ts >= TIMESTAMP '{v}' AND created_ts < TIMESTAMP '{t}'
        ) val_pool
        USING SAMPLE reservoir({val_sample} ROWS) REPEATABLE ({seed + 12})

        UNION ALL

        SELECT *, 'test' AS split
        FROM (
            SELECT * FROM base
            WHERE created_ts >= TIMESTAMP '{t}' AND created_ts < TIMESTAMP '{e}'
        ) test_pool
        USING SAMPLE reservoir({test_sample} ROWS) REPEATABLE ({seed + 13})
    """
    out = con.execute(q).fetchdf()
    con.execute("DROP TABLE token_resolution")
    return out


def add_features(df: pd.DataFrame) -> None:
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


def attach_embeddings(df: pd.DataFrame, n_dims: int, seed: int) -> tuple[pd.DataFrame, list[str], list[int]]:
    index_df = pd.read_parquet(T9A_INDEX, columns=["market_id", "platform"]).reset_index()
    index_df = index_df.rename(columns={"index": "emb_row"})
    index_df = index_df[index_df["platform"] == "polymarket"].copy()
    id_to_row = dict(zip(index_df["market_id"].astype(str), index_df["emb_row"]))

    df["event_id"] = df["event_id"].astype(str)
    df["emb_row"] = df["event_id"].map(id_to_row)

    emb = np.load(T9A_EMBEDDINGS, mmap_mode="r")

    train_rows = (
        df.loc[(df["split"] == "train") & df["emb_row"].notna(), "emb_row"]
        .drop_duplicates()
        .astype(np.int64)
        .to_numpy()
    )
    if len(train_rows) == 0:
        raise RuntimeError("No train events mapped to embeddings for this fold.")

    rng = np.random.default_rng(seed)
    sample_n = min(len(train_rows), 80_000)
    sample_rows = rng.choice(train_rows, size=sample_n, replace=False) if sample_n < len(train_rows) else train_rows
    sample_matrix = np.asarray(emb[sample_rows], dtype=np.float32)
    variances = sample_matrix.var(axis=0)
    if n_dims > variances.shape[0]:
        raise ValueError(f"n_dims={n_dims} exceeds embedding size {variances.shape[0]}")

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
    df.drop(columns=["emb_row"], inplace=True)
    return df, emb_cols, top_dims


def train_and_eval(
    df: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
) -> dict:
    params = dict(CATBOOST_PARAMS)
    params["random_seed"] = seed
    model = CatBoostClassifier(**params)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError("At least one split empty in fold.")

    model.fit(
        train_df[feature_cols].to_numpy(dtype=np.float32, copy=False),
        train_df["won"].to_numpy(dtype=np.int8, copy=False),
        eval_set=(
            val_df[feature_cols].to_numpy(dtype=np.float32, copy=False),
            val_df["won"].to_numpy(dtype=np.int8, copy=False),
        ),
    )

    X = test_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    y = test_df["won"].to_numpy(dtype=np.int8, copy=False)
    implied = test_df["implied_prob"].to_numpy(dtype=np.float64, copy=False)
    p = model.predict_proba(X)[:, 1]

    base_metrics = score_probs(y, implied)
    model_metrics = score_probs(y, p)

    pred_edge = p - implied
    realized_edge = y - implied

    q10 = float(np.quantile(pred_edge, 0.10))
    q90 = float(np.quantile(pred_edge, 0.90))

    top = realized_edge[pred_edge >= q90]
    bottom = realized_edge[pred_edge <= q10]
    long_sig = realized_edge[pred_edge > 0]

    return {
        "counts": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "baseline": base_metrics,
        "model": model_metrics,
        "logloss_improvement": float(base_metrics["logloss"] - model_metrics["logloss"]),
        "brier_improvement": float(base_metrics["brier"] - model_metrics["brier"]),
        "overall_edge": edge_stats(realized_edge),
        "top_decile_edge": edge_stats(top),
        "bottom_decile_edge": edge_stats(bottom),
        "long_signal_edge": edge_stats(long_sig),
    }


def write_outputs(
    args: argparse.Namespace,
    anchor_max_t: pd.Timestamp,
    fold_rows: list[dict],
) -> tuple[Path, Path, Path, Path]:
    suffix = f"_w{args.weeks}_d{args.n_dims}"
    metrics_path = OUT_DIR / f"polymarket_rolling_weekly_metrics{suffix}.json"
    csv_path = OUT_DIR / f"polymarket_rolling_weekly_summary{suffix}.csv"
    fig_path = FIG_DIR / f"9_3_polymarket_rolling_top_decile{suffix}.png"
    report_path = ROOT / "strategy_reports" / f"9_3_polymarket_rolling_weekly{suffix}.md"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(fold_rows)
    df.to_csv(csv_path, index=False)

    payload = {
        "config": {
            "weeks": args.weeks,
            "n_dims": args.n_dims,
            "seed": args.seed,
            "train_sample": args.train_sample,
            "val_sample": args.val_sample,
            "test_sample": args.test_sample,
            "anchor_max_t": str(anchor_max_t),
        },
        "folds": fold_rows,
    }
    metrics_path.write_text(json.dumps(payload, indent=2))

    # Plot top-decile edge across weeks
    dplot = df.copy()
    dplot["test_start"] = pd.to_datetime(dplot["test_start"])
    dplot = dplot.sort_values("test_start")
    fig, ax = plt.subplots(figsize=(11, 5), facecolor="white")
    ax.plot(dplot["test_start"], dplot["top_decile_edge_pp"], marker="o", lw=2)
    ax.axhline(0.0, color="#444444", ls="--", lw=1)
    ax.set_ylabel("Top-Decile Realized Edge (pp)")
    ax.set_title("Polymarket Rolling Weekly Holdouts: Top-Decile Edge")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, facecolor="white")
    plt.close(fig)

    # Report
    pos_top = int((df["top_decile_edge_pp"] > 0).sum())
    pos_logloss = int((df["logloss_improvement"] > 0).sum())
    md = f"""# §9.3: Polymarket Rolling Weekly Holdouts (n_dims={args.n_dims})

Anchor max timestamp (Kalshi-aligned): `{anchor_max_t}`

Weekly folds: `{args.weeks}`

Samples per fold:
- train: {args.train_sample:,}
- val: {args.val_sample:,}
- test: {args.test_sample:,}

## Fold Results

| Test Start | Test End | LogLoss Improvement | AUC (Model) | Top-Decile Edge |
|---|---|---:|---:|---:|
"""
    for _, r in df.sort_values("test_start").iterrows():
        md += (
            f"| {r['test_start']} | {r['test_end']} | "
            f"{r['logloss_improvement']:+.4f} | {r['model_auc']:.4f} | "
            f"{r['top_decile_edge_pp']:+.3f} pp |\n"
        )

    md += f"""

## Summary

- Top-decile edge positive in `{pos_top}/{len(df)}` weeks
- LogLoss improvement positive in `{pos_logloss}/{len(df)}` weeks
- Mean top-decile edge: `{df['top_decile_edge_pp'].mean():+.3f} pp`
- Std top-decile edge: `{df['top_decile_edge_pp'].std(ddof=1):.3f} pp`
- Mean logloss improvement: `{df['logloss_improvement'].mean():+.4f}`

## Artifacts

- JSON: `{metrics_path.relative_to(ROOT)}`
- CSV: `{csv_path.relative_to(ROOT)}`
- Figure: `{fig_path.relative_to(ROOT)}`
"""
    report_path.write_text(md)
    return metrics_path, csv_path, fig_path, report_path


def main() -> None:
    args = parse_args()
    if args.weeks <= 0:
        raise ValueError("--weeks must be > 0")
    if args.n_dims <= 0:
        raise ValueError("--n-dims must be > 0")

    t0 = time.time()
    con = duckdb.connect()
    con.execute("SET threads = 8")

    anchor_max_t = get_kalshi_anchor_max_t(con)
    stderr(f"[info] Anchor max_t: {anchor_max_t}")
    token_resolution_df = build_poly_resolution_table(con)

    fold_rows: list[dict] = []
    for i in range(args.weeks):
        test_end = anchor_max_t - pd.Timedelta(days=7 * i)
        test_start = test_end - pd.Timedelta(days=7)
        val_start = test_start - pd.Timedelta(days=7)

        stderr(
            f"[info] Fold {i + 1}/{args.weeks}: "
            f"val [{val_start} -> {test_start}), test [{test_start} -> {test_end})"
        )

        df = load_week_split(
            con=con,
            token_resolution_df=token_resolution_df,
            val_start=val_start,
            test_start=test_start,
            test_end=test_end,
            seed=args.seed + i * 1000,
            train_sample=args.train_sample,
            val_sample=args.val_sample,
            test_sample=args.test_sample,
        )

        add_features(df)
        df, emb_cols, top_dims = attach_embeddings(df, n_dims=args.n_dims, seed=args.seed + i * 2000)

        feature_cols = [
            "implied_prob",
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

        fold_result = train_and_eval(df, feature_cols=feature_cols, seed=args.seed + i)
        fold_rows.append(
            {
                "fold": i + 1,
                "test_start": str(test_start),
                "test_end": str(test_end),
                "train_n": fold_result["counts"]["train"],
                "val_n": fold_result["counts"]["val"],
                "test_n": fold_result["counts"]["test"],
                "baseline_logloss": fold_result["baseline"]["logloss"],
                "model_logloss": fold_result["model"]["logloss"],
                "logloss_improvement": fold_result["logloss_improvement"],
                "model_auc": fold_result["model"]["auc"],
                "top_decile_edge_pp": fold_result["top_decile_edge"]["mean"] * 100.0,
                "top_decile_n": fold_result["top_decile_edge"]["n"],
                "long_signal_edge_pp": fold_result["long_signal_edge"]["mean"] * 100.0,
                "embed_dims_used": len(top_dims),
            }
        )

    con.close()

    metrics_path, csv_path, fig_path, report_path = write_outputs(args=args, anchor_max_t=anchor_max_t, fold_rows=fold_rows)

    elapsed = time.time() - t0
    stderr(f"[done] Wrote {metrics_path}")
    stderr(f"[done] Wrote {csv_path}")
    stderr(f"[done] Wrote {fig_path}")
    stderr(f"[done] Wrote {report_path}")
    stderr(f"[done] Total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
