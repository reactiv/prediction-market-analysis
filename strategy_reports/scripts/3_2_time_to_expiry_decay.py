"""SS3.2 Time-to-Expiry Decay -- calibration quality vs. time remaining until market close."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import duckdb  # noqa: E402
import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
TRADES_DIR = BASE_DIR / "data" / "kalshi" / "trades"
MARKETS_DIR = BASE_DIR / "data" / "kalshi" / "markets"
FIG_DIR = BASE_DIR / "strategy_reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Time bucket definitions (order matters for plotting)
# ---------------------------------------------------------------------------
BUCKET_ORDER = {
    "0-1h": 1,
    "1-6h": 2,
    "6-24h": 3,
    "1-3d": 4,
    "3-7d": 5,
    "7d+": 6,
}
BUCKET_LABELS = ["0-1h", "1-6h", "6-24h", "1-3d", "3-7d", "7d+"]

# Sequential blue palette: dark (close to expiry) -> light (far from expiry)
# Index 0 = 0-1h (darkest), index 5 = 7d+ (lightest)
BLUE_PALETTE = [
    "#08306b",  # 0-1h  -- very dark navy
    "#2171b5",  # 1-6h
    "#4292c6",  # 6-24h
    "#6baed6",  # 1-3d
    "#9ecae1",  # 3-7d
    "#c6dbef",  # 7d+   -- very light blue
]

# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------
def load_data() -> "duckdb.DuckDBPyRelation":
    """Run the main query and return a DataFrame with per-bucket, per-price stats."""
    con = duckdb.connect()
    sql = f"""
    WITH resolved_markets AS (
        SELECT ticker, result, close_time
        FROM '{MARKETS_DIR}/*.parquet'
        WHERE status = 'finalized'
          AND result IN ('yes', 'no')
          AND close_time IS NOT NULL
    ),
    trade_data AS (
        SELECT
            t.created_time,
            m.close_time,
            EXTRACT(EPOCH FROM (m.close_time - t.created_time)) / 3600.0 AS hours_to_expiry,
            CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS price,
            CASE WHEN t.taker_side = m.result THEN 1.0 ELSE 0.0 END AS won,
            t.count AS contracts
        FROM '{TRADES_DIR}/*.parquet' t
        INNER JOIN resolved_markets m ON t.ticker = m.ticker
    ),
    bucketed AS (
        SELECT
            CASE
                WHEN hours_to_expiry <= 1 THEN '0-1h'
                WHEN hours_to_expiry <= 6 THEN '1-6h'
                WHEN hours_to_expiry <= 24 THEN '6-24h'
                WHEN hours_to_expiry <= 72 THEN '1-3d'
                WHEN hours_to_expiry <= 168 THEN '3-7d'
                ELSE '7d+'
            END AS time_bucket,
            CASE
                WHEN hours_to_expiry <= 1 THEN 1
                WHEN hours_to_expiry <= 6 THEN 2
                WHEN hours_to_expiry <= 24 THEN 3
                WHEN hours_to_expiry <= 72 THEN 4
                WHEN hours_to_expiry <= 168 THEN 5
                ELSE 6
            END AS bucket_order,
            price,
            won,
            contracts
        FROM trade_data
        WHERE hours_to_expiry >= 0
    )
    SELECT
        time_bucket,
        bucket_order,
        price,
        AVG(won) AS win_rate,
        price / 100.0 AS expected_win_rate,
        AVG(won) - price / 100.0 AS excess_return,
        COUNT(*) AS n_trades,
        SUM(contracts) AS total_contracts
    FROM bucketed
    GROUP BY time_bucket, bucket_order, price
    HAVING COUNT(*) >= 100
    ORDER BY bucket_order, price
    """
    df = con.execute(sql).fetchdf()
    con.close()
    return df


# ---------------------------------------------------------------------------
# Compute calibration curves (5-cent buckets for smoothing)
# ---------------------------------------------------------------------------
def compute_calibration_curves(df) -> dict:
    """
    For each time bucket, aggregate price-level data into 5-cent bins.
    Returns {bucket_label: [(midpoint, win_rate, n_trades, total_contracts), ...]}
    """
    curves = {}
    for bucket in BUCKET_LABELS:
        bdf = df[df["time_bucket"] == bucket].copy()
        if bdf.empty:
            continue

        # Create 5-cent bins
        bdf["bin"] = (bdf["price"] // 5) * 5 + 2.5
        # Weighted average by n_trades within each bin
        grouped = bdf.groupby("bin").apply(
            lambda g: {
                "win_rate": (g["win_rate"] * g["n_trades"]).sum() / g["n_trades"].sum(),
                "n_trades": g["n_trades"].sum(),
                "total_contracts": g["total_contracts"].sum(),
            },
            include_groups=False,
        )
        points = []
        for mid, stats in sorted(grouped.items()):
            if stats["n_trades"] >= 100:
                points.append((mid, stats["win_rate"], int(stats["n_trades"]), int(stats["total_contracts"])))
        curves[bucket] = points
    return curves


# ---------------------------------------------------------------------------
# Compute MAE per time bucket
# ---------------------------------------------------------------------------
def compute_mae_per_bucket(df) -> dict:
    """
    Mean absolute error = mean(|win_rate - price/100|) weighted by n_trades
    per time bucket.
    Returns {bucket_label: {mae, n_trades, total_contracts, n_price_points}}
    """
    results = {}
    for bucket in BUCKET_LABELS:
        bdf = df[df["time_bucket"] == bucket]
        if bdf.empty:
            continue
        # Weighted MAE
        abs_err = (bdf["win_rate"] - bdf["expected_win_rate"]).abs()
        w = bdf["n_trades"]
        mae = (abs_err * w).sum() / w.sum()
        results[bucket] = {
            "mae": float(mae),
            "mae_pp": round(float(mae) * 100, 2),  # percentage points
            "n_trades": int(bdf["n_trades"].sum()),
            "total_contracts": int(bdf["total_contracts"].sum()),
            "n_price_points": len(bdf),
        }
    return results


# ---------------------------------------------------------------------------
# Figure 1: Calibration curves per time bucket
# ---------------------------------------------------------------------------
def plot_calibration_curves(curves: dict) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")

    # Perfect calibration line
    ax.plot(
        [0, 100], [0, 100],
        linestyle="--", color="gray", linewidth=1.5,
        label="Perfect calibration", zorder=1,
    )

    # Plot each time bucket with appropriate color
    for i, bucket in enumerate(BUCKET_LABELS):
        if bucket not in curves:
            continue
        pts = curves[bucket]
        if not pts:
            continue
        midpoints = [p[0] for p in pts]
        win_rates = [p[1] * 100 for p in pts]  # convert to percentage
        color = BLUE_PALETTE[i]
        ax.plot(
            midpoints, win_rates,
            color=color, linewidth=2.0, alpha=0.9,
            label=bucket, zorder=2 + i,
        )
        ax.scatter(
            midpoints, win_rates,
            color=color, s=30, zorder=3 + len(BUCKET_LABELS) + i,
            edgecolors="white", linewidth=0.4,
        )

    ax.set_xlabel("Contract Price (cents)", fontsize=12)
    ax.set_ylabel("Win Rate (%)", fontsize=12)
    ax.set_title(
        "Calibration by Time to Expiry",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10, loc="upper left", title="Time to Expiry")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = FIG_DIR / "3_2_calibration_by_time_bucket.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Figure 2: MAE bar chart
# ---------------------------------------------------------------------------
def plot_mae_bars(mae_data: dict) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")

    buckets_present = [b for b in BUCKET_LABELS if b in mae_data]
    maes = [mae_data[b]["mae_pp"] for b in buckets_present]
    colors = [BLUE_PALETTE[BUCKET_LABELS.index(b)] for b in buckets_present]

    bars = ax.bar(range(len(buckets_present)), maes, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(range(len(buckets_present)))
    ax.set_xticklabels(buckets_present, fontsize=11)

    # Add value labels on bars
    for bar, mae_val in zip(bars, maes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{mae_val:.2f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_xlabel("Time to Expiry", fontsize=12)
    ax.set_ylabel("Mean Absolute Calibration Error (pp)", fontsize=12)
    ax.set_title(
        "Calibration Error by Time to Expiry",
        fontsize=14, fontweight="bold",
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    path = FIG_DIR / "3_2_mae_by_time_bucket.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Build JSON summary
# ---------------------------------------------------------------------------
def build_summary(df, curves: dict, mae_data: dict) -> dict:
    """Assemble summary statistics for stdout JSON output."""

    # Per-bucket summary
    bucket_summary = []
    for bucket in BUCKET_LABELS:
        if bucket not in mae_data:
            continue
        m = mae_data[bucket]
        bucket_summary.append({
            "time_bucket": bucket,
            "mae_pp": m["mae_pp"],
            "n_trades": m["n_trades"],
            "total_contracts": m["total_contracts"],
            "n_price_points": m["n_price_points"],
        })

    # Overall stats
    total_trades = int(df["n_trades"].sum())
    total_contracts = int(df["total_contracts"].sum())

    # Improvement ratio: MAE(7d+) / MAE(0-1h)
    if "7d+" in mae_data and "0-1h" in mae_data:
        improvement_ratio = mae_data["7d+"]["mae_pp"] / mae_data["0-1h"]["mae_pp"]
    else:
        improvement_ratio = None

    # Most/least calibrated buckets
    if mae_data:
        best_bucket = min(mae_data, key=lambda b: mae_data[b]["mae_pp"])
        worst_bucket = max(mae_data, key=lambda b: mae_data[b]["mae_pp"])
    else:
        best_bucket = worst_bucket = None

    summary = {
        "total_trades": total_trades,
        "total_contracts": total_contracts,
        "n_time_buckets": len(bucket_summary),
        "best_calibrated_bucket": best_bucket,
        "best_calibrated_mae_pp": mae_data[best_bucket]["mae_pp"] if best_bucket else None,
        "worst_calibrated_bucket": worst_bucket,
        "worst_calibrated_mae_pp": mae_data[worst_bucket]["mae_pp"] if worst_bucket else None,
        "far_to_close_mae_ratio": round(improvement_ratio, 3) if improvement_ratio else None,
        "bucket_summary": bucket_summary,
    }
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading trade data with time-to-expiry buckets...", file=sys.stderr)
    df = load_data()
    print(f"  Rows (bucket x price combos with >=100 trades): {len(df)}", file=sys.stderr)
    print(f"  Time buckets found: {sorted(df['time_bucket'].unique().tolist())}", file=sys.stderr)

    print("Computing calibration curves...", file=sys.stderr)
    curves = compute_calibration_curves(df)
    for b in BUCKET_LABELS:
        if b in curves:
            print(f"  {b}: {len(curves[b])} price bins", file=sys.stderr)

    print("Computing MAE per time bucket...", file=sys.stderr)
    mae_data = compute_mae_per_bucket(df)
    for b in BUCKET_LABELS:
        if b in mae_data:
            print(f"  {b}: MAE = {mae_data[b]['mae_pp']:.2f} pp  ({mae_data[b]['n_trades']:,} trades)", file=sys.stderr)

    print("Plotting calibration curves...", file=sys.stderr)
    plot_calibration_curves(curves)

    print("Plotting MAE bar chart...", file=sys.stderr)
    plot_mae_bars(mae_data)

    print("Building summary...", file=sys.stderr)
    summary = build_summary(df, curves, mae_data)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
