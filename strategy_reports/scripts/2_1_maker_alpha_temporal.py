#!/usr/bin/env python3
"""
Strategy Report 2.1 (Temporal Extension): Maker Alpha by Quarter

Time-stratifies the aggregate +1.12pp maker excess return from the main 2.1
analysis into quarterly buckets to determine whether the edge is stable,
growing, or decaying over the 3-year sample period.

Self-contained: no src.* imports.
"""

import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import duckdb  # noqa: E402

# ── Palette & style ──────────────────────────────────────────────────────────
GREEN = "#2ecc71"
RED = "#e74c3c"
FIG_SIZE = (12, 7)
DPI = 150
GRID_ALPHA = 0.3

TRADES_GLOB = "data/kalshi/trades/*.parquet"
MARKETS_GLOB = "data/kalshi/markets/*.parquet"
FIG_DIR = "strategy_reports/figures"


def main() -> None:
    t0 = time.time()
    con = duckdb.connect()

    print("[info] Running quarterly maker/taker excess return query...", file=sys.stderr)

    sql = f"""
    WITH resolved_markets AS (
        SELECT ticker, result
        FROM '{MARKETS_GLOB}'
        WHERE status = 'finalized'
          AND result IN ('yes', 'no')
    ),
    trade_data AS (
        SELECT
            DATE_TRUNC('quarter', t.created_time) AS quarter,
            -- Taker excess return
            CASE WHEN t.taker_side = m.result THEN 1.0 ELSE 0.0 END
                - (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0
                AS taker_excess,
            -- Maker excess return
            CASE WHEN t.taker_side != m.result THEN 1.0 ELSE 0.0 END
                - (CASE WHEN t.taker_side = 'yes' THEN t.no_price ELSE t.yes_price END) / 100.0
                AS maker_excess,
            -- Volume (taker side)
            t.count * (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0
                AS taker_volume_usd,
            t.count AS contracts
        FROM '{TRADES_GLOB}' t
        INNER JOIN resolved_markets m ON t.ticker = m.ticker
    )
    SELECT
        quarter,
        AVG(maker_excess) AS maker_excess_return,
        AVG(taker_excess) AS taker_excess_return,
        AVG(maker_excess) - AVG(taker_excess) AS maker_taker_gap,
        COUNT(*) AS n_trades,
        SUM(taker_volume_usd) AS total_volume,
        SUM(contracts) AS total_contracts
    FROM trade_data
    GROUP BY quarter
    ORDER BY quarter
    """

    df = con.execute(sql).fetchdf()
    elapsed = time.time() - t0
    print(f"[info] Query done in {elapsed:.1f}s, {len(df)} quarters", file=sys.stderr)

    if df.empty:
        print(json.dumps({"error": "No qualifying trades found"}))
        sys.exit(1)

    # Filter out quarters with negligible trade counts (e.g., 1-trade Q2 2021)
    df = df[df["n_trades"] >= 1000].reset_index(drop=True)
    print(f"[info] {len(df)} quarters after filtering (n_trades >= 1000)", file=sys.stderr)

    # Convert to pp (percentage points) for readability
    df["maker_excess_pp"] = df["maker_excess_return"] * 100
    df["taker_excess_pp"] = df["taker_excess_return"] * 100
    df["gap_pp"] = df["maker_taker_gap"] * 100

    # Build human-readable quarter labels (e.g., "2023-Q1")
    import pandas as pd
    df["quarter_label"] = df["quarter"].apply(
        lambda x: f"{x.year}-Q{(x.month - 1) // 3 + 1}"
        if hasattr(x, "year") else str(x)[:10]
    )

    # Print summary to stderr
    for _, row in df.iterrows():
        print(
            f"  {row['quarter_label']:12s}  maker={row['maker_excess_pp']:+.2f}pp  "
            f"taker={row['taker_excess_pp']:+.2f}pp  gap={row['gap_pp']:.2f}pp  "
            f"trades={int(row['n_trades']):,}",
            file=sys.stderr,
        )

    # ── Figure: Quarterly maker vs taker excess return ───────────────────────
    print("[info] Creating temporal figure...", file=sys.stderr)

    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor="white")

    import numpy as np
    x_pos = np.arange(len(df))
    quarter_labels = df["quarter_label"].tolist()
    maker_vals = df["maker_excess_pp"].values
    taker_vals = df["taker_excess_pp"].values

    ax.plot(x_pos, maker_vals, color=GREEN, marker="o", markersize=7,
            linewidth=2.5, label="Maker Excess Return", zorder=3)
    ax.plot(x_pos, taker_vals, color=RED, marker="s", markersize=7,
            linewidth=2.5, label="Taker Excess Return", zorder=3)

    # Zero line
    ax.axhline(0, color="gray", linewidth=1.2, linestyle="--", alpha=0.7, zorder=2)

    # Shade maker region
    ax.fill_between(x_pos, maker_vals, 0, alpha=0.10, color=GREEN)
    ax.fill_between(x_pos, taker_vals, 0, alpha=0.10, color=RED)

    # Labels and formatting
    ax.set_xlabel("Quarter", fontsize=12)
    ax.set_ylabel("Excess Return (pp)", fontsize=12)

    total_trades = int(df["n_trades"].sum())
    ax.set_title(
        "Maker vs Taker Excess Return by Quarter\n"
        f"(N={total_trades:,} resolved Kalshi trades)",
        fontsize=13, fontweight="bold",
    )

    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=GRID_ALPHA)

    # Format x-axis with quarter labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(quarter_labels, rotation=45, ha="right", fontsize=9)

    fig.tight_layout()

    fig_path = f"{FIG_DIR}/2_1_maker_alpha_temporal.png"
    fig.savefig(fig_path, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"[info] Saved {fig_path}", file=sys.stderr)

    # ── Stdout JSON ──────────────────────────────────────────────────────────
    total_elapsed = time.time() - t0

    quarterly_data = []
    for _, row in df.iterrows():
        quarterly_data.append({
            "quarter": row["quarter_label"],
            "maker_excess_pp": round(float(row["maker_excess_pp"]), 4),
            "taker_excess_pp": round(float(row["taker_excess_pp"]), 4),
            "gap_pp": round(float(row["gap_pp"]), 4),
            "n_trades": int(row["n_trades"]),
            "total_volume": round(float(row["total_volume"]), 2),
            "total_contracts": int(row["total_contracts"]),
        })

    output = {
        "n_quarters": len(quarterly_data),
        "n_trades_total": total_trades,
        "elapsed_seconds": round(total_elapsed, 1),
        "quarterly_data": quarterly_data,
        "figure": fig_path,
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
