#!/usr/bin/env python3
"""
Strategy Report 6.3 (Temporal Extension): YES Bias by Quarter

Time-stratifies the aggregate YES/NO ratio and excess return gap from the
main 6.3 analysis into quarterly buckets to determine whether the YES bias
is persistent, growing, or decaying over the sample period (Jul 2021 - May 2025).

Self-contained: no src.* imports.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import duckdb  # noqa: E402
import numpy as np  # noqa: E402

# -- Palette & style ----------------------------------------------------------
BLUE = "#4C72B0"
RED = "#e74c3c"
GREEN = "#2ecc71"
ORANGE = "#ff7f0e"
FIG_SIZE = (16, 7)
DPI = 150
GRID_ALPHA = 0.3

# -- Paths (relative to repo root) -------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
TRADES_GLOB = str(BASE_DIR / "data" / "kalshi" / "trades" / "*.parquet")
MARKETS_GLOB = str(BASE_DIR / "data" / "kalshi" / "markets" / "*.parquet")
FIG_DIR = BASE_DIR / "strategy_reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    t0 = time.time()
    con = duckdb.connect()

    print("[info] Running quarterly YES bias query...", file=sys.stderr)

    sql = f"""
    WITH resolved_markets AS (
        SELECT ticker, result
        FROM '{MARKETS_GLOB}'
        WHERE status = 'finalized'
          AND result IN ('yes', 'no')
    ),
    trade_level AS (
        SELECT
            DATE_TRUNC('quarter', t.created_time) AS quarter,
            t.taker_side,
            -- Did the taker win?
            CASE WHEN t.taker_side = m.result THEN 1.0 ELSE 0.0 END AS won,
            -- Taker's price (what they paid)
            CASE WHEN t.taker_side = 'yes'
                 THEN t.yes_price / 100.0
                 ELSE t.no_price / 100.0
            END AS taker_price
        FROM '{TRADES_GLOB}' t
        INNER JOIN resolved_markets m ON t.ticker = m.ticker
        WHERE t.count > 0
    )
    SELECT
        quarter,
        -- Counts
        COUNT(*) FILTER (WHERE taker_side = 'yes') AS yes_trades,
        COUNT(*) FILTER (WHERE taker_side = 'no')  AS no_trades,
        -- YES/NO ratio
        COUNT(*) FILTER (WHERE taker_side = 'yes') * 1.0
            / NULLIF(COUNT(*) FILTER (WHERE taker_side = 'no'), 0) AS yes_no_ratio,
        -- YES taker excess return: AVG(won - price) for yes takers
        AVG(CASE WHEN taker_side = 'yes' THEN won - taker_price END) AS yes_excess,
        -- NO taker excess return: AVG(won - price) for no takers
        AVG(CASE WHEN taker_side = 'no'  THEN won - taker_price END) AS no_excess,
        -- Gap: YES excess - NO excess
        AVG(CASE WHEN taker_side = 'yes' THEN won - taker_price END)
          - AVG(CASE WHEN taker_side = 'no' THEN won - taker_price END) AS gap
    FROM trade_level
    GROUP BY quarter
    HAVING COUNT(*) >= 100
    ORDER BY quarter
    """

    df = con.execute(sql).fetchdf()
    con.close()
    elapsed = time.time() - t0
    print(f"[info] Query done in {elapsed:.1f}s, {len(df)} quarters", file=sys.stderr)

    if df.empty:
        print(json.dumps({"error": "No qualifying trades found"}))
        sys.exit(1)

    # Convert to pp (percentage points)
    df["yes_excess_pp"] = df["yes_excess"] * 100
    df["no_excess_pp"] = df["no_excess"] * 100
    df["gap_pp"] = df["gap"] * 100

    # Build quarter labels
    df["quarter_label"] = df["quarter"].apply(
        lambda x: f"{x.year}-Q{(x.month - 1) // 3 + 1}"
        if hasattr(x, "year") else str(x)[:10]
    )

    # Print summary to stderr
    for _, row in df.iterrows():
        print(
            f"  {row['quarter_label']:12s}  "
            f"YES={int(row['yes_trades']):>10,}  "
            f"NO={int(row['no_trades']):>10,}  "
            f"ratio={row['yes_no_ratio']:.2f}  "
            f"yes_ex={row['yes_excess_pp']:+.2f}pp  "
            f"no_ex={row['no_excess_pp']:+.2f}pp  "
            f"gap={row['gap_pp']:+.2f}pp",
            file=sys.stderr,
        )

    # == Figure: 2-panel temporal chart ========================================
    print("[info] Creating temporal figure...", file=sys.stderr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZE, facecolor="white")

    x_pos = np.arange(len(df))
    quarter_labels = df["quarter_label"].tolist()

    # -- Left panel: YES/NO ratio over time ------------------------------------
    ratios = df["yes_no_ratio"].values

    ax1.plot(x_pos, ratios, color=RED, marker="o", markersize=6,
             linewidth=2.5, label="YES/NO ratio", zorder=3)
    ax1.axhline(1.0, color="gray", linewidth=1.2, linestyle="--", alpha=0.7,
                label="Balanced (1.0)", zorder=2)
    ax1.fill_between(x_pos, ratios, 1.0, where=(ratios > 1.0),
                     alpha=0.12, color=RED, interpolate=True)

    ax1.set_xlabel("Quarter", fontsize=12)
    ax1.set_ylabel("YES / NO Trade Count Ratio", fontsize=12)
    ax1.set_title("YES/NO Volume Ratio by Quarter", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10, loc="best")
    ax1.grid(True, alpha=GRID_ALPHA)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(quarter_labels, rotation=45, ha="right", fontsize=8)

    # -- Right panel: Excess returns over time ---------------------------------
    yes_ex = df["yes_excess_pp"].values
    no_ex = df["no_excess_pp"].values

    ax2.plot(x_pos, yes_ex, color=RED, marker="o", markersize=6,
             linewidth=2.5, label="YES taker excess", zorder=3)
    ax2.plot(x_pos, no_ex, color=BLUE, marker="s", markersize=6,
             linewidth=2.5, label="NO taker excess", zorder=3)
    ax2.axhline(0, color="gray", linewidth=1.2, linestyle="--", alpha=0.7,
                zorder=2)

    ax2.set_xlabel("Quarter", fontsize=12)
    ax2.set_ylabel("Excess Return (pp)", fontsize=12)
    ax2.set_title("YES vs NO Taker Excess Return by Quarter", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10, loc="best")
    ax2.grid(True, alpha=GRID_ALPHA)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(quarter_labels, rotation=45, ha="right", fontsize=8)

    fig.tight_layout()

    fig_path = FIG_DIR / "6_3_yes_bias_temporal.png"
    fig.savefig(str(fig_path), dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"[info] Saved {fig_path}", file=sys.stderr)

    # == Stdout JSON ===========================================================
    total_elapsed = time.time() - t0

    quarterly_data = []
    for _, row in df.iterrows():
        quarterly_data.append({
            "quarter": row["quarter_label"],
            "yes_trades": int(row["yes_trades"]),
            "no_trades": int(row["no_trades"]),
            "yes_no_ratio": round(float(row["yes_no_ratio"]), 4),
            "yes_excess_pp": round(float(row["yes_excess_pp"]), 4),
            "no_excess_pp": round(float(row["no_excess_pp"]), 4),
            "gap_pp": round(float(row["gap_pp"]), 4),
        })

    output = {
        "n_quarters": len(quarterly_data),
        "n_trades_total": int(df["yes_trades"].sum() + df["no_trades"].sum()),
        "elapsed_seconds": round(total_elapsed, 1),
        "quarterly_data": quarterly_data,
        "figure": str(fig_path),
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
