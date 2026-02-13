#!/usr/bin/env python3
"""
Strategy Report 4.2: Intra-Event Parity Analysis

Groups markets by event_ticker, sums last_price across all contracts in each
event, and measures deviation from the theoretical parity sum of 100. For
mutually exclusive and exhaustive outcomes, fair prices should sum to 100
(in cents). Deviations indicate embedded vig (overround) or arbitrage
opportunities.

Markets-only query -- no trades join needed.
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import duckdb  # noqa: E402

# ── Palette & style ──────────────────────────────────────────────────────────
BLUE = "#4C72B0"
RED = "#e74c3c"
GREEN = "#2ecc71"
ORANGE = "#ff7f0e"
PURPLE = "#9467bd"
FIG_SIZE = (12, 7)
DPI = 150
GRID_ALPHA = 0.3

PARQUET_GLOB = "data/kalshi/markets/*.parquet"
FIG_DIR = "strategy_reports/figures"

# ── Query ────────────────────────────────────────────────────────────────────
EVENT_STATS_SQL = f"""
WITH event_stats AS (
    SELECT
        event_ticker,
        COUNT(*)            AS n_outcomes,
        SUM(last_price)     AS price_sum,
        SUM(last_price) - 100 AS deviation,
        SUM(volume)         AS total_volume,
        AVG(last_price)     AS avg_price
    FROM '{PARQUET_GLOB}'
    WHERE last_price IS NOT NULL
      AND last_price > 0
      AND event_ticker IS NOT NULL
      AND event_ticker != ''
    GROUP BY event_ticker
    HAVING COUNT(*) >= 2
)
SELECT * FROM event_stats
ORDER BY n_outcomes, event_ticker
"""


def main() -> None:
    con = duckdb.connect()
    df = con.execute(EVENT_STATS_SQL).fetchdf()

    if df.empty:
        print(json.dumps({"error": "No multi-outcome events found"}))
        sys.exit(1)

    total_events = len(df)
    deviations = df["deviation"].values
    n_outcomes_arr = df["n_outcomes"].values

    # ── Global summary stats ─────────────────────────────────────────────
    summary = {
        "total_multi_outcome_events": int(total_events),
        "deviation_mean": round(float(np.mean(deviations)), 2),
        "deviation_median": round(float(np.median(deviations)), 2),
        "deviation_std": round(float(np.std(deviations)), 2),
        "deviation_q25": round(float(np.percentile(deviations, 25)), 2),
        "deviation_q75": round(float(np.percentile(deviations, 75)), 2),
        "pct_positive_deviation": round(
            float(np.mean(deviations > 0) * 100), 2
        ),
        "pct_negative_deviation": round(
            float(np.mean(deviations < 0) * 100), 2
        ),
        "pct_abs_gt_5": round(float(np.mean(np.abs(deviations) > 5) * 100), 2),
        "pct_abs_gt_10": round(
            float(np.mean(np.abs(deviations) > 10) * 100), 2
        ),
    }

    # ── Per-outcome-count breakdown ──────────────────────────────────────
    # Limit to outcome counts with enough data and reasonable range
    max_display_outcomes = 20
    mask_display = n_outcomes_arr <= max_display_outcomes
    outcome_counts = sorted(df.loc[mask_display, "n_outcomes"].unique())

    per_outcome = []
    for n in outcome_counts:
        subset = df.loc[df["n_outcomes"] == n, "deviation"].values
        per_outcome.append({
            "n_outcomes": int(n),
            "count": int(len(subset)),
            "mean_deviation": round(float(np.mean(subset)), 2),
            "median_deviation": round(float(np.median(subset)), 2),
            "std_deviation": round(float(np.std(subset)), 2),
            "pct_abs_gt_5": round(
                float(np.mean(np.abs(subset) > 5) * 100), 2
            ),
        })

    summary["by_outcome_count"] = per_outcome

    # ── Figure 1: Distribution of parity deviations ──────────────────────
    # Focus on the core range for readability; note many high-outcome events
    # have very large deviations, so we clip for the histogram view.
    clip_lo, clip_hi = -150, 500
    clipped = np.clip(deviations, clip_lo, clip_hi)
    n_clipped_lo = int(np.sum(deviations < clip_lo))
    n_clipped_hi = int(np.sum(deviations > clip_hi))

    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor="white")
    bins = np.arange(clip_lo, clip_hi + 5, 5)
    ax.hist(clipped, bins=bins, color=BLUE, edgecolor="white", linewidth=0.3,
            alpha=0.85)
    ax.axvline(0, color=RED, linewidth=2, linestyle="--", label="Parity (0)")
    ax.axvline(float(np.median(deviations)), color=ORANGE, linewidth=2,
               linestyle="-.", label=f"Median ({np.median(deviations):.0f})")

    ax.set_xlabel("Deviation from 100 (price-sum minus 100)", fontsize=12)
    ax.set_ylabel("Number of Events", fontsize=12)
    ax.set_title("Distribution of Intra-Event Parity Deviations\n"
                 f"(N={total_events:,} multi-outcome events; "
                 f"clipped to [{clip_lo}, {clip_hi}], "
                 f"{n_clipped_lo:,} below / {n_clipped_hi:,} above)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=GRID_ALPHA)
    fig.tight_layout()
    path1 = f"{FIG_DIR}/4_2_parity_deviation_distribution.png"
    fig.savefig(path1, dpi=DPI, facecolor="white")
    plt.close(fig)

    # ── Figure 2: Box plot of deviation by outcome count ─────────────────
    box_data = []
    labels = []
    medians_for_line = []
    for n in outcome_counts:
        subset = df.loc[df["n_outcomes"] == n, "deviation"].values
        box_data.append(subset)
        labels.append(str(n))
        medians_for_line.append(float(np.median(subset)))

    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor="white")

    bp = ax.boxplot(
        box_data,
        tick_labels=labels,
        patch_artist=True,
        showfliers=False,
        widths=0.6,
        medianprops=dict(color=RED, linewidth=2),
        whiskerprops=dict(color="#555555"),
        capprops=dict(color="#555555"),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(BLUE)
        patch.set_alpha(0.7)

    # Overlay median line
    x_pos = range(1, len(outcome_counts) + 1)
    ax.plot(x_pos, medians_for_line, color=ORANGE, linewidth=2, marker="o",
            markersize=5, label="Median deviation", zorder=5)
    ax.axhline(0, color=RED, linewidth=1.5, linestyle="--", alpha=0.7,
               label="Parity (0)")

    ax.set_xlabel("Number of Outcomes per Event", fontsize=12)
    ax.set_ylabel("Deviation from 100 (cents)", fontsize=12)
    ax.set_title("Parity Deviation by Outcome Count\n"
                 "(box = IQR, whiskers = 1.5x IQR, fliers hidden)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(axis="y", alpha=GRID_ALPHA)
    fig.tight_layout()
    path2 = f"{FIG_DIR}/4_2_parity_by_outcome_count.png"
    fig.savefig(path2, dpi=DPI, facecolor="white")
    plt.close(fig)

    # ── Stdout JSON ──────────────────────────────────────────────────────
    summary["figures"] = [path1, path2]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
