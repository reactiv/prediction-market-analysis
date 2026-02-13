"""§1.3 Category-Specific Miscalibration — calibration curves & MAE by category."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import duckdb  # noqa: E402
import pandas as pd  # noqa: E402
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
# Category SQL (extract prefix from event_ticker)
# ---------------------------------------------------------------------------
CATEGORY_SQL = """
CASE
    WHEN m.event_ticker IS NULL OR m.event_ticker = '' THEN 'independent'
    WHEN regexp_extract(m.event_ticker, '^([A-Z0-9]+)', 1) = '' THEN 'independent'
    ELSE regexp_extract(m.event_ticker, '^([A-Z0-9]+)', 1)
END
"""

# ---------------------------------------------------------------------------
# Category grouping
# ---------------------------------------------------------------------------
def get_group(cat: str) -> str:
    cat = cat.upper()
    sports = [
        "NFL", "NBA", "MLB", "NHL", "MLS", "NCAA", "SB", "AFC", "NFC",
        "HEISMAN", "WNBA", "PGA", "UFC", "TENNIS", "F1", "BOXING",
        "CRICKET", "SOCCER", "EPL", "NFLGAME", "NBAGAME", "MLBGAME",
        "NHLGAME", "MVENFL", "MVENBA", "ESPORT",
    ]
    politics = [
        "PRES", "DEM", "REP", "GOP", "GOV", "SENATE", "HOUSE", "SCOTUS",
        "POTUS", "ELECT", "TRUMP", "BIDEN", "HARRIS", "PARTY", "PRIM",
        "POLI", "POTY", "MENTION",
    ]
    crypto = ["BTC", "ETH", "SOL", "CRYPTO", "COIN", "DOGE", "XRP", "SHIBA"]
    finance = [
        "NASDAQ", "SP500", "SPX", "SPY", "DJI", "INFL", "CPI", "GDP",
        "FED", "FOMC", "RATE", "ECON", "TREASURY", "BOND", "OIL", "GOLD",
        "COMMODIT", "INX", "USDJPY", "EURUSD",
    ]
    weather = [
        "HURR", "TORNADO", "TEMP", "WEATHER", "RAIN", "SNOW", "CLIMATE",
        "STORM", "CITIES",
    ]
    entertainment = [
        "OSCAR", "EMMY", "GRAMMY", "GOLDEN", "AWARD", "MOVIE", "TV",
        "CELEB", "ROYAL", "MUSIC",
    ]
    science = ["NASA", "SPACE", "AI", "TECH", "LAUNCH", "ROBOT"]
    for prefix in sports:
        if prefix in cat:
            return "Sports"
    for prefix in politics:
        if prefix in cat:
            return "Politics"
    for prefix in crypto:
        if prefix in cat:
            return "Crypto"
    for prefix in finance:
        if prefix in cat:
            return "Finance"
    for prefix in weather:
        if prefix in cat:
            return "Weather"
    for prefix in entertainment:
        if prefix in cat:
            return "Entertainment"
    for prefix in science:
        if prefix in cat:
            return "Science/Tech"
    return "Other"

# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    con = duckdb.connect()
    sql = f"""
    WITH resolved_markets AS (
        SELECT ticker, event_ticker, result
        FROM '{MARKETS_DIR}/*.parquet'
        WHERE status = 'finalized' AND result IN ('yes', 'no')
    ),
    all_positions AS (
        SELECT
            {CATEGORY_SQL} AS category,
            CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS price,
            CASE WHEN t.taker_side = m.result THEN 1 ELSE 0 END AS won
        FROM '{TRADES_DIR}/*.parquet' t
        INNER JOIN resolved_markets m ON t.ticker = m.ticker

        UNION ALL

        SELECT
            {CATEGORY_SQL} AS category,
            CASE WHEN t.taker_side = 'yes' THEN t.no_price ELSE t.yes_price END AS price,
            CASE WHEN t.taker_side != m.result THEN 1 ELSE 0 END AS won
        FROM '{TRADES_DIR}/*.parquet' t
        INNER JOIN resolved_markets m ON t.ticker = m.ticker
    )
    SELECT category, price,
        COUNT(*) AS n,
        SUM(won) AS wins,
        100.0 * SUM(won) / COUNT(*) AS win_rate
    FROM all_positions
    WHERE price BETWEEN 1 AND 99
    GROUP BY category, price
    ORDER BY category, price
    """
    df = con.execute(sql).df()
    con.close()
    return df

# ---------------------------------------------------------------------------
# Aggregate into groups & compute calibration
# ---------------------------------------------------------------------------
def build_calibration(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw categories to groups and re-aggregate by (group, price)."""
    df["group"] = df["category"].apply(get_group)
    grouped = (
        df.groupby(["group", "price"], as_index=False)
        .agg(n=("n", "sum"), wins=("wins", "sum"))
    )
    grouped["win_rate"] = 100.0 * grouped["wins"] / grouped["n"]
    return grouped

def bucket_calibration(df: pd.DataFrame, bucket_width: int = 5) -> pd.DataFrame:
    """Bucket prices into bins, enforce min 100 obs per bucket."""
    df = df.copy()
    df["bucket"] = (df["price"] // bucket_width) * bucket_width + bucket_width / 2.0
    agg = (
        df.groupby(["group", "bucket"], as_index=False)
        .agg(n=("n", "sum"), wins=("wins", "sum"))
    )
    agg["win_rate"] = 100.0 * agg["wins"] / agg["n"]
    # Filter buckets with fewer than 100 observations
    agg = agg[agg["n"] >= 100].copy()
    return agg

def compute_mae(bucketed: pd.DataFrame) -> pd.DataFrame:
    """MAE = mean |win_rate - bucket_midpoint| per group."""
    bucketed = bucketed.copy()
    bucketed["abs_error"] = (bucketed["win_rate"] - bucketed["bucket"]).abs()
    # Weight by number of observations in each bucket
    bucketed["weighted_error"] = bucketed["abs_error"] * bucketed["n"]
    mae = (
        bucketed.groupby("group", as_index=False)
        .agg(
            total_n=("n", "sum"),
            weighted_error_sum=("weighted_error", "sum"),
            n_buckets=("bucket", "count"),
        )
    )
    mae["mae"] = mae["weighted_error_sum"] / mae["total_n"]
    mae = mae.sort_values("mae", ascending=False).reset_index(drop=True)
    return mae

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
COLORS = [
    "#4C72B0",  # blue
    "#e74c3c",  # red
    "#2ecc71",  # green
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#17becf",  # cyan
    "#d62728",  # dark red
]


def plot_calibration_curves(bucketed: pd.DataFrame, top_groups: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")

    for i, grp in enumerate(top_groups):
        subset = bucketed[bucketed["group"] == grp].sort_values("bucket")
        color = COLORS[i % len(COLORS)]
        ax.plot(
            subset["bucket"],
            subset["win_rate"],
            marker="o",
            markersize=4,
            linewidth=1.8,
            label=grp,
            color=color,
            alpha=0.85,
        )

    # Perfect calibration line
    ax.plot(
        [0, 100], [0, 100],
        linestyle="--", color="gray", linewidth=1,
        label="Perfect calibration",
    )

    ax.set_xlabel("Contract Price (cents)", fontsize=12)
    ax.set_ylabel("Win Rate (%)", fontsize=12)
    ax.set_title("Calibration Curves by Category", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = FIG_DIR / "1_3_calibration_by_category.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}", file=sys.stderr)


def plot_mae_bar(mae_df: pd.DataFrame, top_groups: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")

    subset = mae_df[mae_df["group"].isin(top_groups)].sort_values("mae", ascending=True)
    colors = []
    for grp in subset["group"]:
        idx = top_groups.index(grp) if grp in top_groups else 0
        colors.append(COLORS[idx % len(COLORS)])

    bars = ax.barh(subset["group"], subset["mae"], color=colors, edgecolor="white", height=0.6)

    # Add value labels
    for bar, val in zip(bars, subset["mae"]):
        ax.text(
            bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center", fontsize=10,
        )

    ax.set_xlabel("Mean Absolute Error (percentage points)", fontsize=12)
    ax.set_title("Calibration MAE by Category", fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    path = FIG_DIR / "1_3_mae_by_category.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}", file=sys.stderr)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading data...", file=sys.stderr)
    raw = load_data()
    print(f"  Raw rows: {len(raw):,}", file=sys.stderr)

    cal = build_calibration(raw)
    bucketed = bucket_calibration(cal, bucket_width=5)
    mae_df = compute_mae(bucketed)

    # Pick top 8 groups by total trade count
    group_totals = cal.groupby("group")["n"].sum().sort_values(ascending=False)
    top_groups = group_totals.head(8).index.tolist()
    print(f"  Top 8 groups: {top_groups}", file=sys.stderr)

    # Filter bucketed to top 8 for plotting
    bucketed_top = bucketed[bucketed["group"].isin(top_groups)]

    plot_calibration_curves(bucketed_top, top_groups)
    plot_mae_bar(mae_df, top_groups)

    # Build JSON summary
    summary = {
        "total_categories": int(raw["category"].nunique()),
        "total_groups": int(cal["group"].nunique()),
        "top_groups": [],
    }
    for grp in top_groups:
        row = mae_df[mae_df["group"] == grp]
        total_trades = int(group_totals.get(grp, 0))
        mae_val = float(row["mae"].iloc[0]) if len(row) > 0 else None
        summary["top_groups"].append({
            "group": grp,
            "mae_pp": round(mae_val, 2) if mae_val is not None else None,
            "total_trades": total_trades,
        })

    # Per-group bucket details for the most/least calibrated
    most_miscalibrated = mae_df.iloc[0]["group"] if len(mae_df) > 0 else None
    least_miscalibrated = mae_df.iloc[-1]["group"] if len(mae_df) > 0 else None
    summary["most_miscalibrated"] = most_miscalibrated
    summary["least_miscalibrated"] = least_miscalibrated

    # Overall MAE (weighted by trades)
    top_mae = mae_df[mae_df["group"].isin(top_groups)]
    if len(top_mae) > 0:
        overall = (top_mae["mae"] * top_mae["total_n"]).sum() / top_mae["total_n"].sum()
        summary["overall_weighted_mae_pp"] = round(float(overall), 2)

    # Price-range analysis: where is miscalibration worst?
    bucketed_top_copy = bucketed_top.copy()
    bucketed_top_copy["abs_error"] = (bucketed_top_copy["win_rate"] - bucketed_top_copy["bucket"]).abs()
    range_summary = (
        bucketed_top_copy.groupby("bucket")
        .agg(mean_error=("abs_error", "mean"), total_n=("n", "sum"))
        .reset_index()
        .sort_values("mean_error", ascending=False)
        .head(5)
    )
    summary["worst_price_ranges"] = [
        {"bucket": float(r["bucket"]), "mean_error": round(float(r["mean_error"]), 2)}
        for _, r in range_summary.iterrows()
    ]

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
