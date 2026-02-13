"""§1.3 Category-Specific Miscalibration — Temporal Analysis (MAE by category per year)."""

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
# Category grouping (reused from 1_3_category_miscalibration.py)
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
# Query: get per-category, per-year, per-price-bucket data
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
            DATE_TRUNC('year', t.created_time) AS trade_year,
            {CATEGORY_SQL} AS category,
            CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS price,
            CASE WHEN t.taker_side = m.result THEN 1 ELSE 0 END AS won
        FROM '{TRADES_DIR}/*.parquet' t
        INNER JOIN resolved_markets m ON t.ticker = m.ticker

        UNION ALL

        SELECT
            DATE_TRUNC('year', t.created_time) AS trade_year,
            {CATEGORY_SQL} AS category,
            CASE WHEN t.taker_side = 'yes' THEN t.no_price ELSE t.yes_price END AS price,
            CASE WHEN t.taker_side != m.result THEN 1 ELSE 0 END AS won
        FROM '{TRADES_DIR}/*.parquet' t
        INNER JOIN resolved_markets m ON t.ticker = m.ticker
    )
    SELECT
        trade_year,
        category,
        price,
        COUNT(*) AS n,
        SUM(won) AS wins,
        100.0 * SUM(won) / COUNT(*) AS win_rate
    FROM all_positions
    WHERE price BETWEEN 1 AND 99
    GROUP BY trade_year, category, price
    ORDER BY trade_year, category, price
    """
    df = con.execute(sql).df()
    con.close()
    return df

# ---------------------------------------------------------------------------
# Build calibration per year+group
# ---------------------------------------------------------------------------
def build_temporal_calibration(df: pd.DataFrame, bucket_width: int = 5) -> pd.DataFrame:
    """Map categories to groups, bucket prices, compute MAE per year+group."""
    df = df.copy()
    df["group"] = df["category"].apply(get_group)
    df["year"] = df["trade_year"].dt.year

    # Bucket prices
    df["bucket"] = (df["price"] // bucket_width) * bucket_width + bucket_width / 2.0

    # Aggregate by year, group, bucket
    agg = (
        df.groupby(["year", "group", "bucket"], as_index=False)
        .agg(n=("n", "sum"), wins=("wins", "sum"))
    )
    agg["win_rate"] = 100.0 * agg["wins"] / agg["n"]

    # Filter: at least 100 trades per bucket
    agg = agg[agg["n"] >= 100].copy()

    # Compute MAE per year+group (weighted by n)
    agg["abs_error"] = (agg["win_rate"] - agg["bucket"]).abs()
    agg["weighted_error"] = agg["abs_error"] * agg["n"]

    mae = (
        agg.groupby(["year", "group"], as_index=False)
        .agg(
            total_n=("n", "sum"),
            weighted_error_sum=("weighted_error", "sum"),
            n_buckets=("bucket", "count"),
        )
    )
    mae["mae"] = mae["weighted_error_sum"] / mae["total_n"]

    # Filter: at least 500 total trades per year+group
    mae = mae[mae["total_n"] >= 500].copy()
    mae = mae.sort_values(["group", "year"]).reset_index(drop=True)

    return mae

# ---------------------------------------------------------------------------
# Plot: line chart of MAE by category over time
# ---------------------------------------------------------------------------
def plot_temporal_mae(mae_df: pd.DataFrame, groups_to_plot: list[str]) -> None:
    cmap = plt.cm.tab10
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")

    for i, grp in enumerate(groups_to_plot):
        subset = mae_df[mae_df["group"] == grp].sort_values("year")
        if subset.empty:
            continue
        color = cmap(i % 10)
        ax.plot(
            subset["year"],
            subset["mae"],
            marker="o",
            markersize=6,
            linewidth=2.0,
            label=grp,
            color=color,
            alpha=0.9,
        )

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Calibration MAE (percentage points)", fontsize=12)
    ax.set_title(
        "Category-Specific Miscalibration Over Time",
        fontsize=14,
        fontweight="bold",
    )

    # Set x-axis to integer years
    years = sorted(mae_df["year"].unique())
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_xlim(min(years) - 0.3, max(years) + 0.3)

    ax.legend(fontsize=9, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = FIG_DIR / "1_3_category_miscalibration_temporal.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}", file=sys.stderr)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading temporal data...", file=sys.stderr)
    raw = load_data()
    print(f"  Raw rows: {len(raw):,}", file=sys.stderr)

    mae_df = build_temporal_calibration(raw)
    print(f"  Year+group combinations: {len(mae_df):,}", file=sys.stderr)

    # Determine groups to plot: those present in at least 2 years
    group_year_counts = mae_df.groupby("group")["year"].nunique()
    groups_to_plot = group_year_counts[group_year_counts >= 2].index.tolist()
    # Sort by overall MAE descending for legend ordering
    overall_mae = mae_df.groupby("group")["mae"].mean()
    groups_to_plot = sorted(groups_to_plot, key=lambda g: -overall_mae.get(g, 0))
    print(f"  Groups to plot: {groups_to_plot}", file=sys.stderr)

    plot_temporal_mae(mae_df, groups_to_plot)

    # Build JSON summary
    summary: dict = {
        "years": sorted(int(y) for y in mae_df["year"].unique()),
        "groups": groups_to_plot,
        "yearly_mae": {},
    }

    for _, row in mae_df.iterrows():
        year = int(row["year"])
        group = row["group"]
        mae_val = round(float(row["mae"]), 2)
        total_n = int(row["total_n"])
        year_str = str(year)
        if year_str not in summary["yearly_mae"]:
            summary["yearly_mae"][year_str] = {}
        summary["yearly_mae"][year_str][group] = {
            "mae_pp": mae_val,
            "total_trades": total_n,
        }

    # Compute per-group trend: first year vs last year
    trends: dict = {}
    for grp in groups_to_plot:
        subset = mae_df[mae_df["group"] == grp].sort_values("year")
        if len(subset) >= 2:
            first = subset.iloc[0]
            last = subset.iloc[-1]
            change = round(float(last["mae"] - first["mae"]), 2)
            trends[grp] = {
                "first_year": int(first["year"]),
                "first_mae": round(float(first["mae"]), 2),
                "last_year": int(last["year"]),
                "last_mae": round(float(last["mae"]), 2),
                "change_pp": change,
                "direction": "improved" if change < 0 else "worsened" if change > 0 else "stable",
            }
    summary["trends"] = trends

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
