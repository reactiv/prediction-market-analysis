"""Strategy Report 2.4: Spread Dynamics.

Analyzes bid/ask spread characteristics across Kalshi prediction markets,
examining spread distributions, category-level patterns, and the relationship
between spread width and volume.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import duckdb
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MARKETS_GLOB = str(BASE_DIR / "data" / "kalshi" / "markets" / "*.parquet")
FIGURES_DIR = BASE_DIR / "strategy_reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

COLORS = ["#4C72B0", "#e74c3c", "#2ecc71", "#ff7f0e", "#9467bd"]
FIGSIZE = (12, 7)
DPI = 150

# ---------------------------------------------------------------------------
# Category helpers
# ---------------------------------------------------------------------------
CATEGORY_SQL = """
CASE
    WHEN event_ticker IS NULL OR event_ticker = '' THEN 'independent'
    WHEN regexp_extract(event_ticker, '^([A-Z0-9]+)', 1) = '' THEN 'independent'
    ELSE regexp_extract(event_ticker, '^([A-Z0-9]+)', 1)
END
"""


def get_group(cat: str) -> str:
    cat = cat.upper()
    sports = [
        'NFL', 'NBA', 'MLB', 'NHL', 'MLS', 'NCAA', 'SB', 'AFC', 'NFC',
        'HEISMAN', 'WNBA', 'PGA', 'UFC', 'TENNIS', 'F1', 'BOXING',
        'CRICKET', 'SOCCER', 'EPL', 'NFLGAME', 'NBAGAME', 'MLBGAME',
        'NHLGAME', 'MVENFL', 'MVENBA',
    ]
    politics = [
        'PRES', 'DEM', 'REP', 'GOP', 'GOV', 'SENATE', 'HOUSE', 'SCOTUS',
        'POTUS', 'ELECT', 'TRUMP', 'BIDEN', 'HARRIS', 'PARTY', 'PRIM',
        'POLI', 'POTY',
    ]
    crypto = ['BTC', 'ETH', 'SOL', 'CRYPTO', 'COIN', 'DOGE', 'XRP']
    finance = [
        'NASDAQ', 'SP500', 'SPX', 'SPY', 'DJI', 'INFL', 'CPI', 'GDP',
        'FED', 'FOMC', 'RATE', 'ECON', 'TREASURY', 'BOND', 'OIL', 'GOLD',
        'COMMODIT',
    ]
    weather = ['HURR', 'TORNADO', 'TEMP', 'WEATHER', 'RAIN', 'SNOW', 'CLIMATE', 'STORM']
    entertainment = [
        'OSCAR', 'EMMY', 'GRAMMY', 'GOLDEN', 'AWARD', 'MOVIE', 'TV',
        'CELEB', 'ROYAL', 'MUSIC',
    ]
    science = ['NASA', 'SPACE', 'AI', 'TECH', 'LAUNCH', 'ROBOT']
    for prefix in sports:
        if prefix in cat:
            return 'Sports'
    for prefix in politics:
        if prefix in cat:
            return 'Politics'
    for prefix in crypto:
        if prefix in cat:
            return 'Crypto'
    for prefix in finance:
        if prefix in cat:
            return 'Finance'
    for prefix in weather:
        if prefix in cat:
            return 'Weather'
    for prefix in entertainment:
        if prefix in cat:
            return 'Entertainment'
    for prefix in science:
        if prefix in cat:
            return 'Science/Tech'
    return 'Other'


# ---------------------------------------------------------------------------
# Data queries
# ---------------------------------------------------------------------------

def query_spreads(con: duckdb.DuckDBPyConnection):
    """Query spread data from market snapshots."""
    print("Querying spread data from Kalshi markets...", file=sys.stderr)
    df = con.execute(f"""
        SELECT
            ticker,
            event_ticker,
            {CATEGORY_SQL} AS category,
            yes_bid,
            yes_ask,
            (yes_ask - yes_bid) AS spread,
            volume,
            status
        FROM '{MARKETS_GLOB}'
        WHERE yes_bid IS NOT NULL
          AND yes_ask IS NOT NULL
          AND yes_bid > 0
          AND yes_ask > 0
          AND yes_ask >= yes_bid
    """).df()
    print(f"  Found {len(df):,} market snapshots with valid spreads", file=sys.stderr)
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_spread_distribution(df, output_path: Path):
    """Histogram of spread values across all market snapshots."""
    spreads = df["spread"].values

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor("white")

    max_spread = int(np.percentile(spreads, 99))
    bins = np.arange(0, max_spread + 2, 1)

    ax.hist(
        spreads[spreads <= max_spread],
        bins=bins,
        color=COLORS[0],
        edgecolor="white",
        linewidth=0.3,
        alpha=0.85,
    )

    mean_spread = np.mean(spreads)
    median_spread = np.median(spreads)
    ax.axvline(mean_spread, color=COLORS[1], linestyle="--", linewidth=2, label=f"Mean: {mean_spread:.1f}c")
    ax.axvline(median_spread, color=COLORS[2], linestyle="-.", linewidth=2, label=f"Median: {median_spread:.0f}c")

    ax.set_xlabel("Spread (cents)", fontsize=12)
    ax.set_ylabel("Number of Market Snapshots", fontsize=12)
    ax.set_title("Distribution of Bid-Ask Spreads (Kalshi YES Side)", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}K" if x >= 1000 else f"{x:.0f}"))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}", file=sys.stderr)


def plot_spread_by_category(group_stats, output_path: Path):
    """Bar chart of mean and median spread by category group."""
    # Sort by mean spread descending
    groups = sorted(group_stats.keys(), key=lambda g: group_stats[g]["mean_spread"], reverse=True)

    means = [group_stats[g]["mean_spread"] for g in groups]
    medians = [group_stats[g]["median_spread"] for g in groups]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor("white")

    x = np.arange(len(groups))
    width = 0.35

    bars_mean = ax.bar(x - width / 2, means, width, label="Mean Spread", color=COLORS[0],
                       edgecolor="white", linewidth=0.5)
    bars_median = ax.bar(x + width / 2, medians, width, label="Median Spread", color=COLORS[3],
                         edgecolor="white", linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars_mean, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, val in zip(bars_median, medians):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Category", fontsize=12)
    ax.set_ylabel("Spread (cents)", fontsize=12)
    ax.set_title("Bid-Ask Spread by Market Category", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=25, ha="right", fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    con = duckdb.connect()

    # --- Query spread data ---
    df = query_spreads(con)
    con.close()

    # --- Apply category grouping ---
    print("Applying category grouping...", file=sys.stderr)
    df["group"] = df["category"].apply(get_group)

    # --- Overall stats ---
    spreads = df["spread"].values
    overall_stats = {
        "total_snapshots": int(len(df)),
        "mean_spread": round(float(np.mean(spreads)), 2),
        "median_spread": round(float(np.median(spreads)), 2),
        "std_spread": round(float(np.std(spreads)), 2),
        "p25_spread": round(float(np.percentile(spreads, 25)), 2),
        "p75_spread": round(float(np.percentile(spreads, 75)), 2),
        "p95_spread": round(float(np.percentile(spreads, 95)), 2),
        "max_spread": int(np.max(spreads)),
        "pct_spread_1c": round(float(np.mean(spreads == 1) * 100), 2),
        "pct_spread_le_5c": round(float(np.mean(spreads <= 5) * 100), 2),
    }
    print(f"  Overall: mean={overall_stats['mean_spread']}c, median={overall_stats['median_spread']}c",
          file=sys.stderr)

    # --- Spread vs volume correlation ---
    vol_mask = df["volume"] > 0
    if vol_mask.sum() > 0:
        vol_df = df[vol_mask]
        corr_linear = float(np.corrcoef(vol_df["spread"].values.astype(float),
                                         vol_df["volume"].values.astype(float))[0, 1])
        # Log-volume correlation (more meaningful)
        log_vol = np.log10(vol_df["volume"].values.astype(float))
        corr_log = float(np.corrcoef(vol_df["spread"].values.astype(float), log_vol)[0, 1])
    else:
        corr_linear = 0.0
        corr_log = 0.0

    volume_correlation = {
        "n_markets_with_volume": int(vol_mask.sum()),
        "pearson_spread_vs_volume": round(corr_linear, 4),
        "pearson_spread_vs_log_volume": round(corr_log, 4),
    }
    print(f"  Spread-volume correlation: r={corr_linear:.4f}, r(log)={corr_log:.4f}", file=sys.stderr)

    # --- Per-group stats ---
    print("Computing per-group stats...", file=sys.stderr)
    group_stats = {}
    for grp, gdf in df.groupby("group"):
        g_spreads = gdf["spread"].values
        g_vols = gdf["volume"].values
        group_stats[grp] = {
            "n_markets": int(len(gdf)),
            "mean_spread": round(float(np.mean(g_spreads)), 2),
            "median_spread": round(float(np.median(g_spreads)), 2),
            "std_spread": round(float(np.std(g_spreads)), 2),
            "mean_volume": round(float(np.mean(g_vols)), 2),
            "median_volume": round(float(np.median(g_vols)), 2),
        }
        print(f"  {grp}: n={group_stats[grp]['n_markets']}, "
              f"mean_spread={group_stats[grp]['mean_spread']}c, "
              f"mean_vol={group_stats[grp]['mean_volume']:.0f}", file=sys.stderr)

    # --- Generate figures ---
    print("\n--- Generating Figures ---", file=sys.stderr)
    plot_spread_distribution(
        df,
        FIGURES_DIR / "2_4_spread_distribution.png",
    )
    plot_spread_by_category(
        group_stats,
        FIGURES_DIR / "2_4_spread_by_category.png",
    )

    # --- JSON output ---
    output = {
        "report": "2.4 Spread Dynamics",
        "overall": overall_stats,
        "volume_correlation": volume_correlation,
        "by_group": group_stats,
    }
    print(json.dumps(output, indent=2))
    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()
