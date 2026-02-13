#!/usr/bin/env python3
"""
Strategy Report 2.5: Trade Size Segmentation

Computes calibration AND maker excess returns by size bucket
(1 contract, 2-10, 11-100, 101-1000, 1000+). Cross-tabulates with category.
Answers: Are large trades more informed? Do makers earn more against large
or small takers?

Self-contained: no src.* imports, uses inline category resolution.
"""

import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import numpy as np  # noqa: E402
import duckdb  # noqa: E402

# ── Palette & style ──────────────────────────────────────────────────────────
BLUE = "#4C72B0"
GREEN = "#2ecc71"
RED = "#e74c3c"
ORANGE = "#ff7f0e"
GRAY = "#888888"
FIG_SIZE = (12, 7)
DPI = 150
GRID_ALPHA = 0.3

TRADES_GLOB = "data/kalshi/trades/*.parquet"
MARKETS_GLOB = "data/kalshi/markets/*.parquet"
FIG_DIR = "strategy_reports/figures"


def get_group(prefix: str) -> str:
    sports = {'NFL', 'NBA', 'MLB', 'NHL', 'MLS', 'NCAAF', 'NCAAB', 'FIFA', 'WNBA', 'UFC',
              'PGA', 'CL', 'EPL', 'SERA', 'BUND', 'LIGA', 'LIG1', 'MMA', 'BOXING', 'F1',
              'NASCAR', 'ATP', 'WTA', 'BOWL', 'MARCH', 'CFP', 'INDY', 'XFL', 'USFL',
              'KXMVE', 'CUP'}
    politics = {'PRES', 'DEM', 'GOP', 'REP', 'SEN', 'GOV', 'HOUSE', 'EC', 'PRIM', 'PRES24',
                'ELECT', 'PRESO', 'PCON', 'TRUMP', 'BIDEN', 'POTH', 'INX', 'POLI'}
    crypto = {'BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'DOT', 'LINK', 'UNI', 'AVAX',
              'SHIB', 'MATIC', 'CRYPTO', 'COIN', 'DEFI'}
    finance = {'NASDAQ', 'SP500', 'DOW', 'RUSSELL', 'RATE', 'FED', 'CPI', 'GDP', 'JOBS',
               'ECON', 'FOMC', 'YIELD', 'OIL', 'GOLD', 'SILVER', 'HOUSING', 'DEBT',
               'FISCAL', 'TRADE', 'UNEMP', 'RETAIL', 'INF'}
    weather = {'TEMP', 'SNOW', 'RAIN', 'HURRICANE', 'STORM', 'CLIMATE', 'WEATHER', 'WIND',
               'FLOOD', 'DROUGHT', 'HEAT', 'COLD', 'NOAA', 'PRECIP'}
    entertainment = {'OSCAR', 'EMMY', 'GRAMMY', 'GOLDEN', 'TONY', 'NETFLIX', 'SPOTIFY',
                     'MOVIE', 'TV', 'STREAM', 'BILLBOARD', 'BOX', 'BRIT', 'BAFTA', 'SAG',
                     'CANNES'}
    science = {'NASA', 'SPACE', 'LAUNCH', 'MARS', 'MOON', 'ROCKET', 'SATEL', 'COVID',
               'VACCINE', 'FDA', 'WHO', 'VIRUS', 'DISEASE', 'HEALTH', 'PHARMA', 'DRUG',
               'BIOTECH', 'AI', 'TECH', 'APPLE', 'GOOGLE', 'META', 'MSFT', 'AMZN', 'TSLA'}
    p = prefix.upper()
    for kw in sports:
        if kw in p:
            return 'Sports'
    for kw in politics:
        if kw in p:
            return 'Politics'
    for kw in crypto:
        if kw in p:
            return 'Crypto'
    for kw in finance:
        if kw in p:
            return 'Finance'
    for kw in weather:
        if kw in p:
            return 'Weather'
    for kw in entertainment:
        if kw in p:
            return 'Entertainment'
    for kw in science:
        if kw in p:
            return 'Science/Tech'
    return 'Other'


def main() -> None:
    t0 = time.time()
    con = duckdb.connect()

    # ── Step 1: Main query — size bucket stats ──────────────────────────────
    print("[info] Running main size-bucket query...", file=sys.stderr)

    main_sql = f"""
    WITH resolved_markets AS (
        SELECT ticker, result, event_ticker
        FROM '{MARKETS_GLOB}'
        WHERE status = 'finalized' AND result IN ('yes', 'no')
    ),
    trade_data AS (
        SELECT
            t.count AS contracts,
            CASE
                WHEN t.count = 1 THEN '1'
                WHEN t.count <= 10 THEN '2-10'
                WHEN t.count <= 100 THEN '11-100'
                WHEN t.count <= 1000 THEN '101-1K'
                ELSE '1K+'
            END AS size_bucket,
            CASE
                WHEN t.count = 1 THEN 1
                WHEN t.count <= 10 THEN 2
                WHEN t.count <= 100 THEN 3
                WHEN t.count <= 1000 THEN 4
                ELSE 5
            END AS bucket_order,
            -- Taker perspective
            CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS taker_price,
            CASE WHEN t.taker_side = m.result THEN 1.0 ELSE 0.0 END AS taker_won,
            -- Maker perspective
            CASE WHEN t.taker_side = 'yes' THEN t.no_price ELSE t.yes_price END AS maker_price,
            CASE WHEN t.taker_side != m.result THEN 1.0 ELSE 0.0 END AS maker_won,
            regexp_extract(m.event_ticker, '^([A-Z0-9]+)', 1) AS category_prefix
        FROM '{TRADES_GLOB}' t
        INNER JOIN resolved_markets m ON t.ticker = m.ticker
    )
    SELECT
        size_bucket, bucket_order,
        AVG(taker_won) AS taker_win_rate,
        AVG(taker_price / 100.0) AS taker_avg_price,
        AVG(taker_won - taker_price / 100.0) AS taker_excess,
        AVG(maker_won) AS maker_win_rate,
        AVG(maker_price / 100.0) AS maker_avg_price,
        AVG(maker_won - maker_price / 100.0) AS maker_excess,
        COUNT(*) AS n_trades,
        SUM(contracts) AS total_contracts
    FROM trade_data
    GROUP BY size_bucket, bucket_order
    ORDER BY bucket_order
    """

    df_main = con.execute(main_sql).fetchdf()
    elapsed_q1 = time.time() - t0
    print(f"[info] Main query done in {elapsed_q1:.1f}s, {len(df_main)} buckets", file=sys.stderr)
    print(f"[info] Total trades: {df_main['n_trades'].sum():,}", file=sys.stderr)

    if df_main.empty:
        print(json.dumps({"error": "No qualifying trades found"}))
        sys.exit(1)

    # ── Step 2: Cross-tabulation with category ──────────────────────────────
    print("[info] Running category cross-tab query...", file=sys.stderr)

    cross_sql = f"""
    WITH resolved_markets AS (
        SELECT ticker, result, event_ticker
        FROM '{MARKETS_GLOB}'
        WHERE status = 'finalized' AND result IN ('yes', 'no')
    ),
    trade_data AS (
        SELECT
            t.count AS contracts,
            CASE
                WHEN t.count = 1 THEN '1'
                WHEN t.count <= 10 THEN '2-10'
                WHEN t.count <= 100 THEN '11-100'
                WHEN t.count <= 1000 THEN '101-1K'
                ELSE '1K+'
            END AS size_bucket,
            CASE
                WHEN t.count = 1 THEN 1
                WHEN t.count <= 10 THEN 2
                WHEN t.count <= 100 THEN 3
                WHEN t.count <= 1000 THEN 4
                ELSE 5
            END AS bucket_order,
            CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS taker_price,
            CASE WHEN t.taker_side = m.result THEN 1.0 ELSE 0.0 END AS taker_won,
            CASE WHEN t.taker_side = 'yes' THEN t.no_price ELSE t.yes_price END AS maker_price,
            CASE WHEN t.taker_side != m.result THEN 1.0 ELSE 0.0 END AS maker_won,
            regexp_extract(m.event_ticker, '^([A-Z0-9]+)', 1) AS category_prefix
        FROM '{TRADES_GLOB}' t
        INNER JOIN resolved_markets m ON t.ticker = m.ticker
    )
    SELECT
        category_prefix,
        size_bucket, bucket_order,
        AVG(taker_won) AS taker_win_rate,
        AVG(taker_price / 100.0) AS taker_avg_price,
        AVG(taker_won - taker_price / 100.0) AS taker_excess,
        AVG(maker_won) AS maker_win_rate,
        AVG(maker_price / 100.0) AS maker_avg_price,
        AVG(maker_won - maker_price / 100.0) AS maker_excess,
        COUNT(*) AS n_trades,
        SUM(contracts) AS total_contracts
    FROM trade_data
    GROUP BY category_prefix, size_bucket, bucket_order
    ORDER BY category_prefix, bucket_order
    """

    df_cross = con.execute(cross_sql).fetchdf()
    elapsed_q2 = time.time() - t0
    print(f"[info] Cross-tab query done in {elapsed_q2:.1f}s, {len(df_cross)} rows", file=sys.stderr)

    # Map prefixes to category groups
    df_cross["category"] = df_cross["category_prefix"].apply(
        lambda x: get_group(x) if isinstance(x, str) else "Other"
    )

    # Re-aggregate by category group + size bucket
    cat_agg = (
        df_cross.groupby(["category", "size_bucket", "bucket_order"])
        .agg({
            "n_trades": "sum",
            "total_contracts": "sum",
        })
        .reset_index()
    )

    # For weighted averages, we need to weight by n_trades
    # Re-compute weighted taker/maker excess from the raw cross data
    df_cross["w_taker_excess"] = df_cross["taker_excess"] * df_cross["n_trades"]
    df_cross["w_maker_excess"] = df_cross["maker_excess"] * df_cross["n_trades"]
    df_cross["w_taker_win_rate"] = df_cross["taker_win_rate"] * df_cross["n_trades"]
    df_cross["w_maker_win_rate"] = df_cross["maker_win_rate"] * df_cross["n_trades"]
    df_cross["w_taker_avg_price"] = df_cross["taker_avg_price"] * df_cross["n_trades"]
    df_cross["w_maker_avg_price"] = df_cross["maker_avg_price"] * df_cross["n_trades"]

    cat_detailed = (
        df_cross.groupby(["category", "size_bucket", "bucket_order"])
        .agg({
            "n_trades": "sum",
            "total_contracts": "sum",
            "w_taker_excess": "sum",
            "w_maker_excess": "sum",
            "w_taker_win_rate": "sum",
            "w_maker_win_rate": "sum",
            "w_taker_avg_price": "sum",
            "w_maker_avg_price": "sum",
        })
        .reset_index()
    )
    cat_detailed["taker_excess"] = cat_detailed["w_taker_excess"] / cat_detailed["n_trades"]
    cat_detailed["maker_excess"] = cat_detailed["w_maker_excess"] / cat_detailed["n_trades"]
    cat_detailed["taker_win_rate"] = cat_detailed["w_taker_win_rate"] / cat_detailed["n_trades"]
    cat_detailed["maker_win_rate"] = cat_detailed["w_maker_win_rate"] / cat_detailed["n_trades"]
    cat_detailed["taker_avg_price"] = cat_detailed["w_taker_avg_price"] / cat_detailed["n_trades"]
    cat_detailed["maker_avg_price"] = cat_detailed["w_maker_avg_price"] / cat_detailed["n_trades"]

    # Top 5 categories by total volume (contracts)
    cat_volume = cat_detailed.groupby("category")["total_contracts"].sum().sort_values(ascending=False)
    top5_categories = cat_volume.head(5).index.tolist()
    print(f"[info] Top 5 categories by volume: {top5_categories}", file=sys.stderr)

    cat_top5 = cat_detailed[cat_detailed["category"].isin(top5_categories)].copy()

    # ── Figure 1: Excess Returns by Size Bucket (taker vs maker) ────────────
    print("[info] Creating Figure 1: Excess by size...", file=sys.stderr)

    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor="white")

    bucket_labels = df_main["size_bucket"].tolist()
    taker_excess = df_main["taker_excess"].tolist()
    maker_excess = df_main["maker_excess"].tolist()
    x_pos = np.arange(len(bucket_labels))
    width = 0.35

    bars1 = ax.bar(x_pos - width / 2, taker_excess, width, color=BLUE,
                   edgecolor="white", linewidth=0.5, alpha=0.90, label="Taker Excess Return")
    bars2 = ax.bar(x_pos + width / 2, maker_excess, width, color=GREEN,
                   edgecolor="white", linewidth=0.5, alpha=0.90, label="Maker Excess Return")

    # Value labels
    for bar, val in zip(bars1, taker_excess):
        va = "bottom" if val >= 0 else "top"
        offset = 0.002 if val >= 0 else -0.002
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                f"{val:+.2%}", ha="center", va=va, fontsize=10, fontweight="bold",
                color=BLUE)
    for bar, val in zip(bars2, maker_excess):
        va = "bottom" if val >= 0 else "top"
        offset = 0.002 if val >= 0 else -0.002
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                f"{val:+.2%}", ha="center", va=va, fontsize=10, fontweight="bold",
                color=GREEN)

    ax.axhline(0, color="black", linewidth=1, linestyle="-", alpha=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bucket_labels, fontsize=11)
    ax.set_xlabel("Trade Size (Contracts)", fontsize=12)
    ax.set_ylabel("Excess Return (Win Rate - Avg Price Paid)", fontsize=12)

    total_trades = int(df_main["n_trades"].sum())
    ax.set_title(
        "Taker vs Maker Excess Return by Trade Size Bucket\n"
        f"(N={total_trades:,} resolved Kalshi trades)",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=11, loc="best")
    ax.grid(axis="y", alpha=GRID_ALPHA)
    fig.tight_layout()

    path1 = f"{FIG_DIR}/2_5_excess_by_size.png"
    fig.savefig(path1, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"[info] Saved {path1}", file=sys.stderr)

    # ── Figure 2: Maker Excess by Size x Category (heatmap) ────────────────
    print("[info] Creating Figure 2: Size x category heatmap...", file=sys.stderr)

    bucket_order_map = {"1": 1, "2-10": 2, "11-100": 3, "101-1K": 4, "1K+": 5}
    bucket_labels_ordered = ["1", "2-10", "11-100", "101-1K", "1K+"]

    # Build the heatmap matrix
    heatmap_data = np.full((len(top5_categories), len(bucket_labels_ordered)), np.nan)
    n_trades_data = np.full((len(top5_categories), len(bucket_labels_ordered)), 0)

    for _, row in cat_top5.iterrows():
        cat = row["category"]
        bucket = row["size_bucket"]
        if cat in top5_categories and bucket in bucket_labels_ordered:
            r = top5_categories.index(cat)
            c = bucket_labels_ordered.index(bucket)
            heatmap_data[r, c] = row["maker_excess"]
            n_trades_data[r, c] = int(row["n_trades"])

    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor="white")

    # Custom diverging colormap centered at 0
    vmax = np.nanmax(np.abs(heatmap_data)) if not np.all(np.isnan(heatmap_data)) else 0.1
    vmax = max(vmax, 0.01)  # floor

    im = ax.imshow(heatmap_data, cmap="RdYlGn", aspect="auto",
                   vmin=-vmax, vmax=vmax)

    # Add text annotations
    for i in range(len(top5_categories)):
        for j in range(len(bucket_labels_ordered)):
            val = heatmap_data[i, j]
            n = n_trades_data[i, j]
            if np.isnan(val):
                text = "N/A"
                color = GRAY
            else:
                text = f"{val:+.2%}\n(n={n:,})"
                # Choose text color based on background brightness
                color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                    fontweight="bold", color=color)

    ax.set_xticks(np.arange(len(bucket_labels_ordered)))
    ax.set_xticklabels(bucket_labels_ordered, fontsize=11)
    ax.set_yticks(np.arange(len(top5_categories)))
    ax.set_yticklabels(top5_categories, fontsize=11)
    ax.set_xlabel("Trade Size (Contracts)", fontsize=12)
    ax.set_ylabel("Category", fontsize=12)
    ax.set_title(
        "Maker Excess Return by Trade Size and Category\n"
        "(Green = maker profits, Red = maker losses / taker informed)",
        fontsize=13, fontweight="bold"
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Maker Excess Return", fontsize=11)
    fig.tight_layout()

    path2 = f"{FIG_DIR}/2_5_size_by_category.png"
    fig.savefig(path2, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"[info] Saved {path2}", file=sys.stderr)

    # ── Stdout JSON ──────────────────────────────────────────────────────────
    total_elapsed = time.time() - t0

    # Build cross-tab JSON
    cross_tab_json = []
    for _, row in cat_top5.iterrows():
        cross_tab_json.append({
            "category": row["category"],
            "size_bucket": row["size_bucket"],
            "taker_excess": round(float(row["taker_excess"]), 4),
            "maker_excess": round(float(row["maker_excess"]), 4),
            "taker_win_rate": round(float(row["taker_win_rate"]), 4),
            "maker_win_rate": round(float(row["maker_win_rate"]), 4),
            "n_trades": int(row["n_trades"]),
            "total_contracts": int(row["total_contracts"]),
        })

    output = {
        "n_trades_total": int(df_main["n_trades"].sum()),
        "total_contracts": int(df_main["total_contracts"].sum()),
        "elapsed_seconds": round(total_elapsed, 1),
        "size_bucket_stats": [
            {
                "size_bucket": row["size_bucket"],
                "taker_win_rate": round(float(row["taker_win_rate"]), 4),
                "taker_avg_price": round(float(row["taker_avg_price"]), 4),
                "taker_excess": round(float(row["taker_excess"]), 4),
                "maker_win_rate": round(float(row["maker_win_rate"]), 4),
                "maker_avg_price": round(float(row["maker_avg_price"]), 4),
                "maker_excess": round(float(row["maker_excess"]), 4),
                "n_trades": int(row["n_trades"]),
                "total_contracts": int(row["total_contracts"]),
            }
            for _, row in df_main.iterrows()
        ],
        "top5_categories": top5_categories,
        "cross_tab": cross_tab_json,
        "figures": [path1, path2],
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
