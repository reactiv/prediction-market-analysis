#!/usr/bin/env python3
"""
Strategy Report 2.2: Maker Alpha by Category

Computes maker vs taker excess returns by market category. Tests whether
the maker/taker gap varies by category. Hypothesis: the gap should be larger
in categories with more retail participation (sports) and smaller in
categories with more sophisticated participants (finance).

Self-contained: no src.* imports, uses inline category resolution.
"""

import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import duckdb  # noqa: E402

# ── Palette & style ──────────────────────────────────────────────────────────
BLUE = "#4C72B0"
GREEN = "#2ecc71"
RED = "#e74c3c"
FIG_SIZE = (12, 7)
DPI = 150
GRID_ALPHA = 0.3

TRADES_GLOB = "data/kalshi/trades/*.parquet"
MARKETS_GLOB = "data/kalshi/markets/*.parquet"
FIG_DIR = "strategy_reports/figures"

# ── Inline category mapping ─────────────────────────────────────────────────
CATEGORY_KEYWORDS = {
    "Sports": {"NBA", "NFL", "MLB", "NHL", "WNBA", "MLS", "NCAAF", "NCAAB", "UFC", "PGA", "LPGA", "NASCAR", "F1", "TENNIS", "SOCCER", "CRICKET", "BOXING", "GOLF", "RUGBY", "EPL", "SERI", "BUND", "LIGA", "LIGUE", "CHAM"},
    "Politics": {"PRES", "HOUSE", "SENATE", "GOV", "ELECT", "POTUS", "DEM", "REP", "APPR", "TRUMP", "BIDEN", "HARRIS", "PARTY", "VOTE", "INAUG", "PRIM", "CAND"},
    "Crypto": {"BTC", "ETH", "SOL", "DOGE", "XRP", "CRYPTO", "COIN", "DEFI", "NFT"},
    "Finance": {"NASDAQ", "SPX", "INX", "DJIA", "FTSE", "RUSSEL", "EURUSD", "USDJPY", "GBPUSD", "FED", "CPI", "GDP", "UNEMP", "RATE", "YIELD", "BOND", "OIL", "GOLD", "SILVER", "COMMODIT"},
    "Weather": {"TEMP", "RAIN", "SNOW", "HURRICANE", "TORNADO", "DROUGHT", "CLIMATE", "WEATHER", "FLOOD", "WILDFIRE", "STORM", "HEAT"},
    "Entertainment": {"OSCAR", "GRAMMY", "EMMY", "TONY", "GLOBE", "SUPER", "IDOL", "BACHELOR", "SURVIVOR", "MOVIE", "TIKTOK", "YOUTUBE", "SPOTIFY", "TWITCH", "STREAM"},
    "Science/Tech": {"COVID", "FDA", "VACCINE", "VIRUS", "SPACE", "NASA", "LAUNCH", "AI", "TECH", "APPLE", "GOOGLE", "META", "AMAZON", "TSLA", "SPACEX"},
}


def get_group(cat: str) -> str:
    cat_upper = cat.upper()
    for group, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in cat_upper:
                return group
    return "Other"


CATEGORY_SQL = "regexp_extract(m.event_ticker, '^([A-Z0-9]+)', 1)"


def main() -> None:
    t0 = time.time()
    con = duckdb.connect()

    # ── Main query: maker/taker stats by category prefix ─────────────────────
    print("[info] Running maker/taker by category query...", file=sys.stderr)

    sql = f"""
    WITH resolved_markets AS (
        SELECT ticker, event_ticker, result
        FROM '{MARKETS_GLOB}'
        WHERE status = 'finalized'
          AND result IN ('yes', 'no')
    ),
    taker_positions AS (
        SELECT
            {CATEGORY_SQL} AS category,
            CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS price,
            CASE WHEN t.taker_side = m.result THEN 1.0 ELSE 0.0 END AS won,
            t.count AS contracts,
            t.count * (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS volume_usd
        FROM '{TRADES_GLOB}' t
        INNER JOIN resolved_markets m ON t.ticker = m.ticker
    ),
    maker_positions AS (
        SELECT
            {CATEGORY_SQL} AS category,
            CASE WHEN t.taker_side = 'yes' THEN t.no_price ELSE t.yes_price END AS price,
            CASE WHEN t.taker_side != m.result THEN 1.0 ELSE 0.0 END AS won,
            t.count AS contracts,
            t.count * (CASE WHEN t.taker_side = 'yes' THEN t.no_price ELSE t.yes_price END) / 100.0 AS volume_usd
        FROM '{TRADES_GLOB}' t
        INNER JOIN resolved_markets m ON t.ticker = m.ticker
    ),
    taker_stats AS (
        SELECT
            category,
            AVG(won) AS win_rate,
            AVG(price / 100.0) AS avg_price,
            AVG(won - price / 100.0) AS excess_return,
            COUNT(*) AS n_trades,
            SUM(contracts) AS contracts,
            SUM(volume_usd) AS volume_usd,
            SUM(contracts * (won - price / 100.0)) AS pnl
        FROM taker_positions
        GROUP BY category
    ),
    maker_stats AS (
        SELECT
            category,
            AVG(won) AS win_rate,
            AVG(price / 100.0) AS avg_price,
            AVG(won - price / 100.0) AS excess_return,
            COUNT(*) AS n_trades,
            SUM(contracts) AS contracts,
            SUM(volume_usd) AS volume_usd,
            SUM(contracts * (won - price / 100.0)) AS pnl
        FROM maker_positions
        GROUP BY category
    )
    SELECT
        t.category,
        t.win_rate AS taker_win_rate,
        t.avg_price AS taker_avg_price,
        t.excess_return AS taker_excess,
        t.n_trades AS taker_n,
        t.contracts AS taker_contracts,
        t.volume_usd AS taker_volume,
        t.pnl AS taker_pnl,
        mk.win_rate AS maker_win_rate,
        mk.avg_price AS maker_avg_price,
        mk.excess_return AS maker_excess,
        mk.n_trades AS maker_n,
        mk.contracts AS maker_contracts,
        mk.volume_usd AS maker_volume,
        mk.pnl AS maker_pnl
    FROM taker_stats t
    JOIN maker_stats mk ON t.category = mk.category
    ORDER BY t.volume_usd DESC
    """

    df = con.execute(sql).fetchdf()
    elapsed = time.time() - t0
    print(f"[info] Query done in {elapsed:.1f}s, {len(df)} category prefixes", file=sys.stderr)

    if df.empty:
        print(json.dumps({"error": "No qualifying trades found"}))
        sys.exit(1)

    # ── Map prefixes to category groups ──────────────────────────────────────
    df["group"] = df["category"].apply(
        lambda x: get_group(x) if isinstance(x, str) else "Other"
    )

    # ── Aggregate by group (volume-weighted) ─────────────────────────────────
    group_stats = []
    for group in df["group"].unique():
        gd = df[df["group"] == group]

        taker_contracts_total = gd["taker_contracts"].sum()
        maker_contracts_total = gd["maker_contracts"].sum()

        # Volume-weighted excess returns
        taker_vol_weighted = (
            (gd["taker_excess"] * gd["taker_contracts"]).sum() / taker_contracts_total
            if taker_contracts_total > 0 else 0.0
        )
        maker_vol_weighted = (
            (gd["maker_excess"] * gd["maker_contracts"]).sum() / maker_contracts_total
            if maker_contracts_total > 0 else 0.0
        )

        group_stats.append({
            "group": group,
            "taker_excess_pp": taker_vol_weighted * 100,
            "maker_excess_pp": maker_vol_weighted * 100,
            "gap_pp": (maker_vol_weighted - taker_vol_weighted) * 100,
            "taker_n": int(gd["taker_n"].sum()),
            "maker_n": int(gd["maker_n"].sum()),
            "taker_volume": float(gd["taker_volume"].sum()),
            "maker_volume": float(gd["maker_volume"].sum()),
            "taker_pnl": float(gd["taker_pnl"].sum()),
            "maker_pnl": float(gd["maker_pnl"].sum()),
            "taker_contracts": int(taker_contracts_total),
            "maker_contracts": int(maker_contracts_total),
        })

    import pandas as pd
    group_df = pd.DataFrame(group_stats)
    group_df = group_df.sort_values("taker_volume", ascending=False).reset_index(drop=True)

    print(f"[info] {len(group_df)} category groups after mapping", file=sys.stderr)
    for _, row in group_df.iterrows():
        print(
            f"  {row['group']:15s}  taker={row['taker_excess_pp']:+.2f}pp  "
            f"maker={row['maker_excess_pp']:+.2f}pp  gap={row['gap_pp']:.2f}pp  "
            f"trades={row['taker_n']:,}",
            file=sys.stderr,
        )

    # Take top 8 groups by volume for plotting
    top_groups = group_df.head(8)

    # ── Figure 1: Grouped bar chart — Maker vs Taker excess by category ──────
    print("[info] Creating Figure 1: Maker vs Taker by category...", file=sys.stderr)

    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor="white")

    categories = top_groups["group"].tolist()
    taker_vals = top_groups["taker_excess_pp"].tolist()
    maker_vals = top_groups["maker_excess_pp"].tolist()

    x = np.arange(len(categories))
    width = 0.35

    bars_taker = ax.bar(
        x - width / 2, taker_vals, width,
        label="Taker Excess Return", color=RED,
        edgecolor="white", linewidth=0.5, alpha=0.85,
    )
    bars_maker = ax.bar(
        x + width / 2, maker_vals, width,
        label="Maker Excess Return", color=GREEN,
        edgecolor="white", linewidth=0.5, alpha=0.85,
    )

    # Value labels
    for bar, val in zip(bars_taker, taker_vals):
        va = "bottom" if val >= 0 else "top"
        offset = 0.03 if val >= 0 else -0.03
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
            f"{val:+.2f}", ha="center", va=va, fontsize=9, fontweight="bold",
            color=RED,
        )
    for bar, val in zip(bars_maker, maker_vals):
        va = "bottom" if val >= 0 else "top"
        offset = 0.03 if val >= 0 else -0.03
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
            f"{val:+.2f}", ha="center", va=va, fontsize=9, fontweight="bold",
            color=GREEN,
        )

    ax.axhline(0, color="black", linewidth=1, linestyle="-", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=11)
    ax.set_xlabel("Category", fontsize=12)
    ax.set_ylabel("Volume-Weighted Excess Return (pp)", fontsize=12)

    total_trades = int(group_df["taker_n"].sum())
    ax.set_title(
        "Maker vs Taker Excess Return by Category\n"
        f"(N={total_trades:,} resolved Kalshi trades, volume-weighted)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=11, loc="best")
    ax.grid(axis="y", alpha=GRID_ALPHA)
    fig.tight_layout()

    path1 = f"{FIG_DIR}/2_2_maker_taker_by_category.png"
    fig.savefig(path1, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"[info] Saved {path1}", file=sys.stderr)

    # ── Figure 2: Maker-Taker gap by category ───────────────────────────────
    print("[info] Creating Figure 2: Maker-Taker gap by category...", file=sys.stderr)

    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor="white")

    gap_vals = top_groups["gap_pp"].tolist()

    bars_gap = ax.bar(
        x, gap_vals, width=0.6,
        color=BLUE, edgecolor="white", linewidth=0.5, alpha=0.85,
    )

    # Value labels
    for bar, val in zip(bars_gap, gap_vals):
        va = "bottom" if val >= 0 else "top"
        offset = 0.03 if val >= 0 else -0.03
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
            f"{val:.2f}", ha="center", va=va, fontsize=10, fontweight="bold",
            color=BLUE,
        )

    ax.axhline(0, color="black", linewidth=1, linestyle="-", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=11)
    ax.set_xlabel("Category", fontsize=12)
    ax.set_ylabel("Maker - Taker Gap (pp)", fontsize=12)
    ax.set_title(
        "Maker Edge over Taker by Category\n"
        f"(Maker Excess - Taker Excess, volume-weighted, N={total_trades:,})",
        fontsize=13, fontweight="bold",
    )
    ax.grid(axis="y", alpha=GRID_ALPHA)
    fig.tight_layout()

    path2 = f"{FIG_DIR}/2_2_maker_edge_by_category.png"
    fig.savefig(path2, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"[info] Saved {path2}", file=sys.stderr)

    # ── Stdout JSON ──────────────────────────────────────────────────────────
    total_elapsed = time.time() - t0

    category_json = []
    for _, row in group_df.iterrows():
        category_json.append({
            "category": row["group"],
            "taker_excess_pp": round(float(row["taker_excess_pp"]), 4),
            "maker_excess_pp": round(float(row["maker_excess_pp"]), 4),
            "gap_pp": round(float(row["gap_pp"]), 4),
            "taker_n": int(row["taker_n"]),
            "taker_volume": round(float(row["taker_volume"]), 2),
            "maker_volume": round(float(row["maker_volume"]), 2),
            "taker_pnl": round(float(row["taker_pnl"]), 2),
            "maker_pnl": round(float(row["maker_pnl"]), 2),
            "taker_contracts": int(row["taker_contracts"]),
            "maker_contracts": int(row["maker_contracts"]),
        })

    output = {
        "n_trades_total": total_trades,
        "elapsed_seconds": round(total_elapsed, 1),
        "category_stats": category_json,
        "figures": [path1, path2],
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
