#!/usr/bin/env python3
"""
Strategy Report 5.3: Address Clustering

Classifies Polymarket addresses by trading behavior: trade frequency,
average size, and maker vs taker ratio. Identifies archetypes
(market maker, retail, whale, bot/HFT). Compares win rates and excess
returns per cluster.

Self-contained: no src.* imports, uses inline token resolution.
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
RED = "#e74c3c"
GREEN = "#2ecc71"
ORANGE = "#ff7f0e"
PURPLE = "#9467bd"
FIG_SIZE = (12, 7)
DPI = 150
GRID_ALPHA = 0.3

TRADES_GLOB = "data/polymarket/trades/*.parquet"
MARKETS_GLOB = "data/polymarket/markets/*.parquet"
FIG_DIR = "strategy_reports/figures"

CLUSTER_COLORS = {
    "Market Maker": BLUE,
    "Retail": RED,
    "Whale": GREEN,
    "Bot/HFT": ORANGE,
    "Other": PURPLE,
}
CLUSTER_ORDER = ["Market Maker", "Retail", "Whale", "Bot/HFT", "Other"]


def build_token_resolution(con):
    """Build CTF token_id -> won mapping from resolved markets (inline)."""
    markets_df = con.execute(f"""
        SELECT id, clob_token_ids, outcome_prices
        FROM '{MARKETS_GLOB}'
        WHERE closed = true
    """).df()

    token_won = {}
    for _, row in markets_df.iterrows():
        try:
            prices = json.loads(row["outcome_prices"]) if row["outcome_prices"] else None
            if not prices or len(prices) != 2:
                continue
            p0, p1 = float(prices[0]), float(prices[1])
            if p0 > 0.99 and p1 < 0.01:
                winning_outcome = 0
            elif p0 < 0.01 and p1 > 0.99:
                winning_outcome = 1
            else:
                continue
            token_ids = json.loads(row["clob_token_ids"]) if row["clob_token_ids"] else None
            if token_ids and len(token_ids) == 2:
                token_won[token_ids[0]] = winning_outcome == 0
                token_won[token_ids[1]] = winning_outcome == 1
        except Exception:
            continue

    con.execute("CREATE TABLE token_resolution (token_id VARCHAR, won BOOLEAN)")
    con.executemany(
        "INSERT INTO token_resolution VALUES (?, ?)", list(token_won.items())
    )
    return len(token_won)


def main() -> None:
    t0 = time.time()
    con = duckdb.connect()

    # ── Step 1: Token resolution ──────────────────────────────────────────
    n_tokens = build_token_resolution(con)
    print(f"[info] Built resolution table for {n_tokens} tokens", file=sys.stderr)

    # ── Step 2: Compute per-address features from ALL trades ──────────────
    # Single pass over all trades to compute:
    #   - n_trades, n_as_maker, n_as_taker, maker_ratio
    #   - avg_amount (USDC), total_volume
    print("[info] Computing per-address features from all trades...", file=sys.stderr)

    address_features_sql = f"""
        WITH trades_expanded AS (
            SELECT
                t.maker AS maker_addr,
                t.taker AS taker_addr,
                -- USDC amount: whichever side is paying cash
                CASE
                    WHEN t.maker_asset_id = '0'
                        THEN CAST(t.maker_amount AS DOUBLE) / 1e6
                    ELSE CAST(t.taker_amount AS DOUBLE) / 1e6
                END AS usdc_amount
            FROM '{TRADES_GLOB}' t
            WHERE t.taker_amount > 0 AND t.maker_amount > 0
        ),
        maker_stats AS (
            SELECT
                maker_addr AS address,
                COUNT(*) AS n_as_maker,
                0 AS n_as_taker,
                SUM(usdc_amount) AS maker_volume,
                0.0 AS taker_volume
            FROM trades_expanded
            GROUP BY maker_addr
        ),
        taker_stats AS (
            SELECT
                taker_addr AS address,
                0 AS n_as_maker,
                COUNT(*) AS n_as_taker,
                0.0 AS maker_volume,
                SUM(usdc_amount) AS taker_volume
            FROM trades_expanded
            GROUP BY taker_addr
        ),
        combined AS (
            SELECT * FROM maker_stats
            UNION ALL
            SELECT * FROM taker_stats
        )
        SELECT
            address,
            SUM(n_as_maker) AS n_as_maker,
            SUM(n_as_taker) AS n_as_taker,
            SUM(n_as_maker) + SUM(n_as_taker) AS n_trades,
            CAST(SUM(n_as_maker) AS DOUBLE) / (SUM(n_as_maker) + SUM(n_as_taker)) AS maker_ratio,
            (SUM(maker_volume) + SUM(taker_volume)) / (SUM(n_as_maker) + SUM(n_as_taker)) AS avg_amount,
            SUM(maker_volume) + SUM(taker_volume) AS total_volume
        FROM combined
        GROUP BY address
        HAVING (SUM(n_as_maker) + SUM(n_as_taker)) >= 1
    """

    features_df = con.execute(address_features_sql).fetchdf()
    elapsed1 = time.time() - t0
    print(f"[info] Address features computed in {elapsed1:.1f}s, "
          f"{len(features_df):,} addresses", file=sys.stderr)

    # ── Step 3: Compute resolved buyer stats per address ──────────────────
    print("[info] Computing resolved buyer stats per address...", file=sys.stderr)

    buyer_stats_sql = f"""
        WITH buyer_trades AS (
            SELECT
                CASE
                    WHEN t.maker_asset_id = '0' THEN t.maker
                    ELSE t.taker
                END AS address,
                CASE
                    WHEN t.maker_asset_id = '0'
                        THEN CAST(t.maker_amount AS DOUBLE) / CAST(t.taker_amount AS DOUBLE)
                    ELSE CAST(t.taker_amount AS DOUBLE) / CAST(t.maker_amount AS DOUBLE)
                END AS price,
                tr.won
            FROM '{TRADES_GLOB}' t
            INNER JOIN token_resolution tr ON (
                CASE
                    WHEN t.maker_asset_id = '0' THEN t.taker_asset_id
                    ELSE t.maker_asset_id
                END = tr.token_id
            )
            WHERE t.taker_amount > 0 AND t.maker_amount > 0
        )
        SELECT
            address,
            COUNT(*) AS n_resolved_trades,
            AVG(CASE WHEN won THEN 1.0 ELSE 0.0 END) AS win_rate,
            AVG(price) AS avg_price,
            AVG(CASE WHEN won THEN 1.0 ELSE 0.0 END) - AVG(price) AS excess_return
        FROM buyer_trades
        WHERE price > 0 AND price < 1
        GROUP BY address
    """

    buyer_df = con.execute(buyer_stats_sql).fetchdf()
    elapsed2 = time.time() - t0
    print(f"[info] Buyer stats computed in {elapsed2:.1f}s, "
          f"{len(buyer_df):,} addresses with resolved trades", file=sys.stderr)

    # ── Step 4: Merge features with buyer stats ───────────────────────────
    merged = features_df.merge(buyer_df, on="address", how="left")
    print(f"[info] Merged: {len(merged):,} addresses total, "
          f"{merged['win_rate'].notna().sum():,} with resolved buyer data", file=sys.stderr)

    # ── Step 5: Classify addresses into archetypes ────────────────────────
    avg_amount = merged["avg_amount"].values
    n_trades = merged["n_trades"].values
    maker_ratio = merged["maker_ratio"].values

    median_amount = float(np.median(avg_amount))
    p95_amount = float(np.percentile(avg_amount, 95))
    p99_trades = float(np.percentile(n_trades, 99))

    print(f"[info] Classification thresholds:", file=sys.stderr)
    print(f"  median avg_amount = ${median_amount:.2f}", file=sys.stderr)
    print(f"  p95 avg_amount    = ${p95_amount:.2f}", file=sys.stderr)
    print(f"  p99 n_trades      = {p99_trades:.0f}", file=sys.stderr)

    # Apply classification rules (priority order: Bot > Whale > Market Maker > Retail > Other)
    clusters = np.full(len(merged), "Other", dtype=object)

    # Retail: maker_ratio < 0.3 AND avg_amount < median
    retail_mask = (maker_ratio < 0.3) & (avg_amount < median_amount)
    clusters[retail_mask] = "Retail"

    # Market Maker: maker_ratio > 0.7 AND n_trades > 100
    mm_mask = (maker_ratio > 0.7) & (n_trades > 100)
    clusters[mm_mask] = "Market Maker"

    # Whale: avg_amount > p95 AND n_trades > 50
    whale_mask = (avg_amount > p95_amount) & (n_trades > 50)
    clusters[whale_mask] = "Whale"

    # Bot/HFT: n_trades > p99 (overrides others)
    bot_mask = n_trades > p99_trades
    clusters[bot_mask] = "Bot/HFT"

    merged["cluster"] = clusters

    # Print cluster counts
    for c in CLUSTER_ORDER:
        n = int((clusters == c).sum())
        print(f"[info] Cluster '{c}': {n:,} addresses", file=sys.stderr)

    # ── Step 6: Aggregate stats per cluster ───────────────────────────────
    cluster_summary = []
    for cluster_name in CLUSTER_ORDER:
        mask = merged["cluster"] == cluster_name
        subset = merged[mask]
        n_addrs = len(subset)
        if n_addrs == 0:
            continue

        avg_trades = float(subset["n_trades"].mean())
        avg_maker_ratio = float(subset["maker_ratio"].mean())
        avg_avg_amount = float(subset["avg_amount"].mean())
        total_vol = float(subset["total_volume"].sum())

        # Resolved buyer stats (only for addresses that have them)
        resolved_mask = subset["win_rate"].notna()
        n_with_resolved = int(resolved_mask.sum())

        if n_with_resolved > 0:
            resolved_subset = subset[resolved_mask]
            mean_win_rate = float(resolved_subset["win_rate"].mean())
            mean_avg_price = float(resolved_subset["avg_price"].mean())
            mean_excess_return = float(resolved_subset["excess_return"].mean())
            std_excess_return = float(resolved_subset["excess_return"].std())
            se_excess = std_excess_return / np.sqrt(n_with_resolved) if n_with_resolved > 1 else 0
        else:
            mean_win_rate = None
            mean_avg_price = None
            mean_excess_return = None
            std_excess_return = None
            se_excess = None

        cluster_summary.append({
            "cluster": cluster_name,
            "n_addresses": n_addrs,
            "avg_trades": avg_trades,
            "avg_maker_ratio": avg_maker_ratio,
            "avg_amount": avg_avg_amount,
            "total_volume": total_vol,
            "n_with_resolved": n_with_resolved,
            "win_rate": mean_win_rate,
            "avg_price": mean_avg_price,
            "excess_return": mean_excess_return,
            "std_excess_return": std_excess_return,
            "se_excess_return": se_excess,
        })

    # ── Figure 1: Cluster Composition ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE, facecolor="white")

    # Panel A: Number of addresses per cluster
    ax = axes[0]
    cluster_names = [s["cluster"] for s in cluster_summary]
    addr_counts = [s["n_addresses"] for s in cluster_summary]
    colors = [CLUSTER_COLORS[c] for c in cluster_names]

    bars = ax.bar(cluster_names, addr_counts, color=colors, edgecolor="white",
                  linewidth=0.5, alpha=0.85)
    for bar, val in zip(bars, addr_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Number of Addresses", fontsize=12)
    ax.set_title("Addresses per Cluster", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.tick_params(axis="x", rotation=20)

    # Panel B: Total volume per cluster
    ax = axes[1]
    volumes = [s["total_volume"] for s in cluster_summary]
    # Convert to billions for readability
    max_vol = max(volumes)
    if max_vol > 1e9:
        vol_display = [v / 1e9 for v in volumes]
        vol_unit = "Volume ($ Billions)"
        vol_fmt = lambda v: f"${v:.1f}B"
    elif max_vol > 1e6:
        vol_display = [v / 1e6 for v in volumes]
        vol_unit = "Volume ($ Millions)"
        vol_fmt = lambda v: f"${v:.1f}M"
    else:
        vol_display = volumes
        vol_unit = "Volume ($)"
        vol_fmt = lambda v: f"${v:,.0f}"

    bars = ax.bar(cluster_names, vol_display, color=colors, edgecolor="white",
                  linewidth=0.5, alpha=0.85)
    for bar, val in zip(bars, vol_display):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                vol_fmt(val), ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel(vol_unit, fontsize=12)
    ax.set_title("Total Volume per Cluster", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.tick_params(axis="x", rotation=20)

    total_addrs = sum(addr_counts)
    fig.suptitle(
        f"Cluster Composition: {total_addrs:,} Polymarket Addresses",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    path1 = f"{FIG_DIR}/5_3_cluster_composition.png"
    fig.savefig(path1, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[info] Saved {path1}", file=sys.stderr)

    # ── Figure 2: Excess Returns by Cluster ───────────────────────────────
    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor="white")

    # Only plot clusters with resolved data
    plot_clusters = [s for s in cluster_summary if s["excess_return"] is not None]
    plot_names = [s["cluster"] for s in plot_clusters]
    plot_excess = [s["excess_return"] for s in plot_clusters]
    plot_se = [s["se_excess_return"] * 1.96 for s in plot_clusters]  # 95% CI
    plot_colors = [CLUSTER_COLORS[c] for c in plot_names]

    x_pos = np.arange(len(plot_names))
    bars = ax.bar(x_pos, plot_excess, width=0.6, color=plot_colors,
                  edgecolor="white", linewidth=0.5, alpha=0.90,
                  yerr=plot_se, capsize=6,
                  error_kw={"linewidth": 2, "color": "black"})

    ax.axhline(0, color="black", linewidth=1, linestyle="-", alpha=0.5)

    for bar, val, ci in zip(bars, plot_excess, plot_se):
        va = "bottom" if val >= 0 else "top"
        offset = max(abs(ci), 0.001) + 0.002
        y_pos = val + ci + 0.002 if val >= 0 else val - ci - 0.002
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"{val:+.2%}", ha="center", va=va, fontsize=11, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(plot_names, fontsize=11)
    ax.set_ylabel("Excess Return (Win Rate - Avg Price)", fontsize=12)
    ax.set_title(
        "Excess Return by Address Cluster\n"
        "(Mean excess return per cluster, 95% CI shown)",
        fontsize=13, fontweight="bold"
    )
    ax.grid(axis="y", alpha=GRID_ALPHA)
    fig.tight_layout()

    path2 = f"{FIG_DIR}/5_3_cluster_excess_returns.png"
    fig.savefig(path2, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"[info] Saved {path2}", file=sys.stderr)

    # ── Stdout JSON ──────────────────────────────────────────────────────
    total_elapsed = time.time() - t0

    output = {
        "n_total_addresses": int(len(merged)),
        "n_tokens_resolved": n_tokens,
        "elapsed_seconds": round(total_elapsed, 1),
        "classification_thresholds": {
            "median_avg_amount_usdc": round(median_amount, 2),
            "p95_avg_amount_usdc": round(p95_amount, 2),
            "p99_n_trades": round(p99_trades, 0),
        },
        "cluster_summary": [
            {
                "cluster": s["cluster"],
                "n_addresses": s["n_addresses"],
                "pct_of_addresses": round(s["n_addresses"] / len(merged) * 100, 2),
                "avg_trades": round(s["avg_trades"], 1),
                "avg_maker_ratio": round(s["avg_maker_ratio"], 4),
                "avg_amount_usdc": round(s["avg_amount"], 2),
                "total_volume_usdc": round(s["total_volume"], 2),
                "pct_of_volume": round(s["total_volume"] / merged["total_volume"].sum() * 100, 2),
                "n_with_resolved": s["n_with_resolved"],
                "win_rate": round(s["win_rate"], 4) if s["win_rate"] is not None else None,
                "avg_price": round(s["avg_price"], 4) if s["avg_price"] is not None else None,
                "excess_return": round(s["excess_return"], 4) if s["excess_return"] is not None else None,
                "excess_return_se": round(s["se_excess_return"], 4) if s["se_excess_return"] is not None else None,
            }
            for s in cluster_summary
        ],
        "figures": [path1, path2],
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
