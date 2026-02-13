#!/usr/bin/env python3
"""
Strategy Report 5.1: Smart Money Following

Identifies the most profitable addresses on Polymarket by computing
per-address win rates, average purchase prices, and excess returns
(win_rate - avg_price). Compares top-percentile "smart money" to
the median address to quantify edge persistence.

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

MIN_TRADES = 50  # minimum resolved trades for statistical significance


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
            winning_outcome = None
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

    # ── Step 2: Extract buyer-level trade data joined with resolution ─────
    # For each trade, identify the BUYER (the one who buys the token):
    #   maker_asset_id = '0' => maker is buyer, token = taker_asset_id
    #   else => taker is buyer, token = maker_asset_id
    print("[info] Querying trades (this may take several minutes)...", file=sys.stderr)

    address_stats_sql = f"""
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
                CASE
                    WHEN t.maker_asset_id = '0'
                        THEN CAST(t.taker_amount AS DOUBLE)
                    ELSE CAST(t.maker_amount AS DOUBLE)
                END AS token_amount,
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
            COUNT(*)                                         AS n_trades,
            AVG(CASE WHEN won THEN 1.0 ELSE 0.0 END)        AS win_rate,
            AVG(price)                                       AS avg_price,
            AVG(CASE WHEN won THEN 1.0 ELSE 0.0 END) - AVG(price) AS excess_return,
            SUM(CASE WHEN won THEN (1.0 - price) ELSE -price END * token_amount / 1e6) AS total_pnl
        FROM buyer_trades
        GROUP BY address
        HAVING COUNT(*) >= {MIN_TRADES}
        ORDER BY excess_return DESC
    """

    df = con.execute(address_stats_sql).fetchdf()
    elapsed = time.time() - t0
    print(f"[info] Query complete in {elapsed:.1f}s, {len(df)} qualifying addresses", file=sys.stderr)

    if df.empty:
        print(json.dumps({"error": "No qualifying addresses found"}))
        sys.exit(1)

    # ── Step 3: Compute summary statistics ────────────────────────────────
    n_addresses = len(df)
    excess_arr = df["excess_return"].values
    win_rate_arr = df["win_rate"].values
    avg_price_arr = df["avg_price"].values
    pnl_arr = df["total_pnl"].values

    # Percentile thresholds
    p99 = np.percentile(excess_arr, 99)
    p95 = np.percentile(excess_arr, 95)
    p90 = np.percentile(excess_arr, 90)

    top1_mask = excess_arr >= p99
    top5_mask = excess_arr >= p95
    top10_mask = excess_arr >= p90
    rest_mask = excess_arr < p90

    tier_stats = {}
    for label, mask in [
        ("top_1pct", top1_mask),
        ("top_5pct", top5_mask),
        ("top_10pct", top10_mask),
        ("bottom_90pct", rest_mask),
        ("all", np.ones(n_addresses, dtype=bool)),
    ]:
        tier_stats[label] = {
            "n_addresses": int(mask.sum()),
            "mean_excess_return": round(float(excess_arr[mask].mean()), 4),
            "median_excess_return": round(float(np.median(excess_arr[mask])), 4),
            "mean_win_rate": round(float(win_rate_arr[mask].mean()), 4),
            "mean_avg_price": round(float(avg_price_arr[mask].mean()), 4),
            "mean_pnl": round(float(pnl_arr[mask].mean()), 2),
            "median_pnl": round(float(np.median(pnl_arr[mask])), 2),
            "mean_trades": round(float(df.loc[mask, "n_trades"].mean()), 1),
        }

    # Top 20 addresses
    top20 = df.head(20)
    top20_list = []
    for _, row in top20.iterrows():
        top20_list.append({
            "address": row["address"],
            "address_short": row["address"][:6] + "..." + row["address"][-4:],
            "n_trades": int(row["n_trades"]),
            "win_rate": round(float(row["win_rate"]), 4),
            "avg_price": round(float(row["avg_price"]), 4),
            "excess_return": round(float(row["excess_return"]), 4),
            "total_pnl": round(float(row["total_pnl"]), 2),
        })

    # ── Figure 1: Distribution of excess returns ─────────────────────────
    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor="white")

    clip_lo, clip_hi = -0.5, 0.5
    clipped = np.clip(excess_arr, clip_lo, clip_hi)
    bins = np.linspace(clip_lo, clip_hi, 81)

    ax.hist(clipped, bins=bins, color=BLUE, edgecolor="white", linewidth=0.3,
            alpha=0.85, label="Addresses")
    ax.axvline(0, color=RED, linewidth=2, linestyle="--", label="Break-even")
    median_excess = float(np.median(excess_arr))
    ax.axvline(median_excess, color=ORANGE, linewidth=2, linestyle="-.",
               label=f"Median ({median_excess:.3f})")
    ax.axvline(p95, color=GREEN, linewidth=2, linestyle=":",
               label=f"Top 5% threshold ({p95:.3f})")

    ax.set_xlabel("Excess Return (win_rate - avg_price)", fontsize=12)
    ax.set_ylabel("Number of Addresses", fontsize=12)
    ax.set_title(
        f"Distribution of Address Excess Returns\n"
        f"(N={n_addresses:,} addresses with >= {MIN_TRADES} resolved trades)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=GRID_ALPHA)
    fig.tight_layout()
    path1 = f"{FIG_DIR}/5_1_smart_money_distribution.png"
    fig.savefig(path1, dpi=DPI, facecolor="white")
    plt.close(fig)

    # ── Figure 2: Smart Money vs Rest comparison ─────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=FIG_SIZE, facecolor="white")

    tier_labels = ["Top 1%", "Top 5%", "Top 10%", "Bottom 90%"]
    tier_masks = [top1_mask, top5_mask, top10_mask, rest_mask]
    tier_colors = [GREEN, ORANGE, PURPLE, BLUE]

    # Panel A: Mean Excess Return by tier
    ax = axes[0]
    means = [float(excess_arr[m].mean()) for m in tier_masks]
    bars = ax.bar(tier_labels, means, color=tier_colors, edgecolor="white",
                  linewidth=0.5, alpha=0.85)
    ax.axhline(0, color=RED, linewidth=1, linestyle="--", alpha=0.7)
    ax.set_ylabel("Mean Excess Return", fontsize=11)
    ax.set_title("Excess Return by Tier", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=GRID_ALPHA)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)

    # Panel B: Mean Win Rate vs Mean Avg Price
    ax = axes[1]
    wr = [float(win_rate_arr[m].mean()) for m in tier_masks]
    ap = [float(avg_price_arr[m].mean()) for m in tier_masks]
    x_pos = np.arange(len(tier_labels))
    width = 0.35
    ax.bar(x_pos - width / 2, wr, width, color=GREEN, label="Win Rate",
           edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.bar(x_pos + width / 2, ap, width, color=RED, label="Avg Price",
           edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tier_labels, fontsize=9)
    ax.set_ylabel("Rate", fontsize=11)
    ax.set_title("Win Rate vs Avg Price", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.tick_params(axis="x", rotation=30)

    # Panel C: Mean PnL by tier
    ax = axes[2]
    pnls = [float(pnl_arr[m].mean()) for m in tier_masks]
    bars = ax.bar(tier_labels, pnls, color=tier_colors, edgecolor="white",
                  linewidth=0.5, alpha=0.85)
    ax.axhline(0, color=RED, linewidth=1, linestyle="--", alpha=0.7)
    ax.set_ylabel("Mean Est. PnL (USDC)", fontsize=11)
    ax.set_title("Mean PnL by Tier", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=GRID_ALPHA)
    for bar, val in zip(bars, pnls):
        va = "bottom" if val >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"${val:,.0f}", ha="center", va=va, fontsize=9, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)

    fig.suptitle(
        "Smart Money vs Rest: Performance Comparison",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    path2 = f"{FIG_DIR}/5_1_smart_money_vs_rest.png"
    fig.savefig(path2, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    # ── Stdout JSON ──────────────────────────────────────────────────────
    output = {
        "n_qualifying_addresses": n_addresses,
        "min_trades_threshold": MIN_TRADES,
        "elapsed_seconds": round(time.time() - t0, 1),
        "overall": {
            "mean_excess_return": round(float(np.mean(excess_arr)), 4),
            "median_excess_return": round(float(np.median(excess_arr)), 4),
            "std_excess_return": round(float(np.std(excess_arr)), 4),
            "mean_win_rate": round(float(np.mean(win_rate_arr)), 4),
            "mean_avg_price": round(float(np.mean(avg_price_arr)), 4),
            "pct_positive_excess": round(float(np.mean(excess_arr > 0) * 100), 2),
        },
        "tier_stats": tier_stats,
        "top_20_addresses": top20_list,
        "percentile_thresholds": {
            "p99_excess_return": round(float(p99), 4),
            "p95_excess_return": round(float(p95), 4),
            "p90_excess_return": round(float(p90), 4),
        },
        "figures": [path1, path2],
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
