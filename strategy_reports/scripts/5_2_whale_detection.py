#!/usr/bin/env python3
"""
Strategy Report 5.2: Whale Detection

Identifies large trades (top 1% by taker_amount) on Polymarket and compares
win rates of large vs small trades. Determines whether whales have superior
information or simply trade bigger.

Self-contained: no src.* imports, uses inline token resolution.
"""

import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import duckdb  # noqa: E402

# ── Palette & style ──────────────────────────────────────────────────────────
BLUE = "#4C72B0"
GRAY = "#888888"
GREEN = "#2ecc71"
RED = "#e74c3c"
ORANGE = "#ff7f0e"
FIG_SIZE = (12, 7)
DPI = 150
GRID_ALPHA = 0.3

TRADES_GLOB = "data/polymarket/trades/*.parquet"
MARKETS_GLOB = "data/polymarket/markets/*.parquet"
FIG_DIR = "strategy_reports/figures"


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
            if not prices or len(prices) < 2:
                continue

            float_prices = [float(p) for p in prices]
            max_price = max(float_prices)
            min_price = min(float_prices)

            # Check if market is resolved (one outcome near 1.0, others near 0.0)
            if max_price < 0.99 or min_price > 0.01:
                continue

            token_ids = json.loads(row["clob_token_ids"]) if row["clob_token_ids"] else None
            if not token_ids or len(token_ids) != len(float_prices):
                continue

            for tid, p in zip(token_ids, float_prices):
                token_won[tid] = p > 0.99
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
    # For each trade on the CTF Exchange, identify the buyer:
    #   maker_asset_id = '0' => maker pays cash -> maker is buyer, token = taker_asset_id
    #   else => taker pays cash -> taker is buyer, token = maker_asset_id
    # taker_amount is the raw token field; we compute trade size as the
    # cash side of the trade (the USDC amount).
    print("[info] Querying trades (this may take several minutes)...", file=sys.stderr)

    trades_sql = f"""
        WITH buyer_trades AS (
            SELECT
                CASE
                    WHEN t.maker_asset_id = '0' THEN t.taker_asset_id
                    ELSE t.maker_asset_id
                END AS token_id,
                -- Price paid per token (USDC / tokens)
                CASE
                    WHEN t.maker_asset_id = '0'
                        THEN CAST(t.maker_amount AS DOUBLE) / CAST(t.taker_amount AS DOUBLE)
                    ELSE CAST(t.taker_amount AS DOUBLE) / CAST(t.maker_amount AS DOUBLE)
                END AS price,
                -- Token amount (the outcome token quantity)
                CASE
                    WHEN t.maker_asset_id = '0'
                        THEN CAST(t.taker_amount AS DOUBLE) / 1e6
                    ELSE CAST(t.maker_amount AS DOUBLE) / 1e6
                END AS token_amount,
                -- Cash amount (USDC quantity, used for size classification)
                CASE
                    WHEN t.maker_asset_id = '0'
                        THEN CAST(t.maker_amount AS DOUBLE) / 1e6
                    ELSE CAST(t.taker_amount AS DOUBLE) / 1e6
                END AS cash_amount,
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
        SELECT price, token_amount, cash_amount, won
        FROM buyer_trades
        WHERE price > 0 AND price < 1
          AND token_amount > 0
          AND cash_amount > 0
    """

    print("[info] Running main query...", file=sys.stderr)
    df = con.execute(trades_sql).fetchdf()
    elapsed = time.time() - t0
    print(f"[info] Query complete in {elapsed:.1f}s, {len(df):,} buyer trades", file=sys.stderr)

    if df.empty:
        print(json.dumps({"error": "No qualifying trades found"}))
        sys.exit(1)

    # ── Step 3: Calculate percentiles on cash_amount (trade size) ─────────
    price = df["price"].values
    won = df["won"].values.astype(float)
    cash_amount = df["cash_amount"].values

    p50 = np.percentile(cash_amount, 50)
    p90 = np.percentile(cash_amount, 90)
    p95 = np.percentile(cash_amount, 95)
    p99 = np.percentile(cash_amount, 99)

    print(f"[info] Percentiles: p50={p50:.2f}, p90={p90:.2f}, "
          f"p95={p95:.2f}, p99={p99:.2f} USDC", file=sys.stderr)

    # ── Step 4: Classify into size buckets ────────────────────────────────
    def classify_bucket(ca):
        if ca < p50:
            return "Small\n(<p50)"
        elif ca < p90:
            return "Medium\n(p50-p90)"
        elif ca < p99:
            return "Large\n(p90-p99)"
        else:
            return "Whale\n(>p99)"

    bucket_labels_ordered = ["Small\n(<p50)", "Medium\n(p50-p90)", "Large\n(p90-p99)", "Whale\n(>p99)"]
    bucket_labels_table = ["Small (<p50)", "Medium (p50-p90)", "Large (p90-p99)", "Whale (>p99)"]

    # Vectorized bucket assignment
    buckets = np.where(
        cash_amount < p50, 0,
        np.where(cash_amount < p90, 1,
                 np.where(cash_amount < p99, 2, 3))
    )

    # ── Step 5: Compute stats per bucket ──────────────────────────────────
    bucket_stats = []
    for i, (label, label_tbl) in enumerate(zip(bucket_labels_ordered, bucket_labels_table)):
        mask = buckets == i
        n = int(mask.sum())
        if n == 0:
            continue

        avg_price = float(price[mask].mean())
        win_rate = float(won[mask].mean())
        excess_return = win_rate - avg_price
        total_volume = float(cash_amount[mask].sum())

        # 95% CI for win rate (binomial)
        se = np.sqrt(win_rate * (1 - win_rate) / n)
        ci_95 = 1.96 * se

        # 95% CI for excess return (same SE since avg_price is the sample mean,
        # but we treat excess = win_rate - avg_price; the variance of excess
        # is harder to compute exactly, so we use a bootstrap-like approach
        # via the binomial SE on win_rate)
        excess_se = ci_95  # approximate

        bucket_stats.append({
            "label": label,
            "label_table": label_tbl,
            "n_trades": n,
            "avg_price": avg_price,
            "win_rate": win_rate,
            "excess_return": excess_return,
            "excess_se": float(se),
            "excess_ci95": float(ci_95),
            "total_volume": total_volume,
        })

    # ── Step 6: Analyze by price range to control for price level ─────────
    price_ranges = [
        ("0.01-0.20", 0.01, 0.20),
        ("0.20-0.40", 0.20, 0.40),
        ("0.40-0.60", 0.40, 0.60),
        ("0.60-0.80", 0.60, 0.80),
        ("0.80-0.99", 0.80, 0.99),
    ]

    price_range_stats = []
    for pr_label, lo, hi in price_ranges:
        pr_mask = (price >= lo) & (price < hi)
        for i, (label, label_tbl) in enumerate(zip(bucket_labels_ordered, bucket_labels_table)):
            combined_mask = pr_mask & (buckets == i)
            n = int(combined_mask.sum())
            if n < 100:
                continue
            wr = float(won[combined_mask].mean())
            ap = float(price[combined_mask].mean())
            price_range_stats.append({
                "price_range": pr_label,
                "size_bucket": label_tbl,
                "n_trades": n,
                "avg_price": round(ap, 4),
                "win_rate": round(wr, 4),
                "excess_return": round(wr - ap, 4),
            })

    # ── Figure 1: Win Rate vs Expected Win Rate by Size Bucket ────────────
    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor="white")

    labels = [s["label"] for s in bucket_stats]
    win_rates = [s["win_rate"] for s in bucket_stats]
    avg_prices = [s["avg_price"] for s in bucket_stats]
    x_pos = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x_pos - width / 2, win_rates, width, color=BLUE,
                   edgecolor="white", linewidth=0.5, alpha=0.90, label="Actual Win Rate")
    bars2 = ax.bar(x_pos + width / 2, avg_prices, width, color=GRAY,
                   edgecolor="white", linewidth=0.5, alpha=0.70, label="Expected (Avg Price)")

    # Add value labels on bars
    for bar, val in zip(bars1, win_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold",
                color=BLUE)
    for bar, val in zip(bars2, avg_prices):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold",
                color=GRAY)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Rate", fontsize=12)
    ax.set_title(
        "Win Rate vs Expected Win Rate (Avg Price) by Trade Size\n"
        f"(N={len(df):,} resolved buyer trades)",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.set_ylim(0, max(max(win_rates), max(avg_prices)) * 1.15)
    fig.tight_layout()

    path1 = f"{FIG_DIR}/5_2_whale_win_rate.png"
    fig.savefig(path1, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"[info] Saved {path1}", file=sys.stderr)

    # ── Figure 2: Excess Return by Size Bucket with Error Bars ────────────
    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor="white")

    excess_returns = [s["excess_return"] for s in bucket_stats]
    excess_cis = [s["excess_ci95"] for s in bucket_stats]

    bar_colors = [BLUE if er >= 0 else RED for er in excess_returns]
    bars = ax.bar(x_pos, excess_returns, width=0.6, color=bar_colors,
                  edgecolor="white", linewidth=0.5, alpha=0.90,
                  yerr=excess_cis, capsize=6, error_kw={"linewidth": 2, "color": "black"})

    ax.axhline(0, color="black", linewidth=1, linestyle="-", alpha=0.5)

    for bar, val, ci in zip(bars, excess_returns, excess_cis):
        va = "bottom" if val >= 0 else "top"
        offset = 0.003 if val >= 0 else -0.003
        ax.text(bar.get_x() + bar.get_width() / 2, val + ci + offset,
                f"{val:+.2%}", ha="center", va=va, fontsize=11, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Excess Return (Win Rate - Avg Price)", fontsize=12)
    ax.set_title(
        "Excess Return by Trade Size Bucket\n"
        "(Positive = Outperforms Market Implied Probability, 95% CI shown)",
        fontsize=13, fontweight="bold"
    )
    ax.grid(axis="y", alpha=GRID_ALPHA)
    fig.tight_layout()

    path2 = f"{FIG_DIR}/5_2_whale_excess_return.png"
    fig.savefig(path2, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"[info] Saved {path2}", file=sys.stderr)

    # ── Stdout JSON ──────────────────────────────────────────────────────
    total_elapsed = time.time() - t0

    output = {
        "n_trades": len(df),
        "n_tokens_resolved": n_tokens,
        "elapsed_seconds": round(total_elapsed, 1),
        "percentiles": {
            "p50": round(float(p50), 2),
            "p90": round(float(p90), 2),
            "p95": round(float(p95), 2),
            "p99": round(float(p99), 2),
        },
        "bucket_stats": [
            {
                "bucket": s["label_table"],
                "n_trades": s["n_trades"],
                "avg_price": round(s["avg_price"], 4),
                "win_rate": round(s["win_rate"], 4),
                "excess_return": round(s["excess_return"], 4),
                "excess_return_ci95": round(s["excess_ci95"], 4),
                "total_volume_usdc": round(s["total_volume"], 2),
            }
            for s in bucket_stats
        ],
        "price_range_analysis": price_range_stats,
        "figures": [path1, path2],
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
