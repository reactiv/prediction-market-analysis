#!/usr/bin/env python3
"""
Strategy Report 5.1 (Temporal Extension): Persistence of Smart Money Edge

Splits the Polymarket trade data into two time periods by block number and
tests whether addresses identified as "smart money" in Period 1 maintain
their edge in Period 2. Measures temporal persistence via correlation of
excess returns across periods and performance decay of top-percentile tiers.

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
FIG_SIZE = (12, 7)
DPI = 150
GRID_ALPHA = 0.3

TRADES_GLOB = "data/polymarket/trades/*.parquet"
MARKETS_GLOB = "data/polymarket/markets/*.parquet"
FIG_DIR = "strategy_reports/figures"

MIN_TRADES = 30  # minimum resolved trades per period
BLOCK_MIDPOINT = 61_000_000  # approximately mid-2024


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


def compute_period_stats(con, period_filter: str, min_trades: int):
    """Compute per-address stats for a given block range filter."""
    sql = f"""
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
            WHERE t.taker_amount > 0
              AND t.maker_amount > 0
              AND {period_filter}
        )
        SELECT
            address,
            COUNT(*)                                         AS n_trades,
            AVG(CASE WHEN won THEN 1.0 ELSE 0.0 END)        AS win_rate,
            AVG(price)                                       AS avg_price,
            AVG(CASE WHEN won THEN 1.0 ELSE 0.0 END) - AVG(price) AS excess_return
        FROM buyer_trades
        GROUP BY address
        HAVING COUNT(*) >= {min_trades}
    """
    return con.execute(sql).fetchdf()


def main() -> None:
    t0 = time.time()
    con = duckdb.connect()

    # ── Step 1: Token resolution ──────────────────────────────────────────
    n_tokens = build_token_resolution(con)
    print(f"[info] Built resolution table for {n_tokens} tokens", file=sys.stderr)

    # ── Step 2: Compute per-period stats ──────────────────────────────────
    print(f"[info] Computing Period 1 stats (blocks < {BLOCK_MIDPOINT})...", file=sys.stderr)
    p1_df = compute_period_stats(
        con, f"t.block_number < {BLOCK_MIDPOINT}", MIN_TRADES
    )
    print(f"[info] Period 1: {len(p1_df)} qualifying addresses", file=sys.stderr)

    print(f"[info] Computing Period 2 stats (blocks >= {BLOCK_MIDPOINT})...", file=sys.stderr)
    p2_df = compute_period_stats(
        con, f"t.block_number >= {BLOCK_MIDPOINT}", MIN_TRADES
    )
    print(f"[info] Period 2: {len(p2_df)} qualifying addresses", file=sys.stderr)

    # ── Step 3: Join on address ───────────────────────────────────────────
    merged = p1_df.merge(p2_df, on="address", suffixes=("_p1", "_p2"))
    n_both = len(merged)
    print(f"[info] Addresses present in both periods: {n_both}", file=sys.stderr)

    if n_both < 10:
        print(json.dumps({"error": f"Only {n_both} addresses in both periods"}))
        sys.exit(1)

    er_p1 = merged["excess_return_p1"].values
    er_p2 = merged["excess_return_p2"].values

    # Pearson correlation
    corr = float(np.corrcoef(er_p1, er_p2)[0, 1])
    print(f"[info] Correlation(P1, P2 excess return) = {corr:.4f}", file=sys.stderr)

    # ── Step 4: Tier analysis ─────────────────────────────────────────────
    # Define tiers based on P1 excess return percentiles
    p99 = np.percentile(er_p1, 99)
    p95 = np.percentile(er_p1, 95)
    p90 = np.percentile(er_p1, 90)

    tier_results = {}
    for label, mask in [
        ("P1 Top 1%", er_p1 >= p99),
        ("P1 Top 5%", er_p1 >= p95),
        ("P1 Top 10%", er_p1 >= p90),
        ("P1 Bottom 90%", er_p1 < p90),
    ]:
        n_tier = int(mask.sum())
        if n_tier == 0:
            continue
        tier_results[label] = {
            "n_addresses": n_tier,
            "mean_p1_excess_return": round(float(er_p1[mask].mean()), 4),
            "mean_p2_excess_return": round(float(er_p2[mask].mean()), 4),
            "median_p2_excess_return": round(float(np.median(er_p2[mask])), 4),
            "mean_p2_win_rate": round(float(merged.loc[mask, "win_rate_p2"].mean()), 4),
            "mean_p2_avg_price": round(float(merged.loc[mask, "avg_price_p2"].mean()), 4),
            "pct_positive_p2": round(float(np.mean(er_p2[mask] > 0) * 100), 1),
        }

    # ── Step 5: Scatter plot ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor="white")

    # Clip for visualization
    clip_lo, clip_hi = -0.5, 0.5
    x_plot = np.clip(er_p1, clip_lo, clip_hi)
    y_plot = np.clip(er_p2, clip_lo, clip_hi)

    # Scatter all points
    ax.scatter(
        x_plot, y_plot,
        alpha=0.15, s=8, color=BLUE, edgecolors="none",
        label=f"All addresses (N={n_both:,})",
    )

    # Highlight top 5% from P1
    top5_mask = er_p1 >= p95
    ax.scatter(
        x_plot[top5_mask], y_plot[top5_mask],
        alpha=0.5, s=18, color=GREEN, edgecolors="none",
        label=f"P1 Top 5% (N={int(top5_mask.sum()):,})",
    )

    # Diagonal reference line
    diag = np.linspace(clip_lo, clip_hi, 100)
    ax.plot(diag, diag, color=RED, linewidth=1.5, linestyle="--",
            alpha=0.7, label="Perfect persistence (y=x)")

    # Zero lines
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="-", alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="-", alpha=0.5)

    # Add correlation annotation
    ax.text(
        0.05, 0.95,
        f"Pearson r = {corr:.3f}",
        transform=ax.transAxes,
        fontsize=12, fontweight="bold",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    ax.set_xlabel("Period 1 Excess Return (blocks < 61M)", fontsize=12)
    ax.set_ylabel("Period 2 Excess Return (blocks >= 61M)", fontsize=12)
    ax.set_title(
        f"Temporal Persistence of Smart Money Edge\n"
        f"({n_both:,} addresses with >= {MIN_TRADES} resolved trades in each period)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=GRID_ALPHA)
    ax.set_xlim(clip_lo, clip_hi)
    ax.set_ylim(clip_lo, clip_hi)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig_path = f"{FIG_DIR}/5_1_smart_money_temporal.png"
    fig.savefig(fig_path, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"[info] Saved figure to {fig_path}", file=sys.stderr)

    # ── Step 6: Regression-to-mean metrics ────────────────────────────────
    # Mean P2 excess return for all addresses present in both periods
    overall_p2_mean = round(float(er_p2.mean()), 4)

    # Elapsed time
    elapsed = time.time() - t0
    print(f"[info] Complete in {elapsed:.1f}s", file=sys.stderr)

    # ── Stdout JSON ──────────────────────────────────────────────────────
    output = {
        "block_midpoint": BLOCK_MIDPOINT,
        "min_trades_per_period": MIN_TRADES,
        "n_period1_addresses": len(p1_df),
        "n_period2_addresses": len(p2_df),
        "n_both_periods": n_both,
        "correlation_p1_p2": round(corr, 4),
        "overall_p2_mean_excess_return": overall_p2_mean,
        "tier_results": tier_results,
        "percentile_thresholds_p1": {
            "p99": round(float(p99), 4),
            "p95": round(float(p95), 4),
            "p90": round(float(p90), 4),
        },
        "elapsed_seconds": round(elapsed, 1),
        "figure": fig_path,
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
