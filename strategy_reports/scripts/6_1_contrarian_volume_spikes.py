#!/usr/bin/env python3
"""
Strategy Report 6.1: Contrarian Volume Spikes

For each market, identify days with >3x the market's average daily volume.
Compare calibration quality of trades on spike days vs normal days.
Do volume spikes reflect informed trading (better calibration) or uninformed
herding (worse calibration)?

Self-contained: no src.* imports, uses DuckDB for all data access.
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
ORANGE = "#ff7f0e"
FIG_SIZE = (12, 7)
DPI = 150
GRID_ALPHA = 0.3

TRADES_GLOB = "data/kalshi/trades/*.parquet"
MARKETS_GLOB = "data/kalshi/markets/*.parquet"
FIG_DIR = "strategy_reports/figures"

SPIKE_MULT = 3.0    # spike threshold: 3x average daily volume
MIN_DAYS = 3         # market must have at least 3 trading days
PRICE_BUCKETS = 10   # number of calibration buckets


def main() -> None:
    t0 = time.time()
    con = duckdb.connect()

    # ── Step 1: Build resolved market lookup ──────────────────────────────
    print("[info] Building resolved market lookup...", file=sys.stderr)
    con.execute(f"""
        CREATE TEMP TABLE resolved_markets AS
        SELECT DISTINCT ticker, result
        FROM '{MARKETS_GLOB}'
        WHERE result IN ('yes', 'no')
    """)
    n_resolved = con.execute("SELECT COUNT(*) FROM resolved_markets").fetchone()[0]
    print(f"[info] {n_resolved:,} resolved market rows", file=sys.stderr)

    # ── Step 2: Compute daily volume per market-day ───────────────────────
    print("[info] Computing daily volumes per market...", file=sys.stderr)
    con.execute(f"""
        CREATE TEMP TABLE daily_vol AS
        SELECT
            t.ticker,
            CAST(t.created_time AS DATE) AS trade_date,
            COUNT(*) AS daily_trades,
            SUM(t.count) AS daily_contracts
        FROM '{TRADES_GLOB}' t
        INNER JOIN resolved_markets rm ON t.ticker = rm.ticker
        GROUP BY t.ticker, CAST(t.created_time AS DATE)
    """)
    n_market_days = con.execute("SELECT COUNT(*) FROM daily_vol").fetchone()[0]
    print(f"[info] {n_market_days:,} market-days computed", file=sys.stderr)

    # ── Step 3: Compute market averages and classify spike days ───────────
    print("[info] Computing market averages and classifying spike days...", file=sys.stderr)
    con.execute(f"""
        CREATE TEMP TABLE market_avg AS
        SELECT
            ticker,
            AVG(daily_trades) AS avg_daily_trades,
            COUNT(*) AS n_days
        FROM daily_vol
        GROUP BY ticker
        HAVING COUNT(*) >= {MIN_DAYS}
    """)
    n_qualifying = con.execute("SELECT COUNT(*) FROM market_avg").fetchone()[0]
    print(f"[info] {n_qualifying:,} markets with >= {MIN_DAYS} trading days", file=sys.stderr)

    # Tag each market-day as spike or normal
    con.execute(f"""
        CREATE TEMP TABLE classified_days AS
        SELECT
            dv.ticker,
            dv.trade_date,
            dv.daily_trades,
            dv.daily_contracts,
            ma.avg_daily_trades,
            CASE WHEN dv.daily_trades > {SPIKE_MULT} * ma.avg_daily_trades
                 THEN 'spike' ELSE 'normal' END AS day_type
        FROM daily_vol dv
        INNER JOIN market_avg ma ON dv.ticker = ma.ticker
    """)

    # Summary stats on spike vs normal days
    day_summary = con.execute("""
        SELECT
            day_type,
            COUNT(*) AS n_days,
            SUM(daily_trades) AS total_trades,
            SUM(daily_contracts) AS total_contracts,
            AVG(daily_trades) AS avg_daily_trades
        FROM classified_days
        GROUP BY day_type
    """).fetchdf()
    print("[info] Day classification summary:", file=sys.stderr)
    print(day_summary.to_string(), file=sys.stderr)

    # ── Step 4: Join trades with classification and resolution ────────────
    print("[info] Joining trades with spike classification and resolution...", file=sys.stderr)
    con.execute(f"""
        CREATE TEMP TABLE tagged_trades AS
        SELECT
            t.ticker,
            t.yes_price,
            t.no_price,
            t.taker_side,
            t.count AS trade_count,
            cd.day_type,
            rm.result
        FROM '{TRADES_GLOB}' t
        INNER JOIN classified_days cd
            ON t.ticker = cd.ticker
            AND CAST(t.created_time AS DATE) = cd.trade_date
        INNER JOIN resolved_markets rm
            ON t.ticker = rm.ticker
        WHERE t.yes_price > 0 AND t.yes_price < 100
    """)

    n_tagged = con.execute("SELECT COUNT(*) FROM tagged_trades").fetchone()[0]
    print(f"[info] {n_tagged:,} tagged trades", file=sys.stderr)

    # ── Step 5: Compute calibration by price bucket and day_type ──────────
    print("[info] Computing calibration curves...", file=sys.stderr)

    # For each trade, the implied probability = yes_price/100 for yes-takers
    # and no_price/100 = (100-yes_price)/100 for no-takers.
    # A trade "wins" when the taker's side matches the result.
    # We bucket by the taker's implied price.
    calibration_df = con.execute(f"""
        WITH trade_metrics AS (
            SELECT
                day_type,
                -- The taker's price (what they paid, in cents)
                CASE WHEN taker_side = 'yes' THEN yes_price
                     ELSE no_price END AS taker_price,
                -- Did the taker win?
                CASE WHEN taker_side = 'yes' AND result = 'yes' THEN 1
                     WHEN taker_side = 'no'  AND result = 'no'  THEN 1
                     ELSE 0 END AS taker_won,
                trade_count
            FROM tagged_trades
        ),
        bucketed AS (
            SELECT
                day_type,
                -- Bucket: width_bucket puts price 1..99 into N buckets
                LEAST(GREATEST(
                    FLOOR((taker_price - 1) / {100.0 / PRICE_BUCKETS}) + 1,
                    1), {PRICE_BUCKETS}) AS bucket,
                taker_price,
                taker_won,
                trade_count
            FROM trade_metrics
            WHERE taker_price >= 1 AND taker_price <= 99
        )
        SELECT
            day_type,
            bucket,
            -- Midpoint of each bucket (in probability)
            ROUND(AVG(taker_price) / 100.0, 4) AS avg_price,
            -- Win rate (weighted by trade count = number of contracts)
            ROUND(SUM(taker_won * trade_count) * 1.0 / SUM(trade_count), 6) AS win_rate,
            -- Simple unweighted win rate
            ROUND(AVG(taker_won), 6) AS win_rate_unweighted,
            SUM(trade_count) AS total_contracts,
            COUNT(*) AS n_trades
        FROM bucketed
        GROUP BY day_type, bucket
        ORDER BY day_type, bucket
    """).fetchdf()

    print("[info] Calibration data:", file=sys.stderr)
    print(calibration_df.to_string(), file=sys.stderr)

    # ── Step 6: Compute overall statistics ────────────────────────────────
    print("[info] Computing overall statistics...", file=sys.stderr)
    overall_stats = con.execute("""
        WITH trade_metrics AS (
            SELECT
                day_type,
                CASE WHEN taker_side = 'yes' THEN yes_price
                     ELSE no_price END AS taker_price,
                CASE WHEN taker_side = 'yes' AND result = 'yes' THEN 1
                     WHEN taker_side = 'no'  AND result = 'no'  THEN 1
                     ELSE 0 END AS taker_won,
                trade_count
            FROM tagged_trades
            WHERE yes_price >= 1 AND yes_price <= 99
        )
        SELECT
            day_type,
            COUNT(*) AS n_trades,
            SUM(trade_count) AS n_contracts,
            -- Weighted win rate
            ROUND(SUM(taker_won * trade_count) * 1.0 / SUM(trade_count), 6) AS win_rate,
            -- Weighted avg price
            ROUND(SUM(taker_price * trade_count) * 1.0 / SUM(trade_count) / 100.0, 6) AS avg_price,
            -- Weighted excess return (win_rate - avg_price)
            ROUND(SUM(taker_won * trade_count) * 1.0 / SUM(trade_count)
                  - SUM(taker_price * trade_count) * 1.0 / SUM(trade_count) / 100.0, 6) AS excess_return,
            -- MAE: mean absolute error of calibration
            -- (compute later from calibration_df)
            ROUND(AVG(taker_won), 6) AS win_rate_unweighted,
            ROUND(AVG(taker_price) / 100.0, 6) AS avg_price_unweighted
        FROM trade_metrics
        GROUP BY day_type
    """).fetchdf()

    print("[info] Overall stats:", file=sys.stderr)
    print(overall_stats.to_string(), file=sys.stderr)

    # ── Prepare data for plotting ─────────────────────────────────────────
    spike_cal = calibration_df[calibration_df["day_type"] == "spike"].sort_values("bucket")
    normal_cal = calibration_df[calibration_df["day_type"] == "normal"].sort_values("bucket")

    # Compute MAE for each group (mean absolute error across buckets, weighted by contracts)
    def compute_mae(cal_df):
        """MAE = weighted avg of |win_rate - avg_price| across buckets."""
        if cal_df.empty:
            return 0.0
        prices = cal_df["avg_price"].values
        win_rates = cal_df["win_rate"].values
        weights = cal_df["total_contracts"].values
        mae = np.average(np.abs(win_rates - prices), weights=weights)
        return float(mae)

    spike_mae = compute_mae(spike_cal)
    normal_mae = compute_mae(normal_cal)

    # Get stats rows
    spike_stats = overall_stats[overall_stats["day_type"] == "spike"].iloc[0] if len(overall_stats[overall_stats["day_type"] == "spike"]) > 0 else None
    normal_stats = overall_stats[overall_stats["day_type"] == "normal"].iloc[0] if len(overall_stats[overall_stats["day_type"] == "normal"]) > 0 else None

    # ── Figure 1: Calibration curves ──────────────────────────────────────
    print("[info] Creating calibration figure...", file=sys.stderr)
    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor="white")

    # 45-degree reference
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.5,
            label="Perfect calibration", alpha=0.7)

    # Normal days
    if not normal_cal.empty:
        ax.plot(normal_cal["avg_price"].values, normal_cal["win_rate"].values,
                color=BLUE, linewidth=2.5, marker="o", markersize=8,
                label=f"Normal days (MAE={normal_mae:.4f})", zorder=3)

    # Spike days
    if not spike_cal.empty:
        ax.plot(spike_cal["avg_price"].values, spike_cal["win_rate"].values,
                color=ORANGE, linewidth=2.5, marker="s", markersize=8,
                label=f"Spike days (MAE={spike_mae:.4f})", zorder=4)

    ax.set_xlabel("Taker Price (Implied Probability)", fontsize=13)
    ax.set_ylabel("Observed Win Rate", fontsize=13)
    ax.set_title(
        "Calibration: Volume Spike Days vs Normal Days\n"
        f"(Spike = >{SPIKE_MULT:.0f}x market avg daily volume)",
        fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=11, loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=GRID_ALPHA)
    fig.tight_layout()

    path1 = f"{FIG_DIR}/6_1_spike_vs_normal_calibration.png"
    fig.savefig(path1, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"[info] Saved {path1}", file=sys.stderr)

    # ── Figure 2: Excess returns by price bucket ──────────────────────────
    print("[info] Creating excess returns figure...", file=sys.stderr)
    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor="white")

    # Compute excess return per bucket: win_rate - avg_price
    if not normal_cal.empty:
        normal_excess = normal_cal["win_rate"].values - normal_cal["avg_price"].values
        normal_midpoints = normal_cal["avg_price"].values
    else:
        normal_excess = np.array([])
        normal_midpoints = np.array([])

    if not spike_cal.empty:
        spike_excess = spike_cal["win_rate"].values - spike_cal["avg_price"].values
        spike_midpoints = spike_cal["avg_price"].values
    else:
        spike_excess = np.array([])
        spike_midpoints = np.array([])

    bar_width = 0.03

    if len(normal_midpoints) > 0:
        ax.bar(normal_midpoints - bar_width / 2, normal_excess * 100,
               width=bar_width, color=BLUE, alpha=0.85, label="Normal days",
               edgecolor="white", linewidth=0.5)

    if len(spike_midpoints) > 0:
        ax.bar(spike_midpoints + bar_width / 2, spike_excess * 100,
               width=bar_width, color=ORANGE, alpha=0.85, label="Spike days",
               edgecolor="white", linewidth=0.5)

    ax.axhline(0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Price Bucket (Implied Probability)", fontsize=13)
    ax.set_ylabel("Excess Return (pp)", fontsize=13)
    ax.set_title(
        "Excess Return by Price Bucket: Spike vs Normal Days\n"
        "(Positive = taker outperforms implied price)",
        fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1)
    ax.grid(alpha=GRID_ALPHA)
    fig.tight_layout()

    path2 = f"{FIG_DIR}/6_1_spike_excess_returns.png"
    fig.savefig(path2, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"[info] Saved {path2}", file=sys.stderr)

    # ── Build JSON output ─────────────────────────────────────────────────
    elapsed = time.time() - t0

    # Day summary for output
    day_summary_dict = {}
    for _, row in day_summary.iterrows():
        day_summary_dict[row["day_type"]] = {
            "n_days": int(row["n_days"]),
            "total_trades": int(row["total_trades"]),
            "total_contracts": int(row["total_contracts"]),
            "avg_daily_trades": round(float(row["avg_daily_trades"]), 2),
        }

    # Calibration data for output
    cal_output = {}
    for dtype in ["spike", "normal"]:
        subset = calibration_df[calibration_df["day_type"] == dtype].sort_values("bucket")
        cal_output[dtype] = []
        for _, row in subset.iterrows():
            cal_output[dtype].append({
                "bucket": int(row["bucket"]),
                "avg_price": round(float(row["avg_price"]), 4),
                "win_rate": round(float(row["win_rate"]), 6),
                "win_rate_unweighted": round(float(row["win_rate_unweighted"]), 6),
                "n_trades": int(row["n_trades"]),
                "total_contracts": int(row["total_contracts"]),
            })

    # Overall stats for output
    stats_output = {}
    for dtype, stats_row in [("spike", spike_stats), ("normal", normal_stats)]:
        if stats_row is not None:
            stats_output[dtype] = {
                "n_trades": int(stats_row["n_trades"]),
                "n_contracts": int(stats_row["n_contracts"]),
                "win_rate": round(float(stats_row["win_rate"]), 6),
                "avg_price": round(float(stats_row["avg_price"]), 6),
                "excess_return": round(float(stats_row["excess_return"]), 6),
            }

    output = {
        "elapsed_seconds": round(elapsed, 1),
        "spike_multiplier": SPIKE_MULT,
        "min_trading_days": MIN_DAYS,
        "n_qualifying_markets": int(n_qualifying),
        "n_resolved_markets": int(n_resolved),
        "n_tagged_trades": int(n_tagged),
        "day_summary": day_summary_dict,
        "overall_stats": stats_output,
        "mae": {
            "spike": round(spike_mae, 6),
            "normal": round(normal_mae, 6),
            "spike_minus_normal_pp": round((spike_mae - normal_mae) * 100, 2),
        },
        "calibration_by_bucket": cal_output,
        "figures": [path1, path2],
    }

    print(json.dumps(output, indent=2))
    print(f"[info] Done in {elapsed:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
