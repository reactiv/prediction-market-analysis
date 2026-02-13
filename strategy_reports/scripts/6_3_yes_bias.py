"""§6.3 Yes Bias — do people disproportionately buy YES contracts on Kalshi?"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import duckdb  # noqa: E402
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
# Colors
# ---------------------------------------------------------------------------
GREEN = "#2ecc71"
RED = "#e74c3c"

# ---------------------------------------------------------------------------
# Query: per-price-point stats for YES and NO takers (joined with resolved)
# ---------------------------------------------------------------------------
def load_per_price_data() -> tuple[dict, dict]:
    """Return (yes_data, no_data) keyed by price (1-99).

    Each value: {n_trades, total_contracts, wins, win_rate}
    """
    con = duckdb.connect()

    yes_sql = f"""
    WITH resolved_markets AS (
        SELECT ticker, result
        FROM '{MARKETS_DIR}/*.parquet'
        WHERE status = 'finalized'
          AND result IN ('yes', 'no')
    ),
    yes_takers AS (
        SELECT
            t.yes_price AS price,
            CASE WHEN m.result = 'yes' THEN 1.0 ELSE 0.0 END AS won,
            t.count AS contracts
        FROM '{TRADES_DIR}/*.parquet' t
        INNER JOIN resolved_markets m ON t.ticker = m.ticker
        WHERE t.taker_side = 'yes'
          AND t.yes_price BETWEEN 1 AND 99
    )
    SELECT
        price,
        COUNT(*)            AS n_trades,
        SUM(contracts)      AS total_contracts,
        SUM(won)            AS wins_unweighted
    FROM yes_takers
    GROUP BY price
    ORDER BY price
    """
    yes_rows = con.execute(yes_sql).fetchall()

    no_sql = f"""
    WITH resolved_markets AS (
        SELECT ticker, result
        FROM '{MARKETS_DIR}/*.parquet'
        WHERE status = 'finalized'
          AND result IN ('yes', 'no')
    ),
    no_takers AS (
        SELECT
            t.no_price AS price,
            CASE WHEN m.result = 'no' THEN 1.0 ELSE 0.0 END AS won,
            t.count AS contracts
        FROM '{TRADES_DIR}/*.parquet' t
        INNER JOIN resolved_markets m ON t.ticker = m.ticker
        WHERE t.taker_side = 'no'
          AND t.no_price BETWEEN 1 AND 99
    )
    SELECT
        price,
        COUNT(*)            AS n_trades,
        SUM(contracts)      AS total_contracts,
        SUM(won)            AS wins_unweighted
    FROM no_takers
    GROUP BY price
    ORDER BY price
    """
    no_rows = con.execute(no_sql).fetchall()
    con.close()

    def parse(rows):
        data = {}
        for price, n_trades, total_contracts, wins_uw in rows:
            p = int(price)
            n = int(n_trades)
            data[p] = {
                "n_trades": n,
                "total_contracts": int(total_contracts),
                "wins": int(wins_uw),
                "win_rate": float(wins_uw) / n if n > 0 else 0.0,
            }
        return data

    return parse(yes_rows), parse(no_rows)


# ---------------------------------------------------------------------------
# Volume ratio per price point (raw, for Figure 1)
# ---------------------------------------------------------------------------
def compute_volume_ratio(yes_data: dict, no_data: dict, min_obs: int = 100) -> list[dict]:
    """For each price point with enough trades on both sides, compute YES/NO ratio."""
    results = []
    for p in range(1, 100):
        yd = yes_data.get(p)
        nd = no_data.get(p)
        if yd is None or nd is None:
            continue
        if yd["n_trades"] < min_obs or nd["n_trades"] < min_obs:
            continue
        ratio = yd["n_trades"] / nd["n_trades"]
        results.append({
            "price": p,
            "yes_trades": yd["n_trades"],
            "no_trades": nd["n_trades"],
            "ratio": ratio,
        })
    return results


# ---------------------------------------------------------------------------
# Bucket data (5-cent buckets, min observations)
# ---------------------------------------------------------------------------
def bucket_data(data: dict, bucket_width: int = 5, min_obs: int = 100) -> list[dict]:
    """Aggregate price-level data into buckets."""
    buckets: dict[float, dict] = {}
    for p, d in data.items():
        mid = (p // bucket_width) * bucket_width + bucket_width / 2.0
        if mid not in buckets:
            buckets[mid] = {"n_trades": 0, "total_contracts": 0, "wins": 0}
        buckets[mid]["n_trades"] += d["n_trades"]
        buckets[mid]["total_contracts"] += d["total_contracts"]
        buckets[mid]["wins"] += d["wins"]

    result = []
    for mid in sorted(buckets.keys()):
        b = buckets[mid]
        if b["n_trades"] >= min_obs:
            wr = float(b["wins"]) / float(b["n_trades"])
            expected = mid / 100.0
            result.append({
                "bucket_mid": mid,
                "n_trades": b["n_trades"],
                "total_contracts": b["total_contracts"],
                "win_rate": wr,
                "expected_win_rate": expected,
                "excess_return": wr - expected,
            })
    return result


# ---------------------------------------------------------------------------
# Figure 1: YES/NO Volume Ratio by Price
# ---------------------------------------------------------------------------
def plot_volume_ratio(volume_ratio: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")

    prices = [v["price"] for v in volume_ratio]
    ratios = [v["ratio"] for v in volume_ratio]

    ax.plot(prices, ratios, color=GREEN, linewidth=1.8, alpha=0.85, label="YES/NO trade count ratio")

    # Smoothed trendline (rolling average with window ~5)
    if len(ratios) > 5:
        smoothed = np.convolve(ratios, np.ones(5) / 5, mode="valid")
        smoothed_x = prices[2 : 2 + len(smoothed)]
        ax.plot(smoothed_x, smoothed, color=GREEN, linewidth=3, alpha=1.0,
                label="5-point moving average", zorder=6)

    # Reference line at 1.0
    ax.axhline(1.0, color="gray", linewidth=1.5, linestyle="--", label="Balanced (1.0)")

    ax.set_xlabel("Price (cents)", fontsize=12)
    ax.set_ylabel("YES / NO Trade Count Ratio", fontsize=12)
    ax.set_title("YES Bias: Volume Ratio by Price Point", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = FIG_DIR / "6_3_yes_no_volume_ratio.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Figure 2: Calibration Asymmetry (excess return curves)
# ---------------------------------------------------------------------------
def plot_calibration_asymmetry(yes_buckets: list[dict], no_buckets: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")

    # Zero line
    ax.axhline(0, color="gray", linewidth=1.5, linestyle="--", label="Fair pricing (0)")

    # YES takers
    yes_x = [b["bucket_mid"] for b in yes_buckets]
    yes_y = [b["excess_return"] * 100 for b in yes_buckets]  # in percentage points
    ax.plot(yes_x, yes_y, color=GREEN, linewidth=2.5, marker="o",
            markersize=6, label="YES takers excess return", zorder=5)

    # NO takers
    no_x = [b["bucket_mid"] for b in no_buckets]
    no_y = [b["excess_return"] * 100 for b in no_buckets]
    ax.plot(no_x, no_y, color=RED, linewidth=2.5, marker="s",
            markersize=6, label="NO takers excess return", zorder=5)

    # Fill between to highlight gap
    # Find common x range
    common_x = sorted(set(yes_x) & set(no_x))
    if common_x:
        yes_dict = dict(zip(yes_x, yes_y))
        no_dict = dict(zip(no_x, no_y))
        cx = [x for x in common_x]
        cy_yes = [yes_dict[x] for x in cx]
        cy_no = [no_dict[x] for x in cx]
        ax.fill_between(cx, cy_yes, cy_no, alpha=0.1, color="purple", label="Asymmetry gap")

    ax.set_xlabel("Taker Price (cents)", fontsize=12)
    ax.set_ylabel("Excess Return (pp)", fontsize=12)
    ax.set_title("Calibration Asymmetry: YES vs NO Takers", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = FIG_DIR / "6_3_yes_bias_calibration.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Summary table by price range
# ---------------------------------------------------------------------------
def build_range_table(yes_data: dict, no_data: dict,
                      yes_buckets: list[dict], no_buckets: list[dict]) -> list[dict]:
    """Build a table with aggregated price ranges."""
    ranges = [
        ("1-10", 1, 10),
        ("11-20", 11, 20),
        ("21-30", 21, 30),
        ("31-40", 31, 40),
        ("41-50", 41, 50),
        ("51-60", 51, 60),
        ("61-70", 61, 70),
        ("71-80", 71, 80),
        ("81-90", 81, 90),
        ("91-99", 91, 99),
    ]

    table = []
    for label, lo, hi in ranges:
        # Aggregate raw per-price data for trade counts
        yes_trades = sum(yes_data.get(p, {}).get("n_trades", 0) for p in range(lo, hi + 1))
        no_trades = sum(no_data.get(p, {}).get("n_trades", 0) for p in range(lo, hi + 1))
        ratio = yes_trades / no_trades if no_trades > 0 else float("inf")

        # Get excess returns from bucketed data
        def excess_for_range(buckets, lo_bound, hi_bound):
            matching = [b for b in buckets if lo_bound <= b["bucket_mid"] <= hi_bound]
            if not matching:
                return None
            total_n = sum(b["n_trades"] for b in matching)
            if total_n == 0:
                return None
            return sum(b["excess_return"] * b["n_trades"] for b in matching) / total_n

        yes_excess = excess_for_range(yes_buckets, lo, hi)
        no_excess = excess_for_range(no_buckets, lo, hi)

        gap = None
        if yes_excess is not None and no_excess is not None:
            gap = round((yes_excess - no_excess) * 100, 2)

        table.append({
            "price_range": label,
            "yes_trades": yes_trades,
            "no_trades": no_trades,
            "ratio": round(ratio, 2),
            "yes_excess_pp": round(yes_excess * 100, 2) if yes_excess is not None else None,
            "no_excess_pp": round(no_excess * 100, 2) if no_excess is not None else None,
            "gap_pp": gap,
        })
    return table


# ---------------------------------------------------------------------------
# Build overall summary
# ---------------------------------------------------------------------------
def build_summary(yes_data: dict, no_data: dict,
                  yes_buckets: list[dict], no_buckets: list[dict],
                  volume_ratio: list[dict]) -> dict:
    """Build JSON summary for stdout."""

    yes_total = sum(d["n_trades"] for d in yes_data.values())
    no_total = sum(d["n_trades"] for d in no_data.values())
    yes_contracts = sum(d["total_contracts"] for d in yes_data.values())
    no_contracts = sum(d["total_contracts"] for d in no_data.values())

    overall_ratio = yes_total / no_total if no_total > 0 else float("inf")

    def weighted_avg_excess(buckets):
        total_n = sum(b["n_trades"] for b in buckets)
        if total_n == 0:
            return 0.0
        return sum(b["excess_return"] * b["n_trades"] for b in buckets) / total_n

    yes_avg_excess = weighted_avg_excess(yes_buckets)
    no_avg_excess = weighted_avg_excess(no_buckets)

    # Median and max volume ratio
    ratios = [v["ratio"] for v in volume_ratio]
    median_ratio = float(np.median(ratios)) if ratios else 0.0
    max_ratio_entry = max(volume_ratio, key=lambda v: v["ratio"]) if volume_ratio else {}

    # Where is YES buying most disproportionate?
    # Top 5 prices by ratio
    top5_ratio = sorted(volume_ratio, key=lambda v: v["ratio"], reverse=True)[:5]

    range_table = build_range_table(yes_data, no_data, yes_buckets, no_buckets)

    # MAE for each side
    def mae(buckets):
        total_n = sum(b["n_trades"] for b in buckets)
        if total_n == 0:
            return 0.0
        return sum(abs(b["excess_return"]) * b["n_trades"] for b in buckets) / total_n

    summary = {
        "overall": {
            "yes_total_trades": yes_total,
            "no_total_trades": no_total,
            "overall_ratio": round(overall_ratio, 4),
            "yes_total_contracts": yes_contracts,
            "no_total_contracts": no_contracts,
        },
        "volume_ratio_stats": {
            "median_ratio": round(median_ratio, 4),
            "max_ratio": round(max_ratio_entry.get("ratio", 0), 4),
            "max_ratio_price": max_ratio_entry.get("price", 0),
            "top5_prices_by_ratio": [
                {"price": v["price"], "ratio": round(v["ratio"], 2)}
                for v in top5_ratio
            ],
        },
        "excess_return": {
            "yes_weighted_avg_pp": round(yes_avg_excess * 100, 4),
            "no_weighted_avg_pp": round(no_avg_excess * 100, 4),
            "gap_pp": round((yes_avg_excess - no_avg_excess) * 100, 4),
            "yes_mae_pp": round(mae(yes_buckets) * 100, 4),
            "no_mae_pp": round(mae(no_buckets) * 100, 4),
        },
        "range_table": range_table,
        "yes_buckets": [
            {
                "bucket_mid": b["bucket_mid"],
                "n_trades": b["n_trades"],
                "win_rate": round(b["win_rate"], 6),
                "expected": round(b["expected_win_rate"], 4),
                "excess_return_pp": round(b["excess_return"] * 100, 4),
            }
            for b in yes_buckets
        ],
        "no_buckets": [
            {
                "bucket_mid": b["bucket_mid"],
                "n_trades": b["n_trades"],
                "win_rate": round(b["win_rate"], 6),
                "expected": round(b["expected_win_rate"], 4),
                "excess_return_pp": round(b["excess_return"] * 100, 4),
            }
            for b in no_buckets
        ],
    }
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading per-price YES/NO taker data...", file=sys.stderr)
    yes_data, no_data = load_per_price_data()
    print(f"  YES price points: {len(yes_data)}, NO price points: {len(no_data)}", file=sys.stderr)
    print(f"  YES total trades: {sum(d['n_trades'] for d in yes_data.values()):,}", file=sys.stderr)
    print(f"  NO  total trades: {sum(d['n_trades'] for d in no_data.values()):,}", file=sys.stderr)

    print("Computing volume ratio per price point...", file=sys.stderr)
    volume_ratio = compute_volume_ratio(yes_data, no_data, min_obs=100)
    print(f"  Price points with sufficient data: {len(volume_ratio)}", file=sys.stderr)

    print("Bucketing data (5c buckets, min 100 obs)...", file=sys.stderr)
    yes_buckets = bucket_data(yes_data, bucket_width=5, min_obs=100)
    no_buckets = bucket_data(no_data, bucket_width=5, min_obs=100)
    print(f"  YES buckets: {len(yes_buckets)}, NO buckets: {len(no_buckets)}", file=sys.stderr)

    print("Plotting volume ratio...", file=sys.stderr)
    plot_volume_ratio(volume_ratio)

    print("Plotting calibration asymmetry...", file=sys.stderr)
    plot_calibration_asymmetry(yes_buckets, no_buckets)

    print("Building summary...", file=sys.stderr)
    summary = build_summary(yes_data, no_data, yes_buckets, no_buckets, volume_ratio)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
