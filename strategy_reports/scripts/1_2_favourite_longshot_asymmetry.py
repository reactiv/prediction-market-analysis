"""§1.2 Favourite-Longshot Asymmetry — compare calibration for YES vs NO takers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import duckdb  # noqa: E402

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
# Query: per-price win rates for YES-takers and NO-takers
# ---------------------------------------------------------------------------
def load_yes_no_data() -> tuple[dict, dict]:
    """Return side-specific per-price stats with both trade and contract weighting."""
    con = duckdb.connect()

    # YES takers
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
        COUNT(*) AS n_trades,
        SUM(contracts) AS total_contracts,
        SUM(won * contracts) AS wins_weighted,
        SUM(won) AS wins_unweighted,
        COUNT(*) AS n_unweighted
    FROM yes_takers
    GROUP BY price
    ORDER BY price
    """
    yes_rows = con.execute(yes_sql).fetchall()

    # NO takers
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
        COUNT(*) AS n_trades,
        SUM(contracts) AS total_contracts,
        SUM(won * contracts) AS wins_weighted,
        SUM(won) AS wins_unweighted,
        COUNT(*) AS n_unweighted
    FROM no_takers
    GROUP BY price
    ORDER BY price
    """
    no_rows = con.execute(no_sql).fetchall()
    con.close()

    def parse_rows(rows):
        data = {}
        for price, n_trades, total_contracts, wins_w, wins_uw, n_uw in rows:
            p = int(price)
            data[p] = {
                "n_trades": int(n_trades),
                "total_contracts": int(total_contracts),
                "wins_contracts": float(wins_w),
                "wins_unweighted": int(wins_uw),
                "win_rate_contract": float(wins_w) / float(total_contracts) if total_contracts > 0 else 0.0,
                "win_rate_trade": float(wins_uw) / float(n_uw) if n_uw > 0 else 0.0,
            }
        return data

    return parse_rows(yes_rows), parse_rows(no_rows)


# ---------------------------------------------------------------------------
# Bucket data (5-cent buckets, min observations)
# ---------------------------------------------------------------------------
def bucket_data(data: dict, bucket_width: int = 5, min_obs: int = 100) -> list[dict]:
    """Aggregate price-level data into buckets."""
    buckets: dict[float, dict] = {}
    for p, d in data.items():
        mid = (p // bucket_width) * bucket_width + bucket_width / 2.0
        if mid not in buckets:
            buckets[mid] = {
                "n_trades": 0,
                "total_contracts": 0,
                "wins_contracts": 0.0,
                "price_contract_weighted_sum": 0.0,
            }
        buckets[mid]["n_trades"] += d["n_trades"]
        buckets[mid]["total_contracts"] += d["total_contracts"]
        buckets[mid]["wins_contracts"] += d["wins_contracts"]
        buckets[mid]["price_contract_weighted_sum"] += p * d["total_contracts"]

    result = []
    for mid in sorted(buckets.keys()):
        b = buckets[mid]
        if b["n_trades"] >= min_obs and b["total_contracts"] > 0:
            wr = float(b["wins_contracts"]) / float(b["total_contracts"])
            avg_price_cents = float(b["price_contract_weighted_sum"]) / float(b["total_contracts"])
            expected = avg_price_cents / 100.0
            result.append({
                "bucket_mid": mid,
                "avg_price_cents": avg_price_cents,
                "n_trades": b["n_trades"],
                "total_contracts": b["total_contracts"],
                "win_rate": wr,
                "expected_win_rate": expected,
                "excess_return": wr - expected,
            })
    return result


# ---------------------------------------------------------------------------
# Figure 1: Calibration YES vs NO
# ---------------------------------------------------------------------------
def plot_calibration(yes_buckets: list[dict], no_buckets: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")

    # Perfect calibration line
    ax.plot(
        [0, 100], [0, 1.0],
        linestyle="--", color="gray", linewidth=1.5,
        label="Perfect calibration",
    )

    # YES takers
    yes_x = [b["avg_price_cents"] for b in yes_buckets]
    yes_y = [b["win_rate"] for b in yes_buckets]
    ax.plot(yes_x, yes_y, color=GREEN, linewidth=2, marker="o",
            markersize=6, label="YES takers", zorder=5)

    # NO takers
    no_x = [b["avg_price_cents"] for b in no_buckets]
    no_y = [b["win_rate"] for b in no_buckets]
    ax.plot(no_x, no_y, color=RED, linewidth=2, marker="s",
            markersize=6, label="NO takers", zorder=5)

    ax.set_xlabel("Taker Price (cents)", fontsize=12)
    ax.set_ylabel("Actual Win Rate", fontsize=12)
    ax.set_title("Calibration: YES Takers vs NO Takers", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = FIG_DIR / "1_2_calibration_yes_vs_no.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Figure 2: Excess Return YES vs NO
# ---------------------------------------------------------------------------
def plot_excess_return(yes_buckets: list[dict], no_buckets: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")

    # Zero line
    ax.axhline(0, color="gray", linewidth=1.5, linestyle="--")

    # YES takers
    yes_x = [b["avg_price_cents"] for b in yes_buckets]
    yes_y = [b["excess_return"] for b in yes_buckets]
    ax.plot(yes_x, yes_y, color=GREEN, linewidth=2, marker="o",
            markersize=6, label="YES takers", zorder=5)

    # NO takers
    no_x = [b["avg_price_cents"] for b in no_buckets]
    no_y = [b["excess_return"] for b in no_buckets]
    ax.plot(no_x, no_y, color=RED, linewidth=2, marker="s",
            markersize=6, label="NO takers", zorder=5)

    ax.set_xlabel("Taker Price (cents)", fontsize=12)
    ax.set_ylabel("Excess Return (win_rate - price/100)", fontsize=12)
    ax.set_title("Excess Return by Price: YES Takers vs NO Takers", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = FIG_DIR / "1_2_excess_return_yes_vs_no.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Build price-range summary table
# ---------------------------------------------------------------------------
def build_range_table(yes_data: dict, no_data: dict) -> list[dict]:
    """Build a table with aggregated price ranges for the report."""
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

    def agg_for_range(data, lo, hi):
        total_trades = 0
        total_contracts = 0
        total_wins_contracts = 0.0
        total_price_contract_weighted = 0.0
        for p in range(lo, hi + 1):
            d = data.get(p)
            if d is None:
                continue
            total_trades += d["n_trades"]
            total_contracts += d["total_contracts"]
            total_wins_contracts += d["wins_contracts"]
            total_price_contract_weighted += p * d["total_contracts"]
        if total_contracts == 0:
            return None, total_trades
        win_rate = total_wins_contracts / total_contracts
        expected = (total_price_contract_weighted / total_contracts) / 100.0
        weighted_excess = win_rate - expected
        return round(weighted_excess * 100, 2), total_trades

    table = []
    for label, lo, hi in ranges:
        yes_excess, yes_trades = agg_for_range(yes_data, lo, hi)
        no_excess, no_trades = agg_for_range(no_data, lo, hi)
        if yes_excess is not None or no_excess is not None:
            asymmetry = None
            if yes_excess is not None and no_excess is not None:
                asymmetry = round(yes_excess - no_excess, 2)
            table.append({
                "price_range": label,
                "yes_excess_return_pp": yes_excess,
                "no_excess_return_pp": no_excess,
                "asymmetry_pp": asymmetry,
                "yes_trades": yes_trades,
                "no_trades": no_trades,
            })
    return table


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------
def build_summary(yes_data: dict, no_data: dict,
                  yes_buckets: list[dict], no_buckets: list[dict]) -> dict:
    """Build JSON summary."""

    # Overall stats
    yes_total_trades = sum(d["n_trades"] for d in yes_data.values())
    no_total_trades = sum(d["n_trades"] for d in no_data.values())
    yes_total_contracts = sum(d["total_contracts"] for d in yes_data.values())
    no_total_contracts = sum(d["total_contracts"] for d in no_data.values())

    def weighted_avg_excess(data: dict, lo: int = 1, hi: int = 99):
        total_contracts = 0
        total_wins_contracts = 0.0
        total_price_contract_weighted = 0.0
        for p in range(lo, hi + 1):
            d = data.get(p)
            if d is None:
                continue
            total_contracts += d["total_contracts"]
            total_wins_contracts += d["wins_contracts"]
            total_price_contract_weighted += p * d["total_contracts"]
        if total_contracts == 0:
            return 0.0
        win_rate = total_wins_contracts / total_contracts
        expected = (total_price_contract_weighted / total_contracts) / 100.0
        return win_rate - expected

    yes_avg_excess = weighted_avg_excess(yes_data)
    no_avg_excess = weighted_avg_excess(no_data)

    # Low price excess (price <= 20) -- longshots
    def low_price_excess(data):
        return weighted_avg_excess(data, 1, 20)

    # High price excess (price >= 80) -- favourites
    def high_price_excess(data):
        return weighted_avg_excess(data, 80, 99)

    # Mid price excess (20 < price < 80)
    def mid_price_excess(data):
        return weighted_avg_excess(data, 21, 79)

    range_table = build_range_table(yes_data, no_data)

    # Mean Absolute Error (contract-weighted calibration error) for each side
    def mae(data):
        total_contracts = 0
        weighted_abs_err = 0.0
        for p, d in data.items():
            if d["total_contracts"] <= 0:
                continue
            err = abs(d["win_rate_contract"] - (p / 100.0))
            weighted_abs_err += err * d["total_contracts"]
            total_contracts += d["total_contracts"]
        if total_contracts == 0:
            return 0.0
        return weighted_abs_err / total_contracts

    summary = {
        "yes_takers": {
            "total_trades": yes_total_trades,
            "total_contracts": yes_total_contracts,
            "n_buckets": len(yes_buckets),
            "weighted_avg_excess_return": round(yes_avg_excess * 100, 4),
            "low_price_excess_pp": round(low_price_excess(yes_data) * 100, 4),
            "mid_price_excess_pp": round(mid_price_excess(yes_data) * 100, 4),
            "high_price_excess_pp": round(high_price_excess(yes_data) * 100, 4),
            "mae_pp": round(mae(yes_data) * 100, 4),
        },
        "no_takers": {
            "total_trades": no_total_trades,
            "total_contracts": no_total_contracts,
            "n_buckets": len(no_buckets),
            "weighted_avg_excess_return": round(no_avg_excess * 100, 4),
            "low_price_excess_pp": round(low_price_excess(no_data) * 100, 4),
            "mid_price_excess_pp": round(mid_price_excess(no_data) * 100, 4),
            "high_price_excess_pp": round(high_price_excess(no_data) * 100, 4),
            "mae_pp": round(mae(no_data) * 100, 4),
        },
        "asymmetry": {
            "overall_pp": round((yes_avg_excess - no_avg_excess) * 100, 4),
            "low_price_pp": round((low_price_excess(yes_data) - low_price_excess(no_data)) * 100, 4),
            "mid_price_pp": round((mid_price_excess(yes_data) - mid_price_excess(no_data)) * 100, 4),
            "high_price_pp": round((high_price_excess(yes_data) - high_price_excess(no_data)) * 100, 4),
        },
        "range_table": range_table,
        "yes_buckets": [
            {
                "bucket_mid": b["bucket_mid"],
                "avg_price_cents": round(b["avg_price_cents"], 4),
                "n_trades": b["n_trades"],
                "total_contracts": b["total_contracts"],
                "win_rate": round(b["win_rate"], 6),
                "expected": round(b["expected_win_rate"], 4),
                "excess_return_pp": round(b["excess_return"] * 100, 4),
            }
            for b in yes_buckets
        ],
        "no_buckets": [
            {
                "bucket_mid": b["bucket_mid"],
                "avg_price_cents": round(b["avg_price_cents"], 4),
                "n_trades": b["n_trades"],
                "total_contracts": b["total_contracts"],
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
    print("Loading YES/NO taker data...", file=sys.stderr)
    yes_data, no_data = load_yes_no_data()
    print(f"  YES price points: {len(yes_data)}, NO price points: {len(no_data)}", file=sys.stderr)

    print("Bucketing data (5c buckets, min 100 obs)...", file=sys.stderr)
    yes_buckets = bucket_data(yes_data, bucket_width=5, min_obs=100)
    no_buckets = bucket_data(no_data, bucket_width=5, min_obs=100)
    print(f"  YES buckets: {len(yes_buckets)}, NO buckets: {len(no_buckets)}", file=sys.stderr)

    print("Plotting calibration...", file=sys.stderr)
    plot_calibration(yes_buckets, no_buckets)

    print("Plotting excess returns...", file=sys.stderr)
    plot_excess_return(yes_buckets, no_buckets)

    print("Building summary...", file=sys.stderr)
    summary = build_summary(yes_data, no_data, yes_buckets, no_buckets)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
