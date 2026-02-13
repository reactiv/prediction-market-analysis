"""§1.1 Longshot Bias — excess returns from fading longshots on Kalshi."""

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
BLUE = "#4C72B0"
RED = "#e74c3c"
GREEN = "#2ecc71"
ORANGE = "#ff7f0e"
PURPLE = "#9467bd"

# ---------------------------------------------------------------------------
# Query: win rate by price (1-99)
# ---------------------------------------------------------------------------
def load_win_rates() -> dict:
    """Return {price: {n, wins, win_rate}} for prices 1-99."""
    con = duckdb.connect()
    sql = f"""
    WITH resolved_markets AS (
        SELECT ticker, result
        FROM '{MARKETS_DIR}/*.parquet'
        WHERE status = 'finalized' AND result IN ('yes', 'no')
    ),
    all_positions AS (
        -- Taker side
        SELECT
            CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS price,
            CASE WHEN t.taker_side = m.result THEN 1 ELSE 0 END AS won
        FROM '{TRADES_DIR}/*.parquet' t
        INNER JOIN resolved_markets m ON t.ticker = m.ticker
        UNION ALL
        -- Maker side (counterparty)
        SELECT
            CASE WHEN t.taker_side = 'yes' THEN t.no_price ELSE t.yes_price END AS price,
            CASE WHEN t.taker_side != m.result THEN 1 ELSE 0 END AS won
        FROM '{TRADES_DIR}/*.parquet' t
        INNER JOIN resolved_markets m ON t.ticker = m.ticker
    )
    SELECT price,
        COUNT(*) AS n,
        SUM(won) AS wins,
        100.0 * SUM(won) / COUNT(*) AS win_rate
    FROM all_positions
    WHERE price BETWEEN 1 AND 99
    GROUP BY price
    ORDER BY price
    """
    rows = con.execute(sql).fetchall()
    con.close()

    data = {}
    for price, n, wins, win_rate in rows:
        data[int(price)] = {
            "n": int(n),
            "wins": int(wins),
            "win_rate": float(win_rate),
        }
    return data


# ---------------------------------------------------------------------------
# Compute metrics
# ---------------------------------------------------------------------------
def compute_metrics(data: dict) -> dict:
    """
    For each price point compute:
      - excess_return_pp = win_rate - price  (positive => buyers have edge)
      - edge_cents = excess_return_pp  (same numerically; 1 pp = 1 cent/contract)
      - For "fade longshot" strategy:
        If price < 15, the opposing side is at (100 - price) with
        win_rate_opposing = 100 - win_rate.  Compute their excess return.
        Similarly for price > 85.
    """
    prices = sorted(data.keys())
    results = {}
    for p in prices:
        d = data[p]
        wr = d["win_rate"]
        n = d["n"]

        excess_return_pp = wr - p  # positive => buyers win more than price implies
        edge_cents = excess_return_pp  # 1 pp = 1 cent per contract

        # Fade-longshot metrics: what does the OTHER side earn?
        opp_price = 100 - p
        opp_wr = 100.0 - wr
        opp_excess_pp = opp_wr - opp_price  # = -(wr - p) = -excess_return_pp

        results[p] = {
            "n": n,
            "wins": d["wins"],
            "win_rate": wr,
            "excess_return_pp": excess_return_pp,
            "edge_cents": edge_cents,
            "opp_price": opp_price,
            "opp_win_rate": opp_wr,
            "opp_excess_pp": opp_excess_pp,
        }
    return results


# ---------------------------------------------------------------------------
# Bucketed calibration (5-cent buckets, min 100 obs)
# ---------------------------------------------------------------------------
def bucket_calibration(data: dict, bucket_width: int = 5, min_obs: int = 100) -> list[dict]:
    """Aggregate price-level data into buckets."""
    buckets: dict[float, dict] = {}
    for p, d in data.items():
        mid = (p // bucket_width) * bucket_width + bucket_width / 2.0
        if mid not in buckets:
            buckets[mid] = {"n": 0, "wins": 0}
        buckets[mid]["n"] += d["n"]
        buckets[mid]["wins"] += d["wins"]

    result = []
    for mid in sorted(buckets.keys()):
        b = buckets[mid]
        if b["n"] >= min_obs:
            wr = 100.0 * b["wins"] / b["n"]
            result.append({
                "bucket": mid,
                "n": b["n"],
                "wins": b["wins"],
                "win_rate": wr,
                "excess_return_pp": wr - mid,
            })
    return result


# ---------------------------------------------------------------------------
# Figure 1: Excess Returns by Price
# ---------------------------------------------------------------------------
def plot_excess_returns(metrics: dict) -> None:
    prices = sorted(metrics.keys())
    excess = [metrics[p]["excess_return_pp"] for p in prices]
    n_obs = [metrics[p]["n"] for p in prices]

    # Filter out prices with < 100 observations
    filtered_prices = []
    filtered_excess = []
    for p, e, n in zip(prices, excess, n_obs):
        if n >= 100:
            filtered_prices.append(p)
            filtered_excess.append(e)

    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")

    # Color bars: longshot zone (<15) in red, favorite zone (>85) in orange, middle in blue
    colors = []
    for p in filtered_prices:
        if p < 15:
            colors.append(RED)
        elif p > 85:
            colors.append(ORANGE)
        else:
            colors.append(BLUE)

    ax.bar(filtered_prices, filtered_excess, color=colors, width=0.9, alpha=0.85)

    # Shade longshot and favorite zones
    ax.axvspan(0, 15, alpha=0.08, color=RED, label="Longshot zone (<15c)")
    ax.axvspan(85, 100, alpha=0.08, color=ORANGE, label="Favorite zone (>85c)")

    # Zero line
    ax.axhline(0, color="gray", linewidth=1, linestyle="--")

    ax.set_xlabel("Contract Price (cents)", fontsize=12)
    ax.set_ylabel("Excess Return (percentage points)", fontsize=12)
    ax.set_title("Excess Return by Contract Price (Buyer Perspective)", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = FIG_DIR / "1_1_longshot_excess_returns.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Figure 2: Calibration with Longshot Regions
# ---------------------------------------------------------------------------
def plot_calibration(buckets: list[dict]) -> None:
    bucket_mids = [b["bucket"] for b in buckets]
    win_rates = [b["win_rate"] for b in buckets]

    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")

    # Shade longshot and favorite zones
    ax.axvspan(0, 15, alpha=0.08, color=RED, label="Longshot zone (<15c)")
    ax.axvspan(85, 100, alpha=0.08, color=ORANGE, label="Favorite zone (>85c)")

    # Perfect calibration line
    ax.plot(
        [0, 100], [0, 100],
        linestyle="--", color="gray", linewidth=1,
        label="Perfect calibration",
    )

    # Calibration points
    # Color longshot/favorite buckets differently
    for b in buckets:
        mid = b["bucket"]
        wr = b["win_rate"]
        if mid < 15:
            color = RED
        elif mid > 85:
            color = ORANGE
        else:
            color = BLUE
        ax.scatter(mid, wr, color=color, s=60, zorder=5, edgecolors="white", linewidth=0.5)

    # Connect with line
    ax.plot(bucket_mids, win_rates, color=BLUE, linewidth=1.5, alpha=0.7)

    ax.set_xlabel("Contract Price (cents)", fontsize=12)
    ax.set_ylabel("Win Rate (%)", fontsize=12)
    ax.set_title("Calibration Curve with Longshot/Favorite Zones", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = FIG_DIR / "1_1_longshot_calibration.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------
def build_summary(metrics: dict, buckets: list[dict]) -> dict:
    """Build JSON summary with key stats."""

    # Overall stats for longshot zone (<15c buyers)
    longshot_prices = [p for p in metrics if p < 15 and metrics[p]["n"] >= 100]
    favorite_prices = [p for p in metrics if p > 85 and metrics[p]["n"] >= 100]

    def zone_stats(price_list, label):
        if not price_list:
            return {"zone": label, "n_prices": 0}
        total_n = sum(metrics[p]["n"] for p in price_list)
        total_wins = sum(metrics[p]["wins"] for p in price_list)
        weighted_excess = sum(
            metrics[p]["excess_return_pp"] * metrics[p]["n"] for p in price_list
        ) / total_n
        avg_win_rate = 100.0 * total_wins / total_n
        avg_price = sum(p * metrics[p]["n"] for p in price_list) / total_n
        return {
            "zone": label,
            "price_range": f"{min(price_list)}-{max(price_list)}c",
            "n_prices": len(price_list),
            "total_trades": total_n,
            "avg_price": round(avg_price, 2),
            "avg_win_rate": round(avg_win_rate, 2),
            "weighted_excess_return_pp": round(weighted_excess, 2),
            "edge_cents_per_contract": round(weighted_excess, 2),
        }

    longshot_stats = zone_stats(longshot_prices, "longshot (<15c)")
    favorite_stats = zone_stats(favorite_prices, "favorite (>85c)")

    # Fade-longshot strategy: take the OPPOSING side when someone buys at <15c
    # i.e., you are selling YES at price <15c, or equivalently buying NO at (100-price)
    # Your edge = opp_excess_pp for that price
    fade_longshot_prices = [p for p in metrics if p < 15 and metrics[p]["n"] >= 100]
    if fade_longshot_prices:
        total_n = sum(metrics[p]["n"] for p in fade_longshot_prices)
        weighted_fade_edge = sum(
            metrics[p]["opp_excess_pp"] * metrics[p]["n"] for p in fade_longshot_prices
        ) / total_n
        fade_win_rate = sum(
            metrics[p]["opp_win_rate"] * metrics[p]["n"] for p in fade_longshot_prices
        ) / total_n
        fade_longshot = {
            "strategy": "fade_longshot",
            "description": "Bet AGAINST contracts priced <15c (buy NO / sell YES)",
            "total_trades": total_n,
            "avg_opposing_price": round(sum((100 - p) * metrics[p]["n"] for p in fade_longshot_prices) / total_n, 2),
            "avg_win_rate": round(fade_win_rate, 2),
            "weighted_edge_pp": round(weighted_fade_edge, 2),
            "edge_cents_per_contract": round(weighted_fade_edge, 2),
        }
    else:
        fade_longshot = {"strategy": "fade_longshot", "total_trades": 0}

    # Fade-favorite strategy: take the OPPOSING side when someone buys at >85c
    fade_favorite_prices = [p for p in metrics if p > 85 and metrics[p]["n"] >= 100]
    if fade_favorite_prices:
        total_n = sum(metrics[p]["n"] for p in fade_favorite_prices)
        weighted_fade_edge = sum(
            metrics[p]["opp_excess_pp"] * metrics[p]["n"] for p in fade_favorite_prices
        ) / total_n
        fade_win_rate = sum(
            metrics[p]["opp_win_rate"] * metrics[p]["n"] for p in fade_favorite_prices
        ) / total_n
        fade_favorite = {
            "strategy": "fade_favorite",
            "description": "Bet AGAINST contracts priced >85c (buy NO when price >85c)",
            "total_trades": total_n,
            "avg_opposing_price": round(sum((100 - p) * metrics[p]["n"] for p in fade_favorite_prices) / total_n, 2),
            "avg_win_rate": round(fade_win_rate, 2),
            "weighted_edge_pp": round(weighted_fade_edge, 2),
            "edge_cents_per_contract": round(weighted_fade_edge, 2),
        }
    else:
        fade_favorite = {"strategy": "fade_favorite", "total_trades": 0}

    # Breakeven: find price points where excess_return crosses zero
    sorted_prices = sorted(p for p in metrics if metrics[p]["n"] >= 100)
    breakevens = []
    for i in range(len(sorted_prices) - 1):
        p1, p2 = sorted_prices[i], sorted_prices[i + 1]
        e1 = metrics[p1]["excess_return_pp"]
        e2 = metrics[p2]["excess_return_pp"]
        if e1 * e2 < 0:  # sign change
            # Linear interpolation
            bp = p1 + (p2 - p1) * abs(e1) / (abs(e1) + abs(e2))
            breakevens.append(round(bp, 1))

    # Per-price detail for longshot and favorite zones
    longshot_detail = []
    for p in sorted(longshot_prices):
        m = metrics[p]
        longshot_detail.append({
            "price": p,
            "n": m["n"],
            "win_rate": round(m["win_rate"], 2),
            "excess_return_pp": round(m["excess_return_pp"], 2),
            "opposing_win_rate": round(m["opp_win_rate"], 2),
            "opposing_excess_pp": round(m["opp_excess_pp"], 2),
        })

    favorite_detail = []
    for p in sorted(favorite_prices):
        m = metrics[p]
        favorite_detail.append({
            "price": p,
            "n": m["n"],
            "win_rate": round(m["win_rate"], 2),
            "excess_return_pp": round(m["excess_return_pp"], 2),
            "opposing_win_rate": round(m["opp_win_rate"], 2),
            "opposing_excess_pp": round(m["opp_excess_pp"], 2),
        })

    # Bucket-level summary for report table
    bucket_table = []
    for b in buckets:
        mid = b["bucket"]
        if mid <= 15 or mid >= 85:
            bucket_table.append({
                "bucket": mid,
                "n": b["n"],
                "win_rate": round(b["win_rate"], 2),
                "implied_prob": mid,
                "excess_return_pp": round(b["excess_return_pp"], 2),
                "edge_cents": round(b["excess_return_pp"], 2),
            })

    # Total trades
    total_trades = sum(metrics[p]["n"] for p in metrics)

    summary = {
        "total_trades": total_trades,
        "total_prices_with_data": len(metrics),
        "prices_with_100_obs": len([p for p in metrics if metrics[p]["n"] >= 100]),
        "longshot_zone": longshot_stats,
        "favorite_zone": favorite_stats,
        "fade_longshot_strategy": fade_longshot,
        "fade_favorite_strategy": fade_favorite,
        "breakeven_prices": breakevens,
        "longshot_detail": longshot_detail,
        "favorite_detail": favorite_detail,
        "bucket_table": bucket_table,
    }
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading win rate data...", file=sys.stderr)
    data = load_win_rates()
    print(f"  Price points loaded: {len(data)}", file=sys.stderr)

    print("Computing metrics...", file=sys.stderr)
    metrics = compute_metrics(data)

    print("Computing bucketed calibration...", file=sys.stderr)
    buckets = bucket_calibration(data, bucket_width=5, min_obs=100)
    print(f"  Valid buckets: {len(buckets)}", file=sys.stderr)

    print("Plotting excess returns...", file=sys.stderr)
    plot_excess_returns(metrics)

    print("Plotting calibration...", file=sys.stderr)
    plot_calibration(buckets)

    print("Building summary...", file=sys.stderr)
    summary = build_summary(metrics, buckets)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
