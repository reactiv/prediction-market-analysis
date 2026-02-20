"""§1.1 Longshot Bias — excess returns from fading longshots on Kalshi."""

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
BLUE = "#4C72B0"
RED = "#e74c3c"
GREEN = "#2ecc71"
ORANGE = "#ff7f0e"
PURPLE = "#9467bd"

# ---------------------------------------------------------------------------
# Query: win rate by price (1-99)
# ---------------------------------------------------------------------------
def load_win_rates() -> dict:
    """Return per-price trade and contract statistics for prices 1-99."""
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
            CASE WHEN t.taker_side = m.result THEN 1 ELSE 0 END AS won,
            t.count AS contracts
        FROM '{TRADES_DIR}/*.parquet' t
        INNER JOIN resolved_markets m ON t.ticker = m.ticker
        UNION ALL
        -- Maker side (counterparty)
        SELECT
            CASE WHEN t.taker_side = 'yes' THEN t.no_price ELSE t.yes_price END AS price,
            CASE WHEN t.taker_side != m.result THEN 1 ELSE 0 END AS won,
            t.count AS contracts
        FROM '{TRADES_DIR}/*.parquet' t
        INNER JOIN resolved_markets m ON t.ticker = m.ticker
    )
    SELECT price,
        COUNT(*) AS n_trades,
        SUM(contracts) AS n_contracts,
        SUM(won) AS wins_trades,
        SUM(won * contracts) AS wins_contracts,
        100.0 * SUM(won) / COUNT(*) AS win_rate_trade,
        100.0 * SUM(won * contracts) / SUM(contracts) AS win_rate_contract
    FROM all_positions
    WHERE price BETWEEN 1 AND 99
    GROUP BY price
    ORDER BY price
    """
    rows = con.execute(sql).fetchall()
    con.close()

    data = {}
    for (
        price,
        n_trades,
        n_contracts,
        wins_trades,
        wins_contracts,
        win_rate_trade,
        win_rate_contract,
    ) in rows:
        data[int(price)] = {
            "n_trades": int(n_trades),
            "n_contracts": int(n_contracts),
            "wins_trades": int(wins_trades),
            "wins_contracts": float(wins_contracts),
            "win_rate_trade": float(win_rate_trade),
            "win_rate_contract": float(win_rate_contract),
        }
    return data


# ---------------------------------------------------------------------------
# Compute metrics
# ---------------------------------------------------------------------------
def compute_metrics(data: dict) -> dict:
    """
    For each price point compute:
      - excess_return_pp_contract = contract-weighted win_rate - price
      - excess_return_pp_trade = trade-weighted win_rate - price
      - edge_cents = excess_return_pp_contract (1 pp = 1 cent/contract)
      - For "fade longshot" strategy:
        If price < 15, the opposing side is at (100 - price) with
        win_rate_opposing = 100 - win_rate.  Compute their excess return.
        Similarly for price > 85.
    """
    prices = sorted(data.keys())
    results = {}
    for p in prices:
        d = data[p]
        wr_contract = d["win_rate_contract"]
        wr_trade = d["win_rate_trade"]
        n_trades = d["n_trades"]
        n_contracts = d["n_contracts"]

        excess_return_pp_contract = wr_contract - p
        excess_return_pp_trade = wr_trade - p
        edge_cents = excess_return_pp_contract  # 1 pp = 1 cent per contract

        # Fade-longshot metrics: what does the OTHER side earn?
        opp_price = 100 - p
        opp_wr_contract = 100.0 - wr_contract
        opp_wr_trade = 100.0 - wr_trade
        opp_excess_pp_contract = opp_wr_contract - opp_price
        opp_excess_pp_trade = opp_wr_trade - opp_price

        results[p] = {
            "n_trades": n_trades,
            "n_contracts": n_contracts,
            "wins_trades": d["wins_trades"],
            "wins_contracts": d["wins_contracts"],
            "win_rate_trade": wr_trade,
            "win_rate_contract": wr_contract,
            "excess_return_pp_contract": excess_return_pp_contract,
            "excess_return_pp_trade": excess_return_pp_trade,
            "edge_cents": edge_cents,
            "opp_price": opp_price,
            "opp_win_rate_contract": opp_wr_contract,
            "opp_win_rate_trade": opp_wr_trade,
            "opp_excess_pp_contract": opp_excess_pp_contract,
            "opp_excess_pp_trade": opp_excess_pp_trade,
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
            buckets[mid] = {
                "n_trades": 0,
                "n_contracts": 0,
                "wins_contracts": 0.0,
                "price_contract_weighted_sum": 0.0,
            }
        buckets[mid]["n_trades"] += d["n_trades"]
        buckets[mid]["n_contracts"] += d["n_contracts"]
        buckets[mid]["wins_contracts"] += d["wins_contracts"]
        buckets[mid]["price_contract_weighted_sum"] += p * d["n_contracts"]

    result = []
    for mid in sorted(buckets.keys()):
        b = buckets[mid]
        if b["n_trades"] >= min_obs and b["n_contracts"] > 0:
            wr = 100.0 * b["wins_contracts"] / b["n_contracts"]
            avg_price = b["price_contract_weighted_sum"] / b["n_contracts"]
            result.append({
                "bucket_mid": mid,
                "avg_price": avg_price,
                "n_trades": b["n_trades"],
                "n_contracts": b["n_contracts"],
                "win_rate_contract": wr,
                "excess_return_pp_contract": wr - avg_price,
            })
    return result


# ---------------------------------------------------------------------------
# Figure 1: Excess Returns by Price
# ---------------------------------------------------------------------------
def plot_excess_returns(metrics: dict) -> None:
    prices = sorted(metrics.keys())
    excess = [metrics[p]["excess_return_pp_contract"] for p in prices]
    n_obs = [metrics[p]["n_trades"] for p in prices]

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
    bucket_prices = [b["avg_price"] for b in buckets]
    win_rates = [b["win_rate_contract"] for b in buckets]

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
        x = b["avg_price"]
        wr = b["win_rate_contract"]
        if x < 15:
            color = RED
        elif x > 85:
            color = ORANGE
        else:
            color = BLUE
        ax.scatter(x, wr, color=color, s=60, zorder=5, edgecolors="white", linewidth=0.5)

    # Connect with line
    ax.plot(bucket_prices, win_rates, color=BLUE, linewidth=1.5, alpha=0.7)

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
    longshot_prices = [p for p in metrics if p < 15 and metrics[p]["n_trades"] >= 100]
    favorite_prices = [p for p in metrics if p > 85 and metrics[p]["n_trades"] >= 100]

    def zone_stats(price_list, label):
        if not price_list:
            return {"zone": label, "n_prices": 0}
        total_trades = sum(metrics[p]["n_trades"] for p in price_list)
        total_contracts = sum(metrics[p]["n_contracts"] for p in price_list)
        total_wins_trade = sum(metrics[p]["wins_trades"] for p in price_list)
        total_wins_contract = sum(metrics[p]["wins_contracts"] for p in price_list)
        weighted_excess_contract = sum(
            metrics[p]["excess_return_pp_contract"] * metrics[p]["n_contracts"] for p in price_list
        ) / total_contracts
        weighted_excess_trade = sum(
            metrics[p]["excess_return_pp_trade"] * metrics[p]["n_trades"] for p in price_list
        ) / total_trades
        avg_win_rate_contract = 100.0 * total_wins_contract / total_contracts
        avg_win_rate_trade = 100.0 * total_wins_trade / total_trades
        avg_price_contract = sum(p * metrics[p]["n_contracts"] for p in price_list) / total_contracts
        return {
            "zone": label,
            "price_range": f"{min(price_list)}-{max(price_list)}c",
            "n_prices": len(price_list),
            "total_trades": total_trades,
            "total_contracts": total_contracts,
            "avg_price_cents_contract_weighted": round(avg_price_contract, 2),
            "avg_win_rate_pct_contract_weighted": round(avg_win_rate_contract, 2),
            "avg_win_rate_pct_trade_weighted": round(avg_win_rate_trade, 2),
            "weighted_excess_return_pp_contract_weighted": round(weighted_excess_contract, 2),
            "weighted_excess_return_pp_trade_weighted": round(weighted_excess_trade, 2),
            "edge_cents_per_contract": round(weighted_excess_contract, 2),
        }

    longshot_stats = zone_stats(longshot_prices, "longshot (<15c)")
    favorite_stats = zone_stats(favorite_prices, "favorite (>85c)")

    # Fade-longshot strategy: take the OPPOSING side when someone buys at <15c
    # i.e., you are selling YES at price <15c, or equivalently buying NO at (100-price)
    # Your edge = opp_excess_pp for that price
    fade_longshot_prices = [p for p in metrics if p < 15 and metrics[p]["n_trades"] >= 100]
    if fade_longshot_prices:
        total_contracts = sum(metrics[p]["n_contracts"] for p in fade_longshot_prices)
        weighted_fade_edge = sum(
            metrics[p]["opp_excess_pp_contract"] * metrics[p]["n_contracts"] for p in fade_longshot_prices
        ) / total_contracts
        fade_win_rate = sum(
            metrics[p]["opp_win_rate_contract"] * metrics[p]["n_contracts"] for p in fade_longshot_prices
        ) / total_contracts
        fade_longshot = {
            "strategy": "fade_longshot",
            "description": "Bet AGAINST contracts priced <15c (buy NO / sell YES)",
            "total_contracts": total_contracts,
            "avg_opposing_price": round(sum((100 - p) * metrics[p]["n_contracts"] for p in fade_longshot_prices) / total_contracts, 2),
            "avg_win_rate": round(fade_win_rate, 2),
            "weighted_edge_pp": round(weighted_fade_edge, 2),
            "edge_cents_per_contract": round(weighted_fade_edge, 2),
        }
    else:
        fade_longshot = {"strategy": "fade_longshot", "total_trades": 0}

    # Fade-favorite strategy: take the OPPOSING side when someone buys at >85c
    fade_favorite_prices = [p for p in metrics if p > 85 and metrics[p]["n_trades"] >= 100]
    if fade_favorite_prices:
        total_contracts = sum(metrics[p]["n_contracts"] for p in fade_favorite_prices)
        weighted_fade_edge = sum(
            metrics[p]["opp_excess_pp_contract"] * metrics[p]["n_contracts"] for p in fade_favorite_prices
        ) / total_contracts
        fade_win_rate = sum(
            metrics[p]["opp_win_rate_contract"] * metrics[p]["n_contracts"] for p in fade_favorite_prices
        ) / total_contracts
        fade_favorite = {
            "strategy": "fade_favorite",
            "description": "Bet AGAINST contracts priced >85c (buy NO when price >85c)",
            "total_contracts": total_contracts,
            "avg_opposing_price": round(sum((100 - p) * metrics[p]["n_contracts"] for p in fade_favorite_prices) / total_contracts, 2),
            "avg_win_rate": round(fade_win_rate, 2),
            "weighted_edge_pp": round(weighted_fade_edge, 2),
            "edge_cents_per_contract": round(weighted_fade_edge, 2),
        }
    else:
        fade_favorite = {"strategy": "fade_favorite", "total_trades": 0}

    # Breakeven: find price points where excess_return crosses zero
    sorted_prices = sorted(p for p in metrics if metrics[p]["n_trades"] >= 100)
    breakevens = []
    for i in range(len(sorted_prices) - 1):
        p1, p2 = sorted_prices[i], sorted_prices[i + 1]
        e1 = metrics[p1]["excess_return_pp_contract"]
        e2 = metrics[p2]["excess_return_pp_contract"]
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
            "n_trades": m["n_trades"],
            "n_contracts": m["n_contracts"],
            "win_rate_contract": round(m["win_rate_contract"], 2),
            "win_rate_trade": round(m["win_rate_trade"], 2),
            "excess_return_pp_contract": round(m["excess_return_pp_contract"], 2),
            "opposing_win_rate_contract": round(m["opp_win_rate_contract"], 2),
            "opposing_excess_pp_contract": round(m["opp_excess_pp_contract"], 2),
        })

    favorite_detail = []
    for p in sorted(favorite_prices):
        m = metrics[p]
        favorite_detail.append({
            "price": p,
            "n_trades": m["n_trades"],
            "n_contracts": m["n_contracts"],
            "win_rate_contract": round(m["win_rate_contract"], 2),
            "win_rate_trade": round(m["win_rate_trade"], 2),
            "excess_return_pp_contract": round(m["excess_return_pp_contract"], 2),
            "opposing_win_rate_contract": round(m["opp_win_rate_contract"], 2),
            "opposing_excess_pp_contract": round(m["opp_excess_pp_contract"], 2),
        })

    # Bucket-level summary for report table
    bucket_table = []
    for b in buckets:
        bucket_price = b["avg_price"]
        if bucket_price <= 15 or bucket_price >= 85:
            bucket_table.append({
                "bucket_mid": b["bucket_mid"],
                "avg_price_cents": round(bucket_price, 2),
                "n_trades": b["n_trades"],
                "n_contracts": b["n_contracts"],
                "win_rate_contract": round(b["win_rate_contract"], 2),
                "implied_prob_contract_weighted": round(bucket_price, 2),
                "excess_return_pp_contract": round(b["excess_return_pp_contract"], 2),
                "edge_cents_per_contract": round(b["excess_return_pp_contract"], 2),
            })

    total_trades = sum(metrics[p]["n_trades"] for p in metrics)
    total_contracts = sum(metrics[p]["n_contracts"] for p in metrics)

    summary = {
        "total_trades": total_trades,
        "total_contracts": total_contracts,
        "total_prices_with_data": len(metrics),
        "prices_with_100_obs": len([p for p in metrics if metrics[p]["n_trades"] >= 100]),
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
