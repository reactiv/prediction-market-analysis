"""§3.4 Volume Regime Switching — calibration by market volume regime."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import duckdb  # noqa: E402
import pandas as pd  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
TRADES_GLOB = str(BASE_DIR / "data" / "kalshi" / "trades" / "*.parquet")
MARKETS_GLOB = str(BASE_DIR / "data" / "kalshi" / "markets" / "*.parquet")
FIG_DIR = BASE_DIR / "strategy_reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
COLORS = {
    "High":   "#1a3a5c",  # dark blue
    "Medium": "#4C72B0",  # medium blue
    "Low":    "#8cb4d8",  # light blue
}

# ---------------------------------------------------------------------------
# Query: trade-level data with volume regime classification
# ---------------------------------------------------------------------------
def load_data(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load trade-level calibration data bucketed by price and volume regime."""
    sql = f"""
    WITH resolved_markets AS (
        SELECT ticker, result
        FROM '{MARKETS_GLOB}'
        WHERE status = 'finalized' AND result IN ('yes', 'no')
    ),
    market_trade_counts AS (
        SELECT t.ticker, COUNT(*) AS total_trades, SUM(t.count) AS total_contracts
        FROM '{TRADES_GLOB}' t
        INNER JOIN resolved_markets rm ON t.ticker = rm.ticker
        GROUP BY t.ticker
    ),
    market_percentiles AS (
        SELECT
            PERCENTILE_CONT(0.33) WITHIN GROUP (ORDER BY total_trades) AS p33,
            PERCENTILE_CONT(0.67) WITHIN GROUP (ORDER BY total_trades) AS p67
        FROM market_trade_counts
    ),
    classified AS (
        SELECT
            t.ticker,
            CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS price,
            CASE WHEN t.taker_side = m.result THEN 1.0 ELSE 0.0 END AS won,
            t.count AS contracts,
            CASE
                WHEN mc.total_trades <= mp.p33 THEN 'Low'
                WHEN mc.total_trades <= mp.p67 THEN 'Medium'
                ELSE 'High'
            END AS volume_regime,
            CASE
                WHEN mc.total_trades <= mp.p33 THEN 1
                WHEN mc.total_trades <= mp.p67 THEN 2
                ELSE 3
            END AS regime_order
        FROM '{TRADES_GLOB}' t
        INNER JOIN resolved_markets m ON t.ticker = m.ticker
        INNER JOIN market_trade_counts mc ON t.ticker = mc.ticker
        CROSS JOIN market_percentiles mp
    )
    SELECT
        volume_regime, regime_order, price,
        AVG(won) AS win_rate,
        price / 100.0 AS expected_win_rate,
        AVG(won) - price / 100.0 AS excess_return,
        COUNT(*) AS n_trades,
        SUM(contracts) AS total_contracts
    FROM classified
    WHERE price BETWEEN 1 AND 99
    GROUP BY volume_regime, regime_order, price
    HAVING COUNT(*) >= 100
    ORDER BY regime_order, price
    """
    return con.execute(sql).df()


def load_regime_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Summary stats per volume regime: market count, trade count, contracts."""
    sql = f"""
    WITH resolved_markets AS (
        SELECT ticker, result
        FROM '{MARKETS_GLOB}'
        WHERE status = 'finalized' AND result IN ('yes', 'no')
    ),
    market_trade_counts AS (
        SELECT t.ticker, COUNT(*) AS total_trades, SUM(t.count) AS total_contracts
        FROM '{TRADES_GLOB}' t
        INNER JOIN resolved_markets rm ON t.ticker = rm.ticker
        GROUP BY t.ticker
    ),
    market_percentiles AS (
        SELECT
            PERCENTILE_CONT(0.33) WITHIN GROUP (ORDER BY total_trades) AS p33,
            PERCENTILE_CONT(0.67) WITHIN GROUP (ORDER BY total_trades) AS p67
        FROM market_trade_counts
    ),
    classified_markets AS (
        SELECT
            mc.ticker,
            mc.total_trades,
            mc.total_contracts,
            CASE
                WHEN mc.total_trades <= mp.p33 THEN 'Low'
                WHEN mc.total_trades <= mp.p67 THEN 'Medium'
                ELSE 'High'
            END AS volume_regime,
            CASE
                WHEN mc.total_trades <= mp.p33 THEN 1
                WHEN mc.total_trades <= mp.p67 THEN 2
                ELSE 3
            END AS regime_order
        FROM market_trade_counts mc
        INNER JOIN resolved_markets m ON mc.ticker = m.ticker
        CROSS JOIN market_percentiles mp
    )
    SELECT
        volume_regime,
        regime_order,
        COUNT(*) AS n_markets,
        SUM(total_trades) AS total_trades,
        SUM(total_contracts) AS total_contracts
    FROM classified_markets
    GROUP BY volume_regime, regime_order
    ORDER BY regime_order
    """
    return con.execute(sql).df()


def load_percentiles(con: duckdb.DuckDBPyConnection) -> dict:
    """Return the p33 and p67 percentile thresholds."""
    sql = f"""
    WITH resolved_markets AS (
        SELECT ticker, result
        FROM '{MARKETS_GLOB}'
        WHERE status = 'finalized' AND result IN ('yes', 'no')
    ),
    market_trade_counts AS (
        SELECT t.ticker, COUNT(*) AS total_trades
        FROM '{TRADES_GLOB}' t
        INNER JOIN resolved_markets rm ON t.ticker = rm.ticker
        GROUP BY t.ticker
    )
    SELECT
        PERCENTILE_CONT(0.33) WITHIN GROUP (ORDER BY total_trades) AS p33,
        PERCENTILE_CONT(0.67) WITHIN GROUP (ORDER BY total_trades) AS p67
    FROM market_trade_counts
    """
    row = con.execute(sql).fetchone()
    return {"p33": float(row[0]), "p67": float(row[1])}


# ---------------------------------------------------------------------------
# Bucket calibration and MAE
# ---------------------------------------------------------------------------
def bucket_calibration(df: pd.DataFrame, bucket_width: int = 5) -> pd.DataFrame:
    """Aggregate price-level data into wider buckets for smoother curves."""
    df = df.copy()
    df["bucket_mid"] = (df["price"] // bucket_width) * bucket_width + bucket_width / 2.0
    df["price_weighted_trades"] = df["price"] * df["n_trades"]
    agg = (
        df.groupby(["volume_regime", "regime_order", "bucket_mid"], as_index=False)
        .agg(
            n_trades=("n_trades", "sum"),
            total_contracts=("total_contracts", "sum"),
            # Weighted win rate: sum(win_rate * n_trades) / sum(n_trades)
            weighted_wins=("win_rate", lambda x: (x * df.loc[x.index, "n_trades"]).sum()),
            weighted_price=("price_weighted_trades", "sum"),
        )
    )
    agg["win_rate"] = agg["weighted_wins"] / agg["n_trades"]
    agg["avg_price_cents"] = agg["weighted_price"] / agg["n_trades"]
    agg["expected"] = agg["avg_price_cents"] / 100.0
    agg["abs_error"] = (agg["win_rate"] - agg["expected"]).abs()
    agg["excess_return"] = agg["win_rate"] - agg["expected"]
    # Filter tiny buckets
    agg = agg[agg["n_trades"] >= 200].copy()
    return agg


def compute_mae(df: pd.DataFrame) -> pd.DataFrame:
    """Weighted MAE per regime from exact price points (no bucket-midpoint approximation)."""
    df = df.copy()
    df["expected"] = df["price"] / 100.0
    df["abs_error"] = (df["win_rate"] - df["expected"]).abs()
    df["excess_return"] = df["win_rate"] - df["expected"]
    df["weighted_error"] = df["abs_error"] * df["n_trades"]
    mae = (
        df.groupby(["volume_regime", "regime_order"], as_index=False)
        .agg(
            total_trades=("n_trades", "sum"),
            weighted_error_sum=("weighted_error", "sum"),
            n_price_points=("price", "count"),
            avg_excess_return=("excess_return", lambda x: (x * df.loc[x.index, "n_trades"]).sum() / df.loc[x.index, "n_trades"].sum()),
        )
    )
    mae["mae_pp"] = mae["weighted_error_sum"] / mae["total_trades"] * 100  # in percentage points
    mae = mae.sort_values("regime_order").reset_index(drop=True)
    return mae


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_calibration_curves(bucketed: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")

    for regime in ["High", "Medium", "Low"]:
        subset = bucketed[bucketed["volume_regime"] == regime].sort_values("avg_price_cents")
        if subset.empty:
            continue
        ax.plot(
            subset["avg_price_cents"],
            subset["win_rate"] * 100,
            marker="o",
            markersize=4,
            linewidth=2.0,
            label=f"{regime} Volume",
            color=COLORS[regime],
            alpha=0.9,
        )

    # Perfect calibration
    ax.plot(
        [0, 100], [0, 100],
        linestyle="--", color="gray", linewidth=1, alpha=0.7,
        label="Perfect calibration",
    )

    ax.set_xlabel("Contract Price (cents)", fontsize=12)
    ax.set_ylabel("Empirical Win Rate (%)", fontsize=12)
    ax.set_title("Calibration by Volume Regime", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = FIG_DIR / "3_4_calibration_by_volume_regime.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}", file=sys.stderr)


def plot_mae_bar(mae_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")

    mae_sorted = mae_df.sort_values("regime_order")
    regimes = mae_sorted["volume_regime"].tolist()
    mae_vals = mae_sorted["mae_pp"].tolist()
    colors = [COLORS[r] for r in regimes]

    bars = ax.bar(regimes, mae_vals, color=colors, edgecolor="white", width=0.5)

    for bar, val in zip(bars, mae_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.2f} pp",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )

    ax.set_xlabel("Volume Regime", fontsize=12)
    ax.set_ylabel("Mean Absolute Error (percentage points)", fontsize=12)
    ax.set_title("Calibration Error (MAE) by Volume Regime", fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(mae_vals) * 1.35 if mae_vals else 10)
    fig.tight_layout()

    path = FIG_DIR / "3_4_mae_by_regime.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    con = duckdb.connect()

    print("Loading data...", file=sys.stderr)
    df = load_data(con)
    print(f"  Price-regime rows: {len(df):,}", file=sys.stderr)

    regime_summary = load_regime_summary(con)
    print(f"  Regime summary:\n{regime_summary.to_string()}", file=sys.stderr)

    percentiles = load_percentiles(con)
    print(f"  Percentile thresholds: p33={percentiles['p33']:.0f}, p67={percentiles['p67']:.0f}", file=sys.stderr)

    con.close()

    # Bucket and compute MAE
    bucketed = bucket_calibration(df, bucket_width=5)
    mae_df = compute_mae(df)
    print(f"  MAE by regime:\n{mae_df[['volume_regime', 'mae_pp']].to_string()}", file=sys.stderr)

    # Plot
    plot_calibration_curves(bucketed)
    plot_mae_bar(mae_df)

    # Build JSON output
    summary = {
        "percentile_thresholds": {
            "p33_trades": round(percentiles["p33"], 1),
            "p67_trades": round(percentiles["p67"], 1),
        },
        "regimes": [],
    }

    for _, row in regime_summary.iterrows():
        regime = row["volume_regime"]
        mae_row = mae_df[mae_df["volume_regime"] == regime]
        mae_val = float(mae_row["mae_pp"].iloc[0]) if len(mae_row) > 0 else None
        avg_excess = float(mae_row["avg_excess_return"].iloc[0]) * 100 if len(mae_row) > 0 else None
        summary["regimes"].append({
            "regime": regime,
            "n_markets": int(row["n_markets"]),
            "total_trades": int(row["total_trades"]),
            "total_contracts": int(row["total_contracts"]),
            "mae_pp": round(mae_val, 2) if mae_val is not None else None,
            "avg_excess_return_pp": round(avg_excess, 2) if avg_excess is not None else None,
        })

    # Best/worst calibrated
    best = mae_df.loc[mae_df["mae_pp"].idxmin(), "volume_regime"] if len(mae_df) > 0 else None
    worst = mae_df.loc[mae_df["mae_pp"].idxmax(), "volume_regime"] if len(mae_df) > 0 else None
    summary["best_calibrated"] = best
    summary["worst_calibrated"] = worst

    # Per-bucket detail for each regime
    summary["calibration_detail"] = {}
    for regime in ["Low", "Medium", "High"]:
        subset = bucketed[bucketed["volume_regime"] == regime].sort_values("avg_price_cents")
        summary["calibration_detail"][regime] = [
            {
                "bucket_mid": float(r["bucket_mid"]),
                "avg_price_cents": round(float(r["avg_price_cents"]), 2),
                "win_rate_pct": round(float(r["win_rate"]) * 100, 2),
                "expected_pct": round(float(r["expected"]) * 100, 2),
                "error_pp": round(float(r["abs_error"]) * 100, 2),
                "n_trades": int(r["n_trades"]),
            }
            for _, r in subset.iterrows()
        ]

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
