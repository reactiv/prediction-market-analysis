"""Strategy Report 5.4: Contract Type Segmentation.

Analyzes Polymarket trades by contract type (CTF Exchange vs NegRisk CTF Exchange)
to compare volume, trade sizes, and calibration across contract types.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import duckdb
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
TRADES_GLOB = str(BASE_DIR / "data" / "polymarket" / "trades" / "*.parquet")
MARKETS_GLOB = str(BASE_DIR / "data" / "polymarket" / "markets" / "*.parquet")
FIGURES_DIR = BASE_DIR / "strategy_reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

COLORS = ["#4C72B0", "#e74c3c", "#2ecc71", "#ff7f0e", "#9467bd"]
FIGSIZE = (12, 7)
DPI = 150


def build_token_resolution(con: duckdb.DuckDBPyConnection) -> int:
    """Build token_id -> won mapping from resolved markets. Returns count."""
    markets_df = con.execute(f"""
        SELECT id, clob_token_ids, outcome_prices
        FROM '{MARKETS_GLOB}'
        WHERE closed = true
    """).df()

    token_won: dict[str, bool] = {}
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
        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    con.execute("CREATE TABLE token_resolution (token_id VARCHAR, won BOOLEAN)")
    con.executemany(
        "INSERT INTO token_resolution VALUES (?, ?)", list(token_won.items())
    )
    return len(token_won)


def query_contract_types(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Discover the distinct contract types present in the data."""
    print("Discovering contract types...", file=sys.stderr)
    df = con.execute(f"""
        SELECT DISTINCT _contract
        FROM '{TRADES_GLOB}'
    """).df()
    types = sorted(df["_contract"].dropna().tolist())
    print(f"  Found contract types: {types}", file=sys.stderr)
    return types


def query_volume_stats(con: duckdb.DuckDBPyConnection) -> dict:
    """Query per-contract-type volume and trade count."""
    print("Querying volume stats (scanning all trades)...", file=sys.stderr)
    df = con.execute(f"""
        SELECT
            _contract,
            COUNT(*) AS trade_count,
            SUM(
                CASE WHEN maker_asset_id = '0'
                     THEN maker_amount
                     ELSE taker_amount
                END
            ) AS total_cash_volume,
            AVG(
                CASE WHEN maker_asset_id = '0'
                     THEN maker_amount
                     ELSE taker_amount
                END
            ) AS avg_cash_per_trade,
            MEDIAN(
                CASE WHEN maker_asset_id = '0'
                     THEN maker_amount
                     ELSE taker_amount
                END
            ) AS median_cash_per_trade,
            COUNT(DISTINCT
                CASE WHEN maker_asset_id = '0'
                     THEN taker_asset_id
                     ELSE maker_asset_id
                END
            ) AS distinct_tokens
        FROM '{TRADES_GLOB}'
        WHERE maker_amount > 0 AND taker_amount > 0
        GROUP BY _contract
        ORDER BY total_cash_volume DESC
    """).df()
    return df


def query_trade_size_distributions(con: duckdb.DuckDBPyConnection) -> dict:
    """Query trade size distribution per contract type using log-scale buckets."""
    print("Querying trade size distributions...", file=sys.stderr)
    df = con.execute(f"""
        WITH trade_sizes AS (
            SELECT
                _contract,
                CASE WHEN maker_asset_id = '0'
                     THEN maker_amount / 1e6
                     ELSE taker_amount / 1e6
                END AS size_usdc
            FROM '{TRADES_GLOB}'
            WHERE maker_amount > 0 AND taker_amount > 0
        )
        SELECT
            _contract,
            CASE
                WHEN size_usdc < 1 THEN '< $1'
                WHEN size_usdc < 10 THEN '$1-10'
                WHEN size_usdc < 100 THEN '$10-100'
                WHEN size_usdc < 1000 THEN '$100-1K'
                WHEN size_usdc < 10000 THEN '$1K-10K'
                ELSE '$10K+'
            END AS size_bucket,
            CASE
                WHEN size_usdc < 1 THEN 0
                WHEN size_usdc < 10 THEN 1
                WHEN size_usdc < 100 THEN 2
                WHEN size_usdc < 1000 THEN 3
                WHEN size_usdc < 10000 THEN 4
                ELSE 5
            END AS bucket_order,
            COUNT(*) AS cnt
        FROM trade_sizes
        GROUP BY _contract, size_bucket, bucket_order
        ORDER BY _contract, bucket_order
    """).df()
    return df


def query_calibration_by_contract(con: duckdb.DuckDBPyConnection) -> dict:
    """Query calibration data (win rate vs price) per contract type."""
    print("Querying calibration by contract type (this is the slow one)...", file=sys.stderr)
    df = con.execute(f"""
        WITH trade_positions AS (
            SELECT
                _contract,
                CASE
                    WHEN t.maker_asset_id = '0'
                        THEN ROUND(100.0 * t.maker_amount / t.taker_amount)
                    ELSE ROUND(100.0 * t.taker_amount / t.maker_amount)
                END AS price,
                tr.won
            FROM '{TRADES_GLOB}' t
            INNER JOIN token_resolution tr ON (
                CASE WHEN t.maker_asset_id = '0'
                     THEN t.taker_asset_id
                     ELSE t.maker_asset_id
                END = tr.token_id
            )
            WHERE t.taker_amount > 0 AND t.maker_amount > 0
        )
        SELECT
            _contract,
            price,
            COUNT(*) AS total_trades,
            SUM(CASE WHEN won THEN 1 ELSE 0 END) AS wins,
            100.0 * SUM(CASE WHEN won THEN 1 ELSE 0 END) / COUNT(*) AS win_rate
        FROM trade_positions
        WHERE price >= 1 AND price <= 99
        GROUP BY _contract, price
        ORDER BY _contract, price
    """).df()
    return df


def compute_calibration_metrics(df) -> dict:
    """Compute Brier score and ECE for a calibration dataframe."""
    total_trades = df["total_trades"].sum()
    if total_trades == 0:
        return {"brier_score": None, "ece": None, "log_loss": None, "total_resolved_trades": 0}

    brier_sum = 0.0
    ece_sum = 0.0
    log_loss_sum = 0.0
    epsilon = 1e-6

    for _, row in df.iterrows():
        p = row["price"] / 100.0
        wins = row["wins"]
        losses = row["total_trades"] - wins
        actual = row["win_rate"] / 100.0

        brier_sum += wins * (p - 1) ** 2 + losses * p ** 2
        ece_sum += row["total_trades"] * abs(actual - p)

        p_clamped = max(min(p, 1 - epsilon), epsilon)
        log_loss_sum += wins * (-math.log(p_clamped)) + losses * (-math.log(1 - p_clamped))

    return {
        "brier_score": round(brier_sum / total_trades, 4),
        "ece": round(ece_sum / total_trades, 4),
        "log_loss": round(log_loss_sum / total_trades, 4),
        "total_resolved_trades": int(total_trades),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_volume_by_contract_type(volume_df, output_path: Path):
    """Bar chart of volume and trade count by contract type."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)

    contract_types = volume_df["_contract"].tolist()
    colors = COLORS[: len(contract_types)]

    # Left: total cash volume in USDC (amounts are in 1e-6 USDC units)
    volumes_usd = volume_df["total_cash_volume"].values / 1e6
    bars1 = ax1.bar(contract_types, volumes_usd, color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_title("Total Volume by Contract Type", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Total Volume (USD)")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x / 1e9:.1f}B" if x >= 1e9 else f"${x / 1e6:.0f}M"))
    for bar, val in zip(bars1, volumes_usd):
        label = f"${val / 1e9:.2f}B" if val >= 1e9 else f"${val / 1e6:.0f}M"
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 label, ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.set_xticks(range(len(contract_types)))
    ax1.set_xticklabels(contract_types, rotation=15, ha="right", fontsize=9)

    # Right: trade count
    counts = volume_df["trade_count"].values
    bars2 = ax2.bar(contract_types, counts, color=colors, edgecolor="white", linewidth=0.5)
    ax2.set_title("Trade Count by Contract Type", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Number of Trades")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M"))
    for bar, val in zip(bars2, counts):
        label = f"{val / 1e6:.1f}M"
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 label, ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_xticks(range(len(contract_types)))
    ax2.set_xticklabels(contract_types, rotation=15, ha="right", fontsize=9)

    fig.suptitle("Polymarket: Volume by Contract Type", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}", file=sys.stderr)


def plot_trade_size_distribution(size_df, output_path: Path):
    """Grouped bar chart of trade size distributions by contract type."""
    contract_types = sorted(size_df["_contract"].unique())
    buckets = ["< $1", "$1-10", "$10-100", "$100-1K", "$1K-10K", "$10K+"]

    fig, ax = plt.subplots(figsize=FIGSIZE)

    x = np.arange(len(buckets))
    width = 0.8 / len(contract_types)

    for i, ct in enumerate(contract_types):
        ct_data = size_df[size_df["_contract"] == ct].set_index("size_bucket")
        # Normalize to percentages within each contract type
        ct_total = ct_data["cnt"].sum()
        pcts = [(ct_data.loc[b, "cnt"] / ct_total * 100) if b in ct_data.index else 0 for b in buckets]
        offset = (i - (len(contract_types) - 1) / 2) * width
        bars = ax.bar(x + offset, pcts, width, label=ct, color=COLORS[i],
                       edgecolor="white", linewidth=0.5)
        for bar, pct in zip(bars, pcts):
            if pct > 3:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Trade Size (USDC)", fontsize=11)
    ax.set_ylabel("Percentage of Trades", fontsize=11)
    ax.set_title("Trade Size Distribution by Contract Type", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())

    plt.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}", file=sys.stderr)


def plot_calibration_by_contract(cal_df, metrics_by_type: dict, output_path: Path):
    """Scatter plot of win rate vs price per contract type."""
    contract_types = sorted(cal_df["_contract"].unique())

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Perfect calibration line
    ax.plot([0, 100], [0, 100], linestyle="--", color="#888888", linewidth=1.5,
            label="Perfect calibration", zorder=1)

    for i, ct in enumerate(contract_types):
        ct_data = cal_df[cal_df["_contract"] == ct]
        m = metrics_by_type.get(ct, {})
        brier = m.get("brier_score", "?")
        ece = m.get("ece", "?")
        label = f"{ct} (Brier={brier}, ECE={ece})"
        ax.scatter(ct_data["price"], ct_data["win_rate"], s=18, alpha=0.6,
                   color=COLORS[i], label=label, edgecolors="none", zorder=2)

    ax.set_xlabel("Contract Price (cents)", fontsize=11)
    ax.set_ylabel("Win Rate (%)", fontsize=11)
    ax.set_title("Calibration by Contract Type", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    con = duckdb.connect()

    # Step 0: Discover contract types
    contract_types = query_contract_types(con)

    # Step 1: Build token resolution table (needed for calibration)
    print("Building token resolution table...", file=sys.stderr)
    n_tokens = build_token_resolution(con)
    print(f"  Resolved {n_tokens} tokens", file=sys.stderr)

    # Step 2: Volume statistics
    print("\n--- Volume Statistics ---", file=sys.stderr)
    volume_df = query_volume_stats(con)
    print(volume_df.to_string(), file=sys.stderr)

    # Step 3: Trade size distributions
    print("\n--- Trade Size Distributions ---", file=sys.stderr)
    size_df = query_trade_size_distributions(con)
    print(size_df.to_string(), file=sys.stderr)

    # Step 4: Calibration by contract type
    print("\n--- Calibration by Contract Type ---", file=sys.stderr)
    cal_df = query_calibration_by_contract(con)

    # Compute calibration metrics per contract type
    metrics_by_type = {}
    for ct in contract_types:
        ct_cal = cal_df[cal_df["_contract"] == ct]
        if len(ct_cal) > 0:
            metrics_by_type[ct] = compute_calibration_metrics(ct_cal)
    for ct, m in metrics_by_type.items():
        print(f"  {ct}: {m}", file=sys.stderr)

    # Step 5: Generate figures
    print("\n--- Generating Figures ---", file=sys.stderr)
    plot_volume_by_contract_type(
        volume_df,
        FIGURES_DIR / "5_4_volume_by_contract_type.png",
    )
    plot_trade_size_distribution(
        size_df,
        FIGURES_DIR / "5_4_trade_size_by_contract_type.png",
    )
    plot_calibration_by_contract(
        cal_df,
        metrics_by_type,
        FIGURES_DIR / "5_4_calibration_by_contract_type.png",
    )

    # Step 6: JSON output to stdout
    output = {
        "report": "5.4 Contract Type Segmentation",
        "contract_types": contract_types,
        "volume_summary": {},
        "trade_size_summary": {},
        "calibration_summary": {},
    }

    for _, row in volume_df.iterrows():
        ct = row["_contract"]
        output["volume_summary"][ct] = {
            "trade_count": int(row["trade_count"]),
            "total_cash_volume_usdc": round(float(row["total_cash_volume"]) / 1e6, 2),
            "avg_trade_usdc": round(float(row["avg_cash_per_trade"]) / 1e6, 2),
            "median_trade_usdc": round(float(row["median_cash_per_trade"]) / 1e6, 2),
            "distinct_tokens": int(row["distinct_tokens"]),
        }

    for ct in contract_types:
        ct_sizes = size_df[size_df["_contract"] == ct]
        total = ct_sizes["cnt"].sum()
        output["trade_size_summary"][ct] = {
            row["size_bucket"]: {
                "count": int(row["cnt"]),
                "pct": round(row["cnt"] / total * 100, 2),
            }
            for _, row in ct_sizes.iterrows()
        }

    output["calibration_summary"] = metrics_by_type

    print(json.dumps(output, indent=2))
    print("\nDone.", file=sys.stderr)

    con.close()


if __name__ == "__main__":
    main()
