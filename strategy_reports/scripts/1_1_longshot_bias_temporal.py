"""§1.1 Temporal — longshot bias stratified by quarter."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
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
RED = "#e74c3c"
BLUE = "#4C72B0"
GREEN = "#2ecc71"

# ---------------------------------------------------------------------------
# Query: excess return by quarter and price zone (taker positions only)
# ---------------------------------------------------------------------------
QUERY = f"""
WITH resolved_markets AS (
    SELECT ticker, result
    FROM '{MARKETS_DIR}/*.parquet'
    WHERE status = 'finalized' AND result IN ('yes', 'no')
),
taker_positions AS (
    SELECT
        DATE_TRUNC('quarter', t.created_time) AS quarter,
        CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS price_cents,
        CASE WHEN t.taker_side = m.result THEN 1.0 ELSE 0.0 END AS won
    FROM '{TRADES_DIR}/*.parquet' t
    INNER JOIN resolved_markets m ON t.ticker = m.ticker
    WHERE t.count > 0
),
zoned AS (
    SELECT
        quarter,
        price_cents / 100.0 AS price,
        won,
        CASE
            WHEN price_cents < 15 THEN 'longshot'
            WHEN price_cents > 85 THEN 'favourite'
            ELSE 'mid-range'
        END AS zone
    FROM taker_positions
    WHERE price_cents BETWEEN 1 AND 99
)
SELECT
    quarter,
    zone,
    COUNT(*)            AS n_trades,
    AVG(price)          AS avg_price,
    AVG(won)            AS win_rate,
    AVG(won) - AVG(price) AS excess_return
FROM zoned
GROUP BY quarter, zone
HAVING COUNT(*) >= 100
ORDER BY quarter, zone
"""


def main() -> None:
    print("Running temporal longshot-bias query...", file=sys.stderr)
    con = duckdb.connect()
    rows = con.execute(QUERY).fetchall()
    con.close()
    print(f"  Rows returned: {len(rows)}", file=sys.stderr)

    # Organise into {zone: [(quarter, excess_return, n_trades, avg_price, win_rate)]}
    from collections import defaultdict
    zone_data: dict[str, list] = defaultdict(list)
    all_records: list[dict] = []

    for quarter, zone, n_trades, avg_price, win_rate, excess_return in rows:
        zone_data[zone].append({
            "quarter": str(quarter.date()) if hasattr(quarter, "date") else str(quarter),
            "excess_return_pp": round(excess_return * 100, 2),
            "n_trades": int(n_trades),
            "avg_price": round(avg_price * 100, 2),
            "win_rate_pct": round(win_rate * 100, 2),
        })
        all_records.append({
            "quarter": str(quarter.date()) if hasattr(quarter, "date") else str(quarter),
            "zone": zone,
            "excess_return_pp": round(excess_return * 100, 2),
            "n_trades": int(n_trades),
            "avg_price": round(avg_price * 100, 2),
            "win_rate_pct": round(win_rate * 100, 2),
        })

    # ----- Plot -----
    print("Plotting...", file=sys.stderr)
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")

    zone_meta = {
        "longshot":  {"color": RED,   "label": "Longshot (<15c)",   "marker": "o"},
        "mid-range": {"color": BLUE,  "label": "Mid-range (15-85c)", "marker": "s"},
        "favourite": {"color": GREEN, "label": "Favourite (>85c)",  "marker": "^"},
    }

    for zone_name, meta in zone_meta.items():
        pts = zone_data.get(zone_name, [])
        if not pts:
            continue
        import datetime as _dt
        quarters = [_dt.datetime.strptime(p["quarter"], "%Y-%m-%d") for p in pts]
        excess = [p["excess_return_pp"] for p in pts]
        ax.plot(
            quarters, excess,
            color=meta["color"],
            label=meta["label"],
            marker=meta["marker"],
            linewidth=2,
            markersize=7,
        )

    ax.axhline(0, color="gray", linewidth=1, linestyle="--")
    ax.set_xlabel("Quarter", fontsize=12)
    ax.set_ylabel("Excess Return (percentage points)", fontsize=12)
    ax.set_title("Longshot Bias by Quarter (Taker Positions)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-Q"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()

    out_path = FIG_DIR / "1_1_longshot_bias_temporal.png"
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}", file=sys.stderr)

    # ----- JSON to stdout -----
    output = {
        "longshot": zone_data.get("longshot", []),
        "mid_range": zone_data.get("mid-range", []),
        "favourite": zone_data.get("favourite", []),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
