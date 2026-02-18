"""Performance metrics computed from backtest return sequences."""

from __future__ import annotations

from pathlib import Path

import duckdb


def compute_metrics(con: duckdb.DuckDBPyConnection, returns_path: Path) -> dict:
    """Compute standard performance metrics from a returns parquet file.

    Returns a dict suitable for writing into manifest.json.
    """
    p = str(returns_path)

    stats = con.execute(
        f"""
        SELECT
            COUNT(*)                                              AS total_trades,
            SUM(CASE WHEN won THEN 1 ELSE 0 END)                 AS wins,
            AVG(net_return_cents)                                 AS avg_return,
            STDDEV_POP(net_return_cents)                          AS std_return,
            SUM(net_return_cents * contracts)                     AS total_pnl,
            SUM(CASE WHEN won THEN gross_return_cents * contracts
                     ELSE 0 END)                                  AS gross_profit,
            SUM(CASE WHEN NOT won THEN ABS(gross_return_cents) * contracts
                     ELSE 0 END)                                  AS gross_loss,
            AVG(CASE WHEN won THEN net_return_cents END)          AS avg_win,
            AVG(CASE WHEN NOT won THEN net_return_cents END)      AS avg_loss,
            SUM(fee_cents * contracts)                            AS total_fees,
            MIN(timestamp)                                        AS first_trade,
            MAX(timestamp)                                        AS last_trade,
            AVG(time_to_expiry_s) / 3600.0                        AS avg_holding_hours
        FROM read_parquet('{p}')
        """
    ).fetchdf().iloc[0]

    total = int(stats["total_trades"])
    wins = int(stats["wins"])
    win_rate = wins / total if total > 0 else 0.0
    avg_ret = float(stats["avg_return"] or 0)
    std_ret = float(stats["std_return"] or 0)
    gross_profit = float(stats["gross_profit"] or 0)
    gross_loss = float(stats["gross_loss"] or 0)

    sharpe = avg_ret / std_ret if std_ret > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Sortino: downside deviation = sqrt(mean(min(r, 0)^2)) over ALL trades
    dd_dev = con.execute(
        f"""
        SELECT SQRT(AVG(CASE WHEN net_return_cents < 0
                        THEN net_return_cents * net_return_cents
                        ELSE 0 END))
        FROM read_parquet('{p}')
        """
    ).fetchone()[0]
    dd_dev = float(dd_dev) if dd_dev else 0.0
    sortino = avg_ret / dd_dev if dd_dev > 0 else 0.0

    # Max drawdown via cumulative PnL
    dd = con.execute(
        f"""
        WITH cum AS (
            SELECT
                ROW_NUMBER() OVER (ORDER BY timestamp, trade_id) AS rn,
                SUM(net_return_cents * contracts) OVER (
                    ORDER BY timestamp, trade_id
                    ROWS UNBOUNDED PRECEDING
                ) AS cum_pnl
            FROM read_parquet('{p}')
        ),
        peaks AS (
            SELECT
                cum_pnl,
                MAX(cum_pnl) OVER (
                    ORDER BY rn
                    ROWS UNBOUNDED PRECEDING
                ) AS peak_pnl
            FROM cum
        )
        SELECT MIN(cum_pnl - peak_pnl) AS max_drawdown
        FROM peaks
        """
    ).fetchone()[0]
    max_drawdown = float(dd) if dd else 0.0

    return {
        "total_trades": total,
        "wins": wins,
        "losses": total - wins,
        "win_rate": round(win_rate, 6),
        "avg_return_cents": round(avg_ret, 4),
        "std_return_cents": round(std_ret, 4),
        "total_pnl_cents": round(float(stats["total_pnl"] or 0), 2),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "max_drawdown_cents": round(max_drawdown, 2),
        "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else "inf",
        "avg_win_cents": round(float(stats["avg_win"] or 0), 4),
        "avg_loss_cents": round(float(stats["avg_loss"] or 0), 4),
        "avg_holding_hours": round(float(stats["avg_holding_hours"] or 0), 2),
        "total_fees_cents": round(float(stats["total_fees"] or 0), 2),
        "first_trade": str(stats["first_trade"]),
        "last_trade": str(stats["last_trade"]),
    }
