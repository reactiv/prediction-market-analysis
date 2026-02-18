"""Return sequence formatting for downstream Monte Carlo consumption.

Produces the exact input format needed by roan-parity.md TODO 1.1-1.5:
a temporally-ordered sequence of per-trade returns suitable for empirical
distribution analysis, bootstrap resampling, and Kelly position sizing.
"""

from __future__ import annotations

from pathlib import Path

import duckdb


def extract_return_sequence(
    con: duckdb.DuckDBPyConnection,
    returns_path: Path,
) -> Path:
    """Extract ordered return sequence from backtest returns.

    Output schema (one row per trade, ordered by timestamp):
        timestamp, net_return_cents, contracts, won, cumulative_pnl
    """
    seq_path = returns_path.parent / "return_sequence.parquet"
    p = str(returns_path)

    con.execute(
        f"""
        COPY (
            SELECT
                timestamp,
                net_return_cents,
                contracts,
                won,
                SUM(net_return_cents * contracts) OVER (
                    ORDER BY timestamp
                    ROWS UNBOUNDED PRECEDING
                ) AS cumulative_pnl
            FROM read_parquet('{p}')
            ORDER BY timestamp
        ) TO '{seq_path}' (FORMAT PARQUET)
        """
    )

    return seq_path
