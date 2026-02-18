"""Monte Carlo simulation via bootstrap resampling of backtest returns.

Generates N equity curve paths by sampling trades with replacement
from the empirical return distribution. For each path computes:
final PnL, max drawdown, Sharpe ratio, win rate.

Output: mc_paths.parquet + mc_summary.json in the backtest directory.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb
import numpy as np


def run_monte_carlo(
    backtest_dir: Path,
    n_paths: int = 10_000,
    trades_per_path: int | None = None,
    seed: int = 42,
) -> dict:
    """Bootstrap resample returns to generate equity curve distribution.

    Args:
        backtest_dir: Directory containing returns.parquet.
        n_paths: Number of simulated paths.
        trades_per_path: Trades per path. Defaults to min(n_trades, 100_000).
        seed: Random seed for reproducibility.

    Returns:
        Summary dict with percentile statistics.
    """
    returns_path = backtest_dir / "returns.parquet"

    con = duckdb.connect()
    data = con.execute(
        f"""
        SELECT net_return_cents, contracts, won::INTEGER AS won
        FROM read_parquet('{returns_path}')
        """
    ).fetchnumpy()
    con.close()

    returns = data["net_return_cents"].astype(np.float64)
    contracts = data["contracts"].astype(np.float64)
    won = data["won"].astype(np.int32)

    n_trades = len(returns)
    path_len = trades_per_path or min(n_trades, 100_000)

    rng = np.random.default_rng(seed)

    # Pre-allocate result arrays
    final_pnl = np.empty(n_paths)
    max_drawdown = np.empty(n_paths)
    path_sharpe = np.empty(n_paths)
    path_win_rate = np.empty(n_paths)
    path_avg_return = np.empty(n_paths)

    start = time.time()

    for i in range(n_paths):
        idx = rng.integers(0, n_trades, size=path_len)
        sampled_returns = returns[idx]
        sampled_contracts = contracts[idx]
        sampled_won = won[idx]

        trade_pnl = sampled_returns * sampled_contracts
        cum_pnl = np.cumsum(trade_pnl)

        # Max drawdown: largest peak-to-trough decline
        running_max = np.maximum.accumulate(cum_pnl)
        drawdowns = cum_pnl - running_max
        max_drawdown[i] = drawdowns.min()

        final_pnl[i] = cum_pnl[-1]
        std = sampled_returns.std()
        path_sharpe[i] = sampled_returns.mean() / std if std > 0 else 0.0
        path_win_rate[i] = sampled_won.mean()
        path_avg_return[i] = sampled_returns.mean()

    elapsed = time.time() - start

    # Write per-path results to parquet
    paths_path = backtest_dir / "mc_paths.parquet"
    con = duckdb.connect()
    con.execute(
        f"""
        COPY (
            SELECT
                unnest(range({n_paths})) AS path_id,
                unnest($final_pnl) AS final_pnl,
                unnest($max_drawdown) AS max_drawdown,
                unnest($sharpe) AS sharpe,
                unnest($win_rate) AS win_rate,
                unnest($avg_return) AS avg_return
        ) TO '{paths_path}' (FORMAT PARQUET)
        """,
        {
            "final_pnl": final_pnl.tolist(),
            "max_drawdown": max_drawdown.tolist(),
            "sharpe": path_sharpe.tolist(),
            "win_rate": path_win_rate.tolist(),
            "avg_return": path_avg_return.tolist(),
        },
    )
    con.close()

    # Compute summary statistics
    summary = {
        "n_paths": n_paths,
        "trades_per_path": path_len,
        "source_trades": n_trades,
        "elapsed_seconds": round(elapsed, 1),
        "pnl": {
            "mean": round(float(np.mean(final_pnl)), 2),
            "median": round(float(np.median(final_pnl)), 2),
            "std": round(float(np.std(final_pnl)), 2),
            "p5": round(float(np.percentile(final_pnl, 5)), 2),
            "p25": round(float(np.percentile(final_pnl, 25)), 2),
            "p75": round(float(np.percentile(final_pnl, 75)), 2),
            "p95": round(float(np.percentile(final_pnl, 95)), 2),
        },
        "max_drawdown": {
            "median": round(float(np.median(max_drawdown)), 2),
            "p75": round(float(np.percentile(max_drawdown, 25)), 2),
            "p90": round(float(np.percentile(max_drawdown, 10)), 2),
            "p95": round(float(np.percentile(max_drawdown, 5)), 2),
            "p99": round(float(np.percentile(max_drawdown, 1)), 2),
        },
        "sharpe": {
            "median": round(float(np.median(path_sharpe)), 6),
            "p5": round(float(np.percentile(path_sharpe, 5)), 6),
            "p95": round(float(np.percentile(path_sharpe, 95)), 6),
        },
        "win_rate": {
            "median": round(float(np.median(path_win_rate)), 6),
            "std": round(float(np.std(path_win_rate)), 6),
        },
        "avg_return": {
            "median": round(float(np.median(path_avg_return)), 6),
            "std": round(float(np.std(path_avg_return)), 6),
        },
    }

    output_path = backtest_dir / "mc_summary.json"
    output_path.write_text(json.dumps(summary, indent=2))

    return summary
