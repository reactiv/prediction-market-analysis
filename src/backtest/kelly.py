"""Kelly criterion and position sizing for prediction market strategies.

Computes standard Kelly fraction, uncertainty-adjusted Kelly (using
Monte Carlo CV of edge), and fractional Kelly variants. Constrains
sizing to keep 95th percentile drawdown under a configurable limit.

Output: kelly_sizing.json in the backtest directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np


def compute_kelly(
    backtest_dir: Path,
    mc_summary: dict | None = None,
    max_drawdown_pct: float = 0.20,
) -> dict:
    """Compute Kelly criterion position sizes.

    Args:
        backtest_dir: Directory containing returns.parquet.
        mc_summary: Monte Carlo summary dict (from run_monte_carlo).
            If None, estimates CV from the sample.
        max_drawdown_pct: Maximum acceptable drawdown as fraction
            of bankroll at the 95th percentile.

    Returns:
        Sizing recommendations dict.
    """
    returns_path = backtest_dir / "returns.parquet"

    con = duckdb.connect()
    data = con.execute(
        f"""
        SELECT
            net_return_cents,
            won::BOOLEAN AS won,
            entry_price,
            side_taken
        FROM read_parquet('{returns_path}')
        """
    ).fetchnumpy()
    con.close()

    returns = data["net_return_cents"].astype(np.float64)
    won = data["won"].astype(bool)
    entry_prices = data["entry_price"].astype(np.float64)
    side_taken = data["side_taken"]

    # Win probability
    p = float(won.mean())
    q = 1.0 - p

    # Average win and loss magnitudes
    avg_win = float(returns[won].mean()) if won.any() else 0.0
    avg_loss = float(np.abs(returns[~won].mean())) if (~won).any() else 0.0

    # Payoff ratio (b = avg_win / avg_loss)
    b = avg_win / avg_loss if avg_loss > 0 else float("inf")

    # Standard Kelly: f* = (p*b - q) / b
    if b > 0 and b != float("inf"):
        kelly_fraction = (p * b - q) / b
    else:
        kelly_fraction = 0.0

    # Compute CV of edge (coefficient of variation)
    if mc_summary:
        edge_mean = mc_summary["avg_return"]["median"]
        edge_std = mc_summary["avg_return"]["std"]
    else:
        # Estimate from sample standard error
        edge_mean = float(returns.mean())
        edge_std = float(returns.std() / np.sqrt(len(returns)))

    cv_edge = abs(edge_std / edge_mean) if edge_mean != 0 else 1.0

    # Uncertainty-adjusted Kelly: f_adj = f* × (1 - CV_edge)
    kelly_adjusted = kelly_fraction * max(1.0 - cv_edge, 0.0)

    # Average cost per contract (investment required)
    # For NO side: cost = 100 - entry_price
    # For YES side: cost = entry_price
    is_no = side_taken == "no"
    costs = np.where(is_no, 100.0 - entry_prices, entry_prices)
    avg_cost = float(costs.mean())

    # Edge per dollar invested
    edge_per_dollar = float(returns.mean() / avg_cost) if avg_cost > 0 else 0.0

    # Drawdown-constrained sizing
    # If MC says 95th pctile drawdown at 1 contract is D,
    # then at fraction f the drawdown scales by f.
    # max_f such that f * |D| / bankroll <= max_drawdown_pct
    drawdown_constrained = None
    if mc_summary:
        dd_95 = abs(mc_summary["max_drawdown"]["p95"])
        if dd_95 > 0:
            # DD is in cents at the path's trade volume.
            # Per-trade allocation: if bankroll = B cents,
            # fraction f means risking f*B per trade.
            # Scale: drawdown scales linearly with position size.
            drawdown_constrained = {
                "dd_95_cents": round(dd_95, 2),
                "trades_per_path": mc_summary["trades_per_path"],
                "note": (
                    "Drawdown at 1 contract per trade. "
                    "Scale position size to keep dd_95 * size < "
                    f"{max_drawdown_pct:.0%} of bankroll."
                ),
            }

    result = {
        "win_probability": round(p, 6),
        "avg_win_cents": round(avg_win, 4),
        "avg_loss_cents": round(avg_loss, 4),
        "payoff_ratio": round(b, 4) if b != float("inf") else "inf",
        "avg_cost_per_contract": round(avg_cost, 2),
        "edge_per_dollar_invested": round(edge_per_dollar, 6),
        "kelly_fraction": round(kelly_fraction, 6),
        "cv_edge": round(cv_edge, 6),
        "kelly_adjusted": round(kelly_adjusted, 6),
        "half_kelly": round(kelly_adjusted / 2, 6),
        "quarter_kelly": round(kelly_adjusted / 4, 6),
    }

    if drawdown_constrained:
        result["drawdown_constraint"] = drawdown_constrained

    output_path = backtest_dir / "kelly_sizing.json"
    output_path.write_text(json.dumps(result, indent=2))

    return result
