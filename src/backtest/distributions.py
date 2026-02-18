"""Empirical distribution analysis for backtest return sequences.

Computes moments, percentiles, normality tests, and candidate
distribution fits. Output: distribution_summary.json in the
backtest directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
from scipy import stats


def analyze_distribution(backtest_dir: Path) -> dict:
    """Analyze the empirical return distribution from a backtest.

    Reads returns.parquet, computes statistical properties of
    net_return_cents, writes distribution_summary.json.
    """
    returns_path = backtest_dir / "returns.parquet"

    con = duckdb.connect()
    data = con.execute(
        f"SELECT net_return_cents FROM read_parquet('{returns_path}')"
    ).fetchnumpy()
    con.close()

    r = data["net_return_cents"].astype(np.float64)
    n = len(r)

    # Moments
    mean = float(np.mean(r))
    std = float(np.std(r))
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r))

    # Percentiles
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    percentiles = {
        f"p{p}": round(float(np.percentile(r, p)), 4) for p in pcts
    }

    # Normality tests (subsample for Shapiro-Wilk which has N limit)
    rng = np.random.default_rng(42)
    test_sample = r if n <= 5000 else rng.choice(r, 5000, replace=False)
    shapiro_stat, shapiro_p = stats.shapiro(test_sample)
    jb_stat, jb_p = stats.jarque_bera(r)

    # Fit Student-t distribution (random subsample for speed)
    fit_sample = r if n <= 100_000 else rng.choice(r, 100_000, replace=False)
    t_df, t_loc, t_scale = stats.t.fit(fit_sample)
    t_sample = rng.choice(r, min(n, 10_000), replace=False)
    t_ks_stat, t_ks_p = stats.kstest(t_sample, "t", args=(t_df, t_loc, t_scale))

    summary = {
        "n": n,
        "mean": round(mean, 6),
        "std": round(std, 6),
        "skewness": round(skew, 4),
        "kurtosis": round(kurt, 4),
        "percentiles": percentiles,
        "normality": {
            "shapiro_wilk": {
                "statistic": round(float(shapiro_stat), 6),
                "p_value": float(shapiro_p),
            },
            "jarque_bera": {
                "statistic": round(float(jb_stat), 2),
                "p_value": float(jb_p),
            },
            "is_normal": bool(shapiro_p > 0.05 and jb_p > 0.05),
        },
        "student_t_fit": {
            "df": round(float(t_df), 4),
            "loc": round(float(t_loc), 4),
            "scale": round(float(t_scale), 4),
            "ks_statistic": round(float(t_ks_stat), 6),
            "ks_p_value": float(t_ks_p),
        },
    }

    output_path = backtest_dir / "distribution_summary.json"
    output_path.write_text(json.dumps(summary, indent=2, default=str))
    return summary
