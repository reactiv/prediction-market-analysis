"""
T4: Implied probability surfaces for Kalshi threshold markets.

Extracts implied probability distributions from families of threshold contracts
(e.g., "BTC above $85,000"), fits parametric distributions (normal, lognormal),
and tracks temporal evolution of implied moments for major families.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import optimize, stats

from src.transforms._base import Transform
from src.transforms._util import get_tmp_dir

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_threshold(ticker: str) -> float | None:
    """Extract the numeric threshold from a Kalshi ticker like KXBTCD-24NOV15-T85000."""
    m = re.search(r"-T(\d+\.?\d*)", ticker)
    if m:
        return float(m.group(1))
    return None


def _pava_decreasing(values: np.ndarray) -> np.ndarray:
    """Pool Adjacent Violators Algorithm enforcing *non-increasing* sequence.

    Given raw CDF values that should be monotonically non-increasing
    (P(X > x) decreases as x increases), merge adjacent blocks that violate
    this constraint by replacing them with their weighted average.
    """
    n = len(values)
    result = values.astype(float).copy()
    # We want non-increasing.  Negate, apply isotonic (non-decreasing), negate back.
    neg = -result
    # Simple PAVA for non-decreasing
    blocks = [[i, i, neg[i], 1.0] for i in range(n)]  # [start, end, sum, count]
    merged: list[list] = [blocks[0]]
    for i in range(1, n):
        cur = blocks[i]
        while merged and cur[2] / cur[3] < merged[-1][2] / merged[-1][3]:
            prev = merged.pop()
            cur[0] = prev[0]
            cur[2] += prev[2]
            cur[3] += prev[3]
        merged.append(cur)
    out = np.empty(n, dtype=float)
    for block in merged:
        avg = block[2] / block[3]
        out[block[0]: block[1] + 1] = -avg
    # Clip to [0, 1]
    return np.clip(out, 0.0, 1.0)


def _isotonic_decreasing(values: np.ndarray) -> np.ndarray:
    """Enforce non-increasing monotonicity, preferring scipy if available."""
    try:
        from scipy.optimize import isotonic_regression

        # scipy's isotonic_regression fits non-decreasing by default.
        # For non-increasing: negate, fit, negate.
        res = isotonic_regression(-values)
        # res may be a namedtuple with .x or just an array
        arr = res.x if hasattr(res, "x") else np.asarray(res)
        return np.clip(-arr, 0.0, 1.0)
    except (ImportError, AttributeError):
        return _pava_decreasing(values)


def _fit_normal_sf(thresholds: np.ndarray, cdf_vals: np.ndarray):
    """Fit a normal survival function to observed P(X > x) values.

    Returns (mean, std, ks_stat, fitted_sf_values) or None on failure.
    """
    if len(thresholds) < 2:
        return None
    x = thresholds.astype(float)
    y = cdf_vals.astype(float)

    # Initial guesses from weighted percentiles
    # CDF_standard = 1 - y  =>  median ~ threshold where y ~ 0.5
    weights = np.diff(np.concatenate([[0], 1 - y]))
    weights = np.clip(weights, 0, None)
    if weights.sum() > 0:
        mu0 = float(np.average(x, weights=weights + 1e-12))
    else:
        mu0 = float(np.mean(x))
    sigma0 = float(np.std(x)) if np.std(x) > 0 else float(np.ptp(x) / 4 + 1e-6)
    if sigma0 <= 0:
        sigma0 = 1.0

    def cost(params):
        mu, sigma = params
        if sigma <= 0:
            return 1e12
        predicted = stats.norm.sf(x, loc=mu, scale=sigma)
        return float(np.sum((predicted - y) ** 2))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = optimize.minimize(
            cost,
            x0=[mu0, sigma0],
            method="Nelder-Mead",
            options={"maxiter": 200, "xatol": 1e-4, "fatol": 1e-6},
        )

    mu_fit, sigma_fit = res.x
    if sigma_fit <= 0:
        return None

    fitted_sf = stats.norm.sf(x, loc=mu_fit, scale=sigma_fit)
    ks = float(np.max(np.abs(fitted_sf - y)))
    return mu_fit, sigma_fit, ks, fitted_sf


def _fit_lognorm_sf(thresholds: np.ndarray, cdf_vals: np.ndarray):
    """Fit a lognormal survival function to observed P(X > x) values.

    Only applicable when all thresholds are positive.
    Returns (s, loc, scale, ks_stat, fitted_sf_values) or None on failure.
    """
    if len(thresholds) < 2:
        return None
    x = thresholds.astype(float)
    y = cdf_vals.astype(float)

    if np.any(x <= 0):
        return None

    # Initial guesses: log-space mean/std
    log_x = np.log(x)
    mu_log0 = float(np.mean(log_x))
    sigma_log0 = float(np.std(log_x))
    if sigma_log0 <= 0:
        sigma_log0 = 1.0

    # lognorm parameterization: s = sigma_log, loc = 0, scale = exp(mu_log)
    s0 = sigma_log0
    scale0 = float(np.exp(mu_log0))

    def cost(params):
        s, scale = params
        if s <= 0 or scale <= 0:
            return 1e12
        predicted = stats.lognorm.sf(x, s=s, loc=0, scale=scale)
        return float(np.sum((predicted - y) ** 2))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = optimize.minimize(
            cost,
            x0=[s0, scale0],
            method="Nelder-Mead",
            options={"maxiter": 200, "xatol": 1e-4, "fatol": 1e-6},
        )

    s_fit, scale_fit = res.x
    if s_fit <= 0 or scale_fit <= 0:
        return None

    fitted_sf = stats.lognorm.sf(x, s=s_fit, loc=0, scale=scale_fit)
    ks = float(np.max(np.abs(fitted_sf - y)))
    return s_fit, 0.0, scale_fit, ks, fitted_sf


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

class T4ImpliedSurfaces(Transform):
    def __init__(self):
        super().__init__(
            name="t4",
            description="Implied probability surfaces (Kalshi)",
            dependencies=[],
        )

    def _make_connection(self) -> duckdb.DuckDBPyConnection:
        """Create a DuckDB connection with tuned settings."""
        con = duckdb.connect()
        tmp_dir = get_tmp_dir()
        con.execute(f"SET memory_limit='20GB'")
        con.execute(f"SET temp_directory='{tmp_dir}'")
        con.execute("SET preserve_insertion_order=false")
        con.execute("SET threads=4")
        return con

    # -----------------------------------------------------------------------
    # run
    # -----------------------------------------------------------------------

    def run(self):
        self.ensure_output_dir()

        # Clean up old per-event surface files if they exist
        old_surfaces_dir = Path(self.output_dir) / "implied_surfaces"
        if old_surfaces_dir.exists():
            for f in old_surfaces_dir.glob("*.parquet"):
                f.unlink()

        # Clean up old per-event evolution files if they exist
        old_evolution_dir = Path(self.output_dir) / "surface_evolution"
        if old_evolution_dir.exists():
            for f in old_evolution_dir.glob("*.parquet"):
                f.unlink()

        n_events = self._build_cross_sectional_surfaces()
        n_evolution = self._build_temporal_evolution()
        summary_rows = self._build_summary()

        self.write_manifest(
            {
                "events_with_surfaces": n_events,
                "families_with_evolution": n_evolution,
                "summary_rows": summary_rows,
            }
        )

    # -----------------------------------------------------------------------
    # Phase 1 — Cross-sectional surfaces
    # -----------------------------------------------------------------------

    def _build_cross_sectional_surfaces(self) -> int:
        surfaces_path = Path(self.output_dir) / "surfaces.parquet"

        # Resume support: skip if already built
        if surfaces_path.exists():
            con = self._make_connection()
            try:
                count = con.execute(
                    f"SELECT COUNT(DISTINCT event_ticker) FROM read_parquet('{surfaces_path}')"
                ).fetchone()[0]
            finally:
                con.close()
            print(f"Phase 1: resuming — {count} surfaces already built")
            return count

        markets_glob = str(Path(self.base_dir) / "data" / "kalshi" / "markets" / "*.parquet")

        con = self._make_connection()
        try:
            # Find all threshold tickers and group by event
            df_all = con.execute(
                f"""
                SELECT
                    event_ticker,
                    ticker,
                    last_price,
                    result,
                    status,
                    regexp_extract(ticker, '-T(\\d+\\.?\\d*)', 1) AS threshold_str
                FROM read_parquet('{markets_glob}')
                WHERE ticker LIKE '%-T%'
                  AND regexp_extract(ticker, '-T(\\d+\\.?\\d*)', 1) != ''
                ORDER BY event_ticker, threshold_str
                """
            ).fetchdf()
        finally:
            con.close()

        if df_all.empty:
            return 0

        df_all["threshold"] = pd.to_numeric(df_all["threshold_str"], errors="coerce")
        df_all = df_all.dropna(subset=["threshold"])
        df_all = df_all.dropna(subset=["last_price"])

        event_groups = df_all.groupby("event_ticker")

        all_surfaces = []
        n_events = 0

        with self.progress("Building cross-sectional surfaces") as _:
            for event_ticker, group in event_groups:
                group = group.drop_duplicates(subset=["threshold"]).sort_values("threshold")

                # Need >= 3 contracts for a meaningful surface
                if len(group) < 3:
                    continue

                # Filter out events where all contracts have the same price
                if group["last_price"].nunique() <= 1:
                    continue

                thresholds = group["threshold"].values
                raw_cdf = group["last_price"].values / 100.0  # P(X > threshold)

                # Enforce monotonicity (non-increasing)
                mono_cdf = _isotonic_decreasing(raw_cdf)

                # Vectorized PDF computation
                pdf = np.concatenate([-np.diff(mono_cdf), [mono_cdf[-1]]])

                # Fit normal
                normal_result = _fit_normal_sf(thresholds, mono_cdf)
                if normal_result is not None:
                    n_mean, n_std, ks_normal, fitted_normal = normal_result
                else:
                    n_mean = n_std = ks_normal = np.nan
                    fitted_normal = np.full(len(thresholds), np.nan)

                # Fit lognormal
                lognorm_result = _fit_lognorm_sf(thresholds, mono_cdf)
                if lognorm_result is not None:
                    ln_s, ln_loc, ln_scale, ks_lognorm, fitted_lognorm = lognorm_result
                else:
                    ln_s = ln_scale = ks_lognorm = np.nan
                    fitted_lognorm = np.full(len(thresholds), np.nan)

                # Residuals
                residual_normal = mono_cdf - fitted_normal
                residual_lognorm = mono_cdf - fitted_lognorm

                surface_df = pd.DataFrame(
                    {
                        "event_ticker": event_ticker,
                        "ticker": group["ticker"].values,
                        "threshold": thresholds,
                        "last_price": group["last_price"].values,
                        "cdf_raw": raw_cdf,
                        "cdf_monotonic": mono_cdf,
                        "pdf": pdf,
                        "fitted_normal_cdf": fitted_normal,
                        "fitted_lognorm_cdf": fitted_lognorm,
                        "normal_mean": n_mean,
                        "normal_std": n_std,
                        "lognorm_s": ln_s,
                        "lognorm_scale": ln_scale,
                        "ks_normal": ks_normal,
                        "ks_lognorm": ks_lognorm,
                        "residual_normal": residual_normal,
                        "residual_lognorm": residual_lognorm,
                    }
                )

                all_surfaces.append(surface_df)
                n_events += 1

        if all_surfaces:
            combined = pd.concat(all_surfaces, ignore_index=True)
            combined.to_parquet(surfaces_path, index=False)

        return n_events

    # -----------------------------------------------------------------------
    # Phase 2 — Temporal evolution for major families
    # -----------------------------------------------------------------------

    _TARGET_FAMILY_PATTERNS = ["KXBTCD", "KXNASDAQ100U", "KXETHD", "KXINXU"]

    def _build_temporal_evolution(self) -> int:
        evolution_path = Path(self.output_dir) / "evolution.parquet"

        markets_glob = str(Path(self.base_dir) / "data" / "kalshi" / "markets" / "*.parquet")
        trades_glob = str(Path(self.base_dir) / "data" / "kalshi" / "trades" / "*.parquet")

        # Check that trades data exists
        trades_path = Path(self.base_dir) / "data" / "kalshi" / "trades"
        if not trades_path.exists() or not list(trades_path.glob("*.parquet")):
            return 0

        # Build the LIKE conditions for target families
        like_clauses = " OR ".join(
            [f"event_ticker LIKE '%{p}%'" for p in self._TARGET_FAMILY_PATTERNS]
        )

        con = self._make_connection()
        try:
            # Get target event tickers and their threshold contracts
            target_markets = con.execute(
                f"""
                SELECT
                    event_ticker,
                    ticker,
                    regexp_extract(ticker, '-T(\\d+\\.?\\d*)', 1) AS threshold_str
                FROM read_parquet('{markets_glob}')
                WHERE ticker LIKE '%-T%'
                  AND regexp_extract(ticker, '-T(\\d+\\.?\\d*)', 1) != ''
                  AND ({like_clauses})
                ORDER BY event_ticker, threshold_str
                """
            ).fetchdf()
        finally:
            con.close()

        if target_markets.empty:
            return 0

        target_markets["threshold"] = pd.to_numeric(
            target_markets["threshold_str"], errors="coerce"
        )
        target_markets = target_markets.dropna(subset=["threshold"])

        # Build ticker -> (event_ticker, threshold) mapping — vectorized
        ticker_info = dict(zip(
            target_markets["ticker"],
            zip(target_markets["event_ticker"], target_markets["threshold"]),
        ))

        # Group tickers by event
        event_tickers_map: dict[str, dict[str, float]] = {}
        for ticker, (evt, thr) in ticker_info.items():
            event_tickers_map.setdefault(evt, {})[ticker] = thr

        # Filter to events with >= 3 contracts
        event_tickers_map = {
            evt: tickers
            for evt, tickers in event_tickers_map.items()
            if len(tickers) >= 3
        }

        if not event_tickers_map:
            return 0

        # Single trade scan: load ALL target family trades at once
        all_target_tickers = []
        for tickers in event_tickers_map.values():
            all_target_tickers.extend(tickers.keys())

        ticker_list_sql = ", ".join([f"'{t}'" for t in all_target_tickers])

        con = self._make_connection()
        try:
            all_trades_df = con.execute(
                f"""
                SELECT
                    ticker,
                    yes_price,
                    created_time
                FROM read_parquet('{trades_glob}')
                WHERE ticker IN ({ticker_list_sql})
                ORDER BY created_time
                """
            ).fetchdf()
        finally:
            con.close()

        if all_trades_df.empty:
            return 0

        # Build reverse lookup: ticker -> event_ticker
        ticker_to_event = {}
        for evt, tickers in event_tickers_map.items():
            for t in tickers:
                ticker_to_event[t] = evt

        # Tag each trade with its event
        all_trades_df["event_ticker"] = all_trades_df["ticker"].map(ticker_to_event)

        all_evolution_dfs = []
        n_families = 0

        with self.progress("Building temporal evolution") as _:
            for event_ticker, ticker_thresholds in event_tickers_map.items():
                # Filter trades for this event
                event_trades = all_trades_df[
                    all_trades_df["event_ticker"] == event_ticker
                ]

                if event_trades.empty or len(event_trades) < 3:
                    continue

                sorted_items = sorted(ticker_thresholds.items(), key=lambda kv: kv[1])
                sorted_thresholds = np.array([thr for _, thr in sorted_items])
                sorted_ticker_keys = [t for t, _ in sorted_items]
                n_contracts = len(sorted_ticker_keys)

                # Require 70% of contracts (min 3) instead of ALL contracts
                min_required = max(3, int(0.7 * n_contracts))

                # Vectorized pivot + ffill approach
                pivot = event_trades.pivot_table(
                    index="created_time",
                    columns="ticker",
                    values="yes_price",
                    aggfunc="last",
                )
                pivot = pivot.sort_index()
                pivot = pivot.ffill()

                # Filter columns to only our sorted tickers (some may be missing)
                available_tickers = [t for t in sorted_ticker_keys if t in pivot.columns]
                if len(available_tickers) < min_required:
                    continue

                pivot = pivot[available_tickers]

                # Count valid (non-NaN) prices per row
                valid_counts = pivot.notna().sum(axis=1)
                pivot = pivot[valid_counts >= min_required]

                if pivot.empty:
                    continue

                # Subsample every 100th row
                pivot = pivot.iloc[::100]

                if pivot.empty:
                    continue

                # Map available tickers to their thresholds, sorted by threshold
                avail_thresholds = np.array(
                    [ticker_thresholds[t] for t in available_tickers]
                )
                avail_sorted_idx = np.argsort(avail_thresholds)
                avail_thresholds_sorted = avail_thresholds[avail_sorted_idx]
                avail_tickers_sorted = [available_tickers[i] for i in avail_sorted_idx]

                evolution_rows = []

                for trade_time, row in pivot.iterrows():
                    cdf_vals = row[avail_tickers_sorted].values / 100.0

                    if np.any(np.isnan(cdf_vals)):
                        # Use only non-NaN values
                        mask = ~np.isnan(cdf_vals)
                        if mask.sum() < min_required:
                            continue
                        valid_thresholds = avail_thresholds_sorted[mask]
                        cdf_vals = cdf_vals[mask]
                    else:
                        valid_thresholds = avail_thresholds_sorted

                    # Enforce monotonicity
                    mono_cdf = _isotonic_decreasing(cdf_vals)

                    # Skip lognormal in evolution — only fit normal
                    normal_result = _fit_normal_sf(valid_thresholds, mono_cdf)
                    if normal_result is None:
                        continue

                    mu, sigma, _, _ = normal_result

                    # Vectorized PDF computation
                    pdf = np.concatenate([-np.diff(mono_cdf), [mono_cdf[-1]]])

                    pdf_sum = pdf.sum()
                    if pdf_sum > 0:
                        pdf_norm = pdf / pdf_sum
                        emp_mean = np.sum(valid_thresholds * pdf_norm)
                        emp_var = np.sum(
                            (valid_thresholds - emp_mean) ** 2 * pdf_norm
                        )
                        emp_std = np.sqrt(emp_var) if emp_var > 0 else 1e-12
                        emp_skew = float(
                            np.sum(
                                ((valid_thresholds - emp_mean) / emp_std) ** 3
                                * pdf_norm
                            )
                        )
                        emp_kurt = float(
                            np.sum(
                                ((valid_thresholds - emp_mean) / emp_std) ** 4
                                * pdf_norm
                            )
                            - 3.0
                        )
                    else:
                        emp_skew = np.nan
                        emp_kurt = np.nan

                    evolution_rows.append(
                        {
                            "event_ticker": event_ticker,
                            "trade_time": trade_time,
                            "implied_mean": mu,
                            "implied_std": sigma,
                            "implied_skew": emp_skew,
                            "implied_kurtosis": emp_kurt,
                            "n_active_contracts": len(valid_thresholds),
                        }
                    )

                if evolution_rows:
                    all_evolution_dfs.append(pd.DataFrame(evolution_rows))
                    n_families += 1

        if all_evolution_dfs:
            combined = pd.concat(all_evolution_dfs, ignore_index=True)
            combined.to_parquet(evolution_path, index=False)

        return n_families

    # -----------------------------------------------------------------------
    # Phase 3 — Summary (single DuckDB query)
    # -----------------------------------------------------------------------

    def _build_summary(self) -> int:
        surfaces_path = Path(self.output_dir) / "surfaces.parquet"

        if not surfaces_path.exists():
            return 0

        markets_glob = str(
            Path(self.base_dir) / "data" / "kalshi" / "markets" / "*.parquet"
        )

        con = self._make_connection()
        try:
            summary_df = con.execute(
                f"""
                WITH surface_stats AS (
                    SELECT
                        event_ticker,
                        COUNT(*) AS n_contracts,
                        MIN(threshold) AS threshold_min,
                        MAX(threshold) AS threshold_max,
                        FIRST(ks_normal) AS ks_normal,
                        FIRST(ks_lognorm) AS ks_lognorm,
                        FIRST(normal_mean) AS implied_mean,
                        FIRST(normal_std) AS implied_std
                    FROM read_parquet('{surfaces_path}')
                    GROUP BY event_ticker
                ),
                resolutions AS (
                    SELECT
                        event_ticker,
                        result,
                        CAST(regexp_extract(ticker, '-T(\\d+\\.?\\d*)', 1) AS DOUBLE) AS threshold
                    FROM read_parquet('{markets_glob}')
                    WHERE ticker LIKE '%-T%'
                      AND regexp_extract(ticker, '-T(\\d+\\.?\\d*)', 1) != ''
                      AND result IS NOT NULL
                      AND result != ''
                ),
                resolved_bounds AS (
                    SELECT
                        event_ticker,
                        MAX(CASE WHEN result = 'yes' THEN threshold END) AS max_yes_threshold,
                        MIN(CASE WHEN result = 'no' THEN threshold END) AS min_no_threshold
                    FROM resolutions
                    GROUP BY event_ticker
                )
                SELECT
                    s.event_ticker,
                    s.n_contracts,
                    CONCAT(s.threshold_min::VARCHAR, '-', s.threshold_max::VARCHAR) AS threshold_range,
                    CASE
                        WHEN s.ks_normal IS NULL AND s.ks_lognorm IS NULL THEN 'none'
                        WHEN s.ks_lognorm IS NULL THEN 'normal'
                        WHEN s.ks_normal IS NULL THEN 'lognormal'
                        WHEN s.ks_normal <= s.ks_lognorm THEN 'normal'
                        ELSE 'lognormal'
                    END AS best_fit_model,
                    CASE
                        WHEN s.ks_normal IS NULL AND s.ks_lognorm IS NULL THEN NULL
                        WHEN s.ks_lognorm IS NULL THEN s.ks_normal
                        WHEN s.ks_normal IS NULL THEN s.ks_lognorm
                        WHEN s.ks_normal <= s.ks_lognorm THEN s.ks_normal
                        ELSE s.ks_lognorm
                    END AS best_ks,
                    s.implied_mean,
                    s.implied_std,
                    r.max_yes_threshold IS NOT NULL
                        OR r.min_no_threshold IS NOT NULL AS is_resolved,
                    CASE
                        WHEN r.max_yes_threshold IS NOT NULL
                            AND r.min_no_threshold IS NOT NULL
                            THEN (r.max_yes_threshold + r.min_no_threshold) / 2.0
                        WHEN r.max_yes_threshold IS NOT NULL THEN r.max_yes_threshold
                        WHEN r.min_no_threshold IS NOT NULL THEN r.min_no_threshold
                        ELSE NULL
                    END AS actual_outcome
                FROM surface_stats s
                LEFT JOIN resolved_bounds r ON s.event_ticker = r.event_ticker
                ORDER BY s.event_ticker
                """
            ).fetchdf()
        finally:
            con.close()

        if summary_df.empty:
            return 0

        summary_df.to_parquet(
            Path(self.output_dir) / "summary_stats.parquet", index=False
        )

        return len(summary_df)
