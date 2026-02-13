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
            options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-12},
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
            options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-12},
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

    # -----------------------------------------------------------------------
    # run
    # -----------------------------------------------------------------------

    def run(self):
        self.ensure_output_dir()

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
        surfaces_dir = Path(self.output_dir) / "implied_surfaces"
        surfaces_dir.mkdir(parents=True, exist_ok=True)

        markets_glob = str(Path(self.base_dir) / "data" / "kalshi" / "markets" / "*.parquet")

        con = duckdb.connect()
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

                # Compute PDF by differencing adjacent CDF points
                pdf = np.zeros(len(mono_cdf))
                for i in range(len(mono_cdf) - 1):
                    pdf[i] = mono_cdf[i] - mono_cdf[i + 1]
                # Last bucket gets remaining probability mass
                pdf[-1] = mono_cdf[-1]

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

                out_path = surfaces_dir / f"{event_ticker}.parquet"
                surface_df.to_parquet(out_path, index=False)
                n_events += 1

        return n_events

    # -----------------------------------------------------------------------
    # Phase 2 — Temporal evolution for major families
    # -----------------------------------------------------------------------

    _TARGET_FAMILY_PATTERNS = ["KXBTCD", "KXNASDAQ100U", "KXETHD", "KXINXU"]

    def _build_temporal_evolution(self) -> int:
        evolution_dir = Path(self.output_dir) / "surface_evolution"
        evolution_dir.mkdir(parents=True, exist_ok=True)

        markets_glob = str(Path(self.base_dir) / "data" / "kalshi" / "markets" / "*.parquet")
        trades_glob = str(Path(self.base_dir) / "data" / "kalshi" / "trades" / "*.parquet")

        # Check that trades data exists
        trades_path = Path(self.base_dir) / "data" / "kalshi" / "trades"
        if not trades_path.exists() or not list(trades_path.glob("*.parquet")):
            return 0

        con = duckdb.connect()

        # Build the LIKE conditions for target families
        like_clauses = " OR ".join(
            [f"event_ticker LIKE '%{p}%'" for p in self._TARGET_FAMILY_PATTERNS]
        )

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

        # Build ticker -> (event_ticker, threshold) mapping
        ticker_info = {}
        for _, row in target_markets.iterrows():
            ticker_info[row["ticker"]] = (row["event_ticker"], row["threshold"])

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

        n_families = 0

        with self.progress("Building temporal evolution") as _:
            # Process each family sequentially for memory safety
            for event_ticker, ticker_thresholds in event_tickers_map.items():
                family_tickers = list(ticker_thresholds.keys())

                # Build SQL IN clause
                ticker_list_sql = ", ".join([f"'{t}'" for t in family_tickers])

                con = duckdb.connect()
                try:
                    trades_df = con.execute(
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

                if trades_df.empty or len(trades_df) < 3:
                    continue

                # Forward-fill approach: maintain current price for each ticker
                current_prices: dict[str, float] = {}
                evolution_rows = []

                # Sort thresholds for this family
                sorted_tickers = sorted(
                    ticker_thresholds.items(), key=lambda kv: kv[1]
                )
                sorted_thresholds = np.array([thr for _, thr in sorted_tickers])
                sorted_ticker_keys = [t for t, _ in sorted_tickers]
                n_contracts = len(sorted_ticker_keys)

                # Process trades chronologically, subsampling every 100th
                # eligible trade to avoid running scipy fits on millions of rows.
                eligible_count = 0
                for _, trade in trades_df.iterrows():
                    ticker = trade["ticker"]
                    price = trade["yes_price"]
                    trade_time = trade["created_time"]

                    if pd.isna(price):
                        continue

                    current_prices[ticker] = float(price)

                    # Only compute surface when we have prices for all contracts
                    if len(current_prices) < n_contracts:
                        continue

                    eligible_count += 1
                    if eligible_count % 100 != 0:
                        continue

                    # Build current CDF
                    cdf_vals = np.array(
                        [current_prices.get(t, np.nan) for t in sorted_ticker_keys]
                    ) / 100.0

                    if np.any(np.isnan(cdf_vals)):
                        continue

                    # Enforce monotonicity
                    mono_cdf = _isotonic_decreasing(cdf_vals)

                    # Fit normal to get implied mean/std
                    normal_result = _fit_normal_sf(sorted_thresholds, mono_cdf)
                    if normal_result is None:
                        continue

                    mu, sigma, _, _ = normal_result

                    # Compute implied skew and kurtosis from the discrete distribution
                    # Use PDF from differenced CDF
                    pdf = np.zeros(len(mono_cdf))
                    for i in range(len(mono_cdf) - 1):
                        pdf[i] = mono_cdf[i] - mono_cdf[i + 1]
                    pdf[-1] = mono_cdf[-1]

                    pdf_sum = pdf.sum()
                    if pdf_sum > 0:
                        pdf_norm = pdf / pdf_sum
                        emp_mean = np.sum(sorted_thresholds * pdf_norm)
                        emp_var = np.sum(
                            (sorted_thresholds - emp_mean) ** 2 * pdf_norm
                        )
                        emp_std = np.sqrt(emp_var) if emp_var > 0 else 1e-12
                        emp_skew = float(
                            np.sum(
                                ((sorted_thresholds - emp_mean) / emp_std) ** 3
                                * pdf_norm
                            )
                        )
                        emp_kurt = float(
                            np.sum(
                                ((sorted_thresholds - emp_mean) / emp_std) ** 4
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
                            "n_active_contracts": n_contracts,
                        }
                    )

                if not evolution_rows:
                    continue

                evo_df = pd.DataFrame(evolution_rows)
                out_path = evolution_dir / f"{event_ticker}.parquet"
                evo_df.to_parquet(out_path, index=False)
                n_families += 1

        return n_families

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    def _build_summary(self) -> int:
        surfaces_dir = Path(self.output_dir) / "implied_surfaces"

        if not surfaces_dir.exists():
            return 0

        surface_files = list(surfaces_dir.glob("*.parquet"))
        if not surface_files:
            return 0

        summary_rows = []

        with self.progress("Building summary statistics") as _:
            for fpath in surface_files:
                df = pd.read_parquet(fpath)
                if df.empty:
                    continue

                event_ticker = df["event_ticker"].iloc[0]
                n_contracts = len(df)
                threshold_min = float(df["threshold"].min())
                threshold_max = float(df["threshold"].max())
                threshold_range = f"{threshold_min}-{threshold_max}"

                ks_normal = df["ks_normal"].iloc[0]
                ks_lognorm = df["ks_lognorm"].iloc[0]

                # Determine best fit model
                if pd.isna(ks_normal) and pd.isna(ks_lognorm):
                    best_fit = "none"
                    best_ks = np.nan
                elif pd.isna(ks_lognorm):
                    best_fit = "normal"
                    best_ks = ks_normal
                elif pd.isna(ks_normal):
                    best_fit = "lognormal"
                    best_ks = ks_lognorm
                else:
                    if ks_normal <= ks_lognorm:
                        best_fit = "normal"
                        best_ks = ks_normal
                    else:
                        best_fit = "lognormal"
                        best_ks = ks_lognorm

                implied_mean = df["normal_mean"].iloc[0]
                implied_std = df["normal_std"].iloc[0]

                # Check if resolved — look for actual outcome via markets data
                is_resolved = False
                actual_outcome = np.nan
                markets_glob = str(
                    Path(self.base_dir)
                    / "data"
                    / "kalshi"
                    / "markets"
                    / "*.parquet"
                )
                con = duckdb.connect()
                try:
                    resolved_df = con.execute(
                        f"""
                        SELECT
                            ticker,
                            result,
                            regexp_extract(ticker, '-T(\\d+\\.?\\d*)', 1) AS threshold_str
                        FROM read_parquet('{markets_glob}')
                        WHERE event_ticker = '{event_ticker}'
                          AND result IS NOT NULL
                          AND result != ''
                        ORDER BY CAST(regexp_extract(ticker, '-T(\\d+\\.?\\d*)', 1) AS DOUBLE)
                        """
                    ).fetchdf()
                except Exception:
                    resolved_df = pd.DataFrame()
                finally:
                    con.close()

                if not resolved_df.empty:
                    is_resolved = True
                    # The actual outcome lies between the highest YES-resolved
                    # threshold and the lowest NO-resolved threshold.
                    resolved_df["threshold"] = pd.to_numeric(
                        resolved_df["threshold_str"], errors="coerce"
                    )
                    yes_resolved = resolved_df[resolved_df["result"] == "yes"]
                    no_resolved = resolved_df[resolved_df["result"] == "no"]

                    if not yes_resolved.empty and not no_resolved.empty:
                        # Actual value is between max YES threshold and min NO threshold
                        upper = float(no_resolved["threshold"].min())
                        lower = float(yes_resolved["threshold"].max())
                        actual_outcome = (lower + upper) / 2.0
                    elif not yes_resolved.empty:
                        actual_outcome = float(yes_resolved["threshold"].max())
                    elif not no_resolved.empty:
                        actual_outcome = float(no_resolved["threshold"].min())

                summary_rows.append(
                    {
                        "event_ticker": event_ticker,
                        "n_contracts": n_contracts,
                        "threshold_range": threshold_range,
                        "best_fit_model": best_fit,
                        "best_ks": best_ks,
                        "implied_mean": implied_mean,
                        "implied_std": implied_std,
                        "is_resolved": is_resolved,
                        "actual_outcome": actual_outcome,
                    }
                )

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_parquet(
                Path(self.output_dir) / "summary_stats.parquet", index=False
            )

        return len(summary_rows)
