from __future__ import annotations

import itertools
import json
import logging
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from scipy.signal import correlate
from statsmodels.tsa.stattools import grangercausalitytests

from src.transforms._base import Transform
from src.transforms._util import get_tmp_dir

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BAR_RESOLUTION_MIN = 5
LAGS_BARS = 12  # +/- 12 bars = +/- 60 min
ROLLING_WINDOW_BARS = 288  # 24 hours at 5-min resolution
MIN_OVERLAP_PCT = 0.50  # discard pairs with >50% empty bars on either side
GRANGER_MAX_LAG = 12  # 60 min at 5-min bars
TOP_FAMILIES = 10  # max intra-family groups to consider
TOP_CROSS_PER_PLATFORM = 5  # top N markets per platform for cross-platform pairs


class T5LeadLag(Transform):
    def __init__(self) -> None:
        super().__init__(
            name="t5",
            description="Cross-market lead-lag network",
            dependencies=["t1b"],
        )

    # -----------------------------------------------------------------------
    # DuckDB connection (tuned, matching T4 pattern)
    # -----------------------------------------------------------------------
    def _make_connection(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect()
        tmp_dir = get_tmp_dir()
        con.execute("SET memory_limit='20GB'")
        con.execute(f"SET temp_directory='{tmp_dir}'")
        con.execute("SET preserve_insertion_order=false")
        con.execute("SET threads=4")
        return con

    # -----------------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------------
    def run(self) -> None:
        self.ensure_output_dir()

        # Resume: if output + manifest already exist, skip entirely
        summary_path = self.output_dir / "lead_lag_summary.parquet"
        manifest_path = self.output_dir / "manifest.json"
        if summary_path.exists() and manifest_path.exists():
            try:
                meta = json.loads(manifest_path.read_text())
                if meta.get("valid_pairs", 0) > 0:
                    logger.info(
                        "Resuming — lead_lag_summary.parquet already exists with %d valid pairs",
                        meta["valid_pairs"],
                    )
                    return
            except (json.JSONDecodeError, KeyError):
                pass

        pairs = self._curate_pairs()
        logger.info("Curated %d candidate pairs", len(pairs))

        if not pairs:
            logger.warning("No candidate pairs found")
            self.write_manifest({"total_pairs_considered": 0, "valid_pairs": 0})
            return

        # Preload all bar data in 2 queries (one per platform)
        bar_cache = self._preload_bars(pairs)
        logger.info("Preloaded bars for %d markets", len(bar_cache))

        results: list[dict[str, Any]] = []
        with self.progress("Computing lead-lag pairs") as _:
            for i, pair in enumerate(pairs):
                _, market_id_a, _, market_id_b, _ = pair
                logger.info(
                    "Pair %d/%d: %s vs %s", i + 1, len(pairs), market_id_a, market_id_b
                )
                result = self._analyze_pair(pair, bar_cache)
                if result is not None:
                    results.append(result)

        if results:
            df = pd.DataFrame(results)
            df.to_parquet(summary_path, index=False)
            logger.info(
                "Wrote lead_lag_summary.parquet with %d valid pairs", len(results)
            )
        else:
            logger.warning("No valid pairs produced results")

        self.write_manifest(
            {
                "total_pairs_considered": len(pairs),
                "valid_pairs": len(results),
            }
        )

    # -----------------------------------------------------------------------
    # Step 1: Curate market pairs
    # -----------------------------------------------------------------------
    def _curate_pairs(self) -> list[tuple[str, str, str, str, str]]:
        """Return ~20-60 pairs as (platform_a, market_id_a, platform_b, market_id_b, pair_type)."""
        qm_path = self.base_dir / "data" / "transforms" / "t1b" / "qualifying_markets.parquet"

        con = self._make_connection()
        try:
            qm = con.execute(
                """
                SELECT platform, market_id, total_trades
                FROM read_parquet(?)
                ORDER BY total_trades DESC
                """,
                [str(qm_path)],
            ).fetchdf()
        finally:
            con.close()

        pairs: list[tuple[str, str, str, str, str]] = []

        # ----- Intra-family pairs (Kalshi threshold families) -----
        kalshi = qm[qm["platform"] == "kalshi"].copy()
        if not kalshi.empty:
            # Extract family prefix from market_id: e.g. KXBTCD-25MAR0500-T87249.99 → KXBTCD
            kalshi["family"] = kalshi["market_id"].str.split("-").str[0]

            family_counts = (
                kalshi.groupby("family")["market_id"]
                .count()
                .reset_index(name="n_members")
            )
            family_counts = family_counts[family_counts["n_members"] >= 2].sort_values(
                "n_members", ascending=False
            )

            families_used = 0
            for _, fam_row in family_counts.iterrows():
                if families_used >= TOP_FAMILIES:
                    break
                family_name = fam_row["family"]
                members = (
                    kalshi[kalshi["family"] == family_name]
                    .sort_values("total_trades", ascending=False)
                    .head(4)
                )
                member_ids = members["market_id"].tolist()
                for id_a, id_b in itertools.combinations(member_ids, 2):
                    pairs.append(("kalshi", id_a, "kalshi", id_b, "intra_family"))
                families_used += 1

        # ----- Cross-platform pairs -----
        kalshi_top = qm[qm["platform"] == "kalshi"].head(TOP_CROSS_PER_PLATFORM)
        poly_top = qm[qm["platform"] == "polymarket"].head(TOP_CROSS_PER_PLATFORM)

        if not kalshi_top.empty and not poly_top.empty:
            for _, k_row in kalshi_top.iterrows():
                for _, p_row in poly_top.iterrows():
                    pairs.append(
                        (
                            "kalshi",
                            k_row["market_id"],
                            "polymarket",
                            p_row["market_id"],
                            "cross_platform",
                        )
                    )

        # Fallback: intra-platform pairs if nothing else
        if len(pairs) == 0:
            for platform_name in ["kalshi", "polymarket"]:
                plat = qm[qm["platform"] == platform_name].head(6)
                ids = plat["market_id"].tolist()
                for id_a, id_b in itertools.combinations(ids, 2):
                    pairs.append(
                        (platform_name, id_a, platform_name, id_b, "intra_platform")
                    )

        logger.info(
            "Curated %d intra-family + %d cross-platform pairs",
            sum(1 for p in pairs if p[4] == "intra_family"),
            sum(1 for p in pairs if p[4] == "cross_platform"),
        )
        return pairs

    # -----------------------------------------------------------------------
    # Preload bars: 2 DuckDB scans instead of ~100
    # -----------------------------------------------------------------------
    def _preload_bars(
        self, pairs: list[tuple[str, str, str, str, str]]
    ) -> dict[tuple[str, str], pd.DataFrame]:
        """Load all needed bars in one query per platform. Returns {(platform, market_id): df}."""
        # Collect unique (platform, market_id) tuples
        needed: dict[str, set[str]] = {}
        for platform_a, market_id_a, platform_b, market_id_b, _ in pairs:
            needed.setdefault(platform_a, set()).add(market_id_a)
            needed.setdefault(platform_b, set()).add(market_id_b)

        cache: dict[tuple[str, str], pd.DataFrame] = {}
        con = self._make_connection()
        try:
            for platform, market_ids in needed.items():
                bars_dir = (
                    self.base_dir
                    / "data"
                    / "transforms"
                    / "t1b"
                    / platform
                    / "bars_5min"
                )
                if not bars_dir.exists():
                    logger.warning("Bars directory missing: %s", bars_dir)
                    continue

                glob_path = str(bars_dir / "**" / "*.parquet")
                id_list = list(market_ids)

                # Build parameterized IN clause
                placeholders = ", ".join(["?"] * len(id_list))
                query = f"""
                    SELECT ticker, bar_start, close, bar_return
                    FROM read_parquet('{glob_path}')
                    WHERE ticker IN ({placeholders})
                    ORDER BY ticker, bar_start
                """
                try:
                    df = con.execute(query, id_list).fetchdf()
                except Exception:
                    logger.warning(
                        "Failed to preload bars for %s", platform, exc_info=True
                    )
                    continue

                if df.empty:
                    continue

                df["bar_start"] = pd.to_datetime(df["bar_start"], utc=True).dt.tz_localize(None)

                # Split into per-market DataFrames
                for ticker, group in df.groupby("ticker"):
                    cache[(platform, ticker)] = group.reset_index(drop=True)

                logger.info(
                    "Preloaded %d markets from %s (%d total bars)",
                    len(df["ticker"].unique()),
                    platform,
                    len(df),
                )
        finally:
            con.close()

        return cache

    # -----------------------------------------------------------------------
    # Analyze a single pair
    # -----------------------------------------------------------------------
    def _analyze_pair(
        self,
        pair: tuple[str, str, str, str, str],
        bar_cache: dict[tuple[str, str], pd.DataFrame],
    ) -> dict[str, Any] | None:
        platform_a, market_id_a, platform_b, market_id_b, pair_type = pair

        # Look up bars from cache
        bars_a = bar_cache.get((platform_a, market_id_a))
        bars_b = bar_cache.get((platform_b, market_id_b))

        if bars_a is None or bars_b is None:
            return None
        if bars_a.empty or bars_b.empty:
            return None

        # Full outer join on bar_start, using bar_return directly
        merged = pd.merge(
            bars_a[["bar_start", "close", "bar_return"]].rename(
                columns={"close": "close_a", "bar_return": "ret_a"}
            ),
            bars_b[["bar_start", "close", "bar_return"]].rename(
                columns={"close": "close_b", "bar_return": "ret_b"}
            ),
            on="bar_start",
            how="outer",
        ).sort_values("bar_start")

        n_total = len(merged)
        n_valid_a = merged["close_a"].notna().sum()
        n_valid_b = merged["close_b"].notna().sum()

        pct_a = n_valid_a / n_total if n_total > 0 else 0
        pct_b = n_valid_b / n_total if n_total > 0 else 0

        if pct_a < MIN_OVERLAP_PCT or pct_b < MIN_OVERLAP_PCT:
            return None

        # Forward-fill close prices, then forward-fill returns, drop remaining NaNs
        merged = merged.sort_values("bar_start")
        merged["close_a"] = merged["close_a"].ffill()
        merged["close_b"] = merged["close_b"].ffill()
        merged["ret_a"] = merged["ret_a"].fillna(0.0)
        merged["ret_b"] = merged["ret_b"].fillna(0.0)
        merged = merged.dropna(subset=["close_a", "close_b"])

        if len(merged) < ROLLING_WINDOW_BARS:
            return None

        ret_a = merged["ret_a"].values.astype(float)
        ret_b = merged["ret_b"].values.astype(float)
        n_aligned = len(merged)
        pct_overlap = min(pct_a, pct_b)

        # --- Cross-correlation ---
        peak_lag_bars, peak_correlation = self._cross_correlation(ret_a, ret_b)
        peak_lag_minutes = peak_lag_bars * BAR_RESOLUTION_MIN

        # --- Rolling cross-correlation for lag stability ---
        median_lag, lag_std, lag_consistency_pct = self._rolling_cross_correlation(
            ret_a, ret_b
        )

        # --- Directional Granger causality ---
        gc_results = self._granger_causality(ret_a, ret_b, peak_lag_bars)

        return {
            "platform_a": platform_a,
            "market_id_a": market_id_a,
            "platform_b": platform_b,
            "market_id_b": market_id_b,
            "pair_type": pair_type,
            "peak_lag_bars": peak_lag_bars,
            "peak_lag_minutes": peak_lag_minutes,
            "peak_correlation": peak_correlation,
            "granger_a_to_b_f": gc_results["a_to_b_f"],
            "granger_a_to_b_p": gc_results["a_to_b_p"],
            "granger_b_to_a_f": gc_results["b_to_a_f"],
            "granger_b_to_a_p": gc_results["b_to_a_p"],
            "median_lag": median_lag,
            "lag_std": lag_std,
            "lag_consistency_pct": lag_consistency_pct,
            "n_aligned_bars": n_aligned,
            "pct_overlap": round(pct_overlap, 4),
        }

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    @staticmethod
    def _cross_correlation(
        ret_a: np.ndarray, ret_b: np.ndarray
    ) -> tuple[int, float]:
        """Full cross-correlation, return (peak_lag_bars, peak_correlation).

        Positive lag means a leads b (a's past predicts b's future).
        We restrict to lags in [-LAGS_BARS, +LAGS_BARS].
        """
        a = ret_a - ret_a.mean()
        b = ret_b - ret_b.mean()
        norm = np.sqrt(np.sum(a**2) * np.sum(b**2))

        if norm < 1e-12:
            return 0, 0.0

        full_xcorr = correlate(a, b, mode="full") / norm
        n = len(ret_a)
        center = n - 1

        lo = max(center - LAGS_BARS, 0)
        hi = min(center + LAGS_BARS + 1, len(full_xcorr))
        window = full_xcorr[lo:hi]
        lags = np.arange(lo - center, hi - center)

        best_idx = int(np.argmax(np.abs(window)))
        peak_lag = int(lags[best_idx])
        peak_corr = float(window[best_idx])

        return peak_lag, peak_corr

    @staticmethod
    def _rolling_cross_correlation(
        ret_a: np.ndarray, ret_b: np.ndarray
    ) -> tuple[float, float, float]:
        """Rolling cross-correlation with 24h window, 50% overlap stride.

        Returns (median_lag, lag_std, lag_consistency_pct).
        """
        n = len(ret_a)
        if n < ROLLING_WINDOW_BARS:
            return 0.0, 0.0, 0.0

        optimal_lags: list[int] = []

        # 50% overlap: stride = ROLLING_WINDOW_BARS // 2 (was // 4)
        for start in range(0, n - ROLLING_WINDOW_BARS + 1, ROLLING_WINDOW_BARS // 2):
            end = start + ROLLING_WINDOW_BARS
            chunk_a = ret_a[start:end]
            chunk_b = ret_b[start:end]

            a = chunk_a - chunk_a.mean()
            b = chunk_b - chunk_b.mean()
            norm = np.sqrt(np.sum(a**2) * np.sum(b**2))
            if norm < 1e-12:
                continue

            xcorr = correlate(a, b, mode="full") / norm
            center = ROLLING_WINDOW_BARS - 1
            lo = max(center - LAGS_BARS, 0)
            hi = min(center + LAGS_BARS + 1, len(xcorr))
            window = xcorr[lo:hi]
            lags = np.arange(lo - center, hi - center)

            best_idx = int(np.argmax(np.abs(window)))
            optimal_lags.append(int(lags[best_idx]))

        if not optimal_lags:
            return 0.0, 0.0, 0.0

        lags_arr = np.array(optimal_lags)
        median_lag = float(np.median(lags_arr))
        lag_std = float(np.std(lags_arr))

        if median_lag == 0:
            consistency = float(np.mean(lags_arr == 0))
        else:
            median_sign = np.sign(median_lag)
            consistency = float(np.mean(np.sign(lags_arr) == median_sign))

        return median_lag, lag_std, round(consistency, 4)

    @staticmethod
    def _granger_causality(
        ret_a: np.ndarray,
        ret_b: np.ndarray,
        peak_lag_bars: int = 0,
    ) -> dict[str, float | None]:
        """Directional Granger causality based on cross-correlation peak lag.

        If peak_lag > 0: only test a→b (a leads b).
        If peak_lag < 0: only test b→a (b leads a).
        If peak_lag == 0: test both directions.
        """
        result: dict[str, float | None] = {
            "a_to_b_f": None,
            "a_to_b_p": None,
            "b_to_a_f": None,
            "b_to_a_p": None,
        }

        def _run_granger(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
            """Test if y Granger-causes x. Returns (best_f, best_p) or None."""
            data = np.column_stack([x, y])
            try:
                gc = grangercausalitytests(data, maxlag=GRANGER_MAX_LAG, verbose=False)
                best_p = 1.0
                best_f = 0.0
                for lag in range(1, GRANGER_MAX_LAG + 1):
                    f_stat, p_val, _, _ = gc[lag][0]["ssr_ftest"]
                    if p_val < best_p:
                        best_p = p_val
                        best_f = f_stat
                return round(best_f, 6), round(best_p, 6)
            except Exception:
                logger.debug("Granger test failed", exc_info=True)
                return None

        # Test a→b (does a's past predict b's future?)
        if peak_lag_bars >= 0:
            res = _run_granger(ret_b, ret_a)
            if res:
                result["a_to_b_f"], result["a_to_b_p"] = res

        # Test b→a (does b's past predict a's future?)
        if peak_lag_bars <= 0:
            res = _run_granger(ret_a, ret_b)
            if res:
                result["b_to_a_f"], result["b_to_a_p"] = res

        return result
