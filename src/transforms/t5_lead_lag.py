from __future__ import annotations

import itertools
import logging
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from scipy.signal import correlate
from statsmodels.tsa.stattools import grangercausalitytests

from src.transforms._base import Transform

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
    # Public entry point
    # -----------------------------------------------------------------------
    def run(self) -> None:
        self.ensure_output_dir()

        pairs = self._curate_pairs()
        logger.info("Curated %d candidate pairs", len(pairs))

        results: list[dict[str, Any]] = []
        for pair in pairs:
            result = self._analyze_pair(pair)
            if result is not None:
                results.append(result)

        if results:
            df = pd.DataFrame(results)
            df.to_parquet(self.output_dir / "lead_lag_summary.parquet", index=False)
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
    def _curate_pairs(
        self,
    ) -> list[tuple[str, str, str, str, str]]:
        """Return ~20 pairs as (platform_a, market_id_a, platform_b, market_id_b, pair_type)."""
        qm_path = self.base_dir / "data" / "transforms" / "t1b" / "qualifying_markets.parquet"

        con = duckdb.connect()
        qm = con.execute(
            """
            SELECT platform, market_id, event_ticker, total_trades
            FROM read_parquet(?)
            ORDER BY total_trades DESC
            """,
            [str(qm_path)],
        ).fetchdf()
        con.close()

        pairs: list[tuple[str, str, str, str, str]] = []

        # ----- Intra-family pairs (Kalshi threshold families) -----
        kalshi = qm[qm["platform"] == "kalshi"].copy()
        if not kalshi.empty and "event_ticker" in kalshi.columns:
            # Group by event_ticker prefix (first segment before hyphen)
            kalshi["family"] = kalshi["event_ticker"].astype(str).apply(
                lambda t: t.split("-")[0] if "-" in str(t) else str(t)
            )
            family_counts = (
                kalshi.groupby("family")["market_id"]
                .count()
                .reset_index(name="n_members")
            )
            family_counts = family_counts[family_counts["n_members"] >= 2].sort_values(
                "n_members", ascending=False
            )

            # For each family pick top members by volume and create pairs
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

        # If we somehow have zero pairs, try intra-platform for polymarket too
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
    # Step 2-5: Analyze a single pair
    # -----------------------------------------------------------------------
    def _analyze_pair(
        self, pair: tuple[str, str, str, str, str]
    ) -> dict[str, Any] | None:
        platform_a, market_id_a, platform_b, market_id_b, pair_type = pair

        # --- Step 2: Align 5-min bars to common clock ---
        bars_a = self._read_bars(platform_a, market_id_a)
        bars_b = self._read_bars(platform_b, market_id_b)

        if bars_a is None or bars_b is None:
            return None
        if bars_a.empty or bars_b.empty:
            return None

        # Full outer join on bar_start
        merged = pd.merge(
            bars_a[["bar_start", "close"]].rename(columns={"close": "close_a"}),
            bars_b[["bar_start", "close"]].rename(columns={"close": "close_b"}),
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

        # Forward-fill then drop remaining NaNs for aligned returns
        merged = merged.sort_values("bar_start")
        merged["close_a"] = merged["close_a"].ffill()
        merged["close_b"] = merged["close_b"].ffill()
        merged = merged.dropna(subset=["close_a", "close_b"])

        if len(merged) < ROLLING_WINDOW_BARS:
            return None

        # Compute returns
        merged["ret_a"] = merged["close_a"].pct_change()
        merged["ret_b"] = merged["close_b"].pct_change()
        merged = merged.dropna(subset=["ret_a", "ret_b"])

        if len(merged) < ROLLING_WINDOW_BARS:
            return None

        ret_a = merged["ret_a"].values
        ret_b = merged["ret_b"].values
        n_aligned = len(merged)
        pct_overlap = min(pct_a, pct_b)

        # --- Step 3: Cross-correlation ---
        peak_lag_bars, peak_correlation = self._cross_correlation(ret_a, ret_b)
        peak_lag_minutes = peak_lag_bars * BAR_RESOLUTION_MIN

        # --- Rolling cross-correlation for lag stability (Step 5) ---
        median_lag, lag_std, lag_consistency_pct = self._rolling_cross_correlation(
            ret_a, ret_b
        )

        # --- Step 4: Granger causality ---
        gc_results = self._granger_causality(ret_a, ret_b)

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
    def _read_bars(self, platform: str, market_id: str) -> pd.DataFrame | None:
        """Read 5-min bar parquet files for a given market.

        T1B outputs bars into a directory (PER_THREAD_OUTPUT), so we glob for
        all *.parquet files inside the directory and filter to the market_id.
        """
        bars_dir = (
            self.base_dir
            / "data"
            / "transforms"
            / "t1b"
            / platform
            / "bars_5min.parquet"
        )

        if not bars_dir.exists():
            logger.debug("Bars directory does not exist: %s", bars_dir)
            return None

        parquet_files = list(bars_dir.glob("*.parquet"))
        if not parquet_files:
            logger.debug("No parquet files found in %s", bars_dir)
            return None

        con = duckdb.connect()
        try:
            df = con.execute(
                """
                SELECT bar_start, close
                FROM read_parquet(?)
                WHERE market_id = ?
                ORDER BY bar_start
                """,
                [str(bars_dir / "*.parquet"), market_id],
            ).fetchdf()
        except Exception:
            logger.debug(
                "Failed to read bars for %s / %s", platform, market_id, exc_info=True
            )
            return None
        finally:
            con.close()

        if df.empty:
            return None

        # Ensure bar_start is datetime
        df["bar_start"] = pd.to_datetime(df["bar_start"])
        return df

    @staticmethod
    def _cross_correlation(
        ret_a: np.ndarray, ret_b: np.ndarray
    ) -> tuple[int, float]:
        """Full cross-correlation, return (peak_lag_bars, peak_correlation).

        Positive lag means a leads b (a's past predicts b's future).
        We restrict to lags in [-LAGS_BARS, +LAGS_BARS].
        """
        # Normalise to zero-mean, unit-variance for interpretable correlation
        a = ret_a - ret_a.mean()
        b = ret_b - ret_b.mean()
        norm = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))

        if norm < 1e-12:
            return 0, 0.0

        full_xcorr = correlate(a, b, mode="full") / norm
        # full_xcorr has length 2*N-1; index N-1 corresponds to lag 0
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
        """Rolling cross-correlation with 24h window.

        Returns (median_lag, lag_std, lag_consistency_pct).
        """
        n = len(ret_a)
        if n < ROLLING_WINDOW_BARS:
            return 0.0, 0.0, 0.0

        optimal_lags: list[int] = []

        for start in range(0, n - ROLLING_WINDOW_BARS + 1, ROLLING_WINDOW_BARS // 4):
            end = start + ROLLING_WINDOW_BARS
            chunk_a = ret_a[start:end]
            chunk_b = ret_b[start:end]

            a = chunk_a - chunk_a.mean()
            b = chunk_b - chunk_b.mean()
            norm = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
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

        # Consistency: fraction of windows where sign matches the median sign
        if median_lag == 0:
            # Consistent if lag is also zero
            consistency = float(np.mean(lags_arr == 0))
        else:
            median_sign = np.sign(median_lag)
            consistency = float(np.mean(np.sign(lags_arr) == median_sign))

        return median_lag, lag_std, round(consistency, 4)

    @staticmethod
    def _granger_causality(
        ret_a: np.ndarray, ret_b: np.ndarray
    ) -> dict[str, float | None]:
        """Granger causality tests in both directions.

        Returns dict with keys: a_to_b_f, a_to_b_p, b_to_a_f, b_to_a_p.
        """
        result: dict[str, float | None] = {
            "a_to_b_f": None,
            "a_to_b_p": None,
            "b_to_a_f": None,
            "b_to_a_p": None,
        }

        data = np.column_stack([ret_b, ret_a])  # test if a (col 1) causes b (col 0)
        try:
            gc = grangercausalitytests(data, maxlag=GRANGER_MAX_LAG, verbose=False)
            # Pick the lag with the smallest p-value (ssr_ftest)
            best_p = 1.0
            best_f = 0.0
            for lag in range(1, GRANGER_MAX_LAG + 1):
                f_stat, p_val, _, _ = gc[lag][0]["ssr_ftest"]
                if p_val < best_p:
                    best_p = p_val
                    best_f = f_stat
            result["a_to_b_f"] = round(best_f, 6)
            result["a_to_b_p"] = round(best_p, 6)
        except Exception:
            logger.debug("Granger a->b test failed", exc_info=True)

        data_rev = np.column_stack([ret_a, ret_b])  # test if b causes a
        try:
            gc_rev = grangercausalitytests(
                data_rev, maxlag=GRANGER_MAX_LAG, verbose=False
            )
            best_p = 1.0
            best_f = 0.0
            for lag in range(1, GRANGER_MAX_LAG + 1):
                f_stat, p_val, _, _ = gc_rev[lag][0]["ssr_ftest"]
                if p_val < best_p:
                    best_p = p_val
                    best_f = f_stat
            result["b_to_a_f"] = round(best_f, 6)
            result["b_to_a_p"] = round(best_p, 6)
        except Exception:
            logger.debug("Granger b->a test failed", exc_info=True)

        return result
