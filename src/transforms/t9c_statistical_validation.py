from __future__ import annotations

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
LAGS_BARS = 12  # +/- 60 min
MIN_ALIGNED_BARS = 288  # 24h at 5-min resolution
MIN_OVERLAP_PCT = 0.50
GRANGER_MAX_LAG = 12
TAIL_PERCENTILE_LO = 5
TAIL_PERCENTILE_HI = 95


class T9cStatisticalValidation(Transform):
    def __init__(self) -> None:
        super().__init__(
            name="t9c",
            description="Statistical validation of semantically discovered market pairs",
            dependencies=["t9a", "t9b", "t1b"],
        )

    def _make_connection(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect()
        tmp_dir = get_tmp_dir()
        con.execute("SET memory_limit='20GB'")
        con.execute(f"SET temp_directory='{tmp_dir}'")
        con.execute("SET preserve_insertion_order=false")
        con.execute("SET threads=4")
        return con

    def run(self) -> None:
        out = self.ensure_output_dir()

        t9a_dir = self.base_dir / "data" / "transforms" / "t9a"
        t9b_dir = self.base_dir / "data" / "transforms" / "t9b"
        t1b_dir = self.base_dir / "data" / "transforms" / "t1b"

        # Load classified pairs (excluding unrelated)
        classified = pd.read_parquet(t9b_dir / "classified_pairs.parquet")
        classified = classified[classified["relationship_type"] != "unrelated"].copy()
        logger.info("Loaded %d non-unrelated classified pairs", len(classified))

        if classified.empty:
            logger.warning("No classified pairs to validate")
            self._write_empty_outputs(out)
            self.write_manifest({"validated_pairs": 0, "network_edges": 0})
            return

        # Load unified markets for ID resolution
        markets_unified = pd.read_parquet(t9a_dir / "markets_unified.parquet")

        # Load qualifying markets from T1B
        qm = pd.read_parquet(t1b_dir / "qualifying_markets.parquet")

        # Resolve T9a event-level IDs to T1B ticker-level IDs
        resolved_pairs = self._resolve_market_ids(classified, markets_unified, qm)
        logger.info("Resolved %d pairs to T1B tickers", len(resolved_pairs))

        if resolved_pairs.empty:
            logger.warning("No pairs could be resolved to T1B tickers")
            self._write_empty_outputs(out)
            self.write_manifest({"validated_pairs": 0, "network_edges": 0})
            return

        # Preload bars for all needed markets (reuse T5 pattern)
        bar_cache = self._preload_bars(resolved_pairs, t1b_dir)
        logger.info("Preloaded bars for %d markets", len(bar_cache))

        # Analyze each pair
        results: list[dict[str, Any]] = []
        with self.progress("Validating market pairs"):
            for i, (_, row) in enumerate(resolved_pairs.iterrows()):
                if i % 100 == 0:
                    logger.info("Validating pair %d/%d", i + 1, len(resolved_pairs))

                result = self._analyze_pair(row, bar_cache)
                if result is not None:
                    results.append(result)

        # Write outputs
        validated_path = out / "validated_pairs.parquet"
        network_path = out / "correlation_network.parquet"

        if results:
            validated_df = pd.DataFrame(results)
            validated_df.to_parquet(validated_path, index=False)
            logger.info("Wrote %d validated pairs", len(validated_df))

            # Build correlation network (adjacency list)
            network_df = validated_df[[
                "market_a", "market_b", "pearson_r", "cosine_sim",
                "relationship_type", "peak_lag_minutes",
            ]].copy()
            network_df = network_df.rename(columns={"market_a": "source", "market_b": "target"})
            network_df["weight"] = network_df["pearson_r"].abs()
            network_df.to_parquet(network_path, index=False)
            logger.info("Wrote correlation network with %d edges", len(network_df))
        else:
            logger.warning("No pairs had sufficient bar data for validation")
            self._write_empty_outputs(out)

        self.write_manifest({
            "validated_pairs": len(results),
            "network_edges": len(results),
            "pairs_input": len(resolved_pairs),
            "pairs_with_bars": len(results),
        })

    # -----------------------------------------------------------------------
    # ID Resolution
    # -----------------------------------------------------------------------
    def _resolve_market_ids(
        self,
        classified: pd.DataFrame,
        markets_unified: pd.DataFrame,
        qualifying_markets: pd.DataFrame,
    ) -> pd.DataFrame:
        """Resolve T9a event-level market IDs to T1B ticker-level IDs.

        - Kalshi: event_ticker → highest-volume qualifying ticker in that event
        - Polymarket: market_id directly if in qualifying_markets
        """
        con = self._make_connection()
        try:
            # Build Kalshi event_ticker → best qualifying ticker mapping
            kalshi_markets_path = str(self.base_dir / "data" / "kalshi" / "markets" / "*.parquet")
            kalshi_qm = qualifying_markets[qualifying_markets["platform"] == "kalshi"]
            qm_set = set(kalshi_qm["market_id"].tolist())

            # Get event_ticker → ticker mapping from raw markets, filtered to qualifying
            raw_kalshi = con.execute(f"""
                SELECT event_ticker, ticker, volume
                FROM read_parquet('{kalshi_markets_path}')
                WHERE event_ticker IS NOT NULL
                ORDER BY volume DESC NULLS LAST
            """).fetchdf()

            # Keep only qualifying tickers, pick highest-volume per event
            raw_kalshi = raw_kalshi[raw_kalshi["ticker"].isin(qm_set)]
            kalshi_event_to_ticker = (
                raw_kalshi.groupby("event_ticker")
                .first()["ticker"]
                .to_dict()
            )
        finally:
            con.close()

        # Polymarket qualifying set
        poly_qm = qualifying_markets[qualifying_markets["platform"] == "polymarket"]
        poly_qm_set = set(poly_qm["market_id"].tolist())

        # Resolve each pair
        rows = []
        for _, row in classified.iterrows():
            ticker_a = self._resolve_one(
                row["market_a"], row["platform_a"], kalshi_event_to_ticker, poly_qm_set
            )
            ticker_b = self._resolve_one(
                row["market_b"], row["platform_b"], kalshi_event_to_ticker, poly_qm_set
            )
            if ticker_a is None or ticker_b is None:
                continue
            if ticker_a == ticker_b:
                continue

            rows.append({
                **row.to_dict(),
                "ticker_a": ticker_a,
                "ticker_b": ticker_b,
            })

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    @staticmethod
    def _resolve_one(
        market_id: str,
        platform: str,
        kalshi_event_to_ticker: dict[str, str],
        poly_qm_set: set[str],
    ) -> str | None:
        if platform == "kalshi":
            return kalshi_event_to_ticker.get(market_id)
        elif platform == "polymarket":
            return market_id if market_id in poly_qm_set else None
        return None

    # -----------------------------------------------------------------------
    # Bar Preloading (reuse T5 pattern)
    # -----------------------------------------------------------------------
    def _preload_bars(
        self, pairs: pd.DataFrame, t1b_dir
    ) -> dict[tuple[str, str], pd.DataFrame]:
        """Load all needed bars in one query per platform."""
        needed: dict[str, set[str]] = {}
        for _, row in pairs.iterrows():
            needed.setdefault(row["platform_a"], set()).add(row["ticker_a"])
            needed.setdefault(row["platform_b"], set()).add(row["ticker_b"])

        cache: dict[tuple[str, str], pd.DataFrame] = {}
        con = self._make_connection()
        try:
            for platform, market_ids in needed.items():
                bars_dir = t1b_dir / platform / "bars_5min"
                if not bars_dir.exists():
                    logger.warning("Bars directory missing: %s", bars_dir)
                    continue

                glob_path = str(bars_dir / "**" / "*.parquet")
                id_list = list(market_ids)
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
                    logger.warning("Failed to preload bars for %s", platform, exc_info=True)
                    continue

                if df.empty:
                    continue

                df["bar_start"] = pd.to_datetime(df["bar_start"], utc=True).dt.tz_localize(None)

                for ticker, group in df.groupby("ticker"):
                    cache[(platform, ticker)] = group.reset_index(drop=True)

                logger.info(
                    "Preloaded %d markets from %s (%d total bars)",
                    len(df["ticker"].unique()), platform, len(df),
                )
        finally:
            con.close()

        return cache

    # -----------------------------------------------------------------------
    # Pair Analysis
    # -----------------------------------------------------------------------
    def _analyze_pair(
        self, row: pd.Series, bar_cache: dict[tuple[str, str], pd.DataFrame]
    ) -> dict[str, Any] | None:
        bars_a = bar_cache.get((row["platform_a"], row["ticker_a"]))
        bars_b = bar_cache.get((row["platform_b"], row["ticker_b"]))

        if bars_a is None or bars_b is None:
            return None
        if bars_a.empty or bars_b.empty:
            return None

        # Full outer join on bar_start
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

        # Forward-fill and clean
        merged["close_a"] = merged["close_a"].ffill()
        merged["close_b"] = merged["close_b"].ffill()
        merged["ret_a"] = merged["ret_a"].fillna(0.0)
        merged["ret_b"] = merged["ret_b"].fillna(0.0)
        merged = merged.dropna(subset=["close_a", "close_b"])

        if len(merged) < MIN_ALIGNED_BARS:
            return None

        ret_a = merged["ret_a"].values.astype(float)
        ret_b = merged["ret_b"].values.astype(float)
        n_aligned = len(merged)
        pct_overlap = min(pct_a, pct_b)

        # Pearson & Spearman correlation on returns
        pearson_r = float(np.corrcoef(ret_a, ret_b)[0, 1]) if np.std(ret_a) > 1e-12 and np.std(ret_b) > 1e-12 else 0.0
        from scipy.stats import spearmanr
        spearman_r = float(spearmanr(ret_a, ret_b).statistic) if len(ret_a) > 2 else 0.0

        # Price level correlation (more meaningful for prediction markets)
        close_a = merged["close_a"].values.astype(float)
        close_b = merged["close_b"].values.astype(float)
        level_pearson_r = float(np.corrcoef(close_a, close_b)[0, 1]) if np.std(close_a) > 1e-12 and np.std(close_b) > 1e-12 else 0.0
        level_spearman_r = float(spearmanr(close_a, close_b).statistic) if len(close_a) > 2 else 0.0

        # Cross-correlation peak
        peak_lag_bars, peak_xcorr = self._cross_correlation(ret_a, ret_b)
        peak_lag_minutes = peak_lag_bars * BAR_RESOLUTION_MIN

        # Granger causality both directions
        gc = self._granger_causality(ret_a, ret_b)

        # Empirical tail dependence
        tail_upper, tail_lower = self._tail_dependence(ret_a, ret_b)

        return {
            # T9b columns carried forward
            "market_a": row["market_a"],
            "platform_a": row["platform_a"],
            "market_b": row["market_b"],
            "platform_b": row["platform_b"],
            "cosine_sim": row["cosine_sim"],
            "title_a": row.get("title_a", ""),
            "title_b": row.get("title_b", ""),
            "relationship_type": row["relationship_type"],
            "confidence": row.get("confidence", 0.0),
            "reasoning": row.get("reasoning", ""),
            # Validation metrics
            "pearson_r": round(pearson_r, 6),
            "spearman_r": round(spearman_r, 6),
            "level_pearson_r": round(level_pearson_r, 6),
            "level_spearman_r": round(level_spearman_r, 6),
            "peak_lag_bars": peak_lag_bars,
            "peak_lag_minutes": peak_lag_minutes,
            "peak_xcorr": round(peak_xcorr, 6),
            "granger_a_to_b_f": gc["a_to_b_f"],
            "granger_a_to_b_p": gc["a_to_b_p"],
            "granger_b_to_a_f": gc["b_to_a_f"],
            "granger_b_to_a_p": gc["b_to_a_p"],
            "tail_dep_upper": round(tail_upper, 6),
            "tail_dep_lower": round(tail_lower, 6),
            "n_aligned_bars": n_aligned,
            "pct_overlap": round(pct_overlap, 4),
        }

    # -----------------------------------------------------------------------
    # Statistical helpers
    # -----------------------------------------------------------------------
    @staticmethod
    def _cross_correlation(ret_a: np.ndarray, ret_b: np.ndarray) -> tuple[int, float]:
        """Cross-correlation peak within +/- LAGS_BARS window."""
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
    def _granger_causality(ret_a: np.ndarray, ret_b: np.ndarray) -> dict[str, float | None]:
        """Bidirectional Granger causality. Returns best F-stat and p-value each direction."""
        result: dict[str, float | None] = {
            "a_to_b_f": None, "a_to_b_p": None,
            "b_to_a_f": None, "b_to_a_p": None,
        }

        def _run_granger(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
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

        # a→b: does a's past predict b's future?
        res = _run_granger(ret_b, ret_a)
        if res:
            result["a_to_b_f"], result["a_to_b_p"] = res

        # b→a: does b's past predict a's future?
        res = _run_granger(ret_a, ret_b)
        if res:
            result["b_to_a_f"], result["b_to_a_p"] = res

        return result

    @staticmethod
    def _tail_dependence(ret_a: np.ndarray, ret_b: np.ndarray) -> tuple[float, float]:
        """Empirical tail dependence at 5th/95th percentiles.

        Upper tail: P(B > q95 | A > q95)
        Lower tail: P(B < q5 | A < q5)
        """
        if len(ret_a) < 20:
            return 0.0, 0.0

        q_hi_a = np.percentile(ret_a, TAIL_PERCENTILE_HI)
        q_hi_b = np.percentile(ret_b, TAIL_PERCENTILE_HI)
        q_lo_a = np.percentile(ret_a, TAIL_PERCENTILE_LO)
        q_lo_b = np.percentile(ret_b, TAIL_PERCENTILE_LO)

        # Upper tail
        mask_a_hi = ret_a >= q_hi_a
        n_a_hi = mask_a_hi.sum()
        if n_a_hi > 0:
            tail_upper = float((ret_b[mask_a_hi] >= q_hi_b).sum()) / n_a_hi
        else:
            tail_upper = 0.0

        # Lower tail
        mask_a_lo = ret_a <= q_lo_a
        n_a_lo = mask_a_lo.sum()
        if n_a_lo > 0:
            tail_lower = float((ret_b[mask_a_lo] <= q_lo_b).sum()) / n_a_lo
        else:
            tail_lower = 0.0

        return tail_upper, tail_lower

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    def _write_empty_outputs(self, out) -> None:
        validated_cols = [
            "market_a", "platform_a", "market_b", "platform_b", "cosine_sim",
            "title_a", "title_b", "relationship_type", "confidence", "reasoning",
            "pearson_r", "spearman_r", "level_pearson_r", "level_spearman_r",
            "peak_lag_bars", "peak_lag_minutes", "peak_xcorr",
            "granger_a_to_b_f", "granger_a_to_b_p", "granger_b_to_a_f", "granger_b_to_a_p",
            "tail_dep_upper", "tail_dep_lower", "n_aligned_bars", "pct_overlap",
        ]
        network_cols = [
            "source", "target", "weight", "pearson_r", "cosine_sim",
            "relationship_type", "peak_lag_minutes",
        ]
        pd.DataFrame(columns=validated_cols).to_parquet(out / "validated_pairs.parquet", index=False)
        pd.DataFrame(columns=network_cols).to_parquet(out / "correlation_network.parquet", index=False)
