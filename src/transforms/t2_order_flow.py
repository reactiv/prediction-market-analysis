"""T2: Order flow imbalance & price impact.

Builds three derived datasets from T1A enriched trades (Kalshi only):
  1. Impact curves: category-pooled price impact regressions
  2. Rolling OFI: per-market order flow imbalance at 20/50/100 trade windows
  3. VPIN: volume-synchronized probability of informed trading

Performance notes (vs original):
  - Tuned DuckDB connections (memory limit, temp dir, threading)
  - Resume support: skip phases whose output already exists
  - OFI: project only needed columns instead of SELECT *
  - VPIN: reuse T1A's cumulative_volume instead of recomputing
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from src.analysis.kalshi.util.categories import CATEGORY_SQL
from src.transforms._base import Transform
from src.transforms._util import copy_to_parquet, get_tmp_dir


class T2OrderFlow(Transform):
    def __init__(self):
        super().__init__(
            name="t2",
            description="Order flow imbalance & price impact",
            dependencies=["t1a", "t1b"],
        )

    def _make_connection(self) -> duckdb.DuckDBPyConnection:
        """Create a DuckDB connection with tuned settings."""
        con = duckdb.connect()
        tmp_dir = get_tmp_dir()
        con.execute("SET memory_limit='20GB'")
        con.execute(f"SET temp_directory='{tmp_dir}'")
        con.execute("SET preserve_insertion_order=false")
        con.execute("SET threads=4")
        return con

    @property
    def _t1a_glob(self) -> str:
        return str(Path(self.base_dir) / "data" / "transforms" / "t1a" / "kalshi" / "*.parquet")

    @property
    def _qualifying_path(self) -> str:
        return str(Path(self.base_dir) / "data" / "transforms" / "t1b" / "qualifying_markets.parquet")

    def run(self):
        self.ensure_output_dir()
        impact_rows = self._build_impact_curves()
        ofi_rows = self._build_rolling_ofi()
        vpin_rows = self._build_vpin()
        self.write_manifest(
            {
                "impact_curve_rows": impact_rows,
                "ofi_rows": ofi_rows,
                "vpin_rows": vpin_rows,
            }
        )

    def _build_impact_curves(self) -> int:
        """Part 1: Category-pooled impact curves (Kalshi, ALL trades)."""
        output_path = Path(self.output_dir) / "impact_curves.parquet"

        # Resume: skip if already built
        if output_path.exists():
            con = self._make_connection()
            try:
                count = con.execute(
                    f"SELECT COUNT(*) FROM read_parquet('{output_path}')"
                ).fetchone()[0]
            finally:
                con.close()
            print(f"  Impact curves: resuming — {count} rows already built")
            return count

        with self.progress("Building category-pooled impact curves"):
            con = self._make_connection()
            query = f"""
                WITH tagged AS (
                    SELECT
                        event_ticker,
                        count,
                        delta_price,
                        signed_flow,
                        time_to_expiry_seconds,
                        {CATEGORY_SQL} AS category,
                        CASE
                            WHEN count = 1 THEN '1'
                            WHEN count <= 10 THEN '2-10'
                            WHEN count <= 100 THEN '11-100'
                            WHEN count <= 1000 THEN '101-1K'
                            ELSE '1K+'
                        END AS size_bucket,
                        CASE
                            WHEN time_to_expiry_seconds < 3600 THEN '<1h'
                            WHEN time_to_expiry_seconds < 21600 THEN '1-6h'
                            WHEN time_to_expiry_seconds < 86400 THEN '6-24h'
                            WHEN time_to_expiry_seconds < 604800 THEN '1-7d'
                            ELSE '7d+'
                        END AS expiry_bucket
                    FROM read_parquet('{self._t1a_glob}')
                )
                SELECT
                    category,
                    size_bucket,
                    expiry_bucket,
                    regr_slope(delta_price, signed_flow) AS impact_coeff,
                    regr_r2(delta_price, signed_flow) AS r_squared,
                    COUNT(*) AS sample_size,
                    AVG(ABS(delta_price)) AS avg_abs_delta_price,
                    AVG(ABS(signed_flow)) AS avg_abs_flow
                FROM tagged
                GROUP BY category, size_bucket, expiry_bucket
                HAVING COUNT(*) >= 30
                ORDER BY category, size_bucket, expiry_bucket
            """

            df = con.execute(query).df()
            df.to_parquet(str(output_path))
            row_count = len(df)
            con.close()
            return row_count

    def _build_rolling_ofi(self) -> int:
        """Part 2: Per-market rolling OFI for liquid (qualifying) markets."""
        ofi_output_dir = Path(self.output_dir) / "kalshi_ofi"

        # Resume: skip if already built
        if ofi_output_dir.exists() and list(ofi_output_dir.glob("*.parquet")):
            con = self._make_connection()
            try:
                count = con.execute(
                    f"SELECT COUNT(*) FROM read_parquet('{ofi_output_dir}/*.parquet')"
                ).fetchone()[0]
            finally:
                con.close()
            print(f"  OFI: resuming — {count} rows already built")
            return count

        with self.progress("Computing rolling order flow imbalance"):
            ofi_output_dir.mkdir(parents=True, exist_ok=True)
            con = self._make_connection()

            # Project only the columns needed for OFI + downstream analysis
            query = f"""
                WITH qualifying AS (
                    SELECT market_id
                    FROM read_parquet('{self._qualifying_path}')
                    WHERE platform = 'kalshi'
                ),
                t1a_kalshi AS (
                    SELECT
                        t.ticker,
                        t.trade_sequence_num,
                        t.signed_flow,
                        t.norm_price,
                        t.created_time,
                        t.count,
                        t.delta_price,
                        t.cumulative_volume,
                        t.cumulative_net_flow,
                        t.time_to_expiry_seconds
                    FROM read_parquet('{self._t1a_glob}') t
                    SEMI JOIN qualifying q ON t.ticker = q.market_id
                )
                SELECT
                    *,
                    SUM(signed_flow) OVER (
                        PARTITION BY ticker ORDER BY trade_sequence_num
                        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                    ) AS ofi_20,
                    SUM(signed_flow) OVER (
                        PARTITION BY ticker ORDER BY trade_sequence_num
                        ROWS BETWEEN 49 PRECEDING AND CURRENT ROW
                    ) AS ofi_50,
                    SUM(signed_flow) OVER (
                        PARTITION BY ticker ORDER BY trade_sequence_num
                        ROWS BETWEEN 99 PRECEDING AND CURRENT ROW
                    ) AS ofi_100
                FROM t1a_kalshi
            """

            row_count = copy_to_parquet(con, query, str(ofi_output_dir))
            con.close()
            return row_count

    def _build_vpin(self) -> int:
        """Part 3: VPIN (Volume-synchronized Probability of Informed Trading)."""
        vpin_output_dir = Path(self.output_dir) / "kalshi_vpin"

        # Resume: skip if already built
        if vpin_output_dir.exists() and list(vpin_output_dir.glob("*.parquet")):
            con = self._make_connection()
            try:
                count = con.execute(
                    f"SELECT COUNT(*) FROM read_parquet('{vpin_output_dir}/*.parquet')"
                ).fetchone()[0]
            finally:
                con.close()
            print(f"  VPIN: resuming — {count} rows already built")
            return count

        with self.progress("Computing VPIN"):
            vpin_output_dir.mkdir(parents=True, exist_ok=True)
            con = self._make_connection()

            # Use T1A's existing cumulative_volume instead of recomputing
            query = f"""
                WITH qualifying AS (
                    SELECT market_id
                    FROM read_parquet('{self._qualifying_path}')
                    WHERE platform = 'kalshi'
                ),
                t1a_kalshi AS (
                    SELECT
                        t.ticker,
                        t.trade_sequence_num,
                        t.signed_flow,
                        t.count,
                        t.cumulative_volume
                    FROM read_parquet('{self._t1a_glob}') t
                    SEMI JOIN qualifying q ON t.ticker = q.market_id
                ),
                bucket_agg AS (
                    SELECT
                        ticker,
                        FLOOR(cumulative_volume / 50) AS volume_bucket,
                        SUM(CASE WHEN signed_flow > 0 THEN signed_flow ELSE 0 END) AS buy_flow,
                        SUM(CASE WHEN signed_flow < 0 THEN ABS(signed_flow) ELSE 0 END) AS sell_flow,
                        SUM(count) AS bucket_volume
                    FROM t1a_kalshi
                    GROUP BY ticker, volume_bucket
                ),
                vpin AS (
                    SELECT
                        ticker,
                        volume_bucket,
                        buy_flow,
                        sell_flow,
                        bucket_volume,
                        SUM(ABS(buy_flow - sell_flow)) OVER (
                            PARTITION BY ticker ORDER BY volume_bucket
                            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                        ) / NULLIF(SUM(bucket_volume) OVER (
                            PARTITION BY ticker ORDER BY volume_bucket
                            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                        ), 0) AS vpin_20
                    FROM bucket_agg
                )
                SELECT * FROM vpin
            """

            row_count = copy_to_parquet(con, query, str(vpin_output_dir))
            con.close()
            return row_count
