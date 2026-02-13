from __future__ import annotations

import os

import duckdb

from src.analysis.kalshi.util.categories import CATEGORY_SQL
from src.transforms._base import Transform
from src.transforms._util import copy_to_parquet


class T2OrderFlow(Transform):
    def __init__(self):
        super().__init__(
            name="t2",
            description="Order flow imbalance & price impact",
            dependencies=["t1a", "t1b"],
        )

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
        with self.progress("Building category-pooled impact curves"):
            t1a_glob = os.path.join(
                self.base_dir, "data", "transforms", "t1a", "kalshi", "*.parquet"
            )
            output_path = os.path.join(self.output_dir, "impact_curves.parquet")

            con = duckdb.connect()
            query = f"""
                WITH tagged AS (
                    SELECT
                        *,
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
                    FROM read_parquet('{t1a_glob}')
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
            df.to_parquet(output_path)
            row_count = len(df)
            con.close()
            return row_count

    def _build_rolling_ofi(self) -> int:
        """Part 2: Per-market rolling OFI for liquid (qualifying) markets."""
        with self.progress("Computing rolling order flow imbalance"):
            t1a_glob = os.path.join(
                self.base_dir, "data", "transforms", "t1a", "kalshi", "*.parquet"
            )
            qualifying_path = os.path.join(
                self.base_dir,
                "data",
                "transforms",
                "t1b",
                "qualifying_markets.parquet",
            )
            ofi_output_dir = os.path.join(self.output_dir, "kalshi_ofi")
            os.makedirs(ofi_output_dir, exist_ok=True)

            con = duckdb.connect()

            query = f"""
                WITH qualifying AS (
                    SELECT market_id
                    FROM read_parquet('{qualifying_path}')
                    WHERE platform = 'kalshi'
                ),
                t1a_kalshi AS (
                    SELECT t.*
                    FROM read_parquet('{t1a_glob}') t
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

            row_count = copy_to_parquet(con, query, ofi_output_dir)
            con.close()
            return row_count

    def _build_vpin(self) -> int:
        """Part 3: VPIN (Volume-synchronized Probability of Informed Trading)."""
        with self.progress("Computing VPIN"):
            t1a_glob = os.path.join(
                self.base_dir, "data", "transforms", "t1a", "kalshi", "*.parquet"
            )
            qualifying_path = os.path.join(
                self.base_dir,
                "data",
                "transforms",
                "t1b",
                "qualifying_markets.parquet",
            )
            vpin_output_dir = os.path.join(self.output_dir, "kalshi_vpin")
            os.makedirs(vpin_output_dir, exist_ok=True)

            con = duckdb.connect()

            query = f"""
                WITH qualifying AS (
                    SELECT ticker
                    FROM read_parquet('{qualifying_path}')
                    WHERE platform = 'kalshi'
                ),
                t1a_kalshi AS (
                    SELECT t.*,
                        SUM(count) OVER (
                            PARTITION BY t.ticker
                            ORDER BY t.trade_sequence_num
                        ) AS cumulative_volume
                    FROM read_parquet('{t1a_glob}') t
                    SEMI JOIN qualifying q ON t.ticker = q.ticker
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

            row_count = copy_to_parquet(con, query, vpin_output_dir)
            con.close()
            return row_count
