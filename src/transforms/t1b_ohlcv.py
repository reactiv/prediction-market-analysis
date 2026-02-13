from __future__ import annotations

import duckdb

from src.transforms._base import Transform
from src.transforms._util import copy_to_parquet


class T1BOhlcv(Transform):
    def __init__(self):
        super().__init__(
            name="t1b",
            description="Conditional OHLCV bars for liquid markets",
            dependencies=["t1a"],
        )

    def _identify_qualifying_markets(self) -> int:
        """Identify markets with sufficient liquidity and write qualifying_markets.parquet.

        A market qualifies if trades_per_hour > 5 OR total_trades > 100.
        Returns the number of qualifying markets.
        """
        con = duckdb.connect()
        try:
            # Compute per-market stats for Kalshi
            kalshi_stats_sql = f"""
                SELECT
                    'kalshi' AS platform,
                    ticker AS market_id,
                    COUNT(*) AS total_trades,
                    EXTRACT(EPOCH FROM (MAX(created_time) - MIN(created_time))) / 3600.0 AS time_span_hours,
                    CASE
                        WHEN EXTRACT(EPOCH FROM (MAX(created_time) - MIN(created_time))) > 0
                        THEN COUNT(*) / (EXTRACT(EPOCH FROM (MAX(created_time) - MIN(created_time))) / 3600.0)
                        ELSE 0
                    END AS trades_per_hour
                FROM read_parquet('{self.base_dir}/data/transforms/t1a/kalshi/*.parquet')
                GROUP BY ticker
            """

            # Compute per-market stats for Polymarket
            polymarket_stats_sql = f"""
                SELECT
                    'polymarket' AS platform,
                    token_id AS market_id,
                    COUNT(*) AS total_trades,
                    EXTRACT(EPOCH FROM (MAX(timestamp) - MIN(timestamp))) / 3600.0 AS time_span_hours,
                    CASE
                        WHEN EXTRACT(EPOCH FROM (MAX(timestamp) - MIN(timestamp))) > 0
                        THEN COUNT(*) / (EXTRACT(EPOCH FROM (MAX(timestamp) - MIN(timestamp))) / 3600.0)
                        ELSE 0
                    END AS trades_per_hour
                FROM read_parquet('{self.base_dir}/data/transforms/t1a/polymarket/**/*.parquet')
                GROUP BY token_id
            """

            # Union both platforms and filter qualifying markets
            qualifying_sql = f"""
                SELECT * FROM (
                    {kalshi_stats_sql}
                    UNION ALL
                    {polymarket_stats_sql}
                ) combined
                WHERE trades_per_hour > 5 OR total_trades > 100
            """

            output_path = f"{self.output_dir}/qualifying_markets.parquet"
            count = con.execute(f"SELECT COUNT(*) FROM ({qualifying_sql})").fetchone()[0]
            con.execute(
                f"COPY ({qualifying_sql}) TO '{output_path}' (FORMAT PARQUET)"
            )

            return count
        finally:
            con.close()

    def _build_bars(
        self,
        platform: str,
        interval_name: str,
        interval: str,
        qualifying_count: int,
    ) -> int:
        """Build gap-filled OHLCV bars for a given platform and interval.

        Returns the number of bar rows written.
        """
        con = duckdb.connect()
        try:
            # Platform-specific column mappings
            if platform == "kalshi":
                id_col = "ticker"
                time_col = "created_time"
                vol_col = "count"
                vwap_numerator = "norm_price * count"
                vwap_denominator = "count"
            else:
                id_col = "token_id"
                time_col = "timestamp"
                vol_col = "volume"
                vwap_numerator = "norm_price * volume"
                vwap_denominator = "volume"

            t1a_path = f"{self.base_dir}/data/transforms/t1a/{platform}/*.parquet"
            qualifying_path = f"{self.output_dir}/qualifying_markets.parquet"

            # Register the qualifying markets for filtering
            con.execute(f"""
                CREATE TEMP TABLE qualifying AS
                SELECT market_id FROM read_parquet('{qualifying_path}')
                WHERE platform = '{platform}'
            """)

            # Step 2: Generate raw OHLCV bars
            raw_bars_sql = f"""
                SELECT
                    {id_col} AS ticker,
                    time_bucket(INTERVAL '{interval}', {time_col}) AS bar_start,
                    FIRST(norm_price ORDER BY {time_col}) AS open,
                    MAX(norm_price) AS high,
                    MIN(norm_price) AS low,
                    LAST(norm_price ORDER BY {time_col}) AS close,
                    SUM({vol_col}) AS volume,
                    SUM(signed_flow) AS net_flow,
                    COUNT(*) AS trade_count,
                    SUM({vwap_numerator}) / NULLIF(SUM({vwap_denominator}), 0) AS vwap
                FROM read_parquet('{t1a_path}')
                WHERE {id_col} IN (SELECT market_id FROM qualifying)
                GROUP BY {id_col}, bar_start
            """

            con.execute(f"CREATE TEMP TABLE raw_bars AS {raw_bars_sql}")

            # Step 3: Gap filling with generate_series and forward-fill
            # Build a full time grid per ticker, then LEFT JOIN and forward-fill
            gap_filled_sql = f"""
                WITH ticker_bounds AS (
                    SELECT
                        ticker,
                        MIN(bar_start) AS min_start,
                        MAX(bar_start) AS max_start
                    FROM raw_bars
                    GROUP BY ticker
                ),
                time_grid AS (
                    SELECT
                        tb.ticker,
                        gs.bar_start
                    FROM ticker_bounds tb,
                    LATERAL generate_series(tb.min_start, tb.max_start, INTERVAL '{interval}') AS gs(bar_start)
                ),
                joined AS (
                    SELECT
                        g.ticker,
                        g.bar_start,
                        b.open,
                        b.high,
                        b.low,
                        b.close,
                        b.volume,
                        b.net_flow,
                        b.trade_count,
                        b.vwap
                    FROM time_grid g
                    LEFT JOIN raw_bars b
                        ON g.ticker = b.ticker AND g.bar_start = b.bar_start
                ),
                forward_filled AS (
                    SELECT
                        ticker,
                        bar_start,
                        COALESCE(
                            open,
                            LAST_VALUE(close IGNORE NULLS) OVER (
                                PARTITION BY ticker ORDER BY bar_start
                                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                            )
                        ) AS open,
                        COALESCE(
                            high,
                            LAST_VALUE(close IGNORE NULLS) OVER (
                                PARTITION BY ticker ORDER BY bar_start
                                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                            )
                        ) AS high,
                        COALESCE(
                            low,
                            LAST_VALUE(close IGNORE NULLS) OVER (
                                PARTITION BY ticker ORDER BY bar_start
                                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                            )
                        ) AS low,
                        COALESCE(
                            close,
                            LAST_VALUE(close IGNORE NULLS) OVER (
                                PARTITION BY ticker ORDER BY bar_start
                                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                            )
                        ) AS close,
                        COALESCE(volume, 0) AS volume,
                        COALESCE(net_flow, 0) AS net_flow,
                        COALESCE(trade_count, 0) AS trade_count,
                        vwap
                    FROM joined
                )
                SELECT
                    ticker,
                    bar_start,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    net_flow,
                    trade_count,
                    vwap,
                    close - LAG(close) OVER (
                        PARTITION BY ticker ORDER BY bar_start
                    ) AS bar_return,
                    high - low AS bar_range
                FROM forward_filled
            """

            output_dir = f"{self.output_dir}/{platform}/bars_{interval_name}.parquet"
            count = copy_to_parquet(con, gap_filled_sql, output_dir)

            # Clean up temp tables for next call
            con.execute("DROP TABLE IF EXISTS raw_bars")
            con.execute("DROP TABLE IF EXISTS qualifying")

            return count
        finally:
            con.close()

    def run(self):
        self.ensure_output_dir()

        with self.progress("Identifying qualifying markets"):
            qualifying = self._identify_qualifying_markets()

        bar_counts = {}
        for platform in ["kalshi", "polymarket"]:
            for interval_name, interval in [
                ("5min", "5 minutes"),
                ("1h", "1 hour"),
                ("1d", "1 day"),
            ]:
                with self.progress(
                    f"Building {interval_name} bars for {platform}"
                ):
                    count = self._build_bars(
                        platform, interval_name, interval, qualifying
                    )
                    bar_counts[f"{platform}_{interval_name}"] = count

        self.write_manifest(
            {
                "qualifying_markets": qualifying,
                **bar_counts,
            }
        )
