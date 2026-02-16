from __future__ import annotations

import duckdb

from src.transforms._base import Transform
from src.transforms._util import get_tmp_dir

BATCH_SIZE = 5_000


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
                f"COPY ({qualifying_sql}) TO '{output_path}' (FORMAT PARQUET, OVERWRITE true)"
            )

            return count
        finally:
            con.close()

    def _build_raw_bars(
        self,
        platform: str,
        interval_name: str,
        interval: str,
    ) -> str:
        """Phase 1: Single scan of T1A data to produce raw OHLCV bars.

        Writes raw bars (no gap-fill) to a staging parquet file.
        Returns the path to the staging file.
        """
        if platform == "kalshi":
            id_col = "ticker"
            time_col = "created_time"
            vol_col = "count"
            vwap_numerator = "norm_price * count"
            vwap_denominator = "count"
            t1a_glob = "*.parquet"
        else:
            id_col = "token_id"
            time_col = "timestamp"
            vol_col = "volume"
            vwap_numerator = "norm_price * volume"
            vwap_denominator = "volume"
            t1a_glob = "**/*.parquet"

        t1a_path = f"{self.base_dir}/data/transforms/t1a/{platform}/{t1a_glob}"
        qualifying_path = f"{self.output_dir}/qualifying_markets.parquet"
        staging_path = get_tmp_dir() / f"raw_bars_{platform}_{interval_name}.parquet"

        con = duckdb.connect()
        try:
            con.execute(f"SET temp_directory = '{get_tmp_dir()}'")
            con.execute("SET memory_limit = '20GB'")

            con.execute(f"""
                CREATE TEMP TABLE qualifying AS
                SELECT market_id FROM read_parquet('{qualifying_path}')
                WHERE platform = '{platform}'
            """)

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

            con.execute(
                f"COPY ({raw_bars_sql}) TO '{staging_path}' (FORMAT PARQUET, OVERWRITE true)"
            )
            count = con.execute(
                f"SELECT COUNT(*) FROM read_parquet('{staging_path}')"
            ).fetchone()[0]
            print(f"  Raw bars staged: {count:,} rows → {staging_path.name}")
            return str(staging_path)
        finally:
            con.close()

    def _gap_fill_batched(
        self,
        platform: str,
        interval_name: str,
        interval: str,
        raw_bars_path: str,
    ) -> int:
        """Phase 2: Batched gap-fill from the staged raw bars.

        Reads from the small raw bars file (not T1A), so each batch is fast.
        Returns total bar count.
        """
        qualifying_path = f"{self.output_dir}/qualifying_markets.parquet"
        output_base = self.output_dir / platform / f"bars_{interval_name}"
        output_base.mkdir(parents=True, exist_ok=True)
        tmp_dir = get_tmp_dir()

        # Load market IDs that actually have raw bars
        con = duckdb.connect()
        market_ids = [
            row[0]
            for row in con.execute(f"""
                SELECT DISTINCT ticker
                FROM read_parquet('{raw_bars_path}')
                ORDER BY ticker
            """).fetchall()
        ]
        con.close()

        n_batches = (len(market_ids) + BATCH_SIZE - 1) // BATCH_SIZE
        total_count = 0

        for batch_idx in range(n_batches):
            batch_ids = market_ids[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
            batch_dir = output_base / f"batch_{batch_idx}"

            # Resume support: skip batches that already have output
            if batch_dir.exists() and list(batch_dir.glob("*.parquet")):
                batch_con = duckdb.connect()
                count = batch_con.execute(
                    f"SELECT COUNT(*) FROM read_parquet('{batch_dir}/*.parquet')"
                ).fetchone()[0]
                batch_con.close()
                if count > 0:
                    total_count += count
                    print(
                        f"  {platform}/{interval_name} batch {batch_idx + 1}/{n_batches}: "
                        f"{count:,} bars (cached)"
                    )
                    continue

            batch_dir.mkdir(parents=True, exist_ok=True)
            batch_con = duckdb.connect()
            try:
                batch_con.execute(f"SET temp_directory = '{tmp_dir}'")
                batch_con.execute("SET memory_limit = '24GB'")
                batch_con.execute("SET preserve_insertion_order = false")
                batch_con.execute("SET threads = 4")

                # Register batch market IDs
                batch_con.execute("CREATE TEMP TABLE batch_markets (market_id VARCHAR)")
                batch_con.executemany(
                    "INSERT INTO batch_markets VALUES (?)",
                    [(m,) for m in batch_ids],
                )

                # Load only this batch's raw bars from staging
                batch_con.execute(f"""
                    CREATE TEMP TABLE raw_bars AS
                    SELECT * FROM read_parquet('{raw_bars_path}')
                    WHERE ticker IN (SELECT market_id FROM batch_markets)
                """)

                # Gap-fill with generate_series + forward-fill
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
                        LATERAL generate_series(
                            tb.min_start, tb.max_start, INTERVAL '{interval}'
                        ) AS gs(bar_start)
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

                batch_con.execute(f"""
                    COPY ({gap_filled_sql})
                    TO '{batch_dir}'
                    (FORMAT PARQUET, PER_THREAD_OUTPUT true, OVERWRITE true)
                """)

                count = batch_con.execute(
                    f"SELECT COUNT(*) FROM read_parquet('{batch_dir}/*.parquet')"
                ).fetchone()[0]
                total_count += count
                print(
                    f"  {platform}/{interval_name} batch {batch_idx + 1}/{n_batches}: "
                    f"{count:,} bars ({len(batch_ids):,} markets)"
                )
            finally:
                batch_con.close()

        return total_count

    def _build_bars(
        self,
        platform: str,
        interval_name: str,
        interval: str,
    ) -> int:
        """Build gap-filled OHLCV bars: single T1A scan → batched gap-fill."""
        # Phase 1: Single scan of T1A → raw bars staging file
        with self.progress(f"Scanning T1A for {platform} {interval_name} raw bars"):
            raw_bars_path = self._build_raw_bars(platform, interval_name, interval)

        # Phase 2: Batched gap-fill from staging file
        total = self._gap_fill_batched(platform, interval_name, interval, raw_bars_path)

        # Clean up staging file
        try:
            import os
            os.remove(raw_bars_path)
        except OSError:
            pass

        return total

    def run(self):
        self.ensure_output_dir()

        qualifying_path = self.output_dir / "qualifying_markets.parquet"
        if qualifying_path.exists():
            con = duckdb.connect()
            qualifying = con.execute(
                f"SELECT COUNT(*) FROM read_parquet('{qualifying_path}')"
            ).fetchone()[0]
            con.close()
            print(f"Reusing existing qualifying_markets.parquet ({qualifying:,} markets)")
        else:
            with self.progress("Identifying qualifying markets"):
                qualifying = self._identify_qualifying_markets()

        bar_counts = {}
        for platform in ["kalshi", "polymarket"]:
            for interval_name, interval in [
                ("5min", "5 minutes"),
                ("1h", "1 hour"),
                ("1d", "1 day"),
            ]:
                label = f"{platform}_{interval_name}"
                # Skip if this interval already completed
                output_check = self.output_dir / platform / f"bars_{interval_name}"
                if output_check.exists() and list(output_check.glob("**/*.parquet")):
                    con = duckdb.connect()
                    existing = con.execute(
                        f"SELECT COUNT(*) FROM read_parquet('{output_check}/**/*.parquet')"
                    ).fetchone()[0]
                    con.close()
                    if existing > 0:
                        print(f"Skipping {label}: {existing:,} bars already exist")
                        bar_counts[label] = existing
                        continue

                print(f"Building {interval_name} bars for {platform}...")
                count = self._build_bars(platform, interval_name, interval)
                bar_counts[label] = count

        # Clean up any remaining staging files
        tmp_dir = get_tmp_dir()
        for f in tmp_dir.glob("raw_bars_*.parquet"):
            f.unlink(missing_ok=True)

        self.write_manifest(
            {
                "qualifying_markets": qualifying,
                **bar_counts,
            }
        )
