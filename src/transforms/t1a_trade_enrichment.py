"""T1A: Trade-level enrichment with sequential context.

Enriches raw trade data from Kalshi and Polymarket with sequential context
using DuckDB window functions, then writes results to parquet files.

Enrichments include:
- Normalized price (0-100 scale for Kalshi, 0-1 for Polymarket)
- Signed order flow (positive = buy, negative = sell)
- Sequential trade numbering per ticker/token
- Price deltas and time deltas from previous trade
- Cumulative volume and net flow
- Time to expiry
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

from src.transforms._base import Transform
from src.transforms._util import copy_to_parquet, get_tmp_dir


class T1ATradeEnrichment(Transform):
    def __init__(self):
        super().__init__(
            name="t1a",
            description="Trade-level enrichment with sequential context",
            dependencies=[],
        )

    def run(self):
        self.ensure_output_dir()
        kalshi_rows = self._enrich_kalshi()
        poly_rows = self._enrich_polymarket()
        self.write_manifest(
            {
                "kalshi_rows": kalshi_rows,
                "polymarket_rows": poly_rows,
                "total_rows": kalshi_rows + poly_rows,
            }
        )

    def _enrich_kalshi(self) -> int:
        """Enrich Kalshi trades with sequential context via DuckDB window functions."""
        trades_dir = self.base_dir / "data" / "kalshi" / "trades"
        markets_dir = self.base_dir / "data" / "kalshi" / "markets"
        output_dir = self.output_dir / "kalshi"

        con = duckdb.connect()

        query = f"""
        SELECT
            t.trade_id,
            t.ticker,
            m.event_ticker,
            t.count,
            t.yes_price,
            t.no_price,
            t.taker_side,
            t.created_time,
            t.yes_price AS norm_price,
            CASE WHEN t.taker_side = 'yes' THEN t.count ELSE -t.count END AS signed_flow,
            ROW_NUMBER() OVER w AS trade_sequence_num,
            t.yes_price - LAG(t.yes_price) OVER w AS delta_price,
            EPOCH(t.created_time - LAG(t.created_time) OVER w) AS time_since_prev,
            SUM(t.count) OVER (
                PARTITION BY t.ticker ORDER BY t.created_time
                ROWS UNBOUNDED PRECEDING
            ) AS cumulative_volume,
            SUM(CASE WHEN t.taker_side = 'yes' THEN t.count ELSE -t.count END) OVER (
                PARTITION BY t.ticker ORDER BY t.created_time
                ROWS UNBOUNDED PRECEDING
            ) AS cumulative_net_flow,
            ROW_NUMBER() OVER w AS cumulative_trade_count,
            EPOCH(m.close_time - t.created_time) AS time_to_expiry_seconds
        FROM '{trades_dir}/*.parquet' t
        LEFT JOIN '{markets_dir}/*.parquet' m
            ON t.ticker = m.ticker
        WINDOW w AS (PARTITION BY t.ticker ORDER BY t.created_time)
        ORDER BY t.ticker, t.created_time
        """

        with self.progress("Enriching Kalshi trades"):
            total_rows = copy_to_parquet(con, query, output_dir)

        con.close()
        print(f"  Kalshi: {total_rows:,} enriched trades")
        return total_rows

    def _enrich_polymarket(self) -> int:
        """Enrich Polymarket trades with sequential context via DuckDB window functions.

        Processes 388M+ rows in batches by token_id hash to stay within memory:
        Stage 1: Write lean base trades (joins, no windows) to staging parquet.
        Stage 2: Process 10 batches of ~39M rows each through window functions.
        """
        import shutil

        trades_dir = self.base_dir / "data" / "polymarket" / "trades"
        blocks_dir = self.base_dir / "data" / "polymarket" / "blocks"
        markets_dir = self.base_dir / "data" / "polymarket" / "markets"
        output_dir = self.output_dir / "polymarket"
        staging_dir = self.output_dir / "_polymarket_staging"

        tmp_dir = get_tmp_dir()
        n_batches = 10

        # ── Stage 1: Compute base fields and write lean parquet ──
        con = duckdb.connect()
        con.execute(f"SET temp_directory = '{tmp_dir}'")
        con.execute("SET preserve_insertion_order = false")

        with self.progress("Building token-market map"):
            token_market_map = self._build_token_market_map(con, markets_dir)
            token_map_df = pd.DataFrame(
                list(token_market_map.items()), columns=["token_id", "market_id"]
            )
            con.register("token_market_map", token_map_df)

        con.execute(
            f"""
            CREATE TEMP TABLE poly_markets AS
            SELECT id, end_date FROM '{markets_dir}/*.parquet'
            """
        )

        base_query = f"""
        SELECT
            t.block_number,
            t.log_index,
            COALESCE(t.timestamp, b.timestamp)::TIMESTAMP AS timestamp,
            CASE WHEN t.maker_asset_id = '0' THEN t.taker_asset_id ELSE t.maker_asset_id END AS token_id,
            CASE
                WHEN t.maker_asset_id = '0'
                THEN t.maker_amount::DOUBLE / NULLIF(t.taker_amount::DOUBLE, 0)
                ELSE t.taker_amount::DOUBLE / NULLIF(t.maker_amount::DOUBLE, 0)
            END AS norm_price,
            CASE
                WHEN t.maker_asset_id = '0' THEN t.maker_amount::DOUBLE
                ELSE t.taker_amount::DOUBLE
            END AS volume,
            CASE
                WHEN t.maker_asset_id = '0' THEN t.maker_amount::DOUBLE
                ELSE -t.taker_amount::DOUBLE
            END AS signed_flow,
            tmm.market_id,
            pm.end_date
        FROM '{trades_dir}/*.parquet' t
        LEFT JOIN '{blocks_dir}/*.parquet' b ON t.block_number = b.block_number
        LEFT JOIN token_market_map tmm
            ON (CASE WHEN t.maker_asset_id = '0' THEN t.taker_asset_id ELSE t.maker_asset_id END) = tmm.token_id
        LEFT JOIN poly_markets pm ON tmm.market_id = pm.id
        """

        with self.progress("Stage 1: Writing base trades to parquet"):
            copy_to_parquet(con, base_query, staging_dir)

        con.close()

        # ── Stage 2: Process in batches by token_id hash ──
        output_dir.mkdir(parents=True, exist_ok=True)
        total_rows = 0

        for batch_idx in range(n_batches):
            with self.progress(f"Stage 2: Batch {batch_idx + 1}/{n_batches}"):
                con = duckdb.connect()
                con.execute(f"SET temp_directory = '{tmp_dir}'")
                con.execute("SET preserve_insertion_order = false")

                batch_dir = output_dir / f"batch_{batch_idx}"
                enriched_query = f"""
                SELECT
                    block_number,
                    log_index,
                    timestamp,
                    token_id,
                    norm_price,
                    volume,
                    signed_flow,
                    market_id,
                    ROW_NUMBER() OVER w AS trade_sequence_num,
                    norm_price - LAG(norm_price) OVER w AS delta_price,
                    EPOCH(timestamp - LAG(timestamp) OVER w) AS time_since_prev,
                    SUM(volume) OVER (
                        PARTITION BY token_id ORDER BY block_number, log_index
                        ROWS UNBOUNDED PRECEDING
                    ) AS cumulative_volume,
                    SUM(signed_flow) OVER (
                        PARTITION BY token_id ORDER BY block_number, log_index
                        ROWS UNBOUNDED PRECEDING
                    ) AS cumulative_net_flow,
                    ROW_NUMBER() OVER w AS cumulative_trade_count,
                    EPOCH(end_date::TIMESTAMP - timestamp) AS time_to_expiry_seconds
                FROM read_parquet('{staging_dir}/*.parquet')
                WHERE hash(token_id) % {n_batches} = {batch_idx}
                WINDOW w AS (PARTITION BY token_id ORDER BY block_number, log_index)
                """

                batch_rows = copy_to_parquet(con, enriched_query, batch_dir)
                total_rows += batch_rows
                con.close()

        # Clean up staging
        shutil.rmtree(staging_dir, ignore_errors=True)

        print(f"  Polymarket: {total_rows:,} enriched trades")
        return total_rows

    def _build_token_market_map(
        self, con: duckdb.DuckDBPyConnection, markets_dir: Path
    ) -> dict[str, str]:
        """Build token_id -> market_id mapping from Polymarket markets data."""
        markets_df = con.execute(
            f"""
            SELECT id, clob_token_ids
            FROM '{markets_dir}/*.parquet'
            WHERE clob_token_ids IS NOT NULL
            """
        ).df()

        token_market_map: dict[str, str] = {}
        for _, row in markets_df.iterrows():
            market_id = row["id"]
            try:
                token_ids = json.loads(row["clob_token_ids"])
                for token_id in token_ids:
                    token_market_map[str(token_id)] = str(market_id)
            except (json.JSONDecodeError, TypeError):
                continue

        return token_market_map
