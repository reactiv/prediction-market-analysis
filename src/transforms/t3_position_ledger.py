"""T3: Per-address position ledger and summary stats for Polymarket."""

from __future__ import annotations

import json
import os
import shutil

import duckdb
import pandas as pd

from src.transforms._base import Transform
from src.transforms._util import get_tmp_dir


class T3PositionLedger(Transform):
    def __init__(self):
        super().__init__(
            name="t3",
            description="Per-address position ledger (Polymarket)",
            dependencies=[],
        )

    def run(self):
        self.ensure_output_dir()
        ledger_rows = self._build_position_ledger()
        summary_rows = self._build_address_summary()
        self.write_manifest(
            {
                "ledger_rows": ledger_rows,
                "summary_rows": summary_rows,
            }
        )

    # ------------------------------------------------------------------
    # Phase 1 — Position Ledger (two-stage: stage then window)
    # ------------------------------------------------------------------

    def _build_position_ledger(self) -> int:
        """Build the two-sided position ledger from raw trades.

        Each trade produces two ledger entries (buyer + seller).  Running
        cumulative positions and cost bases are computed via DuckDB window
        functions, and the result is written to Hive-partitioned Parquet
        files keyed on ``address_prefix`` (first two hex chars after 0x).

        Stage 1: Single scan of all trades → write trade entries (no window
        functions) to staging, partitioned by first hex nibble.
        Stage 2: For each nibble partition, apply window functions and write
        to final output.
        """
        trades_glob = os.path.join(self.base_dir, "data/polymarket/trades/*.parquet")
        blocks_glob = os.path.join(self.base_dir, "data/polymarket/blocks/*.parquet")
        ledger_dir = os.path.join(self.output_dir, "position_ledger")
        staging_dir = os.path.join(self.output_dir, "_staging_entries")
        tmp_dir = get_tmp_dir()

        # --- Stage 1: Materialize trade entries (no window functions) ---
        # Skip if staging already exists from a previous run
        if os.path.isdir(os.path.join(staging_dir, "nibble=f")):
            print(f"Staging exists, skipping stage 1")
        else:
            os.makedirs(staging_dir, exist_ok=True)
            con = duckdb.connect()
            try:
                con.execute(f"SET temp_directory = '{tmp_dir}'")
                con.execute("SET memory_limit = '16GB'")
                con.execute("SET preserve_insertion_order = false")
                con.execute("SET threads = 4")

                con.execute(
                    f"CREATE VIEW trades AS SELECT * FROM read_parquet('{trades_glob}')"
                )
                con.execute(
                    f"CREATE VIEW blocks AS SELECT * FROM read_parquet('{blocks_glob}')"
                )

                staging_sql = f"""
                COPY (
                    SELECT
                        address,
                        token_id,
                        direction,
                        qty,
                        cost_usdc,
                        block_number,
                        log_index,
                        trade_timestamp,
                        LOWER(SUBSTRING(address, 3, 1)) AS nibble,
                        LOWER(SUBSTRING(address, 3, 2)) AS address_prefix
                    FROM (
                        -- Buyer side
                        SELECT
                            CASE WHEN t.maker_asset_id = '0' THEN t.maker ELSE t.taker END AS address,
                            CASE WHEN t.maker_asset_id = '0' THEN t.taker_asset_id ELSE t.maker_asset_id END AS token_id,
                            'BUY' AS direction,
                            CASE WHEN t.maker_asset_id = '0' THEN t.taker_amount::DOUBLE ELSE t.maker_amount::DOUBLE END AS qty,
                            CASE WHEN t.maker_asset_id = '0' THEN t.maker_amount::DOUBLE ELSE t.taker_amount::DOUBLE END AS cost_usdc,
                            t.block_number,
                            t.log_index,
                            COALESCE(t.timestamp, b.timestamp) AS trade_timestamp
                        FROM trades t
                        LEFT JOIN blocks b ON t.block_number = b.block_number

                        UNION ALL

                        -- Seller side
                        SELECT
                            CASE WHEN t.maker_asset_id = '0' THEN t.taker ELSE t.maker END AS address,
                            CASE WHEN t.maker_asset_id = '0' THEN t.taker_asset_id ELSE t.maker_asset_id END AS token_id,
                            'SELL' AS direction,
                            -(CASE WHEN t.maker_asset_id = '0' THEN t.taker_amount::DOUBLE ELSE t.maker_amount::DOUBLE END) AS qty,
                            -(CASE WHEN t.maker_asset_id = '0' THEN t.maker_amount::DOUBLE ELSE t.taker_amount::DOUBLE END) AS cost_usdc,
                            t.block_number,
                            t.log_index,
                            COALESCE(t.timestamp, b.timestamp) AS trade_timestamp
                        FROM trades t
                        LEFT JOIN blocks b ON t.block_number = b.block_number
                    ) entries
                ) TO '{staging_dir}' (
                    FORMAT PARQUET,
                    PARTITION_BY (nibble),
                    OVERWRITE_OR_IGNORE
                )
                """

                with self.progress("Stage 1: Materializing trade entries"):
                    con.execute(staging_sql)
            finally:
                con.close()

        # --- Stage 2: Window functions per nibble partition ---
        hex_nibbles = [f"{i:x}" for i in range(16)]
        os.makedirs(ledger_dir, exist_ok=True)

        for nibble_idx, nibble in enumerate(hex_nibbles):
            nibble_glob = os.path.join(staging_dir, f"nibble={nibble}", "*.parquet")
            output_file = os.path.join(ledger_dir, f"ledger_{nibble}.parquet")

            if os.path.exists(output_file):
                continue  # Resume support: skip already-written batches

            con = duckdb.connect()
            try:
                con.execute(f"SET temp_directory = '{tmp_dir}'")
                con.execute("SET memory_limit = '28GB'")
                con.execute("SET preserve_insertion_order = false")
                con.execute("SET threads = 1")

                window_sql = f"""
                COPY (
                    SELECT
                        address,
                        token_id,
                        direction,
                        qty,
                        cost_usdc,
                        block_number,
                        log_index,
                        trade_timestamp,
                        SUM(qty) OVER w AS cumulative_position,
                        SUM(cost_usdc) OVER w AS cumulative_cost,
                        SUM(CASE WHEN qty > 0 THEN qty ELSE 0 END) OVER w AS cumulative_buy_qty,
                        SUM(CASE WHEN qty > 0 THEN cost_usdc ELSE 0 END) OVER w AS cumulative_buy_cost,
                        address_prefix
                    FROM read_parquet('{nibble_glob}')
                    WINDOW w AS (
                        PARTITION BY address, token_id
                        ORDER BY block_number, log_index
                        ROWS UNBOUNDED PRECEDING
                    )
                ) TO '{output_file}' (FORMAT PARQUET)
                """

                with self.progress(
                    f"Stage 2: Window functions (batch {nibble_idx + 1}/16, 0x{nibble}...)"
                ):
                    con.execute(window_sql)
            finally:
                con.close()

        # Clean up staging
        shutil.rmtree(staging_dir, ignore_errors=True)

        # Count total rows written.
        count_con = duckdb.connect()
        try:
            total_rows = count_con.execute(
                f"SELECT COUNT(*) FROM read_parquet('{ledger_dir}/*.parquet')"
            ).fetchone()[0]
        finally:
            count_con.close()

        return total_rows

    # ------------------------------------------------------------------
    # Phase 2 — Address Summary Stats
    # ------------------------------------------------------------------

    def _build_token_resolution_df(self, con: duckdb.DuckDBPyConnection) -> None:
        """Parse market data to build a token_id -> resolution mapping.

        Registers a temporary table ``token_resolution`` in *con* with
        columns: ``token_id``, ``market_id``, ``won`` (BOOLEAN).
        """
        markets_glob = os.path.join(
            self.base_dir, "data/polymarket/markets/*.parquet"
        )
        markets_df = con.execute(
            f"""
            SELECT id, clob_token_ids, outcome_prices
            FROM read_parquet('{markets_glob}')
            """
        ).fetchdf()

        rows: list[dict] = []
        for _, row in markets_df.iterrows():
            market_id = row["id"]

            # Parse clob_token_ids — JSON array of token id strings
            try:
                token_ids = json.loads(row["clob_token_ids"])
            except (json.JSONDecodeError, TypeError):
                continue

            # Parse outcome_prices — JSON array of price strings/floats
            try:
                prices = json.loads(row["outcome_prices"])
            except (json.JSONDecodeError, TypeError):
                prices = [None] * len(token_ids)

            for i, tid in enumerate(token_ids):
                price = None
                if prices and i < len(prices):
                    try:
                        price = float(prices[i])
                    except (ValueError, TypeError):
                        price = None

                # A token is considered a winner if its final price > 0.99
                won = price is not None and price > 0.99
                rows.append(
                    {
                        "token_id": str(tid),
                        "market_id": str(market_id),
                        "won": won,
                        "resolved": price is not None,
                    }
                )

        token_res_df = pd.DataFrame(rows)
        con.register("token_resolution", token_res_df)

    def _build_address_summary(self) -> int:
        """Aggregate per-address summary statistics from the position ledger.

        Processes one ledger file at a time (each contains all entries for
        a set of addresses) and concatenates the per-batch summaries.
        """
        ledger_dir = os.path.join(self.output_dir, "position_ledger")
        summary_path = os.path.join(self.output_dir, "address_summary.parquet")
        summary_staging = os.path.join(self.output_dir, "_summary_staging")
        tmp_dir = get_tmp_dir()

        os.makedirs(summary_staging, exist_ok=True)

        # Build token resolution mapping once (shared across batches).
        token_con = duckdb.connect()
        with self.progress("Building token resolution mapping"):
            self._build_token_resolution_df(token_con)
            token_res_df = token_con.execute("SELECT * FROM token_resolution").fetchdf()
        token_con.close()

        hex_nibbles = [f"{i:x}" for i in range(16)]

        for nibble_idx, nibble in enumerate(hex_nibbles):
            ledger_file = os.path.join(ledger_dir, f"ledger_{nibble}.parquet")
            batch_out = os.path.join(summary_staging, f"summary_{nibble}.parquet")

            if os.path.exists(batch_out):
                continue

            con = duckdb.connect()
            try:
                con.execute(f"SET temp_directory = '{tmp_dir}'")
                con.execute("SET memory_limit = '28GB'")
                con.execute("SET preserve_insertion_order = false")
                con.execute("SET threads = 1")

                con.register("token_resolution", token_res_df)
                con.execute(
                    f"CREATE VIEW ledger AS SELECT * FROM read_parquet('{ledger_file}')"
                )

                summary_sql = f"""
                COPY (
                    WITH position_final AS (
                        SELECT
                            l.address,
                            l.token_id,
                            l.cumulative_position AS final_position,
                            l.cumulative_cost AS final_cost,
                            l.cumulative_buy_cost AS final_buy_cost,
                            tr.market_id,
                            tr.won,
                            tr.resolved
                        FROM (
                            SELECT *,
                                ROW_NUMBER() OVER (
                                    PARTITION BY address, token_id
                                    ORDER BY block_number DESC, log_index DESC
                                ) AS rn
                            FROM ledger
                        ) l
                        LEFT JOIN token_resolution tr ON l.token_id = tr.token_id
                        WHERE l.rn = 1
                    ),
                    position_pnl AS (
                        SELECT
                            address,
                            token_id,
                            market_id,
                            resolved,
                            final_position,
                            final_cost,
                            final_buy_cost,
                            CASE
                                WHEN resolved THEN
                                    (CASE WHEN won THEN final_position ELSE 0 END) - final_cost
                                ELSE NULL
                            END AS realized_pnl,
                            CASE
                                WHEN resolved AND final_cost != 0 THEN
                                    CASE
                                        WHEN ((CASE WHEN won THEN final_position ELSE 0 END) - final_cost) > 0
                                        THEN 1
                                        ELSE 0
                                    END
                                ELSE NULL
                            END AS is_profitable
                        FROM position_final
                    ),
                    per_address_trades AS (
                        SELECT
                            l.address,
                            tr.market_id,
                            COUNT(*) AS trade_count,
                            SUM(ABS(l.cost_usdc)) AS abs_volume,
                            MIN(l.trade_timestamp) AS first_trade,
                            MAX(l.trade_timestamp) AS last_trade,
                            COUNT(DISTINCT CAST(l.trade_timestamp AS DATE)) AS active_days
                        FROM ledger l
                        LEFT JOIN token_resolution tr ON l.token_id = tr.token_id
                        GROUP BY l.address, tr.market_id
                    ),
                    per_address_agg AS (
                        SELECT
                            address,
                            COUNT(DISTINCT market_id) AS distinct_markets,
                            SUM(abs_volume) AS total_volume,
                            SUM(trade_count) AS total_trades,
                            MIN(first_trade) AS first_trade,
                            MAX(last_trade) AS last_trade,
                            SUM(active_days) AS active_days_approx
                        FROM per_address_trades
                        GROUP BY address
                    ),
                    per_address_hhi AS (
                        SELECT
                            address,
                            SUM(market_share * market_share) AS herfindahl
                        FROM (
                            SELECT
                                pat.address,
                                pat.abs_volume / NULLIF(paa.total_volume, 0) AS market_share
                            FROM per_address_trades pat
                            JOIN per_address_agg paa ON pat.address = paa.address
                        ) sub
                        GROUP BY address
                    ),
                    per_address_pnl AS (
                        SELECT
                            address,
                            SUM(realized_pnl) AS realized_pnl,
                            AVG(is_profitable) AS win_rate
                        FROM position_pnl
                        WHERE resolved
                        GROUP BY address
                    )
                    SELECT
                        a.address,
                        a.distinct_markets,
                        a.total_volume,
                        a.total_trades,
                        COALESCE(p.realized_pnl, 0) AS realized_pnl,
                        p.win_rate,
                        a.first_trade,
                        a.last_trade,
                        a.active_days_approx AS active_days,
                        COALESCE(h.herfindahl, 0) AS herfindahl,
                        CASE
                            WHEN a.total_trades > 0
                            THEN COALESCE(p.realized_pnl, 0) / a.total_trades
                            ELSE 0
                        END AS avg_return_per_trade,
                        CASE
                            WHEN a.distinct_markets > 0
                            THEN a.total_volume / a.distinct_markets
                            ELSE 0
                        END AS volume_per_market
                    FROM per_address_agg a
                    LEFT JOIN per_address_pnl p ON a.address = p.address
                    LEFT JOIN per_address_hhi h ON a.address = h.address
                ) TO '{batch_out}' (FORMAT PARQUET)
                """

                with self.progress(
                    f"Computing address summary (batch {nibble_idx + 1}/16, 0x{nibble}...)"
                ):
                    con.execute(summary_sql)
            finally:
                con.close()

        # Concatenate all batch summaries into final output.
        merge_con = duckdb.connect()
        try:
            staging_glob = os.path.join(summary_staging, "*.parquet")
            merge_con.execute(
                f"COPY (SELECT * FROM read_parquet('{staging_glob}')) "
                f"TO '{summary_path}' (FORMAT PARQUET)"
            )
            row_count = merge_con.execute(
                f"SELECT COUNT(*) FROM read_parquet('{summary_path}')"
            ).fetchone()[0]
        finally:
            merge_con.close()

        shutil.rmtree(summary_staging, ignore_errors=True)
        return row_count
