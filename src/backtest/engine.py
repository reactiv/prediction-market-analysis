"""BacktestRunner: orchestrates hold-to-resolution strategy evaluation.

Assembles platform adapter SQL, applies strategy filters, computes
per-trade returns with fees, writes results and metrics.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb

from src.backtest.metrics import compute_metrics
from src.backtest.platforms.kalshi import KalshiAdapter
from src.backtest.returns import extract_return_sequence
from src.backtest.strategies._base import StrategyDefinition
from src.transforms._util import get_tmp_dir


class BacktestRunner:
    """Run a hold-to-resolution backtest using DuckDB-native SQL."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or Path(__file__).parent.parent.parent
        self.output_base = self.base_dir / "data" / "backtests"
        self._adapters = {
            "kalshi": KalshiAdapter(self.base_dir),
        }

    def run(self, strategy: StrategyDefinition, force: bool = False) -> Path:
        """Execute a backtest and write results.

        Returns path to the output directory containing:
          - returns.parquet        (full per-trade log)
          - return_sequence.parquet (ordered for Monte Carlo)
          - manifest.json          (config + summary metrics)
        """
        output_dir = self.output_base / strategy.name
        manifest_path = output_dir / "manifest.json"

        if not force and manifest_path.exists():
            print(f"Skipping {strategy.name}: already completed (use --force)")
            return output_dir

        output_dir.mkdir(parents=True, exist_ok=True)

        con = duckdb.connect()
        tmp_dir = get_tmp_dir()
        con.execute(f"SET temp_directory='{tmp_dir}'")
        con.execute("SET memory_limit='20GB'")
        con.execute("SET preserve_insertion_order=false")

        start = time.time()

        # Step 1: Build unified resolved trades SQL
        platform_sqls = []
        for platform_name in strategy.platforms:
            adapter = self._adapters[platform_name]
            platform_sqls.append(adapter.resolved_trades_sql(con))
        union_sql = " UNION ALL ".join(f"({sql})" for sql in platform_sqls)

        # Step 2: Build return computation SQL
        return_sql = self._build_return_sql(strategy, union_sql)

        # Step 3: Write returns
        returns_path = output_dir / "returns.parquet"
        con.execute(f"COPY ({return_sql}) TO '{returns_path}' (FORMAT PARQUET)")

        row_count = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{returns_path}')"
        ).fetchone()[0]
        print(f"  {strategy.name}: {row_count:,} qualifying trades")

        # Step 4: Compute metrics
        metrics = compute_metrics(con, returns_path)

        # Step 5: Extract return sequence for Monte Carlo
        extract_return_sequence(con, returns_path)

        elapsed = time.time() - start

        # Step 6: Write manifest
        manifest_data = {
            "strategy": strategy.name,
            "description": strategy.description,
            "platforms": strategy.platforms,
            "where_clause": strategy.where_clause,
            "take_side": strategy.take_side,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": round(elapsed, 1),
            "qualifying_trades": row_count,
            **metrics,
        }
        manifest_path.write_text(json.dumps(manifest_data, indent=2, default=str))
        print(f"  Metrics: win_rate={metrics['win_rate']:.4f}, "
              f"avg_return={metrics['avg_return_cents']:.4f}c, "
              f"sharpe={metrics['sharpe_ratio']:.4f}")
        print(f"  Completed in {elapsed:.1f}s → {output_dir}")

        con.close()
        return output_dir

    def _build_return_sql(
        self, strategy: StrategyDefinition, union_sql: str
    ) -> str:
        """Build SQL computing per-trade returns with fees."""
        fee_expr = strategy.fee_model.fee_sql("entry_price")

        if strategy.take_side == "no":
            # We buy NO at (100 - yes_price). Taker bought YES at yes_price.
            # entry_price = taker's YES price (what they paid).
            # Our cost = 100 - entry_price.
            # If result='no': payout=100, profit = entry_price cents
            # If result='yes': payout=0, loss = -(100 - entry_price) cents
            side_sql = "'no'"
            won_sql = "result = 'no'"
            gross_sql = (
                "CASE WHEN result = 'no' "
                "THEN entry_price "
                "ELSE -(100.0 - entry_price) "
                "END"
            )
        elif strategy.take_side == "yes":
            side_sql = "'yes'"
            won_sql = "result = 'yes'"
            gross_sql = (
                "CASE WHEN result = 'yes' "
                "THEN (100.0 - entry_price) "
                "ELSE -entry_price "
                "END"
            )
        elif strategy.take_side == "taker":
            side_sql = "taker_side"
            won_sql = "taker_side = result"
            gross_sql = (
                "CASE WHEN taker_side = result "
                "THEN (100.0 - entry_price) "
                "ELSE -entry_price "
                "END"
            )
        else:
            raise ValueError(f"Unknown take_side: {strategy.take_side}")

        extra_cols = f", {strategy.extra_columns}" if strategy.extra_columns else ""

        return f"""
        WITH resolved_trades AS (
            {union_sql}
        ),
        filtered AS (
            SELECT rt.*
                {extra_cols}
            FROM resolved_trades rt
            {strategy.extra_joins}
            WHERE {strategy.where_clause}
        )
        SELECT
            trade_id,
            market_id,
            platform,
            timestamp,
            entry_price,
            {side_sql}              AS side_taken,
            result,
            ({won_sql})::BOOLEAN    AS won,
            ({gross_sql})           AS gross_return_cents,
            ({fee_expr})            AS fee_cents,
            ({gross_sql}) - ({fee_expr}) AS net_return_cents,
            contracts,
            time_to_expiry_s,
            category,
            event_ticker
        FROM filtered
        ORDER BY timestamp
        """
