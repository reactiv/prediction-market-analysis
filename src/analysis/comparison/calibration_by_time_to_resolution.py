"""Compare calibration accuracy by time-to-resolution between Polymarket and Kalshi.

This analysis examines how prediction accuracy varies based on how far from resolution
trades occur. Trades are bucketed by time-to-resolution (1 day, 1 week, 1 month, etc.)
and calibration is computed separately for each bucket.

Key insight: Markets may be well-calibrated close to resolution (when outcomes are
nearly certain) but poorly calibrated for trades made weeks or months in advance.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType

# Bucket size for block-to-timestamp approximation (10800 blocks ~ 6 hours at 2 sec/block)
BLOCK_BUCKET_SIZE = 10800

# Time-to-resolution buckets in hours
TTR_BUCKETS = [
    (0, 24, "< 1 day"),
    (24, 72, "1-3 days"),
    (72, 168, "3-7 days"),
    (168, 336, "1-2 weeks"),
    (336, 720, "2-4 weeks"),
    (720, 2160, "1-3 months"),
    (2160, float("inf"), "> 3 months"),
]


class CalibrationByTimeToResolutionAnalysis(Analysis):
    """Compare calibration accuracy by time-to-resolution between platforms."""

    def __init__(
        self,
        kalshi_trades_dir: Path | str | None = None,
        kalshi_markets_dir: Path | str | None = None,
        polymarket_trades_dir: Path | str | None = None,
        polymarket_legacy_trades_dir: Path | str | None = None,
        polymarket_markets_dir: Path | str | None = None,
        polymarket_blocks_dir: Path | str | None = None,
        collateral_lookup_path: Path | str | None = None,
    ):
        super().__init__(
            name="calibration_by_time_to_resolution",
            description="Compare calibration accuracy by time-to-resolution between platforms",
        )
        base_dir = Path(__file__).parent.parent.parent.parent

        # Kalshi paths
        self.kalshi_trades_dir = Path(kalshi_trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.kalshi_markets_dir = Path(kalshi_markets_dir or base_dir / "data" / "kalshi" / "markets")

        # Polymarket paths
        self.polymarket_trades_dir = Path(polymarket_trades_dir or base_dir / "data" / "polymarket" / "trades")
        self.polymarket_legacy_trades_dir = Path(
            polymarket_legacy_trades_dir or base_dir / "data" / "polymarket" / "legacy_trades"
        )
        self.polymarket_markets_dir = Path(polymarket_markets_dir or base_dir / "data" / "polymarket" / "markets")
        self.polymarket_blocks_dir = Path(polymarket_blocks_dir or base_dir / "data" / "polymarket" / "blocks")
        self.collateral_lookup_path = Path(
            collateral_lookup_path or base_dir / "data" / "polymarket" / "fpmm_collateral_lookup.json"
        )

    def run(self) -> AnalysisOutput:
        """Execute the analysis and return outputs."""
        with self.progress("Loading Kalshi calibration by TTR"):
            kalshi_df = self._load_kalshi_calibration()

        with self.progress("Loading Polymarket calibration by TTR"):
            polymarket_df = self._load_polymarket_calibration()

        # Merge results
        output_df = pd.merge(
            kalshi_df,
            polymarket_df,
            on=["bucket_label", "bucket_order"],
            how="outer",
            suffixes=("_kalshi", "_polymarket"),
        ).sort_values("bucket_order")

        fig = self._create_figure(output_df)
        chart = self._create_chart(output_df)

        return AnalysisOutput(figure=fig, data=output_df, chart=chart)

    def _get_bucket_case_sql(self, hours_col: str) -> str:
        """Generate SQL CASE statement for TTR bucketing."""
        cases = []
        for low, high, label in TTR_BUCKETS:
            if high == float("inf"):
                cases.append(f"WHEN {hours_col} >= {low} THEN '{label}'")
            else:
                cases.append(f"WHEN {hours_col} >= {low} AND {hours_col} < {high} THEN '{label}'")
        return "CASE " + " ".join(cases) + " END"

    def _get_bucket_order_sql(self, hours_col: str) -> str:
        """Generate SQL CASE statement for bucket ordering."""
        cases = []
        for i, (low, high, _) in enumerate(TTR_BUCKETS):
            if high == float("inf"):
                cases.append(f"WHEN {hours_col} >= {low} THEN {i}")
            else:
                cases.append(f"WHEN {hours_col} >= {low} AND {hours_col} < {high} THEN {i}")
        return "CASE " + " ".join(cases) + " END"

    def _load_kalshi_calibration(self) -> pd.DataFrame:
        """Load Kalshi trades and compute calibration by time-to-resolution bucket."""
        con = duckdb.connect()

        bucket_case = self._get_bucket_case_sql("hours_to_resolution")
        order_case = self._get_bucket_order_sql("hours_to_resolution")

        df = con.execute(
            f"""
            WITH resolved_markets AS (
                SELECT ticker, result, close_time
                FROM '{self.kalshi_markets_dir}/*.parquet'
                WHERE result IN ('yes', 'no')
                  AND close_time IS NOT NULL
            ),
            trade_positions AS (
                -- Buyer side (taker)
                SELECT
                    EXTRACT(EPOCH FROM (m.close_time - t.created_time)) / 3600 AS hours_to_resolution,
                    CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS price,
                    CASE WHEN t.taker_side = m.result THEN 1 ELSE 0 END AS won
                FROM '{self.kalshi_trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
                WHERE t.yes_price > 0 AND t.no_price > 0
                  AND t.created_time < m.close_time

                UNION ALL

                -- Seller side (counterparty)
                SELECT
                    EXTRACT(EPOCH FROM (m.close_time - t.created_time)) / 3600 AS hours_to_resolution,
                    CASE WHEN t.taker_side = 'yes' THEN t.no_price ELSE t.yes_price END AS price,
                    CASE WHEN t.taker_side != m.result THEN 1 ELSE 0 END AS won
                FROM '{self.kalshi_trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
                WHERE t.yes_price > 0 AND t.no_price > 0
                  AND t.created_time < m.close_time
            ),
            bucketed AS (
                SELECT
                    {bucket_case} AS bucket_label,
                    {order_case} AS bucket_order,
                    price,
                    won
                FROM trade_positions
                WHERE hours_to_resolution >= 0
            ),
            calibration_by_price AS (
                SELECT
                    bucket_label,
                    bucket_order,
                    price,
                    COUNT(*) AS total,
                    SUM(won) AS wins
                FROM bucketed
                WHERE price >= 1 AND price <= 99
                GROUP BY bucket_label, bucket_order, price
            )
            SELECT
                bucket_label,
                bucket_order,
                SUM(total) AS total_trades,
                AVG(ABS(100.0 * wins / total - price)) AS mean_absolute_deviation,
                SUM(total * POWER(wins::DOUBLE / total - price / 100.0, 2)) / SUM(total) AS brier_score
            FROM calibration_by_price
            GROUP BY bucket_label, bucket_order
            ORDER BY bucket_order
            """
        ).df()

        return df

    def _load_polymarket_calibration(self) -> pd.DataFrame:
        """Load Polymarket trades and compute calibration by time-to-resolution bucket."""
        con = duckdb.connect()

        # Build CTF token_id -> (won, end_timestamp) mapping
        markets_df = con.execute(
            f"""
            SELECT id, clob_token_ids, outcome_prices, end_date
            FROM '{self.polymarket_markets_dir}/*.parquet'
            WHERE closed = true AND end_date IS NOT NULL
            """
        ).df()

        token_data: dict[str, tuple[bool, int]] = {}
        for _, row in markets_df.iterrows():
            try:
                token_ids = json.loads(row["clob_token_ids"]) if row["clob_token_ids"] else None
                prices = json.loads(row["outcome_prices"]) if row["outcome_prices"] else None
                end_date = row["end_date"]
                if not token_ids or not prices or len(token_ids) != 2 or len(prices) != 2:
                    continue
                if pd.isna(end_date):
                    continue
                end_ts = int(end_date.timestamp())
                p0, p1 = float(prices[0]), float(prices[1])
                if p0 > 0.99 and p1 < 0.01:
                    token_data[token_ids[0]] = (True, end_ts)
                    token_data[token_ids[1]] = (False, end_ts)
                elif p0 < 0.01 and p1 > 0.99:
                    token_data[token_ids[0]] = (False, end_ts)
                    token_data[token_ids[1]] = (True, end_ts)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

        con.execute("CREATE TABLE token_resolution (token_id VARCHAR, won BOOLEAN, end_timestamp BIGINT)")
        if token_data:
            con.executemany(
                "INSERT INTO token_resolution VALUES (?, ?, ?)",
                [(k, v[0], v[1]) for k, v in token_data.items()],
            )

        # Load FPMM resolution
        fpmm_data: dict[str, tuple[int, int]] = {}
        fpmm_markets_df = con.execute(
            f"""
            SELECT market_maker_address, outcome_prices, end_date
            FROM '{self.polymarket_markets_dir}/*.parquet'
            WHERE market_maker_address IS NOT NULL AND end_date IS NOT NULL
            """
        ).df()

        for _, row in fpmm_markets_df.iterrows():
            fpmm_addr = row["market_maker_address"]
            try:
                prices = json.loads(row["outcome_prices"]) if row["outcome_prices"] else None
                end_date = row["end_date"]
                if not prices or len(prices) != 2 or pd.isna(end_date):
                    continue
                end_ts = int(end_date.timestamp())
                p0, p1 = float(prices[0]), float(prices[1])
                if p0 > 0.99 and p1 < 0.01:
                    fpmm_data[fpmm_addr.lower()] = (0, end_ts)
                elif p0 < 0.01 and p1 > 0.99:
                    fpmm_data[fpmm_addr.lower()] = (1, end_ts)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

        # Filter to USDC markets only
        if self.collateral_lookup_path.exists():
            with open(self.collateral_lookup_path) as f:
                collateral_lookup = json.load(f)
            usdc_markets = {
                addr.lower() for addr, info in collateral_lookup.items() if info["collateral_symbol"] == "USDC"
            }
            fpmm_data = {k: v for k, v in fpmm_data.items() if k in usdc_markets}

        con.execute("CREATE TABLE fpmm_resolution (fpmm_address VARCHAR, winning_outcome BIGINT, end_timestamp BIGINT)")
        if fpmm_data:
            con.executemany(
                "INSERT INTO fpmm_resolution VALUES (?, ?, ?)",
                [(k, v[0], v[1]) for k, v in fpmm_data.items()],
            )

        # Create blocks lookup table
        con.execute(
            f"""
            CREATE TABLE blocks AS
            SELECT
                block_number // {BLOCK_BUCKET_SIZE} AS bucket,
                EXTRACT(EPOCH FROM FIRST(timestamp::TIMESTAMP))::BIGINT AS timestamp
            FROM '{self.polymarket_blocks_dir}/*.parquet'
            GROUP BY block_number // {BLOCK_BUCKET_SIZE}
            """
        )

        bucket_case = self._get_bucket_case_sql("hours_to_resolution")
        order_case = self._get_bucket_order_sql("hours_to_resolution")

        # CTF trades
        ctf_query = f"""
            SELECT
                (tr.end_timestamp - b.timestamp) / 3600.0 AS hours_to_resolution,
                CASE
                    WHEN t.maker_asset_id = '0' THEN ROUND(100.0 * t.maker_amount / t.taker_amount)
                    ELSE ROUND(100.0 * t.taker_amount / t.maker_amount)
                END AS price,
                CASE WHEN tr.won THEN 1 ELSE 0 END AS won
            FROM '{self.polymarket_trades_dir}/*.parquet' t
            INNER JOIN token_resolution tr ON (
                CASE WHEN t.maker_asset_id = '0' THEN t.taker_asset_id ELSE t.maker_asset_id END = tr.token_id
            )
            JOIN blocks b ON t.block_number // {BLOCK_BUCKET_SIZE} = b.bucket
            WHERE t.taker_amount > 0 AND t.maker_amount > 0
              AND b.timestamp < tr.end_timestamp

            UNION ALL

            SELECT
                (tr.end_timestamp - b.timestamp) / 3600.0 AS hours_to_resolution,
                CASE
                    WHEN t.maker_asset_id = '0' THEN ROUND(100.0 - 100.0 * t.maker_amount / t.taker_amount)
                    ELSE ROUND(100.0 - 100.0 * t.taker_amount / t.maker_amount)
                END AS price,
                CASE WHEN NOT tr.won THEN 1 ELSE 0 END AS won
            FROM '{self.polymarket_trades_dir}/*.parquet' t
            INNER JOIN token_resolution tr ON (
                CASE WHEN t.maker_asset_id = '0' THEN t.taker_asset_id ELSE t.maker_asset_id END = tr.token_id
            )
            JOIN blocks b ON t.block_number // {BLOCK_BUCKET_SIZE} = b.bucket
            WHERE t.taker_amount > 0 AND t.maker_amount > 0
              AND b.timestamp < tr.end_timestamp
        """

        # Legacy FPMM trades
        legacy_query = ""
        if fpmm_data and self.polymarket_legacy_trades_dir.exists():
            legacy_query = f"""
                UNION ALL

                SELECT
                    (r.end_timestamp - COALESCE(t.timestamp, b.timestamp)) / 3600.0 AS hours_to_resolution,
                    ROUND(100.0 * t.amount::DOUBLE / t.outcome_tokens::DOUBLE) AS price,
                    CASE WHEN t.outcome_index = r.winning_outcome THEN 1 ELSE 0 END AS won
                FROM '{self.polymarket_legacy_trades_dir}/*.parquet' t
                INNER JOIN fpmm_resolution r ON LOWER(t.fpmm_address) = r.fpmm_address
                JOIN blocks b ON t.block_number // {BLOCK_BUCKET_SIZE} = b.bucket
                WHERE t.outcome_tokens::DOUBLE > 0
                  AND COALESCE(t.timestamp, b.timestamp) < r.end_timestamp

                UNION ALL

                SELECT
                    (r.end_timestamp - COALESCE(t.timestamp, b.timestamp)) / 3600.0 AS hours_to_resolution,
                    ROUND(100.0 - 100.0 * t.amount::DOUBLE / t.outcome_tokens::DOUBLE) AS price,
                    CASE WHEN t.outcome_index != r.winning_outcome THEN 1 ELSE 0 END AS won
                FROM '{self.polymarket_legacy_trades_dir}/*.parquet' t
                INNER JOIN fpmm_resolution r ON LOWER(t.fpmm_address) = r.fpmm_address
                JOIN blocks b ON t.block_number // {BLOCK_BUCKET_SIZE} = b.bucket
                WHERE t.outcome_tokens::DOUBLE > 0
                  AND COALESCE(t.timestamp, b.timestamp) < r.end_timestamp
            """

        df = con.execute(
            f"""
            WITH trade_positions AS (
                {ctf_query}
                {legacy_query}
            ),
            bucketed AS (
                SELECT
                    {bucket_case} AS bucket_label,
                    {order_case} AS bucket_order,
                    price,
                    won
                FROM trade_positions
                WHERE hours_to_resolution >= 0
            ),
            calibration_by_price AS (
                SELECT
                    bucket_label,
                    bucket_order,
                    price,
                    COUNT(*) AS total,
                    SUM(won) AS wins
                FROM bucketed
                WHERE price >= 1 AND price <= 99
                GROUP BY bucket_label, bucket_order, price
            )
            SELECT
                bucket_label,
                bucket_order,
                SUM(total) AS total_trades,
                AVG(ABS(100.0 * wins / total - price)) AS mean_absolute_deviation,
                SUM(total * POWER(wins::DOUBLE / total - price / 100.0, 2)) / SUM(total) AS brier_score
            FROM calibration_by_price
            GROUP BY bucket_label, bucket_order
            ORDER BY bucket_order
            """
        ).df()

        return df

    def _create_figure(self, df: pd.DataFrame) -> plt.Figure:
        """Create the matplotlib figure."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        labels = df["bucket_label"].tolist()
        x = np.arange(len(labels))
        width = 0.35

        # MAD subplot
        ax1 = axes[0]
        kalshi_mad = df["mean_absolute_deviation_kalshi"].fillna(0).tolist()
        poly_mad = df["mean_absolute_deviation_polymarket"].fillna(0).tolist()

        ax1.bar(x - width / 2, kalshi_mad, width, label="Kalshi", color="#4C72B0")
        ax1.bar(x + width / 2, poly_mad, width, label="Polymarket", color="#55A868")
        ax1.set_xlabel("Time to Resolution")
        ax1.set_ylabel("Mean Absolute Deviation (%)")
        ax1.set_title("Calibration (MAD) by Time to Resolution")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha="right")
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis="y")
        ax1.axhline(y=0, color="#D65F5F", linestyle="--", linewidth=1, alpha=0.7)

        # Brier score subplot
        ax2 = axes[1]
        kalshi_brier = df["brier_score_kalshi"].fillna(0).tolist()
        poly_brier = df["brier_score_polymarket"].fillna(0).tolist()

        ax2.bar(x - width / 2, kalshi_brier, width, label="Kalshi", color="#4C72B0")
        ax2.bar(x + width / 2, poly_brier, width, label="Polymarket", color="#55A868")
        ax2.set_xlabel("Time to Resolution")
        ax2.set_ylabel("Brier Score")
        ax2.set_title("Brier Score by Time to Resolution")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha="right")
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")
        ax2.axhline(y=0.25, color="#D65F5F", linestyle="--", linewidth=1, alpha=0.7, label="Random (0.25)")

        plt.tight_layout()
        return fig

    def _create_chart(self, df: pd.DataFrame) -> ChartConfig:
        """Create the chart configuration for web display."""
        chart_data = []
        for _, row in df.iterrows():
            entry = {"bucket": row["bucket_label"]}
            if pd.notna(row.get("mean_absolute_deviation_kalshi")):
                entry["kalshi_mad"] = round(row["mean_absolute_deviation_kalshi"], 2)
            if pd.notna(row.get("mean_absolute_deviation_polymarket")):
                entry["polymarket_mad"] = round(row["mean_absolute_deviation_polymarket"], 2)
            if pd.notna(row.get("brier_score_kalshi")):
                entry["kalshi_brier"] = round(row["brier_score_kalshi"], 4)
            if pd.notna(row.get("brier_score_polymarket")):
                entry["polymarket_brier"] = round(row["brier_score_polymarket"], 4)
            chart_data.append(entry)

        return ChartConfig(
            type=ChartType.BAR,
            data=chart_data,
            xKey="bucket",
            yKeys=["kalshi_mad", "polymarket_mad"],
            title="Calibration by Time to Resolution: Polymarket vs Kalshi",
            yUnit=UnitType.PERCENT,
            xLabel="Time to Resolution",
            yLabel="Mean Absolute Deviation (%)",
            colors={"kalshi_mad": "#4C72B0", "polymarket_mad": "#55A868"},
        )
