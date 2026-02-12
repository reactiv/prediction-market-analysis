"""Compare Brier score over market lifetime between Polymarket and Kalshi.

Brier score measures the mean squared error between predicted probabilities and outcomes.
Lower scores indicate better predictions. Perfect predictions = 0, random = 0.25.

This analysis calculates Brier score at different time intervals before market resolution
(1 to 100 hours) to show how prediction accuracy improves as resolution approaches.

NOTE: This analysis groups trades by "hours before resolution", which shows convergence
behavior (prices approaching certainty near resolution) rather than true calibration quality.
Low Brier scores near resolution (e.g., ~0.05 at 1 hour) reflect that outcomes are already
known, not that markets are well-calibrated. For true calibration metrics that measure
prediction quality at trade execution time, see cumulative_brier_score_over_time.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType

# Bucket size for block-to-timestamp approximation (10800 blocks ~ 6 hours at 2 sec/block)
BLOCK_BUCKET_SIZE = 10800


class BrierScoreOverTimeAnalysis(Analysis):
    """Compare Brier score at different hours before resolution between platforms."""

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
            name="brier_score_over_time",
            description="Compare Brier score by hours before resolution between platforms",
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
        with self.progress("Loading Kalshi Brier scores"):
            kalshi_df = self._load_kalshi_brier_scores()

        with self.progress("Loading Polymarket Brier scores"):
            polymarket_df = self._load_polymarket_brier_scores()

        # Merge results on hours_before_resolution
        output_df = pd.merge(
            kalshi_df,
            polymarket_df,
            on="hours_before_resolution",
            how="outer",
            suffixes=("_kalshi", "_polymarket"),
        ).sort_values("hours_before_resolution")

        fig = self._create_figure(output_df)
        chart = self._create_chart(output_df)

        return AnalysisOutput(figure=fig, data=output_df, chart=chart)

    def _load_kalshi_brier_scores(self) -> pd.DataFrame:
        """Load Kalshi trades and compute Brier score by hours before resolution."""
        con = duckdb.connect()

        df = con.execute(
            f"""
            WITH resolved_markets AS (
                SELECT ticker, result, close_time
                FROM '{self.kalshi_markets_dir}/*.parquet'
                WHERE result IN ('yes', 'no')
                  AND close_time IS NOT NULL
            ),
            trade_scores AS (
                SELECT
                    -- Hours before resolution (capped at 100)
                    LEAST(
                        FLOOR(EXTRACT(EPOCH FROM (m.close_time - t.created_time)) / 3600),
                        100
                    )::INT AS hours_before_resolution,
                    -- Brier score: (probability - outcome)^2
                    -- For taker side
                    POWER(
                        (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END / 100.0)
                        - (CASE WHEN t.taker_side = m.result THEN 1 ELSE 0 END),
                        2
                    ) AS taker_brier,
                    -- For maker side (counterparty)
                    POWER(
                        (CASE WHEN t.taker_side = 'yes' THEN t.no_price ELSE t.yes_price END / 100.0)
                        - (CASE WHEN t.taker_side != m.result THEN 1 ELSE 0 END),
                        2
                    ) AS maker_brier
                FROM '{self.kalshi_trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
                WHERE t.yes_price > 0
                  AND t.no_price > 0
                  AND t.created_time < m.close_time
            )
            SELECT
                hours_before_resolution,
                AVG(taker_brier) AS taker_brier_score,
                AVG(maker_brier) AS maker_brier_score,
                (AVG(taker_brier) + AVG(maker_brier)) / 2 AS brier_score,
                COUNT(*) AS trade_count
            FROM trade_scores
            WHERE hours_before_resolution >= 1 AND hours_before_resolution <= 100
            GROUP BY hours_before_resolution
            ORDER BY hours_before_resolution
            """
        ).df()

        return df[["hours_before_resolution", "brier_score", "trade_count"]]

    def _load_polymarket_brier_scores(self) -> pd.DataFrame:
        """Load Polymarket trades and compute Brier score by hours before resolution."""
        con = duckdb.connect()
        # Optimize for memory usage
        con.execute("SET preserve_insertion_order = false")

        # Build CTF token_id -> (won, end_timestamp) mapping for resolved markets
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

        # Create CTF token mapping table
        con.execute("CREATE TABLE token_resolution (token_id VARCHAR, won BOOLEAN, end_timestamp BIGINT)")
        if token_data:
            con.executemany(
                "INSERT INTO token_resolution VALUES (?, ?, ?)",
                [(k, v[0], v[1]) for k, v in token_data.items()],
            )

        # Load FPMM resolution from markets parquet files (if market_maker_address exists)
        fpmm_data: dict[str, tuple[int, int]] = {}
        schema_df = con.execute(f"DESCRIBE SELECT * FROM '{self.polymarket_markets_dir}/*.parquet' LIMIT 1").df()
        has_fpmm_field = "market_maker_address" in schema_df["column_name"].values

        if has_fpmm_field:
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

        # Create blocks lookup table (convert timestamp to epoch seconds)
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

        # Step 1: Materialize filtered CTF trades with computed fields
        # Split into two queries for efficient index usage (avoid CASE in join condition)
        con.execute(
            f"""
            CREATE TABLE ctf_trades AS
            -- Case 1: maker_asset_id = '0' (USDC), join on taker_asset_id
            SELECT
                t.block_number // {BLOCK_BUCKET_SIZE} AS bucket,
                tr.won,
                tr.end_timestamp,
                t.maker_amount::DOUBLE / t.taker_amount::DOUBLE AS price
            FROM '{self.polymarket_trades_dir}/*.parquet' t
            INNER JOIN token_resolution tr ON t.taker_asset_id = tr.token_id
            WHERE t.maker_asset_id = '0'
              AND t.taker_amount > 0
              AND t.maker_amount > 0

            UNION ALL

            -- Case 2: maker_asset_id != '0', join on maker_asset_id
            SELECT
                t.block_number // {BLOCK_BUCKET_SIZE} AS bucket,
                tr.won,
                tr.end_timestamp,
                t.taker_amount::DOUBLE / t.maker_amount::DOUBLE AS price
            FROM '{self.polymarket_trades_dir}/*.parquet' t
            INNER JOIN token_resolution tr ON t.maker_asset_id = tr.token_id
            WHERE t.maker_asset_id != '0'
              AND t.taker_amount > 0
              AND t.maker_amount > 0
            """
        )

        # Step 2: Compute Brier scores with block timestamps
        ctf_results = con.execute(
            """
            SELECT
                LEAST(FLOOR((t.end_timestamp - b.timestamp) / 3600), 100)::INT AS hours_before_resolution,
                SUM(POWER(t.price - (CASE WHEN t.won THEN 1.0 ELSE 0.0 END), 2)) AS sum_buyer_brier,
                SUM(POWER((1.0 - t.price) - (CASE WHEN NOT t.won THEN 1.0 ELSE 0.0 END), 2)) AS sum_seller_brier,
                COUNT(*) AS trade_count
            FROM ctf_trades t
            JOIN blocks b ON t.bucket = b.bucket
            WHERE b.timestamp < t.end_timestamp
            GROUP BY hours_before_resolution
            HAVING hours_before_resolution >= 1 AND hours_before_resolution <= 100
            """
        ).df()

        # Step 3: Legacy FPMM trades (if available)
        legacy_results = pd.DataFrame()
        if fpmm_data and self.polymarket_legacy_trades_dir.exists():
            con.execute("CREATE INDEX idx_fpmm ON fpmm_resolution(fpmm_address)")
            legacy_results = con.execute(
                f"""
                SELECT
                    LEAST(FLOOR((r.end_timestamp - COALESCE(t.timestamp, b.timestamp)) / 3600), 100)::INT AS hours_before_resolution,
                    SUM(POWER(
                        (t.amount::DOUBLE / t.outcome_tokens::DOUBLE)
                        - (CASE WHEN t.outcome_index = r.winning_outcome THEN 1.0 ELSE 0.0 END),
                        2
                    )) AS sum_buyer_brier,
                    SUM(POWER(
                        (1.0 - t.amount::DOUBLE / t.outcome_tokens::DOUBLE)
                        - (CASE WHEN t.outcome_index != r.winning_outcome THEN 1.0 ELSE 0.0 END),
                        2
                    )) AS sum_seller_brier,
                    COUNT(*) AS trade_count
                FROM '{self.polymarket_legacy_trades_dir}/*.parquet' t
                INNER JOIN fpmm_resolution r ON LOWER(t.fpmm_address) = r.fpmm_address
                JOIN blocks b ON t.block_number // {BLOCK_BUCKET_SIZE} = b.bucket
                WHERE t.outcome_tokens::DOUBLE > 0
                  AND COALESCE(t.timestamp, b.timestamp) < r.end_timestamp
                GROUP BY hours_before_resolution
                HAVING hours_before_resolution >= 1 AND hours_before_resolution <= 100
                """
            ).df()

        # Combine CTF and legacy results
        if legacy_results.empty:
            combined = ctf_results
        else:
            combined = pd.concat([ctf_results, legacy_results])
            combined = (
                combined.groupby("hours_before_resolution")
                .agg(
                    {
                        "sum_buyer_brier": "sum",
                        "sum_seller_brier": "sum",
                        "trade_count": "sum",
                    }
                )
                .reset_index()
            )

        # Compute final Brier scores
        if combined.empty:
            return pd.DataFrame({"hours_before_resolution": [], "brier_score": [], "trade_count": []})

        combined["brier_score"] = (
            combined["sum_buyer_brier"] / combined["trade_count"]
            + combined["sum_seller_brier"] / combined["trade_count"]
        ) / 2

        return combined[["hours_before_resolution", "brier_score", "trade_count"]].sort_values(
            "hours_before_resolution"
        )

    def _create_figure(self, df: pd.DataFrame) -> plt.Figure:
        """Create the matplotlib figure."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot Kalshi
        kalshi_mask = df["brier_score_kalshi"].notna()
        if kalshi_mask.any():
            ax.plot(
                df.loc[kalshi_mask, "hours_before_resolution"],
                df.loc[kalshi_mask, "brier_score_kalshi"],
                color="#4C72B0",
                linewidth=2,
                label="Kalshi",
            )

        # Plot Polymarket
        poly_mask = df["brier_score_polymarket"].notna()
        if poly_mask.any():
            ax.plot(
                df.loc[poly_mask, "hours_before_resolution"],
                df.loc[poly_mask, "brier_score_polymarket"],
                color="#55A868",
                linewidth=2,
                label="Polymarket",
            )

        ax.set_xlabel("Hours Before Resolution")
        ax.set_ylabel("Brier Score")
        ax.set_title("Brier Score by Hours Before Resolution: Polymarket vs Kalshi")

        # Invert x-axis so time flows left to right (100 hours ago -> resolution)
        ax.invert_xaxis()

        # Add reference line for random guessing (0.25)
        ax.axhline(
            y=0.25,
            color="#D65F5F",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label="Random guessing (0.25)",
        )

        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(100, 1)

        plt.tight_layout()
        return fig

    def _create_chart(self, df: pd.DataFrame) -> ChartConfig:
        """Create the chart configuration for web display."""
        chart_data = []
        for _, row in df.iterrows():
            entry = {"hours_before_resolution": int(row["hours_before_resolution"])}
            if pd.notna(row.get("brier_score_kalshi")):
                entry["kalshi"] = round(row["brier_score_kalshi"], 4)
            if pd.notna(row.get("brier_score_polymarket")):
                entry["polymarket"] = round(row["brier_score_polymarket"], 4)
            chart_data.append(entry)

        return ChartConfig(
            type=ChartType.LINE,
            data=chart_data,
            xKey="hours_before_resolution",
            yKeys=["kalshi", "polymarket"],
            title="Brier Score by Hours Before Resolution: Polymarket vs Kalshi",
            xLabel="Hours Before Resolution",
            yLabel="Brier Score",
            colors={"kalshi": "#4C72B0", "polymarket": "#55A868"},
        )
