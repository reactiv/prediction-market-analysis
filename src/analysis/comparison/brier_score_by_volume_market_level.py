"""Compare market-level Brier score by volume bucket between Polymarket and Kalshi.

This analysis uses market-level aggregation (one observation per market) similar to
the Dune Analytics methodology.

IMPORTANT: We use the LAST TRADE timestamp as the resolution proxy for both platforms.
This ensures a fair comparison because:
- Kalshi's close_time is often set AFTER outcomes are known (sports scores)
- Polymarket's end_date is often set BEFORE outcomes are known (elections)

Using last trade as resolution proxy means we measure "price N hours before trading stopped"
which is consistent across both platforms.

Volume buckets:
- <1K
- 1-5K
- 5-10K
- 10-25K
- 25-50K
- 50-100K
- 100-250K
- 250-500K
- 500K-1M
- 1M+
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType

# Hours before resolution to sample price (24 hours = 1 day)
HOURS_BEFORE_RESOLUTION = 24

# Bucket size for block-to-timestamp approximation
BLOCK_BUCKET_SIZE = 10800

# Volume bucket definitions (lower bound, upper bound, label)
VOLUME_BUCKETS = [
    (0, 1_000, "<1K"),
    (1_000, 5_000, "1-5K"),
    (5_000, 10_000, "5-10K"),
    (10_000, 25_000, "10-25K"),
    (25_000, 50_000, "25-50K"),
    (50_000, 100_000, "50-100K"),
    (100_000, 250_000, "100-250K"),
    (250_000, 500_000, "250-500K"),
    (500_000, 1_000_000, "500K-1M"),
    (1_000_000, float("inf"), "1M+"),
]


class BrierScoreByVolumeMarketLevelAnalysis(Analysis):
    """Compare market-level Brier score by volume bucket between platforms."""

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
            name="brier_score_by_volume_market_level",
            description="Compare market-level Brier score by volume bucket between platforms",
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
        with self.progress("Loading Kalshi market-level Brier scores"):
            kalshi_df = self._load_kalshi_brier_scores()

        with self.progress("Loading Polymarket market-level Brier scores"):
            polymarket_df = self._load_polymarket_brier_scores()

        # Merge results on volume_bucket
        output_df = pd.merge(
            kalshi_df,
            polymarket_df,
            on="volume_bucket",
            how="outer",
            suffixes=("_kalshi", "_polymarket"),
        )

        # Sort by bucket order
        bucket_order = {label: i for i, (_, _, label) in enumerate(VOLUME_BUCKETS)}
        output_df["sort_order"] = output_df["volume_bucket"].map(bucket_order)
        output_df = output_df.sort_values("sort_order").drop(columns=["sort_order"])

        fig = self._create_figure(output_df)
        chart = self._create_chart(output_df)

        return AnalysisOutput(figure=fig, data=output_df, chart=chart)

    def _get_volume_bucket_case(self, volume_col: str) -> str:
        """Generate SQL CASE statement for volume bucketing."""
        cases = []
        for lower, upper, label in VOLUME_BUCKETS:
            if upper == float("inf"):
                cases.append(f"WHEN {volume_col} >= {lower} THEN '{label}'")
            else:
                cases.append(f"WHEN {volume_col} >= {lower} AND {volume_col} < {upper} THEN '{label}'")
        return "CASE " + " ".join(cases) + " END"

    def _load_kalshi_brier_scores(self) -> pd.DataFrame:
        """Load Kalshi market-level Brier scores using price near last trade."""
        con = duckdb.connect()

        volume_bucket_case = self._get_volume_bucket_case("m.volume")

        # Use LAST TRADE timestamp as resolution proxy (not close_time)
        # This makes comparison fair with Polymarket
        df = con.execute(
            f"""
            WITH resolved_markets AS (
                SELECT
                    ticker,
                    result,
                    volume
                FROM '{self.kalshi_markets_dir}/*.parquet'
                WHERE result IN ('yes', 'no')
                  AND volume IS NOT NULL
            ),
            -- Find the last trade timestamp for each market
            market_last_trade AS (
                SELECT
                    t.ticker,
                    MAX(t.created_time) AS last_trade_time
                FROM '{self.kalshi_trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
                WHERE t.yes_price > 0 AND t.yes_price < 100
                GROUP BY t.ticker
            ),
            -- Get trades in the window BEFORE the last trade
            trades_in_window AS (
                SELECT
                    t.ticker,
                    t.yes_price,
                    t.created_time,
                    ROW_NUMBER() OVER (
                        PARTITION BY t.ticker
                        ORDER BY t.created_time DESC
                    ) AS rn
                FROM '{self.kalshi_trades_dir}/*.parquet' t
                INNER JOIN market_last_trade lt ON t.ticker = lt.ticker
                WHERE t.yes_price > 0
                  AND t.yes_price < 100
                  AND t.created_time >= lt.last_trade_time - INTERVAL '{HOURS_BEFORE_RESOLUTION} hours'
                  AND t.created_time < lt.last_trade_time
            ),
            market_prices AS (
                SELECT
                    ticker,
                    yes_price / 100.0 AS price
                FROM trades_in_window
                WHERE rn = 1
            ),
            market_brier AS (
                SELECT
                    {volume_bucket_case} AS volume_bucket,
                    -- Brier score: (price - outcome)^2
                    POWER(
                        mp.price - (CASE WHEN m.result = 'yes' THEN 1.0 ELSE 0.0 END),
                        2
                    ) AS brier_score
                FROM resolved_markets m
                INNER JOIN market_prices mp ON m.ticker = mp.ticker
            )
            SELECT
                volume_bucket,
                AVG(brier_score) AS brier_score,
                COUNT(*) AS market_count
            FROM market_brier
            WHERE volume_bucket IS NOT NULL
            GROUP BY volume_bucket
            """
        ).df()

        return df[["volume_bucket", "brier_score", "market_count"]]

    def _load_polymarket_brier_scores(self) -> pd.DataFrame:
        """Load Polymarket market-level Brier scores using price near last trade."""
        con = duckdb.connect()
        con.execute("SET preserve_insertion_order = false")
        con.execute("SET threads = 4")  # Limit parallelism to reduce memory

        # Build token -> (market_id, volume, is_winning) mapping
        # Note: We use last trade timestamp as resolution proxy, not end_date
        markets_df = con.execute(
            f"""
            SELECT id, clob_token_ids, outcome_prices, volume
            FROM '{self.polymarket_markets_dir}/*.parquet'
            WHERE closed = true
              AND volume IS NOT NULL
            """
        ).df()

        # token_id -> (market_id, volume, is_winning_token)
        token_data: list[tuple[str, str, float, bool]] = []

        for _, row in markets_df.iterrows():
            try:
                token_ids = json.loads(row["clob_token_ids"]) if row["clob_token_ids"] else None
                prices = json.loads(row["outcome_prices"]) if row["outcome_prices"] else None
                volume = row["volume"]
                market_id = row["id"]

                if not token_ids or not prices or len(token_ids) != 2 or len(prices) != 2:
                    continue
                if pd.isna(volume):
                    continue

                p0, p1 = float(prices[0]), float(prices[1])

                if p0 > 0.99 and p1 < 0.01:
                    token_data.append((token_ids[0], market_id, volume, True))
                    token_data.append((token_ids[1], market_id, volume, False))
                elif p0 < 0.01 and p1 > 0.99:
                    token_data.append((token_ids[0], market_id, volume, False))
                    token_data.append((token_ids[1], market_id, volume, True))

            except (json.JSONDecodeError, ValueError, TypeError):
                continue

        if not token_data:
            return pd.DataFrame({"volume_bucket": [], "brier_score": [], "market_count": []})

        # Register token data table (without end_timestamp - we'll compute from trades)
        con.execute("CREATE TABLE token_info (token_id VARCHAR, market_id VARCHAR, volume DOUBLE, is_winning BOOLEAN)")
        con.executemany(
            "INSERT INTO token_info VALUES (?, ?, ?, ?)",
            token_data,
        )
        con.execute("CREATE INDEX idx_token ON token_info(token_id)")

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

        volume_bucket_case = self._get_volume_bucket_case("volume")

        # Two-pass approach:
        # 1. Find last trade timestamp per market
        # 2. Find price in window before last trade
        ctf_results = con.execute(
            f"""
            WITH all_trades AS (
                -- Case 1: maker_asset_id = '0' (USDC), buyer is taker
                SELECT
                    ti.market_id,
                    ti.volume,
                    ti.is_winning,
                    t.maker_amount::DOUBLE / t.taker_amount::DOUBLE AS price,
                    b.timestamp AS trade_timestamp
                FROM '{self.polymarket_trades_dir}/*.parquet' t
                INNER JOIN token_info ti ON t.taker_asset_id = ti.token_id
                JOIN blocks b ON t.block_number // {BLOCK_BUCKET_SIZE} = b.bucket
                WHERE t.maker_asset_id = '0'
                  AND t.taker_amount > 0
                  AND t.maker_amount > 0

                UNION ALL

                -- Case 2: maker_asset_id != '0', seller is taker
                SELECT
                    ti.market_id,
                    ti.volume,
                    ti.is_winning,
                    t.taker_amount::DOUBLE / t.maker_amount::DOUBLE AS price,
                    b.timestamp AS trade_timestamp
                FROM '{self.polymarket_trades_dir}/*.parquet' t
                INNER JOIN token_info ti ON t.maker_asset_id = ti.token_id
                JOIN blocks b ON t.block_number // {BLOCK_BUCKET_SIZE} = b.bucket
                WHERE t.maker_asset_id != '0'
                  AND t.taker_amount > 0
                  AND t.maker_amount > 0
            ),
            -- Step 1: Find last trade timestamp per market
            market_last_trade AS (
                SELECT
                    market_id,
                    MAX(trade_timestamp) AS last_trade_ts
                FROM all_trades
                GROUP BY market_id
            ),
            -- Step 2: Get trades in the window before last trade
            trades_in_window AS (
                SELECT
                    t.market_id,
                    t.volume,
                    t.is_winning,
                    t.price,
                    t.trade_timestamp
                FROM all_trades t
                INNER JOIN market_last_trade lt ON t.market_id = lt.market_id
                WHERE t.trade_timestamp >= lt.last_trade_ts - {HOURS_BEFORE_RESOLUTION * 3600}
                  AND t.trade_timestamp < lt.last_trade_ts
            ),
            -- Step 3: Get price from most recent trade in window
            market_window_price AS (
                SELECT
                    market_id,
                    ANY_VALUE(volume) AS volume,
                    ARG_MAX(price, trade_timestamp) AS window_price,
                    ARG_MAX(is_winning, trade_timestamp) AS window_is_winning
                FROM trades_in_window
                GROUP BY market_id
            ),
            market_brier AS (
                SELECT
                    {volume_bucket_case} AS volume_bucket,
                    -- If trade was for winning token, use price directly
                    -- Otherwise, 1-price is the implied winning token price
                    POWER(
                        (CASE WHEN window_is_winning THEN window_price ELSE 1.0 - window_price END) - 1.0,
                        2
                    ) AS brier_score
                FROM market_window_price
            )
            SELECT
                volume_bucket,
                AVG(brier_score) AS brier_score,
                COUNT(*) AS market_count
            FROM market_brier
            WHERE volume_bucket IS NOT NULL
            GROUP BY volume_bucket
            """
        ).df()

        # Note: Legacy FPMM trades are excluded for simplicity
        # CTF trades represent the vast majority of modern Polymarket volume
        if ctf_results.empty:
            return pd.DataFrame({"volume_bucket": [], "brier_score": [], "market_count": []})

        return ctf_results[["volume_bucket", "brier_score", "market_count"]]

    def _create_figure(self, df: pd.DataFrame) -> plt.Figure:
        """Create the matplotlib figure."""
        fig, ax = plt.subplots(figsize=(12, 6))

        bucket_labels = [label for _, _, label in VOLUME_BUCKETS]
        x = np.arange(len(bucket_labels))
        width = 0.35

        # Get Brier scores for each bucket
        kalshi_scores = []
        polymarket_scores = []
        for label in bucket_labels:
            row = df[df["volume_bucket"] == label]
            kalshi_scores.append(
                row["brier_score_kalshi"].values[0]
                if not row.empty and pd.notna(row["brier_score_kalshi"].values[0])
                else 0
            )
            polymarket_scores.append(
                row["brier_score_polymarket"].values[0]
                if not row.empty and pd.notna(row["brier_score_polymarket"].values[0])
                else 0
            )

        # Create bars
        ax.bar(x - width / 2, kalshi_scores, width, label="Kalshi", color="#4C72B0")
        ax.bar(x + width / 2, polymarket_scores, width, label="Polymarket", color="#55A868")

        ax.set_xlabel("Market Volume (USD)")
        ax.set_ylabel("Brier Score")
        ax.set_title(f"Market-Level Brier Score by Volume ({HOURS_BEFORE_RESOLUTION}h Before Resolution)")
        ax.set_xticks(x)
        ax.set_xticklabels(bucket_labels, rotation=45, ha="right")

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
        ax.grid(True, alpha=0.3, axis="y")

        # Set y-axis limit based on data
        max_score = max(max(kalshi_scores), max(polymarket_scores), 0.1)
        ax.set_ylim(0, min(max_score * 1.2, 0.3))

        plt.tight_layout()
        return fig

    def _create_chart(self, df: pd.DataFrame) -> ChartConfig:
        """Create the chart configuration for web display."""
        chart_data = []
        for _, _, label in VOLUME_BUCKETS:
            row = df[df["volume_bucket"] == label]
            entry = {"volume_bucket": label}
            if not row.empty:
                if pd.notna(row["brier_score_kalshi"].values[0]):
                    entry["kalshi"] = round(row["brier_score_kalshi"].values[0], 4)
                if pd.notna(row["brier_score_polymarket"].values[0]):
                    entry["polymarket"] = round(row["brier_score_polymarket"].values[0], 4)
            chart_data.append(entry)

        return ChartConfig(
            type=ChartType.BAR,
            data=chart_data,
            xKey="volume_bucket",
            yKeys=["kalshi", "polymarket"],
            title=f"Market-Level Brier Score by Volume ({HOURS_BEFORE_RESOLUTION}h Before Resolution)",
            xLabel="Market Volume (USD)",
            yLabel="Brier Score",
            colors={"kalshi": "#4C72B0", "polymarket": "#55A868"},
        )
