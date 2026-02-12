"""Compare Brier score by market volume bucket between Polymarket and Kalshi.

Brier score measures the mean squared error between predicted probabilities and outcomes.
Lower scores indicate better predictions. Perfect predictions = 0, random = 0.25.

This analysis groups markets by their total volume into buckets and calculates the
average Brier score for trades in each bucket, comparing Polymarket vs Kalshi.

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

# Bucket size for block-to-timestamp approximation (10800 blocks ~ 6 hours at 2 sec/block)
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


class BrierScoreByVolumeAnalysis(Analysis):
    """Compare Brier score by market volume bucket between platforms."""

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
            name="brier_score_by_volume",
            description="Compare Brier score by market volume bucket between platforms",
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
        with self.progress("Loading Kalshi Brier scores by volume"):
            kalshi_df = self._load_kalshi_brier_scores()

        with self.progress("Loading Polymarket Brier scores by volume"):
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
        """Load Kalshi trades and compute Brier score by market volume bucket."""
        con = duckdb.connect()

        volume_bucket_case = self._get_volume_bucket_case("m.volume")

        df = con.execute(
            f"""
            WITH resolved_markets AS (
                SELECT ticker, result, volume
                FROM '{self.kalshi_markets_dir}/*.parquet'
                WHERE result IN ('yes', 'no')
                  AND volume IS NOT NULL
            ),
            trade_scores AS (
                SELECT
                    {volume_bucket_case} AS volume_bucket,
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
            )
            SELECT
                volume_bucket,
                AVG(taker_brier) AS taker_brier_score,
                AVG(maker_brier) AS maker_brier_score,
                (AVG(taker_brier) + AVG(maker_brier)) / 2 AS brier_score,
                COUNT(*) AS trade_count
            FROM trade_scores
            WHERE volume_bucket IS NOT NULL
            GROUP BY volume_bucket
            """
        ).df()

        return df[["volume_bucket", "brier_score", "trade_count"]]

    def _load_polymarket_brier_scores(self) -> pd.DataFrame:
        """Load Polymarket trades and compute Brier score by market volume bucket."""
        con = duckdb.connect()
        con.execute("SET preserve_insertion_order = false")

        # Build CTF token_id -> (won, volume) mapping for resolved markets
        markets_df = con.execute(
            f"""
            SELECT id, clob_token_ids, outcome_prices, volume
            FROM '{self.polymarket_markets_dir}/*.parquet'
            WHERE closed = true AND volume IS NOT NULL
            """
        ).df()

        token_data: dict[str, tuple[bool, float]] = {}
        for _, row in markets_df.iterrows():
            try:
                token_ids = json.loads(row["clob_token_ids"]) if row["clob_token_ids"] else None
                prices = json.loads(row["outcome_prices"]) if row["outcome_prices"] else None
                volume = row["volume"]
                if not token_ids or not prices or len(token_ids) != 2 or len(prices) != 2:
                    continue
                if pd.isna(volume):
                    continue
                p0, p1 = float(prices[0]), float(prices[1])
                if p0 > 0.99 and p1 < 0.01:
                    token_data[token_ids[0]] = (True, volume)
                    token_data[token_ids[1]] = (False, volume)
                elif p0 < 0.01 and p1 > 0.99:
                    token_data[token_ids[0]] = (False, volume)
                    token_data[token_ids[1]] = (True, volume)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

        # Create CTF token mapping table
        con.execute("CREATE TABLE token_resolution (token_id VARCHAR, won BOOLEAN, volume DOUBLE)")
        if token_data:
            con.executemany(
                "INSERT INTO token_resolution VALUES (?, ?, ?)",
                [(k, v[0], v[1]) for k, v in token_data.items()],
            )

        # Load FPMM resolution from markets parquet files (if market_maker_address exists)
        fpmm_data: dict[str, tuple[int, float]] = {}
        schema_df = con.execute(f"DESCRIBE SELECT * FROM '{self.polymarket_markets_dir}/*.parquet' LIMIT 1").df()
        has_fpmm_field = "market_maker_address" in schema_df["column_name"].values

        if has_fpmm_field:
            fpmm_markets_df = con.execute(
                f"""
                SELECT market_maker_address, outcome_prices, volume
                FROM '{self.polymarket_markets_dir}/*.parquet'
                WHERE market_maker_address IS NOT NULL AND volume IS NOT NULL
                """
            ).df()

            for _, row in fpmm_markets_df.iterrows():
                fpmm_addr = row["market_maker_address"]
                try:
                    prices = json.loads(row["outcome_prices"]) if row["outcome_prices"] else None
                    volume = row["volume"]
                    if not prices or len(prices) != 2 or pd.isna(volume):
                        continue
                    p0, p1 = float(prices[0]), float(prices[1])
                    if p0 > 0.99 and p1 < 0.01:
                        fpmm_data[fpmm_addr.lower()] = (0, volume)
                    elif p0 < 0.01 and p1 > 0.99:
                        fpmm_data[fpmm_addr.lower()] = (1, volume)
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

        con.execute("CREATE TABLE fpmm_resolution (fpmm_address VARCHAR, winning_outcome BIGINT, volume DOUBLE)")
        if fpmm_data:
            con.executemany(
                "INSERT INTO fpmm_resolution VALUES (?, ?, ?)",
                [(k, v[0], v[1]) for k, v in fpmm_data.items()],
            )

        # Step 1: Materialize filtered CTF trades with computed fields
        con.execute(
            f"""
            CREATE TABLE ctf_trades AS
            -- Case 1: maker_asset_id = '0' (USDC), join on taker_asset_id
            SELECT
                tr.won,
                tr.volume,
                t.maker_amount::DOUBLE / t.taker_amount::DOUBLE AS price
            FROM '{self.polymarket_trades_dir}/*.parquet' t
            INNER JOIN token_resolution tr ON t.taker_asset_id = tr.token_id
            WHERE t.maker_asset_id = '0'
              AND t.taker_amount > 0
              AND t.maker_amount > 0

            UNION ALL

            -- Case 2: maker_asset_id != '0', join on maker_asset_id
            SELECT
                tr.won,
                tr.volume,
                t.taker_amount::DOUBLE / t.maker_amount::DOUBLE AS price
            FROM '{self.polymarket_trades_dir}/*.parquet' t
            INNER JOIN token_resolution tr ON t.maker_asset_id = tr.token_id
            WHERE t.maker_asset_id != '0'
              AND t.taker_amount > 0
              AND t.maker_amount > 0
            """
        )

        # Step 2: Compute Brier scores by volume bucket
        ctf_results = con.execute(
            f"""
            SELECT
                {self._get_volume_bucket_case("t.volume")} AS volume_bucket,
                SUM(POWER(t.price - (CASE WHEN t.won THEN 1.0 ELSE 0.0 END), 2)) AS sum_buyer_brier,
                SUM(POWER((1.0 - t.price) - (CASE WHEN NOT t.won THEN 1.0 ELSE 0.0 END), 2)) AS sum_seller_brier,
                COUNT(*) AS trade_count
            FROM ctf_trades t
            GROUP BY volume_bucket
            HAVING volume_bucket IS NOT NULL
            """
        ).df()

        # Step 3: Legacy FPMM trades (if available)
        legacy_results = pd.DataFrame()
        if fpmm_data and self.polymarket_legacy_trades_dir.exists():
            con.execute("CREATE INDEX idx_fpmm ON fpmm_resolution(fpmm_address)")
            legacy_results = con.execute(
                f"""
                SELECT
                    {self._get_volume_bucket_case("r.volume")} AS volume_bucket,
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
                WHERE t.outcome_tokens::DOUBLE > 0
                GROUP BY volume_bucket
                HAVING volume_bucket IS NOT NULL
                """
            ).df()

        # Combine CTF and legacy results
        if legacy_results.empty:
            combined = ctf_results
        else:
            combined = pd.concat([ctf_results, legacy_results])
            combined = (
                combined.groupby("volume_bucket")
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
            return pd.DataFrame({"volume_bucket": [], "brier_score": [], "trade_count": []})

        combined["brier_score"] = (
            combined["sum_buyer_brier"] / combined["trade_count"]
            + combined["sum_seller_brier"] / combined["trade_count"]
        ) / 2

        return combined[["volume_bucket", "brier_score", "trade_count"]]

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
        ax.set_title("Brier Score by Market Volume: Polymarket vs Kalshi")
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
        ax.set_ylim(0, max(max(kalshi_scores), max(polymarket_scores), 0.25) * 1.1)

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
            title="Brier Score by Market Volume: Polymarket vs Kalshi",
            xLabel="Market Volume (USD)",
            yLabel="Brier Score",
            colors={"kalshi": "#4C72B0", "polymarket": "#55A868"},
        )
