"""Compare price distribution over time between Polymarket and Kalshi.

This analysis shows how the distribution of trade prices has shifted over time,
specifically tracking the percentage of trades at extreme prices (1-20¢ or 80-99¢)
vs mid-range prices (40-60¢).

This helps interpret Brier score and log loss trends: if trading shifts from extreme
prices to mid-range prices, Brier/log loss will naturally increase even if calibration
quality remains constant or improves. The expected Brier for perfectly calibrated
forecasts is E[p(1-p)]:
  - Prices at extremes (5¢/95¢): E[Brier] ≈ 0.05
  - Prices at 50¢: E[Brier] = 0.25

Use this analysis alongside cumulative_brier_score_over_time.py and
calibration_comparison_over_time.py (MAD) to distinguish between:
  - Worsening calibration (MAD increases, price distribution stable)
  - Trading more uncertain events (MAD stable/decreases, mid-range % increases)
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType

# Bucket size for block-to-timestamp approximation (10800 blocks ~ 6 hours at 2 sec/block)
BLOCK_BUCKET_SIZE = 10800


class PriceDistributionOverTimeAnalysis(Analysis):
    """Compare price distribution over time between Polymarket and Kalshi."""

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
            name="price_distribution_over_time",
            description="Compare price distribution over time between platforms",
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
        with self.progress("Loading Kalshi price distribution"):
            kalshi_df = self._load_kalshi_distribution()

        with self.progress("Loading Polymarket price distribution"):
            polymarket_df = self._load_polymarket_distribution()

        # Normalize timezones before merging
        if not kalshi_df.empty and kalshi_df["date"].dt.tz is not None:
            kalshi_df["date"] = kalshi_df["date"].dt.tz_localize(None)
        if not polymarket_df.empty and polymarket_df["date"].dt.tz is not None:
            polymarket_df["date"] = polymarket_df["date"].dt.tz_localize(None)

        # Merge results on date
        output_df = pd.merge(
            kalshi_df,
            polymarket_df,
            on="date",
            how="outer",
            suffixes=("_kalshi", "_polymarket"),
        ).sort_values("date")

        fig = self._create_figure(output_df)
        chart = self._create_chart(output_df)

        return AnalysisOutput(figure=fig, data=output_df, chart=chart)

    def _load_kalshi_distribution(self) -> pd.DataFrame:
        """Load Kalshi trades and compute price distribution by week."""
        con = duckdb.connect()

        df = con.execute(
            f"""
            WITH resolved_markets AS (
                SELECT ticker, result
                FROM '{self.kalshi_markets_dir}/*.parquet'
                WHERE result IN ('yes', 'no')
            ),
            trade_prices AS (
                -- Buyer side (taker)
                SELECT
                    DATE_TRUNC('week', t.created_time) AS week,
                    CASE
                        WHEN t.taker_side = 'yes' THEN t.yes_price
                        ELSE t.no_price
                    END AS price
                FROM '{self.kalshi_trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
                WHERE t.yes_price > 0 AND t.no_price > 0

                UNION ALL

                -- Seller side (counterparty)
                SELECT
                    DATE_TRUNC('week', t.created_time) AS week,
                    CASE
                        WHEN t.taker_side = 'yes' THEN t.no_price
                        ELSE t.yes_price
                    END AS price
                FROM '{self.kalshi_trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
                WHERE t.yes_price > 0 AND t.no_price > 0
            )
            SELECT
                week,
                COUNT(*) AS total_trades,
                SUM(CASE WHEN price <= 20 OR price >= 80 THEN 1 ELSE 0 END) AS extreme_trades,
                SUM(CASE WHEN price >= 40 AND price <= 60 THEN 1 ELSE 0 END) AS midrange_trades
            FROM trade_prices
            WHERE price >= 1 AND price <= 99
            GROUP BY week
            ORDER BY week
            """
        ).df()

        if df.empty:
            return pd.DataFrame({"date": [], "extreme_pct": [], "midrange_pct": [], "total_trades": []})

        df["extreme_pct"] = 100.0 * df["extreme_trades"] / df["total_trades"]
        df["midrange_pct"] = 100.0 * df["midrange_trades"] / df["total_trades"]
        df["date"] = df["week"]

        return df[["date", "extreme_pct", "midrange_pct", "total_trades"]]

    def _load_polymarket_distribution(self) -> pd.DataFrame:
        """Load Polymarket trades and compute price distribution by week."""
        con = duckdb.connect()

        # Build CTF token_id -> won mapping for resolved markets
        markets_df = con.execute(
            f"""
            SELECT id, clob_token_ids, outcome_prices
            FROM '{self.polymarket_markets_dir}/*.parquet'
            WHERE closed = true
            """
        ).df()

        token_ids_set: set[str] = set()
        for _, row in markets_df.iterrows():
            try:
                token_ids = json.loads(row["clob_token_ids"]) if row["clob_token_ids"] else None
                prices = json.loads(row["outcome_prices"]) if row["outcome_prices"] else None
                if not token_ids or not prices or len(token_ids) != 2 or len(prices) != 2:
                    continue
                p0, p1 = float(prices[0]), float(prices[1])
                # Only include resolved markets
                if (p0 > 0.99 and p1 < 0.01) or (p0 < 0.01 and p1 > 0.99):
                    token_ids_set.add(token_ids[0])
                    token_ids_set.add(token_ids[1])
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

        # Register token set
        con.execute("CREATE TABLE resolved_tokens (token_id VARCHAR)")
        if token_ids_set:
            con.executemany("INSERT INTO resolved_tokens VALUES (?)", [(t,) for t in token_ids_set])

        # Load FPMM resolution
        fpmm_addresses: set[str] = set()
        fpmm_markets_df = con.execute(
            f"""
            SELECT market_maker_address, outcome_prices
            FROM '{self.polymarket_markets_dir}/*.parquet'
            WHERE market_maker_address IS NOT NULL
            """
        ).df()

        for _, row in fpmm_markets_df.iterrows():
            fpmm_addr = row["market_maker_address"]
            try:
                prices = json.loads(row["outcome_prices"]) if row["outcome_prices"] else None
                if not prices or len(prices) != 2:
                    continue
                p0, p1 = float(prices[0]), float(prices[1])
                if (p0 > 0.99 and p1 < 0.01) or (p0 < 0.01 and p1 > 0.99):
                    fpmm_addresses.add(fpmm_addr.lower())
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

        # Filter to USDC markets only
        if self.collateral_lookup_path.exists():
            with open(self.collateral_lookup_path) as f:
                collateral_lookup = json.load(f)
            usdc_markets = {
                addr.lower() for addr, info in collateral_lookup.items() if info["collateral_symbol"] == "USDC"
            }
            fpmm_addresses = fpmm_addresses & usdc_markets

        con.execute("CREATE TABLE resolved_fpmm (fpmm_address VARCHAR)")
        if fpmm_addresses:
            con.executemany("INSERT INTO resolved_fpmm VALUES (?)", [(a,) for a in fpmm_addresses])

        # Create blocks lookup table
        con.execute(
            f"""
            CREATE TABLE blocks AS
            SELECT
                block_number // {BLOCK_BUCKET_SIZE} AS bucket,
                FIRST(timestamp::TIMESTAMP) AS timestamp
            FROM '{self.polymarket_blocks_dir}/*.parquet'
            GROUP BY block_number // {BLOCK_BUCKET_SIZE}
            """
        )

        # CTF trades query
        ctf_trades_query = f"""
            SELECT
                DATE_TRUNC('week', b.timestamp) AS week,
                CASE
                    WHEN t.maker_asset_id = '0' THEN ROUND(100.0 * t.maker_amount / t.taker_amount)
                    ELSE ROUND(100.0 * t.taker_amount / t.maker_amount)
                END AS price
            FROM '{self.polymarket_trades_dir}/*.parquet' t
            INNER JOIN resolved_tokens rt ON (
                CASE WHEN t.maker_asset_id = '0' THEN t.taker_asset_id ELSE t.maker_asset_id END = rt.token_id
            )
            JOIN blocks b ON t.block_number // {BLOCK_BUCKET_SIZE} = b.bucket
            WHERE t.taker_amount > 0 AND t.maker_amount > 0

            UNION ALL

            SELECT
                DATE_TRUNC('week', b.timestamp) AS week,
                CASE
                    WHEN t.maker_asset_id = '0' THEN ROUND(100.0 - 100.0 * t.maker_amount / t.taker_amount)
                    ELSE ROUND(100.0 - 100.0 * t.taker_amount / t.maker_amount)
                END AS price
            FROM '{self.polymarket_trades_dir}/*.parquet' t
            INNER JOIN resolved_tokens rt ON (
                CASE WHEN t.maker_asset_id = '0' THEN t.taker_asset_id ELSE t.maker_asset_id END = rt.token_id
            )
            JOIN blocks b ON t.block_number // {BLOCK_BUCKET_SIZE} = b.bucket
            WHERE t.taker_amount > 0 AND t.maker_amount > 0
        """

        # Legacy FPMM trades query
        legacy_trades_query = ""
        if fpmm_addresses and self.polymarket_legacy_trades_dir.exists():
            legacy_trades_query = f"""
                UNION ALL

                SELECT
                    DATE_TRUNC('week', b.timestamp) AS week,
                    ROUND(100.0 * t.amount::DOUBLE / t.outcome_tokens::DOUBLE) AS price
                FROM '{self.polymarket_legacy_trades_dir}/*.parquet' t
                INNER JOIN resolved_fpmm r ON LOWER(t.fpmm_address) = r.fpmm_address
                JOIN blocks b ON t.block_number // {BLOCK_BUCKET_SIZE} = b.bucket
                WHERE t.outcome_tokens::DOUBLE > 0

                UNION ALL

                SELECT
                    DATE_TRUNC('week', b.timestamp) AS week,
                    ROUND(100.0 - 100.0 * t.amount::DOUBLE / t.outcome_tokens::DOUBLE) AS price
                FROM '{self.polymarket_legacy_trades_dir}/*.parquet' t
                INNER JOIN resolved_fpmm r ON LOWER(t.fpmm_address) = r.fpmm_address
                JOIN blocks b ON t.block_number // {BLOCK_BUCKET_SIZE} = b.bucket
                WHERE t.outcome_tokens::DOUBLE > 0
            """

        df = con.execute(
            f"""
            WITH trade_prices AS (
                {ctf_trades_query}
                {legacy_trades_query}
            )
            SELECT
                week,
                COUNT(*) AS total_trades,
                SUM(CASE WHEN price <= 20 OR price >= 80 THEN 1 ELSE 0 END) AS extreme_trades,
                SUM(CASE WHEN price >= 40 AND price <= 60 THEN 1 ELSE 0 END) AS midrange_trades
            FROM trade_prices
            WHERE price >= 1 AND price <= 99
            GROUP BY week
            ORDER BY week
            """
        ).df()

        if df.empty:
            return pd.DataFrame({"date": [], "extreme_pct": [], "midrange_pct": [], "total_trades": []})

        df["extreme_pct"] = 100.0 * df["extreme_trades"] / df["total_trades"]
        df["midrange_pct"] = 100.0 * df["midrange_trades"] / df["total_trades"]
        df["date"] = df["week"]

        return df[["date", "extreme_pct", "midrange_pct", "total_trades"]]

    def _create_figure(self, df: pd.DataFrame) -> plt.Figure:
        """Create the matplotlib figure."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Normalize date column
        df = df.copy()
        if df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)

        # Top plot: Extreme price percentage (1-20¢ or 80-99¢)
        ax1 = axes[0]
        kalshi_mask = df["extreme_pct_kalshi"].notna()
        if kalshi_mask.any():
            ax1.plot(
                df.loc[kalshi_mask, "date"],
                df.loc[kalshi_mask, "extreme_pct_kalshi"],
                color="#4C72B0",
                linewidth=2,
                label="Kalshi",
                alpha=0.8,
            )

        poly_mask = df["extreme_pct_polymarket"].notna()
        if poly_mask.any():
            ax1.plot(
                df.loc[poly_mask, "date"],
                df.loc[poly_mask, "extreme_pct_polymarket"],
                color="#55A868",
                linewidth=2,
                label="Polymarket",
                alpha=0.8,
            )

        ax1.set_ylabel("% of Trades")
        ax1.set_title("Extreme Price Trades (1-20¢ or 80-99¢)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)

        # Bottom plot: Mid-range price percentage (40-60¢)
        ax2 = axes[1]
        if kalshi_mask.any():
            ax2.plot(
                df.loc[kalshi_mask, "date"],
                df.loc[kalshi_mask, "midrange_pct_kalshi"],
                color="#4C72B0",
                linewidth=2,
                label="Kalshi",
                alpha=0.8,
            )

        if poly_mask.any():
            ax2.plot(
                df.loc[poly_mask, "date"],
                df.loc[poly_mask, "midrange_pct_polymarket"],
                color="#55A868",
                linewidth=2,
                label="Polymarket",
                alpha=0.8,
            )

        ax2.set_xlabel("Date")
        ax2.set_ylabel("% of Trades")
        ax2.set_title("Mid-Range Price Trades (40-60¢)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 50)

        fig.suptitle(
            "Price Distribution Over Time: Extreme vs Mid-Range Trading",
            fontsize=14,
            fontweight="bold",
        )
        fig.autofmt_xdate()
        plt.tight_layout()
        return fig

    def _create_chart(self, df: pd.DataFrame) -> ChartConfig:
        """Create the chart configuration for web display."""
        chart_data = []
        for _, row in df.iterrows():
            entry = {"date": row["date"].strftime("%Y-%m-%d")}
            if pd.notna(row.get("extreme_pct_kalshi")):
                entry["kalshi_extreme"] = round(row["extreme_pct_kalshi"], 2)
            if pd.notna(row.get("extreme_pct_polymarket")):
                entry["polymarket_extreme"] = round(row["extreme_pct_polymarket"], 2)
            if pd.notna(row.get("midrange_pct_kalshi")):
                entry["kalshi_midrange"] = round(row["midrange_pct_kalshi"], 2)
            if pd.notna(row.get("midrange_pct_polymarket")):
                entry["polymarket_midrange"] = round(row["midrange_pct_polymarket"], 2)
            chart_data.append(entry)

        return ChartConfig(
            type=ChartType.LINE,
            data=chart_data,
            xKey="date",
            yKeys=["kalshi_extreme", "polymarket_extreme", "kalshi_midrange", "polymarket_midrange"],
            title="Price Distribution Over Time",
            yUnit=UnitType.PERCENT,
            xLabel="Date",
            yLabel="% of Trades",
            colors={
                "kalshi_extreme": "#4C72B0",
                "polymarket_extreme": "#55A868",
                "kalshi_midrange": "#8172B2",
                "polymarket_midrange": "#CCB974",
            },
        )
