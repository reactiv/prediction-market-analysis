"""Compare cumulative Brier score over calendar time between Polymarket and Kalshi.

NOTE: This computes Brier score at trade execution time, which is the correct
methodology for measuring market calibration. Some analyses (e.g., Dune's
polymarket_data table with price_1d_before, or brier_score_over_time.py which groups
by hours before resolution) produce artificially low Brier scores (~0.05) because
markets have already converged toward 0 or 1 as outcomes become obvious.

This analysis answers the question traders care about: "When I buy at X%, does the
outcome happen X% of the time?" Expected Brier score for a well-calibrated market
with trades across all price levels is ~0.17, not 0.05.

INTERPRETATION CAVEAT: Brier score depends on the price distribution of trades, not
just calibration quality. The expected Brier for perfectly calibrated forecasts is
E[p(1-p)], which varies by price distribution:
  - Prices at extremes (5¢/95¢): E[Brier] ≈ 0.05
  - Uniform prices (1-99¢): E[Brier] ≈ 0.17
  - Prices at 50¢: E[Brier] = 0.25

Rising Brier scores may indicate traders are trading more uncertain events (mid-range
prices) rather than making worse predictions. Compare with MAD (calibration_comparison
_over_time.py) and price distribution (price_distribution_over_time.py) to distinguish
between worsening calibration vs. shifting to more uncertain markets.
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


class CumulativeBrierScoreOverTimeAnalysis(Analysis):
    """Compare cumulative Brier score over calendar time between Polymarket and Kalshi."""

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
            name="cumulative_brier_score_over_time",
            description="Compare cumulative Brier score over calendar time between platforms",
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
        with self.progress("Loading Kalshi aggregates"):
            kalshi_df = self._load_kalshi_aggregates()

        with self.progress("Loading Polymarket aggregates"):
            polymarket_df = self._load_polymarket_aggregates()

        with self.progress("Computing Brier scores"):
            kalshi_results = self._compute_cumulative_brier(kalshi_df)
            polymarket_results = self._compute_cumulative_brier(polymarket_df)

        # Merge results on date
        output_df = pd.merge(
            kalshi_results,
            polymarket_results,
            on="date",
            how="outer",
            suffixes=("_kalshi", "_polymarket"),
        ).sort_values("date")

        fig = self._create_figure(output_df)
        chart = self._create_chart(output_df)

        return AnalysisOutput(figure=fig, data=output_df, chart=chart)

    def _load_kalshi_aggregates(self) -> pd.DataFrame:
        """Load Kalshi trades aggregated by week and price directly in SQL."""
        con = duckdb.connect()

        df = con.execute(
            f"""
            WITH resolved_markets AS (
                SELECT ticker, result
                FROM '{self.kalshi_markets_dir}/*.parquet'
                WHERE result IN ('yes', 'no')
            ),
            trade_positions AS (
                -- Buyer side (taker)
                SELECT
                    DATE_TRUNC('week', t.created_time) AS week,
                    CASE
                        WHEN t.taker_side = 'yes' THEN t.yes_price
                        ELSE t.no_price
                    END AS price,
                    CASE WHEN t.taker_side = m.result THEN 1 ELSE 0 END AS won
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
                    END AS price,
                    CASE WHEN t.taker_side != m.result THEN 1 ELSE 0 END AS won
                FROM '{self.kalshi_trades_dir}/*.parquet' t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
                WHERE t.yes_price > 0 AND t.no_price > 0
            )
            SELECT
                week,
                price,
                COUNT(*) AS total,
                SUM(won) AS wins
            FROM trade_positions
            WHERE price >= 1 AND price <= 99
            GROUP BY week, price
            ORDER BY week, price
            """
        ).df()

        return df

    def _load_polymarket_aggregates(self) -> pd.DataFrame:
        """Load Polymarket trades aggregated by week and price directly in SQL."""
        con = duckdb.connect()

        # Build CTF token_id -> won mapping for resolved markets
        markets_df = con.execute(
            f"""
            SELECT id, clob_token_ids, outcome_prices
            FROM '{self.polymarket_markets_dir}/*.parquet'
            WHERE closed = true
            """
        ).df()

        token_won: dict[str, bool] = {}
        for _, row in markets_df.iterrows():
            try:
                token_ids = json.loads(row["clob_token_ids"]) if row["clob_token_ids"] else None
                prices = json.loads(row["outcome_prices"]) if row["outcome_prices"] else None
                if not token_ids or not prices or len(token_ids) != 2 or len(prices) != 2:
                    continue
                p0, p1 = float(prices[0]), float(prices[1])
                if p0 > 0.99 and p1 < 0.01:
                    token_won[token_ids[0]] = True
                    token_won[token_ids[1]] = False
                elif p0 < 0.01 and p1 > 0.99:
                    token_won[token_ids[0]] = False
                    token_won[token_ids[1]] = True
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

        # Register CTF token mapping
        con.execute("CREATE TABLE token_resolution (token_id VARCHAR, won BOOLEAN)")
        con.executemany("INSERT INTO token_resolution VALUES (?, ?)", list(token_won.items()))

        # Load FPMM resolution from markets parquet files (uses market_maker_address field)
        fpmm_resolution: dict[str, int] = {}
        fpmm_markets_df = con.execute(
            f"""
            SELECT market_maker_address, outcome_prices
            FROM '{self.polymarket_markets_dir}/*.parquet'
            WHERE market_maker_address IS NOT NULL
            """
        ).df()

        # Build fpmm_address -> winning_outcome mapping
        for _, row in fpmm_markets_df.iterrows():
            fpmm_addr = row["market_maker_address"]
            try:
                prices = json.loads(row["outcome_prices"]) if row["outcome_prices"] else None
                if not prices or len(prices) != 2:
                    continue
                p0, p1 = float(prices[0]), float(prices[1])
                if p0 > 0.99 and p1 < 0.01:
                    fpmm_resolution[fpmm_addr.lower()] = 0
                elif p0 < 0.01 and p1 > 0.99:
                    fpmm_resolution[fpmm_addr.lower()] = 1
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

        # Filter to USDC markets only
        if self.collateral_lookup_path.exists():
            with open(self.collateral_lookup_path) as f:
                collateral_lookup = json.load(f)
            usdc_markets = {
                addr.lower() for addr, info in collateral_lookup.items() if info["collateral_symbol"] == "USDC"
            }
            fpmm_resolution = {k: v for k, v in fpmm_resolution.items() if k in usdc_markets}

        con.execute("CREATE TABLE fpmm_resolution (fpmm_address VARCHAR, winning_outcome BIGINT)")
        if fpmm_resolution:
            con.executemany("INSERT INTO fpmm_resolution VALUES (?, ?)", list(fpmm_resolution.items()))

        # Create blocks lookup table with larger buckets for speed
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

        # CTF trades query - aggregate by week and price
        ctf_trades_query = f"""
            SELECT
                DATE_TRUNC('week', b.timestamp) AS week,
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

            UNION ALL

            SELECT
                DATE_TRUNC('week', b.timestamp) AS week,
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
        """

        # Legacy FPMM trades query
        legacy_trades_query = ""
        if fpmm_resolution and self.polymarket_legacy_trades_dir.exists():
            legacy_trades_query = f"""
                UNION ALL

                SELECT
                    DATE_TRUNC('week', b.timestamp) AS week,
                    ROUND(100.0 * t.amount::DOUBLE / t.outcome_tokens::DOUBLE) AS price,
                    CASE WHEN t.outcome_index = r.winning_outcome THEN 1 ELSE 0 END AS won
                FROM '{self.polymarket_legacy_trades_dir}/*.parquet' t
                INNER JOIN fpmm_resolution r ON LOWER(t.fpmm_address) = r.fpmm_address
                JOIN blocks b ON t.block_number // {BLOCK_BUCKET_SIZE} = b.bucket
                WHERE t.outcome_tokens::DOUBLE > 0

                UNION ALL

                SELECT
                    DATE_TRUNC('week', b.timestamp) AS week,
                    ROUND(100.0 - 100.0 * t.amount::DOUBLE / t.outcome_tokens::DOUBLE) AS price,
                    CASE WHEN t.outcome_index != r.winning_outcome THEN 1 ELSE 0 END AS won
                FROM '{self.polymarket_legacy_trades_dir}/*.parquet' t
                INNER JOIN fpmm_resolution r ON LOWER(t.fpmm_address) = r.fpmm_address
                JOIN blocks b ON t.block_number // {BLOCK_BUCKET_SIZE} = b.bucket
                WHERE t.outcome_tokens::DOUBLE > 0
            """

        df = con.execute(
            f"""
            WITH trade_positions AS (
                {ctf_trades_query}
                {legacy_trades_query}
            )
            SELECT
                week,
                price,
                COUNT(*) AS total,
                SUM(won) AS wins
            FROM trade_positions
            WHERE price >= 1 AND price <= 99
            GROUP BY week, price
            ORDER BY week, price
            """
        ).df()

        return df

    def _compute_cumulative_brier(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute cumulative Brier score over time from pre-aggregated data.

        Brier score = mean((p - y)²) where p is predicted probability, y is outcome (0 or 1)
        For each price bucket with `total` trades and `wins` wins:
        - wins contribute: (price/100 - 1)² per trade
        - losses contribute: (price/100 - 0)² per trade
        """
        if df.empty:
            return pd.DataFrame({"date": [], "brier_score": []})

        # Normalize week column to timezone-naive
        df = df.copy()
        if df["week"].dt.tz is not None:
            df["week"] = df["week"].dt.tz_localize(None)

        # Pivot to get weekly totals/wins per price
        total_pivot = df.pivot(index="week", columns="price", values="total").fillna(0)
        wins_pivot = df.pivot(index="week", columns="price", values="wins").fillna(0)

        # Compute cumulative sums over weeks
        cumulative_totals = total_pivot.cumsum()
        cumulative_wins = wins_pivot.cumsum()

        weeks = cumulative_totals.index.tolist()
        dates = []
        brier_scores = []

        for week in weeks:
            totals = cumulative_totals.loc[week].values
            wins = cumulative_wins.loc[week].values
            price_values = cumulative_totals.columns.values.astype(float)

            total_trades = totals.sum()
            # Skip if not enough cumulative data
            if total_trades < 1000:
                continue

            # Compute Brier score contribution from each price bucket
            brier_sum = 0.0
            for i, price in enumerate(price_values):
                if totals[i] == 0:
                    continue
                p = price / 100.0  # Convert cents to probability
                bucket_wins = wins[i]
                bucket_losses = totals[i] - wins[i]
                # Wins: (p - 1)², Losses: (p - 0)²
                brier_sum += bucket_wins * (p - 1) ** 2 + bucket_losses * p**2

            brier_score = brier_sum / total_trades

            dates.append(week)
            brier_scores.append(brier_score)

        return pd.DataFrame({"date": dates, "brier_score": brier_scores})

    def _create_figure(self, df: pd.DataFrame) -> plt.Figure:
        """Create the matplotlib figure."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot Kalshi
        kalshi_mask = df["brier_score_kalshi"].notna()
        if kalshi_mask.any():
            ax.plot(
                df.loc[kalshi_mask, "date"],
                df.loc[kalshi_mask, "brier_score_kalshi"],
                color="#4C72B0",
                linewidth=2,
                label="Kalshi",
            )

        # Plot Polymarket
        poly_mask = df["brier_score_polymarket"].notna()
        if poly_mask.any():
            ax.plot(
                df.loc[poly_mask, "date"],
                df.loc[poly_mask, "brier_score_polymarket"],
                color="#55A868",
                linewidth=2,
                label="Polymarket",
            )

        ax.set_xlabel("Date")
        ax.set_ylabel("Brier Score")
        ax.set_title("Cumulative Brier Score Over Time: Polymarket vs Kalshi")

        # Add reference line for theoretical minimum (~0.17 for uniform price distribution)
        ax.axhline(
            y=0.17,
            color="#D65F5F",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label="Theoretical minimum (~0.17)",
        )

        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        fig.autofmt_xdate()

        plt.tight_layout()
        return fig

    def _create_chart(self, df: pd.DataFrame) -> ChartConfig:
        """Create the chart configuration for web display."""
        chart_data = []
        for _, row in df.iterrows():
            entry = {"date": row["date"].strftime("%Y-%m-%d")}
            if pd.notna(row.get("brier_score_kalshi")):
                entry["kalshi"] = round(row["brier_score_kalshi"], 4)
            if pd.notna(row.get("brier_score_polymarket")):
                entry["polymarket"] = round(row["brier_score_polymarket"], 4)
            chart_data.append(entry)

        return ChartConfig(
            type=ChartType.LINE,
            data=chart_data,
            xKey="date",
            yKeys=["kalshi", "polymarket"],
            title="Cumulative Brier Score Over Time: Polymarket vs Kalshi",
            xLabel="Date",
            yLabel="Brier Score",
            colors={"kalshi": "#4C72B0", "polymarket": "#55A868"},
        )
