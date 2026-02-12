"""Compare cumulative log loss over calendar time between Polymarket and Kalshi.

Log loss (cross-entropy) is a proper scoring rule that penalizes confident wrong predictions
more heavily than Brier score. Lower scores indicate better predictions.

NOTE: This computes log loss at trade execution time, which is the correct methodology
for measuring market calibration. Analyses that use price snapshots near resolution
(e.g., price_1d_before) produce artificially low scores because markets have already
converged toward 0 or 1 as outcomes become obvious.

This analysis answers the question traders care about: "When I buy at X%, does the
outcome happen X% of the time?"

INTERPRETATION CAVEAT: Like Brier score, log loss depends on the price distribution of
trades, not just calibration quality. Trades at mid-range prices (40-60¢) naturally have
higher log loss than trades at extreme prices (5¢/95¢), even when perfectly calibrated:
  - Perfectly calibrated 50¢ trade: log loss = log(2) ≈ 0.693
  - Perfectly calibrated 95¢ trade: log loss ≈ 0.05

Rising log loss may indicate traders are trading more uncertain events rather than making
worse predictions. Compare with MAD (calibration_comparison_over_time.py) and price
distribution (price_distribution_over_time.py) to distinguish between worsening
calibration vs. shifting to more uncertain markets.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType

# Bucket size for block-to-timestamp approximation (10800 blocks ~ 6 hours at 2 sec/block)
BLOCK_BUCKET_SIZE = 10800

# Clamp probabilities to avoid log(0)
EPSILON = 1e-6


class CumulativeLogLossOverTimeAnalysis(Analysis):
    """Compare cumulative log loss over calendar time between Polymarket and Kalshi."""

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
            name="cumulative_log_loss_over_time",
            description="Compare cumulative log loss over calendar time between platforms",
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

        with self.progress("Computing log loss"):
            kalshi_results = self._compute_cumulative_log_loss(kalshi_df)
            polymarket_results = self._compute_cumulative_log_loss(polymarket_df)

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

    def _compute_cumulative_log_loss(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute cumulative log loss over time from pre-aggregated data.

        Log loss = -mean(y * log(p) + (1-y) * log(1-p))
        For each price bucket with `total` trades and `wins` wins:
        - wins contribute: -log(p) per trade
        - losses contribute: -log(1-p) per trade
        """
        if df.empty:
            return pd.DataFrame({"date": [], "log_loss": []})

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
        log_losses = []

        for week in weeks:
            totals = cumulative_totals.loc[week].values
            wins = cumulative_wins.loc[week].values
            price_values = cumulative_totals.columns.values.astype(float)

            total_trades = totals.sum()
            # Skip if not enough cumulative data
            if total_trades < 1000:
                continue

            # Compute log loss contribution from each price bucket
            log_loss_sum = 0.0
            for i, price in enumerate(price_values):
                if totals[i] == 0:
                    continue
                # Clamp probability to avoid log(0)
                p = max(min(price / 100.0, 1 - EPSILON), EPSILON)
                bucket_wins = wins[i]
                bucket_losses = totals[i] - wins[i]
                # Wins: -log(p), Losses: -log(1-p)
                log_loss_sum += bucket_wins * (-math.log(p)) + bucket_losses * (-math.log(1 - p))

            log_loss = log_loss_sum / total_trades

            dates.append(week)
            log_losses.append(log_loss)

        return pd.DataFrame({"date": dates, "log_loss": log_losses})

    def _create_figure(self, df: pd.DataFrame) -> plt.Figure:
        """Create the matplotlib figure."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot Kalshi
        kalshi_mask = df["log_loss_kalshi"].notna()
        if kalshi_mask.any():
            ax.plot(
                df.loc[kalshi_mask, "date"],
                df.loc[kalshi_mask, "log_loss_kalshi"],
                color="#4C72B0",
                linewidth=2,
                label="Kalshi",
            )

        # Plot Polymarket
        poly_mask = df["log_loss_polymarket"].notna()
        if poly_mask.any():
            ax.plot(
                df.loc[poly_mask, "date"],
                df.loc[poly_mask, "log_loss_polymarket"],
                color="#55A868",
                linewidth=2,
                label="Polymarket",
            )

        ax.set_xlabel("Date")
        ax.set_ylabel("Log Loss")
        ax.set_title("Cumulative Log Loss Over Time: Polymarket vs Kalshi")

        # Add reference line for random guessing (log(2) ≈ 0.693)
        ax.axhline(
            y=math.log(2),
            color="#D65F5F",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label=f"Random guessing ({math.log(2):.3f})",
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
            if pd.notna(row.get("log_loss_kalshi")):
                entry["kalshi"] = round(row["log_loss_kalshi"], 4)
            if pd.notna(row.get("log_loss_polymarket")):
                entry["polymarket"] = round(row["log_loss_polymarket"], 4)
            chart_data.append(entry)

        return ChartConfig(
            type=ChartType.LINE,
            data=chart_data,
            xKey="date",
            yKeys=["kalshi", "polymarket"],
            title="Cumulative Log Loss Over Time: Polymarket vs Kalshi",
            xLabel="Date",
            yLabel="Log Loss",
            colors={"kalshi": "#4C72B0", "polymarket": "#55A868"},
        )
