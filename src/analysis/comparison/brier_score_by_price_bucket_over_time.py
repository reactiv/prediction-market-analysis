"""Compare Brier score by price bucket over time between Polymarket and Kalshi.

This analysis breaks down Brier score into price buckets (1-10¢, 11-20¢, ..., 91-99¢) to
understand how market calibration varies by price range over time.

KEY INSIGHT: If MAD (calibration) is improving but overall Brier score is increasing, it
suggests markets are getting better at pricing uncertain events (mid-range prices) while
trading volume shifts toward those uncertain events. This analysis confirms that by showing:
- Decreasing Brier in mid-range buckets (31-50¢) = better calibration of uncertain events
- Stable or increasing Brier in extreme buckets (1-10¢, 91-99¢) = no improvement for easy calls

Expected Brier for perfectly calibrated trades varies by price:
- 5¢ trade: E[Brier] = 0.05 * 0.95 = 0.0475
- 50¢ trade: E[Brier] = 0.50 * 0.50 = 0.25
- 95¢ trade: E[Brier] = 0.95 * 0.05 = 0.0475

NOTE: Uses symmetric buckets (1-10 pairs with 91-99, etc.) because a 10¢ YES trade is
economically equivalent to a 90¢ NO trade on the same outcome.
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

# Price buckets: [(min, max, label), ...]
PRICE_BUCKETS = [
    (1, 10, "1-10¢"),
    (11, 20, "11-20¢"),
    (21, 30, "21-30¢"),
    (31, 40, "31-40¢"),
    (41, 50, "41-50¢"),
    (51, 60, "51-60¢"),
    (61, 70, "61-70¢"),
    (71, 80, "71-80¢"),
    (81, 90, "81-90¢"),
    (91, 99, "91-99¢"),
]

# Colors for each bucket (gradient from blue to red)
BUCKET_COLORS = [
    "#1f77b4",  # 1-10¢
    "#2ca02c",  # 11-20¢
    "#ff7f0e",  # 21-30¢
    "#d62728",  # 31-40¢
    "#9467bd",  # 41-50¢
    "#8c564b",  # 51-60¢
    "#e377c2",  # 61-70¢
    "#7f7f7f",  # 71-80¢
    "#bcbd22",  # 81-90¢
    "#17becf",  # 91-99¢
]


class BrierScoreByPriceBucketOverTimeAnalysis(Analysis):
    """Compare Brier score by price bucket over time between Polymarket and Kalshi."""

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
            name="brier_score_by_price_bucket_over_time",
            description="Compare Brier score by price bucket over time between platforms",
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

        with self.progress("Computing Brier scores by bucket"):
            kalshi_results = self._compute_brier_by_bucket(kalshi_df, "kalshi")
            polymarket_results = self._compute_brier_by_bucket(polymarket_df, "polymarket")

        # Combine results
        output_df = pd.concat([kalshi_results, polymarket_results], ignore_index=True)

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

    def _compute_brier_by_bucket(self, df: pd.DataFrame, platform: str) -> pd.DataFrame:
        """Compute cumulative Brier score by price bucket over time.

        Brier score = mean((p - y)²) where p is predicted probability, y is outcome (0 or 1)
        For each price bucket with `total` trades and `wins` wins:
        - wins contribute: (price/100 - 1)² per trade
        - losses contribute: (price/100 - 0)² per trade
        """
        if df.empty:
            return pd.DataFrame({"date": [], "platform": [], "bucket": [], "brier_score": []})

        # Normalize week column to timezone-naive
        df = df.copy()
        if df["week"].dt.tz is not None:
            df["week"] = df["week"].dt.tz_localize(None)

        results = []

        for bucket_min, bucket_max, bucket_label in PRICE_BUCKETS:
            # Filter to prices in this bucket
            bucket_df = df[(df["price"] >= bucket_min) & (df["price"] <= bucket_max)]
            if bucket_df.empty:
                continue

            # Pivot to get weekly totals/wins per price
            total_pivot = bucket_df.pivot(index="week", columns="price", values="total").fillna(0)
            wins_pivot = bucket_df.pivot(index="week", columns="price", values="wins").fillna(0)

            # Compute cumulative sums over weeks
            cumulative_totals = total_pivot.cumsum()
            cumulative_wins = wins_pivot.cumsum()

            weeks = cumulative_totals.index.tolist()

            for week in weeks:
                totals = cumulative_totals.loc[week].values
                wins = cumulative_wins.loc[week].values
                price_values = cumulative_totals.columns.values.astype(float)

                total_trades = totals.sum()
                # Skip if not enough cumulative data for this bucket
                if total_trades < 100:
                    continue

                # Compute Brier score contribution from each price in the bucket
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

                results.append(
                    {
                        "date": week,
                        "platform": platform,
                        "bucket": bucket_label,
                        "brier_score": brier_score,
                    }
                )

        return pd.DataFrame(results)

    def _create_figure(self, df: pd.DataFrame) -> plt.Figure:
        """Create the matplotlib figure with subplots for each platform."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        bucket_labels = [b[2] for b in PRICE_BUCKETS]
        bucket_colors = dict(zip(bucket_labels, BUCKET_COLORS))

        for ax, platform in zip(axes, ["kalshi", "polymarket"]):
            platform_df = df[df["platform"] == platform]

            for bucket_label in bucket_labels:
                bucket_data = platform_df[platform_df["bucket"] == bucket_label].sort_values("date")
                if bucket_data.empty:
                    continue

                ax.plot(
                    bucket_data["date"],
                    bucket_data["brier_score"],
                    color=bucket_colors[bucket_label],
                    linewidth=1.5,
                    label=bucket_label,
                    alpha=0.8,
                )

            ax.set_xlabel("Date")
            ax.set_ylabel("Brier Score")
            ax.set_title(f"Brier Score by Price Bucket: {platform.title()}")
            ax.legend(loc="upper right", fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)

        fig.autofmt_xdate()
        plt.tight_layout()
        return fig

    def _create_chart(self, df: pd.DataFrame) -> ChartConfig:
        """Create the chart configuration for web display."""
        # Pivot data for chart format - one row per date, columns for each platform-bucket combo
        chart_data = []

        # Get unique dates across both platforms
        all_dates = df["date"].unique()

        for date in sorted(all_dates):
            entry = {"date": pd.Timestamp(date).strftime("%Y-%m-%d")}
            date_df = df[df["date"] == date]

            for _, row in date_df.iterrows():
                key = f"{row['platform']}_{row['bucket'].replace('¢', '').replace('-', '_')}"
                entry[key] = round(row["brier_score"], 4)

            chart_data.append(entry)

        # Build yKeys dynamically
        y_keys = []
        colors = {}
        bucket_labels = [b[2] for b in PRICE_BUCKETS]

        for platform in ["kalshi", "polymarket"]:
            for i, bucket_label in enumerate(bucket_labels):
                key = f"{platform}_{bucket_label.replace('¢', '').replace('-', '_')}"
                y_keys.append(key)
                # Use different shades for platforms
                base_color = BUCKET_COLORS[i]
                colors[key] = base_color

        return ChartConfig(
            type=ChartType.LINE,
            data=chart_data,
            xKey="date",
            yKeys=y_keys,
            title="Brier Score by Price Bucket Over Time",
            xLabel="Date",
            yLabel="Brier Score",
            colors=colors,
        )
