"""T6: Rolling calibration scores for Kalshi markets.

Computes daily calibration metrics (MAE between win rate and implied probability)
across rolling windows (7d, 30d, 90d), broken down by category, price bucket,
time-to-expiry bucket, and taker side. The feature date is the *resolution date*
(`close_time`) to avoid using future outcomes at trade time.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from src.analysis.kalshi.util.categories import CATEGORY_SQL
from src.transforms._base import Transform


class T6RollingCalibration(Transform):
    def __init__(self):
        super().__init__(
            name="t6",
            description="Rolling calibration scores (Kalshi)",
            dependencies=[],
        )

    def run(self):
        self.ensure_output_dir()

        with self.progress("Building daily aggregates"):
            daily_df = self._build_daily_aggregates()
            daily_rows = len(daily_df)

        with self.progress("Computing rolling features"):
            features_df = self._compute_rolling_features(daily_df)
            feature_rows = len(features_df)

        with self.progress("Computing regime flags"):
            self._compute_regime_flags(features_df)

        with self.progress("Computing resolution lag"):
            self._compute_resolution_lag()

        self.write_manifest(
            {
                "daily_aggregate_rows": daily_rows,
                "feature_rows": feature_rows,
            }
        )

    def _build_daily_aggregates(self) -> pd.DataFrame:
        """Step 1+2: Join trades with resolved markets and aggregate by resolution day.

        Using resolution day (`close_time`) keeps features causal: a row for date D
        only depends on outcomes that were known by D.
        """
        trades_glob = str(self.base_dir / "data" / "kalshi" / "trades" / "*.parquet")
        markets_glob = str(self.base_dir / "data" / "kalshi" / "markets" / "*.parquet")

        con = duckdb.connect()

        # Load trades and markets into views
        con.execute(
            f"CREATE VIEW trades AS SELECT * FROM read_parquet('{trades_glob}')"
        )
        con.execute(
            f"""
            CREATE VIEW resolved_markets AS
            SELECT * FROM read_parquet('{markets_glob}')
            WHERE status = 'finalized' AND result IN ('yes', 'no')
        """
        )

        # Step 1: Build resolved trade records
        # Step 2: Aggregate to daily level
        query = f"""
            WITH resolved_trades AS (
                SELECT
                    t.trade_id,
                    t.ticker,
                    t.count,
                    t.yes_price,
                    t.no_price,
                    t.taker_side,
                    t.created_time,
                    m.event_ticker,
                    m.status,
                    m.result,
                    m.close_time,
                    m.open_time,
                    {CATEGORY_SQL} AS category,
                    FLOOR(
                        CASE WHEN t.taker_side = 'yes' THEN t.yes_price
                             ELSE t.no_price
                        END / 5
                    ) * 5 AS price_bucket,
                    CASE
                        WHEN EPOCH(m.close_time - t.created_time) < 3600 THEN '<1h'
                        WHEN EPOCH(m.close_time - t.created_time) < 21600 THEN '1-6h'
                        WHEN EPOCH(m.close_time - t.created_time) < 86400 THEN '6-24h'
                        WHEN EPOCH(m.close_time - t.created_time) < 604800 THEN '1-7d'
                        ELSE '7d+'
                    END AS tte_bucket,
                    CASE WHEN t.taker_side = m.result THEN 1 ELSE 0 END AS won,
                    DATE_TRUNC('day', m.close_time) AS trade_date,
                    EPOCH(m.close_time - t.created_time) AS tte_seconds
                FROM trades t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
            )
            SELECT
                trade_date,
                category,
                price_bucket,
                tte_bucket,
                taker_side,
                SUM(count) AS total_trades,
                SUM(won * count) AS total_wins,
                SUM(won * count) * 1.0 / SUM(count) AS win_rate,
                price_bucket / 100.0 AS implied_prob,
                SUM(CASE WHEN taker_side = 'yes' THEN count ELSE 0 END) AS yes_taker_count,
                SUM(CASE WHEN taker_side = 'no' THEN count ELSE 0 END) AS no_taker_count,
                SUM(count) AS total_volume,
                AVG(tte_seconds) AS avg_tte_seconds
            FROM resolved_trades
            GROUP BY trade_date, category, price_bucket, tte_bucket, taker_side
            ORDER BY trade_date, category, price_bucket, tte_bucket, taker_side
        """

        daily_df = con.execute(query).fetchdf()
        con.close()

        # Ensure proper types
        daily_df["trade_date"] = pd.to_datetime(daily_df["trade_date"])
        daily_df["price_bucket"] = daily_df["price_bucket"].astype(int)
        daily_df["implied_prob"] = daily_df["implied_prob"].astype(float)
        daily_df["win_rate"] = daily_df["win_rate"].astype(float)
        daily_df["total_trades"] = daily_df["total_trades"].astype(int)
        daily_df["total_wins"] = daily_df["total_wins"].astype(int)

        # Compute per-row calibration error
        daily_df["calibration_error"] = np.abs(
            daily_df["win_rate"] - daily_df["implied_prob"]
        )
        daily_df["yes_taker_ratio"] = (
            daily_df["yes_taker_count"] / daily_df["total_trades"]
        )

        return daily_df

    def _compute_rolling_features(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """Step 3: Compute rolling metrics per full T6 cell.

        Rolling windows are computed per
        (category, price_bucket, tte_bucket, taker_side) to preserve
        signal granularity and avoid side mixing.
        """
        if daily_df.empty:
            features_df = daily_df.copy()
            for col in [
                "mae_7d",
                "mae_30d",
                "mae_90d",
                "opportunity_score",
                "yes_ratio_7d",
            ]:
                features_df[col] = np.nan
        else:
            group_cols = ["category", "price_bucket", "tte_bucket", "taker_side"]
            rolling_parts: list[pd.DataFrame] = []

            for _, cell_df in daily_df.groupby(group_cols, sort=False):
                cdf = cell_df.sort_values("trade_date").copy().set_index("trade_date")
                weighted_abs_error = cdf["calibration_error"] * cdf["total_trades"]

                for window_days in [7, 30, 90]:
                    window_str = f"{window_days}D"
                    col_suffix = f"_{window_days}d"

                    roll_total_trades = cdf["total_trades"].rolling(
                        window_str, min_periods=1
                    ).sum()
                    roll_weighted_err = weighted_abs_error.rolling(
                        window_str, min_periods=1
                    ).sum()
                    cdf[f"mae{col_suffix}"] = roll_weighted_err / roll_total_trades

                    roll_yes = cdf["yes_taker_count"].rolling(
                        window_str, min_periods=1
                    ).sum()
                    cdf[f"yes_ratio{col_suffix}"] = roll_yes / roll_total_trades

                    cdf[f"volume{col_suffix}"] = cdf["total_volume"].rolling(
                        window_str, min_periods=1
                    ).sum()

                cdf["opportunity_score"] = (
                    cdf["mae_7d"]
                    * np.log1p(cdf["volume_7d"])
                    * (1 + np.abs(cdf["yes_ratio_7d"] - 0.5))
                )
                rolling_parts.append(cdf.reset_index())

            features_df = pd.concat(rolling_parts, ignore_index=True)

        # Select output columns
        output_cols = [
            "trade_date",
            "category",
            "price_bucket",
            "tte_bucket",
            "taker_side",
            "total_trades",
            "total_wins",
            "win_rate",
            "implied_prob",
            "calibration_error",
            "yes_taker_ratio",
            "total_volume",
            "mae_7d",
            "mae_30d",
            "mae_90d",
            "opportunity_score",
            "yes_ratio_7d",
        ]

        # Rename total_volume to volume for output
        features_df = features_df.rename(columns={"total_volume": "volume"})
        output_cols = [c if c != "total_volume" else "volume" for c in output_cols]

        features_df = features_df[output_cols]
        features_df = features_df.sort_values(
            ["trade_date", "category", "price_bucket", "tte_bucket", "taker_side"]
        ).reset_index(drop=True)

        # Write daily features
        output_path = self.output_dir / "daily_features.parquet"
        features_df.to_parquet(output_path, index=False)

        return features_df

    def _compute_regime_flags(self, features_df: pd.DataFrame) -> None:
        """Step 4: Regime flags when 7d MAE crosses above/below 90d MAE."""
        key_cols = ["category", "price_bucket", "tte_bucket", "taker_side"]
        if features_df.empty:
            regime_df = pd.DataFrame(
                columns=[
                    "trade_date",
                    "category",
                    "price_bucket",
                    "tte_bucket",
                    "taker_side",
                    "mae_7d",
                    "mae_90d",
                    "regime_flag",
                ]
            )
            regime_df.to_parquet(
                self.output_dir / "regime_flags.parquet", index=False
            )
            return

        # Get unique cell/date rows with their rolling MAEs
        regime_df = (
            features_df.groupby(["trade_date", *key_cols])
            .agg(mae_7d=("mae_7d", "first"), mae_90d=("mae_90d", "first"))
            .reset_index()
            .sort_values([*key_cols, "trade_date"])
        )

        # Compute regime flag: 1 when 7d > 90d, -1 when 7d < 90d, 0 at start
        def compute_flag(group):
            group = group.copy()
            flags = np.where(
                group["mae_90d"].isna() | group["mae_7d"].isna(),
                0,
                np.where(
                    group["mae_7d"] > group["mae_90d"],
                    1,
                    np.where(group["mae_7d"] < group["mae_90d"], -1, 0),
                ),
            )
            group["regime_flag"] = flags.astype(int)
            return group

        regime_df = (
            regime_df.groupby(key_cols, group_keys=False)
            .apply(compute_flag)
            .reset_index(drop=True)
        )

        output_cols = [
            "trade_date",
            "category",
            "price_bucket",
            "tte_bucket",
            "taker_side",
            "mae_7d",
            "mae_90d",
            "regime_flag",
        ]
        regime_df[output_cols].to_parquet(
            self.output_dir / "regime_flags.parquet", index=False
        )

    def _compute_resolution_lag(self) -> None:
        """Compute per-category resolution lag statistics."""
        trades_glob = str(self.base_dir / "data" / "kalshi" / "trades" / "*.parquet")
        markets_glob = str(self.base_dir / "data" / "kalshi" / "markets" / "*.parquet")

        con = duckdb.connect()
        con.execute(
            f"CREATE VIEW trades AS SELECT * FROM read_parquet('{trades_glob}')"
        )
        con.execute(
            f"""
            CREATE VIEW resolved_markets AS
            SELECT * FROM read_parquet('{markets_glob}')
            WHERE status = 'finalized' AND result IN ('yes', 'no')
        """
        )

        query = f"""
            WITH trade_lags AS (
                SELECT
                    {CATEGORY_SQL} AS category,
                    EPOCH(m.close_time - t.created_time) / 3600.0 AS lag_hours
                FROM trades t
                INNER JOIN resolved_markets m ON t.ticker = m.ticker
            )
            SELECT
                category,
                MEDIAN(lag_hours) AS median_resolution_lag_hours,
                QUANTILE(lag_hours, 0.25) AS p25_lag,
                QUANTILE(lag_hours, 0.75) AS p75_lag,
                COUNT(*) AS total_resolved
            FROM trade_lags
            GROUP BY category
            ORDER BY category
        """

        lag_df = con.execute(query).fetchdf()
        con.close()

        lag_df.to_parquet(
            self.output_dir / "resolution_lag.parquet", index=False
        )
