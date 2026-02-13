"""Analyze whether markets are less well-calibrated early in their lifecycle.

Computes calibration deviation (actual win rate minus implied probability)
at different stages of a market's life, from open to close. Tests the
hypothesis that price discovery takes time and early trades are mispriced.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


LIFECYCLE_BINS = [
    (0.0, 0.05, "0-5%"),
    (0.05, 0.15, "5-15%"),
    (0.15, 0.30, "15-30%"),
    (0.30, 0.50, "30-50%"),
    (0.50, 0.70, "50-70%"),
    (0.70, 0.85, "70-85%"),
    (0.85, 0.95, "85-95%"),
    (0.95, 1.0, "95-100%"),
]

LIFECYCLE_CASE = "\n".join(
    f"WHEN life_pct >= {lo} AND life_pct < {hi} THEN '{label}'"
    for lo, hi, label in LIFECYCLE_BINS[:-1]
) + f"\nWHEN life_pct >= {LIFECYCLE_BINS[-1][0]} THEN '{LIFECYCLE_BINS[-1][2]}'"


class EarlyMarketInefficiencyAnalysis(Analysis):
    """Analyze calibration quality across a market's lifecycle."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="early_market_inefficiency",
            description="Market calibration by lifecycle stage (early vs late)",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        con = duckdb.connect()

        with self.progress("Querying calibration by lifecycle stage"):
            df = con.execute(
                f"""
                WITH resolved_markets AS (
                    SELECT ticker, result, open_time, close_time
                    FROM '{self.markets_dir}/*.parquet'
                    WHERE status = 'finalized'
                      AND result IN ('yes', 'no')
                      AND open_time IS NOT NULL
                      AND close_time IS NOT NULL
                      AND close_time > open_time
                ),
                positions AS (
                    -- Taker side
                    SELECT
                        CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS price,
                        CASE WHEN t.taker_side = m.result THEN 1 ELSE 0 END AS won,
                        LEAST(1.0, GREATEST(0.0,
                            EPOCH(t.created_time - m.open_time) / EPOCH(m.close_time - m.open_time)
                        )) AS life_pct
                    FROM '{self.trades_dir}/*.parquet' t
                    INNER JOIN resolved_markets m ON t.ticker = m.ticker

                    UNION ALL

                    -- Maker side (counterparty)
                    SELECT
                        CASE WHEN t.taker_side = 'yes' THEN t.no_price ELSE t.yes_price END AS price,
                        CASE WHEN t.taker_side != m.result THEN 1 ELSE 0 END AS won,
                        LEAST(1.0, GREATEST(0.0,
                            EPOCH(t.created_time - m.open_time) / EPOCH(m.close_time - m.open_time)
                        )) AS life_pct
                    FROM '{self.trades_dir}/*.parquet' t
                    INNER JOIN resolved_markets m ON t.ticker = m.ticker
                ),
                bucketed AS (
                    SELECT
                        price,
                        won,
                        life_pct,
                        CASE
                            {LIFECYCLE_CASE}
                        END AS stage
                    FROM positions
                    WHERE price BETWEEN 1 AND 99
                )
                SELECT
                    stage,
                    price,
                    COUNT(*) AS n,
                    SUM(won) AS wins,
                    100.0 * SUM(won) / COUNT(*) AS win_rate,
                    100.0 * SUM(won) / COUNT(*) - price AS deviation
                FROM bucketed
                GROUP BY stage, price
                ORDER BY stage, price
                """
            ).df()

        with self.progress("Computing summary statistics"):
            summary = con.execute(
                f"""
                WITH resolved_markets AS (
                    SELECT ticker, result, open_time, close_time
                    FROM '{self.markets_dir}/*.parquet'
                    WHERE status = 'finalized'
                      AND result IN ('yes', 'no')
                      AND open_time IS NOT NULL
                      AND close_time IS NOT NULL
                      AND close_time > open_time
                ),
                positions AS (
                    SELECT
                        CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS price,
                        CASE WHEN t.taker_side = m.result THEN 1 ELSE 0 END AS won,
                        LEAST(1.0, GREATEST(0.0,
                            EPOCH(t.created_time - m.open_time) / EPOCH(m.close_time - m.open_time)
                        )) AS life_pct
                    FROM '{self.trades_dir}/*.parquet' t
                    INNER JOIN resolved_markets m ON t.ticker = m.ticker

                    UNION ALL

                    SELECT
                        CASE WHEN t.taker_side = 'yes' THEN t.no_price ELSE t.yes_price END AS price,
                        CASE WHEN t.taker_side != m.result THEN 1 ELSE 0 END AS won,
                        LEAST(1.0, GREATEST(0.0,
                            EPOCH(t.created_time - m.open_time) / EPOCH(m.close_time - m.open_time)
                        )) AS life_pct
                    FROM '{self.trades_dir}/*.parquet' t
                    INNER JOIN resolved_markets m ON t.ticker = m.ticker
                ),
                bucketed AS (
                    SELECT
                        price,
                        won,
                        CASE
                            {LIFECYCLE_CASE}
                        END AS stage
                    FROM positions
                    WHERE price BETWEEN 1 AND 99
                ),
                by_stage_price AS (
                    SELECT
                        stage,
                        price,
                        COUNT(*) AS n,
                        100.0 * SUM(won) / COUNT(*) - price AS deviation
                    FROM bucketed
                    GROUP BY stage, price
                )
                SELECT
                    stage,
                    SUM(n) AS total_trades,
                    AVG(ABS(deviation)) AS mae,
                    SQRT(AVG(deviation * deviation)) AS rmse,
                    AVG(deviation) AS mean_deviation
                FROM by_stage_price
                GROUP BY stage
                ORDER BY stage
                """
            ).df()

        fig = self._create_figure(df, summary)
        chart = self._create_chart(summary)

        return AnalysisOutput(figure=fig, data=summary, chart=chart)

    def _create_figure(self, df: pd.DataFrame, summary: pd.DataFrame) -> plt.Figure:
        stage_order = [label for _, _, label in LIFECYCLE_BINS]
        summary = summary.copy()
        summary["stage"] = pd.Categorical(summary["stage"], categories=stage_order, ordered=True)
        summary = summary.sort_values("stage")

        cmap = plt.cm.RdYlGn
        n_stages = len(stage_order)
        colors = [cmap(i / (n_stages - 1)) for i in range(n_stages)]

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Left: calibration curves per lifecycle stage (early vs late)
        ax = axes[0]
        early_stages = stage_order[:2]
        late_stages = stage_order[-2:]
        for stage, color, label_prefix in [
            (early_stages, "#e74c3c", "Early"),
            (late_stages, "#2ecc71", "Late"),
        ]:
            mask = df["stage"].isin(stage)
            grouped = df[mask].groupby("price").agg(
                wins=("wins", "sum"), n=("n", "sum")
            ).reset_index()
            grouped["win_rate"] = 100.0 * grouped["wins"] / grouped["n"]
            ax.scatter(
                grouped["price"],
                grouped["win_rate"],
                s=15,
                alpha=0.6,
                color=color,
                label=f"{label_prefix} ({', '.join(stage)})",
            )

        ax.plot([0, 100], [0, 100], linestyle="--", color="gray", linewidth=1, label="Perfect calibration")
        ax.set_xlabel("Contract Price (cents)")
        ax.set_ylabel("Win Rate (%)")
        ax.set_title("Calibration: Early vs Late in Market Lifecycle")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect("equal")
        ax.legend(loc="upper left", fontsize=9)

        # Right: MAE by lifecycle stage
        ax2 = axes[1]
        bars = ax2.bar(
            range(len(summary)),
            summary["mae"],
            color=colors[: len(summary)],
            edgecolor="white",
            linewidth=0.5,
        )
        ax2.set_xticks(range(len(summary)))
        ax2.set_xticklabels(summary["stage"], rotation=45, ha="right", fontsize=9)
        ax2.set_xlabel("Market Lifecycle Stage")
        ax2.set_ylabel("Mean Absolute Calibration Error (pp)")
        ax2.set_title("Calibration Error by Lifecycle Stage")

        for bar, val in zip(bars, summary["mae"]):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()
        return fig

    def _create_chart(self, summary: pd.DataFrame) -> ChartConfig:
        stage_order = [label for _, _, label in LIFECYCLE_BINS]
        summary = summary.copy()
        summary["stage"] = pd.Categorical(summary["stage"], categories=stage_order, ordered=True)
        summary = summary.sort_values("stage")

        chart_data = [
            {
                "stage": row["stage"],
                "MAE": round(row["mae"], 2),
                "RMSE": round(row["rmse"], 2),
                "trades": int(row["total_trades"]),
            }
            for _, row in summary.iterrows()
        ]

        return ChartConfig(
            type=ChartType.BAR,
            data=chart_data,
            xKey="stage",
            yKeys=["MAE", "RMSE"],
            title="Calibration Error by Market Lifecycle Stage",
            yUnit=UnitType.NUMBER,
            xLabel="Lifecycle Stage (% of market duration elapsed)",
            yLabel="Calibration Error (percentage points)",
        )
