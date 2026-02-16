"""Validate T1B OHLCV bars by testing bar-level trading signals.

Uses the 1-hour Kalshi bars to test signals that are only possible with
proper time-aggregated OHLCV data (not available from raw trades alone).

Signals tested:
  1. Bar Net Flow → Next Bar Return (order-flow momentum at bar level)
  2. VWAP-Close Spread → Next Bar Return (institutional vs retail positioning)
  3. Volume Surge + Direction (high-activity bars as directional predictors)
  4. Bar Range Expansion → Volatility Regime (range clustering and return predictability)
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.analysis import Analysis, AnalysisOutput

BASE_DIR = Path(__file__).parent.parent.parent.parent
T1B_KALSHI_1H = BASE_DIR / "data" / "transforms" / "t1b" / "kalshi" / "bars_1h"


class T1BSignalAnalysis(Analysis):
    """Test whether T1B bar-level fields carry tradable signal."""

    def __init__(self):
        super().__init__(
            name="t1b_signal_analysis",
            description="Forward predictive power of T1B OHLCV bar signals (Kalshi 1h)",
        )

    def run(self) -> AnalysisOutput:
        con = duckdb.connect()
        con.execute("SET threads = 8")

        bars_path = f"{T1B_KALSHI_1H}/**/*.parquet"

        # ── Signal 1: Bar Net Flow → Next Bar Return ─────────────
        with self.progress("Signal 1: Bar net flow → next bar return"):
            flow_df = con.execute(f"""
                WITH bars AS (
                    SELECT *,
                        LEAD(bar_return) OVER (
                            PARTITION BY ticker ORDER BY bar_start
                        ) AS next_return
                    FROM read_parquet('{bars_path}')
                    WHERE trade_count > 0
                      AND volume > 0
                ),
                with_signal AS (
                    SELECT *,
                        net_flow / NULLIF(volume, 0) AS flow_imbalance
                    FROM bars
                    WHERE next_return IS NOT NULL
                ),
                bucketed AS (
                    SELECT *,
                        CASE
                            WHEN flow_imbalance < -0.5 THEN '1: <-0.5'
                            WHEN flow_imbalance < -0.2 THEN '2: -0.5 to -0.2'
                            WHEN flow_imbalance < -0.05 THEN '3: -0.2 to -0.05'
                            WHEN flow_imbalance <= 0.05 THEN '4: -0.05 to 0.05'
                            WHEN flow_imbalance <= 0.2 THEN '5: 0.05 to 0.2'
                            WHEN flow_imbalance <= 0.5 THEN '6: 0.2 to 0.5'
                            ELSE '7: >0.5'
                        END AS flow_bucket
                    FROM with_signal
                )
                SELECT
                    flow_bucket,
                    COUNT(*) AS n,
                    AVG(next_return) AS mean_next_return,
                    STDDEV(next_return) / SQRT(COUNT(*)) AS se,
                    100.0 * SUM(CASE WHEN next_return > 0 THEN 1 ELSE 0 END) / COUNT(*) AS pct_positive,
                    100.0 * SUM(CASE WHEN SIGN(next_return) = SIGN(flow_imbalance) THEN 1 ELSE 0 END)
                        / COUNT(*) AS pct_continues
                FROM bucketed
                GROUP BY flow_bucket
                ORDER BY flow_bucket
            """).df()

        # ── Signal 2: VWAP-Close Spread → Next Bar Return ────────
        with self.progress("Signal 2: VWAP-close spread → next bar return"):
            vwap_df = con.execute(f"""
                WITH bars AS (
                    SELECT *,
                        LEAD(bar_return) OVER (
                            PARTITION BY ticker ORDER BY bar_start
                        ) AS next_return,
                        close - vwap AS vwap_spread
                    FROM read_parquet('{bars_path}')
                    WHERE trade_count >= 3
                      AND vwap IS NOT NULL
                      AND bar_range > 0
                ),
                with_normalized AS (
                    SELECT *,
                        vwap_spread / NULLIF(bar_range, 0) AS norm_spread
                    FROM bars
                    WHERE next_return IS NOT NULL
                ),
                bucketed AS (
                    SELECT *,
                        CASE
                            WHEN norm_spread < -0.4 THEN '1: <-0.4 (close << vwap)'
                            WHEN norm_spread < -0.1 THEN '2: -0.4 to -0.1'
                            WHEN norm_spread <= 0.1 THEN '3: -0.1 to 0.1 (close ~ vwap)'
                            WHEN norm_spread <= 0.4 THEN '4: 0.1 to 0.4'
                            ELSE '5: >0.4 (close >> vwap)'
                        END AS spread_bucket
                    FROM with_normalized
                )
                SELECT
                    spread_bucket,
                    COUNT(*) AS n,
                    AVG(next_return) AS mean_next_return,
                    STDDEV(next_return) / SQRT(COUNT(*)) AS se,
                    100.0 * SUM(CASE WHEN next_return > 0 THEN 1 ELSE 0 END) / COUNT(*) AS pct_positive,
                    AVG(norm_spread) AS avg_spread
                FROM bucketed
                GROUP BY spread_bucket
                ORDER BY spread_bucket
            """).df()

        # ── Signal 3: Volume Surge + Direction ────────────────────
        with self.progress("Signal 3: Volume surge → next bar return"):
            surge_df = con.execute(f"""
                WITH bars AS (
                    SELECT *,
                        LEAD(bar_return) OVER w AS next_return,
                        AVG(volume) OVER (
                            PARTITION BY ticker ORDER BY bar_start
                            ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING
                        ) AS avg_vol_12,
                        AVG(trade_count) OVER (
                            PARTITION BY ticker ORDER BY bar_start
                            ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING
                        ) AS avg_tc_12
                    FROM read_parquet('{bars_path}')
                    WHERE trade_count > 0
                    WINDOW w AS (PARTITION BY ticker ORDER BY bar_start)
                ),
                with_surge AS (
                    SELECT *,
                        volume / NULLIF(avg_vol_12, 0) AS vol_ratio,
                        CASE
                            WHEN bar_return > 0 THEN 'up'
                            WHEN bar_return < 0 THEN 'down'
                            ELSE 'flat'
                        END AS bar_direction
                    FROM bars
                    WHERE next_return IS NOT NULL
                      AND avg_vol_12 > 0
                ),
                bucketed AS (
                    SELECT *,
                        CASE
                            WHEN vol_ratio < 0.5 THEN 'low_vol'
                            WHEN vol_ratio < 1.5 THEN 'normal_vol'
                            WHEN vol_ratio < 3.0 THEN 'high_vol'
                            ELSE 'surge'
                        END AS vol_regime
                    FROM with_surge
                )
                SELECT
                    vol_regime,
                    bar_direction,
                    COUNT(*) AS n,
                    AVG(next_return) AS mean_next_return,
                    STDDEV(next_return) / SQRT(COUNT(*)) AS se,
                    100.0 * SUM(CASE WHEN SIGN(next_return) = SIGN(bar_return) THEN 1 ELSE 0 END)
                        / NULLIF(SUM(CASE WHEN bar_return != 0 THEN 1 ELSE 0 END), 0) AS pct_continues
                FROM bucketed
                GROUP BY vol_regime, bar_direction
                ORDER BY vol_regime, bar_direction
            """).df()

        # ── Signal 4: Range Expansion → Next Bar Predictability ───
        with self.progress("Signal 4: Range expansion → next bar behavior"):
            range_df = con.execute(f"""
                WITH bars AS (
                    SELECT *,
                        LEAD(bar_return) OVER w AS next_return,
                        LEAD(bar_range) OVER w AS next_range,
                        AVG(bar_range) OVER (
                            PARTITION BY ticker ORDER BY bar_start
                            ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING
                        ) AS avg_range_12
                    FROM read_parquet('{bars_path}')
                    WHERE trade_count > 0
                    WINDOW w AS (PARTITION BY ticker ORDER BY bar_start)
                ),
                with_regime AS (
                    SELECT *,
                        bar_range / NULLIF(avg_range_12, 0) AS range_ratio
                    FROM bars
                    WHERE next_return IS NOT NULL
                      AND next_range IS NOT NULL
                      AND avg_range_12 > 0
                ),
                bucketed AS (
                    SELECT *,
                        CASE
                            WHEN range_ratio < 0.5 THEN '1: narrow (<0.5x)'
                            WHEN range_ratio < 1.0 THEN '2: below avg (0.5-1x)'
                            WHEN range_ratio < 2.0 THEN '3: above avg (1-2x)'
                            WHEN range_ratio < 4.0 THEN '4: wide (2-4x)'
                            ELSE '5: extreme (>4x)'
                        END AS range_bucket
                    FROM with_regime
                )
                SELECT
                    range_bucket,
                    COUNT(*) AS n,
                    AVG(next_return) AS mean_next_return,
                    AVG(ABS(next_return)) AS mean_abs_next_return,
                    STDDEV(next_return) / SQRT(COUNT(*)) AS se,
                    AVG(next_range / NULLIF(avg_range_12, 0)) AS avg_next_range_ratio,
                    100.0 * SUM(CASE WHEN SIGN(next_return) = SIGN(bar_return) THEN 1 ELSE 0 END)
                        / NULLIF(SUM(CASE WHEN bar_return != 0 THEN 1 ELSE 0 END), 0) AS pct_continues,
                    100.0 * SUM(CASE WHEN next_return > 0 THEN 1 ELSE 0 END) / COUNT(*) AS pct_positive
                FROM bucketed
                GROUP BY range_bucket
                ORDER BY range_bucket
            """).df()

        fig = self._create_figure(flow_df, vwap_df, surge_df, range_df)

        # Build summary table
        summary_rows = []
        for _, r in flow_df.iterrows():
            summary_rows.append({
                "signal": f"Flow: {r['flow_bucket'].split(': ')[1]}",
                "n": int(r["n"]),
                "mean_next_return": round(r["mean_next_return"], 4),
                "se": round(r["se"], 4),
                "pct_continues": round(r["pct_continues"], 2),
            })
        for _, r in vwap_df.iterrows():
            summary_rows.append({
                "signal": f"VWAP: {r['spread_bucket'].split(': ')[1]}",
                "n": int(r["n"]),
                "mean_next_return": round(r["mean_next_return"], 4),
                "se": round(r["se"], 4),
                "pct_continues": None,
            })

        return AnalysisOutput(figure=fig, data=pd.DataFrame(summary_rows))

    def _create_figure(
        self,
        flow_df: pd.DataFrame,
        vwap_df: pd.DataFrame,
        surge_df: pd.DataFrame,
        range_df: pd.DataFrame,
    ) -> plt.Figure:
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle(
            "T1B Signal Analysis: Trading Signals from OHLCV Bars\n"
            "(Kalshi 1h bars — active bars with trades only)",
            fontsize=14, fontweight="bold", y=0.98,
        )

        # ── Panel 1: Net Flow Imbalance → Next Bar Return ─────────
        ax = axes[0, 0]
        x = range(len(flow_df))
        colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(flow_df)))
        bars = ax.bar(
            x, flow_df["mean_next_return"],
            yerr=1.96 * flow_df["se"],
            color=colors, edgecolor="white", linewidth=0.5, capsize=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([b.split(": ")[1] for b in flow_df["flow_bucket"]], fontsize=8)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Bar Flow Imbalance (net_flow / volume)")
        ax.set_ylabel("Mean Next-Bar Return")
        ax.set_title("Signal 1: Bar-Level Flow Imbalance → Next Hour Return")
        for bar, n in zip(bars, flow_df["n"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"n={n:,.0f}", ha="center", va="bottom", fontsize=6,
            )

        # ── Panel 2: VWAP-Close Spread → Next Bar Return ─────────
        ax = axes[0, 1]
        x = range(len(vwap_df))
        colors_v = plt.cm.coolwarm(np.linspace(0.15, 0.85, len(vwap_df)))
        bars = ax.bar(
            x, vwap_df["mean_next_return"],
            yerr=1.96 * vwap_df["se"],
            color=colors_v, edgecolor="white", linewidth=0.5, capsize=3,
        )
        ax.set_xticks(x)
        labels = []
        for b in vwap_df["spread_bucket"]:
            parts = b.split(": ", 1)
            label = parts[1] if len(parts) > 1 else b
            # Truncate long labels
            if "(" in label:
                label = label.split("(")[0].strip()
            labels.append(label)
        ax.set_xticklabels(labels, fontsize=8, rotation=15)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("(Close - VWAP) / Bar Range")
        ax.set_ylabel("Mean Next-Bar Return")
        ax.set_title("Signal 2: VWAP-Close Deviation → Next Hour Return")
        for bar, n in zip(bars, vwap_df["n"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"n={n:,.0f}", ha="center", va="bottom", fontsize=6,
            )

        # ── Panel 3: Volume Surge × Direction → Continuation ──────
        ax = axes[1, 0]
        regime_order = ["low_vol", "normal_vol", "high_vol", "surge"]
        dir_colors = {"up": "#2ecc71", "down": "#e74c3c", "flat": "#95a5a6"}
        width = 0.25
        for i, direction in enumerate(["up", "down", "flat"]):
            sub = surge_df[surge_df["bar_direction"] == direction].copy()
            sub["regime_idx"] = sub["vol_regime"].map(
                {r: j for j, r in enumerate(regime_order)}
            )
            sub = sub.dropna(subset=["regime_idx"]).sort_values("regime_idx")
            offsets = sub["regime_idx"].values + (i - 1) * width
            vals = sub["mean_next_return"].values
            errs = 1.96 * sub["se"].values
            ax.bar(
                offsets, vals, width=width, yerr=errs,
                color=dir_colors[direction], label=f"{direction} bar",
                edgecolor="white", linewidth=0.5, capsize=2,
            )
            for xi, n in zip(offsets, sub["n"].values):
                ax.text(xi, 0, f"{n:,.0f}", ha="center", va="top", fontsize=5, rotation=90)

        ax.set_xticks(range(len(regime_order)))
        ax.set_xticklabels(regime_order, fontsize=9)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Volume Regime (relative to 12-bar average)")
        ax.set_ylabel("Mean Next-Bar Return")
        ax.set_title("Signal 3: Volume Surge × Bar Direction → Next Hour Return")
        ax.legend(fontsize=9)

        # ── Panel 4: Range Expansion → Continuation & Vol Clustering
        ax = axes[1, 1]
        x = range(len(range_df))
        # Dual axis: continuation rate + next range ratio
        color1 = "#3498db"
        color2 = "#e67e22"
        bars1 = ax.bar(
            x, range_df["pct_continues"],
            color=color1, alpha=0.7, edgecolor="white", linewidth=0.5,
            label="Continuation %",
        )
        ax.axhline(50, color="gray", linestyle="--", linewidth=0.8)
        ax.set_ylabel("% Next Bar Continues Direction", color=color1)
        ax.set_ylim(30, 70)

        ax2 = ax.twinx()
        ax2.plot(
            x, range_df["avg_next_range_ratio"],
            color=color2, marker="o", linewidth=2, markersize=8,
            label="Next bar range ratio",
        )
        ax2.axhline(1.0, color=color2, linestyle=":", linewidth=0.8, alpha=0.5)
        ax2.set_ylabel("Avg Next-Bar Range / 12-Bar Avg Range", color=color2)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [b.split(": ")[1] for b in range_df["range_bucket"]],
            fontsize=8,
        )
        ax.set_xlabel("Current Bar Range Regime")
        ax.set_title("Signal 4: Range Expansion → Continuation & Volatility Clustering")
        for bar, n in zip(bars1, range_df["n"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"n={n:,.0f}", ha="center", va="bottom", fontsize=6,
            )

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return fig


if __name__ == "__main__":
    analysis = T1BSignalAnalysis()
    output = analysis.run()

    output_dir = BASE_DIR / "data" / "analysis" / "t1b_signal_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    if output.figure is not None:
        path = output_dir / "t1b_signal_analysis.png"
        output.figure.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Figure saved: {path}")
        plt.close(output.figure)

    if output.data is not None:
        csv_path = output_dir / "t1b_signal_analysis.csv"
        output.data.to_csv(csv_path, index=False)
        print(f"CSV saved: {csv_path}")
        print("\n=== SIGNAL SUMMARY ===")
        print(output.data.to_string(index=False))
