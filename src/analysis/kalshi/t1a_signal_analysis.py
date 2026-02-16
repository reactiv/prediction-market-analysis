"""Assess the value of T1A enrichment by testing candidate trading signals.

Uses the sequential context fields from T1A (delta_price, cumulative_net_flow,
cumulative_volume, time_since_prev, trade_sequence_num, time_to_expiry_seconds)
to construct candidate signals and measure their forward predictive power on
Kalshi resolved markets.

Signals tested:
  1. Order-Flow Imbalance (OFI): cumulative_net_flow / cumulative_volume
  2. Trade-Burst Momentum: signed flow during rapid-fire trade clusters
  3. Price Streak: consecutive same-direction delta_price moves
  4. Time-to-Expiry Regime: interaction of signals with market lifecycle
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.analysis import Analysis, AnalysisOutput

BASE_DIR = Path(__file__).parent.parent.parent.parent
T1A_KALSHI = BASE_DIR / "data" / "transforms" / "t1a" / "kalshi"
MARKETS_DIR = BASE_DIR / "data" / "kalshi" / "markets"


class T1ASignalAnalysis(Analysis):
    """Test whether T1A enrichment fields carry predictive signal."""

    def __init__(self):
        super().__init__(
            name="t1a_signal_analysis",
            description="Forward predictive power of T1A enrichment signals",
        )

    def run(self) -> AnalysisOutput:
        con = duckdb.connect()
        con.execute("SET threads = 8")

        # ── Signal 1: Order-Flow Imbalance ──────────────────────────
        with self.progress("Signal 1: Order-flow imbalance → forward return"):
            ofi_df = con.execute(f"""
                WITH trades AS (
                    SELECT *,
                        LEAD(norm_price) OVER (
                            PARTITION BY ticker ORDER BY trade_sequence_num
                        ) AS next_price,
                        cumulative_net_flow / NULLIF(cumulative_volume, 0) AS ofi_ratio
                    FROM '{T1A_KALSHI}/*.parquet'
                    WHERE cumulative_volume > 0
                      AND trade_sequence_num >= 5
                ),
                with_forward AS (
                    SELECT *,
                        next_price - norm_price AS forward_return
                    FROM trades
                    WHERE next_price IS NOT NULL
                ),
                bucketed AS (
                    SELECT *,
                        CASE
                            WHEN ofi_ratio < -0.6 THEN '1: <-0.6'
                            WHEN ofi_ratio < -0.3 THEN '2: -0.6 to -0.3'
                            WHEN ofi_ratio < -0.1 THEN '3: -0.3 to -0.1'
                            WHEN ofi_ratio <  0.1 THEN '4: -0.1 to 0.1'
                            WHEN ofi_ratio <  0.3 THEN '5: 0.1 to 0.3'
                            WHEN ofi_ratio <  0.6 THEN '6: 0.3 to 0.6'
                            ELSE '7: >0.6'
                        END AS ofi_bucket
                    FROM with_forward
                )
                SELECT
                    ofi_bucket,
                    COUNT(*) AS n,
                    AVG(forward_return) AS mean_fwd_return,
                    STDDEV(forward_return) / SQRT(COUNT(*)) AS se_fwd_return,
                    100.0 * SUM(CASE WHEN forward_return > 0 THEN 1 ELSE 0 END) / COUNT(*) AS pct_positive,
                    AVG(ABS(forward_return)) AS mean_abs_fwd_return
                FROM bucketed
                GROUP BY ofi_bucket
                ORDER BY ofi_bucket
            """).df()

        # ── Signal 2: Trade-Burst Momentum ──────────────────────────
        with self.progress("Signal 2: Trade-burst momentum → forward return"):
            burst_df = con.execute(f"""
                WITH trades AS (
                    SELECT *,
                        LEAD(norm_price) OVER (
                            PARTITION BY ticker ORDER BY trade_sequence_num
                        ) AS next_price,
                        -- rolling 5-trade signed flow
                        SUM(signed_flow) OVER (
                            PARTITION BY ticker ORDER BY trade_sequence_num
                            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                        ) AS recent_flow_5,
                        -- count trades in last 5 with time_since_prev < 10 sec
                        SUM(CASE WHEN time_since_prev < 10 THEN 1 ELSE 0 END) OVER (
                            PARTITION BY ticker ORDER BY trade_sequence_num
                            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                        ) AS burst_count_5
                    FROM '{T1A_KALSHI}/*.parquet'
                    WHERE trade_sequence_num >= 6
                ),
                bursts AS (
                    SELECT *,
                        next_price - norm_price AS forward_return,
                        CASE
                            WHEN burst_count_5 >= 4 AND recent_flow_5 > 0 THEN 'burst_bullish'
                            WHEN burst_count_5 >= 4 AND recent_flow_5 < 0 THEN 'burst_bearish'
                            WHEN burst_count_5 >= 4 AND recent_flow_5 = 0 THEN 'burst_neutral'
                            WHEN burst_count_5 <= 1 THEN 'quiet'
                            ELSE 'moderate'
                        END AS burst_regime
                    FROM trades
                    WHERE next_price IS NOT NULL
                )
                SELECT
                    burst_regime,
                    COUNT(*) AS n,
                    AVG(forward_return) AS mean_fwd_return,
                    STDDEV(forward_return) / SQRT(COUNT(*)) AS se_fwd_return,
                    100.0 * SUM(CASE WHEN forward_return > 0 THEN 1 ELSE 0 END) / COUNT(*) AS pct_positive,
                    AVG(ABS(forward_return)) AS mean_abs_fwd_return
                FROM bursts
                GROUP BY burst_regime
                ORDER BY burst_regime
            """).df()

        # ── Signal 3: Price Streak → Mean Reversion ─────────────────
        with self.progress("Signal 3: Price streak length → forward return"):
            streak_df = con.execute(f"""
                WITH trades AS (
                    SELECT *,
                        LEAD(norm_price) OVER (
                            PARTITION BY ticker ORDER BY trade_sequence_num
                        ) AS next_price,
                        SIGN(delta_price) AS move_dir
                    FROM '{T1A_KALSHI}/*.parquet'
                    WHERE trade_sequence_num >= 2
                ),
                with_forward AS (
                    SELECT *,
                        next_price - norm_price AS forward_return
                    FROM trades
                    WHERE next_price IS NOT NULL
                      AND delta_price != 0
                ),
                with_streak AS (
                    SELECT *,
                        -- count consecutive same-direction moves (look back up to 10)
                        CASE
                            WHEN LAG(move_dir, 1) OVER w != move_dir THEN 1
                            WHEN LAG(move_dir, 2) OVER w != move_dir OR LAG(move_dir, 2) OVER w IS NULL THEN 2
                            WHEN LAG(move_dir, 3) OVER w != move_dir OR LAG(move_dir, 3) OVER w IS NULL THEN 3
                            WHEN LAG(move_dir, 4) OVER w != move_dir OR LAG(move_dir, 4) OVER w IS NULL THEN 4
                            ELSE 5
                        END AS streak_len
                    FROM with_forward
                    WINDOW w AS (PARTITION BY ticker ORDER BY trade_sequence_num)
                )
                SELECT
                    streak_len,
                    move_dir AS streak_direction,
                    COUNT(*) AS n,
                    AVG(forward_return) AS mean_fwd_return,
                    STDDEV(forward_return) / SQRT(COUNT(*)) AS se_fwd_return,
                    100.0 * SUM(CASE WHEN SIGN(forward_return) = move_dir THEN 1 ELSE 0 END)
                        / COUNT(*) AS pct_continues,
                    AVG(ABS(forward_return)) AS mean_abs_fwd_return
                FROM with_streak
                GROUP BY streak_len, move_dir
                ORDER BY streak_len, move_dir
            """).df()

        # ── Signal 4: OFI × Time-to-Expiry Interaction ──────────────
        with self.progress("Signal 4: OFI signal strength by time-to-expiry"):
            tte_df = con.execute(f"""
                WITH trades AS (
                    SELECT *,
                        LEAD(norm_price) OVER (
                            PARTITION BY ticker ORDER BY trade_sequence_num
                        ) AS next_price,
                        cumulative_net_flow / NULLIF(cumulative_volume, 0) AS ofi_ratio,
                        CASE
                            WHEN time_to_expiry_seconds > 7*86400 THEN '1: >7d'
                            WHEN time_to_expiry_seconds > 86400 THEN '2: 1-7d'
                            WHEN time_to_expiry_seconds > 3600 THEN '3: 1-24h'
                            WHEN time_to_expiry_seconds > 600 THEN '4: 10-60m'
                            ELSE '5: <10m'
                        END AS tte_bucket
                    FROM '{T1A_KALSHI}/*.parquet'
                    WHERE cumulative_volume > 0
                      AND trade_sequence_num >= 5
                      AND time_to_expiry_seconds IS NOT NULL
                      AND time_to_expiry_seconds > 0
                ),
                with_forward AS (
                    SELECT *,
                        next_price - norm_price AS forward_return
                    FROM trades
                    WHERE next_price IS NOT NULL
                ),
                with_signal AS (
                    SELECT *,
                        CASE
                            WHEN ofi_ratio > 0.15 THEN 'long'
                            WHEN ofi_ratio < -0.15 THEN 'short'
                            ELSE 'flat'
                        END AS ofi_signal
                    FROM with_forward
                )
                SELECT
                    tte_bucket,
                    ofi_signal,
                    COUNT(*) AS n,
                    AVG(forward_return) AS mean_fwd_return,
                    STDDEV(forward_return) / SQRT(COUNT(*)) AS se_fwd_return,
                    -- signal return: long gets +fwd, short gets -fwd
                    AVG(CASE
                        WHEN ofi_signal = 'long' THEN forward_return
                        WHEN ofi_signal = 'short' THEN -forward_return
                        ELSE 0
                    END) AS signal_return,
                    100.0 * SUM(CASE
                        WHEN ofi_signal = 'long' AND forward_return > 0 THEN 1
                        WHEN ofi_signal = 'short' AND forward_return < 0 THEN 1
                        WHEN ofi_signal = 'flat' THEN 0
                        ELSE 0
                    END) / NULLIF(SUM(CASE WHEN ofi_signal != 'flat' THEN 1 ELSE 0 END), 0) AS hit_rate
                FROM with_signal
                GROUP BY tte_bucket, ofi_signal
                ORDER BY tte_bucket, ofi_signal
            """).df()

        # ── Aggregate summary table ──────────────────────────────────
        with self.progress("Building summary"):
            summary_rows = []

            # OFI extremes
            ofi_long = ofi_df[ofi_df["ofi_bucket"].str.contains(">0.6")]
            ofi_short = ofi_df[ofi_df["ofi_bucket"].str.contains("<-0.6")]
            ofi_neutral = ofi_df[ofi_df["ofi_bucket"].str.contains("-0.1 to 0.1")]
            for label, row_df in [("OFI > 0.6", ofi_long), ("OFI < -0.6", ofi_short), ("OFI neutral", ofi_neutral)]:
                if len(row_df) > 0:
                    r = row_df.iloc[0]
                    summary_rows.append({
                        "signal": label,
                        "n": int(r["n"]),
                        "mean_fwd_return_cents": round(r["mean_fwd_return"], 4),
                        "se": round(r["se_fwd_return"], 4),
                        "pct_positive": round(r["pct_positive"], 2),
                    })

            # Burst regimes
            for _, r in burst_df.iterrows():
                summary_rows.append({
                    "signal": f"Burst: {r['burst_regime']}",
                    "n": int(r["n"]),
                    "mean_fwd_return_cents": round(r["mean_fwd_return"], 4),
                    "se": round(r["se_fwd_return"], 4),
                    "pct_positive": round(r["pct_positive"], 2),
                })

            summary_df = pd.DataFrame(summary_rows)

        fig = self._create_figure(ofi_df, burst_df, streak_df, tte_df)
        return AnalysisOutput(figure=fig, data=summary_df)

    def _create_figure(
        self,
        ofi_df: pd.DataFrame,
        burst_df: pd.DataFrame,
        streak_df: pd.DataFrame,
        tte_df: pd.DataFrame,
    ) -> plt.Figure:
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle(
            "T1A Signal Analysis: Forward Predictive Power of Enrichment Fields\n(Kalshi — resolved markets)",
            fontsize=14, fontweight="bold", y=0.98,
        )

        # ── Panel 1: OFI → forward return ───────────────────────────
        ax = axes[0, 0]
        x = range(len(ofi_df))
        colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(ofi_df)))
        bars = ax.bar(x, ofi_df["mean_fwd_return"], yerr=1.96 * ofi_df["se_fwd_return"],
                       color=colors, edgecolor="white", linewidth=0.5, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([b.split(": ")[1] for b in ofi_df["ofi_bucket"]], fontsize=8)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Order-Flow Imbalance Ratio (cum_net_flow / cum_volume)")
        ax.set_ylabel("Mean Forward Return (cents)")
        ax.set_title("Signal 1: Order-Flow Imbalance → Next-Trade Return")
        for bar, n in zip(bars, ofi_df["n"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"n={n:,.0f}", ha="center", va="bottom", fontsize=6)

        # ── Panel 2: Burst regime ────────────────────────────────────
        ax = axes[0, 1]
        regime_colors = {
            "burst_bearish": "#e74c3c", "burst_bullish": "#2ecc71",
            "burst_neutral": "#95a5a6", "moderate": "#3498db", "quiet": "#f39c12",
        }
        burst_sorted = burst_df.sort_values("burst_regime")
        x = range(len(burst_sorted))
        bar_colors = [regime_colors.get(r, "#999") for r in burst_sorted["burst_regime"]]
        bars = ax.bar(x, burst_sorted["mean_fwd_return"],
                       yerr=1.96 * burst_sorted["se_fwd_return"],
                       color=bar_colors, edgecolor="white", linewidth=0.5, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(burst_sorted["burst_regime"], fontsize=8, rotation=20)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Trade-Burst Regime")
        ax.set_ylabel("Mean Forward Return (cents)")
        ax.set_title("Signal 2: Trade-Burst Momentum → Next-Trade Return")
        for bar, n in zip(bars, burst_sorted["n"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"n={n:,.0f}", ha="center", va="bottom", fontsize=6)

        # ── Panel 3: Price streak → continuation rate ────────────────
        ax = axes[1, 0]
        for direction, marker, color, label in [
            (1.0, "^", "#2ecc71", "Up streaks"),
            (-1.0, "v", "#e74c3c", "Down streaks"),
        ]:
            mask = streak_df["streak_direction"] == direction
            sub = streak_df[mask].sort_values("streak_len")
            ax.errorbar(
                sub["streak_len"], sub["pct_continues"],
                yerr=1.96 * 100 * sub["se_fwd_return"].abs() / sub["mean_abs_fwd_return"].clip(lower=0.01),
                marker=marker, color=color, label=label, capsize=3, linewidth=2, markersize=8,
            )
            for _, row in sub.iterrows():
                ax.annotate(f"n={int(row['n']):,}", (row["streak_len"], row["pct_continues"]),
                            textcoords="offset points", xytext=(0, 10), fontsize=6, ha="center")
        ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, label="No edge (50%)")
        ax.set_xlabel("Consecutive Same-Direction Price Moves")
        ax.set_ylabel("% Next Move Continues Streak")
        ax.set_title("Signal 3: Price Streak → Continuation vs Reversal")
        ax.legend(fontsize=9)
        ax.set_ylim(30, 70)
        ax.set_xticks([1, 2, 3, 4, 5])

        # ── Panel 4: OFI × Time-to-Expiry ───────────────────────────
        ax = axes[1, 1]
        tte_pivot = tte_df[tte_df["ofi_signal"] != "flat"].copy()
        # compute hit rate per TTE bucket for long/short signals combined
        tte_agg = tte_pivot.groupby("tte_bucket").agg(
            n=("n", "sum"),
            hit_rate=("hit_rate", "first"),
            signal_return=("signal_return", "mean"),
        ).reset_index().sort_values("tte_bucket")

        x = range(len(tte_agg))
        tte_colors = plt.cm.coolwarm(np.linspace(0.15, 0.85, len(tte_agg)))
        bars = ax.bar(x, tte_agg["hit_rate"], color=tte_colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([b.split(": ")[1] for b in tte_agg["tte_bucket"]], fontsize=9)
        ax.axhline(50, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Time to Expiry")
        ax.set_ylabel("OFI Signal Hit Rate (%)")
        ax.set_title("Signal 4: OFI Predictive Power by Time-to-Expiry")
        for bar, n in zip(bars, tte_agg["n"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"n={n:,.0f}", ha="center", va="bottom", fontsize=7)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return fig


if __name__ == "__main__":
    analysis = T1ASignalAnalysis()
    output = analysis.run()

    output_dir = BASE_DIR / "data" / "analysis" / "t1a_signal_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    if output.figure is not None:
        path = output_dir / "t1a_signal_analysis.png"
        output.figure.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Figure saved: {path}")
        plt.close(output.figure)

    if output.data is not None:
        csv_path = output_dir / "t1a_signal_analysis.csv"
        output.data.to_csv(csv_path, index=False)
        print(f"CSV saved: {csv_path}")
        print("\n=== SIGNAL SUMMARY ===")
        print(output.data.to_string(index=False))
