#!/usr/bin/env python3
"""
Strategy Report 4.3: Conditional/Correlated Arbitrage

Explores logical pricing consistency across related Kalshi markets within
event families.  Key analyses:

1. **Event family structure** -- How many markets exist per event-ticker
   prefix (family)?  What does the price distribution look like?
2. **Monotonicity violations in threshold markets** -- For families like
   KXBTCD, KXINXU, KXETHD where markets ask "will X be above T?", prices
   MUST decrease as the threshold T increases.  Any violation is a logical
   inconsistency and potential arbitrage.
3. **Cross-family consistency** -- For paired families (e.g. KXBTC range
   vs KXBTCD threshold) on the same underlying and date, do the implied
   distributions agree?
4. **Within-event price variance** -- Distribution of price std-dev within
   events, highlighting high-variance (wide disagreement) vs low-variance
   (concentrated) events.

Markets-only query; no trade data needed.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import duckdb  # noqa: E402

# ── Style constants ──────────────────────────────────────────────────────────
BLUE = "#4C72B0"
RED = "#e74c3c"
GREEN = "#2ecc71"
ORANGE = "#ff7f0e"
PURPLE = "#9467bd"
TEAL = "#17becf"
PINK = "#e377c2"
GREY = "#7f7f7f"
PALETTE = [BLUE, RED, GREEN, ORANGE, PURPLE, TEAL, PINK, GREY, "#bcbd22", "#8c564b"]
FIG_DIR = Path("strategy_reports/figures")
PARQUET_GLOB = "data/kalshi/markets/*.parquet"
DPI = 150
GRID_ALPHA = 0.3

# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_family(event_ticker: str) -> str:
    """Pull the base prefix from an event_ticker, stripping date/ID suffixes."""
    return event_ticker.split("-")[0] if event_ticker else ""


def extract_threshold(ticker: str) -> float | None:
    """
    For threshold-style tickers like KXBTCD-25JAN0317-T97749.99, extract
    the numeric threshold after '-T'.
    """
    m = re.search(r"-T([\d.]+)$", ticker)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def extract_bracket_midpoint(ticker: str) -> float | None:
    """
    For bracket-style tickers like KXBTC-24DEC0117-B98000, extract the
    numeric bracket value after '-B'.
    """
    m = re.search(r"-B([\d.]+)$", ticker)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data():
    """Load all active/liquid Kalshi markets."""
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT
            ticker,
            event_ticker,
            title,
            yes_sub_title,
            status,
            last_price,
            volume,
            yes_bid,
            yes_ask,
            close_time
        FROM '{PARQUET_GLOB}'
        WHERE last_price > 0
          AND volume > 0
          AND event_ticker IS NOT NULL
          AND event_ticker != ''
    """).fetchdf()
    con.close()
    return df


# ── Analysis 1: Event family structure ───────────────────────────────────────

def analyse_family_structure(df):
    """Compute per-family statistics and return top families."""
    df = df.copy()
    df["family"] = df["event_ticker"].apply(extract_family)

    family_stats = (
        df.groupby("family")
        .agg(
            market_count=("ticker", "count"),
            event_count=("event_ticker", "nunique"),
            mean_price=("last_price", "mean"),
            std_price=("last_price", "std"),
            median_price=("last_price", "median"),
            total_volume=("volume", "sum"),
        )
        .reset_index()
        .sort_values("market_count", ascending=False)
    )

    return df, family_stats


# ── Analysis 2: Monotonicity violations in threshold families ────────────────

def analyse_monotonicity(df):
    """
    For threshold families (ticker contains '-T' followed by number),
    check that prices are monotonically non-increasing as the threshold
    increases within each event_ticker.  Each pair where a HIGHER threshold
    has a HIGHER price is a violation.
    """
    # Identify threshold-style markets
    df = df.copy()
    df["threshold"] = df["ticker"].apply(extract_threshold)
    thresh_df = df.dropna(subset=["threshold"]).copy()

    if thresh_df.empty:
        return None, {}

    violations = []
    event_summaries = []

    for evt, grp in thresh_df.groupby("event_ticker"):
        grp_sorted = grp.sort_values("threshold").reset_index(drop=True)
        thresholds = grp_sorted["threshold"].values
        prices = grp_sorted["last_price"].values

        n_violations = 0
        max_violation_cents = 0.0

        for i in range(len(thresholds) - 1):
            if prices[i + 1] > prices[i]:
                # Violation: higher threshold has higher price
                v_size = prices[i + 1] - prices[i]
                n_violations += 1
                max_violation_cents = max(max_violation_cents, v_size)
                violations.append({
                    "event_ticker": evt,
                    "threshold_low": thresholds[i],
                    "price_low_thresh": int(prices[i]),
                    "threshold_high": thresholds[i + 1],
                    "price_high_thresh": int(prices[i + 1]),
                    "violation_cents": int(v_size),
                    "volume_low": int(grp_sorted.iloc[i]["volume"]),
                    "volume_high": int(grp_sorted.iloc[i + 1]["volume"]),
                })

        event_summaries.append({
            "event_ticker": evt,
            "family": extract_family(evt),
            "n_markets": len(grp_sorted),
            "n_violations": n_violations,
            "max_violation_cents": max_violation_cents,
        })

    import pandas as pd
    violations_df = pd.DataFrame(violations) if violations else pd.DataFrame()
    summaries_df = pd.DataFrame(event_summaries)

    # Aggregate by family
    family_violation_stats = (
        summaries_df.groupby("family")
        .agg(
            total_events=("event_ticker", "count"),
            events_with_violations=("n_violations", lambda x: int((x > 0).sum())),
            total_violations=("n_violations", "sum"),
            max_violation_cents=("max_violation_cents", "max"),
        )
        .reset_index()
        .sort_values("total_violations", ascending=False)
    )
    family_violation_stats["pct_events_violated"] = (
        family_violation_stats["events_with_violations"]
        / family_violation_stats["total_events"]
        * 100
    ).round(1)

    return violations_df, {
        "summaries": summaries_df,
        "family_stats": family_violation_stats,
        "total_threshold_events": len(summaries_df),
        "total_events_with_violations": int((summaries_df["n_violations"] > 0).sum()),
        "total_violations": int(summaries_df["n_violations"].sum()),
    }


# ── Analysis 3: Within-event price variance ──────────────────────────────────

def analyse_within_event_variance(df):
    """Compute price variance within each event and each family."""
    import pandas as pd

    event_stats = (
        df.groupby("event_ticker")
        .agg(
            n_markets=("ticker", "count"),
            price_sum=("last_price", "sum"),
            price_mean=("last_price", "mean"),
            price_std=("last_price", "std"),
            price_range=("last_price", lambda x: x.max() - x.min()),
            total_volume=("volume", "sum"),
        )
        .reset_index()
    )
    event_stats["family"] = event_stats["event_ticker"].apply(extract_family)
    event_stats = event_stats.dropna(subset=["price_std"])  # single-market events

    return event_stats


# ── Analysis 4: Cross-family consistency (range vs threshold) ────────────────

def analyse_cross_family(df):
    """
    For paired families like KXBTC (range) vs KXBTCD (threshold), compare
    implied distributions for the same date.
    """
    import pandas as pd

    # Focus on KXBTC vs KXBTCD as a case study
    btc_mask = df["event_ticker"].str.startswith("KXBTC-") | df["event_ticker"].str.startswith("KXBTCD-")
    btc_df = df[btc_mask].copy()

    if btc_df.empty:
        return None

    # Extract date portion
    btc_df["date_part"] = btc_df["event_ticker"].apply(
        lambda x: "-".join(x.split("-")[1:]) if "-" in x else ""
    )
    btc_df["family"] = btc_df["event_ticker"].apply(extract_family)

    # For each date, compare range sum vs threshold structure
    results = []
    for date_part, grp in btc_df.groupby("date_part"):
        range_grp = grp[grp["family"] == "KXBTC"]
        thresh_grp = grp[grp["family"] == "KXBTCD"]
        if len(range_grp) >= 3 and len(thresh_grp) >= 3:
            results.append({
                "date_part": date_part,
                "range_n": len(range_grp),
                "range_sum": int(range_grp["last_price"].sum()),
                "thresh_n": len(thresh_grp),
                "thresh_mean": round(float(thresh_grp["last_price"].mean()), 1),
                "range_max_price": int(range_grp["last_price"].max()),
                "thresh_max_price": int(thresh_grp["last_price"].max()),
            })

    return pd.DataFrame(results) if results else None


# ── Figures ──────────────────────────────────────────────────────────────────

def plot_family_price_distribution(df, family_stats):
    """
    Figure 1: Box plot of last_price distributions for top event families,
    plus a histogram of within-event price standard deviations.
    """
    # Filter to non-sports families for cleaner analysis
    exclude_kw = ["MULTIGAME", "SINGLEGAME", "ESPORTS"]
    mask = ~family_stats["family"].str.contains("|".join(exclude_kw), case=False)
    top_families = family_stats[mask].head(12)["family"].tolist()

    df_plot = df[df["family"].isin(top_families)].copy()

    # Prepare data for box plot: group prices by family
    family_prices = []
    family_labels = []
    family_counts = []
    for fam in top_families:
        prices = df_plot.loc[df_plot["family"] == fam, "last_price"].values
        family_prices.append(prices)
        n_events = df_plot.loc[df_plot["family"] == fam, "event_ticker"].nunique()
        family_labels.append(f"{fam}\n({len(prices):,} mkts, {n_events:,} evts)")
        family_counts.append(len(prices))

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bp = ax.boxplot(
        family_prices,
        labels=family_labels,
        patch_artist=True,
        showfliers=False,
        widths=0.6,
        medianprops=dict(color="black", linewidth=2),
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(PALETTE[i % len(PALETTE)])
        patch.set_alpha(0.7)

    ax.set_ylabel("Last Price (cents)", fontsize=12)
    ax.set_title(
        "Price Distribution Across Top Event Families (excl. sports parlays)",
        fontsize=14,
        fontweight="bold",
    )
    ax.tick_params(axis="x", labelsize=8, rotation=30)
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.set_ylim(-2, 102)
    ax.axhline(50, color=GREY, linestyle="--", alpha=0.4, label="50c midpoint")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "4_3_event_family_price_distribution.png", dpi=DPI)
    plt.close(fig)


def plot_correlation_heatmap(df, mono_results, event_variance):
    """
    Figure 2: Two-panel figure.
    Left: Histogram of monotonicity violations by family.
    Right: Histogram of within-event price standard deviations.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("white")

    # ── Left panel: Monotonicity violations by family ───────────────────
    if mono_results and "family_stats" in mono_results:
        fstats = mono_results["family_stats"]
        fstats = fstats[fstats["total_violations"] > 0].head(15).copy()
        fstats = fstats.sort_values("total_violations", ascending=True)

        colors = [RED if pct > 20 else ORANGE if pct > 5 else GREEN
                  for pct in fstats["pct_events_violated"]]

        bars = ax1.barh(
            range(len(fstats)),
            fstats["total_violations"].values,
            color=colors,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
        )

        ax1.set_yticks(range(len(fstats)))
        ax1.set_yticklabels(
            [f'{row["family"]} ({row["pct_events_violated"]:.0f}%)'
             for _, row in fstats.iterrows()],
            fontsize=9,
        )
        ax1.set_xlabel("Total Monotonicity Violations", fontsize=11)
        ax1.set_title("Monotonicity Violations by Family\n(% = events with violations)", fontsize=12, fontweight="bold")
        ax1.grid(axis="x", alpha=GRID_ALPHA)
        ax1.set_facecolor("white")

        # Add count labels
        for i, (_, row) in enumerate(fstats.iterrows()):
            ax1.text(
                row["total_violations"] + max(fstats["total_violations"].max() * 0.01, 1),
                i, f'{int(row["total_violations"]):,}',
                va="center", fontsize=8,
            )
    else:
        ax1.text(0.5, 0.5, "No threshold markets found", ha="center", va="center",
                 transform=ax1.transAxes, fontsize=14)
        ax1.set_title("Monotonicity Violations", fontsize=12, fontweight="bold")

    # ── Right panel: Within-event price std dev distribution ─────────────
    ev = event_variance.copy()
    # Exclude single-market events and sports parlays
    ev = ev[ev["n_markets"] >= 2]
    exclude_kw = ["MULTIGAME", "SINGLEGAME", "ESPORTS"]
    mask = ~ev["family"].str.contains("|".join(exclude_kw), case=False)
    ev = ev[mask]

    stds = ev["price_std"].dropna().values
    stds = stds[np.isfinite(stds)]

    ax2.set_facecolor("white")
    ax2.hist(stds, bins=80, color=BLUE, alpha=0.75, edgecolor="white", linewidth=0.3)
    ax2.axvline(np.median(stds), color=RED, linestyle="--", linewidth=2,
                label=f"Median = {np.median(stds):.1f}c")
    ax2.axvline(np.mean(stds), color=ORANGE, linestyle="--", linewidth=2,
                label=f"Mean = {np.mean(stds):.1f}c")
    ax2.set_xlabel("Within-Event Price Std Dev (cents)", fontsize=11)
    ax2.set_ylabel("Number of Events", fontsize=11)
    ax2.set_title("Within-Event Price Dispersion\n(multi-market events, excl. parlays)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=GRID_ALPHA)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "4_3_correlation_heatmap.png", dpi=DPI)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...", file=sys.stderr)
    df = load_data()
    print(f"  Loaded {len(df):,} active markets across "
          f"{df['event_ticker'].nunique():,} events", file=sys.stderr)

    # -- Family structure --
    print("Analysing event family structure...", file=sys.stderr)
    df_with_family, family_stats = analyse_family_structure(df)

    # -- Monotonicity --
    print("Checking monotonicity in threshold families...", file=sys.stderr)
    violations_df, mono_results = analyse_monotonicity(df)

    # -- Within-event variance --
    print("Computing within-event variance...", file=sys.stderr)
    event_variance = analyse_within_event_variance(df_with_family)

    # -- Cross-family consistency --
    print("Checking cross-family consistency (BTC)...", file=sys.stderr)
    cross_fam = analyse_cross_family(df)

    # ── Build JSON output ────────────────────────────────────────────────
    output = {}

    # Global stats
    output["total_active_markets"] = int(len(df))
    output["total_events"] = int(df["event_ticker"].nunique())
    output["total_families"] = int(family_stats["family"].nunique())

    # Top families
    exclude_kw = ["MULTIGAME", "SINGLEGAME", "ESPORTS"]
    fstats_clean = family_stats[
        ~family_stats["family"].str.contains("|".join(exclude_kw), case=False)
    ]
    top10 = fstats_clean.head(10)
    output["top_families"] = [
        {
            "family": row["family"],
            "market_count": int(row["market_count"]),
            "event_count": int(row["event_count"]),
            "mean_price": round(float(row["mean_price"]), 1),
            "std_price": round(float(row["std_price"]), 1) if not np.isnan(row["std_price"]) else 0,
        }
        for _, row in top10.iterrows()
    ]

    # Monotonicity results
    if mono_results:
        output["monotonicity"] = {
            "total_threshold_events": mono_results["total_threshold_events"],
            "events_with_violations": mono_results["total_events_with_violations"],
            "pct_events_violated": round(
                mono_results["total_events_with_violations"]
                / max(mono_results["total_threshold_events"], 1)
                * 100,
                2,
            ),
            "total_violations": mono_results["total_violations"],
        }

        # Top violated families
        fv = mono_results["family_stats"]
        fv_top = fv[fv["total_violations"] > 0].head(10)
        output["monotonicity"]["top_violated_families"] = [
            {
                "family": row["family"],
                "total_events": int(row["total_events"]),
                "events_with_violations": int(row["events_with_violations"]),
                "pct_violated": float(row["pct_events_violated"]),
                "total_violations": int(row["total_violations"]),
                "max_violation_cents": int(row["max_violation_cents"]),
            }
            for _, row in fv_top.iterrows()
        ]

        # Worst individual violations
        if violations_df is not None and len(violations_df) > 0:
            worst = violations_df.nlargest(10, "violation_cents")
            output["monotonicity"]["worst_violations"] = [
                {
                    "event_ticker": row["event_ticker"],
                    "threshold_low": float(row["threshold_low"]),
                    "price_at_lower_threshold": int(row["price_low_thresh"]),
                    "threshold_high": float(row["threshold_high"]),
                    "price_at_higher_threshold": int(row["price_high_thresh"]),
                    "violation_cents": int(row["violation_cents"]),
                }
                for _, row in worst.iterrows()
            ]

    # Within-event variance
    ev_clean = event_variance[
        ~event_variance["family"].str.contains("|".join(exclude_kw), case=False)
    ]
    ev_multi = ev_clean[ev_clean["n_markets"] >= 2]
    stds = ev_multi["price_std"].dropna()
    output["within_event_variance"] = {
        "events_analysed": int(len(ev_multi)),
        "median_std": round(float(stds.median()), 2),
        "mean_std": round(float(stds.mean()), 2),
        "q25_std": round(float(stds.quantile(0.25)), 2),
        "q75_std": round(float(stds.quantile(0.75)), 2),
        "pct_high_variance_gt30": round(float((stds > 30).mean() * 100), 1),
    }

    # Cross-family BTC analysis
    if cross_fam is not None and len(cross_fam) > 0:
        output["cross_family_btc"] = {
            "date_pairs_analysed": int(len(cross_fam)),
            "mean_range_sum": round(float(cross_fam["range_sum"].mean()), 1),
            "std_range_sum": round(float(cross_fam["range_sum"].std()), 1),
            "mean_thresh_mean_price": round(float(cross_fam["thresh_mean"].mean()), 1),
            "note": (
                "KXBTC range-bucket markets should sum to ~100 (mutually exclusive). "
                "KXBTCD threshold markets are cumulative (not mutually exclusive). "
                "Consistency check: the KXBTC range that wins should correspond to "
                "the threshold boundary in KXBTCD."
            ),
        }

    # ── Figures ──────────────────────────────────────────────────────────
    print("Plotting figures...", file=sys.stderr)
    plot_family_price_distribution(df_with_family, family_stats)
    plot_correlation_heatmap(df_with_family, mono_results, event_variance)

    # ── Output ───────────────────────────────────────────────────────────
    print(json.dumps(output, indent=2))
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
