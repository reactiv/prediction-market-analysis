"""Generate HTML report for T9 Correlated Market Discovery Pipeline (v2)."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "font.size": 11,
})

ACCENT = "#58a6ff"
ACCENT2 = "#f78166"
ACCENT3 = "#7ee787"
ACCENT4 = "#d2a8ff"
ACCENT5 = "#ff7b72"
COLORS = [ACCENT, ACCENT2, ACCENT3, ACCENT4, ACCENT5, "#ffa657", "#79c0ff"]

BASE = Path(__file__).parent.parent / "data" / "transforms"


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def load_manifest(transform: str) -> dict:
    with open(BASE / transform / "manifest.json") as f:
        return json.load(f)


def plot_similarity_histogram() -> str:
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT
            FLOOR(cosine_sim * 20) / 20 AS bucket,
            COUNT(*) AS cnt
        FROM read_parquet('{BASE}/t9a/similarity_pairs.parquet')
        GROUP BY 1 ORDER BY 1
    """).fetchdf()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df["bucket"], df["cnt"] / 1e6, width=0.045, color=ACCENT, alpha=0.85, edgecolor="#0d1117")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Pairs (millions)")
    ax.set_title("Distribution of 53.6M Semantic Similarity Pairs", fontsize=14, fontweight="bold")
    ax.set_xlim(0.68, 1.02)
    ax.axvline(0.70, color=ACCENT3, ls="--", lw=1.5, label="Cross-platform threshold (0.70)")
    ax.axvline(0.85, color=ACCENT2, ls="--", lw=1.5, label="Intra-platform threshold (0.85)")
    ax.legend(facecolor="#161b22", edgecolor="#30363d")
    ax.grid(axis="y", alpha=0.3)
    return fig_to_b64(fig)


def plot_relationship_types() -> str:
    con = duckdb.connect()
    # Overall
    df = con.execute(f"""
        SELECT relationship_type, COUNT(*) AS cnt
        FROM read_parquet('{BASE}/t9b/classified_pairs.parquet')
        GROUP BY 1 ORDER BY cnt DESC
    """).fetchdf()

    # Cross-platform
    df_cross = con.execute(f"""
        SELECT relationship_type, COUNT(*) AS cnt
        FROM read_parquet('{BASE}/t9b/classified_pairs.parquet')
        WHERE platform_a != platform_b
        GROUP BY 1 ORDER BY cnt DESC
    """).fetchdf()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # All pairs bar chart
    colors_bar = {
        "identical": ACCENT, "hierarchical": ACCENT3, "unrelated": "#8b949e",
        "common_factor": ACCENT4, "inverse": ACCENT5, "causal": ACCENT2,
    }
    c = [colors_bar.get(r, ACCENT) for r in df["relationship_type"]]
    bars = ax1.barh(df["relationship_type"][::-1], df["cnt"][::-1],
                    color=c[::-1], edgecolor="#0d1117")
    ax1.set_xscale("log")
    ax1.set_xlabel("Count (log scale)")
    ax1.set_title("All 50K Classified Pairs", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, df["cnt"][::-1]):
        ax1.text(bar.get_width() * 1.3, bar.get_y() + bar.get_height() / 2,
                 f"{val:,}", va="center", fontsize=10, color="#c9d1d9")
    ax1.grid(axis="x", alpha=0.3)

    # Cross-platform bar chart
    c2 = [colors_bar.get(r, ACCENT) for r in df_cross["relationship_type"]]
    bars2 = ax2.barh(df_cross["relationship_type"][::-1], df_cross["cnt"][::-1],
                     color=c2[::-1], edgecolor="#0d1117")
    ax2.set_xscale("log")
    ax2.set_xlabel("Count (log scale)")
    ax2.set_title("20K Cross-Platform Pairs", fontsize=13, fontweight="bold")
    for bar, val in zip(bars2, df_cross["cnt"][::-1]):
        ax2.text(bar.get_width() * 1.3, bar.get_y() + bar.get_height() / 2,
                 f"{val:,}", va="center", fontsize=10, color="#c9d1d9")
    ax2.grid(axis="x", alpha=0.3)

    fig.tight_layout(w_pad=4)
    return fig_to_b64(fig)


def plot_platform_breakdown() -> str:
    con = duckdb.connect()
    markets = con.execute(f"""
        SELECT platform, COUNT(*) AS cnt
        FROM read_parquet('{BASE}/t9a/markets_unified.parquet')
        GROUP BY 1
    """).fetchdf()

    pairs = con.execute(f"""
        SELECT
            CASE WHEN platform_a = platform_b THEN platform_a ELSE 'cross-platform' END AS ptype,
            COUNT(*) AS cnt
        FROM read_parquet('{BASE}/t9a/similarity_pairs.parquet')
        GROUP BY 1 ORDER BY cnt DESC
    """).fetchdf()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors_m = [ACCENT, ACCENT2]
    ax1.bar(markets["platform"], markets["cnt"] / 1e6, color=colors_m, edgecolor="#0d1117", width=0.5)
    for i, (_, row) in enumerate(markets.iterrows()):
        ax1.text(i, row["cnt"] / 1e6 + 0.02, f'{row["cnt"]:,}', ha="center", fontsize=11, color="#c9d1d9")
    ax1.set_ylabel("Markets (millions)")
    ax1.set_title("1.6M Unified Markets", fontsize=13, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    colors_p = [ACCENT, ACCENT2, ACCENT3]
    ax2.bar(pairs["ptype"], pairs["cnt"] / 1e6, color=colors_p[:len(pairs)], edgecolor="#0d1117", width=0.5)
    for i, (_, row) in enumerate(pairs.iterrows()):
        ax2.text(i, row["cnt"] / 1e6 + 0.2, f'{row["cnt"]:,}', ha="center", fontsize=10, color="#c9d1d9")
    ax2.set_ylabel("Pairs (millions)")
    ax2.set_title("Similarity Pairs by Platform", fontsize=13, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout(w_pad=3)
    return fig_to_b64(fig)


def plot_cluster_sizes() -> str:
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT cluster_id, COUNT(*) AS size
        FROM read_parquet('{BASE}/t9a/clusters.parquet')
        WHERE cluster_id >= 0
        GROUP BY 1 ORDER BY size DESC
        LIMIT 30
    """).fetchdf()

    n_clusters = con.execute(f"""
        SELECT COUNT(DISTINCT cluster_id) FROM read_parquet('{BASE}/t9a/clusters.parquet')
        WHERE cluster_id >= 0
    """).fetchone()[0]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(df))
    ax.bar(x, df["size"], color=ACCENT, alpha=0.85, edgecolor="#0d1117")
    ax.set_xlabel("Cluster Rank")
    ax.set_ylabel("Markets in Cluster")
    ax.set_title(f"Top 30 Clusters by Size ({n_clusters:,} total)", fontsize=14, fontweight="bold")
    ax.set_xticks(list(x)[::5])
    ax.grid(axis="y", alpha=0.3)
    return fig_to_b64(fig)


def plot_validation_stats() -> str:
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT * FROM read_parquet('{BASE}/t9c/validated_pairs.parquet')
    """).fetchdf()

    if df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No validated pairs", ha="center", va="center", fontsize=18, color="#8b949e")
        ax.axis("off")
        return fig_to_b64(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Level correlation (the meaningful metric for prediction markets)
    ax = axes[0, 0]
    has_level = "level_pearson_r" in df.columns
    if has_level:
        vals = df["level_pearson_r"].dropna()
        ax.hist(vals, bins=20, color=ACCENT, alpha=0.85, edgecolor="#0d1117")
        ax.axvline(0, color=ACCENT5, ls="--", lw=1)
        ax.set_xlabel("Pearson r (price levels)")
        ax.set_ylabel("Count")
        ax.set_title("Price Level Correlation", fontsize=13, fontweight="bold")
    else:
        vals = df["pearson_r"].dropna()
        ax.hist(vals, bins=20, color=ACCENT, alpha=0.85, edgecolor="#0d1117")
        ax.axvline(0, color=ACCENT5, ls="--", lw=1)
        ax.set_xlabel("Pearson r (returns)")
        ax.set_ylabel("Count")
        ax.set_title("Return Correlation", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Peak lag
    ax = axes[0, 1]
    vals = df["peak_lag_minutes"].dropna()
    ax.hist(vals, bins=25, color=ACCENT2, alpha=0.85, edgecolor="#0d1117")
    ax.axvline(0, color=ACCENT5, ls="--", lw=1)
    ax.set_xlabel("Peak Lag (minutes)")
    ax.set_ylabel("Count")
    ax.set_title("Cross-Correlation Peak Lag", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Tail dependence scatter
    ax = axes[1, 0]
    ax.scatter(df["tail_dep_upper"], df["tail_dep_lower"], c=ACCENT3, alpha=0.7, s=60, edgecolors="#0d1117")
    ax.plot([0, 1], [0, 1], color="#30363d", ls="--", lw=1)
    ax.set_xlabel("Upper Tail Dependence")
    ax.set_ylabel("Lower Tail Dependence")
    ax.set_title("Tail Dependence (extremes co-movement)", fontsize=13, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    # Granger significance
    ax = axes[1, 1]
    gp_ab = df["granger_a_to_b_p"].dropna()
    gp_ba = df["granger_b_to_a_p"].dropna()
    ax.hist(gp_ab, bins=20, color=ACCENT, alpha=0.6, label="A \u2192 B", edgecolor="#0d1117")
    ax.hist(gp_ba, bins=20, color=ACCENT4, alpha=0.6, label="B \u2192 A", edgecolor="#0d1117")
    ax.axvline(0.05, color=ACCENT5, ls="--", lw=1.5, label="p = 0.05")
    ax.set_xlabel("Granger p-value")
    ax.set_ylabel("Count")
    ax.set_title("Granger Causality Significance", fontsize=13, fontweight="bold")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout(h_pad=3, w_pad=3)
    return fig_to_b64(fig)


def plot_cross_platform_classification() -> str:
    """Stacked bar chart showing cross-platform vs intra-platform by relationship type."""
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT
            relationship_type,
            SUM(CASE WHEN platform_a != platform_b THEN 1 ELSE 0 END) AS cross_platform,
            SUM(CASE WHEN platform_a = platform_b THEN 1 ELSE 0 END) AS intra_platform
        FROM read_parquet('{BASE}/t9b/classified_pairs.parquet')
        WHERE relationship_type != 'unrelated'
        GROUP BY 1
        ORDER BY cross_platform + intra_platform DESC
    """).fetchdf()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(df))
    ax.bar(x, df["cross_platform"], color=ACCENT3, label="Cross-platform", edgecolor="#0d1117")
    ax.bar(x, df["intra_platform"], bottom=df["cross_platform"], color=ACCENT, label="Intra-platform",
           edgecolor="#0d1117", alpha=0.7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["relationship_type"], rotation=20, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Relationship Types: Cross-Platform vs Intra-Platform", fontsize=14, fontweight="bold")
    ax.legend(facecolor="#161b22", edgecolor="#30363d")
    ax.grid(axis="y", alpha=0.3)

    for i, (_, row) in enumerate(df.iterrows()):
        total = row["cross_platform"] + row["intra_platform"]
        ax.text(i, total + 100, f"{total:,}", ha="center", fontsize=9, color="#c9d1d9")

    return fig_to_b64(fig)


def plot_network() -> str:
    """Draw the correlation network as a simple force-directed layout."""
    con = duckdb.connect()
    edges = con.execute(f"""
        SELECT source, target, weight, pearson_r, relationship_type, peak_lag_minutes
        FROM read_parquet('{BASE}/t9c/correlation_network.parquet')
    """).fetchdf()

    if edges.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No validated pairs for network", ha="center", va="center", fontsize=18, color="#8b949e")
        ax.axis("off")
        return fig_to_b64(fig)

    nodes = list(set(edges["source"].tolist() + edges["target"].tolist()))
    node_idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    rng = np.random.default_rng(42)
    pos = rng.uniform(-1, 1, (n, 2))

    degree = np.zeros(n)
    for _, row in edges.iterrows():
        degree[node_idx[row["source"]]] += 1
        degree[node_idx[row["target"]]] += 1

    for _ in range(200):
        for i in range(n):
            for j in range(i + 1, n):
                diff = pos[i] - pos[j]
                dist = max(np.linalg.norm(diff), 0.01)
                force = diff / (dist ** 2) * 0.01
                pos[i] += force
                pos[j] -= force
        for _, row in edges.iterrows():
            i, j = node_idx[row["source"]], node_idx[row["target"]]
            diff = pos[j] - pos[i]
            dist = max(np.linalg.norm(diff), 0.01)
            force = diff * dist * 0.05
            pos[i] += force
            pos[j] -= force

    fig, ax = plt.subplots(figsize=(12, 10))
    for _, row in edges.iterrows():
        i, j = node_idx[row["source"]], node_idx[row["target"]]
        w = max(row["weight"] * 10, 0.5)
        alpha = min(0.3 + row["weight"] * 5, 0.9)
        ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], color=ACCENT, alpha=alpha, lw=w)

    sizes = (degree + 1) * 80
    ax.scatter(pos[:, 0], pos[:, 1], s=sizes, c=ACCENT3, alpha=0.8, edgecolors="#0d1117", zorder=5, lw=0.5)

    for i, node in enumerate(nodes):
        if degree[i] > 1:
            label = node[:20]
            ax.annotate(label, (pos[i, 0], pos[i, 1]), fontsize=7, color="#c9d1d9",
                        ha="center", va="bottom", xytext=(0, 8), textcoords="offset points")

    n_components = len(set(range(n)) - {node_idx[r["source"]] for _, r in edges.iterrows()} - {node_idx[r["target"]] for _, r in edges.iterrows()})
    ax.set_title(f"Correlation Network ({len(edges)} edges, {n} nodes)", fontsize=14, fontweight="bold")
    ax.axis("off")
    return fig_to_b64(fig)


def get_cross_platform_classified() -> str:
    """Top cross-platform classified pairs (non-unrelated)."""
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT title_a, platform_a, title_b, platform_b, cosine_sim,
               relationship_type, confidence
        FROM read_parquet('{BASE}/t9b/classified_pairs.parquet')
        WHERE platform_a != platform_b
          AND relationship_type != 'unrelated'
        ORDER BY cosine_sim DESC
        LIMIT 20
    """).fetchdf()

    rows_html = []
    for _, row in df.iterrows():
        badge_a = f'<span class="badge badge-{row["platform_a"]}">{row["platform_a"]}</span>'
        badge_b = f'<span class="badge badge-{row["platform_b"]}">{row["platform_b"]}</span>'
        rel_class = "good" if row["relationship_type"] == "identical" else ""
        rows_html.append(f"""
        <tr>
            <td style="max-width:300px">{row['title_a'][:100]}</td>
            <td>{badge_a}</td>
            <td style="max-width:300px">{row['title_b'][:100]}</td>
            <td>{badge_b}</td>
            <td class="num">{row['cosine_sim']:.4f}</td>
            <td class="{rel_class}">{row['relationship_type']}</td>
        </tr>""")
    return "\n".join(rows_html)


def get_interesting_pairs() -> str:
    """Non-identical, non-unrelated cross-platform pairs."""
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT title_a, platform_a, title_b, platform_b, cosine_sim,
               relationship_type, confidence, reasoning
        FROM read_parquet('{BASE}/t9b/classified_pairs.parquet')
        WHERE platform_a != platform_b
          AND relationship_type NOT IN ('unrelated', 'identical')
        ORDER BY confidence DESC, cosine_sim DESC
        LIMIT 20
    """).fetchdf()

    rows_html = []
    for _, row in df.iterrows():
        badge_a = f'<span class="badge badge-{row["platform_a"]}">{row["platform_a"]}</span>'
        badge_b = f'<span class="badge badge-{row["platform_b"]}">{row["platform_b"]}</span>'
        rows_html.append(f"""
        <tr>
            <td style="max-width:280px">{row['title_a'][:90]}</td>
            <td>{badge_a}</td>
            <td style="max-width:280px">{row['title_b'][:90]}</td>
            <td>{badge_b}</td>
            <td>{row['relationship_type']}</td>
            <td style="max-width:250px; font-size:11px; color:#8b949e">{row['reasoning'][:120]}</td>
        </tr>""")
    return "\n".join(rows_html)


def get_validated_pairs() -> str:
    """Validated pairs with level correlation."""
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT * FROM read_parquet('{BASE}/t9c/validated_pairs.parquet')
        ORDER BY level_pearson_r DESC
    """).fetchdf()

    if df.empty:
        return "<tr><td colspan='8' style='text-align:center; color:#8b949e'>No pairs with sufficient bar data</td></tr>"

    rows_html = []
    for _, row in df.iterrows():
        level_r = row.get("level_pearson_r", row["pearson_r"])
        rows_html.append(f"""
        <tr>
            <td style="max-width:280px">{row['title_a'][:80]}</td>
            <td style="max-width:280px">{row['title_b'][:80]}</td>
            <td>{row['relationship_type']}</td>
            <td class="num">{level_r:.3f}</td>
            <td class="num">{row['pearson_r']:.3f}</td>
            <td class="num">{row['peak_lag_minutes']:.0f}</td>
            <td class="num">{row['n_aligned_bars']:,}</td>
        </tr>""")
    return "\n".join(rows_html)


def build_report() -> str:
    print("Generating plots...")

    t9b_manifest = load_manifest("t9b")
    t9c_manifest = load_manifest("t9c")

    n_classified = t9b_manifest["total_pairs"]
    n_non_unrelated = t9b_manifest["classified_pairs"]
    type_counts = t9b_manifest["type_counts"]
    n_validated = t9c_manifest["validated_pairs"]

    sim_hist = plot_similarity_histogram()
    platform = plot_platform_breakdown()
    rel_types = plot_relationship_types()
    cross_plat_class = plot_cross_platform_classification()
    clusters = plot_cluster_sizes()
    validation = plot_validation_stats()
    network = plot_network()

    cross_plat_rows = get_cross_platform_classified()
    interesting_rows = get_interesting_pairs()
    validated_rows = get_validated_pairs()

    n_cross_identical = type_counts.get("identical", 0)
    n_hierarchical = type_counts.get("hierarchical", 0)
    n_common = type_counts.get("common_factor", 0)
    n_inverse = type_counts.get("inverse", 0)
    n_causal = type_counts.get("causal", 0)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>T9 Correlated Market Discovery \u2014 Pipeline Report</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #0d1117; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; line-height: 1.6; }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
  h1 {{ font-size: 28px; color: #f0f6fc; margin-bottom: 8px; }}
  h2 {{ font-size: 22px; color: #f0f6fc; margin: 40px 0 16px; border-bottom: 1px solid #21262d; padding-bottom: 8px; }}
  h3 {{ font-size: 17px; color: #c9d1d9; margin: 24px 0 12px; }}
  .subtitle {{ color: #8b949e; font-size: 15px; margin-bottom: 32px; }}
  .hero {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 24px 0; }}
  .stat-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; text-align: center; }}
  .stat-card .number {{ font-size: 32px; font-weight: 700; color: #58a6ff; }}
  .stat-card .number.green {{ color: #7ee787; }}
  .stat-card .number.orange {{ color: #f78166; }}
  .stat-card .label {{ font-size: 13px; color: #8b949e; margin-top: 4px; }}
  .plot {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin: 20px 0; text-align: center; }}
  .plot img {{ max-width: 100%; border-radius: 4px; }}
  .insight {{ background: #1c2128; border-left: 3px solid #58a6ff; padding: 12px 16px; margin: 16px 0; border-radius: 0 6px 6px 0; font-size: 14px; }}
  .insight.warn {{ border-left-color: #f78166; }}
  .insight.good {{ border-left-color: #7ee787; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin: 12px 0; }}
  th {{ background: #161b22; color: #8b949e; text-align: left; padding: 10px 12px; border-bottom: 2px solid #30363d; font-weight: 600; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #21262d; }}
  tr:hover {{ background: #1c2128; }}
  .num {{ font-family: 'SF Mono', 'Fira Code', monospace; text-align: right; }}
  .warn {{ color: #f78166; }}
  .good {{ color: #7ee787; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; }}
  .badge-kalshi {{ background: #1f3a5f; color: #58a6ff; }}
  .badge-polymarket {{ background: #2d1f1f; color: #f78166; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  @media (max-width: 768px) {{ .hero {{ grid-template-columns: repeat(2, 1fr); }} .two-col {{ grid-template-columns: 1fr; }} }}
  .pipeline {{ display: flex; align-items: center; gap: 0; margin: 24px 0; justify-content: center; }}
  .pipeline-step {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px 20px; text-align: center; min-width: 200px; }}
  .pipeline-step .name {{ font-weight: 700; color: #58a6ff; font-size: 15px; }}
  .pipeline-step .desc {{ font-size: 12px; color: #8b949e; }}
  .pipeline-arrow {{ color: #30363d; font-size: 24px; padding: 0 8px; }}
</style>
</head>
<body>
<div class="container">

<h1>Correlated Market Discovery Pipeline</h1>
<p class="subtitle">T9a / T9b / T9c \u2014 Semantic embedding, LLM classification, and statistical validation across 1.6M prediction markets</p>

<div class="pipeline">
  <div class="pipeline-step">
    <div class="name">T9a \u2014 Embed & Cluster</div>
    <div class="desc">OpenAI embeddings + FAISS ANN + HDBSCAN</div>
  </div>
  <div class="pipeline-arrow">\u2192</div>
  <div class="pipeline-step">
    <div class="name">T9b \u2014 Classify</div>
    <div class="desc">Gemini Flash relationship taxonomy</div>
  </div>
  <div class="pipeline-arrow">\u2192</div>
  <div class="pipeline-step">
    <div class="name">T9c \u2014 Validate</div>
    <div class="desc">Level/return correlation, Granger, tail dep.</div>
  </div>
</div>

<div class="hero">
  <div class="stat-card"><div class="number">1.6M</div><div class="label">Markets Embedded</div></div>
  <div class="stat-card"><div class="number">53.6M</div><div class="label">Similarity Pairs Found</div></div>
  <div class="stat-card"><div class="number green">5,832</div><div class="label">Cross-Platform Identical</div></div>
  <div class="stat-card"><div class="number">{n_non_unrelated:,}</div><div class="label">Classified Relationships</div></div>
</div>

<!-- Section: Market Universe -->
<h2>Market Universe</h2>
<div class="plot"><img src="data:image/png;base64,{platform}" /></div>
<div class="insight">Kalshi dominates with 1.2M event-level markets (74.6%) vs Polymarket's 408K (25.4%). Of 104K cross-platform pairs, 20K were prioritized for LLM classification \u2014 yielding 5,832 identical market matches.</div>

<!-- Section: Similarity Distribution -->
<h2>Semantic Similarity Distribution</h2>
<div class="plot"><img src="data:image/png;base64,{sim_hist}" /></div>
<div class="insight"><strong>Dual thresholds</strong> for v2: cross-platform pairs use a lower threshold of 0.70 (green) since different platforms phrase the same events differently. Intra-platform pairs use 0.85 (orange) to filter out trivially similar threshold variants. Pairs with identical titles on the same platform are excluded as trivial recurring market instances.</div>

<!-- Section: Relationship Classification -->
<h2>LLM Relationship Classification</h2>
<div class="plot"><img src="data:image/png;base64,{rel_types}" /></div>

<div class="hero" style="grid-template-columns: repeat(5, 1fr)">
  <div class="stat-card"><div class="number" style="font-size:24px">{n_cross_identical:,}</div><div class="label">Identical</div></div>
  <div class="stat-card"><div class="number green" style="font-size:24px">{n_hierarchical:,}</div><div class="label">Hierarchical</div></div>
  <div class="stat-card"><div class="number" style="font-size:24px; color:#d2a8ff">{n_common:,}</div><div class="label">Common Factor</div></div>
  <div class="stat-card"><div class="number orange" style="font-size:24px">{n_inverse:,}</div><div class="label">Inverse</div></div>
  <div class="stat-card"><div class="number" style="font-size:24px; color:#ffa657">{n_causal:,}</div><div class="label">Causal</div></div>
</div>

<div class="insight good"><strong>v2 improvement:</strong> The updated prompt with cross-platform examples now correctly identifies 5,832 identical cross-platform pairs (up from 0 in v1). The improved prompt also distinguishes 14,595 hierarchical and 1,938 common-factor relationships instead of lumping everything as "identical" (v1 was 99.7% identical).</div>

<!-- Section: Cross-Platform Classification -->
<h2>Cross-Platform Classification Breakdown</h2>
<div class="plot"><img src="data:image/png;base64,{cross_plat_class}" /></div>
<div class="insight">Cross-platform pairs (green) make up a significant fraction of all non-unrelated classifications. The pipeline discovered 5,832 <strong>identical</strong> markets listed on both Kalshi and Polymarket, plus 5,307 <strong>hierarchical</strong> and 1,687 <strong>common-factor</strong> cross-platform relationships.</div>

<!-- Section: Cross-Platform Identical -->
<h2>Cross-Platform Identical Markets</h2>
<p style="color:#8b949e; margin-bottom:12px">Same real-world event listed on both platforms \u2014 discovered purely through semantic similarity + LLM reasoning. Top matches:</p>
<div style="overflow-x:auto">
<table>
  <thead><tr><th>Market A</th><th>Platform</th><th>Market B</th><th>Platform</th><th>Cosine Sim</th><th>Relationship</th></tr></thead>
  <tbody>{cross_plat_rows}</tbody>
</table>
</div>
<div class="insight good">These 5,832 cross-platform identical pairs are the foundation for cross-exchange arbitrage detection \u2014 the same event priced on two different exchanges. The pipeline discovered these automatically without any hardcoded mapping.</div>

<!-- Section: Interesting Cross-Platform Relationships -->
<h2>Non-Identical Cross-Platform Relationships</h2>
<p style="color:#8b949e; margin-bottom:12px">Hierarchical, causal, inverse, and common-factor relationships between markets on different platforms:</p>
<div style="overflow-x:auto">
<table>
  <thead><tr><th>Market A</th><th>Platform</th><th>Market B</th><th>Platform</th><th>Relationship</th><th>Reasoning</th></tr></thead>
  <tbody>{interesting_rows}</tbody>
</table>
</div>
<div class="insight">These non-identical relationships represent more nuanced cross-platform signals: hierarchical pairs where one event implies another, common-factor pairs driven by the same underlying, and inverse pairs that should move in opposite directions.</div>

<!-- Section: Cluster Analysis -->
<h2>Semantic Clusters</h2>
<div class="plot"><img src="data:image/png;base64,{clusters}" /></div>

<!-- Section: Statistical Validation -->
<h2>Statistical Validation</h2>
<p style="color:#8b949e; margin-bottom:12px">{n_validated} pairs with sufficient T1B bar data (\u2265288 bars, \u226550% overlap) validated with level correlation, cross-correlation, Granger causality, and tail dependence.</p>
<div class="plot"><img src="data:image/png;base64,{validation}" /></div>

<div class="two-col">
<div class="insight">
<strong>Level vs Return Correlation:</strong> Price level correlation (mean 0.40) is much more meaningful than return correlation (mean 0.05) for prediction markets. Prices of related markets track each other in levels even when bar-by-bar returns are independent noise.
</div>
<div class="insight warn">
<strong>Data Coverage Bottleneck:</strong> Only {n_validated} of {t9c_manifest.get("pairs_input", 0):,} resolved pairs have sufficient bar data. The 5,832 cross-platform identical pairs from T9b are currently unvalidated because T1B bars don't cover both sides. Expanding T1B coverage would unlock the full arbitrage potential.
</div>
</div>

<!-- Section: Validated Pairs -->
<h3>Validated Pairs (sorted by level correlation)</h3>
<div style="overflow-x:auto">
<table>
  <thead><tr><th>Market A</th><th>Market B</th><th>Type</th><th>Level r</th><th>Return r</th><th>Lag (min)</th><th>Aligned Bars</th></tr></thead>
  <tbody>{validated_rows}</tbody>
</table>
</div>

<!-- Section: Correlation Network -->
<h2>Correlation Network</h2>
<div class="plot"><img src="data:image/png;base64,{network}" /></div>

<!-- Section: Pipeline Stats -->
<h2>Pipeline Performance</h2>
<div class="hero">
  <div class="stat-card"><div class="number" style="font-size:24px">~$6.35</div><div class="label">OpenAI Embedding Cost</div></div>
  <div class="stat-card"><div class="number" style="font-size:24px">&lt;$1</div><div class="label">Gemini Classification Cost</div></div>
  <div class="stat-card"><div class="number" style="font-size:24px">~50 min</div><div class="label">T9a Runtime</div></div>
  <div class="stat-card"><div class="number" style="font-size:24px">~28 min</div><div class="label">T9b + T9c Runtime</div></div>
</div>

</div>
</body>
</html>"""


if __name__ == "__main__":
    output = Path(__file__).parent / "t9_pipeline_report.html"
    html = build_report()
    output.write_text(html)
    print(f"Report written to {output}")
    print(f"Open with: open {output}")
