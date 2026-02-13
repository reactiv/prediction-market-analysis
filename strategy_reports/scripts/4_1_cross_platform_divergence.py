"""
Section 4.1: Kalshi-Polymarket Divergence Analysis
===================================================
Match markets across Kalshi and Polymarket by keyword overlap in titles/questions.
Compare prices where matchable. Look for systematic price divergences.

This is an EXPLORATORY analysis. Cross-platform matching is inherently difficult
because the two platforms use different naming conventions. We use both automated
keyword matching (with strict quality filters) and hand-curated targeted matching
for known overlapping markets.
"""

import glob
import json
import os
import re
import warnings
from collections import Counter, defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KALSHI_DIR = os.path.join(BASE_DIR, "data", "kalshi", "markets")
POLY_DIR = os.path.join(BASE_DIR, "data", "polymarket", "markets")
FIG_DIR = os.path.join(BASE_DIR, "strategy_reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Figure styling constants
# ---------------------------------------------------------------------------
FIG_SIZE = (12, 7)
COLOR_BLUE = "#4C72B0"
COLOR_RED = "#C44E52"
COLOR_GREEN = "#55A868"
COLOR_ORANGE = "#DD8452"
COLOR_PURPLE = "#8172B3"
COLOR_CYAN = "#64B5CD"
COLOR_GRAY = "gray"
DPI = 150

# ---------------------------------------------------------------------------
# Stop words for keyword extraction
# ---------------------------------------------------------------------------
STOP_WORDS = {
    "the", "will", "be", "a", "an", "in", "of", "to", "and", "or", "for",
    "by", "is", "it", "at", "on", "if", "not", "do", "are", "was", "this",
    "that", "with", "from", "has", "have", "had", "its", "any", "before",
    "after", "more", "than", "their", "they", "what", "how", "who", "when",
    "other", "yes", "no", "above", "below", "about", "which", "would",
    "could", "should", "does", "did", "been", "being", "each", "into",
    "all", "also", "but", "can", "may", "might", "must", "new", "per",
    "over", "under", "up", "out", "get", "set", "between", "through",
    "during", "some", "there", "these", "those", "such", "only", "very",
    # Additional generic words that cause false matches
    "win", "next", "party", "rate", "rates", "meeting", "day",
}


def extract_keywords(text: str) -> set:
    """Extract meaningful keywords from a market title/question."""
    if not isinstance(text, str):
        return set()
    text = text.lower()
    # Remove punctuation but keep hyphens and apostrophes within words
    text = re.sub(r"[^\w\s\-']", " ", text)
    words = text.split()
    keywords = set()
    for w in words:
        w = w.strip("-'")
        if len(w) >= 3 and w not in STOP_WORDS and not w.isdigit():
            keywords.add(w)
    return keywords


def parse_poly_prices(outcome_prices_str, outcomes_str):
    """Parse Polymarket outcome_prices JSON string and return YES price (0-100 scale)."""
    try:
        prices = json.loads(outcome_prices_str)
        outcomes = json.loads(outcomes_str)
        for i, outcome in enumerate(outcomes):
            if outcome.lower() == "yes" and i < len(prices):
                return float(prices[i]) * 100
        if len(prices) > 0:
            return float(prices[0]) * 100
    except (json.JSONDecodeError, TypeError, ValueError, IndexError):
        return None
    return None


# ===========================================================================
# 1. Load Data
# ===========================================================================
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

kalshi_files = sorted(glob.glob(os.path.join(KALSHI_DIR, "markets_*.parquet")))
print(f"Found {len(kalshi_files)} Kalshi parquet files")
kalshi_dfs = []
for f in kalshi_files:
    df = pd.read_parquet(f, columns=[
        "ticker", "event_ticker", "title", "last_price", "volume", "status"
    ])
    kalshi_dfs.append(df)
kalshi_all = pd.concat(kalshi_dfs, ignore_index=True)
print(f"Total Kalshi markets loaded: {len(kalshi_all):,}")

poly_files = sorted(glob.glob(os.path.join(POLY_DIR, "markets_*.parquet")))
print(f"Found {len(poly_files)} Polymarket parquet files")
poly_dfs = []
for f in poly_files:
    df = pd.read_parquet(f, columns=[
        "id", "question", "outcomes", "outcome_prices", "volume", "active", "closed"
    ])
    poly_dfs.append(df)
poly_all = pd.concat(poly_dfs, ignore_index=True)
print(f"Total Polymarket markets loaded: {len(poly_all):,}")

# ===========================================================================
# 2. Filter to meaningful markets
# ===========================================================================
print("\n" + "=" * 70)
print("FILTERING MARKETS")
print("=" * 70)

# Kalshi: non-parlay, has price, decent volume
kalshi_filtered = kalshi_all[
    (~kalshi_all["title"].str.startswith("yes ", na=True)) &
    (kalshi_all["last_price"] > 0) &
    (kalshi_all["volume"] > 100)
].copy()
print(f"Kalshi after filtering (non-parlay, price>0, vol>100): {len(kalshi_filtered):,}")

# Polymarket: valid price and question
poly_filtered = poly_all[
    (poly_all["volume"] > 1000) &
    (poly_all["outcome_prices"].notna()) &
    (poly_all["question"].notna())
].copy()
poly_filtered["yes_price"] = poly_filtered.apply(
    lambda r: parse_poly_prices(r["outcome_prices"], r["outcomes"]), axis=1
)
poly_filtered = poly_filtered[
    poly_filtered["yes_price"].notna() & (poly_filtered["yes_price"] > 0)
].copy()
print(f"Polymarket after filtering (vol>1000, valid prices): {len(poly_filtered):,}")

# ===========================================================================
# 3. Extract keywords
# ===========================================================================
print("\n" + "=" * 70)
print("EXTRACTING KEYWORDS")
print("=" * 70)

kalshi_filtered["keywords"] = kalshi_filtered["title"].apply(extract_keywords)
poly_filtered["keywords"] = poly_filtered["question"].apply(extract_keywords)

kalshi_kw_counts = Counter()
for kws in kalshi_filtered["keywords"]:
    kalshi_kw_counts.update(kws)
poly_kw_counts = Counter()
for kws in poly_filtered["keywords"]:
    poly_kw_counts.update(kws)

shared_keywords = set(kalshi_kw_counts.keys()) & set(poly_kw_counts.keys())
print(f"Unique keywords in Kalshi: {len(kalshi_kw_counts):,}")
print(f"Unique keywords in Polymarket: {len(poly_kw_counts):,}")
print(f"Shared keywords: {len(shared_keywords):,}")

# ===========================================================================
# 4. Hand-curated targeted matching
# ===========================================================================
# This is the most reliable approach. We identify specific markets that
# correspond to the same underlying question on both platforms.
# We are very precise to avoid false positives.
print("\n" + "=" * 70)
print("TARGETED MATCHING FOR KNOWN CROSS-PLATFORM EVENTS")
print("=" * 70)

TARGETED_MATCHES = [
    # --- 2024 Presidential Election ---
    {
        "category": "Presidential Election",
        "kalshi_pattern": r"^Will Donald Trump or another Republican win the Presidency\?$",
        "poly_pattern": r"^Will Donald Trump win the 2024 US Presidential Election\?$",
        "note": "Trump wins presidency",
    },
    # --- Fed rate: hold decisions (Kalshi: Hike 0bps = no change) ---
    {
        "category": "Fed Rate Decision",
        "kalshi_pattern": r"Federal Reserve Hike rates by 0bps.*May 2025",
        "poly_pattern": r"No change in Fed interest rates after May 2025",
        "note": "Fed holds May 2025",
    },
    {
        "category": "Fed Rate Decision",
        "kalshi_pattern": r"Federal Reserve Hike rates by 0bps.*July 2025",
        "poly_pattern": r"No change in Fed interest rates after July 2025",
        "note": "Fed holds Jul 2025",
    },
    {
        "category": "Fed Rate Decision",
        "kalshi_pattern": r"Federal Reserve Hike rates by 0bps.*September 2025",
        "poly_pattern": r"No change in Fed interest rates after September 2025",
        "note": "Fed holds Sep 2025",
    },
    {
        "category": "Fed Rate Decision",
        "kalshi_pattern": r"Federal Reserve Hike rates by 0bps.*October 2025",
        "poly_pattern": r"No change in Fed interest rates after October 2025",
        "note": "Fed holds Oct 2025",
    },
    {
        "category": "Fed Rate Decision",
        "kalshi_pattern": r"Federal Reserve Hike rates by 0bps.*December 2025",
        "poly_pattern": r"No change in Fed interest rates after December 2025",
        "note": "Fed holds Dec 2025",
    },
    # --- Fed rate: cut 25bp decisions ---
    {
        "category": "Fed Rate Decision",
        "kalshi_pattern": r"Federal Reserve Cut rates by 25bps.*September 2025",
        "poly_pattern": r"Fed decreases interest rates by 25 bps after September 2025",
        "note": "Fed cuts 25bp Sep 2025",
    },
    {
        "category": "Fed Rate Decision",
        "kalshi_pattern": r"Federal Reserve Cut rates by 25bps.*October 2025",
        "poly_pattern": r"Fed decreases interest rates by 25 bps after October 2025",
        "note": "Fed cuts 25bp Oct 2025",
    },
    {
        "category": "Fed Rate Decision",
        "kalshi_pattern": r"Federal Reserve Cut rates by 25bps.*December 2025",
        "poly_pattern": r"Fed decreases interest rates by 25 bps after December 2025",
        "note": "Fed cuts 25bp Dec 2025",
    },
    # --- Super Bowl ---
    {
        "category": "Super Bowl",
        "kalshi_pattern": r"Philadelphia Eagles win the Super Bowl\?$",
        "poly_pattern": r"Eagles win Super Bowl 2025\?$",
        "note": "Eagles win SB 2025",
    },
    {
        "category": "Super Bowl",
        "kalshi_pattern": r"Kansas City Chiefs win the Super Bowl\?$",
        "poly_pattern": r"Chiefs win Super Bowl 2025\?$",
        "note": "Chiefs win SB 2025",
    },
    # --- TikTok Ban ---
    {
        "category": "TikTok Ban",
        "kalshi_pattern": r"TikTok.*banned.*April 30, 2025",
        "poly_pattern": r"TikTok banned.*US before May 2025",
        "note": "TikTok ban before May 2025",
    },
    # --- Recession ---
    {
        "category": "Recession",
        "kalshi_pattern": r"recession in 2025",
        "poly_pattern": r"recession.*2025",
        "note": "US recession in 2025",
    },
    # --- State Elections: presidential ---
    {
        "category": "State Election",
        "kalshi_pattern": r"Republican party win Georgia in the presidential election",
        "poly_pattern": r"Republican win Georgia Presidential Election",
        "note": "GOP wins Georgia",
    },
    {
        "category": "State Election",
        "kalshi_pattern": r"Republican party win Pennsylvania in the presidential election",
        "poly_pattern": r"Republican win Pennsylvania Presidential Election",
        "note": "GOP wins Pennsylvania",
    },
    {
        "category": "State Election",
        "kalshi_pattern": r"Republican party win Michigan in the presidential election",
        "poly_pattern": r"Republican win Michigan Presidential Election",
        "note": "GOP wins Michigan",
    },
    {
        "category": "State Election",
        "kalshi_pattern": r"Republican party win Wisconsin in the presidential election",
        "poly_pattern": r"Republican win Wisconsin Presidential Election",
        "note": "GOP wins Wisconsin",
    },
    {
        "category": "State Election",
        "kalshi_pattern": r"Republican party win North Carolina in the presidential election",
        "poly_pattern": r"Republican win North Carolina Presidential Election",
        "note": "GOP wins North Carolina",
    },
    {
        "category": "State Election",
        "kalshi_pattern": r"Republican party win Nevada in the presidential election",
        "poly_pattern": r"Republican win Nevada Presidential Election",
        "note": "GOP wins Nevada",
    },
    {
        "category": "State Election",
        "kalshi_pattern": r"Republican party win Arizona in the presidential election",
        "poly_pattern": r"Republican win Arizona Presidential Election",
        "note": "GOP wins Arizona",
    },
    # --- Popular vote ---
    {
        "category": "Popular Vote",
        "kalshi_pattern": r"Trump win.*Electoral College and Trump win.*Popular Vote",
        "poly_pattern": r"Donald Trump win the popular vote.*2024",
        "note": "Trump wins popular vote (note: Kalshi market is joint EC+PV)",
    },
    # --- Virginia Governor ---
    {
        "category": "VA Governor",
        "kalshi_pattern": r"Winsome Earle-Sears win.*2025 Virginia gubernatorial",
        "poly_pattern": r"Winsome Sears win.*Virginia Governor",
        "note": "Sears wins VA Governor",
    },
    # --- NJ Governor ---
    {
        "category": "NJ Governor",
        "kalshi_pattern": r"Mikie Sherrill win.*2025 New Jersey gubernatorial",
        "poly_pattern": r"Mikie Sherrill win.*New Jersey Governor",
        "note": "Sherrill wins NJ Governor",
    },
    # --- Oscars ---
    {
        "category": "Oscars",
        "kalshi_pattern": r"Anora win Best Picture at the Oscars",
        "poly_pattern": r"Anora.*Best Picture.*2025 Oscars",
        "note": "Anora wins Best Picture",
    },
    {
        "category": "Oscars",
        "kalshi_pattern": r"Mikey Madison win Best Actress at the Oscars",
        "poly_pattern": r"Mikey Madison win Best Actress.*2025 Oscars",
        "note": "Mikey Madison wins Best Actress",
    },
    # --- Fed Chair nomination ---
    {
        "category": "Fed Chair",
        "kalshi_pattern": r"Trump.*nominate Kevin Warsh.*Fed Chair",
        "poly_pattern": r"Trump nominate Kevin Warsh.*Fed chair",
        "note": "Trump nominates Warsh as Fed Chair",
    },
    # --- Bitcoin ---
    {
        "category": "Bitcoin",
        "kalshi_pattern": r"When will Bitcoin hit \$100k",
        "poly_pattern": r"Bitcoin hit \$100k in 2024",
        "note": "Bitcoin hits $100k",
    },
    # --- Aliens / UAP ---
    {
        "category": "Aliens/UAP",
        "kalshi_pattern": r"U\.S\. confirm that aliens exist in 2026",
        "poly_pattern": r"US confirm that aliens exist before 2027",
        "note": "US confirms alien existence",
    },
    # --- Sinner Wimbledon ---
    {
        "category": "Tennis",
        "kalshi_pattern": r"Jannik Sinner win.*2025 Wimbledon",
        "poly_pattern": r"Jannik Sinner win Wimbledon 2025",
        "note": "Sinner wins Wimbledon 2025",
    },
    # --- NBA Finals ---
    {
        "category": "NBA Finals",
        "kalshi_pattern": r"Oklahoma City Thunder win the NBA Finals",
        "poly_pattern": r"Oklahoma City Thunder win.*2025 NBA Finals",
        "note": "Thunder win NBA Finals",
    },
    {
        "category": "NBA Finals",
        "kalshi_pattern": r"Boston Celtics win the NBA Finals",
        "poly_pattern": r"Boston Celtics win.*2025 NBA Finals",
        "note": "Celtics win NBA Finals",
    },
    # --- Chilean election ---
    {
        "category": "International Election",
        "kalshi_pattern": r"José Antonio Kast win the Chilean presidential",
        "poly_pattern": r"José Antonio Kast win the Chilean presidential",
        "note": "Kast wins Chilean election",
    },
    # --- Jay Jones VA AG ---
    {
        "category": "VA Attorney General",
        "kalshi_pattern": r"Jay Jones win.*2025 Virginia attorney general",
        "poly_pattern": r"Jay Jones win.*2025 Virginia Attorney General",
        "note": "Jay Jones wins VA AG",
    },
    # --- Liverpool EPL ---
    {
        "category": "Premier League",
        "kalshi_pattern": r"Liverpool win the English Premier League",
        "poly_pattern": r"Liverpool win.*English Premier League",
        "note": "Liverpool wins EPL",
    },
    # --- Donald Trump Time Person of Year ---
    {
        "category": "Time Person of Year",
        "kalshi_pattern": r"Donald Trump be Time Person of the Year in 2024",
        "poly_pattern": r"Donald Trump be TIME.*Person of the Year.*2024",
        "note": "Trump Time POTY 2024",
    },
    # --- Polish Election ---
    {
        "category": "International Election",
        "kalshi_pattern": r"Trzaskowski win the 2025 Polish presidential",
        "poly_pattern": r"Trzaskowski.*Polish.*Presidential",
        "note": "Trzaskowski wins Polish election",
    },
    # --- Wisconsin Supreme Court ---
    {
        "category": "State Election",
        "kalshi_pattern": r"Brad Schimel win the Wisconsin Supreme Court",
        "poly_pattern": r"Wisconsin Supreme Court.*Brad Schimel",
        "note": "Schimel wins WI Supreme Court",
    },
]

targeted_results = []
for match_spec in TARGETED_MATCHES:
    category = match_spec["category"]
    note = match_spec["note"]
    k_pattern = match_spec["kalshi_pattern"]
    p_pattern = match_spec["poly_pattern"]

    k_matches = kalshi_filtered[
        kalshi_filtered["title"].str.contains(k_pattern, case=False, na=False, regex=True)
    ]
    p_matches = poly_filtered[
        poly_filtered["question"].str.contains(p_pattern, case=False, na=False, regex=True)
    ]

    if len(k_matches) > 0 and len(p_matches) > 0:
        k_best = k_matches.nlargest(1, "volume").iloc[0]
        p_best = p_matches.nlargest(1, "volume").iloc[0]
        targeted_results.append({
            "category": category,
            "note": note,
            "kalshi_ticker": k_best["ticker"],
            "kalshi_title": k_best["title"],
            "kalshi_price": k_best["last_price"],
            "kalshi_volume": k_best["volume"],
            "poly_id": p_best["id"],
            "poly_question": p_best["question"],
            "poly_price": p_best["yes_price"],
            "poly_volume": p_best["volume"],
        })
    else:
        if len(k_matches) == 0:
            print(f"  [SKIP] No Kalshi match for: {note}")
        if len(p_matches) == 0:
            print(f"  [SKIP] No Poly match for: {note}")

targeted_df = pd.DataFrame(targeted_results)
if len(targeted_df) > 0:
    targeted_df["price_diff"] = targeted_df["kalshi_price"] - targeted_df["poly_price"]
    targeted_df["abs_diff"] = targeted_df["price_diff"].abs()

print(f"\nTargeted matches found: {len(targeted_df)}")
for _, row in targeted_df.iterrows():
    print(f"\n  [{row['category']}] {row['note']}")
    print(f"    Kalshi:  {str(row['kalshi_title'])[:100]}  (price={row['kalshi_price']:.1f}c)")
    print(f"    Poly:    {str(row['poly_question'])[:100]}  (price={row['poly_price']:.1f}c)")
    print(f"    Diff: {row['price_diff']:+.1f}c")

# ===========================================================================
# 5. Automated keyword matching (strict quality filtering)
# ===========================================================================
print("\n" + "=" * 70)
print("AUTOMATED KEYWORD MATCHING")
print("=" * 70)

# Use top markets by volume on both platforms
kalshi_top = kalshi_filtered.nlargest(5000, "volume").copy()
poly_top = poly_filtered.nlargest(10000, "volume").copy()
print(f"Using top {len(kalshi_top):,} Kalshi / top {len(poly_top):,} Polymarket markets")

# Build inverted index for Polymarket
poly_kw_index = defaultdict(set)
for idx, row in poly_top.iterrows():
    for kw in row["keywords"]:
        poly_kw_index[kw].add(idx)

# Match with high thresholds
MIN_SHARED = 3
matches = []
for k_idx, k_row in kalshi_top.iterrows():
    k_keywords = k_row["keywords"]
    if len(k_keywords) < MIN_SHARED:
        continue
    candidate_counts = Counter()
    for kw in k_keywords:
        for p_idx in poly_kw_index.get(kw, set()):
            candidate_counts[p_idx] += 1
    for p_idx, overlap_count in candidate_counts.items():
        if overlap_count >= MIN_SHARED:
            p_row = poly_top.loc[p_idx]
            shared = k_keywords & p_row["keywords"]
            union_size = len(k_keywords | p_row["keywords"])
            jaccard = len(shared) / union_size if union_size > 0 else 0
            matches.append({
                "kalshi_ticker": k_row["ticker"],
                "kalshi_title": k_row["title"],
                "kalshi_price": k_row["last_price"],
                "kalshi_volume": k_row["volume"],
                "poly_id": p_row["id"],
                "poly_question": p_row["question"],
                "poly_price": p_row["yes_price"],
                "poly_volume": p_row["volume"],
                "shared_keywords": ", ".join(sorted(shared)),
                "n_shared": overlap_count,
                "jaccard": jaccard,
            })

matches_df = pd.DataFrame(matches)
print(f"Raw keyword matches (>= {MIN_SHARED} shared): {len(matches_df):,}")

if len(matches_df) > 0:
    # STRICT filter: Jaccard >= 0.6 for high-confidence matches only
    matches_df = matches_df[matches_df["jaccard"] >= 0.6].copy()
    print(f"After Jaccard >= 0.6 filter: {len(matches_df):,}")

    # Keep best match per Kalshi market
    matches_df = matches_df.sort_values("jaccard", ascending=False).drop_duplicates(
        subset=["kalshi_ticker"], keep="first"
    )
    print(f"Best unique keyword matches (1 per Kalshi): {len(matches_df):,}")

    matches_df["price_diff"] = matches_df["kalshi_price"] - matches_df["poly_price"]
    matches_df["abs_diff"] = matches_df["price_diff"].abs()

    # CRITICAL: Filter out false matches where the underlying question
    # is clearly different (e.g., different specific outcomes/teams/states)
    # We flag matches where one market is settled (price 0-2 or 98-100)
    # and the other is not -- these are often temporal mismatches
    # (same event but different year/variant)
    matches_df["both_settled"] = (
        ((matches_df["kalshi_price"] <= 2) | (matches_df["kalshi_price"] >= 98)) &
        ((matches_df["poly_price"] <= 2) | (matches_df["poly_price"] >= 98))
    )
    matches_df["both_active"] = (
        (matches_df["kalshi_price"] > 2) & (matches_df["kalshi_price"] < 98) &
        (matches_df["poly_price"] > 2) & (matches_df["poly_price"] < 98)
    )
    # Keep: both settled similarly, both active mid-range, or close in price
    matches_df["quality_ok"] = (
        matches_df["both_settled"] |
        matches_df["both_active"] |
        (matches_df["abs_diff"] <= 15)  # close enough to be plausibly same market
    )
    quality_kw = matches_df[matches_df["quality_ok"]].copy()
    print(f"After quality filter (plausible same-market): {len(quality_kw):,}")

    # Show some keyword matches
    for i, (_, row) in enumerate(quality_kw.head(20).iterrows()):
        print(f"\n  [{i+1}] Jaccard={row['jaccard']:.2f}")
        print(f"    K: {str(row['kalshi_title'])[:100]}  ({row['kalshi_price']:.1f}c)")
        print(f"    P: {str(row['poly_question'])[:100]}  ({row['poly_price']:.1f}c)")
        print(f"    Diff: {row['price_diff']:+.1f}c  |  Keywords: {row['shared_keywords']}")
else:
    quality_kw = pd.DataFrame()

# ===========================================================================
# 6. Combine all matches
# ===========================================================================
print("\n" + "=" * 70)
print("COMBINED ANALYSIS")
print("=" * 70)

all_matches = []

# Add targeted matches
if len(targeted_df) > 0:
    for _, row in targeted_df.iterrows():
        all_matches.append({
            "category": row["category"],
            "note": row["note"],
            "kalshi_title": row["kalshi_title"],
            "kalshi_price": row["kalshi_price"],
            "kalshi_volume": row["kalshi_volume"],
            "poly_question": row["poly_question"],
            "poly_price": row["poly_price"],
            "poly_volume": row["poly_volume"],
            "price_diff": row["price_diff"],
            "match_method": "targeted",
        })

# Add quality keyword matches (not already in targeted)
if len(quality_kw) > 0:
    targeted_titles = set(targeted_df["kalshi_title"]) if len(targeted_df) > 0 else set()
    for _, row in quality_kw.iterrows():
        if row["kalshi_title"] not in targeted_titles:
            all_matches.append({
                "category": "keyword_match",
                "note": row["shared_keywords"],
                "kalshi_title": row["kalshi_title"],
                "kalshi_price": row["kalshi_price"],
                "kalshi_volume": row["kalshi_volume"],
                "poly_question": row["poly_question"],
                "poly_price": row["poly_price"],
                "poly_volume": row["poly_volume"],
                "price_diff": row["price_diff"],
                "match_method": "keyword",
            })

all_matches_df = pd.DataFrame(all_matches)
all_matches_df = all_matches_df.drop_duplicates(
    subset=["kalshi_title", "poly_question"], keep="first"
)

print(f"Total unique matched pairs: {len(all_matches_df)}")

# Separate truly comparable pairs from possibly mismatched ones
# (Settled markets are those where the event has already occurred)
if len(all_matches_df) > 0:
    all_matches_df["abs_diff"] = all_matches_df["price_diff"].abs()

    # Classify match quality
    all_matches_df["match_quality"] = "good"
    # Flag where one is settled and other isn't
    all_matches_df.loc[
        ((all_matches_df["kalshi_price"] <= 2) & (all_matches_df["poly_price"] >= 98)) |
        ((all_matches_df["kalshi_price"] >= 98) & (all_matches_df["poly_price"] <= 2)),
        "match_quality"
    ] = "likely_different_market"

    good_matches = all_matches_df[all_matches_df["match_quality"] == "good"]
    print(f"Good quality matches: {len(good_matches)}")
    print(f"Likely different markets (flagged): "
          f"{(all_matches_df['match_quality'] == 'likely_different_market').sum()}")

    diffs = good_matches["price_diff"]
    print(f"\nPrice difference statistics (good matches only, Kalshi - Polymarket, cents):")
    print(f"  N:      {len(diffs)}")
    print(f"  Mean:   {diffs.mean():+.2f}c")
    print(f"  Median: {diffs.median():+.2f}c")
    print(f"  Std:    {diffs.std():.2f}c")
    print(f"  Min:    {diffs.min():+.2f}c")
    print(f"  Max:    {diffs.max():+.2f}c")
    print(f"  MAE:    {diffs.abs().mean():.2f}c")
    if len(diffs) > 0:
        print(f"  Within 5c:  {(diffs.abs() <= 5).sum()} ({(diffs.abs() <= 5).mean()*100:.0f}%)")
        print(f"  Within 10c: {(diffs.abs() <= 10).sum()} ({(diffs.abs() <= 10).mean()*100:.0f}%)")

    # Break down by category
    print("\nBy category:")
    for cat in sorted(good_matches["category"].unique()):
        subset = good_matches[good_matches["category"] == cat]
        print(f"  {cat:30s}: n={len(subset):>3d}, mean_diff={subset['price_diff'].mean():+6.1f}c, "
              f"MAE={subset['price_diff'].abs().mean():5.1f}c")

    # Print all good matches
    print("\n" + "-" * 70)
    print("ALL GOOD-QUALITY MATCHED PAIRS")
    print("-" * 70)
    for i, (_, row) in enumerate(good_matches.sort_values("abs_diff", ascending=False).iterrows()):
        print(f"\n  [{i+1}] {row['category']} | {row.get('note', '')} | diff={row['price_diff']:+.1f}c")
        print(f"    K: {str(row['kalshi_title'])[:100]}  ({row['kalshi_price']:.1f}c)")
        print(f"    P: {str(row['poly_question'])[:100]}  ({row['poly_price']:.1f}c)")
else:
    good_matches = pd.DataFrame()

# ===========================================================================
# 7. Figures
# ===========================================================================
print("\n" + "=" * 70)
print("GENERATING FIGURES")
print("=" * 70)

CATEGORY_COLORS = {
    "Presidential Election": COLOR_BLUE,
    "Fed Rate Decision": COLOR_GREEN,
    "Super Bowl": COLOR_ORANGE,
    "State Election": "#DA8BC3",
    "Popular Vote": COLOR_PURPLE,
    "TikTok Ban": COLOR_CYAN,
    "Recession": COLOR_RED,
    "Oscars": "#FFB482",
    "VA Governor": "#8C8C8C",
    "NJ Governor": "#A1C9F4",
    "Fed Chair": "#CCB974",
    "Bitcoin": "#F7B267",
    "Aliens/UAP": "#B07AA1",
    "Tennis": "#59A14F",
    "NBA Finals": "#E15759",
    "International Election": "#76B7B2",
    "VA Attorney General": "#EDC948",
    "Premier League": "#FF9DA7",
    "Time Person of Year": "#9C755F",
    "keyword_match": "#AAAAAA",
}

# Use good_matches for figures if available, otherwise all_matches
plot_df = good_matches if len(good_matches) > 0 else all_matches_df

# --- Figure 1: Scatter plot ---
fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor="white")

if len(plot_df) > 0:
    ax.plot([0, 100], [0, 100], color=COLOR_GRAY, linestyle="--", linewidth=1.5,
            alpha=0.7, label="Perfect agreement", zorder=1)

    for cat in sorted(plot_df["category"].unique()):
        subset = plot_df[plot_df["category"] == cat]
        color = CATEGORY_COLORS.get(cat, COLOR_GRAY)
        label_name = cat if cat != "keyword_match" else "Keyword match"
        ax.scatter(
            subset["poly_price"], subset["kalshi_price"],
            c=color, s=90, alpha=0.8, edgecolors="white", linewidth=0.6,
            label=f"{label_name} (n={len(subset)})", zorder=2,
        )

    ax.set_xlabel("Polymarket Price (cents)", fontsize=13)
    ax.set_ylabel("Kalshi Price (cents)", fontsize=13)
    ax.set_title("Kalshi vs Polymarket Prices for Matched Markets", fontsize=15, fontweight="bold")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3)

    if len(plot_df) >= 3:
        corr = plot_df["kalshi_price"].corr(plot_df["poly_price"])
        mae = plot_df["price_diff"].abs().mean()
        ax.text(
            0.97, 0.05,
            f"r = {corr:.3f}\nMAE = {mae:.1f}c\nn = {len(plot_df)}",
            transform=ax.transAxes, fontsize=11, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
        )
else:
    ax.text(0.5, 0.5, "No matches found", transform=ax.transAxes,
            fontsize=16, ha="center", va="center", color=COLOR_GRAY)
    ax.set_xlabel("Polymarket Price (cents)", fontsize=13)
    ax.set_ylabel("Kalshi Price (cents)", fontsize=13)
    ax.set_title("Kalshi vs Polymarket Prices for Matched Markets", fontsize=15, fontweight="bold")

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "4_1_price_divergence_scatter.png"), dpi=DPI, facecolor="white")
plt.close(fig)
print("Saved: 4_1_price_divergence_scatter.png")

# --- Figure 2: Histogram of price differences ---
fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor="white")

if len(plot_df) > 0:
    diffs = plot_df["price_diff"].dropna()
    diff_range = max(abs(diffs.min()), abs(diffs.max()), 10)
    n_bins = min(40, max(10, len(diffs) // 2))
    bins = np.linspace(-diff_range - 2, diff_range + 2, n_bins)

    ax.hist(diffs, bins=bins, color=COLOR_BLUE, alpha=0.75, edgecolor="white", linewidth=0.8)
    ax.axvline(0, color=COLOR_GRAY, linestyle="--", linewidth=1.5, alpha=0.7, label="Zero difference")
    ax.axvline(diffs.mean(), color=COLOR_RED, linestyle="-", linewidth=2,
               alpha=0.8, label=f"Mean = {diffs.mean():+.1f}c")
    ax.axvline(diffs.median(), color=COLOR_GREEN, linestyle="-.", linewidth=2,
               alpha=0.8, label=f"Median = {diffs.median():+.1f}c")

    ax.set_xlabel("Price Difference: Kalshi - Polymarket (cents)", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title("Distribution of Cross-Platform Price Divergences", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    ax.text(
        0.97, 0.95,
        f"n = {len(diffs)}\n"
        f"std = {diffs.std():.1f}c\n"
        f"|diff| > 5c: {(diffs.abs() > 5).sum()} ({(diffs.abs() > 5).mean()*100:.0f}%)\n"
        f"|diff| > 10c: {(diffs.abs() > 10).sum()} ({(diffs.abs() > 10).mean()*100:.0f}%)",
        transform=ax.transAxes, fontsize=10, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
    )
else:
    ax.text(0.5, 0.5, "No matches found", transform=ax.transAxes,
            fontsize=16, ha="center", va="center", color=COLOR_GRAY)
    ax.set_xlabel("Price Difference: Kalshi - Polymarket (cents)", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title("Distribution of Cross-Platform Price Divergences", fontsize=15, fontweight="bold")

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "4_1_divergence_distribution.png"), dpi=DPI, facecolor="white")
plt.close(fig)
print("Saved: 4_1_divergence_distribution.png")

# ===========================================================================
# 8. JSON Output
# ===========================================================================
print("\n" + "=" * 70)
print("JSON OUTPUT")
print("=" * 70)

output = {
    "section": "4.1",
    "title": "Kalshi-Polymarket Divergence",
    "data_summary": {
        "kalshi_total_markets": int(len(kalshi_all)),
        "polymarket_total_markets": int(len(poly_all)),
        "kalshi_filtered": int(len(kalshi_filtered)),
        "polymarket_filtered": int(len(poly_filtered)),
        "shared_keywords": int(len(shared_keywords)),
    },
    "matching": {
        "targeted_matches": int(len(targeted_df)),
        "keyword_matches": int(len(quality_kw)) if len(quality_kw) > 0 else 0,
        "total_good_pairs": int(len(good_matches)) if len(good_matches) > 0 else 0,
    },
    "price_divergence": {},
    "matched_markets": [],
    "figures": [
        "figures/4_1_price_divergence_scatter.png",
        "figures/4_1_divergence_distribution.png",
    ],
}

if len(good_matches) > 0:
    diffs = good_matches["price_diff"]
    output["price_divergence"] = {
        "mean_diff_cents": round(float(diffs.mean()), 2),
        "median_diff_cents": round(float(diffs.median()), 2),
        "std_diff_cents": round(float(diffs.std()), 2),
        "mae_cents": round(float(diffs.abs().mean()), 2),
        "max_abs_diff_cents": round(float(diffs.abs().max()), 2),
        "pct_within_5c": round(float((diffs.abs() <= 5).mean() * 100), 1),
        "pct_within_10c": round(float((diffs.abs() <= 10).mean() * 100), 1),
        "correlation": round(float(
            good_matches["kalshi_price"].corr(good_matches["poly_price"])
        ), 4),
    }

    for _, row in good_matches.sort_values("abs_diff", ascending=False).iterrows():
        output["matched_markets"].append({
            "category": row["category"],
            "note": row.get("note", ""),
            "kalshi_title": str(row["kalshi_title"])[:120],
            "poly_question": str(row["poly_question"])[:120],
            "kalshi_price": round(float(row["kalshi_price"]), 1),
            "poly_price": round(float(row["poly_price"]), 1),
            "price_diff": round(float(row["price_diff"]), 1),
            "match_method": row["match_method"],
        })

print(json.dumps(output, indent=2))
print("\nDone.")
