from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.transforms._base import Transform

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COSINE_SIM_CROSS_PLATFORM_THRESHOLD = 0.70  # lower for cross-platform (different phrasing)
COSINE_SIM_INTRA_PLATFORM_THRESHOLD = 0.85  # higher for intra-platform (reduce trivial pairs)
MAX_PAIRS = 50_000
MAX_CROSS_PLATFORM_PAIRS = 20_000  # reserved budget for cross-platform
BATCH_SIZE = 50  # pairs per Gemini request
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_CONCURRENCY = 10  # concurrent Gemini requests

RELATIONSHIP_TYPES = {"identical", "hierarchical", "common_factor", "causal", "inverse", "unrelated"}

SYSTEM_PROMPT = """You are an expert analyst of prediction markets across Kalshi and Polymarket.

Given pairs of market titles, classify the relationship between each pair.

IMPORTANT: Markets on different platforms (Kalshi vs Polymarket) often describe THE SAME EVENT with different phrasing. You must recognize these as "identical" even when the wording differs. Examples:
- Kalshi: "Will Trump attend the UFC 319? | Trump" ↔ Polymarket: "Will Trump attend UFC 319?" → identical
- Kalshi: "Will Carlos Alcaraz win the Six Kings Slam? | Carlos Alcaraz" ↔ Polymarket: "Will Carlos Alcaraz win the Six Kings Slam?" → identical
- Kalshi: "Will Bruce Harrell win the election for the Mayor of Seattle in 2025? | Bruce Harrell" ↔ Polymarket: "Will Bruce Harrell win the 2025 Seattle mayoral election?" → identical
- Kalshi: "Musk out as Tesla CEO before 2026? | Before 2027" ↔ Polymarket: "Musk out as Tesla CEO before 2027?" → hierarchical (different deadlines)

Note that Kalshi titles often include a suffix after " | " (e.g. "| Yes", "| Trump", "| Before 2026") which is a sub-title for a specific contract within an event. Focus on the main question before the pipe.

For each pair, return a JSON object with:
- "relationship": one of "identical", "hierarchical", "common_factor", "causal", "inverse", "unrelated"
- "confidence": float 0.0 to 1.0
- "reasoning": brief explanation (1 sentence)

Relationship definitions:
- identical: Same event or question, possibly different platform, phrasing, or minor wording differences. If two markets would resolve based on the same real-world outcome, they are identical.
- hierarchical: One market implies the other (e.g. "X wins election" implies "X wins primary"), or they cover the same topic with different thresholds/timeframes
- common_factor: Both driven by the same underlying factor (e.g. two oil-related markets)
- causal: One outcome could directly cause the other
- inverse: Negatively correlated (one going up implies the other going down)
- unrelated: No meaningful relationship despite surface similarity

When in doubt between "identical" and "unrelated" for cross-platform pairs, lean toward "identical" if the core question is about the same real-world event.

Return a JSON array with one object per pair, in the same order as the input pairs.
"""


class T9bLLMClassification(Transform):
    def __init__(self) -> None:
        super().__init__(
            name="t9b",
            description="LLM classification of market pair relationships",
            dependencies=["t9a"],
        )

    def run(self) -> None:
        out = self.ensure_output_dir()
        output_path = out / "classified_pairs.parquet"

        # Load similarity pairs from T9a
        t9a_dir = self.base_dir / "data" / "transforms" / "t9a"
        pairs_df = pd.read_parquet(t9a_dir / "similarity_pairs.parquet")
        markets_df = pd.read_parquet(t9a_dir / "markets_unified.parquet", columns=["market_id", "platform", "title"])

        # Dual threshold: lower for cross-platform, higher for intra-platform
        is_cross = pairs_df["platform_a"] != pairs_df["platform_b"]
        cross_mask = is_cross & (pairs_df["cosine_sim"] >= COSINE_SIM_CROSS_PLATFORM_THRESHOLD)
        intra_mask = ~is_cross & (pairs_df["cosine_sim"] >= COSINE_SIM_INTRA_PLATFORM_THRESHOLD)

        cross_pairs = pairs_df[cross_mask].copy()
        intra_pairs = pairs_df[intra_mask].copy()

        logger.info(
            "Cross-platform pairs >= %.2f: %d, Intra-platform pairs >= %.2f: %d",
            COSINE_SIM_CROSS_PLATFORM_THRESHOLD, len(cross_pairs),
            COSINE_SIM_INTRA_PLATFORM_THRESHOLD, len(intra_pairs),
        )

        # Filter trivial intra-platform pairs: same title = recurring market instances
        title_map = dict(zip(markets_df["market_id"], markets_df["title"]))
        if not intra_pairs.empty:
            intra_pairs["_title_a"] = intra_pairs["market_a"].map(title_map)
            intra_pairs["_title_b"] = intra_pairs["market_b"].map(title_map)
            trivial_mask = intra_pairs["_title_a"] == intra_pairs["_title_b"]
            n_trivial = trivial_mask.sum()
            intra_pairs = intra_pairs[~trivial_mask].copy()
            intra_pairs = intra_pairs.drop(columns=["_title_a", "_title_b"])
            logger.info("Filtered %d trivial intra-platform pairs (identical titles)", n_trivial)

        # Budget allocation: prioritize cross-platform pairs
        if len(cross_pairs) > MAX_CROSS_PLATFORM_PAIRS:
            cross_pairs = cross_pairs.nlargest(MAX_CROSS_PLATFORM_PAIRS, "cosine_sim")

        intra_budget = MAX_PAIRS - len(cross_pairs)
        if len(intra_pairs) > intra_budget:
            intra_pairs = intra_pairs.nlargest(intra_budget, "cosine_sim")

        pairs_df = pd.concat([cross_pairs, intra_pairs], ignore_index=True)
        logger.info(
            "Final pair selection: %d cross-platform + %d intra-platform = %d total",
            len(cross_pairs), len(intra_pairs), len(pairs_df),
        )

        if pairs_df.empty:
            logger.warning("No pairs above threshold, writing empty output")
            empty = pd.DataFrame(columns=[
                "market_a", "platform_a", "market_b", "platform_b", "cosine_sim",
                "title_a", "title_b", "relationship_type", "confidence", "reasoning",
            ])
            empty.to_parquet(output_path, index=False)
            self.write_manifest({"total_pairs": 0, "classified_pairs": 0})
            return

        # Join market titles
        title_map = dict(zip(markets_df["market_id"], markets_df["title"]))
        pairs_df["title_a"] = pairs_df["market_a"].map(title_map)
        pairs_df["title_b"] = pairs_df["market_b"].map(title_map)

        # Drop any pairs where we couldn't find titles
        pairs_df = pairs_df.dropna(subset=["title_a", "title_b"])
        pairs_df = pairs_df.reset_index(drop=True)

        # Checkpoint directory for resume
        checkpoint_dir = out / "classification_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        n_batches = (len(pairs_df) + BATCH_SIZE - 1) // BATCH_SIZE

        # Find already-done batches
        done_batches: set[int] = set()
        for f in checkpoint_dir.glob("batch_*.parquet"):
            try:
                idx = int(f.stem.split("_")[1])
                done_batches.add(idx)
            except (IndexError, ValueError):
                pass

        logger.info("%d batches total, %d already done", n_batches, len(done_batches))

        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is required for T9b classification")

        client = genai.Client(api_key=api_key)

        remaining = [i for i in range(n_batches) if i not in done_batches]
        logger.info("%d batches remaining to classify", len(remaining))

        import asyncio

        async def _classify_one(batch_idx: int, sem: asyncio.Semaphore) -> None:
            start = batch_idx * BATCH_SIZE
            end = min(start + BATCH_SIZE, len(pairs_df))
            batch = pairs_df.iloc[start:end]

            async with sem:
                classifications = await asyncio.get_event_loop().run_in_executor(
                    None, self._classify_batch, client, batch
                )

            batch_out = batch.copy()
            batch_out["relationship_type"] = [c.get("relationship", "unrelated") for c in classifications]
            batch_out["confidence"] = [c.get("confidence", 0.0) for c in classifications]
            batch_out["reasoning"] = [c.get("reasoning", "") for c in classifications]
            batch_out.to_parquet(checkpoint_dir / f"batch_{batch_idx:06d}.parquet", index=False)

        async def _classify_all() -> None:
            sem = asyncio.Semaphore(GEMINI_CONCURRENCY)
            pbar = tqdm(total=len(remaining), desc="Classification batches", file=__import__("sys").stderr)

            for wave_start in range(0, len(remaining), GEMINI_CONCURRENCY * 4):
                wave = remaining[wave_start : wave_start + GEMINI_CONCURRENCY * 4]
                tasks = [_classify_one(idx, sem) for idx in wave]
                for coro in asyncio.as_completed(tasks):
                    await coro
                    pbar.update(1)
            pbar.close()

        with self.progress("Classifying market pairs"):
            asyncio.run(_classify_all())

        # Assemble all checkpoints
        all_parts = []
        for batch_idx in range(n_batches):
            part = pd.read_parquet(checkpoint_dir / f"batch_{batch_idx:06d}.parquet")
            all_parts.append(part)

        result = pd.concat(all_parts, ignore_index=True)

        # Validate relationship types
        result["relationship_type"] = result["relationship_type"].apply(
            lambda x: x if x in RELATIONSHIP_TYPES else "unrelated"
        )

        result.to_parquet(output_path, index=False)
        logger.info("Wrote %d classified pairs to %s", len(result), output_path)

        type_counts = result["relationship_type"].value_counts().to_dict()
        self.write_manifest({
            "total_pairs": len(result),
            "classified_pairs": len(result[result["relationship_type"] != "unrelated"]),
            "type_counts": type_counts,
        })

    def _classify_batch(self, client, batch: pd.DataFrame) -> list[dict]:
        """Classify a batch of pairs using Gemini. Returns list of {relationship, confidence, reasoning}."""
        # Build prompt with numbered pairs
        lines = []
        for i, (_, row) in enumerate(batch.iterrows()):
            lines.append(f"Pair {i + 1}:")
            lines.append(f"  Market A: {row['title_a']}")
            lines.append(f"  Market B: {row['title_b']}")
            lines.append("")

        user_prompt = "Classify the relationship for each pair:\n\n" + "\n".join(lines)
        user_prompt += f"\nReturn a JSON array of {len(batch)} objects."

        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=user_prompt,
                config={
                    "system_instruction": SYSTEM_PROMPT,
                    "temperature": 0.1,
                    "response_mime_type": "application/json",
                },
            )

            text = response.text.strip()
            parsed = json.loads(text)

            if isinstance(parsed, list) and len(parsed) == len(batch):
                return parsed

            # If wrong length, pad/truncate
            logger.warning(
                "Gemini returned %d results for %d pairs, adjusting",
                len(parsed) if isinstance(parsed, list) else 0,
                len(batch),
            )
            if isinstance(parsed, list):
                while len(parsed) < len(batch):
                    parsed.append({"relationship": "unrelated", "confidence": 0.0, "reasoning": "parse error"})
                return parsed[: len(batch)]

            return [{"relationship": "unrelated", "confidence": 0.0, "reasoning": "parse error"}] * len(batch)

        except Exception:
            logger.warning("Gemini classification failed for batch", exc_info=True)
            return [{"relationship": "unrelated", "confidence": 0.0, "reasoning": "api error"}] * len(batch)
