from __future__ import annotations

import logging
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.transforms._base import Transform
from src.transforms._util import get_tmp_dir

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072
EMBEDDING_BATCH_SIZE = 2048  # OpenAI max per request
FAISS_NCELLS = 4096
FAISS_NPROBE = 16  # lower nprobe for 3072-dim vectors; >90% recall at sim>=0.70
FAISS_TOP_K = 50
COSINE_SIM_THRESHOLD = 0.70
HDBSCAN_MIN_CLUSTER_SIZE = 5
MIN_TITLE_LENGTH = 10
EMBEDDING_CONCURRENCY = 10  # concurrent OpenAI requests


class T9aSemanticClustering(Transform):
    def __init__(self) -> None:
        super().__init__(
            name="t9a",
            description="Semantic clustering of prediction markets via embeddings",
            dependencies=[],
        )

    def _make_connection(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect()
        tmp_dir = get_tmp_dir()
        con.execute("SET memory_limit='20GB'")
        con.execute(f"SET temp_directory='{tmp_dir}'")
        con.execute("SET preserve_insertion_order=false")
        con.execute("SET threads=4")
        return con

    def run(self) -> None:
        out = self.ensure_output_dir()

        # Phase 1: Extract unified markets
        markets_path = out / "markets_unified.parquet"
        if not markets_path.exists():
            self._extract_unified_markets(markets_path)
        else:
            logger.info("Phase 1 skip: markets_unified.parquet exists")

        # Phase 2: Embed all titles
        embeddings_path = out / "embeddings.npy"
        index_path = out / "embedding_index.parquet"
        if not embeddings_path.exists() or not index_path.exists():
            self._embed_titles(markets_path, embeddings_path, index_path)
        else:
            logger.info("Phase 2 skip: embeddings.npy exists")

        # Phase 3: FAISS ANN search
        pairs_path = out / "similarity_pairs.parquet"
        if not pairs_path.exists():
            self._faiss_search(embeddings_path, index_path, pairs_path)
        else:
            logger.info("Phase 3 skip: similarity_pairs.parquet exists")

        # Phase 4: HDBSCAN clustering
        clusters_path = out / "clusters.parquet"
        if not clusters_path.exists():
            self._hdbscan_clustering(embeddings_path, index_path, pairs_path, clusters_path)
        else:
            logger.info("Phase 4 skip: clusters.parquet exists")

        # Read counts for manifest
        con = self._make_connection()
        try:
            n_markets = con.execute(f"SELECT COUNT(*) FROM read_parquet('{markets_path}')").fetchone()[0]
            n_pairs = con.execute(f"SELECT COUNT(*) FROM read_parquet('{pairs_path}')").fetchone()[0]
            n_clusters = con.execute(
                f"SELECT COUNT(DISTINCT cluster_id) FROM read_parquet('{clusters_path}') WHERE cluster_id >= 0"
            ).fetchone()[0]
        finally:
            con.close()

        self.write_manifest({
            "total_markets": n_markets,
            "similarity_pairs": n_pairs,
            "clusters": n_clusters,
        })

    # -----------------------------------------------------------------------
    # Phase 1: Extract unified markets
    # -----------------------------------------------------------------------
    def _extract_unified_markets(self, output_path: Path) -> None:
        with self.progress("Phase 1: Extracting unified markets"):
            con = self._make_connection()
            try:
                kalshi_markets = str(self.base_dir / "data" / "kalshi" / "markets" / "*.parquet")
                poly_markets = str(self.base_dir / "data" / "polymarket" / "markets" / "*.parquet")

                # Kalshi: aggregate at event_ticker level, pick highest-volume ticker's title
                kalshi_sql = f"""
                    WITH ranked AS (
                        SELECT
                            event_ticker AS market_id,
                            'kalshi' AS platform,
                            title || COALESCE(' | ' || NULLIF(yes_sub_title, ''), '') AS title,
                            event_ticker AS event_group,
                            volume,
                            created_time,
                            status,
                            ROW_NUMBER() OVER (
                                PARTITION BY event_ticker
                                ORDER BY volume DESC NULLS LAST
                            ) AS rn
                        FROM read_parquet('{kalshi_markets}')
                        WHERE event_ticker IS NOT NULL
                    )
                    SELECT market_id, platform, title, event_group, volume, created_time, status
                    FROM ranked
                    WHERE rn = 1
                      AND LENGTH(title) >= {MIN_TITLE_LENGTH}
                """

                # Polymarket: one row per market id
                poly_sql = f"""
                    SELECT
                        id AS market_id,
                        'polymarket' AS platform,
                        question AS title,
                        id AS event_group,
                        volume::BIGINT AS volume,
                        created_at AS created_time,
                        CASE WHEN closed THEN 'closed' ELSE 'active' END AS status
                    FROM read_parquet('{poly_markets}')
                    WHERE LENGTH(question) >= {MIN_TITLE_LENGTH}
                """

                combined = f"""
                    SELECT * FROM ({kalshi_sql})
                    UNION ALL
                    SELECT * FROM ({poly_sql})
                """

                df = con.execute(combined).fetchdf()
                df.to_parquet(output_path, index=False)
                logger.info("Phase 1: wrote %d unified markets to %s", len(df), output_path)
            finally:
                con.close()

    # -----------------------------------------------------------------------
    # Phase 2: Embed all titles
    # -----------------------------------------------------------------------
    def _embed_titles(self, markets_path: Path, embeddings_path: Path, index_path: Path) -> None:
        import asyncio

        from openai import AsyncOpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is required for T9a embeddings")

        client = AsyncOpenAI(api_key=api_key)

        df = pd.read_parquet(markets_path, columns=["market_id", "platform", "title"])
        n_total = len(df)
        titles = df["title"].tolist()

        checkpoint_dir = self.output_dir / "embedding_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Find which batches are already done
        n_batches = (n_total + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE
        done_batches: set[int] = set()
        for f in checkpoint_dir.glob("batch_*.npy"):
            try:
                idx = int(f.stem.split("_")[1])
                done_batches.add(idx)
            except (IndexError, ValueError):
                pass

        remaining = [i for i in range(n_batches) if i not in done_batches]
        logger.info(
            "Phase 2: %d markets, %d batches (%d already done, %d remaining)",
            n_total, n_batches, len(done_batches), len(remaining),
        )

        async def _embed_batch(batch_idx: int, sem: asyncio.Semaphore) -> None:
            start = batch_idx * EMBEDDING_BATCH_SIZE
            end = min(start + EMBEDDING_BATCH_SIZE, n_total)
            batch_titles = [t if t.strip() else "unknown" for t in titles[start:end]]

            async with sem:
                response = await client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch_titles,
                )

            batch_embeddings = np.array(
                [item.embedding for item in response.data],
                dtype=np.float32,
            )
            np.save(checkpoint_dir / f"batch_{batch_idx:06d}.npy", batch_embeddings)

        async def _embed_all() -> None:
            sem = asyncio.Semaphore(EMBEDDING_CONCURRENCY)
            pbar = tqdm(total=len(remaining), desc="Embedding batches", file=__import__("sys").stderr)

            # Process in waves to allow tqdm updates and avoid unbounded memory
            for wave_start in range(0, len(remaining), EMBEDDING_CONCURRENCY * 4):
                wave = remaining[wave_start : wave_start + EMBEDDING_CONCURRENCY * 4]
                tasks = [_embed_batch(idx, sem) for idx in wave]
                for coro in asyncio.as_completed(tasks):
                    await coro
                    pbar.update(1)
            pbar.close()

        with self.progress("Phase 2: Embedding titles"):
            asyncio.run(_embed_all())

        # Assemble all batches into single array
        all_embeddings = []
        for batch_idx in range(n_batches):
            chunk = np.load(checkpoint_dir / f"batch_{batch_idx:06d}.npy")
            all_embeddings.append(chunk)

        embeddings = np.vstack(all_embeddings)
        assert embeddings.shape == (n_total, EMBEDDING_DIM), (
            f"Expected ({n_total}, {EMBEDDING_DIM}), got {embeddings.shape}"
        )

        np.save(embeddings_path, embeddings)
        logger.info("Phase 2: saved embeddings %s to %s", embeddings.shape, embeddings_path)

        # Write index mapping row → market_id
        index_df = df[["market_id", "platform"]].reset_index(drop=True)
        index_df.to_parquet(index_path, index=False)
        logger.info("Phase 2: saved embedding index to %s", index_path)

    # -----------------------------------------------------------------------
    # Phase 3: FAISS ANN search
    # -----------------------------------------------------------------------
    def _faiss_search(self, embeddings_path: Path, index_path: Path, output_path: Path) -> None:
        import faiss

        CHUNK = 50_000  # rows per batch for add/search to limit peak RAM
        TRAIN_SAMPLE = 200_000  # vectors to train IVF centroids

        index_df = pd.read_parquet(index_path)
        n = len(index_df)

        with self.progress("Phase 3: FAISS index build"):
            # Memory-map embeddings to avoid loading all 20GB at once
            embeddings = np.load(embeddings_path, mmap_mode="r")
            assert embeddings.shape == (n, EMBEDDING_DIM)

            ncells = min(FAISS_NCELLS, max(1, n // 40))
            quantizer = faiss.IndexFlatIP(EMBEDDING_DIM)
            index = faiss.IndexIVFFlat(quantizer, EMBEDDING_DIM, ncells, faiss.METRIC_INNER_PRODUCT)

            # Train on random subsample (normalized copy)
            rng = np.random.default_rng(42)
            train_idx = rng.choice(n, size=min(TRAIN_SAMPLE, n), replace=False)
            train_idx.sort()
            train_data = np.array(embeddings[train_idx], dtype=np.float32)
            faiss.normalize_L2(train_data)

            logger.info("Phase 3: training FAISS on %d sample vectors, %d cells", len(train_data), ncells)
            index.train(train_data)
            del train_data

            # Add in chunks (normalize each chunk, then add)
            for start in range(0, n, CHUNK):
                end = min(start + CHUNK, n)
                chunk = np.array(embeddings[start:end], dtype=np.float32)
                faiss.normalize_L2(chunk)
                index.add(chunk)

            index.nprobe = min(FAISS_NPROBE, ncells)
            logger.info("Phase 3: index built with %d vectors", index.ntotal)

        # Search in chunks to limit memory
        faiss.omp_set_num_threads(8)
        n_chunks = (n + CHUNK - 1) // CHUNK
        all_sims = []
        all_ids = []
        with self.progress("Phase 3: FAISS search"):
            for ci, start in enumerate(range(0, n, CHUNK)):
                end = min(start + CHUNK, n)
                logger.info("Phase 3: searching chunk %d/%d (%d-%d)", ci + 1, n_chunks, start, end)
                chunk = np.array(embeddings[start:end], dtype=np.float32)
                faiss.normalize_L2(chunk)
                sims, ids = index.search(chunk, FAISS_TOP_K)
                all_sims.append(sims)
                all_ids.append(ids)

        similarities = np.vstack(all_sims)
        indices = np.vstack(all_ids)
        del all_sims, all_ids

        # Build pairs DataFrame, deduplicate
        with self.progress("Phase 3: Building pair table"):
            market_ids = index_df["market_id"].values
            platforms = index_df["platform"].values

            rows = []
            seen: set[tuple[str, str]] = set()
            for i in range(n):
                for j_pos in range(FAISS_TOP_K):
                    j = indices[i, j_pos]
                    if j < 0 or j == i:
                        continue
                    sim = float(similarities[i, j_pos])
                    if sim < COSINE_SIM_THRESHOLD:
                        continue

                    # Deduplicate (A,B) = (B,A)
                    pair_key = (min(market_ids[i], market_ids[j]), max(market_ids[i], market_ids[j]))
                    if pair_key in seen:
                        continue
                    seen.add(pair_key)

                    rows.append({
                        "market_a": market_ids[i],
                        "platform_a": platforms[i],
                        "market_b": market_ids[j],
                        "platform_b": platforms[j],
                        "cosine_sim": round(sim, 6),
                    })

            pairs_df = pd.DataFrame(rows)
            pairs_df.to_parquet(output_path, index=False)
            logger.info("Phase 3: wrote %d similarity pairs to %s", len(pairs_df), output_path)

    # -----------------------------------------------------------------------
    # Phase 4: HDBSCAN clustering
    # -----------------------------------------------------------------------
    def _hdbscan_clustering(
        self, embeddings_path: Path, index_path: Path, pairs_path: Path, output_path: Path
    ) -> None:
        from sklearn.cluster import HDBSCAN

        with self.progress("Phase 4: HDBSCAN clustering"):
            pairs_df = pd.read_parquet(pairs_path)
            index_df = pd.read_parquet(index_path)
            embeddings = np.load(embeddings_path, mmap_mode="r")

            # Only cluster markets in high-similarity pairs (>=0.80) to keep tractable
            high_sim = pairs_df[pairs_df["cosine_sim"] >= 0.80]
            pair_markets = set(high_sim["market_a"].tolist()) | set(high_sim["market_b"].tolist())
            logger.info("Phase 4: %d markets in high-sim pairs (>=0.80)", len(pair_markets))
            mask = index_df["market_id"].isin(pair_markets)
            subset_indices = np.where(mask.values)[0]

            MAX_HDBSCAN_MARKETS = 50_000
            if len(subset_indices) < HDBSCAN_MIN_CLUSTER_SIZE:
                logger.warning("Phase 4: only %d markets in pairs, skipping clustering", len(subset_indices))
                empty = pd.DataFrame(columns=["market_id", "platform", "cluster_id"])
                empty.to_parquet(output_path, index=False)
                return

            if len(subset_indices) > MAX_HDBSCAN_MARKETS:
                # Too many for HDBSCAN; sample the most-connected markets
                logger.info(
                    "Phase 4: %d markets exceeds %d cap, sampling top-connected",
                    len(subset_indices), MAX_HDBSCAN_MARKETS,
                )
                # Count how many high-sim pairs each market appears in
                from collections import Counter
                counts: Counter = Counter()
                counts.update(high_sim["market_a"].tolist())
                counts.update(high_sim["market_b"].tolist())
                top_markets = {m for m, _ in counts.most_common(MAX_HDBSCAN_MARKETS)}
                mask = index_df["market_id"].isin(top_markets)
                subset_indices = np.where(mask.values)[0]
                logger.info("Phase 4: reduced to %d markets", len(subset_indices))

            subset_embeddings = np.array(embeddings[subset_indices], dtype=np.float32)

            # L2-normalize for cosine metric
            norms = np.linalg.norm(subset_embeddings, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            subset_embeddings = subset_embeddings / norms

            # PCA to 50 dims — HDBSCAN's tree methods degrade above ~20 dims
            from sklearn.decomposition import PCA
            pca_dims = min(50, subset_embeddings.shape[1], subset_embeddings.shape[0])
            logger.info("Phase 4: PCA %d → %d dims", subset_embeddings.shape[1], pca_dims)
            pca = PCA(n_components=pca_dims, random_state=42)
            subset_reduced = pca.fit_transform(subset_embeddings)
            del subset_embeddings

            logger.info("Phase 4: clustering %d markets (%d dims)", len(subset_indices), subset_reduced.shape[1])
            clusterer = HDBSCAN(
                min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
                metric="euclidean",
            )
            labels = clusterer.fit_predict(subset_reduced)

            cluster_df = pd.DataFrame({
                "market_id": index_df.iloc[subset_indices]["market_id"].values,
                "platform": index_df.iloc[subset_indices]["platform"].values,
                "cluster_id": labels,
            })
            cluster_df.to_parquet(output_path, index=False)

            n_clusters = len(set(labels) - {-1})
            n_noise = int((labels == -1).sum())
            logger.info(
                "Phase 4: %d clusters, %d noise points, wrote to %s",
                n_clusters, n_noise, output_path,
            )
