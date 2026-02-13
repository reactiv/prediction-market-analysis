from __future__ import annotations

import logging

import duckdb
import pandas as pd

from src.transforms._base import Transform
from src.transforms._util import copy_to_parquet, get_tmp_dir

logger = logging.getLogger(__name__)


class T7AddressGraph(Transform):
    def __init__(self):
        super().__init__(
            name="t7",
            description="Address interaction graph (Polymarket)",
            dependencies=["t3", "t1b"],
        )

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    def run(self):
        self.ensure_output_dir()
        structural_dir = self.output_dir / "structural"
        temporal_dir = self.output_dir / "temporal"
        structural_dir.mkdir(parents=True, exist_ok=True)
        temporal_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1 — Structural Graph
        edge_rows = self._build_edge_list()
        n_clusters = self._build_communities()
        self._build_cluster_performance()

        # Phase 2 — Temporal Patterns
        self._detect_sybils()
        self._detect_cascades()

        self.write_manifest(
            {
                "edge_list_rows": edge_rows,
                "n_clusters": n_clusters,
            }
        )

    # ------------------------------------------------------------------
    # Phase 1 — Step 1: Build bipartite edge list
    # ------------------------------------------------------------------

    def _build_edge_list(self) -> int:
        """Build (address, token_id, direction, block_number, size) from raw
        Polymarket trades and write to parquet via DuckDB COPY."""

        with self.progress("Building bipartite edge list"):
            trades_glob = str(self.base_dir / "data" / "polymarket" / "trades" / "*.parquet")
            edge_dir = self.output_dir / "structural" / "edge_list.parquet"
            edge_dir.mkdir(parents=True, exist_ok=True)

            tmp_dir = get_tmp_dir()
            con = duckdb.connect()
            con.execute(f"SET temp_directory = '{tmp_dir}'")

            # Each trade produces TWO edge-list rows: one for the buyer, one
            # for the seller.
            #   maker_asset_id = '0' => maker is the buyer (pays USDC),
            #                           taker is the seller (sells tokens)
            #   maker_asset_id != '0' => maker is the seller, taker is the buyer.
            #
            # token_id = the conditional-token asset id (non-USDC side).
            # size     = the USDC amount of the trade.
            query = f"""
                WITH raw AS (
                    SELECT
                        block_number,
                        maker,
                        taker,
                        maker_asset_id,
                        taker_asset_id,
                        maker_amount,
                        taker_amount
                    FROM read_parquet('{trades_glob}')
                ),
                classified AS (
                    SELECT
                        block_number,
                        -- buyer / seller
                        CASE WHEN maker_asset_id = '0' THEN maker ELSE taker END AS buyer,
                        CASE WHEN maker_asset_id = '0' THEN taker ELSE maker END AS seller,
                        -- token id is the non-USDC asset
                        CASE WHEN maker_asset_id = '0' THEN taker_asset_id ELSE maker_asset_id END AS token_id,
                        -- size is the USDC amount
                        CASE WHEN maker_asset_id = '0' THEN maker_amount::DOUBLE ELSE taker_amount::DOUBLE END AS size
                    FROM raw
                )
                -- Unpivot into two rows per trade: one buy, one sell
                SELECT buyer  AS address, token_id, 'buy'  AS direction, block_number, size FROM classified
                UNION ALL
                SELECT seller AS address, token_id, 'sell' AS direction, block_number, size FROM classified
            """

            row_count = copy_to_parquet(con, query, edge_dir)
            con.close()

        logger.info("Edge list: %s rows", f"{row_count:,}")
        return row_count

    # ------------------------------------------------------------------
    # Phase 1 — Steps 2-4: Project graph & community detection
    # ------------------------------------------------------------------

    def _build_communities(self) -> int:
        """Filter top addresses, project address-address graph, run Louvain
        community detection, and write cluster assignments."""

        structural_dir = self.output_dir / "structural"
        edge_glob = str(structural_dir / "edge_list.parquet" / "*.parquet")

        # ---- Step 2: top ~50K most-active addresses --------------------
        with self.progress("Filtering top addresses"):
            tmp_dir = get_tmp_dir()
            con = duckdb.connect()
            con.execute(f"SET temp_directory = '{tmp_dir}'")

            con.execute(f"""
                CREATE OR REPLACE VIEW edge_list AS
                SELECT * FROM read_parquet('{edge_glob}')
            """)

            con.execute("""
                CREATE OR REPLACE TABLE top_addresses AS
                SELECT address, COUNT(DISTINCT token_id) AS market_count
                FROM edge_list
                GROUP BY address
                ORDER BY market_count DESC
                LIMIT 50000
            """)

            top_count = con.execute("SELECT COUNT(*) FROM top_addresses").fetchone()[0]
            logger.info("Top addresses: %s", f"{top_count:,}")

        # ---- Step 3: address-address projection ------------------------
        with self.progress("Projecting address-address graph"):
            projection_query = """
                SELECT
                    a.address AS address_a,
                    b.address AS address_b,
                    COUNT(DISTINCT a.token_id) AS shared_markets,
                    SUM(1) AS interaction_count
                FROM edge_list a
                INNER JOIN edge_list b
                    ON a.token_id = b.token_id
                    AND a.address < b.address
                WHERE a.address IN (SELECT address FROM top_addresses)
                  AND b.address IN (SELECT address FROM top_addresses)
                GROUP BY a.address, b.address
                HAVING COUNT(DISTINCT a.token_id) >= 3
            """

            projection_dir = structural_dir / "address_projection.parquet"
            projection_dir.mkdir(parents=True, exist_ok=True)
            copy_to_parquet(con, projection_query, projection_dir)

            # Read back into pandas for igraph
            proj_df = con.execute(f"""
                SELECT address_a, address_b, shared_markets
                FROM read_parquet('{str(projection_dir / "*.parquet")}')
            """).fetchdf()

        con.close()

        logger.info("Projected edges: %s", f"{len(proj_df):,}")

        # ---- Step 4: Community detection (Louvain) ---------------------
        n_clusters = 0
        with self.progress("Running community detection"):
            try:
                import igraph as ig
            except ImportError:
                logger.warning(
                    "python-igraph not installed — skipping community detection"
                )
                # Write an empty cluster_assignments parquet so downstream
                # steps can still reference the file.
                empty = pd.DataFrame(columns=["address", "cluster_id"])
                empty.to_parquet(structural_dir / "cluster_assignments.parquet", index=False)
                return 0

            if proj_df.empty:
                logger.warning("Projection is empty — no communities to detect")
                empty = pd.DataFrame(columns=["address", "cluster_id"])
                empty.to_parquet(structural_dir / "cluster_assignments.parquet", index=False)
                return 0

            edges = list(zip(proj_df.address_a, proj_df.address_b, proj_df.shared_markets))
            g = ig.Graph.TupleList(edges, weights=True, directed=False)
            communities = g.community_multilevel(weights="weight")

            n_clusters = len(communities)
            logger.info("Communities found: %s", f"{n_clusters:,}")

            # Build cluster assignment dataframe
            assignments: list[dict[str, object]] = []
            for cluster_id, members in enumerate(communities):
                for vertex_idx in members:
                    assignments.append(
                        {
                            "address": g.vs[vertex_idx]["name"],
                            "cluster_id": cluster_id,
                        }
                    )

            ca_df = pd.DataFrame(assignments)
            ca_df.to_parquet(structural_dir / "cluster_assignments.parquet", index=False)
            logger.info("Cluster assignments written: %s addresses", f"{len(ca_df):,}")

        return n_clusters

    # ------------------------------------------------------------------
    # Phase 1 — Step 5: Cluster performance
    # ------------------------------------------------------------------

    def _build_cluster_performance(self):
        """Join cluster assignments with T3 address_summary for per-cluster
        performance stats."""

        with self.progress("Building cluster performance"):
            structural_dir = self.output_dir / "structural"
            ca_path = str(structural_dir / "cluster_assignments.parquet")
            t3_path = str(self.base_dir / "data" / "transforms" / "t3" / "address_summary.parquet")

            con = duckdb.connect()
            query = f"""
                SELECT
                    ca.cluster_id,
                    COUNT(*)                    AS member_count,
                    AVG(s.win_rate)             AS avg_win_rate,
                    SUM(s.total_volume)         AS total_volume,
                    AVG(s.realized_pnl)         AS avg_realized_pnl
                FROM read_parquet('{ca_path}') ca
                INNER JOIN read_parquet('{t3_path}') s
                    ON ca.address = s.address
                GROUP BY ca.cluster_id
                ORDER BY total_volume DESC
            """

            result = con.execute(query).fetchdf()
            out_path = structural_dir / "cluster_performance.parquet"
            result.to_parquet(out_path, index=False)
            con.close()

            logger.info("Cluster performance: %s clusters", f"{len(result):,}")

    # ------------------------------------------------------------------
    # Phase 2 — Step 6 helper: qualifying markets
    # ------------------------------------------------------------------

    def _load_qualifying_market_ids(self, con: duckdb.DuckDBPyConnection) -> None:
        """Register qualifying Polymarket market IDs as a temp table."""
        qualifying_path = str(
            self.base_dir / "data" / "transforms" / "t1b" / "qualifying_markets.parquet"
        )
        con.execute(f"""
            CREATE OR REPLACE TABLE qualifying AS
            SELECT market_id
            FROM read_parquet('{qualifying_path}')
            WHERE platform = 'polymarket'
        """)

    # ------------------------------------------------------------------
    # Phase 2 — Step 7: Sybil detection
    # ------------------------------------------------------------------

    def _detect_sybils(self):
        """Find clusters where 3+ addresses trade same market, same direction,
        within <10 blocks of each other."""

        with self.progress("Detecting sybil candidates"):
            structural_dir = self.output_dir / "structural"
            temporal_dir = self.output_dir / "temporal"
            edge_glob = str(structural_dir / "edge_list.parquet" / "*.parquet")
            ca_path = str(structural_dir / "cluster_assignments.parquet")

            tmp_dir = get_tmp_dir()
            con = duckdb.connect()
            con.execute(f"SET temp_directory = '{tmp_dir}'")

            self._load_qualifying_market_ids(con)

            con.execute(f"""
                CREATE OR REPLACE VIEW edge_list AS
                SELECT * FROM read_parquet('{edge_glob}')
            """)
            con.execute(f"""
                CREATE OR REPLACE VIEW cluster_assignments AS
                SELECT * FROM read_parquet('{ca_path}')
            """)

            query = """
                WITH clustered_trades AS (
                    SELECT
                        e.address,
                        e.token_id,
                        e.direction,
                        e.block_number,
                        ca.cluster_id
                    FROM edge_list e
                    INNER JOIN cluster_assignments ca ON e.address = ca.address
                    WHERE e.token_id IN (SELECT market_id FROM qualifying)
                ),
                potential_sybil AS (
                    SELECT
                        a.cluster_id,
                        a.token_id,
                        a.direction,
                        a.block_number AS block_a,
                        b.block_number AS block_b,
                        a.address AS address_a,
                        b.address AS address_b
                    FROM clustered_trades a
                    INNER JOIN clustered_trades b
                        ON a.cluster_id = b.cluster_id
                        AND a.token_id = b.token_id
                        AND a.direction = b.direction
                        AND ABS(a.block_number - b.block_number) < 10
                        AND a.address < b.address
                )
                SELECT
                    cluster_id,
                    token_id,
                    direction,
                    COUNT(DISTINCT address_a) + COUNT(DISTINCT address_b) AS distinct_addresses,
                    MIN(block_a) AS earliest_block,
                    MAX(block_b) AS latest_block,
                    COUNT(*) AS pair_count
                FROM potential_sybil
                GROUP BY cluster_id, token_id, direction
                HAVING (COUNT(DISTINCT address_a) + COUNT(DISTINCT address_b)) >= 3
                ORDER BY pair_count DESC
            """

            result = con.execute(query).fetchdf()
            out_path = temporal_dir / "sybil_candidates.parquet"
            result.to_parquet(out_path, index=False)
            con.close()

            logger.info("Sybil candidates: %s rows", f"{len(result):,}")

    # ------------------------------------------------------------------
    # Phase 2 — Step 8: Information cascades
    # ------------------------------------------------------------------

    def _detect_cascades(self):
        """For each trade by a cluster member, check if other members in the
        same cluster trade the same token within 50 blocks afterward.

        Bounded to the top 100 most-active clusters to keep runtime reasonable.
        """

        with self.progress("Detecting information cascades"):
            structural_dir = self.output_dir / "structural"
            temporal_dir = self.output_dir / "temporal"
            edge_glob = str(structural_dir / "edge_list.parquet" / "*.parquet")
            ca_path = str(structural_dir / "cluster_assignments.parquet")

            tmp_dir = get_tmp_dir()
            con = duckdb.connect()
            con.execute(f"SET temp_directory = '{tmp_dir}'")

            self._load_qualifying_market_ids(con)

            con.execute(f"""
                CREATE OR REPLACE VIEW edge_list AS
                SELECT * FROM read_parquet('{edge_glob}')
            """)
            con.execute(f"""
                CREATE OR REPLACE VIEW cluster_assignments AS
                SELECT * FROM read_parquet('{ca_path}')
            """)

            query = """
                WITH top_clusters AS (
                    SELECT cluster_id, COUNT(*) AS member_count
                    FROM cluster_assignments
                    GROUP BY cluster_id
                    ORDER BY member_count DESC
                    LIMIT 100
                ),
                clustered_trades AS (
                    SELECT
                        e.address,
                        e.token_id,
                        e.direction,
                        e.block_number,
                        e.size,
                        ca.cluster_id
                    FROM edge_list e
                    INNER JOIN cluster_assignments ca ON e.address = ca.address
                    WHERE ca.cluster_id IN (SELECT cluster_id FROM top_clusters)
                      AND e.token_id IN (SELECT market_id FROM qualifying)
                )
                SELECT
                    a.cluster_id,
                    a.token_id,
                    a.address     AS leader_address,
                    a.direction   AS leader_direction,
                    a.block_number AS leader_block,
                    b.address     AS follower_address,
                    b.direction   AS follower_direction,
                    b.block_number AS follower_block,
                    (b.block_number - a.block_number) AS block_lag,
                    b.size        AS follower_size
                FROM clustered_trades a
                INNER JOIN clustered_trades b
                    ON a.cluster_id = b.cluster_id
                    AND a.token_id = b.token_id
                    AND a.address != b.address
                    AND b.block_number > a.block_number
                    AND b.block_number <= a.block_number + 50
                ORDER BY a.cluster_id, a.token_id, a.block_number, b.block_number
            """

            result = con.execute(query).fetchdf()
            out_path = temporal_dir / "cascade_table.parquet"
            result.to_parquet(out_path, index=False)
            con.close()

            logger.info("Cascade rows: %s", f"{len(result):,}")
