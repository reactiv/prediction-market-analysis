from __future__ import annotations

import duckdb

from src.transforms._base import Transform
from src.transforms._util import copy_to_parquet

ACTIVE_THRESHOLD = 1.0  # trades per hour


class T8Lifecycle(Transform):
    def __init__(self):
        super().__init__(
            name="t8",
            description="Market lifecycle state machine",
            dependencies=["t1a"],
        )

    def run(self):
        self.ensure_output_dir()
        kalshi_dir = self.output_dir / "kalshi"
        kalshi_dir.mkdir(parents=True, exist_ok=True)

        timeline_rows = self._build_state_timeline()
        self._build_per_state_stats()
        self._build_anomalous_transitions()
        self._build_duration_distributions()

        self.write_manifest({
            "timeline_rows": timeline_rows,
        })

    def _build_state_timeline(self) -> int:
        """Build state timeline: assign lifecycle state to every trade."""
        con = duckdb.connect()

        t1a_path = str(self.base_dir / "data" / "transforms" / "t1a" / "kalshi" / "*.parquet")
        markets_path = str(self.base_dir / "data" / "kalshi" / "markets" / "*.parquet")
        output_dir = str(self.output_dir / "kalshi" / "state_timeline.parquet")

        active = ACTIVE_THRESHOLD

        query = f"""
        WITH trades_with_rate AS (
            SELECT
                t.ticker,
                t.created_time,
                t.trade_sequence_num,
                t.norm_price,
                t.signed_flow,
                t.cumulative_volume,
                t.cumulative_trade_count,
                t.time_since_prev,
                t.time_to_expiry_seconds,
                t.count,
                m.status,
                m.open_time,
                COUNT(*) OVER (
                    PARTITION BY t.ticker ORDER BY t.created_time
                    RANGE BETWEEN INTERVAL '6 hours' PRECEDING AND CURRENT ROW
                ) / 6.0 AS trailing_rate_6h
            FROM read_parquet('{t1a_path}') t
            LEFT JOIN read_parquet('{markets_path}') m
                ON t.ticker = m.ticker
        ),
        with_state AS (
            SELECT
                ticker,
                created_time,
                trade_sequence_num,
                norm_price,
                signed_flow,
                cumulative_volume,
                cumulative_trade_count,
                time_since_prev,
                count,
                CASE
                    WHEN status = 'finalized'
                        THEN 'SETTLED'
                    WHEN (norm_price > 90 OR norm_price < 10)
                         AND trailing_rate_6h > {active} * 2
                        THEN 'RESOLVING'
                    WHEN time_to_expiry_seconds < 86400
                        THEN 'APPROACHING_EXPIRY'
                    WHEN trailing_rate_6h > {active} * 5
                        THEN 'HIGH_ACTIVITY'
                    WHEN trailing_rate_6h >= {active}
                         AND cumulative_volume >= 10
                        THEN 'ACTIVE'
                    WHEN time_since_prev > 86400
                         AND cumulative_trade_count > 5
                        THEN 'DORMANT'
                    WHEN cumulative_trade_count <= 5
                         AND EPOCH(created_time - open_time) < 86400
                        THEN 'NEWLY_LISTED'
                    ELSE 'EARLY_TRADING'
                END AS state
            FROM trades_with_rate
        ),
        with_transitions AS (
            SELECT
                ticker,
                created_time,
                trade_sequence_num,
                norm_price,
                state,
                LAG(state) OVER (
                    PARTITION BY ticker ORDER BY created_time
                ) AS prev_state
            FROM with_state
        )
        SELECT
            ticker,
            created_time,
            trade_sequence_num,
            norm_price,
            state,
            prev_state,
            CASE
                WHEN prev_state IS NULL THEN FALSE
                WHEN state != prev_state THEN TRUE
                ELSE FALSE
            END AS is_transition
        FROM with_transitions
        ORDER BY ticker, created_time
        """

        # Create a persistent temp view for downstream queries
        con.execute(f"""
        CREATE OR REPLACE VIEW state_timeline AS {query}
        """)

        with self.progress("Writing state timeline"):
            row_count = copy_to_parquet(con, "SELECT * FROM state_timeline", output_dir)

        # Store connection for reuse in subsequent methods
        self._con = con
        return row_count

    def _build_per_state_stats(self) -> int:
        """Aggregate statistics per (ticker, state) pair."""
        con = self._con
        output_dir = str(self.output_dir / "per_state_stats.parquet")

        t1a_path = str(self.base_dir / "data" / "transforms" / "t1a" / "kalshi" / "*.parquet")
        timeline_path = str(self.output_dir / "kalshi" / "state_timeline.parquet" / "*.parquet")

        query = f"""
        WITH timeline AS (
            SELECT * FROM read_parquet('{timeline_path}')
        ),
        joined AS (
            SELECT
                tl.ticker,
                tl.state,
                tl.created_time,
                tl.norm_price,
                t.signed_flow,
                t.count
            FROM timeline tl
            INNER JOIN read_parquet('{t1a_path}') t
                ON tl.ticker = t.ticker
                AND tl.trade_sequence_num = t.trade_sequence_num
        )
        SELECT
            ticker,
            state,
            COUNT(*) AS trade_count,
            AVG(norm_price) AS avg_price,
            STDDEV(norm_price) AS price_std,
            AVG(signed_flow) AS avg_signed_flow,
            SUM(count) AS total_volume,
            EPOCH(MAX(created_time) - MIN(created_time)) AS duration_seconds,
            MIN(created_time) AS first_entry,
            MAX(created_time) AS last_entry
        FROM joined
        GROUP BY ticker, state
        ORDER BY ticker, first_entry
        """

        with self.progress("Writing per-state stats"):
            row_count = copy_to_parquet(con, query, output_dir)

        return row_count

    def _build_anomalous_transitions(self) -> int:
        """Flag transitions that skip expected lifecycle progression."""
        con = self._con
        output_dir = str(self.output_dir / "anomalous_transitions.parquet")

        timeline_path = str(self.output_dir / "kalshi" / "state_timeline.parquet" / "*.parquet")

        query = f"""
        WITH transitions AS (
            SELECT
                ticker,
                created_time,
                prev_state AS from_state,
                state AS to_state,
                norm_price,
                trade_sequence_num
            FROM read_parquet('{timeline_path}')
            WHERE is_transition = TRUE
              AND prev_state IS NOT NULL
        )
        SELECT
            ticker,
            created_time,
            from_state,
            to_state,
            norm_price,
            trade_sequence_num,
            CASE
                WHEN from_state = 'EARLY_TRADING' AND to_state = 'RESOLVING'
                    THEN 'information_event'
                WHEN from_state = 'NEWLY_LISTED' AND to_state = 'HIGH_ACTIVITY'
                    THEN 'bot_or_manipulation'
                WHEN from_state = 'DORMANT' AND to_state = 'RESOLVING'
                    THEN 'sudden_resolution'
                WHEN from_state = 'ACTIVE' AND to_state = 'NEWLY_LISTED'
                    THEN 'impossible_regression'
            END AS transition_type
        FROM transitions
        WHERE (from_state = 'EARLY_TRADING' AND to_state = 'RESOLVING')
           OR (from_state = 'NEWLY_LISTED' AND to_state = 'HIGH_ACTIVITY')
           OR (from_state = 'DORMANT' AND to_state = 'RESOLVING')
           OR (from_state = 'ACTIVE' AND to_state = 'NEWLY_LISTED')
        ORDER BY ticker, created_time
        """

        with self.progress("Writing anomalous transitions"):
            row_count = copy_to_parquet(con, query, output_dir)

        return row_count

    def _build_duration_distributions(self) -> int:
        """Compute how long markets spend in each state, then aggregate distributions."""
        con = self._con
        output_dir = str(self.output_dir / "state_duration_distributions.parquet")

        timeline_path = str(self.output_dir / "kalshi" / "state_timeline.parquet" / "*.parquet")

        query = f"""
        WITH timeline AS (
            SELECT
                ticker,
                created_time,
                state,
                is_transition,
                prev_state
            FROM read_parquet('{timeline_path}')
        ),
        -- Identify the start of each state stint by marking transitions
        stint_starts AS (
            SELECT
                ticker,
                created_time,
                state,
                SUM(CASE WHEN is_transition OR prev_state IS NULL THEN 1 ELSE 0 END) OVER (
                    PARTITION BY ticker ORDER BY created_time
                    ROWS UNBOUNDED PRECEDING
                ) AS stint_id
            FROM timeline
        ),
        -- Compute duration of each stint
        stint_durations AS (
            SELECT
                ticker,
                state,
                stint_id,
                EPOCH(MAX(created_time) - MIN(created_time)) / 3600.0 AS duration_hours
            FROM stint_starts
            GROUP BY ticker, state, stint_id
        )
        SELECT
            state,
            QUANTILE_CONT(duration_hours, 0.25) AS p25_duration_hours,
            QUANTILE_CONT(duration_hours, 0.50) AS median_duration_hours,
            QUANTILE_CONT(duration_hours, 0.75) AS p75_duration_hours,
            AVG(duration_hours) AS mean_duration_hours,
            COUNT(*) AS count
        FROM stint_durations
        GROUP BY state
        ORDER BY state
        """

        with self.progress("Writing state duration distributions"):
            row_count = copy_to_parquet(con, query, output_dir)

        return row_count
