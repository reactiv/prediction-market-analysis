"""Kalshi platform adapter: raw trades + markets -> resolved trades.

SQL pattern follows t6_rolling_calibration.py:72-104 and
win_rate_by_price.py:37-68 exactly.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from src.analysis.kalshi.util.categories import CATEGORY_SQL


class KalshiAdapter:
    platform_name = "kalshi"

    def __init__(self, base_dir: Path):
        self.trades_dir = base_dir / "data" / "kalshi" / "trades"
        self.markets_dir = base_dir / "data" / "kalshi" / "markets"

    def resolved_trades_sql(self, con: duckdb.DuckDBPyConnection) -> str:
        return f"""
        SELECT
            t.trade_id::VARCHAR                                 AS trade_id,
            t.ticker                                            AS market_id,
            'kalshi'                                            AS platform,
            t.created_time                                      AS timestamp,
            CASE WHEN t.taker_side = 'yes' THEN t.yes_price
                 ELSE t.no_price
            END::DOUBLE                                         AS entry_price,
            t.taker_side                                        AS taker_side,
            t.count::DOUBLE                                     AS contracts,
            m.result                                            AS result,
            EPOCH(m.close_time - t.created_time)                AS time_to_expiry_s,
            {CATEGORY_SQL}                                      AS category,
            m.event_ticker                                      AS event_ticker
        FROM '{self.trades_dir}/*.parquet' t
        INNER JOIN (
            SELECT * FROM '{self.markets_dir}/*.parquet'
            WHERE status = 'finalized' AND result IN ('yes', 'no')
        ) m ON t.ticker = m.ticker
        """
