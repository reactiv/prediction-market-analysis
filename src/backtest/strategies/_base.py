"""Base strategy definition for hold-to-resolution backtesting."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.backtest.fees import ZeroFees


@dataclass(frozen=True)
class StrategyDefinition:
    """A hold-to-resolution strategy defined as composable SQL predicates.

    Attributes:
        name: Human-readable strategy name (used in output paths).
        description: What this strategy does.
        where_clause: SQL WHERE clause filtering resolved trades.
            Available columns: trade_id, market_id, platform, timestamp,
            entry_price, taker_side, contracts, result, time_to_expiry_s,
            category, event_ticker.
        take_side: Which side to take: 'yes', 'no', or 'taker'.
            'yes'/'no' = always take that side regardless of trade direction.
            'taker' = follow the original taker direction.
        fee_model: Fee model with ``fee_sql(price_col)`` method.
        platforms: Which platforms to include.
        extra_joins: Optional SQL JOIN clauses appended after the resolved
            trades CTE (e.g. T6 calibration features).
        extra_columns: Optional SQL SELECT columns from extra joins.
    """

    name: str
    description: str
    where_clause: str
    take_side: str = "taker"
    fee_model: ZeroFees | object = field(default_factory=ZeroFees)
    platforms: list[str] = field(default_factory=lambda: ["kalshi"])
    extra_joins: str = ""
    extra_columns: str = ""
