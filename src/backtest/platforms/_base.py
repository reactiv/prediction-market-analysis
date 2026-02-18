"""Platform-agnostic resolved trade schema and adapter protocol."""

from __future__ import annotations

from typing import Protocol

import duckdb

# Canonical columns that every platform adapter must produce.
RESOLVED_TRADE_COLUMNS = [
    "trade_id",          # str: unique trade identifier
    "market_id",         # str: ticker (Kalshi) or market_id (Polymarket)
    "platform",          # str: 'kalshi' or 'polymarket'
    "timestamp",         # datetime: when the trade occurred
    "entry_price",       # float: price paid in cents (0-100 scale)
    "taker_side",        # str: 'yes' or 'no'
    "contracts",         # float: number of contracts
    "result",            # str: 'yes' or 'no' (market resolution)
    "time_to_expiry_s",  # float: seconds until market close at trade time
    "category",          # str: event category prefix (NULL if unavailable)
    "event_ticker",      # str: parent event ticker (NULL if unavailable)
]


class PlatformAdapter(Protocol):
    """Generates SQL that produces RESOLVED_TRADE_COLUMNS from platform data."""

    @property
    def platform_name(self) -> str: ...

    def resolved_trades_sql(self, con: duckdb.DuckDBPyConnection) -> str:
        """Return a SQL query producing the resolved trade columns.

        The adapter may register temporary tables/views on ``con`` before
        returning the SQL string.
        """
        ...
