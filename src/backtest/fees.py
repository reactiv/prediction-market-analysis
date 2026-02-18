"""Fee models for prediction market platforms.

Each fee model exposes ``fee_sql(price_col)`` which returns a DuckDB SQL
expression computing the per-contract fee in cents.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KalshiFees:
    """Kalshi taker fee: min(price, 100-price) * rate, capped.

    Default: 7% of the lesser side, max 3 cents per contract.
    """

    fee_rate: float = 0.07
    cap_cents: float = 3.0

    def fee_sql(self, price_col: str = "entry_price") -> str:
        return (
            f"LEAST(LEAST({price_col}, 100.0 - {price_col}) "
            f"* {self.fee_rate}, {self.cap_cents})"
        )


@dataclass(frozen=True)
class ZeroFees:
    """No fees — for baseline comparison."""

    def fee_sql(self, price_col: str = "entry_price") -> str:
        return "0.0"
