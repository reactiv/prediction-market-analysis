"""Strategy 1: Fade-Longshot — sell YES (buy NO) on cheap contracts."""

from __future__ import annotations

from src.backtest.fees import KalshiFees
from src.backtest.strategies._base import StrategyDefinition


def fade_longshot(
    max_price: int = 15,
    min_tte_seconds: float = 0,
    categories: list[str] | None = None,
    fee_rate: float = 0.07,
) -> StrategyDefinition:
    """Create a fade-longshot strategy definition.

    Takes the NO side on every taker-YES trade priced below ``max_price``
    cents, holds to resolution.

    Args:
        max_price: Maximum yes_price in cents to qualify.
        min_tte_seconds: Minimum time-to-expiry filter.
        categories: Optional list of category prefixes to restrict to.
        fee_rate: Kalshi taker fee rate.
    """
    conditions = [
        f"entry_price < {max_price}",
        "taker_side = 'yes'",  # we fade the YES taker → take NO
    ]

    if min_tte_seconds > 0:
        conditions.append(f"time_to_expiry_s > {min_tte_seconds}")

    if categories:
        cat_list = ", ".join(f"'{c}'" for c in categories)
        conditions.append(f"category IN ({cat_list})")

    return StrategyDefinition(
        name=f"fade_longshot_lt{max_price}c",
        description=(
            f"Buy NO on taker-YES trades priced < {max_price}c, hold to resolution"
        ),
        where_clause=" AND ".join(conditions),
        take_side="no",
        fee_model=KalshiFees(fee_rate=fee_rate),
        platforms=["kalshi"],
    )
