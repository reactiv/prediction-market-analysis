"""Strategy 2: Calibration Surface — trade mispricing identified by T6.

Uses T6 rolling calibration features to select trades where the
historical mispricing (mae_7d) exceeds a threshold. Takes the NO
side on longshot taker-YES trades in high-mispricing categories.

The JOIN uses a configurable lag (default 7 days) to avoid look-ahead
bias from T6's resolution-aware win rates.
"""

from __future__ import annotations

from pathlib import Path

from src.backtest.fees import KalshiFees
from src.backtest.strategies._base import StrategyDefinition

# TTE bucket CASE expression matching T6's definition
TTE_BUCKET_SQL = """\
CASE
    WHEN rt.time_to_expiry_s < 3600 THEN '<1h'
    WHEN rt.time_to_expiry_s < 21600 THEN '1-6h'
    WHEN rt.time_to_expiry_s < 86400 THEN '6-24h'
    WHEN rt.time_to_expiry_s < 604800 THEN '1-7d'
    ELSE '7d+'
END"""


def calibration_surface(
    max_price: int = 50,
    min_mae_7d: float = 0.03,
    lag_days: int = 7,
    min_trades: int = 50,
    fee_rate: float = 0.07,
) -> StrategyDefinition:
    """Create a calibration surface strategy definition.

    Fades longshot taker-YES trades in (category, price, tte) cells
    where T6's 7-day rolling MAE exceeds ``min_mae_7d``, indicating
    persistent mispricing.

    Args:
        max_price: Maximum yes_price in cents.
        min_mae_7d: Minimum 7-day rolling MAE to qualify.
        lag_days: Days to lag T6 features (avoids look-ahead).
        min_trades: Minimum daily trade count in the T6 cell.
        fee_rate: Kalshi taker fee rate.
    """
    t6_path = str(
        Path(__file__).parent.parent.parent.parent
        / "data"
        / "transforms"
        / "t6"
        / "daily_features.parquet"
    )

    extra_joins = f"""\
INNER JOIN read_parquet('{t6_path}') cal
  ON DATE_TRUNC('day', rt.timestamp) - INTERVAL '{lag_days} day' = cal.trade_date
  AND rt.category = cal.category
  AND FLOOR(rt.entry_price / 5) * 5 = cal.price_bucket
  AND ({TTE_BUCKET_SQL}) = cal.tte_bucket
  AND rt.taker_side = cal.taker_side"""

    where_clause = (
        f"entry_price < {max_price} "
        f"AND rt.taker_side = 'yes' "
        f"AND cal.mae_7d > {min_mae_7d} "
        f"AND cal.total_trades >= {min_trades}"
    )

    extra_columns = (
        "cal.mae_7d AS cal_mae_7d, "
        "cal.calibration_error AS cal_error, "
        "cal.win_rate AS cal_win_rate, "
        "cal.opportunity_score AS cal_opportunity"
    )

    mae_label = f"{min_mae_7d * 100:.0f}pp"

    return StrategyDefinition(
        name=f"cal_surface_lt{max_price}c_mae{mae_label}",
        description=(
            f"Fade longshots in high-MAE cells "
            f"(price < {max_price}c, mae_7d > {mae_label}, "
            f"{lag_days}d lag)"
        ),
        where_clause=where_clause,
        take_side="no",
        fee_model=KalshiFees(fee_rate=fee_rate),
        platforms=["kalshi"],
        extra_joins=extra_joins,
        extra_columns=extra_columns,
    )


def calibration_sweep(
    max_price: int = 50,
    thresholds: list[float] | None = None,
    lag_days: int = 7,
    fee_rate: float = 0.07,
) -> list[StrategyDefinition]:
    """Generate strategies at multiple MAE thresholds for comparison.

    Args:
        max_price: Maximum yes_price in cents.
        thresholds: List of mae_7d thresholds. Defaults to
            0.5pp to 10pp in 0.5pp steps.
        lag_days: Days to lag T6 features.
        fee_rate: Kalshi taker fee rate.

    Returns:
        List of StrategyDefinition objects, one per threshold.
    """
    if thresholds is None:
        thresholds = [i * 0.005 for i in range(1, 21)]  # 0.5pp to 10pp

    return [
        calibration_surface(
            max_price=max_price,
            min_mae_7d=t,
            lag_days=lag_days,
            fee_rate=fee_rate,
        )
        for t in thresholds
    ]
