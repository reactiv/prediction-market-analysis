"""Shared utilities for data transforms."""

from __future__ import annotations

from pathlib import Path

import duckdb


def polymarket_price_sql(alias: str = "t") -> str:
    """SQL expression for normalized Polymarket price (0-1 scale).

    When maker_asset_id == '0', maker is selling USDC for tokens:
        price = maker_amount / taker_amount (USDC per token)
    Otherwise, taker is selling USDC for tokens:
        price = taker_amount / maker_amount
    """
    return f"""
        CASE
            WHEN {alias}.maker_asset_id = '0'
            THEN {alias}.maker_amount::DOUBLE / NULLIF({alias}.taker_amount::DOUBLE, 0)
            ELSE {alias}.taker_amount::DOUBLE / NULLIF({alias}.maker_amount::DOUBLE, 0)
        END
    """


def polymarket_token_id_sql(alias: str = "t") -> str:
    """SQL expression for the outcome token ID in a Polymarket trade."""
    return f"""
        CASE
            WHEN {alias}.maker_asset_id = '0' THEN {alias}.taker_asset_id
            ELSE {alias}.maker_asset_id
        END
    """


def polymarket_volume_sql(alias: str = "t") -> str:
    """SQL expression for trade volume in USDC terms."""
    return f"""
        CASE
            WHEN {alias}.maker_asset_id = '0' THEN {alias}.maker_amount::DOUBLE
            ELSE {alias}.taker_amount::DOUBLE
        END
    """


def polymarket_signed_flow_sql(alias: str = "t") -> str:
    """SQL expression for signed order flow.

    Positive = buying tokens (bullish), negative = selling tokens (bearish).
    """
    return f"""
        CASE
            WHEN {alias}.maker_asset_id = '0'
            THEN {alias}.maker_amount::DOUBLE / NULLIF({alias}.taker_amount::DOUBLE, 0)
            ELSE -({alias}.taker_amount::DOUBLE / NULLIF({alias}.maker_amount::DOUBLE, 0))
        END
    """


def copy_to_parquet(
    con: duckdb.DuckDBPyConnection,
    query: str,
    output_dir: Path | str,
) -> int:
    """Write DuckDB query results to parquet files using native COPY TO.

    Uses DuckDB's PER_THREAD_OUTPUT for parallel writing. Returns row count.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"COPY ({query}) TO '{output_dir}' (FORMAT PARQUET, PER_THREAD_OUTPUT true, OVERWRITE true)"
    )
    count = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{output_dir}/*.parquet')"
    ).fetchone()[0]
    return count


def chunked_query_to_parquet(
    con: duckdb.DuckDBPyConnection,
    query: str,
    output_dir: Path,
    prefix: str = "part",
    chunk_size: int = 500_000,
) -> int:
    """Stream DuckDB query results to numbered parquet files via fetchmany.

    Use this when you need Python-side processing or when COPY TO isn't suitable.
    Returns total row count written.
    """
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = con.execute(query)
    columns = [desc[0] for desc in result.description]

    total_rows = 0
    chunk_idx = 0

    while True:
        rows = result.fetchmany(chunk_size)
        if not rows:
            break

        df = pd.DataFrame(rows, columns=columns)
        output_path = output_dir / f"{prefix}_{chunk_idx:04d}.parquet"
        df.to_parquet(output_path, index=False)
        total_rows += len(df)
        chunk_idx += 1

    return total_rows


def get_base_dir() -> Path:
    """Get the project base directory."""
    return Path(__file__).parent.parent.parent


def get_data_dir() -> Path:
    """Get the data directory."""
    return get_base_dir() / "data"


def get_tmp_dir() -> Path:
    """Get (and create) the temp directory for DuckDB spill."""
    tmp = get_data_dir() / "transforms" / ".tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    return tmp
