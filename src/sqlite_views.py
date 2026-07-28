"""SQL-backed aggregates over the flagged blotter.

The dashboard is a pandas application, so why SQL at all? Because the queries
below are the ones a risk or ops analyst would actually be handed as a saved
view in a surveillance warehouse. Loading the in-memory frame into SQLite and
running real SQL against it keeps those definitions in the language they would
live in downstream, and lets the app show the analyst the exact query behind a
headline number.

The database is in-memory and rebuilt from the current (already filtered)
DataFrame, so the SQL always reflects what is on screen.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Dict, Iterator, Optional

import pandas as pd

from .config import FLAG_COLUMNS

__all__ = [
    "TRADE_TABLE",
    "QUERIES",
    "trade_connection",
    "run_query",
    "desk_exception_league_table",
    "counterparty_exposure_view",
    "monthly_exception_trend",
]

TRADE_TABLE: str = "trades"

#: Columns loaded into SQLite. Kept explicit so the schema is stable regardless
#: of any extra helper columns the caller may have added.
_SQL_COLUMNS = (
    "trade_id",
    "trade_date",
    "trade_month",
    "trader",
    "desk",
    "product_type",
    "counterparty",
    "buy_sell",
    "quantity",
    "price",
    "notional_value",
    "trade_status",
    "days_to_settlement",
    "days_to_amendment",
    *FLAG_COLUMNS,
    "exception_count",
)


#: Named, human-readable queries the app can display alongside their results.
QUERIES: Dict[str, str] = {
    "desk_exception_league_table": f"""
        SELECT
            desk,
            COUNT(*)                                        AS trades,
            ROUND(SUM(notional_value) / 1e6, 1)             AS notional_millions,
            SUM(CASE WHEN exception_count > 0 THEN 1 ELSE 0 END)
                                                            AS flagged_trades,
            SUM(exception_count)                            AS total_exceptions,
            ROUND(
                100.0 * SUM(CASE WHEN exception_count > 0 THEN 1 ELSE 0 END)
                / COUNT(*), 1
            )                                               AS exception_rate_pct,
            SUM(flag_large_notional)                        AS large_notional,
            SUM(flag_late_amendment)                        AS late_amendments,
            SUM(flag_settlement_risk)                       AS settlement_breaches
        FROM {TRADE_TABLE}
        GROUP BY desk
        ORDER BY total_exceptions DESC
    """,
    "counterparty_exposure_view": f"""
        SELECT
            counterparty,
            COUNT(*)                            AS trades,
            ROUND(SUM(notional_value) / 1e6, 1) AS notional_millions,
            ROUND(
                100.0 * SUM(notional_value)
                / (SELECT SUM(notional_value) FROM {TRADE_TABLE}), 2
            )                                   AS pct_of_book,
            COUNT(DISTINCT desk)                AS desks_facing,
            SUM(exception_count)                AS total_exceptions
        FROM {TRADE_TABLE}
        GROUP BY counterparty
        HAVING SUM(notional_value) > 0
        ORDER BY SUM(notional_value) DESC
        LIMIT :limit
    """,
    "monthly_exception_trend": f"""
        SELECT
            trade_month,
            COUNT(*)                                        AS trades,
            SUM(CASE WHEN exception_count > 0 THEN 1 ELSE 0 END)
                                                            AS flagged_trades,
            ROUND(
                100.0 * SUM(CASE WHEN exception_count > 0 THEN 1 ELSE 0 END)
                / COUNT(*), 1
            )                                               AS exception_rate_pct,
            ROUND(SUM(notional_value) / 1e6, 1)             AS notional_millions
        FROM {TRADE_TABLE}
        GROUP BY trade_month
        ORDER BY trade_month
    """,
}


@contextmanager
def trade_connection(df: pd.DataFrame) -> Iterator[sqlite3.Connection]:
    """Yield an in-memory SQLite connection holding the blotter.

    Args:
        df: Flagged blotter — the output of
            :func:`src.surveillance_rules.run_all_rules`.

    Yields:
        A connection with a ``trades`` table and ``sqlite3.Row`` row factory.
        The database is discarded on exit.

    Raises:
        KeyError: If the frame is missing columns the queries depend on.
    """
    missing = [c for c in _SQL_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            "Cannot build the SQLite view; the frame is missing "
            f"{', '.join(missing)}. Run run_all_rules() first."
        )

    payload = df.loc[:, list(_SQL_COLUMNS)].copy()
    payload["trade_date"] = payload["trade_date"].dt.strftime("%Y-%m-%d")
    for column in FLAG_COLUMNS:
        payload[column] = payload[column].astype(int)

    connection = sqlite3.connect(":memory:")
    try:
        connection.row_factory = sqlite3.Row
        payload.to_sql(TRADE_TABLE, connection, index=False, if_exists="replace")
        connection.execute(
            f"CREATE INDEX idx_trades_desk ON {TRADE_TABLE}(desk)"
        )
        connection.execute(
            f"CREATE INDEX idx_trades_cp ON {TRADE_TABLE}(counterparty)"
        )
        yield connection
    finally:
        connection.close()


def run_query(
    df: pd.DataFrame, name: str, params: Optional[Dict[str, object]] = None
) -> pd.DataFrame:
    """Execute one of the named :data:`QUERIES` against the blotter.

    Args:
        df: Flagged blotter.
        name: Key into :data:`QUERIES`.
        params: Bound parameters for the query, e.g. ``{"limit": 10}``.

    Returns:
        The query result as a DataFrame.

    Raises:
        KeyError: If ``name`` is not a known query.
    """
    if name not in QUERIES:
        raise KeyError(
            f"Unknown query {name!r}. Available: {', '.join(sorted(QUERIES))}"
        )
    with trade_connection(df) as connection:
        return pd.read_sql_query(QUERIES[name], connection, params=params or {})


def desk_exception_league_table(df: pd.DataFrame) -> pd.DataFrame:
    """Exception counts and rates per desk, computed in SQL.

    Args:
        df: Flagged blotter.

    Returns:
        One row per desk, ordered by total exceptions descending.
    """
    return run_query(df, "desk_exception_league_table")


def counterparty_exposure_view(df: pd.DataFrame, limit: int = 25) -> pd.DataFrame:
    """Counterparty exposure and share of book, computed in SQL.

    Args:
        df: Flagged blotter.
        limit: Maximum number of counterparties to return.

    Returns:
        One row per counterparty, ordered by gross notional descending.
    """
    return run_query(df, "counterparty_exposure_view", {"limit": limit})


def monthly_exception_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Month-by-month trade count, flagged count and exception rate, in SQL.

    Args:
        df: Flagged blotter.

    Returns:
        One row per ``trade_month``, in chronological order.
    """
    return run_query(df, "monthly_exception_trend")
