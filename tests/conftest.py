"""Shared fixtures: a tiny hand-built blotter plus the real generated dataset.

The hand-built blotter is deliberately small enough that every expected flag can
be reasoned about by eye, which is what makes the rule assertions meaningful.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import TRADES_CSV  # noqa: E402
from src.data_cleaning import clean_trades, load_trades  # noqa: E402


def _row(
    trade_id: str,
    trade_date: str,
    trader: str,
    desk: str,
    product_type: str,
    counterparty: str,
    quantity: int,
    price: float,
    settlement_date: str,
    trade_status: str = "Booked",
    amendment_datetime: "str | None" = None,
    buy_sell: str = "Buy",
) -> dict:
    """Build one blotter row with the derived fields kept consistent."""
    return {
        "trade_id": trade_id,
        "trade_date": trade_date,
        "trader": trader,
        "desk": desk,
        "product_type": product_type,
        "counterparty": counterparty,
        "buy_sell": buy_sell,
        "quantity": quantity,
        "price": price,
        "notional_value": quantity * price,
        "trade_status": trade_status,
        "amendment_flag": amendment_datetime is not None,
        "cancellation_flag": trade_status == "Cancelled",
        "booking_datetime": f"{trade_date} 10:30:00",
        "amendment_datetime": amendment_datetime,
        "settlement_date": settlement_date,
    }


#: A 10-row blotter with one planted example of each pathology.
#:
#: 2026-03-02 is a Monday, so business-day arithmetic across the week is easy to
#: check by hand.
_ROWS = [
    # Three quiet Rates trades facing three different counterparties.
    _row("T00001", "2026-03-02", "Ann", "Rates", "Bond", "Alpha Bank", 1_000, 100.0, "2026-03-03"),
    _row("T00002", "2026-03-03", "Ann", "Rates", "Bond", "Beta Corp", 1_200, 100.0, "2026-03-04"),
    _row("T00003", "2026-03-04", "Ben", "Credit", "Bond", "Gamma LLP", 900, 100.0, "2026-03-05"),
    # One enormous equity trade -> large notional.
    _row("T00004", "2026-03-05", "Ben", "Equities", "Equity", "Delta Ltd", 500_000, 200.0, "2026-03-06"),
    # Amendment booked the next business day -> on time.
    _row(
        "T00005", "2026-03-02", "Cara", "FX", "FX Forward", "Alpha Bank", 1_000, 1.1,
        "2026-03-31", trade_status="Amended", amendment_datetime="2026-03-03 16:00:00",
    ),
    # Amendment booked eight business days later -> late.
    _row(
        "T00006", "2026-03-02", "Cara", "FX", "FX Forward", "Alpha Bank", 1_000, 1.1,
        "2026-03-31", trade_status="Amended", amendment_datetime="2026-03-12 16:00:00",
    ),
    # Cancellation booked six business days later -> late.
    _row(
        "T00007", "2026-03-03", "Cara", "FX", "FX Forward", "Alpha Bank", 1_000, 1.1,
        "2026-04-01", trade_status="Cancelled", amendment_datetime="2026-03-11 09:15:00",
    ),
    # Settles on the trade date -> too fast for a bond.
    _row("T00008", "2026-03-04", "Ann", "Rates", "Bond", "Alpha Bank", 800, 100.0, "2026-03-04"),
    # Settles 20 business days out -> too slow for a bond.
    _row("T00009", "2026-03-04", "Ann", "Rates", "Bond", "Alpha Bank", 700, 100.0, "2026-04-01"),
    # Ann trades far more than her peers -> volume outlier.
    _row("T00010", "2026-03-05", "Ann", "Rates", "Bond", "Alpha Bank", 600, 100.0, "2026-03-06"),
]


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """The hand-built blotter as it would come off a CSV (all strings)."""
    return pd.DataFrame(_ROWS).astype(
        {
            "quantity": "int64",
            "price": "float64",
            "notional_value": "float64",
        }
    )


@pytest.fixture
def sample_trades(raw_frame: pd.DataFrame) -> pd.DataFrame:
    """The hand-built blotter, cleaned and enriched."""
    typed = raw_frame.copy()
    for column in ("trade_date", "settlement_date"):
        typed[column] = pd.to_datetime(typed[column])
    for column in ("booking_datetime", "amendment_datetime"):
        typed[column] = pd.to_datetime(typed[column])
    return clean_trades(typed)


@pytest.fixture(scope="session")
def generated_trades() -> pd.DataFrame:
    """The committed 3,600-row simulated blotter, cleaned."""
    if not TRADES_CSV.exists():
        pytest.skip(
            f"{TRADES_CSV} not present; run `python scripts/generate_trades.py`"
        )
    return clean_trades(load_trades(TRADES_CSV))
