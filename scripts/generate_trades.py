"""Generate the simulated trade blotter at ``data/simulated_trades.csv``.

Everything here is synthetic — trader names and counterparty names come from
Faker, and no real client, counterparty or employee data is involved.

The generator deliberately plants the patterns the surveillance rules are meant
to catch, so the dashboard has something to show out of the box:

* a skewed counterparty distribution (share ~ ``1 / rank ** 0.6``), so a
  handful of names dominate gross notional;
* skewed trader and desk activity, so at least one trader and one desk sit
  outside their peer group;
* a ~1.5% tail of trades sized 10-50x normal, for the large-notional rule;
* ~15% of lifecycle events booked several business days late;
* a small number of trades settling before/at the trade date or far beyond
  convention.

Run with::

    python scripts/generate_trades.py [--rows 3600] [--seed 20260728]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from faker import Faker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402  (path setup must precede the import)
    BUY_SELL,
    DESK_PRODUCT_MIX,
    DESKS,
    PRODUCT_SETTLEMENT_LAG,
    RANDOM_SEED,
    TRADES_CSV,
)

# --------------------------------------------------------------------------- #
# Generation parameters
# --------------------------------------------------------------------------- #

DEFAULT_ROWS: int = 3_600
TRADE_WINDOW: Tuple[str, str] = ("2026-01-01", "2026-06-30")

N_TRADERS: int = 15
N_COUNTERPARTIES: int = 25

#: Relative desk activity. Deliberately uneven so the desk-level volume rule
#: has a genuine outlier (Rates sits ~1.4 sigma above the five-desk mean).
DESK_WEIGHTS: Dict[str, float] = {
    "Rates": 0.32,
    "FX": 0.24,
    "Equities": 0.20,
    "Credit": 0.14,
    "Commodities": 0.10,
}

#: Within a desk, the first trader listed is the desk head and trades most.
#: Two traders overall are boosted hard enough to clear a 2.0 z-score.
TRADER_BOOSTS: Tuple[float, ...] = (3.1, 2.3, 1.0)

#: Counterparty share decays as 1 / rank ** ALPHA. 0.6 puts the largest name at
#: roughly 13% of trades, which is a plausible bank-scale concentration.
COUNTERPARTY_ALPHA: float = 0.6

#: (median quantity, quantity log-sigma, price low, price high, price decimals)
#: Stylised but product-appropriate: bond quantities are face-value units at a
#: clean price near par, FX forwards are notional in currency units at a rate
#: near 1, equities are shares at a share price.
PRODUCT_PRICING: Dict[str, Tuple[float, float, float, float, int]] = {
    "Bond": (25_000, 0.85, 88.0, 112.0, 3),
    "Swap": (12, 0.90, 250_000.0, 1_000_000.0, 2),
    "FX Forward": (2_000_000, 0.80, 0.78, 1.62, 4),
    "Equity": (5_000, 0.95, 12.0, 480.0, 2),
    "Option": (900, 0.90, 45.0, 1_250.0, 2),
}

STATUS_PROBABILITIES: Dict[str, float] = {
    "Booked": 0.88,
    "Amended": 0.09,
    "Cancelled": 0.03,
}

OUTLIER_TRADE_FRACTION: float = 0.015
OUTLIER_SIZE_RANGE: Tuple[float, float] = (10.0, 50.0)

LATE_EVENT_FRACTION: float = 0.15
LATE_EVENT_LAG_DAYS: Tuple[int, int] = (3, 15)

SETTLEMENT_ANOMALY_FRACTION: float = 0.012


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _unique_names(faker: Faker, n: int, kind: str) -> List[str]:
    """Draw ``n`` distinct fake names of the given kind ("person"/"company")."""
    names: List[str] = []
    seen = set()
    guard = 0
    while len(names) < n:
        guard += 1
        if guard > n * 200:
            raise RuntimeError(f"Could not draw {n} unique {kind} names from Faker")
        candidate = faker.name() if kind == "person" else faker.company()
        candidate = candidate.replace(",", "")
        if candidate not in seen:
            seen.add(candidate)
            names.append(candidate)
    return names


def _build_trader_book(faker: Faker) -> pd.DataFrame:
    """Assign fake traders to desks with uneven within-desk activity weights.

    Returns:
        DataFrame with columns ``trader``, ``desk``, ``weight`` where ``weight``
        is the unnormalised probability of a given trade belonging to that
        trader.
    """
    names = _unique_names(faker, N_TRADERS, "person")
    # Three traders per desk across five desks.
    per_desk = N_TRADERS // len(DESKS)
    rows = []
    for desk_idx, desk in enumerate(DESKS):
        desk_traders = names[desk_idx * per_desk : (desk_idx + 1) * per_desk]
        for seat, trader in enumerate(desk_traders):
            boost = TRADER_BOOSTS[min(seat, len(TRADER_BOOSTS) - 1)]
            rows.append(
                {
                    "trader": trader,
                    "desk": desk,
                    "weight": DESK_WEIGHTS[desk] * boost,
                }
            )
    return pd.DataFrame(rows)


def _counterparty_weights(n: int, alpha: float) -> np.ndarray:
    """Return a normalised, deliberately skewed counterparty probability vector."""
    ranks = np.arange(1, n + 1, dtype=float)
    weights = 1.0 / ranks**alpha
    return weights / weights.sum()


def _business_days(start: str, end: str) -> np.ndarray:
    """All Mon-Fri dates in ``[start, end]`` as ``datetime64[D]``."""
    days = pd.bdate_range(start=start, end=end)
    return days.to_numpy(dtype="datetime64[D]")


def _add_business_days(dates: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Roll each date forward (or back) by its business-day offset."""
    return np.busday_offset(dates, offsets, roll="forward")


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


def generate_trades(
    n_rows: int = DEFAULT_ROWS, seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """Build the synthetic blotter.

    Args:
        n_rows: Number of trades to generate. Must be at least 100 for the
            statistical patterns to be meaningful.
        seed: Random seed. The same seed always produces the same blotter.

    Returns:
        DataFrame with the full blotter schema, sorted by ``booking_datetime``.

    Raises:
        ValueError: If ``n_rows`` is below 100.
    """
    if n_rows < 100:
        raise ValueError(f"n_rows must be at least 100, got {n_rows}")

    rng = np.random.default_rng(seed)
    faker = Faker("en_US")
    Faker.seed(seed)

    traders = _build_trader_book(faker)
    counterparties = _unique_names(faker, N_COUNTERPARTIES, "company")
    cp_weights = _counterparty_weights(N_COUNTERPARTIES, COUNTERPARTY_ALPHA)

    # --- who traded, and where -------------------------------------------- #
    trader_probs = traders["weight"].to_numpy() / traders["weight"].sum()
    trader_idx = rng.choice(len(traders), size=n_rows, p=trader_probs)
    trader = traders["trader"].to_numpy()[trader_idx]
    desk = traders["desk"].to_numpy()[trader_idx]

    product = np.empty(n_rows, dtype=object)
    for desk_name, mix in DESK_PRODUCT_MIX.items():
        mask = desk == desk_name
        if mask.any():
            product[mask] = rng.choice(
                list(mix.keys()), size=int(mask.sum()), p=list(mix.values())
            )

    counterparty = rng.choice(counterparties, size=n_rows, p=cp_weights)
    buy_sell = rng.choice(list(BUY_SELL), size=n_rows, p=[0.52, 0.48])

    # --- when -------------------------------------------------------------- #
    calendar = _business_days(*TRADE_WINDOW)
    # Slight upward drift in activity over the window, plus month-end bunching.
    day_weights = np.linspace(0.85, 1.15, len(calendar))
    trade_date = rng.choice(calendar, size=n_rows, p=day_weights / day_weights.sum())

    # Booking time: mostly in the trading day, with a tail into the evening.
    booking_hour = np.clip(rng.normal(13.0, 2.6, size=n_rows), 7.0, 22.5)
    booking_minutes = (booking_hour * 60).astype("int64")
    booking_datetime = trade_date.astype("datetime64[m]") + booking_minutes.astype(
        "timedelta64[m]"
    )

    # --- size -------------------------------------------------------------- #
    quantity = np.zeros(n_rows, dtype=float)
    price = np.zeros(n_rows, dtype=float)
    for product_name, (
        median_qty,
        sigma,
        px_lo,
        px_hi,
        px_dp,
    ) in PRODUCT_PRICING.items():
        mask = product == product_name
        count = int(mask.sum())
        if not count:
            continue
        quantity[mask] = median_qty * rng.lognormal(mean=0.0, sigma=sigma, size=count)
        price[mask] = np.round(rng.uniform(px_lo, px_hi, size=count), px_dp)

    # Inject the fat tail of oversized trades.
    n_outliers = max(1, int(round(n_rows * OUTLIER_TRADE_FRACTION)))
    outlier_idx = rng.choice(n_rows, size=n_outliers, replace=False)
    quantity[outlier_idx] *= rng.uniform(*OUTLIER_SIZE_RANGE, size=n_outliers)

    quantity = np.maximum(np.round(quantity), 1.0)
    notional = np.round(quantity * price, 2)

    # --- lifecycle --------------------------------------------------------- #
    trade_status = rng.choice(
        list(STATUS_PROBABILITIES.keys()),
        size=n_rows,
        p=list(STATUS_PROBABILITIES.values()),
    )
    has_event = trade_status != "Booked"
    cancellation_flag = trade_status == "Cancelled"
    # A cancellation is a post-booking modification, so it carries the amendment
    # flag too; `trade_status` is what distinguishes the two event types.
    amendment_flag = has_event

    n_events = int(has_event.sum())
    is_late = rng.random(n_events) < LATE_EVENT_FRACTION
    event_lag_days = np.where(
        is_late,
        rng.integers(LATE_EVENT_LAG_DAYS[0], LATE_EVENT_LAG_DAYS[1] + 1, size=n_events),
        rng.choice([0, 1, 2], size=n_events, p=[0.55, 0.32, 0.13]),
    )
    event_dates = _add_business_days(
        trade_date[has_event], event_lag_days.astype(int)
    )
    event_minutes = (
        np.clip(rng.normal(15.0, 3.0, size=n_events), 7.5, 23.0) * 60
    ).astype("int64")
    event_datetime_values = event_dates.astype("datetime64[m]") + event_minutes.astype(
        "timedelta64[m]"
    )

    amendment_datetime = np.full(n_rows, np.datetime64("NaT"), dtype="datetime64[m]")
    amendment_datetime[has_event] = event_datetime_values
    # An event can never precede its own booking.
    clash = has_event & (amendment_datetime < booking_datetime)
    amendment_datetime[clash] = booking_datetime[clash] + np.timedelta64(45, "m")

    # --- settlement -------------------------------------------------------- #
    settlement_offset = np.zeros(n_rows, dtype=int)
    for product_name, lags in PRODUCT_SETTLEMENT_LAG.items():
        mask = product == product_name
        count = int(mask.sum())
        if count:
            settlement_offset[mask] = rng.choice(list(lags), size=count)

    n_anomalies = max(2, int(round(n_rows * SETTLEMENT_ANOMALY_FRACTION)))
    anomaly_idx = rng.choice(n_rows, size=n_anomalies, replace=False)
    fast_idx = anomaly_idx[: n_anomalies // 2]
    slow_idx = anomaly_idx[n_anomalies // 2 :]
    # Too fast: same day, or a date keyed in before the trade date.
    settlement_offset[fast_idx] = rng.choice([-1, 0], size=len(fast_idx), p=[0.35, 0.65])
    # Too slow: well beyond the product's convention.
    settlement_offset[slow_idx] = settlement_offset[slow_idx] + rng.integers(
        15, 45, size=len(slow_idx)
    )

    settlement_date = _add_business_days(trade_date, settlement_offset)

    blotter = pd.DataFrame(
        {
            "trade_id": [f"T{i:05d}" for i in range(1, n_rows + 1)],
            "trade_date": pd.to_datetime(trade_date),
            "trader": trader,
            "desk": desk,
            "product_type": product,
            "counterparty": counterparty,
            "buy_sell": buy_sell,
            "quantity": quantity.astype("int64"),
            "price": price,
            "notional_value": notional,
            "trade_status": trade_status,
            "amendment_flag": amendment_flag,
            "cancellation_flag": cancellation_flag,
            "booking_datetime": pd.to_datetime(booking_datetime),
            "amendment_datetime": pd.to_datetime(amendment_datetime),
            "settlement_date": pd.to_datetime(settlement_date),
        }
    )

    blotter = blotter.sort_values("booking_datetime", ignore_index=True)
    # Re-issue IDs so they run in booking order, as a real blotter would.
    blotter["trade_id"] = [f"T{i:05d}" for i in range(1, len(blotter) + 1)]
    return blotter


def write_trades(df: pd.DataFrame, path: Path = TRADES_CSV) -> Path:
    """Write the blotter to CSV, creating the parent directory if needed.

    Args:
        df: Blotter to write.
        path: Destination CSV path.

    Returns:
        The path written to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    return path


def _describe(df: pd.DataFrame) -> str:
    """Return a short console summary of the generated blotter."""
    cp = df.groupby("counterparty")["notional_value"].sum().sort_values(ascending=False)
    top_share = cp.iloc[0] / cp.sum() * 100
    lines = [
        f"rows                : {len(df):,}",
        f"date range          : {df['trade_date'].min():%Y-%m-%d} -> "
        f"{df['trade_date'].max():%Y-%m-%d}",
        f"gross notional      : {df['notional_value'].sum():,.0f}",
        f"traders / desks     : {df['trader'].nunique()} / {df['desk'].nunique()}",
        f"counterparties      : {df['counterparty'].nunique()} "
        f"(largest = {top_share:.1f}% of notional)",
        f"amended / cancelled : {int(df['amendment_flag'].sum()):,} / "
        f"{int(df['cancellation_flag'].sum()):,}",
    ]
    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--rows", type=int, default=DEFAULT_ROWS, help="number of trades to generate"
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED, help="random seed for reproducibility"
    )
    parser.add_argument(
        "--out", type=Path, default=TRADES_CSV, help="output CSV path"
    )
    args = parser.parse_args()

    blotter = generate_trades(n_rows=args.rows, seed=args.seed)
    out_path = write_trades(blotter, args.out)
    print(_describe(blotter))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
