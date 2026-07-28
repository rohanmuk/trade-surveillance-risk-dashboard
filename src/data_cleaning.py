"""Loading, validation and cleaning of the simulated trade blotter.

The two public entry points are :func:`load_trades` (read + type + validate)
and :func:`clean_trades` (repair + derive). They are deliberately separate so
that a caller can inspect the raw, typed blotter before any rows are touched.

Both functions fail loudly: a missing column or an unparseable date raises a
:class:`TradeDataError` with the offending detail in the message rather than
being swallowed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Union

import numpy as np
import pandas as pd

from .config import (
    BUY_SELL,
    DESKS,
    PRODUCT_TYPES,
    TRADE_STATUSES,
)

__all__ = [
    "TradeDataError",
    "CleaningReport",
    "REQUIRED_COLUMNS",
    "load_trades",
    "clean_trades",
]


class TradeDataError(ValueError):
    """Raised when the trade blotter is missing columns or is unparseable."""


#: Columns the blotter must contain, mapped to the dtype family we enforce.
REQUIRED_COLUMNS: Dict[str, str] = {
    "trade_id": "string",
    "trade_date": "date",
    "trader": "string",
    "desk": "string",
    "product_type": "string",
    "counterparty": "string",
    "buy_sell": "string",
    "quantity": "numeric",
    "price": "numeric",
    "notional_value": "numeric",
    "trade_status": "string",
    "amendment_flag": "bool",
    "cancellation_flag": "bool",
    "booking_datetime": "datetime",
    "amendment_datetime": "datetime",
    "settlement_date": "date",
}

_DATE_COLUMNS: Sequence[str] = ("trade_date", "settlement_date")
_DATETIME_COLUMNS: Sequence[str] = ("booking_datetime", "amendment_datetime")
_NUMERIC_COLUMNS: Sequence[str] = ("quantity", "price", "notional_value")
_BOOL_COLUMNS: Sequence[str] = ("amendment_flag", "cancellation_flag")

#: Categorical columns and the canonical vocabulary they are normalised against.
_CATEGORICAL_VOCAB: Dict[str, Sequence[str]] = {
    "desk": DESKS,
    "product_type": PRODUCT_TYPES,
    "buy_sell": BUY_SELL,
    "trade_status": TRADE_STATUSES,
}

#: Rows missing any of these cannot be analysed and are dropped.
_CRITICAL_COLUMNS: Sequence[str] = ("trade_id", "trade_date", "notional_value")

#: Fraction of rows we are willing to drop before treating the file as broken.
_MAX_DROP_FRACTION: float = 0.05

#: The drop-fraction check only applies above this row count. On a handful of
#: rows a 5% rule is noise -- one bad row out of ten is 10% and says nothing
#: about the health of the file.
_MIN_ROWS_FOR_DROP_CHECK: int = 50

_TRUE_TOKENS = {"true", "t", "yes", "y", "1"}
_FALSE_TOKENS = {"false", "f", "no", "n", "0", ""}


@dataclass
class CleaningReport:
    """Record of everything :func:`clean_trades` changed.

    Attached to the cleaned frame as ``df.attrs["cleaning_report"]`` so the
    cleaning signature stays a simple ``DataFrame -> DataFrame``.

    Attributes:
        rows_in: Row count before cleaning.
        rows_out: Row count after cleaning.
        duplicate_trade_ids: Duplicate ``trade_id`` rows removed.
        dropped_missing_critical: Rows dropped for missing critical fields.
        filled_categoricals: Per-column count of blanks filled with ``UNKNOWN``.
        unrecognised_categories: Per-column values not in the canonical vocabulary.
        repaired_amendment_flags: Rows where ``amendment_flag`` disagreed with
            the presence of ``amendment_datetime`` and was corrected.
        recomputed_notionals: Rows where ``notional_value`` did not equal
            ``quantity * price`` within tolerance and was recomputed.
    """

    rows_in: int = 0
    rows_out: int = 0
    duplicate_trade_ids: int = 0
    dropped_missing_critical: int = 0
    filled_categoricals: Dict[str, int] = field(default_factory=dict)
    unrecognised_categories: Dict[str, List[str]] = field(default_factory=dict)
    repaired_amendment_flags: int = 0
    recomputed_notionals: int = 0

    def is_clean(self) -> bool:
        """Return ``True`` when nothing at all had to be repaired or dropped."""
        return (
            self.duplicate_trade_ids == 0
            and self.dropped_missing_critical == 0
            and not self.filled_categoricals
            and not self.unrecognised_categories
            and self.repaired_amendment_flags == 0
            and self.recomputed_notionals == 0
        )

    def summary_lines(self) -> List[str]:
        """Return one human-readable line per non-trivial cleaning action."""
        lines: List[str] = [f"{self.rows_in:,} rows in -> {self.rows_out:,} rows out"]
        if self.duplicate_trade_ids:
            lines.append(f"{self.duplicate_trade_ids:,} duplicate trade_id rows dropped")
        if self.dropped_missing_critical:
            lines.append(
                f"{self.dropped_missing_critical:,} rows dropped for missing "
                f"{', '.join(_CRITICAL_COLUMNS)}"
            )
        for col, n in self.filled_categoricals.items():
            lines.append(f"{n:,} blank '{col}' values filled with UNKNOWN")
        for col, values in self.unrecognised_categories.items():
            lines.append(f"'{col}' has non-standard values: {', '.join(values)}")
        if self.repaired_amendment_flags:
            lines.append(
                f"{self.repaired_amendment_flags:,} amendment_flag values "
                "reconciled with amendment_datetime"
            )
        if self.recomputed_notionals:
            lines.append(
                f"{self.recomputed_notionals:,} notional_value cells recomputed "
                "from quantity * price"
            )
        return lines


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_trades(path: Union[str, Path]) -> pd.DataFrame:
    """Read the trade blotter CSV, enforce dtypes and validate the schema.

    Args:
        path: Path to the blotter CSV.

    Returns:
        The blotter with dates parsed, numerics coerced and flags cast to bool.
        Rows are returned untouched otherwise — no repairs happen here.

    Raises:
        TradeDataError: If the file does not exist, is empty, is missing any of
            :data:`REQUIRED_COLUMNS`, or contains a date/number that cannot be
            parsed.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise TradeDataError(
            f"Trade file not found: {csv_path}. "
            "Run `python scripts/generate_trades.py` to create it."
        )

    try:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=True)
    except pd.errors.EmptyDataError as exc:
        raise TradeDataError(f"Trade file is empty: {csv_path}") from exc
    except pd.errors.ParserError as exc:
        raise TradeDataError(f"Trade file is not valid CSV: {csv_path} ({exc})") from exc

    if df.empty:
        raise TradeDataError(f"Trade file contains a header but no rows: {csv_path}")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise TradeDataError(
            f"Trade file {csv_path} is missing required column(s): "
            f"{', '.join(sorted(missing))}. Found: {', '.join(df.columns)}"
        )

    for col in (*_DATE_COLUMNS, *_DATETIME_COLUMNS):
        df[col] = _parse_datetime_column(df[col], col, csv_path)
    for col in _DATE_COLUMNS:
        df[col] = df[col].dt.normalize()

    for col in _NUMERIC_COLUMNS:
        df[col] = _parse_numeric_column(df[col], col, csv_path)

    for col in _BOOL_COLUMNS:
        df[col] = _parse_bool_column(df[col], col, csv_path)

    for col, kind in REQUIRED_COLUMNS.items():
        if kind == "string":
            df[col] = df[col].astype("object").where(df[col].notna(), None)

    return df


def _parse_datetime_column(
    series: pd.Series, column: str, path: Path
) -> pd.Series:
    """Parse a datetime column, raising with examples if any value is bad."""
    # A malformed value makes pandas fall back to per-element dateutil parsing
    # and warn about it. We turn that case into an exception two lines below,
    # so the warning is pure noise.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(series, errors="coerce")
    newly_bad = parsed.isna() & series.notna() & (series.astype(str).str.strip() != "")
    if newly_bad.any():
        examples = series[newly_bad].head(3).tolist()
        raise TradeDataError(
            f"Column '{column}' in {path} contains {int(newly_bad.sum())} "
            f"unparseable datetime value(s), e.g. {examples}"
        )
    return parsed


def _parse_numeric_column(series: pd.Series, column: str, path: Path) -> pd.Series:
    """Parse a numeric column, raising with examples if any value is bad."""
    parsed = pd.to_numeric(series, errors="coerce")
    newly_bad = parsed.isna() & series.notna() & (series.astype(str).str.strip() != "")
    if newly_bad.any():
        examples = series[newly_bad].head(3).tolist()
        raise TradeDataError(
            f"Column '{column}' in {path} contains {int(newly_bad.sum())} "
            f"non-numeric value(s), e.g. {examples}"
        )
    return parsed


def _parse_bool_column(series: pd.Series, column: str, path: Path) -> pd.Series:
    """Parse a boolean column from the usual true/false spellings."""
    tokens = series.fillna("").astype(str).str.strip().str.lower()
    unknown = sorted(set(tokens) - _TRUE_TOKENS - _FALSE_TOKENS)
    if unknown:
        raise TradeDataError(
            f"Column '{column}' in {path} contains non-boolean value(s): "
            f"{unknown[:5]}"
        )
    return tokens.isin(_TRUE_TOKENS)


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #


def clean_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Repair, normalise and enrich a loaded blotter.

    Performs, in order: duplicate ``trade_id`` removal, dropping of rows with
    missing critical fields, categorical normalisation, notional/flag
    reconciliation, and derivation of ``trade_month``, ``days_to_settlement``
    and ``days_to_amendment``.

    Args:
        df: Output of :func:`load_trades`.

    Returns:
        A new cleaned frame with a reset index. The :class:`CleaningReport` is
        attached as ``df.attrs["cleaning_report"]``.

    Raises:
        TradeDataError: If required columns are absent, or if cleaning would
            discard more than 5% of the rows (which implies a broken file
            rather than a few stray records).
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise TradeDataError(
            f"Cannot clean: missing required column(s) {', '.join(sorted(missing))}"
        )

    out = df.copy()
    report = CleaningReport(rows_in=len(out))

    # 1. Duplicate trade IDs -- keep the first booking of each.
    duplicated = out["trade_id"].duplicated(keep="first")
    report.duplicate_trade_ids = int(duplicated.sum())
    out = out.loc[~duplicated]

    # 2. Rows we cannot analyse at all.
    critical_missing = out[list(_CRITICAL_COLUMNS)].isna().any(axis=1)
    report.dropped_missing_critical = int(critical_missing.sum())
    out = out.loc[~critical_missing]

    dropped = report.duplicate_trade_ids + report.dropped_missing_critical
    if (
        report.rows_in >= _MIN_ROWS_FOR_DROP_CHECK
        and dropped / report.rows_in > _MAX_DROP_FRACTION
    ):
        raise TradeDataError(
            f"Cleaning would drop {dropped:,} of {report.rows_in:,} rows "
            f"({dropped / report.rows_in:.1%}), above the "
            f"{_MAX_DROP_FRACTION:.0%} tolerance. The source file looks corrupt."
        )


    if out.empty:
        raise TradeDataError("No usable rows remain after cleaning.")

    # 3. Categorical hygiene.
    for col, vocab in _CATEGORICAL_VOCAB.items():
        out[col], filled, unknown = _normalise_categorical(out[col], vocab)
        if filled:
            report.filled_categoricals[col] = filled
        if unknown:
            report.unrecognised_categories[col] = unknown

    for col in ("trader", "counterparty"):
        cleaned = out[col].fillna("").astype(str).str.strip().str.replace(
            r"\s+", " ", regex=True
        )
        blanks = int((cleaned == "").sum())
        if blanks:
            report.filled_categoricals[col] = blanks
        out[col] = cleaned.replace("", "UNKNOWN")

    # 4. Internal consistency.
    out["quantity"] = out["quantity"].round().astype("int64")
    expected_notional = out["quantity"] * out["price"]
    mismatch = ~np.isclose(
        out["notional_value"], expected_notional, rtol=1e-6, atol=0.01
    )
    report.recomputed_notionals = int(mismatch.sum())
    out.loc[mismatch, "notional_value"] = expected_notional[mismatch]

    has_amendment_dt = out["amendment_datetime"].notna()
    flag_mismatch = has_amendment_dt != out["amendment_flag"]
    report.repaired_amendment_flags = int(flag_mismatch.sum())
    out["amendment_flag"] = has_amendment_dt
    out["cancellation_flag"] = out["cancellation_flag"] | (
        out["trade_status"] == "Cancelled"
    )

    # 5. Derived analytics columns.
    out["trade_month"] = out["trade_date"].dt.to_period("M").astype(str)
    out["days_to_settlement"] = _business_days_between(
        out["trade_date"], out["settlement_date"]
    )
    out["days_to_amendment"] = _business_days_between(
        out["trade_date"], out["amendment_datetime"]
    )
    out["is_lifecycle_event"] = out["amendment_flag"] | out["cancellation_flag"]

    out = out.reset_index(drop=True)
    report.rows_out = len(out)
    out.attrs["cleaning_report"] = report
    return out


def _normalise_categorical(
    series: pd.Series, vocab: Iterable[str]
) -> "tuple[pd.Series, int, List[str]]":
    """Map a categorical column onto its canonical spellings.

    Matching is case-insensitive and whitespace-tolerant, so ``"fx forward"``
    and ``" FX  Forward "`` both resolve to ``"FX Forward"``. Values outside the
    vocabulary are title-cased and reported rather than dropped.

    Args:
        series: The raw column.
        vocab: Canonical values for this column.

    Returns:
        Tuple of (normalised series, count of blanks filled with ``UNKNOWN``,
        sorted list of values that were not in the vocabulary).
    """
    lookup = {v.lower(): v for v in vocab}
    stripped = (
        series.fillna("").astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    )
    filled = int((stripped == "").sum())

    def _map(value: str) -> str:
        if value == "":
            return "UNKNOWN"
        return lookup.get(value.lower(), value.title())

    mapped = stripped.map(_map)
    unknown = sorted(set(mapped.unique()) - set(vocab) - {"UNKNOWN"})
    return mapped, filled, unknown


def _business_days_between(start: pd.Series, end: pd.Series) -> pd.Series:
    """Count business days from ``start`` to ``end``, NaN where either is null.

    Uses :func:`numpy.busday_count` (Mon-Fri, no holiday calendar). The result
    is signed: a settlement date before the trade date yields a negative count,
    which is exactly the pathology the settlement-risk rule looks for.

    Args:
        start: Series of datetimes (times are ignored).
        end: Series of datetimes (times are ignored).

    Returns:
        Float series of business-day counts, aligned to ``start``'s index.
    """
    result = pd.Series(np.nan, index=start.index, dtype="float64")
    valid = start.notna() & end.notna()
    if not valid.any():
        return result

    start_days = start[valid].dt.normalize().to_numpy(dtype="datetime64[D]")
    end_days = end[valid].dt.normalize().to_numpy(dtype="datetime64[D]")
    result.loc[valid] = np.busday_count(start_days, end_days).astype("float64")
    return result
