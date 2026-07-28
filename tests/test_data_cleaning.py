"""Unit tests for loading, validation and cleaning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import TRADES_CSV
from src.data_cleaning import (
    REQUIRED_COLUMNS,
    TradeDataError,
    clean_trades,
    load_trades,
)


class TestLoadTrades:
    def test_loads_the_committed_dataset_with_the_right_dtypes(self):
        df = load_trades(TRADES_CSV)
        assert len(df) >= 3_000
        assert set(REQUIRED_COLUMNS) <= set(df.columns)
        assert pd.api.types.is_datetime64_any_dtype(df["trade_date"])
        assert pd.api.types.is_datetime64_any_dtype(df["booking_datetime"])
        assert pd.api.types.is_numeric_dtype(df["notional_value"])
        assert df["amendment_flag"].dtype == bool

    def test_missing_file_raises_with_a_hint(self, tmp_path):
        with pytest.raises(TradeDataError, match="generate_trades"):
            load_trades(tmp_path / "nope.csv")

    def test_missing_column_names_the_offender(self, tmp_path):
        df = load_trades(TRADES_CSV).head(20).drop(columns=["counterparty"])
        path = tmp_path / "partial.csv"
        df.to_csv(path, index=False)
        with pytest.raises(TradeDataError, match="counterparty"):
            load_trades(path)

    def test_unparseable_date_raises_rather_than_coercing(self, tmp_path):
        df = load_trades(TRADES_CSV).head(20)
        df["settlement_date"] = df["settlement_date"].astype(object)
        df.loc[df.index[0], "settlement_date"] = "not-a-date"
        path = tmp_path / "baddate.csv"
        df.to_csv(path, index=False)
        with pytest.raises(TradeDataError, match="settlement_date"):
            load_trades(path)

    def test_non_numeric_notional_raises(self, tmp_path):
        df = load_trades(TRADES_CSV).head(20)
        df["notional_value"] = df["notional_value"].astype(object)
        df.loc[df.index[0], "notional_value"] = "lots"
        path = tmp_path / "badnum.csv"
        df.to_csv(path, index=False)
        with pytest.raises(TradeDataError, match="notional_value"):
            load_trades(path)

    def test_non_boolean_flag_raises(self, tmp_path):
        df = load_trades(TRADES_CSV).head(20)
        df["amendment_flag"] = df["amendment_flag"].astype(object)
        df.loc[df.index[0], "amendment_flag"] = "maybe"
        path = tmp_path / "badbool.csv"
        df.to_csv(path, index=False)
        with pytest.raises(TradeDataError, match="amendment_flag"):
            load_trades(path)

    def test_header_only_file_raises(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text(",".join(REQUIRED_COLUMNS) + "\n")
        with pytest.raises(TradeDataError, match="no rows"):
            load_trades(path)


class TestCleanTrades:
    def test_derives_the_helper_columns(self, sample_trades):
        for column in ("trade_month", "days_to_settlement", "days_to_amendment"):
            assert column in sample_trades.columns
        assert sample_trades["trade_month"].iloc[0] == "2026-03"

    def test_days_to_settlement_counts_business_days(self, sample_trades):
        # T00001: Mon 2026-03-02 -> Tue 2026-03-03 is one business day.
        row = sample_trades.loc[sample_trades["trade_id"] == "T00001"].iloc[0]
        assert row["days_to_settlement"] == 1

    def test_settlement_before_the_trade_date_is_negative(self, raw_frame):
        typed = _typed(raw_frame)
        typed.loc[0, "settlement_date"] = pd.Timestamp("2026-02-27")
        cleaned = clean_trades(typed)
        assert cleaned.loc[0, "days_to_settlement"] == -1

    def test_days_to_amendment_is_nan_without_an_amendment(self, sample_trades):
        booked = sample_trades.loc[sample_trades["trade_status"] == "Booked"]
        assert booked["days_to_amendment"].isna().all()

    def test_days_to_amendment_counts_business_days(self, sample_trades):
        # T00006: Mon 2026-03-02 -> Thu 2026-03-12 is eight business days.
        row = sample_trades.loc[sample_trades["trade_id"] == "T00006"].iloc[0]
        assert row["days_to_amendment"] == 8

    def test_duplicate_trade_ids_are_dropped_and_reported(self, raw_frame):
        typed = _typed(raw_frame)
        duplicated = pd.concat([typed, typed.iloc[[0]]], ignore_index=True)
        cleaned = clean_trades(duplicated)
        assert len(cleaned) == len(typed)
        assert cleaned.attrs["cleaning_report"].duplicate_trade_ids == 1

    def test_categorical_casing_is_normalised(self, raw_frame):
        typed = _typed(raw_frame)
        typed.loc[0, "product_type"] = "  bond "
        typed.loc[1, "desk"] = "RATES"
        typed.loc[2, "buy_sell"] = "sell"
        cleaned = clean_trades(typed)
        assert cleaned.loc[0, "product_type"] == "Bond"
        assert cleaned.loc[1, "desk"] == "Rates"
        assert cleaned.loc[2, "buy_sell"] == "Sell"

    def test_fx_forward_keeps_its_capitalisation(self, raw_frame):
        typed = _typed(raw_frame)
        typed.loc[4, "product_type"] = "fx forward"
        cleaned = clean_trades(typed)
        assert cleaned.loc[4, "product_type"] == "FX Forward"

    def test_blank_counterparty_becomes_unknown(self, raw_frame):
        typed = _typed(raw_frame)
        typed.loc[0, "counterparty"] = "   "
        cleaned = clean_trades(typed)
        assert cleaned.loc[0, "counterparty"] == "UNKNOWN"
        assert cleaned.attrs["cleaning_report"].filled_categoricals["counterparty"] == 1

    def test_inconsistent_notional_is_recomputed(self, raw_frame):
        typed = _typed(raw_frame)
        typed.loc[0, "notional_value"] = 1.0
        cleaned = clean_trades(typed)
        expected = cleaned.loc[0, "quantity"] * cleaned.loc[0, "price"]
        assert cleaned.loc[0, "notional_value"] == pytest.approx(expected)
        assert cleaned.attrs["cleaning_report"].recomputed_notionals == 1

    def test_amendment_flag_is_reconciled_with_the_timestamp(self, raw_frame):
        typed = _typed(raw_frame)
        typed.loc[5, "amendment_flag"] = False  # timestamp present, flag wrong
        cleaned = clean_trades(typed)
        assert bool(cleaned.loc[5, "amendment_flag"]) is True
        assert cleaned.attrs["cleaning_report"].repaired_amendment_flags == 1

    def test_rows_missing_critical_fields_are_dropped(self, raw_frame):
        typed = _typed(raw_frame)
        extra = typed.iloc[[0]].copy()
        extra["trade_id"] = "T99999"
        extra["notional_value"] = np.nan
        cleaned = clean_trades(pd.concat([typed, extra], ignore_index=True))
        assert "T99999" not in set(cleaned["trade_id"])
        assert cleaned.attrs["cleaning_report"].dropped_missing_critical == 1

    def test_too_many_bad_rows_raises(self):
        # The tolerance check only engages above 50 rows, so this uses a real
        # slice of the committed dataset rather than the 10-row fixture.
        df = load_trades(TRADES_CSV).head(200).copy()
        df.loc[df.index[:30], "notional_value"] = np.nan
        with pytest.raises(TradeDataError, match="tolerance"):
            clean_trades(df)

    def test_a_few_bad_rows_in_a_small_frame_are_tolerated(self, raw_frame):
        typed = _typed(raw_frame)
        typed.loc[typed.index[0], "notional_value"] = np.nan
        cleaned = clean_trades(typed)
        assert len(cleaned) == len(typed) - 1

    def test_missing_columns_raise(self, raw_frame):
        typed = _typed(raw_frame).drop(columns=["desk"])
        with pytest.raises(TradeDataError, match="desk"):
            clean_trades(typed)

    def test_clean_report_is_quiet_on_good_data(self, sample_trades):
        assert sample_trades.attrs["cleaning_report"].is_clean()

    def test_committed_dataset_needs_no_repairs(self):
        cleaned = clean_trades(load_trades(TRADES_CSV))
        assert cleaned.attrs["cleaning_report"].is_clean()
        assert len(cleaned) >= 3_000


def _typed(raw: pd.DataFrame) -> pd.DataFrame:
    """Parse the raw fixture's date columns without cleaning it."""
    out = raw.copy()
    for column in (
        "trade_date",
        "settlement_date",
        "booking_datetime",
        "amendment_datetime",
    ):
        out[column] = pd.to_datetime(out[column])
    return out
