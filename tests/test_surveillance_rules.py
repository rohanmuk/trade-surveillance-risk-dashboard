"""Unit tests for the five surveillance rules and the orchestrator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import FLAG_COLUMNS, SurveillanceConfig
from src.surveillance_rules import (
    exception_summary,
    flag_counterparty_concentration,
    flag_high_trade_volume,
    flag_large_notional,
    flag_late_amendment_cancellation,
    flag_settlement_risk,
    run_all_rules,
)


def flagged_ids(df: pd.DataFrame, flags: pd.Series) -> set:
    """Trade IDs marked True by a rule."""
    return set(df.loc[flags, "trade_id"])


# --------------------------------------------------------------------------- #
# Rule 1 -- large notional
# --------------------------------------------------------------------------- #


class TestLargeNotional:
    def test_absolute_threshold_flags_only_the_big_trade(self, sample_trades):
        # T00004 is 500,000 x 200 = $100m; every other trade is under $200k.
        result = flag_large_notional(sample_trades, threshold=1_000_000, mode="absolute")
        assert flagged_ids(sample_trades, result.flags) == {"T00004"}

    def test_small_trade_is_not_flagged(self, sample_trades):
        result = flag_large_notional(sample_trades, threshold=1_000_000, mode="absolute")
        small = sample_trades.loc[sample_trades["trade_id"] == "T00001"]
        assert not result.flags.loc[small.index].any()

    def test_percentile_mode_flags_the_upper_tail(self, sample_trades):
        result = flag_large_notional(sample_trades, percentile=0.90, mode="percentile")
        assert result.n_flagged == 1
        assert "T00004" in flagged_ids(sample_trades, result.flags)

    def test_percentile_mode_reports_the_effective_threshold(self, sample_trades):
        result = flag_large_notional(sample_trades, percentile=0.90)
        expected = sample_trades["notional_value"].quantile(0.90)
        assert result.details["effective_threshold"] == pytest.approx(expected)

    def test_by_product_judges_each_product_against_its_own_peers(self, sample_trades):
        result = flag_large_notional(sample_trades, percentile=0.75, by_product=True)
        # The single equity trade cannot exceed its own 75th percentile.
        assert "T00004" not in flagged_ids(sample_trades, result.flags)

    def test_absolute_mode_requires_a_threshold(self, sample_trades):
        with pytest.raises(ValueError, match="threshold is required"):
            flag_large_notional(sample_trades, mode="absolute")

    def test_rejects_unknown_mode(self, sample_trades):
        with pytest.raises(ValueError, match="mode must be"):
            flag_large_notional(sample_trades, mode="magic")

    def test_rejects_out_of_range_percentile(self, sample_trades):
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            flag_large_notional(sample_trades, percentile=1.5)


# --------------------------------------------------------------------------- #
# Rule 2 -- counterparty concentration
# --------------------------------------------------------------------------- #


class TestCounterpartyConcentration:
    def test_shares_sum_to_one_hundred_percent(self, sample_trades):
        result = flag_counterparty_concentration(sample_trades)
        assert result.summary["exposure_pct"].sum() == pytest.approx(100.0)

    def test_dominant_counterparty_is_flagged(self, sample_trades):
        # Delta Ltd faces the single $100m trade -> ~99% of gross notional.
        result = flag_counterparty_concentration(
            sample_trades, top_n=0, exposure_pct_threshold=50.0
        )
        flagged = set(result.summary.loc[result.summary["flagged"], "counterparty"])
        assert flagged == {"Delta Ltd"}

    def test_small_counterparty_is_not_flagged(self, sample_trades):
        result = flag_counterparty_concentration(
            sample_trades, top_n=0, exposure_pct_threshold=50.0
        )
        assert not result.flags.loc[sample_trades["counterparty"] == "Beta Corp"].any()

    def test_top_n_flags_exactly_n_counterparties(self, sample_trades):
        result = flag_counterparty_concentration(
            sample_trades, top_n=2, exposure_pct_threshold=100.0
        )
        assert int(result.summary["flagged"].sum()) == 2

    def test_hhi_is_bounded_and_high_when_concentrated(self, sample_trades):
        result = flag_counterparty_concentration(sample_trades)
        assert 0 < result.details["hhi"] <= 10_000
        assert result.details["hhi"] > 2_500

    def test_rejects_threshold_outside_percentage_range(self, sample_trades):
        with pytest.raises(ValueError, match="between 0 and 100"):
            flag_counterparty_concentration(sample_trades, exposure_pct_threshold=150.0)


# --------------------------------------------------------------------------- #
# Rule 3 -- high trade volume
# --------------------------------------------------------------------------- #


class TestHighTradeVolume:
    def test_busiest_trader_is_the_outlier(self, sample_trades):
        # Ann books 5 of the 10 trades; Ben 2 and Cara 3.
        result = flag_high_trade_volume(
            sample_trades, group_by="trader", z_threshold=1.0
        )
        flagged = set(result.summary.loc[result.summary["flagged"], "group"])
        assert flagged == {"Ann"}

    def test_quiet_trader_is_not_flagged(self, sample_trades):
        result = flag_high_trade_volume(
            sample_trades, group_by="trader", z_threshold=1.0
        )
        assert not result.flags.loc[sample_trades["trader"] == "Ben"].any()

    def test_high_threshold_flags_nobody(self, sample_trades):
        result = flag_high_trade_volume(
            sample_trades, group_by="trader", z_threshold=10.0
        )
        assert result.n_flagged == 0

    def test_flags_apply_to_every_trade_of_an_outlier_group(self, sample_trades):
        result = flag_high_trade_volume(
            sample_trades, group_by="trader", z_threshold=1.0
        )
        ann_trades = int((sample_trades["trader"] == "Ann").sum())
        assert result.n_flagged == ann_trades

    def test_max_attainable_z_is_sqrt_n_minus_one(self, sample_trades):
        result = flag_high_trade_volume(sample_trades, group_by="desk")
        n_groups = result.details["n_groups"]
        assert result.details["max_attainable_z"] == pytest.approx(
            np.sqrt(n_groups - 1)
        )
        assert result.summary["z_score"].max() <= result.details["max_attainable_z"] + 1e-9

    def test_iqr_method_runs_and_agrees_on_the_extreme_case(self, sample_trades):
        result = flag_high_trade_volume(
            sample_trades, group_by="trader", method="iqr", iqr_multiplier=0.5
        )
        assert "Ann" in set(result.summary.loc[result.summary["flagged"], "group"])

    def test_notional_metric_uses_notional_not_counts(self, sample_trades):
        result = flag_high_trade_volume(
            sample_trades, group_by="trader", metric="notional", z_threshold=1.0
        )
        # Ben books the $100m equity trade, so by notional he is the outlier.
        assert set(result.summary.loc[result.summary["flagged"], "group"]) == {"Ben"}

    def test_rejects_unknown_metric(self, sample_trades):
        with pytest.raises(ValueError, match="metric must be"):
            flag_high_trade_volume(sample_trades, metric="vibes")


# --------------------------------------------------------------------------- #
# Rule 4 -- late amendments and cancellations
# --------------------------------------------------------------------------- #


class TestLateAmendment:
    def test_late_amendment_and_late_cancellation_are_flagged(self, sample_trades):
        result = flag_late_amendment_cancellation(
            sample_trades, business_days_threshold=2
        )
        assert flagged_ids(sample_trades, result.flags) == {"T00006", "T00007"}

    def test_next_day_amendment_is_not_flagged(self, sample_trades):
        result = flag_late_amendment_cancellation(
            sample_trades, business_days_threshold=2
        )
        on_time = sample_trades["trade_id"] == "T00005"
        assert not result.flags.loc[on_time].any()

    def test_booked_trades_are_never_flagged(self, sample_trades):
        result = flag_late_amendment_cancellation(
            sample_trades, business_days_threshold=0
        )
        booked = sample_trades["trade_status"] == "Booked"
        assert not result.flags.loc[booked].any()

    def test_raising_the_threshold_removes_flags(self, sample_trades):
        loose = flag_late_amendment_cancellation(
            sample_trades, business_days_threshold=10
        )
        assert loose.n_flagged == 0

    def test_counts_amendments_and_cancellations_separately(self, sample_trades):
        result = flag_late_amendment_cancellation(sample_trades)
        assert result.details["n_amendments"] == 2
        assert result.details["n_cancellations"] == 1

    def test_rejects_negative_threshold(self, sample_trades):
        with pytest.raises(ValueError, match="must be >= 0"):
            flag_late_amendment_cancellation(sample_trades, business_days_threshold=-1)


# --------------------------------------------------------------------------- #
# Rule 5 -- settlement risk
# --------------------------------------------------------------------------- #


class TestSettlementRisk:
    def test_same_day_and_far_dated_bonds_are_flagged(self, sample_trades):
        result = flag_settlement_risk(sample_trades)
        assert {"T00008", "T00009"} <= flagged_ids(sample_trades, result.flags)

    def test_standard_t_plus_one_bond_is_not_flagged(self, sample_trades):
        result = flag_settlement_risk(sample_trades)
        normal = sample_trades["trade_id"] == "T00001"
        assert not result.flags.loc[normal].any()

    def test_product_windows_exempt_forward_dated_fx(self, sample_trades):
        result = flag_settlement_risk(sample_trades)
        fx = sample_trades["product_type"] == "FX Forward"
        assert not result.flags.loc[fx].any()

    def test_flat_window_catches_the_same_fx_trades(self, sample_trades):
        result = flag_settlement_risk(
            sample_trades, min_days=1, max_days=3, product_windows=None
        )
        fx = sample_trades["product_type"] == "FX Forward"
        assert result.flags.loc[fx].all()

    def test_splits_breaches_into_too_fast_and_too_slow(self, sample_trades):
        result = flag_settlement_risk(sample_trades)
        assert result.details["n_too_fast"] >= 1
        assert result.details["n_too_slow"] >= 1

    def test_rejects_inverted_window(self, sample_trades):
        with pytest.raises(ValueError, match="must not exceed"):
            flag_settlement_risk(sample_trades, min_days=5, max_days=1)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


class TestRunAllRules:
    def test_adds_every_flag_column_plus_the_count(self, sample_trades):
        out = run_all_rules(sample_trades)
        for column in FLAG_COLUMNS:
            assert column in out.columns
            assert out[column].dtype == bool
        assert out["exception_count"].dtype == np.dtype("int64")

    def test_exception_count_is_the_row_wise_sum(self, sample_trades):
        out = run_all_rules(sample_trades)
        expected = out[list(FLAG_COLUMNS)].sum(axis=1)
        pd.testing.assert_series_equal(
            out["exception_count"], expected.astype("int64"), check_names=False
        )

    def test_does_not_mutate_the_input(self, sample_trades):
        before = sample_trades.copy()
        run_all_rules(sample_trades)
        pd.testing.assert_frame_equal(sample_trades, before)

    def test_handles_an_empty_frame(self, sample_trades):
        out = run_all_rules(sample_trades.iloc[0:0])
        assert out.empty
        assert set(FLAG_COLUMNS) <= set(out.columns)

    def test_every_rule_fires_on_the_generated_dataset(self, generated_trades):
        out = run_all_rules(generated_trades, SurveillanceConfig())
        for column in FLAG_COLUMNS:
            assert out[column].sum() > 0, f"{column} never fires on the sample data"

    def test_thresholds_move_the_flag_counts(self, generated_trades):
        strict = run_all_rules(
            generated_trades,
            SurveillanceConfig(large_notional_percentile=0.995),
        )
        loose = run_all_rules(
            generated_trades,
            SurveillanceConfig(large_notional_percentile=0.90),
        )
        assert loose["flag_large_notional"].sum() > strict["flag_large_notional"].sum()

    def test_rule_results_are_attached_for_the_ui(self, sample_trades):
        out = run_all_rules(sample_trades)
        assert set(out.attrs["rule_results"]) == set(FLAG_COLUMNS)


class TestExceptionSummary:
    def test_rates_by_group_are_percentages_of_trades(self, sample_trades):
        out = run_all_rules(sample_trades)
        summary = exception_summary(out, group_by="desk")
        assert set(summary["desk"]) == set(out["desk"])
        assert (summary["exception_rate_pct"] <= 100.0).all()
        assert summary["trades"].sum() == len(out)

    def test_works_for_any_categorical_column(self, sample_trades):
        out = run_all_rules(sample_trades)
        summary = exception_summary(out, group_by="product_type")
        assert summary["trades"].sum() == len(out)

    def test_raises_on_an_unflagged_frame(self, sample_trades):
        with pytest.raises(KeyError, match="exception_summary requires"):
            exception_summary(sample_trades, group_by="desk")
