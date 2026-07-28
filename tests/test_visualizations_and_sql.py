"""Smoke tests for the Plotly builders and the SQLite-backed views."""

from __future__ import annotations

import plotly.graph_objects as go
import pytest

from src.sqlite_views import (
    QUERIES,
    counterparty_exposure_view,
    desk_exception_league_table,
    monthly_exception_trend,
    run_query,
    trade_connection,
)
from src.surveillance_rules import run_all_rules
from src.visualizations import (
    empty_figure,
    exception_summary_chart,
    exposure_by_counterparty_chart,
    settlement_timing_chart,
    trade_volume_trend_chart,
    trader_desk_activity_chart,
)


@pytest.fixture(scope="module")
def flagged(generated_trades):
    """The generated blotter with all rules applied."""
    return run_all_rules(generated_trades)


class TestCharts:
    def test_every_builder_returns_a_figure_with_data(self, flagged):
        builders = [
            lambda: exposure_by_counterparty_chart(flagged),
            lambda: trade_volume_trend_chart(flagged),
            lambda: trade_volume_trend_chart(flagged, metric="notional", by_desk=True),
            lambda: exception_summary_chart(flagged, group_by="desk"),
            lambda: exception_summary_chart(flagged, group_by="product_type"),
            lambda: trader_desk_activity_chart(flagged),
            lambda: settlement_timing_chart(flagged),
        ]
        for build in builders:
            fig = build()
            assert isinstance(fig, go.Figure)
            assert len(fig.data) > 0

    def test_charts_degrade_gracefully_on_an_empty_frame(self, flagged):
        empty = flagged.iloc[0:0]
        assert isinstance(exposure_by_counterparty_chart(empty), go.Figure)
        assert isinstance(trade_volume_trend_chart(empty), go.Figure)
        assert isinstance(settlement_timing_chart(empty), go.Figure)
        assert isinstance(trader_desk_activity_chart(empty), go.Figure)

    def test_charts_work_without_the_flag_columns(self, generated_trades):
        assert isinstance(exposure_by_counterparty_chart(generated_trades), go.Figure)
        assert isinstance(trader_desk_activity_chart(generated_trades), go.Figure)

    def test_exception_chart_requires_the_flag_columns(self, generated_trades):
        with pytest.raises(KeyError):
            exception_summary_chart(generated_trades, group_by="desk")

    def test_empty_figure_carries_the_message(self):
        fig = empty_figure("nothing here")
        assert fig.layout.annotations[0].text == "nothing here"

    def test_unknown_metric_raises(self, flagged):
        with pytest.raises(ValueError, match="metric must be"):
            trade_volume_trend_chart(flagged, metric="mood")


class TestSqliteViews:
    def test_desk_league_table_totals_match_the_frame(self, flagged):
        league = desk_exception_league_table(flagged)
        assert league["trades"].sum() == len(flagged)
        assert league["total_exceptions"].sum() == int(flagged["exception_count"].sum())

    def test_counterparty_view_shares_sum_to_one_hundred(self, flagged):
        view = counterparty_exposure_view(flagged, limit=1_000)
        assert view["pct_of_book"].sum() == pytest.approx(100.0, abs=0.5)

    def test_counterparty_view_respects_the_limit(self, flagged):
        assert len(counterparty_exposure_view(flagged, limit=5)) == 5

    def test_monthly_trend_is_chronological_and_complete(self, flagged):
        trend = monthly_exception_trend(flagged)
        assert list(trend["trade_month"]) == sorted(trend["trade_month"])
        assert trend["trades"].sum() == len(flagged)

    def test_connection_exposes_the_trades_table(self, flagged):
        with trade_connection(flagged) as connection:
            (count,) = connection.execute("SELECT COUNT(*) FROM trades").fetchone()
        assert count == len(flagged)

    def test_unflagged_frame_raises_a_helpful_error(self, generated_trades):
        with pytest.raises(KeyError, match="run_all_rules"):
            desk_exception_league_table(generated_trades)

    def test_unknown_query_name_raises(self, flagged):
        with pytest.raises(KeyError, match="Unknown query"):
            run_query(flagged, "select_star")

    def test_all_named_queries_execute(self, flagged):
        for name in QUERIES:
            params = {"limit": 5} if ":limit" in QUERIES[name] else None
            assert not run_query(flagged, name, params).empty
