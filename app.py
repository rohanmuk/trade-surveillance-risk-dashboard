"""Trade Surveillance Risk Dashboard — Streamlit entry point.

This module is deliberately thin: it owns page configuration, the sidebar
controls, filtering and layout. Every calculation lives in :mod:`src` so the
same numbers can be reproduced in the notebook or a unit test without importing
Streamlit.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import pandas as pd
import streamlit as st

from src.config import (
    DESKS,
    ENTITY_LEVEL_FLAGS,
    FLAG_COLUMNS,
    FLAG_LABELS,
    PRODUCT_SETTLEMENT_WINDOW,
    PRODUCT_TYPES,
    QUERY_ROW_LIMIT,
    SurveillanceConfig,
    TRADES_CSV,
    TRADE_LEVEL_FLAGS,
)
from src.data_cleaning import TradeDataError, clean_trades, load_trades
from src.sqlite_views import (
    QUERIES,
    counterparty_exposure_view,
    desk_exception_league_table,
    monthly_exception_trend,
)
from src.surveillance_rules import (
    exception_summary,
    flag_counterparty_concentration,
    flag_high_trade_volume,
    flag_large_notional,
    flag_late_amendment_cancellation,
    flag_settlement_risk,
    run_all_rules,
)
from src.visualizations import (
    empty_figure,
    exception_summary_chart,
    exposure_by_counterparty_chart,
    settlement_timing_chart,
    trade_volume_trend_chart,
    trader_desk_activity_chart,
)

PAGES: Tuple[str, ...] = (
    "Executive Summary",
    "Trade Exceptions",
    "Counterparty Exposure",
    "Trader / Desk Activity",
    "Product-Level Risk Analysis",
    "Raw Trade Data",
)

_MILLIONS = 1_000_000.0


# --------------------------------------------------------------------------- #
# Data plumbing
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Filters:
    """Global population filters. Frozen so it can key ``st.cache_data``."""

    start_date: str
    end_date: str
    desks: Tuple[str, ...]
    products: Tuple[str, ...]


@st.cache_data(show_spinner="Loading trade blotter...")
def load_clean_data(path: str) -> pd.DataFrame:
    """Read and clean the blotter from disk.

    Args:
        path: Path to the blotter CSV (passed explicitly so the cache
            invalidates if the file location changes).

    Returns:
        The cleaned blotter.
    """
    return clean_trades(load_trades(path))


@st.cache_data(show_spinner="Running surveillance rules...")
def get_flagged_data(
    path: str, filters: Filters, config: SurveillanceConfig
) -> pd.DataFrame:
    """Filter the blotter, then run all five rules over the filtered population.

    Filtering happens *before* the rules so that every statistic on every page
    describes the same population: a percentile threshold, a concentration
    share and a peer-group z-score are all relative measures, and computing
    them on the full book while displaying a filtered slice would make the
    numbers on screen mutually inconsistent.

    Args:
        path: Path to the blotter CSV.
        filters: Global population filters.
        config: Rule thresholds.

    Returns:
        The filtered blotter with the six flag columns and ``exception_count``.
    """
    df = load_clean_data(path)
    mask = (
        (df["trade_date"] >= pd.Timestamp(filters.start_date))
        & (df["trade_date"] <= pd.Timestamp(filters.end_date))
        & (df["desk"].isin(filters.desks))
        & (df["product_type"].isin(filters.products))
    )
    return run_all_rules(df.loc[mask].reset_index(drop=True), config)


# --------------------------------------------------------------------------- #
# Small presentation helpers
# --------------------------------------------------------------------------- #


def money(value: float) -> str:
    """Format a notional as a compact currency string."""
    if abs(value) >= 1e9:
        return f"${value / 1e9:,.2f}bn"
    if abs(value) >= 1e6:
        return f"${value / 1e6:,.1f}m"
    if abs(value) >= 1e3:
        return f"${value / 1e3:,.1f}k"
    return f"${value:,.0f}"


def chart(fig, key: str) -> None:
    """Render a Plotly figure full width.

    ``st.plotly_chart`` already stretches to the container by default; passing
    an explicit width would land in its deprecated ``**kwargs``.
    """
    st.plotly_chart(fig, key=key)


def table(df: pd.DataFrame, **kwargs) -> None:
    """Render a DataFrame full width, without the index."""
    st.dataframe(df, width="stretch", hide_index=True, **kwargs)


def sql_expander(query_name: str, label: str = "Show the SQL behind this view") -> None:
    """Reveal the SQL statement backing a query-driven table."""
    with st.expander(label):
        st.caption(
            "The filtered blotter is loaded into an in-memory SQLite table "
            "named `trades`; this is the query that produced the table above."
        )
        st.code(QUERIES[query_name].strip(), language="sql")


def flag_label_map(columns: Sequence[str]) -> dict:
    """Rename flag columns to their human-readable labels for display."""
    return {c: FLAG_LABELS.get(c, c) for c in columns}


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #


def build_sidebar(base: pd.DataFrame) -> "Tuple[str, Filters, SurveillanceConfig]":
    """Render the sidebar and return the page, filters and rule thresholds.

    Args:
        base: The unfiltered cleaned blotter, used to seed the control ranges.

    Returns:
        Tuple of (selected page name, global filters, rule configuration).
    """
    st.sidebar.title("Trade Surveillance")
    st.sidebar.caption("Simulated blotter — no real client or employee data.")

    page = st.sidebar.radio("Page", PAGES, label_visibility="collapsed")
    st.sidebar.divider()

    # ---- Global filters ---------------------------------------------------- #
    st.sidebar.subheader("Global filters")
    min_date = base["trade_date"].min().date()
    max_date = base["trade_date"].max().date()
    date_range = st.sidebar.date_input(
        "Trade date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        help="Applied before the rules run, so all thresholds are relative to "
        "the selected population.",
    )
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date, end_date = date_range
    else:  # the user has picked the first of two dates and is mid-selection
        start_date = end_date = (
            date_range[0] if isinstance(date_range, (tuple, list)) else date_range
        )

    available_desks = [d for d in DESKS if d in set(base["desk"])]
    available_products = [p for p in PRODUCT_TYPES if p in set(base["product_type"])]
    desks = st.sidebar.multiselect("Desk", available_desks, default=available_desks)
    products = st.sidebar.multiselect(
        "Product type", available_products, default=available_products
    )

    filters = Filters(
        start_date=str(start_date),
        end_date=str(end_date),
        desks=tuple(desks) or tuple(available_desks),
        products=tuple(products) or tuple(available_products),
    )

    st.sidebar.divider()
    st.sidebar.subheader("Rule thresholds")
    defaults = SurveillanceConfig()

    with st.sidebar.expander("1. Large notional", expanded=False):
        mode = st.radio(
            "Threshold basis",
            ("percentile", "absolute"),
            format_func=lambda m: (
                "Percentile of the book" if m == "percentile" else "Fixed notional"
            ),
            horizontal=True,
        )
        percentile = st.slider(
            "Percentile", 0.80, 0.999, defaults.large_notional_percentile, 0.005,
            disabled=mode != "percentile",
            help="A trade is flagged when its notional exceeds this quantile of "
            "the filtered population.",
        )
        absolute = st.number_input(
            "Fixed threshold ($)",
            min_value=100_000.0,
            value=defaults.large_notional_threshold,
            step=1_000_000.0,
            disabled=mode != "absolute",
        )

    with st.sidebar.expander("2. Counterparty concentration", expanded=False):
        top_n = st.slider("Flag top N counterparties", 0, 15, defaults.concentration_top_n)
        pct_threshold = st.slider(
            "Share-of-notional threshold (%)",
            0.5,
            25.0,
            defaults.concentration_pct_threshold,
            0.5,
        )

    with st.sidebar.expander("3. Trader / desk volume", expanded=False):
        metric = st.radio(
            "Activity metric",
            ("trade_count", "notional"),
            format_func=lambda m: "Trade count" if m == "trade_count" else "Notional",
            horizontal=True,
        )
        method = st.radio(
            "Outlier method",
            ("zscore", "iqr"),
            format_func=lambda m: "Z-score" if m == "zscore" else "IQR fence",
            horizontal=True,
        )
        trader_z = st.slider(
            "Trader z-score threshold", 0.5, 4.0, defaults.trader_z_threshold, 0.1,
            disabled=method != "zscore",
        )
        desk_z = st.slider(
            "Desk z-score threshold", 0.5, 4.0, defaults.desk_z_threshold, 0.1,
            disabled=method != "zscore",
            help="Kept lower than the trader threshold: with n groups the "
            "largest attainable z-score is sqrt(n-1), which is only 2.0 for "
            "five desks.",
        )
        iqr_multiplier = st.slider(
            "IQR fence multiplier", 0.5, 3.0, defaults.volume_iqr_multiplier, 0.1,
            disabled=method != "iqr",
        )

    with st.sidebar.expander("4. Late amendments", expanded=False):
        late_days = st.slider(
            "Business days after trade date",
            0,
            15,
            defaults.late_amendment_business_days,
            help="Amendments and cancellations booked later than this are flagged.",
        )

    with st.sidebar.expander("5. Settlement risk", expanded=False):
        use_product_windows = st.checkbox(
            "Use per-product settlement windows",
            value=defaults.settlement_use_product_windows,
            help="FX Forwards settle weeks out; cash products settle T+1/T+2. "
            "Turn this off to apply one flat window to every product.",
        )
        settle_min = st.number_input(
            "Minimum business days",
            min_value=-5,
            max_value=30,
            value=defaults.settlement_min_days,
            disabled=use_product_windows,
        )
        settle_max = st.number_input(
            "Maximum business days",
            min_value=0,
            max_value=180,
            value=defaults.settlement_max_days,
            disabled=use_product_windows,
        )
        if use_product_windows:
            st.caption(
                "Active windows: "
                + ", ".join(
                    f"{p} T+{lo}–{hi}"
                    for p, (lo, hi) in PRODUCT_SETTLEMENT_WINDOW.items()
                )
            )

    config = SurveillanceConfig(
        large_notional_mode=mode,
        large_notional_percentile=percentile,
        large_notional_threshold=absolute,
        concentration_top_n=top_n,
        concentration_pct_threshold=pct_threshold,
        volume_metric=metric,
        volume_method=method,
        trader_z_threshold=trader_z,
        desk_z_threshold=desk_z,
        volume_iqr_multiplier=iqr_multiplier,
        late_amendment_business_days=int(late_days),
        settlement_use_product_windows=use_product_windows,
        settlement_min_days=int(settle_min),
        settlement_max_days=int(max(settle_max, settle_min)),
    )
    return page, filters, config


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #


def page_executive_summary(df: pd.DataFrame, config: SurveillanceConfig) -> None:
    """KPI cards and the three headline charts."""
    st.header("Executive Summary")

    trade_level = df[list(TRADE_LEVEL_FLAGS)].any(axis=1)
    entity_level = df[list(ENTITY_LEVEL_FLAGS)].any(axis=1)
    any_flag = df["exception_count"] > 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Trades in scope", f"{len(df):,}")
    c2.metric("Gross notional", money(df["notional_value"].sum()))
    c3.metric(
        "Trade-level exceptions",
        f"{int(trade_level.sum()):,}",
        f"{trade_level.mean() * 100:.1f}% of trades",
        delta_color="off",
    )
    c4.metric(
        "Entity-level alerts",
        f"{int(entity_level.sum()):,}",
        f"{entity_level.mean() * 100:.1f}% of trades",
        delta_color="off",
    )
    c5.metric(
        "Any exception",
        f"{int(any_flag.sum()):,}",
        f"{any_flag.mean() * 100:.1f}% of trades",
        delta_color="off",
    )

    st.caption(
        "**Trade-level** exceptions (large notional, late amendment, settlement "
        "breach) are properties of the individual ticket and can be actioned one "
        "by one. **Entity-level** alerts (counterparty concentration, trader and "
        "desk volume) are properties of a counterparty, trader or desk and "
        "therefore fan out across every trade that entity touched — quoting them "
        "as a per-trade rate overstates the real alert load."
    )

    st.divider()
    left, right = st.columns([3, 2])
    with left:
        by_desk = st.toggle("Split the trend by desk", value=False)
        chart(
            trade_volume_trend_chart(df, metric="trade_count", freq="W", by_desk=by_desk),
            key="exec_trend",
        )
    with right:
        chart(exception_summary_chart(df, group_by="desk"), key="exec_exceptions")

    chart(
        exposure_by_counterparty_chart(
            df,
            exposure=flag_counterparty_concentration(
                df,
                top_n=config.concentration_top_n,
                exposure_pct_threshold=config.concentration_pct_threshold,
            ).summary,
            top_n=15,
        ),
        key="exec_exposure",
    )

    st.divider()
    st.subheader("Desk league table")
    st.caption("Computed in SQL against an in-memory SQLite view of the filtered blotter.")
    league = desk_exception_league_table(df)
    table(
        league,
        column_config={
            "notional_millions": st.column_config.NumberColumn(
                "Notional ($m)", format="%.1f"
            ),
            "exception_rate_pct": st.column_config.ProgressColumn(
                "Exception rate",
                format="%.1f%%",
                min_value=0.0,
                max_value=100.0,
            ),
        },
    )
    sql_expander("desk_exception_league_table")

    monthly = monthly_exception_trend(df)
    if len(monthly) > 1:
        st.subheader("Monthly exception rate")
        st.line_chart(
            monthly.set_index("trade_month")["exception_rate_pct"],
            height=240,
            y_label="Exception rate (%)",
        )


def page_trade_exceptions(df: pd.DataFrame) -> None:
    """Filterable table of every flagged trade."""
    st.header("Trade Exceptions")

    counts = {c: int(df[c].sum()) for c in FLAG_COLUMNS}
    cols = st.columns(len(FLAG_COLUMNS))
    for col, (flag, count) in zip(cols, counts.items()):
        col.metric(FLAG_LABELS[flag], f"{count:,}")

    st.divider()
    f1, f2, f3 = st.columns([2, 1, 1])
    selected_rules = f1.multiselect(
        "Rule type",
        list(FLAG_COLUMNS),
        default=list(FLAG_COLUMNS),
        format_func=lambda c: FLAG_LABELS[c],
    )
    traders = f2.multiselect("Trader", sorted(df["trader"].unique()))
    match_mode = f3.radio(
        "Match", ("Any selected rule", "All selected rules"), horizontal=False
    )

    if not selected_rules:
        st.info("Select at least one rule to see the exceptions it raised.")
        return

    subset = df[list(selected_rules)]
    mask = subset.any(axis=1) if match_mode == "Any selected rule" else subset.all(axis=1)
    if traders:
        mask &= df["trader"].isin(traders)

    flagged = df.loc[mask].sort_values(
        ["exception_count", "notional_value"], ascending=False
    )

    if flagged.empty:
        st.warning(
            "No trades matched this combination of rules and filters. Try "
            "widening the rule selection, or loosening the thresholds in the "
            "sidebar."
        )
        return

    st.caption(
        f"{len(flagged):,} of {len(df):,} trades matched "
        f"({len(flagged) / len(df) * 100:.1f}%)."
    )

    display_columns = [
        "trade_id",
        "trade_date",
        "trader",
        "desk",
        "product_type",
        "counterparty",
        "buy_sell",
        "notional_value",
        "trade_status",
        "days_to_settlement",
        "days_to_amendment",
        "exception_count",
        *selected_rules,
    ]
    view = flagged[display_columns].rename(columns=flag_label_map(selected_rules))
    table(
        view,
        column_config={
            "trade_date": st.column_config.DateColumn("Trade date", format="YYYY-MM-DD"),
            "notional_value": st.column_config.NumberColumn(
                "Notional", format="$%,.0f"
            ),
            "exception_count": st.column_config.NumberColumn("Exceptions", format="%d"),
        },
        height=520,
    )
    st.download_button(
        "Download these exceptions (CSV)",
        data=flagged.to_csv(index=False).encode("utf-8"),
        file_name="trade_exceptions.csv",
        mime="text/csv",
    )


def page_counterparty_exposure(df: pd.DataFrame, config: SurveillanceConfig) -> None:
    """Concentration chart, exposure table and the SQL view behind it."""
    st.header("Counterparty Exposure")

    result = flag_counterparty_concentration(
        df,
        top_n=config.concentration_top_n,
        exposure_pct_threshold=config.concentration_pct_threshold,
    )
    details = result.details

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Counterparties", f"{details['n_counterparties']:,}")
    c2.metric("Largest exposure", f"{details['top1_pct']:.1f}% of notional")
    c3.metric("Top 5 combined", f"{details['top5_pct']:.1f}% of notional")
    c4.metric(
        "HHI",
        f"{details['hhi']:,.0f}",
        help="Herfindahl-Hirschman Index of notional shares. Above 2,500 is "
        "conventionally read as highly concentrated; 10,000 means a single "
        "counterparty faces the entire book.",
    )

    st.caption(
        f"Flagging counterparties above **{config.concentration_pct_threshold:.1f}%** "
        f"of gross notional, or inside the **top {config.concentration_top_n}** by "
        f"notional — {details['n_flagged_counterparties']} of "
        f"{details['n_counterparties']} currently qualify. Adjust both in the sidebar."
    )

    chart(
        exposure_by_counterparty_chart(df, exposure=result.summary, top_n=20),
        key="cp_exposure",
    )

    st.subheader("Exposure table")
    exposure = result.summary.copy()
    exposure["notional_m"] = exposure["notional"] / _MILLIONS
    table(
        exposure[
            [
                "rank",
                "counterparty",
                "trades",
                "notional_m",
                "exposure_pct",
                "cumulative_pct",
                "flagged",
            ]
        ],
        column_config={
            "rank": st.column_config.NumberColumn("Rank", format="%d"),
            "notional_m": st.column_config.NumberColumn("Notional ($m)", format="%.1f"),
            "exposure_pct": st.column_config.ProgressColumn(
                "Share of book",
                format="%.2f%%",
                min_value=0.0,
                max_value=float(max(exposure["exposure_pct"].max(), 1.0)),
            ),
            "cumulative_pct": st.column_config.NumberColumn(
                "Cumulative %", format="%.1f%%"
            ),
            "flagged": st.column_config.CheckboxColumn("Concentrated"),
        },
        height=460,
    )

    st.subheader("Same view, computed in SQL")
    table(counterparty_exposure_view(df, limit=QUERY_ROW_LIMIT))
    sql_expander("counterparty_exposure_view")


def page_trader_desk_activity(df: pd.DataFrame, config: SurveillanceConfig) -> None:
    """Volume outlier detection at trader and desk level."""
    st.header("Trader / Desk Activity")

    trader_result = flag_high_trade_volume(
        df,
        group_by="trader",
        metric=config.volume_metric,
        method=config.volume_method,
        z_threshold=config.trader_z_threshold,
        iqr_multiplier=config.volume_iqr_multiplier,
    )
    desk_result = flag_high_trade_volume(
        df,
        group_by="desk",
        metric=config.volume_metric,
        method=config.volume_method,
        z_threshold=config.desk_z_threshold,
        iqr_multiplier=config.volume_iqr_multiplier,
    )

    metric_label = "trade count" if config.volume_metric == "trade_count" else "notional"
    method_label = (
        f"z-score > {config.trader_z_threshold:.1f} (traders) / "
        f"{config.desk_z_threshold:.1f} (desks)"
        if config.volume_method == "zscore"
        else f"above the Q3 + {config.volume_iqr_multiplier:.1f} x IQR fence"
    )
    st.caption(
        f"Outliers are measured on **{metric_label}** within each peer group, "
        f"{method_label}. Change the metric, method and thresholds in the sidebar."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Flagged traders",
        f"{trader_result.details['n_flagged_groups']} of "
        f"{trader_result.details['n_groups']}",
    )
    c2.metric(
        "Flagged desks",
        f"{desk_result.details['n_flagged_groups']} of "
        f"{desk_result.details['n_groups']}",
    )
    c3.metric(
        "Max attainable desk z",
        f"{desk_result.details['max_attainable_z']:.2f}",
        help="With a sample standard deviation, the largest z-score possible "
        "across n groups is sqrt(n-1). A desk threshold above this can never "
        "fire, however skewed the book is.",
    )

    if (
        config.volume_method == "zscore"
        and config.desk_z_threshold >= desk_result.details["max_attainable_z"]
    ):
        st.warning(
            f"The desk z-score threshold ({config.desk_z_threshold:.1f}) is at or "
            f"above the ceiling of {desk_result.details['max_attainable_z']:.2f} "
            f"for {desk_result.details['n_groups']} desks, so no desk can be "
            "flagged. Lower it, or switch to the IQR method."
        )

    chart(
        trader_desk_activity_chart(df, metric=config.volume_metric),
        key="activity_traders",
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Trader peer group")
        table(
            _format_volume_summary(trader_result.summary),
            column_config=_volume_column_config(),
            height=430,
        )
    with right:
        st.subheader("Desk peer group")
        table(
            _format_volume_summary(desk_result.summary),
            column_config=_volume_column_config(),
        )
        chart(
            trade_volume_trend_chart(df, metric="trade_count", freq="W", by_desk=True),
            key="activity_desk_trend",
        )


def _format_volume_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Trim a volume-rule summary to the columns worth showing."""
    view = summary.copy()
    view["notional_m"] = view["notional"] / _MILLIONS
    return view[["group", "trade_count", "notional_m", "z_score", "flagged"]]


def _volume_column_config() -> dict:
    """Column formatting shared by the trader and desk peer-group tables."""
    return {
        "group": st.column_config.TextColumn("Name"),
        "trade_count": st.column_config.NumberColumn("Trades", format="%d"),
        "notional_m": st.column_config.NumberColumn("Notional ($m)", format="%.1f"),
        "z_score": st.column_config.NumberColumn("Z-score", format="%.2f"),
        "flagged": st.column_config.CheckboxColumn("Outlier"),
    }


def page_product_risk(df: pd.DataFrame, config: SurveillanceConfig) -> None:
    """Exceptions and notional broken down by product type."""
    st.header("Product-Level Risk Analysis")

    summary = exception_summary(df, group_by="product_type")
    view = summary.copy()
    view["notional_m"] = view["notional"] / _MILLIONS
    table(
        view[
            [
                "product_type",
                "trades",
                "notional_m",
                "flagged_trades",
                "total_exceptions",
                "exception_rate_pct",
                *FLAG_COLUMNS,
            ]
        ].rename(columns=flag_label_map(FLAG_COLUMNS)),
        column_config={
            "product_type": st.column_config.TextColumn("Product"),
            "notional_m": st.column_config.NumberColumn("Notional ($m)", format="%.1f"),
            "exception_rate_pct": st.column_config.ProgressColumn(
                "Exception rate", format="%.1f%%", min_value=0.0, max_value=100.0
            ),
        },
    )

    left, right = st.columns(2)
    with left:
        chart(
            exception_summary_chart(df, group_by="product_type"),
            key="product_exceptions",
        )
    with right:
        chart(
            trade_volume_trend_chart(df, metric="notional", freq="W"),
            key="product_notional_trend",
        )

    st.divider()
    st.subheader("Settlement timing")
    settlement = flag_settlement_risk(
        df,
        min_days=config.settlement_min_days,
        max_days=config.settlement_max_days,
        product_windows=(
            PRODUCT_SETTLEMENT_WINDOW if config.settlement_use_product_windows else None
        ),
    )
    s1, s2, s3 = st.columns(3)
    s1.metric("Settling too fast", f"{settlement.details['n_too_fast']:,}")
    s2.metric("Settling too slow", f"{settlement.details['n_too_slow']:,}")
    s3.metric(
        "Dated before the trade",
        f"{settlement.details['n_before_trade_date']:,}",
        help="A settlement date earlier than the trade date is impossible in "
        "practice and almost always a keying error.",
    )
    chart(settlement_timing_chart(df), key="product_settlement")
    table(settlement.summary)

    st.divider()
    st.subheader("Large-notional thresholds by product")
    large = flag_large_notional(
        df,
        threshold=config.large_notional_threshold,
        percentile=config.large_notional_percentile,
        mode=config.large_notional_mode,
    )
    large_view = large.summary.copy()
    for column in ("threshold", "median_notional", "max_notional", "flagged_notional"):
        large_view[column] = large_view[column] / _MILLIONS
    table(
        large_view,
        column_config={
            "product_type": st.column_config.TextColumn("Product"),
            "threshold": st.column_config.NumberColumn("Threshold ($m)", format="%.1f"),
            "median_notional": st.column_config.NumberColumn(
                "Median ($m)", format="%.2f"
            ),
            "max_notional": st.column_config.NumberColumn("Largest ($m)", format="%.1f"),
            "flagged_notional": st.column_config.NumberColumn(
                "Flagged notional ($m)", format="%.1f"
            ),
        },
    )

    st.divider()
    st.subheader("Lifecycle events")
    late = flag_late_amendment_cancellation(
        df, business_days_threshold=config.late_amendment_business_days
    )
    l1, l2, l3 = st.columns(3)
    l1.metric("Amendments", f"{late.details['n_amendments']:,}")
    l2.metric("Cancellations", f"{late.details['n_cancellations']:,}")
    l3.metric("Booked late", f"{late.details['late_event_rate_pct']:.1f}% of events")
    if late.summary.empty:
        st.info("No amendments or cancellations in the filtered population.")
    else:
        table(late.summary)


def page_raw_data(df: pd.DataFrame) -> None:
    """Searchable, exportable view of the full filtered blotter."""
    st.header("Raw Trade Data")

    c1, c2 = st.columns([2, 1])
    search = c1.text_input(
        "Search",
        placeholder="Trade ID, trader, counterparty, product...",
        help="Case-insensitive substring match across the text columns.",
    )
    only_flagged = c2.checkbox("Flagged trades only", value=False)

    view = df
    if only_flagged:
        view = view.loc[view["exception_count"] > 0]
    if search:
        text_columns = [
            "trade_id",
            "trader",
            "desk",
            "product_type",
            "counterparty",
            "trade_status",
        ]
        needle = search.strip().lower()
        matches = pd.Series(False, index=view.index)
        for column in text_columns:
            matches |= view[column].astype(str).str.lower().str.contains(needle, regex=False)
        view = view.loc[matches]

    if view.empty:
        st.warning(
            f"No trades matched '{search}'. Clear the search box or widen the "
            "sidebar filters."
        )
        return

    all_columns = list(view.columns)
    default_columns = [
        c for c in all_columns if c not in FLAG_COLUMNS or c == "exception_count"
    ]
    chosen = st.multiselect("Columns", all_columns, default=default_columns)
    if not chosen:
        st.info("Select at least one column to display.")
        return

    st.caption(f"Showing {len(view):,} of {len(df):,} filtered trades.")
    table(
        view[chosen],
        column_config={
            "trade_date": st.column_config.DateColumn("Trade date", format="YYYY-MM-DD"),
            "settlement_date": st.column_config.DateColumn(
                "Settlement date", format="YYYY-MM-DD"
            ),
            "notional_value": st.column_config.NumberColumn("Notional", format="$%,.0f"),
            "price": st.column_config.NumberColumn("Price", format="%.4f"),
        },
        height=560,
    )
    st.download_button(
        "Download as CSV",
        data=view[chosen].to_csv(index=False).encode("utf-8"),
        file_name="trades_filtered.csv",
        mime="text/csv",
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    """Configure the page, build the sidebar and dispatch to the chosen page."""
    st.set_page_config(
        page_title="Trade Surveillance Risk Dashboard",
        page_icon=":shield:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    try:
        base = load_clean_data(str(TRADES_CSV))
    except TradeDataError as exc:
        st.error(f"Could not load the trade blotter.\n\n{exc}")
        st.stop()
        return

    page, filters, config = build_sidebar(base)

    report = base.attrs.get("cleaning_report")
    if report is not None:
        with st.sidebar.expander("Data quality"):
            for line in report.summary_lines():
                st.write(f"- {line}")
            if report.is_clean():
                st.success("No repairs were needed.")

    df = get_flagged_data(str(TRADES_CSV), filters, config)

    if df.empty:
        st.header(page)
        st.warning(
            "No trades match the current filters. Widen the date range, or "
            "re-select a desk or product in the sidebar."
        )
        st.plotly_chart(empty_figure("Nothing to display"))
        return

    if page == "Executive Summary":
        page_executive_summary(df, config)
    elif page == "Trade Exceptions":
        page_trade_exceptions(df)
    elif page == "Counterparty Exposure":
        page_counterparty_exposure(df, config)
    elif page == "Trader / Desk Activity":
        page_trader_desk_activity(df, config)
    elif page == "Product-Level Risk Analysis":
        page_product_risk(df, config)
    else:
        page_raw_data(df)

    st.sidebar.divider()
    st.sidebar.caption(
        f"{len(df):,} trades in scope · {money(df['notional_value'].sum())} gross "
        f"notional · {int((df['exception_count'] > 0).sum()):,} flagged"
    )


if __name__ == "__main__":
    main()
