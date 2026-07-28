"""Plotly figure builders for the surveillance dashboard.

Every function returns a :class:`plotly.graph_objects.Figure` and none of them
imports Streamlit. Keeping the chart code free of the presentation framework
means the same figures render in the notebook, in a unit test, or in the app.

The colour convention is consistent throughout: red/orange means flagged,
blue/grey means normal — see :data:`src.config.COLORS`.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .config import COLORS, DESK_COLORS, FLAG_COLUMNS, FLAG_LABELS, PLOTLY_TEMPLATE

__all__ = [
    "exposure_by_counterparty_chart",
    "trade_volume_trend_chart",
    "exception_summary_chart",
    "trader_desk_activity_chart",
    "settlement_timing_chart",
    "empty_figure",
]

_MILLIONS = 1_000_000.0


def _base_layout(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    """Apply the shared layout: template, title, margins, legend placement."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title={"text": title, "x": 0.0, "xanchor": "left", "font": {"size": 17}},
        height=height,
        margin={"l": 60, "r": 30, "t": 60, "b": 60},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
        },
        hoverlabel={"align": "left"},
    )
    fig.update_xaxes(gridcolor=COLORS["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=COLORS["grid"], zeroline=False)
    return fig


def empty_figure(message: str = "No data for the current filters") -> go.Figure:
    """Return a blank figure carrying an explanatory message.

    Used instead of raising when a filter combination yields zero rows, so the
    dashboard degrades gracefully rather than erroring.

    Args:
        message: Text to centre on the empty canvas.

    Returns:
        An axis-free figure with a single annotation.
    """
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 14, "color": COLORS["neutral"]},
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=320,
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
    )
    return fig


# --------------------------------------------------------------------------- #
# 1. Counterparty exposure
# --------------------------------------------------------------------------- #


def exposure_by_counterparty_chart(
    df: pd.DataFrame,
    exposure: Optional[pd.DataFrame] = None,
    top_n: int = 15,
) -> go.Figure:
    """Bar chart of gross notional by counterparty, flagged names highlighted.

    A secondary line traces the cumulative share of notional, which is the
    quickest way to read how concentrated the book is.

    Args:
        df: Blotter with ``counterparty``, ``notional_value`` and optionally
            ``flag_counterparty_concentration``.
        exposure: Pre-computed exposure table from
            :func:`src.surveillance_rules.flag_counterparty_concentration`. When
            omitted it is derived from ``df``, and flags fall back to the
            trade-level flag column if present.
        top_n: How many counterparties to display, ranked by notional.

    Returns:
        A Plotly bar-plus-cumulative-line figure.
    """
    if df.empty:
        return empty_figure("No trades to chart")

    if exposure is None:
        exposure = (
            df.groupby("counterparty", as_index=False)
            .agg(notional=("notional_value", "sum"), trades=("notional_value", "size"))
            .sort_values("notional", ascending=False, ignore_index=True)
        )
        total = exposure["notional"].sum()
        exposure["exposure_pct"] = exposure["notional"] / total * 100 if total else 0.0
        exposure["cumulative_pct"] = exposure["exposure_pct"].cumsum()
        if "flag_counterparty_concentration" in df.columns:
            flagged = set(
                df.loc[df["flag_counterparty_concentration"], "counterparty"].unique()
            )
        else:
            flagged = set()
        exposure["flagged"] = exposure["counterparty"].isin(flagged)

    view = exposure.head(top_n).copy()
    colors = np.where(view["flagged"], COLORS["flagged"], COLORS["normal"])

    fig = go.Figure()
    fig.add_bar(
        x=view["counterparty"],
        y=view["notional"] / _MILLIONS,
        marker_color=colors,
        name="Gross notional",
        customdata=np.stack(
            [view["exposure_pct"], view["trades"], view["flagged"]], axis=-1
        ),
        hovertemplate=(
            "<b>%{x}</b><br>Notional: %{y:,.1f}m<br>"
            "Share of book: %{customdata[0]:.2f}%<br>"
            "Trades: %{customdata[1]:,}<extra></extra>"
        ),
    )
    fig.add_scatter(
        x=view["counterparty"],
        y=view["cumulative_pct"],
        yaxis="y2",
        mode="lines+markers",
        line={"color": COLORS["neutral"], "width": 2, "dash": "dot"},
        marker={"size": 5},
        name="Cumulative share",
        hovertemplate="Cumulative: %{y:.1f}%<extra></extra>",
    )
    _base_layout(fig, f"Gross notional by counterparty (top {len(view)})", height=460)
    fig.update_layout(
        yaxis={"title": "Notional (millions)"},
        yaxis2={
            "title": "Cumulative %",
            "overlaying": "y",
            "side": "right",
            "range": [0, 105],
            "showgrid": False,
        },
        xaxis={"tickangle": -35},
        bargap=0.25,
    )
    return fig


# --------------------------------------------------------------------------- #
# 2. Volume trend
# --------------------------------------------------------------------------- #


def trade_volume_trend_chart(
    df: pd.DataFrame,
    metric: str = "trade_count",
    freq: str = "W",
    by_desk: bool = False,
) -> go.Figure:
    """Time series of trading activity, optionally split by desk.

    Args:
        df: Blotter with ``trade_date`` and ``notional_value``.
        metric: ``"trade_count"`` or ``"notional"``.
        freq: Pandas offset alias for resampling — ``"D"``, ``"W"`` or ``"ME"``.
        by_desk: Draw one line per desk instead of a single total.

    Returns:
        A Plotly line figure.

    Raises:
        ValueError: On an unknown ``metric``.
    """
    if df.empty:
        return empty_figure("No trades in the selected period")
    if metric not in {"trade_count", "notional"}:
        raise ValueError(f"metric must be 'trade_count' or 'notional', got {metric!r}")

    is_count = metric == "trade_count"
    label = "Trade count" if is_count else "Notional (millions)"
    scale = 1.0 if is_count else _MILLIONS

    work = df.set_index("trade_date").sort_index()
    fig = go.Figure()

    if by_desk:
        for desk, group in work.groupby("desk"):
            series = (
                group["notional_value"].resample(freq).size()
                if is_count
                else group["notional_value"].resample(freq).sum()
            )
            fig.add_scatter(
                x=series.index,
                y=series.to_numpy() / scale,
                mode="lines",
                name=str(desk),
                line={"width": 2, "color": DESK_COLORS.get(str(desk), COLORS["normal"])},
                hovertemplate=f"<b>{desk}</b><br>%{{x|%d %b %Y}}<br>%{{y:,.1f}}<extra></extra>",
            )
        title = f"{label} over time, by desk"
    else:
        series = (
            work["notional_value"].resample(freq).size()
            if is_count
            else work["notional_value"].resample(freq).sum()
        )
        fig.add_scatter(
            x=series.index,
            y=series.to_numpy() / scale,
            mode="lines",
            name=label,
            line={"width": 2.5, "color": COLORS["normal"]},
            fill="tozeroy",
            fillcolor="rgba(31,119,180,0.12)",
            hovertemplate="%{x|%d %b %Y}<br>%{y:,.1f}<extra></extra>",
        )
        # Overlay flagged activity when the rules have already been applied.
        if "exception_count" in df.columns:
            flagged = work.loc[work["exception_count"] > 0, "notional_value"]
            if not flagged.empty:
                flagged_series = (
                    flagged.resample(freq).size()
                    if is_count
                    else flagged.resample(freq).sum()
                )
                fig.add_scatter(
                    x=flagged_series.index,
                    y=flagged_series.to_numpy() / scale,
                    mode="lines",
                    name="Flagged",
                    line={"width": 2, "color": COLORS["flagged"]},
                    hovertemplate="%{x|%d %b %Y}<br>Flagged: %{y:,.1f}<extra></extra>",
                )
        title = f"{label} over time"

    _base_layout(fig, title)
    fig.update_layout(yaxis={"title": label}, xaxis={"title": ""})
    return fig


# --------------------------------------------------------------------------- #
# 3. Exception breakdown
# --------------------------------------------------------------------------- #


def exception_summary_chart(
    df: pd.DataFrame,
    group_by: str = "desk",
    top_n: int = 12,
    flag_columns: Sequence[str] = FLAG_COLUMNS,
) -> go.Figure:
    """Stacked bar of exceptions per rule, broken down by any category.

    Args:
        df: Output of :func:`src.surveillance_rules.run_all_rules`.
        group_by: Column to break down by — ``"desk"``, ``"product_type"``,
            ``"counterparty"``, ``"trader"``.
        top_n: Cap on the number of groups shown, ranked by total exceptions.
        flag_columns: Which flag columns to stack.

    Returns:
        A stacked horizontal bar figure, one colour per rule.

    Raises:
        KeyError: If ``group_by`` or any flag column is missing.
    """
    missing = [c for c in (group_by, *flag_columns) if c not in df.columns]
    if missing:
        raise KeyError(f"exception_summary_chart is missing column(s): {missing}")
    if df.empty:
        return empty_figure("No trades to summarise")

    grouped = df.groupby(group_by)[list(flag_columns)].sum()
    grouped["_total"] = grouped.sum(axis=1)
    grouped = grouped.sort_values("_total", ascending=True).tail(top_n)

    if grouped["_total"].sum() == 0:
        return empty_figure("No exceptions raised under the current thresholds")

    # Warm palette for trade-level rules, cool for entity-level ones.
    palette = [
        COLORS["flagged"],
        COLORS["accent"],
        COLORS["flagged_soft"],
        COLORS["normal_soft"],
        "#a3312c",
        COLORS["neutral"],
    ]

    fig = go.Figure()
    for idx, column in enumerate(flag_columns):
        fig.add_bar(
            y=grouped.index.astype(str),
            x=grouped[column],
            name=FLAG_LABELS.get(column, column),
            orientation="h",
            marker_color=palette[idx % len(palette)],
            hovertemplate=(
                f"<b>%{{y}}</b><br>{FLAG_LABELS.get(column, column)}: "
                "%{x:,}<extra></extra>"
            ),
        )

    _base_layout(
        fig,
        f"Exceptions by {group_by.replace('_', ' ')}",
        height=max(420, 130 + 32 * len(grouped)),
    )
    # Six rule names will not fit on one line in a narrow column, and a wrapped
    # top legend collides with the title -- so this chart alone puts its legend
    # underneath the plot.
    fig.update_layout(
        barmode="stack",
        xaxis={"title": "Exceptions raised"},
        yaxis={"title": ""},
        margin={"l": 60, "r": 30, "t": 50, "b": 110},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.18,
            "xanchor": "left",
            "x": 0.0,
            "font": {"size": 11},
        },
    )
    return fig


# --------------------------------------------------------------------------- #
# 4. Trader / desk activity
# --------------------------------------------------------------------------- #


def trader_desk_activity_chart(
    df: pd.DataFrame,
    metric: str = "trade_count",
    highlight_flagged: bool = True,
) -> go.Figure:
    """Horizontal bar of trader activity, grouped and coloured by desk.

    Traders flagged by the volume rule get a red outline so the outlier is
    visible without losing the desk colour coding.

    Args:
        df: Blotter, optionally with ``flag_high_volume_trader``.
        metric: ``"trade_count"`` or ``"notional"``.
        highlight_flagged: Outline flagged traders in red.

    Returns:
        A Plotly bar figure sorted by activity.

    Raises:
        ValueError: On an unknown ``metric``.
    """
    if df.empty:
        return empty_figure("No trader activity for the current filters")
    if metric not in {"trade_count", "notional"}:
        raise ValueError(f"metric must be 'trade_count' or 'notional', got {metric!r}")

    is_count = metric == "trade_count"
    work = df.copy()
    if "flag_high_volume_trader" not in work.columns:
        work["flag_high_volume_trader"] = False

    agg = work.groupby(["desk", "trader"], as_index=False).agg(
        trade_count=("notional_value", "size"),
        notional=("notional_value", "sum"),
        flagged=("flag_high_volume_trader", "max"),
    )

    value = agg["trade_count"] if is_count else agg["notional"] / _MILLIONS
    agg["_value"] = value
    agg = agg.sort_values("_value", ascending=True, ignore_index=True)

    line_widths = (
        np.where(agg["flagged"], 2.5, 0.0) if highlight_flagged else np.zeros(len(agg))
    )

    fig = go.Figure()
    fig.add_bar(
        y=agg["trader"],
        x=agg["_value"],
        orientation="h",
        marker={
            "color": [DESK_COLORS.get(d, COLORS["normal"]) for d in agg["desk"]],
            "line": {"color": COLORS["flagged"], "width": line_widths},
        },
        customdata=np.stack([agg["desk"], agg["trade_count"], agg["notional"] / _MILLIONS], axis=-1),
        hovertemplate=(
            "<b>%{y}</b> (%{customdata[0]})<br>"
            "Trades: %{customdata[1]:,}<br>"
            "Notional: %{customdata[2]:,.1f}m<extra></extra>"
        ),
        showlegend=False,
    )

    # Legend proxies so the desk colour key is visible.
    for desk in agg["desk"].drop_duplicates():
        fig.add_bar(
            y=[None],
            x=[None],
            orientation="h",
            name=str(desk),
            marker_color=DESK_COLORS.get(str(desk), COLORS["normal"]),
            showlegend=True,
            hoverinfo="skip",
        )

    label = "Trade count" if is_count else "Notional (millions)"
    _base_layout(
        fig,
        f"Trader activity by desk — {label.lower()}"
        + (" (red outline = flagged)" if highlight_flagged else ""),
        height=max(380, 60 + 26 * len(agg)),
    )
    fig.update_layout(xaxis={"title": label}, yaxis={"title": ""}, barmode="overlay")
    return fig


# --------------------------------------------------------------------------- #
# 5. Settlement timing
# --------------------------------------------------------------------------- #


def settlement_timing_chart(
    df: pd.DataFrame,
    max_display_days: int = 30,
) -> go.Figure:
    """Distribution of business days to settlement, with breaches highlighted.

    FX Forwards legitimately settle weeks out, so the x-axis is clipped and
    anything beyond the cut-off is pooled into a final "N+" bucket; otherwise
    the cash-product spike at T+1/T+2 would be unreadable.

    Args:
        df: Blotter with ``days_to_settlement`` and, ideally,
            ``flag_settlement_risk``.
        max_display_days: Right-hand clip for the histogram.

    Returns:
        An overlaid histogram of normal versus flagged settlement lags.
    """
    if df.empty or "days_to_settlement" not in df.columns:
        return empty_figure("No settlement data for the current filters")

    days = df["days_to_settlement"].dropna()
    if days.empty:
        return empty_figure("No settlement dates available")

    flags = (
        df.loc[days.index, "flag_settlement_risk"]
        if "flag_settlement_risk" in df.columns
        else pd.Series(False, index=days.index)
    )
    clipped = days.clip(upper=max_display_days)

    fig = go.Figure()
    fig.add_histogram(
        x=clipped[~flags],
        name="Within expected window",
        marker_color=COLORS["normal"],
        opacity=0.85,
        xbins={"start": -2.5, "end": max_display_days + 0.5, "size": 1},
        hovertemplate="T+%{x}<br>%{y:,} trades<extra></extra>",
    )
    fig.add_histogram(
        x=clipped[flags],
        name="Outside expected window",
        marker_color=COLORS["flagged"],
        opacity=0.95,
        xbins={"start": -2.5, "end": max_display_days + 0.5, "size": 1},
        hovertemplate="T+%{x}<br>%{y:,} flagged<extra></extra>",
    )
    fig.add_vline(
        x=0,
        line_width=1.5,
        line_dash="dash",
        line_color=COLORS["neutral"],
        annotation_text="trade date",
        annotation_position="top",
    )

    # A log count axis is the only way the handful of breaches stay visible next
    # to the T+1/T+2 spike; the range floor is pushed below zero so that bins
    # containing a single trade still render with visible height.
    bins = np.arange(-2.5, max_display_days + 1.5, 1.0)
    tallest = int(np.histogram(clipped, bins=bins)[0].max())
    upper = float(np.log10(max(tallest, 10))) + 0.25

    _base_layout(fig, "Business days from trade date to settlement")
    fig.update_layout(
        barmode="overlay",
        xaxis={
            "title": f"Business days to settlement (clipped at {max_display_days}+)",
            "dtick": 2,
        },
        yaxis={"title": "Trades (log scale)", "type": "log", "range": [-0.25, upper]},
    )
    return fig
