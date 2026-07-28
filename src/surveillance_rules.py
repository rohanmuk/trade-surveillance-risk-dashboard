"""The five surveillance rules, plus an orchestrator that runs them all.

Each rule is a pure function: it takes the cleaned blotter and threshold
arguments and returns a :class:`RuleResult`. Nothing mutates its input and
nothing reads global state, so the rules are trivially unit-testable and can be
reused from the notebook as easily as from the Streamlit app.

Every threshold is an argument with a default drawn from
:class:`src.config.SurveillanceConfig` — there are no magic numbers buried in
the rule bodies, which is what lets the app expose them as sidebar controls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_SETTLEMENT_WINDOW,
    FLAG_COLUMNS,
    PRODUCT_SETTLEMENT_WINDOW,
    SurveillanceConfig,
)

__all__ = [
    "RuleResult",
    "flag_large_notional",
    "flag_counterparty_concentration",
    "flag_high_trade_volume",
    "flag_late_amendment_cancellation",
    "flag_settlement_risk",
    "run_all_rules",
    "exception_summary",
]


@dataclass
class RuleResult:
    """The output of a single surveillance rule.

    Attributes:
        flags: Boolean Series aligned to the input frame's index; ``True``
            means the trade is an exception under this rule.
        summary: Rule-specific breakdown — one row per counterparty, per
            trader, per product, and so on. Always non-empty in shape even when
            no trade is flagged.
        details: Scalar diagnostics (effective threshold, HHI, and similar)
            that the UI surfaces alongside the table.
    """

    flags: pd.Series
    summary: pd.DataFrame
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_flagged(self) -> int:
        """Number of trades flagged by this rule."""
        return int(self.flags.sum())


def _require_columns(df: pd.DataFrame, columns: Sequence[str], rule: str) -> None:
    """Raise a clear error if ``df`` is missing any column ``rule`` needs."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"{rule} requires column(s) {', '.join(missing)}; "
            "did you run clean_trades() first?"
        )


def _empty_flags(df: pd.DataFrame) -> pd.Series:
    """An all-False boolean Series aligned to ``df``."""
    return pd.Series(False, index=df.index, dtype=bool)


# --------------------------------------------------------------------------- #
# Rule 1 -- large notional
# --------------------------------------------------------------------------- #


def flag_large_notional(
    df: pd.DataFrame,
    threshold: Optional[float] = None,
    percentile: float = 0.99,
    mode: str = "percentile",
    by_product: bool = False,
) -> RuleResult:
    """Flag trades whose notional is unusually large.

    Two modes are supported. ``"absolute"`` compares against a fixed currency
    threshold, which is how a limits framework is usually written. ``"percentile"``
    compares against a quantile of the observed distribution, which adapts to
    the book and is the better default when products span very different
    notional scales.

    Args:
        df: Cleaned blotter.
        threshold: Absolute notional cut-off. Required when ``mode="absolute"``;
            ignored otherwise.
        percentile: Quantile in ``(0, 1)`` used when ``mode="percentile"``.
        mode: ``"percentile"`` or ``"absolute"``.
        by_product: When ``True`` in percentile mode, the quantile is taken
            within each ``product_type`` so a large swap is judged against
            other swaps rather than against equities.

    Returns:
        :class:`RuleResult` whose ``summary`` has one row per product type with
        the effective threshold and flag count.

    Raises:
        ValueError: On an unknown mode, a percentile outside ``(0, 1)``, or a
            missing/non-positive absolute threshold.
        KeyError: If required columns are absent.
    """
    _require_columns(df, ["notional_value", "product_type"], "flag_large_notional")

    if mode not in {"percentile", "absolute"}:
        raise ValueError(
            f"mode must be 'percentile' or 'absolute', got {mode!r}"
        )
    if mode == "percentile" and not 0.0 < percentile < 1.0:
        raise ValueError(f"percentile must be strictly between 0 and 1, got {percentile}")
    if mode == "absolute":
        if threshold is None:
            raise ValueError("threshold is required when mode='absolute'")
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")

    notional = df["notional_value"]

    if mode == "absolute":
        cutoffs = pd.Series(float(threshold), index=df.index)
    elif by_product:
        cutoffs = df.groupby("product_type")["notional_value"].transform(
            lambda s: s.quantile(percentile)
        )
    else:
        cutoffs = pd.Series(float(notional.quantile(percentile)), index=df.index)

    flags = (notional > cutoffs).fillna(False).astype(bool)

    summary = (
        df.assign(_flag=flags, _cutoff=cutoffs)
        .groupby("product_type", as_index=False)
        .agg(
            trades=("notional_value", "size"),
            threshold=("_cutoff", "max"),
            median_notional=("notional_value", "median"),
            max_notional=("notional_value", "max"),
            flagged_trades=("_flag", "sum"),
            flagged_notional=("notional_value", lambda s: s[flags.loc[s.index]].sum()),
        )
        .sort_values("flagged_notional", ascending=False, ignore_index=True)
    )

    return RuleResult(
        flags=flags,
        summary=summary,
        details={
            "mode": mode,
            "percentile": percentile if mode == "percentile" else None,
            "by_product": by_product,
            "effective_threshold": float(cutoffs.min()) if by_product else float(cutoffs.iloc[0]),
            "flagged_notional": float(notional[flags].sum()),
        },
    )


# --------------------------------------------------------------------------- #
# Rule 2 -- counterparty concentration
# --------------------------------------------------------------------------- #


def flag_counterparty_concentration(
    df: pd.DataFrame,
    top_n: int = 5,
    exposure_pct_threshold: float = 7.5,
    combine: str = "or",
) -> RuleResult:
    """Flag trades facing counterparties that dominate the book's notional.

    Concentration is measured as each counterparty's share of gross notional.
    A counterparty is flagged if its share exceeds ``exposure_pct_threshold``,
    if it ranks inside the top ``top_n``, or both, depending on ``combine``.

    Args:
        df: Cleaned blotter.
        top_n: Number of top counterparties by notional to treat as concentrated.
            Set to ``0`` to rely on the percentage threshold alone.
        exposure_pct_threshold: Percent-of-total-notional threshold (0-100).
        combine: How to combine the two criteria — ``"or"`` (default),
            ``"and"``, or ``"threshold"`` to ignore the rank test entirely.

    Returns:
        :class:`RuleResult` whose ``summary`` is the full exposure table (one
        row per counterparty, ranked) and whose ``details`` carries the
        Herfindahl-Hirschman Index of the notional distribution.

    Raises:
        ValueError: On a negative ``top_n``, a threshold outside 0-100, or an
            unknown ``combine`` value.
        KeyError: If required columns are absent.
    """
    _require_columns(
        df, ["counterparty", "notional_value"], "flag_counterparty_concentration"
    )
    if top_n < 0:
        raise ValueError(f"top_n must be >= 0, got {top_n}")
    if not 0.0 <= exposure_pct_threshold <= 100.0:
        raise ValueError(
            f"exposure_pct_threshold must be between 0 and 100, got {exposure_pct_threshold}"
        )
    if combine not in {"or", "and", "threshold"}:
        raise ValueError(f"combine must be 'or', 'and' or 'threshold', got {combine!r}")

    exposure = (
        df.groupby("counterparty", as_index=False)
        .agg(
            notional=("notional_value", "sum"),
            trades=("notional_value", "size"),
            avg_notional=("notional_value", "mean"),
        )
        .sort_values("notional", ascending=False, ignore_index=True)
    )

    total = float(exposure["notional"].sum())
    exposure["exposure_pct"] = (
        exposure["notional"] / total * 100.0 if total > 0 else 0.0
    )
    exposure["cumulative_pct"] = exposure["exposure_pct"].cumsum()
    exposure["rank"] = np.arange(1, len(exposure) + 1)

    above_threshold = exposure["exposure_pct"] > exposure_pct_threshold
    in_top_n = exposure["rank"] <= top_n

    if combine == "or":
        exposure["flagged"] = above_threshold | in_top_n
    elif combine == "and":
        exposure["flagged"] = above_threshold & in_top_n
    else:
        exposure["flagged"] = above_threshold

    flagged_names = set(exposure.loc[exposure["flagged"], "counterparty"])
    flags = df["counterparty"].isin(flagged_names)

    # HHI on percentage shares: 10,000 = single counterparty, ~0 = fully diverse.
    hhi = float((exposure["exposure_pct"] ** 2).sum())

    return RuleResult(
        flags=flags,
        summary=exposure,
        details={
            "total_notional": total,
            "hhi": hhi,
            "n_counterparties": int(len(exposure)),
            "n_flagged_counterparties": int(exposure["flagged"].sum()),
            "top1_pct": float(exposure["exposure_pct"].iloc[0]) if len(exposure) else 0.0,
            "top5_pct": float(exposure["exposure_pct"].head(5).sum()),
            "top_n": top_n,
            "exposure_pct_threshold": exposure_pct_threshold,
        },
    )


# --------------------------------------------------------------------------- #
# Rule 3 -- high trade volume
# --------------------------------------------------------------------------- #


def flag_high_trade_volume(
    df: pd.DataFrame,
    group_by: str = "trader",
    metric: str = "trade_count",
    method: str = "zscore",
    z_threshold: float = 2.0,
    iqr_multiplier: float = 1.5,
) -> RuleResult:
    """Flag traders or desks whose activity is an outlier within their peer group.

    Args:
        df: Cleaned blotter.
        group_by: Column to aggregate over — typically ``"trader"`` or ``"desk"``.
        metric: ``"trade_count"`` (number of trades) or ``"notional"`` (gross
            notional traded).
        method: ``"zscore"`` for a mean/standard-deviation test, or ``"iqr"``
            for a Tukey upper-fence test, which is more robust when one group
            is extreme enough to inflate the standard deviation.
        z_threshold: Z-score above which a group is an outlier.
        iqr_multiplier: Fence multiplier ``Q3 + k * IQR`` when ``method="iqr"``.

    Returns:
        :class:`RuleResult`. ``flags`` is at trade level (every trade belonging
        to an outlier group is flagged); ``summary`` is one row per group with
        its metric, z-score and flag.

    Raises:
        ValueError: On an unknown ``metric`` or ``method``.
        KeyError: If ``group_by`` is not a column of ``df``.

    Note:
        The z-score uses the sample standard deviation (``ddof=1``), for which
        the largest attainable score across *n* groups is ``sqrt(n - 1)``. With
        five desks that ceiling is 2.00, so a threshold of 2.0 can never fire on
        desks no matter how skewed the book is. Use a lower threshold (the app
        defaults desks to 1.2) or the IQR method for small group counts.
    """
    _require_columns(df, [group_by, "notional_value"], "flag_high_trade_volume")
    if metric not in {"trade_count", "notional"}:
        raise ValueError(f"metric must be 'trade_count' or 'notional', got {metric!r}")
    if method not in {"zscore", "iqr"}:
        raise ValueError(f"method must be 'zscore' or 'iqr', got {method!r}")

    grouped = (
        df.groupby(group_by, as_index=False)
        .agg(
            trade_count=("notional_value", "size"),
            notional=("notional_value", "sum"),
            avg_notional=("notional_value", "mean"),
        )
        .rename(columns={group_by: "group"})
    )
    grouped.insert(0, "group_by", group_by)

    values = grouped["trade_count"] if metric == "trade_count" else grouped["notional"]
    grouped["metric_value"] = values

    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    grouped["z_score"] = (values - mean) / std if std > 0 else 0.0

    q1, q3 = float(values.quantile(0.25)), float(values.quantile(0.75))
    iqr = q3 - q1
    upper_fence = q3 + iqr_multiplier * iqr

    if method == "zscore":
        grouped["flagged"] = grouped["z_score"] > z_threshold
        cutoff = mean + z_threshold * std if std > 0 else float("inf")
    else:
        grouped["flagged"] = values > upper_fence
        cutoff = upper_fence

    grouped = grouped.sort_values("metric_value", ascending=False, ignore_index=True)

    flagged_groups = set(grouped.loc[grouped["flagged"], "group"])
    flags = df[group_by].isin(flagged_groups)

    max_attainable_z = float(np.sqrt(len(values) - 1)) if len(values) > 1 else 0.0

    return RuleResult(
        flags=flags,
        summary=grouped,
        details={
            "group_by": group_by,
            "metric": metric,
            "method": method,
            "mean": mean,
            "std": std,
            "cutoff": float(cutoff),
            "upper_fence": upper_fence,
            "n_groups": int(len(grouped)),
            "n_flagged_groups": int(grouped["flagged"].sum()),
            "max_attainable_z": max_attainable_z,
        },
    )


# --------------------------------------------------------------------------- #
# Rule 4 -- late amendments and cancellations
# --------------------------------------------------------------------------- #


def flag_late_amendment_cancellation(
    df: pd.DataFrame,
    business_days_threshold: int = 2,
) -> RuleResult:
    """Flag lifecycle events that happened long after the trade was struck.

    An amendment or cancellation booked days after the trade date is an
    operational-risk signal: it can indicate a booking error caught late, a
    mismarked economic term, or — in the worst case — a position being rewritten
    after the fact. The lag is measured in business days from ``trade_date`` to
    ``amendment_datetime``.

    Args:
        df: Cleaned blotter (must include the derived ``days_to_amendment``).
        business_days_threshold: Lag, in business days, above which the event is
            late. ``2`` means "later than T+2".

    Returns:
        :class:`RuleResult` whose ``summary`` breaks late events down by desk
        and event type.

    Raises:
        ValueError: If the threshold is negative.
        KeyError: If required columns are absent.
    """
    _require_columns(
        df,
        ["days_to_amendment", "amendment_flag", "cancellation_flag", "desk", "trade_status"],
        "flag_late_amendment_cancellation",
    )
    if business_days_threshold < 0:
        raise ValueError(
            f"business_days_threshold must be >= 0, got {business_days_threshold}"
        )

    has_event = df["amendment_flag"] | df["cancellation_flag"]
    lag = df["days_to_amendment"]
    flags = (has_event & (lag > business_days_threshold)).fillna(False).astype(bool)

    event_type = np.where(
        df["cancellation_flag"], "Cancellation", np.where(has_event, "Amendment", "None")
    )

    detail = df.assign(_flag=flags, _event=event_type, _lag=lag).loc[has_event]
    summary = (
        detail.groupby(["desk", "_event"], as_index=False)
        .agg(
            events=("_flag", "size"),
            late_events=("_flag", "sum"),
            median_lag_days=("_lag", "median"),
            max_lag_days=("_lag", "max"),
        )
        .rename(columns={"_event": "event_type"})
        .sort_values("late_events", ascending=False, ignore_index=True)
    )
    if not summary.empty:
        summary["late_rate_pct"] = summary["late_events"] / summary["events"] * 100.0

    return RuleResult(
        flags=flags,
        summary=summary,
        details={
            "business_days_threshold": business_days_threshold,
            "n_lifecycle_events": int(has_event.sum()),
            "n_amendments": int((has_event & ~df["cancellation_flag"]).sum()),
            "n_cancellations": int(df["cancellation_flag"].sum()),
            "median_lag_days": float(lag[has_event].median()) if has_event.any() else float("nan"),
            "late_event_rate_pct": (
                float(flags.sum() / has_event.sum() * 100.0) if has_event.any() else 0.0
            ),
        },
    )


# --------------------------------------------------------------------------- #
# Rule 5 -- settlement risk
# --------------------------------------------------------------------------- #


def flag_settlement_risk(
    df: pd.DataFrame,
    min_days: int = 1,
    max_days: int = 3,
    product_windows: Optional[Mapping[str, "tuple[int, int]"]] = PRODUCT_SETTLEMENT_WINDOW,
    default_window: "tuple[int, int]" = DEFAULT_SETTLEMENT_WINDOW,
) -> RuleResult:
    """Flag trades settling outside the expected window for their product.

    Both tails matter. Settling too fast (or before the trade date, which is
    impossible in practice) points at a booking error; settling far beyond
    convention points at a fail, a mis-keyed date, or an off-market bilateral
    arrangement that ought to be reviewed.

    Args:
        df: Cleaned blotter (must include the derived ``days_to_settlement``).
        min_days: Minimum acceptable business days to settlement, used when
            ``product_windows`` is ``None`` or a product is unmapped.
        max_days: Maximum acceptable business days to settlement, same caveat.
        product_windows: Per-product ``(min, max)`` business-day windows. Pass
            ``None`` to apply the flat ``min_days``/``max_days`` to everything.
        default_window: Fallback window for products absent from the mapping.

    Returns:
        :class:`RuleResult` whose ``summary`` has one row per product with the
        applied window and a split of too-fast versus too-slow breaches.

    Raises:
        ValueError: If ``min_days > max_days``.
        KeyError: If required columns are absent.
    """
    _require_columns(
        df, ["days_to_settlement", "product_type"], "flag_settlement_risk"
    )
    if min_days > max_days:
        raise ValueError(
            f"min_days ({min_days}) must not exceed max_days ({max_days})"
        )

    products = df["product_type"]
    if product_windows is None:
        lower = pd.Series(float(min_days), index=df.index)
        upper = pd.Series(float(max_days), index=df.index)
    else:
        fallback = default_window if default_window is not None else (min_days, max_days)
        lower = products.map(
            lambda p: float(product_windows.get(p, fallback)[0])
        )
        upper = products.map(
            lambda p: float(product_windows.get(p, fallback)[1])
        )

    days = df["days_to_settlement"]
    too_fast = (days < lower).fillna(False)
    too_slow = (days > upper).fillna(False)
    missing = days.isna()
    flags = (too_fast | too_slow | missing).astype(bool)

    summary = (
        df.assign(
            _fast=too_fast, _slow=too_slow, _missing=missing, _lo=lower, _hi=upper
        )
        .groupby("product_type", as_index=False)
        .agg(
            trades=("days_to_settlement", "size"),
            expected_min=("_lo", "min"),
            expected_max=("_hi", "max"),
            median_days=("days_to_settlement", "median"),
            too_fast=("_fast", "sum"),
            too_slow=("_slow", "sum"),
            missing_date=("_missing", "sum"),
        )
    )
    summary["flagged_trades"] = (
        summary["too_fast"] + summary["too_slow"] + summary["missing_date"]
    )
    summary = summary.sort_values(
        "flagged_trades", ascending=False, ignore_index=True
    )

    return RuleResult(
        flags=flags,
        summary=summary,
        details={
            "uses_product_windows": product_windows is not None,
            "min_days": min_days,
            "max_days": max_days,
            "n_too_fast": int(too_fast.sum()),
            "n_too_slow": int(too_slow.sum()),
            "n_missing": int(missing.sum()),
            "n_before_trade_date": int((days < 0).fillna(False).sum()),
        },
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def run_all_rules(
    df: pd.DataFrame,
    config: Optional[SurveillanceConfig] = None,
) -> pd.DataFrame:
    """Apply all five rules and return the blotter with flag columns appended.

    The high-volume rule is applied twice — once grouped by trader and once by
    desk — because a quiet desk can still hide a hyperactive trader and vice
    versa. That yields the six flag columns listed in
    :data:`src.config.FLAG_COLUMNS`.

    Args:
        df: Cleaned blotter.
        config: Thresholds. Defaults to :class:`SurveillanceConfig` defaults.

    Returns:
        A copy of ``df`` with one boolean column per flag plus an integer
        ``exception_count`` (the row-wise sum). The individual
        :class:`RuleResult` objects are attached as
        ``df.attrs["rule_results"]`` for callers that want the summaries.

    Raises:
        KeyError: If the frame has not been through :func:`clean_trades`.
    """
    cfg = config or SurveillanceConfig()
    out = df.copy()

    if out.empty:
        for col in FLAG_COLUMNS:
            out[col] = pd.Series(dtype=bool)
        out["exception_count"] = pd.Series(dtype="int64")
        out.attrs["rule_results"] = {}
        return out

    results: Dict[str, RuleResult] = {}

    results["flag_large_notional"] = flag_large_notional(
        out,
        threshold=cfg.large_notional_threshold,
        percentile=cfg.large_notional_percentile,
        mode=cfg.large_notional_mode,
    )
    results["flag_counterparty_concentration"] = flag_counterparty_concentration(
        out,
        top_n=cfg.concentration_top_n,
        exposure_pct_threshold=cfg.concentration_pct_threshold,
    )
    results["flag_high_volume_trader"] = flag_high_trade_volume(
        out,
        group_by="trader",
        metric=cfg.volume_metric,
        method=cfg.volume_method,
        z_threshold=cfg.trader_z_threshold,
        iqr_multiplier=cfg.volume_iqr_multiplier,
    )
    results["flag_high_volume_desk"] = flag_high_trade_volume(
        out,
        group_by="desk",
        metric=cfg.volume_metric,
        method=cfg.volume_method,
        z_threshold=cfg.desk_z_threshold,
        iqr_multiplier=cfg.volume_iqr_multiplier,
    )
    results["flag_late_amendment"] = flag_late_amendment_cancellation(
        out, business_days_threshold=cfg.late_amendment_business_days
    )
    results["flag_settlement_risk"] = flag_settlement_risk(
        out,
        min_days=cfg.settlement_min_days,
        max_days=cfg.settlement_max_days,
        product_windows=(
            PRODUCT_SETTLEMENT_WINDOW if cfg.settlement_use_product_windows else None
        ),
    )

    for column in FLAG_COLUMNS:
        out[column] = results[column].flags.astype(bool).to_numpy()

    out["exception_count"] = out[list(FLAG_COLUMNS)].sum(axis=1).astype("int64")
    out.attrs["rule_results"] = results
    out.attrs.setdefault("cleaning_report", df.attrs.get("cleaning_report"))
    return out


def exception_summary(df: pd.DataFrame, group_by: str = "desk") -> pd.DataFrame:
    """Aggregate exception counts and rates by any categorical column.

    Args:
        df: Output of :func:`run_all_rules`.
        group_by: Column to group on, e.g. ``"desk"``, ``"product_type"``,
            ``"counterparty"``, ``"trader"``.

    Returns:
        One row per group with trade count, notional, count per rule, total
        exceptions, number of trades with at least one exception, and the
        exception rate as a percentage of trades.

    Raises:
        KeyError: If ``group_by`` or the flag columns are absent.
    """
    _require_columns(df, [group_by, "exception_count", *FLAG_COLUMNS], "exception_summary")

    aggregations: Dict[str, "tuple[str, str]"] = {
        "trades": ("trade_id", "size"),
        "notional": ("notional_value", "sum"),
    }
    aggregations.update({col: (col, "sum") for col in FLAG_COLUMNS})
    aggregations["total_exceptions"] = ("exception_count", "sum")

    summary = df.groupby(group_by, as_index=False).agg(**aggregations)
    flagged_trades = (
        df.assign(_any=df["exception_count"] > 0)
        .groupby(group_by, as_index=False)["_any"]
        .sum()
        .rename(columns={"_any": "flagged_trades"})
    )
    summary = summary.merge(flagged_trades, on=group_by, how="left")
    summary["exception_rate_pct"] = (
        summary["flagged_trades"] / summary["trades"] * 100.0
    )
    return summary.sort_values(
        "total_exceptions", ascending=False, ignore_index=True
    )
