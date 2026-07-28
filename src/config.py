"""Central configuration: paths, domain constants, default thresholds, colours.

Every filesystem path and every "magic number" used by the surveillance rules
lives here so that nothing downstream hardcodes a path or an unexplained
constant. Paths are all derived from :data:`PROJECT_ROOT`, which is resolved
relative to this file, so the project is portable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
IMAGES_DIR: Path = PROJECT_ROOT / "images"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
TRADES_CSV: Path = DATA_DIR / "simulated_trades.csv"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #

RANDOM_SEED: int = 20260728

# --------------------------------------------------------------------------- #
# Domain vocabulary
#
# These are the canonical spellings for the categorical columns. Cleaning
# normalises incoming values against them case-insensitively, which is why we
# can keep correct capitalisation such as "FX Forward" rather than naively
# title-casing everything.
# --------------------------------------------------------------------------- #

DESKS: Tuple[str, ...] = ("Rates", "FX", "Credit", "Equities", "Commodities")

PRODUCT_TYPES: Tuple[str, ...] = ("Bond", "Swap", "FX Forward", "Equity", "Option")

BUY_SELL: Tuple[str, ...] = ("Buy", "Sell")

TRADE_STATUSES: Tuple[str, ...] = ("Booked", "Amended", "Cancelled")

#: Which products each desk is allowed to trade, with sampling weights.
DESK_PRODUCT_MIX: Dict[str, Dict[str, float]] = {
    "Rates": {"Bond": 0.55, "Swap": 0.45},
    "FX": {"FX Forward": 0.80, "Option": 0.20},
    "Credit": {"Bond": 0.65, "Swap": 0.35},
    "Equities": {"Equity": 0.75, "Option": 0.25},
    "Commodities": {"Swap": 0.55, "Option": 0.45},
}

# --------------------------------------------------------------------------- #
# Settlement conventions
#
# `days_to_settlement` is measured in BUSINESS days (np.busday_count), so these
# windows read as "T+min .. T+max". FX Forwards are genuinely forward-dated
# (1M / 2M / 3M), which is why their window is far wider than the cash
# products'.
# --------------------------------------------------------------------------- #

#: Normal settlement lag, in business days, used by the data generator.
PRODUCT_SETTLEMENT_LAG: Dict[str, Tuple[int, ...]] = {
    "Bond": (1,),
    "Swap": (2,),
    "Equity": (1,),
    "Option": (1,),
    "FX Forward": (21, 42, 63),
}

#: Inclusive (min, max) business-day window a product is expected to settle in.
PRODUCT_SETTLEMENT_WINDOW: Dict[str, Tuple[int, int]] = {
    "Bond": (1, 2),
    "Swap": (1, 3),
    "Equity": (1, 2),
    "Option": (1, 2),
    "FX Forward": (15, 70),
}

#: Fallback window for any product not present in the map above.
DEFAULT_SETTLEMENT_WINDOW: Tuple[int, int] = (1, 3)

# --------------------------------------------------------------------------- #
# Default rule thresholds
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SurveillanceConfig:
    """Tunable thresholds for the five surveillance rules.

    The dataclass is frozen so it is hashable, which lets Streamlit use it as a
    ``st.cache_data`` key. Every field is a scalar for the same reason.

    Attributes:
        large_notional_mode: ``"percentile"`` or ``"absolute"``.
        large_notional_percentile: Quantile in ``(0, 1)`` used in percentile mode.
        large_notional_threshold: Absolute notional cut-off used in absolute mode.
        concentration_top_n: Counterparties ranked in the top N are flagged.
        concentration_pct_threshold: Percent-of-total-notional flagging threshold.
        volume_metric: ``"trade_count"`` or ``"notional"``.
        volume_method: ``"zscore"`` or ``"iqr"``.
        trader_z_threshold: Z-score cut-off when grouping by trader.
        desk_z_threshold: Z-score cut-off when grouping by desk (see note below).
        volume_iqr_multiplier: Tukey fence multiplier when ``volume_method="iqr"``.
        late_amendment_business_days: Amendments later than this many business
            days after the trade date are flagged.
        settlement_use_product_windows: Use :data:`PRODUCT_SETTLEMENT_WINDOW`
            instead of the flat min/max below.
        settlement_min_days: Flat minimum business days to settlement.
        settlement_max_days: Flat maximum business days to settlement.

    Note:
        ``desk_z_threshold`` defaults lower than ``trader_z_threshold`` on
        purpose. With a sample standard deviation the largest attainable
        z-score for *n* groups is ``sqrt(n - 1)``; with five desks that ceiling
        is 2.00, so a 2.0 threshold could never fire. See
        :func:`src.surveillance_rules.flag_high_trade_volume`.
    """

    large_notional_mode: str = "percentile"
    large_notional_percentile: float = 0.99
    large_notional_threshold: float = 25_000_000.0

    concentration_top_n: int = 5
    concentration_pct_threshold: float = 7.5

    volume_metric: str = "trade_count"
    volume_method: str = "zscore"
    trader_z_threshold: float = 2.0
    desk_z_threshold: float = 1.2
    volume_iqr_multiplier: float = 1.5

    late_amendment_business_days: int = 2

    settlement_use_product_windows: bool = True
    settlement_min_days: int = 1
    settlement_max_days: int = 3


#: Column names added by :func:`src.surveillance_rules.run_all_rules`, in the
#: order they are appended. ``exception_count`` is the row-wise sum of these.
FLAG_COLUMNS: Tuple[str, ...] = (
    "flag_large_notional",
    "flag_counterparty_concentration",
    "flag_high_volume_trader",
    "flag_high_volume_desk",
    "flag_late_amendment",
    "flag_settlement_risk",
)

#: Flags raised by something intrinsic to the individual trade. These are the
#: ones an ops analyst can action ticket-by-ticket, so the headline "exception
#: rate" is computed from this subset.
TRADE_LEVEL_FLAGS: Tuple[str, ...] = (
    "flag_large_notional",
    "flag_late_amendment",
    "flag_settlement_risk",
)

#: Flags raised by the behaviour of an *entity* (a counterparty, a trader, a
#: desk) rather than by the trade itself. They necessarily fan out across every
#: trade that entity touched, so quoting them as a per-trade rate overstates the
#: alert load — they are reported as counts of flagged entities instead.
ENTITY_LEVEL_FLAGS: Tuple[str, ...] = (
    "flag_counterparty_concentration",
    "flag_high_volume_trader",
    "flag_high_volume_desk",
)

#: Human-readable labels for the flag columns, used in the UI and charts.
FLAG_LABELS: Dict[str, str] = {
    "flag_large_notional": "Large notional",
    "flag_counterparty_concentration": "Counterparty concentration",
    "flag_high_volume_trader": "High volume (trader)",
    "flag_high_volume_desk": "High volume (desk)",
    "flag_late_amendment": "Late amendment / cancellation",
    "flag_settlement_risk": "Settlement risk",
}

# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #

#: Shared colour scheme: red/orange for flagged, blue/grey for normal.
COLORS: Dict[str, str] = {
    "flagged": "#d62728",
    "flagged_soft": "#ff9f43",
    "normal": "#1f77b4",
    "normal_soft": "#9aa5b1",
    "neutral": "#4a5568",
    "accent": "#2c7fb8",
    "grid": "#e2e8f0",
}

#: Consistent qualitative palette for desk-level series.
DESK_COLORS: Dict[str, str] = {
    "Rates": "#1f77b4",
    "FX": "#2c7fb8",
    "Credit": "#6b7fa3",
    "Equities": "#4a5568",
    "Commodities": "#9aa5b1",
}

PLOTLY_TEMPLATE: str = "plotly_white"

#: Row cap applied to the SQL-backed exposure view in the app.
QUERY_ROW_LIMIT: int = 25
