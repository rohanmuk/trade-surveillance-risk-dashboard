"""Trade Surveillance Risk Dashboard — analytics package.

Layout:
    :mod:`src.config`              paths, domain constants, default thresholds
    :mod:`src.data_cleaning`       load, validate and enrich the blotter
    :mod:`src.surveillance_rules`  the five rules and the orchestrator
    :mod:`src.visualizations`      Plotly figure builders (Streamlit-free)
    :mod:`src.sqlite_views`        SQL-backed aggregates over the cleaned data
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = [
    "config",
    "data_cleaning",
    "surveillance_rules",
    "visualizations",
    "sqlite_views",
]
