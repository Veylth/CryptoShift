"""Dashboard components module."""

from .overview import render_overview_tab
from .anomaly_explorer import render_anomaly_explorer_tab
from .model_performance import render_performance_tab
from .alerts_table import render_alerts_table_tab
from .feature_analysis import render_feature_analysis_tab

__all__ = [
    "render_overview_tab",
    "render_anomaly_explorer_tab",
    "render_performance_tab",
    "render_alerts_table_tab",
    "render_feature_analysis_tab",
]
