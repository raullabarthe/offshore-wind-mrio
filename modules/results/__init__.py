"""
Results analysis module for Impact Analysis section.

Contains functions for:
- Building the paper summary indicator table
- Computing strategic ratios
- Creating capture plots (absolute, average, scatter, etc.)
"""

from .results import (
    # Table builders
    build_all_indicators_table_effect_and_comp_columns_from_by_origin,

    # Plotting functions
    plot_and_save_capture_sublevel_avg_connected_power,
    plot_and_save_eu27_capture_share_figures,

    # Strategic ratios
    compute_strategic_ratios,
    compute_strategic_ratios_subcomponents,

)

__all__ = [
    # Table builders
    "build_all_indicators_table_effect_and_comp_columns_from_by_origin",

    # Plotting functions
    "plot_and_save_capture_sublevel_avg_connected_power",
    "plot_and_save_eu27_capture_share_figures",

    # Strategic ratios
    "compute_strategic_ratios",
    "compute_strategic_ratios_subcomponents",
]
