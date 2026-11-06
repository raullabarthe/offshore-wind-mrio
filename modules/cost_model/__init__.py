"""
Cost Model Module
=================
Offshore wind farm cost modeling: inputs, calculations, and visualization.

Structure
---------
inputs
    Reference data: BVG benchmark, NREL data, design parameters (TurbineDesign, PlantConfig)

helpers
    Utility functions: conversions, aggregations (sum_block_kg, kg_to_t, etc.)

model
    Core calculations: turbine masses, cost breakdowns, Devex/BoP/Installation/OPEX/ABEX

cost_main
    Legacy compatibility layer

charts
    Visualization helpers and saved plots
"""

# Input data & parameters
from .inputs import (
    costs_bvg,
    NREL_ROWS,
    TurbineBVG,
    SiteBVG,
    PlantBVG,
    TurbineDesign,
    PlantConfig,
)

# Helper utilities

# Core model calculations
from .model import (
    turbine_masses,
    build_turbine_supply,
    build_cost_breakdown,
    calculate_devex,
    calculate_bop,
    calculate_installation,
    calculate_opex,
    calculate_abex,
)

# Visualization & export
from .charts import (
    PARENTS_TABLEAU,
    build_bvg_rows_from_costs_bvg,
    build_chart_df_from_model,
    build_chart_df_from_snap,
    plot_model_double_donut,
)

__all__ = [
    # Inputs
    "costs_bvg",
    "NREL_ROWS",
    "TurbineBVG",
    "SiteBVG",
    "PlantBVG",
    "TurbineDesign",
    "PlantConfig",
    # Helpers
    # Model calculations
    "turbine_masses",
    "build_turbine_supply",
    "build_cost_breakdown",
    "calculate_devex",
    "calculate_bop",
    "calculate_installation",
    "calculate_opex",
    "calculate_abex",
    # Charts
    "PARENTS_TABLEAU",
    "build_bvg_rows_from_costs_bvg",
    "build_chart_df_from_model",
    "build_chart_df_from_snap",
    "plot_model_double_donut",
]

__version__ = "2.0.0"
__author__ = "Offshore wind value-chain MRIO research project"

