"""Cost-model input data, design defaults, benchmark costs, and FX helpers."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET
import pandas as pd


# ============================================================================
# EXCHANGE RATES LOAD
# ============================================================================

# Sources: 
# https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/eurofxref-graph-usd.en.html
# https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/eurofxref-graph-gbp.en.html

def parse_ecb_fx_xml(file_path: Path) -> pd.DataFrame:
    """
    Parse an ECB SDMX CompactData FX XML file and return:
    columns: date, year, value, currency
    """

    tree = ET.parse(file_path)
    root = tree.getroot()

    # Helper to remove namespace prefix
    def tag_name(elem):
        return elem.tag.split("}")[-1]

    # Extract currency code from <Series> or <Group>
    currency_code = None
    for elem in root.iter():
        if tag_name(elem) in {"Series", "Group"} and "CURRENCY" in elem.attrib:
            currency_code = elem.attrib["CURRENCY"]
            break
    if currency_code is None:
        currency_code = file_path.stem.upper()

    rows = []

    # Parse all <Obs>
    for elem in root.iter():
        if tag_name(elem) != "Obs":
            continue

        date_str = elem.attrib.get("TIME_PERIOD")
        value_str = elem.attrib.get("OBS_VALUE")

        if not date_str or not value_str:
            continue

        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue

        try:
            value = float(value_str)
        except ValueError:
            continue

        rows.append(
            {
                "date": dt,
            "year": dt.year,
                "value": value,
                "currency": currency_code,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["date", "year", "value", "currency"])
    return pd.DataFrame(rows)


def build_fx_dictionary(base_dir: Path | str = None) -> dict:
    """
    Walk exchange rates directory, read all XML files, compute yearly average FX,
    and return a dictionary:
        {currency: {year: avg_rate}}
    
    Args:
        base_dir: Path to exchange rates directory. If None, uses EXCHANGE_ROOT from config.
                  If that doesn't exist, returns empty dict.
    """
    
    # Import config here to avoid circular imports
    try:
        from ..config import EXCHANGE_ROOT
        default_dir = EXCHANGE_ROOT if EXCHANGE_ROOT and EXCHANGE_ROOT.exists() else None
    except (ImportError, AttributeError):
        default_dir = None

    if base_dir is None:
        base_dir = default_dir
    else:
        base_dir = Path(base_dir)
    
    # If still no valid directory, return empty dictionary
    if base_dir is None:
        return {}
    
    # Ensure base_dir is a Path object
    if not isinstance(base_dir, Path):
        base_dir = Path(base_dir)

    if not base_dir.exists():
        return {}

    # Find all XML files recursively
    xml_files = list(base_dir.rglob("*.xml"))
    if not xml_files:
        return {}

    all_data = []

    for fpath in xml_files:
        df = parse_ecb_fx_xml(fpath)
        if not df.empty:
            all_data.append(df)

    if not all_data:
        return {}

    combined = pd.concat(all_data, ignore_index=True)

    # Compute average per currency-year
    avg_df = (
        combined.groupby(["currency", "year"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": "avg_rate"})
    )

    # Convert to dictionary of dictionaries
    fx_dict = {}
    for _, row in avg_df.iterrows():
        cur = row["currency"]
        yr = int(row["year"])
        rate = float(row["avg_rate"])

        fx_dict.setdefault(cur, {})
        fx_dict[cur][yr] = rate

    return fx_dict

# ============================================================================
# TURBINE DESIGN & PLANT CONFIGURATION
# ============================================================================

@dataclass
class TurbineDesign:
    """
    European Offshore Wind Farm turbine design.

    Default values based on IRENA (2025) and DTU 10MW reference turbine.
    
    Source: https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2025/Jul/IRENA_TEC_RPGC_in_2024_2025.pdf
    """
    n_turbines: int = round(1000 / 10)  # 100 turbines for 1 GW farm
    rated_rpm: float = 9.6
    rotor_diameter: float = 178.3
    rated_power: float = 10  # MW
    hub_height: float = 119  # m
    drivetrain_type: str = "dd"  # 'dd' (direct drive), 'hs' (high-speed), 'ms' (medium-speed)
    foundation_type: str = "monopile"


@dataclass
class PlantConfig:
    """
    Wind farm site and operational configuration.
    
    Defines the location, size, and operational parameters for the
    offshore wind farm installation.
    """
    destination: str = "BE"  # Country code
    site_distance_km: float = 75  # Distance from port to site (km)
    water_depth_m: float = 50  # Average water depth at site (m)
    project_duration_years: float = 25  # Assumed project lifetime
    decommissioning_factor: float = 0.05  # Decommissioning as % of CAPEX

# Source: BVG Offshore Wind Cost Model (2025)
# https://guidetoanoffshorewindfarm.com/wind-farm-costs/

costs_bvg = {
    "Development and project management": {
        "Total": 155_300,
        "Development & consenting services": {
            "Total": 97_400,
            "Environmental impact assessments": 10_700,
            "Development activities and other consenting services": 61_600,
            "Environmental surveys": {
                "Total": 8_700,
                "Animal surveys (benthic, fish, shellfish, mammals and birds)": 7_500,
                "Onshore environmental surveys": 1_200,
            },
            "Resource and metocean assessment": {
                "Total": 7_000,
                "Structure": 3_500,
                "Sensors": 900,
                "Maintenance": 660,
            },
            "Geological and hydrographical surveys": {
                "Total": 9_400,
                "Geophysical surveys": 2_500,
                "Geotechnical surveys": 5_000,
                "Hydrographic surveys": 1_900,
            },
        },
        "Engineering & consultancy": 9_500,
        "Project management": 48_400,
    },
    "Turbine": {
        "Total": 1_214_000,
        "Nacelle": 780_000,
        "Rotor": 340_000,
        "Tower": 94_000,
    },
    "BoP": {
        "Total": 1_086_000,
        "Array cables": 76_000,
        "Export cables": 270_000,
        "Cable accessories": 36_000,
        "Foundation": {
            "Total": 420_000,
            "Monopile": 266_000,
            "Transition piece": 124_000,
            "Corrosion protection": 28_000,
            "Scour protection": 5_400,
        },
        "OSS": {
            "Total": 240_000,
            "HVAC electrical system": 79_800,
            "Auxiliary systems": 13_300,
            "Topside structure": 123_200,
            "Foundation": 22_000,
        },
        "Onshore substation": {
            "Total": 44_000,
            "Electrical system": 30_800,
            "Buildings, access and security": 13_000,
        },
    },
    "Installation and commissioning": {
        "Total": 1_014_200,
        "Inbound transport": 87_000,
        "Foundation installation": {
            "Total": 165_000,
            "Monopile installation vessel": 75_000,
            "TP installation vessel": 41_000,
            "Technician services": 1_000,
            "Marshalling port": 10_400,
            "Other": 37_000,
        },
        "Turbine installation": {
            "Total": 132_000,
            "Turbine installation vessel": 98_000,
            "Technician services": 7_000,
            "Marshalling port": 9_200,
            "Other": 17_000,
        },
        "Offshore cable installation": 193_000,
        "Onshore export cable installation": 8_400,
        "Offshore substation installation": 43_400,
        "Onshore substation construction": 29_400,
        "Offshore logistics": {
            "Total": 13_000,
            "Sea-based support": 6_000,
            "Marine coordination": 2_000,
            "Weather forecasting and metocean data": 600,
            "Marine safety and rescue": 4_400,
        },
        "Contingency and insurance": 343_000,
    },
    "Operation, maintenance and service": {
        "Total": 85_320,
        "Operations, maintenance and service port": 400,
        "Operations": {
            "Total": 38_120,
            "Operations control centre": 1_300,
            "Training": 2_600,
            "Onshore logistics": 1_300,
            "Technical resource (onshore and off)": 6_600,
            "Admin and support staff (onshore)": 7_900,
            "Insurance": 11_220,
            "Offshore logistics": 7_200,
        },
        "Maintenance & service": {
            "Total": 47_200,
            "Turbine maintenance & service": 34_200,
            "BoP maintenance & service": 11_600,
            "Statutory inspections": 1_000,
        },
    },
    "Decommissioning": {
        "Total": 426_000,
        "Turbine decommissioning": 105_000,
        "Foundation decommissioning": 132_000,
        "Cable decommissioning": 155_000,
        "Substation decommissioning": 35_000,
    },
}

@dataclass
class TurbineBVG:
    """
    BVG reference turbine:
    - 1,000 MW wind farm
    - 15 MW turbines
    """
    n_turbines: int = round(1000 / 15)  # ~67
    rated_power: float = 15.0           # MW
    rotor_diameter: float = 230.0       # m


@dataclass
class SiteBVG:
    """
    BVG reference site:
    - 50 m water depth
    - 75 km to offshore substation/grid/landfall
    """
    depth: float = 50.0                # m
    distance: float = 75.0             # km (offshore distance)
    distance_to_landfall: float = 75.0 # km


@dataclass
class PlantBVG:
    """
    BVG reference plant layout parameters.
    Used ONLY to derive benchmark cable lengths and substation assumptions.
    """
    layout: Optional[str] = None
    num_turbines: int = 67
    row_spacing: float = 7.0              # in rotor diameters (D)
    turbine_spacing: float = 9.0          # in D
    substation_distance: Optional[float] = None

    # --- CABLE assumptions (from BVG report) ---
    array_cable_factor: float = 1.1       # 10% added length for array cables
    array_cable_cost: float = costs_bvg["BoP"]["Array cables"]  # £ / MW
    export_cable_cost: float = costs_bvg["BoP"]["Export cables"]
    export_length_factor: float = 1.1     # 10% added length for export cables
    n_export_cables: int = 4              # BVG 1 GW reference case


# NREL benchmark shares (2019 Cost of Wind Energy Review)
# Stehly, T., Beiter, P., & Duffy, P. (2020). NREL/TP-5000-78471
# https://docs.nrel.gov/docs/fy21osti/78471.pdf
NREL_ROWS = [
    {"Parent": "O&M", "Cost": "O&M", "NREL": 34.3},

    {"Parent": "Turbine", "Cost": "Turbine", "NREL": 21.0},

    {"Parent": "BoP", "Cost": "Substructure & Foundation", "NREL": 13.2},
    {"Parent": "BoP", "Cost": "Electrical Infrastructure", "NREL": 12.3},
    {"Parent": "Installation", "Cost": "Assembly and Installation", "NREL": 3.2},
    {"Parent": "Installation", "Cost": "Port and Staging, Logistics, Transportation", "NREL": 0.9},

    {"Parent": "Devex", "Cost": "Development", "NREL": 2.2},
    {"Parent": "Devex", "Cost": "Project Management", "NREL": 1.1},

    {"Parent": "Installation", "Cost": "Lease Price", "NREL": 1.4},
    {"Parent": "Installation", "Cost": "Plant Commissioning", "NREL": 0.7},
    {"Parent": "ABEX", "Cost": "Decommissioning", "NREL": 0.9},

    {"Parent": "Installation", "Cost": "Contingency", "NREL": 5.1},
    {"Parent": "Installation", "Cost": "Construction Finance", "NREL": 3.0},
    {"Parent": "Installation", "Cost": "Insurance During Construction", "NREL": 0.7},
]



