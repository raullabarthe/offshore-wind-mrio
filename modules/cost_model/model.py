"""Core offshore-wind cost calculations used to build the final-demand vector."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from .inputs import TurbineDesign, PlantConfig, costs_bvg

# =============================================================================
# MATERIAL PRICES & GENERATOR PARAMETERS
# =============================================================================

# DTU optimized spoked-arm PMSG-DD generator breakdown:
#   Optimized Generator Designs for the DTU 10-MW Offshore Wind Turbine using GeneratorSE
#   Table 11 from: https://docs.nrel.gov/docs/fy17osti/67444.pdf
#   - DTU generator cost factor = 1.79
#   - All masses below correspond to the optimized 10 MW DTU PMSG-DD
#
# GeneratorSE: A Sizing Tool for Variable-Speed Wind Turbine Generators (NREL):
#   Material prices from: https://docs.nrel.gov/docs/fy18osti/66462.pdf

# Material prices GeneratorSE - NREL (2017 USD per metric ton)
copper_price = 4_786.24    # USD/mT
magnet_price = 95_000      # USD/mT
steel_struct = 501.39      # USD/mT (electrical steel)
steel_elec = 556           # USD/mT (structural steel: spokes + arms)

# Generator material masses (DTU PMSG-DD 10 MW)
# Source: NREL 67444.pdf
copper_mass = 9.0      # tons
magnet_mass = 4.96     # tons
iron_mass = 54.0       # tons  (electrical steel)
arms_mass = 164.0      # tons  (structural steel: spokes + arms)

# Generator manufacturing / overhead factor
gen_cost_factor = 1.79

# DTU drivetrain components (from Table 11 – USD/kW)
# Source: DTU Cost Model Table 11 (PMSG-DD)
dt_main_shaft = 148.5      # USD/kW
dt_main_bearings = 34.3    # USD/kW
dt_coupling_brake = 2.8    # USD/kW

# Rated turbine size (default)
rated_power_mw = 10.0      # 10 MW turbine


def compute_drivetrain_shares(drivetrain_bom_per_kw: float | None = None):
    """
    Compute generator internal material shares and drivetrain component shares.

    Returns a dict mapping component names to their fractional share of the
    total DTU generator cost (materials + assembly + drivetrain components).
    """

    # 1. Generator raw material costs (USD)
    gen_material_costs_usd = {
        "copper": copper_price * copper_mass,
        "magnets": magnet_price * magnet_mass,
        "iron_elec": steel_elec * iron_mass,
        "steel_struct": steel_struct * arms_mass,
    }

    # Total material cost (USD)
    mat_cost_usd = sum(gen_material_costs_usd.values())

    # Convert to USD/kW
    gen_material_costs_per_kw = {
        name: cost_usd / (rated_power_mw * 1_000)
        for name, cost_usd in gen_material_costs_usd.items()
    }

    mat_cost_per_kw = mat_cost_usd / (rated_power_mw * 1_000)

    # Total generator cost (materials + assembly)
    gen_total_per_kw = mat_cost_per_kw * gen_cost_factor

    # Generator assembly fraction
    gen_assembly_per_kw = mat_cost_per_kw * (gen_cost_factor - 1.0)

    # 2. DTU 2018 generator total
    total_dtu_per_kw = (
        gen_total_per_kw +
        dt_main_shaft +
        dt_main_bearings +
        dt_coupling_brake
    )

    # 3. Components for shares
    components_per_kw = {
        # Generator breakdown
        "Generator - Assembly": gen_assembly_per_kw,
        "Generator - Magnets": gen_material_costs_per_kw["magnets"],
        "Generator - Copper": gen_material_costs_per_kw["copper"],
        "Generator - Iron (elect)": gen_material_costs_per_kw["iron_elec"],
        "Generator - Iron (arms)": gen_material_costs_per_kw["steel_struct"],

        # Drivetrain
        "Main shaft": dt_main_shaft,
        "Main bearings": dt_main_bearings,
        "Coupling & brake": dt_coupling_brake,
    }

    # Shares relative to the total DTU generator cost
    component_shares = {
        name: cost_kw / total_dtu_per_kw
        for name, cost_kw in components_per_kw.items()
    }

    return component_shares


# Pre-computed generator shares
gen_shares = compute_drivetrain_shares()

# Yaw bearing is 50% total yaw cost for small (2-3 MW turbine) Source: https://docs.nrel.gov/docs/fy07osti/40566.pdf 
# Yaw bearing (with gear) is 80% total yaw cost  Source: https://docs.nrel.gov/docs/fy12osti/52707.pdf
# Yaw allocation assumption:
# Bearing ≈ 60%
# Drive & brake ≈ 20%
# Gear type ≈ 20%
yaw_shares = {
    "Bearing": 0.60,
    "Drive & brake": 0.20,
    "Gear type": 0.20
}


# ============================================================================
# TURBINE MASS CALCULATIONS
# ============================================================================
# ----------------------------
# Helpers
# ----------------------------
def _sum_block_kg(block: Dict[str, np.ndarray]) -> float:
    """Sum all subcomponents for turbine index 0; return kg."""
    return float(sum(v[0] for v in block.values()))

def _kg_to_t(x: float) -> float:
    """kg -> tonnes."""
    return x / 1000.0

def scalar_sum(x) -> float:
    """Sum array or return scalar."""
    if hasattr(x, 'sum'):
        return float(x.sum())
    elif isinstance(x, (int, float)):
        return float(x)
    elif isinstance(x, dict):
        return sum(scalar_sum(v) for v in x.values())
    else:
        return 0.0

def mass_kg(x, mass_unit) -> float:
    """Return scalar mass in kg."""
    val = scalar_sum(x)
    if mass_unit == "t":
        val *= 1000.0
    return val

def sum_block_kg(block: Dict[str, np.ndarray]) -> float:
    """
    Sum all subcomponents for turbine index 0; return kg.
    
    Parameters
    ----------
    block : Dict[str, np.ndarray]
        Dictionary where each value is an array of masses (kg)
        
    Returns
    -------
    float
        Sum of masses for first turbine (kg)
    """
    return float(sum(v[0] for v in block.values()))


def kg_to_t(x: float) -> float:
    """
    Convert kilograms to tonnes.
    
    Parameters
    ----------
    x : float
        Mass in kilograms
        
    Returns
    -------
    float
        Mass in tonnes (kg / 1000)
    """
    return x / 1000.0

# ----------------------------
# Masses by subcomponents
# ----------------------------

def turbine_masses(des: TurbineDesign) -> Dict[str, Any]:
    """
    Calculate turbine component masses based on design parameters.
    
    Uses empirical scaling equations from NREL and DTU models.
    
    Parameters
    ----------
    des : TurbineDesign
        Turbine design parameters
        
    Returns
    -------
    Dict[str, Any]
        Nested dictionary of component masses (kg) for each turbine
        
    References
    ----------
    - NREL offshore wind cost models (Beiter et al.)
    - DTU 10MW Reference Turbine (Bak et al.)
    """
    n, D, P, H, rpm = (
        int(des.n_turbines),
        float(des.rotor_diameter),
        float(des.rated_power),
        float(des.hub_height),
        float(des.rated_rpm),
    )
    A = np.pi * (D / 2.0) ** 2
    T = P / (rpm * np.pi / 30.0) * 1.1

    masses: Dict[str, Any] = {}
    masses["3_blades"] = np.full(n, 0.3 * D ** 2.5)
    masses["hub"] = {
        "structure": np.full(n, 6.0e3 + 0.1 * D ** 2.5),
        "pitch_bearings": np.full(n, 5.0e2 + 0.07 * D ** 2.5),
        "pitch_system": np.full(n, 5.0e2 + 0.03 * D ** 2.5),
        "secondary": np.full(n, 7.0e2 + 15.0 * D),
    }
    masses["tower"] = {
        "structure": np.full(n, 0.25 * (A * H)),
        "internal": np.full(n, 1.0e3 + 100.0 * H),
        "cabling": np.full(n, 25.0 * (P * H)),
        "secondary": np.full(n, 1.0e3 + 500.0 * P),
        "transformer": np.full(n, 2.5e3 * P),
    }
    if des.drivetrain_type == "hs":
        masses["drivetrain"] = {
            "bedplate": np.full(n, 2.4 * D ** 2.0),
            "main_shaft": np.full(n, 0.02 * D ** 2.8),
            "main_bearings": np.full(n, 0.02 * D ** 2.5),
            "bearing_housing": np.full(n, 0.03 * D ** 2.5),
            "gearbox": np.full(n, 15.0e3 * T),
            "coupling_&_brake": np.full(n, 500.0 + 50.0 * (T * rpm / 1.5)),
            "generator": np.full(n, 1.0e3 + 400.0 * (T * rpm / 1.5)),
        }
    elif des.drivetrain_type == "dd":
        masses["drivetrain"] = {
            "bedplate": np.full(n, 1.2 * D ** 2.0),
            "generator": np.full(n, 35.0e3 * (T ** (2.0 / 3.0))),
        }
    else:  # 'ms'
        masses["drivetrain"] = {
            "bedplate": np.full(n, 1.2 * D ** 2.0),
            "main_shaft": np.full(n, 0.02 * D ** 2.8),
            "main_bearings": np.full(n, 0.02 * D ** 2.5),
            "bearing_housing": np.full(n, 0.03 * D ** 2.5),
            "gearbox": np.full(n, 1.0e4 * T),
            "coupling_&_brake": np.full(n, 30.0 * (T * rpm / 0.4)),
            "generator": np.full(n, 6.0e3 + 100.0 * (T * rpm / 0.4)),
        }
    masses["nacelle"] = {
        "cooling": np.full(n, 500.0 * P),
        "converter": np.full(n, 1.0e3 * P),
        "controller": np.full(n, 200.0 + 100.0 * P),
        "yaw": np.full(n, 0.1 * D ** 2.5),
        "canopy": np.full(n, 1.0e3 + 1.5e3 * P),
        "secondary": np.full(n, 1.0e3 + 1.0e3 * P),
    }
    return masses


# ============================================================================
# TURBINE SUPPLY COST
# ============================================================================

def build_turbine_supply(
    des: "TurbineDesign",
    masses: Dict[str, Any],
    *,
    unit_costs: Optional[Dict[str, Dict[str, float]]] = None,  # €/kg overrides
    mass_unit: str = "kg",
    include_subtotals: bool = True,
    currency: str = "EUR",
    cost_factors: Optional[Dict[str, float]] = None,
    eur_gbp: float = 1.17,          # GBP -> EUR
    calibrate_to_bvg: bool = True,  # single global scaling to BVG
) -> dict:
    """
    Build turbine supply cost breakdown (full wind farm).

    Logic:
    - Use DTU-style €/kg for Blades, Hub, Nacelle, Tower, Drivetrain.
    - Mass inputs may be arrays (per turbine); mass_kg() collapses to total kg.
    - First compute an internal (model) total.
    - Then (if calibrate_to_bvg=True) scale ALL turbine supply costs so that:
        Model_total(farm) == BVG_WT_Total[GBP/MW] * eur_gbp * rated_power_MW * n_turbines
    - No per-component (Rotor/Nacelle/Tower) calibration.
    """

    # --- 1. Cost factors ---
    cf = {
        "direct_labor": 0.03,
        "direct_overhead": 0.10,
        "sga_only": 0.05,
        "overhead": 0.05,
        "r_and_d": 0.03,
    }
    if cost_factors:
        cf.update(cost_factors)

    def pack_component(bom: float) -> dict:
        """Direct on BoM; indirect on direct_production."""
        bom = float(bom)
        dl = cf["direct_labor"] * bom
        do = cf["direct_overhead"] * bom
        dp = bom + dl + do
        return {
            "BoM": bom,
            "Direct": {
                "direct_labor": dl,
                "direct_overhead": do,
                "direct_production": dp,
            },
            "Indirect": {
                "sga_only": cf["sga_only"] * dp,
                "overhead": cf["overhead"] * dp,
                "r_and_d": cf["r_and_d"] * dp,
            },
        }

    def leaf_total(p: dict) -> float:
        return p["Direct"]["direct_production"] + sum(p["Indirect"].values())

    # --- 2. Base €/kg (DTU-style, as in your original) ---
    uc = {
        "Blades": {"Blades": 12.0},
        "Hub": {
            "Structure": 2.5,
            "Pitch bearings": 8.0,
            "Pitch system": 8.0,
            "Secondary": 8.0,
        },
        "Nacelle": {
            "Cooling": 8.0,
            "Converter": 30.0,
            "Controller": 50.0,
            "Yaw": 6.0,
            "Canopy": 10.0,
            "Secondary": 10.0,
        },
        "Tower": {
            "Structure": 1.5,
            "Internal": 8.0,
            "Cabling": 8.0,
            "Secondary": 10.0,
            "Transformer": 8.0,
        },
        "Drivetrain": {
            "Bedplate": 2.5,
            "Main shaft": 5.0,
            "Main bearings": 15.0,
            "Bearing housing": 2.5,
            "Gearbox": 8.0,
            "Coupling & brake": 8.0,
            "Generator": 8.0,
        },
    }
    if unit_costs:
        for block, leaves in unit_costs.items():
            uc.setdefault(block, {}).update(leaves)

    # --- 3. Masses (arrays -> total kg for the whole farm) ---
    blades_mass = mass_kg(masses.get("3_blades", 0.0), mass_unit)

    hub = masses.get("hub", {})
    hub_mass = {
        "Structure":      mass_kg(hub.get("structure", 0.0), mass_unit),
        "Pitch bearings": mass_kg(hub.get("pitch_bearings", 0.0), mass_unit),
        "Pitch system":   mass_kg(hub.get("pitch_system", 0.0), mass_unit),
        "Secondary":      mass_kg(hub.get("secondary", 0.0), mass_unit),
    }

    nac = masses.get("nacelle", {})
    nac_mass = {
        "Cooling":   mass_kg(nac.get("cooling", 0.0), mass_unit),
        "Converter": mass_kg(nac.get("converter", 0.0), mass_unit),
        "Controller":mass_kg(nac.get("controller", 0.0), mass_unit),
        "Yaw":       mass_kg(nac.get("yaw", 0.0), mass_unit),
        "Canopy":    mass_kg(nac.get("canopy", 0.0), mass_unit),
        "Secondary": mass_kg(nac.get("secondary", 0.0), mass_unit),
    }

    tower = masses.get("tower", {})
    tower_mass = {
        "Structure":   mass_kg(tower.get("structure", 0.0), mass_unit),
        "Internal":    mass_kg(tower.get("internal", 0.0), mass_unit),
        "Cabling":     mass_kg(tower.get("cabling", 0.0), mass_unit),
        "Secondary":   mass_kg(tower.get("secondary", 0.0), mass_unit),
        "Transformer": mass_kg(tower.get("transformer", 0.0), mass_unit),
    }

    drv = masses.get("drivetrain", {})
    drv_mass = {
        "Bedplate":         mass_kg(drv.get("bedplate", 0.0), mass_unit),
        "Main shaft":       mass_kg(drv.get("main_shaft", 0.0), mass_unit),
        "Main bearings":    mass_kg(drv.get("main_bearings", 0.0), mass_unit),
        "Bearing housing":  mass_kg(drv.get("bearing_housing", 0.0), mass_unit),
        "Gearbox":          mass_kg(drv.get("gearbox", 0.0), mass_unit),
        "Coupling & brake": mass_kg(drv.get("coupling_&_brake", 0.0), mass_unit),
        "Generator":        mass_kg(drv.get("generator", 0.0), mass_unit),
    }
    # Only keep drivetrain components that exist (as in your original)
    drv_mass = {k: v for k, v in drv_mass.items() if v > 0.0}

    # --- 3a. Split Yaw into subcomponents BEFORE cost calculation ---
    yaw_mass = nac_mass.get("Yaw", 0.0)
    if yaw_mass > 0.0:
        nac_mass["Yaw - Bearing"]       = yaw_mass * yaw_shares["Bearing"]
        nac_mass["Yaw - Drive & brake"] = yaw_mass * yaw_shares["Drive & brake"]
        nac_mass["Yaw - Gear type"]     = yaw_mass * yaw_shares["Gear type"]
        nac_mass["Yaw"] = 0.0  # so pack_block ignores "Yaw"

        # All yaw subcomponents use the same €/kg as original Yaw
        yaw_uc = uc["Nacelle"]["Yaw"]
        uc["Nacelle"]["Yaw - Bearing"]       = yaw_uc
        uc["Nacelle"]["Yaw - Drive & brake"] = yaw_uc
        uc["Nacelle"]["Yaw - Gear type"]     = yaw_uc

    # --- 3b. Adjust Drivetrain €/kg for DTU generator group BEFORE BoM -> Direct/Indirect ---
    # DTU "generator" group = Generator + Main shaft + Main bearings + Coupling & brake
    gen_mass = drv_mass.get("Generator", 0.0)
    if gen_mass > 0.0:
        drv_mass["Generator - Assembly"]    = gen_mass * gen_shares["Generator - Assembly"]
        drv_mass["Generator - Copper"]      = gen_mass * gen_shares["Generator - Copper"]
        drv_mass["Generator - Iron (arms)"] = gen_mass * gen_shares["Generator - Iron (arms)"]
        drv_mass["Generator - Iron (elect)"]= gen_mass * gen_shares["Generator - Iron (elect)"]
        drv_mass["Generator - Magnets"]     = gen_mass * gen_shares["Generator - Magnets"]
        drv_mass["Main bearings"]           = gen_mass * gen_shares["Main bearings"]
        drv_mass["Main shaft"]              = gen_mass * gen_shares["Main shaft"]
        drv_mass["Coupling & brake"]        = gen_mass * gen_shares["Coupling & brake"]
        drv_mass["Generator"] = 0.0  # so pack_block ignores "Yaw"

        # All generator subcomponents use the same €/kg as original generator
        gen_uc = uc["Drivetrain"]["Generator"]
        uc["Drivetrain"]["Generator - Assembly"]    = gen_uc
        uc["Drivetrain"]["Generator - Copper"]      = gen_uc
        uc["Drivetrain"]["Generator - Iron (arms)"] = gen_uc
        uc["Drivetrain"]["Generator - Iron (elect)"]= gen_uc
        uc["Drivetrain"]["Generator - Magnets"]     = gen_uc
        uc["Drivetrain"]["Main bearings"]           = gen_uc
        uc["Drivetrain"]["Main shaft"]              = gen_uc
        uc["Drivetrain"]["Coupling & brake"]        = gen_uc

    # --- 4. Unscaled breakdown (farm-level, DTU €/kg) ---
    def pack_block(block_name: str, mdict: Dict[str, float]) -> Dict[str, dict]:
        """Use only components that have both mass and €/kg defined and mass > 0."""
        umap = uc[block_name]
        return {
            name: pack_component(mdict[name] * umap[name])
            for name in umap.keys()
            if mdict.get(name, 0.0) > 0.0
        }

    result = {
        "Blades": {
            "Blades": pack_component(blades_mass * uc["Blades"]["Blades"])
        } if blades_mass > 0.0 else {},
        "Hub":        pack_block("Hub", hub_mass),
        "Nacelle":    pack_block("Nacelle", nac_mass),
        "Tower":      pack_block("Tower", tower_mass),
        "Drivetrain": pack_block("Drivetrain", drv_mass),
    }

    raw_total = sum(
        leaf_total(leaf)
        for block in result.values()
        for leaf in block.values()
    )

    # --- 5. Global BVG scaling: match total WT CAPEX (farm) ---
    if calibrate_to_bvg and raw_total > 0:
        rated = float(des.rated_power)
        n_t = getattr(des, "n_turbines", 1)  # trust design; no guessing from arrays
        bvg_gbp_per_mw = costs_bvg["Turbine"]["Total"]  # GBP/MW
        target_total = bvg_gbp_per_mw * eur_gbp * rated * n_t  # EUR, all turbines
        f = target_total / raw_total if target_total > 0 else 1.0

        # Scale every monetary field by f
        for block in result.values():
            for leaf in block.values():
                leaf["BoM"] *= f
                for k in leaf["Direct"]:
                    leaf["Direct"][k] *= f
                for k in leaf["Indirect"]:
                    leaf["Indirect"][k] *= f

    # --- 6. Subtotals & grand total (after scaling) ---
    if include_subtotals:
        subtotals = {
            name: sum(leaf_total(leaf) for leaf in block.values())
            for name, block in result.items()
        }
        result["Supply_components"] = {
            **subtotals,
            "currency": currency,
            "scope": "farm",
        }
        result["Grand_total"] = {
            "value": sum(subtotals.values()),
            "currency": currency,
            "scope": "farm",
        }

    return result


def sum_bom_direct_indirect(node: dict) -> float:
    """
    Sum BoM + Direct + Indirect costs from cost node.
    
    Aggregates material costs (BoM), direct manufacturing costs,
    and indirect/overhead costs from a cost breakdown node.
    
    Parameters
    ----------
    node : dict
        Cost node with structure: {"BoM": float, "Direct": {}, "Indirect": {}}
        
    Returns
    -------
    float
        Total cost (BoM + Direct + Indirect)
    """
    bom = float(node.get("BoM", 0.0))

    direct = node.get("Direct", {}) or {}
    direct_sum = sum(float(v) for k, v in direct.items() if k != "direct_production")

    indirect = node.get("Indirect", {}) or {}
    indirect_sum = sum(float(v) for k, v in indirect.items() if k != "overhead")

    return bom + direct_sum + indirect_sum


# ============================================================================
# DEVEX, BOP, INSTALLATION, OPEX, ABEX - BVG-BASED CALCULATIONS
# ============================================================================

def calculate_devex(
    rated_power_arr: np.ndarray,
    *,
    eur_gbp: float = 1.17,
) -> Dict[str, Any]:
    """
    Development and Project Management costs from BVG benchmark.
    
    Scales BVG costs (GBP/MW) to absolute EUR values based on total
    installed capacity.
    
    Parameters
    ----------
    rated_power_arr : np.ndarray
        Array of rated powers (MW) for each turbine
    eur_gbp : float
        EUR/GBP conversion rate
        
    Returns
    -------
    Dict[str, Any]
        Cost breakdown with keys: Total, Development & consenting services,
        Engineering & consultancy, Project management (all in EUR)
    """
    total_mw = float(rated_power_arr.sum())
    if total_mw <= 0:
        return {
            "Total": 0.0,
            "Development & consenting services": 0.0,
            "Engineering & consultancy": 0.0,
            "Project management": 0.0
        }

    fx = eur_gbp
    dev_mw = costs_bvg["Development and project management"]
    dcs_mw = float(dev_mw["Development & consenting services"]["Total"])
    eng_pm = float(dev_mw["Engineering & consultancy"])
    pm_pm  = float(dev_mw["Project management"])

    dcs_abs   = fx * total_mw * dcs_mw
    eng_abs   = fx * total_mw * eng_pm
    pmc_abs   = fx * total_mw * pm_pm
    dev_total = dcs_abs + eng_abs + pmc_abs

    return {
        "Total": dev_total,
        "Development & consenting services": dcs_abs,
        "Engineering & consultancy": eng_abs,
        "Project management": pmc_abs,
    }


def calculate_bop(
    rated_power_arr: np.ndarray,
    *,
    eur_gbp: float = 1.17,
) -> Dict[str, Any]:
    """
    Balance of Plant (Supply) costs from BVG benchmark.
    
    Covers Foundation, Export cables, Array cables, Offshore substation,
    and other BoP components. Scales BVG values (GBP/MW) to absolute EUR.
    
    Parameters
    ----------
    rated_power_arr : np.ndarray
        Array of rated powers (MW) for each turbine
    eur_gbp : float
        EUR/GBP conversion rate
        
    Returns
    -------
    Dict[str, Any]
        Cost breakdown with keys: Total, Foundation, Export cables,
        Array cables, OSS, Other (all in EUR)
    """
    total_mw = float(rated_power_arr.sum())
    if total_mw <= 0:
        return {
            "Total": 0.0,
            "Foundation": 0.0,
            "Export cables": 0.0,
            "Array cables": 0.0,
            "OSS": 0.0,
            "Other": 0.0,
        }

    fx = eur_gbp
    bop_mw = costs_bvg["BoP"]

    # GBP/MW (first level)
    foundations_mw = float(bop_mw["Foundation"]["Total"])
    export_mw      = float(bop_mw["Export cables"])
    array_mw       = float(bop_mw["Array cables"])
    oss_mw         = float(bop_mw["OSS"]["Total"])
    cable_acc_mw   = float(bop_mw.get("Cable accessories", 0.0))

    # Allocate cable accessories to export/array by their BVG weights
    denom_cables = (export_mw + array_mw) or 1.0
    export_share = export_mw / denom_cables
    array_share  = array_mw  / denom_cables
    export_mw   += cable_acc_mw * export_share
    array_mw    += cable_acc_mw * array_share

    # Residual (Cable accessories + Onshore substation + other items)
    bop_total_mw = float(bop_mw["Total"])
    other_mw = bop_total_mw - (foundations_mw + export_mw + array_mw + oss_mw)

    # EUR absolute
    foundations_abs = fx * foundations_mw * total_mw
    export_abs      = fx * export_mw * total_mw
    array_abs       = fx * array_mw * total_mw
    oss_abs         = fx * oss_mw * total_mw
    other_abs       = fx * other_mw * total_mw
    total_abs       = foundations_abs + export_abs + array_abs + oss_abs + other_abs

    return {
        "Total": total_abs,
        "Foundation": foundations_abs,
        "Export cables": export_abs,
        "Array cables": array_abs,
        "OSS": oss_abs,
        "Other": other_abs,
    }


def calculate_installation(
    rated_power_arr: np.ndarray,
    *,
    eur_gbp: float = 1.17,
) -> Dict[str, Any]:
    """
    Installation & Commissioning costs from BVG benchmark.
    
    Intelligently splits offshore cable installation between export and array
    using the supply-side cable proportions. Scales BVG values (GBP/MW) to
    absolute EUR.
    
    Parameters
    ----------
    rated_power_arr : np.ndarray
        Array of rated powers (MW) for each turbine
    eur_gbp : float
        EUR/GBP conversion rate
        
    Returns
    -------
    Dict[str, Any]
        Cost breakdown with keys: Total, Export cables, Array cables,
        Foundation, Turbine, OSS, Other (all in EUR)
        
    Notes
    -----
    - Offshore cable installation is split using BoP cable supply shares
    - Onshore export cable installation is fully attributed to Export cables
    - OSS includes both offshore substation installation and onshore construction
    """
    total_mw = float(rated_power_arr.sum())
    if total_mw <= 0:
        return {
            "Total": 0.0,
            "Export cables": 0.0,
            "Array cables": 0.0,
            "Foundation": 0.0,
            "Turbine": 0.0,
            "OSS": 0.0,
            "Other": 0.0,
        }

    fx = eur_gbp
    inst_mw = costs_bvg["Installation and commissioning"]
    bop_mw  = costs_bvg["BoP"]

    # --- BoP cable supply shares (GBP/MW) ---
    export_supply_mw = float(bop_mw["Export cables"])
    array_supply_mw  = float(bop_mw["Array cables"])
    denom = (export_supply_mw + array_supply_mw) or 1.0
    export_share = export_supply_mw / denom
    array_share  = array_supply_mw  / denom

    # --- Installation first-level totals (GBP/MW) ---
    inst_total_mw = float(inst_mw["Total"])

    offshore_cable_install_mw = float(inst_mw.get("Offshore cable installation", 0.0))
    onshore_export_install_mw = float(inst_mw.get("Onshore export cable installation", 0.0))

    foundation_mw = float(inst_mw["Foundation installation"]["Total"])
    turbine_mw    = float(inst_mw["Turbine installation"]["Total"])

    oss_mw = float(inst_mw.get("Offshore substation installation", 0.0))

    # --- Split offshore cable installation into export vs array ---
    export_offshore_mw = offshore_cable_install_mw * export_share
    array_offshore_mw  = offshore_cable_install_mw * array_share

    export_cables_mw = export_offshore_mw + onshore_export_install_mw
    array_cables_mw  = array_offshore_mw

    # --- Residual Other (keeps totals consistent with BVG Total) ---
    known_mw = foundation_mw + turbine_mw + oss_mw + export_cables_mw + array_cables_mw
    other_mw = inst_total_mw - known_mw

    # EUR absolute
    def to_eur(x_mw: float) -> float:
        return fx * float(x_mw) * total_mw

    return {
        "Total": to_eur(inst_total_mw),
        "Export cables": to_eur(export_cables_mw),
        "Array cables": to_eur(array_cables_mw),
        "Foundation": to_eur(foundation_mw),
        "Turbine": to_eur(turbine_mw),
        "OSS": to_eur(oss_mw),
        "Other": to_eur(other_mw),
    }


def calculate_opex(
    rated_power_arr: np.ndarray,
    project_duration_years: float = 25.0,
    eur_gbp: float = 1.1812,
) -> Dict[str, Any]:
    """
    Operation & Maintenance costs over lifetime from BVG benchmark.
    
    Splits OPEX into Operations and Maintenance & service categories,
    with both annual and lifetime (25-30 years) breakdowns.
    
    Parameters
    ----------
    rated_power_arr : np.ndarray
        Array of rated powers (MW) for each turbine
    project_duration_years : float
        Project lifetime in years (typically 25-30)
    eur_gbp : float
        EUR/GBP conversion rate
        
    Returns
    -------
    Dict[str, Any]
        Cost breakdown with keys: annual_breakdown, lifetime_breakdown
        Each containing Operations and Maintenance & service (in EUR)
        
    Notes
    -----
    - BVG provides GBP/MW/year values
    - Lifetime costs are calculated as: annual_cost * project_duration_years
    """
    total_capacity_mw = float(np.sum(rated_power_arr))
    if total_capacity_mw <= 0:
        return {"annual_breakdown": {}, "lifetime_breakdown": {}}

    om = costs_bvg["Operation, maintenance and service"]

    ops_port_pm = float(om.get("Operations, maintenance and service port", 0.0))
    operations = om.get("Operations", {}) or {}
    maintenance = om.get("Maintenance & service", {}) or {}

    # Sum GBP/MW-year values (ignore "Total" if present)
    operations_pm = ops_port_pm + sum(
        float(v) for k, v in operations.items() if k != "Total"
    )
    maintenance_pm = sum(
        float(v) for k, v in maintenance.items() if k != "Total"
    )

    annual_breakdown = {
        "Operations": operations_pm * total_capacity_mw * eur_gbp,
        "Maintenance & service": maintenance_pm * total_capacity_mw * eur_gbp,
    }

    lifetime_breakdown = {
        k: v * project_duration_years for k, v in annual_breakdown.items()
    }

    return {
        "annual_breakdown": annual_breakdown,
        "lifetime_breakdown": lifetime_breakdown,
    }


def calculate_abex(
    rated_power_arr: np.ndarray,
    eur_gbp_tc: float,
) -> Dict[str, Any]:
    """
    Decommissioning (ABEX) costs from BVG benchmark.
    
    Simplified version using BVG total decommissioning cost, scaled by
    total installed capacity.
    
    Parameters
    ----------
    rated_power_arr : np.ndarray
        Array of rated powers (MW) for each turbine
    eur_gbp_tc : float
        EUR/GBP conversion rate
        
    Returns
    -------
    Dict[str, Any]
        Cost breakdown with key: ABEX (total decommissioning cost in EUR)
        
    Notes
    -----
    - Vessel costs represent 60-80% of decommissioning costs
    - Source: ORE Catapult "End-of-Life decision planning in offshore wind"
    """
    def F(x):
        try:
            return float(x)
        except Exception:
            return 0.0

    dec_bvg = costs_bvg["Decommissioning"]

    # Total installed capacity (MW)
    total_mw = float(np.sum(rated_power_arr))

    # BVG total coefficient in GBP/MW
    dec_total_pm = F(dec_bvg.get("Total"))

    # Total ABEX cost: GBP/MW * MW * (EUR/GBP)
    total = dec_total_pm * total_mw * eur_gbp_tc

    return {
        "ABEX": total,
    }


# ============================================================================
# COMPLETE COST BREAKDOWN
# ============================================================================

def build_cost_breakdown(
    turbine_supply_cost: Dict[str, Any],
    plant: PlantConfig,
    design: TurbineDesign,
    project_duration_years: float = 25.0,
    eur_gbp: float = 1.17,
) -> Dict[str, Any]:
    """
    Build complete wind farm cost breakdown.

    Combines:
    - Turbine supply costs (factory gate)
    - Development & management (Devex)
    - Balance of Plant (BoP)
    - Installation & commissioning
    - Operation & maintenance (O&M) - lifetime
    - Decommissioning (ABEX)

    Parameters
    ----------
    turbine_supply_cost : dict
        Output from build_turbine_supply()
    plant : PlantConfig
        Wind farm configuration
    design : TurbineDesign
        Turbine specifications
    project_duration_years : float
        Project lifetime (typically 25-30 years)
    eur_gbp : float
        EUR/GBP conversion rate

    Returns
    -------
    dict
        Complete cost structure:
        {
            "Devex": {...},
            "Turbine": {...},
            "BoP": {...},
            "Installation": {...},
            "O&M": {...},
            "ABEX": {...},
        }
    """
    n_t = int(design.n_turbines or 0)
    rated_power_arr = np.full(n_t, float(design.rated_power), dtype=float)
    
    # --- BVG-based blocks (EUR) ---
    devex = calculate_devex(rated_power_arr, eur_gbp=eur_gbp)

    bop_bvg = calculate_bop(rated_power_arr, eur_gbp=eur_gbp)
    inst_bvg = calculate_installation(rated_power_arr, eur_gbp=eur_gbp)
    opex = calculate_opex(
        rated_power_arr,
        project_duration_years=project_duration_years,
        eur_gbp=eur_gbp,
    )
    abex = calculate_abex(
        rated_power_arr=rated_power_arr,
        eur_gbp_tc=eur_gbp,
    )

    # --- Keep EXACT output labels ---
    bop = {
        "Foundation":  float(bop_bvg.get("Foundation", 0.0)),
        "Export cables":       float(bop_bvg.get("Export cables", 0.0)),
        "Array cables":        float(bop_bvg.get("Array cables", 0.0)),
        "OSS":                 float(bop_bvg.get("OSS", 0.0)),
        "Other":           float(bop_bvg.get("Other", 0.0)),
    }

    installation = {
        "Export cables": float(inst_bvg.get("Export cables", 0.0)),  
        "Array cables": float(inst_bvg.get("Array cables", 0.0)),
        "Foundation":   float(inst_bvg.get("Foundation", 0.0)),
        "Turbine":      float(inst_bvg.get("Turbine", 0.0)),
        "OSS":          float(inst_bvg.get("OSS", 0.0)),
        "Other":        float(inst_bvg.get("Other", 0.0)),
    }

    return {
        "Devex": devex,
        "Turbine": turbine_supply_cost,
        "BoP": bop,
        "Installation": installation,
        "O&M": opex,
        "ABEX": abex,
    }
