"""Turbine sourcing allocation from OEM orders, suppliers, and facilities."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from modules.config import CORE_DEPLOYMENT
from modules.country import NAME_TO_CODE, COUNTRY_COORDS, EU27

from .inputs import CG_STD_MAP, oem_hq_country, ownership_oem, bridge
from .helpers import (
    normalize_country_code,
    pick_assembly_facility_country,
    pick_tower_assembly_country,
    tower_assembly_country_shares,
    get_import_total_fraction,
    get_import_country_fractions,
)

# --------------------------------------------------
# OEM market share by deployment country
# --------------------------------------------------

def prepare_oem_share_by_country(df: pd.DataFrame):
    df = df.copy()
    if "Country of deployment" in df.columns:
        df["country_code"] = df["Country of deployment"].map(NAME_TO_CODE)
    elif "deployment_country" in df.columns:
        df["country_code"] = df["deployment_country"].astype(str).str.strip()
    else:
        raise KeyError("Expected 'Country of deployment' or 'deployment_country' in OEM orders.")

    totals = df.groupby("country_code")["Capacity (MW)"].sum().astype(float)

    share_df = (
        df.groupby(["country_code", "OEM"])["Capacity (MW)"]
          .sum()
          .div(totals)
          .reset_index(name="share")
    )

    share_dict = (
        share_df.pivot(index="country_code", columns="OEM", values="share")
               .apply(lambda r: r.dropna().to_dict(), axis=1)
               .to_dict()
    )

    return share_dict, totals.to_dict(), share_df, df


def oem_suppliers(oem_supply, component_group_map=None):
    """
    Build OEM-specific supplier summary table from a preloaded dataframe.

    Parameters
    ----------
    oem_supply : pd.DataFrame
        OEM supply dataframe already loaded.
    component_group_map : dict or None
        Optional mapping to standardize component_group (e.g. CG_STD_MAP).

    Returns
    -------
    summary_table : pd.DataFrame
        Pivot table with OEMs as columns and components as rows.
    """

    df = oem_supply.copy()

    # Standardize component groups if mapping is provided
    if component_group_map is not None:
        df["component_group"] = df["component_group"].replace(component_group_map)

    # Remove Gearbox rows
    df = df[df["sub_component"] != "Gearbox"]

    # Mark OEM-internal sourcing explicitly
    df["country_clean"] = df.apply(
        lambda r: "OEM" if r["is_oem"] else r["country_origin"],
        axis=1
    )

    # Group and aggregate countries per OEM
    summary = (
        df.groupby(["component_group", "sub_component", "OEM"])["country_clean"]
          .apply(lambda x: ", ".join(sorted(set(x.dropna()))))
          .reset_index()
    )

    # Pivot table
    summary_table = (
        summary
        .pivot(
            index=["component_group", "sub_component"],
            columns="OEM",
            values="country_clean"
        )
        .fillna("–")
    )

    return summary_table, oem_supply

def prepare_bom_facilities(
    facilities: pd.DataFrame,
    keep,
    deploy=None,
    include_eu27: bool = False,
    eu27_set: set | None = None,
    share_thresh: float | None = None,
) -> pd.DataFrame:
    fac = facilities.copy()

    # 1) Keep only Operating + Idle (if available)
    if "Status" in fac.columns:
        fac = fac[fac["Status"].isin(["Operating", "Idle"])].copy()

    # 2) Standardize category_key
    fac["category_key"] = fac["Component"].replace({
        "Towers": "Tower",
        "Tower": "Tower",
        "Converter": "Converters",
        "Converters": "Converters",
    })

    # 3) Keep only selected components
    fac = fac[fac["category_key"].isin(keep)].copy()

    # 4) Normalize country code (build country_origin)
    if "country_origin" in fac.columns:
        fac["country_origin"] = fac["country_origin"].apply(normalize_country_code)
    else:
        fac["country_origin"] = fac["Country"].apply(normalize_country_code)

    # 5) Clean Number
    if "Number" not in fac.columns:
        raise ValueError("Expected column 'Number' in facilities.")
    fac["Number"] = pd.to_numeric(fac["Number"], errors="coerce")
    fac = fac.dropna(subset=["Number"]).copy()
    fac = fac[fac["Number"] != 0].copy()

    # 6) Collapse Operating+Idle (Status disappears) by summing Number
    #    Important: keep category_key in the groupby so it survives.
    fac = (
        fac
        .groupby(["category_key", "country_origin"], as_index=False)
        .agg({"Number": "sum"})
    )

    # 7) Compute shares within each component
    totals = fac.groupby("category_key")["Number"].transform("sum").astype(float)
    fac["share_cat"] = np.where(totals > 0, fac["Number"] / totals, 0.0)

    # 8) Pre-filter countries (for plotting/keep list)
    if include_eu27 or deploy:
        allowed = set()
        if include_eu27:
            allowed |= (eu27_set or set(EU27))
        if deploy:
            allowed |= set(deploy)
        fac = fac[fac["country_origin"].isin(allowed)].copy()

    return fac

# share_exclusion = 0.001
# KEEP = {"Castings", "Forgings", "Converters", "Tower"}

# EXCLUDE_GLOBAL = {"RU", "UA", "TR"}
# EXCLUDE_TOWER  = {"AT", "HR", "CZ"}   # tower-only exclusions
# DEPLOY         = set(CORE_DEPLOYMENT)

# bom_facilities = prepare_bom_facilities(
#     bom_fac,
#     keep=KEEP,
#     exclude_global=EXCLUDE_GLOBAL,
#     exclude_tower=EXCLUDE_TOWER,
#     deploy=DEPLOY,
#     share_thresh=share_exclusion,
# )

# --------------------------------------------------
# 4. Split turbine costs by OEM and flatten
# --------------------------------------------------

def scale_numeric_leaf(value, factor):
    return value * factor if isinstance(value, (int, float)) else value

def scale_nested_dict(d, factor):
    if isinstance(d, dict):
        return {k: scale_nested_dict(v, factor) for k, v in d.items()}
    return scale_numeric_leaf(d, factor)

def split_turbine_costs_by_oem(cost_by_country, share_dict):
    result = defaultdict(dict)
    for country, country_data in cost_by_country.items():
        if country not in share_dict or "Turbine" not in country_data:
            continue
        for oem, share in share_dict[country].items():
            result[country][oem] = {"Turbine": scale_nested_dict(country_data["Turbine"], share)}
    return result


def flatten_turbine_costs(cost_by_country_oem):
    rows = []
    for country, oem_dict in cost_by_country_oem.items():
        for oem, data in oem_dict.items():
            turbine = data.get("Turbine", {})
            for component_group, group_data in turbine.items():
                if component_group in ["Supply_components", "Grand_total"] or not isinstance(group_data, dict):
                    continue
                for sub_component, sub_data in group_data.items():
                    if not isinstance(sub_data, dict):
                        continue
                    bom_value = sub_data.get("BoM", 0.0)
                    if isinstance(bom_value, (int, float)) and bom_value != 0:
                        rows.append({
                            "deployment_country": country,
                            "OEM": oem,
                            "component_group": component_group,
                            "sub_component": sub_component,
                            "cost_block": "BoM",
                            "cost_type": "BoM",
                            "value": float(bom_value),
                        })
                    direct = sub_data.get("Direct", {})
                    if isinstance(direct, dict):
                        for ct in ["direct_labor", "direct_overhead"]:
                            v = direct.get(ct, 0.0)
                            if isinstance(v, (int, float)) and v != 0:
                                rows.append({
                                    "deployment_country": country,
                                    "OEM": oem,
                                    "component_group": component_group,
                                    "sub_component": sub_component,
                                    "cost_block": "Assembly",
                                    "cost_type": ct,
                                    "value": float(v),
                                })
                    indirect = sub_data.get("Indirect", {})
                    if isinstance(indirect, dict):
                        for ct in ["sga_only", "overhead", "r_and_d"]:
                            v = indirect.get(ct, 0.0)
                            if isinstance(v, (int, float)) and v != 0:
                                rows.append({
                                    "deployment_country": country,
                                    "OEM": oem,
                                    "component_group": component_group,
                                    "sub_component": sub_component,
                                    "cost_block": "Indirect",
                                    "cost_type": ct,
                                    "value": float(v),
                                })
    return rows


# --------------------------------------------------
# 5. Add category_key (for fallback / industry bridge)
# --------------------------------------------------

def add_category_to_df_turbine(df_turbine, bridge):
    df = df_turbine.copy()
    df["category_key"] = df.apply(
        lambda r: bridge.get((r["component_group"], r["sub_component"])),
        axis=1,
    )
    return df


# --------------------------------------------------
# 6. Import-share filter for plotting
# --------------------------------------------------

def filter_import_shares_for_turbine(df_turbine_cat, import_shares, import_category):
    """
    Return only import-share categories that are actually used
    by the (component_group, sub_component) pairs in df_turbine_cat.
    """
    pairs = df_turbine_cat[["component_group", "sub_component"]].drop_duplicates()
    used = set()
    for _, r in pairs.iterrows():
        cat = import_category.get((r["component_group"], r["sub_component"]))
        if cat and cat != "Turbines":
            used.add(cat)
    return {k: v for k, v in import_shares.items() if k in used}


# --------------------------------------------------
# 7. Filter BoM facilities by component deployment
# --------------------------------------------------

def filter_bom_facilities_by_component_deploy(
    bom_facilities: pd.DataFrame,
    df_turbine_cat: pd.DataFrame,
    *,
    include_eu27: bool = True,
    eu27_set: set | None = None,
    share_thresh: float | None = None,
) -> pd.DataFrame:
    """
    Expand BoM facilities by deployment country and apply per-deployment filters.

    Rules:
      - Always include facilities from the deployment_country,
        regardless of share_thresh.
      - share_thresh (if provided) is applied within each (category_key, deployment).
    """
    # Note: EU27 / deploy pre-filtering should be handled upstream
    # in prepare_bom_facilities when desired.

    dep_by_cat = (
        df_turbine_cat.dropna(subset=["category_key"])
        .groupby("category_key")["deployment_country"]
        .apply(lambda s: set(s.astype(str)))
        .to_dict()
    )

    out = []
    for cat, dep_countries in dep_by_cat.items():
        fac_cat = bom_facilities[bom_facilities["category_key"] == cat].copy()
        if fac_cat.empty:
            continue

        for dep in dep_countries:
            dep = str(dep)
            fac_dep = fac_cat.copy()
            if fac_dep.empty:
                continue

            totals = fac_dep.groupby("category_key")["Number"].transform("sum").astype(float)
            fac_dep["share_cat"] = np.where(totals > 0, fac_dep["Number"] / totals, 0.0)

            if share_thresh is not None:
                keep_mask = fac_dep["share_cat"] >= share_thresh
                keep_mask = keep_mask | (fac_dep["country_origin"] == dep)
                fac_dep = fac_dep[keep_mask].copy()
                if fac_dep.empty:
                    continue
                totals = fac_dep.groupby("category_key")["Number"].transform("sum").astype(float)
                fac_dep["share_cat"] = np.where(totals > 0, fac_dep["Number"] / totals, 0.0)

            fac_dep["deployment_country"] = dep
            out.append(fac_dep)

    if not out:
        return bom_facilities.iloc[0:0].copy()
    result = pd.concat(out, ignore_index=True)

    # Sanity check: shares sum to 1 per (deployment_country, category_key)
    share_sums = (
        result.groupby(["deployment_country", "category_key"])["share_cat"]
        .sum()
    )
    bad = share_sums[(share_sums < 0.999) | (share_sums > 1.001)]
    if not bad.empty:
        print(
            "Warning: share_cat does not sum to 1 for some (deployment_country, category_key):"
        )
        for idx, val in bad.items():
            print(f"  {idx}: {val:.6f}")

    # Sanity check: missing deployment-country/category_key combinations
    expected = set(
        df_turbine_cat.dropna(subset=["category_key"])[
            ["deployment_country", "category_key"]
        ]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    present = set(
        result[["deployment_country", "category_key"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    missing = expected - present
    if missing:
        print(
            "Warning: missing (deployment_country, category_key) after "
            "share_thresh filtering:"
        )
        for dep, cat in sorted(missing):
            print(f"  ({dep}, {cat})")

    return result


# --------------------------------------------------
# 8. Allocation function
# --------------------------------------------------

def allocate_row(row, oem_supply, assembly_facilities, bom_facilities,
                 ownership_oem, oem_hq_country, country_coords):

    dep = row["deployment_country"]
    oem = row["OEM"]
    cg  = row["component_group"]
    sc  = row["sub_component"]          # keep as you have it
    cb  = row["cost_block"]
    val = float(row["value"])
    cat = row["category_key"]

    out = []

    # --------------------------------------------------
    # Assembly country (proximity; used only for Assembly + some Indirect)
    # --------------------------------------------------
    if cg == "Tower":
        asm_shares = tower_assembly_country_shares(assembly_facilities)
        asm_country = pick_tower_assembly_country(
            base_country=dep,
            facilities=assembly_facilities,
            country_coords=country_coords,
        )
    else:
        asm_country = (
            pick_assembly_facility_country(
                base_country=dep,
                oem=oem,
                cg=cg,
                facilities=assembly_facilities,
                oem_hq_country=oem_hq_country,
                country_coords=country_coords,
            )
            or dep
        )

    # --------------------------------------------------
    # STEP 1 — IMPORT SHARE (applies to all cost blocks in your model)
    # --------------------------------------------------
    import_frac = get_import_total_fraction(cg, sc)        # fraction of TOTAL value
    import_src  = get_import_country_fractions(cg, sc)     # {country: frac_of_total}

    import_val   = val * import_frac
    domestic_val = val - import_val

    # Imported part
    if import_val > 0 and import_src:
        for country, frac in import_src.items():
            if frac <= 0:
                continue
            out.append({
                **row,
                "origin_country": country,
                "origin_type": f"{cb} - Imports",
                "value_alloc": val * frac,
            })

    # If nothing left domestically, we’re done
    if domestic_val <= 0:
        return out

    # --------------------------------------------------
    # Helpers for BoM allocations (domestic_val)
    # --------------------------------------------------
    def alloc_by_facility_shares(category_key, origin_type, amount=None):
        """Allocate using bom_facilities shares (share_cat) for the given category_key."""
        base_value = domestic_val if amount is None else float(amount)
        if base_value <= 0:
            return True

        fac_cat = bom_facilities[bom_facilities["category_key"] == category_key]
        if "deployment_country" in fac_cat.columns:
            fac_cat = fac_cat[fac_cat["deployment_country"] == dep]
        if fac_cat.empty:
            return False

        shares = fac_cat.groupby("country_origin")["share_cat"].sum()
        shares = shares[shares > 0]
        if shares.empty:
            return False

        shares = shares / shares.sum()

        for c, w in shares.items():
            out.append({
                **row,
                "origin_country": c,
                "origin_type": origin_type,
                "value_alloc": base_value * float(w),
            })
        return True

    # --------------------------------------------------
    # Helper: known supplier weights (OEM-specific)
    # --------------------------------------------------
    def _known_supplier_weights():
        mask_sup = (
            (oem_supply["OEM"] == oem)
            & (oem_supply["component_group"] == cg)
            & (oem_supply["sub_component"] == sc)
        )
        supply_rows = oem_supply.loc[mask_sup].copy()
        if supply_rows.empty:
            return None
        supply_rows["is_oem"] = supply_rows["is_oem"].fillna(False).astype(bool)

        # external supplier counts by country
        ext = supply_rows[~supply_rows["is_oem"]]
        ext_counts = ext["country_origin"].dropna().value_counts().astype(float)

        # inhouse supplier count -> assembly facility country
        n_oem = int(supply_rows["is_oem"].sum())

        weights = {}
        if n_oem > 0:
            if pd.isna(asm_country):
                raise ValueError(
                    f"Known supplier marked in-house but asm_country is NA "
                    f"(dep={dep}, oem={oem}, cg={cg}, sc={sc})."
                )
            weights[asm_country] = weights.get(asm_country, 0.0) + float(n_oem)

        for c, n in ext_counts.items():
            weights[c] = weights.get(c, 0.0) + float(n)

        total = sum(weights.values())
        if total <= 0:
            return None
        return {c: float(n) / total for c, n in weights.items()}

    # --------------------------------------------------
    # INDIRECT COSTS (domestic part)
    # --------------------------------------------------
    if cb == "Indirect":
        if cg == "Tower":
            weights = _known_supplier_weights()
            if weights:
                for c, w in weights.items():
                    out.append({
                        **row,
                        "origin_country": c,
                        "origin_type": "Indirect - Tower",
                        "value_alloc": domestic_val * float(w),
                    })
            elif asm_shares:
                for c, w in asm_shares.items():
                    out.append({
                        **row,
                        "origin_country": c,
                        "origin_type": "Indirect - Tower",
                        "value_alloc": domestic_val * float(w),
                    })
            else:
                raise ValueError(
                    f"Tower Indirect allocation failed: no known suppliers and no tower facility shares "
                    f"(dep={dep}, oem={oem}, cg={cg}, sc={sc})."
                )
        else:
            out.append({
                **row,
                "origin_country": oem_hq_country.get(oem),
                "origin_type": "Indirect - OEM HQ",
                "value_alloc": domestic_val,
            })
        return out

    # --------------------------------------------------
    # ASSEMBLY COSTS (domestic part; proximity only)
    # --------------------------------------------------
    if cb == "Assembly":
        if cg == "Tower":
            weights = _known_supplier_weights()
            if weights:
                for c, w in weights.items():
                    out.append({
                        **row,
                        "origin_country": c,
                        "origin_type": "Assembly - Known facility",
                        "value_alloc": domestic_val * float(w),
                    })
            elif asm_shares:
                for c, w in asm_shares.items():
                    out.append({
                        **row,
                        "origin_country": c,
                        "origin_type": "Assembly - Known facility",
                        "value_alloc": domestic_val * float(w),
                    })
            else:
                raise ValueError(
                    f"Tower Assembly allocation failed: no known suppliers and no tower facility shares "
                    f"(dep={dep}, oem={oem}, cg={cg}, sc={sc})."
                )
        else:
            out.append({
                **row,
                "origin_country": asm_country,
                "origin_type": "Assembly - Known facility",
                "value_alloc": domestic_val,
            })
        return out

    # --------------------------------------------------
    # BoM COSTS (domestic part)
    # Order (Tower and non-Tower identical): oem_supply -> facilities -> (no fallback)
    # --------------------------------------------------
    if cb == "BoM":

        # --------------------------------------------------
        # Ownership split for configured category_key values.
        # --------------------------------------------------
        supp_value = domestic_val

        if pd.notna(cat) and cat in ownership_oem:
            oem_share = ownership_oem[cat] / 100.0
            oem_value = domestic_val * oem_share
            supp_value = domestic_val - oem_value

            if oem_value > 0:
                oem_country = (
                    pick_assembly_facility_country(
                        base_country=asm_country,
                        oem=oem,
                        cg=cg,
                        facilities=assembly_facilities,
                        oem_hq_country=oem_hq_country,
                        country_coords=country_coords,
                    )
                    or oem_hq_country.get(oem)
                    or asm_country
                )
                out.append({
                    **row,
                    "origin_country": oem_country,
                    "origin_type": "BoM - Unknown - Inhouse",
                    "value_alloc": oem_value,
                })

        # --------------------------------------------------
        # 1) OEM-specific suppliers (oem_supply) — FIXED
        #    Shares = supplier counts by country_origin.
        #    If is_oem == True -> allocate that count to asm_country.
        # --------------------------------------------------
        mask_sup = (
            (oem_supply["OEM"] == oem)
            & (oem_supply["component_group"] == cg)
            & (oem_supply["sub_component"] == sc)
        )
        supply_rows = oem_supply.loc[mask_sup].copy()

        if not supply_rows.empty:
            supply_rows["is_oem"] = supply_rows["is_oem"].fillna(False).astype(bool)

            # external supplier counts by country
            ext = supply_rows[~supply_rows["is_oem"]]
            ext_counts = ext["country_origin"].dropna().value_counts().astype(float)

            # inhouse supplier count -> assembly facility country
            n_oem = int(supply_rows["is_oem"].sum())

            weights = {}

            if n_oem > 0:
                if pd.isna(asm_country):
                    raise ValueError(
                        f"BoM: in-house supplier found but asm_country is NA "
                        f"(dep={dep}, oem={oem}, cg={cg}, sc={sc})."
                    )
                weights[asm_country] = weights.get(asm_country, 0.0) + float(n_oem)

            for c, n in ext_counts.items():
                weights[c] = weights.get(c, 0.0) + float(n)

            total = sum(weights.values())
            if total <= 0:
                raise ValueError(
                    f"BoM: OEM suppliers found but no valid allocation weights "
                    f"(dep={dep}, oem={oem}, cg={cg}, sc={sc})."
                )

            for c, n in weights.items():
                out.append({
                    **row,
                    "origin_country": c,
                    "origin_type": "BoM - OEM known supplier",
                    "value_alloc": supp_value * (n / total),
                })

            return out

        # --------------------------------------------------
        # 2) Industry facilities by category_key
        # --------------------------------------------------
        if pd.isna(cat):
            raise ValueError(
                f"BoM has no OEM suppliers and category_key is NA "
                f"(dep={dep}, oem={oem}, cg={cg}, sc={sc})."
            )

        ok = alloc_by_facility_shares(
            category_key=cat,
            origin_type="BoM - Unknown - Outsourced",
            amount=supp_value,
        )
        if ok:
            return out

        # --------------------------------------------------
        # 3) No fallback by design
        # --------------------------------------------------
        raise ValueError(
            f"BoM allocation failed: no OEM suppliers and no bom_facilities shares "
            f"for category_key={cat} (dep={dep}, oem={oem}, cg={cg}, sc={sc})."
        )

    raise ValueError(f"Unknown cost_block={cb} for row: {dict(row)}")


def allocate_all(
    df_turbine_cat,
    oem_supply,
    assembly_facilities,
    bom_facilities,
    ownership_oem,
    oem_hq_country,
    country_coords=COUNTRY_COORDS,
):

    out_rows = []
    for _, row in df_turbine_cat.iterrows():
        out_rows.extend(
            allocate_row(
                row,
                oem_supply,
                assembly_facilities,
                bom_facilities,
                ownership_oem,
                oem_hq_country,
                country_coords,
            )
        )
    return pd.DataFrame(out_rows)


def build_turbine_trade(df_alloc: pd.DataFrame) -> pd.DataFrame:
    """Build turbine trade long table from allocation results."""
    turbine_trade = (
        df_alloc.groupby(
            ["deployment_country", "component_group", "sub_component", "origin_country"],
            as_index=False
        )["value_alloc"].sum()
    )

    turbine_trade["Final_share"] = (
        turbine_trade.groupby(
            ["deployment_country", "component_group", "sub_component"]
        )["value_alloc"].transform(lambda x: x / x.sum())
    )

    return turbine_trade.rename(columns={
        "deployment_country": "Country_deploy",
        "origin_country": "Country_origin",
        "component_group": "CostCategory",
        "sub_component": "CostType",
    })


def _read_b2_sheet(workbook_path: str, sheet_name: str) -> pd.DataFrame:
    try:
        df = pd.read_excel(workbook_path, sheet_name=sheet_name, header=0)
    except ImportError as exc:
        if "openpyxl" not in str(exc).lower():
            raise
        from openpyxl import load_workbook

        wb = load_workbook(workbook_path, read_only=True, data_only=True)
        ws = wb[sheet_name]
        rows = ws.iter_rows(values_only=True)
        header = list(next(rows))
        df = pd.DataFrame(rows, columns=header)

    df = df.dropna(how="all").copy()
    df = df.loc[:, [c for c in df.columns if c is not None and not str(c).startswith("Unnamed:")]]
    return df


def _coerce_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "1": True, "yes": True, "y": True, "false": False, "0": False, "no": False, "n": False})
        .fillna(False)
        .astype(bool)
    )


def load_b2_oem_orders(workbook_path: str, *, sheet_name: str = "OEM_orders") -> pd.DataFrame:
    """Load OEM order observations from Supplementary Material B2."""
    df = _read_b2_sheet(workbook_path, sheet_name)
    required = {"OEM", "Turbine model", "deployment_country", "Capacity (MW)", "Year"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in B2 OEM_orders: {sorted(missing)}")
    df["deployment_country"] = df["deployment_country"].astype(str).str.strip()
    df["Capacity (MW)"] = pd.to_numeric(df["Capacity (MW)"], errors="coerce")
    return df


def load_b2_known_suppliers(workbook_path: str, *, sheet_name: str = "known_suppliers") -> pd.DataFrame:
    """Load OEM-specific known supplier inputs from Supplementary Material B2."""
    df = _read_b2_sheet(workbook_path, sheet_name)
    df = df.rename(columns={"CostType": "component_group", "CostSubType": "sub_component"})
    required = {"Model", "OEM", "source_component", "component_group", "sub_component", "is_oem", "Supplier", "EU location", "country_origin"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in B2 known_suppliers: {sorted(missing)}")
    df["component_group"] = df["component_group"].replace(CG_STD_MAP)
    df["country_origin"] = df["country_origin"].apply(normalize_country_code)
    df["is_oem"] = _coerce_bool_series(df["is_oem"])
    return df


def load_b2_assembly_facilities(workbook_path: str, *, sheet_name: str = "assembly_facilities") -> pd.DataFrame:
    """Load assembly and HQ facility inputs from Supplementary Material B2."""
    df = _read_b2_sheet(workbook_path, sheet_name)
    df = df.rename(columns={"CostType": "component_group"})
    required = {"Assembler", "Factory", "component_group", "country_origin", "Size (jobs)", "lat", "lon"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in B2 assembly_facilities: {sorted(missing)}")
    df["component_group"] = df["component_group"].replace(CG_STD_MAP)
    df["country_origin"] = df["country_origin"].apply(normalize_country_code)
    return df


def load_b2_general_facilities(workbook_path: str, *, sheet_name: str = "general_facilities") -> pd.DataFrame:
    """Load general manufacturing facility inputs from Supplementary Material B2."""
    df = _read_b2_sheet(workbook_path, sheet_name)
    df = df.rename(
        columns={
            "country_name": "Country",
            "FacilitiesCategory": "Component",
            "status": "Status",
            "number": "Number",
        }
    )
    required = {"Country", "country_origin", "Component", "Status", "Number"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in B2 general_facilities: {sorted(missing)}")
    df["country_origin"] = df["country_origin"].apply(normalize_country_code)
    df["Number"] = pd.to_numeric(df["Number"], errors="coerce")
    return df


def load_b2_turbine_inputs(workbook_path: str) -> dict:
    """
    Load public B2 input sheets used by the turbine allocation process.

    This preserves the allocation chain while replacing the four separate
    turbine source workbooks with the journal supplementary B2 workbook.
    """
    return {
        "oem_orders": load_b2_oem_orders(workbook_path),
        "known_suppliers": load_b2_known_suppliers(workbook_path),
        "assembly_facilities": load_b2_assembly_facilities(workbook_path),
        "general_facilities": load_b2_general_facilities(workbook_path),
    }


def build_turbine_allocations(
    cost_by_country: dict,
    *,
    share_dict_override: dict | None = None,
    assembly_facilities: pd.DataFrame | None = None,
    bom_facilities_override: pd.DataFrame | None = None,
    ownership_oem_override: dict | None = None,
    oem_hq_override: dict | None = None,
    country_coords_override: dict | None = None,
    component_group_map: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build allocation table (df_alloc) and trade table (turbine_trade)
    from cost_by_country.
    """
    share_dict_use = share_dict_override or share_dict
    cg_map = component_group_map or CG_STD_MAP
    assembly_fac_use = assembly_facilities if assembly_facilities is not None else assembly_fac
    bom_fac_use = bom_facilities_override if bom_facilities_override is not None else bom_facilities
    ownership_use = ownership_oem_override or ownership_oem
    oem_hq_use = oem_hq_override or oem_hq_country
    coords_use = country_coords_override or COUNTRY_COORDS

    cost_by_country_oem = split_turbine_costs_by_oem(cost_by_country, share_dict_use)
    df_turbine = pd.DataFrame(flatten_turbine_costs(cost_by_country_oem))
    if df_turbine.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_turbine["component_group"] = df_turbine["component_group"].replace(cg_map)
    df_turbine_cat = add_category_to_df_turbine(df_turbine, bridge)

    df_alloc = allocate_all(
        df_turbine_cat=df_turbine_cat,
        oem_supply=oem_supply,
        assembly_facilities=assembly_fac_use,
        bom_facilities=bom_fac_use,
        ownership_oem=ownership_use,
        oem_hq_country=oem_hq_use,
        country_coords=coords_use,
    )

    turbine_trade = build_turbine_trade(df_alloc)
    return df_alloc, turbine_trade


# ============================================================================
# Data Loading Helper
# ============================================================================

def load_supply_chain_data(data_path: str | None = None):
    """
    Load turbine supply-chain inputs from Supplementary Material B2.
    
    Parameters
    ----------
    data_path : str, optional
        Path to the B2 workbook or to the directory containing it. If None,
        uses ``SUPPLY_SOURCES / "turbine"`` from config.
        
    Returns
    -------
    dict
        Dictionary with loaded and prepared data:
        - turbine_orders, oem_share_df, share_dict, capacity_by_country
        - oem_supply, summary_table
        - assembly_fac, bom_facilities
    """
    from modules.config import SUPPLY_SOURCES

    default_name = "Supplementary_Material_B2_Turbine_Supply_Chain_Sourcing_Inputs.xlsx"
    path = Path(data_path) if data_path is not None else Path(SUPPLY_SOURCES) / "turbine"
    workbook_path = path if path.suffix.lower() == ".xlsx" else path / default_name

    b2_inputs = load_b2_turbine_inputs(str(workbook_path))

    turbine_orders_raw = b2_inputs["oem_orders"]
    share_dict, capacity_by_country, oem_share_df, turbine_orders = prepare_oem_share_by_country(turbine_orders_raw)

    oem_supply_raw = b2_inputs["known_suppliers"]
    summary_table, oem_supply = oem_suppliers(oem_supply_raw, component_group_map=CG_STD_MAP)

    assembly_fac = b2_inputs["assembly_facilities"]

    bom_fac_raw = b2_inputs["general_facilities"]

    KEEP = {"Castings", "Forgings", "Converters", "Tower"}
    DEPLOY = set(CORE_DEPLOYMENT)

    bom_facilities = prepare_bom_facilities(
        bom_fac_raw,
        keep=KEEP,
        deploy=DEPLOY,
        include_eu27=True,
    )
    
    return {
        "turbine_orders": turbine_orders,
        "oem_share_df": oem_share_df,
        "share_dict": share_dict,
        "capacity_by_country": capacity_by_country,
        "oem_supply": oem_supply,
        "summary_table": summary_table,
        "assembly_fac": assembly_fac,
        "bom_facilities": bom_facilities,
    }
