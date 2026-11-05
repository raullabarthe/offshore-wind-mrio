"""Wind-farm supplier processing and sourcing matrix construction."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from modules.country import NAME_TO_CODE

# -----------------------------------------------------------------------------
# Data loading helpers
# -----------------------------------------------------------------------------


def load_excel(path: str, sheet_name: str) -> pd.DataFrame:
    """Load one worksheet from an Excel workbook."""
    return pd.read_excel(path, sheet_name=sheet_name, header=0)

def load_data(
    base_dir: str,
    wf_filename: str = "4COffshore_windfarms.xlsx",
    suppliers_filename: str = "4COffshore_suppliers.xlsx",
    no_country_filename: str = "4COffshore_no_country.xlsx",
    suppliers_class_filename: str = "helper_cost_type.xlsx",
    suppliers_review_filename: str = "4COffshore_manual.xlsx",
    
    # Optional sheet names can be customised; defaults reflect the original script.
    wf_sheet: str = "Windfarm Project Details",
    suppliers_sheet: str = "Windfarm Supply Chain",
    no_country_sheet: str = "No country",
    suppliers_class_sheet: str = "SupplierMapping",
    suppliers_review_sheet: str = "Suppliers",
) -> Dict[str, pd.DataFrame]:
    """
    Load all required Excel datasets from a base directory.

    This function centralizes the loading of the external spreadsheets used in
    the analysis.  It keeps I/O separate from the data‑processing logic,
    allowing the core functions to operate on pre‑loaded DataFrames.

    Parameters
    ----------
    base_dir : str
        Directory containing the Excel files.
    wf_filename : str, optional
        Filename for the windfarm project details workbook.
    suppliers_filename : str, optional
        Filename for the windfarm supply chain workbook.
    no_country_filename : str, optional
        Filename for the table of stakeholders missing country information.
    suppliers_class_filename : str, optional
        Filename for the supplier type classification workbook.
    suppliers_review_filename : str, optional
    wf_sheet, suppliers_sheet, no_country_sheet,
    suppliers_class_sheet, suppliers_review_sheet, 
        Worksheet names for each Excel file.

    Returns
    -------
    dict
        A dictionary with keys corresponding to the dataset names and values
        equal to the loaded DataFrames.
    """
    datasets: Dict[str, pd.DataFrame] = {}
    datasets["wind_farms"] = load_excel(f"{base_dir}/{wf_filename}", wf_sheet)
    datasets["suppliers"] = load_excel(f"{base_dir}/{suppliers_filename}", suppliers_sheet)
    datasets["no_country"] = load_excel(f"{base_dir}/{no_country_filename}", no_country_sheet)
    datasets["suppliers_class"] = load_excel(f"{base_dir}/{suppliers_class_filename}", suppliers_class_sheet)
    datasets["suppliers_reviewed"] = load_excel(f"{base_dir}/{suppliers_review_filename}", suppliers_review_sheet)
    return datasets


# -----------------------------------------------------------------------------
# Cleaning functions
# -----------------------------------------------------------------------------
def clean_suppliers(suppliers: pd.DataFrame, no_country: pd.DataFrame,
                    core_deployment: List[str]) -> pd.DataFrame:
    """
    Filter, merge and clean the suppliers dataframe.

    This function fills missing country information from the ``no_country``
    mapping, restricts suppliers to core deployment countries, drops unused
    columns, and derives a year-of-source column for convenience.

    Parameters
    ----------
    suppliers : DataFrame
        Raw suppliers table loaded from 4C Offshore.
    no_country : DataFrame
        Table mapping WindfarmStakeholderId to OrganisationCountry for
        stakeholders missing this field.
    name_to_code : dict
        Mapping from country names to ISO alpha‑2 codes.
    core_deployment : list
        List of ISO codes for core deployment countries to retain.

    Returns
    -------
    DataFrame
        Cleaned suppliers table.
    """
    # Map missing organisation country from the auxiliary table
    no_country_map = no_country.set_index("WindfarmStakeholderId")["OrganisationCountry"]
    suppliers = suppliers.copy()
    suppliers["OrganisationCountry"] = suppliers.apply(
        lambda r: no_country_map.get(r["WindfarmStakeholderId"], r["OrganisationCountry"]),
        axis=1,
    )

    # Convert DateOfSource to year
    suppliers["YearOfSource"] = pd.to_datetime(suppliers["DateOfSource"], errors="coerce").dt.year

    # Filter by core deployment countries
    suppliers["CountryCode"] = suppliers["Country"].map(NAME_TO_CODE)
    suppliers = suppliers[suppliers["CountryCode"].isin(core_deployment)]

    # Drop unused columns
    cols_to_drop = [
        "WindfarmStatus", "Source", "Stake", "CountUnit", "DateOfSource",
        "OrganisationAddress", "OrganisationCity", "Website",
    ]
    suppliers = suppliers.drop(columns=[c for c in cols_to_drop if c in suppliers.columns], errors="ignore")
    return suppliers


def clean_windfarms(
    wind_farms: pd.DataFrame,
    windfarm_ids: List[int],
    core_deployment: List[str],
    min_capacity: float = 30.0,
    allowed_status: List[str] | None = None,
) -> pd.DataFrame:
    """
    Filter the windfarm projects dataframe to relevant projects.

    Only projects that appear in the suppliers table, are located in core
    deployment countries, have capacity exceeding ``min_capacity`` and are in
    one of the allowed status categories are retained.

    Parameters
    ----------
    wind_farms : DataFrame
        Raw windfarm projects table.
    windfarm_ids : list of int
        Windfarm IDs that appear in the suppliers table.
    name_to_code : dict
        Mapping from country names to ISO alpha‑2 codes.
    core_deployment : list
        List of ISO codes for core deployment countries.
    min_capacity : float, optional
        Minimum capacity (MW) threshold for inclusion.
    allowed_status : list of str, optional
        List of allowed windfarm status values (case sensitive).  If None,
        defaults to the categories used in the original analysis.

    Returns
    -------
    DataFrame
        Filtered windfarm projects.
    """
    wf = wind_farms[wind_farms["WindfarmId"].isin(windfarm_ids)].copy()
    wf["CountryCode"] = wf["CountryName"].map(NAME_TO_CODE)
    wf = wf[wf["CountryCode"].isin(core_deployment)]

    # Filter by capacity
    capacity_col = "CapacityMWMax"
    wf[capacity_col] = pd.to_numeric(wf[capacity_col], errors="coerce")
    wf = wf[wf[capacity_col] > min_capacity]

    # Filter by status
    if allowed_status is None:
        allowed_status = [
            "Fully Commissioned",
            "Under Construction",
            "Pre-Construction",
            "Partial Generation/Under Construction & Decommissioned",
        ]
    wf = wf[wf["WindfarmStatus"].isin(allowed_status)]
    return wf


# -----------------------------------------------------------------------------
# Classification and mapping
# -----------------------------------------------------------------------------
def classify_suppliers(
    suppliers: pd.DataFrame,
    suppliers_class: pd.DataFrame,
    default_category: str = "To review",
    default_type: str = "To review",
) -> pd.DataFrame:
    """
    Assign cost categories and cost types to suppliers based on their stakeholder type.

    Parameters
    ----------
    suppliers : DataFrame
        Cleaned suppliers table with a ``StakeholderType`` column.
    suppliers_class : DataFrame
        Table mapping stakeholder types to cost category and type.  This table
        should have columns ``Supplier type``, ``Cost category`` and ``Cost type``.
    default_category : str, optional
        Category assigned when a stakeholder type is not found in the mapping.
    default_type : str, optional
        Type assigned when a stakeholder type is not found in the mapping.

    Returns
    -------
    DataFrame
        Suppliers table augmented with ``CostCategory`` and ``CostType`` columns.
    """
    mapping = suppliers_class.rename(
        columns={
            "StakeholderType": "SupplierTypeMap",
            "CostCategory": "CostCategory",
            "CostType": "CostType",
        }
    )

    df = suppliers.copy()
    if "StakeholderType" not in df.columns:
        raise KeyError("Column 'StakeholderType' is missing in suppliers DataFrame.")

    # Merge cost categories by supplier type
    df = df.merge(
        mapping,
        how="left",
        left_on="StakeholderType",
        right_on="SupplierTypeMap",
    ).drop(columns=["SupplierTypeMap"])

    # Fill missing values
    df["CostCategory"] = df["CostCategory"].fillna(default_category)
    df["CostType"] = df["CostType"].fillna(default_type)

    # Ensure StakeDescription exists
    if "StakeDescription" not in df.columns:
        df["StakeDescription"] = ""
    return df

def apply_reviewed_mapping(
    suppliers_cost: pd.DataFrame,
    suppliers_reviewed: pd.DataFrame,
    *,
    id_col: str = "WindfarmStakeholderId",
) -> pd.DataFrame:
    revised = suppliers_reviewed[
        (suppliers_reviewed["CostCategory"] != "To review") |
        (suppliers_reviewed["CostType"] != "To review")
    ][[id_col, "CostCategory", "CostType"]].copy()

    out = suppliers_cost.copy().set_index(id_col)
    out.update(revised.set_index(id_col))
    return out.reset_index()



def update_classification_with_review(
    suppliers_cost: pd.DataFrame,
    suppliers_reviewed: pd.DataFrame,
) -> pd.DataFrame:
    """
    Update supplier classifications with manually reviewed entries.

    Entries in ``suppliers_reviewed`` override the ``CostCategory`` and
    ``CostType`` columns for matching ``WindfarmStakeholderId`` values.

    Parameters
    ----------
    suppliers_cost : DataFrame
        Suppliers table with initial cost classifications.
    suppliers_reviewed : DataFrame
        Manually reviewed table containing ``WindfarmStakeholderId``,
        ``CostCategory`` and ``CostType`` columns.

    Returns
    -------
    DataFrame
        Updated suppliers table.
    """
    df = suppliers_cost.set_index("WindfarmStakeholderId").copy()
    revised = suppliers_reviewed[
        (suppliers_reviewed["CostCategory"] != "To review") |
        (suppliers_reviewed["CostType"] != "To review")
    ][["WindfarmStakeholderId", "CostCategory", "CostType"]]
    revised = revised.set_index("WindfarmStakeholderId")
    df.update(revised)
    return df.reset_index()


# -----------------------------------------------------------------------------
# Trade dynamics
# -----------------------------------------------------------------------------
def compute_trade_dynamics(
    suppliers_cost: pd.DataFrame,
    name_to_code: Dict[str, str],
    core_row_order: List[str],
    intermediate_map: Dict[Tuple[str, str], str] | None = None,
) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
    """
    Compute trade dynamics (counts and shares) for each cost category and type.

    """

    df = suppliers_cost.copy()
    # Normalize country fields to improve ISO mapping
    for col in ("OrganisationCountry", "Country", "CountryCode"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"": np.nan, "nan": np.nan, "None": np.nan})

    def _to_iso(val: object) -> str | None:
        if pd.isna(val):
            return None
        v = str(val).strip()
        # if already ISO alpha-2, keep
        if len(v) == 2 and v.upper() in set(NAME_TO_CODE.values()):
            return v.upper()
        return NAME_TO_CODE.get(v)

    # Map names to ISO codes (origin = OrganisationCountry, deploy = CountryCode/Country)
    df["OrgCode"] = df["OrganisationCountry"].map(_to_iso)
    if "CountryCode" in df.columns:
        df["Country_Deploy"] = df["CountryCode"].map(_to_iso)
        df.loc[df["Country_Deploy"].isna(), "Country_Deploy"] = df["Country"].map(_to_iso)
    else:
        df["Country_Deploy"] = df["Country"].map(_to_iso)
    df["Country_Origin"] = df["OrgCode"]

    df = df.dropna(subset=["Country_Origin", "Country_Deploy", "CostCategory", "CostType"])

    # Assign intermediate cost level
    if intermediate_map is None:
        df["CostGroup"] = df["CostType"]
    else:
        df["CostGroup"] = df.apply(
            lambda r: intermediate_map.get((r["CostCategory"], r["CostType"]), r["CostType"]),
            axis=1,
        )

    def _compute_matrix(subset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        counts = pd.crosstab(subset["Country_Origin"], subset["Country_Deploy"])
        # Reorder rows: core countries first, then others (sorted) and rest of world
        extra_rows = [r for r in counts.index if r not in core_row_order and r != "RoW"]
        extra_rows_sorted = sorted(extra_rows)
        index_order = core_row_order + extra_rows_sorted
        if "RoW" in counts.index:
            index_order.append("RoW")
        counts = counts.reindex(index_order).fillna(0).astype(int)
        shares = counts.div(counts.sum(axis=0), axis=1).fillna(0.0)
        return counts, shares

    trade_dynamics: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = {}
    for cat, cat_df in df.groupby("CostCategory"):
        cat_dict: Dict[str, Dict[str, pd.DataFrame]] = {}
        # Total counts/shares for each cost category (aggregated across types)
        counts_total, shares_total = _compute_matrix(cat_df)
        cat_dict["_TOTAL"] = {
            "counts": counts_total,
            "shares": shares_total.style.format("{:.0%}"),
        }
        # Counts/shares by intermediate group
        for group, gdf in cat_df.groupby("CostGroup"):
            counts, shares = _compute_matrix(gdf)
            cat_dict[group] = {
                "counts": counts,
                "shares": shares.style.format("{:.0%}"),
            }
        trade_dynamics[cat] = cat_dict
    return trade_dynamics


def trade_dynamics_to_long(
    trade_dynamics: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
    *,
    period_label: str,
    exclude_cats: set[str] | None = None,
    keep_all_cats: set[str] | None = None,
    keep_groups_by_cat: Dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    exclude_cats = exclude_cats or set()
    keep_all_cats = keep_all_cats or set()
    keep_groups_by_cat = keep_groups_by_cat or {}

    rows = []
    for cost_cat, groups in trade_dynamics.items():
        if cost_cat in exclude_cats:
            continue

        # if no keep rules provided, keep everything (except _TOTAL)
        no_keep_rules = (len(keep_all_cats) == 0 and len(keep_groups_by_cat) == 0)

        for cost_group, obj in groups.items():
            if cost_group == "_TOTAL":
                continue

            if not no_keep_rules:
                keep = False
                if cost_cat in keep_all_cats:
                    keep = True
                if cost_cat in keep_groups_by_cat and cost_group in keep_groups_by_cat[cost_cat]:
                    keep = True
                if not keep:
                    continue

            counts = obj["counts"]
            col_sums = counts.sum(axis=0)
            shares = counts.div(col_sums, axis=1).fillna(0.0)

            for origin in counts.index:
                for deploy in counts.columns:
                    rows.append({
                        "CostCategory":   cost_cat,
                        "CostType":       cost_group,
                        "Period":         period_label,
                        "Country_origin": origin,
                        "Country_deploy": deploy,
                        "Counts":         int(counts.loc[origin, deploy]),
                        "Share":          float(shares.loc[origin, deploy]),
                    })

    df = pd.DataFrame(rows)
    # Normalize ISO codes (avoid hidden spaces)
    for col in ("Country_origin", "Country_deploy"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df



# -----------------------------------------------------------------------------
# Turbine trade table
# -----------------------------------------------------------------------------
def build_turbine_trade_long(df_alloc: pd.DataFrame) -> pd.DataFrame:
    """
    Convert allocated turbine cost data into a trade long table.

    Parameters
    ----------
    df_alloc : DataFrame
        Allocation table containing columns ``deployment_country``,
        ``component_group``, ``sub_component``, ``origin_country`` and
        ``value_alloc``.

    Returns
    -------
    DataFrame
        Trade long table with columns ``CostCategory`` (fixed to "Turbine"),
        ``CostType`` (component group), ``Country_deploy``, ``Country_origin`` and
        ``Final_share`` representing the share of each origin in total cost.
    """
    g = df_alloc.groupby(
        ["deployment_country", "component_group", "origin_country"],
        as_index=False
    )["value_alloc"].sum()
    g["total_cg_dep"] = g.groupby(
        ["deployment_country", "component_group"]
    )["value_alloc"].transform("sum")
    g["Final_share"] = g["value_alloc"] / g["total_cg_dep"]
    g = g.rename(columns={
        "deployment_country": "Country_deploy",
        "origin_country": "Country_origin",
        "component_group": "CostType",
    })
    g["CostCategory"] = "Turbine"
    return g[["CostCategory", "CostType", "Country_deploy", "Country_origin", "Final_share"]]



# -----------------------------------------------------------------------------
# Trade matrices
# -----------------------------------------------------------------------------
def _compute_trade_matrix(subset: pd.DataFrame, core_row_order: List[str], row_rest_of_world: str = "RoW") -> pd.DataFrame:
    """
    Compute a pivoted share matrix for a given subset of trade data.

    Rows correspond to origin countries and columns correspond to deployment
    countries.  Rows and columns are reordered according to the provided core
    row order, with any extra origins sorted alphabetically and the rest of
    world appended last.

    Parameters
    ----------
    subset : DataFrame
        Trade long table containing columns ``Country_origin``, ``Country_deploy`` and
        ``Final_share``.
    core_row_order : list
        List of core ISO country codes determining the preferred row order.
    row_rest_of_world : str, optional
        ISO code representing the rest of world.

    Returns
    -------
    DataFrame
        Pivot table with origins as rows, deployments as columns and shares as values.
    """
    sub = subset.copy()
    for col in ("Country_origin", "Country_deploy"):
        if col in sub.columns:
            sub[col] = sub[col].astype(str).str.strip()

    mat = sub.pivot_table(
        index="Country_origin",
        columns="Country_deploy",
        values="Final_share",
        aggfunc="sum",
        fill_value=0.0,
    )
    # Reorder columns to core row order
    mat = mat.reindex(columns=core_row_order, fill_value=0.0)
    existing_rows = list(mat.index)
    extra_rows = [r for r in existing_rows if r not in core_row_order and r != row_rest_of_world]
    extra_rows_sorted = sorted(extra_rows)
    row_order = core_row_order + extra_rows_sorted
    if row_rest_of_world in existing_rows:
        row_order.append(row_rest_of_world)
    mat = mat.reindex(row_order)
    mat = mat.fillna(0.0)
    return mat


def _compute_counts_matrix(
    subset: pd.DataFrame,
    core_row_order: List[str],
    row_rest_of_world: str = "RoW",
    value_col: str = "Counts",
) -> pd.DataFrame:
    """
    Compute a pivoted counts matrix for a given subset of trade data.

    Rows correspond to origin countries and columns correspond to deployment
    countries. Rows/columns are reordered to core_row_order, then extra origins
    alphabetically, and rest of world last if present.
    """
    sub = subset.copy()
    for col in ("Country_origin", "Country_deploy"):
        if col in sub.columns:
            sub[col] = sub[col].astype(str).str.strip()

    mat = sub.pivot_table(
        index="Country_origin",
        columns="Country_deploy",
        values=value_col,
        aggfunc="sum",
        fill_value=0.0,
    )
    mat = mat.reindex(columns=core_row_order, fill_value=0.0)
    existing_rows = list(mat.index)
    extra_rows = [r for r in existing_rows if r not in core_row_order and r != row_rest_of_world]
    extra_rows_sorted = sorted(extra_rows)
    row_order = core_row_order + extra_rows_sorted
    if row_rest_of_world in existing_rows:
        row_order.append(row_rest_of_world)
    mat = mat.reindex(row_order)
    mat = mat.fillna(0.0)
    return mat


def build_trade_matrices(
    trade_all_final: pd.DataFrame,
    core_row_order: List[str],
    row_rest_of_world: str = "RoW",
    period: str | None = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Construct trade share matrices for all cost categories and types.

    Parameters
    ----------
    trade_all_final : DataFrame
        Long table of trade data. If ``Final_share`` is present, it is used
        directly. Otherwise, if ``period`` is provided, it is computed from
        the corresponding counts column in the Steen trade table.
    core_row_order : list
        Preferred order of countries (rows and columns) in the matrices.
    row_rest_of_world : str, optional
        ISO code representing rest of world.

    Returns
    -------
    dict
        Nested dictionary ``lc_matrices[category][type] = matrix`` where
        each matrix is a DataFrame of final shares.
    """
    df = trade_all_final.copy()
    # normalize key columns to avoid hidden spaces / mismatches
    for col in ("CostCategory", "CostType", "Country_origin", "Country_deploy"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    if "Final_share" in df.columns:
        df["Final_share"] = pd.to_numeric(df["Final_share"], errors="coerce").fillna(0.0)

    if "Final_share" not in df.columns:
        if period is None:
            raise KeyError("Final_share not found and no period provided to compute shares.")

        col_map = {
            "2000-2019": "Counts (2000-2019)",
            "2008-2019": "Counts (2008-2019)",
            "2015-2019": "Counts (2015-2019)",
        }
        col = col_map.get(period, period)
        if col not in df.columns:
            raise KeyError(f"Counts column not found for period '{period}': {col}")

        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        totals = df.groupby(["CostCategory", "CostType", "Country_deploy"])[col].transform("sum")
        df["Final_share"] = np.where(totals > 0, df[col] / totals, 0.0)

    lc_matrices: Dict[str, Dict[str, pd.DataFrame]] = {}
    for (cc, ct), g in df.groupby(["CostCategory", "CostType"]):
        mat = _compute_trade_matrix(g, core_row_order, row_rest_of_world)
        if cc not in lc_matrices:
            lc_matrices[cc] = {}
        lc_matrices[cc][ct] = mat
    return lc_matrices

def replace_trade_with_steen(
    trade_matrices: Dict[str, Dict[str, pd.DataFrame]],
    steen_matrices: Dict[str, Dict[str, pd.DataFrame]],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Replace trade_matrices entries with Steen matrices when keys match
    (same CostCategory and CostType). Other entries remain unchanged.
    """
    from copy import deepcopy
    out = deepcopy(trade_matrices)
    for cc, types in steen_matrices.items():
        if cc not in out:
            continue
        for ct, mat in types.items():
            if ct in out[cc]:
                out[cc][ct] = mat
    return out

def build_trade_matrices_from_contracts(
    suppliers_cost: pd.DataFrame,
    *,
    df_alloc: pd.DataFrame,
    period_label: str,
    year_min: int | None = None,
    year_max: int | None = None,
    include_unknown_years: bool | None = None,
    core_row_order: List[str],
    row_rest_of_world: str = "RoW",
    exclude_abex: bool = True,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Build trade matrices from contract data, optionally filtered by YearOfSource.

    Parameters
    ----------
    suppliers_cost : DataFrame
        Cleaned + classified suppliers table.
    df_alloc : DataFrame
        Turbine allocation dataframe (for turbine trade).
    period_label : str
        Period label for build_trade_dynamics (e.g., "2015-2019").
    year_min, year_max : int, optional
        Filter YearOfSource between [year_min, year_max].
    core_row_order : list
        Preferred order of countries (rows and columns) in the matrices.
    row_rest_of_world : str, optional
        ISO code representing rest of world.
    exclude_abex : bool
        If True, drop ABEX from trade matrices.
    """
    sc = suppliers_cost.copy()
    if year_min is not None or year_max is not None:
        if "YearOfSource" not in sc.columns:
            raise KeyError("YearOfSource not found in suppliers_cost.")
        ys = sc["YearOfSource"]
        if include_unknown_years is None:
            known_min = ys.min(skipna=True)
            known_max = ys.max(skipna=True)
            include_unknown_years = (
                year_min is not None and year_max is not None
                and pd.notna(known_min) and pd.notna(known_max)
                and year_min <= known_min and year_max >= known_max
            )
        if year_min is not None and year_max is not None:
            mask = ys.between(year_min, year_max)
        elif year_min is not None:
            mask = ys >= year_min
        else:
            mask = ys <= year_max
        if include_unknown_years:
            mask = mask | ys.isna()
        sc = sc[mask]

    trade_4c_long = build_trade_dynamics(sc, period_label=period_label)
    trade_4c_shares = apply_shares(trade_4c_long)

    # Optionally drop ABEX from 4C trades
    if exclude_abex:
        trade_4c_shares = trade_4c_shares.loc[
            trade_4c_shares["CostCategory"] != "ABEX"
        ].copy()

    # Build matrices from contracts (4C)
    trade_matrices = build_trade_matrices(
        trade_all_final=trade_4c_shares,
        core_row_order=core_row_order,
        row_rest_of_world=row_rest_of_world,
    )

    # Add turbine matrices from model allocation (kept separate to avoid any mix-up)
    turbine_trade_long = build_turbine_trade_long(df_alloc)
    turbine_matrices = build_trade_matrices(
        trade_all_final=turbine_trade_long,
        core_row_order=core_row_order,
        row_rest_of_world=row_rest_of_world,
    )
    if "Turbine" in turbine_matrices:
        trade_matrices["Turbine"] = turbine_matrices["Turbine"]

    return trade_matrices


def load_b1_sourcing_long(
    workbook_path: str,
    *,
    sheet_name: str = "sourcing_long",
) -> pd.DataFrame:
    """
    Load the public Supplementary Material B1 sourcing_long sheet.

    This is the public entry point for the wind-farm sourcing layer. It starts
    from the processed audit-trail workbook instead of the restricted raw TGS
    4C Offshore supplier-contract files.
    """
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

    required = {
        "period",
        "component",
        "subcomponent",
        "dep_country",
        "org_country",
        "share_total",
        "final_local_c",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in B1 sourcing_long: {sorted(missing)}")

    df = df.dropna(subset=["period", "component", "subcomponent", "dep_country", "org_country"]).copy()
    for col in ["period", "component", "subcomponent", "dep_country", "org_country"]:
        df[col] = df[col].astype(str).str.strip()
    for col in ["share_total", "final_local_c"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def add_b1_final_share(
    sourcing_long: pd.DataFrame,
    *,
    base_col: str = "share_total",
    local_col: str = "final_local_c",
    out_col: str = "share_final_normalized",
) -> pd.DataFrame:
    """
    Reconstruct normalized post-benchmark sourcing shares from B1.

    The B1 workbook stores the post-benchmark domestic share in
    ``final_local_c``. This function sets each deployment country's domestic
    share to that value and rescales the non-domestic block to preserve a
    column sum of one.
    """
    required = {
        "period",
        "component",
        "subcomponent",
        "dep_country",
        "org_country",
        base_col,
        local_col,
    }
    missing = required - set(sourcing_long.columns)
    if missing:
        raise KeyError(f"Missing columns for B1 final-share reconstruction: {sorted(missing)}")

    out = sourcing_long.copy()
    out[out_col] = 0.0

    group_cols = ["period", "component", "subcomponent", "dep_country"]
    for _, idx in out.groupby(group_cols, sort=False).groups.items():
        idx = list(idx)
        g = out.loc[idx]
        dep = str(g["dep_country"].iloc[0]).strip()
        is_domestic = g["org_country"].astype(str).str.strip().eq(dep)

        base = pd.to_numeric(g[base_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        domestic_base_sum = float(base.loc[is_domestic].sum())
        nondomestic_base_sum = float(base.loc[~is_domestic].sum())

        local_values = pd.to_numeric(g.loc[is_domestic, local_col], errors="coerce").dropna()
        if len(local_values):
            domestic_final = float(local_values.max())
        else:
            domestic_final = domestic_base_sum
        domestic_final = min(max(domestic_final, 0.0), 1.0)

        if domestic_base_sum > 0:
            out.loc[g.index[is_domestic], out_col] = (
                base.loc[is_domestic] / domestic_base_sum * domestic_final
            ).to_numpy()
        elif is_domestic.any():
            out.loc[g.index[is_domestic], out_col] = domestic_final / int(is_domestic.sum())

        if nondomestic_base_sum > 0:
            out.loc[g.index[~is_domestic], out_col] = (
                base.loc[~is_domestic] / nondomestic_base_sum * (1.0 - domestic_final)
            ).to_numpy()
        else:
            out.loc[g.index[~is_domestic], out_col] = 0.0

    return out


def build_trade_matrices_from_b1_sourcing(
    workbook_path: str,
    *,
    core_row_order: List[str],
    df_alloc: pd.DataFrame | None = None,
    period: str = "2000-2025",
    share_version: str = "final",
    row_rest_of_world: str = "RoW",
    include_turbine: bool = True,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Build sourcing matrices from public Supplementary Material B1.

    ``share_version="total"`` uses the reviewed contract shares before the
    local-content correction. ``share_version="final"`` reconstructs the
    post-correction shares from ``share_total`` and ``final_local_c``.
    """
    sourcing_long = load_b1_sourcing_long(workbook_path)
    sourcing_long = sourcing_long.loc[sourcing_long["period"].eq(str(period))].copy()
    if sourcing_long.empty:
        raise ValueError(f"No rows found in B1 sourcing_long for period {period!r}.")

    version = str(share_version).strip().lower()
    if version in {"total", "pre", "reviewed"}:
        share_col = "share_total"
    elif version in {"final", "post", "adjusted"}:
        share_col = "share_final_normalized"
        sourcing_long = add_b1_final_share(sourcing_long, out_col=share_col)
    else:
        if share_version not in sourcing_long.columns:
            raise ValueError(
                "share_version must be one of 'total', 'final', or a column name "
                f"in sourcing_long; got {share_version!r}."
            )
        share_col = share_version

    trade_long = sourcing_long.rename(
        columns={
            "component": "CostCategory",
            "subcomponent": "CostType",
            "dep_country": "Country_deploy",
            "org_country": "Country_origin",
            share_col: "Final_share",
        }
    )[["CostCategory", "CostType", "Country_deploy", "Country_origin", "Final_share"]].copy()
    trade_long["Final_share"] = pd.to_numeric(trade_long["Final_share"], errors="coerce").fillna(0.0)

    trade_matrices = build_trade_matrices(
        trade_all_final=trade_long,
        core_row_order=core_row_order,
        row_rest_of_world=row_rest_of_world,
    )

    if include_turbine:
        if df_alloc is None:
            raise ValueError("df_alloc is required when include_turbine=True.")
        turbine_trade_long = build_turbine_trade_long(df_alloc)
        turbine_matrices = build_trade_matrices(
            trade_all_final=turbine_trade_long,
            core_row_order=core_row_order,
            row_rest_of_world=row_rest_of_world,
        )
        if "Turbine" in turbine_matrices:
            trade_matrices["Turbine"] = turbine_matrices["Turbine"]

    return trade_matrices


def build_contract_counts_long(
    suppliers_cost: pd.DataFrame,
    *,
    period_label: str,
    year_min: int | None = None,
    year_max: int | None = None,
    include_unknown_years: bool | None = None,
    core_row_order: List[str] | None = None,
    exclude_cats: set[str] | None = None,
    keep_all_cats: set[str] | None = None,
    keep_groups_by_cat: Dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    """
    Build a trade-long table with Counts and Share from contracts, optionally filtered by year.
    Output columns match trade_steen-style format:
    CostCategory, CostType, Country_origin, Country_deploy, Counts, Share.
    """
    sc = suppliers_cost.copy()
    if year_min is not None or year_max is not None:
        if "YearOfSource" not in sc.columns:
            raise KeyError("YearOfSource not found in suppliers_cost.")
        ys = sc["YearOfSource"]
        if include_unknown_years is None:
            known_min = ys.min(skipna=True)
            known_max = ys.max(skipna=True)
            include_unknown_years = (
                year_min is not None and year_max is not None
                and pd.notna(known_min) and pd.notna(known_max)
                and year_min <= known_min and year_max >= known_max
            )
        if year_min is not None and year_max is not None:
            mask = ys.between(year_min, year_max)
        elif year_min is not None:
            mask = ys >= year_min
        else:
            mask = ys <= year_max
        if include_unknown_years:
            mask = mask | ys.isna()
        sc = sc[mask]

    if core_row_order is None:
        core_row_order = ["GB", "BE", "NL", "DK", "DE"]

    td = compute_trade_dynamics(
        suppliers_cost=sc,
        name_to_code=NAME_TO_CODE,
        core_row_order=core_row_order,
    )
    return trade_dynamics_to_long(
        td,
        period_label=period_label,
        exclude_cats=exclude_cats,
        keep_all_cats=keep_all_cats,
        keep_groups_by_cat=keep_groups_by_cat,
    )


def build_counts_matrices(
    trade_long: pd.DataFrame,
    *,
    core_row_order: List[str],
    row_rest_of_world: str = "RoW",
    value_col: str = "Counts",
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Build count matrices (no normalization) from a trade-long table.
    """
    required = {"CostCategory", "CostType", "Country_origin", "Country_deploy", value_col}
    missing = required - set(trade_long.columns)
    if missing:
        raise KeyError(f"Missing columns in trade_long: {sorted(missing)}")

    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    for (cc, ct), g in trade_long.groupby(["CostCategory", "CostType"]):
        mat = _compute_counts_matrix(
            g,
            core_row_order=core_row_order,
            row_rest_of_world=row_rest_of_world,
            value_col=value_col,
        )
        if cc not in out:
            out[cc] = {}
        out[cc][ct] = mat
    return out


def build_contract_counts_matrices(
    suppliers_cost: pd.DataFrame,
    *,
    period_label: str,
    year_min: int | None = None,
    year_max: int | None = None,
    core_row_order: List[str],
    row_rest_of_world: str = "RoW",
    exclude_cats: set[str] | None = None,
    keep_all_cats: set[str] | None = None,
    keep_groups_by_cat: Dict[str, set[str]] | None = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Build count matrices directly from contracts (not shares), with optional year filtering.
    """
    trade_long = build_contract_counts_long(
        suppliers_cost,
        period_label=period_label,
        year_min=year_min,
        year_max=year_max,
        core_row_order=core_row_order,
        exclude_cats=exclude_cats,
        keep_all_cats=keep_all_cats,
        keep_groups_by_cat=keep_groups_by_cat,
    )
    return build_counts_matrices(
        trade_long,
        core_row_order=core_row_order,
        row_rest_of_world=row_rest_of_world,
        value_col="Counts",
    )


def build_steen_counts_matrices(
    trade_steen: pd.DataFrame,
    *,
    period: str,
    core_row_order: List[str],
    row_rest_of_world: str = "RoW",
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Build count matrices from the Steen trade table for a given period.
    Period can be '2000-2019', '2008-2019', '2015-2019', or a direct column name.
    """
    col_map = {
        "2000-2019": "Counts (2000-2019)",
        "2008-2019": "Counts (2008-2019)",
        "2015-2019": "Counts (2015-2019)",
    }
    col = col_map.get(period, period)
    if col not in trade_steen.columns:
        raise KeyError(f"Counts column not found for period '{period}': {col}")

    df = trade_steen.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df = df.rename(columns={col: "Counts"})
    return build_counts_matrices(
        df,
        core_row_order=core_row_order,
        row_rest_of_world=row_rest_of_world,
        value_col="Counts",
    )

def _collect_common_points(
    trade_matrices: Dict[str, Dict[str, pd.DataFrame]],
    steen_matrices: Dict[str, Dict[str, pd.DataFrame]],
    group: str,
    *,
    drop_zeros: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if group not in trade_matrices or group not in steen_matrices:
        return np.array([]), np.array([])

    xs, ys = [], []
    common_ct = set(trade_matrices[group]) & set(steen_matrices[group])
    for ct in common_ct:
        mt = trade_matrices[group][ct]
        ms = steen_matrices[group][ct]
        common_o = [o for o in mt.index if o in ms.index]
        common_d = [d for d in mt.columns if d in ms.columns]
        for o in common_o:
            for d in common_d:
                x = mt.loc[o, d]
                y = ms.loc[o, d]
                if not np.isfinite(x) or not np.isfinite(y):
                    continue
                if drop_zeros and (x == 0 or y == 0):
                    continue
                xs.append(float(x))
                ys.append(float(y))
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def fit_group_linear_regression(
    trade_matrices: Dict[str, Dict[str, pd.DataFrame]],
    steen_matrices: Dict[str, Dict[str, pd.DataFrame]],
    group: str,
    *,
    drop_zeros: bool = True,
    min_points: int = 5,
) -> dict:
    """
    Fit y = a + b x for a given group (e.g., BoP or Installation),
    using overlapping cells between trade_matrices (x) and steen_matrices (y).
    """
    xs, ys = _collect_common_points(trade_matrices, steen_matrices, group, drop_zeros=drop_zeros)
    if xs.size < max(2, min_points):
        raise ValueError(f"Not enough points to fit regression for {group}: n={xs.size}")
    slope, intercept = np.polyfit(xs, ys, 1)
    r = float(np.corrcoef(xs, ys)[0, 1]) if xs.size > 1 else np.nan
    return {
        "group": group,
        "slope": float(slope),
        "intercept": float(intercept),
        "r": r,
        "n": int(xs.size),
    }


def _apply_reg_to_matrix(
    mat: pd.DataFrame,
    *,
    slope: float,
    intercept: float,
    preserve_zeros: bool = True,
    clip: tuple[float, float] = (0.0, 1.0),
    renormalize: bool = True,
) -> pd.DataFrame:
    mat_adj = intercept + slope * mat
    if preserve_zeros:
        mat_adj = mat_adj.mask(mat == 0, 0.0)
    if clip is not None:
        lo, hi = clip
        mat_adj = mat_adj.clip(lower=lo, upper=hi)
    if renormalize:
        col_sums = mat_adj.sum(axis=0).replace(0, np.nan)
        mat_adj = mat_adj.div(col_sums, axis=1).fillna(0.0)
    return mat_adj


def apply_regression_to_missing_components(
    trade_matrices: Dict[str, Dict[str, pd.DataFrame]],
    steen_matrices: Dict[str, Dict[str, pd.DataFrame]],
    *,
    group: str,
    slope: float,
    intercept: float,
    clip: tuple[float, float] = (0.0, 1.0),
    renormalize: bool = True,
    preserve_zeros: bool = True,
) -> tuple[Dict[str, Dict[str, pd.DataFrame]], list[str]]:
    """
    Apply a group-level regression to components missing in Steen.
    Returns a new matrices dict and the list of missing CostTypes adjusted.
    """
    from copy import deepcopy
    out = deepcopy(trade_matrices)
    if group not in out:
        return out, []

    missing_ct = sorted(set(out[group]) - set(steen_matrices.get(group, {})))
    for ct in missing_ct:
        mat = out[group][ct].copy()
        out[group][ct] = _apply_reg_to_matrix(
            mat,
            slope=slope,
            intercept=intercept,
            preserve_zeros=preserve_zeros,
            clip=clip,
            renormalize=renormalize,
        )

    return out, missing_ct
    

def apply_group_regression_to_group(
    trade_matrices: Dict[str, Dict[str, pd.DataFrame]],
    *,
    group: str,
    slope: float,
    intercept: float,
    preserve_zeros: bool = True,
    clip: tuple[float, float] = (0.0, 1.0),
    renormalize: bool = True,
) -> tuple[Dict[str, Dict[str, pd.DataFrame]], list[str]]:
    """
    Apply a group-level regression to all components in the group.
    Returns updated matrices and list of adjusted components.
    """
    from copy import deepcopy
    out = deepcopy(trade_matrices)
    if group not in out:
        return out, []

    adjusted = []
    for ct, mat in out[group].items():
        out[group][ct] = _apply_reg_to_matrix(
            mat,
            slope=slope,
            intercept=intercept,
            preserve_zeros=preserve_zeros,
            clip=clip,
            renormalize=renormalize,
        )
        adjusted.append(ct)

    return out, adjusted


def fit_component_regressions_by_group(
    trade_matrices: Dict[str, Dict[str, pd.DataFrame]],
    steen_matrices: Dict[str, Dict[str, pd.DataFrame]],
    *,
    group: str,
    drop_zeros: bool = True,
    min_points: int = 5,
    force_intercept_zero: bool = True,
) -> dict:
    """
    Fit y = a + b x for each component within a group using overlapping cells.
    Returns dict: {component: reg_dict}.
    """
    if group not in trade_matrices or group not in steen_matrices:
        return {}

    regs = {}
    common_ct = set(trade_matrices[group]) & set(steen_matrices[group])
    for ct in sorted(common_ct):
        mt = trade_matrices[group][ct]
        ms = steen_matrices[group][ct]
        common_o = [o for o in mt.index if o in ms.index]
        common_d = [d for d in mt.columns if d in ms.columns]

        xs, ys = [], []
        for o in common_o:
            for d in common_d:
                x = mt.loc[o, d]
                y = ms.loc[o, d]
                if not np.isfinite(x) or not np.isfinite(y):
                    continue
                if drop_zeros and (x == 0 or y == 0):
                    continue
                xs.append(float(x))
                ys.append(float(y))

        xs = np.array(xs, dtype=float)
        ys = np.array(ys, dtype=float)
        if xs.size < max(2, min_points):
            continue

        if force_intercept_zero:
            denom = float(np.sum(xs * xs))
            slope = float(np.sum(xs * ys) / denom) if denom > 0 else 0.0
            intercept = 0.0
        else:
            slope, intercept = np.polyfit(xs, ys, 1)

        r = float(np.corrcoef(xs, ys)[0, 1]) if xs.size > 1 else np.nan
        regs[ct] = {
            "group": group,
            "component": ct,
            "slope": float(slope),
            "intercept": float(intercept),
            "r": r,
            "n": int(xs.size),
        }

    return regs


def apply_component_regressions_to_group(
    trade_matrices: Dict[str, Dict[str, pd.DataFrame]],
    *,
    group: str,
    component_regs: dict,
    preserve_zeros: bool = True,
    clip: tuple[float, float] = (0.0, 1.0),
    renormalize: bool = True,
) -> tuple[Dict[str, Dict[str, pd.DataFrame]], list[str]]:
    """
    Apply component-level regressions to all components with a fitted reg.
    Returns updated matrices and list of adjusted components.
    """
    from copy import deepcopy
    out = deepcopy(trade_matrices)
    if group not in out:
        return out, []

    adjusted = []
    for ct, mat in out[group].items():
        reg = component_regs.get(ct)
        if reg is None:
            continue
        out[group][ct] = _apply_reg_to_matrix(
            mat,
            slope=reg["slope"],
            intercept=reg["intercept"],
            preserve_zeros=preserve_zeros,
            clip=clip,
            renormalize=renormalize,
        )
        adjusted.append(ct)

    return out, adjusted


def fit_component_regressions_by_groups(
    trade_matrices: Dict[str, Dict[str, pd.DataFrame]],
    steen_matrices: Dict[str, Dict[str, pd.DataFrame]],
    *,
    groups: list[str] | tuple[str, ...],
    drop_zeros: bool = True,
    min_points: int = 5,
    force_intercept_zero: bool = True,
) -> dict:
    """
    Fit component-level regressions for multiple groups.
    Returns dict: {group: {component: reg_dict}}.
    """
    out: dict = {}
    for g in groups:
        regs = fit_component_regressions_by_group(
            trade_matrices,
            steen_matrices,
            group=g,
            drop_zeros=drop_zeros,
            min_points=min_points,
            force_intercept_zero=force_intercept_zero,
        )
        if regs:
            out[g] = regs
    return out


def fit_group_regressions(
    trade_matrices: Dict[str, Dict[str, pd.DataFrame]],
    steen_matrices: Dict[str, Dict[str, pd.DataFrame]],
    *,
    groups: list[str] | tuple[str, ...],
    drop_zeros: bool = True,
    min_points: int = 5,
) -> dict:
    """
    Fit group-level regressions for a list of groups.
    Returns dict: {group: reg_dict}.
    """
    out: dict = {}
    for g in groups:
        try:
            reg = fit_group_linear_regression(
                trade_matrices,
                steen_matrices,
                g,
                drop_zeros=drop_zeros,
                min_points=min_points,
            )
        except ValueError:
            continue
        out[g] = reg
    return out


def apply_component_regressions_to_counts(
    contract_counts: Dict[str, Dict[str, pd.DataFrame]],
    regs_by_group: dict,
    *,
    groups: list[str] | tuple[str, ...] | None = None,
    clip_min: float | None = 0.0,
    keep_unmodeled: bool = False,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Apply component-level regressions (assets ~ contracts) to count matrices.

    Parameters
    ----------
    contract_counts : dict
        Nested dict {Group: {Component: counts DataFrame}}
    regs_by_group : dict
        Output of fit_component_regressions_by_groups.
    groups : list/tuple, optional
        Subset of groups to apply. Defaults to all in regs_by_group.
    clip_min : float or None
        If set, clip values below this to clip_min (e.g., 0.0).
    """
    from copy import deepcopy

    if groups is None:
        groups = list(regs_by_group.keys())

    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    for g in groups:
        if g not in contract_counts:
            continue
        regs = regs_by_group.get(g, {})
        if not regs:
            continue

        out[g] = {}
        for ct, mat in contract_counts[g].items():
            reg = regs.get(ct)
            if reg is None:
                if keep_unmodeled:
                    out[g][ct] = mat.copy()
                continue
            slope = float(reg.get("slope", 0.0))
            intercept = float(reg.get("intercept", 0.0))
            adjusted = mat * slope + intercept
            if clip_min is not None:
                adjusted = adjusted.clip(lower=clip_min)
            out[g][ct] = adjusted

    return out


def estimate_missing_components_with_group_regression(
    trade_matrices_target: Dict[str, Dict[str, pd.DataFrame]],
    trade_matrices_base: Dict[str, Dict[str, pd.DataFrame]],
    steen_matrices_base: Dict[str, Dict[str, pd.DataFrame]],
    *,
    groups: list[str] | tuple[str, ...],
    drop_zeros: bool = True,
    min_points: int = 5,
    preserve_zeros: bool = True,
    clip: tuple[float, float] = (0.0, 1.0),
    renormalize: bool = True,
) -> tuple[Dict[str, Dict[str, pd.DataFrame]], dict, dict]:
    """
    Estimate missing components in target matrices using group-level regression
    fitted on base (contracts vs assets) matrices.

    Returns:
        updated_matrices, reg_summary_by_group, adjusted_components_by_group
    """
    from copy import deepcopy

    out = deepcopy(trade_matrices_target)
    reg_summary: dict = {}
    adjusted: dict = {}

    for g in groups:
        reg = fit_group_linear_regression(
            trade_matrices_base,
            steen_matrices_base,
            g,
            drop_zeros=drop_zeros,
            min_points=min_points,
        )
        reg_summary[g] = reg

        out, missing = apply_regression_to_missing_components(
            out,
            steen_matrices_base,
            group=g,
            slope=reg["slope"],
            intercept=reg["intercept"],
            preserve_zeros=preserve_zeros,
            clip=clip,
            renormalize=renormalize,
        )
        adjusted[g] = missing

    return out, reg_summary, adjusted


__all__ = [
    "load_data",
    "clean_suppliers",
    "clean_windfarms",
    "classify_suppliers",
    "update_classification_with_review",
    "compute_trade_dynamics",
    "build_turbine_trade_long",
    "build_trade_matrices",
    "replace_trade_with_steen",
    "build_trade_matrices_from_contracts",
    "load_b1_sourcing_long",
    "add_b1_final_share",
    "build_trade_matrices_from_b1_sourcing",
    "build_contract_counts_long",
    "build_counts_matrices",
    "build_contract_counts_matrices",
    "build_steen_counts_matrices",
    "compute_windfarm_filter_stats",
    "compute_expired_stats",
    "compute_mapping_stats",
    "build_report_enumerate",
    "fit_group_linear_regression",
    "apply_regression_to_missing_components",
    "apply_group_regression_to_group",
    "fit_component_regressions_by_group",
    "fit_component_regressions_by_groups",
    "apply_component_regressions_to_group",
    "fit_group_regressions",
    "apply_component_regressions_to_counts",
    "estimate_missing_components_with_group_regression",
    "audit_trade_matrix_column_sums",
    "trade_matrices_to_sourcing_long",
    "compare_trade_matrices_sourcing_long",
    "contract_counts_to_sourcing_long",
    "reviewed_contract_ids",
    "filter_final_supplier_contracts",
    "filter_sourcing_estimation_contracts",
    "manual_review_contract_sensitivity",
    "build_component_flows_checked",
    "apply_uk_benchmark_k",
]

def build_trade_dynamics(
    suppliers_cost: pd.DataFrame,
    *,
    period_label: str,
    core_row_order: List[str] | None = None,
    exclude_cats: set[str] | None = None,
    keep_all_cats: set[str] | None = None,
    keep_groups_by_cat: Dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    if core_row_order is None:
        core_row_order = ["GB","BE","NL","DK","DE"]

    # Drop 4C Offshore aggregated turbine row (CostType == "Turbine") so
    # turbine trade comes only from the model allocation.
    sc = suppliers_cost.copy()
    if "CostCategory" in sc.columns and "CostType" in sc.columns:
        sc = sc[~((sc["CostCategory"] == "Turbine") & (sc["CostType"] == "Turbine"))]

    td = compute_trade_dynamics(
        suppliers_cost=sc,
        name_to_code=NAME_TO_CODE,
        core_row_order=core_row_order,
    )
    return trade_dynamics_to_long(
        td,
        period_label=period_label,
        exclude_cats=exclude_cats,
        keep_all_cats=keep_all_cats,
        keep_groups_by_cat=keep_groups_by_cat,
    )



def apply_shares(trade_long: pd.DataFrame) -> pd.DataFrame:
    df = trade_long.copy()
    df["Final_share"] = df["Share"].astype(float)
    return df

def compute_windfarm_filter_stats(
    wf: pd.DataFrame,
    sup: pd.DataFrame,
    *,
    core_deployment: List[str],
    allowed_status: List[str],
    min_capacity: float = 30.0,
) -> dict:
    """
    Compute wind farm filter stats used in the report.
    """
    wf_core = wf.copy()
    wf_core["CountryCode"] = wf_core["CountryName"].map(NAME_TO_CODE)
    wf_core = wf_core.loc[wf_core["CountryCode"].isin(core_deployment)]

    wf_core["CapacityMWMax"] = pd.to_numeric(wf_core["CapacityMWMax"], errors="coerce")
    wf_comm = wf_core.loc[wf_core["CapacityMWMax"] > min_capacity]

    wf_final = wf_comm.loc[wf_comm["WindfarmId"].isin(sup["WindfarmId"])]

    wf_status = wf_final.loc[wf_final["WindfarmStatus"].isin(allowed_status)]
    status_counts = wf_status["WindfarmStatus"].value_counts().to_dict()

    contracts_after_status = sup.loc[sup["WindfarmId"].isin(wf_status["WindfarmId"])].shape[0]

    return {
        "raw_wf": len(wf),
        "raw_contracts": len(sup),
        "core_wf": len(wf_core),
        "comm_wf": len(wf_comm),
        "with_contract_wf": len(wf_final),
        "status_wf": len(wf_status),
        "status_counts": status_counts,
        "contracts_after_status": contracts_after_status,
    }


def compute_expired_stats(suppliers_df: pd.DataFrame) -> dict:
    """
    Compute counts before/after removing IsExpired == 'Yes'.
    """
    exp = suppliers_df["IsExpired"].astype(str).str.strip().str.lower()
    n_before = len(suppliers_df)
    n_after = int((exp != "yes").sum())
    drop_n = n_before - n_after
    drop_pct = 100 * (n_after / n_before) if n_before else 0.0
    return {
        "n_before": n_before,
        "n_after": n_after,
        "drop_n": drop_n,
        "drop_pct": drop_pct,
    }


def compute_mapping_stats(suppliers_cost: pd.DataFrame) -> dict:
    """
    Compute mapping coverage stats based on CostCategory/CostType == 'To review'.
    """
    n_total = len(suppliers_cost)
    n_to_review = (
        (suppliers_cost["CostCategory"].eq("To review")) |
        (suppliers_cost["CostType"].eq("To review"))
    ).sum()
    n_mapped = n_total - n_to_review
    pct_mapped = 100 * (n_mapped / n_total) if n_total else 0.0
    return {
        "n_total": n_total,
        "n_mapped": n_mapped,
        "n_to_review": int(n_to_review),
        "pct_mapped": pct_mapped,
    }


def build_report_enumerate(
    wf_stats: dict,
    expired_stats: dict,
    mapping_stats: dict,
    *,
    label_counts: str = "tab:appendix_supplier_contract_counts",
    label_country: str = "tab:supplier_contracts_country",
    sourcing_usage_stats: dict | None = None,
) -> str:
    """
    Build LaTeX enumerate text for the report summary.
    """
    sc = wf_stats.get("status_counts", {})
    def _get(name, default=0):
        return sc.get(name, default)

    sourcing_item = ""
    if sourcing_usage_stats is not None:
        auto_other = sourcing_usage_stats.get("automatic_other")
        auto_used = sourcing_usage_stats.get("automatic_used")
        manual_used = sourcing_usage_stats.get("manual_used")
        final_used = sourcing_usage_stats.get("final_used")

        if all(v is not None for v in [auto_other, auto_used, manual_used, final_used]):
            sourcing_item = (
                r"""  \item After all contracts are assigned to a cost type, """
                + f"{auto_other:,}"
                + r""" automatically mapped entries fall under \textit{Other}, including owners, shareholders, non-direct suppliers, turbine suppliers, and records without a usable supplier country. Together with other entries outside the sourcing-contract universe, this leaves """
                + f"{auto_used:,}"
                + r""" automatically mapped contracts for the sourcing estimation. Manual review adds """
                + f"{manual_used:,}"
                + r""" relevant contracts, yielding """
                + f"{final_used:,}"
                + r""" contracts used to estimate sourcing shares."""
                + "\n"
            )

    txt = r"""\begin{enumerate}
  \item Raw 4C Offshore data include """ + f"{wf_stats['raw_wf']:,}" + r""" wind farms and """ + f"{wf_stats['raw_contracts']:,}" + r""" contracts.
  \item Base filters are applied: core deployment countries (BE, DE, DK, NL, UK) yield """ + f"{wf_stats['core_wf']:,}" + r""" wind farms; commercial-scale projects ($>$30~MW) yield """ + f"{wf_stats['comm_wf']:,}" + r"""; requiring at least one contract yields """ + f"{wf_stats['with_contract_wf']:,}" + r""" wind farms.
  \item Status filtering retains """ + f"{wf_stats['status_wf']:,}" + r""" wind farms with effective or imminent execution: \textit{fully commissioned} (""" + f"{_get('Fully Commissioned'):,}" + r"""), \textit{under construction} (""" + f"{_get('Under Construction'):,}" + r"""), \textit{pre-construction} (""" + f"{_get('Pre-Construction'):,}" + r"""), and \textit{partial generation/under construction} (""" + f"{_get('Partial Generation/Under Construction'):,}" + r"""). For these projects, """ + f"{wf_stats['contracts_after_status']:,}" + r""" contracts are available.
  \item Some entries are expired (i.e., replaced by another supplier for the same service). To avoid double counting, we retain """ + f"{expired_stats['n_after']:,}" + r""" contracts (""" + f"{expired_stats['drop_pct']:.1f}" + r"""\%).
  \item A mapping between 4C Offshore stakeholder types and cost-model categories is defined, covering """ + f"{mapping_stats['n_mapped']:,}" + r""" contracts (""" + f"{mapping_stats['pct_mapped']:.2f}" + r"""\%); see Appendix Table~\ref{""" + label_counts + r"""}).
  \item The remaining """ + f"{mapping_stats['n_to_review']:,}" + r""" contracts -- under generic stakeholder types labelled as \textit{Other} -- are manually re-mapped based on their description fields.
""" + sourcing_item + r"""  \item The resulting allocation by deployment country is reported in Table~\ref{""" + label_country + r"""}.
\end{enumerate}
"""
    return txt

import pandas as pd
import numpy as np

def audit_trade_matrix_column_sums(
    trade_matrices: dict,
    *,
    required_deploy: list[str] | None = None,
    tol: float = 1e-6,
    share_version: str | None = None,
) -> pd.DataFrame:
    """
    Audit whether origin-country sourcing shares sum to one by deployment column.

    This is the shared column-sum check used by audit_trade_matrices and by
    contract sourcing exports.
    """
    rows = []

    for cc, types_dict in trade_matrices.items():
        if not isinstance(types_dict, dict):
            continue
        for ct, mat in types_dict.items():
            if not isinstance(mat, pd.DataFrame):
                continue

            cols_check = (
                [c for c in required_deploy if c in mat.columns]
                if required_deploy is not None
                else list(mat.columns)
            )
            if not cols_check:
                continue

            col_sums = mat[cols_check].sum(axis=0)
            for dep, s in col_sums.items():
                s_float = float(s) if np.isfinite(s) else np.nan
                diff = s_float - 1.0 if np.isfinite(s_float) else np.nan
                ok = bool(np.isfinite(s_float) and abs(diff) <= tol)
                issue = None
                if not np.isfinite(s_float):
                    issue = "non-finite"
                elif not ok:
                    issue = f"|sum-1|>{tol}"

                row = {
                    "CostCategory": cc,
                    "CostType": ct,
                    "Country_deploy": dep,
                    "ColSum": s_float,
                    "DiffFrom1": diff,
                    "OK": ok,
                    "Issue": issue,
                }
                if share_version is not None:
                    row = {"ShareVersion": share_version, **row}
                rows.append(row)

    columns = ["CostCategory", "CostType", "Country_deploy", "ColSum", "DiffFrom1", "OK", "Issue"]
    if share_version is not None:
        columns = ["ShareVersion"] + columns
    if not rows:
        return pd.DataFrame(columns=columns)

    out = pd.DataFrame(rows)
    return out.sort_values(
        [c for c in ["ShareVersion", "OK", "CostCategory", "CostType", "Country_deploy"] if c in out.columns],
        kind="mergesort",
    ).reset_index(drop=True)


def audit_trade_matrices(
    trade_matrices: dict,
    *,
    required_deploy: list[str],
    tol: float = 1e-6,
    require_exact_cols: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Audita:
      A) columnas de deployment presentes por matriz
      B) suma de shares por columna ≈ 1

    Returns:
      missing_cols_df: filas con columnas faltantes
      colsum_issues_df: filas con columnas cuya suma se aleja de 1
    """
    missing_rows = []
    for cc, types_dict in trade_matrices.items():
        for ct, mat in types_dict.items():
            if not isinstance(mat, pd.DataFrame):
                continue

            # --- (A) columnas faltantes / extra ---
            cols = list(mat.columns)
            missing = [c for c in required_deploy if c not in cols]
            extra   = [c for c in cols if c not in required_deploy]

            if missing or (require_exact_cols and extra):
                missing_rows.append({
                    "CostCategory": cc,
                    "CostType": ct,
                    "MissingDeployCols": missing,
                    "ExtraDeployCols": extra,
                })

    missing_cols_df = pd.DataFrame(missing_rows).sort_values(["CostCategory","CostType"], kind="mergesort") \
        if missing_rows else pd.DataFrame(columns=["CostCategory","CostType","MissingDeployCols","ExtraDeployCols"])

    colsum_audit_df = audit_trade_matrix_column_sums(
        trade_matrices,
        required_deploy=required_deploy,
        tol=tol,
    )
    colsum_issues_df = (
        colsum_audit_df.loc[~colsum_audit_df["OK"], ["CostCategory", "CostType", "Country_deploy", "ColSum", "Issue"]]
        .sort_values(["CostCategory", "CostType", "Country_deploy"], kind="mergesort")
        .reset_index(drop=True)
        if not colsum_audit_df.empty
        else pd.DataFrame(columns=["CostCategory","CostType","Country_deploy","ColSum","Issue"])
    )

    return missing_cols_df, colsum_issues_df


def trade_matrices_to_sourcing_long(
    trade_matrices: dict,
    *,
    period: str | None = None,
    share_col: str = "share",
) -> pd.DataFrame:
    """
    Convert nested sourcing-share matrices to a long table.

    Input shape:
        {component: {subcomponent: DataFrame(origin_country x deployment_country)}}

    Output columns:
        period, component, subcomponent, origin_country, deployment_country, share_col
    """
    rows = []

    for component, sub_map in trade_matrices.items():
        if not isinstance(sub_map, dict):
            continue
        for subcomponent, mat in sub_map.items():
            if not isinstance(mat, pd.DataFrame):
                continue

            df = mat.copy()
            df.index = df.index.astype(str)
            df.columns = df.columns.astype(str)

            long = (
                df.rename_axis(index="origin_country")
                  .reset_index()
                  .melt(
                      id_vars="origin_country",
                      var_name="deployment_country",
                      value_name=share_col,
                  )
            )
            long.insert(0, "component", str(component))
            long.insert(1, "subcomponent", str(subcomponent))
            if period is not None:
                long.insert(0, "period", str(period))
            rows.append(long)

    base_cols = ["component", "subcomponent", "origin_country", "deployment_country", share_col]
    if period is not None:
        base_cols = ["period"] + base_cols
    if not rows:
        return pd.DataFrame(columns=base_cols)

    out = pd.concat(rows, ignore_index=True)
    out[share_col] = pd.to_numeric(out[share_col], errors="coerce").fillna(0.0)

    sort_cols = [c for c in ["period", "component", "subcomponent", "deployment_country", "origin_country"] if c in out.columns]
    return out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def compare_trade_matrices_sourcing_long(
    trade_before: dict,
    trade_after: dict,
    *,
    period: str | None = None,
    before_col: str = "share_pre",
    after_col: str = "share_post",
) -> pd.DataFrame:
    """
    Build a single long sourcing table with before/after shares side by side.
    Missing before/after cells are filled with zero.
    """
    before = trade_matrices_to_sourcing_long(
        trade_before,
        period=period,
        share_col=before_col,
    )
    after = trade_matrices_to_sourcing_long(
        trade_after,
        period=period,
        share_col=after_col,
    )

    keys = [c for c in ["period", "component", "subcomponent", "origin_country", "deployment_country"] if c in before.columns or c in after.columns]
    out = before.merge(after, on=keys, how="outer")
    out[before_col] = pd.to_numeric(out[before_col], errors="coerce").fillna(0.0)
    out[after_col] = pd.to_numeric(out[after_col], errors="coerce").fillna(0.0)
    out["share_delta"] = out[after_col] - out[before_col]

    return out.sort_values(
        [c for c in ["period", "component", "subcomponent", "deployment_country", "origin_country"] if c in out.columns],
        kind="mergesort",
    ).reset_index(drop=True)


def contract_counts_to_sourcing_long(
    suppliers_cost: pd.DataFrame,
    *,
    period_label: str,
    count_col: str = "contract_count",
    year_min: int | None = None,
    year_max: int | None = None,
    include_unknown_years: bool | None = None,
    core_row_order: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build contract counts with the same long keys used by sourcing_long.
    """
    counts = build_contract_counts_long(
        suppliers_cost,
        period_label=period_label,
        year_min=year_min,
        year_max=year_max,
        include_unknown_years=include_unknown_years,
        core_row_order=core_row_order,
    )
    if counts.empty:
        return pd.DataFrame(columns=["period", "component", "subcomponent", "origin_country", "deployment_country", count_col])

    out = counts.rename(
        columns={
            "Period": "period",
            "CostCategory": "component",
            "CostType": "subcomponent",
            "Country_origin": "origin_country",
            "Country_deploy": "deployment_country",
            "Counts": count_col,
        }
    )
    keep = ["period", "component", "subcomponent", "origin_country", "deployment_country", count_col]
    out = out[keep].copy()
    out[count_col] = pd.to_numeric(out[count_col], errors="coerce").fillna(0).astype(int)
    for c in ["period", "component", "subcomponent", "origin_country", "deployment_country"]:
        out[c] = out[c].astype(str).str.strip()
    return out


def reviewed_contract_ids(
    suppliers_reviewed: pd.DataFrame,
    *,
    id_col: str = "WindfarmStakeholderId",
) -> set[str]:
    """
    Return IDs of contracts with a non-placeholder manual review assignment.
    """
    required = {id_col, "CostCategory", "CostType"}
    missing = required - set(suppliers_reviewed.columns)
    if missing:
        raise KeyError(f"Missing columns in suppliers_reviewed: {sorted(missing)}")

    reviewed = suppliers_reviewed[
        (suppliers_reviewed["CostCategory"].astype(str).str.strip() != "To review") |
        (suppliers_reviewed["CostType"].astype(str).str.strip() != "To review")
    ]
    return set(reviewed[id_col].dropna().astype(str).str.strip())


def filter_final_supplier_contracts(
    suppliers_cost: pd.DataFrame,
    *,
    exclude_categories: tuple[str, ...] = ("ABEX", "Other", "To review"),
    exclude_types: tuple[str, ...] = ("To review",),
) -> pd.DataFrame:
    """
    Apply the final supply-chain contract filter used before building shares.
    """
    df = suppliers_cost.copy()
    if "CostCategory" not in df.columns or "CostType" not in df.columns:
        raise KeyError("suppliers_cost must include CostCategory and CostType.")

    return df[
        (~df["CostCategory"].astype(str).str.strip().isin(exclude_categories)) &
        (~df["CostType"].astype(str).str.strip().isin(exclude_types))
    ].copy()


def filter_sourcing_estimation_contracts(
    suppliers_cost: pd.DataFrame,
    *,
    exclude_categories: tuple[str, ...] = ("ABEX", "Other", "To review"),
    exclude_types: tuple[str, ...] = ("To review",),
    drop_4c_turbine_aggregate: bool = True,
    require_usable_countries: bool = True,
) -> pd.DataFrame:
    """
    Apply the final contract universe used to estimate sourcing shares.

    This starts from reviewed/classified contracts, removes categories outside
    the sourcing model, drops the 4C aggregate turbine supplier rows because
    turbine sourcing is model-allocated from df_alloc, and optionally keeps only
    rows with origin/deployment countries that can be converted to ISO codes.
    """
    df = filter_final_supplier_contracts(
        suppliers_cost,
        exclude_categories=exclude_categories,
        exclude_types=exclude_types,
    )

    if drop_4c_turbine_aggregate:
        df = df[
            ~(
                df["CostCategory"].astype(str).str.strip().eq("Turbine") &
                df["CostType"].astype(str).str.strip().eq("Turbine")
            )
        ].copy()

    if require_usable_countries:
        valid_iso = set(NAME_TO_CODE.values())

        def _to_iso(val: object) -> str | None:
            if pd.isna(val):
                return None
            v = str(val).strip()
            if len(v) == 2 and v.upper() in valid_iso:
                return v.upper()
            return NAME_TO_CODE.get(v)

        origin = df["OrganisationCountry"].map(_to_iso)
        if "CountryCode" in df.columns:
            deploy = df["CountryCode"].map(_to_iso)
            deploy = deploy.where(deploy.notna(), df["Country"].map(_to_iso))
        else:
            deploy = df["Country"].map(_to_iso)
        df = df.loc[origin.notna() & deploy.notna()].copy()

    return df


def manual_review_contract_sensitivity(
    suppliers: pd.DataFrame,
    suppliers_class: pd.DataFrame,
    suppliers_reviewed: pd.DataFrame,
    *,
    df_alloc: pd.DataFrame,
    core_row_order: list[str],
    out_path=None,
    period_label: str = "2000-2025",
    year_min: int | None = 2000,
    year_max: int | None = 2025,
    include_unknown_years: bool | None = None,
    deploy: str = "GB",
    id_col: str = "WindfarmStakeholderId",
    exclude_expired: bool = True,
    apply_benchmark: bool = True,
    tol: float = 1e-6,
    engine: str | None = None,
) -> dict:
    """
    Sensitivity of final sourcing shares to including manually reviewed contracts.

    Counterfactual:
      - Exclude contracts that required manual review.

    Baseline:
      - Include those contracts after applying the reviewed mapping.
      - Optionally apply the UK benchmark k_c as a separate final step.

    This measures the influence of adding the reviewed observations, not the
    correctness of the reviewed labels.
    """
    base = suppliers.copy()
    if exclude_expired and "IsExpired" in base.columns:
        exp = base["IsExpired"].astype(str).str.strip().str.lower()
        base = base.loc[exp != "yes"].copy()

    classified = classify_suppliers(base, suppliers_class)
    manual_ids = reviewed_contract_ids(suppliers_reviewed, id_col=id_col)
    id_as_str = classified[id_col].astype(str).str.strip()

    suppliers_excl_manual = classified.loc[~id_as_str.isin(manual_ids)].copy()
    suppliers_with_manual = apply_reviewed_mapping(
        classified,
        suppliers_reviewed,
        id_col=id_col,
    )

    suppliers_excl_manual_final = filter_final_supplier_contracts(suppliers_excl_manual)
    suppliers_with_manual_final = filter_final_supplier_contracts(suppliers_with_manual)

    shares_excl_manual = build_trade_matrices_from_contracts(
        suppliers_excl_manual_final,
        df_alloc=df_alloc,
        period_label=period_label,
        year_min=year_min,
        year_max=year_max,
        include_unknown_years=include_unknown_years,
        core_row_order=core_row_order,
    )
    shares_with_manual = build_trade_matrices_from_contracts(
        suppliers_with_manual_final,
        df_alloc=df_alloc,
        period_label=period_label,
        year_min=year_min,
        year_max=year_max,
        include_unknown_years=include_unknown_years,
        core_row_order=core_row_order,
    )

    k_excl_manual = pd.DataFrame()
    k_with_manual = pd.DataFrame()
    if apply_benchmark:
        shares_excl_manual_post_kc, k_excl_manual = apply_uk_benchmark_k(
            shares_excl_manual,
            deploy=deploy,
        )
        shares_with_manual_post_kc, k_with_manual = apply_uk_benchmark_k(
            shares_with_manual,
            deploy=deploy,
        )
    else:
        shares_excl_manual_post_kc = shares_excl_manual
        shares_with_manual_post_kc = shares_with_manual

    before_col = "share_excl_manual_contracts"
    manual_col = "share_post_manual"
    kc_col = "share_post_kc"
    count_before_col = "contract_count_excl_manual"
    count_after_col = "contract_count_with_manual"
    counts_excl_manual_long = contract_counts_to_sourcing_long(
        suppliers_excl_manual_final,
        period_label=period_label,
        count_col=count_before_col,
        year_min=year_min,
        year_max=year_max,
        include_unknown_years=include_unknown_years,
        core_row_order=core_row_order,
    )
    counts_with_manual_long = contract_counts_to_sourcing_long(
        suppliers_with_manual_final,
        period_label=period_label,
        count_col=count_after_col,
        year_min=year_min,
        year_max=year_max,
        include_unknown_years=include_unknown_years,
        core_row_order=core_row_order,
    )
    sourcing_long = trade_matrices_to_sourcing_long(
        shares_excl_manual,
        period=period_label,
        share_col=before_col,
    )
    sourcing_manual = trade_matrices_to_sourcing_long(
        shares_with_manual,
        period=period_label,
        share_col=manual_col,
    )
    sourcing_kc = trade_matrices_to_sourcing_long(
        shares_with_manual_post_kc,
        period=period_label,
        share_col=kc_col,
    )
    key_cols = ["period", "component", "subcomponent", "origin_country", "deployment_country"]
    sourcing_long = sourcing_long.merge(sourcing_manual, on=key_cols, how="outer")
    sourcing_long = sourcing_long.merge(sourcing_kc, on=key_cols, how="outer")
    for col in [before_col, manual_col, kc_col]:
        sourcing_long[col] = pd.to_numeric(sourcing_long[col], errors="coerce").fillna(0.0)
    sourcing_long = sourcing_long.merge(
        counts_excl_manual_long[key_cols + [count_before_col]],
        on=key_cols,
        how="left",
    )
    sourcing_long = sourcing_long.merge(
        counts_with_manual_long[key_cols + [count_after_col]],
        on=key_cols,
        how="left",
    )
    sourcing_long[count_before_col] = pd.to_numeric(sourcing_long[count_before_col], errors="coerce").fillna(0).astype(int)
    sourcing_long[count_after_col] = pd.to_numeric(sourcing_long[count_after_col], errors="coerce").fillna(0).astype(int)
    sourcing_long["share_delta_manual"] = sourcing_long[manual_col] - sourcing_long[before_col]
    sourcing_long["share_delta_kc"] = sourcing_long[kc_col] - sourcing_long[manual_col]
    sourcing_long["share_delta_total"] = sourcing_long[kc_col] - sourcing_long[before_col]
    sourcing_long["abs_delta_manual"] = sourcing_long["share_delta_manual"].abs()
    sourcing_long["abs_delta_kc"] = sourcing_long["share_delta_kc"].abs()
    sourcing_long["abs_delta_total"] = sourcing_long["share_delta_total"].abs()
    sourcing_long["share_delta"] = sourcing_long["share_delta_manual"]
    sourcing_long["abs_delta"] = sourcing_long["abs_delta_manual"]
    sort_cols = ["period", "component", "subcomponent", "deployment_country", "origin_country"]
    sourcing_long = sourcing_long.sort_values(
        sort_cols,
        kind="mergesort",
    ).reset_index(drop=True)

    sensitivity_summary = (
        sourcing_long
        .groupby(["component", "subcomponent", "deployment_country"], as_index=False)
        .agg(
            l1_delta_manual=("abs_delta_manual", "sum"),
            tv_distance_manual=("abs_delta_manual", lambda s: 0.5 * float(s.sum())),
            max_abs_delta_manual=("abs_delta_manual", "max"),
            n_origins_changed_manual=("abs_delta_manual", lambda s: int((s > tol).sum())),
            l1_delta_kc=("abs_delta_kc", "sum"),
            tv_distance_kc=("abs_delta_kc", lambda s: 0.5 * float(s.sum())),
            max_abs_delta_kc=("abs_delta_kc", "max"),
            n_origins_changed_kc=("abs_delta_kc", lambda s: int((s > tol).sum())),
            l1_delta_total=("abs_delta_total", "sum"),
            tv_distance_total=("abs_delta_total", lambda s: 0.5 * float(s.sum())),
            max_abs_delta_total=("abs_delta_total", "max"),
            n_origins_changed_total=("abs_delta_total", lambda s: int((s > tol).sum())),
        )
        .assign(
            l1_delta=lambda d: d["l1_delta_manual"],
            tv_distance=lambda d: d["tv_distance_manual"],
            max_abs_delta=lambda d: d["max_abs_delta_manual"],
            n_origins_changed=lambda d: d["n_origins_changed_manual"],
        )
        .sort_values("tv_distance_manual", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )

    sourcing_sum_audit = pd.concat(
        [
            audit_trade_matrix_column_sums(
                shares_excl_manual,
                tol=tol,
                share_version=before_col,
            ),
            audit_trade_matrix_column_sums(
                shares_with_manual,
                tol=tol,
                share_version=manual_col,
            ),
            audit_trade_matrix_column_sums(
                shares_with_manual_post_kc,
                tol=tol,
                share_version=kc_col,
            ),
        ],
        ignore_index=True,
    )
    sourcing_sum_issues = sourcing_sum_audit[~sourcing_sum_audit["OK"]].reset_index(drop=True)

    export = None
    if out_path is not None:
        from pathlib import Path
        import importlib.util

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sourcing_nonzero = sourcing_long[
            sourcing_long[[before_col, manual_col, kc_col, "abs_delta_manual", "abs_delta_kc"]].abs().sum(axis=1) > tol
        ].reset_index(drop=True)

        def _exclude_turbine_for_excel(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            if "component" in out.columns:
                out = out[out["component"].astype(str).str.strip() != "Turbine"]
            if "CostCategory" in out.columns:
                out = out[out["CostCategory"].astype(str).str.strip() != "Turbine"]
            return out.reset_index(drop=True)

        sheets = {
            "sourcing_long": _exclude_turbine_for_excel(sourcing_long),
            "sourcing_nonzero": _exclude_turbine_for_excel(sourcing_nonzero),
            "sensitivity_summary": _exclude_turbine_for_excel(sensitivity_summary),
            "sourcing_sum_audit": _exclude_turbine_for_excel(sourcing_sum_audit),
            "sourcing_sum_issues": _exclude_turbine_for_excel(sourcing_sum_issues),
        }

        writer_engine = engine
        if writer_engine is None:
            writer_engine = "xlsxwriter" if importlib.util.find_spec("xlsxwriter") else "openpyxl"

        with pd.ExcelWriter(out_path, engine=writer_engine) as writer:
            for sheet_name, df_sheet in sheets.items():
                df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)

                if writer_engine == "xlsxwriter":
                    ws = writer.sheets[sheet_name]
                    n_rows, n_cols = df_sheet.shape
                    if n_cols:
                        ws.freeze_panes(1, 0)
                        workbook = writer.book
                        pct_fmt = workbook.add_format({"num_format": "0.000%"})
                        int_fmt = workbook.add_format({"num_format": "0"})
                        for i, col in enumerate(df_sheet.columns):
                            lengths = df_sheet[col].astype(str).str.len()
                            q95 = lengths.quantile(0.95) if n_rows else len(str(col))
                            q95 = 0 if pd.isna(q95) else int(q95)
                            width = max(11, min(32, max(len(str(col)), q95) + 2))
                            col_name = str(col)
                            fmt = None
                            if col_name.startswith("share") or col_name.startswith("abs_delta") or col_name.startswith("tv_distance") or col_name.startswith("l1_delta") or col_name.startswith("max_abs_delta") or col_name == "diff_from_1":
                                fmt = pct_fmt
                            elif col_name.startswith("contract_count") or col_name.startswith("n_origins"):
                                fmt = int_fmt
                            ws.set_column(i, i, width, fmt)
                        if n_rows:
                            ws.add_table(
                                0,
                                0,
                                n_rows,
                                n_cols - 1,
                                {
                                    "name": f"tbl_{sheet_name[:20]}",
                                    "columns": [{"header": str(c)} for c in df_sheet.columns],
                                    "style": "Table Style Medium 2",
                                },
                            )
                elif writer_engine == "openpyxl":
                    ws = writer.sheets[sheet_name]
                    ws.freeze_panes = "A2"
                    if df_sheet.shape[1] and df_sheet.shape[0]:
                        from openpyxl.worksheet.table import Table, TableStyleInfo

                        ref = f"A1:{ws.cell(row=df_sheet.shape[0] + 1, column=df_sheet.shape[1]).coordinate}"
                        tab = Table(displayName=f"tbl_{sheet_name[:20]}", ref=ref)
                        tab.tableStyleInfo = TableStyleInfo(
                            name="TableStyleMedium2",
                            showFirstColumn=False,
                            showLastColumn=False,
                            showRowStripes=True,
                            showColumnStripes=False,
                        )
                        ws.add_table(tab)

        export = {
            "path": out_path,
            "excel_excludes_component": "Turbine",
            "sourcing_long": sheets["sourcing_long"],
            "sourcing_nonzero": sheets["sourcing_nonzero"],
            "sensitivity_summary": sheets["sensitivity_summary"],
            "sourcing_sum_audit": sheets["sourcing_sum_audit"],
            "sourcing_sum_issues": sheets["sourcing_sum_issues"],
        }

    n_manual_in_scope = int(id_as_str.isin(manual_ids).sum())
    with_manual_ids = suppliers_with_manual_final[id_col].astype(str).str.strip()
    excl_manual_ids = suppliers_excl_manual_final[id_col].astype(str).str.strip()
    diagnostics = {
        "n_after_expired_filter": int(len(base)),
        "n_manual_reviewed_ids_total": int(len(manual_ids)),
        "n_manual_reviewed_ids_in_scope": n_manual_in_scope,
        "n_final_contracts_excl_manual": int(len(suppliers_excl_manual_final)),
        "n_final_contracts_with_manual": int(len(suppliers_with_manual_final)),
        "n_final_manual_reviewed_contracts_included": int(with_manual_ids.isin(manual_ids).sum()),
        "n_final_contracts_added_relative_to_excl_manual": int(len(suppliers_with_manual_final) - len(suppliers_excl_manual_final)),
        "n_final_contracts_excl_manual_still_reviewed_ids": int(excl_manual_ids.isin(manual_ids).sum()),
        "apply_benchmark": bool(apply_benchmark),
        "benchmark_deploy": deploy if apply_benchmark else None,
    }

    return {
        "diagnostics": diagnostics,
        "suppliers_excl_manual_final": suppliers_excl_manual_final,
        "suppliers_with_manual_final": suppliers_with_manual_final,
        "shares_excl_manual": shares_excl_manual,
        "shares_with_manual": shares_with_manual,
        "shares_excl_manual_post_kc": shares_excl_manual_post_kc,
        "shares_with_manual_post_kc": shares_with_manual_post_kc,
        "shares_excl_manual_final": shares_excl_manual_post_kc,
        "shares_with_manual_final": shares_with_manual_post_kc,
        "k_excl_manual": k_excl_manual,
        "k_with_manual": k_with_manual,
        "counts_excl_manual_long": counts_excl_manual_long,
        "counts_with_manual_long": counts_with_manual_long,
        "sourcing_long": sourcing_long,
        "sensitivity_summary": sensitivity_summary,
        "sourcing_sum_audit": sourcing_sum_audit,
        "sourcing_sum_issues": sourcing_sum_issues,
        "export": export,
    }



# ------------------ build flows: project (L1 turbine) + turbine detail (L0) ------------------
def build_hier(flows):
    out = {}
    for dep, compdict in flows.items():
        out.setdefault(dep, {})
        for name, origins in compdict.items():
            parent, child = [x.strip() for x in name.split(":", 1)]
            out[dep].setdefault(parent, {})
            out[dep][parent][child] = origins
    return out


def build_component_flows(
    cost_by_country: dict,
    trade_matrices: dict,
    df_alloc: pd.DataFrame,
) -> tuple[dict, dict, dict, list]:
    """
    Build comprehensive component flows from cost model and turbine allocations.
    
    This function integrates:
    1. Turbine flows (from df_alloc) at two levels:
       - L1: Turbine: {component_group} (e.g., "Turbine: Blades")
       - L0: {component_group}: {sub_component} (e.g., "Drivetrain: Bedplate")
    2. Wind farm component flows (Devex, BoP, Installation, O&M) from cost_by_country
    
    Parameters
    ----------
    cost_by_country : dict
        Cost breakdown by deployment country. Structure:
        {country: {"Devex": {...}, "BoP": {...}, "Installation": {...}, "O&M": {...}}}
    trade_matrices : dict
        Trade share matrices by cost category. Structure:
        {category: {sub_component: {deployment_country: {origin_country: share}}}}
    df_alloc : pd.DataFrame
        Turbine allocation dataframe with columns:
        - deployment_country
        - component_group (e.g., "Blades", "Hub", "Nacelle", etc.)
        - sub_component (e.g., "Bedplate", "Main shaft", etc.)
        - origin_country
        - value_alloc
    
    Returns
    -------
    flows_by_component_hier_all : dict
        Hierarchical flows for project-level (L1) visualization.
        Structure: {deployment_country: {parent: {child: {origin_country: value}}}}
    flows_turbine_detail_hier : dict
        Hierarchical flows for turbine detail (L0) visualization.
        Structure: {deployment_country: {parent: {child: {origin_country: value}}}}
    flows_turbine_L1 : dict
        Raw turbine L1 flows (before hierarchy).
        Structure: {deployment_country: {component_name: {origin_country: value}}}
    all_origins : list
        Sorted list of all unique origin countries across all flows.
    
    Notes
    -----
    - SKIP_KEYS = {"Total", "total", "TOTAL"} are excluded from processing
    - O&M costs use "lifetime_breakdown" if available, otherwise the top-level dict
    - Missing matrices in trade_matrices will raise KeyError
    
    Example
    -------
    >>> flows_hier, turbine_hier, turbine_L1, origins = build_component_flows(
    ...     cost_by_country, trade_matrices, df_alloc
    ... )
    >>> # Use flows_hier for sankey diagrams
    >>> # Use turbine_hier for detailed turbine breakdowns
    >>> # Use origins for legend/color mapping
    """
    from collections import defaultdict
    
    # =========================================================================
    # Helper functions
    # =========================================================================
    def build_hier(flows):
        """Convert flat flows to hierarchical structure."""
        out = {}
        for dep, compdict in flows.items():
            out.setdefault(dep, {})
            for name, origins in compdict.items():
                parent, child = [x.strip() for x in name.split(":", 1)]
                out[dep].setdefault(parent, {})
                out[dep][parent][child] = origins
        return out
    
    def get_amt(block, k):
        """Extract amount from nested or simple dict."""
        v = block[k]
        if isinstance(v, dict) and "Total" in v:
            return float(v["Total"])
        return float(v)
    
    def add_component(dep, component_name, amt, shares_dict):
        """Add component flow with origin country shares."""
        for origin, sh in shares_dict.items():
            flows_by_component[dep][component_name][origin] += amt * float(sh)
    
    SKIP_KEYS = {"Total", "total", "TOTAL"}
    
    # =========================================================================
    # 1. Build turbine flows from df_alloc
    # =========================================================================
    
    # Turbine L1: Project-level aggregation (Turbine: Blades, Turbine: Hub, ...)
    flows_turbine_L1 = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for (dep, block, origin), g in df_alloc.groupby(
        ["deployment_country", "component_group", "origin_country"]
    ):
        flows_turbine_L1[dep][f"Turbine: {block}"][origin] += float(g["value_alloc"].sum())
    
    # Turbine L0: Detailed component breakdown (Drivetrain: Bedplate, ...)
    flows_turbine_L0 = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for (dep, block, part, origin), g in df_alloc.groupby(
        ["deployment_country", "component_group", "sub_component", "origin_country"]
    ):
        flows_turbine_L0[dep][f"{block}: {part}"][origin] += float(g["value_alloc"].sum())
    
    flows_turbine_detail_hier = build_hier(flows_turbine_L0)
    
    # =========================================================================
    # 2. Build wind farm component flows from cost_by_country
    # =========================================================================
    
    flows_by_component = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    
    for dep, data in cost_by_country.items():
        
        # ---- Devex (Development & Engineering) ----
        if "Devex" in data:
            devex = data["Devex"]
            tm_devex = trade_matrices["Devex"]
            
            for subkey in devex.keys():
                if subkey in SKIP_KEYS:
                    continue
                if subkey not in tm_devex:
                    raise KeyError(
                        f"[MISSING MATRIX] {dep=} Devex/{subkey=} not found in trade_matrices"
                    )
                
                amt = get_amt(devex, subkey)
                shares = tm_devex[subkey][dep]
                add_component(dep, f"Devex: {subkey}", amt, shares)
        
        # ---- BoP (Balance of Plant) ----
        if "BoP" in data:
            bop = data["BoP"]
            tm_bop = trade_matrices["BoP"]
            
            for subkey in bop.keys():
                if subkey in SKIP_KEYS:
                    continue
                if subkey not in tm_bop:
                    continue
                
                amt = get_amt(bop, subkey)
                shares = tm_bop[subkey][dep]
                add_component(dep, f"BoP: {subkey}", amt, shares)
        
        # ---- Installation ----
        if "Installation" in data:
            inst = data["Installation"]
            tm_inst = trade_matrices["Installation"]
            
            for subkey in inst.keys():
                if subkey in SKIP_KEYS:
                    continue
                if subkey not in tm_inst:
                    continue
                
                amt = get_amt(inst, subkey)
                shares = tm_inst[subkey][dep]
                add_component(dep, f"Installation: {subkey}", amt, shares)
        
        # ---- O&M (Operations & Maintenance) ----
        if "O&M" in data:
            om_block = data["O&M"]
            # Use lifetime_breakdown if available, otherwise use top-level
            om = om_block.get("lifetime_breakdown", om_block)
            tm_om = trade_matrices["O&M"]
            
            for subkey in om.keys():
                if subkey in SKIP_KEYS:
                    continue
                if subkey not in tm_om:
                    continue
                
                amt = get_amt(om, subkey)
                shares = tm_om[subkey][dep]
                add_component(dep, f"O&M: {subkey}", amt, shares)
    
    # =========================================================================
    # 3. Inject turbine L1 flows into project flows
    # =========================================================================
    
    for dep, compdict in flows_turbine_L1.items():
        for comp, origins in compdict.items():
            for origin, val in origins.items():
                flows_by_component[dep][comp][origin] += val
    
    flows_by_component_hier_all = build_hier(flows_by_component)
    
    # =========================================================================
    # 4. Extract all unique origin countries
    # =========================================================================
    
    all_origins = set()
    
    # From project-level flows
    for dep in flows_by_component_hier_all:
        for parent in flows_by_component_hier_all[dep]:
            for child in flows_by_component_hier_all[dep][parent]:
                all_origins.update(
                    map(str, flows_by_component_hier_all[dep][parent][child].keys())
                )
    
    # From turbine detail flows
    for dep in flows_turbine_detail_hier:
        for parent in flows_turbine_detail_hier[dep]:
            for child in flows_turbine_detail_hier[dep][parent]:
                all_origins.update(
                    map(str, flows_turbine_detail_hier[dep][parent][child].keys())
                )
    
    all_origins = sorted(all_origins)
    
    return (
        flows_by_component_hier_all,
        flows_turbine_detail_hier,
        flows_turbine_L1,
        all_origins,
    )


def build_component_flows_checked(
    *,
    cost_by_country: dict,
    trade_matrices: dict,
    df_alloc: pd.DataFrame,
    required_deploy: list[str],
    tol: float = 1e-6,
) -> tuple[dict, dict, dict, list, pd.DataFrame, pd.DataFrame]:
    """
    Build component flows and audit trade matrices in one call.

    Returns:
      flows_by_component_hier_all, flows_turbine_detail_hier, flows_turbine_L1, all_origins,
      missing_cols_df, colsum_issues_df
    """
    missing_cols_df, colsum_issues_df = audit_trade_matrices(
        trade_matrices,
        required_deploy=required_deploy,
        tol=tol,
    )
    flows = build_component_flows(
        cost_by_country=cost_by_country,
        trade_matrices=trade_matrices,
        df_alloc=df_alloc,
    )
    return (*flows, missing_cols_df, colsum_issues_df)


def _norm_gb_label_sa(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("&", " and ")
    s = s.replace("-", " ")
    s = " ".join(s.split())
    if s == "oss":
        s = "offshore substation"
    if s.endswith("cables"):
        s = s[:-1]
    return s


def _get_gb_inputs_sa():
    import importlib
    import modules.supply_chain.inputs as sc_inputs
    if not hasattr(sc_inputs, "GB_local") or not hasattr(sc_inputs, "GB_CATEGORY_MAP"):
        sc_inputs = importlib.reload(sc_inputs)
    if not hasattr(sc_inputs, "GB_local") or not hasattr(sc_inputs, "GB_CATEGORY_MAP"):
        raise ImportError("GB_local / GB_CATEGORY_MAP not found in modules.supply_chain.inputs")
    return sc_inputs.GB_local, sc_inputs.GB_CATEGORY_MAP


def _map_gb_category_sa(cost_cat: str, cost_type: str | None) -> str | None:
    GB_local, GB_CATEGORY_MAP = _get_gb_inputs_sa()
    cc_raw_in = str(cost_cat).split(":", 1)[0].strip()
    ct_raw_in = str(cost_type).split(":", 1)[-1].strip() if cost_type is not None else None
    cc = _norm_gb_label_sa(cc_raw_in)
    ct = _norm_gb_label_sa(ct_raw_in) if ct_raw_in is not None else None

    for (cc_raw, ct_raw), target in GB_CATEGORY_MAP.items():
        if ct_raw is None:
            continue
        if _norm_gb_label_sa(cc_raw) == cc and _norm_gb_label_sa(ct_raw) == ct:
            return target

    for (cc_raw, ct_raw), target in GB_CATEGORY_MAP.items():
        if ct_raw is not None:
            continue
        if _norm_gb_label_sa(cc_raw) == cc:
            return target

    # fallback: map subcomponents of aggregated benchmark categories
    if cc in {"devex", "o and m", "turbine", "abex"}:
        if cc == "o and m":
            return "O&M"
        return cost_cat

    return None


def apply_uk_benchmark_k(
    trade_matrices: dict,
    *,
    deploy: str = "GB",
) -> tuple[dict, pd.DataFrame]:
    """
    Apply UK benchmark (BVG) by computing a per-component k factor from UK.
    For the UK column, the domestic share is set to the benchmark and the
    remaining shares are renormalized to sum to 1 - benchmark.
    For other deployments, the domestic share is corrected as
    LC' = k_c * LC_contract, and the non-domestic block is rescaled to
    fill the remaining share (1 - LC').

    Returns:
        adjusted_matrices, k_df
    """
    from copy import deepcopy

    GB_local, _ = _get_gb_inputs_sa()
    bench_data = GB_local.get("data", {})
    bench_norm = {_norm_gb_label_sa(k): v for k, v in bench_data.items()}

    def _bench_for(cc: str, ct: str | None):
        mapped = _map_gb_category_sa(cc, ct)
        bench = bench_norm.get(_norm_gb_label_sa(mapped), np.nan) if mapped else np.nan
        if bench is not None:
            bench = pd.to_numeric(bench, errors="coerce")
            if isinstance(bench, (pd.Series, pd.DataFrame)):
                bench = bench.values.flatten()[0] if bench.size else np.nan
        return mapped, bench
    def _resolve_dep(mat: pd.DataFrame, dep: str) -> tuple[str | None, str | None]:
        dep_col = dep if dep in mat.columns else None
        dep_row = dep if dep in mat.index else None
        if (dep_col is None or dep_row is None) and dep == "GB":
            if dep_col is None and "UK" in mat.columns:
                dep_col = "UK"
            if dep_row is None and "UK" in mat.index:
                dep_row = "UK"
        if (dep_col is None or dep_row is None) and dep == "UK":
            if dep_col is None and "GB" in mat.columns:
                dep_col = "GB"
            if dep_row is None and "GB" in mat.index:
                dep_row = "GB"
        return dep_col, dep_row

    out = deepcopy(trade_matrices)
    rows = []

    for cc, types in out.items():
        # Turbine is model-allocated (df_alloc). Do not scale.
        if str(cc) == "Turbine":
            for ct, mat in types.items():
                rows.append({
                    "Category": cc,
                    "Component": ct,
                    "Deploy": deploy,
                    "LC_contract": np.nan,
                    "LC_benchmark": np.nan,
                    "BenchmarkCategory": None,
                    "k": 1.0,
                    "Applied": False,
                })
            continue

        for ct, mat in types.items():
            if not isinstance(mat, pd.DataFrame):
                continue
            dep_col, dep_row = _resolve_dep(mat, deploy)
            if dep_col is None or dep_row is None:
                rows.append({
                    "Category": cc, "Component": ct, "Deploy": deploy,
                    "LC_contract": np.nan, "LC_benchmark": np.nan, "k": np.nan, "Applied": False
                })
                continue

            mapped, bench = _bench_for(cc, ct)

            lc_contract_uk = float(mat.loc[dep_row, dep_col])
            lc_bench_uk = float(bench) / 100.0 if not pd.isna(bench) else np.nan

            applied = False
            k = 1.0

            if np.isfinite(lc_contract_uk) and np.isfinite(lc_bench_uk) and lc_contract_uk > 0.0:
                # k corrects the domestic share directly
                k = lc_bench_uk / lc_contract_uk
                applied = True

            # apply k to all deployments: correct domestic LC, rescale international
            for d in mat.columns:
                if d == dep_col and np.isfinite(lc_bench_uk):
                    # UK: enforce benchmark domestic, rescale international
                    off_sum = mat.loc[mat.index != dep_row, d].sum()
                    if off_sum > 0:
                        mat.loc[mat.index != dep_row, d] *= (1.0 - lc_bench_uk) / off_sum
                    if dep_row in mat.index:
                        mat.loc[dep_row, d] = lc_bench_uk
                    continue

                # Other deployments: correct domestic by k_c
                if d in mat.index:
                    lc_contract_d = float(mat.loc[d, d])
                    lc_new = lc_contract_d * k
                    # clip to [0, 1]
                    lc_new = min(max(lc_new, 0.0), 1.0)
                else:
                    lc_new = np.nan

                # rescale international block to sum to (1 - lc_new)
                off_sum = mat.loc[mat.index != d, d].sum()
                if np.isfinite(lc_new):
                    if off_sum > 0:
                        mat.loc[mat.index != d, d] *= (1.0 - lc_new) / off_sum
                    else:
                        # no international block to rescale
                        mat.loc[mat.index != d, d] = 0.0
                    if d in mat.index:
                        mat.loc[d, d] = lc_new
                else:
                    # fallback: renormalize if something went wrong
                    col_sum = mat.loc[:, d].sum()
                    if col_sum > 0:
                        mat.loc[:, d] = mat.loc[:, d] / col_sum

            rows.append({
                "Category": cc,
                "Component": ct,
                "Deploy": deploy,
                "LC_contract": lc_contract_uk,
                "LC_benchmark": lc_bench_uk,
                "BenchmarkCategory": mapped,
                "k": float(k),
                "Applied": bool(applied),
            })

    k_df = pd.DataFrame(rows)
    return out, k_df
