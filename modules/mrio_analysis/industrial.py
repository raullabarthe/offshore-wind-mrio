"""Cost-to-industry allocation and industrial-composition figures."""

import os.path as op
import os
import textwrap
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from collections import defaultdict
from modules.config import HELPERS_DIR
from modules.mrio_analysis.mrio import load_classifications, make_final_demand_vector, summarize_deployments_by_origin_mrio
from modules.cost_model.inputs import costs_bvg
from modules.latex_style import (
    apply_house_style,
    save_and_close,
    set_house_style,
    tex_safe,
    tex_bold,
    get_text_color,
)

ind_codes, final_codes, primary_inputs, ind_labels = load_classifications(HELPERS_DIR)

# Explicit industry color map (used by plots)
COLOR_IND = {
    "B": "#6B6B6B",
    "C16": "#6A4F4F",
    "C18": "#3A2E2E",
    "C19": "#000000",
    "C20": "#8801AA",
    "C21": "#DC6BF9",
    "C22": "#E8ACF7",
    "C23": "#DEF3CE",
    "C24": "#BDF39C",
    "C25": "#28CF28DF",
    "C26": "#0D8C46C8",
    "C27": "#06492696",
    "C28": "#002315",
    "C30": "#A9ECE3",
    "C33": "#ACCCD8B5",
    "D35": "#7A1F4B",
    "E37T39": "#9137BA",
    "F": "#A84C78",
    "G45": "#533200",
    "H49": "#023B90",
    "H50": "#1D5BAC",
    "H52": "#1E7BEC",
    "H53": "#6BACFB",
    "K64": "#C3C002",
    "K65": "#848200",
    "K66": "#585500",
    "M69_70": "#CE4949",
    "M71": "#870101",
    "M72": "#330101",
    "M74_75": "#440000",
    "N78": "#B67102",
    "N80T82": "#935800",
    "Others": "#B3ADAD",
}

# Helper function for loading Excel files
def _read_excel_sheet(path: str | Path, sheet_name: str) -> pd.DataFrame:
    try:
        return pd.read_excel(path, sheet_name=sheet_name)
    except ImportError as exc:
        if "openpyxl" not in str(exc):
            raise
        from openpyxl import load_workbook

        wb = load_workbook(path, read_only=True, data_only=True)
        ws = wb[sheet_name]
        rows = ws.iter_rows(values_only=True)
        headers = [str(x).strip() if x is not None else "" for x in next(rows)]
        records = []
        for values in rows:
            if any(value is not None for value in values):
                records.append(
                    {headers[i]: values[i] if i < len(values) else None for i in range(len(headers))}
                )
        wb.close()
        return pd.DataFrame(records)


def load_excel(filename: str, sheet_name: str, root: str) -> pd.DataFrame:
    """Load an Excel file from the specified root directory."""
    # Convert Path to string if needed
    root = str(root) if isinstance(root, Path) else root
    path = op.join(root, filename)
    return _read_excel_sheet(path, sheet_name)

# ==============================================================
# 0) BRIDGES
# ==============================================================
C1_BRIDGE_FILENAME = "Supplementary_Material_C1_NACE2_Cost_Mapping.xlsx"


def _find_c1_bridge_path(root: str | Path) -> Path | None:
    root = Path(root)
    candidates = [
        root / C1_BRIDGE_FILENAME,
        root / "Industrial breakdown" / C1_BRIDGE_FILENAME,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def get_ind_root(root) -> str:
    """
    Get the Industrial breakdown directory path.
    Handles both string and Path objects as input.
    """
    # Convert Path to string if needed
    root = str(root) if isinstance(root, Path) else root

    c1_path = _find_c1_bridge_path(root)
    if c1_path is not None:
        return str(c1_path.parent)

    # Try to find Industrial breakdown directory
    ind_root = op.join(root, "Industrial breakdown")

    if op.isdir(ind_root):
        return ind_root

    # If not found in subdirectory, check if root itself is the Industrial breakdown directory
    if op.isdir(root) and "bridge_matrix.xlsx" in os.listdir(root):
        return root

    raise FileNotFoundError(
        f"Could not find 'Industrial breakdown' directory or bridge_matrix.xlsx in: {root}\n"
        f"Expected path: {ind_root}"
    )


def _to_cpa_code(code: object) -> str:
    c = str(code).strip()
    if not c.startswith("CPA_"):
        c = f"CPA_{c}"
    return _normalize_cpa(c)


def _c1_bvg_root(component_group: str) -> str:
    return _bvg_root(_canon_component_group(component_group))


def _is_c1_parent_row(row: pd.Series, component_items: set[str]) -> bool:
    root = _c1_bvg_root(row["component_group"])
    item = str(row["cost_item"]).strip()
    node = costs_bvg.get(root, {})
    if not isinstance(node, dict) or item not in node or not isinstance(node[item], dict):
        return False
    children = {str(k).strip() for k in node[item] if k != "Total"}
    return bool(children & component_items)


def _c1_most_disaggregated(mapping: pd.DataFrame) -> pd.DataFrame:
    df = mapping.copy()
    df["component_group"] = df["component_group"].astype(str).str.strip()
    df["sub_component"] = df["sub_component"].astype(str).str.strip()
    df["cost_item"] = df["cost_item"].astype(str).str.strip()

    pieces = []
    for _, group in df.groupby(["component_group", "sub_component"], dropna=False):
        items = set(group["cost_item"])
        parent_mask = group.apply(lambda row: _is_c1_parent_row(row, items), axis=1)
        pieces.append(group.loc[~parent_mask].copy())

    return pd.concat(pieces, ignore_index=True) if pieces else df.iloc[0:0].copy()


def _load_bridges_from_c1(c1_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = _read_excel_sheet(c1_path, sheet_name="mapping_long").copy()
    required = {
        "Scope",
        "CostCategory",
        "CostType",
        "cost_block",
        "cost_item",
        "Component share (%)",
        "NACE 2-digits",
        "NACE 4-digits",
        "NACE 4-digits description",
    }
    missing = required - set(mapping.columns)
    if missing:
        raise KeyError(f"Missing columns in C1 mapping_long: {sorted(missing)}")

    plant = mapping[mapping["Scope"].astype(str).str.strip().eq("Plant")].copy()
    plant = plant.rename(
        columns={
            "CostCategory": "component_group",
            "CostType": "sub_component",
            "NACE 2-digits": "industry_code",
        }
    )
    plant = _c1_most_disaggregated(plant)
    plant["industry_code"] = plant["industry_code"].map(_to_cpa_code)
    plant["share"] = pd.to_numeric(plant["Component share (%)"], errors="coerce").fillna(0.0) * 100.0
    totals = plant.groupby(["component_group", "sub_component"])["share"].transform("sum")
    plant["share"] = np.where(totals > 0, plant["share"] / totals * 100.0, 0.0)
    bridge_long = (
        plant.groupby(["component_group", "sub_component", "industry_code"], as_index=False)["share"]
        .sum()
    )

    turbine = mapping[mapping["Scope"].astype(str).str.strip().eq("Turbine")].copy()
    turbine = turbine.rename(
        columns={
            "CostCategory": "cost_parent",
            "CostType": "component_group",
            "cost_item": "sub_component",
            "NACE 2-digits": "industry_code",
            "NACE 4-digits": "NACE 4-digit",
            "NACE 4-digits description": "nace_label",
        }
    )
    turbine["industry_code"] = turbine["industry_code"].map(_to_cpa_code)
    for c in ["cost_parent", "component_group", "cost_block", "sub_component", "industry_code"]:
        turbine[c] = turbine[c].astype(str).str.strip()
    bridge_tur = (
        turbine[["cost_parent", "component_group", "cost_block", "sub_component", "industry_code", "NACE 4-digit", "nace_label"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return bridge_tur, bridge_long


def load_bridges(ind_root: str):
    c1_path = _find_c1_bridge_path(ind_root)
    if c1_path is not None:
        return _load_bridges_from_c1(c1_path)

    bridge_tur = load_excel("bridge_matrix.xlsx", "turbine_bridge", ind_root).copy()
    for c in ["component_group","cost_block","sub_component","industry_code"]:
        bridge_tur[c] = bridge_tur[c].astype(str).str.strip()

    bridge_farm = load_excel("bridge_matrix.xlsx", "plant_bridge", ind_root).copy()
    # Detect new BVG-based bridge format (has bvg_level_* + cpa_parent)
    new_cols = {"bvg_level_2","bvg_level_3","cpa_parent"}
    if new_cols.issubset(set(bridge_farm.columns)):
        bridge_long = _build_bridge_from_bvg(bridge_farm)
    else:
        bridge_long = (
            bridge_farm
            .melt(id_vars=["component_group","sub_component"], var_name="industry_code", value_name="share")
            .query("share > 0")
            .copy()
        )
        for c in ["component_group","sub_component","industry_code"]:
            bridge_long[c] = bridge_long[c].astype(str).str.strip()

    return bridge_tur, bridge_long


def _sum_numeric_leaves(d: dict) -> float:
    total = 0.0
    for k, v in d.items():
        if k == "Total":
            continue
        if isinstance(v, dict):
            total += _sum_numeric_leaves(v)
        elif isinstance(v, (int, float, np.number)):
            total += float(v)
    return total


def _normalize_cpa(code: str) -> str:
    if code is None:
        return None
    c = str(code).strip()
    if not c:
        return None

    # Collapse to MRIO-available aggregates if needed
    collapse = {
        "CPA_M69": "CPA_M69_70",
        "CPA_M70": "CPA_M69_70",
        "CPA_M74": "CPA_M74_75",
        "CPA_M75": "CPA_M74_75",
        "CPA_N80": "CPA_N80T82",
        "CPA_N81": "CPA_N80T82",
        "CPA_N82": "CPA_N80T82",
        "CPA_J59": "CPA_J59_60",
        "CPA_J60": "CPA_J59_60",
        "CPA_J62": "CPA_J62_63",
        "CPA_J63": "CPA_J62_63",
        "CPA_C10": "CPA_C10T12",
        "CPA_C11": "CPA_C10T12",
        "CPA_C12": "CPA_C10T12",
        "CPA_C13": "CPA_C13T15",
        "CPA_C14": "CPA_C13T15",
        "CPA_C15": "CPA_C13T15",
        "CPA_C31": "CPA_C31_32",
        "CPA_C32": "CPA_C31_32",
        "CPA_E37": "CPA_E37T39",
        "CPA_E38": "CPA_E37T39",
        "CPA_E39": "CPA_E37T39",
        "CPA_Q87": "CPA_Q87_88",
        "CPA_Q88": "CPA_Q87_88",
        "CPA_R90": "CPA_R90T92",
        "CPA_R91": "CPA_R90T92",
        "CPA_R92": "CPA_R90T92",
    }
    c = collapse.get(c, c)

    if c not in ind_codes:
        raise KeyError(f"CPA code not in MRIO classification: {c}")
    return c


def _bvg_root(component_group: str) -> str:
    cg = str(component_group).strip()
    mapping = {
        "Devex": "Development and project management",
        "Development and project management": "Development and project management",
        "BoP": "BoP",
        "Installation": "Installation and commissioning",
        "Installation and commissioning": "Installation and commissioning",
        "O&M": "Operation, maintenance and service",
        "Operation, maintenance and service": "Operation, maintenance and service",
        "ABEX": "Decommissioning",
        "Decommissioning": "Decommissioning",
    }
    return mapping.get(cg, cg)


def _bvg_value(root: str, lvl2: str | None, lvl3: str | None, *, allow_missing_l3: bool = False) -> float:
    if root not in costs_bvg:
        raise KeyError(f"Unknown BVG root '{root}'.")
    node = costs_bvg[root]

    l2 = (str(lvl2).strip() if lvl2 is not None else "")
    l3 = (str(lvl3).strip() if lvl3 is not None else "")

    if not l2:
        if isinstance(node, dict):
            if "Total" in node:
                return float(node["Total"])
            return float(_sum_numeric_leaves(node))
        return float(node)

    if l2 not in node:
        raise KeyError(f"BVG level_2 '{l2}' not found under '{root}'.")
    child = node[l2]

    if not l3 or l3 == l2:
        if isinstance(child, dict):
            if "Total" in child:
                return float(child["Total"])
            return float(_sum_numeric_leaves(child))
        return float(child)

    if isinstance(child, dict) and l3 in child:
        v = child[l3]
        if isinstance(v, dict):
            if "Total" in v:
                return float(v["Total"])
            return float(_sum_numeric_leaves(v))
        return float(v)

    if allow_missing_l3:
        return 0.0
    raise KeyError(f"BVG level_3 '{l3}' not found under '{root} -> {l2}'.")

def _canon_component_group(cg: str) -> str:
    c = str(cg).strip()
    mapping = {
        "Development and project management": "Devex",
        "Installation and commissioning": "Installation",
        "Operation, maintenance and service": "O&M",
        "Decommissioning": "ABEX",
    }
    return mapping.get(c, c)


def _default_subcomponent(cg: str, lvl2: str) -> str:
    cg = str(cg).strip()
    l2 = str(lvl2).strip()
    if cg == "Devex":
        return l2
    if cg == "BoP":
        if l2 in {"Array cables", "Export cables", "Foundation", "OSS"}:
            return l2
        if l2 == "Onshore substation":
            return "Other"
        if l2 == "Cable accessories":
            return ""  # handled by split
        return "Other"
    if cg == "Installation":
        if l2 == "Foundation installation":
            return "Foundation"
        if l2 == "Turbine installation":
            return "Turbine"
        if l2 == "Offshore cable installation":
            return ""  # handled by split
        if l2 == "Onshore export cable installation":
            return "Export cables"
        if l2 in {"Offshore substation installation", "Onshore substation construction"}:
            return "OSS"
        if l2 in {"Inbound transport", "Offshore logistics", "Contingency and insurance"}:
            return "Other"
        return "Other"
    if cg == "O&M":
        if l2 == "Operations":
            return "Operations"
        if l2 == "Maintenance & service":
            return "Maintenance & service"
        if l2 == "Operations, maintenance and service port":
            return "Operations"
        return l2
    return l2


def _build_bridge_from_bvg(bridge_farm: pd.DataFrame) -> pd.DataFrame:
    df = bridge_farm.copy()
    for c in ["component_group","sub_component","bvg_level_2","bvg_level_3","cpa_parent"]:
        df[c] = df[c].astype(str).str.strip()
        df.loc[df[c].isin(["nan", "None", ""]), c] = ""

    # Canonical component_group names (align with model)
    df["component_group"] = df["component_group"].map(_canon_component_group)

    # Fill default sub_component when missing or equals bvg_level_2
    sc_missing = df["sub_component"].eq("") | (df["sub_component"] == df["bvg_level_2"])
    df.loc[sc_missing, "sub_component"] = df.loc[sc_missing].apply(
        lambda r: _default_subcomponent(r["component_group"], r["bvg_level_2"]),
        axis=1,
    )

    # Force specific BVG level_2 -> model sub_component mapping (override if needed)
    mask_inst = df["component_group"].eq("Installation")
    df.loc[mask_inst & df["bvg_level_2"].eq("Foundation installation"), "sub_component"] = "Foundation"
    df.loc[mask_inst & df["bvg_level_2"].eq("Turbine installation"), "sub_component"] = "Turbine"
    df.loc[mask_inst & df["bvg_level_2"].eq("Onshore export cable installation"), "sub_component"] = "Export cables"
    df.loc[mask_inst & df["bvg_level_2"].eq("Offshore substation installation"), "sub_component"] = "OSS"
    df.loc[
        mask_inst
        & df["bvg_level_2"].eq("Onshore substation construction")
        & df["sub_component"].eq("")
    , "sub_component"] = "OSS"

    # Expand special splits if sub_component not explicitly set
    extra_rows = []
    for _, r in df.iterrows():
        cg = r["component_group"]
        sc = r["sub_component"]
        l2 = r["bvg_level_2"]
        if cg == "BoP" and l2 == "Cable accessories" and sc not in {"Array cables", "Export cables"}:
            for target in ["Array cables", "Export cables"]:
                rr = r.copy()
                rr["sub_component"] = target
                extra_rows.append(rr)
            continue
        if cg == "Installation" and l2 == "Offshore cable installation" and sc not in {"Array cables", "Export cables"}:
            for target in ["Array cables", "Export cables"]:
                rr = r.copy()
                rr["sub_component"] = target
                extra_rows.append(rr)
            continue
        extra_rows.append(r)

    df = pd.DataFrame(extra_rows)

    # Drop total-like rows when more detailed rows exist for same (cg, sc, level_2)
    df["_is_detail"] = df["bvg_level_3"].ne("") & df["bvg_level_3"].ne(df["bvg_level_2"])
    has_detail = df.groupby(["component_group","sub_component","bvg_level_2"])["_is_detail"].transform("any")
    is_total_like = df["bvg_level_3"].eq("") | (df["bvg_level_3"] == df["bvg_level_2"])
    df = df[~(has_detail & is_total_like)].copy()
    df = df.drop(columns=["_is_detail"], errors="ignore")

    # Base shares for cable splits (match build_cost_breakdown logic)
    export_base = float(costs_bvg["BoP"].get("Export cables", 0.0))
    array_base  = float(costs_bvg["BoP"].get("Array cables", 0.0))
    denom = (export_base + array_base) or 1.0
    export_share = export_base / denom
    array_share  = array_base  / denom

    cable_acc = float(costs_bvg["BoP"].get("Cable accessories", 0.0))
    offshore_cable_install = float(costs_bvg["Installation and commissioning"].get("Offshore cable installation", 0.0))

    values = []
    missing = []
    for _, r in df.iterrows():
        cg = str(r["component_group"]).strip()
        sc = str(r["sub_component"]).strip()
        root = _bvg_root(cg)
        lvl2 = r["bvg_level_2"]
        lvl3 = r["bvg_level_3"]

        # Special split: Cable accessories -> Export/Array by BoP shares
        if root == "BoP" and lvl2 == "Cable accessories":
            if sc == "Export cables":
                val = cable_acc * export_share
            elif sc == "Array cables":
                val = cable_acc * array_share
            else:
                val = cable_acc
        # Special split: Offshore cable installation -> Export/Array by BoP shares
        elif root == "Installation and commissioning" and lvl2 == "Offshore cable installation":
            if sc == "Export cables":
                val = offshore_cable_install * export_share
            elif sc == "Array cables":
                val = offshore_cable_install * array_share
            else:
                val = offshore_cable_install
        else:
            # Check missing level_3 explicitly
            missing_l3 = False
            if lvl3 and lvl3 != lvl2:
                child = costs_bvg.get(root, {}).get(lvl2, {})
                if isinstance(child, dict) and (lvl3 not in child):
                    missing_l3 = True
            if missing_l3:
                missing.append((root, lvl2, lvl3))
            val = _bvg_value(root, lvl2, lvl3, allow_missing_l3=True)

        values.append(float(val))

    df["bvg_value"] = values

    if missing:
        uniq = sorted(set(missing))
        print("[bridge] Missing BVG level_3 entries (treated as 0 in share weights):")
        for root, l2, l3 in uniq:
            print(f"  - {root} -> {l2} -> {l3}")
    df["industry_code"] = df["cpa_parent"].map(_normalize_cpa)

    # Aggregate to (component_group, sub_component, industry_code)
    agg = (
        df.groupby(["component_group","sub_component","industry_code"], as_index=False)["bvg_value"]
        .sum()
    )

    # Compute shares within each sub_component
    totals = agg.groupby(["component_group","sub_component"])["bvg_value"].transform("sum")
    agg["share"] = np.where(totals > 0, agg["bvg_value"] / totals * 100.0, 0.0)

    # sanity check
    check = agg.groupby(["component_group","sub_component"])["share"].sum()
    bad = check[(check - 100.0).abs() > 1e-6]
    if len(bad):
        print("[bridge] Warning: shares not summing to 100% for:")
        for (cg, sc), v in bad.items():
            print(f"  - {cg} / {sc}: {v:.4f}%")

    return agg[["component_group","sub_component","industry_code","share"]].copy()


def flows_to_cost_df(flows_hier_all: dict) -> pd.DataFrame:
    rec = []
    for dep, parents in flows_hier_all.items():
        for cg, children in parents.items():
            for sc, origins in children.items():
                for origin, val in origins.items():
                    rec.append({
                        "deployment_country": dep,
                        "component_group": cg,
                        "sub_component": sc,
                        "origin_country": origin,
                        "value": float(val),
                    })
    df = pd.DataFrame(rec)
    return df[df["value"].abs() > 1e-9].copy()

def allocate_farm_costs(cost_df: pd.DataFrame, bridge_long: pd.DataFrame) -> pd.DataFrame:
    df = cost_df.query("component_group != 'Turbine'").copy()
    df["component_group"] = df["component_group"].astype(str).str.strip()
    df["sub_component"]   = df["sub_component"].astype(str).str.strip()
    df = df.merge(bridge_long, how="left", on=["component_group","sub_component"])
    miss = df["share"].isna()
    if miss.any():
        bad = df.loc[miss, ["component_group","sub_component"]].drop_duplicates()
        raise KeyError(f"Unmapped (component_group, sub_component) in plant_bridge:\n{bad.to_string(index=False)}")

    df["value_alloc"] = df["value"] * df["share"] / 100.0
    df = df[["deployment_country","component_group","sub_component","origin_country","industry_code","value_alloc"]]
    df["industry_code"] = df["industry_code"].astype("string").str.strip().fillna("UNMAPPED")
    df["cost_block"] = "__NA__"
    return df


def allocate_turbine_costs(
    df_alloc: pd.DataFrame,
    bridge_tur: pd.DataFrame,
    *,
    keep_oem: bool = False,
) -> pd.DataFrame:
    df = df_alloc.drop(columns=["industry_code"], errors="ignore").copy()

    if "cost_block" not in df.columns:
        raise KeyError("df_alloc must contain a 'cost_block' column for turbine allocation.")

    df["component_group"] = df["component_group"].astype(str).str.strip()
    df["sub_component"]   = df["sub_component"].astype(str).str.strip()
    df["cost_block"]      = df["cost_block"].astype(str).str.strip()

    bt = bridge_tur.copy()
    for c in ["component_group","cost_block","sub_component","industry_code"]:
        bt[c] = bt[c].astype(str).str.strip()

    # Pattern A: (cg='Turbine', cost_block=<turbine block>, sub_component=<part>)
    a = df.copy()
    a["_cg"], a["_cb"], a["_sc"] = "Turbine", a["component_group"], a["sub_component"]
    bta = bt.rename(columns={"component_group":"_cg","cost_block":"_cb","sub_component":"_sc"})[["_cg","_cb","_sc","industry_code"]]
    a = a.merge(bta, how="left", on=["_cg","_cb","_sc"])

    # Pattern B: (cg=<turbine block>, cost_block=<Assembly/BoM/...>, sub_component=<part>)
    b = df.copy()
    b["_cg"], b["_cb"], b["_sc"] = b["component_group"], b["cost_block"], b["sub_component"]
    btb = bt.rename(columns={"component_group":"_cg","cost_block":"_cb","sub_component":"_sc"})[["_cg","_cb","_sc","industry_code"]]
    b = b.merge(btb, how="left", on=["_cg","_cb","_sc"])

    ok_a = a["industry_code"].notna().mean() if len(a) else 0.0
    ok_b = b["industry_code"].notna().mean() if len(b) else 0.0
    dfm = a if ok_a >= ok_b else b

    out_cols = [
        "deployment_country",
        "component_group",
        "sub_component",
        "cost_block",
        "origin_country",
        "industry_code",
        "value_alloc",
    ]
    if keep_oem and "OEM" in dfm.columns:
        out_cols.insert(1, "OEM")
    out = dfm[out_cols].copy()

    out["industry_code"] = out["industry_code"].astype("string").str.strip().fillna("UNMAPPED")

    group_cols = [
        "deployment_country", "component_group", "sub_component", "cost_block",
        "origin_country", "industry_code",
    ]
    if keep_oem and "OEM" in out.columns:
        group_cols.insert(1, "OEM")
    out = out.groupby(
        group_cols,
        as_index=False
    )["value_alloc"].sum()

    return out


def build_final_cost_df(cost_df_farm: pd.DataFrame, df_turbine: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([cost_df_farm, df_turbine], ignore_index=True)
    df["cost_block"] = df["cost_block"].astype("string").fillna("__NA__")

    df = df.groupby(
        ["deployment_country","component_group","sub_component","cost_block","origin_country","industry_code"],
        as_index=False
    )["value_alloc"].sum()

    df["deployment_country"] = df["deployment_country"].astype(str).str.strip().replace({"UK":"GB"})
    df["origin_country"]     = df["origin_country"].astype(str).str.strip().replace({"UK":"GB","RoW":"ROW"})
    df["industry_code"]      = df["industry_code"].astype("string").str.strip().fillna("UNMAPPED")
    return df


# ==============================================================
# 2) THREE LEVELS
#   L0: max disaggregation
#   L1: OPEX combined via OPEX_L0_TO_L1; turbine blocks -> Turbine
#   L2: collapse to component group (OPEX -> OPEX)
# ==============================================================
# OPEX category mapping
OPEX_L0_TO_L1 = {
    "Operations": "Operations",
    "Offshore logistics": "Operations",
    "Turbine maintenance and service": "Maintenance & service",
    "BoP maintenance and service": "Maintenance & service",
}


TURBINE_BLOCKS = {"Blades","Hub","Nacelle","Tower","Drivetrain"}

def add_3_levels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["component_group_L0"] = df["component_group"].astype(str).str.strip()
    df["sub_component_L0"]   = df["sub_component"].astype(str).str.strip()

    df["component_group_L1"] = df["component_group_L0"]
    df["sub_component_L1"]   = df["sub_component_L0"]

    m = df["component_group_L0"].isin(TURBINE_BLOCKS)
    df.loc[m, "component_group_L1"] = "Turbine"
    df.loc[m, "sub_component_L1"]   = df.loc[m, "component_group_L0"]

    m = df["component_group_L0"].eq("O&M")
    df.loc[m, "sub_component_L1"] = df.loc[m, "sub_component_L0"].map(OPEX_L0_TO_L1).fillna(df.loc[m, "sub_component_L0"])

    df["component_group_L2"] = df["component_group_L1"]
    df["sub_component_L2"]   = df["component_group_L2"]
    return df

# ==============================================================
# 3) BRIDGE OVERRIDES
# ==============================================================
def apply_bridge_overrides(bridge_long: pd.DataFrame, overrides: list[dict],
                          group_cols=("component_group","sub_component"),
                          code_col="industry_code", share_col="share") -> pd.DataFrame:
    out = bridge_long.copy()
    for c in [group_cols[0], group_cols[1], code_col]:
        out[c] = out[c].astype(str).str.strip()

    for ov in overrides:
        cg, sc = str(ov[group_cols[0]]).strip(), str(ov[group_cols[1]]).strip()
        out = out.loc[~((out[group_cols[0]] == cg) & (out[group_cols[1]] == sc))].copy()

    rows = []
    for ov in overrides:
        cg, sc = str(ov[group_cols[0]]).strip(), str(ov[group_cols[1]]).strip()
        for code, sh in ov["shares"].items():
            rows.append({group_cols[0]:cg, group_cols[1]:sc, code_col:str(code).strip(), share_col:float(sh)})

    if rows:
        out = pd.concat([out, pd.DataFrame(rows)], ignore_index=True)

    return out.query(f"{share_col} > 0").copy()

def extract_component_metadata(df: pd.DataFrame, level: int = 1, include_L0: bool = True):
    cg_col, sc_col = f"component_group_L{level}", f"sub_component_L{level}"
    cg = df[cg_col].astype(str)
    sc = df[sc_col].astype(str)

    component_groups = ["Total"] + sorted(cg.unique())
    component_groups = list(dict.fromkeys(component_groups))

    subcomponents_dict = (
        pd.DataFrame({"cg": cg, "sc": sc})
        .drop_duplicates()
        .groupby("cg")["sc"]
        .apply(lambda x: sorted(x.unique()))
        .to_dict()
    )

    if include_L0:
        cg0 = df["component_group_L0"].astype(str)
        sc0 = df["sub_component_L0"].astype(str)
        subcomponents_L0 = (
            pd.DataFrame({"cg0": cg0, "sc0": sc0})
            .drop_duplicates()
            .groupby("cg0")["sc0"]
            .apply(lambda x: sorted(x.unique()))
            .to_dict()
        )
        subcomponents_dict["__L0__"] = subcomponents_L0

    return component_groups, subcomponents_dict

# ==============================================================
# 4) INDUSTRIAL PIPELINE
# ==============================================================
# Source: The regional economic impacts of offshore wind energy developments in Scotland
# https://linkinghub.elsevier.com/retrieve/pii/S0960148120309757


def run_industrial(
    ROOT: str,
    flows_by_component_hier_all: dict,
    df_alloc: pd.DataFrame,
    bridge_overrides: list[dict] | None = None,
    level: int = 1,
    ind_ind: bool = False,
):
    bridge_tur, bridge_long = load_bridges(get_ind_root(ROOT))
    if ind_ind:
        def _strip_cpa(x):
            s = str(x).strip()
            if s.startswith("CPA_"):
                return s[4:]
            return s
        for df in (bridge_tur, bridge_long):
            if "industry_code" in df.columns:
                df["industry_code"] = df["industry_code"].map(_strip_cpa)
    if bridge_overrides:
        bridge_long = apply_bridge_overrides(bridge_long, bridge_overrides)

    cost_df = flows_to_cost_df(flows_by_component_hier_all)
    farm = allocate_farm_costs(cost_df, bridge_long)
    turb = allocate_turbine_costs(df_alloc, bridge_tur)
    final = build_final_cost_df(farm, turb)
    final = add_3_levels(final)

    component_groups, subcomponents_dict = extract_component_metadata(final, level, include_L0=True)
    return final, component_groups, subcomponents_dict


# ==============================================================
# 6) BRIDGE SENSITIVITY
# ==============================================================

# ==============================================================
# 7) INDUSTRY-GROUP MRIO RESULTS
# ==============================================================

def results_by_origin_industry_groups(
    final_cost_df: pd.DataFrame,
    industry_groups: dict,
    *,
    core,
    mrio,
    year: int = 2023,
    L1_MRIO=None,
    VA_MRIO=None,
    LB_MRIO=None,
    W_MRIO=None,
    level: int = 1,
    include_total: bool = True,
):
    """
    Build a results_by_origin dict keyed by industry-group name.

    Each value matches summarize_deployments_by_origin_mrio output:
    one row per (Deployment, Origin) with Output/GVA/Income/Jobs cols.
    """
    if core is None:
        raise ValueError("Pass core=... deployment list.")
    if mrio is None:
        raise ValueError("Pass mrio=... (MRIO table).")
    if L1_MRIO is None or VA_MRIO is None or LB_MRIO is None:
        raise ValueError("Pass L1_MRIO, VA_MRIO, LB_MRIO (and W_MRIO if needed).")

    df = final_cost_df.copy()
    df["industry_code"] = df["industry_code"].astype(str).str.strip()

    def _expand_codes(codes):
        codes = {str(c).strip() for c in codes}
        return codes | {f"CPA_{c}" for c in codes}

    out = {}
    for group_name, codes in industry_groups.items():
        code_set = _expand_codes(codes)
        df_g = df[df["industry_code"].isin(code_set)].copy()

        y_by_dep = {
            dep: make_final_demand_vector(
                mrio, df_g, dep,
                component_group=None,
                sub_component=None,
                level=level,
                mode="annual",
            )
            for dep in core
        }

        out[group_name] = summarize_deployments_by_origin_mrio(
            y_by_dep, year, L1_MRIO, VA_MRIO, LB_MRIO, W_MRIO
        )

    if include_total:
        y_all = {
            dep: make_final_demand_vector(
                mrio, df, dep,
                component_group=None,
                sub_component=None,
                level=level,
                mode="annual",
            )
            for dep in core
        }
        out["Total"] = summarize_deployments_by_origin_mrio(
            y_all, year, L1_MRIO, VA_MRIO, LB_MRIO, W_MRIO
        )

    return out

OTHER_LABEL="Other"
def _trunc(s,n=120): s=str(s); return s if len(s)<=n else s[:n-1]+"…"
def _hex2rgb(h): h=h.lstrip("#"); return tuple(int(h[i:i+2],16)/255 for i in (0,2,4))
def _blend(a,b,t): return tuple((1-t)*a[i]+t*b[i] for i in range(3))
def _tail(code):
    s=str(code)
    return s.split("_",1)[1] if s.startswith("CPA_") and "_" in s else s


def _short_component_label(s: str) -> str:
    s = str(s)
    mapping = {
        "Development & consenting services": "Dev. & consenting",
        "Engineering & consultancy": "Eng. & consult.",
        "Project management": "Proj. mgmt.",
        "Maintenance & service": "Maint./serv.",
        "Array cables": "Array cab.",
        "Export cables": "Exp. cab.",
    }
    return mapping.get(s, s)

def _label_for_code(code: str, labels: dict) -> str:
    s = str(code)
    if s in labels:
        return labels[s]
    if not s.startswith("CPA_") and f"CPA_{s}" in labels:
        return labels[f"CPA_{s}"]
    if s.startswith("CPA_") and s[4:] in labels:
        return labels[s[4:]]
    return s


def _sobrio_palette_for_codes(codes: list[str]):
    def _classify_tail(t: str):
        t = str(t)
        # Red family: M*, plus K* and P*
        if t.startswith("M") or t.startswith("K") or t.startswith("P"):
            if t == "M71":
                order = (0, t)
            elif t.startswith("M"):
                order = (1, t)
            elif t.startswith("K"):
                order = (2, t)
            else:
                order = (3, t)
            return "RED", order
        # Green family: selected C codes (C28 darkest)
        if t in {"C23", "C25", "C26", "C27", "C28"}:
            order_map = {"C28": 0, "C27": 1, "C26": 2, "C25": 3, "C23": 4}
            return "GREEN", (order_map.get(t, 99), t)
        # Orange family: C33, D35, F* (F darkest)
        if t == "C33" or t == "D35" or t.startswith("F"):
            if t.startswith("F"):
                order = (0, t)
            elif t == "D35":
                order = (1, t)
            else:
                order = (2, t)
            return "ORANGE", order
        # Blue family: H49, H50, H52 (H50 darkest)
        if t in {"H49", "H50", "H52"}:
            order_map = {"H50": 0, "H49": 1, "H52": 2}
            return "BLUE", (order_map.get(t, 99), t)
        return "OTHER", (99, t)

    def _ramp(dark, light, n):
        if n <= 1:
            return [_hex2rgb(dark)]
        dark_rgb = _hex2rgb(dark)
        light_rgb = _hex2rgb(light)
        return [_blend(dark_rgb, light_rgb, i / (n - 1)) for i in range(n)]

    palette_defs = {
        "RED": ("#5A1F1F", "#E1B9B9"),
        "GREEN": ("#1F4D3B", "#CFE6D8"),
        "ORANGE": ("#7A4A1B", "#E9D1B0"),
        "BLUE": ("#1F3C5B", "#B7C7E3"),
        "OTHER": ("#4C4C4C", "#D9D9D9"),
    }

    groups = defaultdict(list)
    for c in codes:
        t = _tail(c)
        fam, key = _classify_tail(t)
        groups[fam].append((c, key))

    out = {}
    for fam, items in groups.items():
        items_sorted = [c for c, _ in sorted(items, key=lambda x: x[1])]
        dark, light = palette_defs[fam]
        ramp = _ramp(dark, light, len(items_sorted))
        for c, col in zip(items_sorted, ramp):
            out[c] = col
    return out

def plot_componentgroup_industry(
    df, ind_labels,
    deploy_country=None,
    group_col="component_group_L1",
    industry_col="industry_code",
    value_col="value_alloc",
    group_order=("Devex","BoP","Turbine","Installation","O&M"),
    include_total=True,
    max_industries=12, legend_cap=140, legend_wrap: int | None = None,
    fig_w=7.8, fig_h=4.8,
    legend_space=0.72, legend_x=1.18, legend_y=0.5, legend_loc="center left",
    other_style=("white","black",0.4),
    pct_font=7, xtick_font=9,
    legend_font=6, legend_title_font=6,
    total_label="Total",
    show_total_label=False,
    color_ind: dict | None = None,
    **_ignore,
):
    set_house_style()
    face_other,edge_other,lw_other = other_style

    d=df.copy()
    if deploy_country is not None:
        d=d.loc[d["deployment_country"].astype(str)==str(deploy_country)].copy()
    if d.empty: raise ValueError("No rows after filtering")

    groups_present=list(d[group_col].dropna().astype(str).unique())
    ordered_groups=[g for g in group_order if g in groups_present] or groups_present

    ind_tot=(d.groupby(industry_col,dropna=False)[value_col].sum().sort_values(ascending=False))
    ind_tot.index=ind_tot.index.astype(str)

    base=list(ind_tot.index) if (max_industries is None or len(ind_tot)<=max_industries) else list(ind_tot.index[:max_industries])+[OTHER_LABEL]
    use_other=(OTHER_LABEL in base)
    _bucket=lambda x: str(x) if ((not use_other) or str(x) in base) else OTHER_LABEL

    codes=[x for x in base if x!=OTHER_LABEL]
    if color_ind is None:
        color_ind = COLOR_IND
    code_color = _sobrio_palette_for_codes(codes)
    if color_ind:
        for k, v in color_ind.items():
            k = str(k)
            code_color[k] = v
            code_color[f"CPA_{k}"] = v

    def _color(c):
        if c==OTHER_LABEL: return face_other
        return code_color.get(c,(0.35,0.35,0.35))

    rows=[]
    for g in ordered_groups:
        dd=d[d[group_col].astype(str)==str(g)]
        bv=defaultdict(float)
        for ind,v in dd.groupby(industry_col)[value_col].sum().items(): bv[_bucket(ind)] += float(v)
        tot=sum(bv.values()) or 1.0
        rows.append((g,{o:(bv.get(o,0.0)/tot) for o in base}))

    if include_total:
        totv=defaultdict(float)
        for ind,v in d.groupby(industry_col)[value_col].sum().items(): totv[_bucket(ind)] += float(v)
        gtot=sum(totv.values()) or 1.0
        rows.append(("Total",{o:(totv.get(o,0.0)/gtot) for o in base}))

    if include_total:
        rows = [r for r in rows if r[0] == "Total"] + [r for r in rows if r[0] != "Total"]

    labels=[g for g,_ in rows]
    rowdicts=[rv for _,rv in rows]
    n=len(labels)

    fig,ax=plt.subplots(figsize=(fig_w,fig_h),dpi=150)

    for i,(lbl,rv) in enumerate(zip(labels,rowdicts)):
        bottom=0.0
        order=[o for o,_ in sorted(((k,v) for k,v in rv.items() if k!=OTHER_LABEL), key=lambda kv:kv[1], reverse=True)]
        if use_other: order.append(OTHER_LABEL)
        for o in order:
            h=float(rv.get(o,0.0))
            if h<=0: continue
            col=_color(o)
            ax.bar(i,h,bottom=bottom,color=col,
                   edgecolor="white" if o!=OTHER_LABEL else edge_other,
                   linewidth=0.5 if o!=OTHER_LABEL else lw_other)
            if h>=0.04:
                if h < 0.999:
                    ax.text(i,bottom+h/2,tex_safe(f"{h*100:.0f}%"),
                            ha="center",va="center",fontsize=pct_font+2,color=get_text_color(col))
            bottom+=h

    ax.set_ylim(0,1)
    ax.set_yticks([0,.25,.5,.75,1])
    ax.set_yticklabels([tex_safe(x) for x in ["0%","25%","50%","75%","100%"]])
    ax.set_xticks(np.arange(n))
    xt = [("" if (not show_total_label and str(x) == str(total_label)) else x) for x in labels]
    ax.set_xticklabels([tex_safe(x) for x in xt], fontsize=xtick_font)
    ax.set_ylabel(tex_bold("Share of value by industry"))
    ax.set_xlabel(tex_bold("Component group"))

    def _lab_raw(o):
        if o == OTHER_LABEL:
            return OTHER_LABEL
        return f"{o} â€” {_label_for_code(o, ind_labels)}"
    legend_codes = sorted([x for x in base if x != OTHER_LABEL], key=lambda o: _lab_raw(o).casefold())
    if use_other:
        legend_codes.append(OTHER_LABEL)
    def _lab(o):
        if o == OTHER_LABEL:
            lab = OTHER_LABEL
        else:
            lab = f"{o} — {_trunc(_label_for_code(o, ind_labels),legend_cap)}"
        if legend_wrap and legend_wrap > 0:
            lab = "\n".join(textwrap.wrap(lab, width=legend_wrap))
        return lab
    patches=[ mpatches.Patch(facecolor=_color(o), edgecolor=edge_other if o==OTHER_LABEL else "white",
                             linewidth=lw_other if o==OTHER_LABEL else 0.5, label=tex_safe(_lab(o)))
              for o in legend_codes ]

    ax.legend(handles=patches, title=tex_bold("Industry"),
              bbox_to_anchor=(legend_x, legend_y), loc=legend_loc,
              fontsize=legend_font, title_fontsize=legend_title_font,
              frameon=True, framealpha=0.9, facecolor="white", edgecolor="#CCCCCC")

    apply_house_style(ax,grid=True); ax.grid(axis="y",alpha=0.2,linestyle=":",linewidth=0.5)
    fig.subplots_adjust(right=1-legend_space); fig.tight_layout()
    return fig,ax

def plot_componentgroup_subcomponent_industry(
    df, ind_labels,
    deploy_country=None,
    group_col="component_group_L1",
    subcomponent_col="sub_component_L1",
    industry_col="industry_code",
    value_col="value_alloc",
    group_order=("Devex","Turbine","BoP","Installation","O&M"),
    max_industries=12, legend_cap=140, legend_wrap: int | None = None,
    fig_w=12.2, fig_h=5.0, dpi=160,
    legend_space=0.30, legend_x=1.02, legend_y=0.5, legend_loc="center left",
    other_style=("white","black",0.4),
    pct_font=8, xtick_font=10, group_font=14,
    legend_font=8, legend_title_font=8,
    label_min_share=0.04,
    include_total=True,
    total_label="Total",
    show_total_label=False,
    short_labels=True,
    group_label_offset=-0.35,
    color_ind: dict | None = None,
):
    """
    Plot industry shares by subcomponent with hierarchical x-axis
    (component group blocks + subcomponent labels), similar to
    plot_capture_sublevel_avg_connected_power.
    """
    set_house_style()
    face_other, edge_other, lw_other = other_style

    d = df.copy()
    if deploy_country is not None:
        d = d.loc[d["deployment_country"].astype(str) == str(deploy_country)].copy()
    if d.empty:
        raise ValueError("No rows after filtering")

    groups_present = list(d[group_col].dropna().astype(str).unique())
    ordered_groups = [g for g in group_order if g in groups_present] or groups_present

    ind_tot = d.groupby(industry_col, dropna=False)[value_col].sum().sort_values(ascending=False)
    ind_tot.index = ind_tot.index.astype(str)

    base = (
        list(ind_tot.index)
        if (max_industries is None or len(ind_tot) <= max_industries)
        else list(ind_tot.index[:max_industries]) + [OTHER_LABEL]
    )
    use_other = (OTHER_LABEL in base)
    _bucket = lambda x: str(x) if ((not use_other) or str(x) in base) else OTHER_LABEL

    codes = [x for x in base if x != OTHER_LABEL]
    if color_ind is None:
        color_ind = COLOR_IND
    code_color = _sobrio_palette_for_codes(codes)
    if color_ind:
        for k, v in color_ind.items():
            k = str(k)
            code_color[k] = v
            code_color[f"CPA_{k}"] = v

    def _color(c):
        if c == OTHER_LABEL: return face_other
        return code_color.get(c, (0.35, 0.35, 0.35))

    rows = []
    for g in ordered_groups:
        sub_present = sorted(d.loc[d[group_col].astype(str) == str(g), subcomponent_col].dropna().astype(str).unique())
        for sc in sub_present:
            dd = d[(d[group_col].astype(str) == str(g)) & (d[subcomponent_col].astype(str) == str(sc))]
            bv = defaultdict(float)
            for ind, v in dd.groupby(industry_col)[value_col].sum().items():
                bv[_bucket(ind)] += float(v)
            tot = sum(bv.values()) or 1.0
            rows.append((g, sc, {o: (bv.get(o, 0.0) / tot) for o in base}))

    if include_total:
        bv = defaultdict(float)
        for ind, v in d.groupby(industry_col)[value_col].sum().items():
            bv[_bucket(ind)] += float(v)
        tot = sum(bv.values()) or 1.0
        rows = [(total_label, total_label, {o: (bv.get(o, 0.0) / tot) for o in base})] + rows
        ordered_groups = [total_label] + ordered_groups

    # build x positions with group gaps (hierarchical like capture plot)
    x, xt, centers, seps, k = [], [], [], [], 0
    plot_rows = []
    for gi, g in enumerate(ordered_groups):
        sub_rows = [r for r in rows if r[0] == g]
        if not sub_rows:
            continue
        start = k
        for _, sc, rv in sub_rows:
            x.append(k)
            xt.append(sc)
            plot_rows.append(rv)
            k += 1
        end = k - 1
        centers.append((g, (start + end) / 2))
        if gi < len(ordered_groups) - 1:
            seps.append(k)
        k += 1

    if not plot_rows:
        raise ValueError("No rows to plot after processing.")

    vals = np.array([[rv.get(o, 0.0) for o in base] for rv in plot_rows], dtype=float)
    xs = x[:vals.shape[0]]

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    idx_map = {o: j for j, o in enumerate(base)}

    for i in range(vals.shape[0]):
        bottom = 0.0
        # order within each subcomponent by descending share
        order = sorted(
            [o for o in base if o != OTHER_LABEL],
            key=lambda o: float(vals[i, idx_map[o]]),
            reverse=True,
        )
        if use_other:
            order.append(OTHER_LABEL)

        for o in order:
            j = idx_map[o]
            w = float(vals[i, j])
            if w <= 0:
                continue
            col = _color(o)
            ax.bar(
                xs[i], w, bottom=bottom, width=0.85,
                color=col,
                edgecolor="white" if o != OTHER_LABEL else edge_other,
                linewidth=0.5 if o != OTHER_LABEL else lw_other,
            )
            if w >= label_min_share and w < 0.999:
                ax.text(
                    xs[i], bottom + w / 2,
                    tex_safe(f"{w*100:.0f}%"),
                    ha="center", va="center", fontsize=pct_font,
                    color=get_text_color(col),
                )
            bottom += w

    for s in seps:
        ax.axvline(s, color="black", linewidth=1.0, linestyle="--", alpha=0.6)

    ax.set_ylim(0, 1)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_yticklabels([tex_safe(x) for x in ["0%","25%","50%","75%","100%"]])

    ax.set_xticks(xs)
    xt_disp = [_short_component_label(t) for t in xt[:vals.shape[0]]] if short_labels else xt[:vals.shape[0]]
    ax.set_xticklabels([tex_safe(t) for t in xt_disp], fontsize=xtick_font, rotation=45, ha="right")
    ax.tick_params(axis="x", length=0)

    for comp, c in centers:
        if (not show_total_label) and str(comp) == str(total_label):
            continue
        ax.text(c, group_label_offset, tex_safe(str(comp)), transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=group_font, fontweight="bold")

    ax.set_ylabel(tex_bold("Industry share [%]"))

    def _lab_raw(o):
        if o == OTHER_LABEL:
            return OTHER_LABEL
        return f"{o} â€” {_label_for_code(o, ind_labels)}"
    legend_codes = sorted([x for x in base if x != OTHER_LABEL], key=lambda o: _lab_raw(o).casefold())
    if use_other:
        legend_codes.append(OTHER_LABEL)
    def _lab(o):
        if o == OTHER_LABEL:
            lab = OTHER_LABEL
        else:
            lab = f"{o} — {_trunc(_label_for_code(o, ind_labels), legend_cap)}"
        if legend_wrap and legend_wrap > 0:
            lab = "\n".join(textwrap.wrap(lab, width=legend_wrap))
        return lab
    patches = [
        mpatches.Patch(facecolor=_color(o), edgecolor=edge_other if o == OTHER_LABEL else "white",
                       linewidth=lw_other if o == OTHER_LABEL else 0.5, label=tex_safe(_lab(o)))
        for o in legend_codes
    ]
    ax.legend(handles=patches, title=tex_bold("Industry"),
              bbox_to_anchor=(legend_x, legend_y), loc=legend_loc,
              fontsize=legend_font, title_fontsize=legend_title_font,
              frameon=True, framealpha=0.9, facecolor="white", edgecolor="#CCCCCC")

    apply_house_style(ax, grid=True)
    ax.grid(axis="y", alpha=0.2, linestyle=":", linewidth=0.5)
    ax.set_title(None)
    fig.subplots_adjust(bottom=0.32, right=1-legend_space)
    fig.tight_layout()
    return fig, ax

def plot_and_save_componentgroup_industry(
    deploy_country,
    df,
    save,
    close,
    filename=None,
    *,
    fig_name: str | None = None,
    fig_title: str | None = None,
    ind_ind: bool | None = None,
    ind_labels_override: dict | None = None,
    color_ind: dict | None = None,
    **kw,
):
    if filename is not None and fig_name is not None:
        raise ValueError("Use either filename or fig_name, not both.")

    output_filename = filename or fig_name or "4.7_EU_industry_componentgroup.pdf"
    valid_plot_suffixes = {".pdf", ".png", ".pgf"}
    if fig_name is not None and Path(str(output_filename)).suffix.lower() not in valid_plot_suffixes:
        output_filename = f"{output_filename}.pdf"

    use_sub = kw.pop("use_subcomponents", None)
    if use_sub is None:
        use_sub = ("sub_component_L1" in df.columns)

    labels = ind_labels_override
    if labels is None and ind_ind is not None:
        _, _, _, labels = load_classifications(HELPERS_DIR, ind_ind, set_globals=False)
    if labels is None:
        labels = ind_labels

    if use_sub:
        fig,_=plot_componentgroup_subcomponent_industry(
            df, labels, deploy_country=deploy_country, color_ind=color_ind, **kw
        )
    else:
        fig,_=plot_componentgroup_industry(
            df, labels, deploy_country=deploy_country, color_ind=color_ind, **kw
        )
    if fig_title:
        fig.suptitle(fig_title, y=1.02)
    return save_and_close(fig=fig,filename=output_filename, save=save, close=close)


def plot_eu27_composition_groups(
    deploy_country,
    final_cost_df_agg: pd.DataFrame,
    save,
    close,
    filename=None,
    *,
    fig_name: str | None = None,
    fig_title: str | None = None,
    ind_ind: bool | None = None,
    ind_labels_override: dict | None = None,
    color_ind: dict | None = None,
    **kw,
):
    """
    Backwards-compatible alias: delegates to plot_and_save_componentgroup_industry.
    """
    defaults = {
        "max_industries": 20,
        "fig_w": 13,
        "fig_h": 5,
        "legend_cap": 35,
        "pct_font": 8,
        "legend_wrap": 50,
        "legend_font": 10,
        "legend_title_font": 12,
        "legend_x": 0.99,
        "legend_y": 0.5,
        "group_label_offset": -0.22,
    }
    defaults.update(kw)
    return plot_and_save_componentgroup_industry(
        deploy_country,
        final_cost_df_agg,
        save,
        close,
        filename=filename,
        fig_name=fig_name,
        fig_title=fig_title,
        ind_ind=ind_ind,
        ind_labels_override=ind_labels_override,
        color_ind=color_ind,
        **defaults,
    )



def _lighten(color, amount=0.35):
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)
