"""Cost-model figures used by the paper run."""

from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modules.latex_style import (
    apply_house_style,
    save_and_close,
    set_house_style,
    tex_safe,
    adjust_lightness,
    get_text_color,
)
from .inputs import costs_bvg

# ============================================================================
# COLOR PALETTES
# ============================================================================

PALETTE = {
    "primary": "#2F3EEA",
    "secondary": "#6B7FD7",
    "highlight": "#990000",
    "dark": "#4A4A4A",
    "light": "#E5E5E5",
}

PALETTE_CARDO = {
    "primary": "#1F3A5F",
    "secondary": "#4A6FA5",
    "highlight": "#7A2E2E",
    "dark": "#2C2C2C",
    "light": "#DADADA",
}

PARENTS_TABLEAU = {
    "Devex": "#4E79A7",
    "Turbine": "#F28E2B",
    "BoP": "#59A14F",
    "Installation": "#E15759",
    "O&M": "#9A31AC",
    "ABEX": "#D6D5D5",
}

PALETTE_MUTED = [
    "#332288",
    "#88CCEE",
    "#44AA99",
    "#117733",
    "#999933",
    "#DDCC77",
    "#CC6677",
    "#882255",
    "#AA4499",
    "#BBBBBB",
]

# ============================================================================
# MAIN CHARTS (Turbine and Wind Farm)
# ============================================================================

def plot_turbine_double_ring(
    turbine_supply_cost: dict,
    *,
    filename: str = "A.3.pdf",
    min_label_pct: float = 7.0,
    inner_fontsize: int = 8,
    outer_fontsize: int = 12,
    inner_group_fontsize: int | None = 14,
    legend_fontsize: float = 8,
    save: bool = True,
    close: bool = True,
):
    set_house_style()
    if inner_group_fontsize is None:
        inner_group_fontsize = outer_fontsize

    tur = turbine_supply_cost
    supply = tur["Supply_components"]

    # -------------------------
    # 1) Parent totals
    # -------------------------
    parent_labels, parent_values = [], []
    for k, v in supply.items():
        if isinstance(v, (int, float)):
            parent_labels.append(k)
            parent_values.append(v)

    parent_values = np.array(parent_values, dtype=float)
    parent_labels_tex = [tex_safe(s) for s in parent_labels]

    # -------------------------
    # 2) Children by parent
    # -------------------------
    children_by_parent = {"Blades": {"Blades": supply["Blades"]}}

    children_by_parent["Hub"] = {
        name: sub["BoM"] + sub["Direct"]["direct_production"] + sub["Indirect"]["overhead"]
        for name, sub in tur["Hub"].items()
    }
    children_by_parent["Nacelle"] = {
        name: sub["BoM"] + sub["Direct"]["direct_production"] + sub["Indirect"]["overhead"]
        for name, sub in tur["Nacelle"].items()
    }
    children_by_parent["Tower"] = {
        name: sub["BoM"] + sub["Direct"]["direct_production"] + sub["Indirect"]["overhead"]
        for name, sub in tur["Tower"].items()
    }

    drv_children, other_metals_total = {}, 0.0
    for name, sub in tur["Drivetrain"].items():
        value = sub["BoM"] + sub["Direct"]["direct_production"] + sub["Indirect"]["overhead"]
        if name in ["Generator - Copper", "Generator - Iron (arms)", "Generator - Iron (elect)"]:
            other_metals_total += value
        else:
            drv_children[name] = value
    if other_metals_total > 0:
        drv_children["Generator - Other metals"] = other_metals_total
    children_by_parent["Drivetrain"] = drv_children

    # -------------------------
    # 3) Parent colors (fixed per component; same as current output)
    # -------------------------
    turbine_component_colors = {
        "Blades": "#1F3A5F",     # PALETTE_CARDO primary
        "Hub": "#2C2C2C",        # PALETTE_CARDO dark
        "Nacelle": "#2E6F40",    # pine green
        "Tower": "#BFBFBF",      # darker light gray for contrast
        "Drivetrain": "#7A2E2E", # PALETTE_CARDO highlight
    }

    tab20 = plt.get_cmap("tab20")
    seed_colors = [
        PALETTE_CARDO["primary"],
        PALETTE_CARDO["secondary"],
        PALETTE_CARDO["highlight"],
        PALETTE_CARDO["dark"],
        PALETTE_CARDO["light"],
    ]
    base_colors_by_label = {
        lbl: (seed_colors[i] if i < len(seed_colors) else tab20(i % 20))
        for i, lbl in enumerate(parent_labels)
    }

    parent_colors = {
        lbl: turbine_component_colors.get(lbl, base_colors_by_label.get(lbl, PALETTE_CARDO["secondary"]))
        for lbl in parent_labels
    }

    # -------------------------
    # 4) Flatten children
    # -------------------------
    child_labels, child_values, child_colors = [], [], []
    child_rel_to_parent_pct = []

    def _order_children(parent_label: str, children: dict) -> list[tuple[str, float]]:
        items = list(children.items())

        def is_gen(name: str) -> bool:
            return parent_label == "Drivetrain" and "Generator" in name

        def is_yaw(name: str) -> bool:
            return parent_label == "Nacelle" and "Yaw" in name

        gen = [(k, v) for k, v in items if is_gen(k)]
        yaw = [(k, v) for k, v in items if is_yaw(k)]
        rest = [(k, v) for k, v in items if not is_gen(k) and not is_yaw(k)]

        groups: list[tuple[str, list[tuple[str, float]]]] = []
        if len(gen) >= 2:
            groups.append(("__GEN__", gen))
        else:
            rest += gen
        if len(yaw) >= 2:
            groups.append(("__YAW__", yaw))
        else:
            rest += yaw

        for k, v in rest:
            groups.append((k, [(k, v)]))

        def group_sum(g):
            return sum(v for _, v in g[1])

        ordered: list[tuple[str, float]] = []
        for _, group in sorted(groups, key=group_sum, reverse=True):
            ordered.extend(sorted(group, key=lambda kv: kv[1], reverse=True))
        return ordered

    for p_label, p_value in zip(parent_labels, parent_values):
        raw_children = children_by_parent[p_label]
        ordered = _order_children(p_label, raw_children)
        names = [k for k, _ in ordered]
        raw_vals = np.array([v for _, v in ordered], dtype=float)

        rel_fracs = raw_vals / raw_vals.sum() if raw_vals.sum() > 0 else np.zeros_like(raw_vals)
        scaled_vals = raw_vals / raw_vals.sum() * p_value if raw_vals.sum() > 0 else raw_vals

        base_color = parent_colors[p_label]

        for idx, (cname, cval, rel) in enumerate(zip(names, scaled_vals, rel_fracs)):
            child_labels.append(tex_safe(f"{p_label} – {cname}"))
            child_values.append(cval)
            child_rel_to_parent_pct.append(float(rel) * 100.0)
            factor = 1.0 if len(names) == 1 else (0.6 + 0.8 * idx / (len(names) - 1))
            child_colors.append(adjust_lightness(base_color, factor))

    child_values = np.array(child_values, dtype=float)

    # -------------------------
    # 5) Label helpers
    # -------------------------
    def _fmt_pct(pct: float) -> str:
        return f"{pct:.0f}\\%" if pct >= 1 else f"{pct:.1f}\\%"

    def _make_autopct(min_pct: float):
        def _autopct(pct):
            if pct < min_pct:
                return ""
            return _fmt_pct(pct)
        return _autopct

    def _make_autopct_from_list(pcts: list[float], min_pct: float):
        it = iter(pcts)
        def _autopct(_pct_from_total):
            try:
                pct = next(it)
            except StopIteration:
                return ""
            if pct < min_pct:
                return ""
            return _fmt_pct(pct)
        return _autopct

    # -------------------------
    # 6) Plot
    # -------------------------
    fig, ax = plt.subplots(figsize=(9, 9))
    start_angle = 90

    wedges_inner, _, autotexts_inner = ax.pie(
        child_values,
        labels=None,
        autopct=_make_autopct_from_list(child_rel_to_parent_pct, min_label_pct),
        radius=1.0,
        colors=child_colors,
        startangle=start_angle,
        wedgeprops=dict(width=0.3, edgecolor="white"),
        pctdistance=0.82,
        textprops=dict(fontsize=inner_fontsize, fontweight="bold"),
    )

    for w, t in zip(wedges_inner, autotexts_inner):
        if t.get_text():
            t.set_color(get_text_color(w.get_facecolor()))

    wedges_outer, _, autotexts_outer = ax.pie(
        parent_values,
        labels=None,
        autopct=_make_autopct(min_label_pct),
        radius=1.3,
        colors=[parent_colors[l] for l in parent_labels],
        startangle=start_angle,
        wedgeprops=dict(width=0.3, edgecolor="white"),
        pctdistance=0.9,
        textprops=dict(fontsize=outer_fontsize, fontweight="bold"),
    )

    for w, t in zip(wedges_outer, autotexts_outer):
        if t.get_text():
            t.set_color(get_text_color(w.get_facecolor()))

    r_label = 0.45
    DEVEX_OFFSET = 0.18
    for w, p, lbl in zip(wedges_outer, parent_labels, parent_labels_tex):
        ang = np.deg2rad((w.theta1 + w.theta2) / 2.0)
        r = r_label + (DEVEX_OFFSET if p == "Devex" else 0.0)
        xx, yy = r * np.cos(ang), r * np.sin(ang)
        ax.text(xx, yy, lbl, ha="center", va="center", color="black", fontweight="bold",
                fontsize=inner_group_fontsize)

    ax.set(aspect="equal")

    legend = ax.legend(
        wedges_inner,
        child_labels,
        title=tex_safe("Subcomponents"),
        loc="center left",
        bbox_to_anchor=(1.15, 0.5),
        frameon=False,
        fontsize=legend_fontsize,
    )
    legend.get_title().set_fontweight("bold")

    apply_house_style(ax, grid=False)

    outpath = save_and_close(fig=fig, filename=filename, save=save, close=close, legacy_tight=True)
    return outpath

def plot_model_double_donut(
    cost_break,
    *,
    filename: str = "A.2.pdf",
    min_label_pct: float = 5.0,
    inner_fontsize: int = 8,
    outer_fontsize: int = 12,
    inner_group_fontsize: int | None = 14,
    legend_fontsize: float = 9.2,
    save: bool = True,
    close: bool = True,
):
    """
    Plot double donut: inner (detailed costs) + outer (aggregated costs).

    Parameters
    ----------
    chart_df : pd.DataFrame
        Cost breakdown with columns: Parent, Cost, Model
    filename : str
        Output filename
    """
    set_house_style()
    if inner_group_fontsize is None:
        inner_group_fontsize = outer_fontsize
    chart_df = build_chart_df_from_model(cost_break)
    df = chart_df.copy()
    df = df[["Parent", "Cost", "Model"]].copy()
    df["Model"] = np.asarray(df["Model"], dtype=float)
    df = df[df["Model"] > 0].copy()
    # Exclude ABEX from this chart
    df = df[df["Parent"] != "ABEX"].copy()

    parent_order = ["Devex", "Turbine", "BoP", "Installation", "O&M"]
    parent_order = [p for p in parent_order if p in df["Parent"].unique()]

    parent_totals = (
        df.groupby("Parent", as_index=False)["Model"]
        .sum()
        .set_index("Parent")["Model"]
        .reindex(parent_order)
        .fillna(0.0)
    )
    parent_labels = parent_totals.index.tolist()
    parent_values = parent_totals.values.astype(float)
    parent_labels_tex = [tex_safe(s) for s in parent_labels]

    children_by_parent = {
        p: dict(zip(sub["Cost"], sub["Model"]))
        for p in parent_labels
        for sub in [df[df["Parent"] == p]]
    }

    # Explicit, more sober tones (same hues as original palette)
    parent_colors = {
        "Devex": "#3B5C7A",        # muted blue
        "Turbine": "#C97A2B",      # muted orange
        "BoP": "#4C7A54",          # muted green
        "Installation": "#B24A4A", # muted red
        "O&M": "#7A4E8A",          # muted purple
        "ABEX": "#CFCFCF",         # light gray
    }
    parent_colors = {p: parent_colors.get(p, "#999999") for p in parent_labels}

    child_labels, child_values, child_colors = [], [], []
    child_rel_to_parent_pct = []

    def _order_children(parent_label: str, children: dict) -> list[tuple[str, float]]:
        items = list(children.items())

        def is_gen(name: str) -> bool:
            return parent_label == "Drivetrain" and "Generator" in name

        def is_yaw(name: str) -> bool:
            return parent_label == "Nacelle" and "Yaw" in name

        gen = [(k, v) for k, v in items if is_gen(k)]
        yaw = [(k, v) for k, v in items if is_yaw(k)]
        rest = [(k, v) for k, v in items if not is_gen(k) and not is_yaw(k)]

        groups: list[tuple[str, list[tuple[str, float]]]] = []
        if len(gen) >= 2:
            groups.append(("__GEN__", gen))
        else:
            rest += gen
        if len(yaw) >= 2:
            groups.append(("__YAW__", yaw))
        else:
            rest += yaw

        for k, v in rest:
            groups.append((k, [(k, v)]))

        def group_sum(g):
            return sum(v for _, v in g[1])

        ordered: list[tuple[str, float]] = []
        for _, group in sorted(groups, key=group_sum, reverse=True):
            ordered.extend(sorted(group, key=lambda kv: kv[1], reverse=True))
        return ordered

    for p_label, p_value in zip(parent_labels, parent_values):
        raw_children = children_by_parent.get(p_label, {})
        ordered = _order_children(p_label, raw_children)
        names = [k for k, _ in ordered]
        raw_vals = np.array([v for _, v in ordered], dtype=float)

        denom = raw_vals.sum()
        rel_fracs = raw_vals / denom if denom > 0 else np.zeros_like(raw_vals)
        scaled_vals = (raw_vals / denom * p_value) if denom > 0 else raw_vals

        base_color = parent_colors[p_label]
        for idx, (cname, cval, rel) in enumerate(zip(names, scaled_vals, rel_fracs)):
            child_labels.append(tex_safe(f"{p_label} – {cname}"))
            child_values.append(cval)
            child_rel_to_parent_pct.append(float(rel) * 100.0)
            factor = 1.0 if len(names) == 1 else (0.6 + 0.8 * idx / (len(names) - 1))
            child_colors.append(adjust_lightness(base_color, factor))

    child_values = np.array(child_values, dtype=float)

    def _fmt_pct(pct: float) -> str:
        return f"{pct:.0f}\\%" if pct >= 1 else f"{pct:.1f}\\%"

    def _make_autopct(min_pct: float):
        def _autopct(pct):
            if pct < min_pct:
                return ""
            return _fmt_pct(pct)
        return _autopct

    def _make_autopct_from_list(pcts: list[float], min_pct: float):
        it = iter(pcts)
        def _autopct(_pct_from_total):
            try:
                pct = next(it)
            except StopIteration:
                return ""
            if pct < min_pct:
                return ""
            return _fmt_pct(pct)
        return _autopct

    fig, ax = plt.subplots(figsize=(9, 9))
    start_angle = 90

    wedges_inner, _, autotexts_inner = ax.pie(
        child_values,
        labels=None,
        autopct=_make_autopct_from_list(child_rel_to_parent_pct, min_label_pct),
        radius=1.0,
        colors=child_colors,
        startangle=start_angle,
        wedgeprops=dict(width=0.3, edgecolor="white"),
        pctdistance=0.82,
        textprops=dict(fontsize=inner_fontsize, fontweight="bold"),
    )

    for w, t in zip(wedges_inner, autotexts_inner):
        if t.get_text():
            t.set_color(get_text_color(w.get_facecolor()))

    wedges_outer, _, autotexts_outer = ax.pie(
        parent_values,
        labels=None,
        autopct=_make_autopct(min_label_pct),
        radius=1.3,
        colors=[parent_colors[p] for p in parent_labels],
        startangle=start_angle,
        wedgeprops=dict(width=0.3, edgecolor="white"),
        pctdistance=0.9,
        textprops=dict(fontsize=outer_fontsize, fontweight="bold"),
    )

    for w, t in zip(wedges_outer, autotexts_outer):
        if t.get_text():
            t.set_color(get_text_color(w.get_facecolor()))

    r_label = 0.45
    DEVEX_OFFSET = 0.12
    for w, p, lbl in zip(wedges_outer, parent_labels, parent_labels_tex):
        ang = np.deg2rad((w.theta1 + w.theta2) / 2.0)
        r = r_label + (DEVEX_OFFSET if p == "Devex" else 0.0)
        xx, yy = r * np.cos(ang), r * np.sin(ang)
        ax.text(xx, yy, lbl, ha="center", va="center", color="black", fontweight="bold",
                fontsize=inner_group_fontsize)

    ax.set(aspect="equal")

    legend = ax.legend(
        wedges_inner,
        child_labels,
        title=tex_safe("Components"),
        loc="center left",
        bbox_to_anchor=(1.15, 0.5),
        frameon=False,
        fontsize=legend_fontsize,
    )
    legend.get_title().set_fontweight("bold")

    apply_house_style(ax, grid=False)

    outpath = save_and_close(fig=fig, filename=filename, save=save, close=close, legacy_tight=True)
    return outpath

# ============================================================================
# BENCHMARK CHARTS (BVG vs Model vs NREL)
# ============================================================================



def canon_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.strip()
         .str.replace(r"\s+", " ", regex=True)
    )


def canon_df(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = canon_series(out[c])
    return out


def build_bvg_rows_from_costs_bvg(costs_bvg: dict, opex_years: int = 20) -> list[dict]:
    # -----------------------------
    # DEVEX -> single bucket (sum of its sub-buckets)
    # -----------------------------
    dev = costs_bvg["Development and project management"]
    dev_total = (
        float(dev["Development & consenting services"]["Total"])
        + float(dev["Engineering & consultancy"])
        + float(dev["Project management"])
    )

    # -----------------------------
    # TURBINE -> Nacelle, Rotor, Tower
    # -----------------------------
    wt = costs_bvg["Turbine"]
    turbine_nacelle = float(wt["Nacelle"])
    turbine_rotor   = float(wt["Rotor"])
    turbine_tower   = float(wt["Tower"])

    # -----------------------------
    # BALANCE OF PLANT -> Foundations, Export, Array, OSS, Other (residual)
    # -----------------------------
    bop = costs_bvg["BoP"]
    bop_foundations = float(bop["Foundation"]["Total"])
    bop_export      = float(bop["Export cables"])
    bop_array       = float(bop["Array cables"])

    oss_block = bop.get("Offshore substation", bop.get("OSS"))
    bop_oss = float(oss_block["Total"]) if isinstance(oss_block, dict) and "Total" in oss_block else 0.0

    bop_total = float(bop["Total"])
    bop_other = bop_total - (bop_foundations + bop_export + bop_array + bop_oss)

    # -----------------------------
    # INSTALLATION -> Export cables, Array cables, Foundation, Turbine, OSS, Other
    # -----------------------------
    inst = costs_bvg["Installation and commissioning"]
    inst_total = float(inst["Total"])

    denom = (bop_export + bop_array) or 1.0
    export_share = bop_export / denom
    array_share  = bop_array  / denom

    offshore_cable_install = float(inst.get("Offshore cable installation", 0.0))
    onshore_export_install = float(inst.get("Onshore export cable installation", 0.0))

    export_offshore = offshore_cable_install * export_share
    array_offshore  = offshore_cable_install * array_share

    inst_export_cables = export_offshore + onshore_export_install
    inst_array_cables  = array_offshore

    inst_foundation = float(inst["Foundation installation"]["Total"])
    inst_turbine    = float(inst["Turbine installation"]["Total"])

    inst_oss = float(inst["Offshore substation installation"])

    inst_other = inst_total - (
        inst_export_cables + inst_array_cables + inst_foundation + inst_turbine + inst_oss
    )

    # -----------------------------
    # OPEX -> split operations & maintenance
    # -----------------------------
    om = costs_bvg["Operation, maintenance and service"]
    operations = om["Operations"]
    maintenance = om["Maintenance & service"]

    ops_port = float(om.get("Operations, maintenance and service port", 0.0))
    ops_annual = ops_port + sum(float(v) for k, v in operations.items() if k != "Total")
    maint_annual = sum(float(v) for k, v in maintenance.items() if k != "Total")

    ops_life = ops_annual * opex_years
    maint_life = maint_annual * opex_years

    # -----------------------------
    # ABEX -> Decommissioning (single bucket)
    # -----------------------------
    abex_total = float(costs_bvg["Decommissioning"]["Total"])

    # -----------------------------
    # TOTAL lifetime cost (denominator)
    # -----------------------------
    total_lifetime = (
        dev_total
        + (turbine_nacelle + turbine_rotor + turbine_tower)
        + (bop_foundations + bop_export + bop_array + bop_oss + bop_other)
        + (inst_export_cables + inst_array_cables + inst_foundation + inst_turbine + inst_oss + inst_other)
        + (ops_life + maint_life)
        + abex_total
    )

    def pct(x):
        return 100.0 * float(x) / float(total_lifetime) if total_lifetime else 0.0

    bvg_rows = [
        {"Parent": "Devex",            "Cost": "Devex",                  "BVG": pct(dev_total)},
        {"Parent": "Turbine",          "Cost": "Nacelle",                "BVG": pct(turbine_nacelle)},
        {"Parent": "Turbine",          "Cost": "Rotor",                  "BVG": pct(turbine_rotor)},
        {"Parent": "Turbine",          "Cost": "Tower",                  "BVG": pct(turbine_tower)},
        {"Parent": "BoP", "Cost": "Foundation",            "BVG": pct(bop_foundations)},
        {"Parent": "BoP", "Cost": "Export cables",          "BVG": pct(bop_export)},
        {"Parent": "BoP", "Cost": "Array cables",           "BVG": pct(bop_array)},
        {"Parent": "BoP", "Cost": "OSS",                    "BVG": pct(bop_oss)},
        {"Parent": "BoP", "Cost": "Other",       "BVG": pct(bop_other)},
        {"Parent": "Installation",     "Cost": "Export cables",          "BVG": pct(inst_export_cables)},
        {"Parent": "Installation",     "Cost": "Array cables",           "BVG": pct(inst_array_cables)},
        {"Parent": "Installation",     "Cost": "Foundation",             "BVG": pct(inst_foundation)},
        {"Parent": "Installation",     "Cost": "Turbine",                "BVG": pct(inst_turbine)},
        {"Parent": "Installation",     "Cost": "OSS",                    "BVG": pct(inst_oss)},
        {"Parent": "Installation",     "Cost": "Other",                  "BVG": pct(inst_other)},
        {"Parent": "O&M",              "Cost": "Operations",             "BVG": pct(ops_life)},
        {"Parent": "O&M",              "Cost": "Maintenance & service",  "BVG": pct(maint_life)},
        {"Parent": "ABEX",             "Cost": "Decommissioning",        "BVG": pct(abex_total)},
    ]

    df = pd.DataFrame(bvg_rows)
    print("Sum BVG shares (%):", df["BVG"].sum())
    print("Installation residual Other (GBP/MW):", inst_other)
    print("BoP residual Other BoP (GBP/MW):", bop_other)

    return bvg_rows




def build_bvg_template_df(bvg_rows: list[dict]) -> pd.DataFrame:
    bvg_df = canon_df(pd.DataFrame(bvg_rows), ["Parent", "Cost"])
    template = bvg_df[["Parent", "Cost"]].drop_duplicates().copy()
    return bvg_df, template


def build_model_agg_from_snap(
    snap: dict,
    template: pd.DataFrame,
    *,
    build_table_func: Callable[[dict], pd.DataFrame],
) -> pd.DataFrame:
    df_full = build_table_func(snap).copy()
    df_full = canon_df(df_full, ["Group", "Item", "Subgroup1"])

    turb = df_full[df_full["Group"] == "Turbine"].copy()

    def map_turb_cost(s1: str):
        if s1 in ("Blades", "Hub"):
            return "Rotor"
        if s1 in ("Drivetrain", "Nacelle"):
            return "Nacelle"
        if s1 == "Tower":
            return "Tower"
        return np.nan

    turb["Cost"] = turb["Subgroup1"].map(map_turb_cost)
    turb = turb.dropna(subset=["Cost"])

    turb_agg = (
        turb.groupby("Cost", as_index=False)["Percent_of_Total_%"]
            .sum()
            .rename(columns={"Percent_of_Total_%": "Model"})
    )
    turb_agg["Parent"] = "Turbine"

    rest = df_full[df_full["Group"] != "Turbine"].copy()

    parent_map = {
        "Devex": "Devex",
        "BoP": "BoP",
        "Installation": "Installation",
        "OPEX_lifetime": "O&M",
        "ABEX": "ABEX",
    }
    rest["Parent"] = rest["Group"].replace(parent_map)

    name_map_rest = {
        "Foundation": "Foundation",
        "Other": "Other",
        "OSS": "OSS",
        "Offshore Substation": "OSS",
        "Offshore substation": "OSS",

        "Export cables": "Export cables",
        "Array cables": "Array cables",
        "Foundation": "Foundation",
        "Turbine": "Turbine",
        "OSS": "OSS",
        "Other": "Other",

        "Operations control centre": "Operations",
        "Training": "Operations",
        "Onshore logistics": "Operations",
        "Technical resource (onshore and off)": "Operations",
        "Admin and support staff (onshore)": "Operations",
        "Offshore logistics": "Operations",
        "Insurance": "Operations",
        "Operations, maintenance and service port": "Operations",

        "Turbine maintenance and service": "Maintenance & service",
        "BoP maintenance and service": "Maintenance & service",
        "Statutory inspections": "Maintenance & service",
    }

    rest.loc[rest["Parent"] == "Devex", "Item"] = "Devex"
    rest.loc[rest["Parent"] == "ABEX", "Item"] = "Decommissioning"

    rest = rest.rename(columns={"Item": "Cost"})
    rest["Cost"] = rest["Cost"].replace(name_map_rest)

    rest_agg = (
        rest.dropna(subset=["Cost"])
            .groupby(["Parent", "Cost"], as_index=False)["Percent_of_Total_%"]
            .sum()
            .rename(columns={"Percent_of_Total_%": "Model"})
    )

    model_agg = pd.concat(
        [turb_agg[["Parent", "Cost", "Model"]],
         rest_agg[["Parent", "Cost", "Model"]]],
        ignore_index=True,
    )
    model_agg = canon_df(model_agg, ["Parent", "Cost"])

    out = template.merge(model_agg, on=["Parent", "Cost"], how="left")
    out["Model"] = out["Model"].fillna(0.0)
    return out


def build_chart_df_from_snap(
    snap: dict,
    bvg_rows: list[dict],
    *,
    build_table_func: Callable[[dict], pd.DataFrame],
) -> pd.DataFrame:
    bvg_df, template = build_bvg_template_df(bvg_rows)

    model_aligned = build_model_agg_from_snap(snap, template, build_table_func=build_table_func)

    chart_df = template.merge(bvg_df[["Parent", "Cost", "BVG"]], on=["Parent", "Cost"], how="left")
    chart_df = chart_df.merge(model_aligned[["Parent", "Cost", "Model"]], on=["Parent", "Cost"], how="left")

    chart_df["BVG"] = chart_df["BVG"].fillna(0.0)
    chart_df["Model"] = chart_df["Model"].fillna(0.0)
    return chart_df


def turbine_parts_first_level(cost_breakdown: dict) -> dict:
    turb = cost_breakdown.get("Turbine", {}) or {}

    exclude = {"Supply_components", "Grand_total"}
    parts = [k for k in turb.keys() if k not in exclude]

    out = {}
    for part in parts:
        part_node = turb.get(part, {}) or {}

        part_total = 0.0
        for sub_name, sub_node in part_node.items():
            if isinstance(sub_node, dict):
                part_total += sum_bom_direct_indirect(sub_node)

        out[part] = part_total

    return out


def build_chart_df_from_model(cost_breakdown: dict) -> pd.DataFrame:
    rows = []

    if "Devex" in cost_breakdown:
        dev = cost_breakdown["Devex"]
        if isinstance(dev, dict):
            dev_total = float(dev.get("Total", dev.get("total", 0.0)))
        else:
            dev_total = float(dev)
        if dev_total > 0:
            rows.append({"Parent": "Devex", "Cost": "Devex", "Model": dev_total})

    turb_parts = turbine_parts_first_level(cost_breakdown)
    for part, val in turb_parts.items():
        if val > 0:
            rows.append({"Parent": "Turbine", "Cost": part, "Model": val})

    for parent in ["BoP", "Installation", "ABEX"]:
        block = cost_breakdown.get(parent, {}) or {}
        if isinstance(block, dict):
            for k, v in block.items():
                if k == "Grand_total":
                    continue
                if isinstance(v, (int, float)) and float(v) > 0:
                    rows.append({"Parent": parent, "Cost": k, "Model": float(v)})

    om = cost_breakdown.get("O&M", {}) or {}
    if isinstance(om, dict):
        annual = om.get("lifetime_breakdown", {}) or {}
        for k, v in annual.items():
            if isinstance(v, (int, float)) and float(v) > 0:
                rows.append({"Parent": "O&M", "Cost": k, "Model": float(v)})

    return pd.DataFrame(rows)


def sum_bom_direct_indirect(node: dict) -> float:
    bom = float(node.get("BoM", 0.0))

    direct = node.get("Direct", {}) or {}
    direct_sum = sum(float(v) for k, v in direct.items() if k != "direct_production")

    indirect = node.get("Indirect", {}) or {}
    indirect_sum = sum(float(v) for k, v in indirect.items() if k != "overhead")

    return bom + direct_sum + indirect_sum


