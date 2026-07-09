"""Sourcing reallocation scenarios, resilience accounting, and scenario figures."""

from modules.mrio_analysis.mrio import build_y_shock_levels, summarize_deployments_by_origin_mrio
from modules.mrio_analysis.industrial import run_industrial
from modules.config import HELPERS_DIR, CORE_DEPLOYMENT

from typing import Dict, Tuple
from dataclasses import dataclass
import builtins
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from modules.country import COLOR_MAP, NAME_TO_CODE, CODE_TO_NAME, EUROPE_CODES, EU27
from modules.latex_style import (
    apply_house_style,
    save_and_close,
    set_house_style,
    tex_safe,
    tex_bold,
    get_text_color,
)


def _save_and_close_exact_aware(
    *,
    fig,
    filename,
    save,
    close,
    exact_size=False,
):
    if not exact_size:
        return save_and_close(fig=fig, filename=filename, save=save, close=close)

    from pathlib import Path
    from modules.config import CLOSE_FIGURES, IMG_ROOT, SAVE_FIGURES

    fig = fig or plt.gcf()
    save = SAVE_FIGURES if save is None else save
    close = CLOSE_FIGURES if close is None else close
    outpath = None

    if save:
        outpath = Path(IMG_ROOT) / filename
        try:
            fig.set_layout_engine("none")
        except Exception:
            pass
        fig.savefig(outpath)
        print(f"Saved to: {outpath}")

    if close:
        plt.close(fig)

    return outpath


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ScenarioDelta:
    # Container for scenario results with cost modifications.
    delta_pp: float
    final_cost_df_cf: "pd.DataFrame"
    stats_turbine: dict
    stats_farm: dict

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

LEVELS = (0, 1, 2)  # Default aggregation levels


SUBCOMPONENT_LABEL_ABBREVIATIONS = {
    "Development & consenting services": "Dev. & consenting",
    "Engineering & consultancy": "Eng. & consult.",
    "Project management": "Proj. mgmt.",
    "Maintenance & service": "Maint./serv.",
    "Array cables": "Array cab.",
    "Export cables": "Exp. cab.",
    "Generator - Assembly": "Gen. asm.",
    "Generator - Copper": "Gen. Cu",
    "Generator - Iron (arms)": "Gen. Fe arms",
    "Generator - Iron (elect)": "Gen. Fe elect.",
    "Generator - Magnets": "Gen. magnets",
    "Yaw - Bearing": "Yaw bearing",
    "Yaw - Drive & brake": "Yaw drive/brake",
    "Yaw - Gear type": "Yaw gear",
    "Coupling & brake": "Coupling/brake",
    "Pitch bearings": "Pitch bear.",
    "Main bearings": "Main bear.",
}


def _short_subcomponent_label(label) -> str:
    text = str(label)
    return SUBCOMPONENT_LABEL_ABBREVIATIONS.get(text, text)


def run_results_by_origin_from_final_cost_df(
    final_cost_df: "pd.DataFrame",
    *,
    L1_MRIO, VA_MRIO, LB_MRIO, W_MRIO,
    core=None,
    mrio=None,
    annualize: bool = True,
    lifetime_years: int = 25,
):
    def _normalize_origin_country_to_mrio(df: "pd.DataFrame"):
        if mrio is None or "origin_country" not in df.columns:
            return df
        idx = mrio.index
        if isinstance(idx, pd.MultiIndex):
            if "origin_region" in idx.names:
                lvl = "origin_region"
            elif "origin" in idx.names:
                lvl = "origin"
            elif "region" in idx.names:
                lvl = "region"
            else:
                lvl = 0
            mrio_origin_set = set(idx.get_level_values(lvl).astype(str).unique())
        else:
            mrio_origin_set = set(idx.astype(str).unique())

        out = df.copy()
        out["origin_country"] = (
            out["origin_country"]
              .astype(str).str.strip()
              .replace({"UK": "GB", "RoW": "ROW"})
              .where(lambda s: s.isin(mrio_origin_set), "ROW")
        )
        return out

    if core is None:
        core = CORE_DEPLOYMENT
    if mrio is None:
        raise ValueError("Pass mrio=... to run_results_by_origin_from_final_cost_df().")

    df_use = final_cost_df
    if not bool(annualize):
        df_use = final_cost_df.copy()
        if "value_alloc" in df_use.columns:
            df_use["value_alloc"] = df_use["value_alloc"].astype(float) * float(lifetime_years)
    df_use = _normalize_origin_country_to_mrio(df_use)
    y_shock_lvl, y_shock_sub_lvl = build_y_shock_levels(df_use, core, mrio)

    results_by_origin_lvl = {
        lvl: {cg: summarize_deployments_by_origin_mrio(y, 2023, L1_MRIO, VA_MRIO, LB_MRIO, W_MRIO)
              for cg, y in y_shock_lvl[lvl].items()}
        for lvl in LEVELS
    }
    results_by_origin_sub_lvl = {
        lvl: {k: summarize_deployments_by_origin_mrio(y, 2023, L1_MRIO, VA_MRIO, LB_MRIO, W_MRIO)
              for k, y in y_shock_sub_lvl[lvl].items()}
        for lvl in LEVELS
    }
    return results_by_origin_lvl, results_by_origin_sub_lvl



def run_mrio_for_scenarios(
    scenarios: Dict[int, "ScenarioDelta"],
    *,
    L1_MRIO,
    VA_MRIO,
    LB_MRIO,
    W_MRIO,
    core=None,
    mrio=None,
    annualize: bool = True,
    lifetime_years: int = 25,
) -> Dict[int, dict]:
    """
    Runs the MRIO model for a set of scenarios.
    Uses global LEVELS variable.
    """
    out: Dict[int, dict] = {}

    for d_pp, scn in scenarios.items():
        r_lvl, r_sub = run_results_by_origin_from_final_cost_df(
            scn.final_cost_df_cf,
            L1_MRIO=L1_MRIO,
            VA_MRIO=VA_MRIO,
            LB_MRIO=LB_MRIO,
            W_MRIO=W_MRIO,
            core=core,
            mrio=mrio,
            annualize=annualize,
            lifetime_years=lifetime_years,
        )

        out[d_pp] = {
            "results_by_origin_lvl": r_lvl,
            "results_by_origin_sub_lvl": r_sub,
        }

    return out


def trace_upstream_trade_effect_sources(
    scenarios_delta: Dict[int, ScenarioDelta],
    *,
    delta_target: int,
    delta_base: int = 0,
    lost_market_countries: tuple[str, ...] | list[str] | None = None,
    lvl: int = 0,
    L1_MRIO: pd.DataFrame,
    VA_MRIO: pd.Series,
    LB_MRIO: pd.Series,
    W_MRIO: pd.Series | None = None,
    mrio: pd.DataFrame,
    core: tuple[str, ...] | list[str] | None = None,
    connected_country: dict | None = None,
    avg_dep_order: tuple[str, ...] | list[str] | None = None,
    weight_key: str = "Power (MW)",
    eu_countries: tuple[str, ...] | list[str] | set[str] = EU27,
    components_keep: tuple[str, ...] | list[str] | set[str] | None = None,
    subcomponents_keep: tuple[str, ...] | list[str] | set[str] | None = None,
    ind_labels: dict | None = None,
    annualize: bool = False,
    lifetime_years: float = 25.0,
    include_direct_market_rows: bool = False,
    min_abs_output: float = 0.0,
    min_abs_gva: float = 0.0,
    min_abs_jobs: float = 0.0,
    top_n: int | None = None,
    sort_by: str = "GVA",
) -> pd.DataFrame:
    """
    Trace which EU27 country-industries lose upstream activity when demand in
    selected non-EU markets contracts between two already-built delta scenarios.
    If ``lost_market_countries`` is None, all non-EU markets with a negative
    final-demand delta are included and ranked ex-post.

    This does not build new scenarios. It reuses ``ScenarioDelta.final_cost_df_cf``
    for ``delta_base`` and ``delta_target``, rebuilds the existing final-demand
    vectors, isolates the negative part of the target-market shock, and applies
    the Leontief matrix:

        source_loss = L1 @ min(delta_y_market, 0)

    Returned losses are signed. Negative Output/GVA/Jobs values are losses.
    """
    if int(delta_base) not in scenarios_delta:
        raise KeyError(f"delta_base={delta_base} not found in scenarios_delta.")
    if int(delta_target) not in scenarios_delta:
        raise KeyError(f"delta_target={delta_target} not found in scenarios_delta.")
    if not isinstance(L1_MRIO.index, pd.MultiIndex) or not isinstance(L1_MRIO.columns, pd.MultiIndex):
        raise ValueError("L1_MRIO must have MultiIndex rows and columns.")
    if mrio is None:
        raise ValueError("Pass mrio=MRIO.")

    core = tuple(str(c).strip() for c in (core or CORE_DEPLOYMENT))
    avg_dep_order = tuple(str(c).strip() for c in (avg_dep_order or core))
    eu_set = {str(c).strip().upper().replace("UK", "GB") for c in (eu_countries or ())}
    components_keep_set = None if components_keep is None else {str(c).strip() for c in components_keep}
    subcomponents_keep_set = None if subcomponents_keep is None else {str(c).strip() for c in subcomponents_keep}

    def _norm_country(c: str) -> str:
        c = str(c).strip().upper()
        return {"UK": "GB", "ROW": "ROW", "ROW_REST": "ROW", "REST OF WORLD": "ROW", "ROW.": "ROW", "ROW ": "ROW", "ROW\n": "ROW", "ROW\t": "ROW", "ROW\r": "ROW", "ROW": "ROW", "RoW".upper(): "ROW"}.get(c, c)

    lost_markets = None
    if lost_market_countries is not None:
        lost_markets = tuple(dict.fromkeys(_norm_country(c) for c in lost_market_countries if str(c).strip()))
        if not lost_markets:
            raise ValueError("lost_market_countries cannot be empty when provided.")

    def _origin_level(idx: pd.MultiIndex) -> str | int:
        for name in ("origin_region", "origin", "region", "country", "from_region", "src_region", "i_region"):
            if name in idx.names:
                return name
        return 0

    def _industry_level(idx: pd.MultiIndex) -> str | int:
        for name in ("industry", "sector", "product", "prd", "code", "i_sector"):
            if name in idx.names:
                return name
        return 1 if idx.nlevels > 1 else 0

    row_origin_level = _origin_level(L1_MRIO.index)
    row_ind_level = _industry_level(L1_MRIO.index)
    col_origin_level = _origin_level(L1_MRIO.columns)
    col_ind_level = _industry_level(L1_MRIO.columns)

    mrio_origin_level = _origin_level(mrio.index) if isinstance(mrio.index, pd.MultiIndex) else 0
    mrio_origin_set = set(mrio.index.get_level_values(mrio_origin_level).astype(str)) if isinstance(mrio.index, pd.MultiIndex) else set(mrio.index.astype(str))

    def _normalize_final_cost_df(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "origin_country" in out.columns:
            s = out["origin_country"].astype(str).str.strip().str.upper().replace({"UK": "GB", "ROW": "ROW", "ROW_REST": "ROW"})
            out["origin_country"] = s.where(s.isin(mrio_origin_set), "ROW")
        if not bool(annualize) and "value_alloc" in out.columns:
            out["value_alloc"] = out["value_alloc"].astype(float) * float(lifetime_years)
        return out

    base_df = _normalize_final_cost_df(scenarios_delta[int(delta_base)].final_cost_df_cf)
    targ_df = _normalize_final_cost_df(scenarios_delta[int(delta_target)].final_cost_df_cf)
    _, y_base_sub = build_y_shock_levels(base_df, list(core), mrio)
    _, y_targ_sub = build_y_shock_levels(targ_df, list(core), mrio)

    keys = set(y_base_sub.get(lvl, {}).keys()) | set(y_targ_sub.get(lvl, {}).keys())
    skip_self = _self_aggregate_keys_to_skip(keys)
    keys = [
        k for k in keys
        if isinstance(k, tuple)
        and len(k) >= 2
        and not _is_aggregate_component_key(k)
        and k not in skip_self
    ]
    if components_keep_set is not None:
        keys = [k for k in keys if str(k[0]).strip() in components_keep_set]
    if subcomponents_keep_set is not None:
        keys = [k for k in keys if str(k[1]).strip() in subcomponents_keep_set]
    keys = sorted(keys, key=lambda k: (str(k[0]), str(k[1])))

    def _weight(dep: str) -> float:
        if connected_country is None:
            return 1.0
        try:
            return float(connected_country.get(dep, {}).get(weight_key, 0.0))
        except Exception:
            return 0.0

    dep_weights = {dep: _weight(dep) for dep in avg_dep_order}
    if sum(dep_weights.values()) <= 0:
        dep_weights = {dep: 1.0 for dep in avg_dep_order}
    w_sum = sum(dep_weights.values()) or 1.0

    VA = pd.to_numeric(VA_MRIO, errors="coerce").reindex(L1_MRIO.index).fillna(0.0)
    LB = pd.to_numeric(LB_MRIO, errors="coerce").reindex(L1_MRIO.index).fillna(0.0)
    if W_MRIO is not None:
        W = pd.to_numeric(W_MRIO, errors="coerce").reindex(L1_MRIO.index).fillna(0.0).replace(0.0, np.nan)
    else:
        W = pd.Series(np.nan, index=L1_MRIO.index)

    row_origins = L1_MRIO.index.get_level_values(row_origin_level).astype(str).str.upper().str.replace("UK", "GB")
    row_industries = L1_MRIO.index.get_level_values(row_ind_level).astype(str)
    col_origins = L1_MRIO.columns.get_level_values(col_origin_level).astype(str).str.upper().str.replace("UK", "GB")
    col_industries = L1_MRIO.columns.get_level_values(col_ind_level).astype(str)
    eu_row_mask = np.asarray(row_origins.isin(eu_set), dtype=bool)
    if include_direct_market_rows:
        source_row_mask = np.ones(len(L1_MRIO.index), dtype=bool)
    else:
        source_row_mask = eu_row_mask

    def _industry_label(code: str) -> str:
        if not ind_labels:
            return str(code)
        code = str(code)
        variants = [code]
        if code.startswith("CPA_"):
            variants.append(code[4:])
        else:
            variants.append("CPA_" + code)
        variants.extend([code.replace("T", "-"), code.replace("-", "T"), code.replace("_", "-"), code.replace("-", "_")])
        for v in variants:
            if v in ind_labels and pd.notna(ind_labels[v]):
                return str(ind_labels[v])
        return code

    rows = []
    zero_template = pd.Series(0.0, index=L1_MRIO.columns)
    for comp, sub in keys:
        base_dep_map = y_base_sub.get(lvl, {}).get((comp, sub), {})
        targ_dep_map = y_targ_sub.get(lvl, {}).get((comp, sub), {})
        y_delta = zero_template.copy()
        for dep, wd in dep_weights.items():
            y0 = base_dep_map.get(dep, zero_template).reindex(L1_MRIO.columns).fillna(0.0)
            y1 = targ_dep_map.get(dep, zero_template).reindex(L1_MRIO.columns).fillna(0.0)
            y_delta = y_delta.add((float(wd) / w_sum) * (y1 - y0), fill_value=0.0)

        neg_mask = np.asarray(y_delta.to_numpy(dtype=float) < 0.0, dtype=bool)
        if lost_markets is None:
            market_mask = ~np.asarray(col_origins.isin(eu_set), dtype=bool)
        else:
            market_mask = np.asarray(col_origins.isin(lost_markets), dtype=bool)
        trace_mask = neg_mask & market_mask
        if not trace_mask.any():
            continue

        loss_cols = pd.DataFrame({
            "lost_market_country": col_origins[trace_mask],
            "lost_market_industry_code": col_industries[trace_mask],
            "value": y_delta.iloc[np.where(trace_mask)[0]].to_numpy(dtype=float),
        }, index=L1_MRIO.columns[trace_mask])

        for (market, market_industry), g_loss in loss_cols.groupby(
            ["lost_market_country", "lost_market_industry_code"],
            sort=False,
        ):
            y_loss = zero_template.copy()
            y_loss.loc[g_loss.index] = g_loss["value"].to_numpy(dtype=float)
            market_contraction = float(y_loss.sum())
            if abs(market_contraction) <= 1e-12:
                continue

            output = L1_MRIO.dot(y_loss).reindex(L1_MRIO.index).fillna(0.0)
            output = output.where(source_row_mask, 0.0)
            gva = output.mul(VA, axis=0)
            income = output.mul(LB, axis=0)
            jobs = income.divide(W, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

            tmp = pd.DataFrame({
                "source_country": row_origins,
                "source_industry_code": row_industries,
                "Output_loss": output.to_numpy(dtype=float),
                "GVA_loss": gva.to_numpy(dtype=float),
                "Jobs_loss": jobs.to_numpy(dtype=float),
            }, index=L1_MRIO.index)
            tmp = tmp[
                (tmp["Output_loss"].abs() > float(min_abs_output))
                | (tmp["GVA_loss"].abs() > float(min_abs_gva))
                | (tmp["Jobs_loss"].abs() > float(min_abs_jobs))
            ].copy()
            if tmp.empty:
                continue
            tmp = (
                tmp.groupby(["source_country", "source_industry_code"], as_index=False)
                [["Output_loss", "GVA_loss", "Jobs_loss"]]
                .sum()
            )
            tmp["Component"] = str(comp)
            tmp["Subcomponent"] = str(sub)
            tmp["lost_market_country"] = str(market)
            tmp["lost_market_industry_code"] = str(market_industry)
            tmp["lost_market_industry"] = _industry_label(str(market_industry))
            tmp["market_contraction"] = market_contraction
            tmp["source_industry"] = tmp["source_industry_code"].map(_industry_label)
            rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=[
            "Component", "Subcomponent", "lost_market_country",
            "lost_market_industry_code", "lost_market_industry",
            "source_country", "source_industry_code", "source_industry",
            "Output_loss", "GVA_loss", "Jobs_loss", "market_contraction",
        ])

    out = pd.concat(rows, ignore_index=True)
    out = out[[
        "Component", "Subcomponent", "lost_market_country",
        "lost_market_industry_code", "lost_market_industry",
        "source_country", "source_industry_code", "source_industry",
        "Output_loss", "GVA_loss", "Jobs_loss", "market_contraction",
    ]]
    sort_col_map = {
        "OUTPUT": "Output_loss",
        "GVA": "GVA_loss",
        "JOBS": "Jobs_loss",
    }
    sort_col = sort_col_map.get(str(sort_by).upper(), "GVA_loss")
    out["_sort_abs"] = out[sort_col].abs()
    out = out.sort_values("_sort_abs", ascending=False).drop(columns="_sort_abs").reset_index(drop=True)
    if top_n is not None:
        out = out.head(int(top_n)).copy()
    return out

# ============================================================
# DROP-IN: EU-level import substitution (direct only) + plots
# ============================================================

EU_LABEL, UK_LABEL, ROE_LABEL, ROW_LABEL = "EU27", "United Kingdom", "RoE", "RoW"
LOW_INCOME = {
    "TR", "BR", "CN",
    "MX", "ZA", "RU", "SA",
    "ID", "IN",
}


# Turbine component blocks used in df_alloc (when "Turbine" is passed as filter)
TURBINE_BLOCKS = {"Blades", "Hub", "Nacelle", "Tower", "Drivetrain"}


def _is_aggregate_component_key(key) -> bool:
    # True for total/ALL keys that should not be mixed with detailed rows.
    if not (isinstance(key, tuple) and len(key) >= 2):
        return False
    comp = str(key[0]).strip()
    subc = str(key[1]).strip()
    comp_cf = comp.casefold()
    subc_cf = subc.casefold()
    return comp_cf in {"all", "total"} or subc_cf in {"all", "total"}


def _self_aggregate_keys_to_skip(keys) -> set:
    """
    Self keys like ("Turbine", "Turbine") are aggregates only when the same
    component also has detailed children at that level.
    """
    clean = []
    for key in keys:
        if not (isinstance(key, tuple) and len(key) >= 2):
            continue
        if _is_aggregate_component_key(key):
            continue
        comp = str(key[0]).strip()
        subc = str(key[1]).strip()
        clean.append((key, comp.casefold(), subc.casefold()))

    comps_with_children = {
        comp_cf
        for _, comp_cf, subc_cf in clean
        if comp_cf != subc_cf
    }
    return {
        key
        for key, comp_cf, subc_cf in clean
        if comp_cf == subc_cf and comp_cf in comps_with_children
    }

# -----------------------------
# 0) Helpers
# -----------------------------
def norm_cc(x: str) -> str:
    return str(x).strip().upper()


def lighten_color(hex_color: str, amount: float = 0.35) -> str:
    if not isinstance(hex_color, str) or not hex_color.startswith("#") or len(hex_color) != 7:
        return hex_color
    r = int(hex_color[1:3], 16); g = int(hex_color[3:5], 16); b = int(hex_color[5:7], 16)
    r2 = int(r + (255 - r) * amount)
    g2 = int(g + (255 - g) * amount)
    b2 = int(b + (255 - b) * amount)
    return f"#{r2:02X}{g2:02X}{b2:02X}"

def darken_color(hex_color: str, amount: float = 0.25) -> str:
    if not isinstance(hex_color, str) or not hex_color.startswith("#") or len(hex_color) != 7:
        return hex_color
    r = int(hex_color[1:3], 16); g = int(hex_color[3:5], 16); b = int(hex_color[5:7], 16)
    r2 = int(r * (1.0 - amount))
    g2 = int(g * (1.0 - amount))
    b2 = int(b * (1.0 - amount))
    return f"#{r2:02X}{g2:02X}{b2:02X}"

def _prep_proc_df(df: pd.DataFrame, *, level: int = 1) -> pd.DataFrame:
    cg_col = f"component_group_L{level}"
    sc_col = f"sub_component_L{level}"

    out = df.copy()
    out["origin_country"]     = out["origin_country"].astype(str).str.strip().str.upper()
    out["deployment_country"] = out["deployment_country"].astype(str).str.strip().str.upper()
    out[cg_col] = out[cg_col].astype(str).str.strip()
    out[sc_col] = out[sc_col].astype(str).str.strip()
    out["industry_code"] = out["industry_code"].astype(str).str.strip()
    out["cost_block"]    = out["cost_block"].astype(str).str.strip().fillna("__NA__")
    out["value_alloc"]   = pd.to_numeric(out["value_alloc"], errors="coerce").fillna(0.0)
    return out

def _bucket_macro(o, *, EU27):
    o = str(o).strip().upper()
    if o in EU27:
        return EU_LABEL
    return ROW_LABEL

def _bucket_detailed(o, *, EU27, split_iso=(), split_names=None):
    split_names = split_names or {}
    o = str(o).strip().upper()
    if o in split_iso:
        return split_names.get(o, o)
    if o in EU27:
        return EU_LABEL
    return ROW_LABEL


# -----------------------------
# 0b) Capacity mix from results
# ----------------------------


def build_gva_share_map(
    results_by_origin_lvl: dict,
    lvl: int,
    *,
    EU27=(),
    components: list[str] | tuple[str, ...] | None = None,
    year: int = 2023,
    sector: str = "ALL",
    connected_country: dict | None = None,
    weight_key: str = "Power (MW)",
    gva_col: str = "GVA_Total",
) -> dict:
    """
    Build GVA share maps by component (category-level).
    Returns: {component: {"EU": {country: share}, "NON_EU": {...}}}
    Shares are computed from results_by_origin_lvl using optional deployment weights.
    """
    EU27 = {str(x).strip().upper() for x in EU27}
    if lvl not in results_by_origin_lvl:
        return {}

    if components is None:
        components = list(results_by_origin_lvl[lvl].keys())

    def _norm(d: dict) -> dict:
        s = float(sum(d.values()))
        if s <= 0:
            return {}
        return {k: float(v) / s for k, v in d.items() if float(v) > 0}

    out: dict = {}

    for comp in components:
        if comp not in results_by_origin_lvl[lvl]:
            continue
        df = results_by_origin_lvl[lvl][comp].copy()
        if "Year" in df.columns:
            df = df[df["Year"] == year]
        if "Sector" in df.columns:
            df = df[df["Sector"].astype(str) == str(sector)]

        df["Deployment"] = df["Deployment"].astype(str).str.strip().str.upper()
        df["Country"] = df["Country"].astype(str).str.strip().str.upper()

        if gva_col not in df.columns:
            continue

        if connected_country is not None:
            w_map = {str(k).upper(): float(v.get(weight_key, 0.0)) for k, v in connected_country.items()}
            df["_w"] = df["Deployment"].map(w_map).fillna(0.0)
            if float(df["_w"].sum()) <= 0:
                df["_w"] = 1.0
        else:
            df["_w"] = 1.0

        df["_val"] = pd.to_numeric(df[gva_col], errors="coerce").fillna(0.0) * df["_w"]
        g = df.groupby("Country")["_val"].sum()

        eu = {c: float(g.get(c, 0.0)) for c in g.index if c in EU27}
        non = {c: float(g.get(c, 0.0)) for c in g.index if c not in EU27}

        out[str(comp)] = {"EU": _norm(eu), "NON_EU": _norm(non)}

    return out


def build_gva_share_map_sub(
    results_by_origin_sub_lvl: dict,
    lvl: int,
    *,
    EU27=(),
    components: list[str] | tuple[str, ...] | None = None,
    year: int = 2023,
    sector: str = "ALL",
    connected_country: dict | None = None,
    weight_key: str = "Power (MW)",
    gva_col: str = "GVA_Total",
) -> dict:
    """
    Build GVA share maps by subcomponent (component, subcomponent).
    Returns: {(component, subcomponent): {"EU": {country: share}, "NON_EU": {...}}}
    Shares are computed from results_by_origin_sub_lvl using optional deployment weights.
    """
    EU27 = {str(x).strip().upper() for x in EU27}
    if lvl not in results_by_origin_sub_lvl:
        return {}

    if components is not None:
        comp_set = {str(x).strip() for x in components}
    else:
        comp_set = None

    def _norm(d: dict) -> dict:
        s = float(sum(d.values()))
        if s <= 0:
            return {}
        return {k: float(v) / s for k, v in d.items() if float(v) > 0}

    out: dict = {}

    for (comp, subc), df in results_by_origin_sub_lvl[lvl].items():
        comp = str(comp)
        if comp_set is not None and comp not in comp_set:
            continue

        d = df.copy()
        if "Year" in d.columns:
            d = d[d["Year"] == year]
        if "Sector" in d.columns:
            d = d[d["Sector"].astype(str) == str(sector)]

        if gva_col not in d.columns:
            continue

        d["Deployment"] = d["Deployment"].astype(str).str.strip().str.upper()
        d["Country"] = d["Country"].astype(str).str.strip().str.upper()

        if connected_country is not None:
            w_map = {str(k).upper(): float(v.get(weight_key, 0.0)) for k, v in connected_country.items()}
            d["_w"] = d["Deployment"].map(w_map).fillna(0.0)
            if float(d["_w"].sum()) <= 0:
                d["_w"] = 1.0
        else:
            d["_w"] = 1.0

        d["_val"] = pd.to_numeric(d[gva_col], errors="coerce").fillna(0.0) * d["_w"]
        g = d.groupby("Country")["_val"].sum()

        eu = {c: float(g.get(c, 0.0)) for c in g.index if c in EU27}
        non = {c: float(g.get(c, 0.0)) for c in g.index if c not in EU27}

        out[(comp, str(subc))] = {"EU": _norm(eu), "NON_EU": _norm(non)}

    return out


def build_gva_share_map_bucket_df_alloc(
    df_alloc: pd.DataFrame,
    *,
    EU27=(),
    exclude_origin=("EXPORTROW",),
    gva_col: str | None = None,
) -> dict:
    """
    Build GVA/value share maps by bucket from df_alloc.
    Bucket = (deployment_country, OEM, component_group, sub_component, cost_block).
    If gva_col is provided and present, use it; otherwise use value_alloc.
    """
    EU27 = {str(x).strip().upper() for x in EU27}
    exclude_origin = {str(x).strip().upper() for x in (exclude_origin or ())}

    def _norm(d: dict) -> dict:
        s = float(sum(d.values()))
        if s <= 0:
            return {}
        return {k: float(v) / s for k, v in d.items() if float(v) > 0}

    df = df_alloc.copy()
    for c in ["deployment_country","OEM","component_group","sub_component","cost_block","origin_country"]:
        if c not in df.columns:
            raise KeyError(f"Missing column: {c}")
        df[c] = df[c].astype(str).str.strip()
    df["deployment_country"] = df["deployment_country"].str.upper().replace({"UK":"GB"})
    df["origin_country"] = df["origin_country"].str.upper().replace({"UK":"GB","RoW":"ROW"})
    df["cost_block"] = df["cost_block"].fillna("__NA__")

    val_col = None
    if gva_col and gva_col in df.columns:
        val_col = gva_col
    elif "value_alloc" in df.columns:
        val_col = "value_alloc"
    else:
        raise KeyError("df_alloc missing value_alloc and gva_col not found.")

    df[val_col] = pd.to_numeric(df[val_col], errors="coerce").fillna(0.0)
    df = df.loc[~df["origin_country"].isin(exclude_origin)].copy()

    B = ["deployment_country","OEM","component_group","sub_component","cost_block"]
    out: dict = {}
    g = df.groupby(B + ["origin_country"], dropna=False)[val_col].sum().reset_index()

    for key, sub in g.groupby(B, dropna=False):
        by_o = sub.set_index("origin_country")[val_col].to_dict()
        eu = {o: float(v) for o, v in by_o.items() if o in EU27}
        non = {o: float(v) for o, v in by_o.items() if o not in EU27}
        out[key] = {"EU": _norm(eu), "NON_EU": _norm(non)}

    return out


def build_gva_share_map_bucket_flows(
    flows_by_component_hier_all: dict,
    *,
    EU27=(),
    gva_col: str | None = None,
) -> dict:
    """
    Build GVA/value share maps by bucket from flows_by_component_hier_all.
    Bucket = (deployment, parent, child).
    Returns: {bucket_key: {"EU": {country: share}, "NON_EU": {...}}}
    """
    EU27 = {str(x).strip().upper() for x in EU27}

    def _norm(d: dict) -> dict:
        s = float(sum(d.values()))
        if s <= 0:
            return {}
        return {k: float(v) / s for k, v in d.items() if float(v) > 0}

    out: dict = {}
    for dep, parents in flows_by_component_hier_all.items():
        for parent, childs in parents.items():
            for child, od in childs.items():
                by_o = {str(o).strip().upper(): float(v) for o, v in od.items()}
                eu = {o: float(v) for o, v in by_o.items() if o in EU27}
                non = {o: float(v) for o, v in by_o.items() if o not in EU27}
                out[(str(dep).strip().upper(), str(parent).strip(), str(child).strip())] = {
                    "EU": _norm(eu),
                    "NON_EU": _norm(non),
                }
    return out


# -----------------------------
# 1) Counterfactual: Δ-pp LC shift (direct only), industry-fixed
# -----------------------------
def apply_delta_lc_shift_direct_turbine(
    df_alloc: pd.DataFrame,
    *,
    group_set,
    delta_pp: float,
    exclude_origin=("EXPORTROW",),
    eps: float = 1e-12,
    receiver_policy: str = "baseline_mix",   
    capacity_map: dict | None = None,  # deprecated (ignored)
    gva_share_map: dict | None = None,
    gva_share_map_bucket: dict | None = None,
    single_receiver_country: str | None = None,  
    single_donor_country: str | None = None,
    component_filter: set[str] | list[str] | None = None,
    receiver_whitelist: set[str] | list[str] | None = None,
    donor_whitelist: set[str] | list[str] | None = None,
    eu_spend_reduction_pct: float = 0.0,
    preserve_total: bool = True,
):
    import numpy as np
    import pandas as pd

    def norm(x): return str(x).strip().upper()
    def _expand_component_filter(cf):
        if not cf:
            return None
        items = {str(x).strip() for x in cf if str(x).strip()}
        items_cf = {x.casefold() for x in items}
        if ("turbine" in items_cf) or ("turbines" in items_cf):
            items.update(TURBINE_BLOCKS)
        return items

    group_set = {norm(x) for x in group_set}
    exclude_origin = {norm(x) for x in exclude_origin}
    receiver_whitelist = {norm(x) for x in (receiver_whitelist or [])} or None
    donor_whitelist = {norm(x) for x in (donor_whitelist or [])} or None
    single_donor_country_norm = None if single_donor_country is None else norm(single_donor_country)
    component_filter = _expand_component_filter(component_filter)

    df = df_alloc.copy()

    # normalize
    for c in ["deployment_country","OEM","component_group","sub_component","cost_block","origin_country"]:
        if c not in df.columns:
            raise KeyError(f"df_alloc missing required column '{c}'")
        df[c] = df[c].astype(str).str.strip()
    df["deployment_country"] = df["deployment_country"].str.upper().replace({"UK":"GB"})
    df["origin_country"] = df["origin_country"].str.upper().replace({"UK":"GB","RoW":"ROW"})
    df["value_alloc"] = pd.to_numeric(df["value_alloc"], errors="coerce").fillna(0.0)

    # keep excluded rows untouched
    mask_excl = df["origin_country"].isin(exclude_origin)
    df_excl = df.loc[mask_excl].copy()
    d0 = df.loc[~mask_excl].copy()

    # optional: restrict to selected component groups (others remain unchanged)
    if component_filter is not None:
        comp_cf = {str(x).casefold() for x in component_filter}
        mask_comp = d0["component_group"].astype(str).str.strip().str.casefold().isin(comp_cf)
        d_other = d0.loc[~mask_comp].copy()
        d0 = d0.loc[mask_comp].copy()
    else:
        d_other = None

    if d0.empty:
        df_new = d0.copy()
        if len(df_excl):
            df_new = pd.concat([df_new, df_excl], ignore_index=True)
        if d_other is not None and len(d_other):
            df_new = pd.concat([df_new, d_other], ignore_index=True)
        stats = {
            "delta_pp": float(delta_pp),
            "moved_abs_total": 0.0,
            "unshiftable_abs_total": 0.0,
            "group_set_size_effective": 0,
            "non_group_origins_present": [],
            "receiver_policy": str(receiver_policy),
            "single_receiver_country": None if single_receiver_country is None else norm(single_receiver_country),
            "single_donor_country": single_donor_country_norm,
            "receiver_whitelist": sorted(receiver_whitelist) if receiver_whitelist else [],
            "donor_whitelist": sorted(donor_whitelist) if donor_whitelist else [],
            "eu_spend_reduction_pct": float(eu_spend_reduction_pct or 0.0),
            "preserve_total": bool(preserve_total),
        }
        return df_new, stats

    # effective universes
    orig_universe = set(d0["origin_country"].unique())
    group_eff = group_set & orig_universe
    non_group_eff = orig_universe - group_eff

    # bucket (A): per OEM
    B = ["deployment_country","OEM","component_group","sub_component","cost_block"]

    # ---- 1) need per bucket
    tot = d0.groupby(B, dropna=False)["value_alloc"].sum().rename("T").reset_index()
    grp = (
        d0.loc[d0["origin_country"].isin(group_eff)]
          .groupby(B, dropna=False)["value_alloc"].sum()
          .rename("G").reset_index()
    )
    need_df = tot.merge(grp, on=B, how="left").fillna({"G":0.0})
    need_df["s0"] = np.where(need_df["T"] > eps, need_df["G"]/need_df["T"], 0.0)
    need_df["s_target"] = np.clip(need_df["s0"] + float(delta_pp), 0.0, 1.0)
    need_df["need"] = (need_df["s_target"] - need_df["s0"]) * need_df["T"]
    need_map = {tuple(r[k] for k in B): float(r["need"]) for _, r in need_df.iterrows()}

    # ---- 2) GVA-share weights (category-level, symmetric)
    def _parent_key(cg: str) -> str:
        c = str(cg).strip()
        return "Turbine" if c in TURBINE_BLOCKS else c

    def _norm_weights(d: dict) -> dict:
        s = float(sum(d.values()))
        if s <= eps:
            return {}
        return {k: float(v) / s for k, v in d.items() if float(v) > 0}

    def _weights_for_group(
        parent_raw: str,
        parent_fallback: str,
        subc: str | None,
        group: str,
        allowed: set[str] | None = None,
    ) -> dict:
        base = {}
        if gva_share_map_bucket is not None and bucket_key in gva_share_map_bucket:
            base = gva_share_map_bucket[bucket_key].get(group, {}) or {}
        if not base and gva_share_map:
            if subc is not None:
                if (parent_raw, str(subc)) in gva_share_map:
                    base = gva_share_map[(parent_raw, str(subc))].get(group, {}) or {}
                elif (parent_fallback, str(subc)) in gva_share_map:
                    base = gva_share_map[(parent_fallback, str(subc))].get(group, {}) or {}
            if not base:
                if parent_raw in gva_share_map:
                    base = gva_share_map[parent_raw].get(group, {}) or {}
                elif parent_fallback in gva_share_map:
                    base = gva_share_map[parent_fallback].get(group, {}) or {}
        if allowed is not None:
            base = {o: float(base.get(o, 0.0)) for o in allowed if float(base.get(o, 0.0)) > 0}
        if receiver_whitelist is not None:
            base = {o: v for o, v in base.items() if o in receiver_whitelist}
        if not base and str(group) == "NON_EU":
            base = {"ROW": 1.0}
        return _norm_weights(base)

    def _allocate_reduction(by_o: pd.Series, donors: list[str], weights: dict, total: float) -> dict:
        alloc = {o: 0.0 for o in donors}
        rem = float(total)
        active = {o for o in donors if float(by_o.get(o, 0.0)) > eps and float(weights.get(o, 0.0)) > 0}
        weights = {o: float(weights.get(o, 0.0)) for o in active}

        while rem > eps and active:
            wsum = float(sum(weights[o] for o in active))
            if wsum <= eps:
                break
            capped = []
            for o in list(active):
                cap = float(by_o.get(o, 0.0)) - alloc[o]
                if cap <= eps:
                    capped.append(o)
                    continue
                amt = rem * (weights[o] / wsum)
                if amt >= cap - eps:
                    alloc[o] += cap
                    rem -= cap
                    capped.append(o)
                else:
                    alloc[o] += amt
            for o in capped:
                active.discard(o)
            if not capped:
                rem = 0.0
                break

        if rem > eps:
            rem_caps = {o: float(by_o.get(o, 0.0)) - alloc[o] for o in donors}
            rem_caps = {o: v for o, v in rem_caps.items() if v > eps}
            cap_sum = float(sum(rem_caps.values()))
            if cap_sum > eps:
                for o, cap in rem_caps.items():
                    alloc[o] += rem * (cap / cap_sum)
                rem = 0.0
        return alloc

    # ---- 3) Execute per bucket (preserve total in bucket)
    parts = []
    unshiftable = 0.0

    for key, g in d0.groupby(B, dropna=False):
        need = float(need_map.get(key, 0.0))
        if abs(need) <= eps:
            parts.append(g)
            continue

        dep, oem, cg, sc, cb = key
        move_into_group = (need > 0)
        move = abs(need)

        by_o = g.groupby("origin_country")["value_alloc"].sum()

        donors_set = non_group_eff if move_into_group else group_eff
        donor_filter = donor_whitelist
        if move_into_group and single_donor_country_norm is not None:
            donor_filter = (
                {single_donor_country_norm}
                if donor_filter is None
                else (donor_filter | {single_donor_country_norm})
            )
        if donor_filter is not None:
            donors_set = donors_set & donor_filter
        donors = [o for o in by_o.index if o in donors_set]
        donor_sum = float(by_o.loc[donors].sum()) if donors else 0.0
        if donor_sum <= eps:
            unshiftable += move
            parts.append(g)
            continue
        move = min(move, donor_sum)

        parent_raw = str(cg).strip()
        parent_key = _parent_key(cg)
        bucket_key = (dep, oem, cg, sc, cb)
        donor_group = "NON_EU" if move_into_group else "EU"
        recv_group = "EU" if move_into_group else "NON_EU"

        donor_w = _weights_for_group(parent_raw, parent_key, sc, donor_group, allowed=set(donors))
        if not donor_w:
            donor_w = _norm_weights({o: float(by_o.loc[o]) for o in donors})

        # receivers: single-country override only for offshoring (non‑EU)
        receivers = []
        recv_w = {}
        if receiver_policy == "single_country" and (not move_into_group):
            if single_receiver_country is not None:
                oN = norm(single_receiver_country)
                if oN in non_group_eff and (receiver_whitelist is None or oN in receiver_whitelist):
                    receivers = [oN]
                    recv_w = {oN: 1.0}

        if not receivers:
            recv_w = _weights_for_group(parent_raw, parent_key, sc, recv_group, allowed=None)
            if receiver_whitelist is not None:
                recv_w = {o: v for o, v in recv_w.items() if o in receiver_whitelist}
            if not recv_w:
                # fallback to current by_o within destination group
                dest_set = group_eff if move_into_group else non_group_eff
                recv_w = _norm_weights({o: float(by_o.loc[o]) for o in by_o.index if o in dest_set})
            receivers = list(recv_w.keys())

        if not receivers:
            unshiftable += move
            parts.append(g)
            continue

        # add missing receiver rows if needed
        missing = [o for o in receivers if o not in by_o.index]
        if missing:
            proto = g.iloc[[0]].copy()
            if "origin_type" in proto.columns:
                proto["origin_type"] = "CF - created"
            proto["value_alloc"] = 0.0
            if "value" in proto.columns:
                proto["value"] = 0.0
            adds = []
            for o in missing:
                a = proto.copy()
                a["origin_country"] = o
                adds.append(a)
            g = pd.concat([g] + adds, ignore_index=True)
            by_o = g.groupby("origin_country")["value_alloc"].sum()

        new_by_o = by_o.copy()

        # donor reductions using GVA-share weights (capped by available amounts)
        alloc = _allocate_reduction(by_o, donors, donor_w, move)
        for o, red in alloc.items():
            new_by_o.loc[o] = float(new_by_o.get(o, 0.0)) - float(red)

        # receiver additions using GVA-share weights
        recv_w = _norm_weights(recv_w)
        if not recv_w:
            unshiftable += move
            parts.append(g)
            continue
        wsum = float(sum(recv_w.values()))
        pct = float(eu_spend_reduction_pct or 0.0)
        apply_markup = (pct > 0.0) and (float(delta_pp) != 0.0)

        if apply_markup and move_into_group:
            # Reshoring: increase amount reallocated from low-income donors
            adj_total = 0.0
            for o, red in alloc.items():
                if norm(o) in LOW_INCOME:
                    adj_total += float(red) / max(1e-12, (1.0 - pct))
                else:
                    adj_total += float(red)
            for o, wi in recv_w.items():
                new_by_o.loc[o] = float(new_by_o.get(o, 0.0)) + adj_total * float(wi) / wsum
        elif apply_markup and (not move_into_group):
            # Offshoring: reduce amount reallocated to low-income receivers
            for o, wi in recv_w.items():
                factor = (1.0 - pct) if (norm(o) in LOW_INCOME) else 1.0
                new_by_o.loc[o] = float(new_by_o.get(o, 0.0)) + move * float(wi) / wsum * factor
        else:
            for o, wi in recv_w.items():
                new_by_o.loc[o] = float(new_by_o.get(o, 0.0)) + move * float(wi) / wsum

        g2 = g.copy()
        for oc, sr in g2.groupby("origin_country"):
            base = sr["value_alloc"].to_numpy()
            s = float(base.sum())
            tgt = float(new_by_o.get(oc, 0.0))
            g2.loc[sr.index, "value_alloc"] = (base / s * tgt) if s > eps else (tgt / len(sr))

        if preserve_total:
            T0 = float(by_o.sum())
            T1 = float(g2["value_alloc"].sum())
            if T1 > eps and T0 > eps:
                g2["value_alloc"] *= T0 / T1

        parts.append(g2)

    df_new = pd.concat(parts, ignore_index=True)
    if len(df_excl):
        df_new = pd.concat([df_new, df_excl], ignore_index=True)
    if d_other is not None and len(d_other):
        df_new = pd.concat([df_new, d_other], ignore_index=True)

    K = B + ["origin_country"]
    b = d0.groupby(K, dropna=False)["value_alloc"].sum().rename("base").reset_index()
    s = df_new.loc[~df_new["origin_country"].isin(exclude_origin)] \
              .groupby(K, dropna=False)["value_alloc"].sum().rename("scn").reset_index()
    m = b.merge(s, on=K, how="outer").fillna(0.0)
    moved_abs_total = 0.5 * float((m["scn"] - m["base"]).abs().sum())

    stats = {
        "delta_pp": float(delta_pp),
        "moved_abs_total": float(moved_abs_total),
        "unshiftable_abs_total": float(unshiftable),
        "group_set_size_effective": int(len(group_eff)),
        "non_group_origins_present": sorted(list(non_group_eff)),
        "receiver_policy": str(receiver_policy),  # NEW
        "single_receiver_country": None if single_receiver_country is None else norm(single_receiver_country),
        "single_donor_country": single_donor_country_norm,
        "receiver_whitelist": sorted(receiver_whitelist) if receiver_whitelist else [],
        "donor_whitelist": sorted(donor_whitelist) if donor_whitelist else [],
        "eu_spend_reduction_pct": float(eu_spend_reduction_pct or 0.0),
        "preserve_total": bool(preserve_total),
    }
    return df_new, stats

# ============================================================
# 2) Minimal patch: apply_delta_lc_shift_direct_farm
# ============================================================
def apply_delta_lc_shift_direct_farm(
    flows_by_component_hier_all: dict,
    *,
    group_set,
    delta_pp: float,
    eps: float = 1e-12,
    receiver_policy: str = "baseline_mix", 
    gva_share_map: dict | None = None,
    gva_share_map_bucket: dict | None = None,
    single_receiver_country: str | None = None, 
    single_donor_country: str | None = None,
    component_filter: set[str] | list[str] | None = None,
    receiver_whitelist: set[str] | list[str] | None = None,
    donor_whitelist: set[str] | list[str] | None = None,
    eu_spend_reduction_pct: float = 0.0,
    preserve_total: bool = True,
):
    from collections import defaultdict
    import copy

    def norm(x): return str(x).strip().upper()

    GROUP = {norm(x) for x in group_set}
    receiver_whitelist = {norm(x) for x in (receiver_whitelist or [])} or None
    donor_whitelist = {norm(x) for x in (donor_whitelist or [])} or None
    single_donor_country_norm = None if single_donor_country is None else norm(single_donor_country)
    component_filter = {str(x).strip() for x in (component_filter or [])} or None
    new_flows = copy.deepcopy(flows_by_component_hier_all)

    all_origins = set()
    for dep, parents in flows_by_component_hier_all.items():
        for parent, childs in parents.items():
            for child, od in childs.items():
                all_origins.update({norm(o) for o in od.keys()})

    group_eff = GROUP & all_origins
    non_group_eff = all_origins - group_eff

    def _norm_weights(d: dict) -> dict:
        s = float(sum(d.values()))
        if s <= eps:
            return {}
        return {k: float(v) / s for k, v in d.items() if float(v) > 0}

    def _weights_for_group(parent: str, subc: str | None, group: str, allowed: set[str] | None = None) -> dict:
        base = {}
        if gva_share_map_bucket is not None and bucket_key in gva_share_map_bucket:
            base = gva_share_map_bucket[bucket_key].get(group, {}) or {}
        if not base and gva_share_map:
            if subc is not None and (parent, str(subc)) in gva_share_map:
                base = gva_share_map[(parent, str(subc))].get(group, {}) or {}
            elif parent in gva_share_map:
                base = gva_share_map[parent].get(group, {}) or {}
        if allowed is not None:
            base = {o: float(base.get(o, 0.0)) for o in allowed if float(base.get(o, 0.0)) > 0}
        if receiver_whitelist is not None:
            base = {o: v for o, v in base.items() if o in receiver_whitelist}
        if not base and str(group) == "NON_EU":
            base = {"ROW": 1.0}
        return _norm_weights(base)

    def _allocate_reduction(by_o: dict, donors: list[str], weights: dict, total: float) -> dict:
        alloc = {o: 0.0 for o in donors}
        rem = float(total)
        active = {o for o in donors if float(by_o.get(o, 0.0)) > eps and float(weights.get(o, 0.0)) > 0}
        weights = {o: float(weights.get(o, 0.0)) for o in active}

        while rem > eps and active:
            wsum = float(sum(weights[o] for o in active))
            if wsum <= eps:
                break
            capped = []
            for o in list(active):
                cap = float(by_o.get(o, 0.0)) - alloc[o]
                if cap <= eps:
                    capped.append(o)
                    continue
                amt = rem * (weights[o] / wsum)
                if amt >= cap - eps:
                    alloc[o] += cap
                    rem -= cap
                    capped.append(o)
                else:
                    alloc[o] += amt
            for o in capped:
                active.discard(o)
            if not capped:
                rem = 0.0
                break

        if rem > eps:
            rem_caps = {o: float(by_o.get(o, 0.0)) - alloc[o] for o in donors}
            rem_caps = {o: v for o, v in rem_caps.items() if v > eps}
            cap_sum = float(sum(rem_caps.values()))
            if cap_sum > eps:
                for o, cap in rem_caps.items():
                    alloc[o] += rem * (cap / cap_sum)
                rem = 0.0
        return alloc

    moved_abs_total = 0.0
    unshiftable_abs_total = 0.0

    for dep, parents in flows_by_component_hier_all.items():
        for parent, childs in parents.items():
            if component_filter is not None and str(parent).strip() not in component_filter:
                continue
            for child, od in childs.items():
                bucket_key = (str(dep).strip().upper(), str(parent).strip(), str(child).strip())
                by_o = {norm(o): float(v) for o, v in od.items() if float(v) != 0.0}
                T = sum(by_o.values())
                if T <= eps:
                    continue

                G = sum(v for o, v in by_o.items() if o in group_eff)
                s0 = G / T
                s_target = min(1.0, max(0.0, s0 + float(delta_pp)))
                need = (s_target - s0) * T
                if abs(need) <= eps:
                    continue

                move_into_group = (need > 0)
                move = abs(need)

                donors_set = non_group_eff if move_into_group else group_eff
                donor_filter = donor_whitelist
                if move_into_group and single_donor_country_norm is not None:
                    donor_filter = (
                        {single_donor_country_norm}
                        if donor_filter is None
                        else (donor_filter | {single_donor_country_norm})
                    )
                if donor_filter is not None:
                    donors_set = donors_set & donor_filter
                donors = [o for o in by_o.keys() if o in donors_set]
                donor_sum = sum(by_o[o] for o in donors) if donors else 0.0
                if donor_sum <= eps:
                    unshiftable_abs_total += move
                    continue
                move = min(move, donor_sum)

                parent_key = str(parent).strip()
                donor_group = "NON_EU" if move_into_group else "EU"
                recv_group = "EU" if move_into_group else "NON_EU"

                donor_w = _weights_for_group(parent_key, child, donor_group, allowed=set(donors))
                if not donor_w:
                    donor_w = _norm_weights({o: float(by_o.get(o, 0.0)) for o in donors})

                # receivers: single-country override only for offshoring (non‑EU)
                receivers = []
                recv_w = {}
                if receiver_policy == "single_country" and (not move_into_group):
                    if single_receiver_country is not None:
                        oN = norm(single_receiver_country)
                        if oN in non_group_eff and (receiver_whitelist is None or oN in receiver_whitelist):
                            receivers = [oN]
                            recv_w = {oN: 1.0}

                if not receivers:
                    recv_w = _weights_for_group(parent_key, child, recv_group, allowed=None)
                    if receiver_whitelist is not None:
                        recv_w = {o: v for o, v in recv_w.items() if o in receiver_whitelist}
                    if not recv_w:
                        dest_set = group_eff if move_into_group else non_group_eff
                        recv_w = _norm_weights({o: float(by_o.get(o, 0.0)) for o in by_o.keys() if o in dest_set})
                    receivers = list(recv_w.keys())

                if not receivers:
                    unshiftable_abs_total += move
                    continue

                for o in receivers:
                    if o not in by_o:
                        by_o[o] = 0.0

                new_by_o = dict(by_o)

                # donor reductions using GVA-share weights (capped by available amounts)
                alloc = _allocate_reduction(by_o, donors, donor_w, move)
                for o, red in alloc.items():
                    new_by_o[o] = new_by_o.get(o, 0.0) - float(red)

                # receiver additions using GVA-share weights
                recv_w = _norm_weights(recv_w)
                if not recv_w:
                    unshiftable_abs_total += move
                    continue
                wsum = float(sum(recv_w.values()))
                pct = float(eu_spend_reduction_pct or 0.0)
                apply_markup = (pct > 0.0) and (float(delta_pp) != 0.0)

                if apply_markup and move_into_group:
                    # Reshoring: increase amount reallocated from low-income donors
                    adj_total = 0.0
                    for o, red in alloc.items():
                        if norm(o) in LOW_INCOME:
                            adj_total += float(red) / max(1e-12, (1.0 - pct))
                        else:
                            adj_total += float(red)
                    for o, wi in recv_w.items():
                        new_by_o[o] = new_by_o.get(o, 0.0) + adj_total * (wi / wsum)
                elif apply_markup and (not move_into_group):
                    # Offshoring: reduce amount reallocated to low-income receivers
                    for o, wi in recv_w.items():
                        factor = (1.0 - pct) if (norm(o) in LOW_INCOME) else 1.0
                        new_by_o[o] = new_by_o.get(o, 0.0) + move * (wi / wsum) * factor
                else:
                    for o, wi in recv_w.items():
                        new_by_o[o] = new_by_o.get(o, 0.0) + move * (wi / wsum)

                if preserve_total:
                    T1 = sum(new_by_o.values())
                    if T1 > eps:
                        scale = T / T1
                        for o in new_by_o:
                            new_by_o[o] *= scale

                bucket = new_flows[dep][parent][child]
                for o in list(bucket.keys()):
                    bucket[o] = 0.0
                for o, v in new_by_o.items():
                    bucket[o] = float(v)

                moved_abs_total += move

    stats = {
        "delta_pp": float(delta_pp),
        "moved_abs_total": float(moved_abs_total),
        "unshiftable_abs_total": float(unshiftable_abs_total),
        "group_set_size_effective": int(len(group_eff)),
        "non_group_origins_present": sorted(list(non_group_eff)),
        "receiver_policy": str(receiver_policy),
        "single_receiver_country": None if single_receiver_country is None else norm(single_receiver_country),
        "single_donor_country": single_donor_country_norm,
        "receiver_whitelist": sorted(receiver_whitelist) if receiver_whitelist else [],
        "donor_whitelist": sorted(donor_whitelist) if donor_whitelist else [],
        "eu_spend_reduction_pct": float(eu_spend_reduction_pct or 0.0),
        "preserve_total": bool(preserve_total),
    }
    return new_flows, stats

def build_delta_scenarios(
    *,
    group_set,
    df_alloc: pd.DataFrame,
    flows_by_component_hier_all: dict,
    ind_root: str | None = None,
    deltas_pp=range(-20, 21),
    receiver_policy: str = "baseline_mix",   
    gva_share_map: dict | None = None,
    gva_share_map_bucket: dict | None = None,
    single_receiver_country: str | None = None,  
    single_donor_country: str | None = None,
    ind_ind: bool = False,
    component_filter_farm: set[str] | list[str] | None = None,
    component_filter_turbine: set[str] | list[str] | None = None,
    receiver_whitelist: set[str] | list[str] | None = None,
    donor_whitelist: set[str] | list[str] | None = None,
    eu_spend_reduction_pct: float = 0.0,
    preserve_total: bool = True,
) -> Dict[int, ScenarioDelta]:
    scenarios: Dict[int, ScenarioDelta] = {}

    def _expand_component_filter(cf):
        if not cf:
            return None
        items = {str(x).strip() for x in cf if str(x).strip()}
        items_cf = {x.casefold() for x in items}
        if ("turbine" in items_cf) or ("turbines" in items_cf):
            items.update(TURBINE_BLOCKS)
        return items

    # optional component filters
    component_filter_farm = {str(x).strip() for x in (component_filter_farm or [])} or None
    component_filter_turbine = _expand_component_filter(component_filter_turbine)

    def _filter_flows(flows, keep):
        out = {}
        for dep, parents in flows.items():
            p2 = {cg: childs for cg, childs in parents.items() if str(cg).strip() in keep}
            out[dep] = p2
        return out

    flows_src = flows_by_component_hier_all
    if component_filter_farm is not None:
        flows_src = _filter_flows(flows_by_component_hier_all, component_filter_farm)

    df_alloc_src = df_alloc
    if component_filter_turbine is not None and "component_group" in df_alloc.columns:
        comp_cf = {str(x).casefold() for x in component_filter_turbine}
        mask = df_alloc["component_group"].astype(str).str.strip().str.casefold().isin(comp_cf)
        df_alloc_src = df_alloc.loc[mask].copy()

    for d_pp in deltas_pp:
        delta = d_pp / 100.0

        df_alloc_cf, stats_turb = apply_delta_lc_shift_direct_turbine(
            df_alloc_src,
            group_set=group_set,
            delta_pp=delta,
            receiver_policy=receiver_policy,
            gva_share_map=gva_share_map,
            gva_share_map_bucket=gva_share_map_bucket,
            single_receiver_country=single_receiver_country,  
            single_donor_country=single_donor_country,
            component_filter=component_filter_turbine,
            receiver_whitelist=receiver_whitelist,
            donor_whitelist=donor_whitelist,
            eu_spend_reduction_pct=eu_spend_reduction_pct,
            preserve_total=preserve_total,
        )

        flows_cf, stats_farm = apply_delta_lc_shift_direct_farm(
            flows_src,
            group_set=group_set,
            delta_pp=delta,
            receiver_policy=receiver_policy,
            gva_share_map=gva_share_map,
            gva_share_map_bucket=gva_share_map_bucket,
            single_receiver_country=single_receiver_country,  # NEW
            single_donor_country=single_donor_country,
            component_filter=component_filter_farm,
            receiver_whitelist=receiver_whitelist,
            donor_whitelist=donor_whitelist,
            eu_spend_reduction_pct=eu_spend_reduction_pct,
            preserve_total=preserve_total,
        )

        if ind_root is None:
            ind_root = str(HELPERS_DIR)

        final_cost_df_cf, component_groups, subcomponents_dict = run_industrial(
            ROOT=ind_root,
            flows_by_component_hier_all=flows_cf,
            df_alloc=df_alloc_cf,
            ind_ind=ind_ind,
        )

        scenarios[d_pp] = ScenarioDelta(
            delta_pp=delta,
            final_cost_df_cf=final_cost_df_cf,
            stats_turbine=stats_turb,
            stats_farm=stats_farm,
        )

    return scenarios


import os
import pickle
from dataclasses import asdict
from typing import Dict, Any
import pandas as pd

# -----------------------------
# Cache helpers (pickle)
# -----------------------------
def _cache_paths(cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)
    return {
        "scenarios_delta": os.path.join(cache_dir, "scenarios_delta.pkl"),
        "scenarios_result": os.path.join(cache_dir, "scenarios_result.pkl"),
    }

def _pickle_save(obj: Any, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)

def _pickle_load(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)

def save_cache(
    *,
    scenarios_delta: Dict[int, "ScenarioDelta"],
    scenarios_result: Dict[int, dict],
    cache_dir: str = "cache_delta",
) -> dict:
    """
    Saves:
      - scenarios_result: directly
      - scenarios_delta: only the minimum needed (final_cost_df_cf + stats), so it's robust.
    """
    paths = _cache_paths(cache_dir)

    delta_pack = {}
    for k, scn in scenarios_delta.items():
        delta_pack[k] = {
            "delta_pp": scn.delta_pp,
            "final_cost_df_cf": scn.final_cost_df_cf,
            "stats_turbine": scn.stats_turbine,
            "stats_farm": scn.stats_farm,
        }

    _pickle_save(delta_pack, paths["scenarios_delta"])
    _pickle_save(scenarios_result, paths["scenarios_result"])

    return {"ok": True, "paths": paths}

def load_cache(
    *,
    cache_dir: str = "cache_delta",
) -> tuple[Dict[int, "ScenarioDelta"], Dict[int, dict]]:
    paths = _cache_paths(cache_dir)

    delta_pack = _pickle_load(paths["scenarios_delta"])
    scenarios_result = _pickle_load(paths["scenarios_result"])

    scenarios_delta = {}
    for k, d in delta_pack.items():
        scenarios_delta[k] = ScenarioDelta(
            delta_pp=d["delta_pp"],
            final_cost_df_cf=d["final_cost_df_cf"],
            stats_turbine=d["stats_turbine"],
            stats_farm=d["stats_farm"],
        )

    return scenarios_delta, scenarios_result

def _canonical_cache_dir_name(cache_dir: str) -> str:
    """
    Keep scenario cache names stable even if old notebook cells pass a wage hash
    suffix such as Europe_resilience_wages_88194cd2.
    """
    cache_dir = str(cache_dir)
    parent, name = os.path.split(cache_dir)
    name2 = re.sub(r"_wages_[0-9a-fA-F]{8}$", "", name)
    return os.path.join(parent, name2) if parent else name2


def _resolve_cache_dir(cache_dir: str, base_dir=None) -> str:
    cache_dir = _canonical_cache_dir_name(cache_dir)
    if base_dir is None:
        return str(cache_dir)
    return os.path.join(str(base_dir), str(cache_dir))

def maybe_run_or_load(
    flows_by_component_hier_all,
    df_alloc,
    ind_root: str | None = None,
    ROOT: str | None = None,
    cache_dir: str = "cache_delta",
    cache_base_dir=None,                    
    force_recompute: bool = False,
    verbose: bool = True,
    EUROPE=None,
    deltas_pp=range(-5, 6),
    L1_MRIO=None, VA_MRIO=None, LB_MRIO=None, W_MRIO=None,
    core=None,
    mrio=None,
    receiver_policy: str = "baseline_mix",
    capacity_map: dict | None = None,
    capacity_from_results: bool = False,
    results_by_origin_lvl: dict | None = None,
    results_by_origin_sub_lvl: dict | None = None,
    capacity_lvl: int | None = None,
    capacity_components: list[str] | tuple[str, ...] | None = None,
    capacity_targets: list[str] | tuple[str, ...] | dict | None = None,
    capacity_include_row: bool = False,
    capacity_year: int = 2023,
    capacity_sector: str = "ALL",
    capacity_gva_col: str = "GVA_Total",
    EU27=(),
    connected_country: dict | None = None,
    weight_key: str = "Power (MW)",
    single_receiver_country: str | None = None,
    single_donor_country: str | None = None,
    ind_ind: bool | None = None,
    component_filter_farm: set[str] | list[str] | None = None,
    component_filter_turbine: set[str] | list[str] | None = None,
    receiver_whitelist: set[str] | list[str] | None = None,
    donor_whitelist: set[str] | list[str] | None = None,
    eu_spend_reduction_pct: float = 0.0,
    preserve_total: bool = True,
    annualize: bool = True,
    lifetime_years: int = 25,
    **kwargs,
):
    # Backwards/forwards compatibility for renamed params
    if "mrio" in kwargs and mrio is None:
        mrio = kwargs.pop("mrio")
    if "core" in kwargs and core is None:
        core = kwargs.pop("core")

    if ind_root is None:
        if ROOT is not None:
            ind_root = str(ROOT)
        else:
            ind_root = str(HELPERS_DIR)

    cache_dir = _resolve_cache_dir(cache_dir, cache_base_dir)   # NEW
    paths = _cache_paths(cache_dir)
    have_cache = os.path.exists(paths["scenarios_delta"]) and os.path.exists(paths["scenarios_result"])

    if have_cache and not force_recompute:
        if verbose:
            print(f"[maybe_run_or_load] Loading cache: {cache_dir}")
        scenarios_delta, scenarios_result = load_cache(cache_dir=cache_dir)
        return scenarios_delta, scenarios_result

    if verbose:
        reason = "force_recompute=True" if force_recompute else "cache files missing"
        print(f"[maybe_run_or_load] Recomputing ({reason}): {cache_dir}")
        print(f"[maybe_run_or_load] scenarios_delta exists: {os.path.exists(paths['scenarios_delta'])}")
        print(f"[maybe_run_or_load] scenarios_result exists: {os.path.exists(paths['scenarios_result'])}")

    # Auto-detect industry code prefix to match MRIO (CPA_ vs bare codes)
    if ind_ind is None:
        if mrio is not None:
            try:
                lvl = pd.Index(mrio.index.get_level_values(1)).astype(str)
                share_cpa = float((lvl.str.startswith("CPA_")).mean()) if len(lvl) else 0.0
                ind_ind = (share_cpa < 0.5)
            except Exception:
                ind_ind = False
        else:
            ind_ind = False

    # Build GVA share maps (category-level) from observed results
    gva_share_map = None
    gva_share_map_bucket = None
    if (results_by_origin_sub_lvl is not None or results_by_origin_lvl is not None) and capacity_lvl is not None:
        try:
            comps = set()
            if component_filter_turbine:
                comps.update({str(x).strip() for x in component_filter_turbine})
            if component_filter_farm:
                comps.update({str(x).strip() for x in component_filter_farm})
            if capacity_components is not None:
                comps = {str(x).strip() for x in capacity_components}
            if not comps:
                comps = set(results_by_origin_lvl.get(capacity_lvl, {}).keys())

            comps_sub = set(comps)
            # Always include Turbine aggregate if any turbine blocks are used
            if any(str(x).strip().casefold() in {"turbine", "turbines"} for x in comps) or (comps & TURBINE_BLOCKS):
                comps.add("Turbine")
                # Ensure sub-level map includes turbine blocks (e.g., Drivetrain)
                comps_sub.update(TURBINE_BLOCKS)

            if results_by_origin_sub_lvl is not None:
                gva_share_map = build_gva_share_map_sub(
                    results_by_origin_sub_lvl,
                    capacity_lvl,
                    EU27=EU27,
                    components=sorted(comps_sub),
                    year=capacity_year,
                    sector=capacity_sector,
                    connected_country=connected_country,
                    weight_key=weight_key,
                    gva_col=capacity_gva_col,
                )
            else:
                gva_share_map = build_gva_share_map(
                    results_by_origin_lvl,
                    capacity_lvl,
                    EU27=EU27,
                    components=sorted(comps),
                    year=capacity_year,
                    sector=capacity_sector,
                    connected_country=connected_country,
                    weight_key=weight_key,
                    gva_col=capacity_gva_col,
                )
        except Exception:
            gva_share_map = None

    # Build bucket-level share maps (deployment × OEM × component_group × sub_component × cost_block)
    try:
        gva_share_map_bucket = build_gva_share_map_bucket_df_alloc(
            df_alloc,
            EU27=EU27,
            exclude_origin=("EXPORTROW",),
            gva_col=capacity_gva_col,
        )
    except Exception:
        gva_share_map_bucket = None
    # Add bucket-level maps for farm flows (deployment × parent × child)
    try:
        farm_bucket_map = build_gva_share_map_bucket_flows(
            flows_by_component_hier_all,
            EU27=EU27,
            gva_col=capacity_gva_col,
        )
        if gva_share_map_bucket is None:
            gva_share_map_bucket = farm_bucket_map
        else:
            gva_share_map_bucket.update(farm_bucket_map)
    except Exception:
        pass

    scenarios_delta = build_delta_scenarios(
        group_set=EUROPE,
        df_alloc=df_alloc,
        flows_by_component_hier_all=flows_by_component_hier_all,
        ind_root=ind_root,
        deltas_pp=deltas_pp,
        receiver_policy=receiver_policy,
        gva_share_map=gva_share_map,
        gva_share_map_bucket=gva_share_map_bucket,
        single_receiver_country=single_receiver_country,
        single_donor_country=single_donor_country,
        ind_ind=bool(ind_ind),
        component_filter_farm=component_filter_farm,
        component_filter_turbine=component_filter_turbine,
        receiver_whitelist=receiver_whitelist,
        donor_whitelist=donor_whitelist,
        eu_spend_reduction_pct=eu_spend_reduction_pct,
        preserve_total=preserve_total,
    )
    scenarios_result = run_mrio_for_scenarios(
        scenarios_delta,
        L1_MRIO=L1_MRIO, VA_MRIO=VA_MRIO, LB_MRIO=LB_MRIO, W_MRIO=W_MRIO,
        core=core,
        mrio=mrio,
        annualize=annualize,
        lifetime_years=lifetime_years,
    )

    save_cache(
        scenarios_delta=scenarios_delta,
        scenarios_result=scenarios_result,
        cache_dir=cache_dir,
    )
    return scenarios_delta, scenarios_result




def build_gross_expansion_scenarios_from_net(
    net_scenarios_delta: Dict[int, "ScenarioDelta"],
    *,
    deltas_pp: list[int] | tuple[int, ...] | None = None,
    value_col: str = "value_alloc",
) -> Dict[int, "ScenarioDelta"]:
    """
    Build gross expansion scenarios from an existing net reallocation scenario.

    The net scenario combines additions to receiving origins and reductions from
    donor origins. The gross expansion scenario keeps the baseline shock and adds
    only positive row-level changes relative to delta=0. This isolates the MRIO
    effect of additional procurement assigned to receiving suppliers without
    subtracting the activity removed from offshore suppliers.
    """
    if 0 not in net_scenarios_delta:
        raise KeyError("net_scenarios_delta must contain delta 0 as baseline.")

    if deltas_pp is None:
        deltas_pp = tuple(sorted(int(k) for k in net_scenarios_delta.keys()))
    else:
        deltas_pp = tuple(int(k) for k in deltas_pp)

    base_df = net_scenarios_delta[0].final_cost_df_cf.copy()
    if value_col not in base_df.columns:
        raise KeyError(f"Missing value column {value_col!r} in baseline final_cost_df.")

    def _gross_df(target_df: pd.DataFrame) -> pd.DataFrame:
        if value_col not in target_df.columns:
            raise KeyError(f"Missing value column {value_col!r} in target final_cost_df.")

        common_cols = [c for c in base_df.columns if c in target_df.columns]
        key_cols = [c for c in common_cols if c != value_col]
        if not key_cols:
            raise ValueError("Could not infer non-value key columns for final_cost_df.")

        base_g = (
            base_df[key_cols + [value_col]]
            .groupby(key_cols, dropna=False, as_index=False)[value_col]
            .sum()
            .rename(columns={value_col: "_base_value"})
        )
        targ_g = (
            target_df[key_cols + [value_col]]
            .groupby(key_cols, dropna=False, as_index=False)[value_col]
            .sum()
            .rename(columns={value_col: "_target_value"})
        )

        merged = base_g.merge(targ_g, on=key_cols, how="outer")
        merged[["_base_value", "_target_value"]] = merged[["_base_value", "_target_value"]].fillna(0.0)
        positive_delta = (merged["_target_value"] - merged["_base_value"]).clip(lower=0.0)
        merged[value_col] = merged["_base_value"] + positive_delta
        merged = merged[merged[value_col].abs() > 1e-12].copy()
        return merged[key_cols + [value_col]].copy()

    out: Dict[int, ScenarioDelta] = {}
    for d in deltas_pp:
        if d not in net_scenarios_delta:
            raise KeyError(f"delta {d} not found in net_scenarios_delta.")
        net_scn = net_scenarios_delta[d]
        gross_df = base_df.copy() if int(d) == 0 else _gross_df(net_scn.final_cost_df_cf)
        out[int(d)] = ScenarioDelta(
            delta_pp=net_scn.delta_pp,
            final_cost_df_cf=gross_df,
            stats_turbine={
                **(net_scn.stats_turbine or {}),
                "gross_expansion_only": True,
                "source_delta_pp": int(d),
            },
            stats_farm={
                **(net_scn.stats_farm or {}),
                "gross_expansion_only": True,
                "source_delta_pp": int(d),
            },
        )
    return out


def build_eu27_side_scenarios_from_net(
    net_scenarios_delta: Dict[int, "ScenarioDelta"],
    *,
    eu_countries,
    deltas_pp: list[int] | tuple[int, ...] | None = None,
    value_col: str = "value_alloc",
    origin_col: str = "origin_country",
) -> Dict[int, "ScenarioDelta"]:
    """
    Build scenarios that isolate the EU27 side of the signed reallocation.

    Non-EU27 origins are kept at baseline. EU27-origin rows take their
    counterfactual value, so reshoring is a positive EU27 expansion and
    offshoring is a negative EU27 contraction.
    """
    if 0 not in net_scenarios_delta:
        raise KeyError("net_scenarios_delta must contain delta 0 as baseline.")

    if deltas_pp is None:
        deltas_pp = tuple(sorted(int(k) for k in net_scenarios_delta.keys()))
    else:
        deltas_pp = tuple(int(k) for k in deltas_pp)

    eu_set = {str(x).strip().upper().replace("UK", "GB") for x in (eu_countries or [])}
    if not eu_set:
        raise ValueError("eu_countries must contain at least one country code.")

    base_df = net_scenarios_delta[0].final_cost_df_cf.copy()
    if value_col not in base_df.columns:
        raise KeyError(f"Missing value column {value_col!r} in baseline final_cost_df.")
    if origin_col not in base_df.columns:
        raise KeyError(f"Missing origin column {origin_col!r} in baseline final_cost_df.")

    def _eu27_side_df(target_df: pd.DataFrame) -> pd.DataFrame:
        if value_col not in target_df.columns:
            raise KeyError(f"Missing value column {value_col!r} in target final_cost_df.")
        if origin_col not in target_df.columns:
            raise KeyError(f"Missing origin column {origin_col!r} in target final_cost_df.")

        common_cols = [c for c in base_df.columns if c in target_df.columns]
        key_cols = [c for c in common_cols if c != value_col]
        if origin_col not in key_cols:
            raise KeyError(f"Origin column {origin_col!r} is not available among final_cost_df keys.")

        base_g = (
            base_df[key_cols + [value_col]]
            .groupby(key_cols, dropna=False, as_index=False)[value_col]
            .sum()
            .rename(columns={value_col: "_base_value"})
        )
        targ_g = (
            target_df[key_cols + [value_col]]
            .groupby(key_cols, dropna=False, as_index=False)[value_col]
            .sum()
            .rename(columns={value_col: "_target_value"})
        )

        merged = base_g.merge(targ_g, on=key_cols, how="outer")
        merged[["_base_value", "_target_value"]] = merged[["_base_value", "_target_value"]].fillna(0.0)
        origin_norm = merged[origin_col].astype(str).str.strip().str.upper().replace({"UK": "GB"})
        is_eu = origin_norm.isin(eu_set)
        merged[value_col] = np.where(is_eu, merged["_target_value"], merged["_base_value"])
        merged = merged[merged[value_col].abs() > 1e-12].copy()
        return merged[key_cols + [value_col]].copy()

    out: Dict[int, ScenarioDelta] = {}
    for d in deltas_pp:
        if d not in net_scenarios_delta:
            raise KeyError(f"delta {d} not found in net_scenarios_delta.")
        net_scn = net_scenarios_delta[d]
        eu_side_df = base_df.copy() if int(d) == 0 else _eu27_side_df(net_scn.final_cost_df_cf)
        out[int(d)] = ScenarioDelta(
            delta_pp=net_scn.delta_pp,
            final_cost_df_cf=eu_side_df,
            stats_turbine={
                **(net_scn.stats_turbine or {}),
                "eu27_side_reallocation_only": True,
                "source_delta_pp": int(d),
            },
            stats_farm={
                **(net_scn.stats_farm or {}),
                "eu27_side_reallocation_only": True,
                "source_delta_pp": int(d),
            },
        )
    return out


def maybe_run_gross_expansion_from_net(
    *,
    net_scenarios_delta: Dict[int, "ScenarioDelta"],
    cache_dir: str = "gross_expansion",
    cache_base_dir=None,
    force_recompute: bool = False,
    deltas_pp: list[int] | tuple[int, ...] = (0, 10),
    L1_MRIO=None,
    VA_MRIO=None,
    LB_MRIO=None,
    W_MRIO=None,
    core=None,
    mrio=None,
    annualize: bool = False,
    lifetime_years: int = 25,
    verbose: bool = True,
):
    # Build/load gross expansion MRIO results derived from net scenarios.
    cache_dir = _resolve_cache_dir(cache_dir, cache_base_dir)
    paths = _cache_paths(cache_dir)
    have_cache = os.path.exists(paths["scenarios_delta"]) and os.path.exists(paths["scenarios_result"])

    if have_cache and not force_recompute:
        if verbose:
            print(f"[maybe_run_gross_expansion_from_net] Loading cache: {cache_dir}")
        return load_cache(cache_dir=cache_dir)

    if verbose:
        reason = "force_recompute=True" if force_recompute else "cache files missing"
        print(f"[maybe_run_gross_expansion_from_net] Recomputing ({reason}): {cache_dir}")

    gross_delta = build_gross_expansion_scenarios_from_net(
        net_scenarios_delta,
        deltas_pp=deltas_pp,
    )
    gross_result = run_mrio_for_scenarios(
        gross_delta,
        L1_MRIO=L1_MRIO,
        VA_MRIO=VA_MRIO,
        LB_MRIO=LB_MRIO,
        W_MRIO=W_MRIO,
        core=core,
        mrio=mrio,
        annualize=annualize,
        lifetime_years=lifetime_years,
    )
    save_cache(
        scenarios_delta=gross_delta,
        scenarios_result=gross_result,
        cache_dir=cache_dir,
    )
    return gross_delta, gross_result


def maybe_run_eu27_side_from_net(
    *,
    net_scenarios_delta: Dict[int, "ScenarioDelta"],
    eu_countries,
    cache_dir: str = "eu27_side_reallocation",
    cache_base_dir=None,
    force_recompute: bool = False,
    deltas_pp: list[int] | tuple[int, ...] = (0, 10),
    L1_MRIO=None,
    VA_MRIO=None,
    LB_MRIO=None,
    W_MRIO=None,
    core=None,
    mrio=None,
    annualize: bool = False,
    lifetime_years: int = 25,
    verbose: bool = False,
):
    # Build/load MRIO results that isolate the EU27 side of net reallocation.
    cache_dir = _resolve_cache_dir(cache_dir, cache_base_dir)
    paths = _cache_paths(cache_dir)
    have_cache = os.path.exists(paths["scenarios_delta"]) and os.path.exists(paths["scenarios_result"])

    if have_cache and not force_recompute:
        if verbose:
            print(f"[maybe_run_eu27_side_from_net] Loading cache: {cache_dir}")
        return load_cache(cache_dir=cache_dir)

    if verbose:
        reason = "force_recompute=True" if force_recompute else "cache files missing"
        print(f"[maybe_run_eu27_side_from_net] Recomputing ({reason}): {cache_dir}")

    eu_side_delta = build_eu27_side_scenarios_from_net(
        net_scenarios_delta,
        eu_countries=eu_countries,
        deltas_pp=deltas_pp,
    )
    eu_side_result = run_mrio_for_scenarios(
        eu_side_delta,
        L1_MRIO=L1_MRIO,
        VA_MRIO=VA_MRIO,
        LB_MRIO=LB_MRIO,
        W_MRIO=W_MRIO,
        core=core,
        mrio=mrio,
        annualize=annualize,
        lifetime_years=lifetime_years,
    )
    save_cache(
        scenarios_delta=eu_side_delta,
        scenarios_result=eu_side_result,
        cache_dir=cache_dir,
    )
    return eu_side_delta, eu_side_result


def build_panels_origin_metrics(
    results_by_origin_sub_lvl: dict,
    *,
    lvl_plot: int = 1,
    year: int = 2023,
    sector: str = "ALL",
    EU27=(),
    connected_country: dict | None = None,
    weight_key: str = "Power (MW)",
    exclude_deployments=("GB",),
    exclude_keys=(("Total", "ALL"),),
    dep_col: str = "Deployment",
    org_col: str = "Country",
    # metric columns expected in origin tables
    col_output_direct: str = "Output_Direct",
    col_jobs_total: str = "Jobs_Total",
    col_gva_total: str = "GVA_Total",
    # (optional: keep in absolutes; shares are only for the 3 above)
    col_jobs_direct: str = "Jobs_Direct",
    col_gva_direct: str = "GVA_Direct",
    # Split selected non-European countries into their own regions; everything else remains RoW.
    split_row=(),  # e.g. ("CN", "KR", "IN", "ZA")
    split_row_labels: dict | None = None,
    # NEW: override output_direct with baseline values (for counterfactual consistency)
    output_direct_override: pd.DataFrame | None = None,
):
    """
    Returns TWO dataframes:
      1) panel_sub: one row per (Component, Subcomponent, Region)
         with weighted-average ABSOLUTES (GB excluded from averaging, Total/ALL excluded)
         + shares within each (Component, Subcomponent) across regions.

      2) panel_comp: one row per (Component, Region)
         computed by summing panel_sub absolutes over Subcomponent
         + shares within each Component across regions.

    Shares (only three):
      share_sub_output_direct, share_sub_jobs_total, share_sub_gva_total
    """

    EU27 = {str(x).strip().upper() for x in EU27}
    exclude_deployments = {str(x).strip().upper() for x in exclude_deployments}
    exclude_keys = set(exclude_keys)

    split_row = {str(x).strip().upper() for x in split_row}
    split_row_labels = {
        code: ("China" if code == "CN" else CODE_TO_NAME.get(code, code))
        for code in split_row
    } | (split_row_labels or {})

    abs_cols_src = {
        "abs_output_direct": col_output_direct,
        "abs_jobs_direct": col_jobs_direct,
        "abs_jobs_total": col_jobs_total,
        "abs_gva_direct": col_gva_direct,
        "abs_gva_total": col_gva_total,
    }

    def _normalize_component(name: str) -> str:
        s = str(name).strip()
        s_cf = s.casefold()
        alias = {
            "balance of plant": "BoP",
            "balance of plant (bop)": "BoP",
            "bop": "BoP",
        }
        return alias.get(s_cf, s)

    def bucket_region(country: str) -> str:
        c = str(country).strip().upper()
        if c in {"GB", "UK"}:
            return UK_LABEL
        if c in EU27:
            return EU_LABEL
        if c in split_row:
            return split_row_labels.get(c, c)
        if c in EUROPE_CODES:
            return ROE_LABEL
        return ROW_LABEL

    def deployment_weight(dep: str) -> float:
        if connected_country is None:
            return 1.0
        d = str(dep).strip().upper()
        return float(connected_country.get(d, {}).get(weight_key, 0.0))

    rows = []
    src = results_by_origin_sub_lvl[lvl_plot]
    skip_self = _self_aggregate_keys_to_skip(src.keys())

    # Precompute union of deployments per component (after filters)
    dep_list_by_comp = {}
    for key, df in src.items():
        if key in exclude_keys or _is_aggregate_component_key(key) or key in skip_self:
            continue
        if not (isinstance(key, tuple) and len(key) >= 2):
            continue

        comp = _normalize_component(key[0])

        d0 = df.copy()
        if "Year" in d0.columns:
            d0 = d0[d0["Year"] == year]
        if "Sector" in d0.columns:
            d0 = d0[d0["Sector"].astype(str) == str(sector)]
        if d0.empty:
            continue

        deps = d0[dep_col].astype(str).str.strip().str.upper().unique().tolist()
        dep_list_by_comp.setdefault(comp, set()).update(deps)

    for key, df in src.items():
        if key in exclude_keys or _is_aggregate_component_key(key) or key in skip_self:
            continue
        if not (isinstance(key, tuple) and len(key) >= 2):
            continue

        component, subcomponent = _normalize_component(key[0]), key[1]

        d = df.copy()
        if "Year" in d.columns:
            d = d[d["Year"] == year]
        if "Sector" in d.columns:
            d = d[d["Sector"].astype(str) == str(sector)]

        d[dep_col] = d[dep_col].astype(str).str.strip().str.upper()
        d[org_col] = d[org_col].astype(str).str.strip().str.upper()
        d["Region"] = d[org_col].map(bucket_region)

        deps_all = dep_list_by_comp.get(component, set())
        if not deps_all:
            deps_all = set(d[dep_col].unique())

        dep_list = [x for x in sorted(deps_all) if x not in exclude_deployments]
        if not dep_list:
            dep_list = list(sorted(deps_all))

        w_map = {dep: deployment_weight(dep) for dep in dep_list}
        w_sum = builtins.sum(w_map.values()) or 1.0

        # Dynamic regions: EU27, UK, RoE, optional selected countries, residual RoW.
        # Explicit split countries are removed from RoE/RoW by bucket_region above.
        region_list = [EU_LABEL, UK_LABEL, ROE_LABEL]
        base_regions = set(region_list + [ROW_LABEL])
        region_list += sorted([
            split_row_labels.get(x, x)
            for x in split_row
            if split_row_labels.get(x, x) not in base_regions
        ])
        region_list += [ROW_LABEL]
        acc = {r: {k: 0.0 for k in abs_cols_src} for r in region_list}

        for dep, wd in w_map.items():
            w = wd / w_sum
            dd = d[d[dep_col] == dep]
            if dd.empty:
                continue

            g = dd.groupby("Region", as_index=False)[list(abs_cols_src.values())].sum()
            g = g.rename(columns={v: k for k, v in abs_cols_src.items()})

            for region in region_list:
                rr = g[g["Region"] == region]
                if rr.empty:
                    continue
                for k in abs_cols_src:
                    acc[region][k] += w * float(rr[k].iloc[0])

        for region in region_list:
            rows.append({
                "Component": component,
                "Subcomponent": subcomponent,
                "Region": region,
                **{k: float(acc[region][k]) for k in abs_cols_src},
            })

    panel_sub = pd.DataFrame(rows)

    # Apply output_direct_override if provided (for counterfactual consistency)
    if output_direct_override is not None:
        panel_sub = panel_sub.merge(
            output_direct_override[["Component", "Subcomponent", "Region", "abs_output_direct"]],
            on=["Component", "Subcomponent", "Region"],
            how="left",
            suffixes=("_scenario", "_baseline")
        )
        panel_sub["abs_output_direct"] = panel_sub["abs_output_direct_baseline"].fillna(panel_sub["abs_output_direct_scenario"])
        panel_sub = panel_sub.drop(columns=["abs_output_direct_scenario", "abs_output_direct_baseline"])

    den_sub = (
        panel_sub.groupby(["Component", "Subcomponent"], as_index=False)[
            ["abs_output_direct", "abs_jobs_total", "abs_gva_total"]
        ].sum()
        .rename(columns={
            "abs_output_direct": "_den_output_direct",
            "abs_jobs_total": "_den_jobs_total",
            "abs_gva_total": "_den_gva_total",
        })
    )
    panel_sub = panel_sub.merge(den_sub, on=["Component", "Subcomponent"], how="left")

    panel_sub["share_sub_output_direct"] = panel_sub["abs_output_direct"] / panel_sub["_den_output_direct"].replace(0.0, pd.NA)
    panel_sub["share_sub_jobs_total"]    = panel_sub["abs_jobs_total"]    / panel_sub["_den_jobs_total"].replace(0.0, pd.NA)
    panel_sub["share_sub_gva_total"]     = panel_sub["abs_gva_total"]     / panel_sub["_den_gva_total"].replace(0.0, pd.NA)

    panel_sub[["share_sub_output_direct", "share_sub_jobs_total", "share_sub_gva_total"]] = (
        panel_sub[["share_sub_output_direct", "share_sub_jobs_total", "share_sub_gva_total"]].fillna(0.0)
    )
    panel_sub = panel_sub.drop(columns=["_den_output_direct", "_den_jobs_total", "_den_gva_total"])

    for c in ["share_world_output_direct","share_world_jobs_total","share_world_gva_total",
              "share_output_direct","share_jobs_total","share_gva_total"]:
        if c in panel_sub.columns:
            panel_sub = panel_sub.drop(columns=[c])

    panel_comp = (
        panel_sub.groupby(["Component", "Region"], as_index=False)[
            list(abs_cols_src.keys())
        ].sum()
    )

    den_comp = (
        panel_comp.groupby(["Component"], as_index=False)[
            ["abs_output_direct", "abs_jobs_total", "abs_gva_total"]
        ].sum()
        .rename(columns={
            "abs_output_direct": "_den_output_direct",
            "abs_jobs_total": "_den_jobs_total",
            "abs_gva_total": "_den_gva_total",
        })
    )
    panel_comp = panel_comp.merge(den_comp, on=["Component"], how="left")

    panel_comp["share_sub_output_direct"] = panel_comp["abs_output_direct"] / panel_comp["_den_output_direct"].replace(0.0, pd.NA)
    panel_comp["share_sub_jobs_total"]    = panel_comp["abs_jobs_total"]    / panel_comp["_den_jobs_total"].replace(0.0, pd.NA)
    panel_comp["share_sub_gva_total"]     = panel_comp["abs_gva_total"]     / panel_comp["_den_gva_total"].replace(0.0, pd.NA)

    panel_comp[["share_sub_output_direct", "share_sub_jobs_total", "share_sub_gva_total"]] = (
        panel_comp[["share_sub_output_direct", "share_sub_jobs_total", "share_sub_gva_total"]].fillna(0.0)
    )
    panel_comp = panel_comp.drop(columns=["_den_output_direct", "_den_jobs_total", "_den_gva_total"])

    return panel_sub, panel_comp


def build_baseline_panels(
    scenarios_result: dict,
    *,
    split_row=(),
    split_row_labels: dict | None = None,
    lvl_plot: int = 1,
    year: int = 2023,
    sector: str = "ALL",
    EU27=(),
    connected_country: dict | None = None,
    weight_key: str = "Power (MW)",
    exclude_deployments=("GB",),
    exclude_keys=(("Total", "ALL"),),
    scenario_name: str = "baseline",
    add_eu_indirect_split: bool = False,
    eu_split_labels: tuple[str, str] | None = None,
):
    """
    Build baseline panels (sub + comp) from scenarios_result dict (delta_pp -> result dict).
    """
    sub_list, comp_list = [], []

    eu_split_df = None
    eu_eu_label, eu_row_label = eu_split_labels or (f"{EU_LABEL} (from EU)", f"{EU_LABEL} (from RoW)")
    if add_eu_indirect_split:
        eu_split_df = split_eu_indirect_by_deployment(
            scenarios_result,
            EU27=EU27,
            lvl_plot=lvl_plot,
            year=year,
            sector=sector,
            connected_country=connected_country,
            weight_key=weight_key,
            exclude_deployments=exclude_deployments,
        )
    for delta_pp, res in scenarios_result.items():
        origin = res.get("results_by_origin_sub_lvl")
        if origin is None:
            continue

        panel_sub, panel_comp = build_panels_origin_metrics(
            origin,
            lvl_plot=lvl_plot,
            year=year,
            sector=sector,
            EU27=EU27,
            connected_country=connected_country,
            weight_key=weight_key,
            exclude_deployments=exclude_deployments,
            exclude_keys=exclude_keys,
            split_row=split_row,
            split_row_labels=split_row_labels,
        )
        panel_sub = panel_sub.assign(delta_pp=delta_pp, scenario=scenario_name)
        panel_comp = panel_comp.assign(delta_pp=delta_pp, scenario=scenario_name)

        if add_eu_indirect_split and eu_split_df is not None:
            sd = eu_split_df[eu_split_df["delta_pp"] == delta_pp].copy()
            if not sd.empty:
                eu_base = panel_sub.loc[panel_sub["Region"] == EU_LABEL, [
                    "Component", "Subcomponent",
                    "abs_output_direct", "abs_gva_direct", "abs_jobs_direct",
                ]].drop_duplicates(subset=["Component", "Subcomponent"])

                sd = sd.merge(eu_base, left_on=["component", "subcomponent"], right_on=["Component", "Subcomponent"], how="left")
                sd[["abs_output_direct", "abs_gva_direct", "abs_jobs_direct"]] = sd[
                    ["abs_output_direct", "abs_gva_direct", "abs_jobs_direct"]
                ].fillna(0.0)

                rows = []
                for _, r in sd.iterrows():
                    comp = r["component"]
                    subc = r["subcomponent"]
                    out_dir = float(r.get("abs_output_direct", 0.0))
                    gva_dir = float(r.get("abs_gva_direct", 0.0))
                    jobs_dir = float(r.get("abs_jobs_direct", 0.0))

                    out_ie = float(r.get("output_indirect_eu_from_eu", 0.0))
                    out_ir = float(r.get("output_indirect_eu_from_row", 0.0))
                    gva_ie = float(r.get("gva_indirect_eu_from_eu", 0.0))
                    gva_ir = float(r.get("gva_indirect_eu_from_row", 0.0))
                    jobs_ie = float(r.get("jobs_indirect_eu_from_eu", 0.0))
                    jobs_ir = float(r.get("jobs_indirect_eu_from_row", 0.0))

                    rows.append({
                        "Component": comp,
                        "Subcomponent": subc,
                        "Region": eu_eu_label,
                        "abs_output_direct": out_dir,
                        "abs_output_total": out_dir + out_ie,
                        "abs_gva_direct": gva_dir,
                        "abs_gva_total": gva_dir + gva_ie,
                        "abs_jobs_direct": jobs_dir,
                        "abs_jobs_total": jobs_dir + jobs_ie,
                    })
                    rows.append({
                        "Component": comp,
                        "Subcomponent": subc,
                        "Region": eu_row_label,
                        "abs_output_direct": 0.0,
                        "abs_output_total": out_ir,
                        "abs_gva_direct": 0.0,
                        "abs_gva_total": gva_ir,
                        "abs_jobs_direct": 0.0,
                        "abs_jobs_total": jobs_ir,
                    })

                split_sub = pd.DataFrame(rows)
                if not split_sub.empty:
                    split_sub = split_sub.assign(delta_pp=delta_pp, scenario=scenario_name)
                    share_cols = [c for c in panel_sub.columns if c.startswith("share_")]
                    for c in share_cols:
                        split_sub[c] = 0.0
                    panel_sub = pd.concat([panel_sub, split_sub], ignore_index=True)

                    abs_cols = [c for c in panel_sub.columns if c.startswith("abs_")]
                    split_comp = (
                        split_sub.groupby(["Component", "Region"], as_index=False)[abs_cols].sum()
                    )
                    split_comp = split_comp.assign(delta_pp=delta_pp, scenario=scenario_name)
                    share_cols_comp = [c for c in panel_comp.columns if c.startswith("share_")]
                    for c in share_cols_comp:
                        split_comp[c] = 0.0
                    panel_comp = pd.concat([panel_comp, split_comp], ignore_index=True)
        sub_list.append(panel_sub)
        comp_list.append(panel_comp)

    PANEL_SUB = pd.concat(sub_list, ignore_index=True) if sub_list else pd.DataFrame()
    PANEL_COMP = pd.concat(comp_list, ignore_index=True) if comp_list else pd.DataFrame()
    return PANEL_SUB, PANEL_COMP


def split_eu_indirect_by_deployment(
    scenarios_result: dict,
    *,
    EU27=(),
    lvl_plot: int = 1,
    year: int = 2023,
    sector: str = "ALL",
    connected_country: dict | None = None,
    weight_key: str = "Power (MW)",
    exclude_deployments=(),
) -> pd.DataFrame:
    """
    Split EU27 indirect effects by where the final-demand shock is deployed (EU vs RoW).
    Returns a DataFrame with EU-origin indirect GVA/Jobs split by deployment region.
    """
    EU27 = {str(x).strip().upper() for x in EU27}

    def _normalize_component(name: str) -> str:
        s = str(name).strip()
        s_cf = s.casefold()
        alias = {
            "balance of plant": "BoP",
            "balance of plant (bop)": "BoP",
            "bop": "BoP",
        }
        return alias.get(s_cf, s)

    def _dep_weight(dep: str) -> float:
        if connected_country is None:
            return 1.0
        d = str(dep).strip().upper()
        return float(connected_country.get(d, {}).get(weight_key, 0.0))

    rows = []
    for delta_pp, res in scenarios_result.items():
        origin = res.get("results_by_origin_sub_lvl", {}).get(lvl_plot, {})
        skip_self = _self_aggregate_keys_to_skip(origin.keys())
        for key, df in origin.items():
            if _is_aggregate_component_key(key) or key in skip_self:
                continue
            if not (isinstance(key, tuple) and len(key) >= 2):
                continue
            d = df.copy()
            if "Year" in d.columns:
                d = d[d["Year"] == year]
            if "Sector" in d.columns:
                d = d[d["Sector"].astype(str) == str(sector)]

            dep_all = sorted(d["Deployment"].astype(str).str.strip().str.upper().unique())
            dep_excl = {str(x).strip().upper() for x in (exclude_deployments or [])}
            dep_list = [x for x in dep_all if x not in dep_excl]
            if not dep_list:
                dep_list = dep_all

            w_map = {dep: _dep_weight(dep) for dep in dep_list}
            w_sum = float(sum(w_map.values())) or 1.0
            w_norm = {dep: float(w_map[dep]) / w_sum for dep in dep_list}

            gva_eu_from_eu = 0.0
            gva_eu_from_row = 0.0
            jobs_eu_from_eu = 0.0
            jobs_eu_from_row = 0.0
            out_eu_from_eu = 0.0
            out_eu_from_row = 0.0

            for dep, w in w_norm.items():
                dep_u = str(dep).strip().upper()
                dep_is_eu = dep_u in EU27

                sub = d[
                    (d["Deployment"].astype(str).str.strip().str.upper() == dep_u) &
                    (d["Country"].astype(str).str.strip().str.upper().isin(EU27))
                ]
                if sub.empty:
                    continue
                gva_ind = float(sub["GVA_Indirect"].sum())
                jobs_ind = float(sub["Jobs_Indirect"].sum())
                out_ind = float(sub["Output_Indirect"].sum())

                if dep_is_eu:
                    gva_eu_from_eu += w * gva_ind
                    jobs_eu_from_eu += w * jobs_ind
                    out_eu_from_eu += w * out_ind
                else:
                    gva_eu_from_row += w * gva_ind
                    jobs_eu_from_row += w * jobs_ind
                    out_eu_from_row += w * out_ind

            rows.append({
                "delta_pp": delta_pp,
                "component": _normalize_component(key[0]),
                "subcomponent": str(key[1]).strip(),
                "gva_indirect_eu_from_eu": gva_eu_from_eu,
                "gva_indirect_eu_from_row": gva_eu_from_row,
                "jobs_indirect_eu_from_eu": jobs_eu_from_eu,
                "jobs_indirect_eu_from_row": jobs_eu_from_row,
                "output_indirect_eu_from_eu": out_eu_from_eu,
                "output_indirect_eu_from_row": out_eu_from_row,
            })

    return pd.DataFrame(rows)

def mu_sensitivity(
    *,
    mu0,
    eps: float = 0.01,
    deltas: tuple[int, int] | list[int] = (-10, 10),
    base_kwargs: dict | None = None,
    cache_prefix: str = "sens_mu",
    force_recompute: bool = False,
    components: list[str] | tuple[str, ...] | None = None,
    subcomponents: list[str] | tuple[str, ...] | None = None,
    year: int = 2023,
    sector: str = "ALL",
    lvl_plot: int = 1,
):
    """
    Sensibilidad de E (EU27) respecto a mu, usando diferencias centrales.
    Devuelve:
      - raw: valores E para mu0-eps y mu0+eps
      - sens: derivadas dE/dmu por delta
    """
    if base_kwargs is None:
        raise ValueError("base_kwargs es obligatorio.")

    mu_lo = float(mu0) - float(eps)
    mu_hi = float(mu0) + float(eps)
    if not (0.0 <= mu_lo < 1.0 and 0.0 <= mu_hi < 1.0):
        raise ValueError("mu0±eps debe estar en [0,1). Ajusta mu0 o eps.")

    def _run(mu: float) -> pd.DataFrame:
        cache_dir = f"{cache_prefix}_mu_{mu:.4f}_d{min(deltas)}_{max(deltas)}".replace(".", "p")
        _, sr = maybe_run_or_load(
            **base_kwargs,
            deltas_pp=list(deltas),
            eu_spend_reduction_pct=float(mu),
            cache_dir=cache_dir,
            force_recompute=force_recompute,
        )

        panel_sub, _ = build_baseline_panels(
            sr,
            split_row=(),
            split_row_labels={},
            lvl_plot=lvl_plot,
            year=year,
            sector=sector,
            EU27=base_kwargs.get("EU27", ()),
            connected_country=base_kwargs.get("connected_country", None),
            weight_key=base_kwargs.get("weight_key", "Power (MW)"),
        )

        df = panel_sub.copy()
        if components:
            df = df[df["Component"].isin(components)]
        if subcomponents:
            df = df[df["Subcomponent"].isin(subcomponents)]

        df = df[df["Region"] == EU_LABEL]

        g = (
            df.groupby("delta_pp", as_index=False)
              .agg(
                  gva_direct=("abs_gva_direct", "sum"),
                  gva_total=("abs_gva_total", "sum"),
                  jobs_direct=("abs_jobs_direct", "sum"),
                  jobs_total=("abs_jobs_total", "sum"),
              )
        )
        g["gva_indirect"] = g["gva_total"] - g["gva_direct"]
        g["jobs_indirect"] = g["jobs_total"] - g["jobs_direct"]
        g["mu"] = float(mu)
        return g

    lo = _run(mu_lo)
    hi = _run(mu_hi)

    sens = lo.merge(hi, on="delta_pp", suffixes=("_lo", "_hi"))
    denom = (mu_hi - mu_lo)

    for m in ["gva_direct", "gva_indirect", "jobs_direct", "jobs_indirect"]:
        sens[f"d{m}_dmu"] = (sens[f"{m}_hi"] - sens[f"{m}_lo"]) / denom

    raw = pd.concat([lo, hi], ignore_index=True)
    return raw, sens
def plot_global_totals_by_region_direct_indirect(
    PANEL_SUB: pd.DataFrame,
    *,
    components: list[str] | tuple[str, ...] | None = None,
    scenario: str | None = "baseline",
    split_eu_indirect: bool = False,
    eu_indirect_split_df: pd.DataFrame | None = None,
    scenarios_result: dict | None = None,
    EU27=(),
    connected_country: dict | None = None,
    weight_key: str = "Power (MW)",
    exclude_deployments=("GB",),
    lvl_plot: int = 1,
    year: int = 2023,
    sector: str = "ALL",
    delta_col: str = "delta_pp",
    region_col: str = "Region",
    comp_col: str = "Component",
    scenario_col: str = "scenario",
    gva_direct_col: str = "abs_gva_direct",
    gva_total_col: str = "abs_gva_total",
    jobs_direct_col: str = "abs_jobs_direct",
    jobs_total_col: str = "abs_jobs_total",
    delta_min: int | None = -100,
    delta_max: int | None = 100,
    fig_w: float = 9.0,
    fig_h: float = 5.0,
    hspace: float = 0.10,
    wspace: float = 0.15,
    y_as_million: bool = True,
    x_tick_step: int = 20,
    label_step: int = 20,
    label_start_offset: int = 0,
    label_min_gva: float = 0.0,
    label_min_jobs: float = 0.0,
    label_min_gva_direct: float | None = None,
    label_min_gva_indirect: float | None = None,
    label_min_jobs_direct: float | None = None,
    label_min_jobs_indirect: float | None = None,
    label_at_zero: bool = True,
    show_total_labels: bool = False,
    show_value_labels: bool = True,
    value_format: str | None = "k",
    value_font: int = 8,
    legend_y: float = -1.01,
    legend_x: float = 0.50,
    legend_loc: str = "lower center",
    legend_title: str | None = None,
    legend_ncol: int | None = 8,
    edge_label_dx: float = 1.15,
    total_label_font: int | None = None,
    total_label_offset_px: float = 0.0,
    xtick_font: int = 8,
    ytick_font: int = 8,
    ytick_format: str | None = "k",
    legend_font: int = 9,
    legend_title_font: int = 9,
    ytitle_font: int = 9,
    alpha_fill: float = 0.85,
    show_edges: bool = False,
    show_total_line: bool = False,
    split_row=(),
    split_row_labels: dict | None = None,
    split_row_order: list[str] | tuple[str, ...] | None = None,
    show_non_eu_labels: bool = True,
    non_eu_label_min_abs_jobs: float = 0.0,
    non_eu_label_min_abs_gva: float = 0.0,
):
    """
    Baseline-style stacked area plot by region, split into direct/indirect effects.
    Produces 2x2 panels:
      - GVA direct / GVA indirect
      - Jobs direct / Jobs indirect
    A black line shows global total in each panel.
    """
    set_house_style()

    split_row = [str(x).strip().upper() for x in split_row]
    split_row_labels = {
        code: ("China" if code == "CN" else CODE_TO_NAME.get(code, code))
        for code in split_row
    } | (split_row_labels or {})
    split_regions_raw = [split_row_labels.get(x, x) for x in split_row]

    ROW_FC = COLOR_MAP.get("RoW") or COLOR_MAP.get(ROW_LABEL) or "#E6E6E6"
    EU_FC = COLOR_MAP.get("EU27") or COLOR_MAP.get(EU_LABEL) or "#4C78A8"
    ROE_FC = COLOR_MAP.get("EU_dep") or COLOR_MAP.get("RoE") or COLOR_MAP.get(ROE_LABEL) or "#4F6BD1"

    STYLE_MAP = {
        EU_LABEL: dict(
            fc=EU_FC,
            ec=EU_FC,
            ls="solid", lw=0.6
        ),
        UK_LABEL: dict(
            fc=COLOR_MAP.get("GB") or COLOR_MAP.get("UK") or "#5C1F1F",
            ec=COLOR_MAP.get("GB") or COLOR_MAP.get("UK") or "#5C1F1F",
            ls="solid", lw=0.6
        ),
        ROE_LABEL: dict(
            fc=ROE_FC,
            ec=ROE_FC,
            ls="solid", lw=0.6
        ),
        ROW_LABEL: dict(
            fc=ROW_FC,
            ec="black",
            ls=(0, (3, 2)),
            lw=0.6
        ),
    }

    for iso, lab in zip(split_row, split_regions_raw):
        c = COLOR_MAP.get(iso) or COLOR_MAP.get(lab) or "#BBBBBB"
        STYLE_MAP[lab] = dict(fc=c, ec=c, ls="solid", lw=0.6)

    df = PANEL_SUB.copy()
    if scenario is not None and scenario_col in df.columns:
        df = df[df[scenario_col].astype(str) == str(scenario)].copy()

    if components is not None and comp_col in df.columns:
        comp_set = {str(x).strip() for x in components}
        df = df[df[comp_col].astype(str).isin(comp_set)].copy()

    if df.empty:
        avail_comps = sorted({str(x) for x in PANEL_SUB.get(comp_col, pd.Series(dtype=str)).dropna().unique().tolist()})
        avail_scen = sorted({str(x) for x in PANEL_SUB.get(scenario_col, pd.Series(dtype=str)).dropna().unique().tolist()})
        raise ValueError(
            "No data after applying scenario/components filters. "
            f"Requested components={components}, scenario={scenario}. "
            f"Available components={avail_comps}. Available scenarios={avail_scen}."
        )

    df[delta_col] = df[delta_col].astype(int)

    if delta_min is None:
        if df[delta_col].dropna().empty:
            raise ValueError("No valid delta values found after filtering.")
        delta_min = int(df[delta_col].min())
    if delta_max is None:
        delta_max = int(df[delta_col].max())

    df = df[(df[delta_col] >= int(delta_min)) & (df[delta_col] <= int(delta_max))].copy()

    EU_EU_LABEL = f"{EU_LABEL} (from EU)"
    EU_ROW_LABEL = f"{EU_LABEL} (from RoW)"
    if split_eu_indirect:
        if (EU_EU_LABEL not in df[region_col].unique()) or (EU_ROW_LABEL not in df[region_col].unique()):
            raise ValueError("EU split rows not found. Build panels with add_eu_indirect_split=True.")
        df = df[df[region_col] != EU_LABEL].copy()
        eu_labels = [EU_EU_LABEL, EU_ROW_LABEL]
    else:
        df = df[~df[region_col].isin([EU_EU_LABEL, EU_ROW_LABEL])].copy()
        eu_labels = [EU_LABEL]

    agg = (
        df.groupby([delta_col, region_col], as_index=False)
          .agg(
              GVA_D=(gva_direct_col, "sum"),
              GVA_T=(gva_total_col, "sum"),
              JOB_D=(jobs_direct_col, "sum"),
              JOB_T=(jobs_total_col, "sum"),
          )
    )
    agg["GVA_I"] = agg["GVA_T"] - agg["GVA_D"]
    agg["JOB_I"] = agg["JOB_T"] - agg["JOB_D"]

    deltas = np.array(sorted(agg[delta_col].unique()), dtype=int)

    # Order split regions by importance (descending), keep EU/UK first, RoW last
    if split_regions_raw:
        tot_by_region = (
            agg.groupby(region_col, as_index=False)[["GVA_T", "JOB_T"]].sum()
        )
        s = tot_by_region[tot_by_region[region_col].isin(split_regions_raw)].copy()
        s["abs_gva"] = s["GVA_T"].abs()
        s["abs_jobs"] = s["JOB_T"].abs()

        gva_den = float(s["abs_gva"].sum()) or 1.0
        jobs_den = float(s["abs_jobs"].sum()) or 1.0

        s["share_gva"] = s["abs_gva"] / gva_den
        s["share_jobs"] = s["abs_jobs"] / jobs_den
        s["score"] = 0.5 * s["share_gva"] + 0.5 * s["share_jobs"]

        if split_row_order:
            order_codes = [str(x).strip().upper() for x in split_row_order]
            order_labels = [split_row_labels.get(c, c) for c in order_codes if c in split_row]
            split_rank = order_labels
        else:
            split_rank = split_regions_raw

        base_regions = set(eu_labels + [UK_LABEL, ROE_LABEL, ROW_LABEL])
        split_regions = [
            r for r in (split_rank + [r for r in split_regions_raw if r not in split_rank])
            if r not in base_regions
        ]
    else:
        split_regions = []

    if split_eu_indirect:
        eu_fc = COLOR_MAP.get("EU27") or COLOR_MAP.get(EU_LABEL) or "#4C78A8"
        STYLE_MAP[EU_EU_LABEL] = dict(fc=eu_fc, ec=eu_fc, ls="solid", lw=0.6)
        STYLE_MAP[EU_ROW_LABEL] = dict(fc=darken_color(eu_fc, 0.20), ec=darken_color(eu_fc, 0.20), ls="solid", lw=0.6)
        ORIGIN_ORDER = eu_labels + [UK_LABEL, ROE_LABEL] + split_regions + [ROW_LABEL]
    else:
        ORIGIN_ORDER = eu_labels + [UK_LABEL, ROE_LABEL] + split_regions + [ROW_LABEL]

    grid = pd.MultiIndex.from_product([deltas, ORIGIN_ORDER], names=[delta_col, region_col]).to_frame(index=False)
    agg = grid.merge(agg, on=[delta_col, region_col], how="left").fillna(0.0)

    def pivot(metric: str) -> pd.DataFrame:
        return (
            agg.pivot(index=delta_col, columns=region_col, values=metric)
               .reindex(index=deltas, columns=ORIGIN_ORDER)
               .fillna(0.0)
        )

    gva_d = pivot("GVA_D")
    gva_i = pivot("GVA_I")
    jobs_d = pivot("JOB_D")
    jobs_i = pivot("JOB_I")

    # Clip negative residuals (e.g., RoW balancing artifacts) for plotting only
    gva_d = gva_d.clip(lower=0.0)
    gva_i = gva_i.clip(lower=0.0)
    jobs_d = jobs_d.clip(lower=0.0)
    jobs_i = jobs_i.clip(lower=0.0)

    if y_as_million:
        gva_d = gva_d / 1e6
        gva_i = gva_i / 1e6
        gva_unit = "mEUR"
    else:
        gva_unit = "EUR"

    def fmt_int(v: float) -> str:
        return f"{v:,.0f}"

    def fmt_k(v: float) -> str:
        av = abs(float(v))
        if av >= 1000.0:
            val = float(v) / 1000.0
            s = f"{val:.0f}" if abs(val) >= 10.0 else f"{val:.1f}".rstrip("0").rstrip(".")
            return f"{s}k"
        return f"{v:,.0f}"

    def fmt_label(v: float) -> str:
        if value_format == "k":
            if abs(float(v)) >= 1000.0:
                val = float(v) / 1000.0
                s = f"{val:.0f}" if abs(val) >= 10.0 else f"{val:.1f}".rstrip("0").rstrip(".")
                return f"{s}k"
            return f"{float(v):.0f}"
        return f"{float(v):.0f}"

    fig, axes = plt.subplots(
        nrows=2, ncols=2,
        figsize=(fig_w, fig_h),
        dpi=150,
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": hspace, "wspace": wspace},
        constrained_layout=False,
    )
    ax11, ax12 = axes[0]
    ax21, ax22 = axes[1]

    dx_abs = 5.0 * float(edge_label_dx) if len(deltas) >= 2 else 0.0

    def x_interior(xi: float, x0: float, x1: float) -> float:
        if dx_abs <= 0:
            return xi
        if abs(xi - x0) < 1e-12:
            return xi + dx_abs
        if abs(xi - x1) < 1e-12:
            return xi - dx_abs
        return xi

    def label_mask(xvals: np.ndarray) -> np.ndarray:
        step = max(int(label_step), 1)
        base = int(delta_min) + int(label_start_offset)
        mask = ((xvals.astype(int) - base) % step) == 0
        if len(xvals) > 0:
            mask = mask | (xvals.astype(int) == int(xvals[0]))
            mask = mask | (xvals.astype(int) == int(xvals[-1]))
        if label_at_zero and (0 in xvals.astype(int)):
            mask = mask | (xvals.astype(int) == 0)
        return mask

    def y_shift_px(ax, x, y, dy_px: float) -> float:
        xd, yd = ax.transData.transform((x, y))
        yd2 = yd + dy_px
        x2, y2 = ax.transData.inverted().transform((xd, yd2))
        return y2

    def stack_area(ax, mat, ylabel: str, *, non_eu_threshold: float, label_min: float):
        x = mat.index.values.astype(float)
        x0, x1 = float(x[0]), float(x[-1])

        bottoms = np.zeros_like(x, dtype=float)
        stacks = {}

        for reg in ORIGIN_ORDER:
            y = mat[reg].values.astype(float)
            st = STYLE_MAP[reg]

            ax.fill_between(x, bottoms, bottoms + y, alpha=alpha_fill, color=st["fc"], linewidth=0.0)
            if show_edges:
                ax.plot(x, bottoms + y, color=st["ec"], linewidth=st["lw"], linestyle=st["ls"])

            stacks[reg] = (bottoms.copy(), y.copy())
            bottoms += y

        total = mat.sum(axis=1).values.astype(float)
        if show_total_line:
            ax.plot(x, total, color="black", linewidth=1.8, zorder=6)

        ax.set_ylabel(tex_bold(ylabel), fontsize=ytitle_font)
        apply_house_style(ax, grid=True)
        ax.grid(axis="y", alpha=0.2, linestyle=":", linewidth=0.5)

        if not show_value_labels:
            return

        mask = label_mask(x)

        if show_total_labels:
            for i in np.where(mask)[0]:
                xi = x_interior(float(x[i]), x0, x1)

                fnt = int(total_label_font) if total_label_font is not None else int(value_font)
                yi = total[i] * 1.01
                yi = y_shift_px(ax, xi, yi, float(total_label_offset_px))
                lbl = fmt_label(total[i])

                ax.text(
                    xi, yi,
                    tex_safe(lbl),
                    ha="center", va="bottom",
                    fontsize=fnt,
                    fontweight="bold",
                    color="black",
                    clip_on=False
                )

        if show_non_eu_labels:
            label_regions = eu_labels + [UK_LABEL, ROE_LABEL] + split_regions + [ROW_LABEL]
        else:
            label_regions = eu_labels

        for reg in label_regions:
            bottom, height = stacks[reg]
            y_mid = bottom + 0.5 * height
            tc = get_text_color(STYLE_MAP[reg]["fc"])

            for i in np.where(mask)[0]:
                if height[i] <= 0:
                    continue
                if float(height[i]) < float(label_min):
                    continue
                if (reg not in eu_labels) and (float(height[i]) < float(non_eu_threshold)):
                    continue
                xi = x_interior(float(x[i]), x0, x1)
                lbl = fmt_label(height[i])
                ax.text(
                    xi, y_mid[i],
                    tex_safe(lbl),
                    ha="center", va="center",
                    fontsize=value_font,
                    color=tc,
                    alpha=0.92,
                    fontweight="bold" if reg in eu_labels else "normal",
                    clip_on=True
                )

    gva_d_min = label_min_gva if label_min_gva_direct is None else label_min_gva_direct
    gva_i_min = label_min_gva if label_min_gva_indirect is None else label_min_gva_indirect
    jobs_d_min = label_min_jobs if label_min_jobs_direct is None else label_min_jobs_direct
    jobs_i_min = label_min_jobs if label_min_jobs_indirect is None else label_min_jobs_indirect

    stack_area(ax11, gva_d, f"GVA Direct [{gva_unit}]", non_eu_threshold=float(non_eu_label_min_abs_gva), label_min=float(gva_d_min))
    stack_area(ax12, gva_i, f"GVA Indirect [{gva_unit}]", non_eu_threshold=float(non_eu_label_min_abs_gva), label_min=float(gva_i_min))
    stack_area(ax21, jobs_d, "Employment Direct [job-years]", non_eu_threshold=float(non_eu_label_min_abs_jobs), label_min=float(jobs_d_min))
    stack_area(ax22, jobs_i, "Employment Indirect [job-years]", non_eu_threshold=float(non_eu_label_min_abs_jobs), label_min=float(jobs_i_min))

    for ax in (ax11, ax12, ax21, ax22):
        ax.set_ylim(bottom=0.0)

    if x_tick_step is None or int(x_tick_step) <= 0:
        xticks = deltas.astype(int)
    else:
        step = int(x_tick_step)
        tick_lo = int(delta_min)
        tick_hi = int(delta_max)
        xticks = np.arange(tick_lo, tick_hi + step, step, dtype=int)

    ax21.set_xlabel(tex_bold("Policy shock (Δpp)"), fontsize=ytick_font)
    ax22.set_xlabel(tex_bold("Policy shock (Δpp)"), fontsize=ytick_font)
    ax21.set_xticks(xticks)
    ax22.set_xticks(xticks)
    ax21.set_xticklabels([str(int(x)) for x in xticks], fontsize=xtick_font)
    ax22.set_xticklabels([str(int(x)) for x in xticks], fontsize=xtick_font)

    for ax in (ax11, ax12, ax21, ax22):
        ax.tick_params(axis="y", labelsize=ytick_font)
        ax.set_xlim(float(delta_min) - 2.0, float(delta_max) + 2.0)
        if ytick_format == "k":
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: fmt_k(v)))

    patches = [
        mpatches.Patch(
            facecolor=STYLE_MAP[r]["fc"],
            edgecolor=STYLE_MAP[r]["ec"],
            linewidth=STYLE_MAP[r]["lw"],
            linestyle=STYLE_MAP[r]["ls"],
            label=tex_safe(r),
        )
        for r in ORIGIN_ORDER
    ]
    legend_y_eff = 0.98 + float(legend_y)
    if legend_y >= 0:
        legend_y_eff = min(0.98, max(0.05, legend_y_eff))
    leg = fig.legend(
        handles=patches,
        title=tex_bold(legend_title) if legend_title else None,
        loc=legend_loc,
        bbox_to_anchor=(float(legend_x), float(legend_y_eff)),
        bbox_transform=fig.transFigure,
        borderaxespad=0.2,
        fontsize=legend_font,
        title_fontsize=legend_title_font,
        ncol=legend_ncol,
        frameon=False,
        framealpha=1.0,
        handlelength=1.1,
        handletextpad=0.4,
        labelspacing=0.3,
        borderpad=0.4,
    )
    leg.set_zorder(100)
    for t in leg.get_texts():
        t.set_fontweight("bold")
    leg.get_frame().set_zorder(100)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.10, hspace=hspace, wspace=wspace)
    try:
        fig.set_layout_engine("none")
    except Exception:
        pass

    return fig, (ax11, ax12, ax21, ax22)


def plot_and_save_global_totals_by_region_direct_indirect(
    *,
    PANEL_SUB: pd.DataFrame,
    filename: str,
    fig_w: float | None = None,
    fig_h: float | None = None,
    save: bool | None = None,
    close: bool | None = None,
    **kw,
):
    if fig_w is not None:
        kw["fig_w"] = fig_w
    if fig_h is not None:
        kw["fig_h"] = fig_h
    fig, _ = plot_global_totals_by_region_direct_indirect(PANEL_SUB, **kw)
    return save_and_close(fig=fig, filename=filename, save=save, close=close)


def build_eu27_gross_net_multiplier_decomposition(
    PANEL_NET: pd.DataFrame,
    PANEL_GROSS: pd.DataFrame,
    *,
    indicator: str = "GVA",
    delta_base: int = 0,
    delta_target: int = 10,
    scenario: str | None = "baseline",
    scenario_col: str = "scenario",
    delta_col: str = "delta_pp",
    region_col: str = "Region",
    comp_col: str = "Component",
    subcomp_col: str = "Subcomponent",
    output_direct_col: str = "abs_output_direct",
    gva_direct_col: str = "abs_gva_direct",
    gva_total_col: str = "abs_gva_total",
    jobs_direct_col: str = "abs_jobs_direct",
    jobs_total_col: str = "abs_jobs_total",
    eu_label: str = EU_LABEL,
    drop_matching_component_subcomponent: bool = True,
) -> pd.DataFrame:
    """Build the point-in-delta gross/net decomposition used by the resilience plots."""
    ind_u = str(indicator).upper().strip()
    if ind_u == "GVA":
        direct_col = gva_direct_col
        total_col = gva_total_col
        denom_scale = 1.0
        indicator_label = "GVA"
    elif ind_u == "JOBS":
        direct_col = jobs_direct_col
        total_col = jobs_total_col
        denom_scale = 1_000_000.0
        indicator_label = "Jobs"
    else:
        raise ValueError("indicator must be 'GVA' or 'Jobs'.")

    key = [comp_col, subcomp_col]

    def _prep(panel: pd.DataFrame, label: str) -> pd.DataFrame:
        df = panel.copy()
        if scenario is not None and scenario_col in df.columns:
            df = df[df[scenario_col].astype(str) == str(scenario)].copy()
        needed = [delta_col, region_col, comp_col, subcomp_col, output_direct_col, direct_col, total_col]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns in {label} panel: {missing}")
        df[delta_col] = pd.to_numeric(df[delta_col], errors="coerce").fillna(0).astype(int)
        df[region_col] = df[region_col].astype(str).str.strip()
        df[comp_col] = df[comp_col].astype(str).str.strip()
        df[subcomp_col] = df[subcomp_col].astype(str).str.strip()
        for c in (output_direct_col, direct_col, total_col):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        comp_cf = df[comp_col].str.casefold()
        sub_cf = df[subcomp_col].str.casefold()
        keep = ~comp_cf.isin({"all", "total"}) & ~sub_cf.isin({"all", "total"})
        if drop_matching_component_subcomponent:
            keep &= comp_cf != sub_cf
        return df.loc[keep, needed].copy()

    net = _prep(PANEL_NET, "net")
    gross = _prep(PANEL_GROSS, "gross")
    for label, panel in (("net", net), ("gross", gross)):
        available = set(panel[delta_col].unique())
        missing = {int(delta_base), int(delta_target)} - available
        if missing:
            raise KeyError(f"{label} panel is missing delta(s): {sorted(missing)}.")

    den = (
        net[net[delta_col] == int(delta_base)]
        .groupby(key, as_index=False)[output_direct_col]
        .sum()
        .rename(columns={output_direct_col: "_den_output"})
    )

    def _eu(panel: pd.DataFrame, prefix: str) -> pd.DataFrame:
        eu = (
            panel[panel[region_col] == eu_label]
            .groupby([delta_col, *key], as_index=False)[[direct_col, total_col]]
            .sum()
            .rename(columns={direct_col: f"_{prefix}_direct", total_col: f"_{prefix}_total"})
        )
        base = (
            eu[eu[delta_col] == int(delta_base)][key + [f"_{prefix}_direct", f"_{prefix}_total"]]
            .rename(columns={
                f"_{prefix}_direct": f"_{prefix}_base_direct",
                f"_{prefix}_total": f"_{prefix}_base_total",
            })
        )
        target = eu[eu[delta_col] == int(delta_target)].copy()
        merged = target.merge(base, on=key, how="outer")
        merged[delta_col] = int(delta_target)
        return merged

    out = _eu(gross, "gross").merge(_eu(net, "net"), on=[delta_col, *key], how="outer")
    out = out.merge(den, on=key, how="left").fillna(0.0)
    denom = (out["_den_output"].astype(float) / float(denom_scale)).replace(0.0, np.nan)

    out["direct_gross"] = (out["_gross_direct"] - out["_gross_base_direct"]) / denom
    out["_total_gross"] = (out["_gross_total"] - out["_gross_base_total"]) / denom
    out["indirect_gross"] = out["_total_gross"] - out["direct_gross"]
    out["total_net"] = (out["_net_total"] - out["_net_base_total"]) / denom
    out["trade_linkage_adjustment"] = out["total_net"] - out["_total_gross"]
    out["indicator"] = indicator_label
    out["delta_base"] = int(delta_base)
    out["delta_target"] = int(delta_target)

    out = out.rename(columns={comp_col: "Component", subcomp_col: "Subcomponent"})
    cols = [
        "Component",
        "Subcomponent",
        "direct_gross",
        "indirect_gross",
        "trade_linkage_adjustment",
        "total_net",
        "indicator",
        "delta_base",
        "delta_target",
    ]
    return out[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def plot_eu27_gross_net_multiplier_decomposition(
    PANEL_NET: pd.DataFrame,
    PANEL_GROSS: pd.DataFrame,
    *,
    indicator: str = "GVA",
    delta_base: int = 0,
    delta_target: int = 10,
    components_keep: tuple[str, ...] | list[str] | None = None,
    subcomponents_keep: tuple[str, ...] | list[str] | None = None,
    component_order: tuple[str, ...] | list[str] | None = None,
    subcomponent_order: tuple[str, ...] | list[str] | dict[str, list[str]] | None = None,
    ylab_main: str | None = None,
    figsize: tuple[float, float] = (12.0617, 3.8818),
    dpi: int = 160,
    bar_width: float = 0.85,
    label_min_abs: float = 0.0,
    show_labels: bool = True,
    show_legend: bool = True,
    legend_loc: str = "lower right",
    legend_x: float = 0.98,
    legend_y: float = 0.14,
    legend_ncol: int = 1,
    xtick_font: int = 10,
    ytick_font: int = 10,
    ytitle_font: int = 12,
    component_label_font: int = 14,
    component_label_offset: float = -0.35,
    legend_font: int = 10,
    legend_title_font: int = 9,
    value_font: int = 9,
    net_marker_size: float = 4.3,
    net_marker_edgewidth: float = 2.2,
    legend_direct_label: str | None = None,
    legend_indirect_label: str | None = None,
    legend_linkage_label: str | None = None,
    legend_total_label: str | None = None,
    legend_unit: str | None = None,
    trade_loss_label_offset_points: float = 10.0,
    subcomponent_label_map: dict[str, str] | None = None,
    drop_matching_component_subcomponent: bool = True,
):
    # Plot direct/indirect gross effects and overlay the net total effect.
    set_house_style()
    plot_df = build_eu27_gross_net_multiplier_decomposition(
        PANEL_NET,
        PANEL_GROSS,
        indicator=indicator,
        delta_base=delta_base,
        delta_target=delta_target,
        drop_matching_component_subcomponent=drop_matching_component_subcomponent,
    )

    if components_keep:
        keep = {str(c).strip() for c in components_keep}
        plot_df = plot_df[plot_df["Component"].isin(keep)].copy()
    if subcomponents_keep:
        keep = {str(c).strip() for c in subcomponents_keep}
        plot_df = plot_df[plot_df["Subcomponent"].isin(keep)].copy()
    if plot_df.empty:
        raise ValueError("No rows to plot after component/subcomponent filter.")

    comps = plot_df["Component"].unique().tolist()
    if component_order is None:
        pref_main = ["Devex", "Turbine", "BoP", "Installation", "O&M"]
        pref_turb = ["Blades", "Hub", "Drivetrain", "Nacelle", "Tower"]
        if any(c in comps for c in pref_main):
            comp_order = [c for c in pref_main if c in comps] + [c for c in comps if c not in pref_main]
        elif any(c in comps for c in pref_turb):
            comp_order = [c for c in pref_turb if c in comps] + [c for c in comps if c not in pref_turb]
        else:
            comp_order = sorted(comps)
    else:
        base = [c for c in component_order if c in comps]
        comp_order = base + [c for c in comps if c not in set(base)]

    x, xt, centers, seps, rows, k = [], [], [], [], [], 0
    for gi, comp in enumerate(comp_order):
        subdf = plot_df[plot_df["Component"] == comp].copy()
        if subcomponent_order is not None:
            if isinstance(subcomponent_order, dict):
                order_list = subcomponent_order.get(comp)
            else:
                order_list = list(subcomponent_order)
            if order_list:
                sub_set = set(subdf["Subcomponent"].tolist())
                ordered = [s for s in order_list if s in sub_set]
                ordered += [s for s in subdf["Subcomponent"].tolist() if s not in set(ordered)]
                subdf = subdf.set_index("Subcomponent").loc[ordered].reset_index()
            else:
                subdf = subdf.sort_values("Subcomponent")
        else:
            subdf = subdf.sort_values("Subcomponent")

        start = k
        for _, r in subdf.iterrows():
            x.append(k)
            xt.append(r["Subcomponent"])
            rows.append(r)
            k += 1
        end = k - 1
        centers.append((comp, (start + end) / 2))
        if gi < len(comp_order) - 1:
            seps.append(k)
        k += 1

    plot_df = pd.DataFrame(rows).reset_index(drop=True)
    xs = np.asarray(x[:len(plot_df)], dtype=float)

    default_subcomponent_label_map = dict(SUBCOMPONENT_LABEL_ABBREVIATIONS)
    if subcomponent_label_map:
        default_subcomponent_label_map.update({
            str(k): str(v) for k, v in subcomponent_label_map.items()
        })

    ind_u = str(indicator).upper().strip()
    legend_unit_use = legend_unit
    if legend_unit_use is None:
        legend_unit_use = "EUR/EUR" if ind_u == "GVA" else "job-years/mEUR"
    legend_direct_label_use = legend_direct_label or "Direct"
    legend_indirect_label_use = legend_indirect_label or "Indirect"
    legend_linkage_label_use = legend_linkage_label or "Upstream trade effects"
    legend_total_label_use = legend_total_label or "Net effect"

    eu_fc = COLOR_MAP.get("EU27") or COLOR_MAP.get(EU_LABEL) or "#4C78A8"
    indirect_fc = lighten_color(eu_fc, 0.35)
    net_fc = COLOR_MAP.get("GB") or COLOR_MAP.get("UK") or "#5C1F1F"
    linkage_edge = net_fc
    linkage_fc = "#F6D6DA"

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    def _fmt(v: float) -> str:
        v = float(v)
        if v == 0.0:
            return "0"
        if abs(v) >= 1.0:
            return f"{v:,.0f}"
        for d in range(1, 7):
            if round(abs(v), d) != 0:
                return f"{v:,.{d}f}"
        return f"{v:,.6f}"

    direct_vals = plot_df["direct_gross"].to_numpy(dtype=float)
    indirect_vals = plot_df["indirect_gross"].to_numpy(dtype=float)
    net_vals = plot_df["total_net"].to_numpy(dtype=float)

    pos_base = np.zeros_like(xs, dtype=float)
    neg_base = np.zeros_like(xs, dtype=float)
    stack_info = []

    linkage_vals = plot_df["trade_linkage_adjustment"].to_numpy(dtype=float)

    for vals, color, hatch, edge_override, negative_only in (
        (direct_vals, eu_fc, None, None, False),
        (indirect_vals, indirect_fc, None, None, False),
        (linkage_vals, linkage_fc, "///", linkage_edge, False),
    ):
        edge = edge_override if edge_override is not None else darken_color(color, 0.25)
        y_pos = np.zeros_like(vals, dtype=float) if negative_only else np.where(vals > 0, vals, 0.0)
        y_neg = np.where(vals < 0, vals, 0.0)
        base_pos = pos_base.copy()
        base_neg = neg_base.copy()

        if np.any(np.abs(y_pos) > 1e-12):
            ax.bar(xs, y_pos, bottom=base_pos, width=bar_width, color=color, edgecolor=edge, linewidth=0.6, hatch=hatch)
        if np.any(np.abs(y_neg) > 1e-12):
            ax.bar(xs, y_neg, bottom=base_neg, width=bar_width, color=color, edgecolor=edge, linewidth=0.6, hatch=hatch)

        stack_info.append((vals, base_pos, base_neg, color, hatch is not None))
        pos_base += y_pos
        neg_base += y_neg

    if show_labels:
        for vals, base_pos, base_neg, color, is_trade_loss in stack_info:
            for i, v in enumerate(vals):
                if abs(v) < float(label_min_abs):
                    continue
                if is_trade_loss:
                    y_edge = (base_pos[i] + v) if v >= 0 else (base_neg[i] + v)
                    ax.annotate(
                        tex_safe(_fmt(v)),
                        xy=(xs[i], y_edge),
                        xytext=(0, -abs(float(trade_loss_label_offset_points))),
                        textcoords="offset points",
                        ha="center",
                        va="top",
                        fontsize=value_font,
                        color=linkage_edge,
                        zorder=9,
                        clip_on=False,
                    )
                    continue
                else:
                    y = (base_pos[i] + v / 2.0) if v >= 0 else (base_neg[i] + v / 2.0)
                    va = "center"
                    label_color = get_text_color(color)
                ax.text(
                    xs[i], y,
                    tex_safe(_fmt(v)),
                    ha="center",
                    va=va,
                    fontsize=value_font,
                    color=label_color,
                    zorder=9,
                    clip_on=True,
                )

    ax.scatter(
        xs,
        net_vals,
        facecolors="white",
        edgecolors=net_fc,
        marker="D",
        s=38,
        linewidths=2.6,
        zorder=6,
        label=tex_safe("Net total"),
    )

    for s in seps:
        ax.axvline(s, color="black", linewidth=1.0, linestyle="--", alpha=0.6)

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    if ylab_main is None:
        if ind_u == "GVA":
            ylab_main = f"EU27 GVA change ({delta_base}->{delta_target}) [EUR/EUR]"
        else:
            ylab_main = f"EU27 employment change ({delta_base}->{delta_target}) [job-years/mEUR]"
    elif legend_unit_use and "[" not in str(ylab_main):
        ylab_main = f"{ylab_main} [{legend_unit_use}]"
    ax.set_ylabel(tex_bold(ylab_main), fontsize=ytitle_font)
    ax.set_xticks(xs)
    xt_disp = [
        default_subcomponent_label_map.get(str(t), _short_subcomponent_label(t))
        for t in xt[:len(plot_df)]
    ]
    ax.set_xticklabels([tex_safe(t) for t in xt_disp], fontsize=xtick_font, rotation=45, ha="right")
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=ytick_font)

    for comp, c in centers:
        ax.text(
            c,
            component_label_offset,
            tex_safe(str(comp)),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=component_label_font,
            fontweight="bold",
        )

    apply_house_style(ax, grid=True)
    ax.grid(axis="y", alpha=0.2, linestyle=":", linewidth=0.5)

    pos_env = np.maximum(0.0, direct_vals) + np.maximum(0.0, indirect_vals) + np.maximum(0.0, linkage_vals)
    neg_env = np.minimum(0.0, direct_vals) + np.minimum(0.0, indirect_vals) + np.minimum(0.0, linkage_vals)
    ymin = float(min(0.0, np.nanmin(neg_env) if len(neg_env) else 0.0, np.nanmin(net_vals) if len(net_vals) else 0.0))
    ymax = float(max(0.0, np.nanmax(pos_env) if len(pos_env) else 0.0, np.nanmax(net_vals) if len(net_vals) else 0.0))
    yrng = ymax - ymin
    ypad = 0.12 * (yrng if yrng > 0 else 1.0)
    ax.set_ylim(ymin - ypad, ymax + ypad)

    if show_legend:
        handles = [
            mpatches.Patch(facecolor=eu_fc, edgecolor=darken_color(eu_fc, 0.25), label=tex_safe(str(legend_direct_label_use))),
            mpatches.Patch(facecolor=indirect_fc, edgecolor=darken_color(indirect_fc, 0.25), label=tex_safe(str(legend_indirect_label_use))),
            mpatches.Patch(facecolor=linkage_fc, edgecolor=linkage_edge, hatch="///", linewidth=0.6, label=tex_safe(str(legend_linkage_label_use))),
            plt.Line2D(
                [0], [0],
                marker="D",
                color=net_fc,
                markerfacecolor="white",
                markeredgecolor=net_fc,
                linestyle="None",
                markeredgewidth=float(net_marker_edgewidth),
                markersize=max(3.0, float(net_marker_size) + 1.0),
                label=tex_safe(str(legend_total_label_use)),
            ),
        ]
        leg = ax.legend(
            handles=handles,
            loc=legend_loc,
            bbox_to_anchor=(legend_x, legend_y),
            bbox_transform=ax.transAxes,
            fontsize=legend_font,
            title_fontsize=legend_title_font,
            ncol=legend_ncol,
            frameon=False,
            framealpha=1.0,
            handlelength=1.1,
            handletextpad=0.4,
            labelspacing=0.3,
            borderpad=0.4,
        )
    fig.subplots_adjust(bottom=0.36)
    fig.tight_layout()
    return fig, ax, plot_df


def plot_and_save_eu27_gross_net_multiplier_decomposition(
    *,
    PANEL_NET: pd.DataFrame,
    PANEL_GROSS: pd.DataFrame,
    indicator: str = "GVA",
    filename: str | None = None,
    save: bool | None = None,
    close: bool | None = None,
    exact_size: bool = True,
    **kw,
):
    if filename is None:
        safe_ind = str(indicator).strip().lower()
        filename = f"3.7_eu27_gross_net_decomposition_{safe_ind}.pdf"

    ind_u = str(indicator).strip().upper()
    filename_l = str(filename).casefold()
    is_l0 = "_l0" in filename_l
    is_l0_neg = is_l0 and "_neg" in filename_l

    paper_defaults = {
        "show_labels": False,
        "value_font": 10,
        "trade_loss_label_offset_points": 4 if ind_u == "JOBS" else 8,
        "legend_loc": "lower right",
        "legend_x": 0.98,
        "legend_y": 0.01 if is_l0_neg else 0.06,
        "legend_ncol": 1,
        "component_label_offset": -0.33 if is_l0 else -0.30,
        "ylab_main": "EU27 emp. var." if ind_u == "JOBS" else "EU27 GVA var.",
        "label_min_abs": (30 if is_l0 else 10) if ind_u == "JOBS" else (5 if is_l0 else 2),
    }
    if is_l0:
        paper_defaults.update({
            "net_marker_size": 3.0,
            "net_marker_edgewidth": 1.8,
            "drop_matching_component_subcomponent": False,
        })
    for key, value in paper_defaults.items():
        kw.setdefault(key, value)

    fig, _, _ = plot_eu27_gross_net_multiplier_decomposition(
        PANEL_NET,
        PANEL_GROSS,
        indicator=indicator,
        **kw,
    )
    return _save_and_close_exact_aware(
        fig=fig,
        filename=filename,
        save=save,
        close=close,
        exact_size=exact_size,
    )


def build_eu27_gross_net_delta_area_df(
    PANEL_NET: pd.DataFrame,
    PANEL_GROSS: pd.DataFrame,
    *,
    indicator: str = "GVA",
    value_mode: str = "multiplier",
    delta_base: int = 0,
    scenario: str | None = "baseline",
    scenario_col: str = "scenario",
    delta_col: str = "delta_pp",
    region_col: str = "Region",
    comp_col: str = "Component",
    output_direct_col: str = "abs_output_direct",
    gva_direct_col: str = "abs_gva_direct",
    gva_total_col: str = "abs_gva_total",
    jobs_direct_col: str = "abs_jobs_direct",
    jobs_total_col: str = "abs_jobs_total",
    eu_label: str = EU_LABEL,
) -> pd.DataFrame:
    # Build EU27 gross/net multiplier decomposition over all available deltas.
    ind_u = str(indicator).upper().strip()
    mode = str(value_mode).strip().lower()
    if mode not in {"multiplier", "ratio", "absolute", "abs"}:
        raise ValueError("value_mode must be 'multiplier' or 'absolute'.")
    absolute_mode = mode in {"absolute", "abs"}

    if ind_u == "GVA":
        direct_col = gva_direct_col
        total_col = gva_total_col
        denom_scale = 1.0
        abs_scale = 1_000_000.0
    elif ind_u == "JOBS":
        direct_col = jobs_direct_col
        total_col = jobs_total_col
        denom_scale = 1_000_000.0
        abs_scale = 1_000.0
    else:
        raise ValueError("indicator must be 'GVA' or 'Jobs'.")

    def _prep(panel: pd.DataFrame) -> pd.DataFrame:
        df = panel.copy()
        if scenario is not None and scenario_col in df.columns:
            df = df[df[scenario_col].astype(str) == str(scenario)].copy()
        needed = [delta_col, region_col, comp_col, output_direct_col, direct_col, total_col]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns in panel: {missing}")
        df[delta_col] = pd.to_numeric(df[delta_col], errors="coerce").fillna(0).astype(int)
        df[region_col] = df[region_col].astype(str).str.strip()
        df[comp_col] = df[comp_col].astype(str).str.strip()
        for c in (output_direct_col, direct_col, total_col):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        comp_cf = df[comp_col].str.casefold()
        df = df[~comp_cf.isin({"all", "total"})].copy()
        return df[needed].copy()

    net = _prep(PANEL_NET)
    gross = _prep(PANEL_GROSS)
    if int(delta_base) not in set(net[delta_col].unique()):
        raise KeyError(f"net panel is missing delta_base={delta_base}.")
    if int(delta_base) not in set(gross[delta_col].unique()):
        raise KeyError(f"gross panel is missing delta_base={delta_base}.")

    def _with_total(df: pd.DataFrame) -> pd.DataFrame:
        totals = (
            df.groupby([delta_col, region_col], as_index=False)[[output_direct_col, direct_col, total_col]]
            .sum()
        )
        totals[comp_col] = "Total"
        return pd.concat([df, totals[df.columns]], ignore_index=True)

    net = _with_total(net)
    gross = _with_total(gross)

    key = [comp_col]
    den = (
        net[net[delta_col] == int(delta_base)]
        .groupby(key, as_index=False)[output_direct_col]
        .sum()
        .rename(columns={output_direct_col: "_den_output"})
    )

    def _eu(panel: pd.DataFrame, prefix: str) -> pd.DataFrame:
        eu = (
            panel[panel[region_col] == eu_label]
            .groupby([delta_col, comp_col], as_index=False)[[direct_col, total_col]]
            .sum()
            .rename(columns={direct_col: f"_{prefix}_direct", total_col: f"_{prefix}_total"})
        )
        base = (
            eu[eu[delta_col] == int(delta_base)][[comp_col, f"_{prefix}_direct", f"_{prefix}_total"]]
            .rename(columns={
                f"_{prefix}_direct": f"_{prefix}_base_direct",
                f"_{prefix}_total": f"_{prefix}_base_total",
            })
        )
        out = eu.merge(base, on=comp_col, how="left")
        return out

    out = _eu(gross, "gross").merge(_eu(net, "net"), on=[delta_col, comp_col], how="outer")
    out = out.merge(den, on=comp_col, how="left").fillna(0.0)

    if absolute_mode:
        denom = float(abs_scale)
    else:
        denom = (out["_den_output"].astype(float) / float(denom_scale)).replace(0.0, np.nan)

    out["direct_gross"] = (out["_gross_direct"] - out["_gross_base_direct"]) / denom
    out["total_gross"] = (out["_gross_total"] - out["_gross_base_total"]) / denom
    out["gross_indirect"] = out["total_gross"] - out["direct_gross"]
    out["net_effect"] = (out["_net_total"] - out["_net_base_total"]) / denom
    out["upstream_trade_adjustment"] = out["net_effect"] - out["total_gross"]
    out["indicator"] = "GVA" if ind_u == "GVA" else "Jobs"
    out["value_mode"] = "absolute" if absolute_mode else "multiplier"
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_eu27_resilience_accounting_table(
    PANEL_NET: pd.DataFrame,
    PANEL_EU_SIDE: pd.DataFrame,
    *,
    component: str | None = None,
    component_order: list[str] | tuple[str, ...] | None = None,
    component_labels: dict | None = None,
    delta_base: int = 0,
    delta_resil: int | list[int] | tuple[int, ...] = 100,
    scenario_label_template: str = "Delta = {delta}",
    scenario: str | None = "baseline",
    scenario_col: str = "scenario",
    delta_col: str = "delta_pp",
    region_col: str = "Region",
    comp_col: str = "Component",
    gva_direct_col: str = "abs_gva_direct",
    gva_total_col: str = "abs_gva_total",
    jobs_direct_col: str = "abs_jobs_direct",
    jobs_total_col: str = "abs_jobs_total",
    eu_label: str = EU_LABEL,
    gva_to_million: bool = True,
    jobs_to_thousand: bool = True,
    include_pct_change: bool = True,
    parenthesize_trade_adjustment: bool = False,
    indent_effect_rows: bool = False,
    labels: dict | None = None,
) -> pd.DataFrame:
    """
    Build an EU27-only component accounting table for the resilience scenario.

    Rows decompose the full signed net effect into the EU27-side direct
    change, the gross EU27 upstream effect, upstream trade effects,
    and the net effect. Columns are broad components.
    """
    if isinstance(delta_resil, (list, tuple, set, np.ndarray, pd.Index)):
        parts = []
        for delta in delta_resil:
            delta_i = int(delta)
            part = build_eu27_resilience_accounting_table(
                PANEL_NET,
                PANEL_EU_SIDE,
                component=component,
                component_order=component_order,
                component_labels=component_labels,
                delta_base=delta_base,
                delta_resil=delta_i,
                scenario_label_template=scenario_label_template,
                scenario=scenario,
                scenario_col=scenario_col,
                delta_col=delta_col,
                region_col=region_col,
                comp_col=comp_col,
                gva_direct_col=gva_direct_col,
                gva_total_col=gva_total_col,
                jobs_direct_col=jobs_direct_col,
                jobs_total_col=jobs_total_col,
                eu_label=eu_label,
                gva_to_million=gva_to_million,
                jobs_to_thousand=jobs_to_thousand,
                include_pct_change=include_pct_change,
                parenthesize_trade_adjustment=parenthesize_trade_adjustment,
                indent_effect_rows=indent_effect_rows,
                labels=labels,
            ).copy()
            part["Scenario"] = scenario_label_template.format(delta=delta_i)
            part = part.set_index("Scenario", append=True)
            part = part.reorder_levels(["Scenario"] + [n for n in part.index.names if n != "Scenario"])
            parts.append(part)
        return pd.concat(parts)

    labels = labels or {}
    if component_order is None:
        component_order = ["Total", "Devex", "Turbine", "BoP", "Installation", "O&M"]
    component_labels = {"Installation": "Inst."} | (component_labels or {})
    indent = r"\hspace{0.8em}" if indent_effect_rows else ""
    base_label = labels.get("base", "EU27 base (0)")
    direct_label = labels.get("direct", "Direct (1)")
    gross_indirect_label = labels.get("gross_indirect", "Indirect (2)")
    gross_label = labels.get("gross", "Gross effect (1+2)")
    gross_pct_label = labels.get("gross_pct", "Var % Gross")
    trade_label = labels.get("trade_losses", "Up. trade effects (3)")
    net_label = labels.get("net", "Net effect (1+2+3)")
    net_pct_label = labels.get("net_pct", "Var % Net")
    direct_row = f"{indent}{direct_label}"
    gross_indirect_row = f"{indent}{gross_indirect_label}"
    trade_row = f"{indent}{trade_label}"

    def _prep(panel: pd.DataFrame) -> pd.DataFrame:
        df = panel.copy()
        if scenario is not None and scenario_col in df.columns:
            df = df[df[scenario_col].astype(str) == str(scenario)].copy()
        needed = [delta_col, region_col, comp_col, gva_direct_col, gva_total_col, jobs_direct_col, jobs_total_col]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns in panel: {missing}")

        df[delta_col] = pd.to_numeric(df[delta_col], errors="coerce").fillna(0).astype(int)
        df[region_col] = df[region_col].astype(str).str.strip()
        df[comp_col] = df[comp_col].astype(str).str.strip()
        for c in (gva_direct_col, gva_total_col, jobs_direct_col, jobs_total_col):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        df = df[needed].copy()

        comp_cf = df[comp_col].str.casefold()
        no_total = df[~comp_cf.isin({"all", "total"})].copy()
        totals = (
            no_total
            .groupby([delta_col, region_col], as_index=False)[
                [gva_direct_col, gva_total_col, jobs_direct_col, jobs_total_col]
            ]
            .sum()
        )
        totals[comp_col] = "Total"
        return pd.concat([no_total, totals[needed]], ignore_index=True)

    net = _prep(PANEL_NET)
    eu_side = _prep(PANEL_EU_SIDE)

    if component_order is None:
        preferred = ["Total", "Devex", "BoP", "Turbine", "Installation", "O&M", "Devex+O&M"]
        available = net[comp_col].dropna().astype(str).unique().tolist()
        component_order = [c for c in preferred if c in available] + [c for c in available if c not in preferred]
    else:
        available = set(net[comp_col].dropna().astype(str).unique().tolist())
        component_order = [str(c).strip() for c in component_order if str(c).strip() in available]

    if not component_order:
        raise ValueError("No components available for the accounting table.")

    def _col_label(comp: str) -> str:
        return component_labels.get(comp, comp)

    def _pick(panel: pd.DataFrame, comp: str, delta: int, direct_col: str, total_col: str) -> tuple[float, float]:
        sub = panel[
            (panel[delta_col] == int(delta))
            & (panel[region_col] == eu_label)
            & (panel[comp_col] == comp)
        ]
        if sub.empty:
            avail = sorted(panel[comp_col].dropna().astype(str).unique().tolist())
            raise KeyError(f"No EU27 rows found for component={comp!r}, delta={delta}. Available components: {avail}")
        direct = float(sub[direct_col].sum())
        total = float(sub[total_col].sum())
        return direct, total

    rows = {}
    specs = [
        ("GVA", gva_direct_col, gva_total_col, 1e6 if gva_to_million else 1.0),
        ("Employment", jobs_direct_col, jobs_total_col, 1e3 if jobs_to_thousand else 1.0),
    ]

    for impact, direct_col, total_col, scale in specs:
        row_names = [
            base_label,
            direct_row,
            gross_indirect_row,
            gross_label,
            trade_row,
            net_label,
        ]
        if include_pct_change:
            row_names.append(gross_pct_label)
            row_names.append(net_pct_label)

        for row_name in row_names:
            rows[(impact, row_name)] = {}

        for comp in component_order:
            eu_base_direct, eu_base_total = _pick(eu_side, comp, delta_base, direct_col, total_col)
            eu_resil_direct, eu_resil_total = _pick(eu_side, comp, delta_resil, direct_col, total_col)
            _, net_base_total = _pick(net, comp, delta_base, direct_col, total_col)
            _, net_resil_total = _pick(net, comp, delta_resil, direct_col, total_col)

            direct_effect = eu_resil_direct - eu_base_direct
            gross_total_effect = eu_resil_total - eu_base_total
            gross_indirect = gross_total_effect - direct_effect
            net_effect = net_resil_total - net_base_total
            upstream_trade_effects = gross_total_effect - net_effect
            gross_pct = 100.0 * gross_total_effect / net_base_total if abs(net_base_total) > 1e-12 else np.nan
            net_pct = 100.0 * net_effect / net_base_total if abs(net_base_total) > 1e-12 else np.nan

            col = _col_label(comp)
            rows[(impact, base_label)][col] = net_base_total / scale
            rows[(impact, direct_row)][col] = direct_effect / scale
            rows[(impact, gross_indirect_row)][col] = gross_indirect / scale
            rows[(impact, gross_label)][col] = gross_total_effect / scale
            rows[(impact, trade_row)][col] = -upstream_trade_effects / scale
            rows[(impact, net_label)][col] = net_effect / scale
            if include_pct_change:
                rows[(impact, gross_pct_label)][col] = gross_pct
                rows[(impact, net_pct_label)][col] = net_pct

    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index = pd.MultiIndex.from_tuples(out.index, names=["Impact", "Effects"])
    ordered_cols = [_col_label(c) for c in component_order if _col_label(c) in out.columns]
    out = out.reindex(columns=ordered_cols)

    if parenthesize_trade_adjustment:
        def _fmt_accounting_loss(x) -> str:
            try:
                if pd.isna(x):
                    return ""
            except Exception:
                pass
            v = abs(float(x))
            if v >= 1.0:
                return rf"(\num{{{int(round(v))}}})"
            return rf"(\num{{{v:.1f}}})"

        out = out.astype(object)
        trade_mask = out.index.get_level_values("Effects").astype(str) == str(trade_row)
        if trade_mask.any():
            out.loc[trade_mask] = out.loc[trade_mask].map(_fmt_accounting_loss)

    return out
def plot_eu27_gross_net_delta_area_component(
    PANEL_NET: pd.DataFrame,
    PANEL_EU_SIDE: pd.DataFrame,
    *,
    component: str,
    value_mode: str = "absolute",
    delta_base: int = 0,
    delta_min: int | None = -100,
    delta_max: int | None = 100,
    fig_w: float = 9.0,
    fig_h: float = 5.0,
    label_at_zero: bool = True,
    show_value_labels: bool = True,
    show_net_labels: bool = True,
    show_edges: bool = False,
    x_tick_step: int = 20,
    label_step: int = 20,
    label_start_offset: int = 0,
    edge_label_dx: float = 1.15,
    label_min_gva: float = 0.0,
    label_min_jobs: float = 0.0,
    ylim_gva: tuple[float, float] | None = None,
    ylim_jobs: tuple[float, float] | None = None,
    ytick_format: str | None = "k",
    ytick_font: int = 8,
    xtick_font: int = 8,
    value_format: str = "k",
    value_font: int = 10,
    net_label_offset_px: float = 14.0,
    ytitle_font: int = 8,
    legend_loc: str = "lower right",
    legend_x: float = 0.98,
    legend_y: float = 0.14,
    legend_title: str | None = None,
    legend_ncol: int = 1,
    legend_font: int = 9,
    legend_title_font: int = 9,
    hspace: float = 0.10,
    wspace: float = 0.15,
    alpha_fill: float = 0.85,
):
    # Plot one component as two D.3-style panels: GVA and Jobs.
    set_house_style()

    component = str(component).strip()
    mode = str(value_mode).strip().lower()
    absolute_mode = mode in {"absolute", "abs"}
    df_gva = build_eu27_gross_net_delta_area_df(
        PANEL_NET,
        PANEL_EU_SIDE,
        indicator="GVA",
        value_mode=value_mode,
        delta_base=delta_base,
    )
    df_jobs = build_eu27_gross_net_delta_area_df(
        PANEL_NET,
        PANEL_EU_SIDE,
        indicator="Jobs",
        value_mode=value_mode,
        delta_base=delta_base,
    )

    if component not in set(df_gva["Component"].astype(str)):
        available = sorted(df_gva["Component"].dropna().astype(str).unique().tolist())
        raise ValueError(f"Component {component!r} not available. Available components: {available}")

    all_deltas = sorted(set(df_gva["delta_pp"].astype(int)).union(set(df_jobs["delta_pp"].astype(int))))
    if delta_min is None:
        delta_min = int(min(all_deltas))
    if delta_max is None:
        delta_max = int(max(all_deltas))

    deltas = np.array([d for d in all_deltas if int(delta_min) <= int(d) <= int(delta_max)], dtype=int)
    if len(deltas) == 0:
        raise ValueError("No delta values available in the requested range.")

    dx_abs = 5.0 * float(edge_label_dx) if len(deltas) >= 2 else 0.0

    def x_interior(xi: float, x0: float, x1: float) -> float:
        if dx_abs <= 0:
            return xi
        if abs(xi - x0) < 1e-12:
            return xi + dx_abs
        if abs(xi - x1) < 1e-12:
            return xi - dx_abs
        return xi

    def label_mask(xvals: np.ndarray) -> np.ndarray:
        step = max(int(label_step), 1)
        base = int(delta_min) + int(label_start_offset)
        mask = ((xvals.astype(int) - base) % step) == 0
        if len(xvals) > 0:
            mask = mask | (xvals.astype(int) == int(xvals[0]))
            mask = mask | (xvals.astype(int) == int(xvals[-1]))
        if label_at_zero and (0 in xvals.astype(int)):
            mask = mask | (xvals.astype(int) == 0)
        return mask

    def fmt_label(v: float) -> str:
        x = float(v)
        ax = abs(x)
        if ax >= 10.0:
            return f"{x:,.0f}"
        if ax >= 1.0:
            return f"{x:,.1f}"
        if ax == 0.0:
            return "0"
        for decimals in range(1, 8):
            if round(ax, decimals) != 0.0:
                return f"{x:,.{decimals}f}"
        return f"{x:,.8f}"

    def fmt_k(v: float) -> str:
        av = abs(float(v))
        if av >= 1000.0:
            val = float(v) / 1000.0
            s = f"{val:.0f}" if abs(val) >= 10.0 else f"{val:.1f}".rstrip("0").rstrip(".")
            return f"{s}k"
        return f"{float(v):.0f}" if av >= 1.0 else f"{float(v):.2f}".rstrip("0").rstrip(".")

    eu_fc = COLOR_MAP.get("EU27") or COLOR_MAP.get(EU_LABEL) or "#4C78A8"
    indirect_fc = lighten_color(eu_fc, 0.35)
    net_fc = COLOR_MAP.get("GB") or COLOR_MAP.get("UK") or "#5C1F1F"
    adjustment_edge = net_fc
    adjustment_fc = "#F6D6DA"

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(fig_w, fig_h),
        dpi=150,
        sharex=True,
        constrained_layout=False,
        gridspec_kw={"hspace": hspace, "wspace": wspace},
    )

    series = [
        ("direct_gross", eu_fc, None, None),
        ("gross_indirect", indirect_fc, None, None),
        ("upstream_trade_adjustment", adjustment_fc, "///", adjustment_edge),
    ]

    def _plot_one(ax, df, ylabel: str, label_min_abs: float, ylim: tuple[float, float] | None):
        sub = df[df["Component"].astype(str) == component].copy()
        piv = (
            sub.pivot_table(index="delta_pp", values=[s[0] for s in series] + ["net_effect"], aggfunc="sum")
            .reindex(deltas)
            .fillna(0.0)
        )
        x = piv.index.values.astype(float)
        x0, x1 = float(x[0]), float(x[-1])
        pos_base = np.zeros_like(x, dtype=float)
        neg_base = np.zeros_like(x, dtype=float)
        stack_info = []

        for col, color, hatch, edge_override in series:
            vals = piv[col].values.astype(float)
            edge = edge_override if edge_override is not None else darken_color(color, 0.25)
            y_pos = np.where(vals > 0, vals, 0.0)
            y_neg = np.where(vals < 0, vals, 0.0)
            base_pos = pos_base.copy()
            base_neg = neg_base.copy()

            if np.any(np.abs(y_pos) > 1e-12):
                ax.fill_between(
                    x,
                    base_pos,
                    base_pos + y_pos,
                    facecolor=color,
                    edgecolor=edge if hatch else "none",
                    hatch=hatch,
                    alpha=alpha_fill,
                    linewidth=0.0,
                )
                if show_edges and hatch is None:
                    ax.plot(x, base_pos + y_pos, color=edge, linewidth=0.6)
            if np.any(np.abs(y_neg) > 1e-12):
                ax.fill_between(
                    x,
                    base_neg,
                    base_neg + y_neg,
                    facecolor=color,
                    edgecolor=edge if hatch else "none",
                    hatch=hatch,
                    alpha=alpha_fill,
                    linewidth=0.0,
                )
                if show_edges and hatch is None:
                    ax.plot(x, base_neg + y_neg, color=edge, linewidth=0.6)

            stack_info.append((vals, base_pos, base_neg, color))
            pos_base += y_pos
            neg_base += y_neg

        net_vals = piv["net_effect"].values.astype(float)
        ax.plot(x, net_vals, color=net_fc, linewidth=1.5, zorder=6)
        ax.scatter(
            x,
            net_vals,
            facecolors="white",
            edgecolors=net_fc,
            marker="D",
            s=14,
            linewidths=1.8,
            zorder=7,
        )

        if show_value_labels or show_net_labels:
            mask = label_mask(x.astype(int))
            mask = mask & (x.astype(int) != 0)
            for i in np.where(mask)[0]:
                v = net_vals[i]
                if abs(v) < float(label_min_abs):
                    continue
                xi = x_interior(float(x[i]), x0, x1)
                xd, yd = ax.transData.transform((xi, v))
                yd += float(net_label_offset_px) if x[i] >= 0 else -float(net_label_offset_px)
                _, y_label = ax.transData.inverted().transform((xd, yd))

                ax.text(
                    xi,
                    y_label,
                    tex_safe(fmt_label(v)),
                    ha="center",
                    va="top" if x[i] < 0 else "bottom",
                    fontsize=value_font,
                    color=net_fc,
                    clip_on=True,
                    zorder=10,
                )

        ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
        ylabel_out = ylabel if ("$" in str(ylabel) or "\\" in str(ylabel)) else tex_bold(ylabel)
        ax.set_ylabel(ylabel_out, fontsize=ytitle_font, fontweight="bold")
        apply_house_style(ax, grid=True)
        ax.grid(axis="y", alpha=0.2, linestyle=":", linewidth=0.5)
        ax.tick_params(axis="y", labelsize=ytick_font)
        ax.tick_params(axis="x", labelsize=xtick_font)
        if ytick_format == "k":
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: fmt_k(v)))
        ax.set_xlim(float(delta_min) - 2.0, float(delta_max) + 2.0)

        if ylim is not None:
            ax.set_ylim(float(ylim[0]), float(ylim[1]))
        else:
            ymin = float(min(0.0, np.nanmin(neg_base) if len(neg_base) else 0.0, np.nanmin(net_vals) if len(net_vals) else 0.0))
            ymax = float(max(0.0, np.nanmax(pos_base) if len(pos_base) else 0.0, np.nanmax(net_vals) if len(net_vals) else 0.0))
            yrng = ymax - ymin
            ypad = 0.12 * (yrng if yrng > 0 else 1.0)
            ax.set_ylim(ymin - ypad, ymax + ypad)

    if absolute_mode:
        gva_ylabel = r"$\Delta$ EU27 GVA [million EUR]"
        jobs_ylabel = r"$\Delta$ EU27 employment [thousand job-years]"
    else:
        gva_ylabel = r"$\Delta$ EU27 GVA [EUR/EUR]"
        jobs_ylabel = r"$\Delta$ EU27 employment [job-years/mEUR]"

    _plot_one(axes[0], df_gva, gva_ylabel, float(label_min_gva), ylim_gva)
    _plot_one(axes[1], df_jobs, jobs_ylabel, float(label_min_jobs), ylim_jobs)

    for ax in axes:
        ax.set_xlabel(tex_bold("Policy shock (delta pp)"), fontsize=ytick_font)
        if x_tick_step is None or int(x_tick_step) <= 0:
            xticks = deltas.astype(int)
        else:
            step = int(x_tick_step)
            tick_lo = int(np.floor(float(delta_min) / step) * step)
            tick_hi = int(np.ceil(float(delta_max) / step) * step)
            xticks = np.arange(tick_lo, tick_hi + step, step, dtype=int)
            xticks = xticks[(xticks >= int(delta_min)) & (xticks <= int(delta_max))]
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(int(t)) for t in xticks], fontsize=xtick_font)

    handles = [
        mpatches.Patch(facecolor=eu_fc, edgecolor=darken_color(eu_fc, 0.25), label=tex_safe("Direct")),
        mpatches.Patch(facecolor=indirect_fc, edgecolor=darken_color(indirect_fc, 0.25), label=tex_safe("Indirect")),
        mpatches.Patch(facecolor=adjustment_fc, edgecolor=adjustment_edge, hatch="///", linewidth=0.6, label=tex_safe("Upstream trade effects")),
        plt.Line2D([0], [0], color=net_fc, marker="D", markerfacecolor="white", markeredgecolor=net_fc, markeredgewidth=1.8, linewidth=1.5, markersize=4.0, label=tex_safe("Net effect")),
    ]
    if str(legend_loc).strip().lower() == "lower right":
        legend_y_eff = float(legend_y) if float(legend_y) >= 0 else 0.14
    else:
        legend_y_eff = 0.98 + float(legend_y)
        if legend_y >= 0:
            legend_y_eff = min(0.98, max(0.05, legend_y_eff))
    fig.legend(
        handles=handles,
        title=tex_bold(legend_title) if legend_title else None,
        loc=legend_loc,
        bbox_to_anchor=(float(legend_x), float(legend_y_eff)),
        bbox_transform=fig.transFigure,
        borderaxespad=0.2,
        fontsize=legend_font,
        title_fontsize=legend_title_font,
        ncol=legend_ncol,
        frameon=False,
        framealpha=1.0,
        handlelength=1.1,
        handletextpad=0.4,
        labelspacing=0.3,
        borderpad=0.4,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.12, hspace=hspace, wspace=wspace)
    try:
        fig.set_layout_engine("none")
    except Exception:
        pass

    return fig, axes, {"GVA": df_gva, "Jobs": df_jobs}


def compute_eu27_gross_net_delta_area_component_ylims(
    *,
    PANEL_NET: pd.DataFrame,
    PANEL_EU_SIDE: pd.DataFrame,
    components,
    value_mode: str = "absolute",
    delta_base: int = 0,
    delta_min: int | None = -100,
    delta_max: int | None = 100,
    pad_frac: float = 0.12,
) -> dict[str, tuple[float, float]]:
    # Compute common y-axis limits across component delta-area plots.
    series_cols = [
        "direct_gross",
        "gross_indirect",
        "upstream_trade_adjustment",
    ]
    components = [str(c).strip() for c in components]
    out: dict[str, tuple[float, float]] = {}

    for indicator in ("GVA", "Jobs"):
        df = build_eu27_gross_net_delta_area_df(
            PANEL_NET,
            PANEL_EU_SIDE,
            indicator=indicator,
            value_mode=value_mode,
            delta_base=delta_base,
        )

        all_deltas = sorted(df["delta_pp"].astype(int).unique().tolist())
        if not all_deltas:
            out[indicator] = (-1.0, 1.0)
            continue
        lo = int(min(all_deltas)) if delta_min is None else int(delta_min)
        hi = int(max(all_deltas)) if delta_max is None else int(delta_max)
        deltas = np.array([d for d in all_deltas if lo <= int(d) <= hi], dtype=int)

        ymin = 0.0
        ymax = 0.0
        for component in components:
            sub = df[df["Component"].astype(str) == component].copy()
            if sub.empty:
                continue
            piv = (
                sub.pivot_table(index="delta_pp", values=series_cols + ["net_effect"], aggfunc="sum")
                .reindex(deltas)
                .fillna(0.0)
            )
            pos_base = np.zeros(len(piv), dtype=float)
            neg_base = np.zeros(len(piv), dtype=float)
            for col in series_cols:
                vals = piv[col].values.astype(float)
                pos_base += np.where(vals > 0, vals, 0.0)
                neg_base += np.where(vals < 0, vals, 0.0)

            net_vals = piv["net_effect"].values.astype(float)
            ymin = float(min(ymin, np.nanmin(neg_base), np.nanmin(net_vals)))
            ymax = float(max(ymax, np.nanmax(pos_base), np.nanmax(net_vals)))

        yrng = ymax - ymin
        ypad = float(pad_frac) * (yrng if yrng > 0 else 1.0)
        out[indicator] = (ymin - ypad, ymax + ypad)

    return out


def plot_and_save_eu27_gross_net_delta_area_component(
    *,
    PANEL_NET: pd.DataFrame,
    PANEL_EU_SIDE: pd.DataFrame,
    component: str,
    filename: str,
    save: bool | None = None,
    close: bool | None = None,
    **kw,
):
    fig, _, _ = plot_eu27_gross_net_delta_area_component(
        PANEL_NET,
        PANEL_EU_SIDE,
        component=component,
        **kw,
    )
    return save_and_close(fig=fig, filename=filename, save=save, close=close)


def plot_and_save_d3_offshoring_figures(
    *,
    PANEL_SUB: pd.DataFrame,
    split_row=(),
    save: bool | None = None,
    close: bool | None = None,
):
    """Save the paper's D.3 offshoring figures for the calibrated component set."""
    component_plots = [
        ("Turbine", "D.3_offshoring_Turbine.pdf", 50, 100, 600, 3000),
        ("BoP", "D.3_offshoring_BoP.pdf", 50, 90, 600, 1200),
        ("Devex+O&M", "D.3_offshoring_Devex_OM.pdf", 50, 50, 700, 1500),
        ("Installation", "D.3_offshoring_Installation.pdf", 50, 50, 400, 750),
        ("Total", "D.3_offshoring_Total.pdf", 150, 400, 1500, 8000),
    ]
    outs = []
    for component, filename, min_gva_d, min_gva_i, min_jobs_d, min_jobs_i in component_plots:
        outs.append(plot_and_save_global_totals_by_region_direct_indirect(
            PANEL_SUB=PANEL_SUB,
            components=[component],
            split_row=split_row,
            filename=filename,
            label_min_gva_direct=min_gva_d,
            label_min_gva_indirect=min_gva_i,
            label_min_jobs_direct=min_jobs_d,
            label_min_jobs_indirect=min_jobs_i,
            save=save,
            close=close,
        ))
    return outs


def plot_and_save_eu27_component_delta_area_figures(
    *,
    PANEL_NET: pd.DataFrame,
    PANEL_EU_SIDE: pd.DataFrame,
    save: bool | None = None,
    close: bool | None = None,
):
    """Save the paper's 3.4 EU27 gross/net delta-area component figures."""
    component_plots = [
        ("Turbine", "3.4_eu27_gross_net_delta_area_Turbine.pdf"),
        ("BoP", "3.4_eu27_gross_net_delta_area_BoP.pdf"),
        ("Devex+O&M", "3.4_eu27_gross_net_delta_area_Devex_OM.pdf"),
        ("Installation", "3.4_eu27_gross_net_delta_area_Installation.pdf"),
    ]
    ylims = compute_eu27_gross_net_delta_area_component_ylims(
        PANEL_NET=PANEL_NET,
        PANEL_EU_SIDE=PANEL_EU_SIDE,
        components=[component for component, _ in component_plots],
    )
    outs = []
    for component, filename in component_plots:
        outs.append(plot_and_save_eu27_gross_net_delta_area_component(
            PANEL_NET=PANEL_NET,
            PANEL_EU_SIDE=PANEL_EU_SIDE,
            component=component,
            filename=filename,
            ylim_gva=ylims["GVA"],
            ylim_jobs=ylims["Jobs"],
            save=save,
            close=close,
        ))
    return outs


def plot_and_save_eu27_component_multiplier_figures(
    *,
    PANEL_NET: pd.DataFrame,
    PANEL_GROSS: pd.DataFrame,
    delta_resil: int = 100,
    delta_out: int = -100,
    save: bool | None = None,
    close: bool | None = None,
):
    """Save the paper's 3.5 component gross/net multiplier figures."""
    specs = [
        ("GVA", int(delta_resil), "3.5_eu27_delta_mult_gva_split_comp.pdf"),
        ("Jobs", int(delta_resil), "3.5_eu27_delta_mult_jobs_split_comp.pdf"),
        ("GVA", int(delta_out), "3.5_eu27_delta_mult_gva_split_comp_neg.pdf"),
        ("Jobs", int(delta_out), "3.5_eu27_delta_mult_jobs_split_comp_neg.pdf"),
    ]
    outs = []
    for indicator, delta_target, filename in specs:
        outs.append(plot_and_save_eu27_gross_net_multiplier_decomposition(
            PANEL_NET=PANEL_NET,
            PANEL_GROSS=PANEL_GROSS,
            indicator=indicator,
            delta_target=delta_target,
            filename=filename,
            save=save,
            close=close,
        ))
    return outs


def plot_and_save_eu27_turbine_subcomponent_multiplier_figures(
    *,
    PANEL_NET: pd.DataFrame,
    PANEL_GROSS: pd.DataFrame,
    delta_resil: int = 100,
    delta_out: int = -100,
    save: bool | None = None,
    close: bool | None = None,
):
    """Save the paper's 3.6 turbine subcomponent gross/net multiplier figures."""
    turbine_order = ["Blades", "Hub", "Drivetrain", "Nacelle", "Tower"]
    specs = [
        ("GVA", int(delta_resil), "3.6_eu27_delta_mult_gva_split_comp_L0.pdf"),
        ("Jobs", int(delta_resil), "3.6_eu27_delta_mult_jobs_split_comp_L0.pdf"),
        ("GVA", int(delta_out), "3.6_eu27_delta_mult_gva_split_comp_L0_neg.pdf"),
        ("Jobs", int(delta_out), "3.6_eu27_delta_mult_jobs_split_comp_L0_neg.pdf"),
    ]
    outs = []
    for indicator, delta_target, filename in specs:
        outs.append(plot_and_save_eu27_gross_net_multiplier_decomposition(
            PANEL_NET=PANEL_NET,
            PANEL_GROSS=PANEL_GROSS,
            indicator=indicator,
            delta_target=delta_target,
            components_keep=turbine_order,
            component_order=turbine_order,
            filename=filename,
            save=save,
            close=close,
        ))
    return outs


def build_resilience_region_composition_share_df(
    PANEL_SUB: pd.DataFrame,
    *,
    indicator: str = "GVA",
    delta_target: int = 100,
    scenario: str | None = "baseline",
    scenario_col: str = "scenario",
    delta_col: str = "delta_pp",
    region_col: str = "Region",
    comp_col: str = "Component",
    subcomp_col: str = "Subcomponent",
    gva_col: str = "abs_gva_total",
    jobs_col: str = "abs_jobs_total",
    region_order: tuple[str, ...] = (EU_LABEL, UK_LABEL, ROE_LABEL, ROW_LABEL),
    components_keep: tuple[str, ...] | list[str] | set[str] | None = None,
    drop_total: bool = True,
) -> pd.DataFrame:
    # Build regional composition shares by subcomponent for one resilience delta.
    df = PANEL_SUB.copy()
    if scenario is not None and scenario_col in df.columns:
        df = df[df[scenario_col].astype(str) == str(scenario)].copy()
    if df.empty:
        raise ValueError("No data after scenario filter.")

    ind_u = str(indicator).upper().strip()
    if ind_u == "GVA":
        metric_col = gva_col
        indicator_label = "GVA"
    elif ind_u == "JOBS":
        metric_col = jobs_col
        indicator_label = "Employment"
    else:
        raise ValueError("indicator must be 'GVA' or 'Jobs'.")

    for col in (delta_col, region_col, comp_col, subcomp_col, metric_col):
        if col not in df.columns:
            raise KeyError(f"Missing column {col!r} in PANEL_SUB.")

    df[delta_col] = pd.to_numeric(df[delta_col], errors="coerce").fillna(0).astype(int)
    df[comp_col] = df[comp_col].astype(str).str.strip()
    df[subcomp_col] = df[subcomp_col].astype(str).str.strip()
    df[region_col] = df[region_col].astype(str).str.strip()
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce").fillna(0.0)

    if components_keep:
        keep = {str(c).strip() for c in components_keep}
        df = df[df[comp_col].isin(keep) | df[subcomp_col].isin(keep)].copy()
        if df.empty:
            raise ValueError(f"No rows after components_keep filter: {sorted(keep)}")

    if drop_total:
        comp_cf = df[comp_col].str.casefold()
        sub_cf = df[subcomp_col].str.casefold()
        df = df[
            ~comp_cf.isin({"all", "total"})
            & ~sub_cf.isin({"all", "total"})
            & ~comp_cf.eq(sub_cf)
        ].copy()

    df = df[df[delta_col] == int(delta_target)].copy()
    if df.empty:
        raise ValueError(f"No data for delta_target={delta_target}.")

    agg = df.groupby([comp_col, subcomp_col, region_col], as_index=False)[metric_col].sum()
    total = (
        agg.groupby([comp_col, subcomp_col], as_index=False)[metric_col]
        .sum()
        .rename(columns={metric_col: "_global"})
    )
    out = agg.merge(total, on=[comp_col, subcomp_col], how="left")
    out["share_pct"] = np.where(
        out["_global"].astype(float) != 0.0,
        100.0 * out[metric_col].astype(float) / out["_global"].astype(float),
        0.0,
    )
    out["indicator"] = indicator_label

    keys = out[[comp_col, subcomp_col]].drop_duplicates()
    rows = []
    base_regions = list(region_order)
    for _, key in keys.iterrows():
        comp = key[comp_col]
        sub = key[subcomp_col]
        d = out[(out[comp_col] == comp) & (out[subcomp_col] == sub)]
        present = d[region_col].astype(str).drop_duplicates().tolist()
        regions = base_regions + [r for r in present if r not in set(base_regions)]
        for region in regions:
            m = d[d[region_col].astype(str) == str(region)]
            rows.append({
                "Component": comp,
                "Subcomponent": sub,
                "Region": region,
                "share_pct": 0.0 if m.empty else float(m["share_pct"].sum()),
                "value": 0.0 if m.empty else float(m[metric_col].sum()),
                "global_value": 0.0 if m.empty else float(m["_global"].iloc[0]),
                "indicator": indicator_label,
                "delta_pp": int(delta_target),
            })
    return pd.DataFrame(rows)


def build_eu27_resilience_capture_share_df(
    *,
    indicator: str = "GVA",
    delta_base: int = 0,
    delta_target: int = 100,
    scenario: str | None = "baseline",
    scenario_col: str = "scenario",
    delta_col: str = "delta_pp",
    region_col: str = "Region",
    comp_col: str = "Component",
    subcomp_col: str = "Subcomponent",
    gva_col: str = "abs_gva_total",
    jobs_col: str = "abs_jobs_total",
    eu_label: str = EU_LABEL,
    components_keep: tuple[str, ...] | list[str] | set[str] | None = None,
    drop_total: bool = True,
) -> pd.DataFrame:
    # Build EU27 base, resilience delta, and RoW leakage shares by subcomponent.
    df = PANEL_SUB.copy()
    if scenario is not None and scenario_col in df.columns:
        df = df[df[scenario_col].astype(str) == str(scenario)].copy()
    if df.empty:
        raise ValueError("No data after scenario filter.")

    ind_u = str(indicator).upper().strip()
    if ind_u == "GVA":
        metric_col = gva_col
        indicator_label = "GVA"
    elif ind_u == "JOBS":
        metric_col = jobs_col
        indicator_label = "Employment"
    else:
        raise ValueError("indicator must be 'GVA' or 'Jobs'.")

    for col in (delta_col, region_col, comp_col, subcomp_col, metric_col):
        if col not in df.columns:
            raise KeyError(f"Missing column {col!r} in PANEL_SUB.")

    df[delta_col] = pd.to_numeric(df[delta_col], errors="coerce").fillna(0).astype(int)
    df[comp_col] = df[comp_col].astype(str).str.strip()
    df[subcomp_col] = df[subcomp_col].astype(str).str.strip()
    df[region_col] = df[region_col].astype(str).str.strip()
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce").fillna(0.0)

    if components_keep:
        keep = {str(c).strip() for c in components_keep}
        comp_match = df[comp_col].isin(keep)
        subcomp_match = df[subcomp_col].isin(keep)
        df = df[comp_match | subcomp_match].copy()
        if df.empty:
            raise ValueError(
                f"No rows after components_keep filter. Checked both {comp_col!r} and {subcomp_col!r}: {sorted(keep)}"
            )

    if drop_total:
        comp_cf = df[comp_col].str.casefold()
        sub_cf = df[subcomp_col].str.casefold()
        df = df[
            ~comp_cf.isin({"all", "total"})
            & ~sub_cf.isin({"all", "total"})
            & ~comp_cf.eq(sub_cf)
        ].copy()

    needed_deltas = {int(delta_base), int(delta_target)}
    missing = needed_deltas - set(df[delta_col].unique())
    if missing:
        raise KeyError(f"Missing delta(s) {sorted(missing)} in {delta_col}.")

    agg = (
        df[df[delta_col].isin(needed_deltas)]
        .groupby([delta_col, comp_col, subcomp_col, region_col], as_index=False)[metric_col]
        .sum()
    )
    total = (
        agg.groupby([delta_col, comp_col, subcomp_col], as_index=False)[metric_col]
        .sum()
        .rename(columns={metric_col: "_global"})
    )
    eu = (
        agg[agg[region_col] == eu_label]
        .groupby([delta_col, comp_col, subcomp_col], as_index=False)[metric_col]
        .sum()
        .rename(columns={metric_col: "_eu27"})
    )
    share = total.merge(eu, on=[delta_col, comp_col, subcomp_col], how="left").fillna({"_eu27": 0.0})
    share["eu27_share"] = (
        share["_eu27"].astype(float)
        .div(share["_global"].replace(0.0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=0.0, upper=1.0)
        * 100.0
    )

    piv = (
        share.pivot_table(
            index=[comp_col, subcomp_col],
            columns=delta_col,
            values="eu27_share",
            aggfunc="first",
        )
        .fillna(0.0)
    )

    out = piv.reset_index().rename(columns={
        comp_col: "Component",
        subcomp_col: "Subcomponent",
        int(delta_base): "EU27_base_pct",
        int(delta_target): "EU27_resil_pct",
    })
    if "EU27_base_pct" not in out.columns:
        out["EU27_base_pct"] = 0.0
    if "EU27_resil_pct" not in out.columns:
        out["EU27_resil_pct"] = 0.0

    out["delta_resilience_pct"] = out["EU27_resil_pct"] - out["EU27_base_pct"]
    out["RoW_leakage_pct"] = 100.0 - out["EU27_resil_pct"]
    out["indicator"] = indicator_label
    return out
