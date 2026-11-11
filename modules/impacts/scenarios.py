"""Sourcing reallocation scenarios, resilience accounting, and scenario figures."""

from modules.mrio_analysis.mrio import (
    build_y_shock_levels,
    make_final_demand_vector,
    _apply_zero_output_industry_fallbacks,
)
from modules.mrio_analysis.industrial import (
    run_industrial,
    load_bridges,
    get_ind_root,
    allocate_turbine_costs,
    allocate_farm_costs,
    flows_to_cost_df,
    add_3_levels,
)
from modules.config import HELPERS_DIR, CORE_DEPLOYMENT

from typing import Dict, Iterable
from dataclasses import dataclass
import builtins
import pandas as pd
import numpy as np
import re
import tempfile
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from modules.country import COLOR_MAP, CODE_TO_NAME, EUROPE_CODES, EU27
from modules.latex_style import (
    apply_house_style,
    save_and_close,
    set_house_style,
    tex_safe,
    tex_bold,
    get_text_color,
)


# A low-cost supplier is assumed to charge (1 - m) times the EU price.
# Each sourcing scenario calibrates and values equivalent quantities at its
# own markup.  REFERENCE_MARKUP remains only as a backwards-compatible default
# for helper signatures; scenario execution passes its explicit markup.
REFERENCE_MARKUP = 0.30
PRICE_ADJUSTMENT_MODEL_VERSION = "quantity-recalibrated-mu-v1"
# Donor-side sourcing weights are part of the scenario's economic provenance:
# they use baseline total GVA first, with capped-availability fallback.
SOURCING_ALLOCATION_MODEL_VERSION = "baseline-total-gva-donor-eu27-target-v1"


def _reference_price_factor(country: str, low_income, *, markup: float = REFERENCE_MARKUP) -> float:
    """Return the relative price used for equivalent-production quantities."""
    country_norm = str(country).strip().upper().replace("UK", "GB").replace("ROW", "ROW")
    low_income_norm = {
        str(value).strip().upper().replace("UK", "GB")
        for value in (low_income or ())
    }
    factor = 1.0 - float(markup) if country_norm in low_income_norm else 1.0
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError("Reference markup must produce strictly positive relative prices.")
    return float(factor)


def _scenario_markup(value: float | None) -> float:
    """Validate the single markup used for one scenario calibration."""
    markup = float(0.0 if value is None else value)
    if not np.isfinite(markup) or not 0.0 <= markup < 1.0:
        raise ValueError("Scenario markup must satisfy 0 <= mu < 1.")
    return markup


def _reference_value_by_origin(q_by_origin: dict, low_income, *, markup: float = REFERENCE_MARKUP) -> dict:
    return {
        country: float(quantity) * _reference_price_factor(country, low_income, markup=markup)
        for country, quantity in q_by_origin.items()
    }


def _reference_transfer_value(
    quantity: float,
    by_origin: dict,
    origins,
    low_income,
    *,
    markup: float = REFERENCE_MARKUP,
) -> float:
    quantity = float(quantity)
    available_quantity = float(sum(float(by_origin.get(origin, 0.0)) for origin in origins))
    if available_quantity <= 0.0:
        return 0.0
    available_value = float(sum(
        float(by_origin.get(origin, 0.0))
        * _reference_price_factor(origin, low_income, markup=markup)
        for origin in origins
    ))
    return quantity * available_value / available_quantity


def _reference_quantity_preserve_total_guard(preserve_total: bool, delta_pp: float) -> None:
    if bool(preserve_total) and abs(float(delta_pp)) > 1e-12:
        raise ValueError(
            "preserve_total=True is incompatible with quantity-based sourcing: "
            "monetary normalization would violate equivalent-quantity conservation. "
            "Use preserve_total=False."
        )


def _adjust_transfer_for_price_category(
    amount: float,
    *,
    source_low_income: bool,
    destination_low_income: bool,
    discount: float,
) -> float:
    """Convert a quantity-equivalent transfer into nominal expenditure.

    ``amount`` is the nominal expenditure at the source price.  Transfers
    between equal price categories are unchanged.  A low-income source being
    replaced by an EU source is divided by ``1 - discount``; an EU source
    moved to a low-income destination is multiplied by ``1 - discount``.
    """
    amount = float(amount)
    discount = float(discount or 0.0)
    if discount < 0.0 or discount >= 1.0:
        raise ValueError("eu_spend_reduction_pct must satisfy 0 <= discount < 1")
    if not discount or bool(source_low_income) == bool(destination_low_income):
        return amount
    if source_low_income and not destination_low_income:
        return amount / (1.0 - discount)
    return amount * (1.0 - discount)


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
    from modules.config import CLOSE_FIGURES, IMG_ROOT, SAVE_FIGURES, project_relative_path

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
        print(f"Saved to: {project_relative_path(outpath)}")

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


MRIO_RESULT_COLUMNS = [
    "Deployment", "Country", "Year", "Sector",
    "Output_Direct", "Output_Indirect", "Output_Total",
    "GVA_Direct", "GVA_Indirect", "GVA_Total",
    "Income_Direct", "Income_Indirect", "Income_Total",
    "Jobs_Direct", "Jobs_Indirect", "Jobs_Total",
]


def build_component_hierarchy_crosswalk(
    final_cost_df: pd.DataFrame,
    *,
    value_col: str = "value_alloc",
    require_nonzero: bool = True,
) -> pd.DataFrame:
    """Build and validate the canonical terminal-observation hierarchy.

    The L0 pair is the terminal observation supplied by the industrial
    allocation. L1 and L2 are read from the same source rows; no hierarchy is
    inferred from display names. Parent rows that coexist with children are
    rejected so they cannot be counted twice.
    """
    required = [
        value_col,
        "component_group_L0", "sub_component_L0",
        "component_group_L1", "sub_component_L1",
        "component_group_L2", "sub_component_L2",
    ]
    missing = [c for c in required if c not in final_cost_df.columns]
    if missing:
        raise KeyError(f"final_cost_df is missing hierarchy columns: {missing}")

    df = final_cost_df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)
    if require_nonzero:
        df = df[df[value_col].abs() > 1e-12].copy()
    if df.empty:
        raise ValueError("No nonzero allocation records are available for the hierarchy crosswalk.")

    hierarchy_cols = [
        "component_group_L0", "sub_component_L0",
        "component_group_L1", "sub_component_L1",
        "component_group_L2", "sub_component_L2",
    ]
    for col in hierarchy_cols:
        df[col] = df[col].astype(str).str.strip()

    leaf_cols = ["component_group_L0", "sub_component_L0"]
    crosswalk = df[leaf_cols + hierarchy_cols[2:]].drop_duplicates().copy()
    counts = crosswalk.groupby(leaf_cols, dropna=False).size()
    ambiguous = counts[counts > 1]
    if not ambiguous.empty:
        examples = crosswalk.merge(ambiguous.rename("n"), on=leaf_cols, how="inner")
        raise ValueError(
            "A terminal L0 observation maps to multiple parent hierarchies: "
            f"{examples.to_dict('records')}"
        )

    crosswalk = crosswalk.sort_values(leaf_cols, kind="stable").reset_index(drop=True)
    crosswalk.insert(0, "leaf_id", np.arange(len(crosswalk), dtype=int))

    # Detect a genuine source-record parent alongside its children.  Matching
    # labels across levels are not evidence of double counting: e.g. each BoP
    # source observation remains ``BoP / Array cables`` at L1 and only rolls
    # up to ``BoP / BoP`` at L2.  A duplicate parent is instead a *separate
    # source observation* such as ``Turbine / Turbine`` coexisting with raw
    # Blades, Drivetrain, Hub, Nacelle, or Tower observations.
    #
    # Prefer the industrial source columns retained by run_industrial.  The
    # L0 fallback keeps the validator usable for deliberately minimal inputs
    # in tests and for backwards-compatible callers.
    source_group_col = "component_group" if "component_group" in df.columns else "component_group_L0"
    source_sub_col = "sub_component" if "sub_component" in df.columns else "sub_component_L0"
    source_records = df[
        [source_group_col, source_sub_col, "component_group_L1"]
    ].drop_duplicates().copy()
    source_records.columns = ["_source_group", "_source_sub", "_l1_group"]
    for l1_group, group in source_records.groupby("_l1_group", dropna=False):
        # A source record is an aggregate only when it explicitly names the
        # L1 parent as both its component and subcomponent.  Ordinary BoP
        # leaves do not meet this condition because their source subcomponent
        # remains Array cables, Foundation, etc.
        parent_rows = group[
            group["_source_group"].eq(str(l1_group))
            & group["_source_sub"].eq(str(l1_group))
        ]
        if parent_rows.empty:
            continue
        children = group[
            ~(
                group["_source_group"].eq(str(l1_group))
                & group["_source_sub"].eq(str(l1_group))
            )
        ]
        if not children.empty:
            raise ValueError(
                "A source aggregate is present together with its terminal children: "
                f"parent={parent_rows.to_dict('records')}; "
                f"children={children.to_dict('records')}"
            )

    expenditure_by_leaf = df.groupby(leaf_cols, dropna=False)[value_col].sum()
    if not np.isclose(
        float(expenditure_by_leaf.sum()),
        float(df[value_col].sum()),
        atol=1e-6,
        rtol=1e-10,
    ):
        raise ValueError("Terminal hierarchy expenditure does not equal source expenditure.")

    for level in LEVELS:
        cols = [f"component_group_L{level}", f"sub_component_L{level}"]
        hierarchy_total = df.groupby(cols, dropna=False)[value_col].sum().sum()
        if not np.isclose(float(hierarchy_total), float(df[value_col].sum()), atol=1e-6, rtol=1e-10):
            raise ValueError(f"Hierarchy level L{level} does not conserve total expenditure.")

    return crosswalk


def validate_component_hierarchy_crosswalk(
    final_cost_df: pd.DataFrame,
    crosswalk: pd.DataFrame | None = None,
    *,
    value_col: str = "value_alloc",
) -> dict:
    """Return conservation diagnostics for a canonical component crosswalk."""
    crosswalk = crosswalk if crosswalk is not None else build_component_hierarchy_crosswalk(
        final_cost_df, value_col=value_col
    )
    leaf_cols = ["component_group_L0", "sub_component_L0"]
    df = final_cost_df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)
    df = df[df[value_col].abs() > 1e-12].copy()
    source_total = float(df[value_col].sum())
    leaf_total = float(
        df.groupby(leaf_cols, dropna=False)[value_col].sum().sum()
    )
    level_totals = {}
    for level in LEVELS:
        cols = [f"component_group_L{level}", f"sub_component_L{level}"]
        level_totals[level] = float(df.groupby(cols, dropna=False)[value_col].sum().sum())
    return {
        "n_terminal_observations": int(len(crosswalk)),
        "source_total": source_total,
        "leaf_total": leaf_total,
        "level_totals": level_totals,
        "conserved": bool(
            np.isclose(source_total, leaf_total, atol=1e-6, rtol=1e-10)
            and all(np.isclose(source_total, x, atol=1e-6, rtol=1e-10) for x in level_totals.values())
        ),
    }


def _mrio_origin_labels(index: pd.Index) -> pd.Index:
    if isinstance(index, pd.MultiIndex):
        for candidate in ("origin_region", "origin", "region", "country", "from_region", "src_region"):
            if candidate in index.names:
                return pd.Index(index.get_level_values(candidate).astype(str))
        return pd.Index(index.get_level_values(0).astype(str))
    return pd.Index(index.astype(str))


def _empty_mrio_result_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=MRIO_RESULT_COLUMNS)


def _aggregate_detailed_mrio_results(
    detailed: pd.DataFrame,
    crosswalk: pd.DataFrame,
    *,
    year: int,
) -> tuple[dict, dict]:
    """Reconstruct legacy L0/L1/L2 dictionaries from detailed impacts."""
    metric_cols = MRIO_RESULT_COLUMNS[4:]
    group_cols = ["Deployment", "Country", "Year", "Sector"]

    def _sum(mask: pd.Series) -> pd.DataFrame:
        if not bool(mask.any()):
            return _empty_mrio_result_frame()
        out = detailed.loc[mask, group_cols + metric_cols].groupby(group_cols, as_index=False)[metric_cols].sum()
        return out[MRIO_RESULT_COLUMNS]

    by_lvl = {}
    by_sub = {}
    for level in LEVELS:
        cg_col = f"component_group_L{level}"
        sc_col = f"sub_component_L{level}"
        by_lvl[level] = {"Total": _sum(pd.Series(True, index=detailed.index))}
        by_sub[level] = {("Total", "ALL"): _sum(pd.Series(True, index=detailed.index))}
        for component in crosswalk[cg_col].drop_duplicates().tolist():
            leaf_ids = set(crosswalk.loc[crosswalk[cg_col].eq(component), "leaf_id"])
            by_lvl[level][str(component)] = _sum(detailed["leaf_id"].isin(leaf_ids))
        pairs = crosswalk[[cg_col, sc_col]].drop_duplicates().itertuples(index=False, name=None)
        for component, subcomponent in pairs:
            leaf_ids = set(
                crosswalk.loc[
                    crosswalk[cg_col].eq(component) & crosswalk[sc_col].eq(subcomponent),
                    "leaf_id",
                ]
            )
            by_sub[level][(str(component), str(subcomponent))] = _sum(detailed["leaf_id"].isin(leaf_ids))
    return by_lvl, by_sub


def evaluate_detailed_mrio_results(
    final_cost_df_cf: pd.DataFrame,
    *,
    L1_MRIO: pd.DataFrame,
    VA_MRIO: pd.Series,
    LB_MRIO: pd.Series,
    W_MRIO: pd.Series | None = None,
    core: list[str] | tuple[str, ...],
    mrio: pd.DataFrame,
    annualize: bool = True,
    lifetime_years: int = 25,
    year: int = 2023,
    eu_source_countries: tuple[str, ...] | list[str] | set[str] = EU27,
) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    """Evaluate terminal hierarchy observations and aggregate afterward.

    Final demand is retained in two exhaustive source partitions (EU27 and
    non-EU27).  This is sufficient for the signed EU27-side decomposition and
    is algebraically identical to evaluating every source country separately,
    while avoiding one Leontief column per origin country.
    """
    if mrio is None or L1_MRIO is None or VA_MRIO is None or LB_MRIO is None:
        raise ValueError("L1_MRIO, VA_MRIO, LB_MRIO, and mrio are required.")
    corrected = _apply_zero_output_industry_fallbacks(final_cost_df_cf, mrio, verbose=False)
    crosswalk = build_component_hierarchy_crosswalk(corrected)
    validate_component_hierarchy_crosswalk(corrected, crosswalk)

    leaves = crosswalk.to_dict("records")
    mrio_index = L1_MRIO.columns
    if not L1_MRIO.index.equals(mrio_index):
        raise ValueError("L1_MRIO must have aligned row and column indices.")
    index = mrio_index
    origin_labels = _mrio_origin_labels(index)
    origin_values = pd.Index(origin_labels.unique())
    va = pd.to_numeric(VA_MRIO, errors="coerce").reindex(index).fillna(0.0).to_numpy(dtype=float)
    lb = pd.to_numeric(LB_MRIO, errors="coerce").reindex(index).fillna(0.0).to_numpy(dtype=float)
    if W_MRIO is not None:
        # W_MRIO is retained as the API name for compatibility.  It now
        # contains FTE coefficients measured as FTE per million EUR output.
        fte_coefficients = (
            pd.to_numeric(W_MRIO, errors="coerce")
            .reindex(index)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
    else:
        fte_coefficients = None

    response = L1_MRIO.to_numpy(dtype=float)
    # ``make_final_demand_vector`` uses gross output only for its zero-output
    # diagnostic.  It is invariant across terminal leaves and source origins,
    # so compute it once for the batched detailed evaluation.
    gross_output = pd.to_numeric(mrio.sum(axis=1), errors="coerce")
    eu_source_set = {
        str(country).strip().upper().replace("UK", "GB")
        for country in (eu_source_countries or ())
    }
    rows = []
    for deployment in core:
        demand_columns = []
        for leaf in leaves:
            leaf_df = corrected[
                (corrected["deployment_country"].astype(str) == str(deployment))
                & corrected["component_group_L0"].astype(str).eq(str(leaf["component_group_L0"]))
                & corrected["sub_component_L0"].astype(str).eq(str(leaf["sub_component_L0"]))
            ]
            source_norm = (
                leaf_df["origin_country"].astype(str).str.strip().str.upper()
                .replace({"UK": "GB"})
            )
            for source_country, source_mask in (
                ("__EU27_SOURCE__", source_norm.isin(eu_source_set)),
                ("__NON_EU_SOURCE__", ~source_norm.isin(eu_source_set)),
            ):
                if not bool(source_mask.any()):
                    continue
                source_df = leaf_df.loc[source_mask]
                y = make_final_demand_vector(
                    mrio,
                    source_df,
                    str(deployment),
                    component_group=leaf["component_group_L0"],
                    sub_component=leaf["sub_component_L0"],
                    level=0,
                    lifetime_years=lifetime_years,
                    mode="annual" if annualize else "total",
                    gross_output=gross_output,
                )
                demand_columns.append(
                    (int(leaf["leaf_id"]), source_country,
                     y.reindex(index).fillna(0.0).to_numpy(dtype=float))
                )
        if not demand_columns:
            continue
        Y = np.column_stack([column[2] for column in demand_columns])
        total = response @ Y
        direct = Y
        indirect = total - direct
        gva = total * va[:, None]
        gva_direct = direct * va[:, None]
        income = total * lb[:, None]
        income_direct = direct * lb[:, None]
        if fte_coefficients is None:
            jobs = np.zeros_like(income)
            jobs_direct = np.zeros_like(income_direct)
        else:
            # Final demand and MRIO output are in EUR; W_MRIO is FTE per
            # million EUR.  Convert the output base before applying it.
            jobs = total * (fte_coefficients / 1_000_000.0)[:, None]
            jobs_direct = direct * (fte_coefficients / 1_000_000.0)[:, None]

        for j, (leaf_id, source_country, _) in enumerate(demand_columns):
            leaf = next(item for item in leaves if int(item["leaf_id"]) == leaf_id)
            for country in origin_values:
                mask = origin_labels == country
                if not bool(mask.any()):
                    continue
                rows.append({
                    "Deployment": str(deployment),
                    "SourceCountry": str(source_country),
                    "Country": str(country),
                    "Year": int(year),
                    "Sector": "ALL",
                    "leaf_id": int(leaf_id),
                    **{col: leaf[col] for col in crosswalk.columns if col != "leaf_id"},
                    "Output_Direct": float(direct[mask, j].sum()),
                    "Output_Indirect": float(indirect[mask, j].sum()),
                    "Output_Total": float(total[mask, j].sum()),
                    "GVA_Direct": float(gva_direct[mask, j].sum()),
                    "GVA_Indirect": float((gva - gva_direct)[mask, j].sum()),
                    "GVA_Total": float(gva[mask, j].sum()),
                    "Income_Direct": float(income_direct[mask, j].sum()),
                    "Income_Indirect": float((income - income_direct)[mask, j].sum()),
                    "Income_Total": float(income[mask, j].sum()),
                    "Jobs_Direct": float(jobs_direct[mask, j].sum()),
                    "Jobs_Indirect": float((jobs - jobs_direct)[mask, j].sum()),
                    "Jobs_Total": float(jobs[mask, j].sum()),
                })

    detailed = pd.DataFrame(rows)
    if detailed.empty:
        raise ValueError("The detailed MRIO evaluation produced no results.")
    by_lvl, by_sub = _aggregate_detailed_mrio_results(detailed, crosswalk, year=year)
    return by_lvl, by_sub, detailed, crosswalk


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
    fallback_verbose: bool = True,
    return_detailed: bool = False,
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

    df_use = _normalize_origin_country_to_mrio(final_cost_df)
    results_by_origin_lvl, results_by_origin_sub_lvl, detailed, crosswalk = evaluate_detailed_mrio_results(
        df_use,
        L1_MRIO=L1_MRIO,
        VA_MRIO=VA_MRIO,
        LB_MRIO=LB_MRIO,
        W_MRIO=W_MRIO,
        core=list(core),
        mrio=mrio,
        annualize=annualize,
        lifetime_years=lifetime_years,
    )
    if return_detailed:
        return results_by_origin_lvl, results_by_origin_sub_lvl, detailed, crosswalk
    return results_by_origin_lvl, results_by_origin_sub_lvl


def _validate_scenario_result_frames(results_by_origin: dict, *, context: str) -> None:
    """Check result splits and component totals without changing result frames."""
    split_cols = [
        ("Output_Direct", "Output_Indirect", "Output_Total"),
        ("GVA_Direct", "GVA_Indirect", "GVA_Total"),
    ]
    for lvl, frames in results_by_origin.items():
        total_key = "Total" if "Total" in frames else ("Total", "ALL")
        total = frames.get(total_key)
        if total is None or "Deployment" not in total.columns:
            continue
        detail = [
            frame for key, frame in frames.items()
            if key != total_key
            and not _is_aggregate_component_key(key)
            and key not in _self_aggregate_keys_to_skip(frames.keys())
        ]
        for frame in frames.values():
            for direct, indirect, total_col in split_cols:
                if {direct, indirect, total_col}.issubset(frame.columns):
                    if not np.allclose(
                        (frame[direct] + frame[indirect]).to_numpy(dtype=float),
                        frame[total_col].to_numpy(dtype=float),
                        atol=1e-6,
                        rtol=1e-9,
                    ):
                        raise ValueError(f"Scenario result split failed for {context}, level {lvl}.")
        if not detail:
            continue
        key_cols = [c for c in ("Deployment", "Country", "Year", "Sector") if c in total.columns]
        cols = [c for triple in split_cols for c in triple if c in total.columns]
        total_g = total.groupby(key_cols, dropna=False)[cols].sum().sort_index()
        detail_g = pd.concat(detail, ignore_index=True).groupby(key_cols, dropna=False)[cols].sum().sort_index()
        detail_g = detail_g.reindex(total_g.index).fillna(0.0)
        if not np.allclose(total_g.to_numpy(dtype=float), detail_g.to_numpy(dtype=float), atol=1e-6, rtol=1e-9):
            raise ValueError(f"Scenario component sum failed for {context}, level {lvl}.")



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
    fallback_verbose: bool | None = None,
    progress: bool = False,
) -> Dict[int, dict]:
    """
    Runs the MRIO model for a set of scenarios.
    Uses global LEVELS variable.
    """
    out: Dict[int, dict] = {}

    total_scenarios = len(scenarios)
    for scenario_i, (d_pp, scn) in enumerate(scenarios.items(), start=1):
        if progress:
            print(
                f"[scenario MRIO] delta {int(d_pp):+d} pp "
                f"({scenario_i}/{total_scenarios})",
                flush=True,
            )
        evaluated = run_results_by_origin_from_final_cost_df(
            scn.final_cost_df_cf,
            L1_MRIO=L1_MRIO,
            VA_MRIO=VA_MRIO,
            LB_MRIO=LB_MRIO,
            W_MRIO=W_MRIO,
            core=core,
            mrio=mrio,
            annualize=annualize,
            lifetime_years=lifetime_years,
            fallback_verbose=True if fallback_verbose is None else fallback_verbose,
            return_detailed=True,
        )
        # Keep compatibility with callers/tests that replace the legacy
        # two-value helper with a lightweight stub.
        if len(evaluated) == 2:
            r_lvl, r_sub = evaluated
            detailed = None
            crosswalk = None
        else:
            r_lvl, r_sub, detailed, crosswalk = evaluated
        _validate_scenario_result_frames(r_lvl, context=f"delta={d_pp}, level results")
        _validate_scenario_result_frames(r_sub, context=f"delta={d_pp}, subcomponent results")

        out[d_pp] = {
            "results_by_origin_lvl": r_lvl,
            "results_by_origin_sub_lvl": r_sub,
            # Retain the terminal source-level impacts for exact derived
            # decompositions such as the EU27-side reallocation.  Consumers
            # of the legacy dictionaries are unaffected by these private keys.
            "_detailed": detailed,
            "_crosswalk": crosswalk,
        }

    return out



# ============================================================
# DROP-IN: EU-level import substitution (direct only) + plots
# ============================================================

EU_LABEL, UK_LABEL, ROE_LABEL = "EU27", "United Kingdom", "RoE"
OTHER_EU27_LABEL = "Other EU27"
OTHER_ORIGINS_LABEL = "Other origins"
ROW_LABEL = "RoW residual"


def _validate_reallocation(
    values: dict,
    group_eff: set[str],
    expected_share: float | None,
    *,
    eps: float,
    context: str,
) -> None:
    """Validate one sourcing bucket without changing its values."""
    tol = max(float(eps), 1e-10)
    if any(float(v) < -tol for v in values.values()):
        raise ValueError(f"Negative sourcing allocation in {context}.")
    total = float(sum(float(v) for v in values.values()))
    if total <= tol:
        return
    share = float(sum(float(v) for o, v in values.items() if o in group_eff) / total)
    if share < -tol or share > 1.0 + tol:
        raise ValueError(f"Invalid EU27 sourcing share in {context}: {share:.12g}.")
    if expected_share is not None and not np.isclose(
        share, float(expected_share), atol=tol, rtol=1e-10
    ):
        raise ValueError(
            f"EU27 sourcing target mismatch in {context}: "
            f"expected {float(expected_share):.12g}, got {share:.12g}."
        )


def _snap_roundoff_negative_allocations(
    values: dict,
    *,
    total_quantity: float,
    context: str,
) -> None:
    """Snap only scale-relative floating-point negative residuals to zero."""
    scale = max(1.0, abs(float(total_quantity)))
    roundoff_tol = max(64.0 * np.finfo(float).eps * scale, 1e-12)
    for country, value in list(values.items()):
        value_f = float(value)
        if value_f < 0.0 and abs(value_f) <= roundoff_tol:
            values[country] = 0.0
        elif value_f < -roundoff_tol:
            raise ValueError(
                f"Negative sourcing allocation in {context}: "
                f"{country}={value_f:.12g} below roundoff tolerance "
                f"{roundoff_tol:.12g}."
            )


def _validate_donor_reduction(
    by_o: dict,
    donors: list[str],
    alloc: dict,
    move: float,
    *,
    eps: float,
    context: str,
) -> None:
    tol = max(float(eps), abs(float(move)) * 1e-10)
    for origin, reduction in alloc.items():
        available = float(by_o.get(origin, 0.0))
        if float(reduction) < -tol or float(reduction) > available + tol:
            raise ValueError(f"Donor reduction exceeds allocation in {context}: {origin}.")
    if not np.isclose(float(sum(alloc.values())), float(move), atol=tol, rtol=1e-10):
        raise ValueError(
            f"Donor reductions do not equal intended transfer in {context}: "
            f"intended={float(move):.12g}, actual={float(sum(alloc.values())):.12g}."
        )


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

# -----------------------------
# 0b) Capacity mix from results
# ----------------------------


def build_bucket_baseline_gva_weights(
    *,
    df_alloc: pd.DataFrame,
    flows_by_component_hier_all: dict,
    ind_root: str,
    ind_ind: bool,
    mrio: pd.DataFrame,
    L1_MRIO: pd.DataFrame,
    VA_MRIO: pd.Series,
    EU27=(),
    low_income=(),
    lifetime_years: int = 25,
    eps: float = 1e-12,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
    """Build recipient weights from one baseline MRIO shock per exact bucket.

    The returned map is recipient-only. Donors are weighted separately by their
    existing bucket expenditure in the reallocation functions.  A fallback to
    valid baseline expenditure is recorded when a group has no positive GVA.

    ``low_income`` is accepted for consistent propagation from the scenario
    runner. It does not alter these recipient weights, which remain based on
    exact bucket-level baseline upstream GVA.
    """
    if mrio is None or L1_MRIO is None or VA_MRIO is None:
        raise ValueError("Bucket GVA weights require MRIO, L1_MRIO, and VA_MRIO.")

    bridge_tur, bridge_long = load_bridges(get_ind_root(ind_root))
    if ind_ind:
        for bridge in (bridge_tur, bridge_long):
            if "industry_code" in bridge.columns:
                bridge["industry_code"] = bridge["industry_code"].map(
                    lambda value: str(value).strip()[4:]
                    if str(value).strip().startswith("CPA_") else str(value).strip()
                )

    turbine = allocate_turbine_costs(df_alloc, bridge_tur, keep_oem=True)
    farm = allocate_farm_costs(flows_to_cost_df(flows_by_component_hier_all), bridge_long)
    turbine["OEM"] = turbine["OEM"].astype(str).str.strip()
    farm["OEM"] = "__NA__"
    mapped = pd.concat([turbine, farm], ignore_index=True, sort=False)

    expected_turbine = set()
    if not df_alloc.empty:
        expected_turbine = set(
            map(tuple, df_alloc.loc[df_alloc["value_alloc"].astype(float).abs() > eps,
                                    ["deployment_country", "OEM", "component_group", "sub_component", "cost_block"]]
                .astype(str).itertuples(index=False, name=None))
        )
    actual_turbine = set(
        map(tuple, turbine[["deployment_country", "OEM", "component_group", "sub_component", "cost_block"]]
            .astype(str).itertuples(index=False, name=None))
    )
    if expected_turbine - actual_turbine:
        raise ValueError(
            "Could not reconstruct exact turbine baseline buckets: "
            f"{sorted(expected_turbine - actual_turbine)[:5]}"
        )

    expected_farm = set()
    for dep, parents in flows_by_component_hier_all.items():
        for parent, children in parents.items():
            if str(parent).strip() == "Turbine":
                continue
            for child, origins in children.items():
                if any(abs(float(value)) > eps for value in origins.values()):
                    expected_farm.add((str(dep).strip(), "__NA__", str(parent).strip(), str(child).strip()))
    actual_farm = set(
        map(tuple, farm[["deployment_country", "OEM", "component_group", "sub_component"]]
            .astype(str).itertuples(index=False, name=None))
    )
    if expected_farm - actual_farm:
        raise ValueError(
            "Could not reconstruct exact farm baseline buckets: "
            f"{sorted(expected_farm - actual_farm)[:5]}"
        )
    mapped = add_3_levels(mapped)

    idx = mrio.index
    if not isinstance(idx, pd.MultiIndex) or idx.nlevels < 2:
        raise ValueError("MRIO.index must be a country-industry MultiIndex.")
    if not idx.is_unique:
        raise ValueError("MRIO.index must be unique for bucket final-demand construction.")
    if not isinstance(L1_MRIO.index, pd.MultiIndex) or not isinstance(L1_MRIO.columns, pd.MultiIndex):
        raise ValueError("L1_MRIO must have country-industry MultiIndex rows and columns.")
    if not L1_MRIO.index.is_unique or not L1_MRIO.columns.is_unique:
        raise ValueError("L1_MRIO rows and columns must have unique country-industry indices.")
    if not VA_MRIO.index.is_unique:
        raise ValueError("VA_MRIO.index must be unique and aligned with L1_MRIO.index.")

    def _index_with_names(index, target):
        if isinstance(target, pd.MultiIndex):
            if not isinstance(index, pd.MultiIndex) or index.nlevels != target.nlevels:
                raise ValueError("Country-industry indices have incompatible structures.")
            return pd.MultiIndex.from_tuples(list(index), names=target.names)
        return pd.Index(index, name=target.name)

    va_index = _index_with_names(VA_MRIO.index, L1_MRIO.index)
    va_values = pd.to_numeric(VA_MRIO, errors="coerce")
    if va_values.isna().any():
        raise ValueError("VA_MRIO contains missing or non-numeric aligned coefficients.")
    va_aligned = pd.Series(va_values.to_numpy(dtype=float), index=va_index, name=VA_MRIO.name)
    missing_va = L1_MRIO.index.difference(va_aligned.index)
    if len(missing_va):
        raise ValueError(
            "VA_MRIO is missing L1_MRIO industry rows, e.g. "
            f"{list(missing_va[:5])}."
        )
    va_aligned = va_aligned.reindex(L1_MRIO.index)

    mrio_country = idx.get_level_values(0).astype(str).str.strip().str.upper()
    mrio_origin_set = set(mrio_country)
    mapped_origin = mapped["origin_country"].astype(str).str.strip().str.upper()
    mapped_origin = mapped_origin.replace({"UK": "GB"})
    off_scope_origins = set(mapped_origin) - mrio_origin_set
    known_off_scope = off_scope_origins & set(CODE_TO_NAME)
    invalid_origins = sorted(off_scope_origins - known_off_scope)
    if invalid_origins:
        raise ValueError(
            "Bucket industrial allocation contains country codes absent from MRIO: "
            f"{invalid_origins}. Only valid ISO country codes may be aggregated to ROW."
        )
    if known_off_scope:
        mapped_origin = mapped_origin.replace({country: "ROW" for country in known_off_scope})
    mapped["origin_country"] = mapped_origin
    mrio_industry = idx.get_level_values(1).astype(str).str.strip()
    gross_output = pd.to_numeric(mrio.sum(axis=1), errors="coerce").fillna(0.0)
    mrio_codes = set(mrio_industry)

    def _align(code):
        code = str(code).strip()
        if code in mrio_codes:
            return code
        alt = code[4:] if code.startswith("CPA_") else f"CPA_{code}"
        return alt if alt in mrio_codes else code

    eu = {str(country).strip().upper().replace("UK", "GB") for country in (EU27 or ())}
    recipient_map = {}
    metadata = {}
    baseline_rows = []
    recipient_rows = []

    bucket_specs = [
        ("turbine", ["deployment_country", "OEM", "component_group", "sub_component", "cost_block"]),
        ("farm", ["deployment_country", "OEM", "component_group", "sub_component"]),
    ]
    expected_annual_final_demand = (
        float(pd.to_numeric(mapped["value_alloc"], errors="coerce").fillna(0.0).sum())
        / float(lifetime_years)
    )
    reconstructed_annual_final_demand = 0.0
    for scenario_type, bucket_cols in bucket_specs:
        subset = mapped.loc[mapped["OEM"].eq("__NA__") if scenario_type == "farm" else ~mapped["OEM"].eq("__NA__")]
        for key, bucket in subset.groupby(bucket_cols, dropna=False):
            key = tuple(key) if isinstance(key, tuple) else (key,)
            dep = str(key[0]).strip().upper()
            bucket_key = key if scenario_type == "turbine" else (key[0], key[2], key[3])
            bucket_before_fallback = bucket.copy()
            bucket = _apply_zero_output_industry_fallbacks(bucket, mrio, verbose=False)
            industry_fallback = []
            for country_code, source_code, target_code in (("JP", "C33", "C27"), ("CA", "C33", "C27")):
                before_mask = (
                    bucket_before_fallback["origin_country"].astype(str).str.strip().str.upper().eq(country_code)
                    & bucket_before_fallback["industry_code"].astype(str).str.strip().str.replace("CPA_", "", regex=False).eq(source_code)
                )
                after_target = (
                    bucket["origin_country"].astype(str).str.strip().str.upper().eq(country_code)
                    & bucket["industry_code"].astype(str).str.strip().str.replace("CPA_", "", regex=False).eq(target_code)
                )
                if bool(before_mask.any()) and bool(after_target.any()):
                    industry_fallback.append(f"{country_code}:{source_code}->{target_code}")
            industry_fallback = ";".join(industry_fallback)
            bucket = bucket.loc[bucket["value_alloc"].astype(float).abs() > eps].copy()
            if bucket.empty:
                continue

            y = make_final_demand_vector(
                mrio, bucket, dep, level=1, lifetime_years=lifetime_years, mode="annual"
            )
            if not y.index.is_unique:
                raise ValueError(
                    f"Final-demand vector has duplicate country-industry entries for bucket {bucket_key}."
                )
            y_index = _index_with_names(y.index, L1_MRIO.columns)
            y_for_alignment = pd.Series(
                pd.to_numeric(y, errors="coerce").to_numpy(dtype=float),
                index=y_index,
                name=y.name,
            )
            if y_for_alignment.isna().any():
                raise ValueError(f"Final-demand vector contains non-numeric values for bucket {bucket_key}.")
            missing = y_for_alignment.index.difference(L1_MRIO.columns)
            missing_values = y_for_alignment.reindex(missing).fillna(0.0)
            missing_nonzero = missing_values[missing_values.abs() > eps]
            if len(missing_nonzero):
                raise ValueError(
                    "Bucket final demand contains nonzero entries outside L1_MRIO.columns: "
                    f"{list(missing_nonzero.index[:5])}; "
                    f"y index names={y.index.names}, L1 columns={L1_MRIO.columns.names}, "
                    f"MRIO index={mrio.index.names}."
                )
            y = y_for_alignment.reindex(L1_MRIO.columns, fill_value=0.0)
            if not np.isclose(float(y.sum()), float(y_for_alignment.sum()), atol=1e-6, rtol=1e-10):
                raise ValueError(
                    f"Final-demand alignment changed bucket expenditure for {bucket_key}: "
                    f"before={float(y_for_alignment.sum()):,.6f}, after={float(y.sum()):,.6f}."
                )
            reconstructed_annual_final_demand += float(y.sum())

            x = L1_MRIO @ y
            va = va_aligned.reindex(x.index)
            direct = va * y.reindex(x.index).fillna(0.0)
            total = va * x
            indirect = total - direct
            countries = x.index.get_level_values(0).astype(str).str.strip().str.upper()
            codes = {_align(code) for code in bucket["industry_code"].astype(str)}

            def _valid(country_code):
                mask = mrio_country == country_code
                return bool(mask.any()) and all(
                    bool((mask & (mrio_industry == code) & (gross_output > eps)).any())
                    for code in codes
                )

            country_frame = pd.DataFrame({
                "country": countries,
                "direct": direct.to_numpy(dtype=float),
                "indirect": indirect.to_numpy(dtype=float),
                "total": total.to_numpy(dtype=float),
            }).groupby("country", as_index=False)[["direct", "indirect", "total"]].sum()
            country_frame["valid_industry_mapping"] = country_frame["country"].map(_valid)
            for _, row in country_frame.iterrows():
                detail = {
                    "scenario_type": scenario_type,
                    "deployment": dep,
                    "OEM": key[1],
                    "component": key[2],
                    "subcomponent": key[3],
                    "cost_block": key[4] if scenario_type == "turbine" else "__NA__",
                    "producing_country": row["country"],
                    "GVA_Direct": float(row["direct"]),
                    "GVA_Indirect": float(row["indirect"]),
                    "GVA_Total": float(row["total"]),
                    "valid_industry_mapping": bool(row["valid_industry_mapping"]),
                    "positive_output_for_bucket": bool(row["valid_industry_mapping"]),
                    "industry_fallback": industry_fallback,
                }
                baseline_rows.append(detail)

            by_origin = bucket.groupby("origin_country")["value_alloc"].sum()
            by_origin.index = pd.Index([
                "GB" if str(country).strip().upper() == "UK" else str(country).strip().upper()
                for country in by_origin.index
            ])
            country_totals = country_frame.set_index("country")["total"].to_dict()
            group_data = {}
            for group, group_set in (("EU", eu), ("NON_EU", None)):
                group_candidates = country_frame.loc[
                    country_frame["valid_industry_mapping"]
                    & country_frame["country"].map(
                        lambda value: value in group_set if group_set is not None else value not in eu
                    )
                ]
                positive = {
                    str(row["country"]): float(row["total"])
                    for _, row in group_candidates.iterrows()
                    if float(row["total"]) > eps
                }
                positive_sum = float(sum(positive.values()))
                if positive_sum > eps:
                    shares = {country: value / positive_sum for country, value in positive.items()}
                    source = "baseline_total_gva"
                    fallback = industry_fallback
                else:
                    valid_origins = {
                        country: float(value)
                        for country, value in by_origin.items()
                        if country in (group_set if group_set is not None else set(by_origin.index) - eu)
                        and _valid(str(country).strip().upper())
                        and float(value) > eps
                    }
                    value_sum = float(sum(valid_origins.values()))
                    shares = {country: value / value_sum for country, value in valid_origins.items()} if value_sum > eps else {}
                    source = "fallback:baseline_tier1_expenditure_no_positive_gva" if shares else "unavailable:no_valid_recipient"
                    fallback = ";".join(filter(None, (source if shares else "", industry_fallback)))
                if industry_fallback:
                    source = f"{source};industry_fallback={industry_fallback}"
                group_data[group] = shares
                for _, candidate in group_candidates.iterrows():
                    country = str(candidate["country"])
                    share = float(shares.get(country, 0.0))
                    recipient_rows.append({
                        "scenario_type": scenario_type,
                        "deployment": dep,
                        "OEM": key[1],
                        "component": key[2],
                        "subcomponent": key[3],
                        "cost_block": key[4] if scenario_type == "turbine" else "__NA__",
                        "receiving_group": group,
                        "country": country,
                        "baseline_GVA": float(country_totals.get(country, 0.0)),
                        "recipient_weight": share,
                        "weight_source": source if share > 0 else "excluded_nonpositive_gva",
                        "fallback_rule": fallback if share > 0 else "zero_or_negative_gva_excluded",
                        "industry_fallback": industry_fallback,
                    })
                metadata.setdefault(bucket_key, {})[group] = {
                    "source": source,
                    "fallback_rule": fallback,
                }
            recipient_map[bucket_key] = group_data

    if not np.isclose(
        reconstructed_annual_final_demand,
        expected_annual_final_demand,
        atol=1e-6,
        rtol=1e-10,
    ):
        raise ValueError(
            "Reconstructed bucket final demand does not reconcile with the mapped baseline: "
            f"expected={expected_annual_final_demand:,.6f}, "
            f"reconstructed={reconstructed_annual_final_demand:,.6f}."
        )

    return recipient_map, pd.DataFrame(baseline_rows), pd.DataFrame(recipient_rows), metadata


# -----------------------------
# 1) Counterfactual: Δ-pp LC shift (direct only), industry-fixed
# -----------------------------
def apply_delta_lc_shift_direct_turbine(
    df_alloc: pd.DataFrame,
    *,
    group_set,
    low_income=(),
    delta_pp: float,
    exclude_origin=("EXPORTROW",),
    eps: float = 1e-12,
    receiver_policy: str = "baseline_mix",   
    capacity_map: dict | None = None,  # deprecated (ignored)
    gva_share_map: dict | None = None,
    gva_share_map_bucket: dict | None = None,
    weight_metadata: dict | None = None,
    single_receiver_country: str | None = None,  
    single_donor_country: str | None = None,
    component_filter: set[str] | list[str] | None = None,
    receiver_whitelist: set[str] | list[str] | None = None,
    donor_whitelist: set[str] | list[str] | None = None,
    eu_spend_reduction_pct: float = 0.0,
    preserve_total: bool = True,
    weight_audit: list[dict] | None = None,
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

    low_income = {norm(x) for x in (low_income or ())}
    scenario_markup = _scenario_markup(eu_spend_reduction_pct)
    _reference_quantity_preserve_total_guard(preserve_total, delta_pp)
    group_set = {norm(x) for x in group_set}
    exclude_origin = {norm(x) for x in exclude_origin}
    receiver_whitelist = {norm(x) for x in (receiver_whitelist or [])} or None
    donor_whitelist = {norm(x) for x in (donor_whitelist or [])} or None
    single_donor_country_norm = None if single_donor_country is None else norm(single_donor_country)
    component_filter = _expand_component_filter(component_filter)
    audit_rows = weight_audit if weight_audit is not None else []
    positive_gva_zero_expenditure_donors = []

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
            "eu_spend_reduction_pct": scenario_markup,
            "scenario_markup": scenario_markup,
            "preserve_total": bool(preserve_total),
            "pricing_model_version": PRICE_ADJUSTMENT_MODEL_VERSION,
        }
        return df_new, stats

    # effective universes
    orig_universe = set(d0["origin_country"].unique())
    # Keep the full EU27 membership for quota accounting.  A country may have
    # zero initial Tier-1 sourcing and still be an eligible EU27 receiver when
    # its bucket-level upstream-GVA weight is positive.
    group_eff = set(group_set)
    non_group_eff = orig_universe - group_eff

    # bucket (A): per OEM
    B = ["deployment_country","OEM","component_group","sub_component","cost_block"]

    # ---- 1) need per bucket
    d0["_reference_price"] = d0["origin_country"].map(
        lambda country: _reference_price_factor(
            country, low_income, markup=scenario_markup
        )
    )
    d0["_equivalent_quantity"] = d0["value_alloc"] / d0["_reference_price"]
    tot = d0.groupby(B, dropna=False)["_equivalent_quantity"].sum().rename("T").reset_index()
    grp = (
        d0.loc[d0["origin_country"].isin(group_eff)]
          .groupby(B, dropna=False)["_equivalent_quantity"].sum()
          .rename("G").reset_index()
    )
    need_df = tot.merge(grp, on=B, how="left").fillna({"G":0.0})
    need_df["s0"] = np.where(need_df["T"] > eps, need_df["G"]/need_df["T"], 0.0)
    need_df["s_target"] = np.clip(need_df["s0"] + float(delta_pp), 0.0, 1.0)
    need_df["need"] = (need_df["s_target"] - need_df["s0"]) * need_df["T"]
    need_map = {tuple(r[k] for k in B): float(r["need"]) for _, r in need_df.iterrows()}
    target_map = {tuple(r[k] for k in B): float(r["s_target"]) for _, r in need_df.iterrows()}

    # ---- 2) Recipient weights (bucket-level upstream GVA)
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
        return_source: bool = False,
    ) -> dict | tuple[dict, str]:
        base = {}
        source = "none"
        if gva_share_map_bucket is not None and bucket_key in gva_share_map_bucket:
            base = gva_share_map_bucket[bucket_key].get(group, {}) or {}
            if base:
                source = (weight_metadata or {}).get(bucket_key, {}).get(group, {}).get(
                    "source", "baseline_total_gva"
                )
        if not base and gva_share_map:
            if subc is not None:
                if (parent_raw, str(subc)) in gva_share_map:
                    base = gva_share_map[(parent_raw, str(subc))].get(group, {}) or {}
                    if base:
                        source = "baseline_gva_component_subcomponent"
                elif (parent_fallback, str(subc)) in gva_share_map:
                    base = gva_share_map[(parent_fallback, str(subc))].get(group, {}) or {}
                    if base:
                        source = "baseline_gva_component_subcomponent"
            if not base:
                if parent_raw in gva_share_map:
                    base = gva_share_map[parent_raw].get(group, {}) or {}
                    if base:
                        source = "baseline_gva_component"
                elif parent_fallback in gva_share_map:
                    base = gva_share_map[parent_fallback].get(group, {}) or {}
                    if base:
                        source = "baseline_gva_component"
        if allowed is not None:
            base = {o: float(base.get(o, 0.0)) for o in allowed if float(base.get(o, 0.0)) > 0}
        result = _norm_weights(base)
        return (result, source if result else "none") if return_source else result

    def _allocate_reduction(by_o: pd.Series, donors: list[str], weights: dict, total: float) -> dict:
        alloc = {o: 0.0 for o in donors}
        rem = float(total)
        active = {o for o in donors if float(by_o.get(o, 0.0)) > eps and float(weights.get(o, 0.0)) > 0}
        weights = {o: float(weights.get(o, 0.0)) for o in active}

        while rem > eps and active:
            wsum = float(sum(weights[o] for o in active))
            if wsum <= eps:
                break
            proposed = {
                o: rem * (weights[o] / wsum)
                for o in active
            }
            capped = [
                o for o in active
                if proposed[o] >= float(by_o.get(o, 0.0)) - alloc[o] - eps
            ]
            if not capped:
                for o, amt in proposed.items():
                    alloc[o] += amt
                rem = 0.0
                break
            for o in capped:
                cap = float(by_o.get(o, 0.0)) - alloc[o]
                alloc[o] += max(0.0, cap)
                rem -= max(0.0, cap)
            for o in capped:
                active.discard(o)

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
    donor_reduction_total = 0.0
    fallback_donor_reallocation_total = 0.0

    for key, g in d0.groupby(B, dropna=False):
        need = float(need_map.get(key, 0.0))
        if abs(need) <= eps:
            parts.append(g)
            continue

        dep, oem, cg, sc, cb = key
        move_into_group = (need > 0)
        move = abs(need)

        by_o = g.groupby("origin_country")["_equivalent_quantity"].sum()
        by_o_value = g.groupby("origin_country")["value_alloc"].sum()

        donors_set = non_group_eff if move_into_group else group_eff
        donor_filter = donor_whitelist
        if move_into_group and single_donor_country_norm is not None:
            donor_filter = {single_donor_country_norm}
        if donor_filter is not None:
            donors_set = donors_set & donor_filter
        donors = [o for o in by_o.index if o in donors_set and float(by_o.loc[o]) > eps]
        donor_sum = float(by_o.loc[donors].sum()) if donors else 0.0
        if donor_sum <= eps:
            unshiftable += float(sum(
                float(by_o.get(country, 0.0))
                * _reference_price_factor(country, low_income, markup=scenario_markup)
                for country in donors_set
            ))
            parts.append(g)
            continue
        move = min(move, donor_sum)

        parent_raw = str(cg).strip()
        parent_key = _parent_key(cg)
        bucket_key = (dep, oem, cg, sc, cb)
        donor_group = "NON_EU" if move_into_group else "EU"
        recv_group = "EU" if move_into_group else "NON_EU"

        donor_gva_all, _ = _weights_for_group(
            parent_raw, parent_key, sc, donor_group, allowed=None, return_source=True
        )
        zero_expenditure_gva = sorted(
            o for o, weight in donor_gva_all.items()
            if float(weight) > 0.0 and float(by_o.get(o, 0.0)) <= eps
        )
        if zero_expenditure_gva:
            positive_gva_zero_expenditure_donors.append({
                "deployment": dep,
                "OEM": oem,
                "component": cg,
                "subcomponent": sc,
                "cost_block": cb,
                "group": donor_group,
                "countries": zero_expenditure_gva,
            })
        donor_w, donor_source = _weights_for_group(
            parent_raw, parent_key, sc, donor_group,
            allowed=set(donors), return_source=True
        )
        if not donor_w:
            donor_w = _norm_weights({o: float(by_o_value.loc[o]) for o in donors})
            donor_source = "fallback:baseline_tier1_expenditure_no_positive_gva"

        # receivers: single-country override only for offshoring (non‑EU)
        receivers = []
        recv_w = {}
        if receiver_policy == "single_country" and (not move_into_group):
            if single_receiver_country is not None:
                oN = norm(single_receiver_country)
                if oN in non_group_eff and (receiver_whitelist is None or oN in receiver_whitelist):
                    receivers = [oN]
                    recv_w = {oN: 1.0}
                    recv_source = "single_receiver_override"

        if not receivers:
            recv_w, recv_source = _weights_for_group(
                parent_raw, parent_key, sc, recv_group, allowed=None, return_source=True
            )
            if receiver_whitelist is not None:
                recv_w = {o: v for o, v in recv_w.items() if o in receiver_whitelist}
            if not recv_w:
                # fallback to current by_o within destination group
                dest_set = group_eff if move_into_group else non_group_eff
                recv_w = _norm_weights({o: float(by_o_value.loc[o]) for o in by_o.index if o in dest_set})
                recv_source = "fallback:baseline_tier1_expenditure_no_positive_gva"
            receivers = list(recv_w.keys())

        if not receivers:
            unshiftable += _reference_transfer_value(
                move, by_o.to_dict(), donors, low_income, markup=scenario_markup
            )
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
            proto["_equivalent_quantity"] = 0.0
            adds = []
            for o in missing:
                a = proto.copy()
                a["origin_country"] = o
                a["_reference_price"] = _reference_price_factor(
                    o, low_income, markup=scenario_markup
                )
                adds.append(a)
            g = pd.concat([g] + adds, ignore_index=True)
            by_o = g.groupby("origin_country")["_equivalent_quantity"].sum()
            by_o_value = g.groupby("origin_country")["value_alloc"].sum()

        new_by_o = by_o.copy()

        # donor reductions using GVA-share weights (capped by available
        # equivalent quantities)
        alloc = _allocate_reduction(by_o, donors, donor_w, move)
        _validate_donor_reduction(
            by_o, donors, alloc, move, eps=eps,
            context=f"turbine {dep}/{oem}/{cg}/{sc}/{cb}",
        )
        alloc_reference_value = _reference_value_by_origin(
            alloc, low_income, markup=scenario_markup
        )
        donor_reduction_total += float(sum(alloc_reference_value.values()))
        fallback_donor_reallocation_total += float(sum(
            alloc_reference_value.get(country, 0.0) for country, amount in alloc.items()
            if country not in donor_w and float(amount) > eps
        ))
        for o, red in alloc.items():
            new_by_o.loc[o] = float(new_by_o.get(o, 0.0)) - float(red)

        # receiver additions using GVA-share weights.  All additions are in
        # equivalent quantities; conversion to fixed reference-price money is
        # performed only when the final allocation frame is rebuilt below.
        recv_w = _norm_weights(recv_w)
        if not recv_w:
            unshiftable += move
            parts.append(g)
            continue
        wsum = float(sum(recv_w.values()))
        for o, wi in recv_w.items():
            new_by_o.loc[o] = float(new_by_o.get(o, 0.0)) + move * float(wi) / wsum

        raw_receiver_total = float(move)
        move_reference_value = float(sum(alloc_reference_value.values()))
        audit_start = len(audit_rows)
        # If GVA-weighted donors exhaust their available Tier-1 expenditure,
        # _allocate_reduction redistributes the remainder by available
        # expenditure.  Keep those fallback countries visible in the audit.
        audit_donor_weights = dict(donor_w)
        for country, amount in alloc.items():
            if float(amount) > eps and country not in audit_donor_weights:
                audit_donor_weights[country] = 0.0
        for role, weights, source, group_name in (
            ("donor", audit_donor_weights, donor_source, donor_group),
            ("receiver", recv_w, recv_source, recv_group),
        ):
            for country, requested_weight in weights.items():
                price = _reference_price_factor(
                    country, low_income, markup=scenario_markup
                )
                initial = float(by_o.get(country, 0.0)) * price
                final = float(new_by_o.get(country, 0.0)) * price
                change = final - initial
                pre_markup = (
                    raw_receiver_total * float(requested_weight) / wsum
                    * price
                    if role == "receiver" else 0.0
                )
                audit_rows.append({
                    "weight_role": role,
                    "weight_source": source,
                    "aggregation_level": "deployment/OEM/component/subcomponent/cost_block",
                    "deployment": dep,
                    "OEM": oem,
                    "component": cg,
                    "subcomponent": sc,
                    "cost_block": cb,
                    "delta_pp": float(delta_pp) * 100.0,
                    "country": country,
                    "group": group_name,
                    "requested_weight": float(requested_weight),
                    "requested_transfer": move_reference_value,
                    "initial_expenditure": initial,
                    "actual_donor_reduction": max(-change, 0.0) if role == "donor" else 0.0,
                    "receiver_addition_before_markup": pre_markup,
                    "markup_adjustment": (change - pre_markup) if role == "receiver" else 0.0,
                    "final_expenditure": final,
                    "actual_receiver_addition": max(change, 0.0) if role == "receiver" else 0.0,
                })

        _validate_reallocation(
            new_by_o.to_dict(), group_eff,
            target_map[key],
            eps=eps,
            context=f"turbine {dep}/{oem}/{cg}/{sc}/{cb}",
        )

        g2 = g.copy()
        for oc, sr in g2.groupby("origin_country"):
            base = sr["value_alloc"].to_numpy()
            s = float(base.sum())
            tgt_q = float(new_by_o.get(oc, 0.0))
            tgt = tgt_q * _reference_price_factor(oc, low_income, markup=scenario_markup)
            g2.loc[sr.index, "value_alloc"] = (base / s * tgt) if s > eps else (tgt / len(sr))

        g2_by_o = g2.groupby("origin_country")["value_alloc"].sum().to_dict()
        for row in audit_rows[audit_start:]:
            final = float(g2_by_o.get(row["country"], 0.0))
            change = final - float(row["initial_expenditure"])
            row["final_expenditure"] = final
            row["actual_donor_reduction"] = max(-change, 0.0) if row["weight_role"] == "donor" else 0.0
            row["actual_receiver_addition"] = max(change, 0.0) if row["weight_role"] == "receiver" else 0.0
            if row["weight_role"] == "receiver":
                row["markup_adjustment"] = change - float(row["receiver_addition_before_markup"])
        g2_by_q = {
            country: float(value)
            / _reference_price_factor(country, low_income, markup=scenario_markup)
            for country, value in g2_by_o.items()
        }
        _validate_reallocation(
            g2_by_q, group_eff,
            target_map[key],
            eps=eps,
            context=f"turbine {dep}/{oem}/{cg}/{sc}/{cb}",
        )

        parts.append(g2)

    df_new = pd.concat(parts, ignore_index=True)
    df_new = df_new.drop(columns=["_reference_price", "_equivalent_quantity"], errors="ignore")
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
        "donor_reduction_abs_total": float(donor_reduction_total),
        "unshiftable_abs_total": float(unshiftable),
        "group_set_size_effective": int(len(group_eff)),
        "non_group_origins_present": sorted(list(non_group_eff)),
        "receiver_policy": str(receiver_policy),  # NEW
        "single_receiver_country": None if single_receiver_country is None else norm(single_receiver_country),
        "single_donor_country": single_donor_country_norm,
        "receiver_whitelist": sorted(receiver_whitelist) if receiver_whitelist else [],
        "donor_whitelist": sorted(donor_whitelist) if donor_whitelist else [],
        "eu_spend_reduction_pct": scenario_markup,
        "scenario_markup": scenario_markup,
        "preserve_total": bool(preserve_total),
        "pricing_model_version": PRICE_ADJUSTMENT_MODEL_VERSION,
        "sourcing_allocation_model_version": SOURCING_ALLOCATION_MODEL_VERSION,
        "positive_gva_zero_expenditure_donors": positive_gva_zero_expenditure_donors,
        "fallback_donor_reallocation_abs_total": float(fallback_donor_reallocation_total),
    }
    if weight_audit is not None:
        stats["weight_audit"] = audit_rows
    return df_new, stats

# ============================================================
# 2) Minimal patch: apply_delta_lc_shift_direct_farm
# ============================================================
def apply_delta_lc_shift_direct_farm(
    flows_by_component_hier_all: dict,
    *,
    group_set,
    low_income=(),
    delta_pp: float,
    eps: float = 1e-12,
    receiver_policy: str = "baseline_mix", 
    gva_share_map: dict | None = None,
    gva_share_map_bucket: dict | None = None,
    weight_metadata: dict | None = None,
    single_receiver_country: str | None = None, 
    single_donor_country: str | None = None,
    component_filter: set[str] | list[str] | None = None,
    receiver_whitelist: set[str] | list[str] | None = None,
    donor_whitelist: set[str] | list[str] | None = None,
    eu_spend_reduction_pct: float = 0.0,
    preserve_total: bool = True,
    weight_audit: list[dict] | None = None,
):
    from collections import defaultdict
    import copy

    def norm(x): return str(x).strip().upper()

    low_income = {norm(x) for x in (low_income or ())}
    scenario_markup = _scenario_markup(eu_spend_reduction_pct)
    GROUP = {norm(x) for x in group_set}
    _reference_quantity_preserve_total_guard(preserve_total, delta_pp)
    receiver_whitelist = {norm(x) for x in (receiver_whitelist or [])} or None
    donor_whitelist = {norm(x) for x in (donor_whitelist or [])} or None
    single_donor_country_norm = None if single_donor_country is None else norm(single_donor_country)
    component_filter = {str(x).strip() for x in (component_filter or [])} or None
    audit_rows = weight_audit if weight_audit is not None else []
    positive_gva_zero_expenditure_donors = []
    fallback_donor_reallocation_total = 0.0
    new_flows = copy.deepcopy(flows_by_component_hier_all)

    all_origins = set()
    for dep, parents in flows_by_component_hier_all.items():
        for parent, childs in parents.items():
            for child, od in childs.items():
                all_origins.update({norm(o) for o in od.keys()})

    # Keep the full EU27 membership for quota accounting, including countries
    # that can be added as zero-baseline eligible receivers.
    group_eff = set(GROUP)
    non_group_eff = all_origins - group_eff

    def _norm_weights(d: dict) -> dict:
        s = float(sum(d.values()))
        if s <= eps:
            return {}
        return {k: float(v) / s for k, v in d.items() if float(v) > 0}

    def _weights_for_group(parent: str, subc: str | None, group: str, allowed: set[str] | None = None,
                           return_source: bool = False) -> dict | tuple[dict, str]:
        base = {}
        source = "none"
        if gva_share_map_bucket is not None and bucket_key in gva_share_map_bucket:
            base = gva_share_map_bucket[bucket_key].get(group, {}) or {}
            if base:
                source = (weight_metadata or {}).get(bucket_key, {}).get(group, {}).get(
                    "source", "baseline_total_gva"
                )
        if not base and gva_share_map:
            if subc is not None and (parent, str(subc)) in gva_share_map:
                base = gva_share_map[(parent, str(subc))].get(group, {}) or {}
                if base:
                    source = "baseline_gva_component_subcomponent"
            elif parent in gva_share_map:
                base = gva_share_map[parent].get(group, {}) or {}
                if base:
                    source = "baseline_gva_component"
        if allowed is not None:
            base = {o: float(base.get(o, 0.0)) for o in allowed if float(base.get(o, 0.0)) > 0}
        result = _norm_weights(base)
        return (result, source if result else "none") if return_source else result

    def _allocate_reduction(by_o: dict, donors: list[str], weights: dict, total: float) -> dict:
        alloc = {o: 0.0 for o in donors}
        rem = float(total)
        active = {o for o in donors if float(by_o.get(o, 0.0)) > eps and float(weights.get(o, 0.0)) > 0}
        weights = {o: float(weights.get(o, 0.0)) for o in active}

        while rem > eps and active:
            wsum = float(sum(weights[o] for o in active))
            if wsum <= eps:
                break
            proposed = {
                o: rem * (weights[o] / wsum)
                for o in active
            }
            capped = [
                o for o in active
                if proposed[o] >= float(by_o.get(o, 0.0)) - alloc[o] - eps
            ]
            if not capped:
                for o, amt in proposed.items():
                    alloc[o] += amt
                rem = 0.0
                break
            for o in capped:
                cap = float(by_o.get(o, 0.0)) - alloc[o]
                alloc[o] += max(0.0, cap)
                rem -= max(0.0, cap)
            for o in capped:
                active.discard(o)

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
    donor_reduction_total = 0.0

    for dep, parents in flows_by_component_hier_all.items():
        for parent, childs in parents.items():
            if component_filter is not None and str(parent).strip() not in component_filter:
                continue
            for child, od in childs.items():
                bucket_key = (str(dep).strip().upper(), str(parent).strip(), str(child).strip())
                by_o_value = {norm(o): float(v) for o, v in od.items() if float(v) != 0.0}
                by_o = {
                    country: value
                    / _reference_price_factor(country, low_income, markup=scenario_markup)
                    for country, value in by_o_value.items()
                }
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
                    donor_filter = {single_donor_country_norm}
                if donor_filter is not None:
                    donors_set = donors_set & donor_filter
                donors = [o for o in by_o.keys() if o in donors_set and float(by_o[o]) > eps]
                donor_sum = sum(by_o[o] for o in donors) if donors else 0.0
                if donor_sum <= eps:
                    unshiftable_abs_total += _reference_transfer_value(
                        move, by_o, donors, low_income, markup=scenario_markup
                    )
                    continue
                move = min(move, donor_sum)

                parent_key = str(parent).strip()
                donor_group = "NON_EU" if move_into_group else "EU"
                recv_group = "EU" if move_into_group else "NON_EU"

                donor_gva_all, _ = _weights_for_group(
                    parent_key, child, donor_group, allowed=None, return_source=True
                )
                zero_expenditure_gva = sorted(
                    o for o, weight in donor_gva_all.items()
                    if float(weight) > 0.0 and float(by_o.get(o, 0.0)) <= eps
                )
                if zero_expenditure_gva:
                    positive_gva_zero_expenditure_donors.append({
                        "deployment": dep,
                        "component": parent,
                        "subcomponent": child,
                        "group": donor_group,
                        "countries": zero_expenditure_gva,
                    })
                donor_w, donor_source = _weights_for_group(
                    parent_key, child, donor_group,
                    allowed=set(donors), return_source=True
                )
                if not donor_w:
                    donor_w = _norm_weights({o: float(by_o_value.get(o, 0.0)) for o in donors})
                    donor_source = "fallback:baseline_tier1_expenditure_no_positive_gva"

                # receivers: single-country override only for offshoring (non‑EU)
                receivers = []
                recv_w = {}
                if receiver_policy == "single_country" and (not move_into_group):
                    if single_receiver_country is not None:
                        oN = norm(single_receiver_country)
                        if oN in non_group_eff and (receiver_whitelist is None or oN in receiver_whitelist):
                            receivers = [oN]
                            recv_w = {oN: 1.0}
                            recv_source = "single_receiver_override"

                if not receivers:
                    recv_w, recv_source = _weights_for_group(
                        parent_key, child, recv_group, allowed=None, return_source=True
                    )
                    if receiver_whitelist is not None:
                        recv_w = {o: v for o, v in recv_w.items() if o in receiver_whitelist}
                    if not recv_w:
                        dest_set = group_eff if move_into_group else non_group_eff
                        recv_w = _norm_weights({o: float(by_o_value.get(o, 0.0)) for o in by_o.keys() if o in dest_set})
                        recv_source = "fallback:baseline_tier1_expenditure_no_positive_gva"
                    receivers = list(recv_w.keys())

                if not receivers:
                    unshiftable_abs_total += _reference_transfer_value(
                        move, by_o, donors, low_income, markup=scenario_markup
                    )
                    continue

                for o in receivers:
                    if o not in by_o:
                        by_o[o] = 0.0
                        by_o_value[o] = 0.0

                new_by_o = dict(by_o)

                # donor reductions using GVA-share weights (capped by available
                # equivalent quantities)
                alloc = _allocate_reduction(by_o, donors, donor_w, move)
                _validate_donor_reduction(
                    by_o, donors, alloc, move, eps=eps,
                    context=f"farm {dep}/{parent}/{child}",
                )
                alloc_reference_value = _reference_value_by_origin(
                    alloc, low_income, markup=scenario_markup
                )
                donor_reduction_total += float(sum(alloc_reference_value.values()))
                fallback_donor_reallocation_total += float(sum(
                    alloc_reference_value.get(country, 0.0) for country, amount in alloc.items()
                    if country not in donor_w and float(amount) > eps
                ))
                for o, red in alloc.items():
                    new_by_o[o] = new_by_o.get(o, 0.0) - float(red)

                # receiver additions using GVA-share weights in equivalent
                # quantities; markup never enters MRIO demand construction.
                recv_w = _norm_weights(recv_w)
                if not recv_w:
                    unshiftable_abs_total += _reference_transfer_value(
                        move, by_o, donors, low_income, markup=scenario_markup
                    )
                    continue
                wsum = float(sum(recv_w.values()))
                for o, wi in recv_w.items():
                    new_by_o[o] = new_by_o.get(o, 0.0) + move * (wi / wsum)

                _snap_roundoff_negative_allocations(
                    new_by_o,
                    total_quantity=T,
                    context=f"farm {dep}/{parent}/{child}",
                )

                move_reference_value = float(sum(alloc_reference_value.values()))

                audit_start = len(audit_rows)
                # If GVA-weighted donors exhaust their available Tier-1 expenditure,
                # _allocate_reduction redistributes the remainder by available
                # expenditure.  Keep those fallback countries visible in the audit.
                audit_donor_weights = dict(donor_w)
                for country, amount in alloc.items():
                    if float(amount) > eps and country not in audit_donor_weights:
                        audit_donor_weights[country] = 0.0
                for role, weights, source, group_name in (
                    ("donor", audit_donor_weights, donor_source, donor_group),
                    ("receiver", recv_w, recv_source, recv_group),
                ):
                    for country, requested_weight in weights.items():
                        price = _reference_price_factor(
                            country, low_income, markup=scenario_markup
                        )
                        initial = float(by_o.get(country, 0.0)) * price
                        final = float(new_by_o.get(country, 0.0)) * price
                        pre_markup = (
                            float(move) * float(requested_weight) / wsum * price
                            if role == "receiver" else 0.0
                        )
                        audit_rows.append({
                            "weight_role": role,
                            "weight_source": source,
                            "aggregation_level": "deployment/component/subcomponent",
                            "deployment": dep,
                            "OEM": None,
                            "component": parent,
                            "subcomponent": child,
                            "cost_block": None,
                            "delta_pp": float(delta_pp) * 100.0,
                            "country": country,
                            "group": group_name,
                            "requested_weight": float(requested_weight),
                            "requested_transfer": move_reference_value,
                            "initial_expenditure": initial,
                            "actual_donor_reduction": max(initial - final, 0.0) if role == "donor" else 0.0,
                            "receiver_addition_before_markup": pre_markup,
                            "markup_adjustment": (final - initial - pre_markup) if role == "receiver" else 0.0,
                            "final_expenditure": final,
                            "actual_receiver_addition": max(final - initial, 0.0) if role == "receiver" else 0.0,
                        })

                _validate_reallocation(
                    new_by_o, group_eff,
                    float(s_target),
                    eps=eps,
                    context=f"farm {dep}/{parent}/{child}",
                )

                for row in audit_rows[audit_start:]:
                    country = row["country"]
                    final = float(new_by_o.get(country, 0.0)) * _reference_price_factor(
                        country, low_income, markup=scenario_markup
                    )
                    change = final - float(row["initial_expenditure"])
                    row["final_expenditure"] = final
                    row["actual_donor_reduction"] = max(-change, 0.0) if row["weight_role"] == "donor" else 0.0
                    row["actual_receiver_addition"] = max(change, 0.0) if row["weight_role"] == "receiver" else 0.0
                    if row["weight_role"] == "receiver":
                        row["markup_adjustment"] = change - float(row["receiver_addition_before_markup"])

                _validate_reallocation(
                    new_by_o, group_eff,
                    float(s_target),
                    eps=eps,
                    context=f"farm {dep}/{parent}/{child}",
                )

                bucket = new_flows[dep][parent][child]
                old_by_value = {
                    norm(country): float(value)
                    for country, value in bucket.items()
                }
                for o in list(bucket.keys()):
                    bucket[o] = 0.0
                for o, v in new_by_o.items():
                    bucket[o] = float(v) * _reference_price_factor(
                        o, low_income, markup=scenario_markup
                    )

                moved_abs_total += 0.5 * float(sum(
                    abs(float(bucket.get(country, 0.0)) - float(old_by_value.get(country, 0.0)))
                    for country in set(old_by_value) | set(bucket)
                ))

    stats = {
        "delta_pp": float(delta_pp),
        "moved_abs_total": float(moved_abs_total),
        "donor_reduction_abs_total": float(donor_reduction_total),
        "unshiftable_abs_total": float(unshiftable_abs_total),
        "group_set_size_effective": int(len(group_eff)),
        "non_group_origins_present": sorted(list(non_group_eff)),
        "receiver_policy": str(receiver_policy),
        "single_receiver_country": None if single_receiver_country is None else norm(single_receiver_country),
        "single_donor_country": single_donor_country_norm,
        "receiver_whitelist": sorted(receiver_whitelist) if receiver_whitelist else [],
        "donor_whitelist": sorted(donor_whitelist) if donor_whitelist else [],
        "eu_spend_reduction_pct": scenario_markup,
        "scenario_markup": scenario_markup,
        "preserve_total": bool(preserve_total),
        "pricing_model_version": PRICE_ADJUSTMENT_MODEL_VERSION,
        "sourcing_allocation_model_version": SOURCING_ALLOCATION_MODEL_VERSION,
        "positive_gva_zero_expenditure_donors": positive_gva_zero_expenditure_donors,
        "fallback_donor_reallocation_abs_total": float(fallback_donor_reallocation_total),
    }
    if weight_audit is not None:
        stats["weight_audit"] = audit_rows
    return new_flows, stats

def build_delta_scenarios(
    *,
    group_set,
    df_alloc: pd.DataFrame,
    flows_by_component_hier_all: dict,
    low_income=(),
    ind_root: str | None = None,
    deltas_pp=range(-20, 21),
    receiver_policy: str = "baseline_mix",   
    gva_share_map: dict | None = None,
    gva_share_map_bucket: dict | None = None,
    weight_metadata: dict | None = None,
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

    for d_pp in deltas_pp:
        delta = d_pp / 100.0

        audit_turbine = []
        audit_farm = []

        df_alloc_cf, stats_turb = apply_delta_lc_shift_direct_turbine(
            df_alloc,
            group_set=group_set,
            low_income=low_income,
            delta_pp=delta,
            receiver_policy=receiver_policy,
            gva_share_map=gva_share_map,
            gva_share_map_bucket=gva_share_map_bucket,
            weight_metadata=weight_metadata,
            single_receiver_country=single_receiver_country,  
            single_donor_country=single_donor_country,
            component_filter=component_filter_turbine,
            receiver_whitelist=receiver_whitelist,
            donor_whitelist=donor_whitelist,
            eu_spend_reduction_pct=eu_spend_reduction_pct,
            preserve_total=preserve_total,
            weight_audit=audit_turbine,
        )

        flows_cf, stats_farm = apply_delta_lc_shift_direct_farm(
            flows_by_component_hier_all,
            group_set=group_set,
            low_income=low_income,
            delta_pp=delta,
            receiver_policy=receiver_policy,
            gva_share_map=gva_share_map,
            gva_share_map_bucket=gva_share_map_bucket,
            weight_metadata=weight_metadata,
            single_receiver_country=single_receiver_country,  # NEW
            single_donor_country=single_donor_country,
            component_filter=component_filter_farm,
            receiver_whitelist=receiver_whitelist,
            donor_whitelist=donor_whitelist,
            eu_spend_reduction_pct=eu_spend_reduction_pct,
            preserve_total=preserve_total,
            weight_audit=audit_farm,
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
import hashlib
import shutil
from typing import Any

# -----------------------------
# Cache helpers (pickle)
# -----------------------------
SCENARIO_CACHE_VERSION = "eu27-sourcing-v10-quantity-recalibrated-mu-eu27-target-bucket-total-gva-donor"

# Production cache names are part of the reproducible scenario contract.  The
# notebook used to construct descriptive names that had never been populated,
# even though the corresponding production calculations were already present
# under these stable names.  Keep the mapping explicit: it is intentionally
# not a fuzzy cache lookup, so a scenario cannot be silently substituted for a
# different markup or component filter.
SCENARIO_CACHE_REGISTRY: dict[tuple[str, float], str] = {
    ("macro", 0.30): "v10_quantity_recalibrated_mu/EU27_resilience_Turbine_BoP",
    ("macro", 0.10): "v10_quantity_recalibrated_mu/EU27_resilience_Turbine_BoP_markup_low",
    ("macro", 0.50): "v10_quantity_recalibrated_mu/EU27_resilience_Turbine_BoP_markup_high",
    ("marginal", 0.30): "v10_quantity_recalibrated_mu/EU27_resilience_marginal_all_components_central",
}

# Optimized terminal/source detail is an additive artifact.  Legacy aggregate
# results remain immutable and usable for consumers that do not need the
# source-country decomposition.
OPTIMIZED_DETAIL_CACHE_FILENAME = "optimized_detailed_v1.pkl"

# Baseline recipient weights do not depend on delta, markup, or the scenario
# family.  Keep a process-local cache so the notebook's central, sensitivity,
# and marginal runs reuse the exact same detailed bucket calculation.
_BASELINE_WEIGHT_CACHE: dict[str, tuple] = {}


def _fingerprint(value) -> str:
    h = hashlib.sha256()
    if isinstance(value, pd.DataFrame):
        h.update(repr((value.shape, value.index.names, value.columns.names)).encode())
        h.update(pd.util.hash_pandas_object(value, index=True).to_numpy().tobytes())
    elif isinstance(value, pd.Series):
        h.update(repr((value.shape, value.index.names, value.name)).encode())
        h.update(pd.util.hash_pandas_object(value, index=True).to_numpy().tobytes())
    elif isinstance(value, dict):
        for key in sorted(value, key=lambda x: repr(x)):
            h.update(repr(key).encode())
            h.update(_fingerprint(value[key]).encode())
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in sorted(value, key=lambda x: repr(x)) if isinstance(value, (set, frozenset)) else value:
            h.update(_fingerprint(item).encode())
    else:
        h.update(repr(value).encode())
    return h.hexdigest()


def _mapping_source_signature(ind_root: str | None) -> list[tuple[str, int, int]]:
    if not ind_root or not os.path.isdir(ind_root):
        return []
    entries = []
    for root, _, files in os.walk(ind_root):
        for name in sorted(files):
            if name.lower().endswith((".xlsx", ".xls", ".csv")):
                path = os.path.join(root, name)
                stat = os.stat(path)
                entries.append((os.path.relpath(path, ind_root), stat.st_size, stat.st_mtime_ns))
    return entries


def _scenario_cache_signature(**kwargs) -> str:
    payload = {
        "version": SCENARIO_CACHE_VERSION,
        **kwargs,
    }
    return _fingerprint(payload)


def _cache_paths(cache_dir: str, *, create: bool = True):
    if create:
        os.makedirs(cache_dir, exist_ok=True)
    return {
        "scenarios_delta": os.path.join(cache_dir, "scenarios_delta.pkl"),
        "scenarios_result": os.path.join(cache_dir, "scenarios_result.pkl"),
        "metadata": os.path.join(cache_dir, "metadata.pkl"),
    }


def _cache_metadata_is_current(cache_dir: str) -> bool:
    """Return whether a derived cache belongs to the current price model."""
    path = _cache_paths(cache_dir, create=False)["metadata"]
    if not os.path.exists(path):
        return False
    try:
        metadata = _pickle_load(path)
    except Exception:
        return False
    return bool(
        isinstance(metadata, dict)
        and metadata.get("version") == SCENARIO_CACHE_VERSION
        and metadata.get("pricing_model_version") == PRICE_ADJUSTMENT_MODEL_VERSION
        and metadata.get("sourcing_allocation_model_version") == SOURCING_ALLOCATION_MODEL_VERSION
    )


def _optimized_detail_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, OPTIMIZED_DETAIL_CACHE_FILENAME)

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
    cache_signature: str | None = None,
    cache_base_signature: str | None = None,
    cache_deltas: list[int] | tuple[int, ...] | None = None,
) -> dict:
    """
    Saves:
      - scenarios_result: directly
      - scenarios_delta: only the minimum needed (final_cost_df_cf + stats), so it's robust.
    """
    paths = _cache_paths(cache_dir)

    # Force-recompute writes remain additive: retain the first pre-existing
    # cache generation before replacing the active files.  This protects old
    # legacy results while keeping the established cache directory/name.
    for name, path in paths.items():
        if not os.path.exists(path):
            continue
        legacy_path = path.replace(".pkl", ".legacy.pkl")
        if not os.path.exists(legacy_path):
            shutil.copy2(path, legacy_path)

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
    _pickle_save(
        {
            "version": SCENARIO_CACHE_VERSION,
            "sourcing_allocation_model_version": SOURCING_ALLOCATION_MODEL_VERSION,
            "pricing_model_version": PRICE_ADJUSTMENT_MODEL_VERSION,
            "signature": cache_signature,
            "base_signature": cache_base_signature,
            "deltas": None if cache_deltas is None else [int(x) for x in cache_deltas],
        },
        paths["metadata"],
    )

    return {"ok": True, "paths": paths}

def load_cache(
    *,
    cache_dir: str = "cache_delta",
) -> tuple[Dict[int, "ScenarioDelta"], Dict[int, dict]]:
    paths = _cache_paths(cache_dir, create=False)

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


def _legacy_aggregate_results_available(
    scenarios_delta: dict,
    scenarios_result: dict,
    requested_deltas: tuple[int, ...],
) -> bool:
    """Return whether a cache has the two legacy, reusable stages (A and B)."""
    return all(
        delta in scenarios_delta
        and delta in scenarios_result
        and isinstance(scenarios_result[delta], dict)
        and "results_by_origin_lvl" in scenarios_result[delta]
        and "results_by_origin_sub_lvl" in scenarios_result[delta]
        for delta in requested_deltas
    )


def _cache_markup_matches(
    scenarios_delta: dict,
    requested_deltas: tuple[int, ...],
    markup: float,
) -> bool:
    """Check the auditable markup recorded with an existing allocation cache."""
    for delta in requested_deltas:
        scenario = scenarios_delta.get(delta)
        if scenario is None:
            return False
        stats = (scenario.stats_turbine or {}) | (scenario.stats_farm or {})
        recorded = stats.get("eu_spend_reduction_pct")
        if recorded is not None and not np.isclose(float(recorded), float(markup)):
            return False
    return True


def _cache_pricing_model_matches(
    scenarios_delta: dict,
    requested_deltas: tuple[int, ...],
) -> bool:
    """Require explicit provenance for bidirectional supplier pricing."""
    for delta in requested_deltas:
        scenario = scenarios_delta.get(delta)
        if scenario is None:
            return False
        stats = (scenario.stats_turbine or {}) | (scenario.stats_farm or {})
        if stats.get("pricing_model_version") != PRICE_ADJUSTMENT_MODEL_VERSION:
            return False
    return True


def _cache_allocation_model_matches(
    scenarios_delta: dict,
    requested_deltas: tuple[int, ...],
) -> bool:
    """Require provenance for baseline-total-GVA donor allocation weights."""
    for delta in requested_deltas:
        scenario = scenarios_delta.get(delta)
        if scenario is None:
            return False
        stats = (scenario.stats_turbine or {}) | (scenario.stats_farm or {})
        if stats.get("sourcing_allocation_model_version") != SOURCING_ALLOCATION_MODEL_VERSION:
            return False
    return True






def _load_optimized_detail_artifact(cache_dir: str) -> dict[int, dict]:
    """Load the additive optimized-detail artifact, if it exists."""
    path = _optimized_detail_path(cache_dir)
    if not os.path.exists(path):
        return {}
    payload = _pickle_load(path)
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError(f"Unsupported optimized detail artifact: {path}")
    details = payload.get("details", {})
    if not isinstance(details, dict):
        raise ValueError(f"Invalid optimized detail artifact: {path}")
    return {int(delta): value for delta, value in details.items()}


def _attach_optimized_details(
    scenarios_result: dict[int, dict],
    details: dict[int, dict],
) -> dict[int, dict]:
    """Attach additive detail to an in-memory copy of aggregate results."""
    out = {}
    for delta, result in scenarios_result.items():
        updated = dict(result)
        detail = details.get(int(delta))
        if isinstance(detail, dict):
            updated.update({
                key: detail[key]
                for key in (
                    "_detailed",
                    "_crosswalk",
                    "results_by_origin_lvl",
                    "results_by_origin_sub_lvl",
                )
                if key in detail
            })
        out[int(delta)] = updated
    return out


def upgrade_cached_detailed_results(
    *,
    cache_dir: str,
    cache_base_dir=None,
    deltas_pp: Iterable[int],
    L1_MRIO,
    VA_MRIO,
    LB_MRIO=None,
    W_MRIO=None,
    core=None,
    mrio=None,
    annualize: bool = True,
    lifetime_years: int = 25,
    verbose: bool = True,
) -> dict[int, dict]:
    """Evaluate only missing optimized detail from cached final-demand inputs.

    This is deliberately an economic-results upgrade: it reads the existing
    ``scenarios_delta.pkl`` and writes *only*
    ``optimized_detailed_v1.pkl``.  Legacy aggregate results and sourcing
    allocations are never replaced.
    """
    resolved = _resolve_cache_dir(cache_dir, cache_base_dir)
    scenarios_delta, scenarios_result = load_cache(cache_dir=resolved)
    requested = tuple(dict.fromkeys(int(x) for x in deltas_pp))
    if not _legacy_aggregate_results_available(scenarios_delta, scenarios_result, requested):
        raise RuntimeError(
            "Cannot upgrade optimized detail because cached sourcing and aggregate "
            f"results are incomplete for deltas {list(requested)}: {resolved}."
        )
    existing = _load_optimized_detail_artifact(resolved)
    missing = [delta for delta in requested if delta not in existing]
    if missing:
        if verbose:
            print(
                "[upgrade_cached_detailed_results] Evaluating optimized detail from "
                f"cached final demand for deltas {missing}: {resolved}"
            )
        evaluated = run_mrio_for_scenarios(
            {delta: scenarios_delta[delta] for delta in missing},
            L1_MRIO=L1_MRIO,
            VA_MRIO=VA_MRIO,
            LB_MRIO=LB_MRIO,
            W_MRIO=W_MRIO,
            core=core,
            mrio=mrio,
            annualize=annualize,
            lifetime_years=lifetime_years,
            fallback_verbose=False,
            progress=bool(verbose),
        )
        existing.update({
            int(delta): {
                "_detailed": evaluated[delta]["_detailed"],
                "_crosswalk": evaluated[delta]["_crosswalk"],
                # Use the refreshed aggregate pair together with refreshed
                # detail in-memory.  In particular this prevents historical
                # Jobs (with an unknown wage fingerprint) being compared to a
                # newly evaluated EU27-side Jobs decomposition.
                "results_by_origin_lvl": evaluated[delta]["results_by_origin_lvl"],
                "results_by_origin_sub_lvl": evaluated[delta]["results_by_origin_sub_lvl"],
            }
            for delta in missing
        })
        _pickle_save(
            {
                "version": 1,
                "source_cache_version": SCENARIO_CACHE_VERSION,
                "details": existing,
            },
            _optimized_detail_path(resolved),
        )
    return _attach_optimized_details(
        {delta: scenarios_result[delta] for delta in requested},
        existing,
    )

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


def _normalize_scenario_execution_mode(
    execution_mode: str | None,
    *,
    force_recompute: bool,
) -> str:
    if execution_mode is None:
        return "force_recompute" if force_recompute else "cache_only"
    mode = str(execution_mode).strip().lower()
    if mode not in {"cache_only", "compute_missing", "force_recompute"}:
        raise ValueError(
            "execution_mode must be one of 'cache_only', 'compute_missing', or 'force_recompute'."
        )
    if force_recompute and mode != "force_recompute":
        raise ValueError(
            "force_recompute=True conflicts with execution_mode="
            f"{execution_mode!r}; use execution_mode='force_recompute'."
        )
    return mode

def maybe_run_or_load(
    flows_by_component_hier_all,
    df_alloc,
    ind_root: str | None = None,
    ROOT: str | None = None,
    cache_dir: str = "cache_delta",
    cache_base_dir=None,                    
    force_recompute: bool = False,
    execution_mode: str | None = None,
    require_detailed: bool = False,
    allow_legacy_aggregate: bool = True,
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
    LOW_INCOME=(),
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
    execution_mode = _normalize_scenario_execution_mode(
        execution_mode,
        force_recompute=force_recompute,
    )
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

    # The counterfactual delta is defined on EU27 procurement shares.  Keep
    # the legacy EUROPE parameter as a compatibility check, but never use the
    # broader geographical EUROPE constant as the sourcing group.
    sourcing_group = {
        str(country).strip().upper().replace("UK", "GB")
        for country in (EU27 or ())
        if str(country).strip()
    }
    if not sourcing_group:
        raise ValueError("EU27 must be provided as the sourcing target group.")
    if EUROPE is not None:
        requested_group = {
            str(country).strip().upper().replace("UK", "GB")
            for country in EUROPE
            if str(country).strip()
        }
        if requested_group != sourcing_group:
            raise ValueError(
                "The sourcing target group must equal EU27; broader EUROPE "
                "sets are not valid for delta-based reallocations."
            )

    # Auto-detect industry code prefix to match MRIO (CPA_ vs bare codes).
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

    signature_args = dict(
        deltas_pp=list(deltas_pp),
        group_set=sourcing_group,
        df_alloc=df_alloc,
        flows=flows_by_component_hier_all,
        mapping_sources=_mapping_source_signature(ind_root),
        ind_ind=bool(ind_ind),
        receiver_policy=receiver_policy,
        single_receiver_country=single_receiver_country,
        single_donor_country=single_donor_country,
        component_filter_farm=component_filter_farm,
        component_filter_turbine=component_filter_turbine,
        receiver_whitelist=receiver_whitelist,
        donor_whitelist=donor_whitelist,
        eu_spend_reduction_pct=eu_spend_reduction_pct,
        preserve_total=preserve_total,
        annualize=annualize,
        lifetime_years=lifetime_years,
        connected_country=connected_country,
        weight_key=weight_key,
        low_income=LOW_INCOME,
        capacity_args=(capacity_lvl, capacity_components, capacity_targets, capacity_include_row,
                       capacity_year, capacity_sector, capacity_gva_col),
        observed_results=(results_by_origin_lvl, results_by_origin_sub_lvl),
        L1_MRIO=L1_MRIO,
        VA_MRIO=VA_MRIO,
        LB_MRIO=LB_MRIO,
        W_MRIO=W_MRIO,
        mrio=mrio,
    )
    cache_signature = _scenario_cache_signature(**signature_args)
    cache_base_signature = _scenario_cache_signature(
        **{**signature_args, "deltas_pp": None}
    )

    cache_dir = _resolve_cache_dir(cache_dir, cache_base_dir)
    paths = _cache_paths(cache_dir, create=False)
    have_cache = os.path.exists(paths["scenarios_delta"]) and os.path.exists(paths["scenarios_result"])
    try:
        cache_metadata = _pickle_load(paths["metadata"]) if os.path.exists(paths["metadata"]) else {}
    except Exception:
        cache_metadata = {}
    cache_valid = have_cache and cache_metadata.get("signature") == cache_signature
    legacy_cache_files = have_cache and not cache_metadata.get("base_signature")
    write_cache_dir = cache_dir
    if (
        execution_mode == "compute_missing"
        and legacy_cache_files
        and not force_recompute
    ):
        # A partial legacy cache may need new sourcing or aggregate results.
        # Preserve the old A/B files and write the completed generation beside
        # them.  Detail-only upgrades use their additive artifact above.
        write_cache_dir = f"{cache_dir}_optimized_v1"

    requested_deltas = tuple(dict.fromkeys(int(x) for x in deltas_pp))
    cached_delta = None
    cached_result = None
    cached_detail = {}
    if have_cache:
        try:
            cached_delta, cached_result = load_cache(cache_dir=cache_dir)
            cached_detail = _load_optimized_detail_artifact(cache_dir)
            cached_result = _attach_optimized_details(cached_result, cached_detail)
        except Exception:
            cached_delta = None
            cached_result = None
            cached_detail = {}

    legacy_aggregate_available = (
        cached_delta is not None
        and cached_result is not None
        and _legacy_aggregate_results_available(cached_delta, cached_result, requested_deltas)
        and _cache_markup_matches(cached_delta, requested_deltas, eu_spend_reduction_pct)
        and _cache_pricing_model_matches(cached_delta, requested_deltas)
        and _cache_allocation_model_matches(cached_delta, requested_deltas)
    )
    detailed_available = legacy_aggregate_available and all(
        isinstance(cached_result[delta].get("_detailed"), pd.DataFrame)
        and isinstance(cached_result[delta].get("_crosswalk"), pd.DataFrame)
        for delta in requested_deltas
    )
    if (
        execution_mode == "compute_missing"
        and not force_recompute
        and have_cache
        and cache_metadata.get("base_signature") == cache_base_signature
    ):
        cached_keys = {int(x) for x in cached_delta}
        missing_deltas = [x for x in requested_deltas if x not in cached_keys]
        if not missing_deltas:
            if verbose:
                print(f"[maybe_run_or_load] Loading compatible delta subset: {cache_dir}")
            return (
                {x: cached_delta[x] for x in requested_deltas},
                {x: cached_result[x] for x in requested_deltas},
            )
        deltas_to_compute = missing_deltas
    else:
        deltas_to_compute = list(requested_deltas)

    if cache_valid and execution_mode != "force_recompute" and (
        not require_detailed or detailed_available
    ):
        if verbose:
            print(f"[maybe_run_or_load] Loading cache: {cache_dir}")
        return cached_delta, cached_result

    # Old production caches predate the full signature schema and optimized
    # source detail.  Their final-demand and aggregate-result stages remain
    # valid when the recorded allocation markup matches; do not force a
    # sourcing rebuild merely because presentation or result-schema metadata
    # has changed.
    if (
        allow_legacy_aggregate
        and legacy_aggregate_available
        and execution_mode != "force_recompute"
        and (not require_detailed or detailed_available)
    ):
        if verbose:
            detail_note = "with optimized detail" if detailed_available else "without optimized detail"
            print(f"[maybe_run_or_load] Loading legacy aggregate cache ({detail_note}): {cache_dir}")
        return (
            {delta: cached_delta[delta] for delta in requested_deltas},
            {delta: cached_result[delta] for delta in requested_deltas},
        )

    if (
        execution_mode == "compute_missing"
        and legacy_aggregate_available
        and require_detailed
        and not detailed_available
    ):
        upgraded = upgrade_cached_detailed_results(
            cache_dir=cache_dir,
            deltas_pp=requested_deltas,
            L1_MRIO=L1_MRIO,
            VA_MRIO=VA_MRIO,
            LB_MRIO=LB_MRIO,
            W_MRIO=W_MRIO,
            core=core,
            mrio=mrio,
            annualize=annualize,
            lifetime_years=lifetime_years,
            verbose=verbose,
        )
        return (
            {delta: cached_delta[delta] for delta in requested_deltas},
            upgraded,
        )

    if execution_mode == "cache_only":
        if have_cache and not _cache_pricing_model_matches(cached_delta or {}, requested_deltas):
            reason = (
                "cached sourcing uses an older pricing model; it is invalid under "
                f"{PRICE_ADJUSTMENT_MODEL_VERSION}"
            )
        elif have_cache and not _cache_allocation_model_matches(cached_delta or {}, requested_deltas):
            reason = (
                "cached sourcing uses an older expenditure-weighted donor allocation model; "
                "it is invalid under "
                f"{SOURCING_ALLOCATION_MODEL_VERSION}"
            )
        elif have_cache and require_detailed and legacy_aggregate_available:
            reason = (
                "legacy aggregate results are available, but optimized source detail is missing. "
                "Use execution_mode='compute_missing' to evaluate detail from the cached "
                "final_cost_df_cf without rebuilding sourcing"
            )
        elif have_cache:
            reason = "metadata signature does not match the current scenario specification"
        else:
            missing_files = [name for name, path in paths.items() if name != "metadata" and not os.path.exists(path)]
            reason = f"missing cache files: {missing_files}"
        raise RuntimeError(
            "Scenario execution is cache_only and no compatible cache is available. "
            f"Cache: {cache_dir}. Reason: {reason}. "
            "Set execution_mode='compute_missing' or 'force_recompute' explicitly to calculate it."
        )

    if verbose:
        reason = "force_recompute" if execution_mode == "force_recompute" else ("cache invalid" if have_cache else "cache files missing")
        print(f"[maybe_run_or_load] Computing ({reason}): {cache_dir}")

    if mrio is None or L1_MRIO is None or VA_MRIO is None:
        raise ValueError(
            "The sourcing scenario now requires MRIO, L1_MRIO, and VA_MRIO "
            "to calculate bucket-level baseline upstream-GVA recipient weights."
        )

    # Build exact bucket-level baseline weights once. No aggregate component
    # result is substituted when an exact bucket can be reconstructed.  The
    # key deliberately excludes delta and markup, but includes every input
    # that can affect the detailed baseline response.
    weight_cache_key = _scenario_cache_signature(
        df_alloc=df_alloc,
        flows=flows_by_component_hier_all,
        mapping_sources=_mapping_source_signature(ind_root),
        ind_ind=bool(ind_ind),
        mrio=mrio,
        L1_MRIO=L1_MRIO,
        VA_MRIO=VA_MRIO,
        EU27=EU27,
        low_income=LOW_INCOME,
        lifetime_years=lifetime_years,
    )
    if weight_cache_key in _BASELINE_WEIGHT_CACHE:
        (
            gva_share_map_bucket,
            baseline_upstream_gva,
            recipient_weights,
            weight_metadata,
        ) = _BASELINE_WEIGHT_CACHE[weight_cache_key]
    else:
        (
            gva_share_map_bucket,
            baseline_upstream_gva,
            recipient_weights,
            weight_metadata,
        ) = build_bucket_baseline_gva_weights(
            df_alloc=df_alloc,
            flows_by_component_hier_all=flows_by_component_hier_all,
            low_income=LOW_INCOME,
            ind_root=ind_root,
            ind_ind=bool(ind_ind),
            mrio=mrio,
            L1_MRIO=L1_MRIO,
            VA_MRIO=VA_MRIO,
            EU27=EU27,
            lifetime_years=lifetime_years,
        )
        _BASELINE_WEIGHT_CACHE[weight_cache_key] = (
            gva_share_map_bucket,
            baseline_upstream_gva,
            recipient_weights,
            weight_metadata,
        )

    scenarios_delta = build_delta_scenarios(
        group_set=sourcing_group,
        df_alloc=df_alloc,
        flows_by_component_hier_all=flows_by_component_hier_all,
        low_income=LOW_INCOME,
        ind_root=ind_root,
        deltas_pp=deltas_to_compute,
        receiver_policy=receiver_policy,
        gva_share_map=None,
        gva_share_map_bucket=gva_share_map_bucket,
        weight_metadata=weight_metadata,
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
    audit_anchor = scenarios_delta.get(0) or next(iter(scenarios_delta.values()))
    audit_anchor.stats_turbine["_baseline_upstream_gva"] = baseline_upstream_gva.to_dict("records")
    audit_anchor.stats_turbine["_recipient_weights"] = recipient_weights.to_dict("records")
    scenarios_result = run_mrio_for_scenarios(
        scenarios_delta,
        L1_MRIO=L1_MRIO, VA_MRIO=VA_MRIO, LB_MRIO=LB_MRIO, W_MRIO=W_MRIO,
        core=core,
        mrio=mrio,
        annualize=annualize,
        lifetime_years=lifetime_years,
        fallback_verbose=False,
        progress=bool(verbose),
    )

    # In compute_missing mode, retain compatible cached deltas and append only
    # the newly evaluated cases.  This keeps the cache complete for the
    # requested scenario family and avoids silently discarding prior work.
    if cached_delta is not None and cached_result is not None:
        merged_delta = {
            delta: cached_delta[delta]
            for delta in requested_deltas
            if delta in cached_delta
        }
        merged_result = {
            delta: cached_result[delta]
            for delta in requested_deltas
            if delta in cached_result
        }
        merged_delta.update(scenarios_delta)
        merged_result.update(scenarios_result)
        scenarios_delta = merged_delta
        scenarios_result = merged_result

    save_cache(
        scenarios_delta=scenarios_delta,
        scenarios_result=scenarios_result,
        cache_dir=write_cache_dir,
        cache_signature=cache_signature,
        cache_base_signature=cache_base_signature,
        cache_deltas=requested_deltas,
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


def _derive_eu27_side_results_from_detailed(
    net_scenarios_delta: Dict[int, "ScenarioDelta"],
    net_results_delta: Dict[int, dict],
    *,
    eu_countries,
    deltas_pp: list[int] | tuple[int, ...],
) -> dict[int, dict] | None:
    """Derive EU27-side results from cached source-level net responses.

    ``evaluate_detailed_mrio_results`` retains the sourcing origin for every
    terminal demand column.  Since the MRIO response is linear, only the EU27
    source-origin change needs to be applied to the complete baseline.  This
    avoids a second MRIO evaluation when the net cache was produced by the
    current engine; legacy caches without ``_detailed`` fall back to the
    existing side-run path.
    """
    if 0 not in net_results_delta:
        return None
    requested = tuple(int(x) for x in deltas_pp)
    if any(
        delta not in net_results_delta
        or not isinstance(net_results_delta[delta].get("_detailed"), pd.DataFrame)
        or not isinstance(net_results_delta[delta].get("_crosswalk"), pd.DataFrame)
        for delta in requested
    ):
        return None

    eu_set = {str(x).strip().upper().replace("UK", "GB") for x in (eu_countries or [])}
    if not eu_set:
        return None
    metric_cols = MRIO_RESULT_COLUMNS[4:]
    hierarchy_cols = [
        "component_group_L0", "sub_component_L0",
        "component_group_L1", "sub_component_L1",
        "component_group_L2", "sub_component_L2",
    ]
    key_cols = [
        "Deployment", "SourceCountry", "Country", "Year", "Sector",
        *hierarchy_cols,
    ]
    baseline_crosswalk = net_results_delta[0]["_crosswalk"]
    baseline = net_results_delta[0]["_detailed"]
    if any(col not in baseline.columns for col in ("SourceCountry", *key_cols, *metric_cols)):
        return None
    baseline = baseline[key_cols + metric_cols].copy()
    baseline["SourceCountry"] = baseline["SourceCountry"].astype(str).str.strip()
    out = {}
    for delta in requested:
        target = net_results_delta[delta]["_detailed"][key_cols + metric_cols].copy()
        target["SourceCountry"] = target["SourceCountry"].astype(str).str.strip()
        base_g = baseline.groupby(key_cols, dropna=False, as_index=False)[metric_cols].sum()
        target_g = target.groupby(key_cols, dropna=False, as_index=False)[metric_cols].sum()
        merged = base_g.merge(
            target_g,
            on=key_cols,
            how="outer",
            suffixes=("_base", "_target"),
        )
        for col in metric_cols:
            merged[f"{col}_base"] = pd.to_numeric(merged[f"{col}_base"], errors="coerce").fillna(0.0)
            merged[f"{col}_target"] = pd.to_numeric(merged[f"{col}_target"], errors="coerce").fillna(0.0)
        source_label = merged["SourceCountry"].astype(str).str.upper()
        is_eu = source_label.isin(eu_set) | source_label.eq("__EU27_SOURCE__")
        side = merged[key_cols].copy()
        for col in metric_cols:
            change = merged[f"{col}_target"] - merged[f"{col}_base"]
            side[col] = merged[f"{col}_base"] + np.where(is_eu, change, 0.0)
        # ``leaf_id`` is local to a scenario's observed leaves and can be
        # renumbered when a full offshoring case removes a leaf.  Join and
        # aggregate through stable hierarchy labels, then recover the baseline
        # canonical leaf id.  The baseline contains the complete project.
        crosswalk = baseline_crosswalk
        leaf_lookup = crosswalk[[
            "leaf_id", "component_group_L0", "sub_component_L0"
        ]].drop_duplicates()
        side = side.merge(
            leaf_lookup,
            on=["component_group_L0", "sub_component_L0"],
            how="left",
            validate="many_to_one",
        )
        if side["leaf_id"].isna().any():
            missing_leaves = side.loc[
                side["leaf_id"].isna(),
                ["component_group_L0", "sub_component_L0"],
            ].drop_duplicates()
            raise ValueError(
                "EU27-side detailed results contain leaves absent from the baseline crosswalk: "
                f"{missing_leaves.to_dict('records')}"
            )
        side["leaf_id"] = side["leaf_id"].astype(int)
        # Match the legacy side-scenario structure: a terminal leaf whose
        # complete EU27-side level is exactly zero is absent from its
        # final-demand scenario rather than retained as a zero-valued result
        # node (e.g. a fully offshored Blades leaf).
        active_leaf_ids = set(
            side.groupby("leaf_id", dropna=False)["Output_Direct"]
            .sum()
            .abs()
            .gt(1e-9)
            .loc[lambda values: values]
            .index.astype(int)
        )
        side = side[side["leaf_id"].isin(active_leaf_ids)].copy()
        crosswalk = crosswalk[crosswalk["leaf_id"].isin(active_leaf_ids)].copy()
        by_lvl, by_sub = _aggregate_detailed_mrio_results(
            side,
            crosswalk,
            year=2023,
        )
        out[delta] = {
            "results_by_origin_lvl": by_lvl,
            "results_by_origin_sub_lvl": by_sub,
            "_detailed": side,
            "_crosswalk": crosswalk,
        }
    return out


def maybe_run_gross_expansion_from_net(
    *,
    net_scenarios_delta: Dict[int, "ScenarioDelta"],
    cache_dir: str = "gross_expansion",
    cache_base_dir=None,
    force_recompute: bool = False,
    execution_mode: str | None = None,
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
    execution_mode = _normalize_scenario_execution_mode(
        execution_mode,
        force_recompute=force_recompute,
    )
    cache_dir = _resolve_cache_dir(cache_dir, cache_base_dir)
    paths = _cache_paths(cache_dir, create=False)
    have_cache = os.path.exists(paths["scenarios_delta"]) and os.path.exists(paths["scenarios_result"])
    if have_cache and execution_mode != "force_recompute" and not _cache_metadata_is_current(cache_dir):
        have_cache = False
    requested_deltas = tuple(dict.fromkeys(int(x) for x in deltas_pp))

    if have_cache and execution_mode != "force_recompute":
        cached_delta, cached_result = load_cache(cache_dir=cache_dir)
        missing_deltas = [
            delta for delta in requested_deltas
            if delta not in cached_delta or delta not in cached_result
        ]
        if missing_deltas and execution_mode == "cache_only":
            raise RuntimeError(
                "Gross-expansion execution is cache_only but the cache is missing "
                f"deltas {missing_deltas}. Cache: {cache_dir}."
            )
        if not missing_deltas:
            return (
                {delta: cached_delta[delta] for delta in requested_deltas},
                {delta: cached_result[delta] for delta in requested_deltas},
            )
        deltas_to_compute = missing_deltas
    else:
        deltas_to_compute = list(requested_deltas)

    if execution_mode == "cache_only":
        raise RuntimeError(
            "Gross-expansion execution is cache_only and no compatible cache is available. "
            f"Cache: {cache_dir}. Set execution_mode='compute_missing' or "
            "'force_recompute' explicitly to calculate it."
        )

    if have_cache and execution_mode == "compute_missing":
        if verbose:
            print(f"[maybe_run_gross_expansion_from_net] Computing missing deltas {deltas_to_compute}: {cache_dir}")
    elif verbose:
        reason = "force_recompute=True" if execution_mode == "force_recompute" else "cache files missing"
        print(f"[maybe_run_gross_expansion_from_net] Recomputing ({reason}): {cache_dir}")

    gross_delta = build_gross_expansion_scenarios_from_net(
        net_scenarios_delta,
        deltas_pp=deltas_to_compute,
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
    if have_cache and execution_mode == "compute_missing":
        cached_delta.update(gross_delta)
        cached_result.update(gross_result)
        gross_delta = {delta: cached_delta[delta] for delta in requested_deltas}
        gross_result = {delta: cached_result[delta] for delta in requested_deltas}
    save_cache(
        scenarios_delta=gross_delta,
        scenarios_result=gross_result,
        cache_dir=cache_dir,
    )
    return gross_delta, gross_result


def maybe_run_eu27_side_from_net(
    *,
    net_scenarios_delta: Dict[int, "ScenarioDelta"],
    net_results_delta: Dict[int, dict] | None = None,
    eu_countries,
    cache_dir: str = "eu27_side_reallocation",
    cache_base_dir=None,
    force_recompute: bool = False,
    execution_mode: str | None = None,
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
    progress: bool = True,
    allow_legacy_side_cache: bool = False,
):
    # Build/load MRIO results that isolate the EU27 side of net reallocation.
    execution_mode = _normalize_scenario_execution_mode(
        execution_mode,
        force_recompute=force_recompute,
    )
    requested_deltas = tuple(dict.fromkeys(int(x) for x in deltas_pp))

    # An optimized net response contains an exact EU27/non-EU27 source
    # partition.  Prefer this in-memory derivation to any historical side
    # cache: historical side allocations can belong to a different sourcing
    # scenario even when their cache names look similar.
    if net_results_delta is not None:
        derived = _derive_eu27_side_results_from_detailed(
            net_scenarios_delta,
            net_results_delta,
            eu_countries=eu_countries,
            deltas_pp=requested_deltas,
        )
        if derived is not None:
            eu_side_delta = build_eu27_side_scenarios_from_net(
                net_scenarios_delta,
                eu_countries=eu_countries,
                deltas_pp=requested_deltas,
            )
            if verbose:
                print("[maybe_run_eu27_side_from_net] Derived EU27-side results from net source detail")
            return eu_side_delta, derived

    cache_dir = _resolve_cache_dir(cache_dir, cache_base_dir)
    paths = _cache_paths(cache_dir, create=False)
    have_cache = os.path.exists(paths["scenarios_delta"]) and os.path.exists(paths["scenarios_result"])
    if (
        have_cache
        and execution_mode != "force_recompute"
        and (not allow_legacy_side_cache or not _cache_metadata_is_current(cache_dir))
    ):
        # Do not replace an unverifiable historical side cache.  A deliberate
        # fallback MRIO run, if ever required, gets its own versioned cache.
        cache_dir = f"{cache_dir}_optimized_v1"
        paths = _cache_paths(cache_dir, create=False)
        have_cache = os.path.exists(paths["scenarios_delta"]) and os.path.exists(paths["scenarios_result"])
        if have_cache and not _cache_metadata_is_current(cache_dir):
            have_cache = False

    if have_cache and execution_mode != "force_recompute" and allow_legacy_side_cache:
        cached_delta, cached_result = load_cache(cache_dir=cache_dir)
        cached_keys = set(cached_delta).intersection(cached_result)
        missing_deltas = [delta for delta in requested_deltas if delta not in cached_keys]
        if not missing_deltas:
            selected_delta = {delta: cached_delta[delta] for delta in requested_deltas}
            selected_result = {delta: cached_result[delta] for delta in requested_deltas}
            if verbose:
                print(f"[maybe_run_eu27_side_from_net] Loading cache: {cache_dir}")
            return selected_delta, selected_result
        if execution_mode == "cache_only":
            raise RuntimeError(
                "EU27-side execution is cache_only but the cache is missing "
                f"deltas {missing_deltas}. Cache: {cache_dir}."
            )
        deltas_to_compute = missing_deltas
    else:
        deltas_to_compute = list(requested_deltas)

    if have_cache and execution_mode == "force_recompute" and verbose:
        print(f"[maybe_run_eu27_side_from_net] Recomputing (force_recompute): {cache_dir}")
    elif have_cache and execution_mode == "compute_missing" and verbose:
        print(
            f"[maybe_run_eu27_side_from_net] Computing missing deltas "
            f"{deltas_to_compute}: {cache_dir}"
        )
    elif have_cache and execution_mode != "force_recompute":
        # The cache-only branch above already handled a complete cache.  This
        # guard documents that only compute_missing can reach the calculation.
        pass

    if execution_mode == "cache_only":
        detail_reason = (
            "The net cache has aggregate legacy results but no optimized source detail; "
            "run execution_mode='compute_missing' to derive source detail from its cached "
            "final_cost_df_cf. Historical EU27-side caches are intentionally not used unless "
            "their final-demand inputs have been verified compatible."
        )
        raise RuntimeError(
            "EU27-side execution is cache_only and no compatible current-net result is available. "
            f"Cache: {cache_dir}. {detail_reason}"
        )

    if verbose and not have_cache:
        reason = "force_recompute" if execution_mode == "force_recompute" else "cache files missing"
        print(f"[maybe_run_eu27_side_from_net] Computing ({reason}): {cache_dir}")

    eu_side_delta = build_eu27_side_scenarios_from_net(
        net_scenarios_delta,
        eu_countries=eu_countries,
        deltas_pp=deltas_to_compute,
    )
    eu_side_result = None
    if net_results_delta is not None:
        eu_side_result = _derive_eu27_side_results_from_detailed(
            net_scenarios_delta,
            net_results_delta,
            eu_countries=eu_countries,
            deltas_pp=deltas_to_compute,
        )
    if eu_side_result is None:
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
            progress=progress,
        )
    elif progress:
        for i, delta in enumerate(deltas_to_compute, start=1):
            print(
                f"[scenario EU27-side] delta {int(delta):+d} pp "
                f"({i}/{len(deltas_to_compute)}) derived from net response",
                flush=True,
            )
    save_cache(
        scenarios_delta=eu_side_delta,
        scenarios_result=eu_side_result,
        cache_dir=cache_dir,
    )
    if have_cache and execution_mode == "compute_missing":
        cached_delta.update(eu_side_delta)
        cached_result.update(eu_side_result)
        save_cache(
            scenarios_delta=cached_delta,
            scenarios_result=cached_result,
            cache_dir=cache_dir,
        )
        return (
            {delta: cached_delta[delta] for delta in requested_deltas},
            {delta: cached_result[delta] for delta in requested_deltas},
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

    split_row = {
        str(x).strip().upper()
        for x in split_row
        if str(x).strip().upper() not in {"ROW", "X01"}
    }
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
        if c in {"ROW", "X01"}:
            return ROW_LABEL
        if c in {"GB", "UK"}:
            return UK_LABEL
        if c in EU27:
            return EU_LABEL
        if c in split_row:
            return split_row_labels.get(c, c)
        if c in EUROPE_CODES:
            return ROE_LABEL
        return OTHER_ORIGINS_LABEL

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

        # Dynamic regions: EU27, UK, RoE, selected countries, unshown origins,
        # and the explicit FIGARO ROW model region.
        region_list = [EU_LABEL, UK_LABEL, ROE_LABEL]
        base_regions = set(region_list + [OTHER_ORIGINS_LABEL, ROW_LABEL])
        region_list += sorted([
            split_row_labels.get(x, x)
            for x in split_row
            if split_row_labels.get(x, x) not in base_regions
        ])
        region_list += [ROW_LABEL, OTHER_ORIGINS_LABEL]
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

def _prepare_counterfactual_region_panel(
    PANEL_D3: pd.DataFrame,
    *,
    component: str,
    split_row=(),
    split_row_labels: dict | None = None,
):
    """Prepare mutually exclusive regional totals using the existing D.3 buckets."""
    required = {
        "Component", "Region", "delta_pp", "scenario",
        "abs_gva_direct", "abs_gva_total", "abs_jobs_direct", "abs_jobs_total",
    }
    missing = sorted(required.difference(PANEL_D3.columns))
    if missing:
        raise KeyError(f"PANEL_D3 is missing columns: {missing}.")

    frame = PANEL_D3.loc[
        PANEL_D3["scenario"].astype(str).eq("baseline")
        & PANEL_D3["Component"].astype(str).str.strip().eq(str(component))
    ].copy()
    if frame.empty:
        raise ValueError(f"PANEL_D3 has no baseline rows for {component!r}.")

    split_codes = [
        str(code).strip().upper()
        for code in (split_row or ())
        if str(code).strip().upper() not in {"ROW", "X01"}
    ]
    split_labels = {
        code: ("China" if code == "CN" else CODE_TO_NAME.get(code, code))
        for code in split_codes
    } | (split_row_labels or {})
    split_regions = list(dict.fromkeys(split_labels[code] for code in split_codes))
    split_codes_set = set(split_codes)
    selected_uk = bool(split_codes_set & {"GB", "UK"})

    frame["Region"] = frame["Region"].astype(str).str.strip()
    frame["Region"] = frame["Region"].replace({"UK": UK_LABEL, "GB": UK_LABEL})
    eu_split_labels = {f"{EU_LABEL} (from EU)", f"{EU_LABEL} (from RoW)"}
    frame = frame.loc[~frame["Region"].isin(eu_split_labels)].copy()
    if not selected_uk:
        frame["Region"] = frame["Region"].replace(
            {ROE_LABEL: OTHER_ORIGINS_LABEL, UK_LABEL: OTHER_ORIGINS_LABEL}
        )
    else:
        frame["Region"] = frame["Region"].replace({ROE_LABEL: OTHER_ORIGINS_LABEL})

    base_regions = {EU_LABEL, UK_LABEL, ROE_LABEL, ROW_LABEL, OTHER_ORIGINS_LABEL}
    recognized_regions = base_regions | set(split_regions)
    frame.loc[~frame["Region"].isin(recognized_regions), "Region"] = OTHER_ORIGINS_LABEL

    frame["delta_pp"] = pd.to_numeric(frame["delta_pp"], errors="raise").astype(int)
    value_cols = ["abs_gva_direct", "abs_gva_total", "abs_jobs_direct", "abs_jobs_total"]
    for col in value_cols:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    if not np.isfinite(frame[value_cols].to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite regional values in {component!r}.")

    frame["_gva_indirect"] = frame["abs_gva_total"] - frame["abs_gva_direct"]
    frame["_jobs_indirect"] = frame["abs_jobs_total"] - frame["abs_jobs_direct"]
    regional = (
        frame.groupby(["delta_pp", "Region"], as_index=False)[
            value_cols + ["_gva_indirect", "_jobs_indirect"]
        ].sum()
    )
    if not np.allclose(
        regional["abs_gva_direct"] + regional["_gva_indirect"],
        regional["abs_gva_total"], rtol=1e-11, atol=1e-7,
    ) or not np.allclose(
        regional["abs_jobs_direct"] + regional["_jobs_indirect"],
        regional["abs_jobs_total"], rtol=1e-11, atol=1e-9,
    ):
        raise AssertionError(f"Direct plus indirect does not reproduce {component!r} regional totals.")

    deltas = np.array(sorted(regional["delta_pp"].unique()), dtype=int)
    split_rank = [split_labels[code] for code in split_codes]
    order = [EU_LABEL]
    if selected_uk:
        order.append(UK_LABEL)
    order += [r for r in split_rank if r not in base_regions]
    order += [ROW_LABEL, OTHER_ORIGINS_LABEL]
    order = list(dict.fromkeys(order))
    activity = (
        regional.groupby("Region")[["abs_gva_total", "abs_jobs_total"]]
        .agg(lambda series: float(np.abs(series.to_numpy(dtype=float)).max()))
        .max(axis=1)
    )
    order = [r for r in order if float(activity.get(r, 0.0)) > 1e-12]
    if not order:
        raise ValueError(f"No nonzero regional totals remain for {component!r}.")
    unknown = set(regional["Region"].unique()).difference(order)
    if unknown:
        raise AssertionError(f"Regional activity was omitted from the D.3 order: {sorted(unknown)}.")

    expected_global = frame.groupby("delta_pp")[["abs_gva_total", "abs_jobs_total"]].sum().reindex(deltas)
    actual_global = regional.groupby("delta_pp")[["abs_gva_total", "abs_jobs_total"]].sum().reindex(deltas)
    for col in expected_global.columns:
        if not np.allclose(expected_global[col], actual_global[col], rtol=1e-11, atol=1e-7):
            raise AssertionError(f"Geographic aggregation changed {col} for {component!r}.")

    grid = pd.MultiIndex.from_product(
        [deltas, order], names=["delta_pp", "Region"]
    ).to_frame(index=False)
    regional = grid.merge(regional, on=["delta_pp", "Region"], how="left").fillna(0.0)
    gva = (
        regional.pivot(index="delta_pp", columns="Region", values="abs_gva_total")
        .reindex(index=deltas, columns=order).fillna(0.0) / 1_000_000.0
    )
    employment = (
        regional.pivot(index="delta_pp", columns="Region", values="abs_jobs_total")
        .reindex(index=deltas, columns=order).fillna(0.0)
    )
    if (gva.to_numpy(dtype=float) < -1e-8).any() or (employment.to_numpy(dtype=float) < -1e-8).any():
        raise ValueError(f"Absolute regional levels are negative for {component!r}; refusing to clip them.")

    eu_fc = COLOR_MAP.get("EU27") or COLOR_MAP.get(EU_LABEL) or "#4C78A8"
    roe_fc = COLOR_MAP.get("EU_dep") or COLOR_MAP.get("RoE") or COLOR_MAP.get(ROE_LABEL) or "#4F6BD1"
    styles = {
        EU_LABEL: dict(fc=eu_fc, ec=eu_fc, ls="solid", lw=0.6),
        UK_LABEL: dict(fc=COLOR_MAP.get("GB") or COLOR_MAP.get("UK") or "#5C1F1F", ec=COLOR_MAP.get("GB") or COLOR_MAP.get("UK") or "#5C1F1F", ls="solid", lw=0.6),
        ROE_LABEL: dict(fc=roe_fc, ec=roe_fc, ls="solid", lw=0.6),
        ROW_LABEL: dict(fc="white", ec="black", ls=(0, (3, 2)), lw=0.6),
        OTHER_ORIGINS_LABEL: dict(fc="black", ec="black", ls="solid", lw=0.6),
    }
    for code in split_codes:
        label = split_labels[code]
        color = COLOR_MAP.get(code) or COLOR_MAP.get(label) or "#BBBBBB"
        styles[label] = dict(fc=color, ec=color, ls="solid", lw=0.6)
    return {"deltas": deltas, "regions": order, "gva": gva, "employment": employment, "styles": styles}


def _format_counterfactual_annotation(value: float) -> str:
    """Format plotted annotations, rounding thousands once to one decimal."""
    number = float(value)
    if abs(number) < 1000.0:
        return f"{number:.0f}"
    thousands = f"{number / 1000.0:.1f}"
    if thousands.endswith(".0"):
        thousands = thousands[:-2]
    return f"{thousands}k"


def _nudge_counterfactual_annotations(ax, artists, *, edge_padding_pt=5.0, gap_pt=0.5):
    """Keep plotted value labels inside an axis and separate actual collisions."""
    artists = list(artists)
    if not artists:
        return

    fig = ax.figure
    px_per_pt = fig.dpi / 72.0

    def move_x(artist, points):
        x, y = artist.get_position()
        display_x, display_y = ax.transData.transform((x, y))
        new_x = ax.transData.inverted().transform(
            (display_x + points * px_per_pt, display_y)
        )[0]
        artist.set_x(float(new_x))

    edge_padding = edge_padding_pt * px_per_pt
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in artists:
        bbox = artist.get_window_extent(renderer)
        if bbox.x0 < ax.bbox.x0:
            move_x(artist, (ax.bbox.x0 - bbox.x0) / px_per_pt)
        elif bbox.x1 > ax.bbox.x1 - edge_padding:
            move_x(artist, (ax.bbox.x1 - edge_padding - bbox.x1) / px_per_pt)

    gap = gap_pt * px_per_pt
    for _ in range(max(1, len(artists) * 2)):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ordered = sorted(artists, key=lambda item: item.get_window_extent(renderer).x0)
        collision = None
        for left, right in zip(ordered, ordered[1:]):
            left_box = left.get_window_extent(renderer)
            right_box = right.get_window_extent(renderer)
            if left_box.overlaps(right_box):
                collision = (left, right, left_box, right_box)
                break
        if collision is None:
            break

        left, right, left_box, right_box = collision
        needed = (left_box.x1 - right_box.x0 + gap) / px_per_pt
        right_room = (ax.bbox.x1 - edge_padding - right_box.x1) / px_per_pt
        left_room = (left_box.x0 - ax.bbox.x0 - edge_padding) / px_per_pt
        if right_room >= needed:
            move_x(right, needed)
        elif left_room >= needed:
            move_x(left, -needed)
        else:
            left_move = min(max(0.0, left_room), 0.5 * needed)
            right_move = min(max(0.0, right_room), needed - left_move)
            if left_move + right_move <= 0.0:
                break
            move_x(left, -left_move)
            move_x(right, right_move)


def _separate_consecutive_counterfactual_labels(ax, records):
    """Resolve rendered collisions between labels at consecutive 20-pp deltas."""
    from matplotlib.transforms import Bbox

    fig = ax.figure
    px_per_pt = fig.dpi / 72.0
    collision_padding_x = 1.0 * px_per_pt
    collision_padding_y = 1.5 * px_per_pt
    gap = 1.0 * px_per_pt

    def move_x(artist, points):
        x, y = artist.get_position()
        display_x, display_y = ax.transData.transform((x, y))
        artist.set_x(float(ax.transData.inverted().transform(
            (display_x + points * px_per_pt, display_y)
        )[0]))

    by_delta = {}
    for delta, artist in records:
        by_delta.setdefault(int(delta), []).append(artist)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = {artist: artist.get_window_extent(renderer).frozen() for _, artist in records}
    for delta in sorted(by_delta):
        next_delta = delta + 20
        if next_delta not in by_delta:
            continue
        for left in by_delta[delta]:
            for right in by_delta[next_delta]:
                left_box = boxes[left]
                right_box = boxes[right]
                vertical_overlap = (
                    min(left_box.y1, right_box.y1) + 2 * collision_padding_y
                    > max(left_box.y0, right_box.y0)
                )
                horizontal_gap = right_box.x0 - left_box.x1
                if not vertical_overlap or horizontal_gap >= 2 * collision_padding_x + gap:
                    continue

                needed = (2 * collision_padding_x + gap - horizontal_gap) / px_per_pt
                right_room = (ax.bbox.x1 - 5.0 * px_per_pt - right_box.x1) / px_per_pt
                left_room = (left_box.x0 - ax.bbox.x0) / px_per_pt
                if right_room >= needed:
                    move_x(right, needed)
                    boxes[right] = Bbox.from_extents(
                        right_box.x0 + needed * px_per_pt, right_box.y0,
                        right_box.x1 + needed * px_per_pt, right_box.y1,
                    )
                elif left_room >= needed:
                    move_x(left, -needed)
                    boxes[left] = Bbox.from_extents(
                        left_box.x0 - needed * px_per_pt, left_box.y0,
                        left_box.x1 - needed * px_per_pt, left_box.y1,
                    )
                else:
                    left_move = min(max(0.0, left_room), 0.5 * needed)
                    right_move = min(max(0.0, right_room), needed - left_move)
                    if left_move + right_move > 0.0:
                        move_x(left, -left_move)
                        move_x(right, right_move)
                        boxes[left] = Bbox.from_extents(
                            left_box.x0 - left_move * px_per_pt, left_box.y0,
                            left_box.x1 - left_move * px_per_pt, left_box.y1,
                        )
                        boxes[right] = Bbox.from_extents(
                            right_box.x0 + right_move * px_per_pt, right_box.y0,
                            right_box.x1 + right_move * px_per_pt, right_box.y1,
                        )


def _position_net_labels_without_dropping(ax, candidates, blockers):
    """Resolve rendered Net-label collisions horizontally, preserving their y offset."""
    fig = ax.figure
    px_per_pt = fig.dpi / 72.0
    edge = 1.0 * px_per_pt
    gap = 2.0 * px_per_pt
    max_horizontal_nudge_pt = 18.0
    horizontal_step_pt = 0.25

    def boxes_conflict(first, second):
        return (
            first.x0 < second.x1 + gap and first.x1 + gap > second.x0
            and first.y0 < second.y1 + gap and first.y1 + gap > second.y0
        )

    def shift_artist_x(artist, dx_pt=0.0):
        x, y = artist.get_position()
        artist.set_position((x + dx_pt, y))

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    blocker_boxes = [artist.get_window_extent(renderer).frozen() for artist in blockers]
    accepted_boxes = []
    original_offsets = {artist: artist.get_position() for _delta, _value, artist in candidates}
    # Place edge labels first so interior labels can yield horizontally when
    # a plateau brings their rendered boxes close to an endpoint.
    placement_order = sorted(candidates, key=lambda item: (-abs(item[0]), item[0]))
    for _delta, _value, artist in placement_order:
        # Start from the plotted label's original offset. Its vertical offset
        # remains fixed; only small horizontal moves are considered.
        original_x, original_y = original_offsets[artist]
        preferred_direction = -1.0 if _delta >= 80 else (1.0 if _delta <= -80 else 1.0)
        displacements = [0.0]
        for step in np.arange(horizontal_step_pt, max_horizontal_nudge_pt + 1e-9, horizontal_step_pt):
            displacements.extend((preferred_direction * float(step), -preferred_direction * float(step)))
        placed = False
        for dx_pt in displacements:
            artist.set_position((original_x + dx_pt, original_y))
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bbox = artist.get_window_extent(renderer)
            inside_axes = (
                bbox.x0 >= ax.bbox.x0 + edge
                and bbox.x1 <= ax.bbox.x1 - edge
                and bbox.y0 >= ax.bbox.y0 + edge
                and bbox.y1 <= ax.bbox.y1 - edge
            )
            if inside_axes and not any(
                boxes_conflict(bbox, other) for other in blocker_boxes + accepted_boxes
            ):
                accepted_boxes.append(bbox.frozen())
                placed = True
                break
        if not placed:
            artist.set_position((original_x, original_y))
            raise RuntimeError(
                f"Cannot place Net effect label at delta {_delta} without overlap "
                f"using a horizontal-only adjustment of at most {max_horizontal_nudge_pt:g} pt."
            )

    # Validate final rendered extents after all text has settled. This guards
    # against renderer-dependent overlaps and confirms the vertical offsets
    # were not altered by collision handling.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    final_boxes = []
    for _delta, _value, artist in candidates:
        if not np.isclose(artist.get_position()[1], original_offsets[artist][1], atol=1e-12):
            raise AssertionError(f"Net label at delta {_delta} changed its vertical offset.")
        bbox = artist.get_window_extent(renderer)
        if (
            bbox.x0 < ax.bbox.x0 + edge or bbox.x1 > ax.bbox.x1 - edge
            or bbox.y0 < ax.bbox.y0 + edge or bbox.y1 > ax.bbox.y1 - edge
        ):
            raise AssertionError(f"Net label at delta {_delta} is clipped by the axes.")
        final_boxes.append((_delta, bbox.frozen()))
    for index, (delta, bbox) in enumerate(final_boxes):
        for other_delta, other_bbox in final_boxes[index + 1:]:
            if boxes_conflict(bbox, other_bbox):
                raise AssertionError(
                    f"Rendered Net labels at deltas {delta} and {other_delta} overlap."
                )
        if any(boxes_conflict(bbox, blocker) for blocker in blocker_boxes):
            raise AssertionError(f"Net label at delta {delta} overlaps an area annotation.")


def _plot_counterfactual_regional_total(
    ax,
    data: dict,
    *,
    indicator: str,
    label_min: float,
    value_font: int = 8,
    ytick_font: int = 8,
):
    """Render one D.3-style total regional stack, retaining its labels and colors."""
    from matplotlib.ticker import FuncFormatter

    is_gva = str(indicator).upper() == "GVA"
    matrix = data["gva"] if is_gva else data["employment"]
    deltas = data["deltas"].astype(int)
    regions = data["regions"]
    styles = data["styles"]
    bottoms = np.zeros(len(deltas), dtype=float)
    stacks = {}
    for region in regions:
        values = matrix[region].to_numpy(dtype=float)
        style = styles[region]
        ax.fill_between(
            deltas.astype(float), bottoms, bottoms + values,
            alpha=0.85, color=style["fc"],
            edgecolor=style["ec"] if region == ROW_LABEL else "none",
            linewidth=style["lw"] if region == ROW_LABEL else 0.0,
            linestyle=style["ls"] if region == ROW_LABEL else "solid",
        )
        stacks[region] = (bottoms.copy(), values.copy())
        bottoms += values

    if is_gva:
        ylabel = "Global GVA [million EUR]"
    else:
        ylabel = "Global employment [FTE-years]"
    ax.set_ylabel(tex_bold(ylabel), fontsize=9)
    apply_house_style(ax, grid=True)
    ax.grid(axis="y", alpha=0.2, linestyle=":", linewidth=0.5)
    ax.set_ylim(bottom=0.0)

    mask = ((deltas - int(deltas[0])) % 20) == 0
    mask |= (deltas == int(deltas[0])) | (deltas == int(deltas[-1])) | (deltas == 0)
    edge_offset = 5.75 if len(deltas) > 1 else 0.0
    label_records = []
    for region in regions:
        bottom, values = stacks[region]
        for idx in np.flatnonzero(mask):
            value = float(values[idx])
            if value < float(label_min):
                continue
            x = float(deltas[idx])
            if idx == 0:
                x += edge_offset
            elif idx == len(deltas) - 1:
                x -= edge_offset
            label = _format_counterfactual_annotation(value)
            color = get_text_color(data["styles"][region]["fc"])
            artist = ax.text(
                x, float(bottom[idx] + 0.5 * value), label,
                ha="center", va="center", fontsize=value_font, color=color,
                alpha=0.92, fontweight="normal",
                clip_on=True,
            )
            label_records.append((int(deltas[idx]), artist))

    def fmt_axis(value, _position):
        value = float(value)
        if abs(value) >= 1000.0:
            scaled = value / 1000.0
            return f"{scaled:.0f}k" if abs(scaled) >= 10 else f"{scaled:.1f}".rstrip("0").rstrip(".") + "k"
        return f"{value:.0f}"
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_axis))
    ax.tick_params(axis="y", labelsize=ytick_font)
    ax.set_xlim(float(deltas[0]) - 2.0, float(deltas[-1]) + 2.0)
    return label_records


def _validate_counterfactual_component_totals(PANEL_D3: pd.DataFrame) -> None:
    """Ensure the saved Total rows equal all component rows at every region and delta."""
    frame = PANEL_D3.loc[PANEL_D3["scenario"].astype(str).eq("baseline")].copy()
    keys = ["delta_pp", "Region"]
    for col in ("abs_gva_total", "abs_jobs_total"):
        total = frame.loc[frame["Component"].astype(str).eq("Total")].groupby(keys)[col].sum()
        parts = frame.loc[~frame["Component"].astype(str).eq("Total")].groupby(keys)[col].sum()
        comparison = pd.concat([total.rename("total"), parts.rename("parts")], axis=1).fillna(0.0)
        if not np.allclose(comparison["total"], comparison["parts"], rtol=1e-10, atol=1e-7):
            raise AssertionError(f"PANEL_D3 Total does not reconcile with component {col}.")


def plot_and_save_eu27_counterfactual_figures(
    *,
    PANEL_D3: pd.DataFrame,
    PANEL_NET: pd.DataFrame,
    PANEL_EU_SIDE: pd.DataFrame,
    split_row=(),
    save: bool | None = None,
    close: bool | None = None,
):
    """Save global regional totals beside absolute EU27 effects and net levels."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    set_house_style()
    _validate_counterfactual_component_totals(PANEL_D3)
    regional = {
        scope: _prepare_counterfactual_region_panel(
            PANEL_D3, component=scope, split_row=split_row
        )
        for scope in ("Total", "Turbine", "BoP")
    }
    reference_deltas = set(regional["Total"]["deltas"].astype(int))
    ordered_deltas = np.array(sorted(reference_deltas), dtype=int)
    for scope, panel in regional.items():
        if set(panel["deltas"].astype(int)) != reference_deltas:
            raise ValueError(f"Regional delta coverage differs for {scope}.")

    def _scope_effect(frame, scope, *, name):
        selected = frame.loc[frame["Component"].astype(str).eq(scope)].copy()
        if selected.empty:
            raise ValueError(f"{name} has no rows for scope {scope!r}.")
        selected["delta_pp"] = pd.to_numeric(selected["delta_pp"], errors="raise").astype(int)
        grouped = selected.groupby("delta_pp").sum(numeric_only=True).sort_index()
        if set(grouped.index.astype(int)) != reference_deltas:
            raise ValueError(f"{name} delta coverage differs for {scope!r}.")
        return grouped.reindex(ordered_deltas)

    def _absolute_eu27_levels(panel, *, scope, indicator, name):
        columns = {
            "GVA": ("abs_gva_direct", "abs_gva_total", 1_000_000.0),
            "Jobs": ("abs_jobs_direct", "abs_jobs_total", 1.0),
        }
        direct_col, total_col, scale = columns[indicator]
        required = {"scenario", "delta_pp", "Component", "Region", direct_col, total_col}
        missing = sorted(required.difference(panel.columns))
        if missing:
            raise KeyError(f"{name} is missing columns required for absolute EU27 levels: {missing}.")
        frame = panel.loc[
            panel["scenario"].astype(str).eq("baseline")
            & panel["Region"].astype(str).str.strip().eq(EU_LABEL)
        ].copy()
        if scope == "Total":
            frame = frame.loc[~frame["Component"].astype(str).str.casefold().isin({"all", "total"})]
        else:
            frame = frame.loc[frame["Component"].astype(str).str.strip().eq(scope)]
        if frame.empty:
            raise ValueError(f"{name} has no EU27 rows for {scope!r}.")
        frame["delta_pp"] = pd.to_numeric(frame["delta_pp"], errors="raise").astype(int)
        values = frame.groupby("delta_pp")[[direct_col, total_col]].sum().sort_index() / scale
        if set(values.index.astype(int)) != reference_deltas:
            raise ValueError(f"{name} delta coverage differs for {scope!r}.")
        values = values.reindex(ordered_deltas).astype(float)
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"{name} contains non-finite EU27 levels for {scope!r}.")
        return values[direct_col], values[total_col]

    central_changes = {
        indicator: build_eu27_gross_net_delta_area_df(
            PANEL_NET, PANEL_EU_SIDE, indicator=indicator, value_mode="absolute"
        )
        for indicator in ("GVA", "Jobs")
    }
    # The 3.4 definitions use the EU-side scenario for gross Direct + Indirect
    # levels and the full net scenario for the final EU27 result.  At delta=0,
    # those scenarios coincide: upstream is therefore the observed residual
    # Net - gross, not a separately added baseline already embedded in Indirect.
    absolute = {}
    for scope in ("Total", "Turbine", "BoP"):
        absolute[scope] = {}
        for indicator, metric in (("GVA", "gva"), ("Jobs", "employment")):
            direct, gross_total = _absolute_eu27_levels(
                PANEL_EU_SIDE, scope=scope, indicator=indicator, name="PANEL_EU_SIDE"
            )
            _, net = _absolute_eu27_levels(
                PANEL_NET, scope=scope, indicator=indicator, name="PANEL_NET"
            )
            indirect = gross_total - direct
            upstream = net - gross_total
            left_net = regional[scope][metric][EU_LABEL].reindex(ordered_deltas).astype(float)
            if not np.allclose(net, left_net, rtol=2e-12, atol=1e-7):
                error = float(np.max(np.abs(net.to_numpy() - left_net.to_numpy())))
                raise AssertionError(
                    f"Absolute EU27 {indicator} net levels do not match the EU27 region "
                    f"in D.3 for {scope}; max absolute error={error:.6g}."
                )
            if not np.allclose(direct + indirect + upstream, net, rtol=1e-12, atol=1e-8):
                raise AssertionError(f"Absolute Direct + Indirect + Upstream does not equal Net for {scope}/{indicator}.")
            if not np.isclose(float(upstream.loc[0]), 0.0, rtol=0.0, atol=1e-8):
                raise AssertionError(
                    f"At delta=0 the observed EU-side and net baselines differ for {scope}/{indicator}; "
                    f"upstream residual={float(upstream.loc[0]):.12g}."
                )
            if (direct < -1e-8).any() or (indirect < -1e-8).any() or (net < -1e-8).any():
                raise ValueError(f"Absolute EU27 {indicator} Direct, Indirect, or Net levels are negative for {scope}.")

            # Confirm that the absolute-level transformation preserves the
            # original 3.4 change definitions, including the upstream residual.
            change = _scope_effect(
                central_changes[indicator], scope, name=f"3.4 {indicator} decomposition"
            )
            change_scale = 1000.0 if indicator == "Jobs" else 1.0
            for level, field in (
                (direct, "direct_gross"),
                (indirect, "gross_indirect"),
                (upstream, "upstream_trade_adjustment"),
                (net, "net_effect"),
            ):
                if not np.allclose(
                    level - float(level.loc[0]),
                    change[field] * change_scale,
                    rtol=2e-10, atol=1e-7,
                ):
                    raise AssertionError(
                        f"Absolute {field} levels do not reproduce the existing 3.4 changes "
                        f"for {scope}/{indicator}."
                    )

            absolute[scope][indicator] = {
                "direct": direct,
                "indirect": indirect,
                "gross_total": gross_total,
                "upstream": upstream,
                "net": net,
            }

    eu_fc = COLOR_MAP.get("EU27") or COLOR_MAP.get(EU_LABEL) or "#4C78A8"
    net_fc = COLOR_MAP.get("GB") or COLOR_MAP.get("UK") or "#5C1F1F"
    def _format_axis(value, _position):
        value = float(value)
        if abs(value) >= 1000.0:
            scaled = value / 1000.0
            return f"{scaled:.0f}k" if abs(scaled) >= 10 else f"{scaled:.1f}".rstrip("0").rstrip(".") + "k"
        return f"{value:.0f}"

    region_handles = []
    for region in regional["Total"]["regions"]:
        style = regional["Total"]["styles"][region]
        label = CODE_TO_NAME.get("GB", "UK") if region == UK_LABEL else region
        region_handles.append(mpatches.Patch(
            facecolor=style["fc"], edgecolor=style["ec"], linewidth=style["lw"],
            linestyle=style["ls"], label=tex_safe(label),
        ))
    direct_fc = eu_fc
    indirect_fc = lighten_color(eu_fc, 0.55)
    indirect_edge_fc = darken_color(lighten_color(eu_fc, 0.35), 0.25)
    trade_fc = "#F6D6DA"
    effect_area_handles = [
        Patch(facecolor=direct_fc, edgecolor=darken_color(direct_fc, 0.25), label="Direct"),
        Patch(facecolor=indirect_fc, edgecolor=indirect_edge_fc, label="Indirect"),
        Patch(facecolor=trade_fc, edgecolor=net_fc, hatch="///", label="Upstream trade effects"),
    ]

    outputs = {}
    for scope in ("Total", "Turbine", "BoP"):
        fig = plt.figure(figsize=(662.78 / 72.0, 373.0 / 72.0), dpi=150)
        grid = fig.add_gridspec(
            2, 2, left=0.0850, right=0.9693, bottom=0.15, top=0.9807,
            width_ratios=(1.0, 1.0), hspace=0.10, wspace=0.15,
        )
        axes = np.asarray([
            [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])],
            [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])],
        ])
        right_panel_payload = {}
        net_label_candidates = {}
        net_legend_handle = None
        area_label_min = {
            "GVA": {"Total": 250.0, "Turbine": 100.0, "BoP": 75.0},
            "Jobs": {"Total": 5500.0, "Turbine": 1500.0, "BoP": 1200.0},
        }
        left_label_series = {}
        left_label_series[0] = _plot_counterfactual_regional_total(
            axes[0, 0], regional[scope], indicator="GVA",
            label_min=area_label_min["GVA"][scope], value_font=8, ytick_font=8,
        )
        left_label_series[1] = _plot_counterfactual_regional_total(
            axes[1, 0], regional[scope], indicator="Jobs",
            label_min=area_label_min["Jobs"][scope], value_font=8, ytick_font=8,
        )

        for indicator, row, metric, ylabel in (
            ("GVA", 0, "gva", "EU27 GVA [million EUR]"),
            ("Jobs", 1, "employment", "EU27 employment [FTE-years]"),
        ):
            ax = axes[row, 1]
            rows_effect = absolute[scope][indicator]
            direct = rows_effect["direct"].reindex(ordered_deltas).to_numpy(dtype=float)
            gross_total = rows_effect["gross_total"].reindex(ordered_deltas).to_numpy(dtype=float)
            net = rows_effect["net"].reindex(ordered_deltas).to_numpy(dtype=float)
            net_annotation_deltas = np.arange(-100, 101, 20, dtype=int)
            net_annotation_idx = np.flatnonzero(np.isin(ordered_deltas, net_annotation_deltas))
            if len(ordered_deltas) != 41 or not np.array_equal(
                ordered_deltas[net_annotation_idx], net_annotation_deltas
            ):
                raise ValueError("Net labels require 41 deltas and all 11 configured 20-pp points.")
            ax.fill_between(ordered_deltas, 0.0, direct, color=direct_fc,
                            edgecolor=darken_color(direct_fc, 0.25), linewidth=0.45, zorder=2)
            ax.fill_between(ordered_deltas, direct, gross_total, color=indirect_fc,
                            edgecolor=indirect_edge_fc, linewidth=0.45, zorder=2)
            ax.fill_between(ordered_deltas, gross_total, net, facecolor=trade_fc,
                            edgecolor=net_fc, linewidth=0.45, hatch="///", zorder=3)
            ax.plot(
                ordered_deltas, net, color=net_fc, linewidth=1.15,
                zorder=6, label="_nolegend_",
            )
            net_legend_handle = Line2D(
                [], [], color=net_fc, linewidth=1.15, marker="D", markersize=3.5,
                markerfacecolor="white", markeredgecolor=net_fc, markeredgewidth=1.2,
                label="Net effect",
            )
            net_label_candidates[row] = []
            for idx in net_annotation_idx:
                value = float(net[idx])
                label = _format_counterfactual_annotation(value)
                if idx == 0:
                    text_offset, horizontal_alignment = (4, 4), "left"
                elif idx == len(ordered_deltas) - 1:
                    text_offset, horizontal_alignment = (0, 4), "right"
                else:
                    text_offset, horizontal_alignment = (0, 4), "center"
                artist = ax.annotate(
                    label, (ordered_deltas[idx], value), xytext=text_offset,
                    textcoords="offset points", ha=horizontal_alignment, va="bottom",
                    fontsize=8, color=net_fc, clip_on=True,
                )
                net_label_candidates[row].append(
                    (int(ordered_deltas[idx]), value, artist)
                )
            right_panel_payload[row] = (ax, direct, gross_total, net)
            ax.set_ylabel(tex_bold(ylabel), fontsize=9)
            ax.yaxis.set_major_formatter(FuncFormatter(_format_axis))
            apply_house_style(ax, grid=True)
            ax.grid(axis="y", alpha=0.2, linestyle=":", linewidth=0.45)
            ax.tick_params(axis="both", labelsize=8)
            ax.set_xlim(-102.0, 102.0)
            ax.set_xticks(np.arange(-100, 101, 20, dtype=int))
            if row == 0:
                ax.tick_params(axis="x", labelbottom=False)

        axes[1, 0].set_ylabel(tex_bold("Global employment [FTE-years]"), fontsize=9)
        axes[0, 0].set_ylabel(tex_bold("Global GVA [million EUR]"), fontsize=9)
        for ax in axes[0, :]:
            ax.tick_params(axis="x", labelbottom=False)
        for ax in axes[1, :]:
            ax.set_xlabel(r"\textbf{\boldmath$\Delta$ EU27 sourcing (pp)}", fontsize=8)
        for ax in axes.flat:
            ax.set_xticks(np.arange(-100, 101, 20, dtype=int))

        # Share the global D.3 scale across both columns for each indicator.
        for row, indicator, metric in ((0, "GVA", "gva"), (1, "Jobs", "employment")):
            global_total = regional[scope][metric].sum(axis=1).reindex(ordered_deltas).astype(float)
            eu_paths = absolute[scope][indicator]["net"].reindex(ordered_deltas).to_numpy(dtype=float)
            ymax = max(float(global_total.max()), float(np.nanmax(eu_paths))) * 1.025
            if not np.isfinite(ymax) or ymax <= 0.0:
                raise ValueError(f"Invalid shared global/EU27 {indicator} scale for {scope}.")
            ticks = MaxNLocator(nbins=5).tick_values(0.0, ymax)
            ticks = ticks[(ticks >= 0.0) & (ticks <= ymax)]
            for ax in axes[row, :]:
                ax.set_ylim(0.0, ymax)
                ax.set_yticks(ticks)
                ax.tick_params(axis="x", labelsize=8)
                ax.tick_params(axis="y", labelsize=8, labelleft=True)

        # Use the same 20-pp candidates and absolute area minima as the D.3
        # stacks. The D.3 rule has no additional rendered-height cutoff.
        area_label_deltas = ordered_deltas[
            ((ordered_deltas - int(ordered_deltas[0])) % 20 == 0)
            | (ordered_deltas == int(ordered_deltas[0]))
            | (ordered_deltas == int(ordered_deltas[-1]))
            | (ordered_deltas == 0)
        ]
        area_label_candidates = []
        area_label_delta_set = set(area_label_deltas.astype(int))
        for indicator, row, _metric, _ylabel in (
            ("GVA", 0, "gva", "EU27 GVA [million EUR]"),
            ("Jobs", 1, "employment", "EU27 employment [FTE-years]"),
        ):
            ax, direct, gross_total, _net = right_panel_payload[row]
            minimum = area_label_min[indicator][scope]
            for idx, delta in enumerate(ordered_deltas):
                if int(delta) not in area_label_delta_set:
                    continue
                segments = (
                    ("Direct", 0.0, float(direct[idx]), direct_fc),
                    ("Indirect", float(direct[idx]), float(gross_total[idx]), indirect_fc),
                )
                for segment_name, lower, upper, color in segments:
                    height = upper - lower
                    if height < minimum:
                        continue
                    text = _format_counterfactual_annotation(height)
                    x_value = float(delta)
                    if idx == 0:
                        x_value += 5.75
                    elif idx == len(ordered_deltas) - 1:
                        x_value -= 5.75
                    artist = ax.text(
                        x_value, 0.5 * (lower + upper), text,
                        ha="center", va="center", fontsize=8,
                        color=get_text_color(color), alpha=0.92,
                        clip_on=True, zorder=7,
                    )
                    area_label_candidates.append({
                        "artist": artist,
                        "axes": ax,
                    })

        for row in (0, 1):
            ax = axes[row, 1]
            _nudge_counterfactual_annotations(
                ax,
                [item["artist"] for item in area_label_candidates if item["axes"] is ax],
            )
        for row in (0, 1):
            _nudge_counterfactual_annotations(axes[row, 0], axes[row, 0].texts)
            _separate_consecutive_counterfactual_labels(
                axes[row, 0], left_label_series[row]
            )

        # Keep all 11 Net label-marker pairs; resolve overlap by repositioning labels.
        for row in (0, 1):
            ax = axes[row, 1]
            area_artists = [
                item["artist"]
                for item in area_label_candidates if item["axes"] is ax
            ]
            _position_net_labels_without_dropping(
                ax, net_label_candidates[row], area_artists
            )
            paired_deltas = [item[0] for item in net_label_candidates[row]]
            if not np.array_equal(paired_deltas, net_annotation_deltas.tolist()):
                raise AssertionError("Net labels no longer match the configured 11 deltas.")
            if len(net_label_candidates[row]) != 11:
                raise AssertionError("Each Net panel must retain exactly 11 label-marker pairs.")
            ax.plot(
                [item[0] for item in net_label_candidates[row]],
                [item[1] for item in net_label_candidates[row]],
                linestyle="None", marker="D", markersize=3.5,
                markerfacecolor="white", markeredgecolor=net_fc,
                markeredgewidth=1.2, zorder=6, label="_nolegend_",
            )

        # Center the geographic legend across the left plotting-column width;
        # keep the effects legend at its natural width in the right column.
        left_plot = axes[0, 0].get_position()
        right_plot = axes[0, 1].get_position()
        left_x_center = 0.5 * (left_plot.x0 + left_plot.x1)
        right_x_center = 0.5 * (right_plot.x0 + right_plot.x1)
        footer_y = 0.018
        region_legend = fig.legend(
            handles=region_handles, loc="lower left",
            bbox_to_anchor=(left_plot.x0, footer_y, left_plot.width, 0.0),
            bbox_transform=fig.transFigure, ncol=6, mode="expand",
            frameon=False, prop={"family": "serif", "size": 7},
            handlelength=0.8, handletextpad=0.2, columnspacing=0.25,
            labelspacing=0.12, borderaxespad=0.0,
        )
        if net_legend_handle is None:
            raise RuntimeError("Net effect line handle was not created.")
        effect_legend = fig.legend(
            handles=effect_area_handles + [net_legend_handle], loc="lower center",
            labels=["Direct", "Indirect", "Upstream trade effects", "Net effect"],
            bbox_to_anchor=(right_x_center, footer_y), bbox_transform=fig.transFigure,
            ncol=4, frameon=False, prop={"family": "serif", "size": 7},
            handlelength=0.9, handletextpad=0.25, columnspacing=0.4,
            labelspacing=0.12, borderaxespad=0.0,
        )

        # Center the one-row effects legend on the measured center of the
        # two-row geographic legend.  The legends retain their natural widths.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        region_bbox = region_legend.get_window_extent(renderer)
        effect_bbox = effect_legend.get_window_extent(renderer)
        vertical_shift = 0.5 * (
            region_bbox.y0 + region_bbox.y1 - effect_bbox.y0 - effect_bbox.y1
        )
        effect_anchor = fig.transFigure.transform((right_x_center, footer_y))
        effect_anchor = fig.transFigure.inverted().transform(
            (effect_anchor[0], effect_anchor[1] + vertical_shift)
        )
        effect_legend.set_bbox_to_anchor(effect_anchor, transform=fig.transFigure)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        region_bbox = region_legend.get_window_extent(renderer)
        effect_bbox = effect_legend.get_window_extent(renderer)
        if abs(
            0.5 * (region_bbox.y0 + region_bbox.y1 - effect_bbox.y0 - effect_bbox.y1)
        ) > 0.05:
            raise AssertionError("The Figure 3.4 legend centers are not vertically aligned.")
        left_plot_width = left_plot.width * fig.get_figwidth() * fig.dpi
        right_plot_width = right_plot.width * fig.get_figwidth() * fig.dpi
        left_center_px = fig.transFigure.transform((left_x_center, footer_y))[0]
        right_center_px = fig.transFigure.transform((right_x_center, footer_y))[0]
        if abs(0.5 * (region_bbox.x0 + region_bbox.x1) - left_center_px) > 0.05:
            raise AssertionError("The geographic legend is not centered under the left plots.")
        if abs(0.5 * (effect_bbox.x0 + effect_bbox.x1) - right_center_px) > 0.05:
            raise AssertionError("The effects legend is not centered under the right plots.")
        region_font_sizes = {text.get_fontsize() for text in region_legend.get_texts()}
        effect_font_sizes = {text.get_fontsize() for text in effect_legend.get_texts()}
        if region_font_sizes != effect_font_sizes:
            raise AssertionError("The Figure 3.4 legend font sizes must match.")
        if region_bbox.width < 0.98 * left_plot_width:
            raise AssertionError("The geographic legend does not span the left plotting column.")
        if region_bbox.width > left_plot_width + 0.5:
            raise ValueError("The geographic legend exceeds the left plotting-column width.")
        if effect_bbox.width > right_plot_width + 0.5:
            raise ValueError("The effects legend exceeds the right plotting-column width.")

        try:
            fig.set_layout_engine("none")
        except Exception:
            pass
        outputs[scope] = save_and_close(
            fig=fig, filename=f"3.4_EU27_counterfactual_{scope}.pdf", save=save, close=close,
            exact_size=True,
        )
    return outputs














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
    include_scenario_row: bool = False,
    parenthesize_trade_adjustment: bool = False,
    indent_effect_rows: bool = False,
    labels: dict | None = None,
    procurement_savings: pd.DataFrame | None = None,
    markup_net_effects: pd.DataFrame | None = None,
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
                include_scenario_row=include_scenario_row,
                parenthesize_trade_adjustment=parenthesize_trade_adjustment,
                indent_effect_rows=indent_effect_rows,
                labels=labels,
                procurement_savings=procurement_savings,
                markup_net_effects=markup_net_effects,
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
    gross_pct_label = labels.get("gross_pct", "Var %")
    trade_label = labels.get("trade_losses", "Up. trade effects (3)")
    net_label = labels.get("net", "Net effect (1+2+3)")
    net_pct_label = labels.get("net_pct", "Var %")
    cost_base_label = labels.get("cost_base", "Baseline cost")
    cost_variation_label = labels.get("cost_variation", "Cost variation")
    cost_scenario_label = labels.get("cost_scenario", "Scenario cost")
    cost_pct_label = labels.get("cost_pct", "Var %")
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

    def _markup_mu(value) -> float:
        text = str(value).strip().casefold()
        aliases = {"low": 0.1, "central": 0.3, "high": 0.5}
        if text in aliases:
            return aliases[text]
        for prefix in ("mu=", "mu_", "mu "):
            if text.startswith(prefix):
                text = text[len(prefix):]
                break
        try:
            mu_value = float(text)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid markup case {value!r}; expected a numeric mu.") from exc
        if not np.isfinite(mu_value) or not (0.0 <= mu_value < 1.0):
            raise ValueError(f"Invalid markup case {value!r}; expected 0 <= mu < 1.")
        return float(mu_value)

    def _markup_key(value) -> str:
        return f"mu={_markup_mu(value):g}"

    def _markup_label(mu_value: float) -> str:
        return rf"$\mu={_format_resilience_markup_value(mu_value)}$"

    central_markup_key = _markup_key(0.3)
    has_markup_sensitivity = markup_net_effects is not None
    sensitivity = None
    sensitivity_cases = []
    if has_markup_sensitivity:
        required_sensitivity_columns = {
            "markup", "delta_pp", "Component", "indicator", "net_effect",
        }
        missing_sensitivity_columns = sorted(
            required_sensitivity_columns.difference(markup_net_effects.columns)
        )
        if missing_sensitivity_columns:
            raise KeyError(
                "markup_net_effects is missing columns: "
                f"{missing_sensitivity_columns}."
            )
        sensitivity = markup_net_effects.copy()
        sensitivity["markup"] = sensitivity["markup"].map(_markup_key)
        sensitivity["Component"] = sensitivity["Component"].astype(str).str.strip()
        sensitivity["indicator"] = sensitivity["indicator"].astype(str).str.strip()
        sensitivity["delta_pp"] = pd.to_numeric(sensitivity["delta_pp"], errors="raise").astype(int)
        sensitivity["net_effect"] = pd.to_numeric(sensitivity["net_effect"], errors="raise")
        available_markup_cases = {
            key: _markup_mu(key.removeprefix("mu="))
            for key in sensitivity["markup"].dropna().unique()
        }
        sensitivity_cases = [
            (key, _markup_label(mu_value), mu_value)
            for key, mu_value in sorted(available_markup_cases.items(), key=lambda item: item[1])
            if not np.isclose(mu_value, 0.3, atol=1e-12, rtol=0.0)
        ]
    sensitivity_row_labels = [label for _, label, _ in sensitivity_cases]

    net_pct_display_label = (
        f"{net_pct_label} ($\\mu=0.3$)" if has_markup_sensitivity else net_pct_label
    )
    cost_pct_display_label = (
        f"{cost_pct_label} ($\\mu=0.3$)" if has_markup_sensitivity else cost_pct_label
    )

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
        if include_scenario_row:
            row_names.append("EU27 scen.")
        if include_pct_change:
            row_names.append(net_pct_display_label)
            if has_markup_sensitivity:
                row_names.extend(sensitivity_row_labels)
            elif not has_markup_sensitivity:
                row_names.insert(-1, gross_pct_label)

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
            if include_scenario_row:
                rows[(impact, "EU27 scen.")][col] = (net_base_total + net_effect) / scale
            if include_pct_change:
                rows[(impact, net_pct_display_label)][col] = net_pct
            if has_markup_sensitivity:
                sensitivity_indicator = "GVA" if impact == "GVA" else "Jobs"
                sensitivity_pct_values = []
                for markup_key, sensitivity_label, _ in sensitivity_cases:
                    selected = sensitivity[
                        sensitivity["markup"].eq(markup_key)
                        & sensitivity["delta_pp"].eq(int(delta_resil))
                        & sensitivity["Component"].eq(comp)
                        & sensitivity["indicator"].eq(sensitivity_indicator)
                    ]
                    if len(selected) != 1:
                        raise ValueError(
                            "Expected exactly one markup net-effect row for "
                            f"impact={impact!r}, markup={markup_key!r}, "
                            f"component={comp!r}, delta={delta_resil}; found {len(selected)}."
                        )
                    markup_net_effect = float(selected.iloc[0]["net_effect"])
                    baseline_in_output_units = net_base_total / scale
                    if abs(baseline_in_output_units) <= 1e-12:
                        sensitivity_pct_values.append(np.nan)
                    else:
                        sensitivity_pct_values.append(
                            100.0 * markup_net_effect / baseline_in_output_units
                        )
                if include_pct_change:
                    for (_, sensitivity_label, _), sensitivity_value in zip(
                        sensitivity_cases, sensitivity_pct_values
                    ):
                        rows[(impact, sensitivity_label)][col] = sensitivity_value

    if procurement_savings is not None:
        required_cost_columns = {
            "markup",
            "delta_pp",
            "Component",
            "baseline_global_gva_total_eur",
            "procurement_savings_million_eur",
        }
        missing_cost_columns = sorted(required_cost_columns.difference(procurement_savings.columns))
        if missing_cost_columns:
            raise KeyError(
                "procurement_savings is missing columns: "
                f"{missing_cost_columns}."
            )

        cost_df = procurement_savings.copy()
        cost_baselines = procurement_savings.attrs.get("procurement_cost_baselines_eur")
        if not isinstance(cost_baselines, dict):
            raise ValueError(
                "procurement_savings lacks fixed-reference baseline costs by markup; "
                "rebuild it with build_eu27_procurement_savings_df()."
            )
        cost_df["markup"] = cost_df["markup"].map(_markup_key)
        cost_df["Component"] = cost_df["Component"].astype(str).str.strip()
        cost_df["delta_pp"] = pd.to_numeric(cost_df["delta_pp"], errors="raise").astype(int)
        for column in ("baseline_global_gva_total_eur", "procurement_savings_million_eur"):
            cost_df[column] = pd.to_numeric(cost_df[column], errors="raise")

        cost_baselines = {
            _markup_key(markup): values
            for markup, values in cost_baselines.items()
        }

        def _cost_row(component: str, delta: int, markup: str = central_markup_key) -> pd.Series:
            markup_key = _markup_key(markup)
            selected = cost_df[
                cost_df["Component"].eq(str(component))
                & cost_df["delta_pp"].eq(int(delta))
                & cost_df["markup"].eq(markup_key)
            ]
            if selected.empty:
                raise KeyError(
                    "No procurement-savings row found for "
                    f"markup={markup_key!r}, component={component!r}, delta={delta}."
                )
            if len(selected) != 1:
                raise ValueError(
                    "Expected exactly one procurement-savings row for "
                    f"markup={markup_key!r}, component={component!r}, delta={delta}; found {len(selected)}."
                )
            return selected.iloc[0]

        cost_row_names = [
            cost_base_label,
            cost_variation_label,
            cost_scenario_label,
            cost_pct_display_label,
        ]
        if has_markup_sensitivity:
            cost_row_names += sensitivity_row_labels
        for row_name in cost_row_names:
            rows[("Procurement cost", row_name)] = {}

        for comp in component_order:
            base_row = _cost_row(comp, delta_base, central_markup_key)
            scenario_row = _cost_row(comp, delta_resil, central_markup_key)
            try:
                baseline_eur = float(cost_baselines[central_markup_key][comp])
            except (KeyError, TypeError, ValueError) as exc:
                raise KeyError(
                    f"procurement_savings lacks the central baseline cost for component {comp!r}."
                ) from exc
            baseline_million = baseline_eur / 1e6
            scenario_baseline_million = baseline_million
            if not np.isclose(baseline_million, scenario_baseline_million, atol=1e-9, rtol=1e-12):
                raise ValueError(
                    "Central procurement baseline differs between delta_base and "
                    f"delta={delta_resil} for component={comp!r}."
                )
            if abs(baseline_million) <= 1e-12:
                raise ValueError(f"Central procurement baseline is zero for component={comp!r}.")

            cost_variation = -float(scenario_row["procurement_savings_million_eur"])
            scenario_cost = baseline_million + cost_variation
            cost_pct = 100.0 * cost_variation / baseline_million
            col = _col_label(comp)
            rows[("Procurement cost", cost_base_label)][col] = baseline_million
            rows[("Procurement cost", cost_variation_label)][col] = cost_variation
            rows[("Procurement cost", cost_scenario_label)][col] = scenario_cost
            rows[("Procurement cost", cost_pct_display_label)][col] = cost_pct
            if has_markup_sensitivity:
                for markup_key, sensitivity_label, _ in sensitivity_cases:
                    markup_base_row = _cost_row(comp, delta_base, markup_key)
                    markup_scenario_row = _cost_row(comp, delta_resil, markup_key)
                    try:
                        markup_baseline_million = float(cost_baselines[markup_key][comp]) / 1e6
                    except (KeyError, TypeError, ValueError) as exc:
                        raise KeyError(
                            f"procurement_savings lacks the {markup_key} baseline cost for component {comp!r}."
                        ) from exc
                    if abs(markup_baseline_million) <= 1e-12:
                        markup_cost_pct = np.nan
                    else:
                        markup_variation = -float(
                            markup_scenario_row["procurement_savings_million_eur"]
                        )
                        markup_cost_pct = (
                            100.0 * markup_variation / markup_baseline_million
                        )
                    rows[("Procurement cost", sensitivity_label)][col] = markup_cost_pct

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
def build_eu27_net_delta_effect_df(
    PANEL_NET: pd.DataFrame,
    *,
    indicator: str = "GVA",
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
    """Build absolute net effects directly from one net scenario panel.

    This is intentionally independent of the EU27-side/gross panel. It is used
    for markup sensitivity, where only net MRIO outcomes are required.
    """
    ind_u = str(indicator).upper().strip()
    if ind_u == "GVA":
        total_col = gva_total_col
        abs_scale = 1_000_000.0
        indicator_label = "GVA"
    elif ind_u == "JOBS":
        total_col = jobs_total_col
        abs_scale = 1_000.0
        indicator_label = "Jobs"
    else:
        raise ValueError("indicator must be 'GVA' or 'Jobs'.")

    needed = [delta_col, region_col, comp_col, output_direct_col, total_col]
    df = PANEL_NET.copy()
    if scenario is not None and scenario_col in df.columns:
        df = df[df[scenario_col].astype(str) == str(scenario)].copy()
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in net panel: {missing}")
    df[delta_col] = pd.to_numeric(df[delta_col], errors="coerce").fillna(0).astype(int)
    df[region_col] = df[region_col].astype(str).str.strip()
    df[comp_col] = df[comp_col].astype(str).str.strip()
    for col in (output_direct_col, total_col):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    comp_cf = df[comp_col].str.casefold()
    df = df[~comp_cf.isin({"all", "total"})].copy()

    available_deltas = set(df[delta_col].unique())
    if int(delta_base) not in available_deltas:
        raise KeyError(f"net panel is missing delta_base={delta_base}.")

    totals = (
        df.groupby([delta_col, region_col], as_index=False)[[output_direct_col, total_col]]
        .sum()
    )
    totals[comp_col] = "Total"
    df = pd.concat([df[needed], totals[needed]], ignore_index=True)

    eu = (
        df[df[region_col] == eu_label]
        .groupby([delta_col, comp_col], as_index=False)[[output_direct_col, total_col]]
        .sum()
    )
    base = (
        eu[eu[delta_col] == int(delta_base)][[comp_col, output_direct_col, total_col]]
        .rename(columns={
            output_direct_col: "_base_output",
            total_col: "_base_total",
        })
    )
    out = eu.merge(base, on=comp_col, how="left")
    if out[["_base_output", "_base_total"]].isna().any().any():
        missing_components = sorted(
            out.loc[out["_base_total"].isna(), comp_col].astype(str).unique().tolist()
        )
        raise KeyError(
            f"Net panel is missing baseline values for components: {missing_components}"
        )
    out["net_effect"] = (out[total_col] - out["_base_total"]) / float(abs_scale)
    out["indicator"] = indicator_label
    out = out.rename(columns={delta_col: "delta_pp", comp_col: "Component"})
    return out[["delta_pp", "Component", "net_effect", "indicator"]].copy()


def _normalize_resilience_markup_values(values: Iterable[float]) -> tuple[float, ...]:
    """Validate, round, sort, and deduplicate accounting markup values."""
    normalized = []
    for value in values:
        mu = float(value)
        if not np.isfinite(mu) or not 0.0 <= mu < 1.0:
            raise ValueError("Accounting markup values must satisfy 0 <= mu < 1.")
        mu = round(mu, 12)
        if not any(np.isclose(mu, existing, atol=1e-12, rtol=0.0) for existing in normalized):
            normalized.append(mu)
    return tuple(sorted(normalized))


def _format_resilience_markup_value(value: float) -> str:
    """Return one stable decimal representation for labels and cache names."""
    text = f"{float(value):.12f}".rstrip("0").rstrip(".")
    return text or "0"


def _resilience_markup_key(value: float) -> str:
    return f"mu={_format_resilience_markup_value(value)}"


def _resilience_markup_cache_token(value: float) -> str:
    aliases = {0.0: "mu_0_0", 0.2: "mu_0_2", 0.4: "mu_0_4"}
    for alias_value, alias in aliases.items():
        if np.isclose(float(value), alias_value, atol=1e-12, rtol=0.0):
            return alias
    return f"mu_{_format_resilience_markup_value(value).replace('.', '_')}"


def _resilience_accounting_note(markup_values: Iterable[float]) -> str:
    """Build the accounting note without making claims about an arbitrary grid."""
    values = _normalize_resilience_markup_values((*markup_values, REFERENCE_MARKUP))
    default_grid = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    if len(values) == len(default_grid) and all(
        np.isclose(a, b, atol=1e-12, rtol=0.0) for a, b in zip(values, default_grid)
    ):
        sensitivity_text = (
            r"Percentage variations are reported for $\mu=0$--$0.5$ in increments of 0.1, "
            r"relative to the unchanged baseline."
        )
    else:
        labels = ", ".join(rf"$\mu={_format_resilience_markup_value(value)}$" for value in values)
        sensitivity_text = (
            f"Percentage variations are reported for {labels}, relative to the unchanged baseline."
        )
    return (
        r"\textit{Note:} Total baseline and scenario levels cover the full OWF; only Turbine and BoP "
        r"procurement are reallocated. GVA and employment refer to EU27 impacts, while procurement costs "
        r"cover total project expenditure. GVA and costs are in EUR million; employment is in thousand FTE-years. "
        r"Net equals gross plus upstream trade effects. Absolute values correspond to $\mu=0.3$. "
        + sensitivity_text
        + r" Negative values are shown in parentheses."
    )


def prepare_resilience_markup_sensitivities(
    *,
    central_scenarios: tuple[dict[int, "ScenarioDelta"], dict[int, dict]],
    accounting_markups: Iterable[float],
    cache_base_dir,
    figure_deltas: Iterable[int],
    scenario_recompute: bool,
    accounting_recompute: bool,
    EUROPE,
    EU27,
    LOW_INCOME,
    df_alloc: pd.DataFrame,
    flows_by_component_hier_all,
    ROOT: str,
    L1_MRIO,
    VA_MRIO,
    LB_MRIO=None,
    W_MRIO=None,
    mrio=None,
    core=None,
    annualize: bool = False,
    lifetime_years: int = 25,
    receiver_policy: str = "baseline_mix",
    results_by_origin_lvl: dict | None = None,
    results_by_origin_sub_lvl: dict | None = None,
    capacity_lvl: int | None = None,
    connected_country: dict | None = None,
    component_filter_turbine: set[str] | list[str] | None = None,
    component_filter_farm: set[str] | list[str] | None = None,
    preserve_total: bool = False,
    exclude_deployments: Iterable[str] = (),
    figure_markup_low: float = 0.10,
    figure_markup_central: float = 0.30,
    figure_markup_high: float = 0.50,
    delta_base: int = 0,
    delta_out: int = -100,
    delta_resil: int = 100,
    accounting_component_level: int = 2,
    preloaded_scenarios: dict[float, tuple[dict[int, "ScenarioDelta"], dict[int, dict]]] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Prepare figure and accounting markup sensitivity results from v10 caches.

    Figure markup cases use the notebook's normal recomputation policy and the
    full figure delta grid. Accounting-only cases use only ``delta_out`` and
    ``delta_resil`` and an independent recomputation policy. Existing caches
    with broader delta coverage are copied to a temporary staging directory if
    an endpoint must be recomputed, so a subset request can never replace a
    broader production cache.
    """
    central_delta, central_result = central_scenarios
    central_mu = float(figure_markup_central)
    if not np.isclose(central_mu, REFERENCE_MARKUP, atol=1e-12, rtol=0.0):
        raise ValueError("The resilience accounting central markup must be mu=0.30.")
    central_mu = REFERENCE_MARKUP
    selected_values = _normalize_resilience_markup_values(accounting_markups)
    accounting_values = _normalize_resilience_markup_values((*selected_values, central_mu))
    figure_deltas = tuple(dict.fromkeys(int(delta) for delta in figure_deltas))
    accounting_deltas = tuple(dict.fromkeys((int(delta_out), int(delta_resil))))
    if len(accounting_deltas) != 2 or accounting_deltas[0] == accounting_deltas[1]:
        raise ValueError("Accounting endpoint deltas must contain distinct out and resilience values.")
    if int(delta_base) in accounting_deltas:
        raise ValueError("delta_base must be separate from the accounting endpoint deltas.")
    if int(delta_base) not in central_delta or int(delta_base) not in central_result:
        raise KeyError(f"Central scenarios must contain the accounting baseline delta={delta_base}.")

    preloaded = {}
    for value, scenarios in (preloaded_scenarios or {}).items():
        normalized = _normalize_resilience_markup_values((value,))
        if len(normalized) != 1:
            raise ValueError(f"Invalid preloaded markup value: {value!r}.")
        preloaded[normalized[0]] = scenarios
    preloaded[central_mu] = (central_delta, central_result)

    europe_arg = EUROPE if EUROPE is not None else EU27
    common_kwargs = dict(
        flows_by_component_hier_all=flows_by_component_hier_all,
        df_alloc=df_alloc,
        ROOT=ROOT,
        EUROPE=europe_arg,
        EU27=EU27,
        LOW_INCOME=LOW_INCOME,
        L1_MRIO=L1_MRIO,
        VA_MRIO=VA_MRIO,
        LB_MRIO=LB_MRIO,
        W_MRIO=W_MRIO,
        receiver_policy=receiver_policy,
        results_by_origin_lvl=results_by_origin_lvl,
        results_by_origin_sub_lvl=results_by_origin_sub_lvl,
        capacity_lvl=capacity_lvl,
        connected_country=connected_country,
        component_filter_turbine=component_filter_turbine,
        component_filter_farm=component_filter_farm,
        eu_spend_reduction_pct=central_mu,
        preserve_total=preserve_total,
        mrio=mrio,
        core=core,
        annualize=annualize,
        lifetime_years=lifetime_years,
    )

    def _run_case(
        *,
        markup: float,
        cache_dir: str,
        deltas: Iterable[int],
        force_recompute: bool,
        require_detailed: bool,
        cache_base=cache_base_dir,
    ):
        kwargs = dict(common_kwargs)
        kwargs["eu_spend_reduction_pct"] = float(markup)
        kwargs.update(
            cache_dir=cache_dir,
            cache_base_dir=cache_base,
            deltas_pp=tuple(dict.fromkeys(int(delta) for delta in deltas)),
            force_recompute=bool(force_recompute),
            execution_mode=("force_recompute" if force_recompute else "compute_missing"),
            require_detailed=require_detailed,
            verbose=verbose,
        )
        return maybe_run_or_load(**kwargs)

    figure_scenarios = dict(preloaded)
    figure_markup_cases = {
        "central": central_mu,
        "low": float(figure_markup_low),
        "high": float(figure_markup_high),
    }
    figure_cache_status = {}
    for name in ("low", "high"):
        markup = figure_markup_cases[name]
        normalized = _normalize_resilience_markup_values((markup,))[0]
        if normalized in figure_scenarios:
            figure_cache_status[name] = "loaded"
            continue
        figure_scenarios[normalized] = _run_case(
            markup=normalized,
            cache_dir=f"EU27_resilience_Turbine_BoP_markup_{name}",
            deltas=figure_deltas,
            force_recompute=scenario_recompute,
            require_detailed=not scenario_recompute,
        )
        figure_cache_status[name] = "calculated" if scenario_recompute else "loaded_or_calculated_missing"

    for normalized, scenarios in figure_scenarios.items():
        preloaded.setdefault(normalized, scenarios)

    sd_markup = {
        "central": figure_scenarios[central_mu][0],
        "low": figure_scenarios[_normalize_resilience_markup_values((figure_markup_low,))[0]][0],
        "high": figure_scenarios[_normalize_resilience_markup_values((figure_markup_high,))[0]][0],
    }
    sr_markup = {
        "central": figure_scenarios[central_mu][1],
        "low": figure_scenarios[_normalize_resilience_markup_values((figure_markup_low,))[0]][1],
        "high": figure_scenarios[_normalize_resilience_markup_values((figure_markup_high,))[0]][1],
    }
    procurement_savings_figure = build_eu27_procurement_savings_df(
        SR_MARKUP=sr_markup,
        SD_MARKUP=sd_markup,
        low_income=LOW_INCOME,
    )

    figure_net_parts = []
    for name in ("low", "high"):
        _, panel = build_baseline_panels(
            sr_markup[name],
            EU27=EU27,
            connected_country=connected_country,
            exclude_deployments=exclude_deployments,
            lvl_plot=accounting_component_level,
        )
        panel = panel.loc[panel["scenario"].astype(str).eq("baseline")].copy()
        for indicator in ("GVA", "Jobs"):
            effects = build_eu27_net_delta_effect_df(panel, indicator=indicator, delta_base=delta_base)
            effects["markup"] = name
            figure_net_parts.append(effects)
    markup_net_effects_figure = pd.concat(figure_net_parts, ignore_index=True)

    accounting_sd = {}
    accounting_sr = {}
    accounting_status = {}

    def _endpoint_case(markup: float):
        normalized = _normalize_resilience_markup_values((markup,))[0]
        key = _resilience_markup_key(normalized)
        if normalized in preloaded and (
            np.isclose(normalized, central_mu, atol=1e-12, rtol=0.0)
            or not accounting_recompute
        ):
            accounting_status[key] = "loaded"
            return preloaded[normalized]

        cache_name = (
            "EU27_resilience_Turbine_BoP_accounting_"
            f"{_resilience_markup_cache_token(normalized)}"
        )
        resolved = _resolve_cache_dir(cache_name, cache_base_dir)
        paths = _cache_paths(resolved, create=False)
        required = set(accounting_deltas)
        cached_keys = set()
        cached_delta = None
        cached_result = None
        try:
            if all(os.path.exists(paths[name]) for name in ("scenarios_delta", "scenarios_result")):
                cached_delta, cached_result = load_cache(cache_dir=resolved)
                cached_keys = set(cached_delta).intersection(cached_result)
        except Exception:
            cached_keys = set()
        has_extra = bool(cached_keys - required)
        validation_keys = tuple(sorted(cached_keys.intersection(required)))
        cache_compatible = bool(
            validation_keys
            and _cache_metadata_is_current(resolved)
            and _cache_markup_matches(cached_delta, validation_keys, normalized)
            and _cache_pricing_model_matches(cached_delta, validation_keys)
            and _cache_allocation_model_matches(cached_delta, validation_keys)
            and all(
                isinstance(cached_result[delta].get("_detailed"), pd.DataFrame)
                and isinstance(cached_result[delta].get("_crosswalk"), pd.DataFrame)
                for delta in validation_keys
            )
        )
        complete_and_reusable = (
            not accounting_recompute
            and required.issubset(cached_keys)
            and cache_compatible
        )
        stage = (has_extra and not complete_and_reusable) or (
            bool(cached_keys) and not cache_compatible
        )
        if stage:
            with tempfile.TemporaryDirectory(prefix="resilience_accounting_cache_") as tmp:
                staged = os.path.join(tmp, os.path.basename(resolved))
                if cache_compatible:
                    shutil.copytree(resolved, staged)
                else:
                    os.makedirs(staged, exist_ok=True)
                result = _run_case(
                    markup=normalized,
                    cache_dir=staged,
                    deltas=accounting_deltas,
                    force_recompute=accounting_recompute,
                    require_detailed=True,
                    cache_base=None,
                )
        else:
            result = _run_case(
                markup=normalized,
                cache_dir=cache_name,
                deltas=accounting_deltas,
                force_recompute=accounting_recompute,
                require_detailed=True,
            )
        accounting_status[key] = "loaded" if complete_and_reusable else "calculated"
        return result

    for markup in accounting_values:
        normalized = _normalize_resilience_markup_values((markup,))[0]
        case = _endpoint_case(normalized)
        case_delta, case_result = case
        case_delta = dict(case_delta)
        case_result = dict(case_result)
        case_delta[int(delta_base)] = central_delta[int(delta_base)]
        case_result[int(delta_base)] = central_result[int(delta_base)]
        key = _resilience_markup_key(normalized)
        accounting_sd[key] = case_delta
        accounting_sr[key] = case_result

    accounting_net_parts = []
    accounting_proc_parts = []
    accounting_baselines = {}
    central_trim = {
        delta: accounting_sd[_resilience_markup_key(central_mu)][delta]
        for delta in (*accounting_deltas, int(delta_base))
    }
    high_case = {
        delta: figure_scenarios[_normalize_resilience_markup_values((figure_markup_high,))[0]][0][delta]
        for delta in (*accounting_deltas, int(delta_base))
    }
    high_result_case = {
        delta: figure_scenarios[_normalize_resilience_markup_values((figure_markup_high,))[0]][1][delta]
        for delta in (*accounting_deltas, int(delta_base))
    }
    for markup in accounting_values:
        key = _resilience_markup_key(markup)
        case_panel = build_baseline_panels(
            accounting_sr[key],
            EU27=EU27,
            connected_country=connected_country,
            exclude_deployments=exclude_deployments,
            lvl_plot=accounting_component_level,
        )[1]
        case_panel = case_panel.loc[case_panel["scenario"].astype(str).eq("baseline")].copy()
        for indicator in ("GVA", "Jobs"):
            effects = build_eu27_net_delta_effect_df(case_panel, indicator=indicator, delta_base=delta_base)
            effects["markup"] = key
            accounting_net_parts.append(effects)

        case_delta = {delta: accounting_sd[key][delta] for delta in (*accounting_deltas, int(delta_base))}
        case_result = {delta: accounting_sr[key][delta] for delta in (*accounting_deltas, int(delta_base))}
        proc = build_eu27_procurement_savings_df(
            SR_MARKUP={
                "central": {delta: central_result[delta] for delta in (*accounting_deltas, int(delta_base))},
                "low": case_result,
                "high": high_result_case,
            },
            SD_MARKUP={"central": central_trim, "low": case_delta, "high": high_case},
            low_income=LOW_INCOME,
        )
        accounting_baselines[key] = proc.attrs["procurement_cost_baselines_eur"]["low"]
        part = proc.loc[proc["markup"].astype(str).str.casefold().eq("low")].copy()
        part["markup"] = key
        accounting_proc_parts.append(part)

    markup_net_effects_accounting = pd.concat(accounting_net_parts, ignore_index=True)
    procurement_savings_accounting = pd.concat(accounting_proc_parts, ignore_index=True)
    procurement_savings_accounting.attrs["procurement_cost_baselines_eur"] = accounting_baselines

    return {
        "sd_markup": sd_markup,
        "sr_markup": sr_markup,
        "markup_net_effects": markup_net_effects_figure,
        "procurement_savings": procurement_savings_figure,
        "sd_accounting": accounting_sd,
        "sr_accounting": accounting_sr,
        "markup_net_effects_accounting": markup_net_effects_accounting,
        "procurement_savings_accounting": procurement_savings_accounting,
        "accounting_note": _resilience_accounting_note(selected_values),
        "accounting_cache_status": accounting_status,
        "figure_cache_status": figure_cache_status,
        "accounting_markups": accounting_values,
    }


def build_eu27_procurement_savings_df(
    SR_MARKUP: dict[str, dict],
    SD_MARKUP: dict[str, dict] | None = None,
    *,
    components: tuple[str, ...] | list[str] = ("Total", "Turbine", "BoP"),
    results_level: int = 2,
    delta_base: int = 0,
    identity_rtol: float = 1e-9,
    identity_atol: float = 1e-4,
    low_income=None,
    reference_markup: float = REFERENCE_MARKUP,
) -> pd.DataFrame:
    """Build procurement savings from scenario-calibrated monetary allocations.

    GVA and output diagnostics are retained, while procurement savings are
    calculated directly from each scenario's ``value_alloc``.  The public
    columns remain unchanged.
    """
    if not isinstance(SR_MARKUP, dict) or not SR_MARKUP:
        raise ValueError("SR_MARKUP must contain central, low, and high result mappings.")
    required_markups = {"central", "low", "high"}
    missing_markups = sorted(required_markups.difference(SR_MARKUP))
    if missing_markups:
        raise ValueError(f"SR_MARKUP is missing markup cases: {missing_markups}")
    if SD_MARKUP is not None:
        missing_cost_markups = sorted(required_markups.difference(SD_MARKUP))
        if missing_cost_markups:
            raise ValueError(f"SD_MARKUP is missing markup cases: {missing_cost_markups}")
    else:
        raise ValueError("SD_MARKUP is required for reference-price procurement-cost calculations.")

    components = [str(component).strip() for component in components]
    if len(set(components)) != len(components):
        raise ValueError("components must not contain duplicates.")

    rows = []
    for markup_name in ("low", "central", "high"):
        result_by_delta = SR_MARKUP[markup_name]
        cost_by_delta = None if SD_MARKUP is None else SD_MARKUP[markup_name]
        if not isinstance(result_by_delta, dict):
            raise TypeError(f"SR_MARKUP[{markup_name!r}] must be a delta-to-result mapping.")

        if int(delta_base) not in result_by_delta:
            raise KeyError(f"SR_MARKUP[{markup_name!r}] is missing delta={delta_base}.")
        available_deltas = sorted(int(delta) for delta in result_by_delta)
        baseline_result = result_by_delta[int(delta_base)]
        baseline_components = baseline_result.get("results_by_origin_lvl", {}).get(results_level, {})
        if not isinstance(baseline_components, dict):
            raise KeyError(
                f"SR_MARKUP[{markup_name!r}] has no results_by_origin_lvl[{results_level}] mapping."
            )

        for delta in available_deltas:
            result = result_by_delta[delta]
            level_results = result.get("results_by_origin_lvl", {}).get(results_level, {})
            if not isinstance(level_results, dict):
                raise KeyError(
                    f"SR_MARKUP[{markup_name!r}][{delta}] has no "
                    f"results_by_origin_lvl[{results_level}] mapping."
                )
            scenario_delta = None if cost_by_delta is None else cost_by_delta.get(delta)
            cost_df = None if scenario_delta is None else scenario_delta.final_cost_df_cf

            for component in components:
                result_df = level_results.get(component)
                if not isinstance(result_df, pd.DataFrame):
                    raise KeyError(
                        f"SR_MARKUP[{markup_name!r}][{delta}] is missing level-{results_level} "
                        f"component {component!r}."
                    )
                required_result_cols = {"GVA_Total", "Output_Direct"}
                missing_result_cols = sorted(required_result_cols.difference(result_df.columns))
                if missing_result_cols:
                    raise KeyError(
                        f"SR_MARKUP[{markup_name!r}][{delta}] component {component!r} "
                        f"is missing columns {missing_result_cols}."
                    )

                global_gva = float(pd.to_numeric(result_df["GVA_Total"], errors="raise").sum())
                global_output = float(pd.to_numeric(result_df["Output_Direct"], errors="raise").sum())

                allocation_total = np.nan
                if cost_df is not None:
                    if "value_alloc" not in cost_df.columns:
                        raise KeyError(
                            f"SD_MARKUP[{markup_name!r}][{delta}] final_cost_df_cf is missing "
                            "'value_alloc'."
                        )
                    if component == "Total":
                        allocation = cost_df
                    else:
                        hierarchy_col = "component_group_L2"
                        if hierarchy_col not in cost_df.columns:
                            raise KeyError(
                                f"SD_MARKUP[{markup_name!r}][{delta}] final_cost_df_cf is missing "
                                f"'{hierarchy_col}' for component-level verification."
                            )
                        allocation = cost_df[
                            cost_df[hierarchy_col].astype(str).str.strip().eq(component)
                        ]
                    allocation_total = float(
                        pd.to_numeric(allocation["value_alloc"], errors="raise").sum()
                    )

                discrepancy_output = global_gva - global_output
                discrepancy_allocation = (
                    np.nan if pd.isna(allocation_total) else global_gva - allocation_total
                )
                checks = [
                    ("Output_Direct", discrepancy_output, global_output),
                ]
                if not pd.isna(discrepancy_allocation):
                    checks.append(("value_alloc", discrepancy_allocation, allocation_total))
                # These are diagnostics only.  GVA, gross output, and
                # procurement expenditure are not interchangeable identities
                # under the fixed-reference-quantity specification.

                baseline_component = baseline_components.get(component)
                if not isinstance(baseline_component, pd.DataFrame):
                    raise KeyError(
                        f"SR_MARKUP[{markup_name!r}][{delta_base}] is missing level-{results_level} "
                        f"component {component!r}."
                    )
                baseline_gva = float(
                    pd.to_numeric(baseline_component["GVA_Total"], errors="raise").sum()
                )
                markup_value = {"low": 0.1, "central": 0.3, "high": 0.5}[markup_name]
                baseline_cost = _reference_price_cost_from_df(
                    cost_by_delta[delta_base].final_cost_df_cf,
                    markup=markup_value,
                    low_income=low_income,
                    reference_markup=reference_markup,
                    mask=(
                        None
                        if component == "Total"
                        else cost_by_delta[delta_base].final_cost_df_cf["component_group_L2"].astype(str).str.strip().eq(component)
                    ),
                ) if cost_by_delta is not None else np.nan
                scenario_cost = _reference_price_cost_from_df(
                    cost_df,
                    markup=markup_value,
                    low_income=low_income,
                    reference_markup=reference_markup,
                    mask=(
                        None
                        if component == "Total"
                        else cost_df["component_group_L2"].astype(str).str.strip().eq(component)
                    ),
                )

                rows.append({
                    "markup": markup_name,
                    "delta_pp": int(delta),
                    "Component": component,
                    "global_gva_total_eur": global_gva,
                    "global_output_direct_eur": global_output,
                    "allocation_value_eur": allocation_total,
                    "identity_discrepancy_output_eur": discrepancy_output,
                    "identity_discrepancy_allocation_eur": discrepancy_allocation,
                    "baseline_global_gva_total_eur": baseline_gva,
                    "procurement_savings_million_eur": (baseline_cost - scenario_cost) / 1_000_000.0,
                })

    data = pd.DataFrame(rows)
    if SD_MARKUP is not None:
        baseline_costs = {}
        scenario_costs = {}
        for markup_name, markup_value in (("low", 0.1), ("central", 0.3), ("high", 0.5)):
            baseline_delta = SD_MARKUP[markup_name][int(delta_base)]
            baseline_df = baseline_delta.final_cost_df_cf
            baseline_costs[markup_name] = {}
            scenario_costs[markup_name] = {}
            for component in components:
                mask = (
                    None
                    if component == "Total"
                    else baseline_df["component_group_L2"].astype(str).str.strip().eq(component)
                )
                baseline_costs[markup_name][component] = _reference_price_cost_from_df(
                    baseline_df,
                    markup=markup_value,
                    low_income=low_income,
                    reference_markup=reference_markup,
                    mask=mask,
                )
                scenario_costs[markup_name][component] = {
                    int(delta): _reference_price_cost_from_df(
                        SD_MARKUP[markup_name][delta].final_cost_df_cf,
                        markup=markup_value,
                        low_income=low_income,
                        reference_markup=reference_markup,
                        mask=(
                            None
                            if component == "Total"
                            else SD_MARKUP[markup_name][delta].final_cost_df_cf["component_group_L2"].astype(str).str.strip().eq(component)
                        ),
                    )
                    for delta in available_deltas
                }
        data.attrs["procurement_cost_baselines_eur"] = baseline_costs
        data.attrs["procurement_cost_scenarios_eur"] = scenario_costs
        data.attrs["procurement_cost_reference_markup"] = float(reference_markup)
    return data










# Broad component groups used in the saved EU27 sourcing scenarios.  These
# deliberately stop at the five turbine components and five BoP components.
EU27_COMPONENT_SOURCING_SPECS = (
    ("Turbine", "Drivetrain", "Drivetrain"),
    ("Turbine", "Blades", "Blades"),
    ("Turbine", "Nacelle", "Nacelle"),
    ("Turbine", "Tower", "Tower"),
    ("Turbine", "Hub", "Hub"),
    ("BoP", "Array", "Array cables"),
    ("BoP", "Export", "Export cables"),
    ("BoP", "Foundation", "Foundation"),
    ("BoP", "OSS", "OSS"),
    ("BoP", "Other", "Other"),
)
EU27_COMPONENT_SOURCING_COLORS = {
    "Drivetrain": "#4E79A7",
    "Blades": "#F28E2B",
    "Nacelle": "#E15759",
    "Tower": "#76B7B2",
    "Hub": "#59A14F",
    "Array": "#EDC948",
    "Export": "#B07AA1",
    "Foundation": "#FF9DA7",
    "OSS": "#9C755F",
    "Other": "#79706E",
}

DEFAULT_LOW_INCOME = frozenset({
    "RU", "RS", "TR", "MK", "ME", "AL", "JP",
    "KR", "SA", "CN", "AR", "BR", "ZA", "MX", "ID", "ROW", "IN",
})


def _reference_price_cost_from_df(
    cost_df: pd.DataFrame,
    *,
    markup: float,
    low_income=None,
    reference_markup: float = REFERENCE_MARKUP,
    mask: pd.Series | None = None,
) -> float:
    """Return the already-calibrated monetary allocation without revaluation.

    ``markup`` and ``reference_markup`` remain accepted for compatibility with
    existing callers, but scenario allocations are now calibrated and valued
    at the same scenario markup before they reach this helper.
    """
    if not isinstance(cost_df, pd.DataFrame) or "value_alloc" not in cost_df.columns:
        raise TypeError("A final_cost_df_cf DataFrame with value_alloc is required.")
    selected = cost_df if mask is None else cost_df.loc[mask]
    if selected.empty:
        return 0.0
    values = pd.to_numeric(selected["value_alloc"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Scenario procurement allocation contains non-finite values.")
    return float(values.sum())


def _eu27_sourcing_field(value, name, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _eu27_sourcing_component_cost(
    scenario_delta,
    component,
    *,
    deployment="EU27_AVG",
    markup: float = REFERENCE_MARKUP,
    low_income=None,
    reference_markup: float = REFERENCE_MARKUP,
):
    cost_df = _eu27_sourcing_field(scenario_delta, "final_cost_df_cf")
    if not isinstance(cost_df, pd.DataFrame):
        raise TypeError("Each saved scenario must contain final_cost_df_cf as a DataFrame.")
    required = {"deployment_country", "component_group", "sub_component", "value_alloc"}
    missing = sorted(required.difference(cost_df.columns))
    if missing:
        raise KeyError(f"final_cost_df_cf is missing required columns: {missing}")
    spec = next((row for row in EU27_COMPONENT_SOURCING_SPECS if row[1] == component), None)
    if spec is None:
        raise KeyError(f"Unknown EU27 sourcing component {component!r}.")
    group, _, subcomponent = spec
    mask = cost_df["deployment_country"].astype("string").str.strip().eq(str(deployment))
    if group == "Turbine":
        mask &= cost_df["component_group"].astype("string").str.strip().eq(component)
    else:
        mask &= cost_df["component_group"].astype("string").str.strip().eq("BoP")
        mask &= cost_df["sub_component"].astype("string").str.strip().eq(subcomponent)
    if not mask.any():
        raise KeyError(
            f"No saved procurement rows for deployment={deployment!r}, component={component!r}."
        )
    values = pd.to_numeric(cost_df.loc[mask, "value_alloc"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite procurement values found for component {component!r}.")
    return _reference_price_cost_from_df(
        cost_df,
        markup=markup,
        low_income=low_income,
        reference_markup=reference_markup,
        mask=mask,
    )


def _eu27_sourcing_project_cost(
    scenario_delta,
    *,
    deployment="EU27_AVG",
    markup: float = REFERENCE_MARKUP,
    low_income=None,
    reference_markup: float = REFERENCE_MARKUP,
):
    """Return the full saved project procurement cost for one deployment."""
    cost_df = _eu27_sourcing_field(scenario_delta, "final_cost_df_cf")
    if not isinstance(cost_df, pd.DataFrame):
        raise TypeError("Each saved scenario must contain final_cost_df_cf as a DataFrame.")
    required = {"deployment_country", "value_alloc"}
    missing = sorted(required.difference(cost_df.columns))
    if missing:
        raise KeyError(f"final_cost_df_cf is missing required columns: {missing}")
    mask = cost_df["deployment_country"].astype("string").str.strip().eq(str(deployment))
    if not mask.any():
        raise KeyError(f"No saved procurement rows for deployment={deployment!r}.")
    values = pd.to_numeric(cost_df.loc[mask, "value_alloc"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite procurement values found in the full project cost.")
    return _reference_price_cost_from_df(
        cost_df,
        markup=markup,
        low_income=low_income,
        reference_markup=reference_markup,
        mask=mask,
    )


def _eu27_sourcing_result_component(result, component):
    levels = _eu27_sourcing_field(result, "results_by_origin_lvl")
    sublevels = _eu27_sourcing_field(result, "results_by_origin_sub_lvl")
    if isinstance(levels, dict):
        level_zero = levels.get(0)
    elif isinstance(levels, (list, tuple)) and levels:
        level_zero = levels[0]
    else:
        level_zero = None
    if isinstance(sublevels, dict):
        sublevel_zero = sublevels.get(0)
    elif isinstance(sublevels, (list, tuple)) and sublevels:
        sublevel_zero = sublevels[0]
    else:
        sublevel_zero = None

    spec = next((row for row in EU27_COMPONENT_SOURCING_SPECS if row[1] == component), None)
    if spec is None:
        raise KeyError(f"Unknown EU27 sourcing component {component!r}.")
    group, _, subcomponent = spec
    if group == "Turbine":
        frame = level_zero.get(component) if isinstance(level_zero, dict) else None
    else:
        frame = sublevel_zero.get(("BoP", subcomponent)) if isinstance(sublevel_zero, dict) else None
    if not isinstance(frame, pd.DataFrame):
        raise KeyError(f"Saved MRIO results lack the required {component!r} component table.")
    return frame


def _eu27_sourcing_impact(result, component, indicator, *, deployment="EU27_AVG"):
    indicator = str(indicator).strip().upper()
    value_col = {"GVA": "GVA_Total", "JOBS": "Jobs_Total"}.get(indicator)
    if value_col is None:
        raise ValueError("indicator must be 'GVA' or 'Jobs'.")
    frame = _eu27_sourcing_result_component(result, component)
    required = {"Deployment", "Country", value_col}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Saved {component!r} results are missing columns: {missing}")
    mask = frame["Deployment"].astype("string").str.strip().eq(str(deployment))
    mask &= frame["Country"].astype("string").str.strip().str.upper().isin(set(EU27))
    if not mask.any():
        raise KeyError(f"No EU27 impact rows for deployment={deployment!r}, component={component!r}.")
    values = pd.to_numeric(frame.loc[mask, value_col], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite {value_col} values found for component {component!r}.")
    return float(values.sum())


def _eu27_sourcing_project_impact(result, indicator, *, deployment="EU27_AVG"):
    """Return the saved full-project EU27 impact for one indicator."""
    indicator = str(indicator).strip().upper()
    value_col = {"GVA": "GVA_Total", "JOBS": "Jobs_Total"}.get(indicator)
    if value_col is None:
        raise ValueError("indicator must be 'GVA' or 'Jobs'.")
    levels = _eu27_sourcing_field(result, "results_by_origin_lvl")
    if isinstance(levels, dict):
        level_zero = levels.get(0)
    elif isinstance(levels, (list, tuple)) and levels:
        level_zero = levels[0]
    else:
        level_zero = None
    frame = level_zero.get("Total") if isinstance(level_zero, dict) else None
    if not isinstance(frame, pd.DataFrame):
        raise KeyError("Saved MRIO results lack the full-project Total table.")
    required = {"Deployment", "Country", value_col}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Saved full-project results are missing columns: {missing}")
    mask = frame["Deployment"].astype("string").str.strip().eq(str(deployment))
    mask &= frame["Country"].astype("string").str.strip().str.upper().isin(set(EU27))
    if not mask.any():
        raise KeyError(f"No EU27 full-project impact rows for deployment={deployment!r}.")
    values = pd.to_numeric(frame.loc[mask, value_col], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite full-project {value_col} values found.")
    return float(values.sum())


def _eu27_sourcing_markup(scenario_delta):
    values = []
    for stat_name in ("stats_turbine", "stats_farm"):
        stats = _eu27_sourcing_field(scenario_delta, stat_name)
        value = _eu27_sourcing_field(stats, "eu_spend_reduction_pct") if stats is not None else None
        if value is not None and np.isfinite(float(value)):
            values.append(float(value))
    if not values or not np.allclose(values, values[0], rtol=0.0, atol=1e-12):
        raise ValueError("Saved scenario lacks one consistent, finite markup value.")
    return values[0]


def build_eu27_component_strategic_quadrant_data(
    *,
    scenarios_delta: dict,
    scenarios_result: dict,
    markup: float,
    indicator: str = "GVA",
    annualize: bool = False,
    delta_offshoring: int = -100,
    delta_reshoring: int = 100,
) -> pd.DataFrame:
    """Build all-ten-component, median-split EU27 sourcing quadrant data.

    The stored outcomes include the saved markup assumption.  Procurement
    expenditure and MRIO impacts are read directly from matching saved
    scenario objects; this helper does not rerun or alter the model.
    """
    if not isinstance(scenarios_delta, dict) or not isinstance(scenarios_result, dict):
        raise TypeError("scenarios_delta and scenarios_result must be delta-to-result mappings.")
    needed = {0, int(delta_offshoring), int(delta_reshoring)}
    if not needed.issubset(scenarios_delta) or not needed.issubset(scenarios_result):
        raise KeyError(f"Saved scenario mappings must include deltas {sorted(needed)}.")
    actual_markup = _eu27_sourcing_markup(scenarios_delta[0])
    if not np.isclose(actual_markup, float(markup), rtol=0.0, atol=1e-12):
        raise ValueError(f"Requested markup {markup:g} does not match saved markup {actual_markup:g}.")
    indicator_key = str(indicator).strip().upper()
    if indicator_key not in {"GVA", "JOBS"}:
        raise ValueError("indicator must be 'GVA' or 'Jobs'.")
    if int(delta_offshoring) >= 0 or int(delta_reshoring) <= 0:
        raise ValueError("Offshoring delta must be negative and reshoring delta must be positive.")

    rows = []
    for group, component, _subcomponent in EU27_COMPONENT_SOURCING_SPECS:
        base_cost = _eu27_sourcing_component_cost(scenarios_delta[0], component)
        if not np.isfinite(base_cost) or base_cost <= 0.0:
            raise ValueError(f"Baseline procurement expenditure must be positive for {component!r}.")
        base_impact = _eu27_sourcing_impact(scenarios_result[0], component, indicator_key)
        off_impact = _eu27_sourcing_impact(
            scenarios_result[int(delta_offshoring)], component, indicator_key
        )
        re_impact = _eu27_sourcing_impact(
            scenarios_result[int(delta_reshoring)], component, indicator_key
        )
        if indicator_key == "GVA":
            x = (base_impact - off_impact) / base_cost * 100.0
            y = (re_impact - base_impact) / base_cost * 100.0
            exposure_unit = opportunity_unit = "% of baseline component expenditure"
        else:
            baseline_spend_million = base_cost / 1_000_000.0
            x = (base_impact - off_impact) / baseline_spend_million
            y = (re_impact - base_impact) / baseline_spend_million
            employment_unit = "FTE" if annualize else "FTE-years"
            exposure_unit = opportunity_unit = f"{employment_unit} per million EUR"
        rows.append({
            "component_group": group,
            "component": component,
            "markup": float(markup),
            "indicator": indicator_key,
            "annualize": bool(annualize),
            "baseline_expenditure_eur": base_cost,
            "baseline_impact": base_impact,
            "offshoring_delta_pp": int(delta_offshoring),
            "offshoring_impact": off_impact,
            "reshoring_delta_pp": int(delta_reshoring),
            "reshoring_impact": re_impact,
            "offshoring_exposure": float(x),
            "reshoring_opportunity": float(y),
            "exposure_unit": exposure_unit,
            "opportunity_unit": opportunity_unit,
        })
    data = pd.DataFrame(rows)
    if data["component"].tolist() != [row[1] for row in EU27_COMPONENT_SOURCING_SPECS]:
        raise AssertionError("Strategic quadrant data does not contain the expected 10 components.")
    if not np.isfinite(data[["offshoring_exposure", "reshoring_opportunity"]].to_numpy()).all():
        raise ValueError("Strategic quadrant data contains non-finite points.")
    return data


def plot_eu27_component_strategic_quadrant(
    data: pd.DataFrame,
    *,
    filename: str,
    save: bool | None = None,
    close: bool | None = None,
):
    """Plot ten component-level EU27 exposure/opportunity points."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    required = {
        "component_group", "component", "markup", "indicator", "annualize",
        "baseline_expenditure_eur", "offshoring_exposure", "reshoring_opportunity",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise KeyError(f"Strategic quadrant data is missing columns: {missing}")
    expected = [row[1] for row in EU27_COMPONENT_SOURCING_SPECS]
    if data["component"].tolist() != expected:
        raise ValueError("Quadrant plot requires each of the 10 broad components exactly once and in order.")
    indicator = str(data["indicator"].iloc[0]).upper()
    if data["indicator"].astype(str).str.upper().nunique() != 1:
        raise ValueError("Quadrant points must use the same indicator.")
    annualize = bool(data["annualize"].iloc[0])
    if data["annualize"].astype(bool).nunique() != 1:
        raise ValueError("Quadrant points must use the same annualization setting.")
    x = pd.to_numeric(data["offshoring_exposure"], errors="raise").to_numpy(dtype=float)
    y = pd.to_numeric(data["reshoring_opportunity"], errors="raise").to_numpy(dtype=float)
    expenditure = pd.to_numeric(data["baseline_expenditure_eur"], errors="raise").to_numpy(dtype=float)
    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(expenditure).all()):
        raise ValueError("Quadrant plot values must be finite.")
    if (expenditure <= 0.0).any():
        raise ValueError("Baseline expenditure must be positive for every component.")

    if data["markup"].nunique() != 1:
        raise ValueError("All quadrant observations must use the same baseline markup.")
    median_x = float(np.median(x))
    median_y = float(np.median(y))
    x_upper = max(float(x.max()), 0.0)
    y_upper = max(float(y.max()), 0.0)
    x_lower = min(float(x.min()), 0.0)
    y_lower = min(float(y.min()), 0.0)
    x_pad = max((x_upper - x_lower) * 0.12, abs(x_upper) * 0.035, 0.05)
    y_pad = max((y_upper - y_lower) * 0.12, abs(y_upper) * 0.035, 0.05)

    set_house_style()
    fig, ax = plt.subplots(figsize=(12.0617, 3.8818), dpi=150)
    for row in data.itertuples(index=False):
        component = str(row.component)
        ax.scatter(
            float(row.offshoring_exposure),
            float(row.reshoring_opportunity),
            s=30,
            color=EU27_COMPONENT_SOURCING_COLORS[component],
            marker="o",
            edgecolor="white", linewidth=0.55, alpha=0.95,
            zorder=3,
        )
    ax.axvline(median_x, color="#444444", linewidth=0.9, linestyle="--", alpha=0.8, zorder=1)
    ax.axhline(median_y, color="#444444", linewidth=0.9, linestyle="--", alpha=0.8, zorder=1)
    ax.set_xlim(x_lower - x_pad, x_upper + x_pad)
    ax.set_ylim(y_lower - y_pad, y_upper + y_pad)
    if indicator == "GVA":
        ax.set_xlabel(r"\textbf{$-\Delta GVA_{\mathrm{EU27},c}^{\mathrm{off}}/Y_c$ [\%]}", fontsize=9)
        ax.set_ylabel(r"\textbf{$\Delta GVA_{\mathrm{EU27},c}^{\mathrm{res}}/Y_c$ [\%]}", fontsize=9)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))
    else:
        employment_unit = "FTE" if annualize else "FTE-years"
        ax.set_xlabel(rf"\textbf{{$-\Delta FTE_{{\mathrm{{EU27}},c}}^{{\mathrm{{off}}}}/(Y_c/10^6)$ [{employment_unit} per million EUR]}}", fontsize=9)
        ax.set_ylabel(rf"\textbf{{$\Delta FTE_{{\mathrm{{EU27}},c}}^{{\mathrm{{res}}}}/(Y_c/10^6)$ [{employment_unit} per million EUR]}}", fontsize=9)
    apply_house_style(ax, grid=True)
    component_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor=EU27_COMPONENT_SOURCING_COLORS[component],
            markeredgecolor="white", markeredgewidth=0.5,
            markersize=5.2, label=component,
        )
        for _group, component, _ in EU27_COMPONENT_SOURCING_SPECS
    ]
    ax.legend(
        handles=component_handles,
        loc="upper left", ncol=5, frameon=True, framealpha=0.88,
        facecolor="white", edgecolor="#CCCCCC", fontsize=8.0,
        handletextpad=0.35, labelspacing=0.25, columnspacing=0.75,
        borderpad=0.35, handlelength=0.8,
    )
    ax.tick_params(axis="both", labelsize=8)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.22, top=0.96)
    path = save_and_close(fig=fig, filename=filename, save=save, close=close, exact_size=True)
    return fig, ax, path


def _eu27_tradeoff_normalization_scope(value) -> str:
    """Validate the Figure 3.5 normalization mode."""
    scope = "absolute" if value is None else str(value).strip().lower()
    if scope not in {"absolute", "category", "project"}:
        raise ValueError("normalization_scope must be 'absolute', 'category', or 'project'.")
    return scope


def plot_eu27_aggregate_cost_impact_curves(
    data: pd.DataFrame,
    *,
    indicator: str,
    filename: str,
    baseline_markup: float,
    axis_limits: dict[str, tuple[float, float]],
    normalization_scope: str | None = "project",
    annualize: bool | None = None,
    save: bool | None = None,
    close: bool | None = None,
):
    """Plot aggregate Turbine, BoP, and Total project trade-off curves."""
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D

    scope = _eu27_tradeoff_normalization_scope(normalization_scope)
    if scope != "absolute":
        data = _normalize_eu27_tradeoff_data(data, normalization_scope=scope)
    indicator_key = str(indicator).strip().upper()
    if indicator_key not in {"GVA", "JOBS"}:
        raise ValueError("indicator must be 'GVA' or 'Jobs'.")
    series = ("Turbine", "BoP", "Total project")
    if set(data["component_group"].astype(str).unique()) != set(series):
        raise ValueError(f"Aggregate plot must contain exactly these series: {series}.")
    if annualize is None:
        annualize = bool(data["annualize"].astype(bool).iloc[0])
    if data["annualize"].astype(bool).nunique() != 1 or bool(data["annualize"].iloc[0]) != bool(annualize):
        raise ValueError("Requested annualization does not match the saved aggregate data.")
    if indicator_key == "JOBS" and bool(annualize):
        raise ValueError("These manuscript employment curves report cumulative FTE-years.")

    markups = sorted(pd.to_numeric(data["markup"], errors="raise").unique().tolist())
    if (
        len(markups) != 3
        or not np.isclose(markups[1], float(baseline_markup), rtol=0.0, atol=1e-12)
    ):
        raise ValueError(
            "Curves must contain low, configured central, and high markup cases, "
            f"with central markup {baseline_markup:g}."
        )
    low_markup, _central_markup, high_markup = markups
    baseline_costs = {}
    for aggregate in series:
        aggregate_rows = data.loc[
            data["component_group"].astype(str).eq(aggregate)
            & np.isclose(data["markup"].astype(float), float(baseline_markup), rtol=0.0, atol=1e-12)
        ]
        if aggregate_rows.empty or aggregate_rows["baseline_expenditure_eur"].nunique(dropna=False) != 1:
            raise ValueError(f"{aggregate}: expected one positive baseline expenditure.")
        cost = float(aggregate_rows["baseline_expenditure_eur"].iloc[0])
        if not np.isfinite(cost) or cost <= 0.0:
            raise ValueError(f"{aggregate}: invalid baseline expenditure.")
        baseline_costs[aggregate] = cost

    x_col = "cost_change_million_eur"
    y_col = "gva_change_million_eur" if indicator_key == "GVA" else "employment_change"
    if scope == "absolute":
        x_label = r"\boldmath$\Delta\,\mathrm{Procurement\ cost}\;[\mathrm{million\ EUR}]$"
        if indicator_key == "GVA":
            y_label = r"\boldmath$\Delta\,\mathrm{EU27\ GVA}\;[\mathrm{million\ EUR}]$"
        else:
            y_label = r"\boldmath$\Delta\,\mathrm{EU27\ employment}\;[\mathrm{FTE\text{-}years}]$"
    else:
        x_label = r"\boldmath$\Delta\,\mathrm{Procurement\ cost}\;[\%\ \mathrm{var}]$"
        if indicator_key == "GVA":
            y_label = r"\boldmath$\Delta\,\mathrm{EU27\ GVA}\;[\%\ \mathrm{var}]$"
        else:
            y_label = r"\boldmath$\Delta\,\mathrm{EU27\ employment}\;[\%\ \mathrm{var}]$"

    if not isinstance(axis_limits, dict) or set(axis_limits) != {"xlim", "ylim"}:
        raise ValueError("axis_limits must contain the shared 'xlim' and 'ylim' tuples.")
    validated_axis_limits = {}
    for key in ("xlim", "ylim"):
        bounds = tuple(float(value) for value in axis_limits[key])
        if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError(f"Invalid shared {key}: {axis_limits[key]!r}.")
        validated_axis_limits[key] = bounds

    aggregate_colors = {
        "Turbine": EU27_COMPONENT_SOURCING_COLORS["Drivetrain"],
        "BoP": "#6543A1",
        "Total project": "#41464D",
    }
    set_house_style()
    fig, ax = plt.subplots(figsize=(4.35, 3.25), dpi=150)
    curves = {}
    plotted_values = []
    for aggregate in series:
        aggregate_rows = data.loc[data["component_group"].astype(str).eq(aggregate)]
        curve = aggregate_rows.loc[
            np.isclose(aggregate_rows["markup"].astype(float), float(baseline_markup), rtol=0.0, atol=1e-12)
        ].sort_values("delta_pp")
        low_curve = aggregate_rows.loc[
            np.isclose(aggregate_rows["markup"].astype(float), low_markup, rtol=0.0, atol=1e-12)
        ].sort_values("delta_pp")
        high_curve = aggregate_rows.loc[
            np.isclose(aggregate_rows["markup"].astype(float), high_markup, rtol=0.0, atol=1e-12)
        ].sort_values("delta_pp")
        expected_deltas = list(range(-100, 101, 5))
        deltas = curve["delta_pp"].astype(int).tolist()
        if deltas != expected_deltas or low_curve["delta_pp"].astype(int).tolist() != expected_deltas or high_curve["delta_pp"].astype(int).tolist() != expected_deltas:
            raise ValueError(f"{aggregate}: expected the 41 saved delta observations from -100 to +100.")
        x_values = curve[x_col].to_numpy(dtype=float)
        y_values = curve[y_col].to_numpy(dtype=float)
        low_values = low_curve[[x_col, y_col]].to_numpy(dtype=float)
        high_values = high_curve[[x_col, y_col]].to_numpy(dtype=float)
        values = np.column_stack([x_values, y_values])
        if not np.isfinite(values).all() or not np.isfinite(low_values).all() or not np.isfinite(high_values).all():
            raise ValueError(f"{aggregate}: trade-off coordinates are non-finite.")
        zero_idx = deltas.index(0)
        if not np.allclose(values[zero_idx], [0.0, 0.0], rtol=0.0, atol=1e-10):
            raise AssertionError(f"{aggregate}: delta=0 must be exactly the origin.")
        plotted_values.append(values)
        curves[aggregate] = (deltas, x_values, y_values)
        band_vertices = np.stack(
            (low_values[:-1], low_values[1:], high_values[1:], high_values[:-1]),
            axis=1,
        )
        ax.add_collection(PolyCollection(
            band_vertices,
            facecolors=[aggregate_colors[aggregate]],
            edgecolors="none",
            alpha=0.10,
            label="_nolegend_",
            zorder=2,
        ))
        ax.plot(
            x_values, y_values,
            color=aggregate_colors[aggregate],
            linestyle="-", linewidth=1.15,
            zorder=3,
        )

    plotted_values = np.vstack(plotted_values)
    cost_min, cost_max = min(baseline_costs.values()), max(baseline_costs.values())
    if np.isclose(cost_min, cost_max, rtol=0.0, atol=1e-12):
        marker_area = {aggregate: 28.0 for aggregate in series}
    else:
        marker_area = {
            aggregate: 12.0 + 34.0 * (baseline_costs[aggregate] - cost_min) / (cost_max - cost_min)
            for aggregate in series
        }
    for aggregate in series:
        deltas, x_values, y_values = curves[aggregate]
        color = aggregate_colors[aggregate]
        zero_idx = deltas.index(0)
        baseline_xy = np.array([x_values[zero_idx], y_values[zero_idx]])
        for delta, marker, face in ((-100, "o", "white"), (100, "^", color)):
            endpoint_idx = deltas.index(delta)
            endpoint_xy = np.array([x_values[endpoint_idx], y_values[endpoint_idx]])
            if np.allclose(endpoint_xy, baseline_xy, rtol=0.0, atol=1e-10):
                continue
            ax.scatter(
                endpoint_xy[0], endpoint_xy[1], s=marker_area[aggregate], marker=marker,
                facecolors=face, edgecolors=color, linewidths=0.85, zorder=5,
            )
        ax.scatter(
            baseline_xy[0], baseline_xy[1], s=10.0, marker="o",
            facecolors=color, edgecolors=color, linewidths=0.5, zorder=6,
        )

    ax.axhline(0.0, color="#777777", linewidth=0.6, alpha=0.6, zorder=1)
    ax.axvline(0.0, color="#777777", linewidth=0.6, alpha=0.6, zorder=1)
    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel(y_label, fontsize=8.5)
    apply_house_style(ax, grid=True)
    ax.grid(False, axis="y", which="minor")
    if indicator_key == "GVA":
        ax.grid(True, axis="y", which="major", linewidth=0.4, alpha=0.3)
        ax.tick_params(axis="y", which="minor", length=0)
    ax.tick_params(axis="both", labelsize=7.5)

    def _format_tick(value, _position):
        value = float(value)
        if abs(value) < 1e-10:
            return "0"
        if abs(value) >= 100.0:
            return f"{value:,.0f}"
        return f"{value:,.1f}".rstrip("0").rstrip(".")

    ax.xaxis.set_major_formatter(plt.FuncFormatter(_format_tick))
    def _format_y_tick(value, _position):
        value = float(value)
        if abs(value) >= 1000.0:
            scaled = value / 1000.0
            return f"{scaled:.0f}k" if abs(scaled) >= 10 else f"{scaled:.1f}k".rstrip("0").rstrip(".")
        return f"{value:.0f}"

    ax.yaxis.set_major_formatter(plt.FuncFormatter(_format_y_tick))
    aggregate_handles = [
        Line2D(
            [0], [0], color=aggregate_colors[aggregate], linewidth=1.2, label=aggregate,
        )
        for aggregate in series
    ]
    delta_handles = [
        Line2D(
            [0], [0], linestyle="none", marker=marker,
            markerfacecolor=face, markeredgecolor="#444444", markeredgewidth=0.7,
            markersize=4.2, label=label,
        )
        for marker, face, label in (
            ("o", "white", r"$\delta=-100$"),
            ("^", "#444444", r"$\delta=+100$"),
        )
    ]
    ax.legend(
        handles=aggregate_handles + delta_handles,
        title=tex_bold("Aggregate"),
        title_fontproperties={"family": "serif", "size": 6.6},
        loc="lower right", bbox_to_anchor=(0.98, 0.02), bbox_transform=ax.transAxes,
        ncol=1, frameon=True, fancybox=False, framealpha=0.94,
        facecolor="white", edgecolor="#B8B8B8",
        prop={"family": "serif", "size": 6.6},
        handlelength=1.0, handletextpad=0.45, columnspacing=0.6,
        labelspacing=0.22, borderpad=0.45, borderaxespad=0.0,
    )
    ax.set_xlim(*validated_axis_limits["xlim"])
    ax.set_ylim(*validated_axis_limits["ylim"])
    fig.subplots_adjust(left=0.145, right=0.99, bottom=0.13, top=0.985)
    path = save_and_close(fig=fig, filename=filename, save=save, close=close, exact_size=True)
    return fig, ax, path


def build_eu27_component_tradeoff_curve_data(
    *,
    scenarios_delta_by_markup: dict,
    scenarios_result_by_markup: dict,
    expected_markups: tuple[float, ...],
    expected_deltas: tuple[int, ...] = tuple(range(-100, 101, 5)),
    annualize: bool = False,
    low_income=None,
    reference_markup: float = REFERENCE_MARKUP,
) -> pd.DataFrame:
    """Extract saved component cost, EU27 GVA and employment changes.

    Every row corresponds to an existing (delta, markup) cache entry.  Cost
    and impact changes are measured from that markup's own delta-zero result.
    """
    if not isinstance(scenarios_delta_by_markup, dict) or not isinstance(scenarios_result_by_markup, dict):
        raise TypeError("Scenario inputs must map markup values to saved delta mappings.")
    normalized_delta = {float(key): value for key, value in scenarios_delta_by_markup.items()}
    normalized_result = {float(key): value for key, value in scenarios_result_by_markup.items()}
    required_markups = tuple(float(value) for value in expected_markups)
    if set(normalized_delta) != set(required_markups) or set(normalized_result) != set(required_markups):
        raise ValueError(f"Expected exactly the saved markup cases {required_markups}.")
    expected_delta_set = {int(value) for value in expected_deltas}
    if 0 not in expected_delta_set or len(expected_delta_set) != len(expected_deltas):
        raise ValueError("expected_deltas must contain unique values including zero.")

    rows = []
    for markup in required_markups:
        delta_map = normalized_delta[markup]
        result_map = normalized_result[markup]
        if not isinstance(delta_map, dict) or not isinstance(result_map, dict):
            raise TypeError(f"Saved markup {markup:g} must contain delta mappings.")
        delta_keys = {int(key) for key in delta_map}
        result_keys = {int(key) for key in result_map}
        if delta_keys != expected_delta_set or result_keys != expected_delta_set:
            raise ValueError(
                f"Markup {markup:g} does not contain the complete expected delta set "
                f"({len(expected_delta_set)} observations)."
            )
        delta_lookup = {int(key): value for key, value in delta_map.items()}
        result_lookup = {int(key): value for key, value in result_map.items()}
        actual_markup = _eu27_sourcing_markup(delta_lookup[0])
        if not np.isclose(actual_markup, markup, rtol=0.0, atol=1e-12):
            raise ValueError(f"Cache key μ={markup:g} contains saved μ={actual_markup:g}.")

        baseline_delta = delta_lookup[0]
        baseline_result = result_lookup[0]
        project_baseline_cost = _eu27_sourcing_project_cost(
            baseline_delta,
            markup=markup,
            low_income=low_income,
            reference_markup=reference_markup,
        )
        project_baseline_gva = _eu27_sourcing_project_impact(baseline_result, "GVA")
        project_baseline_jobs = _eu27_sourcing_project_impact(baseline_result, "Jobs")
        if project_baseline_cost <= 0.0:
            raise ValueError(f"Full-project baseline expenditure is not positive for markup {markup:g}.")
        if project_baseline_gva == 0.0 or project_baseline_jobs == 0.0:
            raise ValueError(f"Full-project baseline impact is zero for markup {markup:g}.")
        for group, component, _subcomponent in EU27_COMPONENT_SOURCING_SPECS:
            base_cost = _eu27_sourcing_component_cost(
                baseline_delta,
                component,
                markup=markup,
                low_income=low_income,
                reference_markup=reference_markup,
            )
            if base_cost <= 0.0:
                raise ValueError(f"Baseline expenditure is not positive for {component!r}.")
            base_gva = _eu27_sourcing_impact(baseline_result, component, "GVA")
            base_jobs = _eu27_sourcing_impact(baseline_result, component, "Jobs")
            for delta in sorted(expected_delta_set):
                current_cost = _eu27_sourcing_component_cost(
                    delta_lookup[delta],
                    component,
                    markup=markup,
                    low_income=low_income,
                    reference_markup=reference_markup,
                )
                current_gva = _eu27_sourcing_impact(result_lookup[delta], component, "GVA")
                current_jobs = _eu27_sourcing_impact(result_lookup[delta], component, "Jobs")
                cost_change = (current_cost - base_cost) / 1_000_000.0
                gva_change = (current_gva - base_gva) / 1_000_000.0
                employment_change = current_jobs - base_jobs
                if delta == 0:
                    cost_change = 0.0
                    gva_change = 0.0
                    employment_change = 0.0
                rows.append({
                    "markup": markup,
                    "delta_pp": int(delta),
                    "component_group": group,
                    "component": component,
                    "baseline_expenditure_eur": base_cost,
                    "baseline_gva_eur": base_gva,
                    "baseline_employment": base_jobs,
                    "project_baseline_expenditure_eur": project_baseline_cost,
                    "project_baseline_gva_eur": project_baseline_gva,
                    "project_baseline_employment": project_baseline_jobs,
                    "cost_change_million_eur": float(cost_change),
                    "gva_change_million_eur": float(gva_change),
                    "employment_change": float(employment_change),
                    "employment_unit": "FTE" if annualize else "FTE-years",
                    "annualize": bool(annualize),
                })

    data = pd.DataFrame(rows)
    expected_rows = len(required_markups) * len(expected_delta_set) * len(EU27_COMPONENT_SOURCING_SPECS)
    if len(data) != expected_rows or not np.isfinite(
        data[["cost_change_million_eur", "gva_change_million_eur", "employment_change"]].to_numpy()
    ).all():
        raise ValueError("Saved scenario observations are incomplete or contain non-finite values.")
    baseline = data.loc[data["delta_pp"].eq(0)]
    if not (baseline[["cost_change_million_eur", "gva_change_million_eur", "employment_change"]] == 0.0).all().all():
        raise AssertionError("Every saved markup/component path must begin exactly at (0, 0).")
    return data


def build_eu27_aggregate_tradeoff_curve_data(
    data: pd.DataFrame,
    *,
    scenarios_delta_by_markup: dict,
    scenarios_result_by_markup: dict,
    low_income=None,
    reference_markup: float = REFERENCE_MARKUP,
) -> pd.DataFrame:
    """Aggregate trade-offs, sourcing the project row from full saved scenarios."""
    required = {
        "markup", "delta_pp", "component_group", "component",
        "baseline_expenditure_eur", "baseline_gva_eur", "baseline_employment",
        "cost_change_million_eur", "gva_change_million_eur", "employment_change",
        "annualize", "employment_unit",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise KeyError(f"Component trade-off data are missing aggregation columns: {missing}")
    expected_components = {row[1] for row in EU27_COMPONENT_SOURCING_SPECS}
    if set(data["component"].astype(str).unique()) != expected_components:
        raise ValueError("Aggregation requires exactly the ten saved component observations.")
    if set(data["component_group"].astype(str).unique()) != {"Turbine", "BoP"}:
        raise ValueError("Aggregation requires both Turbine and BoP component groups.")

    value_columns = (
        "baseline_expenditure_eur", "baseline_gva_eur", "baseline_employment",
        "cost_change_million_eur", "gva_change_million_eur", "employment_change",
    )
    rows = []
    group_order = ("Turbine", "BoP", "Total project")
    if not isinstance(scenarios_delta_by_markup, dict) or not isinstance(scenarios_result_by_markup, dict):
        raise TypeError("Full-project aggregate sourcing requires scenario mappings.")
    normalized_delta = {float(key): value for key, value in scenarios_delta_by_markup.items()}
    normalized_result = {float(key): value for key, value in scenarios_result_by_markup.items()}
    for (markup, delta), observations in data.groupby(
        ["markup", "delta_pp"], sort=True, dropna=False
    ):
        observations = observations.copy()
        if len(observations) != len(expected_components) or set(observations["component"].astype(str)) != expected_components:
            raise ValueError(f"Markup {markup:g}, delta {int(delta)} lacks the complete component set.")
        if markup not in normalized_delta or markup not in normalized_result:
            raise KeyError(f"Missing full-project scenario mapping for markup {markup:g}.")
        delta_mapping = normalized_delta[markup]
        result_mapping = normalized_result[markup]
        if not isinstance(delta_mapping, dict) or not isinstance(result_mapping, dict):
            raise TypeError(f"Saved markup {markup:g} must contain delta mappings.")
        if int(delta) not in delta_mapping or int(delta) not in result_mapping or 0 not in delta_mapping or 0 not in result_mapping:
            raise KeyError(f"Full-project scenario mappings for markup {markup:g} must contain delta={int(delta)} and delta=0.")
        baseline_delta = delta_mapping[0]
        baseline_result = result_mapping[0]
        baseline_cost = _eu27_sourcing_project_cost(
            baseline_delta,
            markup=markup,
            low_income=low_income,
            reference_markup=reference_markup,
        )
        baseline_gva = _eu27_sourcing_project_impact(baseline_result, "GVA")
        baseline_jobs = _eu27_sourcing_project_impact(baseline_result, "Jobs")
        current_cost = _eu27_sourcing_project_cost(
            delta_mapping[int(delta)],
            markup=markup,
            low_income=low_income,
            reference_markup=reference_markup,
        )
        current_gva = _eu27_sourcing_project_impact(result_mapping[int(delta)], "GVA")
        current_jobs = _eu27_sourcing_project_impact(result_mapping[int(delta)], "Jobs")
        project_values = {
            "baseline_expenditure_eur": float(baseline_cost),
            "baseline_gva_eur": float(baseline_gva),
            "baseline_employment": float(baseline_jobs),
            "cost_change_million_eur": 0.0 if int(delta) == 0 else float((current_cost - baseline_cost) / 1_000_000.0),
            "gva_change_million_eur": 0.0 if int(delta) == 0 else float((current_gva - baseline_gva) / 1_000_000.0),
            "employment_change": 0.0 if int(delta) == 0 else float(current_jobs - baseline_jobs),
        }
        for column in ("cost_change_million_eur", "gva_change_million_eur", "employment_change"):
            component_sum = float(pd.to_numeric(observations[column], errors="raise").sum())
            if not np.isclose(
                project_values[column], component_sum,
                rtol=1e-9, atol=1e-6,
            ):
                raise ValueError(
                    f"Full-project {column} does not reconcile to Turbine + BoP "
                    f"for markup={markup:g}, delta={int(delta)}."
                )
        for group in group_order:
            selected = (
                observations
                if group == "Total project"
                else observations.loc[observations["component_group"].astype(str).eq(group)]
            )
            if selected.empty:
                raise ValueError(f"No observations available for aggregate {group!r}.")
            annualize_values = selected["annualize"].astype(bool).unique()
            employment_units = selected["employment_unit"].astype(str).unique()
            if len(annualize_values) != 1 or len(employment_units) != 1:
                raise ValueError(f"Aggregate {group!r} has inconsistent employment metadata.")
            row = {
                "markup": float(markup),
                "delta_pp": int(delta),
                "component_group": group,
                "component": group,
                "annualize": bool(annualize_values[0]),
                "employment_unit": employment_units[0],
                "project_baseline_expenditure_eur": float(project_values["baseline_expenditure_eur"]),
                "project_baseline_gva_eur": float(project_values["baseline_gva_eur"]),
                "project_baseline_employment": float(project_values["baseline_employment"]),
            }
            for column in value_columns:
                values = pd.to_numeric(selected[column], errors="raise")
                if not np.isfinite(values.to_numpy(dtype=float)).all():
                    raise ValueError(f"Aggregate {group!r} contains non-finite {column} values.")
                row[column] = (
                    float(project_values[column])
                    if group == "Total project"
                    else float(values.sum())
                )
            rows.append(row)

    aggregate = pd.DataFrame(rows)
    expected_rows = data[["markup", "delta_pp"]].drop_duplicates().shape[0] * len(group_order)
    if len(aggregate) != expected_rows:
        raise ValueError("Aggregate trade-off observations are incomplete.")
    baseline = aggregate.loc[aggregate["delta_pp"].eq(0)]
    if not (baseline[["cost_change_million_eur", "gva_change_million_eur", "employment_change"]] == 0.0).all().all():
        raise AssertionError("Every aggregate path must begin exactly at (0, 0).")
    return aggregate


def _normalize_eu27_tradeoff_data(
    data: pd.DataFrame,
    *,
    normalization_scope: str,
) -> pd.DataFrame:
    """Return a presentation-only copy expressed as category/project percentages."""
    required = {
        "markup",
        "delta_pp",
        "component",
        "baseline_expenditure_eur",
        "baseline_gva_eur",
        "baseline_employment",
        "cost_change_million_eur",
        "gva_change_million_eur",
        "employment_change",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise KeyError(f"Trade-off data are missing normalization columns: {missing}")

    scope = str(normalization_scope).strip().lower()
    if scope not in {"category", "project"}:
        raise ValueError("normalization_scope must be 'category' or 'project'.")
    if scope == "project":
        project_baseline_columns = [
            "project_baseline_expenditure_eur",
            "project_baseline_gva_eur",
            "project_baseline_employment",
        ]
        missing_project = sorted(set(project_baseline_columns).difference(data.columns))
        if missing_project:
            raise KeyError(
                "Project-normalized trade-off data are missing full-project baseline columns: "
                f"{missing_project}"
            )

    out = data.copy()
    baseline_rows = out.loc[
        np.isclose(out["delta_pp"].astype(float), 0.0, rtol=0.0, atol=1e-12)
    ].copy()
    if baseline_rows.empty:
        raise ValueError("Trade-off data must contain delta=0 baseline rows for normalization.")

    baseline_columns = ["baseline_expenditure_eur", "baseline_gva_eur", "baseline_employment"]
    if scope == "category":
        baseline_rows["_normalization_key"] = list(
            zip(baseline_rows["markup"].astype(float), baseline_rows["component_group"].astype(str))
        )
        grouped = baseline_rows.groupby("_normalization_key", sort=False)[baseline_columns].sum()
        baseline_values = {
            key: tuple(float(value) for value in row)
            for key, row in grouped.iterrows()
        }
        row_keys = list(zip(out["markup"].astype(float), out["component_group"].astype(str)))
    else:
        project_rows = baseline_rows[["markup"] + project_baseline_columns].copy()
        grouped = project_rows.groupby(project_rows["markup"].astype(float), sort=False)
        baseline_values = {}
        for markup, rows in grouped:
            if any(rows[column].nunique(dropna=False) != 1 for column in project_baseline_columns):
                raise ValueError(f"Full-project baseline must be unique for markup {markup:g}.")
            baseline_values[float(markup)] = tuple(
                float(rows[column].iloc[0]) for column in project_baseline_columns
            )
        row_keys = out["markup"].astype(float).tolist()

    denominators = [baseline_values.get(key) for key in row_keys]
    if any(values is None for values in denominators):
        missing_keys = sorted({key for key, values in zip(row_keys, denominators) if values is None})
        raise ValueError(f"Missing delta=0 baselines for normalization: {missing_keys}")
    denominator_array = np.asarray(denominators, dtype=float)
    if not np.isfinite(denominator_array).all():
        raise ValueError("Trade-off baseline values for normalization must be finite.")
    if (denominator_array[:, 0] <= 0.0).any():
        raise ValueError("Component baseline expenditure must be positive for normalization.")
    if (denominator_array[:, 1:] == 0.0).any():
        raise ValueError("Component baseline GVA and employment must be non-zero for normalization.")

    out["cost_change_million_eur"] = (
        pd.to_numeric(out["cost_change_million_eur"], errors="raise")
        / (denominator_array[:, 0] / 1_000_000.0)
        * 100.0
    )
    out["gva_change_million_eur"] = (
        pd.to_numeric(out["gva_change_million_eur"], errors="raise")
        / (denominator_array[:, 1] / 1_000_000.0)
        * 100.0
    )
    out["employment_change"] = (
        pd.to_numeric(out["employment_change"], errors="raise")
        / denominator_array[:, 2]
        * 100.0
    )
    if not np.isfinite(
        out[["cost_change_million_eur", "gva_change_million_eur", "employment_change"]]
        .to_numpy(dtype=float)
    ).all():
        raise ValueError("Normalized trade-off coordinates contain non-finite values.")
    return out


def eu27_component_tradeoff_axis_limits(
    data: pd.DataFrame,
    *,
    indicator: str,
    baseline_markup: float,
    padding: float = 0.04,
    normalization_scope: str | None = None,
    required_series: tuple[str, ...] = ("Turbine", "BoP"),
    component_group: str | None = None,
) -> dict[str, tuple[float, float]]:
    """Calculate limits, optionally scoping X to one component group.

    Y limits are always calculated from all required series so that the
    Turbine and BoP component plots remain comparable within an indicator.
    """
    indicator_key = str(indicator).strip().upper()
    if indicator_key not in {"GVA", "JOBS"}:
        raise ValueError("indicator must be 'GVA' or 'Jobs'.")
    if not np.isfinite(float(padding)) or not 0.0 <= float(padding) <= 0.1:
        raise ValueError("padding must be between 0 and 0.1 of the observed range.")
    if not {"markup", "component_group", "cost_change_million_eur", "gva_change_million_eur", "employment_change"}.issubset(data.columns):
        raise KeyError("Trade-off data are missing columns needed to calculate axis limits.")
    scope = _eu27_tradeoff_normalization_scope(normalization_scope)
    if scope != "absolute":
        data = _normalize_eu27_tradeoff_data(
            data,
            normalization_scope=scope,
        )
    markups = sorted(pd.to_numeric(data["markup"], errors="raise").unique().tolist())
    if len(markups) != 3 or not np.isclose(markups[1], float(baseline_markup), rtol=0.0, atol=1e-12):
        raise ValueError("Axis limits require low, central, and high markup observations.")
    expected_series = {str(value) for value in required_series}
    if set(data["component_group"].astype(str).unique()) != expected_series:
        raise ValueError(f"Axis limits require exactly these series: {sorted(expected_series)}.")
    if component_group is None:
        x_data = data
    else:
        group = str(component_group).strip()
        if group not in expected_series:
            raise ValueError(
                f"component_group must be one of the required series: {sorted(expected_series)}."
            )
        x_data = data.loc[data["component_group"].astype(str).eq(group)]
        if x_data.empty:
            raise ValueError(f"No trade-off observations found for component group {group!r}.")

    central = data.loc[
        np.isclose(data["markup"].astype(float), float(baseline_markup), rtol=0.0, atol=1e-12)
    ]
    y_col = "gva_change_million_eur" if indicator_key == "GVA" else "employment_change"
    values = central[["cost_change_million_eur", y_col]].to_numpy(dtype=float)
    x_observations = x_data["cost_change_million_eur"].to_numpy(dtype=float)
    y_observations = data[y_col].to_numpy(dtype=float)
    if (
        values.size == 0
        or x_observations.size == 0
        or not np.isfinite(values).all()
        or not np.isfinite(x_observations).all()
        or not np.isfinite(y_observations).all()
    ):
        raise ValueError("Axis observations are empty or non-finite.")

    def _limits(frame, column):
        low = float(frame[column].min())
        high = float(frame[column].max())
        span = high - low
        pad = span * float(padding) if span > 0 else max(abs(low), abs(high), 1.0) * float(padding)
        return (low - pad, high + pad)

    limits = {
        "xlim": _limits(x_data, "cost_change_million_eur"),
        "ylim": _limits(data, y_col),
    }
    # Ensure all plotted low/high paths remain inside the padded limits.
    for key, column in (("xlim", "cost_change_million_eur"), ("ylim", y_col)):
        all_values = (
            x_data[column].to_numpy(dtype=float)
            if key == "xlim"
            else data[column].to_numpy(dtype=float)
        )
        low, high = limits[key]
        margin = (high - low) * 0.002
        if all_values.min() < low:
            low = float(all_values.min() - margin)
        if all_values.max() > high:
            high = float(all_values.max() + margin)
        limits[key] = (low, high)
    return limits


def plot_eu27_component_cost_impact_curves(
    data: pd.DataFrame,
    *,
    component_group: str,
    indicator: str,
    filename: str,
    baseline_markup: float,
    axis_limits: dict[str, tuple[float, float]],
    normalization_scope: str | None = None,
    annualize: bool | None = None,
    save: bool | None = None,
    close: bool | None = None,
):
    """Plot saved component procurement changes against net EU27 effects."""
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D

    scope = _eu27_tradeoff_normalization_scope(normalization_scope)
    if scope != "absolute":
        data = _normalize_eu27_tradeoff_data(
            data,
            normalization_scope=scope,
        )

    group = str(component_group).strip()
    if group not in {"Turbine", "BoP"}:
        raise ValueError("component_group must be 'Turbine' or 'BoP'.")
    indicator_key = str(indicator).strip().upper()
    if indicator_key not in {"GVA", "JOBS"}:
        raise ValueError("indicator must be 'GVA' or 'Jobs'.")
    components = [row[1] for row in EU27_COMPONENT_SOURCING_SPECS if row[0] == group]
    subset = data.loc[data["component_group"].astype(str).eq(group)].copy()
    if set(subset["component"].astype(str).unique()) != set(components):
        raise ValueError(f"{group} plot must contain exactly these five components: {components}.")
    component_colors = EU27_COMPONENT_SOURCING_COLORS.copy()
    band_alpha = 0.10
    if group == "BoP":
        component_colors.update({
            "Array": "#C4A000",
            "Export": "#6543A1",
            "Foundation": "#C32B83",
            "OSS": "#92512D",
            "Other": "#41464D",
        })
        # Slightly strengthen only the BoP envelope fills; central curves,
        # markers, and the existing low/high envelope geometry stay unchanged.
        band_alpha = 0.16
    if annualize is None:
        annualize = bool(subset["annualize"].astype(bool).iloc[0])
    if subset["annualize"].astype(bool).nunique() != 1 or bool(subset["annualize"].iloc[0]) != bool(annualize):
        raise ValueError("Requested annualization does not match the saved curve data.")
    if indicator_key == "JOBS" and bool(annualize):
        raise ValueError("These manuscript employment curves report cumulative FTE-years.")
    markups = sorted(pd.to_numeric(data["markup"], errors="raise").unique().tolist())
    if (
        len(markups) != 3
        or not np.isclose(markups[1], float(baseline_markup), rtol=0.0, atol=1e-12)
    ):
        raise ValueError(
            "Curves must contain low, configured central, and high markup cases, "
            f"with central markup {baseline_markup:g}."
        )
    low_markup, _central_markup, high_markup = markups
    baseline_costs = {}
    for component in (row[1] for row in EU27_COMPONENT_SOURCING_SPECS):
        component_rows = data.loc[
            data["component"].astype(str).eq(component)
            & np.isclose(data["markup"].astype(float), float(baseline_markup), rtol=0.0, atol=1e-12)
        ]
        if component_rows.empty or component_rows["baseline_expenditure_eur"].nunique(dropna=False) != 1:
            raise ValueError(f"{component}: expected one positive baseline component expenditure.")
        cost = float(component_rows["baseline_expenditure_eur"].iloc[0])
        if not np.isfinite(cost) or cost <= 0.0:
            raise ValueError(f"{component}: invalid baseline component expenditure.")
        baseline_costs[component] = cost

    x_col = "cost_change_million_eur"
    y_col = "gva_change_million_eur" if indicator_key == "GVA" else "employment_change"
    if scope != "absolute":
        x_label = r"\boldmath$\Delta\,\mathrm{Procurement\ cost}\;[\%\ \mathrm{var}]$"
        if indicator_key == "GVA":
            y_label = r"\boldmath$\Delta\,\mathrm{EU27\ GVA}\;[\%\ \mathrm{var}]$"
        else:
            y_label = r"\boldmath$\Delta\,\mathrm{EU27\ employment}\;[\%\ \mathrm{var}]$"
    else:
        x_label = r"\boldmath$\Delta\,\mathrm{Procurement\ cost}\;[\mathrm{million\ EUR}]$"
        if indicator_key == "GVA":
            y_label = r"\boldmath$\Delta\,\mathrm{EU27\ GVA}\;[\mathrm{million\ EUR}]$"
        else:
            y_label = r"\boldmath$\Delta\,\mathrm{EU27\ employment}\;[\mathrm{FTE\text{-}years}]$"

    if not isinstance(axis_limits, dict) or set(axis_limits) != {"xlim", "ylim"}:
        raise ValueError("axis_limits must contain the shared 'xlim' and 'ylim' tuples.")
    validated_axis_limits = {}
    for key in ("xlim", "ylim"):
        bounds = tuple(float(value) for value in axis_limits[key])
        if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError(f"Invalid shared {key}: {axis_limits[key]!r}.")
        validated_axis_limits[key] = bounds

    set_house_style()
    fig, ax = plt.subplots(figsize=(4.35, 3.25), dpi=150)
    plotted_values = []
    curves = {}
    component_costs = {}
    draw_components = list(components)
    if group == "Turbine" and "Blades" in draw_components:
        draw_components = [component for component in draw_components if component != "Blades"] + ["Blades"]
    for component in draw_components:
        component_rows = subset.loc[subset["component"].astype(str).eq(component)]
        curve = component_rows.loc[
            np.isclose(component_rows["markup"].astype(float), float(baseline_markup), rtol=0.0, atol=1e-12)
        ].sort_values("delta_pp")
        low_curve = component_rows.loc[
            np.isclose(component_rows["markup"].astype(float), low_markup, rtol=0.0, atol=1e-12)
        ].sort_values("delta_pp")
        high_curve = component_rows.loc[
            np.isclose(component_rows["markup"].astype(float), high_markup, rtol=0.0, atol=1e-12)
        ].sort_values("delta_pp")
        deltas = curve["delta_pp"].astype(int).tolist()
        if deltas != list(range(-100, 101, 5)):
            raise ValueError(f"{component}: expected the 41 saved delta observations from -100 to +100.")
        expected_deltas = list(range(-100, 101, 5))
        if (
            low_curve["delta_pp"].astype(int).tolist() != expected_deltas
            or high_curve["delta_pp"].astype(int).tolist() != expected_deltas
        ):
            raise ValueError(f"{component}: low/high markup cases must contain the same 41 deltas.")
        if curve["baseline_expenditure_eur"].nunique(dropna=False) != 1:
            raise ValueError(f"{component}: baseline expenditure must be constant along its curve.")
        baseline_cost = float(curve["baseline_expenditure_eur"].iloc[0])
        if not np.isfinite(baseline_cost) or baseline_cost <= 0.0:
            raise ValueError(f"{component}: invalid baseline component expenditure.")
        component_costs[component] = baseline_cost

        x_values = curve[x_col].to_numpy(dtype=float)
        y_values = curve[y_col].to_numpy(dtype=float)
        values = np.column_stack([x_values, y_values])
        if not np.isfinite(values).all():
            raise ValueError(f"{component}: baseline curve contains non-finite observations.")
        low_values = low_curve[[x_col, y_col]].to_numpy(dtype=float)
        high_values = high_curve[[x_col, y_col]].to_numpy(dtype=float)
        if not np.isfinite(low_values).all() or not np.isfinite(high_values).all():
            raise ValueError(f"{component}: low/high uncertainty coordinates are non-finite.")
        zero_idx = deltas.index(0)
        if not np.allclose(values[zero_idx], [0.0, 0.0], rtol=0.0, atol=1e-10):
            raise AssertionError(f"{component}: delta=0 must be exactly the origin.")

        plotted_values.append(values)
        curves[component] = (deltas, x_values, y_values)
        band_vertices = np.stack(
            (low_values[:-1], low_values[1:], high_values[1:], high_values[:-1]),
            axis=1,
        )
        band = PolyCollection(
            band_vertices,
            facecolors=[component_colors[component]],
            edgecolors="none",
            alpha=band_alpha,
            label="_nolegend_",
            zorder=2,
        )
        ax.add_collection(band, autolim=False)
        ax.plot(
            x_values, y_values,
            color=component_colors[component],
            linestyle="-", linewidth=1.15,
            zorder=4 if group == "Turbine" and component == "Blades" else 3,
        )

    plotted_values = np.vstack(plotted_values)
    cost_min, cost_max = min(baseline_costs.values()), max(baseline_costs.values())
    if np.isclose(cost_min, cost_max, rtol=0.0, atol=1e-12):
        marker_area = {component: 28.0 for component in components}
    else:
        marker_area = {
            component: 12.0 + 34.0 * (baseline_costs[component] - cost_min) / (cost_max - cost_min)
            for component in components
        }

    for component in draw_components:
        deltas, x_values, y_values = curves[component]
        color = component_colors[component]
        zero_idx = deltas.index(0)
        baseline_xy = np.array([x_values[zero_idx], y_values[zero_idx]])
        for delta, marker, face in ((-100, "o", "white"), (100, "^", color)):
            endpoint_idx = deltas.index(delta)
            endpoint_xy = np.array([x_values[endpoint_idx], y_values[endpoint_idx]])
            # A capped endpoint may genuinely coincide with the baseline.
            coincides_with_baseline = np.allclose(endpoint_xy, baseline_xy, rtol=0.0, atol=1e-10)
            blades_reshoring_endpoint = (
                group == "Turbine" and component == "Blades" and delta == 100
            )
            if coincides_with_baseline and not blades_reshoring_endpoint:
                continue
            ax.scatter(
                endpoint_xy[0], endpoint_xy[1], s=marker_area[component], marker=marker,
                facecolors=face, edgecolors=color, linewidths=0.85,
                zorder=8 if blades_reshoring_endpoint else 5,
            )
    # Keep the shared origin marker visible, then redraw the Blades +100
    # endpoint on top when its feasible path terminates at the baseline.
    for component in draw_components:
        if group == "Turbine" and component == "Blades":
            continue
        deltas, x_values, y_values = curves[component]
        zero_idx = deltas.index(0)
        ax.scatter(
            x_values[zero_idx], y_values[zero_idx], s=10.0, marker="o",
            facecolors=component_colors[component],
            edgecolors=component_colors[component], linewidths=0.5, zorder=6,
        )
    blades_baseline = None
    if group == "Turbine" and "Blades" in curves:
        deltas, x_values, y_values = curves["Blades"]
        zero_idx = deltas.index(0)
        blades_baseline = (x_values[zero_idx], y_values[zero_idx])
        color = EU27_COMPONENT_SOURCING_COLORS["Blades"]
        ax.scatter(*blades_baseline, s=10.0, marker="o", facecolors=color,
                   edgecolors=color, linewidths=0.5, zorder=7)
        positive_idx = deltas.index(100)
        if np.allclose(
            [x_values[positive_idx], y_values[positive_idx]],
            blades_baseline, rtol=0.0, atol=1e-10,
        ):
            ax.scatter(x_values[positive_idx], y_values[positive_idx],
                       s=marker_area["Blades"], marker="^", facecolors=color,
                       edgecolors=color, linewidths=0.85, zorder=9)

    ax.axhline(0.0, color="#777777", linewidth=0.6, alpha=0.6, zorder=1)
    ax.axvline(0.0, color="#777777", linewidth=0.6, alpha=0.6, zorder=1)
    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel(y_label, fontsize=8.5)
    apply_house_style(ax, grid=True)
    ax.grid(False, axis="y", which="minor")
    if indicator_key == "GVA":
        ax.grid(True, axis="y", which="major", linewidth=0.4, alpha=0.3)
        ax.tick_params(axis="y", which="minor", length=0)
    ax.tick_params(axis="both", labelsize=7.5)

    def _format_tick(value, _position):
        value = float(value)
        if abs(value) < 1e-10:
            return "0"
        if abs(value) >= 100.0:
            return f"{value:,.0f}"
        return f"{value:,.1f}".rstrip("0").rstrip(".")

    ax.xaxis.set_major_formatter(plt.FuncFormatter(_format_tick))
    def _format_y_tick(value, _position):
        value = float(value)
        if abs(value) >= 1000.0:
            scaled = value / 1000.0
            return f"{scaled:.0f}k" if abs(scaled) >= 10 else f"{scaled:.1f}k".rstrip("0").rstrip(".")
        return f"{value:.0f}"

    ax.yaxis.set_major_formatter(plt.FuncFormatter(_format_y_tick))

    component_handles = [
        Line2D(
            [0], [0], color=component_colors[component],
            linewidth=1.2,
            label=next(
                display_label
                for _group, source_component, display_label in EU27_COMPONENT_SOURCING_SPECS
                if source_component == component
            ),
        )
        for component in components
    ]
    delta_handles = [
        Line2D(
            [0], [0], linestyle="none", marker=marker,
            markerfacecolor=face, markeredgecolor="#444444", markeredgewidth=0.7,
            markersize=4.2, label=label,
        )
        for marker, face, label in (
            ("o", "white", r"$\delta=-100$"),
            ("^", "#444444", r"$\delta=+100$"),
        )
    ]
    ax.legend(
        handles=component_handles + delta_handles,
        title=tex_bold("Component"),
        title_fontproperties={"family": "serif", "size": 6.6},
        loc="lower right", bbox_to_anchor=(0.98, 0.02), bbox_transform=ax.transAxes,
        ncol=1, frameon=True, fancybox=False, framealpha=0.94,
        facecolor="white", edgecolor="#B8B8B8",
        prop={"family": "serif", "size": 6.6},
        handlelength=1.0, handletextpad=0.45, columnspacing=0.6,
        labelspacing=0.22, borderpad=0.45, borderaxespad=0.0,
    )

    ax.set_xlim(*validated_axis_limits["xlim"])
    ax.set_ylim(*validated_axis_limits["ylim"])

    observed_x = subset[x_col].to_numpy(dtype=float)
    observed_x_min = float(observed_x.min())
    observed_x_max = float(observed_x.max())
    fig.subplots_adjust(left=0.145, right=0.99, bottom=0.13, top=0.985)
    print(
        f"[Figure 3.5] {group}/{indicator_key}: "
        f"observed X=[{observed_x_min:.6g}, {observed_x_max:.6g}], "
        f"calculated xlim={validated_axis_limits['xlim']!r}, "
        f"ax.get_xlim()={ax.get_xlim()!r}"
    )
    path = save_and_close(fig=fig, filename=filename, save=save, close=close, exact_size=True)
    return fig, ax, path
