
"""Baseline impact tables, ratios, and capture figures."""

from __future__ import annotations

from pathlib import Path
from typing import List

import re
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors

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

    import matplotlib.pyplot as plt
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


# Default industry color map (fallback if industrial.COLOR_IND cannot be imported)
COLOR_IND_DEFAULT = {
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

INDUSTRY_LABEL_FALLBACKS = {
    "D35": "Electricity, gas, steam and air conditioning supply",
    "CPA_D35": "Electricity, gas, steam and air conditioning supply",
}


def _industry_label(code, ind_labels: dict | None = None) -> str:
    """Resolve an industry label across CPA/bare and hyphen/underscore code variants."""
    code_s = str(code).strip()
    if not ind_labels:
        return INDUSTRY_LABEL_FALLBACKS.get(code_s, code_s)

    labels = {str(k).strip(): v for k, v in ind_labels.items()}
    variants = [
        code_s,
        code_s[4:] if code_s.startswith("CPA_") else f"CPA_{code_s}",
        code_s.replace("-", "_"),
        code_s.replace("_", "-"),
        code_s.replace("T", "-"),
        code_s.replace("-", "T"),
    ]

    for key in variants:
        if key in labels and pd.notna(labels[key]):
            label = str(labels[key]).strip()
            if label and label not in {code_s, f"CPA_{code_s}"}:
                return label

    return INDUSTRY_LABEL_FALLBACKS.get(code_s, code_s)






from modules.country import (
    EU27,
    EUROPE_CODES,
    COLOR_MAP,
    CODE_TO_NAME,
    canonical_name,
)


def _is_aggregate_component_key(key) -> bool:
    """True for total/ALL keys that should not be mixed with detailed rows."""
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
    component also has detailed children at that level. If they are the only
    key for a component, they are the component-level observation and must stay.
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


# ---------- BY ORIGIN -> component dict (keep Origin, Country=Deployment) ----------
def by_origin_to_component_dict(by_origin_lvl: dict) -> dict:
    """
    Returns {Component: df_long} where:
      - Country == Deployment (this is what your table uses as columns)
      - Origin == original Country (origin country/region) for EU27 filtering
      - NOT collapsed yet (still has one row per (Deployment, Origin))
    """
    comp_map = {}
    skip_self = _self_aggregate_keys_to_skip(by_origin_lvl.keys())
    for (comp, subc), df in by_origin_lvl.items():
        # Keep ("Total", "ALL") here: component-column tables use it as the
        # explicit Total component. Detailed/export paths filter it separately.
        if (comp, subc) != ("Total", "ALL") and (
            _is_aggregate_component_key((comp, subc)) or (comp, subc) in skip_self
        ):
            continue
        comp_map.setdefault(comp, []).append(df)

    out = {}
    for comp, dfs in comp_map.items():
        big = pd.concat(dfs, ignore_index=True).copy()

        # keep origin in a separate column
        big["Origin"] = big["Country"]
        # table columns must be deployments
        big["Country"] = big["Deployment"]

        num_cols = big.select_dtypes(include=[np.number]).columns.tolist()
        keep_cols = ["Country", "Origin"] + num_cols

        out[comp] = big[keep_cols]
    return out


_SCENARIO_RESULTS_COLUMNS = [
    "scenario", "component_group", "subcomponent", "deployment",
    "origin_country", "origin_region", "indicator", "effect_type",
    "value", "unit", "connected_power_mw", "delta_pp", "markup_mu",
]
_EXCEL_MAX_ROWS = 1_048_576


def _scenario_field(value, name, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _scenario_markup_from_delta_mapping(scenarios_delta: dict) -> float:
    """Return the single markup recorded in a saved delta mapping."""
    values = []
    for scenario in scenarios_delta.values():
        for stats_name in ("stats_turbine", "stats_farm"):
            stats = _scenario_field(scenario, stats_name, {}) or {}
            value = _scenario_field(stats, "eu_spend_reduction_pct")
            if value is not None and np.isfinite(float(value)):
                values.append(float(value))
    if not values or not np.allclose(values, values[0], rtol=0.0, atol=1e-12):
        raise ValueError("Saved scenario mapping lacks one consistent, finite markup value.")
    return float(values[0])


def _scenario_detailed_result_frame(result, *, level: int = 0) -> pd.DataFrame:
    """Return the saved terminal result frame, with a legacy-map fallback."""
    detailed = _scenario_field(result, "_detailed")
    if isinstance(detailed, pd.DataFrame) and not detailed.empty:
        return detailed.copy()

    result_sub = _scenario_field(result, "results_by_origin_sub_lvl")
    if not isinstance(result_sub, dict):
        raise ValueError("Saved scenario result lacks _detailed and results_by_origin_sub_lvl.")
    level_map = result_sub.get(level)
    if not isinstance(level_map, dict):
        raise ValueError(f"Saved scenario result lacks results_by_origin_sub_lvl[{level}].")

    frames = []
    for key, frame in level_map.items():
        if key == ("Total", "ALL") or not isinstance(key, tuple) or len(key) < 2:
            continue
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        component, subcomponent = str(key[0]), str(key[1])
        out = frame.copy()
        out["component_group_L0"] = component
        out["sub_component_L0"] = subcomponent
        out["component_group_L1"] = component
        out["sub_component_L1"] = subcomponent
        out["component_group_L2"] = component
        out["sub_component_L2"] = subcomponent
        frames.append(out)
    if not frames:
        raise ValueError("Saved scenario result contains no detailed subcomponent frames.")
    return pd.concat(frames, ignore_index=True, sort=False)


def _normalize_scenario_origin(value) -> str:
    value = str(value).strip().upper()
    return {"UK": "GB", "ROW": "ROW", "X01": "ROW", "RO W": "ROW"}.get(value, value)


def _scenario_origin_region(country: str, EU27) -> str:
    from modules.country import EUROPE_CODES

    country = _normalize_scenario_origin(country)
    eu27 = {str(value).strip().upper().replace("UK", "GB") for value in (EU27 or ())}
    if country in {"ROW", "X01"}:
        return "RoW"
    if country in eu27:
        return "EU27"
    return "RoE" if country in set(EUROPE_CODES) else "RoW"


def _scenario_results_from_saved_detail(
    *,
    scenarios_delta: dict,
    scenarios_result: dict,
    EU27,
    connected_power_mw: float,
    annualize: bool,
    lifetime_years: float,
    year: int,
    sector: str,
    markup_mu: float,
) -> tuple[pd.DataFrame, dict]:
    """Build the long Results export from saved detailed MRIO result frames."""
    if not isinstance(scenarios_delta, dict) or not isinstance(scenarios_result, dict):
        raise TypeError("Saved scenario deltas and results must both be delta-to-result mappings.")
    if set(scenarios_delta) != set(scenarios_result):
        missing_results = sorted(set(scenarios_delta).difference(scenarios_result))
        extra_results = sorted(set(scenarios_result).difference(scenarios_delta))
        raise ValueError(
            "Scenario delta/result keys do not match: "
            f"missing_results={missing_results}, extra_results={extra_results}."
        )
    if not np.isfinite(float(connected_power_mw)) or float(connected_power_mw) <= 0.0:
        raise ValueError("connected_power_mw must be a positive finite value from the current configuration.")

    markup_saved = _scenario_markup_from_delta_mapping(scenarios_delta)
    if not np.isclose(float(markup_saved), float(markup_mu), rtol=0.0, atol=1e-12):
        raise ValueError(
            f"Requested markup_mu={markup_mu:g} does not match saved markup {markup_saved:g}."
        )

    expected_columns = {
        "Deployment", "Country", "component_group_L1", "sub_component_L1",
        "GVA_Direct", "GVA_Indirect", "GVA_Total",
        "Jobs_Direct", "Jobs_Indirect", "Jobs_Total",
    }
    output_frames = []
    diagnostics = {
        "max_split_gva_million_eur": 0.0,
        "max_split_jobs_fte_years": 0.0,
        "n_scenarios": 0,
        "n_rows": 0,
    }

    for delta, result in sorted(scenarios_result.items(), key=lambda item: float(item[0])):
        detailed = _scenario_detailed_result_frame(result)
        missing = expected_columns.difference(detailed.columns)
        if missing:
            raise KeyError(f"Saved detailed result for delta={delta} is missing columns: {sorted(missing)}")

        detailed = detailed.copy()
        detailed["Deployment"] = detailed["Deployment"].astype(str).str.strip()
        deployment_values = set(detailed["Deployment"])
        if deployment_values != {"EU27_AVG"}:
            raise ValueError(
                f"Results for delta={delta} contain deployments {sorted(deployment_values)}; "
                "the supplementary export requires only EU27_AVG."
            )
        if year is not None and "Year" in detailed.columns:
            detailed = detailed.loc[detailed["Year"] == year].copy()
        if sector is not None and "Sector" in detailed.columns:
            detailed = detailed.loc[detailed["Sector"].astype(str).eq(str(sector))].copy()
        if detailed.empty:
            raise ValueError(f"No detailed results remain for delta={delta}, year={year}, sector={sector!r}.")

        group_cols = ["Deployment", "component_group_L1", "sub_component_L1", "Country"]
        metric_cols = [
            "GVA_Direct", "GVA_Indirect", "GVA_Total",
            "Jobs_Direct", "Jobs_Indirect", "Jobs_Total",
        ]
        grouped = detailed.groupby(group_cols, dropna=False)[metric_cols].sum().reset_index()
        diagnostics["n_rows"] += int(len(grouped) * 6)
        gva_split = (grouped["GVA_Direct"] + grouped["GVA_Indirect"] - grouped["GVA_Total"]).abs().max()
        jobs_split = (grouped["Jobs_Direct"] + grouped["Jobs_Indirect"] - grouped["Jobs_Total"]).abs().max()
        diagnostics["max_split_gva_million_eur"] = max(
            diagnostics["max_split_gva_million_eur"], float(gva_split) / 1_000_000.0
        )
        diagnostics["max_split_jobs_fte_years"] = max(
            diagnostics["max_split_jobs_fte_years"], float(jobs_split)
        )

        scenario_delta = float(delta)
        if np.isclose(scenario_delta, 0.0):
            scenario_label = "base" if np.isclose(float(markup_mu), 0.3) else f"base (mu={markup_mu:g})"
        elif np.isclose(scenario_delta, -100.0):
            scenario_label = "full offshoring"
        elif np.isclose(scenario_delta, 100.0):
            scenario_label = "full resilience"
        else:
            scenario_label = f"delta={scenario_delta:+g} pp"
        if not np.isclose(float(markup_mu), 0.3):
            scenario_label = f"{scenario_label} (mu={markup_mu:g})"

        base = grouped[["Deployment", "component_group_L1", "sub_component_L1", "Country"]].copy()
        base = base.rename(columns={
            "component_group_L1": "component_group",
            "sub_component_L1": "subcomponent",
            "Country": "origin_country",
        })
        base["scenario"] = scenario_label
        base["deployment"] = "EU27_AVG"
        base["origin_country"] = base["origin_country"].map(_normalize_scenario_origin)
        base["origin_region"] = base["origin_country"].map(
            lambda country: _scenario_origin_region(country, EU27)
        )
        base["connected_power_mw"] = float(connected_power_mw)
        base["delta_pp"] = scenario_delta
        base["markup_mu"] = float(markup_mu)
        base = base.drop(columns=["Deployment"])

        for indicator, prefix, unit, scale in (
            (
                "GVA", "GVA",
                "million EUR per year" if annualize else f"million EUR, cumulative {lifetime_years:g}-year",
                1_000_000.0,
            ),
            (
                "Employment", "Jobs",
                "FTE per year" if annualize else f"FTE-years, cumulative {lifetime_years:g}-year",
                1.0,
            ),
        ):
            values = grouped[[f"{prefix}_Direct", f"{prefix}_Indirect", f"{prefix}_Total"]].copy()
            values.columns = ["Direct", "Indirect", "Total"]
            long = values.melt(var_name="effect_type", value_name="value")
            long["value"] = pd.to_numeric(long["value"], errors="raise") / scale
            repeated_base = pd.concat([base] * 3, ignore_index=True)
            repeated_base["indicator"] = indicator
            repeated_base["unit"] = unit
            output_frames.append(pd.concat([repeated_base.reset_index(drop=True), long], axis=1))
        diagnostics["n_scenarios"] += 1

    out = pd.concat(output_frames, ignore_index=True)[_SCENARIO_RESULTS_COLUMNS]
    if out.empty:
        raise ValueError("No Results observations were built from the saved detailed scenario results.")
    out["component_group"] = pd.Categorical(
        out["component_group"],
        categories=["Devex", "Turbine", "BoP", "Installation", "O&M"],
        ordered=True,
    )
    out = out.sort_values(
        ["delta_pp", "component_group", "subcomponent", "origin_country", "indicator", "effect_type"],
        kind="mergesort",
    ).reset_index(drop=True)
    out["component_group"] = out["component_group"].astype(str)
    key = [
        "scenario", "component_group", "subcomponent", "deployment", "origin_country",
        "indicator", "effect_type", "delta_pp", "markup_mu",
    ]
    if out.duplicated(key).any():
        raise ValueError("Results contains duplicate scenario/component/origin/indicator/effect keys.")
    if not np.isfinite(pd.to_numeric(out["value"], errors="coerce")).all():
        raise ValueError("Results contains non-finite numeric values.")
    return out, diagnostics


def _validate_saved_scenario_result_reconciliation(
    *,
    scenarios_result: dict,
    year: int,
    sector: str,
) -> dict:
    """Reconcile the saved detailed result with saved component aggregates."""
    max_global_gva = 0.0
    max_global_jobs = 0.0
    max_component_gva = 0.0
    max_component_jobs = 0.0
    max_subcomponent_gva = 0.0
    max_subcomponent_jobs = 0.0

    for delta, result in scenarios_result.items():
        detailed = _scenario_detailed_result_frame(result)
        if year is not None and "Year" in detailed.columns:
            detailed = detailed.loc[detailed["Year"] == year]
        if sector is not None and "Sector" in detailed.columns:
            detailed = detailed.loc[detailed["Sector"].astype(str).eq(str(sector))]
        detailed = detailed.copy()
        detailed["Country"] = detailed["Country"].map(_normalize_scenario_origin)

        ref_levels = _scenario_field(result, "results_by_origin_lvl") or {}
        ref_sublevels = _scenario_field(result, "results_by_origin_sub_lvl") or {}
        if 1 not in ref_levels or 1 not in ref_sublevels:
            raise ValueError(f"Saved result for delta={delta} lacks level-1 results for reconciliation.")

        def _compare(reference, actual, metric_cols):
            if reference is None:
                raise ValueError(f"Missing reference frame for delta={delta}.")
            ref = reference.copy()
            ref["Country"] = ref["Country"].map(_normalize_scenario_origin)
            if year is not None and "Year" in ref.columns:
                ref = ref.loc[ref["Year"] == year]
            if sector is not None and "Sector" in ref.columns:
                ref = ref.loc[ref["Sector"].astype(str).eq(str(sector))]
            ref_values = ref.groupby("Country", dropna=False)[metric_cols].sum()
            actual_values = actual.groupby("Country", dropna=False)[metric_cols].sum()
            joined = ref_values.join(actual_values, how="outer", lsuffix="_ref", rsuffix="_actual").fillna(0.0)
            return {
                column: float((joined[f"{column}_ref"] - joined[f"{column}_actual"]).abs().max())
                if not joined.empty else 0.0
                for column in metric_cols
            }

        metric_cols = [
            "GVA_Direct", "GVA_Indirect", "GVA_Total",
            "Jobs_Direct", "Jobs_Indirect", "Jobs_Total",
        ]
        detail = detailed.groupby(["Country"], dropna=False)[metric_cols].sum().reset_index()
        errors = _compare(ref_levels[1].get("Total"), detail, metric_cols)
        max_global_gva = max(max_global_gva, max(errors[k] for k in errors if k.startswith("GVA")))
        max_global_jobs = max(max_global_jobs, max(errors[k] for k in errors if k.startswith("Jobs")))

        comp_actual = detailed.groupby(["component_group_L1", "Country"], dropna=False)[metric_cols].sum().reset_index()
        for component, actual_component in comp_actual.groupby("component_group_L1", dropna=False):
            ref_component = ref_levels[1].get(str(component))
            if ref_component is None:
                raise ValueError(f"Missing saved component reference {component!r} for delta={delta}.")
            errors = _compare(ref_component, actual_component, metric_cols)
            max_component_gva = max(max_component_gva, max(v for k, v in errors.items() if k.startswith("GVA")))
            max_component_jobs = max(max_component_jobs, max(v for k, v in errors.items() if k.startswith("Jobs")))

        for key, ref_sub in ref_sublevels[1].items():
            if key == ("Total", "ALL"):
                continue
            component, subcomponent = str(key[0]), str(key[1])
            actual_sub = detailed.loc[
                detailed["component_group_L1"].astype(str).eq(component)
                & detailed["sub_component_L1"].astype(str).eq(subcomponent)
            ]
            errors = _compare(ref_sub, actual_sub, metric_cols)
            max_subcomponent_gva = max(max_subcomponent_gva, max(v for k, v in errors.items() if k.startswith("GVA")))
            max_subcomponent_jobs = max(max_subcomponent_jobs, max(v for k, v in errors.items() if k.startswith("Jobs")))

    return {
        "max_global_gva_eur": max_global_gva,
        "max_global_jobs_fte_years": max_global_jobs,
        "max_component_gva_eur": max_component_gva,
        "max_component_jobs_fte_years": max_component_jobs,
        "max_subcomponent_gva_eur": max_subcomponent_gva,
        "max_subcomponent_jobs_fte_years": max_subcomponent_jobs,
    }




def export_scenario_weight_audit_xlsx(
    *,
    baseline_results_by_origin_sub_lvl: dict | None = None,
    scenarios_delta: dict,
    scenarios_result: dict | None = None,
    scenarios_delta_by_markup: dict | None = None,
    scenarios_result_by_markup: dict | None = None,
    baseline_upstream_gva: pd.DataFrame | None = None,
    recipient_weights: pd.DataFrame | None = None,
    EU27=(),
    connected_country: dict | None = None,
    connected_power_mw: float | None = None,
    weight_key: str = "Power (MW)",
    baseline_level: int = 0,
    annualize: bool = False,
    lifetime_years: float = 25.0,
    report_root=None,
    filename: str = "Supplementary_Material_D_Detailed_Results.xlsx",
    year: int = 2023,
    sector: str = "ALL",
):
    """Export detailed saved scenario results and the recorded sourcing audit.

    The scenario functions store their effective donor/receiver weights in the
    optional ``weight_audit`` field of their statistics.  The ``Results`` sheets
    are built only from saved detailed MRIO result frames; this exporter does
    not recalculate or alter the scenario.

    ``scenarios_delta_by_markup`` and ``scenarios_result_by_markup`` may contain
    the central case and compatible sensitivity cases.  The central markup is
    written to ``Results`` and additional markups to explicit sibling sheets.
    """
    from modules.config import RESULTS_SUPPLEMENTARY

    if report_root is None:
        report_root = RESULTS_SUPPLEMENTARY
    report_root = Path(report_root)
    report_root.mkdir(parents=True, exist_ok=True)

    if connected_power_mw is None:
        if connected_country and "EU27_AVG" in connected_country:
            connected_power_mw = connected_country["EU27_AVG"].get(weight_key)
        if connected_power_mw is None:
            raise ValueError(
                "connected_power_mw must be supplied from the current EU27_AVG configuration; "
                "old deployment-specific capacities cannot be inferred safely."
            )

    if baseline_upstream_gva is None or recipient_weights is None:
        for scenario in scenarios_delta.values():
            stats = getattr(scenario, "stats_turbine", {}) if not isinstance(scenario, dict) else scenario.get("stats_turbine", {})
            if stats.get("_baseline_upstream_gva") is not None:
                baseline_upstream_gva = pd.DataFrame(stats["_baseline_upstream_gva"])
                recipient_weights = pd.DataFrame(stats.get("_recipient_weights", []))
                break
    if baseline_upstream_gva is None or recipient_weights is None:
        raise ValueError("Bucket-level baseline GVA details are unavailable; recompute scenarios before exporting.")

    audit_parts = []
    for delta, scenario in scenarios_delta.items():
        stats_items = []
        if hasattr(scenario, "stats_turbine"):
            stats_items.append(("turbine", scenario.stats_turbine))
            stats_items.append(("farm", scenario.stats_farm))
        elif isinstance(scenario, dict):
            stats_items.append(("turbine", scenario.get("stats_turbine", {})))
            stats_items.append(("farm", scenario.get("stats_farm", {})))
        for source_type, stats in stats_items:
            for row in stats.get("weight_audit", []) or []:
                out = dict(row)
                out["scenario_type"] = source_type
                out["delta"] = delta
                audit_parts.append(out)
    if any(float(delta) != 0.0 for delta in scenarios_delta) and not audit_parts:
        raise ValueError(
            "Scenario statistics contain no weight audit. Recompute scenarios with the current code before exporting."
        )
    if audit_parts:
        raw_audit = pd.DataFrame(audit_parts)
        donor_over = raw_audit.loc[
            raw_audit["actual_donor_reduction"] > raw_audit["initial_expenditure"] + 1e-8
        ]
        if not donor_over.empty:
            raise ValueError("Scenario weight audit exceeds available donor expenditure.")
    actual_weights = pd.DataFrame(audit_parts)
    if not actual_weights.empty:
        actual_weights["fallback_rule"] = actual_weights["weight_source"].where(
            actual_weights["weight_source"].astype(str).str.startswith("fallback:"), ""
        )
        group_cols = [
            "delta", "scenario_type", "weight_role", "aggregation_level", "deployment",
            "OEM", "component", "subcomponent", "cost_block", "group",
        ]
        actual_weights["weight_sum"] = actual_weights.groupby(group_cols, dropna=False)["requested_weight"].transform("sum")
        actual_weights["realized_transfer"] = np.where(
            actual_weights["weight_role"].eq("donor"),
            actual_weights["actual_donor_reduction"],
            actual_weights["actual_receiver_addition"],
        )
        actual_weights["realized_transfer_share"] = (
            actual_weights["realized_transfer"] /
            actual_weights.groupby(group_cols, dropna=False)["realized_transfer"].transform("sum").replace(0.0, np.nan)
        ).fillna(0.0)
        bad = actual_weights.loc[~np.isclose(actual_weights["weight_sum"], 1.0, atol=1e-8)]
        if not bad.empty:
            raise ValueError("Scenario weight audit contains a group whose requested weights do not sum to one.")

    audit_columns = [
        "delta", "scenario_type", "weight_role", "weight_source", "fallback_rule", "aggregation_level",
        "deployment", "OEM", "component", "subcomponent", "cost_block", "group", "country",
        "requested_weight", "realized_transfer", "realized_transfer_share", "weight_sum",
    ]
    actual_weights = actual_weights.reindex(columns=audit_columns)
    reallocation_columns = [
        "delta", "scenario_type", "weight_role", "deployment", "OEM", "component",
        "subcomponent", "cost_block", "group", "country", "requested_transfer",
        "initial_expenditure", "actual_donor_reduction", "receiver_addition_before_markup",
        "markup_adjustment", "actual_receiver_addition", "final_expenditure",
    ]
    reallocation_audit = pd.DataFrame(audit_parts).reindex(columns=reallocation_columns)

    # Build the long economic Results export from saved MRIO result maps.
    if scenarios_delta_by_markup is None:
        scenarios_delta_by_markup = {"central": scenarios_delta}
    if scenarios_result_by_markup is None:
        if scenarios_result is None:
            raise ValueError(
                "Detailed scenario results are required for the Results sheet. "
                "Pass scenarios_result or scenarios_result_by_markup; baseline_level alone is insufficient."
            )
        scenarios_result_by_markup = {"central": scenarios_result}
    if set(scenarios_delta_by_markup) != set(scenarios_result_by_markup):
        raise ValueError("scenarios_delta_by_markup and scenarios_result_by_markup must have identical case keys.")

    case_specs = []
    central_markup = 0.3
    for case_key in scenarios_delta_by_markup:
        delta_mapping = scenarios_delta_by_markup[case_key]
        result_mapping = scenarios_result_by_markup[case_key]
        markup_mu = _scenario_markup_from_delta_mapping(delta_mapping)
        case_specs.append((markup_mu, delta_mapping, result_mapping))
    case_specs.sort(key=lambda item: (not np.isclose(item[0], central_markup), item[0]))
    if not any(np.isclose(markup_mu, central_markup) for markup_mu, _, _ in case_specs):
        raise ValueError("The central markup mu=0.3 is missing from the supplied scenario result maps.")

    # Preflight all cases without retaining their large long-format frames.
    case_diagnostics = []
    for markup_mu, delta_mapping, result_mapping in case_specs:
        frame, diagnostics = _scenario_results_from_saved_detail(
            scenarios_delta=delta_mapping,
            scenarios_result=result_mapping,
            EU27=EU27,
            connected_power_mw=float(connected_power_mw),
            annualize=bool(annualize),
            lifetime_years=float(lifetime_years),
            year=year,
            sector=sector,
            markup_mu=markup_mu,
        )
        reconciliation = _validate_saved_scenario_result_reconciliation(
            scenarios_result=result_mapping,
            year=year,
            sector=sector,
        )
        if len(frame) != diagnostics["n_rows"]:
            raise ValueError("Results row-count diagnostic does not match the built frame.")
        case_diagnostics.append((markup_mu, diagnostics, reconciliation))
        del frame

    total_result_rows = sum(diagnostics["n_rows"] for _, diagnostics, _ in case_diagnostics)
    combine_markups = total_result_rows + 1 <= _EXCEL_MAX_ROWS

    path = report_root / filename
    import os
    import tempfile
    from openpyxl import Workbook

    temp_handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.", suffix=".tmp.xlsx", dir=str(report_root), delete=False
    )
    temp_path = Path(temp_handle.name)
    temp_handle.close()

    def _write_frame(write_only_sheet, frame, *, include_header=True):
        if include_header:
            write_only_sheet.append(list(frame.columns))
        for row in frame.itertuples(index=False, name=None):
            values = []
            for value in row:
                if pd.isna(value):
                    values.append(None)
                elif isinstance(value, np.generic):
                    values.append(value.item())
                else:
                    values.append(value)
            write_only_sheet.append(values)

    sheet_records = []
    workbook = Workbook(write_only=True)
    try:
        if combine_markups:
            results_sheet = workbook.create_sheet(title="Results")
            first_frame = True
            for markup_mu, delta_mapping, result_mapping in case_specs:
                frame, _ = _scenario_results_from_saved_detail(
                    scenarios_delta=delta_mapping,
                    scenarios_result=result_mapping,
                    EU27=EU27,
                    connected_power_mw=float(connected_power_mw),
                    annualize=bool(annualize),
                    lifetime_years=float(lifetime_years),
                    year=year,
                    sector=sector,
                    markup_mu=markup_mu,
                )
                _write_frame(results_sheet, frame, include_header=first_frame)
                first_frame = False
                del frame
            sheet_records.append({
                "sheet": "Results",
                "markup_mu": "all supplied cases",
                "rows": int(total_result_rows),
                "note": "All compatible markup cases fit below Excel's row limit.",
            })
        else:
            for markup_mu, delta_mapping, result_mapping in case_specs:
                frame, _ = _scenario_results_from_saved_detail(
                    scenarios_delta=delta_mapping,
                    scenarios_result=result_mapping,
                    EU27=EU27,
                    connected_power_mw=float(connected_power_mw),
                    annualize=bool(annualize),
                    lifetime_years=float(lifetime_years),
                    year=year,
                    sector=sector,
                    markup_mu=markup_mu,
                )
                sheet_name = (
                    "Results"
                    if np.isclose(markup_mu, central_markup)
                    else f"Results_mu_{markup_mu:g}".replace(".", "_")
                )
                if len(frame) + 1 > _EXCEL_MAX_ROWS:
                    raise ValueError(
                        f"Results for markup_mu={markup_mu:g} contain {len(frame):,} rows "
                        "and exceed Excel's row limit."
                    )
                _write_frame(workbook.create_sheet(title=sheet_name), frame)
                sheet_records.append({
                    "sheet": sheet_name,
                    "markup_mu": float(markup_mu),
                    "rows": int(len(frame)),
                    "note": "Markup sensitivity; explicit partition by markup",
                })
                del frame

        if len(sheet_records) > 1:
            _write_frame(workbook.create_sheet(title="Results_Index"), pd.DataFrame(sheet_records))
        _write_frame(workbook.create_sheet(title="Baseline_Upstream_GVA"), baseline_upstream_gva)
        _write_frame(workbook.create_sheet(title="Recipient_Weights"), recipient_weights)
        _write_frame(workbook.create_sheet(title="Actual_Scenario_Weights"), actual_weights)
        _write_frame(workbook.create_sheet(title="Reallocation_Audit"), reallocation_audit)
        workbook.save(temp_path)
        os.replace(temp_path, path)
    except Exception:
        workbook.close()
        if temp_path.exists():
            temp_path.unlink()
        raise
    finally:
        workbook.close()
    print(
        "[scenario results export] "
        f"{total_result_rows:,} Results rows across "
        f"{len(case_specs)} markup case(s); workbook={path.name}"
    )
    for markup_mu, diagnostics, reconciliation in case_diagnostics:
        print(
            f"[scenario results validation] mu={markup_mu:g}, deltas={diagnostics['n_scenarios']}, "
            f"max split GVA={diagnostics['max_split_gva_million_eur']:.3e} million EUR, "
            f"max split Employment={diagnostics['max_split_jobs_fte_years']:.3e} FTE-years, "
            f"max global GVA reconciliation={reconciliation['max_global_gva_eur']:.3e} EUR, "
            f"max subcomponent GVA reconciliation={reconciliation['max_subcomponent_gva_eur']:.3e} EUR."
        )
    return path

# ---------- EU27 filters MUST be on Origin ----------




# ---------- existing table builders (UNCHANGED) ----------


def build_all_indicators_table_from_by_origin(
    results_by_origin_sub_lvl: dict,
    lvl: int,
    core: List[str],
    indicators=("Output", "GVA", "Income", "Jobs"),
    relative=None,
    denom_metric="Output",
    denom_type="Total",
    country_col="Country",   # Country == Deployment in the long dicts
    eur_to_million=True,
    decimals_eur_mn=2,
    decimals_ratio=4,
    scope="global",          # "global" | "eu27"
    connected_country: dict | None = None,
    weight_key: str = "Power (MW)",
    add_indicator_space: bool = True,
    eu_pct_decimals: int = 1,
    component_order: list[str] | None = None,
    type_order: list[str] | None = None,
    annualize: bool = True,
    lifetime_years: float = 25.0,
    per_mw: bool = False,
):
    """
    Returns table with columns (Indicator, Metric) where Metric = EU27, RoW, Global, EU27%.
    Values are weighted-average across deployments (weights from connected_country if provided).
    """
    # Long format: one row per (Deployment, Origin)
    base_long = by_origin_to_component_dict(results_by_origin_sub_lvl[lvl])

    # weights for connected-capacity average
    core_set = {str(c) for c in (core or [])}
    if connected_country is not None:
        w_map = {str(k): float(v.get(weight_key, 0.0)) for k, v in connected_country.items()}
        if core_set:
            w_map = {d: w for d, w in w_map.items() if d in core_set}
        w_map = {d: w for d, w in w_map.items() if w > 0}
    else:
        w_map = {d: 1.0 for d in core_set} if core_set else {}

    def _weights_for_deps(deps: list[str]) -> dict[str, float]:
        if w_map:
            return {d: w_map.get(d, 0.0) for d in deps}
        return {d: 1.0 for d in deps}

    types = list(type_order) if type_order else ["Direct", "Indirect", "Total"]
    ind_cols = {ind: [f"{ind}_{t}" for t in types] for ind in indicators}

    rows = []
    for comp, df in base_long.items():
        d = df.copy()
        # keep only core deployments if provided
        if core_set:
            d = d[d[country_col].astype(str).isin(core_set)]
        if d.empty:
            continue

        d["Origin"] = d["Origin"].astype(str)
        orig = d["Origin"].astype(str).str.strip().str.upper()
        d["_bucket"] = np.where(
            orig.isin(EU27),
            "EU27",
            np.where(
                orig.isin({"GB", "UK"}),
                "UK",
                np.where(orig.isin(set(EUROPE_CODES)), "RoE", "RoW"),
            ),
        )

        num_cols = [c for c in d.columns if "_" in c and c.split("_", 1)[0] in set(indicators)]
        g = d.groupby([country_col, "_bucket"], as_index=False)[num_cols].sum(numeric_only=True)

        deps = sorted(g[country_col].astype(str).unique().tolist())
        w_dep = _weights_for_deps(deps)
        w_sum = sum(w_dep.values())
        if w_sum <= 0.0:
            w_dep = {d: 1.0 for d in deps}
            w_sum = float(len(deps)) if deps else 1.0

        for ind in indicators:
            for t in types:
                col = f"{ind}_{t}"
                if col not in num_cols:
                    continue
                piv = g.pivot_table(index=country_col, columns="_bucket", values=col, aggfunc="sum").fillna(0.0)
                for dep in deps:
                    if dep not in piv.index:
                        piv.loc[dep] = 0.0
                eu = float((piv.get("EU27", 0.0) * pd.Series(w_dep)).sum() / w_sum)
                uk = float((piv.get("UK", 0.0) * pd.Series(w_dep)).sum() / w_sum)
                roe = float((piv.get("RoE", 0.0) * pd.Series(w_dep)).sum() / w_sum)
                rw = float((piv.get("RoW", 0.0) * pd.Series(w_dep)).sum() / w_sum)
                gl = eu + uk + roe + rw
                eu_pct = 0.0 if gl == 0 else (eu / gl) * 100.0
                europe_pct = 0.0 if gl == 0 else ((eu + uk + roe) / gl) * 100.0
                row_pct = 0.0 if gl == 0 else (rw / gl) * 100.0

                rows.append({
                    "Component": comp,
                    "Type": t,
                    "Indicator": ind,
                    "EU27": eu,
                    "EU27%": eu_pct,
                    "UK": uk,
                    "RoE": roe,
                    "Europe %": europe_pct,
                    "RoW": rw,
                    "RoW %": row_pct,
                })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)

    # scale monetary indicators to million EUR
    money_inds = {"Output", "GVA", "Income"}
    if eur_to_million:
        m = out["Indicator"].isin(money_inds)
        for c in ["EU27", "UK", "RoE", "RoW"]:
            out.loc[m, c] = out.loc[m, c] / 1_000_000.0

    # convert annual -> lifetime if requested (applies to all numeric indicators)
    if not annualize:
        for c in ["EU27", "UK", "RoE", "RoW"]:
            out[c] = out[c].astype(float) * float(lifetime_years)

    # convert to per-MW if requested (uses connected capacity sum)
    if per_mw and connected_country is not None:
        mw_vals = []
        for k, v in connected_country.items():
            if core_set and str(k) not in core_set:
                continue
            try:
                mw_vals.append(float(v.get(weight_key, 0.0)))
            except Exception:
                mw_vals.append(0.0)
        total_mw = sum(mw_vals)
        if total_mw > 0:
            for c in ["EU27", "UK", "RoE", "RoW"]:
                out[c] = out[c].astype(float) / total_mw

    # keep numeric precision; formatting handled in df_to_latex
    for c in ["EU27", "UK", "RoE", "RoW", "EU27%", "Europe %", "RoW %"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # pivot to MultiIndex columns (Indicator, Country)
    out = out.pivot_table(
        index=["Component", "Type"],
        columns=["Indicator"],
        values=["EU27", "EU27%", "UK", "RoE", "Europe %", "RoW", "RoW %"],
        aggfunc="first",
    )

    # reorder columns to (Indicator, Country)
    out = out.swaplevel(0, 1, axis=1)
    country_order = ["EU27", "EU27%", "UK", "RoE", "Europe %", "RoW", "RoW %"]
    ordered_cols = []
    for ind in indicators:
        for c in country_order:
            if (ind, c) in out.columns:
                ordered_cols.append((ind, c))
    leftovers = [c for c in out.columns if c not in ordered_cols]
    out = out.reindex(columns=ordered_cols + leftovers)

    # optional space column inside each indicator group (unused now)
    if add_indicator_space:
        pass

    # reorder index levels if requested (preserve custom order via categorical index)
    if component_order is None:
        component_order = ["Devex", "BoP", "Turbine", "Installation", "O&M", "Total"]
    if type_order is None:
        type_order = ["Direct", "Indirect", "Total"]

    idx = out.index
    comp_vals = idx.get_level_values(0).astype(str).tolist()
    type_vals = idx.get_level_values(1).astype(str).tolist()

    comp_order = [c for c in component_order if c in comp_vals] + [c for c in comp_vals if c not in set(component_order)]
    type_order_use = [t for t in type_order if t in type_vals] + [t for t in type_vals if t not in set(type_order)]

    comp_cat = pd.CategoricalIndex(comp_vals, categories=comp_order, ordered=True)
    type_cat = pd.CategoricalIndex(type_vals, categories=type_order_use, ordered=True)
    out.index = pd.MultiIndex.from_arrays([comp_cat, type_cat], names=out.index.names)
    out = out.sort_index(level=[0, 1], sort_remaining=False)

    return out


def build_all_indicators_table_comp_columns_from_by_origin(
    results_by_origin_sub_lvl: dict,
    *,
    lvl: int,
    core: List[str],
    connected_country: dict | None = None,
    weight_key: str = "Power (MW)",
    indicators: tuple = ("Output", "GVA", "Income", "Jobs"),
    component_order: list[str] | None = None,
    type_order: list[str] | None = None,
    metric_order: list[str] | None = None,
    include_eu_pct: bool = True,
    row_region: bool = False,
    region_order: list[str] | None = None,
    add_row_eu_pct: bool = False,
    annualize: bool = True,
    lifetime_years: float = 25.0,
    per_mw: bool = False,
    eur_to_million: bool = True,
    jobs_to_thousand: bool = True,
):
    """
    Same values as build_all_indicators_table_from_by_origin, but reshaped:
      - Rows: (Type, Impact) or (Region, Impact) if row_region=True
      - Columns: (Component, Metric) where Metric = EU27, Global, EU27%
    """
    base = build_all_indicators_table_from_by_origin(
        results_by_origin_sub_lvl,
        lvl=lvl,
        core=core,
        connected_country=connected_country,
        weight_key=weight_key,
        indicators=indicators,
        eur_to_million=eur_to_million,
        annualize=annualize,
        lifetime_years=lifetime_years,
        per_mw=per_mw,
    )
    if base is None or base.empty:
        return pd.DataFrame()

    # Ensure Global metric exists (sum of EU27 + UK + RoE + RoW)
    if isinstance(base.columns, pd.MultiIndex):
        for ind in indicators:
            if (ind, "Global") not in base.columns:
                parts = [c for c in [(ind, "EU27"), (ind, "UK"), (ind, "RoE"), (ind, "RoW")] if c in base.columns]
                if parts:
                    base[(ind, "Global")] = base[parts].fillna(0.0).sum(axis=1)
        if jobs_to_thousand:
            job_metrics = [
                c for c in base.columns
                if len(c) >= 2 and str(c[0]) == "Jobs" and "%" not in str(c[1])
            ]
            if job_metrics:
                base.loc[:, job_metrics] = base.loc[:, job_metrics] / 1_000.0

    if component_order is None:
        component_order = ["Total", "Devex", "Turbine", "BoP", "Installation", "O&M"]
    if type_order is None:
        type_order = ["Direct", "Total"]

    # filter types
    type_keep = [t for t in type_order if t in base.index.get_level_values(1).astype(str).unique()]
    base = base.loc[(slice(None), type_keep), :]

    indicator_order = list(indicators)
    region_metric_default = ["EU27", "UK", "RoE", "RoW", "Global", "EU27%", "Europe %", "RoW %"]
    if metric_order is None:
        metric_order = list(region_order) if (row_region and region_order is not None) else region_metric_default
    elif row_region:
        region_metric_set = {"Global", "EU27", "EU27%", "UK", "RoE", "Europe %", "RoW", "RoW %"}
        if set(map(str, metric_order)).issubset(region_metric_set):
            metric_order = list(metric_order)
    metrics = metric_order[:]
    if not include_eu_pct:
        metrics = [m for m in metrics if m != "EU27%"]
    impact_label = {"Jobs": "Employment"}

    records = []
    for (comp, typ), row in base.iterrows():
        comp = str(comp)
        typ = str(typ)
        for ind in indicator_order:
            ind_label = impact_label.get(ind, ind)
            for m in metrics:
                val = row.get((ind, m))
                records.append(
                    {"Component": comp, "Type": typ, "Impact": ind_label, "Metric": m, "Value": val}
                )

    df = pd.DataFrame.from_records(records)
    out = df.pivot_table(
        index=["Type", "Impact"],
        columns=["Component", "Metric"],
        values="Value",
        aggfunc="first",
    )

    # reorder columns
    comp_present = out.columns.get_level_values(0).astype(str).unique().tolist()
    comp_order = [c for c in component_order if c in comp_present] + [c for c in comp_present if c not in set(component_order)]
    col_order = []
    for comp in comp_order:
        for m in metrics:
            if (comp, m) in out.columns:
                col_order.append((comp, m))
    leftovers = [c for c in out.columns if c not in col_order]
    out = out.reindex(columns=col_order + leftovers)

    # if only one metric, flatten columns to a single header row
    if len(metrics) == 1:
        out.columns = out.columns.get_level_values(0)

    # optional: move Region (metric) to rows and drop Type
    if row_region:
        if isinstance(out.columns, pd.MultiIndex):
            # keep only the first type (expected: "Total") if multiple
            if isinstance(out.index, pd.MultiIndex):
                types_present = out.index.get_level_values(0).astype(str).unique().tolist()
                keep_type = type_order[0] if type_order and type_order[0] in types_present else (types_present[0] if types_present else None)
                if keep_type is not None:
                    out = out.loc[(keep_type, slice(None)), :]

            # move Metric to rows
            out = out.stack(level=1, future_stack=True)
            out.index = out.index.set_names(["Type", "Impact", "Region"])
            # drop Type level (now constant)
            out = out.droplevel(0)
            # swap to make Region the first index level
            out = out.swaplevel(0, 1)
            out.index = out.index.set_names(["Region", "Impact"])

            # order regions (as rows) and impacts
            if region_order is None:
                region_order = ["EU27", "Global"]
            region_vals = out.index.get_level_values(0).astype(str).unique().tolist()
            reg_order = [r for r in region_order if r in region_vals] + [r for r in region_vals if r not in set(region_order)]

            impact_vals = out.index.get_level_values(1).astype(str).unique().tolist()
            impact_order = [impact_label.get(i, i) for i in indicator_order]
            impact_order_use = [i for i in impact_order if i in impact_vals] + [i for i in impact_vals if i not in set(impact_order)]

            reg_cat = pd.CategoricalIndex(out.index.get_level_values(0), categories=reg_order, ordered=True)
            impact_cat = pd.CategoricalIndex(out.index.get_level_values(1), categories=impact_order_use, ordered=True)
            out.index = pd.MultiIndex.from_arrays([reg_cat, impact_cat], names=out.index.names)
            out = out.sort_index(level=[0, 1], sort_remaining=False)

            # add EU27% row (EU27 / Global) if requested
            if add_row_eu_pct:
                try:
                    eu = out.loc[("EU27", slice(None)), :]
                    gl = out.loc[("Global", slice(None)), :]
                    # align on Impact level
                    eu.index = eu.index.droplevel(0)
                    gl.index = gl.index.droplevel(0)
                    eu_pct = (eu / gl.replace(0, np.nan)) * 100.0
                    eu_pct = eu_pct.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    # build MultiIndex with Region="EU27%"
                    eu_pct.index = pd.MultiIndex.from_product([["EU27%"], eu_pct.index], names=out.index.names)
                    out = pd.concat([out, eu_pct], axis=0)
                except Exception:
                    pass

            # reorder regions after possible EU27% insertion
            if region_order is None:
                region_order = ["EU27", "Global", "EU27%"] if add_row_eu_pct else ["EU27", "Global"]
            region_vals = out.index.get_level_values(0).astype(str).unique().tolist()
            reg_order = [r for r in region_order if r in region_vals] + [r for r in region_vals if r not in set(region_order)]

            impact_vals = out.index.get_level_values(1).astype(str).unique().tolist()
            impact_order = [impact_label.get(i, i) for i in indicator_order]
            impact_order_use = [i for i in impact_order if i in impact_vals] + [i for i in impact_vals if i not in set(impact_order)]

            reg_cat = pd.CategoricalIndex(out.index.get_level_values(0), categories=reg_order, ordered=True)
            impact_cat = pd.CategoricalIndex(out.index.get_level_values(1), categories=impact_order_use, ordered=True)
            out.index = pd.MultiIndex.from_arrays([reg_cat, impact_cat], names=out.index.names)
            out = out.sort_index(level=[0, 1], sort_remaining=False)

            # Final row layout for report tables: Impact first, Region second.
            region_report_order = list(region_order) if region_order is not None else list(metrics)
            region_vals = out.index.get_level_values(0).astype(str).unique().tolist()
            reg_order = [r for r in region_report_order if r in region_vals] + [
                r for r in region_vals if r not in set(region_report_order)
            ]

            impact_vals = out.index.get_level_values(1).astype(str).unique().tolist()
            impact_order = [impact_label.get(i, i) for i in indicator_order]
            impact_order_use = [i for i in impact_order if i in impact_vals] + [
                i for i in impact_vals if i not in set(impact_order)
            ]

            impact_cat = pd.CategoricalIndex(out.index.get_level_values(1), categories=impact_order_use, ordered=True)
            reg_cat = pd.CategoricalIndex(out.index.get_level_values(0), categories=reg_order, ordered=True)
            out.index = pd.MultiIndex.from_arrays([impact_cat, reg_cat], names=["Impact", "Region"])
            out = out.sort_index(level=[0, 1], sort_remaining=False)
            return out

    # order index
    type_present = out.index.get_level_values(0).astype(str).unique().tolist()
    type_order_use = [t for t in type_order if t in type_present] + [t for t in type_present if t not in set(type_order)]
    impact_present = out.index.get_level_values(1).astype(str).unique().tolist()
    impact_order = [impact_label.get(i, i) for i in indicator_order]
    impact_order_use = [i for i in impact_order if i in impact_present] + [i for i in impact_present if i not in set(impact_order)]

    type_cat = pd.CategoricalIndex(out.index.get_level_values(0), categories=type_order_use, ordered=True)
    impact_cat = pd.CategoricalIndex(out.index.get_level_values(1), categories=impact_order_use, ordered=True)
    out.index = pd.MultiIndex.from_arrays([type_cat, impact_cat], names=["Type", "Impact"])
    out = out.sort_index(level=[0, 1], sort_remaining=False)

    return out


def build_all_indicators_table_effect_columns_from_by_origin(
    results_by_origin_sub_lvl: dict,
    *,
    lvl: int,
    core: List[str],
    connected_country: dict | None = None,
    weight_key: str = "Power (MW)",
    indicators: tuple = ("Output", "GVA", "Income", "Jobs"),
    effect_order: list[str] | None = None,
    metric_order: list[str] | None = None,
    region_order: list[str] | None = None,
    include_eu_pct: bool = True,
    annualize: bool = True,
    lifetime_years: float = 25.0,
    per_mw: bool = False,
    eur_to_million: bool = True,
    jobs_to_thousand: bool = False,
    total_component_label: str = "Total",
):
    """
    Report table with rows (Impact, Region) and columns Direct/Indirect/Total.
    Uses the aggregate Total component when available.
    """
    if effect_order is None:
        effect_order = ["Direct", "Indirect", "Total"]

    region_default = ["EU27", "UK", "RoE", "RoW", "Global", "EU27%", "Europe %", "RoW %"]
    if metric_order is None:
        metric_order = list(region_order) if region_order is not None else region_default
    else:
        metric_order = list(metric_order)
    if not include_eu_pct:
        metric_order = [m for m in metric_order if m != "EU27%"]

    base = build_all_indicators_table_from_by_origin(
        results_by_origin_sub_lvl,
        lvl=lvl,
        core=core,
        connected_country=connected_country,
        weight_key=weight_key,
        indicators=indicators,
        type_order=effect_order,
        eur_to_million=eur_to_million,
        annualize=annualize,
        lifetime_years=lifetime_years,
        per_mw=per_mw,
    )
    if base is None or base.empty:
        return pd.DataFrame()

    if isinstance(base.columns, pd.MultiIndex):
        for ind in indicators:
            if (ind, "Global") not in base.columns:
                parts = [c for c in [(ind, "EU27"), (ind, "UK"), (ind, "RoE"), (ind, "RoW")] if c in base.columns]
                if parts:
                    base[(ind, "Global")] = base[parts].fillna(0.0).sum(axis=1)
        if jobs_to_thousand:
            job_metrics = [
                c for c in base.columns
                if len(c) >= 2 and str(c[0]) == "Jobs" and "%" not in str(c[1])
            ]
            if job_metrics:
                base.loc[:, job_metrics] = base.loc[:, job_metrics] / 1_000.0

    comp_vals = base.index.get_level_values(0).astype(str)
    if total_component_label in set(comp_vals):
        selected = base.loc[(total_component_label, slice(None)), :].copy()
        selected.index = selected.index.droplevel(0)
    else:
        selected = base.groupby(level=1).sum(numeric_only=True)

    effect_present = selected.index.astype(str).unique().tolist()
    effect_order_use = [e for e in effect_order if e in effect_present] + [
        e for e in effect_present if e not in set(effect_order)
    ]
    selected = selected.reindex(effect_order_use)

    impact_label = {"Jobs": "Employment"}
    records = []
    for ind in indicators:
        ind_label = impact_label.get(ind, ind)
        for region in metric_order:
            for effect in effect_order_use:
                val = selected.loc[effect].get((ind, region), np.nan)
                records.append(
                    {
                        "Impact": ind_label,
                        "Region": region,
                        "Effect": effect,
                        "Value": val,
                    }
                )

    df = pd.DataFrame.from_records(records)
    out = df.pivot_table(
        index=["Impact", "Region"],
        columns="Effect",
        values="Value",
        aggfunc="first",
    )

    out = out.reindex(columns=[e for e in effect_order if e in out.columns])
    impact_order = [impact_label.get(i, i) for i in indicators]
    impact_vals = out.index.get_level_values(0).astype(str).unique().tolist()
    impact_order_use = [i for i in impact_order if i in impact_vals] + [
        i for i in impact_vals if i not in set(impact_order)
    ]
    region_vals = out.index.get_level_values(1).astype(str).unique().tolist()
    region_order_use = [r for r in metric_order if r in region_vals] + [
        r for r in region_vals if r not in set(metric_order)
    ]

    impact_cat = pd.CategoricalIndex(out.index.get_level_values(0), categories=impact_order_use, ordered=True)
    region_cat = pd.CategoricalIndex(out.index.get_level_values(1), categories=region_order_use, ordered=True)
    out.index = pd.MultiIndex.from_arrays([impact_cat, region_cat], names=["Impact", "Region"])
    out = out.sort_index(level=[0, 1], sort_remaining=False)
    out.columns.name = None
    return out


def build_all_indicators_table_effect_and_comp_columns_from_by_origin(
    results_by_origin_sub_lvl: dict,
    *,
    lvl: int = 2,
    core: List[str],
    connected_country: dict | None = None,
    weight_key: str = "Power (MW)",
    indicators: tuple = ("Output", "GVA", "Income", "Jobs"),
    component_order: list[str] | None = None,
    effect_order: list[str] | None = None,
    metric_order: list[str] | None = None,
    region_order: list[str] | None = None,
    include_eu_pct: bool = True,
    annualize: bool = True,
    lifetime_years: float = 25.0,
    per_mw: bool = False,
    eur_to_million: bool = True,
    jobs_to_thousand: bool = True,
):
    """
    Combined report table with rows (Impact, Region) and columns:
    Direct, Indirect, Total, Devex, BoP, Turbine, Installation, O&M.

    Direct and Indirect are total-project effect columns. Component columns
    report total effects by component.
    """
    if component_order is None:
        component_order = ["Total", "Devex", "BoP", "Turbine", "Installation", "O&M"]
    if effect_order is None:
        effect_order = ["Direct", "Indirect"]

    effect_cols = build_all_indicators_table_effect_columns_from_by_origin(
        results_by_origin_sub_lvl,
        lvl=lvl,
        core=core,
        connected_country=connected_country,
        weight_key=weight_key,
        indicators=indicators,
        effect_order=["Direct", "Indirect", "Total"],
        metric_order=metric_order,
        region_order=region_order,
        include_eu_pct=include_eu_pct,
        annualize=annualize,
        lifetime_years=lifetime_years,
        per_mw=per_mw,
        eur_to_million=eur_to_million,
        jobs_to_thousand=jobs_to_thousand,
        total_component_label="Total",
    )
    effect_keep = [c for c in effect_order if c in effect_cols.columns]
    effect_cols = effect_cols.reindex(columns=effect_keep)

    comp_cols = build_all_indicators_table_comp_columns_from_by_origin(
        results_by_origin_sub_lvl,
        lvl=lvl,
        core=core,
        connected_country=connected_country,
        weight_key=weight_key,
        indicators=indicators,
        component_order=component_order,
        type_order=["Total"],
        metric_order=metric_order,
        include_eu_pct=include_eu_pct,
        row_region=True,
        region_order=region_order,
        add_row_eu_pct=False,
        annualize=annualize,
        lifetime_years=lifetime_years,
        per_mw=per_mw,
        eur_to_million=eur_to_million,
        jobs_to_thousand=jobs_to_thousand,
    )
    comp_keep = [c for c in component_order if c in comp_cols.columns]
    comp_cols = comp_cols.reindex(columns=comp_keep)

    out = pd.concat([effect_cols, comp_cols], axis=1)
    if isinstance(effect_cols.index, pd.MultiIndex):
        out = out.reindex(effect_cols.index)

    # The regular table builders intentionally aggregate all non-European
    # origins into RoW. For this table only, retain that aggregate and derive
    # a China/Others child breakdown from the unrounded country-level rows.
    origin_detail = by_origin_to_component_dict(results_by_origin_sub_lvl[lvl])
    china_codes = {
        str(code).strip().upper()
        for code, name in CODE_TO_NAME.items()
        if str(name).strip().casefold() == "china"
    }
    china_codes.add("CN")

    core_set = {str(c) for c in (core or [])}
    if connected_country is not None:
        weight_map = {
            str(k): float(v.get(weight_key, 0.0))
            for k, v in connected_country.items()
        }
        if core_set:
            weight_map = {d: w for d, w in weight_map.items() if d in core_set}
        weight_map = {d: w for d, w in weight_map.items() if w > 0}
    else:
        weight_map = {d: 1.0 for d in core_set} if core_set else {}

    def _weighted_origin_values(origin_codes: set[str]) -> dict[str, dict[str, float]]:
        values: dict[str, dict[str, float]] = {}
        for component, frame in origin_detail.items():
            d = frame.copy()
            if core_set:
                d = d[d["Country"].astype(str).isin(core_set)]
            if d.empty:
                continue
            origin = d["Origin"].astype(str).str.strip().str.upper()
            d = d[origin.isin(origin_codes)]
            if d.empty:
                continue

            num_cols = [
                c for c in d.columns
                if "_" in c and c.split("_", 1)[0] in set(indicators)
            ]
            grouped = d.groupby("Country", as_index=False)[num_cols].sum(numeric_only=True)
            deps = sorted(grouped["Country"].astype(str).unique().tolist())
            weights = {dep: weight_map.get(dep, 0.0) for dep in deps}
            weight_sum = sum(weights.values())
            if weight_sum <= 0.0:
                weights = {dep: 1.0 for dep in deps}
                weight_sum = float(len(deps)) if deps else 1.0

            component_values = values.setdefault(str(component), {})
            for column in num_cols:
                series = grouped.set_index("Country")[column].astype(float)
                component_values[column] = float(
                    sum(float(series.get(dep, 0.0)) * weights[dep] for dep in deps)
                    / weight_sum
                )

        money_inds = {"Output", "GVA", "Income"}
        for component_values in values.values():
            for column in list(component_values):
                indicator = column.split("_", 1)[0]
                if eur_to_million and indicator in money_inds:
                    component_values[column] /= 1_000_000.0
                if not annualize:
                    component_values[column] *= float(lifetime_years)
                if jobs_to_thousand and indicator == "Jobs":
                    component_values[column] /= 1_000.0

        if per_mw and connected_country is not None:
            total_mw = sum(
                float(v.get(weight_key, 0.0))
                for k, v in connected_country.items()
                if not core_set or str(k) in core_set
            )
            if total_mw > 0:
                for component_values in values.values():
                    for column in component_values:
                        component_values[column] /= total_mw
        return values

    china_values = _weighted_origin_values(china_codes)
    data_columns = list(out.columns)

    def _china_row(indicator: str) -> pd.Series:
        total_values = china_values.get("Total", {})
        result = {}
        for column in data_columns:
            if column in {"Direct", "Indirect"}:
                source_column = f"{indicator}_{column}"
                result[column] = total_values.get(source_column, np.nan)
            else:
                source_component = "Installation" if column == "Inst." else str(column)
                source_column = f"{indicator}_Total"
                result[column] = china_values.get(source_component, {}).get(source_column, np.nan)
        return pd.Series(result, index=data_columns, dtype=float)

    impact_labels = {"Jobs": "Employment"}
    for indicator in indicators:
        impact = impact_labels.get(indicator, indicator)
        if (impact, "RoW") not in out.index:
            continue
        row = out.loc[(impact, "RoW")].astype(float)
        china = _china_row(indicator)
        others = row - china
        global_row = out.loc[(impact, "Global")].astype(float)
        for region, values in (("China", china), ("Others", others)):
            out.loc[(impact, region), data_columns] = values
        for region, values in (("China %", china), ("Others %", others)):
            pct = (values / global_row.replace(0, np.nan)) * 100.0
            out.loc[(impact, region), data_columns] = pct.replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0.0)

    # Keep the finalized regional layout: children immediately follow RoW,
    # while Output is the only block without percentage rows.
    region_order_base = ["EU27", "UK", "RoE", "RoW", "Global"]
    pct_order = ["EU27%", "Europe %", "RoW %", "China %", "Others %"]
    ordered_rows = []
    for impact in out.index.get_level_values(0).astype(str).unique().tolist():
        regions = region_order_base[:]
        if impact != "Output":
            regions += pct_order
        for region in regions:
            if (impact, region) in out.index:
                ordered_rows.append((impact, region))
    out = out.reindex(pd.MultiIndex.from_tuples(ordered_rows, names=out.index.names))
    return out


EU_LABEL, UK_LABEL, ROE_LABEL = "EU27", "United Kingdom", "RoE"
OTHER_EU27_LABEL = "Other EU27"
OTHER_ORIGINS_LABEL = "Other origins"
ROW_LABEL = "RoW residual"
EU_DEPLOY_LABEL = "EU27 deployment"
EU_NONDEPLOY_LABEL = "EU27 non-deployment"
def _lighten(color, amount=0.35):
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)




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


def _capture_split_origin_label(iso: str, labels: dict | None = None) -> str:
    iso = str(iso).strip().upper()
    labels = labels or {}
    return labels.get(iso) or CODE_TO_NAME.get(iso) or canonical_name(iso) or iso


def build_capture_sublevel_avg_connected_power_df(
    results_by_origin_sub_lvl,
    connected_country,
    *,
    indicator="Jobs",
    typ="Total",
    mode="mult",
    year=2023,
    sector="ALL",
    avg_dep_order=("BE", "NL", "DE", "DK", "GB"),
    weight_key="Power (MW)",
    group_eu_deploy=False,
    eu_deploy_set=("BE", "NL", "DK", "DE"),
    split_origin_countries=("CN",),
    split_origin_country_labels=None,
    deployment_col="Deployment",
    origin_col="Country",
    year_col="Year",
    sector_col="Sector",
):
    """Build weighted-average capture values by subcomponent and origin bucket."""
    split_origin_country_labels = split_origin_country_labels or {}
    split_origin_countries = tuple(
        str(c).strip().upper()
        for c in (split_origin_countries or ())
        if str(c).strip().upper() not in {"ROW", "X01"}
    )
    split_labels = [_capture_split_origin_label(c, split_origin_country_labels) for c in split_origin_countries]

    has_eu_split = any(c in EU27 for c in split_origin_countries)
    eu_label = (
        EU_NONDEPLOY_LABEL
        if group_eu_deploy
        else OTHER_EU27_LABEL if has_eu_split else EU_LABEL
    )
    dep_label = EU_DEPLOY_LABEL if group_eu_deploy else None
    origin_list = [eu_label]
    if dep_label is not None:
        origin_list.append(dep_label)
    origin_list += split_labels + [ROW_LABEL, OTHER_ORIGINS_LABEL]

    ind_u = str(indicator).upper()
    mode = str(mode).lower().strip()
    if mode not in {"mult", "total", "share"}:
        raise ValueError("mode must be 'mult', 'total', or 'share'.")

    if mode == "mult":
        if ind_u not in {"JOBS", "GVA"}:
            raise ValueError("indicator must be 'Jobs' or 'GVA'.")
        num_col = "Jobs_Total" if ind_u == "JOBS" else "GVA_Total"
        shock_col = "Output_Direct"
        needed = {deployment_col, origin_col, num_col, shock_col}
    else:
        col = f"{indicator}_{typ}"
        needed = {deployment_col, origin_col, col}

    eu_deploy_set = {str(c).strip().upper() for c in (eu_deploy_set or ())}
    def _bucket(o):
        o = str(o).strip().upper().replace("UK", "GB")
        if group_eu_deploy and o in eu_deploy_set:
            return EU_DEPLOY_LABEL
        if o in {"ROW", "X01"}:
            return ROW_LABEL
        if o in split_origin_countries:
            return _capture_split_origin_label(o, split_origin_country_labels)
        if o in EU27:
            return EU_NONDEPLOY_LABEL if group_eu_deploy else eu_label
        return OTHER_ORIGINS_LABEL

    def _w(dep):
        dep = str(dep).strip().upper()
        return float(connected_country.get(dep, {}).get(weight_key, 0.0))

    avg_list = tuple(str(d).strip().upper() for d in avg_dep_order)
    w_map = {d: _w(d) for d in avg_list}
    # A zero total weight must not silently turn a valid capture panel into
    # all zeros. When no requested deployment has a usable capacity weight,
    # fall back to an equal-weight average of those same deployments.
    if not any(abs(float(weight)) > 1e-12 for weight in w_map.values()):
        w_map = {d: 1.0 for d in avg_list}
    w_sum = sum(w_map.values())

    skip = _self_aggregate_keys_to_skip(results_by_origin_sub_lvl.keys())
    keys = [
        k for k in results_by_origin_sub_lvl.keys()
        if k != ("Total", "ALL")
        and isinstance(k, tuple)
        and len(k) >= 2
        and not _is_aggregate_component_key(k)
        and k not in skip
    ]

    rows = []
    for key in keys:
        comp, sub = key[0], key[1]
        df = results_by_origin_sub_lvl[key].copy()

        if year is not None and year_col in df.columns:
            df = df[df[year_col] == year]
        if sector is not None and sector_col in df.columns:
            df = df[df[sector_col].astype(str) == str(sector)]

        missing = needed - set(df.columns)
        if missing:
            raise KeyError(f"Missing columns {missing} in key {(comp, sub)}")

        df[deployment_col] = df[deployment_col].astype(str).str.strip().str.upper()
        df[origin_col] = df[origin_col].astype(str).str.strip().str.upper().str.replace("UK", "GB")
        df["__bucket__"] = df[origin_col].map(_bucket)

        acc = {lab: 0.0 for lab in origin_list}
        for dep in avg_list:
            dd = df[df[deployment_col] == dep]
            if dd.empty:
                continue
            w = w_map.get(dep, 0.0) / w_sum

            if mode == "mult":
                shock_total = float(pd.to_numeric(dd[shock_col], errors="coerce").fillna(0.0).sum())
                if abs(shock_total) < 1e-12:
                    continue
                grouped = (
                    dd.assign(__num__=pd.to_numeric(dd[num_col], errors="coerce").fillna(0.0))
                    .groupby("__bucket__")["__num__"]
                    .sum()
                )
                if ind_u == "JOBS":
                    vals = grouped / (shock_total / 1_000_000.0)
                else:
                    vals = grouped / shock_total
            else:
                grouped = (
                    dd.assign(__val__=pd.to_numeric(dd[col], errors="coerce").fillna(0.0))
                    .groupby("__bucket__")["__val__"]
                    .sum()
                )
                if mode == "share":
                    total = float(grouped.sum())
                    vals = grouped / total if abs(total) > 1e-12 else grouped * 0.0
                else:
                    vals = grouped

            for lab, val in vals.items():
                if lab in acc:
                    acc[lab] += w * float(val)

        row = {"Component": str(comp), "Subcomponent": str(sub)}
        for lab in origin_list:
            row[lab] = float(acc.get(lab, 0.0))
        if mode == "share":
            total = sum(row[lab] for lab in origin_list)
            if abs(total) > 1e-12:
                for lab in origin_list:
                    row[lab] = row[lab] / total
        rows.append(row)

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        return pd.DataFrame(columns=["Component", "Subcomponent", *origin_list]), origin_list
    return df_out, origin_list


def compute_industry_group_multipliers_weighted(
    industry_groups: dict,
    connected_country: dict,
    *,
    mrio,
    L1_MRIO,
    VA_MRIO,
    LB_MRIO,
    W_MRIO=None,
    indicator="GVA",
    shock_eur=1_000_000.0,
    include_deployments=None,
    exclude_deployments=(),
    weight_key="Power (MW)",
    weight_mode: str = "capacity",  # "capacity" | "equal"
    level: str = "group",   # "group" | "industry"
):
    """
    Compute EU27/UK/RoW multipliers per industry group using isolated shocks
    (shock_eur per industry code), then average across deployments weighted
    by connected_country[weight_key].
    """
    from modules.mrio_analysis.mrio import compute_full_effects_mrio

    ind_u = str(indicator).upper()
    if ind_u not in {"GVA", "JOBS", "OUTPUT"}:
        raise ValueError("indicator must be 'GVA', 'Jobs', or 'Output'.")

    # deployments + weights
    if include_deployments is None:
        include_deployments = list(connected_country.keys())
    include_set = {str(d) for d in include_deployments}
    excl_set = {str(d) for d in (exclude_deployments or ())}
    dep_list = sorted(include_set - excl_set)

    weights = {str(k): float(v.get(weight_key, 0.0)) for k, v in connected_country.items()}
    if str(weight_mode).lower() == "equal":
        weights = {k: 1.0 for k in dep_list}
    else:
        weights = {k: w for k, w in weights.items() if w > 0 and k in dep_list}

    idx = mrio.index
    if not isinstance(idx, pd.MultiIndex) or idx.nlevels < 2:
        raise ValueError("MRIO index must be a MultiIndex with origin and industry.")

    # origin + industry levels
    origin_lvl = 0
    ind_lvl = 1
    origins = idx.get_level_values(origin_lvl).astype(str)
    industries = idx.get_level_values(ind_lvl).astype(str)

    # filter out ExportROW like other MRIO helpers
    base_idx = idx[origins != "ExportROW"]
    base_zero = pd.Series(0.0, index=base_idx, name="final_demand")

    origin_set = set(base_idx.get_level_values(origin_lvl).astype(str))
    def _dep_code(dep):
        d = str(dep)
        if d in origin_set:
            return d
        if d == "GB" and "UK" in origin_set:
            return "UK"
        if d == "UK" and "GB" in origin_set:
            return "GB"
        return d

    def _group_sum(x: pd.Series) -> pd.Series:
        return x.groupby(level=origin_lvl).sum()

    def _weighted_avg(dep_rows: list[dict]) -> dict | None:
        if not dep_rows:
            return None
        dep_df = pd.DataFrame(dep_rows)
        if str(weight_mode).lower() == "equal":
            dep_df["w"] = 1.0
        else:
            dep_df["w"] = dep_df["Deployment"].map(weights).fillna(0.0)
            dep_df = dep_df[dep_df["w"] > 0]
        if dep_df.empty:
            return None
        wsum = dep_df["w"].sum()
        return {
            "EU27": (dep_df["EU27"] * dep_df["w"]).sum() / wsum,
            "UK":   (dep_df["UK"]   * dep_df["w"]).sum() / wsum,
            "RoW":  (dep_df["RoW"]  * dep_df["w"]).sum() / wsum,
        }

    rows = []
    for gname, codes in industry_groups.items():
        codes = [str(c).strip() for c in codes]

        if level.lower() == "industry":
            for code in codes:
                if code not in set(industries):
                    continue
                dep_rows = []
                for dep in dep_list:
                    dep_code = _dep_code(dep)
                    if dep_code not in origin_set:
                        continue

                    y = base_zero.copy()
                    key = (dep_code, code)
                    if key not in y.index:
                        continue
                    y.loc[key] = float(shock_eur)

                    eff = compute_full_effects_mrio(
                        results_dict=None, country=None, year=None,
                        direct_output=y, L1=L1_MRIO, VA=VA_MRIO, LB=LB_MRIO, W=W_MRIO
                    )
                    if ind_u == "GVA":
                        tot = _group_sum(eff["GVA"]["Total"])
                    elif ind_u == "OUTPUT":
                        tot = _group_sum(eff["Output"]["Total"])
                    else:
                        tot = _group_sum(eff["Jobs"]["Total"])

                    eu = sum(v for k, v in tot.items() if str(k) in EU27)
                    uk = sum(v for k, v in tot.items() if str(k) in {"UK", "GB"})
                    row = sum(v for k, v in tot.items() if str(k) not in EU27 and str(k) not in {"UK", "GB"})

                    if ind_u in {"GVA", "OUTPUT"}:
                        denom = float(shock_eur)
                        eu_m, uk_m, row_m = eu / denom, uk / denom, row / denom
                    else:
                        denom = float(shock_eur) / 1e6
                        eu_m, uk_m, row_m = eu / denom, uk / denom, row / denom

                    dep_rows.append({"Deployment": dep, "EU27": eu_m, "UK": uk_m, "RoW": row_m})

                avg = _weighted_avg(dep_rows)
                if avg is None:
                    continue
                rows.append({"Group": gname, "Industry": code, **avg})

        else:
            dep_rows = []
            for dep in dep_list:
                dep_code = _dep_code(dep)
                if dep_code not in origin_set:
                    continue

                total_effect = {}
                shock_total = 0.0

                for code in codes:
                    if code not in set(industries):
                        continue
                    y = base_zero.copy()
                    key = (dep_code, code)
                    if key not in y.index:
                        continue
                    y.loc[key] = float(shock_eur)
                    shock_total += float(shock_eur)

                    eff = compute_full_effects_mrio(
                        results_dict=None, country=None, year=None,
                        direct_output=y, L1=L1_MRIO, VA=VA_MRIO, LB=LB_MRIO, W=W_MRIO
                    )

                    if ind_u == "GVA":
                        tot = _group_sum(eff["GVA"]["Total"])
                    elif ind_u == "OUTPUT":
                        tot = _group_sum(eff["Output"]["Total"])
                    else:
                        tot = _group_sum(eff["Jobs"]["Total"])

                    for k, v in tot.items():
                        total_effect[k] = total_effect.get(k, 0.0) + float(v)

                if shock_total <= 0:
                    continue

                eu = sum(v for k, v in total_effect.items() if str(k) in EU27)
                uk = sum(v for k, v in total_effect.items() if str(k) in {"UK", "GB"})
                row = sum(v for k, v in total_effect.items() if str(k) not in EU27 and str(k) not in {"UK", "GB"})

                if ind_u in {"GVA", "OUTPUT"}:
                    denom = shock_total
                    eu_m, uk_m, row_m = eu / denom, uk / denom, row / denom
                else:
                    denom = shock_total / 1e6
                    eu_m, uk_m, row_m = eu / denom, uk / denom, row / denom

                dep_rows.append({"Deployment": dep, "EU27": eu_m, "UK": uk_m, "RoW": row_m})

            avg = _weighted_avg(dep_rows)
            if avg is None:
                continue
            rows.append({"Group": gname, **avg})

    return pd.DataFrame(rows)




def compute_industry_effects_by_industry(
    industries: list[str],
    *,
    mrio,
    L1_MRIO,
    VA_MRIO,
    LB_MRIO,
    W_MRIO=None,
    indicator="Output",
    effect="Indirect",
    shock_eur=1_000_000.0,
    include_deployments=None,
):
    """
    Compute country-averaged, global industry multipliers.

    indicator : "Output", "GVA", or "Jobs"
    effect    : "Total", "Direct", or "Indirect"

    Each selected country-industry pair receives the same final
    demand shock. Results are aggregated across producing countries
    and averaged equally across the selected deployment countries.

    Indirect = Total - Direct, including within-industry feedback.
    """
    from modules.mrio_analysis.mrio import compute_full_effects_mrio

    indicator = str(indicator).strip().capitalize()
    effect = str(effect).strip().capitalize()

    if indicator not in {"Output", "GVA", "Jobs"}:
        raise ValueError("indicator must be Output, GVA, or Jobs.")

    if effect not in {"Total", "Direct", "Indirect"}:
        raise ValueError("effect must be Total, Direct, or Indirect.")

    if not industries or not include_deployments:
        raise ValueError(
            "industries and include_deployments must be non-empty."
        )

    if shock_eur <= 0:
        raise ValueError("shock_eur must be positive.")

    idx = mrio.index
    if not isinstance(idx, pd.MultiIndex) or idx.nlevels < 2:
        raise ValueError(
            "MRIO index must be a country-industry MultiIndex."
        )

    origin_codes = idx.get_level_values(0).astype(str)
    base_idx = idx[origin_codes != "ExportROW"]
    available = set(base_idx.tolist())

    industries = list(dict.fromkeys(
        str(c).strip() for c in industries
    ))
    deployments = list(dict.fromkeys(
        str(d).strip() for d in include_deployments
    ))

    missing = [
        (dep, code)
        for dep in deployments
        for code in industries
        if (dep, code) not in available
    ]
    if missing:
        raise ValueError(
            f"Country-industry pairs missing from MRIO: {missing}"
        )

    y_zero = pd.Series(
        0.0, index=base_idx, name="final_demand"
    )

    results = {}

    for code in industries:
        effects = []

        for dep in deployments:
            y = y_zero.copy()
            y.loc[(dep, code)] = float(shock_eur)

            eff = compute_full_effects_mrio(
                results_dict=None,
                country=None,
                year=None,
                direct_output=y,
                L1=L1_MRIO,
                VA=VA_MRIO,
                LB=LB_MRIO,
                W=W_MRIO,
            )

            values = (
                eff[indicator][effect]
                .groupby(level=1)
                .sum()
                .div(float(shock_eur))
            )

            effects.append(values)

        results[code] = (
            pd.concat(effects, axis=1)
            .fillna(0.0)
            .mean(axis=1)
        )

    out = pd.DataFrame(results).T.fillna(0.0)
    out.index.name = "Industry"

    return out


def plot_industry_multipliers_twin(
    df_ind: pd.DataFrame,
    *,
    indicator="Output",
    effect="Indirect",
    industries: list[str] | None = None,
    figsize=(9.0, 5.0),
    fig_w: float | None = 9.0,
    fig_h: float | None = 5.0,
    dpi=150,
    filename="C.1.2.pdf",
    save=None,
    close=None,
    sort_desc=True,
    max_industries: int | None = None,
    label_min=0.0,
    value_decimals=1,
    ind_labels: dict | None = None,
    legend_cap=40,
    legend_wrap: int | None = 50,
    legend_font=7,
    legend_title_font=7,
    legend_x=0.99,
    legend_y=0.5,
    group_label_offset=-0.03,
    show_group_labels: bool = False,
    top_n: int | None = 4,
    color_ind: dict | None = None,
    show_xlabel: bool = False,
    label_min_share: float | None = 0.20,
):
    """
    Twin-format plot to match plot_eu27_labor_capital_ratio, using multipliers.
    """
    df = df_ind.copy()

    indicator = str(indicator).strip().capitalize()
    effect = str(effect).strip().capitalize()

    if effect not in {"Total", "Direct", "Indirect"}:
        raise ValueError("effect must be Total, Direct, or Indirect.")

    if indicator not in {"Output", "GVA", "Jobs"}:
        raise ValueError("indicator must be Output, GVA, or Jobs.")
    
    if label_min_share is not None:
        label_min = float(label_min_share)

    # If df is a matrix (index=Industry, columns=prd_use), plot composition by industry
    if "Industry" not in df.columns:
        if industries is not None:
            keep = [str(x) for x in industries]
            df = df.loc[df.index.astype(str).isin(keep)]

        if fig_w is not None and fig_h is not None:
            figsize = (float(fig_w), float(fig_h))

        df["__parent__"] = df.index.astype(str).str[0]
        num = df.select_dtypes(include=[np.number])
        df["_total"] = num.sum(axis=1)
        if sort_desc:
            # alphabetic within group
            df["_sort_key"] = df.index.astype(str)
            df = df.sort_values(["__parent__", "_sort_key"], ascending=[True, True])
        if max_industries is not None:
            df = df.head(int(max_industries))

        plot_df = df.drop(columns=["__parent__", "_total"], errors="ignore")
        # ensure we only plot numeric industry columns
        plot_df = plot_df.select_dtypes(include=[np.number])

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        y = np.arange(len(plot_df))

        # Palette by parent letter (variations within each letter)
        def _blend(c1: str, c2: str, t: float) -> str:
            h1 = str(c1).lstrip("#"); h2 = str(c2).lstrip("#")
            if len(h1) != 6 or len(h2) != 6:
                return c1
            r1, g1, b1 = int(h1[0:2], 16), int(h1[2:4], 16), int(h1[4:6], 16)
            r2, g2, b2 = int(h2[0:2], 16), int(h2[2:4], 16), int(h2[4:6], 16)
            r = int(r1 + (r2 - r1) * t)
            g = int(g1 + (g2 - g1) * t)
            b = int(b1 + (b2 - b1) * t)
            return f"#{r:02X}{g:02X}{b:02X}"

        def _tint(base: str, amount: float) -> str:
            # blend base toward white
            return _blend(base, "#FFFFFF", amount)

        parent_base = {
            "A": "#2F6F6D", "B": "#6B6B6B", "C": "#1F4D3B", "D": "#7A1F4B",
            "E": "#4C78A8", "F": "#6A4300", "G": "#C07505", "H": "#0B2B63",
            "I": "#4C4C4C", "J": "#1F6F9D", "K": "#DEDB08", "L": "#3D5A80",
            "M": "#6E0606", "N": "#4A7A1B", "O": "#4C1F5C", "P": "#5B5C1F",
            "Q": "#6F1F3C", "R": "#3C6F1F", "S": "#1F3C6F", "T": "#6F3C1F",
            "U": "#3C1F6F",
        }

        def _num_key(code: str) -> int:
            m = re.search(r"(\d+)", str(code))
            return int(m.group(1)) if m else -1

        # precompute row orders (top_n) to build legend-driven palette
        row_orders = []
        used_cols = set()
        for _, row in plot_df.iterrows():
            vals = row.astype(float)
            order = vals.sort_values(ascending=False).index.tolist()
            others_val = 0.0
            if top_n is not None and top_n > 0 and len(order) > top_n:
                keep = order[:int(top_n)]
                drop = order[int(top_n):]
                others_val = float(vals.loc[drop].sum()) if drop else 0.0
                order = keep
            for col in order:
                if float(vals[col]) != 0:
                    used_cols.add(str(col))
            row_orders.append((order, others_val))

        # deterministic colors per code within each parent letter (only legend codes)
        used_cols_order = [c for c in plot_df.columns.astype(str).tolist() if c in used_cols and c != "Others"]
        by_parent = {}
        for c in used_cols_order:
            by_parent.setdefault(c[:1], []).append(c)
        code_color = {}
        for p, codes in by_parent.items():
            if p == "C":
                base_hi = "#043E28"      # exact for C28
                base_lo = "#4A7A1B"      # light green for C<25
                low = [c for c in codes if _num_key(c) != -1 and _num_key(c) <= 24]
                high = [c for c in codes if _num_key(c) != -1 and _num_key(c) >= 25]
                other = [c for c in codes if c not in set(low + high)]
                # C<=24: palette from base_lo (C24 light -> C19 dark)
                if low:
                    nums = [_num_key(c) for c in low if _num_key(c) != -1]
                    n = len(low)
                    if n == 1:
                        code_color[low[0]] = base_lo
                    else:
                        light = _tint(base_lo, 0.70)
                        hi = max(nums) if nums else 24
                        lo = min(nums) if nums else 1
                        for c in low:
                            if str(c) == "C19":
                                code_color[c] = "#000000"
                                continue
                            num = _num_key(c)
                            if num == -1 or hi == lo:
                                t = 0.0
                            else:
                                # C24 light -> lower darker
                                t = (hi - num) / float(hi - lo)
                            code_color[c] = _blend(light, base_lo, t)
                # C>=25: palette from base_hi (C28 exact)
                if high:
                    nums = [_num_key(c) for c in high if _num_key(c) != -1]
                    n = len(high)
                    if n == 1:
                        code_color[high[0]] = base_hi
                    else:
                        light = _tint(base_hi, 0.55)
                        hi = max(nums) if nums else 28
                        lo = min(nums) if nums else 25
                        for c in high:
                            num = _num_key(c)
                            if num == -1 or hi == lo:
                                t = 1.0
                            else:
                                t = (num - lo) / float(hi - lo)  # C25 light -> C28 dark
                            code_color[c] = _blend(light, base_hi, t)
                # override C28 exact color
                if "C28" in codes:
                    code_color["C28"] = base_hi
                for c in other:
                    code_color[c] = base_hi
            else:
                base = parent_base.get(p, "#4C78A8")
                n = len(codes)
                if n == 1:
                    code_color[codes[0]] = base
                else:
                    light_amt = 0.65 if p == "H" else 0.85
                    light = _tint(base, light_amt)
                    for i, c in enumerate(codes):
                        t = i / max(1, n - 1)  # 0 light, 1 dark
                        code_color[c] = _blend(light, base, t)

        # allow explicit color overrides
        if color_ind is None:
            try:
                from modules.mrio_analysis.industrial import COLOR_IND as _COLOR_IND
                color_ind = _COLOR_IND
            except Exception:
                color_ind = COLOR_IND_DEFAULT
        if color_ind:
            for k, v in color_ind.items():
                k = str(k)
                code_color[k] = v
                if not k.startswith("CPA_"):
                    code_color[f"CPA_{k}"] = v

        # draw per-row segments ordered by contribution (desc)
        for i, (order, others_val) in enumerate(row_orders):
            vals = plot_df.iloc[i].astype(float)
            left = 0.0
            for col in order:
                v = float(vals[col])
                if v == 0:
                    continue
                c = code_color.get(str(col), "#B0B0B0")
                ax.barh(i, v, left=left, color=c, edgecolor="none", linewidth=0.0)
                left += v
            if others_val > 0:
                others_color = (color_ind or {}).get("Others", "#D6D6D6")
                ax.barh(i, others_val, left=left, color=others_color, edgecolor="none", linewidth=0.0)
                left += others_val

        ax.set_yticks(y)
        ax.set_yticklabels([tex_safe(str(x)) for x in plot_df.index.tolist()])
        ax.invert_yaxis()
        if show_xlabel:
            labels = {
                "Output": "Output [EUR/EUR]",
                "GVA": "GVA [EUR/EUR]",
                "Jobs": "Employment [FTE-years/mEUR]",
            }

            ax.set_xlabel(
                tex_bold(f"Global {effect.lower()} {labels[indicator]}")
            )

        parents = df["__parent__"].to_numpy()
        bounds = []
        start = 0
        for i in range(1, len(parents)):
            if parents[i] != parents[i-1]:
                bounds.append((parents[i-1], start, i-1))
                start = i
        if len(parents):
            bounds.append((parents[-1], start, len(parents)-1))

        for _, start, end in bounds:
            if end < len(df) - 1:
                ax.axhline(end + 0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.6)

        if show_group_labels:
            for parent, start, end in bounds:
                center = (start + end) / 2
                ax.text(group_label_offset, center, tex_bold(parent), transform=ax.get_yaxis_transform(),
                        ha="right", va="center")

        if ind_labels:
            from matplotlib.patches import Patch
            legend_items = []
            cols = sorted(used_cols)
            for code in cols:
                lab = _industry_label(code, ind_labels)
                lab = str(lab)
                if legend_cap and len(lab) > legend_cap:
                    lab = lab[: legend_cap - 3] + "..."
                if legend_wrap and legend_wrap > 0:
                    lab = "\n".join(textwrap.wrap(lab, width=legend_wrap))
                legend_items.append((code, f"{code} --- {lab}"))
            if top_n is not None and top_n > 0:
                legend_items.append(("Others", "Others"))
            patches = [
                Patch(facecolor=("#B0B0B0" if code == "Others" else code_color.get(str(code), "#B0B0B0")),
                      edgecolor="white", label=tex_safe(lbl))
                for (code, lbl) in legend_items
            ]
            ax.legend(handles=patches, title=tex_bold("Industries in effects"),
                      loc="center left", bbox_to_anchor=(legend_x, legend_y),
                      frameon=True, framealpha=0.9, facecolor="white", edgecolor="#CCCCCC",
                      fontsize=legend_font, title_fontsize=legend_title_font)

        apply_house_style(ax, grid=True)
        ax.grid(axis="x", alpha=0.2, linestyle=":", linewidth=0.5)
        fig.tight_layout()
        if (save is not None) or (close is not None):
            save_and_close(fig=fig, filename=filename, save=save, close=close)
        return fig, ax

    if industries is not None:
        keep = [str(x) for x in industries]
        df = df[df["Industry"].astype(str).isin(keep)].copy()

    if fig_w is not None and fig_h is not None:
        figsize = (float(fig_w), float(fig_h))

    df = df.set_index("Industry")
    df["__parent__"] = df.index.astype(str).str[0]

    ind_u = str(indicator).upper()
    if ind_u not in {"GVA", "JOBS", "OUTPUT"}:
        raise ValueError("indicator must be 'GVA', 'Jobs', or 'Output'.")

    # total multiplier as sum of EU27/UK/RoW
    df["ratio"] = df[["EU27", "UK", "RoW"]].sum(axis=1)

    if sort_desc:
        df = df.sort_values(["__parent__", "ratio"], ascending=[True, False])
    if max_industries is not None:
        df = df.head(int(max_industries))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    y = np.arange(len(df))
    vals = df["ratio"].to_numpy(dtype=float)

    try:
        from modules.mrio_analysis.industrial import _sobrio_palette_for_codes
        code_color = _sobrio_palette_for_codes(df.index.astype(str).tolist())
    except Exception:
        code_color = {}
    colors = [code_color.get(str(c), "#4C78A8") for c in df.index.tolist()]

    ax.barh(y, vals, color=colors, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels([tex_safe(str(x)) for x in df.index.tolist()])
    ax.invert_yaxis()

    # group separators + centered parent labels
    parents = df["__parent__"].to_numpy()
    bounds = []
    start = 0
    for i in range(1, len(parents)):
        if parents[i] != parents[i-1]:
            bounds.append((parents[i-1], start, i-1))
            start = i
    if len(parents):
        bounds.append((parents[-1], start, len(parents)-1))

    for _, start, end in bounds:
        if end < len(df) - 1:
            ax.axhline(end + 0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.6)

    if show_group_labels:
        for parent, start, end in bounds:
            center = (start + end) / 2
            ax.text(group_label_offset, center, tex_bold(parent), transform=ax.get_yaxis_transform(),
                    ha="right", va="center")

    for i, v in enumerate(vals):
        if v < label_min:
            continue
        ax.text(v, i, tex_safe(f"{v:.{value_decimals}f}"),
                va="center", ha="left", fontsize=9)

    if ind_u == "GVA":
        ylab = "GVA multiplier [EUR/EUR]"
    elif ind_u == "OUTPUT":
        ylab = "Output multiplier [EUR/EUR]"
    else:
        ylab = "Employment multiplier [FTE-years/mEUR]"
    ax.set_xlabel(tex_bold(ylab))

    if ind_labels:
        from matplotlib.patches import Patch
        legend_items = []
        for code in df.index.tolist():
            lab = _industry_label(code, ind_labels)
            lab = str(lab)
            if legend_cap and len(lab) > legend_cap:
                lab = lab[: legend_cap - 3] + "..."
            if legend_wrap and legend_wrap > 0:
                lab = "\n".join(textwrap.wrap(lab, width=legend_wrap))
            legend_items.append((code, f"{code} --- {lab}"))
        patches = [
            Patch(facecolor=code_color.get(code, "#B0B0B0"), edgecolor="white", label=tex_safe(lbl))
            for code, lbl in legend_items
        ]
        ax.legend(handles=patches, title=tex_bold("Industry codes"),
                  loc="center left", bbox_to_anchor=(legend_x, legend_y),
                  frameon=True, framealpha=0.9, facecolor="white", edgecolor="#CCCCCC",
                  fontsize=legend_font, title_fontsize=legend_title_font)

    apply_house_style(ax, grid=True)
    ax.grid(axis="x", alpha=0.2, linestyle=":", linewidth=0.5)
    fig.tight_layout()
    if (save is not None) or (close is not None):
        save_and_close(fig=fig, filename=filename, save=save, close=close)
    return fig, ax

def plot_capture_sublevel_avg_connected_power(
    results_by_origin_sub_lvl,
    connected_country,
    *,
    indicator="Jobs",
    typ="Total",
    mode="mult",
    year=2023,
    sector="ALL",
    avg_dep_order=("BE", "NL", "DE", "DK", "GB"),
    weight_key="Power (MW)",
    group_eu_deploy=False,
    eu_deploy_set=("BE", "NL", "DK", "DE"),
    split_origin_countries=("CN",),
    split_origin_country_labels=None,
    figsize=(12.0617, 3.8818),
    dpi=160,
    label_min_share=0.04,
    label_min_abs_gva=0.05,
    label_min_abs_jobs=5.0,
    label_font=10,
    xtick_font=10,
    ytick_font=10,
    ytitle_font=12,
    comp_label_font=14,
    comp_label_offset=-0.35,
    show_value_labels=True,
    show_legend=True,
):
    """Plot weighted-average value capture by component or subcomponent."""
    set_house_style()

    df_abs, origin_list = build_capture_sublevel_avg_connected_power_df(
        results_by_origin_sub_lvl,
        connected_country,
        indicator=indicator,
        typ=typ,
        mode=mode,
        year=year,
        sector=sector,
        avg_dep_order=avg_dep_order,
        weight_key=weight_key,
        group_eu_deploy=group_eu_deploy,
        eu_deploy_set=eu_deploy_set,
        split_origin_countries=split_origin_countries,
        split_origin_country_labels=split_origin_country_labels,
    )

    split_origin_country_labels = split_origin_country_labels or {}
    split_origin_countries = tuple(
        str(c).strip().upper()
        for c in (split_origin_countries or ())
        if str(c).strip().upper() not in {"ROW", "X01"}
    )
    split_labels = [_capture_split_origin_label(c, split_origin_country_labels) for c in split_origin_countries]

    eu_label = (
        EU_NONDEPLOY_LABEL
        if group_eu_deploy
        else OTHER_EU27_LABEL if any(c in EU27 for c in split_origin_countries) else EU_LABEL
    )
    split_label_to_iso = dict(zip(split_labels, split_origin_countries))
    eu_base = COLOR_MAP.get("EU27") or "#33467F"

    def _country_color(label):
        """Resolve the same shared country color used by weighted sourcing."""
        if label == eu_label:
            return eu_base
        if label == EU_DEPLOY_LABEL:
            return COLOR_MAP.get("EU_dep") or _lighten(eu_base, 0.35)
        if label == UK_LABEL:
            return COLOR_MAP.get("GB") or "#BDBDBD"
        iso = split_label_to_iso.get(label, label)
        return COLOR_MAP.get(iso) or COLOR_MAP.get(label) or "#BDBDBD"

    def _aggregate_style():
        return dict(fc="white", ec="black", ls=(0, (3, 2)), lw=0.6)

    def _style(label):
        if label == ROW_LABEL:
            return _aggregate_style()
        if label == OTHER_ORIGINS_LABEL:
            return dict(fc="black", ec="black", ls="solid", lw=0.6)
        if label == OTHER_EU27_LABEL:
            return dict(fc=_lighten(eu_base, 0.45), ec="#6B7FA3", ls="solid", lw=0.6)
        color = _country_color(label)
        return dict(fc=color, ec=color, ls="solid", lw=0.6)

    ind_u = str(indicator).upper()
    if mode == "share":
        label_min_abs = float(label_min_share)
    else:
        label_min_abs = float(label_min_abs_gva) if ind_u == "GVA" else float(label_min_abs_jobs)

    def _fmt(v):
        if mode == "share":
            return f"{float(v)*100:.0f}%"
        if mode == "total" and ind_u in {"OUTPUT", "GVA", "INCOME"}:
            return f"{float(v):,.1f}"
        if ind_u == "JOBS":
            return f"{float(v):,.0f}"
        return f"{float(v):,.1f}"

    df_abs = df_abs.copy()
    df_abs["Component"] = df_abs["Component"].astype(str)
    df_abs["Subcomponent"] = df_abs["Subcomponent"].astype(str)

    pref = ["Devex", "Turbine", "BoP", "Installation", "O&M"]
    comp_order = [c for c in pref if c in df_abs["Component"].unique().tolist()]
    comp_order += [c for c in df_abs["Component"].unique().tolist() if c not in comp_order]

    x, xt, centers, seps, rows, k = [], [], [], [], [], 0
    for gi, comp in enumerate(comp_order):
        subdf = df_abs[df_abs["Component"] == comp].sort_values(["Subcomponent"]).copy()
        if subdf.empty:
            continue
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
    # Keep the stacked order aligned with the weighted-sourcing convention:
    # EU27, explicitly split countries, then the remaining country groups.
    plot_origin_order = [
        label
        for label in (
            [eu_label]
            + split_labels
            + [EU_DEPLOY_LABEL, UK_LABEL, ROW_LABEL, OTHER_ORIGINS_LABEL]
        )
        if label in origin_list
    ]
    plot_origin_order += [label for label in origin_list if label not in plot_origin_order]
    active_labels = [
        label for label in plot_origin_order
        if float(np.nanmax(np.abs(plot_df[label].to_numpy(dtype=float)))) > 1e-12
    ]
    plot_origin_order = active_labels
    vals = plot_df[plot_origin_order].to_numpy(dtype=float)
    xs = np.asarray(x[:vals.shape[0]], dtype=float)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    bottoms = np.zeros(vals.shape[0])

    for j, lab in enumerate(plot_origin_order):
        seg = vals[:, j]
        st = _style(lab)
        ax.bar(
            xs,
            seg,
            bottom=bottoms,
            width=0.85,
            color=st["fc"],
            edgecolor=st["ec"],
            linewidth=st["lw"],
            linestyle=st["ls"],
        )
        bottoms += seg

    if show_value_labels:
        for i in range(vals.shape[0]):
            cum = 0.0
            for j, lab in enumerate(plot_origin_order):
                w = float(vals[i, j])
                if abs(w) < 1e-12:
                    continue
                if (
                    mode == "share" and w < label_min_share
                ) or (
                    mode != "share"
                    and lab in {UK_LABEL, OTHER_ORIGINS_LABEL, ROW_LABEL, OTHER_EU27_LABEL, *split_labels}
                    and abs(w) < label_min_abs
                ):
                    cum += w
                    continue
                st = _style(lab)
                tc = "white" if lab == OTHER_ORIGINS_LABEL else ("black" if lab == ROW_LABEL else get_text_color(st["fc"]))
                ax.text(
                    xs[i],
                    cum + w / 2,
                    tex_safe(_fmt(w)),
                    ha="center",
                    va="center",
                    fontsize=label_font,
                    color=tc,
                )
                cum += w

    for s in seps:
        ax.axvline(s, color="black", linewidth=1.0, linestyle="--", alpha=0.6)

    if mode == "share":
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels([tex_safe(x) for x in ["0%", "25%", "50%", "75%", "100%"]], fontsize=ytick_font)
    else:
        totals = vals.sum(axis=1)
        ymax = max(float(np.max(totals)) if len(totals) else 0.0, 1.0)
        ax.set_ylim(0, ymax * 1.10)
        ax.tick_params(axis="y", labelsize=ytick_font)

    if mode == "mult":
        if ind_u == "GVA":
            ylab = "GVA multiplier [EUR/EUR]"
        elif ind_u == "JOBS":
            ylab = "Employment multiplier [FTE-years/mEUR]"
        else:
            ylab = f"{indicator} captured by origin group"
    elif mode == "share":
        ylab = f"{indicator} [%]"
    else:
        ylab = f"{indicator} [mEUR]" if ind_u in {"OUTPUT", "GVA", "INCOME"} else "Employment [FTE-years]"
    ax.set_ylabel(tex_bold(ylab), fontsize=ytitle_font)

    ax.set_xticks(xs)
    xt_disp = [_short_subcomponent_label(t) for t in xt[:vals.shape[0]]]
    ax.set_xticklabels([tex_safe(t) for t in xt_disp], fontsize=xtick_font, rotation=45, ha="right")
    ax.tick_params(axis="x", length=0)

    for comp, c in centers:
        ax.text(
            c,
            comp_label_offset,
            tex_safe(str(comp)),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=comp_label_font,
            fontweight="bold",
        )

    patches = []
    legend_labels = {
        eu_label: eu_label,
        UK_LABEL: CODE_TO_NAME.get("GB", UK_LABEL),
    }
    patches = []
    for lab in plot_origin_order:
        st = _style(lab)
        patches.append(mpatches.Patch(
            facecolor=st["fc"],
            edgecolor=st["ec"],
            linewidth=st["lw"],
            linestyle=st["ls"],
            label=tex_safe(legend_labels.get(lab, lab)),
        ))

    if show_legend and patches:
        if mode == "mult":
            ax.legend(
                handles=patches,
                title=tex_bold("Origin"),
                loc="upper right",
                fontsize=8,
                title_fontsize=8,
                frameon=True,
                framealpha=0.9,
                facecolor="white",
                edgecolor="#CCCCCC",
            )
        else:
            ax.legend(
                handles=patches,
                title=tex_bold("Origin"),
                bbox_to_anchor=(0.98, 0.5),
                loc="center left",
                fontsize=8,
                title_fontsize=8,
                frameon=True,
                framealpha=0.9,
                facecolor="white",
                edgecolor="#CCCCCC",
            )

    apply_house_style(ax, grid=True)
    ax.grid(axis="y", alpha=0.2, linestyle=":", linewidth=0.5)
    ax.set_title(None)
    fig.subplots_adjust(bottom=0.32)
    fig.tight_layout()
    return fig, ax



def plot_and_save_capture_sublevel_avg_connected_power(
    results_by_origin_sub_lvl,
    connected_country,
    *,
    indicators=("Jobs", "GVA"),
    year=2023,
    sector="ALL",
    typ="Total",
    mode="mult",
    avg_dep_order=("BE", "NL", "DE", "DK", "GB"),
    weight_key="Power (MW)",
    group_eu_deploy=False,
    eu_deploy_set=("BE", "NL", "DK", "DE"),
    split_origin_countries=("CN",),
    split_origin_country_labels=None,
    prefix="5.4_EU_capture_sublevel_avg",
    show_value_labels=True,
    show_legend=True,
    save,
    close,
    **kw,
):
    outs = []
    for ind in indicators:
        fig, _ = plot_capture_sublevel_avg_connected_power(
            results_by_origin_sub_lvl,
            connected_country,
            indicator=ind,
            typ=typ,
            mode=mode,
            show_value_labels=show_value_labels,
            show_legend=show_legend,
            year=year,
            sector=sector,
            avg_dep_order=avg_dep_order,
            weight_key=weight_key,
            group_eu_deploy=group_eu_deploy,
            eu_deploy_set=eu_deploy_set,
            split_origin_countries=split_origin_countries,
            split_origin_country_labels=split_origin_country_labels,
            **kw,
        )
        outs.append(save_and_close(fig=fig, filename=f"{prefix}_{ind}.pdf", save=save, close=close))
    return outs


def plot_and_save_eu27_capture_share_figures(
    results_by_origin_sub_lvl: dict,
    connected_country: dict,
    *,
    indicators=("Jobs", "GVA"),
    year: int = 2023,
    sector: str = "ALL",
    avg_dep_order=("BE", "NL", "DE", "DK", "GB"),
    split_origin_countries=("CN",),
    component_level: int = 1,
    turbine_subcomponent_level: int = 0,
    turbine_components=("Blades", "Hub", "Drivetrain", "Nacelle", "Tower"),
    component_prefix="3.2_EU_capture_comp_share_avg",
    turbine_prefix="3.3_EU_capture_subcomp_share_avg",
    save=None,
    close=None,
):
    """Save the paper's EU27 value-capture share figures."""
    common = dict(
        connected_country=connected_country,
        indicators=indicators,
        year=year,
        sector=sector,
        group_eu_deploy=False,
        avg_dep_order=avg_dep_order,
        mode="share",
        split_origin_countries=split_origin_countries,
        show_value_labels=True,
        label_min_abs_gva=0.1,
        label_min_abs_jobs=0.1,
        save=save,
        close=close,
    )

    component_paths = plot_and_save_capture_sublevel_avg_connected_power(
        results_by_origin_sub_lvl=results_by_origin_sub_lvl[component_level],
        prefix=component_prefix,
        comp_label_offset=-0.32,
        **common,
    )

    turbine_set = set(turbine_components)
    turbine_results = {
        key: value
        for key, value in results_by_origin_sub_lvl[turbine_subcomponent_level].items()
        if key[0] in turbine_set
    }
    turbine_paths = plot_and_save_capture_sublevel_avg_connected_power(
        results_by_origin_sub_lvl=turbine_results,
        prefix=turbine_prefix,
        label_font=8,
        comp_label_offset=-0.34,
        **common,
    )

    return {
        "component": component_paths,
        "turbine_subcomponent": turbine_paths,
    }

### RESEARCH QUESTION 2: STRATEGIC COMPONENTS



def compute_strategic_ratios(
    results_by_origin: dict,
    eu_origins,
    core_deployment,                 # <-- NEW: deployment countries
    year=2023,
    sector="ALL",
    year_col="Year",
    sector_col="Sector",
    deployment_col="Deployment",
    origin_col="Country",            # country where effects materialize
    output_direct_col="Output_Direct",
    output_indirect_col="Output_Indirect",
):
    eu_set = set(map(str, eu_origins))
    dep_set = set(map(str, core_deployment))

    rows = []

    for comp, df in results_by_origin.items():
        if str(comp).strip().lower() == "total":
            continue

        tmp = df.copy()

        if year is not None and year_col in tmp.columns:
            tmp = tmp[tmp[year_col] == year]
        if sector is not None and sector_col in tmp.columns:
            tmp = tmp[tmp[sector_col].astype(str) == str(sector)]

        # Required columns
        missing = [c for c in (output_direct_col, output_indirect_col) if c not in tmp.columns]
        if missing:
            raise KeyError(f"[{comp}] Missing columns: {missing}")

        # Region flags
        tmp["_is_eu"] = tmp[origin_col].astype(str).isin(eu_set)
        tmp["_is_dep_ctry"] = tmp[origin_col].astype(str).isin(dep_set)

        g = tmp.groupby(deployment_col, as_index=False)

        # Global direct/indirect
        GD = g[[output_direct_col]].sum(numeric_only=True).rename(columns={output_direct_col: "GD"})
        GI = g[[output_indirect_col]].sum(numeric_only=True).rename(columns={output_indirect_col: "GI"})

        # EU totals (direct/indirect)
        EU_D = (
            tmp[tmp["_is_eu"]]
            .groupby(deployment_col, as_index=False)[[output_direct_col]]
            .sum(numeric_only=True)
            .rename(columns={output_direct_col: "EU_D"})
        )
        EU_I = (
            tmp[tmp["_is_eu"]]
            .groupby(deployment_col, as_index=False)[[output_indirect_col]]
            .sum(numeric_only=True)
            .rename(columns={output_indirect_col: "EU_I"})
        )

        # RoW totals (direct/indirect)
        RoW_D = (
            tmp[~tmp["_is_eu"]]
            .groupby(deployment_col, as_index=False)[[output_direct_col]]
            .sum(numeric_only=True)
            .rename(columns={output_direct_col: "RoW_D"})
        )
        RoW_I = (
            tmp[~tmp["_is_eu"]]
            .groupby(deployment_col, as_index=False)[[output_indirect_col]]
            .sum(numeric_only=True)
            .rename(columns={output_indirect_col: "RoW_I"})
        )

        # EU split: deployment vs non-deployment (direct + indirect)
        EU_D_dep = (
            tmp[tmp["_is_eu"] & tmp["_is_dep_ctry"]]
            .groupby(deployment_col, as_index=False)[[output_direct_col]]
            .sum(numeric_only=True)
            .rename(columns={output_direct_col: "EU_D_dep"})
        )
        EU_D_nondep = (
            tmp[tmp["_is_eu"] & ~tmp["_is_dep_ctry"]]
            .groupby(deployment_col, as_index=False)[[output_direct_col]]
            .sum(numeric_only=True)
            .rename(columns={output_direct_col: "EU_D_nondep"})
        )
        EU_I_dep = (
            tmp[tmp["_is_eu"] & tmp["_is_dep_ctry"]]
            .groupby(deployment_col, as_index=False)[[output_indirect_col]]
            .sum(numeric_only=True)
            .rename(columns={output_indirect_col: "EU_I_dep"})
        )
        EU_I_nondep = (
            tmp[tmp["_is_eu"] & ~tmp["_is_dep_ctry"]]
            .groupby(deployment_col, as_index=False)[[output_indirect_col]]
            .sum(numeric_only=True)
            .rename(columns={output_indirect_col: "EU_I_nondep"})
        )

        out = GD.merge(GI, on=deployment_col, how="outer")
        out = out.merge(EU_D, on=deployment_col, how="left")
        out = out.merge(EU_I, on=deployment_col, how="left")
        out = out.merge(RoW_D, on=deployment_col, how="left")
        out = out.merge(RoW_I, on=deployment_col, how="left")
        out = out.merge(EU_D_dep, on=deployment_col, how="left")
        out = out.merge(EU_D_nondep, on=deployment_col, how="left")
        out = out.merge(EU_I_dep, on=deployment_col, how="left")
        out = out.merge(EU_I_nondep, on=deployment_col, how="left")

        out = out.fillna({
            "EU_D": 0.0, "EU_I": 0.0, "RoW_D": 0.0, "RoW_I": 0.0,
            "EU_D_dep": 0.0, "EU_D_nondep": 0.0, "EU_I_dep": 0.0, "EU_I_nondep": 0.0
        })

        # Totals
        out["Gx"] = out["GD"] + out["GI"]
        out["Ex"] = out["EU_D"] + out["EU_I"]
        out["Rx"] = out["RoW_D"] + out["RoW_I"]

        # EU non-deployment totals (spillovers target)
        out["EU_nondep_x"] = out["EU_D_nondep"] + out["EU_I_nondep"]
        out["EU_dep_x"] = out["EU_D_dep"] + out["EU_I_dep"]

        # Ratios aligned with your methodology
        out["LC"] = out["EU_D"] / out["GD"].replace(0, np.nan)          # LC_c = E^{D,x}/G^{D,x}
        out["I"]  = out["Gx"]   / out["GD"].replace(0, np.nan)          # I_c  = G^{x}/G^{D,x}
        out["EU_sp"] = out["EU_nondep_x"] / out["Ex"].replace(0, np.nan)
        out["EU_cap"]  = out["Ex"] / out["Gx"].replace(0, np.nan)
        out["r"]  = out["Rx"]   / out["Gx"].replace(0, np.nan)          # r_c  = R^{x}/G^{x}

        out["Component"] = str(comp)
        rows.append(out)

    ratios = pd.concat(rows, ignore_index=True)

    # Weight share per deployment scenario (using direct output = expenditure proxy)
    denom = (
        ratios.groupby([deployment_col], as_index=False)["GD"]
        .sum()
        .rename(columns={"GD": "GD_total_non_total"})
    )
    ratios = ratios.merge(denom, on=[deployment_col], how="left")
    ratios["w_share"] = ratios["GD"] / ratios["GD_total_non_total"].replace(0, np.nan)

    ratios = ratios.rename(columns={deployment_col: "Deployment"})
    ratios = ratios[[
        "Component", "Deployment",
        "GD", "GI", "Gx",
        "EU_D", "EU_I", "Ex",
        "RoW_D", "RoW_I", "Rx",
        "EU_D_dep", "EU_I_dep", "EU_dep_x",
        "EU_D_nondep", "EU_I_nondep", "EU_nondep_x",
        "LC", "I", "r",
        "EU_sp",
        "EU_cap",
        "w_share"
    ]].sort_values(["Deployment", "Component"])

    return ratios


def compute_strategic_ratios_subcomponents(
    results_by_origin_sub: dict,
    eu_origins,
    core_deployment,
    year=2023,
    sector="ALL",
    year_col="Year",
    sector_col="Sector",
    deployment_col="Deployment",
    origin_col="Country",
    output_direct_col="Output_Direct",
    output_indirect_col="Output_Indirect",
    # --- EU27 capture (optional) ---
    capture_metrics: tuple[str, ...] = ("Output_Total", "GVA_Total", "Jobs_Total"),
    capture_connected_country: dict | None = None,
    capture_weight_key: str = "Power (MW)",
    capture_exclude_deployments=("GB",),
    capture_from_scenarios: dict | None = None,
    capture_deltas: tuple[int, ...] | None = None,
    capture_lvl: int | None = None,
    capture_as_percent: bool = False,
    capture_percent_decimals: int = 1,
    capture_prefix: str = "EU27_capture",
    capture_include_base: bool = True,
):
    eu_set = set(map(str, eu_origins))
    dep_set = set(map(str, core_deployment))

    rows = []

    for key, df in results_by_origin_sub.items():
        # Key parsing: keep BOTH columns
        if isinstance(key, tuple) and len(key) >= 2:
            parent_component = str(key[0]).strip()
            subcomponent = str(key[1]).strip()
        else:
            parent_component = "Unknown"
            subcomponent = str(key).strip()

        # Skip totals and self-aggregate rows at either level
        if (
            parent_component.casefold() in {"all", "total"}
            or subcomponent.casefold() in {"all", "total"}
            or parent_component.casefold() == subcomponent.casefold()
        ):
            continue

        tmp = df.copy()

        if year is not None and year_col in tmp.columns:
            tmp = tmp[tmp[year_col] == year]
        if sector is not None and sector_col in tmp.columns:
            tmp = tmp[tmp[sector_col].astype(str) == str(sector)]

        missing = [c for c in (output_direct_col, output_indirect_col) if c not in tmp.columns]
        if missing:
            raise KeyError(f"[{key}] Missing columns: {missing}")

        # Flags
        tmp["_is_eu"] = tmp[origin_col].astype(str).isin(eu_set)
        tmp["_is_dep_ctry"] = tmp[origin_col].astype(str).isin(dep_set)

        g = tmp.groupby(deployment_col, as_index=False)

        # Global direct/indirect
        GD = g[[output_direct_col]].sum(numeric_only=True).rename(columns={output_direct_col: "GD"})
        GI = g[[output_indirect_col]].sum(numeric_only=True).rename(columns={output_indirect_col: "GI"})

        # EU totals (direct/indirect)
        EU_D = (
            tmp[tmp["_is_eu"]]
            .groupby(deployment_col, as_index=False)[[output_direct_col]]
            .sum(numeric_only=True)
            .rename(columns={output_direct_col: "EU_D"})
        )
        EU_I = (
            tmp[tmp["_is_eu"]]
            .groupby(deployment_col, as_index=False)[[output_indirect_col]]
            .sum(numeric_only=True)
            .rename(columns={output_indirect_col: "EU_I"})
        )

        # RoW totals (direct/indirect)
        RoW_D = (
            tmp[~tmp["_is_eu"]]
            .groupby(deployment_col, as_index=False)[[output_direct_col]]
            .sum(numeric_only=True)
            .rename(columns={output_direct_col: "RoW_D"})
        )
        RoW_I = (
            tmp[~tmp["_is_eu"]]
            .groupby(deployment_col, as_index=False)[[output_indirect_col]]
            .sum(numeric_only=True)
            .rename(columns={output_indirect_col: "RoW_I"})
        )

        # EU split: deployment vs non-deployment (direct + indirect)
        EU_D_dep = (
            tmp[tmp["_is_eu"] & tmp["_is_dep_ctry"]]
            .groupby(deployment_col, as_index=False)[[output_direct_col]]
            .sum(numeric_only=True)
            .rename(columns={output_direct_col: "EU_D_dep"})
        )
        EU_D_nondep = (
            tmp[tmp["_is_eu"] & ~tmp["_is_dep_ctry"]]
            .groupby(deployment_col, as_index=False)[[output_direct_col]]
            .sum(numeric_only=True)
            .rename(columns={output_direct_col: "EU_D_nondep"})
        )
        EU_I_dep = (
            tmp[tmp["_is_eu"] & tmp["_is_dep_ctry"]]
            .groupby(deployment_col, as_index=False)[[output_indirect_col]]
            .sum(numeric_only=True)
            .rename(columns={output_indirect_col: "EU_I_dep"})
        )
        EU_I_nondep = (
            tmp[tmp["_is_eu"] & ~tmp["_is_dep_ctry"]]
            .groupby(deployment_col, as_index=False)[[output_indirect_col]]
            .sum(numeric_only=True)
            .rename(columns={output_indirect_col: "EU_I_nondep"})
        )

        out = GD.merge(GI, on=deployment_col, how="outer")
        out = out.merge(EU_D, on=deployment_col, how="left")
        out = out.merge(EU_I, on=deployment_col, how="left")
        out = out.merge(RoW_D, on=deployment_col, how="left")
        out = out.merge(RoW_I, on=deployment_col, how="left")
        out = out.merge(EU_D_dep, on=deployment_col, how="left")
        out = out.merge(EU_D_nondep, on=deployment_col, how="left")
        out = out.merge(EU_I_dep, on=deployment_col, how="left")
        out = out.merge(EU_I_nondep, on=deployment_col, how="left")

        out = out.fillna({
            "EU_D": 0.0, "EU_I": 0.0, "RoW_D": 0.0, "RoW_I": 0.0,
            "EU_D_dep": 0.0, "EU_D_nondep": 0.0, "EU_I_dep": 0.0, "EU_I_nondep": 0.0
        })

        # Totals
        out["Gx"] = out["GD"] + out["GI"]
        out["Ex"] = out["EU_D"] + out["EU_I"]
        out["Rx"] = out["RoW_D"] + out["RoW_I"]

        # EU non-deployment totals
        out["EU_nondep_x"] = out["EU_D_nondep"] + out["EU_I_nondep"]
        out["EU_dep_x"] = out["EU_D_dep"] + out["EU_I_dep"]

        # Ratios
        out["LC"] = out["EU_D"] / out["GD"].replace(0, np.nan)          # LC_c = E^{D,x}/G^{D,x}
        out["I"]  = out["Gx"]   / out["GD"].replace(0, np.nan)          # I_c  = G^{x}/G^{D,x}
        out["EU_sp"] = out["EU_nondep_x"] / out["Ex"].replace(0, np.nan)
        out["EU_cap"]  = out["Ex"] / out["Gx"].replace(0, np.nan)
        out["r"]  = out["Rx"]   / out["Gx"].replace(0, np.nan)          # r_c  = R^{x}/G^{x}

        # KEEP BOTH identifiers
        out["Component"] = parent_component
        out["Subcomponent"] = subcomponent

        rows.append(out)

    ratios = pd.concat(rows, ignore_index=True)

    # w_share: same definition/name as lvl2 (within each deployment scenario)
    denom = (
        ratios.groupby([deployment_col], as_index=False)["GD"]
        .sum()
        .rename(columns={"GD": "GD_total_non_total"})
    )
    ratios = ratios.merge(denom, on=[deployment_col], how="left")
    ratios["w_share"] = ratios["GD"] / ratios["GD_total_non_total"].replace(0, np.nan)

    ratios = ratios.rename(columns={deployment_col: "Deployment"})

    # Output columns: same metrics as ratios_lvl2, plus Subcomponent
    ratios = ratios[[
        "Component", "Subcomponent", "Deployment",
        "GD", "GI", "Gx",
        "EU_D", "EU_I", "Ex",
        "RoW_D", "RoW_I", "Rx",
        "EU_D_dep", "EU_I_dep", "EU_dep_x",
        "EU_D_nondep", "EU_I_nondep", "EU_nondep_x",
        "LC", "I", "r",
        "EU_sp",
        "EU_cap",
        "w_share"
    ]].sort_values(["Deployment", "Component", "Subcomponent"])

    # ------------------------------------------------------------------
    # EU27 capture (weighted across deployments) for Output/GVA/Jobs totals
    # ------------------------------------------------------------------
    def _compute_capture(results_by_origin_sub_local: dict) -> pd.DataFrame:
        if not results_by_origin_sub_local:
            return pd.DataFrame()

        # Precompute union of deployments per component (after filters)
        dep_list_by_comp: dict[str, set[str]] = {}
        for key, df in results_by_origin_sub_local.items():
            if not (isinstance(key, tuple) and len(key) >= 2):
                continue
            parent_component = str(key[0]).strip()
            subcomponent = str(key[1]).strip()
            if (
                parent_component.casefold() in {"total", "all"}
                or subcomponent.casefold() in {"total", "all"}
                or parent_component.casefold() == subcomponent.casefold()
            ):
                continue

            d0 = df.copy()
            if year is not None and year_col in d0.columns:
                d0 = d0[d0[year_col] == year]
            if sector is not None and sector_col in d0.columns:
                d0 = d0[d0[sector_col].astype(str) == str(sector)]
            if d0.empty:
                continue

            deps = d0[deployment_col].astype(str).str.strip().str.upper().unique().tolist()
            dep_list_by_comp.setdefault(parent_component, set()).update(deps)

        eu_set_local = set(map(str, eu_origins))
        excl_set = {str(x).strip().upper() for x in capture_exclude_deployments}

        def _bucket_region(country: str) -> str:
            c = str(country).strip().upper()
            if c == "GB":
                return "United Kingdom"
            if c in eu_set_local:
                return "EU27"
            return "RoW"

        def _weight(dep: str) -> float:
            if capture_connected_country is None:
                return 1.0
            d = str(dep).strip().upper()
            return float(capture_connected_country.get(d, {}).get(capture_weight_key, 0.0))

        rows_cap = []

        for key, df in results_by_origin_sub_local.items():
            if not (isinstance(key, tuple) and len(key) >= 2):
                continue
            parent_component = str(key[0]).strip()
            subcomponent = str(key[1]).strip()
            if (
                parent_component.casefold() in {"total", "all"}
                or subcomponent.casefold() in {"total", "all"}
                or parent_component.casefold() == subcomponent.casefold()
            ):
                continue

            d = df.copy()
            if year is not None and year_col in d.columns:
                d = d[d[year_col] == year]
            if sector is not None and sector_col in d.columns:
                d = d[d[sector_col].astype(str) == str(sector)]
            if d.empty:
                continue

            missing = [m for m in capture_metrics if m not in d.columns]
            if missing:
                raise KeyError(f"[{key}] Missing columns: {missing}")

            d[deployment_col] = d[deployment_col].astype(str).str.strip().str.upper()
            d[origin_col] = d[origin_col].astype(str).str.strip().str.upper()
            d["Region"] = d[origin_col].map(_bucket_region)

            deps_all = dep_list_by_comp.get(parent_component, set()) or set(d[deployment_col].unique())
            dep_list = [x for x in sorted(deps_all) if x not in excl_set]
            if not dep_list:
                dep_list = list(sorted(deps_all))

            w_map = {dep: _weight(dep) for dep in dep_list}
            w_sum = float(sum(w_map.values()))
            if w_sum <= 0:
                w_map = {dep: 1.0 for dep in dep_list}
                w_sum = float(len(dep_list)) if dep_list else 1.0

            eu_tot = {m: 0.0 for m in capture_metrics}
            gl_tot = {m: 0.0 for m in capture_metrics}

            for dep in dep_list:
                w = float(w_map.get(dep, 0.0)) / float(w_sum)
                dd = d[d[deployment_col] == dep]
                if dd.empty:
                    continue

                g = dd.groupby("Region", as_index=False)[list(capture_metrics)].sum()
                total_vals = g[list(capture_metrics)].sum()
                eu_vals = g[g["Region"] == "EU27"][list(capture_metrics)].sum()

                for m in capture_metrics:
                    eu_tot[m] += w * float(eu_vals.get(m, 0.0))
                    gl_tot[m] += w * float(total_vals.get(m, 0.0))

            row = {"Component": parent_component, "Subcomponent": subcomponent}
            for m in capture_metrics:
                denom = gl_tot[m]
                row[f"{capture_prefix}_{m}"] = (eu_tot[m] / denom) if denom else 0.0
            rows_cap.append(row)

        out_cap = pd.DataFrame(rows_cap)
        if out_cap.empty:
            return out_cap

        if capture_as_percent:
            for m in capture_metrics:
                col = f"{capture_prefix}_{m}"
                out_cap[col] = (out_cap[col] * 100.0).round(int(capture_percent_decimals))

        return out_cap

    # Base capture from provided results_by_origin_sub
    if capture_include_base:
        cap_base = _compute_capture(results_by_origin_sub)
        if not cap_base.empty:
            ratios = ratios.merge(cap_base, on=["Component", "Subcomponent"], how="left")

    # Optional: capture from scenarios (multiple deltas)
    if capture_from_scenarios is not None:
        deltas = capture_deltas if capture_deltas is not None else (0, 100)

        def _resolve_results_by_origin_sub(delta: int) -> dict | None:
            res = capture_from_scenarios.get(delta)
            if res is None:
                return None
            if isinstance(res, dict) and "results_by_origin_sub_lvl" in res:
                if capture_lvl is None:
                    raise ValueError("capture_lvl is required when capture_from_scenarios uses results_by_origin_sub_lvl.")
                return res["results_by_origin_sub_lvl"].get(capture_lvl)
            return res

        for dlt in deltas:
            rbos = _resolve_results_by_origin_sub(int(dlt))
            if not rbos:
                continue
            cap_df = _compute_capture(rbos)
            if cap_df.empty:
                continue

            suffix = f"_d{int(dlt)}"
            rename_map = {
                f"{capture_prefix}_{m}": f"{capture_prefix}_{m}{suffix}"
                for m in capture_metrics
            }
            cap_df = cap_df.rename(columns=rename_map)
            ratios = ratios.merge(cap_df, on=["Component", "Subcomponent"], how="left")

    return ratios

import textwrap

