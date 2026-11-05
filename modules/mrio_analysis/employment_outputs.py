"""Presentation, reporting, supplementary export, and visualization outputs.

This module owns employment-facing tables, the C2 workbook, descriptive wage
reporting, and employment-intensity figures.  The source harmonization and
FTE40 allocation methodology remain in :mod:`employment`.
"""

from __future__ import annotations

import hashlib
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modules.config import HELPERS_DIR
from modules.latex_style import apply_house_style, save_and_close, set_house_style, tex_safe
from .employment import (
    _a14_groups_for_code,
    _add_cpa_prefix,
    _aggregate_coverage_map,
    _aggregate_source_priority,
    _canon_map,
    _c2_effective_blocks,
    _clean_industry_code,
    _detect_aggregate_profile,
    _effective_resolution_metadata,
    _is_industry_matrix,
    _isic3_l2_to_nace_rev2_a64_map,
    _isic4_l2_to_nace_rev2_a64_map,
    _split_coverage_codes,
    _split_coverage_string,
    _standardize_fte_source,
    _strip_cpa_prefix,
    aggregate_coverage,
    class_prd_ava,
)


def build_fte40_source_decision_report(
    explicit_geos,
    *,
    employment_industry: pd.DataFrame | None,
    hours_total: pd.DataFrame | None,
    hours_mean: pd.DataFrame | None,
    target_industries: set[str],
) -> pd.DataFrame:
    """Summarize the labour sources actually passed to the FTE40 builder."""
    def summary(frame, geo, value_name, fallback):
        if frame is None or frame.empty or "geo" not in frame.columns:
            return fallback, pd.NA, 0
        group = frame.loc[frame["geo"].astype(str).str.upper().eq(geo)].copy()
        if group.empty:
            return fallback, pd.NA, 0
        metadata, _ = _effective_resolution_metadata(
            group,
            target_industries=target_industries,
            value_name=value_name,
            source_name=fallback,
        )
        sources = sorted(
            value for value in group.get("source_db", pd.Series(dtype=str)).dropna().astype(str).unique()
        )
        years = pd.to_numeric(
            group.get("year", pd.Series(dtype=float)), errors="coerce"
        ).dropna().astype(int).unique()
        source = "; ".join(sources) if sources else fallback
        year = int(years[0]) if len(years) == 1 else "; ".join(map(str, sorted(years))) if len(years) else pd.NA
        return source, year, int(metadata.get("effective_resolution", 0))

    rows = []
    for geo in sorted({str(value).strip().upper() for value in explicit_geos}):
        employment_source, employment_year, employment_resolution = summary(
            employment_industry, geo, "employment", "Unavailable"
        )
        hours_source, hours_year, hours_resolution = summary(
            hours_total, geo, "hours", "Unavailable"
        )
        if hours_source == "Unavailable":
            hours_source, hours_year, hours_resolution = summary(
                hours_mean, geo, "hours", "Employment-based route (no industrial hours)"
            )
            if hours_source != "Employment-based route (no industrial hours)":
                hours_source = f"{hours_source} (employment-based route)"
        rows.append(
            {
                "Country": geo,
                "Employment source": employment_source,
                "Employment year": employment_year,
                "Employment resolution": employment_resolution,
                "Hours source": hours_source,
                "Hours year": hours_year,
                "Hours resolution": hours_resolution,
                "FTE40 anchor": "ILOSTAT FTE40 (2024)",
            }
        )
    return pd.DataFrame(rows)


def build_fte40_allocation_method_comparison(
    before_fte_industry: pd.DataFrame | None,
    after_fte_industry: pd.DataFrame,
    *,
    explicit_geos,
    target_industries,
) -> dict[str, pd.DataFrame]:
    """Compare final allocation provenance without changing any values.

    The four headline categories are mutually exclusive.  Annual-hours parent
    residuals remain in the annual-hours category, while ``detail`` separates
    them from exact annual-hours observations.  ROW is reported separately
    because its World-minus-explicit residual is not an explicit-country labor
    source.
    """
    categories = (
        "Direct observed annual hours",
        "Observed employment × mean weekly hours",
        "Observed industrial employment",
        "FIGARO-output-based residual allocation",
    )
    method_category = {
        "EU_H_DIRECT": categories[0],
        "EU_H_PARENT": categories[0],
        "EMP_X_HOURS": categories[1],
        "EMP_SHARE": categories[2],
        "EMP_PARENT_OUTPUT": categories[3],
        "EMP_NATIONAL_RESIDUAL_OUTPUT": categories[3],
        "NATIONAL_RESIDUAL_OUTPUT": categories[3],
        "ZERO_OUTPUT_UNRESOLVED": categories[3],
    }
    method_detail = {
        "EU_H_DIRECT": "direct annual-hours observation",
        "EU_H_PARENT": "annual-hours parent residual",
        "EMP_X_HOURS": "employment × compatible mean hours",
        "EMP_SHARE": "direct industrial employment",
        "EMP_PARENT_OUTPUT": "employment parent residual split by FIGARO output",
        "EMP_NATIONAL_RESIDUAL_OUTPUT": "national residual split by FIGARO output",
        "NATIONAL_RESIDUAL_OUTPUT": "national residual split by FIGARO output",
        "ZERO_OUTPUT_UNRESOLVED": "zero-output unresolved; no FTE allocated",
    }
    explicit = sorted({str(value).strip().upper() for value in explicit_geos})
    industries = sorted({str(value).strip() for value in target_industries})
    expected_n = len(explicit) * len(industries)

    def _prepare(frame: pd.DataFrame | None) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame(columns=["geo", "industry", "category", "detail"])
        out = frame.copy()
        out["geo"] = out["geo"].astype(str).str.strip().str.upper()
        out["industry"] = out["industry"].astype(str).str.strip()
        out["method"] = out.get("allocation_method", pd.Series("", index=out.index)).astype(str)
        out["category"] = out["method"].map(method_category).fillna(categories[3])
        out["detail"] = out["method"].map(method_detail).fillna(
            "unclassified residual allocation"
        )
        return out

    def _counts(frame: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for category in categories:
            count = int(frame["category"].eq(category).sum()) if not frame.empty else 0
            rows.append({"Allocation method": category, "country-industries": count})
        result = pd.DataFrame(rows)
        result["%"] = 100.0 * result["country-industries"] / expected_n if expected_n else 0.0
        return result

    before = _prepare(before_fte_industry).loc[lambda frame: frame["geo"].isin(explicit)]
    after = _prepare(after_fte_industry).loc[lambda frame: frame["geo"].isin(explicit)]
    if not after.empty:
        after_pairs = after[["geo", "industry"]].drop_duplicates()
        if len(after_pairs) != expected_n or len(after) != expected_n:
            raise ValueError(
                "FTE40 allocation-method comparison requires exactly one row "
                f"per explicit country-industry ({expected_n}); got {len(after)} rows."
            )
    before_counts = _counts(before).rename(
        columns={"country-industries": "Before: country-industries", "%": "Before: %"}
    )
    after_counts = _counts(after).rename(
        columns={"country-industries": "After: country-industries", "%": "After: %"}
    )
    global_table = before_counts.merge(after_counts, on="Allocation method", how="outer")
    for column in (
        "Before: country-industries", "After: country-industries",
    ):
        global_table[column] = global_table[column].fillna(0).astype(int)
    for column in ("Before: %", "After: %"):
        global_table[column] = global_table[column].fillna(0.0)
    global_table.attrs["denominator"] = expected_n

    country_rows = []
    for geo in explicit:
        row = {"Country": geo}
        after_geo = after.loc[after["geo"].eq(geo)]
        counts = after_geo["category"].value_counts() if not after_geo.empty else pd.Series(dtype=int)
        for category in categories:
            short = category.replace("Direct observed ", "Direct ").replace(
                "Observed ", ""
            ).replace("FIGARO-output-based residual allocation", "FIGARO residual")
            row[f"After: {short} %"] = 100.0 * float(counts.get(category, 0)) / len(industries)
        country_rows.append(row)
    country_table = pd.DataFrame(country_rows)
    country_table.attrs["denominator_per_country"] = len(industries)

    detail_rows = []
    for prefix, frame in (("Before", before), ("After", after)):
        counts = frame["detail"].value_counts() if not frame.empty else pd.Series(dtype=int)
        for detail, count in sorted(counts.items()):
            detail_rows.append({
                "Period": prefix,
                "Allocation detail": detail,
                "Country-industries": int(count),
            })
    detail_table = pd.DataFrame(detail_rows)

    row_frame = _prepare(after_fte_industry).loc[lambda frame: frame["geo"].eq("ROW")]
    row_rows = []
    if not row_frame.empty:
        row_counts = row_frame["method"].value_counts()
        for method, count in row_counts.items():
            row_rows.append({
                "Allocation method": method,
                "Allocation detail": method_detail.get(str(method), "ROW residual allocation"),
                "ROW industries": int(count),
                "ROW %": 100.0 * int(count) / len(industries) if industries else 0.0,
            })
    row_table = pd.DataFrame(row_rows)
    row_table.attrs["denominator"] = len(industries)
    return {
        "global": global_table,
        "country": country_table,
        "detail": detail_table,
        "row": row_table,
    }



def _wages_cache_path(
    emp_core,
    E_MRIO,
    mrio_prd_set,
    *,
    ind_ind: bool,
    total_code: str | None,
    total_countries,
    cache_path: str | Path | None = None,
    cache_key: str | None = None,
) -> Path:
    if cache_path is not None:
        return Path(cache_path)

    emp_rows = len(emp_core) if hasattr(emp_core, "__len__") else 0
    emp_cols = len(emp_core.columns) if hasattr(emp_core, "columns") else 0
    mrio_prd_len = len(mrio_prd_set) if mrio_prd_set is not None else 0
    e_len = len(E_MRIO) if hasattr(E_MRIO, "__len__") else 0
    tc = ",".join(map(str, total_countries)) if total_countries is not None else ""

    if cache_key is None:
        key_src = f"rule=isic4_A14_A21_agg_sector_csv_v15_l2_connected_components|ind={int(bool(ind_ind))}|total={total_code}|tc={tc}|emp={emp_rows}x{emp_cols}|mrio_prd={mrio_prd_len}|E={e_len}"
        key = hashlib.sha1(key_src.encode("utf-8")).hexdigest()[:10]
        fname = f"wages_{key}.pkl"
    else:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(cache_key)).strip("_") or "wages"
        fname = f"wages_{safe}.pkl"

    return Path(HELPERS_DIR) / "cache" / fname


def _global_employment_signature(global_employment: pd.DataFrame | None) -> str | None:
    if global_employment is None:
        return None
    if not isinstance(global_employment, pd.DataFrame):
        raise TypeError("global_employment must be a pandas DataFrame")
    columns = [
        column
        for column in ("nace_r2", "ECO", "EC2", "OBS_VALUE", "TIME_PERIOD", "dataflow_id")
        if column in global_employment.columns
    ]
    if not columns:
        return "empty"
    frame = global_employment.loc[:, columns].copy().astype(str)
    frame = frame.sort_values(columns).reset_index(drop=True)
    payload = pd.util.hash_pandas_object(frame, index=False).values.tobytes()
    return hashlib.sha1(payload).hexdigest()

def build_wages(
    emp_core,
    E_MRIO,
    mrio_prd_set,
    *,
    total_countries=("CN","JP","CA"),
    total_code=None,
    ind_ind: bool | None = None,
    use_cache: bool = False,
    recompute: bool | None = None,
    cache_path: str | Path | None = None,
    cache_key: str | None = None,
    return_audit: bool = False,
    return_emp_mrio: bool = False,
    global_employment: pd.DataFrame | None = None,
):

    import pandas as pd, numpy as np
    from functools import lru_cache

    def implicit_employment_from_wages(wage_series, compensation_series):
        """Return implicit employment in thousand persons/year.

        ``D1`` compensation is expressed in million EUR and the returned
        wages are expressed in EUR per person-year. Dividing by wage in kEUR
        therefore gives employment in thousand persons/year.
        """
        compensation = compensation_series.reindex(wage_series.index)
        wage_k_eur = pd.to_numeric(wage_series, errors="coerce") / 1_000.0
        compensation = pd.to_numeric(compensation, errors="coerce")
        employment = compensation.div(wage_k_eur).where(wage_k_eur > 0)
        employment = employment.mask(compensation.eq(0), 0.0)
        employment.name = "emp_mrio"
        return employment

    ind_ind = _is_industry_matrix(mrio_prd_set, ind_ind)
    if total_code is None:
        total_code = "CPA_TOTAL" if ind_ind else "CPA_TOTAL"
    global_signature = _global_employment_signature(global_employment)

    # --- wages cache ---
    if recompute is None:
        recompute = False
    cache_file = None
    if use_cache:
        cache_file = _wages_cache_path(
            emp_core,
            E_MRIO,
            mrio_prd_set,
            ind_ind=ind_ind,
            total_code=total_code,
            total_countries=total_countries,
            cache_path=cache_path,
            cache_key=cache_key,
        )
        if (not recompute) and cache_file.exists():
            try:
                with cache_file.open("rb") as f:
                    payload = pickle.load(f)
                wages = payload.get("wages")
                report = payload.get("report")
                empty = payload.get("empty")
                wage_audit = payload.get("wage_audit")
                if (
                    wages is not None
                    and report is not None
                    and empty is not None
                    and (not return_audit or wage_audit is not None)
                    and payload.get("global_employment_signature") == global_signature
                ):
                    emp_mrio = payload.get("emp_mrio")
                    if return_emp_mrio and emp_mrio is None:
                        emp_mrio = implicit_employment_from_wages(wages, E_MRIO)
                    if return_emp_mrio:
                        print(
                            "Global implicit MRIO employment: "
                            f"{emp_mrio.sum(min_count=1):,.3f} thousand persons/year"
                        )
                    if return_audit:
                        if return_emp_mrio:
                            return wages, emp_mrio, report, empty, wage_audit
                        return wages, report, empty, wage_audit
                    if return_emp_mrio:
                        return wages, emp_mrio, report, empty
                    return wages, report, empty
            except Exception:
                pass

    canon = _canon_map(mrio_prd_set)
    canon[total_code] = total_code

    # Work in CPA_* space for ind_ind, then strip at the end
    if ind_ind:
        to_cpa = {str(p): _add_cpa_prefix(p) for p in mrio_prd_set}
        canon_cpa = {_add_cpa_prefix(k): _add_cpa_prefix(v) for k, v in canon.items()}
        P_raw = (class_prd_ava.set_index("prd_ava_code")["parent_code"].dropna().astype(str).to_dict())
        P = {canon_cpa.get(k, k): canon_cpa.get(v, v) for k, v in P_raw.items()}
    else:
        to_cpa = None
        P_raw = (class_prd_ava.set_index("prd_ava_code")["parent_code"].dropna().astype(str).to_dict())
        P = {canon.get(k, k): canon.get(v, v) for k, v in P_raw.items()}
    ch = {}
    for c,p in P.items(): ch.setdefault(p, []).append(c)

    @lru_cache(None)
    def desc(c):
        s={c}
        for k in ch.get(c,[]): s |= desc(k)
        return s

    def non_overlapping_observed_descendants(g, parent_code):
        candidates = {
            j for j in desc(parent_code)
            if j != parent_code
            and (g, j) in empd
            and (g, j) in d1d
        }

        # If both an observed aggregate and its children exist, keep the
        # more disaggregated observed codes to avoid double counting D1/E.
        return sorted(
            j for j in candidates
            if not any(other != j and other in desc(j) for other in candidates)
        )

    prds = list(mrio_prd_set)
    aggregate_priority = _aggregate_source_priority(mrio_prd_set, ind_ind=ind_ind)

    if ind_ind:
        prds_cpa = [to_cpa[p] for p in prds]
        d1 = (E_MRIO.rename("D1_mrio").to_frame().reset_index())
        d1["prd_use"] = d1["prd_use"].map(lambda x: _add_cpa_prefix(x))
        d1 = (d1.query("prd_use in @prds_cpa")
                .set_index(["destination_region","prd_use"]))
        emp_frame = emp_core.copy()
        emp_frame["nace_r2"] = emp_frame["nace_r2"].map(_add_cpa_prefix)
        emp = emp_frame.groupby(["geo","nace_r2"])["OBS_VALUE"].sum()
    else:
        d1 = (E_MRIO.rename("D1_mrio").to_frame().reset_index()
                     .query("prd_use in @prds")
                     .set_index(["destination_region","prd_use"]))
        emp_frame = emp_core.copy()
        emp = emp_frame.groupby(["geo","nace_r2"])["OBS_VALUE"].sum()
    d1d, empd = d1["D1_mrio"].to_dict(), emp.to_dict()

    coverage_col = "covers_ind_ava_codes"

    @lru_cache(None)
    def observed_row_coverage_for_geo(g):
        if coverage_col not in emp_frame.columns:
            return {}
        rows = emp_frame.loc[emp_frame["geo"].astype(str).eq(str(g))]
        out = {}
        for _, row in rows.iterrows():
            src = str(row["nace_r2"]).strip()
            if ind_ind:
                src = _add_cpa_prefix(src)
            covers = _split_coverage_string(row.get(coverage_col))
            norm_covers = set()
            for c in covers:
                norm_covers.add(_add_cpa_prefix(c) if ind_ind else c)
            if norm_covers:
                out.setdefault(src, set()).update(norm_covers)
        return out

    @lru_cache(None)
    def observed_codes_for_geo(g):
        return frozenset(code for gg, code in empd.keys() if gg == g)

    @lru_cache(None)

    def parent_proxy_vals(g, parent_code):
        """
        Parent-level wage proxy from observed descendants.

        This does not reconstruct the full parent.
        It only uses descendant codes for which both MRIO compensation
        and observed employment are available.

        Example:
            JP, C21 missing -> use wage_C from observed C descendants.
        """
        parent_code = str(parent_code)

        candidates = non_overlapping_observed_descendants(g, parent_code)

        if not candidates:
            return None, None

        d = sum(float(d1d[(g, j)]) for j in candidates)
        e = sum(float(empd[(g, j)]) for j in candidates)

        if e > 0 and d >= 0:
            return d, e

        return None, None

    def aggregate_expanded_for_geo(g):
        aggregate_cov = _aggregate_coverage_map(
            mrio_prd_set,
            ind_ind=ind_ind,
            observed_codes=set(observed_codes_for_geo(g)),
        )
        for src, covers in observed_row_coverage_for_geo(g).items():
            aggregate_cov.setdefault(src, set()).update(covers)
        return {
            src: set().union(*(desc(c) for c in covers)) if covers else set()
            for src, covers in aggregate_cov.items()
        }

    def d1_sum_for_emp_code(g, code):
        if _strip_cpa_prefix(code) == "TOTAL":
            return sum(d1d.get((g, j), 0.0) for j in d1.index.get_level_values("prd_use").unique())

        candidates = aggregate_expanded_for_geo(g).get(code)
        if candidates is None:
            candidates = set(desc(code)) if code in P or code in ch else {code}
        return sum(d1d.get((g, j), 0.0) for j in candidates)

    @lru_cache(None)
    def aggregate_components_for_geo(g):
        coverage = {
            str(src): {str(t) for t in covered}
            for src, covered in aggregate_expanded_for_geo(g).items()
            if (g, src) in empd and covered
        }
        if not coverage:
            return tuple()

        target_to_sources = {}
        for src, targets_cov in coverage.items():
            for target in targets_cov:
                target_to_sources.setdefault(target, set()).add(src)

        seen_sources = set()
        components = []
        for src0 in sorted(coverage):
            if src0 in seen_sources:
                continue
            sources = set()
            targets_cov = set()
            stack = [src0]
            while stack:
                src = stack.pop()
                if src in seen_sources:
                    continue
                seen_sources.add(src)
                sources.add(src)
                for target in coverage.get(src, set()):
                    if target not in targets_cov:
                        targets_cov.add(target)
                        stack.extend(target_to_sources.get(target, set()) - seen_sources)
            if sources and targets_cov:
                priority = min(aggregate_priority.get(s, 99) for s in sources)
                components.append((priority, frozenset(sources), frozenset(targets_cov)))

        components.sort(key=lambda x: (x[0], len(x[2]), len(x[1]), ";".join(sorted(x[1]))))
        return tuple(components)

    def aggregate_emp_component_candidates(g, ind):
        out = []
        for priority, sources, targets_cov in aggregate_components_for_geo(g):
            if ind in targets_cov:
                out.append((priority, len(targets_cov), len(sources), sources, targets_cov))
        out.sort(key=lambda x: (x[0], x[1], x[2], ";".join(sorted(x[3]))))
        return [(sources, targets_cov) for _, _, _, sources, targets_cov in out]

    def aggregate_emp_candidates(g, ind):
        out = []
        for src, covered in aggregate_expanded_for_geo(g).items():
            if (g, src) in empd and ind in covered:
                out.append((len(covered), aggregate_priority.get(src, 99), src))
        out.sort(key=lambda x: (x[1], x[0], str(x[2])))
        return [src for _, _, src in out]

    @lru_cache(None)
    def country_employment_metadata(g):
        rows = emp_frame.loc[
            emp_frame["geo"].astype(str).eq(str(g))
        ]
        if rows.empty:
            return "-", tuple(), tuple(), rows

        codes = set(rows["nace_r2"].dropna().astype(str))
        dataflows = tuple(sorted(
            str(x).strip()
            for x in rows.get("dataflow_id", pd.Series(dtype=object)).dropna().unique()
            if str(x).strip()
        ))
        sources = tuple(sorted(
            str(x).strip()
            for x in rows.get("source_db", pd.Series(dtype=object)).dropna().unique()
            if str(x).strip()
        ))
        profile = _detect_aggregate_profile(codes) or (
            "TOTAL" if not codes - {"TOTAL"} else "DETAIL"
        )
        return profile, dataflows, sources, rows

    def _audit_trace(
        *,
        rule,
        comp_codes=(),
        emp_codes=(),
        used_d=None,
        used_e=None,
    ):
        return {
            "rule": str(rule),
            "comp_codes": tuple(sorted({str(x) for x in comp_codes})),
            "emp_codes": tuple(sorted({str(x) for x in emp_codes})),
            "used_d": used_d,
            "used_e": used_e,
        }

    def _reported_comp_codes(g, targets):
        """Keep audit compensation codes limited to actual MRIO industries."""
        available = {
            code
            for geo, code in d1d
            if str(geo) == str(g)
        }
        reported = set(targets) & available
        return reported or set(targets)

    def _parent_aggregate_rule(sources):
        labels = sorted({
            _strip_cpa_prefix(source)
            for source in sources
            if str(source).strip()
        })
        if labels:
            return (
                "Observed parent employment at "
                f"{';'.join(labels)} -> Aggregated wage"
            )
        return "Observed employment -> Aggregated wage"

    def resolve_vals(g, ind):
        # 1. Direct exact match
        if (g, ind) in empd and (g, ind) in d1d:
            return (
                d1d[(g, ind)],
                empd[(g, ind)],
                _audit_trace(
                    rule="Observed employment -> Direct calculation",
                    comp_codes=(ind,),
                    emp_codes=(ind,),
                    used_d=d1d[(g, ind)],
                    used_e=empd[(g, ind)],
                ),
            )

        # 2. Explicit aggregate/reconstructed coverage, if available.
        # Many-to-many source rows are not assigned partially. Instead,
        # connected source-target blocks are reconstructed as:
        #   sum(D1 over target coverage) / sum(employment over observed sources).
        for sources, targets_cov in aggregate_emp_component_candidates(g, ind):
            e = sum(float(empd[(g, c)]) for c in sources if (g, c) in empd)
            d = sum(float(d1d.get((g, j), 0.0)) for j in targets_cov)
            if e > 0 and d > 0:
                return (
                    d,
                    e,
                    _audit_trace(
                        rule=_parent_aggregate_rule(sources),
                        comp_codes=_reported_comp_codes(g, targets_cov),
                        emp_codes=sources,
                        used_d=d,
                        used_e=e,
                    ),
                )

        # 3. Parent fallback
        c = ind
        while c in P:
            c = P[c]

            # 3a. If parent employment is explicitly available, use it
            if (g, c) in empd:
                e = empd[(g, c)]
                d = d1_sum_for_emp_code(g, c)
                if e > 0 and d > 0:
                    targets = aggregate_expanded_for_geo(g).get(c)
                    if not targets:
                        targets = set(desc(c)) if c in P or c in ch else {ind}
                    return (
                        d,
                        e,
                        _audit_trace(
                            rule=(
                                "Observed parent employment at "
                                f"{_strip_cpa_prefix(c)} -> Aggregated wage"
                            ),
                            comp_codes=_reported_comp_codes(g, targets),
                            emp_codes=(c,),
                            used_d=d,
                            used_e=e,
                        ),
                    )

            # 3b. If parent is not explicitly available, build parent wage
            #     from observed descendants under that parent.
            d, e = parent_proxy_vals(g, c)
            if d is not None and e is not None:
                candidates = non_overlapping_observed_descendants(g, c)
                return (
                    d,
                    e,
                    _audit_trace(
                        rule=(
                            "Parent proxy from observed descendants under "
                            f"{_strip_cpa_prefix(c)} -> Aggregated wage"
                        ),
                        comp_codes=candidates,
                        emp_codes=candidates,
                        used_d=d,
                        used_e=e,
                    ),
                )

        return None, None, _audit_trace(
            rule="No direct employment resolution",
            comp_codes=(ind,),
            emp_codes=(),
        )

    def use_vals(g, ind):
        d, e, _ = resolve_vals(g, ind)
        return d, e

    w = d1.copy()
    w["E_obs"] = pd.Series(empd).reindex(w.index).astype("Float64")
    w["wage"]  = w["D1_mrio"].div(w["E_obs"]).where(w["E_obs"] > 0).astype("Float64")
    zero_emp_direct = (
        w["E_obs"].notna()
        & np.isclose(w["E_obs"].astype(float), 0.0, atol=1e-12, rtol=0.0)
    )
    w.loc[zero_emp_direct, "wage"] = 0.0
    zero_d1 = w["D1_mrio"].notna() & np.isclose(w["D1_mrio"].astype(float), 0.0, atol=1e-12, rtol=0.0)
    w.loc[zero_d1, "wage"] = 0.0

    d1_tot = d1.reset_index().groupby("destination_region")["D1_mrio"].sum()
    e_tot = (
        emp.reset_index()
        .query("nace_r2 == @total_code")
        .groupby("geo")["OBS_VALUE"]
        .sum()
    )
    tot_ratio = d1_tot.div(e_tot.reindex(d1_tot.index).replace(0, pd.NA)).dropna().to_dict()

    def bloc_wage(countries, ind):
        D=E=0.0
        for gg in countries:
            d,e = use_vals(gg, ind)
            if d is not None and e is not None and e > 0:
                D += d; E += e
        return (D/E) if E > 0 else pd.NA

    all_geos = [x for x in w.index.get_level_values(0).unique().tolist() if x != "ROW"]
    row_w = {i: bloc_wage(all_geos, i) for i in prds}
    if ("ROW", prds[0]) in w.index:
        for i in prds: w.loc[("ROW", i), "wage"] = row_w[i]

    def fb(idx):
        gg, ind = idx
        d,e = use_vals(gg, ind)
        if d is not None and e is not None and e > 0:
            return d/e
        if gg in tot_ratio:
            return tot_ratio[gg]
        return pd.NA

    m = w["wage"].isna()
    w.loc[m, "wage"] = pd.Series([fb(i) for i in w.index[m]], index=w.index[m]).astype("Float64")

    # promedio pais final (sin inf: solo si emp_total > 0)
    dsum = w.groupby(level=0)["D1_mrio"].sum(min_count=1)
    esum = (emp.reset_index().query("nace_r2==@total_code").groupby("geo")["OBS_VALUE"].sum()
              .reindex(dsum.index))
    esum = esum.fillna(w.groupby(level=0)["E_obs"].sum(min_count=1))
    cavg = (dsum / esum.replace(0, pd.NA)).to_dict()

    m = w["wage"].isna()
    w.loc[m, "wage"] = w.index.get_level_values(0)[m].map(cavg).astype("Float64")

    # limpieza final
    w["wage"] = w["wage"].replace([np.inf, -np.inf], pd.NA).astype("Float64")

    # Estimate ROW at the end from the completed explicit-region wage vector.
    # When global ILO employment is supplied, use its A14 totals to determine
    # the ROW residual and allocate that residual across FIGARO industries.
    # Without that input, retain the historical explicit-region wage proxy.
    row_mask = w.index.get_level_values("destination_region") == "ROW"
    global_row_employment = {}
    global_row_group_by_target = {}
    global_row_source = None
    global_row_dataflow = None
    global_row_profile = None
    global_source_total = None
    if row_mask.any():
        explicit = w.loc[~row_mask].copy()
        valid = (
            explicit["D1_mrio"].notna()
            & explicit["wage"].notna()
            & (explicit["wage"] > 0)
        )
        d_exp = explicit.loc[valid, "D1_mrio"]
        e_exp = d_exp.div(explicit.loc[valid, "wage"])
        d_by_ind = d_exp.groupby(level="prd_use").sum(min_count=1)
        e_by_ind = e_exp.groupby(level="prd_use").sum(min_count=1)
        row_ratio = d_by_ind.div(e_by_ind.replace(0, pd.NA))
        agg_e = e_by_ind.sum(min_count=1)
        agg_ratio = d_by_ind.sum(min_count=1) / agg_e if pd.notna(agg_e) and agg_e > 0 else pd.NA

        if global_employment is not None:
            global_df = global_employment.copy()
            if "SEX" in global_df.columns:
                total_sex = global_df["SEX"].astype(str).eq("SEX_T")
                if total_sex.any():
                    global_df = global_df.loc[total_sex].copy()
            if "nace_r2" not in global_df.columns:
                industry_col = next(
                    (column for column in ("ECO", "EC2") if column in global_df.columns),
                    None,
                )
                if industry_col is not None:
                    global_df["nace_r2"] = (
                        global_df[industry_col]
                        .astype(str)
                        .str.extract(r"(?:ECO|EC2)_ISIC4_(.+)$", expand=False)
                    )
            global_df["nace_r2"] = global_df["nace_r2"].map(_clean_industry_code)
            global_df["OBS_VALUE"] = pd.to_numeric(
                global_df["OBS_VALUE"], errors="coerce"
            )
            global_df["a14_group"] = global_df["nace_r2"].map(
                lambda code: next(iter(_a14_groups_for_code(code)), pd.NA)
            )
            global_df = global_df.loc[
                global_df["a14_group"].notna()
                & global_df["OBS_VALUE"].notna()
            ].copy()

            global_by_group = global_df.groupby("a14_group")["OBS_VALUE"].sum()
            global_source_total = float(global_by_group.sum(min_count=1))
            global_row_source = ";".join(sorted({
                str(value).strip()
                for value in global_df.get("source_db", pd.Series(dtype=object)).dropna()
                if str(value).strip()
            })) or "ILOSTAT_SDMX"
            global_row_dataflow = ";".join(sorted({
                str(value).strip()
                for value in global_df.get("dataflow_id", pd.Series(dtype=object)).dropna()
                if str(value).strip()
            })) or "DF_EMP_2EMP_SEX_ECO_NB"
            global_row_profile = "ISIC4_A14"

            target_to_group = {}
            for idx in w.index[row_mask]:
                target = str(idx[1])
                groups = _a14_groups_for_code(_strip_cpa_prefix(target))
                if len(groups) == 1:
                    target_to_group[target] = next(iter(groups))

            explicit_by_group = {}
            unmatched_explicit = 0.0
            for target, value in e_by_ind.items():
                group = target_to_group.get(str(target))
                if group is None:
                    unmatched_explicit += float(value)
                else:
                    explicit_by_group[group] = explicit_by_group.get(group, 0.0) + float(value)

            residual_by_group = {
                group: float(global_by_group.get(group, 0.0)) - explicit_by_group.get(group, 0.0)
                for group in global_by_group.index
            }
            if unmatched_explicit > 1e-9:
                print(
                    "[ILOSTAT] global employment reconciliation excludes "
                    f"{unmatched_explicit:,.3f} thousand explicit-region employment "
                    "because its FIGARO industry could not be mapped to A14."
                )
            negative_residuals = {
                group: value
                for group, value in residual_by_group.items()
                if value < -1e-9
            }
            if negative_residuals:
                print(
                    "[ILOSTAT] global employment residual is negative for A14 group(s): "
                    f"{negative_residuals}; retaining the explicit-region wage proxy "
                    "for those groups."
                )

            for idx in w.index[row_mask]:
                target = str(idx[1])
                group = target_to_group.get(target)
                d_value = pd.to_numeric(w.loc[idx, "D1_mrio"], errors="coerce")
                d_target = float(d_value) if pd.notna(d_value) else 0.0
                if d_target <= 0:
                    w.loc[idx, "wage"] = 0.0
                    global_row_employment[target] = 0.0
                    if group is not None:
                        global_row_group_by_target[target] = group
                    continue
                if group is None or residual_by_group.get(group, -1.0) < 0:
                    fill = row_ratio.get(idx[1], pd.NA)
                    if pd.isna(fill):
                        fill = agg_ratio
                    w.loc[idx, "wage"] = fill
                    continue

                targets = [
                    candidate
                    for candidate, candidate_group in target_to_group.items()
                    if candidate_group == group
                ]
                weights = {}
                for candidate in targets:
                    value = e_by_ind.get(candidate, np.nan)
                    if pd.notna(value) and float(value) > 0:
                        weights[candidate] = float(value)
                    else:
                        candidate_idx = ("ROW", candidate)
                        candidate_value = pd.to_numeric(
                            w.loc[candidate_idx, "D1_mrio"], errors="coerce"
                        )
                        candidate_d = (
                            float(candidate_value)
                            if pd.notna(candidate_value)
                            else 0.0
                        )
                        if candidate_d > 0:
                            weights[candidate] = candidate_d
                weight_total = sum(weights.values())
                residual = residual_by_group.get(group, 0.0)
                allocation = (
                    residual * weights.get(target, 0.0) / weight_total
                    if weight_total > 0 and residual >= 0
                    else 0.0
                )
                global_row_employment[target] = allocation
                global_row_group_by_target[target] = group
                if allocation > 0:
                    w.loc[idx, "wage"] = d_target / allocation
                else:
                    w.loc[idx, "wage"] = 0.0

            print(
                "ILOSTAT global employment target: "
                f"{global_source_total:,.3f} thousand persons/year"
            )
            print(
                "Explicit-region implicit employment used in residual: "
                f"{e_exp.sum(min_count=1):,.3f} thousand persons/year"
            )
            print(
                "ROW residual employment allocated from global ILO data: "
                f"{sum(global_row_employment.values()):,.3f} thousand persons/year"
            )
        else:
            for idx in w.index[row_mask]:
                if pd.notna(w.loc[idx, "D1_mrio"]) and w.loc[idx, "D1_mrio"] == 0:
                    w.loc[idx, "wage"] = 0.0
                    continue
                fill = row_ratio.get(idx[1], pd.NA)
                if pd.isna(fill):
                    fill = agg_ratio
                w.loc[idx, "wage"] = fill

    w.loc[zero_d1, "wage"] = 0.0
    w.loc[zero_emp_direct, "wage"] = 0.0
    w["wage"] = w["wage"].replace([np.inf, -np.inf], pd.NA).astype("Float64")

    empty = w[w["wage"].isna()]
    report = (w["wage"].notna().groupby(level=0)
              .agg(n_total="size", n_ok="sum")
              .assign(share_ok=lambda d: d.n_ok/d.n_total))

    wage_audit = None
    if return_audit:
        audit_rows = []

        def _safe_float(value):
            try:
                return float(value) if pd.notna(value) else np.nan
            except (TypeError, ValueError):
                return np.nan

        def _audit_codes(values):
            return ";".join(sorted({
                _strip_cpa_prefix(value)
                for value in values
                if value is not None and str(value).strip()
            }))

        for idx in w.index:
            country, ind = idx
            country = str(country)
            ind = str(ind)
            d_target = _safe_float(w.loc[idx, "D1_mrio"])
            e_observed = _safe_float(w.loc[idx, "E_obs"])
            wage_unscaled = _safe_float(w.loc[idx, "wage"])

            trace = None
            used_d = None
            used_e = None

            if country == "ROW":
                target_key = str(idx[1])
                if global_employment is not None and target_key in global_row_employment:
                    group = global_row_group_by_target.get(target_key, "A14")
                    rule = (
                        "Global ILO A14 employment residual allocated by industry "
                        "weights -> Estimated wage"
                    )
                    comp_codes = (ind,)
                    emp_codes = (group,)
                    emp_resolution = global_row_profile or "ISIC4_A14"
                    emp_dataset = global_row_dataflow or "DF_EMP_2EMP_SEX_ECO_NB"
                    emp_source = global_row_source or "ILOSTAT_SDMX"
                    used_d = d_target
                    used_e = global_row_employment[target_key]
                else:
                    rule = (
                        "ROW inferred employment from explicit-region wages "
                        "-> Estimated wage"
                    )
                    comp_codes = (ind,)
                    emp_codes = (ind,)
                    emp_resolution = "-"
                    emp_dataset = "-"
                    emp_source = "MRIO_D1_PLUS_EXPLICIT_WAGES"
                    used_d = d_target
                    used_e = (
                        d_target / wage_unscaled
                        if np.isfinite(d_target)
                        and np.isfinite(wage_unscaled)
                        and wage_unscaled > 0
                        else 0.0
                    )
            elif np.isfinite(d_target) and np.isclose(d_target, 0.0):
                trace = _audit_trace(
                    rule="Zero compensations -> Wage set to 0",
                    comp_codes=(ind,),
                    emp_codes=(ind,),
                    used_d=d_target,
                    used_e=0.0 if not np.isfinite(e_observed) else e_observed,
                )
            elif np.isfinite(e_observed) and np.isclose(e_observed, 0.0):
                trace = _audit_trace(
                    rule="Zero employment -> Wage set to 0",
                    comp_codes=(ind,),
                    emp_codes=(ind,),
                    used_d=d_target,
                    used_e=e_observed,
                )
            elif np.isfinite(e_observed) and e_observed > 0:
                trace = _audit_trace(
                    rule="Observed employment -> Direct calculation",
                    comp_codes=(ind,),
                    emp_codes=(ind,),
                    used_d=d_target,
                    used_e=e_observed,
                )
            else:
                resolved_d, resolved_e, resolved_trace = resolve_vals(country, ind)
                if (
                    resolved_d is not None
                    and resolved_e is not None
                    and resolved_e > 0
                ):
                    trace = resolved_trace
                elif country in tot_ratio and np.isfinite(wage_unscaled) and wage_unscaled > 0:
                    trace = _audit_trace(
                        rule="Country total employment ratio -> Estimated wage",
                        comp_codes=(ind,),
                        emp_codes=(total_code,),
                        used_d=d_target,
                        used_e=d_target / wage_unscaled,
                    )
                elif np.isfinite(wage_unscaled) and wage_unscaled > 0:
                    trace = _audit_trace(
                        rule="Country average wage -> Estimated wage",
                        comp_codes=(ind,),
                        emp_codes=(),
                        used_d=d_target,
                        used_e=d_target / wage_unscaled,
                    )
                else:
                    trace = resolved_trace

            if trace is not None:
                profile, all_dataflows, all_sources, country_rows = country_employment_metadata(country)
                source_codes = set(trace.get("emp_codes", ()))
                selected_rows = country_rows.loc[
                    country_rows["nace_r2"].astype(str).isin(source_codes)
                ] if not country_rows.empty else country_rows

                if not selected_rows.empty:
                    dataflows = tuple(sorted(
                        str(x).strip()
                        for x in selected_rows.get("dataflow_id", pd.Series(dtype=object)).dropna().unique()
                        if str(x).strip()
                    ))
                    sources = tuple(sorted(
                        str(x).strip()
                        for x in selected_rows.get("source_db", pd.Series(dtype=object)).dropna().unique()
                        if str(x).strip()
                    ))
                else:
                    dataflows = all_dataflows
                    sources = all_sources

                emp_resolution = profile
                emp_dataset = ";".join(dataflows) if dataflows else "-"
                emp_source = ";".join(sources) if sources else "-"
                comp_codes = trace.get("comp_codes", (ind,))
                emp_codes = trace.get("emp_codes", ())
                rule = trace.get("rule", "")
                used_d = trace.get("used_d")
                used_e = trace.get("used_e")

            final_wage = (
                wage_unscaled * 1_000
                if np.isfinite(wage_unscaled)
                else np.nan
            )
            if used_d is None or not np.isfinite(_safe_float(used_d)):
                used_d = d_target
            if used_e is None or not np.isfinite(_safe_float(used_e)):
                used_e = (
                    d_target / wage_unscaled
                    if np.isfinite(d_target)
                    and np.isfinite(wage_unscaled)
                    and wage_unscaled > 0
                    else np.nan
                )

            audit_rows.append({
                "country": country,
                "industry (NACE2)": _strip_cpa_prefix(ind),
                "final_wage": final_wage,
                "final_comp": _safe_float(used_d),
                "comp_codes": _audit_codes(comp_codes),
                "final_emp": _safe_float(used_e),
                "emp_codes": _audit_codes(emp_codes),
                "emp_resolution": emp_resolution,
                "emp_dataset": emp_dataset,
                "emp_source": emp_source,
                "calc_note": rule,
            })

        wage_audit = pd.DataFrame(audit_rows)

    w["wage"] = w["wage"] * 1_000

    if ind_ind:
        w.index = pd.MultiIndex.from_tuples(
            [(i[0], _strip_cpa_prefix(i[1])) for i in w.index],
            names=w.index.names
        )
        empty.index = pd.MultiIndex.from_tuples(
            [(i[0], _strip_cpa_prefix(i[1])) for i in empty.index],
            names=empty.index.names
        )

    emp_mrio = implicit_employment_from_wages(w["wage"], w["D1_mrio"])
    if return_emp_mrio:
        if global_source_total is not None:
            discrepancy = float(emp_mrio.sum(min_count=1)) - global_source_total
            print(
                "Global employment reconciliation (MRIO implicit - ILO source): "
                f"{discrepancy:,.3f} thousand persons/year"
            )
        print(
            "Global implicit MRIO employment: "
            f"{emp_mrio.sum(min_count=1):,.3f} thousand persons/year"
        )

    # --- wages cache save ---
    if use_cache and cache_file is not None:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with cache_file.open("wb") as f:
                pickle.dump({
                    "wages": w["wage"],
                    "emp_mrio": emp_mrio,
                    "report": report,
                    "empty": empty,
                    "wage_audit": wage_audit,
                    "global_employment_signature": global_signature,
                }, f)
        except Exception:
            pass

    if return_audit:
        if return_emp_mrio:
            return w["wage"], emp_mrio, report, empty, wage_audit
        return w["wage"], report, empty, wage_audit
    if return_emp_mrio:
        return w["wage"], emp_mrio, report, empty
    return w["wage"], report, empty

def _export_fte40_supplementary_material(
    *,
    fte_industry: pd.DataFrame,
    fte_audit: pd.DataFrame | None,
    emp_core: pd.DataFrame,
    hours_total: pd.DataFrame | None = None,
    hours_mean: pd.DataFrame | None = None,
    output_path: str | Path,
    template_path: str | Path | None = None,
) -> Path:
    """Export the FTE40/output audit using the established C2 workbook layout."""
    required = {
        "geo", "industry", "FTE_country_total", "FTE_country_industry",
        "allocation_method", "source_code", "source_db", "source_year",
        "dataflow_id", "population_scope", "source_coverage",
    }
    missing = sorted(required - set(fte_industry.columns))
    if missing:
        raise KeyError(f"fte_industry is missing required C2 columns: {missing}.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet_names = [
        "README",
        "wages_calculation",
        "sources_resolution",
        "resolution_profiles",
    ]
    detail = fte_industry.copy()
    detail["geo"] = detail["geo"].astype(str).str.strip()
    detail["industry"] = detail["industry"].astype(str).str.strip()
    if "FIGARO_output" not in detail:
        detail["FIGARO_output"] = pd.NA
    if "FTE_intensity" not in detail:
        detail["FTE_intensity"] = pd.NA
    if "source_observation" not in detail:
        detail["source_observation"] = pd.NA
    for column in [
        "parent_source_observation", "parent_residual_observation",
        "national_employment_total", "FIGARO_GVA", "parent_GVA_total",
        "parent_output_total", "GVA_allocation_share", "output_allocation_block",
        "output_allocation_share", "allocation_block", "effective_allocation_weight",
        "GVA_used_as_weight", "allocation_operation", "weight_unit",
        "original_source_code", "original_class_system", "employment_observation",
        "mean_weekly_hours", "hours_source_code", "hours_dataflow_id",
        "hours_source_year", "employment_hours_compatibility",
        "estimated_child_labor_volume", "source_conflicts",
        "national_employment_conflicts", "national_employment_source_year",
        "national_hours_total", "national_hours_source_year", "national_hours_status",
        "employment_constraint_status",
        "final_country_share", "constraint_level", "observation_level", "weight_role",
        "weight_coverage", "gva_role", "allocation_raw_numerator",
        "allocation_raw_denominator", "allocation_raw_unit",
        "stage1_raw_numerator", "stage1_raw_denominator", "stage1_weight_unit",
        "stage1_normalized_share", "stage1_resulting_fte", "stage2_operation",
        "stage2_allocation_block", "stage2_normalized_share", "stage2_resulting_fte",
        "stage1_operation", "stage1_weight_source", "stage2_weight_source",
        "stage2_raw_numerator", "stage2_raw_denominator", "stage2_weight_unit",
    ]:
        if column not in detail:
            detail[column] = pd.NA

    # Prefer the actual populated observation columns.  Some Eurostat frames
    # retain an empty compatibility column alongside OBS_VALUE; selecting that
    # empty column would erase valid observations from the export.
    source_observation_raw = detail["source_observation"].copy()
    detail["source_observation"] = _c2_first_available(
        detail,
        ["source_observation", "employment_observation", "estimated_child_labor_volume"],
    )
    detail["original_observation_unit"] = np.select(
        [
            detail["allocation_method"].isin({"EU_H_DIRECT", "EU_H_PARENT"}),
            detail["allocation_method"].eq("EMP_X_HOURS"),
            detail["allocation_method"].isin({"EMP_SHARE", "EMP_PARENT_OUTPUT", "EMP_NATIONAL_RESIDUAL_OUTPUT"}),
            detail["allocation_method"].eq("NATIONAL_RESIDUAL_OUTPUT"),
        ],
        [
            "thousand hours/year",
            "thousand person-hours/week",
            "thousand employed persons",
            "thousand FTE",
        ],
        default="source-specific",
    )
    detail["final_country_share_pct"] = 100.0 * pd.to_numeric(
        detail["final_country_share"], errors="coerce"
    )
    detail["actual_weight_numerator"] = pd.to_numeric(
        detail["allocation_raw_numerator"], errors="coerce"
    )
    detail["actual_weight_denominator"] = pd.to_numeric(
        detail["allocation_raw_denominator"], errors="coerce"
    )
    gva_weight = detail["GVA_used_as_weight"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    detail.loc[gva_weight & detail["actual_weight_numerator"].isna(), "actual_weight_numerator"] = pd.to_numeric(
        detail.loc[gva_weight & detail["actual_weight_numerator"].isna(), "FIGARO_GVA"], errors="coerce"
    )
    detail.loc[gva_weight & detail["actual_weight_denominator"].isna(), "actual_weight_denominator"] = pd.to_numeric(
        detail.loc[gva_weight & detail["actual_weight_denominator"].isna(), "parent_GVA_total"], errors="coerce"
    )
    detail["actual_weight_share"] = (
        detail["actual_weight_numerator"]
        / detail["actual_weight_denominator"].replace(0, np.nan)
    )

    detail["parent employment share (%)"] = (
        100.0
        * pd.to_numeric(detail.get("parent_source_observation"), errors="coerce")
        / pd.to_numeric(detail.get("national_employment_total"), errors="coerce").replace(0, np.nan)
    )
    detail["parent FTE mass (thousand)"] = (
        pd.to_numeric(detail.get("FTE_country_total"), errors="coerce")
        * detail["parent employment share (%)"] / 100.0
    )
    detail["remaining parent FTE mass (thousand)"] = (
        pd.to_numeric(detail.get("FTE_country_total"), errors="coerce")
        * pd.to_numeric(detail.get("parent_residual_observation"), errors="coerce")
        / pd.to_numeric(detail.get("national_employment_total"), errors="coerce").replace(0, np.nan)
    )
    detail["effective allocation weight (%)"] = (
        100.0 * pd.to_numeric(detail.get("effective_allocation_weight"), errors="coerce")
    )
    detail["GVA allocation share (%)"] = (
        100.0 * pd.to_numeric(detail.get("GVA_allocation_share"), errors="coerce")
    )

    # The calculation engine stores the selected source metadata in one
    # record.  Build separate hours and employment display blocks here so the
    # publication worksheet does not present a synthetic fallback value as an
    # original observation.  Missing source metadata remains blank.
    target_industries = set(detail["industry"].dropna().astype(str))

    def _c2_hours_metadata(frame, source_name):
        if frame is None or frame.empty:
            return {}
        standardized = _standardize_fte_source(
            frame,
            target_industries=target_industries,
            value_name="hours",
            source_name=source_name,
        )
        lookup = {}
        for row in standardized.to_dict("records"):
            metadata = {
                "class_system": row.get("original_class_system", row.get("class_system", pd.NA)),
                "dataflow_id": row.get("dataflow_id", pd.NA),
                "source_db": row.get("source_db", pd.NA),
                "source_year": row.get("source_year", pd.NA),
            }
            for code in (row.get("original_source_code"), row.get("source_code")):
                if code is not None and not pd.isna(code):
                    lookup.setdefault(
                        (str(row.get("geo", "")).strip(), str(code).strip()),
                        [],
                    ).append(metadata)
        return lookup

    hours_total_metadata = _c2_hours_metadata(hours_total, "annual-hours source")
    hours_mean_metadata = _c2_hours_metadata(hours_mean, "mean-hours source")

    def _c2_metadata(lookup, geo, code, year, dataflow):
        if code is None or pd.isna(code):
            return {}
        codes = [str(code).strip()]
        if ";" in codes[0]:
            codes = [value.strip() for value in codes[0].split(";") if value.strip()]
        candidates = []
        for source_code in codes:
            candidates.extend(lookup.get((str(geo).strip(), source_code), []))
        if not candidates:
            return {}
        for candidate in candidates:
            same_flow = (
                pd.isna(dataflow)
                or pd.isna(candidate.get("dataflow_id"))
                or str(candidate.get("dataflow_id")) == str(dataflow)
            )
            same_year = (
                pd.isna(year)
                or pd.isna(candidate.get("source_year"))
                or str(candidate.get("source_year")) == str(year)
            )
            if same_flow and same_year:
                return candidate
        return candidates[0]

    def _c2_value(value):
        return pd.NA if value is None or pd.isna(value) else value

    hours_rows = []
    employment_rows = []
    readable_methods = []
    hours_methods = {"EU_H_DIRECT", "EU_H_PARENT"}
    employment_methods = {
        "EMP_X_HOURS", "EMP_SHARE", "EMP_PARENT_OUTPUT",
        "EMP_NATIONAL_RESIDUAL_OUTPUT",
    }
    for index, row in detail.iterrows():
        method = str(row.get("allocation_method", ""))
        geo = row.get("geo", pd.NA)
        source_code = row.get("original_source_code", row.get("source_code", pd.NA))
        source_observation = source_observation_raw.loc[index]
        source_year = row.get("source_year", pd.NA)
        dataflow = row.get("dataflow_id", pd.NA)
        source_db = row.get("source_db", pd.NA)
        source_class = row.get("original_class_system", pd.NA)
        weight_source = str(row.get("allocation_weight_source", ""))
        gva_used = bool(row.get("GVA_used_as_weight", False))

        hours_code = hours_observation = hours_unit = hours_class = pd.NA
        hours_flow = hours_db = pd.NA
        employment_code = employment_observation = employment_unit = pd.NA
        employment_class = employment_flow = employment_db = pd.NA

        if method in hours_methods:
            hours_code = _c2_value(source_code)
            hours_observation = _c2_value(source_observation)
            hours_unit = "thousand hours/year"
            metadata = _c2_metadata(
                hours_total_metadata,
                geo,
                hours_code,
                source_year,
                dataflow,
            )
            hours_class = _c2_value(
                source_class if not pd.isna(source_class) else metadata.get("class_system", pd.NA)
            )
            hours_flow = _c2_value(
                dataflow if not pd.isna(dataflow) else metadata.get("dataflow_id", pd.NA)
            )
            hours_db = _c2_value(
                source_db if not pd.isna(source_db) else metadata.get("source_db", pd.NA)
            )
        elif method == "EMP_X_HOURS":
            hours_code = _c2_value(row.get("hours_source_code", pd.NA))
            hours_observation = _c2_value(row.get("mean_weekly_hours", pd.NA))
            hours_unit = "hours per employed person per week"
            metadata = _c2_metadata(
                hours_mean_metadata,
                geo,
                hours_code,
                row.get("hours_source_year", pd.NA),
                row.get("hours_dataflow_id", pd.NA),
            )
            hours_class = _c2_value(metadata.get("class_system", source_class))
            hours_flow = _c2_value(
                row.get("hours_dataflow_id", metadata.get("dataflow_id", pd.NA))
            )
            hours_db = _c2_value(metadata.get("source_db", pd.NA))

        if method in employment_methods:
            employment_code = _c2_value(source_code)
            employment_observation = _c2_value(
                row.get("employment_observation", pd.NA)
            )
            if pd.isna(employment_observation):
                employment_observation = _c2_value(source_observation)
            employment_unit = "thousand employed persons"
            employment_class = _c2_value(source_class)
            employment_flow = _c2_value(dataflow)
            employment_db = _c2_value(source_db)

        hours_rows.append(
            {
                "Original hours source code": hours_code,
                "Original hours observation": hours_observation,
                "Original hours observation unit": hours_unit,
                "Original hours classification/revision": hours_class,
                "Original hours dataflow": hours_flow,
                "Original hours source database": hours_db,
            }
        )
        employment_rows.append(
            {
                "Original employment source code": employment_code,
                "Original employment observation": employment_observation,
                "Original employment observation unit": employment_unit,
                "Original employment classification/revision": employment_class,
                "Original employment dataflow": employment_flow,
                "Original employment source database": employment_db,
            }
        )

        if method == "EU_H_DIRECT":
            readable_methods.append("Direct annual-hours observation")
        elif method == "EMP_X_HOURS":
            readable_methods.append("Employment x mean weekly hours")
        elif method == "EMP_SHARE":
            readable_methods.append("Employment-only allocation")
        elif method == "EMP_NATIONAL_RESIDUAL_OUTPUT":
            readable_methods.append("National employment residual allocation using MRIO gross output")
        elif method in {"EU_H_PARENT", "EMP_PARENT_OUTPUT"}:
            if weight_source == "MRIO gross output":
                readable_methods.append("Parent residual allocation using MRIO gross output")
            else:
                readable_methods.append("Parent residual allocation using source weights")
        elif method == "NATIONAL_RESIDUAL_OUTPUT":
            readable_methods.append("National residual allocation using MRIO gross output")
        elif method == "ZERO_OUTPUT_UNRESOLVED":
            readable_methods.append("Excluded: zero modeled output")
        else:
            readable_methods.append("Other allocation method")

    hours_frame = pd.DataFrame(hours_rows, index=detail.index)
    employment_frame = pd.DataFrame(employment_rows, index=detail.index)
    detail["display_allocation_method"] = readable_methods
    detail["Source year"] = detail["source_year"].to_numpy()
    for column in hours_frame.columns:
        detail[column] = hours_frame[column]
    for column in employment_frame.columns:
        detail[column] = employment_frame[column]

    fte_sheet = detail.rename(
        columns={
            "geo": "country",
            "industry": "industry (NACE2)",
            "FTE_country_total": "National FTE40 anchor (thousand)",
            "source_code": "Parent/source industry group",
            "FTE_country_industry": "Estimated FTE (thousand)",
            "FIGARO_output": "FIGARO output (million EUR)",
            "FIGARO_GVA": "FIGARO GVA (million EUR)",
            "FTE_intensity": "FTE per million EUR",
            "final_country_share": "National FTE40 share (%)",
            "display_allocation_method": "Allocation method",
            "allocation_weight_source": "Allocation weight source",
            "allocation_operation": "Allocation operation",
            "allocation_block": "Residual allocation block",
            "parent_output_total": "Residual block output (million EUR)",
            "output_allocation_share": "Residual output share",
            "parent_source_observation": "Parent source observation",
            "parent_residual_observation": "Parent residual observation",
            "constraint_level": "Constraint level",
        }
    )
    fte_sheet["National FTE40 share (%)"] = detail["final_country_share_pct"].to_numpy()
    fte_sheet["Allocation method"] = detail["display_allocation_method"].to_numpy()
    fte_columns = [
        # A. Results and allocation
        "country", "industry (NACE2)", "Estimated FTE (thousand)",
        "FTE per million EUR", "FIGARO output (million EUR)",
        "FIGARO GVA (million EUR)", "National FTE40 anchor (thousand)",
        "National FTE40 share (%)", "Source year", "Allocation method",
        # B. Original hours and source information
        "Original hours source code", "Original hours observation",
        "Original hours observation unit", "Original hours classification/revision",
        "Original hours dataflow", "Original hours source database",
        # C. Original employment and source information
        "Original employment source code", "Original employment observation",
        "Original employment observation unit", "Original employment classification/revision",
        "Original employment dataflow", "Original employment source database",
        # D. Industrial constraint
        "Parent/source industry group", "Parent source observation",
        "Parent residual observation", "Residual allocation block",
        "Residual block output (million EUR)", "Residual output share",
        "Allocation weight source", "Allocation operation", "Constraint level",
    ]
    fte_sheet = fte_sheet.reindex(columns=fte_columns).sort_values(
        ["country", "industry (NACE2)"]
    )
    coefficient = pd.to_numeric(
        fte_sheet["FTE per million EUR"], errors="coerce"
    )
    gross_output = pd.to_numeric(
        fte_sheet["FIGARO output (million EUR)"], errors="coerce"
    )
    zero_output_missing_coefficient = coefficient.isna() & gross_output.eq(0)
    if (coefficient.isna() & ~gross_output.eq(0)).any():
        raise ValueError(
            "C2 export found a missing FTE per million EUR value for a "
            "non-zero FIGARO output row."
        )
    fte_sheet.loc[zero_output_missing_coefficient, "FTE per million EUR"] = 0.0

    sources_inventory = _make_c2_public_source_inventory(
        emp_core=emp_core,
        hours_total=hours_total,
        hours_mean=hours_mean,
        fte_industry=fte_industry,
        fte_audit=fte_audit,
    )
    resolution_profiles = _make_c2_resolution_profile_sheet()
    anchors = (
        fte_industry[["geo", "FTE_country_total"]]
        .drop_duplicates()
        .rename(
            columns={
                "geo": "country",
                "FTE_country_total": "FTE anchor (thousand)",
            }
        )
    )
    global_anchor = float(anchors["FTE anchor (thousand)"].sum())
    max_error = (
        float(fte_audit["reconciliation_error"].abs().max())
        if fte_audit is not None and "reconciliation_error" in fte_audit
        else pd.NA
    )
    readme = pd.DataFrame([
        {
            "Section": "Workbook title",
            "Content": "Supplementary Material C2 - FTE40/output calculation",
        },
        {
            "Section": "FTE40 hierarchy",
            "Content": (
                "The FTE40 anchor is the national full-time-equivalent benchmark. "
                "Industrial observations are resolved to FIGARO/NACE Rev. 2 A64 "
                "through the source classification and coverage profiles shown in "
                "resolution_profiles."
            ),
        },
        {
            "Section": "Scope",
            "Content": (
                f"{anchors.loc[anchors['country'].ne('ROW'), 'country'].nunique()} explicit "
                f"FIGARO countries plus ROW; {fte_sheet['industry (NACE2)'].nunique()} "
                "country-industry observations per 64-industry FIGARO coverage."
            ),
        },
        {
            "Section": "Country calibration",
            "Content": (
                "FTE per million EUR = 1,000 x allocated FTE40 (thousand) / "
                "FIGARO gross output (million EUR)."
            ),
        },
        {
            "Section": "Parent-child constraints",
            "Content": (
                "Observed detailed children are allocated first. Their mass is removed "
                "from observed parents, and only the remaining parent mass is allocated "
                "across unresolved children."
            ),
        },
        {
            "Section": "Observed residual allocation",
            "Content": (
                "Unresolved children receive the residual from the closest compatible observed "
                "parent, or the national FTE40 residual, allocated by their combined FIGARO "
                "gross output. FIGARO GVA is retained as an input diagnostic only, not as an "
                "employment allocation weight."
            ),
        },
        {
            "Section": "Zero-output coefficient convention",
            "Content": (
                "When FIGARO gross output is zero, FTE per million EUR is exported as a "
                "numeric zero because the coefficient is undefined. This convention does "
                "not imply zero allocated FTE40; some zero-output industries retain positive "
                "allocated FTE40."
            ),
        },
        {
            "Section": "Source inventory",
            "Content": (
                "sources_resolution distinguishes selected sources, available but unused "
                "sources, and alternative candidates. Mapped FIGARO coverage counts covered "
                "industries, not independently observed industries. Used in final FTE40 "
                "allocation is Yes only when the source metadata matches a source actually "
                "used by the final hierarchy; a selected source can therefore be marked "
                "No when a higher-priority compatible source supplied the final allocation."
            ),
        },
        {
            "Section": "ROW treatment",
            "Content": "ROW is World FTE40 less the 49 explicit FIGARO country anchors.",
        },
        {
            "Section": "Validation",
            "Content": (
                f"{len(fte_industry):,} country-industry observations; maximum country "
                f"reconciliation error {max_error}. Global allocated FTE anchor total: "
                f"{global_anchor:.6f} thousand."
            ),
        },
    ])
    missing_hours_database = (
        detail["allocation_method"].eq("EMP_X_HOURS")
        & hours_frame["Original hours source database"].isna()
    )
    if missing_hours_database.any():
        readme = pd.concat(
            [
                readme,
                pd.DataFrame([
                    {
                        "Section": "Source metadata limitation",
                        "Content": (
                            f"A separate hours-source database identifier is unavailable in "
                            f"the supplied FTE40 records for {int(missing_hours_database.sum()):,} "
                            "employment x mean-hours rows; those cells are left blank rather "
                            "than inferred or replaced with a synthetic value."
                        ),
                    }
                ]),
            ],
            ignore_index=True,
        )

    frames = {
        "README": readme,
        "wages_calculation": fte_sheet,
        "sources_resolution": sources_inventory,
        "resolution_profiles": resolution_profiles,
    }
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet in sheet_names:
            frames[sheet].to_excel(writer, sheet_name=sheet, index=False)
        _format_c2_workbook(writer.book)
    return output


def export_wage_supplementary_material(
    wage_audit: pd.DataFrame,
    emp_core: pd.DataFrame,
    output_path: str | Path,
    *,
    template_path: str | Path | None = None,
    fte_industry: pd.DataFrame | None = None,
    fte_audit: pd.DataFrame | None = None,
    hours_total: pd.DataFrame | None = None,
    hours_mean: pd.DataFrame | None = None,
) -> Path:
    """Export the C2 workbook using the established interface.

    When ``fte_industry`` is supplied, the workbook documents the FTE40/output
    calculation.  The wage branch remains only for backward-compatible tests
    and callers during migration.
    """
    if fte_industry is not None:
        return _export_fte40_supplementary_material(
            fte_industry=fte_industry,
            fte_audit=fte_audit,
            hours_total=hours_total,
            hours_mean=hours_mean,
            emp_core=emp_core,
            output_path=output_path,
            template_path=template_path,
        )
    required_wage_columns = [
        "country",
        "industry (NACE2)",
        "final_wage",
        "final_comp",
        "comp_codes",
        "final_emp",
        "emp_codes",
        "emp_resolution",
        "emp_dataset",
        "emp_source",
        "calc_note",
    ]
    missing = [c for c in required_wage_columns if c not in wage_audit.columns]
    if missing:
        raise KeyError(
            "wage_audit is missing required C2 columns: "
            f"{missing}."
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    sheet_names = [
        "README",
        "wages_calculation",
        "sources_resolution",
        "resolution_profiles",
    ]
    if template_path is not None:
        template_sheets = pd.ExcelFile(template_path).sheet_names
        sheet_names = [s for s in template_sheets if s in sheet_names]
        sheet_names += [s for s in [
            "README",
            "wages_calculation",
            "sources_resolution",
            "resolution_profiles",
        ] if s not in sheet_names]

    wages_sheet = wage_audit.loc[:, required_wage_columns].copy()
    employment_resolution = _make_employment_resolution_sheet(emp_core).rename(
        columns={"NACE2": "emp_resolution"}
    )
    active_profiles = set(
        employment_resolution["emp_resolution"].dropna().astype(str).str.strip()
    )
    resolution_profiles = _aggregate_coverage_detail_sheet(
        active_profiles=active_profiles
    )
    countries = sorted(wages_sheet["country"].dropna().astype(str).unique())
    explicit = [c for c in countries if c != "ROW"]
    industry_count = wages_sheet["industry (NACE2)"].nunique()
    unresolved = int(wages_sheet["final_wage"].isna().sum())

    readme = pd.DataFrame([
        {
            "Section": "Workbook title",
            "Content": "Supplementary Material C2 - Wages Calculation",
        },
        {
            "Section": "Purpose",
            "Content": (
                "Documents compensation, employment inputs, final wage "
                "coefficients, and the exact calculation rule used by build_wages."
            ),
        },
        {
            "Section": "Scope",
            "Content": (
                f"Covers {len(explicit)} explicit FIGARO countries plus "
                f"ROW across {industry_count} industries."
            ),
        },
        {
            "Section": "Data source",
            "Content": (
                "Compensation is taken from MRIO D1. Employment is sourced "
                "from Eurostat or ILOSTAT according to country coverage."
            ),
        },
        {
            "Section": "ROW treatment",
            "Content": (
                "ROW wages use the explicit-region wage proxy calculated in "
                "build_wages; they do not use country-specific employment data."
            ),
        },
        {
            "Section": "Unresolved wages",
            "Content": str(unresolved),
        },
    ])

    frames = {
        "README": readme,
        "wages_calculation": wages_sheet,
        "sources_resolution": employment_resolution,
        "resolution_profiles": resolution_profiles,
    }

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet in sheet_names:
            frames[sheet].to_excel(writer, sheet_name=sheet, index=False)
        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions

    return output


def _aggregate_coverage_detail_sheet(
    *,
    active_profiles: set[str] | None = None,
) -> pd.DataFrame:
    cols = ["coverage_profile", "source_code", "covers_ind_ava_codes"]
    if aggregate_coverage is None or aggregate_coverage.empty:
        return pd.DataFrame(columns=cols)

    out = aggregate_coverage.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = pd.NA
    out = out.loc[:, cols].copy()
    out["coverage_profile"] = out["coverage_profile"].astype("string").str.strip()
    out["source_code"] = out["source_code"].astype("string").str.strip()
    out["covers_ind_ava_codes"] = out["covers_ind_ava_codes"].astype("string").str.strip()
    if active_profiles is not None:
        active = {
            str(profile).strip()
            for profile in active_profiles
            if pd.notna(profile) and str(profile).strip()
        }
        out = out.loc[out["coverage_profile"].isin(active)].copy()
    return out.loc[out["source_code"].notna()].sort_values(["coverage_profile", "source_code"])


def _make_employment_resolution_sheet(emp_core: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "NACE2",
        "class_system",
        "n_countries",
        "countries",
        "source_dbs",
        "dataflow_ids",
        "years",
        "n_codes_min",
        "n_codes_max",
        "country_code_counts",
    ]
    if emp_core is None or emp_core.empty:
        return pd.DataFrame(columns=cols)

    df = emp_core.copy()
    df["geo"] = df["geo"].astype(str).str.strip()
    df["nace_r2"] = df.get("source_code", df.get("nace_r2")).astype(
        str
    ).str.strip().map(_strip_cpa_prefix)
    if "source_db" not in df.columns:
        df["source_db"] = pd.NA
    if "dataflow_id" not in df.columns:
        df["dataflow_id"] = pd.NA
    df["TIME_PERIOD"] = df.get("year", df.get("TIME_PERIOD", pd.NA))

    country_rows = []
    for geo, g in df.groupby("geo", dropna=False):
        codes_all = {c for c in g["nace_r2"].dropna().astype(str) if c and c != "nan"}
        codes = codes_all - {"TOTAL"} or codes_all
        dataflows = {str(x).strip() for x in g["dataflow_id"].dropna().unique() if str(x).strip()}
        source_dbs = {str(x).strip() for x in g["source_db"].dropna().unique() if str(x).strip()}
        years = {str(x).strip() for x in g["TIME_PERIOD"].dropna().unique() if str(x).strip()}
        profile_label = _detect_aggregate_profile(codes_all) or (
            "TOTAL" if not codes else "DETAIL"
        )
        class_values = sorted({
            str(value).strip()
            for value in g.get("class_system", pd.Series(dtype=object)).dropna().unique()
            if str(value).strip()
        })
        class_system = class_values[0] if len(class_values) == 1 else "MIXED"
        country_rows.append({
            "geo": str(geo),
            "coverage_profile": profile_label,
            "class_system": class_system,
            "source_dbs": ";".join(sorted(source_dbs)),
            "dataflow_ids": ";".join(sorted(dataflows)),
            "years": ";".join(sorted(years)),
            "n_codes": len(codes),
            "codes": ";".join(sorted(codes)),
        })

    country_detail = pd.DataFrame(country_rows)
    if country_detail.empty:
        return pd.DataFrame(columns=cols)

    def join_unique(s):
        vals = sorted({str(x).strip() for x in s if pd.notna(x) and str(x).strip()})
        return ";".join(vals)

    out = (
        country_detail
        .groupby(["coverage_profile", "class_system"], as_index=False, dropna=False)
        .agg(
            countries=("geo", lambda s: ";".join(sorted(map(str, s)))),
            n_countries=("geo", "nunique"),
            source_dbs=("source_dbs", join_unique),
            dataflow_ids=("dataflow_ids", join_unique),
            years=("years", join_unique),
            n_codes_min=("n_codes", "min"),
            n_codes_max=("n_codes", "max"),
            country_code_counts=("geo", lambda s: ""),
        )
    )

    count_map = (
        country_detail
        .assign(pair=lambda d: d["geo"].astype(str) + "=" + d["n_codes"].astype(str))
        .groupby(["coverage_profile", "class_system"], dropna=False)["pair"]
        .agg(lambda s: ";".join(sorted(s)))
        .reset_index()
    )
    out = out.drop(columns=["country_code_counts"]).merge(
        count_map,
        on=["coverage_profile", "class_system"],
        how="left",
    ).rename(columns={"pair": "country_code_counts"})
    out = out.rename(columns={"coverage_profile": "NACE2"})

    return out.loc[:, cols].sort_values(["n_countries", "NACE2"], ascending=[False, True])




def _c2_first_available(frame: pd.DataFrame, columns, *, default=pd.NA) -> pd.Series:
    """Return the first populated value among alternative source columns."""
    result = pd.Series(default, index=frame.index, dtype="object")
    for column in columns:
        if column not in frame.columns:
            continue
        candidate = frame[column]
        missing = result.isna() | result.astype("string").str.strip().isin({"", "<NA>", "nan"})
        result = result.where(~missing, candidate)
    return result


def _c2_prepare_source_frame(
    frame: pd.DataFrame | None,
    *,
    measure: str,
    value_name: str,
    default_unit: str,
    default_source_db: str,
    target_industries: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare raw and mapped rows for the C2 source-resolution report."""
    raw_columns = [
        "geo", "raw_code", "value", "unit", "source_db", "dataflow_id",
        "source_year", "class_system", "population_scope", "measure",
        "candidate_only",
    ]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=raw_columns), pd.DataFrame()

    source = frame.copy()
    raw = pd.DataFrame(index=source.index)
    raw["geo"] = _c2_first_available(source, ["geo", "REF_AREA", "country", "originArea"])
    raw["raw_code"] = _c2_first_available(
        source,
        ["source_code", "raw_code", "nace_r2_raw", "nace_r2", "industry", "industry_code", "ECO", "EC2", "prd_use"],
    )
    raw["value"] = pd.to_numeric(
        _c2_first_available(source, [value_name, "OBS_VALUE", "obs_value", "value"]),
        errors="coerce",
    )
    raw["source_db"] = _c2_first_available(source, ["source_db", "dataset"], default=default_source_db)
    raw["dataflow_id"] = _c2_first_available(source, ["dataflow_id", "indicator", "dataflow"])
    raw["source_year"] = pd.to_numeric(
        _c2_first_available(source, ["source_year", "TIME_PERIOD", "year", "time"]),
        errors="coerce",
    )
    raw["class_system"] = [
        _identify_source_classification(
            source.loc[index],
            source_db=str(raw.at[index, "source_db"]),
            dataflow=str(raw.at[index, "dataflow_id"]),
            code=str(raw.at[index, "raw_code"]),
        )
        for index in source.index
    ]
    raw["population_scope"] = _c2_first_available(source, ["population_scope", "pop_scope"])
    raw["measure"] = measure
    raw["candidate_only"] = False
    raw["geo"] = raw["geo"].astype("string").str.strip().str.upper().replace({"X01": "ROW"})
    raw["raw_code"] = raw["raw_code"].astype("string").str.strip()
    raw = raw.loc[raw["geo"].notna() & raw["raw_code"].notna()]
    raw = raw.loc[~raw["raw_code"].str.upper().isin({"", "NAN"})].copy()
    raw["unit"] = default_unit

    # Use the same correspondence and hierarchy resolution as the allocator.
    standard = raw.rename(columns={"raw_code": "nace_r2", "value": "OBS_VALUE"}).copy()
    mapped = _standardize_fte_source(
        standard,
        target_industries=target_industries,
        value_name="OBS_VALUE",
        source_name=default_source_db,
    )
    def _resolution_key_part(value) -> str:
        if pd.isna(value):
            return ""
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.notna(numeric):
            return str(int(numeric)) if float(numeric).is_integer() else str(numeric)
        return str(value).strip()

    if not mapped.empty:
        mapped["measure"] = measure
        mapped["unit"] = default_unit
        mapped["_resolution_key"] = (
            mapped["geo"].map(_resolution_key_part).str.upper().str.strip()
            + "|" + mapped["source_db"].map(_resolution_key_part)
            + "|" + mapped["dataflow_id"].map(_resolution_key_part)
            + "|" + mapped["source_year"].map(_resolution_key_part)
            + "|" + mapped["class_system"].map(_resolution_key_part)
        )
    raw["_resolution_key"] = (
        raw["geo"].map(_resolution_key_part).str.upper().str.strip()
        + "|" + raw["source_db"].map(_resolution_key_part)
        + "|" + raw["dataflow_id"].map(_resolution_key_part)
        + "|" + raw["source_year"].map(_resolution_key_part)
        + "|" + raw["class_system"].map(_resolution_key_part)
    )
    return raw.loc[:, raw_columns + ["_resolution_key"]], mapped


def _c2_selected_measure_status(
    country: str,
    measure: str,
    *,
    fte_industry: pd.DataFrame,
) -> tuple[str, str]:
    rows = fte_industry.loc[fte_industry["geo"].astype(str).str.upper().eq(str(country).upper())]
    methods = set(rows.get("allocation_method", pd.Series(dtype=object)).astype(str))
    if measure == "Annual total hours":
        selected = bool(methods & {"EU_H_DIRECT", "EU_H_PARENT"})
        return ("selected" if selected else "unused", "annual-hours constraint" if selected else "not selected by the FTE40 hierarchy")
    if measure == "Mean weekly hours":
        selected = bool(methods & {"EMP_X_HOURS"}) or bool(rows.get("hours_source_code", pd.Series(dtype=object)).notna().any())
        return ("selected" if selected else "unused", "employment × mean-hours allocation" if selected else "not selected by the FTE40 hierarchy")
    if measure == "Industrial employment":
        selected = bool(methods & {
            "EMP_X_HOURS", "EMP_SHARE", "EMP_PARENT_OUTPUT",
            "EMP_NATIONAL_RESIDUAL_OUTPUT",
        })
        return ("selected" if selected else "unused", "employment constraint or labour-volume weight" if selected else "not selected by the FTE40 hierarchy")
    return "unused", "not an industrial allocation source"


def _make_c2_sources_resolution(
    *,
    emp_core: pd.DataFrame | None,
    hours_total: pd.DataFrame | None,
    hours_mean: pd.DataFrame | None,
    fte_industry: pd.DataFrame,
    fte_audit: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    target_industries = set(fte_industry["industry"].astype(str).str.strip())
    source_specs = [
        ("Annual total hours", hours_total, "OBS_VALUE", "thousand hours/year", "Eurostat"),
        ("Industrial employment", emp_core, "employment", "thousand employed persons", "employment source"),
        ("Mean weekly hours", hours_mean, "hours", "hours per employed person per week", "ILOSTAT"),
    ]
    raw_frames = []
    mapped_frames = []
    for measure, frame, value_name, unit, source_name in source_specs:
        raw, mapped = _c2_prepare_source_frame(
            frame,
            measure=measure,
            value_name=value_name,
            default_unit=unit,
            default_source_db=source_name,
            target_industries=target_industries,
        )
        raw_frames.append(raw)
        if mapped is not None and not mapped.empty:
            mapped_frames.append(mapped)

    # Preserve candidate ILO source records without mixing them with final
    # country-industry allocations. Candidate values are not used as model inputs.
    if fte_audit is not None:
        availability = fte_audit.attrs.get("source_availability_audit", pd.DataFrame())
        if availability is not None and not availability.empty:
            candidate_rows = []
            for row in availability.to_dict("records"):
                measure_text = str(row.get("measure", ""))
                measure = "Mean weekly hours" if "hour" in measure_text.lower() else "Industrial employment"
                codes = _split_coverage_codes(row.get("original_source_codes", ""))
                if not codes:
                    continue
                for code in codes:
                    candidate_rows.append({
                        "geo": row.get("geo", pd.NA), "raw_code": code, "value": 1.0,
                        "unit": "candidate source code", "source_db": "ILOSTAT",
                        "dataflow_id": row.get("indicator", pd.NA),
                        "source_year": row.get("source_year", pd.NA),
                        "class_system": row.get("class_system", pd.NA),
                        "population_scope": pd.NA, "measure": measure,
                        "candidate_only": True,
                    })
            if candidate_rows:
                candidate_frame = pd.DataFrame(candidate_rows)
                candidate_raw, candidate_mapped = _c2_prepare_source_frame(
                    candidate_frame,
                    measure="candidate",
                    value_name="value",
                    default_unit="candidate source code",
                    default_source_db="ILOSTAT",
                    target_industries=target_industries,
                )
                candidate_raw["measure"] = candidate_frame["measure"].reindex(candidate_raw.index).to_numpy()
                raw_frames.append(candidate_raw)
                if candidate_mapped is not None and not candidate_mapped.empty:
                    candidate_mapped["measure"] = "candidate"
                    mapped_frames.append(candidate_mapped)

    candidate_resolution_lookup = {}
    if fte_audit is not None:
        availability = fte_audit.attrs.get("source_availability_audit", pd.DataFrame())
        if availability is not None and not availability.empty:
            for row in availability.to_dict("records"):
                measure_label = str(row.get("measure", ""))
                measure_label = (
                    "Mean weekly hours" if "hour" in measure_label.lower()
                    else "Industrial employment"
                )
                lookup_key = (
                    measure_label,
                    str(row.get("geo", "")),
                    str(row.get("indicator", row.get("dataflow_id", ""))),
                    str(row.get("source_year", "")),
                    str(row.get("class_system", "")),
                )
                candidate_resolution_lookup[lookup_key] = row

    raw_all = pd.concat([frame for frame in raw_frames if not frame.empty], ignore_index=True) if any(not frame.empty for frame in raw_frames) else pd.DataFrame()
    mapped_all = pd.concat(mapped_frames, ignore_index=True) if mapped_frames else pd.DataFrame()
    if raw_all.empty:
        columns = [
            "Measure", "Source database", "Dataflow / indicator", "Original classification/revision",
            "Effective FIGARO resolution", "Named source profile", "Number of countries", "Countries", "Years represented",
            "Minimum A64 coverage", "Maximum A64 coverage", "Countries where source is used",
        ]
        return pd.DataFrame(columns=columns), pd.DataFrame(), set()

    inventory_rows = []
    group_cols = ["measure", "geo", "source_db", "dataflow_id", "source_year", "class_system"]
    for key, raw_group in raw_all.groupby(group_cols, dropna=False):
        measure, country, source_db, dataflow, year, class_system = key
        # ``_resolution_key`` identifies a source observation, not the
        # report's display measure.  The latter is deliberately different
        # for employment, total hours and mean hours even when the underlying
        # source row has the same metadata.
        key_text = "|".join(str(value) for value in key[1:])
        mapped_group = mapped_all.loc[mapped_all.get("_resolution_key", pd.Series(dtype=str)).eq(key_text)] if not mapped_all.empty else pd.DataFrame()
        # Candidate rows are diagnostics only; they are not observations used by the model.
        usable = raw_group.loc[~raw_group["candidate_only"].astype(bool)]
        source_codes = {
            str(value).strip() for value in raw_group["raw_code"].dropna()
            if str(value).strip().upper() not in {"TOTAL", "NAN"}
        }
        valid_codes = set(mapped_group["source_code"].astype(str)) if not mapped_group.empty else set()
        blocks = _c2_effective_blocks(mapped_group)
        covered = set().union(
            *(set(targets) for targets in mapped_group["targets"] if isinstance(targets, tuple))
        ) if not mapped_group.empty else set()
        named_profile, completeness = _c2_profile_label(
            mapped=mapped_group,
            n_covered=len(covered),
            n_target=len(target_industries),
        )
        candidate_key = (
            "Mean weekly hours" if "hour" in str(measure).lower() else "Industrial employment",
            str(country),
            str(dataflow),
            str(year),
            str(class_system),
        )
        candidate_diag = candidate_resolution_lookup.get(candidate_key)
        if candidate_diag is not None:
            named_profile = candidate_diag.get(
                "named_source_profile", candidate_diag.get("resolution_profile", named_profile)
            )
            completeness = str(candidate_diag.get("resolution_completeness", completeness))
            n_valid_observations = int(candidate_diag.get("valid_source_observations", len(usable)))
            n_mapped_observations = int(candidate_diag.get("mapped_source_observations", len(mapped_group)))
            n_blocks = int(candidate_diag.get(
                "effective_resolution", candidate_diag.get("independent_blocks", len(blocks))
            ))
            n_exact = int(candidate_diag.get("exact_A64", 0))
            n_aggregate = int(candidate_diag.get("aggregate_blocks", 0))
            n_covered = int(candidate_diag.get("A64_covered", len(covered)))
            unmapped_codes = str(candidate_diag.get("unmapped_codes", ""))
            ambiguous_codes = str(candidate_diag.get("ambiguous_codes", ""))
        else:
            n_valid_observations = int(len(usable))
            n_mapped_observations = int(len(mapped_group))
            n_blocks = int(len(blocks))
            n_exact = int(sum(len(block["effective_targets"]) == 1 and block["source_is_exact"] for block in blocks))
            n_aggregate = int(sum(not (len(block["effective_targets"]) == 1 and block["source_is_exact"]) for block in blocks))
            n_covered = int(len(covered))
            unmapped_codes = ";".join(sorted(source_codes - valid_codes))
            ambiguous_codes = ""
        status, reason = _c2_selected_measure_status(str(country), str(measure), fte_industry=fte_industry)
        if bool(raw_group["candidate_only"].all()):
            if candidate_diag is not None and bool(candidate_diag.get("selected", False)):
                status = "selected"
                reason = str(candidate_diag.get("selection_reason", "selected by effective resolution"))
            else:
                status = "candidate / unused"
                reason = "candidate retained for source-resolution comparison"
        selected_rows = fte_industry.loc[fte_industry["geo"].astype(str).str.upper().eq(str(country).upper())]
        compatible_pairs = pd.NA
        if fte_audit is not None and not fte_audit.empty and "geo" in fte_audit:
            audit_rows = fte_audit.loc[fte_audit["geo"].astype(str).str.upper().eq(str(country).upper())]
            if not audit_rows.empty:
                compatible_pairs = audit_rows.get("n_compatible_emp_hours_pairs", pd.Series(pd.NA)).iloc[0]
        inventory_rows.append({
            "Country": str(country), "Measure": str(measure), "Statistical unit": str(raw_group["unit"].iloc[0]),
            "Source database": str(source_db), "Dataflow / indicator": str(dataflow),
            "Original classification/revision": str(class_system), "Available year": year,
            "Effective FIGARO resolution": n_blocks,
            "Named source profile": named_profile,
            "Resolution completeness": completeness,
            "n_raw_codes": len(source_codes), "n_valid_source_codes": len(valid_codes),
            "n_valid_source_observations": n_valid_observations,
            "n_mapped_source_observations": n_mapped_observations,
            "n_effective_industrial_blocks": n_blocks,
            "n_exact_A64": n_exact,
            "n_aggregate_blocks": n_aggregate,
            "n_A64_covered": n_covered, "Exact A64 industries": ";".join(sorted({next(iter(block["effective_targets"])) for block in blocks if len(block["effective_targets"]) == 1 and block["source_is_exact"]})),
            "Aggregate blocks": ";".join(f"{block['source_code']}->{len(block['effective_targets'])}" for block in blocks if not (len(block["effective_targets"]) == 1 and block["source_is_exact"])),
            "Unmapped or ambiguous source observations": ";".join(filter(None, [unmapped_codes, ambiguous_codes])),
            "Selection status": status, "Selection reason": reason,
            "Compatible employment-hours pairs": compatible_pairs,
            "FTE allocation coverage": int(selected_rows["industry"].nunique()) if not selected_rows.empty else 0,
        })

    inventory = pd.DataFrame(inventory_rows)
    if not inventory.empty:
        inventory["Maximum available candidate"] = False
        for (_country, _measure), group in inventory.groupby(["Country", "Measure"], dropna=False):
            score = group[
                ["n_A64_covered", "n_effective_industrial_blocks", "n_exact_A64", "n_aggregate_blocks"]
            ].copy()
            score["n_aggregate_blocks"] = -pd.to_numeric(score["n_aggregate_blocks"], errors="coerce")
            score = score.apply(pd.to_numeric, errors="coerce").fillna(-1)
            best_index = tuple(score.sort_values(
                ["n_A64_covered", "n_effective_industrial_blocks", "n_exact_A64", "n_aggregate_blocks"],
                ascending=False,
            ).index[:1])
            inventory.loc[list(best_index), "Maximum available candidate"] = True
    summary = (
        inventory.groupby(
            ["Measure", "Source database", "Dataflow / indicator", "Original classification/revision", "Effective FIGARO resolution", "Named source profile"],
            dropna=False,
            as_index=False,
        )
        .agg(
            **{
                "Number of countries": ("Country", "nunique"),
                "Countries": ("Country", lambda values: ";".join(sorted(set(map(str, values))))),
                "Years represented": ("Available year", lambda values: ";".join(sorted({str(value) for value in values if pd.notna(value) and str(value) != "<NA>"}))),
                "Minimum A64 coverage": ("n_A64_covered", "min"),
                "Maximum A64 coverage": ("n_A64_covered", "max"),
                "Countries where source is used": ("Selection status", lambda values: int(sum(str(value) in {"selected", "partially used"} for value in values))),
            }
        )
        .sort_values(["Measure", "Number of countries", "Effective FIGARO resolution"], ascending=[True, False, True])
    )
    profiles = set(inventory["Named source profile"].dropna().astype(str))
    if mapped_all is not None and not mapped_all.empty:
        for _key, mapped_group in mapped_all.groupby(
            ["_resolution_key"], dropna=False
        ):
            raw_profile = _detect_aggregate_profile(
                set(mapped_group["source_code"].astype(str))
            )
            if raw_profile in _aggregate_profile_source_sets():
                profiles.add(raw_profile)
    return summary, inventory, profiles


def _c2_text(value) -> str:
    """Return a stable display string for a supplementary workbook cell."""
    if value is None or value is pd.NA:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _c2_measure_label(value) -> str:
    text = _c2_text(value).lower()
    if "hour" in text and ("mean" in text or "weekly" in text):
        return "Mean weekly hours"
    if "hour" in text:
        return "Annual total hours"
    if "employ" in text:
        return "Industrial employment"
    return _c2_text(value) or "Unspecified measure"


def _c2_classification_label(value) -> str:
    text = _c2_text(value)
    normalized = re.sub(r"[ ._-]+", "", text).upper()
    if normalized in {"ISIC4", "ISICREV4"}:
        return "ISIC Rev. 4"
    return text


def _make_c2_public_source_inventory(
    *,
    emp_core: pd.DataFrame | None,
    hours_total: pd.DataFrame | None,
    hours_mean: pd.DataFrame | None,
    fte_industry: pd.DataFrame,
    fte_audit: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build one publication-facing source inventory from the audit metadata.

    The source-availability audit is produced by the existing source-selection
    logic.  This function only reshapes it for publication: it does not select
    sources, remap industries, or recalculate any FTE values.
    """
    columns = [
        "Country", "Measure", "Selection status", "Used in final FTE40 allocation",
        "Source priority", "Classification / revision",
        "Independent industrial blocks", "Mapped FIGARO industries (of 64)",
        "Selection reason", "Source database", "Dataflow / indicator", "Source year",
        "Resolution profile", "Resolution completeness",
        "Valid source observations", "Original source codes",
        "Unmapped or ambiguous source observations",
        "FTE allocation coverage (of 64)",
    ]

    availability = pd.DataFrame()
    if fte_audit is not None:
        candidate = fte_audit.attrs.get("source_availability_audit", pd.DataFrame())
        if candidate is not None and not candidate.empty:
            availability = candidate.copy()

    # Preserve compatibility with callers that provide raw source frames but
    # not the audit attribute.  The legacy helper supplies the same source
    # decisions, and is used only as a presentation fallback here.
    if availability.empty:
        _, legacy_inventory, _ = _make_c2_sources_resolution(
            emp_core=emp_core,
            hours_total=hours_total,
            hours_mean=hours_mean,
            fte_industry=fte_industry,
            fte_audit=fte_audit,
        )
        if legacy_inventory.empty:
            return pd.DataFrame(columns=columns)
        availability = legacy_inventory.rename(
            columns={
                "Country": "geo",
                "Measure": "measure",
                "Source database": "source_db",
                "Dataflow / indicator": "dataflow_id",
                "Available year": "source_year",
                "Original classification/revision": "class_system",
                "Effective FIGARO resolution": "independent_blocks",
                "Named source profile": "named_source_profile",
                "Resolution completeness": "resolution_completeness",
                "n_valid_source_observations": "valid_source_observations",
                "n_A64_covered": "A64_covered",
                "Original source codes": "original_source_codes",
                "Selection status": "selection_status",
                "Selection reason": "selection_reason",
            }
        )
        availability["selected"] = availability.get(
            "selection_status", pd.Series("", index=availability.index)
        ).astype(str).eq("selected")

    # A country has at most one row per genuine source candidate.  Candidate
    # rows without any source/database/indicator/code identifier are synthetic
    # implementation records and are deliberately omitted.
    employment_methods = {
        "EMP_X_HOURS", "EMP_SHARE", "EMP_PARENT_OUTPUT",
        "EMP_NATIONAL_RESIDUAL_OUTPUT",
    }
    hours_methods = {"EU_H_DIRECT", "EU_H_PARENT"}

    def actual_fte40_use(row: dict) -> str:
        country = _c2_text(row.get("geo")).upper()
        measure = _c2_measure_label(row.get("measure"))
        selected_rows = fte_industry.loc[
            fte_industry["geo"].astype(str).str.upper().eq(country)
        ]
        if selected_rows.empty:
            return "not used"
        dataflow = _c2_text(row.get("dataflow_id")) or _c2_text(row.get("indicator"))
        source_db = _c2_text(row.get("source_db")) or _c2_text(row.get("source"))
        source_year = _c2_text(row.get("source_year"))

        def same_year(left, right) -> bool:
            if not left or not right:
                return True
            left_num = pd.to_numeric(pd.Series([left]), errors="coerce").iloc[0]
            right_num = pd.to_numeric(pd.Series([right]), errors="coerce").iloc[0]
            if pd.notna(left_num) and pd.notna(right_num):
                return int(left_num) == int(right_num)
            return str(left) == str(right)

        def same_metadata(candidate_rows: pd.DataFrame, *, flow_column: str) -> bool:
            if candidate_rows.empty:
                return False
            for candidate in candidate_rows.to_dict("records"):
                candidate_flow = _c2_text(candidate.get(flow_column))
                candidate_db = _c2_text(candidate.get("source_db"))
                candidate_year = _c2_text(
                    candidate.get("source_year", candidate.get("source_year"))
                )
                flow_match = not dataflow or not candidate_flow or dataflow == candidate_flow
                db_match = not source_db or not candidate_db or source_db == candidate_db
                year_match = same_year(source_year, candidate_year)
                if flow_match and db_match and year_match:
                    return True
            return False

        if measure == "Annual total hours":
            return "used" if same_metadata(
                selected_rows.loc[selected_rows["allocation_method"].isin(hours_methods)],
                flow_column="dataflow_id",
            ) else "not used"
        if measure == "Mean weekly hours":
            candidates = selected_rows.loc[
                selected_rows["allocation_method"].eq("EMP_X_HOURS")
            ].copy()
            if candidates.empty:
                return "not used"
            for candidate in candidates.to_dict("records"):
                flow_match = (
                    not dataflow
                    or not _c2_text(candidate.get("hours_dataflow_id"))
                    or dataflow == _c2_text(candidate.get("hours_dataflow_id"))
                )
                year_match = (
                    not source_year
                    or not _c2_text(candidate.get("hours_source_year"))
                    or same_year(source_year, _c2_text(candidate.get("hours_source_year")))
                )
                if flow_match and year_match:
                    return "used"
            return "not used"
        if measure == "Industrial employment":
            return "used" if same_metadata(
                selected_rows.loc[selected_rows["allocation_method"].isin(employment_methods)],
                flow_column="dataflow_id",
            ) else "not used"
        return "not used"

    records = []
    for row in availability.to_dict("records"):
        source_db = _c2_text(row.get("source_db"))
        source_name = _c2_text(row.get("source"))
        dataflow = _c2_text(row.get("dataflow_id")) or _c2_text(row.get("indicator"))
        original_codes = _c2_text(row.get("original_source_codes"))
        if not any((source_db, source_name, dataflow, original_codes)):
            continue

        selected = bool(row.get("selected", False))
        if selected:
            status = "selected"
            priority = 1
        elif source_db or source_name:
            status = "available but unused"
            priority = 2
        else:
            status = "alternative candidate"
            priority = 3

        if not source_db:
            source_db = source_name or (
                "OECD (candidate)"
                if dataflow.upper().startswith("OECD:")
                else "ILOSTAT (candidate)" if dataflow else "Unidentified source"
            )

        actual_use = actual_fte40_use(row)
        if actual_use == "used":
            actual_use = "Yes"
        elif selected:
            actual_use = "No — selected source not used by final hierarchy"
        else:
            actual_use = "No"

        records.append({
            "Country": _c2_text(row.get("geo")),
            "Measure": _c2_measure_label(row.get("measure")),
            "Selection status": status,
            "Used in final FTE40 allocation": actual_use,
            "Source priority": priority,
            "Classification / revision": _c2_classification_label(row.get("class_system")),
            "Independent industrial blocks": row.get(
                "independent_blocks", row.get("effective_resolution", pd.NA)
            ),
            "Mapped FIGARO industries (of 64)": row.get("A64_covered", pd.NA),
            "Selection reason": _c2_text(row.get("selection_reason")),
            "Source database": source_db,
            "Dataflow / indicator": dataflow,
            "Source year": row.get("source_year", pd.NA),
            "Resolution profile": _c2_text(
                row.get("named_source_profile") or row.get("resolution_profile")
            ),
            "Resolution completeness": _c2_text(row.get("resolution_completeness")),
            "Valid source observations": row.get(
                "valid_source_observations", pd.NA
            ),
            "Original source codes": original_codes,
            "Unmapped or ambiguous source observations": ";".join(
                value for value in (
                    _c2_text(row.get("unmapped_codes")),
                    _c2_text(row.get("ambiguous_codes")),
                ) if value
            ),
        })

    inventory = pd.DataFrame(records)
    if inventory.empty:
        return pd.DataFrame(columns=columns)

    coverage = (
        fte_industry.assign(_country=fte_industry["geo"].astype(str).str.strip())
        .groupby("_country")["industry"]
        .nunique()
        .to_dict()
    )
    inventory["FTE allocation coverage (of 64)"] = inventory["Country"].map(coverage)
    inventory["_measure_order"] = inventory["Measure"].map({
        "Annual total hours": 1,
        "Industrial employment": 2,
        "Mean weekly hours": 3,
    }).fillna(99)
    inventory = inventory.drop_duplicates(
        subset=[
            "Country", "Measure", "Source database", "Dataflow / indicator",
            "Source year", "Classification / revision", "Resolution profile",
            "Original source codes", "Selection status",
        ],
        keep="first",
    )
    inventory = inventory.sort_values(
        ["Country", "_measure_order", "Source priority", "Source database", "Dataflow / indicator"],
        kind="stable",
    ).drop(columns="_measure_order")
    return inventory.reindex(columns=columns).reset_index(drop=True)


def _make_c2_resolution_profile_sheet() -> pd.DataFrame:
    """Export the actual industrial correspondences used or referenced by C2."""
    columns = [
        "Mapping family", "Source classification", "Source code",
        "Target classification", "Target code", "Target count",
        "Profile", "Mapping type", "Used by FTE40 source selection", "Notes",
    ]
    rows = []

    def add(
        family,
        source_classification,
        source_code,
        target_classification,
        target_code,
        *,
        target_count=None,
        profile="",
        mapping_type="",
        used="Yes",
        notes="",
    ):
        rows.append({
            "Mapping family": family,
            "Source classification": source_classification,
            "Source code": source_code,
            "Target classification": target_classification,
            "Target code": target_code,
            "Target count": target_count,
            "Profile": profile,
            "Mapping type": mapping_type,
            "Used by FTE40 source selection": used,
            "Notes": notes,
        })

    for family, source_classification, mapper, profile in [
        (
            "ISIC4_L2_TO_NACE_REV2_A64",
            "ISIC Rev. 4 detailed",
            _isic4_l2_to_nace_rev2_a64_map(),
            "ISIC4_DETAILED",
        ),
    ]:
        if not mapper:
            add(
                family,
                source_classification,
                "MISSING",
                "FIGARO / NACE Rev. 2 A64",
                "MISSING",
                profile=profile,
                mapping_type="Missing correspondence",
                used="No",
                notes="No correspondence was available in the implementation at export time.",
            )
            continue
        for source_code, targets in sorted(mapper.items()):
            targets = sorted(targets)
            for target_code in targets:
                add(
                    family,
                    source_classification,
                    source_code,
                    "FIGARO / NACE Rev. 2 A64",
                    target_code,
                    target_count=len(targets),
                    profile=profile,
                    mapping_type="Detailed crosswalk",
                    notes="Used by the existing classified-source coverage resolver.",
                )

    if aggregate_coverage is not None and not aggregate_coverage.empty:
        for row in aggregate_coverage.to_dict("records"):
            profile = _c2_text(row.get("coverage_profile"))
            source_code = _c2_text(row.get("source_code"))
            if not profile or not source_code or source_code == "__ISIC4_L2_CROSSWALK__":
                continue
            targets = [
                code for code in _split_coverage_codes(row.get("covers_ind_ava_codes"))
                if _c2_text(code)
            ]
            for target_code in targets:
                add(
                    family=profile,
                    source_classification=(
                        "ISIC Rev. 4 aggregate"
                        if profile.startswith("ISIC4_")
                        else "Aggregated employment profile"
                    ),
                    source_code=source_code,
                    target_classification="FIGARO / NACE Rev. 2 A64",
                    target_code=target_code,
                    target_count=len(targets),
                    profile=profile,
                    mapping_type="Aggregated coverage profile",
                    notes="Used by the existing aggregate-coverage resolver.",
                )

    return pd.DataFrame(rows, columns=columns).drop_duplicates().sort_values(
        ["Mapping family", "Source code", "Target code"], kind="stable"
    ).reset_index(drop=True)


def _format_c2_workbook(workbook) -> None:
    """Apply consistent, presentation-only formatting to the four C2 sheets."""
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from numbers import Number

    accounting_zero = '_(* #,##0_);_(* (#,##0);_(* 0_);_(@_)'
    accounting_one = '_(* #,##0.0_);_(* (#,##0.0);_(* 0.0_);_(@_)'
    number_zero = '0'

    header_fill = PatternFill("solid", fgColor="1F4E78")
    alternate_fill = PatternFill("solid", fgColor="F3F6F9")
    header_font = Font(name="Aptos", size=10, bold=True, color="FFFFFF")
    body_font = Font(name="Aptos", size=10, color="000000")
    border = Border(bottom=Side(style="thin", color="D9E2F3"))

    for worksheet in workbook.worksheets:
        worksheet.sheet_view.showGridLines = False
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        worksheet.sheet_properties.pageSetUpPr.fitToPage = True
        worksheet.page_setup.fitToWidth = 1
        worksheet.page_setup.fitToHeight = 0
        worksheet.page_setup.orientation = "landscape"
        worksheet.row_dimensions[1].height = 20

        for cell in worksheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=False)
            cell.border = border

        for row_number in range(2, worksheet.max_row + 1):
            fill = alternate_fill if row_number % 2 == 0 else PatternFill(fill_type=None)
            worksheet.row_dimensions[row_number].height = 18
            for cell in worksheet[row_number]:
                cell.font = body_font
                cell.fill = fill
                cell.alignment = Alignment(vertical="top", wrap_text=False)
                cell.border = border

        if worksheet.title == "README":
            worksheet.freeze_panes = "A2"
            worksheet.column_dimensions["A"].width = 28
            worksheet.column_dimensions["B"].width = 120
        else:
            for column in worksheet.iter_cols(1, worksheet.max_column):
                header = _c2_text(column[0].value).lower()
                width = max(12, min(34, len(_c2_text(column[0].value)) + 3))
                if "reason" in header or "notes" in header or "source codes" in header:
                    width = 42
                elif "coverage" in header or "observation" in header:
                    width = 30
                worksheet.column_dimensions[column[0].column_letter].width = width

                if worksheet.title == "wages_calculation":
                    if header == "national fte40 share (%)":
                        number_format = accounting_one
                    elif header == "source year":
                        number_format = number_zero
                    elif any(
                        token in header
                        for token in (
                            "estimated fte", "fte per", "figaro output", "figaro gva",
                            "national fte40 anchor", "observation",
                        )
                    ):
                        number_format = accounting_zero
                    else:
                        number_format = "@"
                else:
                    if "year" in header:
                        number_format = number_zero
                    elif any(token in header for token in ("share", "percentage", "(%)")):
                        number_format = accounting_one
                    elif any(
                        token in header
                        for token in (
                            "count", "priority", "blocks", "industries", "coverage",
                            "observations", "target count",
                        )
                    ):
                        number_format = accounting_zero
                    else:
                        number_format = "@"

                for cell in column[1:]:
                    if isinstance(cell.value, Number) and not isinstance(cell.value, bool):
                        cell.number_format = number_format
                    else:
                        cell.number_format = "@"






def plot_wages_dispersion_boxplot_sorted(
    wages: pd.Series,
    *,
    eu_countries: set[str] | None = None,
    group_sets: dict[str, set[str]] | None = None,
    emp_mrio: pd.Series | None = None,
    eu27_countries: set[str] | None = None,
    filename="wages_dispersion_boxplot_sorted.pdf",
    save_figures=True,
    close_figures=True,
    figsize=(11, 5),
    dpi=150,
    stat="median",   # "median" o "p75"
    ylabel: str = "Compensations (kEUR/FTE-year)",
    show_eu27_median_line: bool = True,
):
    set_house_style()

    w = wages.copy()
    w.name = "wage"
    w = w.reset_index()
    w["destination_region"] = w["destination_region"].astype(str).str.strip()
    w["wage"] = pd.to_numeric(w["wage"], errors="coerce")
    w = w[w["wage"] > 0].copy()
    w["wage"] = w["wage"] / 1_000.0

    employment = None
    if emp_mrio is not None:
        employment = emp_mrio.rename("employment").reset_index()
        employment["destination_region"] = employment["destination_region"].astype(str).str.strip()
        employment["employment"] = pd.to_numeric(
            employment["employment"], errors="coerce"
        )
        w = w.merge(
            employment,
            on=["destination_region", "prd_use"],
            how="left",
        )
        w["employment"] = w["employment"].fillna(0.0)

    # -------- robust statistic per region --------
    if stat == "median":
        score = w.groupby("destination_region")["wage"].median()
    elif stat == "p75":
        score = w.groupby("destination_region")["wage"].quantile(0.75)
    else:
        raise ValueError("stat must be 'median' or 'p75'")

    # -------- assign mutually exclusive country groups --------
    if group_sets is None:
        group_sets = {"EUROPE": set(eu_countries or set())}

    group_order = list(group_sets)
    assigned = set()
    group_regions = {}
    for group_name in group_order:
        members = {str(country).strip().upper() for country in group_sets[group_name]}
        overlap = members & assigned
        if overlap:
            raise ValueError(
                f"Country groups must be mutually exclusive; overlapping countries: "
                f"{sorted(overlap)}"
            )
        assigned.update(members)
        group_regions[group_name] = score[
            score.index.astype(str).str.upper().isin(members)
        ].sort_values(ascending=False)

    row_regions = score[
        ~score.index.astype(str).str.upper().isin(assigned)
    ].sort_values(ascending=False)
    group_regions["ROW"] = row_regions
    if "ROW" not in group_order:
        group_order.append("ROW")

    def _weighted_boxplot_stats(values, weights):
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        valid = (
            np.isfinite(values)
            & np.isfinite(weights)
            & (weights > 0)
        )
        values = values[valid]
        weights = weights[valid]
        if values.size == 0 or weights.sum() <= 0:
            return None

        order = np.argsort(values)
        values = values[order]
        weights = weights[order]
        cumulative = np.cumsum(weights) / weights.sum()

        def weighted_quantile(q):
            index = int(np.searchsorted(cumulative, q, side="left"))
            return float(values[min(index, len(values) - 1)])

        q1 = weighted_quantile(0.25)
        median = weighted_quantile(0.50)
        q3 = weighted_quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        in_whisker = values[(values >= lower) & (values <= upper)]
        whislo = float(in_whisker.min()) if in_whisker.size else float(values.min())
        whishi = float(in_whisker.max()) if in_whisker.size else float(values.max())
        return {
            "q1": q1,
            "med": median,
            "q3": q3,
            "whislo": whislo,
            "whishi": whishi,
            "fliers": [],
        }

    def _weighted_distribution(region_mask):
        subset = w.loc[region_mask]
        if subset.empty:
            return None
        if employment is None:
            values = subset["wage"].to_numpy(dtype=float)
            return _weighted_boxplot_stats(values, np.ones(values.size))
        return _weighted_boxplot_stats(
            subset["wage"].to_numpy(dtype=float),
            subset["employment"].to_numpy(dtype=float),
        )

    special_group = None
    special_eu27 = {
        str(country).strip().upper()
        for country in (eu27_countries or set())
        if str(country).strip()
    }
    if special_eu27:
        for group_name in group_order:
            members = {
                str(country).strip().upper()
                for country in group_sets[group_name]
            }
            if special_eu27.issubset(members):
                special_group = group_name
                break

    specs = []
    group_ranges = []
    position = 0
    for group_name in group_order:
        labels = group_regions[group_name].index.tolist()
        start = position
        for region in labels:
            subset = w["destination_region"].eq(region)
            specs.append((region, subset, False))
            position += 1
        if group_name == special_group:
            subset = w["destination_region"].str.upper().isin(special_eu27)
            if subset.any():
                specs.append(("EU27", subset, True))
                position += 1
        if position > start:
            group_ranges.append((group_name, start + 1, position))

    eu27_median = None
    eu27_position = None
    if employment is not None and special_group is not None:
        eu27_stats = _weighted_distribution(w["destination_region"].str.upper().isin(special_eu27))
        if eu27_stats is not None:
            eu27_median = eu27_stats["med"]

    from matplotlib.cbook import boxplot_stats

    stats = []
    labels = []
    for label, mask, weighted in specs:
        if weighted:
            current = _weighted_distribution(mask)
        else:
            values = w.loc[mask, "wage"].to_numpy(dtype=float)
            current = boxplot_stats(values, whis=1.5)[0] if values.size else None
        if current is None:
            continue
        stats.append(current)
        labels.append(label)
        if label == "EU27":
            eu27_position = len(stats)

    # -------- plotting --------
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    bp = ax.bxp(
        stats,
        positions=np.arange(1, len(stats) + 1),
        showfliers=False,
    )
    for median in bp["medians"]:
        median.set_color("#8B0000")

    # Visual separators and group labels replacing the old x-axis label.
    group_centers = []
    for group_name, start, end in group_ranges:
        group_centers.append((group_name, (start + end) / 2))
        if end < len(stats):
            ax.axvline(end + 0.5, color="0.65", linewidth=1)

    if show_eu27_median_line and eu27_median is not None:
        ax.plot(
            [eu27_position, len(stats) + 0.5],
            [eu27_median, eu27_median],
            color="#8B0000",
            linestyle="--",
            linewidth=0.9,
            alpha=0.65,
            zorder=0,
        )

    group_label_size = ax.xaxis.label.get_size()
    for group_name, center in group_centers:
        ax.text(
            center,
            -0.13,
            tex_safe(group_name),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=group_label_size,
            fontweight="bold",
            clip_on=False,
        )

    ax.set_ylabel(tex_safe(ylabel), fontweight="bold")
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels([tex_safe(label) for label in labels])
    ax.tick_params(axis="x", rotation=45)

    apply_house_style(ax, grid=True)
    fig.tight_layout()

    return save_and_close(fig=fig, filename=filename, save=save_figures, close=close_figures)
