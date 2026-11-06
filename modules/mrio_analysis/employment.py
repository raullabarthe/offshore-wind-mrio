"""Employment and wage extension for the MRIO impact model."""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import re

import numpy as np
import pandas as pd
import pycountry

from modules.config import HELPERS_DIR
from .source_utils import _INDUSTRIAL_SOURCE_COLUMNS, _fte40_area_to_iso2

CLASSIFICATION_CROSSWALK_DIR = HELPERS_DIR / "classifications"


def load_fte40_results_cache(
    cache_path: str | Path,
    *,
    metadata: dict,
    expected_pairs: tuple[tuple[str, str], ...],
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Return a validated notebook FTE40 cache, or ``None`` when it is stale.

    The cache contains only final allocation results.  Pickle is intentional:
    it preserves the audit metadata held in ``DataFrame.attrs``.
    """
    try:
        payload = pd.read_pickle(cache_path)
    except Exception:  # Corrupt or incompatible local cache: recompute safely.
        return None

    if not isinstance(payload, dict) or payload.get("metadata") != metadata:
        return None
    fte_industry = payload.get("fte_industry")
    fte_audit = payload.get("fte_audit")
    required_fte = {"geo", "industry", "FTE_country_industry", "FTE_intensity"}
    if (
        not isinstance(fte_industry, pd.DataFrame)
        or not isinstance(fte_audit, pd.DataFrame)
        or not required_fte.issubset(fte_industry.columns)
        or not {"geo", "reconciliation_error"}.issubset(fte_audit.columns)
    ):
        return None

    actual_pairs = tuple(
        (str(geo), str(industry))
        for geo, industry in fte_industry[["geo", "industry"]].itertuples(index=False, name=None)
    )
    if len(actual_pairs) != len(expected_pairs) or set(actual_pairs) != set(expected_pairs):
        return None
    return fte_industry, fte_audit

ind_to_prod = pd.read_csv(HELPERS_DIR / "ind_to_prd.csv")
class_prd_ava = pd.read_csv(HELPERS_DIR / "class_prd_ava.csv")
class_ind_ava = pd.read_csv(HELPERS_DIR / "class_ind_ava.csv")
try:
    aggregate_coverage = pd.read_csv(
        CLASSIFICATION_CROSSWALK_DIR / "isic4_A14_A21_agg_sector.csv",
        comment="#",
    )
except FileNotFoundError:
    aggregate_coverage = pd.DataFrame(columns=["source_code", "covers_ind_ava_codes"])

if "coverage_profile" in aggregate_coverage.columns:
    aggregate_coverage["coverage_profile"] = (
        aggregate_coverage["coverage_profile"]
        .astype(str)
        .str.strip()
    )

ISIC4_L2_TO_NACE_REV2_A64_PATH = CLASSIFICATION_CROSSWALK_DIR / "isic4_l2_to_nace_rev2_a64.csv"
ISIC3_L2_TO_NACE_REV2_A64_PATH = CLASSIFICATION_CROSSWALK_DIR / "isic3_l2_to_nace_rev2_a64.csv"

ILO_GLOBAL_A14_CODES = (
    "A", "B", "C", "DE", "F", "G", "HJ", "I", "K", "LMN", "O", "P", "Q", "RSTU",
)
ILO_FTE40_BASIS = "JBF_FTE_FTE40"

def _is_industry_matrix(mrio_prd_set, ind_ind: bool | None) -> bool:
    if ind_ind is not None:
        return bool(ind_ind)
    return not any(str(c).startswith("CPA_") for c in mrio_prd_set)

def _canon_map(mrio_prd_set) -> dict[str, str]:
    canon = {}
    for c in mrio_prd_set:
        c = str(c)
        if "T" in c:
            canon[c.replace("T", "-")] = c
        if "_" in c:
            canon[c.replace("_", "-")] = c
        if "-" in c:
            canon[c.replace("-", "T")] = c
            canon[c.replace("-", "_")] = c
    return canon

def _add_cpa_prefix(x: str) -> str:
    if x is None:
        return x
    s = str(x).strip()
    if s == "TOTAL":
        return "CPA_TOTAL"
    if s.startswith("CPA_"):
        return s
    return f"CPA_{s}"

def _strip_cpa_prefix(x: str) -> str:
    if x is None:
        return x
    s = str(x).strip()
    if s == "CPA_TOTAL":
        return "TOTAL"
    return s[4:] if s.startswith("CPA_") else s

def _clean_industry_code(x: str) -> str:
    s = str(x).strip().upper()
    s = s.replace("CPA_", "")
    s = re.sub(r"^(EC2|ECO)_(ISIC3|ISIC4)_", "", s)
    s = re.sub(r"^ECO_AGGREGATE_\d+(?:_AGG)?_", "", s)
    s = re.sub(r"^ECO_SECTOR_\d+_", "", s)
    s = re.sub(r"^ECO_AGNAG_(?:\d+_)?", "", s)
    s = re.sub(r"^ECO_(AGGREGATE|SECTOR|AGNAG)_", "", s)
    return s.replace("-", "_")

def _industry_code_divisions(code: str) -> set[int]:
    s = _clean_industry_code(code)
    if s in {"TOTAL", "X", ""}:
        return set()

    section_ranges = {
        "A": range(1, 4),
        "B": range(5, 10),
        "C": range(10, 34),
        "D": range(35, 36),
        "E": range(36, 40),
        "F": range(41, 44),
        "G": range(45, 48),
        "H": range(49, 54),
        "I": range(55, 57),
        "J": range(58, 64),
        "K": range(64, 67),
        "L": range(68, 69),
        "M": range(69, 76),
        "N": range(77, 83),
        "O": range(84, 85),
        "P": range(85, 86),
        "Q": range(86, 89),
        "R": range(90, 94),
        "S": range(94, 97),
        "T": range(97, 99),
        "U": range(99, 100),
    }
    if s in section_ranges:
        return set(section_ranges[s])

    m = re.fullmatch(r"([A-Z]+)(\d{2})(?:[T_](?:[A-Z]+)?(\d{2}))?", s)
    if m:
        a = int(m.group(2))
        b = int(m.group(3)) if m.group(3) else a
        return set(range(a, b + 1))

    m = re.fullmatch(r"(\d{2})(?:[._]?\d*)?", s)
    if m:
        return {int(m.group(1))}

    return set()

def _target_by_nace2_division(mrio_prd_set, *, ind_ind: bool) -> dict[int, str]:
    plain_targets = [_strip_cpa_prefix(str(c)) for c in mrio_prd_set]
    candidates = []
    for target in plain_targets:
        divs = _industry_code_divisions(target)
        if divs:
            candidates.append((target, divs))

    out = {}
    for div in range(1, 100):
        hits = [(target, divs) for target, divs in candidates if div in divs]
        if not hits:
            continue
        hits.sort(key=lambda x: (len(x[1]), len(str(x[0])), str(x[0])))
        target = hits[0][0]
        out[div] = _add_cpa_prefix(target) if not ind_ind else target
    return out

def _read_l2_crosswalk(path: Path, source_col: str, target_col: str) -> dict[str, set[str]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str, comment="#").fillna("")
    if not {source_col, target_col}.issubset(df.columns):
        return {}
    out = {}
    for _, row in df.iterrows():
        src = _clean_industry_code(row[source_col])
        targets = {
            _clean_industry_code(c)
            for c in _split_coverage_codes(row[target_col])
        }
        targets = {c for c in targets if c}
        if src and targets:
            out.setdefault(src, set()).update(targets)
    return out

@lru_cache(None)
def _isic4_l2_to_nace_rev2_a64_map() -> dict[str, set[str]]:
    return _read_l2_crosswalk(
        ISIC4_L2_TO_NACE_REV2_A64_PATH,
        "isic4_l2",
        "nace_rev2_a64",
    )

@lru_cache(None)
def _isic3_l2_to_nace_rev2_a64_map() -> dict[str, set[str]]:
    return _read_l2_crosswalk(
        ISIC3_L2_TO_NACE_REV2_A64_PATH,
        "isic3_l2",
        "nace_rev2_a64",
    )

def _normalize_crosswalk_targets(targets: set[str], mrio_prd_set, *, ind_ind: bool) -> set[str]:
    plain_mrio = {_strip_cpa_prefix(str(c).strip()) for c in mrio_prd_set}
    div_to_target = _target_by_nace2_division(tuple(map(str, mrio_prd_set)), ind_ind=ind_ind)
    out = set()
    for target in targets:
        target = _clean_industry_code(target)
        if not target:
            continue
        if target in plain_mrio:
            out.add(_add_cpa_prefix(target) if not ind_ind else target)
            continue
        divs = _industry_code_divisions(target)
        mapped = {div_to_target[d] for d in divs if d in div_to_target}
        if mapped:
            out.update(mapped)
    return out


@lru_cache(None)
def _class_ind_ava_target_map() -> dict[str, set[str]]:
    """Return descendants of each NACE hierarchy node at FIGARO A64."""
    if class_ind_ava is None or class_ind_ava.empty:
        return {}
    codes = {
        _clean_industry_code(code): row
        for _, row in class_ind_ava.iterrows()
        for code in [row.get("ind_ava_code")]
        if pd.notna(code)
    }
    parent = {
        code: _clean_industry_code(row.get("parent_code"))
        for code, row in codes.items()
        if pd.notna(row.get("parent_code"))
    }
    children: dict[str, set[str]] = {}
    for child, ancestor in parent.items():
        children.setdefault(ancestor, set()).add(child)

    @lru_cache(None)
    def descendants(code: str) -> frozenset[str]:
        out = {code}
        for child in children.get(code, set()):
            out.update(descendants(child))
        return frozenset(out)

    result = {}
    for code in codes:
        result[code] = {
            target
            for target in descendants(code)
            if bool(codes[target].get("A*64", 0))
        }
    return result


def _existing_industry_targets(
    code: str,
    target_industries: set[str],
    *,
    ind_ind: bool,
) -> set[str]:
    """Resolve one source code using the existing AVA hierarchy first."""
    clean_code = _clean_industry_code(code)
    clean_targets = {_clean_industry_code(target) for target in target_industries}
    exact = {target for target in target_industries if _clean_industry_code(target) == clean_code}
    if exact:
        return exact

    hierarchy_targets = _class_ind_ava_target_map().get(clean_code, set())
    hierarchy_targets &= clean_targets
    if hierarchy_targets:
        return {
            target
            for target in target_industries
            if _clean_industry_code(target) in hierarchy_targets
        }

    return _normalize_crosswalk_targets(
        {clean_code},
        tuple(target_industries),
        ind_ind=ind_ind,
    )

def _identify_source_classification(
    row: pd.Series | None = None,
    *,
    source_db: str = "",
    dataflow: str = "",
    code: str = "",
    profile: str | None = None,
    industry_col: str | None = None,
) -> str:
    """Return the canonical source classification family from metadata."""
    explicit = row.get("class_system", pd.NA) if row is not None else pd.NA
    text = " ".join(
        str(value) for value in (explicit, profile, industry_col, source_db, dataflow, code)
        if pd.notna(value)
    ).upper().replace(" ", "")
    if "ISIC3" in text or "ISICREV.3" in text:
        return "ISIC3"
    if "ISIC4" in text or "ISICREV.4" in text:
        return "ISIC4"
    if "ECO_AGGREGATE" in text or "ECO_AGNAG" in text:
        return "ECO_AGGREGATE"
    if "ECO_SECTOR" in text:
        return "ECO_SECTOR"
    if "NAMA_10_A64_E" in text or "EUROSTAT" in text or "NACEREV" in text or "NACE_R2" in text:
        return "NACE_R2"
    return "UNKNOWN"


def _official_classification_coverage(
    code: str,
    class_system: str | None,
    mrio_prd_set,
    *,
    ind_ind: bool,
) -> set[str]:
    """Resolve an explicitly classified code through the official bridge."""
    system = _identify_source_classification(
        source_db=str(class_system or ""),
        code=code,
    )
    source = _clean_industry_code(code)
    if system not in {"ISIC3", "ISIC4"} or source in _aggregate_source_codes_raw() or source == "TOTAL":
        return set()

    crosswalk = (
        _isic3_l2_to_nace_rev2_a64_map()
        if system == "ISIC3"
        else _isic4_l2_to_nace_rev2_a64_map()
    )
    nace_targets = crosswalk.get(source, set())
    if nace_targets:
        return _normalize_crosswalk_targets(nace_targets, tuple(map(str, mrio_prd_set)), ind_ind=ind_ind)
    if system == "ISIC3":
        return set()

    # ISIC4 section-level rows such as A, B, C are handled directly from
    # their section/division coverage. No downloaded correspondence tables
    # are used here.
    div_to_target = _target_by_nace2_division(tuple(map(str, mrio_prd_set)), ind_ind=ind_ind)
    divs = _industry_code_divisions(source)
    return {div_to_target[d] for d in divs if d in div_to_target}



def _isic3_to_mrio_code(code: str) -> str | None:
    """Return the unambiguous FIGARO target for one ISIC Rev. 3 code.

    Retained for the legacy NACE-to-CPA helper.
    """
    targets = _ISIC3_TO_NACE_REV2_A64.get(_clean_industry_code(code), set())
    return next(iter(targets)) if len(targets) == 1 else None


def _split_coverage_string(x) -> set[str]:
    if pd.isna(x) or str(x).strip() == "":
        return set()
    return {p.strip() for p in re.split(r"[;,]", str(x)) if p.strip()}

def _split_coverage_codes(x) -> list[str]:
    if pd.isna(x):
        return []
    return [p.strip() for p in re.split(r"[;,]", str(x)) if p and p.strip()]


_ISIC3_TO_NACE_REV2_A64 = _isic3_l2_to_nace_rev2_a64_map()


def _a14_groups_for_code(code) -> set[str]:
    """Return the A14 group(s) containing a FIGARO industry code."""
    plain = _clean_industry_code(code)
    if not plain or plain in {"TOTAL", "X"}:
        return set()

    if (
        not aggregate_coverage.empty
        and "coverage_profile" in aggregate_coverage.columns
        and "source_code" in aggregate_coverage.columns
    ):
        rows = aggregate_coverage.loc[
        aggregate_coverage["coverage_profile"].astype(str).eq("ISIC4_A14")
        ]
        for _, row in rows.iterrows():
            source = str(row["source_code"]).strip()
            if plain == source or plain in _split_coverage_codes(row.get("covers_ind_ava_codes")):
                return {source}

    return set()

def _ind_to_prd_map() -> dict[str, str]:
    if ind_to_prod is None or ind_to_prod.empty:
        return {}
    return (
        ind_to_prod
        .loc[:, ["ind_ava_code", "prd_ava_code"]]
        .assign(
            ind_ava_code=lambda d: d["ind_ava_code"].astype(str).str.strip(),
            prd_ava_code=lambda d: d["prd_ava_code"].astype(str).str.strip(),
        )
        .set_index("ind_ava_code")["prd_ava_code"]
        .to_dict()
    )

def _aggregate_source_codes_raw() -> set[str]:
    if aggregate_coverage is None or aggregate_coverage.empty or "source_code" not in aggregate_coverage.columns:
        return set()
    return {
        str(c).strip()
        for c in aggregate_coverage["source_code"].dropna()
        if str(c).strip()
    }

def _normalize_ind_code_for_wage_space(
    code,
    *,
    ind_ind: bool,
    canon: dict,
    canon_cpa: dict | None = None,
    as_source: bool = False,
) -> str:
    raw = str(code).strip()
    if raw in {"TOTAL", "CPA_TOTAL"}:
        return "CPA_TOTAL" if ind_ind else "CPA_TOTAL"
    if ind_ind:
        # Industry-by-industry FIGARO uses IND_AVA codes without the CPA
        # prefix.  Aggregate coverage is defined in that same code space.
        plain = _strip_cpa_prefix(raw)
        return canon.get(plain, plain)

    if as_source and _strip_cpa_prefix(raw) in _aggregate_source_codes_raw():
        return _strip_cpa_prefix(raw)

    if raw.startswith("CPA_"):
        mapped = raw
    else:
        mapped = _ind_to_prd_map().get(raw)
        if mapped is None:
            mapped = f"CPA_{raw}" if f"CPA_{raw}" in canon.values() else raw
    return canon.get(mapped, mapped)

def _aggregate_profile_col() -> str | None:
    if aggregate_coverage is None or aggregate_coverage.empty:
        return None
    for col in ("coverage_profile", "profile", "coverage_group", "group"):
        if col in aggregate_coverage.columns:
            return col
    return None

def _aggregate_coverage_entries(mrio_prd_set, *, ind_ind: bool) -> list[dict]:
    if aggregate_coverage is None or aggregate_coverage.empty:
        return []

    canon = _canon_map(mrio_prd_set)
    canon["CPA_TOTAL"] = "CPA_TOTAL"
    canon_cpa = {_add_cpa_prefix(k): _add_cpa_prefix(v) for k, v in canon.items()} if ind_ind else None
    profile_col = _aggregate_profile_col()

    entries = []
    for _, row in aggregate_coverage.iterrows():
        src = row.get("source_code")
        if pd.isna(src):
            continue
        src_norm = _normalize_ind_code_for_wage_space(src, ind_ind=ind_ind, canon=canon, canon_cpa=canon_cpa, as_source=True)
        covers = {
            _normalize_ind_code_for_wage_space(c, ind_ind=ind_ind, canon=canon, canon_cpa=canon_cpa)
            for c in _split_coverage_codes(row.get("covers_ind_ava_codes"))
        }
        covers = {c for c in covers if c}
        profile = row.get(profile_col) if profile_col else None
        profile = str(profile).strip() if pd.notna(profile) else None
        entries.append({"source": src_norm, "covers": covers, "profile": profile})

    if profile_col:
        profile_sources = {}
        for entry in entries:
            if entry["profile"] is not None:
                profile_sources.setdefault(entry["profile"], set()).add(entry["source"])
        for entry in entries:
            entry["profile_sources"] = profile_sources.get(entry["profile"], {entry["source"]})
    else:
        for entry in entries:
            entry["profile_sources"] = {entry["source"]}

    return entries

def _aggregate_coverage_map(
    mrio_prd_set,
    *,
    ind_ind: bool,
    observed_codes: set[str] | None = None,
) -> dict[str, set[str]]:
    entries = _aggregate_coverage_entries(mrio_prd_set, ind_ind=ind_ind)
    canon = _canon_map(mrio_prd_set)
    canon["CPA_TOTAL"] = "CPA_TOTAL"
    canon_cpa = {_add_cpa_prefix(k): _add_cpa_prefix(v) for k, v in canon.items()} if ind_ind else None

    if observed_codes is not None:
        observed_raw = {str(c).strip() for c in observed_codes if pd.notna(c)}
        observed_uses_cpa = any(c.startswith("CPA_") for c in observed_raw)
        observed_codes = {
            _normalize_ind_code_for_wage_space(c, ind_ind=ind_ind, canon=canon, canon_cpa=canon_cpa, as_source=True)
            for c in observed_raw
        }
        observed_codes = {c for c in observed_codes if c}
        profiled = [e for e in entries if e.get("profile")]
        matching_profiles = {
            e["profile"]: e["profile_sources"]
            for e in profiled
            if observed_codes and observed_codes.issubset(e["profile_sources"])
        }
        if len(matching_profiles) == 1:
            best_profile = next(iter(matching_profiles))
            entries = [
                e for e in entries
                if not e.get("profile") or e.get("profile") == best_profile
            ]
        elif not matching_profiles:
            entries = [e for e in entries if not e.get("profile")]
        # Multiple containing profiles are genuinely ambiguous.  Keep all
        # observed mappings rather than silently choosing one nominal family.

    out: dict[str, set[str]] = {}
    for entry in entries:
        out.setdefault(entry["source"], set()).update(entry["covers"])

    if observed_codes is not None:
        out = {src: covers for src, covers in out.items() if src in observed_codes}

    if observed_codes is not None and ind_ind and not observed_uses_cpa:
        out = {
            _strip_cpa_prefix(src): {_strip_cpa_prefix(c) for c in covers}
            for src, covers in out.items()
        }
    return out

def _aggregate_source_priority(mrio_prd_set, *, ind_ind: bool) -> dict[str, int]:
    canon = _canon_map(mrio_prd_set)
    canon["CPA_TOTAL"] = "CPA_TOTAL"
    canon_cpa = {_add_cpa_prefix(k): _add_cpa_prefix(v) for k, v in canon.items()} if ind_ind else None

    out: dict[str, int] = {}
    if aggregate_coverage is not None and not aggregate_coverage.empty:
        for _, row in aggregate_coverage.iterrows():
            src = row.get("source_code")
            if pd.notna(src):
                src_norm = _normalize_ind_code_for_wage_space(src, ind_ind=ind_ind, canon=canon, canon_cpa=canon_cpa, as_source=True)
                out[src_norm] = min(out.get(src_norm, 99), 0)
                if ind_ind:
                    src_plain = _strip_cpa_prefix(src_norm)
                    out[src_plain] = min(out.get(src_plain, 99), 0)

    for ind_code in _ind_to_prd_map():
        src_norm = _normalize_ind_code_for_wage_space(ind_code, ind_ind=ind_ind, canon=canon, canon_cpa=canon_cpa)
        out[src_norm] = min(out.get(src_norm, 99), 1)
        if ind_ind:
            src_plain = _strip_cpa_prefix(src_norm)
            out[src_plain] = min(out.get(src_plain, 99), 1)
    return out

_SESSION=None





def _ensure_geo_col(df: pd.DataFrame) -> pd.DataFrame:
    if "geo" in df.columns:
        return df
    if "geo" in df.index.names:
        return df.reset_index()
    candidates = ["geo\t", "GEO", "geo_t", "country", "COUNTRY", "ref_area", "REF_AREA"]
    for c in candidates:
        if c in df.columns:
            out = df.copy()
            out["geo"] = out[c].astype(str).str.strip()
            # If likely ISO3, map to ISO2 where possible
            if out["geo"].str.len().median() == 3:
                def _iso3_to_iso2(x):
                    cc = pycountry.countries.get(alpha_3=str(x).upper())
                    return cc.alpha_2 if cc else x
                out["geo"] = out["geo"].map(_iso3_to_iso2)
            return out
    raise KeyError(f"Missing 'geo' column. Columns found: {list(df.columns)}")






def _first_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    exact = {str(column).strip().lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in exact:
            return exact[candidate.lower()]
    return None


def prepare_fte40_employment_inputs(
    eurostat_employment: pd.DataFrame | None,
    ilostat_employment: pd.DataFrame | None,
    *,
    target_industries: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select standardized employment sources before FTE allocation."""
    sources = []
    for source_label, frame in (
        ("Eurostat", eurostat_employment), ("ILOSTAT", ilostat_employment)
    ):
        if frame is None or frame.empty:
            sources.append((source_label, pd.DataFrame(columns=_INDUSTRIAL_SOURCE_COLUMNS)))
            continue
        missing = set(_INDUSTRIAL_SOURCE_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(
                f"{source_label} employment input is not standardized; missing {sorted(missing)}."
            )
        sources.append((source_label, frame[_INDUSTRIAL_SOURCE_COLUMNS].copy()))

    candidates = []
    for source_label, source_frame in sources:
        if source_frame.empty:
            continue
        group_cols = ["geo", "source_db", "dataflow_id", "year", "class_system"]
        for key, group in source_frame.groupby(group_cols, dropna=False, sort=False):
            metadata, _mapped = _effective_resolution_metadata(
                group,
                target_industries=target_industries,
                value_name="employment",
                source_name=source_label,
            )
            actual_source_label = (
                "OECD"
                if group["source_db"].astype(str).str.contains("OECD", case=False, na=False).any()
                else source_label
            )
            candidates.append({
                "source_label": actual_source_label,
                "frame": group.copy(),
                "geo": key[0],
                "source_db": key[1],
                "dataflow_id": key[2],
                "source_year": key[3],
                "class_system": key[4],
                "original_source_codes": ";".join(sorted(group["source_code"].astype(str).unique())),
                **metadata,
            })

    if not candidates:
        empty = pd.DataFrame(columns=_INDUSTRIAL_SOURCE_COLUMNS)
        return empty, pd.DataFrame()

    inventory = pd.DataFrame(
        [{
            "geo": candidate["geo"],
            "source": candidate["source_label"],
            "source_db": candidate["source_db"],
            "dataflow_id": candidate["dataflow_id"],
            "source_year": candidate["source_year"],
            "class_system": candidate["class_system"],
            "original_source_codes": candidate["original_source_codes"],
            "resolution_profile": candidate["resolution_profile"],
            "named_source_profile": candidate["named_source_profile"],
            "effective_resolution": candidate["effective_resolution"],
            "resolution_completeness": candidate["resolution_completeness"],
            "valid_source_observations": candidate["valid_source_observations"],
            "mapped_source_observations": candidate["mapped_source_observations"],
            "independent_blocks": candidate["independent_blocks"],
            "exact_A64": candidate["exact_A64"],
            "aggregate_blocks": candidate["aggregate_blocks"],
            "A64_covered": candidate["A64_covered"],
            "unmapped_codes": ";".join(candidate["unmapped_codes"]),
            "ambiguous_codes": ";".join(candidate["ambiguous_codes"]),
            "selected": False,
            "selection_reason": "not selected",
        } for candidate in candidates]
    )

    selected_frames = []
    for geo, group in inventory.groupby("geo", dropna=False, sort=False):
        candidate_indexes = group.index.tolist()
        best_index = max(
            candidate_indexes,
            key=lambda index: (
                inventory.at[index, "effective_resolution"],
                inventory.at[index, "exact_A64"],
                inventory.at[index, "independent_blocks"],
                inventory.at[index, "A64_covered"],
                -inventory.at[index, "aggregate_blocks"],
                inventory.at[index, "source"] == "Eurostat",
            ),
        )
        inventory.loc[best_index, "selected"] = True
        inventory.loc[best_index, "selection_reason"] = (
            "selected: highest effective industrial resolution; Eurostat priority on tie"
        )
        selected_frames.append(candidates[best_index]["frame"])

    selected = pd.concat(selected_frames, ignore_index=True)
    selected.attrs["source_resolution_inventory"] = inventory.copy()
    selected.attrs["source_availability_audit"] = inventory.assign(
        measure=np.where(
            inventory["source"].astype(str).str.upper().eq("OECD"),
            "OECD employment",
            "ILOSTAT employment",
        ),
        indicator=inventory["dataflow_id"],
        selected=inventory["selected"],
    )
    return selected[_INDUSTRIAL_SOURCE_COLUMNS], inventory


def _standardize_fte_source(
    frame: pd.DataFrame | None,
    *,
    target_industries: set[str],
    value_name: str,
    source_name: str,
) -> pd.DataFrame:
    """Convert an industrial source to a small, explicit long-form table."""
    if frame is None or frame.empty:
        return pd.DataFrame(
            columns=[
                "geo", "source_code", "targets", value_name,
                "source_is_exact", "source_db", "source_year", "dataflow_id", "key_used",
                "population_scope", "class_system", "source_coverage",
            ]
        )
    df = frame.copy()
    df.attrs = {}
    geo_col = _first_column(df, ("geo", "REF_AREA", "country", "originArea", "destination_region"))
    industry_col = _first_column(
        df,
        (
            "source_code",
            "nace_r2_raw",
            "nace_r2",
            "industry",
            "industry_code",
            "ECO",
            "EC2",
            "originIi",
            "finalIi",
            "prd_use",
        ),
    )
    if "nace_r2_raw" in df.columns and "nace_r2" in df.columns:
        df = df.copy()
        df["_fte_industry_code"] = df["nace_r2_raw"].where(
            df["nace_r2_raw"].notna(),
            df["nace_r2"],
        )
        industry_col = "_fte_industry_code"
    value_col = _first_column(df, (value_name, "OBS_VALUE", "obs_value", "value"))
    if geo_col is None or industry_col is None or value_col is None:
        raise ValueError(f"{source_name} lacks geo, industry, or value columns.")

    ind_ind = _is_industry_matrix(target_industries, None)
    target_set = set(target_industries)
    group_cols = [column for column in (
        geo_col, "class_system", "source_db", "dataflow_id", "year", "source_year",
    ) if column in df.columns]
    group_keys = (
        df[group_cols].astype("string").fillna("").agg("\x1f".join, axis=1)
        if group_cols else pd.Series("", index=df.index, dtype="string")
    )
    aggregate_maps = {}
    for key, group in df.groupby(group_keys, sort=False):
        observed_codes = {
            _clean_industry_code(code)
            for code in group[industry_col].dropna()
            if _clean_industry_code(code) not in {"", "TOTAL", "X"}
        }
        aggregate_maps[str(key)] = _aggregate_coverage_map(
            target_set,
            ind_ind=ind_ind,
            observed_codes=observed_codes,
        )
    aggregate_fallback = _aggregate_coverage_map(target_set, ind_ind=ind_ind)

    rows = []
    for index, row in df.iterrows():
        geo = _fte40_area_to_iso2(row[geo_col]) or str(row[geo_col]).strip().upper()
        if geo == "X01":
            geo = "ROW"
        raw_code = str(row[industry_col]).strip()
        code = _clean_industry_code(raw_code)
        if code in {"TOTAL", "X", ""}:
            continue
        class_system = str(row.get("class_system", "")).upper().replace(" ", "")
        aggregate_map = aggregate_maps.get(str(group_keys.at[index]), {})
        if class_system == "ISIC3":
            targets = _official_classification_coverage(
                code, "ISIC3", target_set, ind_ind=ind_ind,
            )
        else:
            aggregate_targets = aggregate_map.get(
                code, aggregate_fallback.get(code, ())
            )
            targets = set().union(*(
                _existing_industry_targets(target, target_set, ind_ind=ind_ind)
                or {target}
                for target in aggregate_targets
            )) if aggregate_targets else set()
            targets.update(
                _existing_industry_targets(
                    code, target_set, ind_ind=ind_ind,
                )
            )
        targets &= target_set
        if not targets:
            mapped = _ind_to_prd_map().get(code)
            if mapped is not None:
                targets = {
                    target for target in target_set
                    if _clean_industry_code(target) == _clean_industry_code(mapped)
                }
        if not targets:
            continue
        source_is_exact = len(targets) == 1 and any(
            code == _clean_industry_code(target)
            or code == _clean_industry_code(_strip_cpa_prefix(target))
            or (
                _industry_code_divisions(code)
                == _industry_code_divisions(_strip_cpa_prefix(target))
                and bool(_industry_code_divisions(code))
            )
            for target in targets
        )
        value = pd.to_numeric(pd.Series([row[value_col]]), errors="coerce").iloc[0]
        if pd.isna(value):
            continue
        source_db = row.get("source_db", row.get("dataset", source_name))
        source_year = row.get(
            "source_year",
            row.get("TIME_PERIOD", row.get("time", row.get("year", pd.NA))),
        )
        source_year = pd.to_numeric(pd.Series([source_year]), errors="coerce").iloc[0]
        rows.append(
            {
                "geo": geo,
                "source_code": code,
                "original_source_code": raw_code,
                "targets": tuple(sorted(targets)),
                value_name: float(value),
                "source_is_exact": source_is_exact,
                "source_db": source_db,
                "source_year": source_year,
                "dataflow_id": row.get("dataflow_id", pd.NA),
                "key_used": row.get("key_used", pd.NA),
                "population_scope": row.get(
                    "population_scope",
                    "national-accounts employment" if "eurostat" in str(source_db).lower() else pd.NA,
                ),
                "class_system": row.get("class_system", pd.NA),
                "original_class_system": row.get("class_system", pd.NA),
                "source_coverage": ";".join(sorted(targets)),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "geo", "source_code", "original_source_code", "targets", value_name,
            "source_is_exact", "source_db", "source_year", "dataflow_id", "key_used",
            "population_scope", "class_system", "original_class_system", "source_coverage",
        ],
    )


def _configured_resolution_targets() -> set[str]:
    """Return the configured FIGARO A64 industry targets for diagnostics.

    The notebook supplies the exact ``prd_use`` target set used by the active
    MRIO.  This fallback keeps the public loaders usable in isolation and is
    derived from the existing AVA hierarchy rather than from code shape.
    """
    if class_ind_ava is None or class_ind_ava.empty:
        return set()
    flag = pd.to_numeric(class_ind_ava.get("A*64"), errors="coerce").eq(1)
    return {
        _clean_industry_code(code)
        for code in class_ind_ava.loc[flag, "ind_ava_code"].dropna()
        if _clean_industry_code(code) not in {"TOTAL", "X", ""}
    }


def _effective_resolution_metadata(
    frame: pd.DataFrame | None,
    *,
    target_industries: set[str] | None,
    value_name: str,
    source_name: str,
) -> tuple[dict, pd.DataFrame]:
    """Measure mapped industrial resolution without counting overlaps twice."""
    target_set = {
        _clean_industry_code(code)
        for code in (target_industries or _configured_resolution_targets())
        if _clean_industry_code(code) not in {"TOTAL", "X", ""}
    }
    empty = {
        "resolution_profile": None,
        "named_source_profile": None,
        "effective_resolution": 0,
        "resolution_completeness": "empty",
        "valid_source_observations": 0,
        "mapped_source_observations": 0,
        "independent_blocks": 0,
        "exact_A64": 0,
        "aggregate_blocks": 0,
        "A64_covered": 0,
        "unmapped_codes": tuple(),
        "ambiguous_codes": tuple(),
        "n_unmapped": 0,
        "n_ambiguous": 0,
        "target_count": len(target_set),
    }
    if frame is None or frame.empty:
        return empty, pd.DataFrame()

    source = frame.copy()
    if "nace_r2" not in source.columns and "source_code" in source.columns:
        source["nace_r2"] = source["source_code"]
    elif "nace_r2" not in source.columns:
        # ILO raw extracts are normalized by the source module.  Keep this
        # fallback lazy to avoid a module-level employment/ILOSTAT cycle.
        from .ilostat import process_ilo_industry_measure

        source = process_ilo_industry_measure(
            source,
            source_db=source_name,
            measure_name=value_name,
        )
    code_series = source.get("nace_r2", pd.Series(pd.NA, index=source.index))
    value_series = source.get(
        value_name,
        source.get("OBS_VALUE", pd.Series(index=source.index, dtype=float)),
    )
    valid = source.loc[
        code_series.notna()
        & ~code_series.astype(str).str.upper().isin({"", "TOTAL", "X", "NAN"})
        & pd.to_numeric(value_series, errors="coerce").notna()
    ].copy()
    if "OBS_VALUE" not in valid.columns:
        valid["OBS_VALUE"] = pd.to_numeric(value_series.reindex(valid.index), errors="coerce")
    mapped = _standardize_fte_source(
        valid,
        target_industries=target_set,
        value_name="OBS_VALUE",
        source_name=source_name,
    )
    source_codes = set(valid["nace_r2"].astype(str).str.strip()) if not valid.empty else set()
    mapped_codes = set(mapped["source_code"].astype(str).str.strip()) if not mapped.empty else set()
    blocks = _c2_effective_blocks(mapped)
    covered = set().union(
        *(set(targets) for targets in mapped["targets"] if isinstance(targets, tuple))
    ) if not mapped.empty else set()

    ambiguous_codes = set()
    if not mapped.empty:
        target_variants = mapped.groupby("source_code", dropna=False)["targets"].agg(
            lambda values: {tuple(value) for value in values}
        )
        ambiguous_codes = {
            str(code) for code, variants in target_variants.items() if len(variants) > 1
        }

    if not mapped.empty:
        named_profile, completeness = _c2_profile_label(
            mapped=mapped,
            n_covered=len(covered),
            n_target=len(target_set),
        )
    else:
        named_profile = None
        completeness = "unmapped"

    metadata = {
        "resolution_profile": named_profile,
        "named_source_profile": named_profile,
        "effective_resolution": int(len(blocks)),
        "resolution_completeness": completeness,
        "valid_source_observations": int(len(valid)),
        "mapped_source_observations": int(len(mapped)),
        "independent_blocks": int(len(blocks)),
        "exact_A64": int(
            sum(
                len(block["effective_targets"]) == 1
                and block["source_is_exact"]
                for block in blocks
            )
        ),
        "aggregate_blocks": int(
            sum(
                not (
                    len(block["effective_targets"]) == 1
                    and block["source_is_exact"]
                )
                for block in blocks
            )
        ),
        "A64_covered": int(len(covered)),
        "unmapped_codes": tuple(sorted(source_codes - mapped_codes)),
        "ambiguous_codes": tuple(sorted(ambiguous_codes)),
        "n_unmapped": int(len(source_codes - mapped_codes)),
        "n_ambiguous": int(len(ambiguous_codes)),
        "target_count": len(target_set),
    }
    return metadata, mapped


def _c2_effective_blocks(mapped: pd.DataFrame) -> list[dict]:
    """Return the finest feasible non-overlapping partition of mapped observations."""
    if mapped is None or mapped.empty:
        return []
    rows = []
    for (code, targets, exact), group in mapped.groupby(
        ["source_code", "targets", "source_is_exact"], dropna=False
    ):
        target_set = set(targets) if isinstance(targets, tuple) else set()
        if target_set:
            rows.append({
                "source_code": str(code),
                "targets": target_set,
                "source_is_exact": bool(exact),
            })
    unique = {}
    for row in rows:
        key = frozenset(row["targets"])
        old = unique.get(key)
        if old is None or (row["source_is_exact"], row["source_code"]) > (
            old["source_is_exact"], old["source_code"]
        ):
            unique[key] = row

    candidates = [
        row for row in unique.values()
        if not any(other["targets"] < row["targets"] for other in unique.values())
    ]
    if not candidates:
        return []

    conflicts = {
        index: {
            other_index for other_index, other in enumerate(candidates)
            if other_index != index and row["targets"] & other["targets"]
        }
        for index, row in enumerate(candidates)
    }
    components = []
    unseen = set(range(len(candidates)))
    while unseen:
        stack = [unseen.pop()]
        component = set(stack)
        while stack:
            index = stack.pop()
            neighbours = conflicts[index] & unseen
            unseen -= neighbours
            component |= neighbours
            stack.extend(neighbours)
        components.append(sorted(component))

    selected_indexes = []
    for component in components:
        if len(component) == 1:
            selected_indexes.extend(component)
            continue
        local = {index: conflicts[index] & set(component) for index in component}

        def score(indexes):
            chosen = [candidates[index] for index in indexes]
            return (
                len(chosen),
                sum(bool(row["source_is_exact"]) for row in chosen),
                sum(len(row["targets"]) for row in chosen),
                tuple(sorted(row["source_code"] for row in chosen)),
            )

        @lru_cache(maxsize=None)
        def solve(available_key):
            available = set(available_key)
            if not available:
                return ()
            if all(not (local[index] & available) for index in available):
                return tuple(sorted(available))
            index = max(available, key=lambda item: len(local[item] & available))
            with_index = (index,) + solve(tuple(sorted(
                available - {index} - local[index]
            )))
            without_index = solve(tuple(sorted(available - {index})))
            return with_index if score(with_index) > score(without_index) else without_index

        selected_indexes.extend(solve(tuple(component)))

    selected = [
        {**candidates[index], "effective_targets": set(candidates[index]["targets"])}
        for index in selected_indexes
    ]
    return sorted(
        selected,
        key=lambda row: (len(row["effective_targets"]), row["source_code"]),
    )


def _aggregate_profile_source_sets() -> dict[str, set[str]]:
    if aggregate_coverage is None or aggregate_coverage.empty:
        return {}
    if not {"coverage_profile", "source_code"}.issubset(aggregate_coverage.columns):
        return {}
    out = {}
    for profile, group in aggregate_coverage.groupby("coverage_profile", dropna=True):
        sources = {
            _strip_cpa_prefix(str(code).strip())
            for code in group["source_code"]
            if pd.notna(code) and str(code).strip()
        }
        if sources:
            out[str(profile)] = sources
    return out


def _detect_aggregate_profile(codes: set[str]) -> str | None:
    observed = {_strip_cpa_prefix(str(code).strip()) for code in codes if pd.notna(code)}
    observed.discard("TOTAL")
    matches = [
        (profile, sources)
        for profile, sources in _aggregate_profile_source_sets().items()
        if observed and observed.issubset(sources)
    ]
    return matches[0][0] if len(matches) == 1 else None


def _c2_profile_label(
    *,
    mapped: pd.DataFrame,
    n_covered: int,
    n_target: int,
) -> tuple[str | None, str]:
    observed_codes = (
        set(mapped["source_code"].dropna().astype(str))
        if mapped is not None and "source_code" in mapped.columns
        else set()
    )
    class_systems = set(
        mapped.get("class_system", pd.Series(dtype=object)).dropna().astype(str)
    ) if mapped is not None and not mapped.empty else set()
    named_profile = (
        _detect_aggregate_profile(observed_codes)
        if "ISIC3" not in class_systems else None
    )
    if named_profile is None and mapped is not None and not mapped.empty:
        isic4_detailed_codes = {
            _clean_industry_code(code)
            for code in observed_codes
            if _clean_industry_code(code) in _isic4_l2_to_nace_rev2_a64_map()
        }
        isic3_detailed_codes = {
            _clean_industry_code(code)
            for code in observed_codes
            if _clean_industry_code(code) in _isic3_l2_to_nace_rev2_a64_map()
        }
        if class_systems == {"ISIC4"} and isic4_detailed_codes:
            named_profile = "ISIC4_DETAILED"
        elif class_systems == {"ISIC3"} and isic3_detailed_codes:
            named_profile = "ISIC3_DETAILED"
    completeness = "complete" if n_covered >= n_target else "partial"
    return named_profile, completeness


def _source_weights(
    source: pd.DataFrame,
    *,
    target_industries: list[str],
    value_name: str,
    fallback_weights: dict[str, dict[str, float]] | None = None,
    conflict_log: list[dict] | None = None,
    strict_parent_conflicts: bool = False,
) -> dict[str, dict[str, float]]:
    """Create non-overlapping industry shares from one additive source."""
    result: dict[str, dict[str, float]] = {}
    if source.empty:
        return result

    for geo, group in source.groupby("geo", sort=False):
        rows = []
        for row in group.itertuples(index=False):
            targets = [target for target in row.targets if target in target_industries]
            if not targets:
                continue
            rows.append(
                (
                    len(targets),
                    tuple(sorted(targets)),
                    float(getattr(row, value_name)),
                    bool(getattr(row, "source_is_exact", False)),
                )
            )

        selected: dict[str, float] = {}
        occupied: set[str] = set()
        # Narrow observations take precedence over parent observations.
        for width, targets, value, source_is_exact in sorted(
            rows, key=lambda item: (item[0], not item[3], item[1])
        ):
            if width != 1:
                continue
            target = targets[0]
            if target not in selected or source_is_exact:
                selected[target] = value
            occupied.add(target)

        for width, targets, value, _source_is_exact in sorted(rows, key=lambda item: item[0]):
            if width <= 1:
                continue
            missing_targets = [target for target in targets if target not in occupied]
            if not missing_targets:
                continue
            # Parent observations include any children already observed.  Only
            # the unresolved residual is allocated to the remaining children.
            residual = value - sum(selected.get(target, 0.0) for target in targets)
            if residual < -1e-9:
                conflict = {
                    "geo": geo,
                    "source": value_name,
                    "parent_targets": ";".join(targets),
                    "parent_value": value,
                    "observed_children": sum(selected.get(target, 0.0) for target in targets),
                    "residual": residual,
                    "rule": "parent excluded; detailed observations retained",
                }
                if conflict_log is not None:
                    conflict_log.append(conflict)
                if strict_parent_conflicts:
                    raise ValueError(
                        f"Parent {targets} is smaller than its observed children for {geo}."
                    )
                continue
            allocation = {
                target: (fallback_weights or {}).get(geo, {}).get(target, 0.0)
                for target in missing_targets
            }
            if sum(allocation.values()) <= 0:
                if conflict_log is not None:
                    conflict_log.append(
                        {
                            "geo": geo,
                            "source": value_name,
                            "parent_targets": ";".join(targets),
                            "parent_value": value,
                            "observed_children": sum(
                                selected.get(target, 0.0) for target in targets
                            ),
                            "residual": value - sum(
                                selected.get(target, 0.0) for target in targets
                            ),
                            "rule": "parent residual unresolved: no positive MRIO output",
                        }
                    )
                continue
            denom = sum(allocation.values())
            for target in missing_targets:
                selected[target] = selected.get(target, 0.0) + residual * allocation[target] / denom
                occupied.add(target)

        selected = {target: max(float(value), 0.0) for target, value in selected.items()}
        total = sum(selected.values())
        # A partial source has no information on its mass relative to the
        # unobserved industries.  Do not silently complete it with GVA and
        # then return it as a national labour-share vector.  Parent-level
        # constraints are handled explicitly by _resolve_fte40_country(),
        # where a compatible national total can establish their mass.
        if total > 0 and occupied == set(target_industries):
            result[geo] = {
                target: selected.get(target, 0.0) / total
                for target in target_industries
            }
    return result


def _output_weights_for_geo(
    output_source: pd.DataFrame | None,
    *,
    geo: str,
    target_industries: set[str],
) -> dict[str, float]:
    """Return positive exact A64 MRIO output values for one geography."""
    if output_source is None or output_source.empty:
        return {}
    group = output_source.loc[output_source["geo"].eq(geo)]
    weights: dict[str, float] = {}
    for row in group.itertuples(index=False):
        targets = tuple(
            target for target in getattr(row, "targets", ())
            if target in target_industries
        )
        value = pd.to_numeric(
            pd.Series([getattr(row, "output", np.nan)]), errors="coerce"
        ).iloc[0]
        # Aggregate output observations cannot be copied to every child.
        if len(targets) != 1 or pd.isna(value) or not np.isfinite(float(value)):
            continue
        weights[targets[0]] = weights.get(targets[0], 0.0) + max(float(value), 0.0)
    return {target: value for target, value in weights.items() if value > 0}


def _mean_hours_employment_product(
    employment: pd.DataFrame | None,
    hours_mean: pd.DataFrame | None,
    *,
    target_industries: set[str],
    allow_year_mismatch: bool = False,
) -> pd.DataFrame:
    """Build relative ``employment × mean-hours`` allocation weights.

    Mean hours are weekly observations.  Their product with employment is
    used only as a relative industry weight; it is never interpreted as an
    annual national-hours total or mixed with the Eurostat annual-hours
    denominator.
    """
    if employment is None or hours_mean is None or employment.empty or hours_mean.empty:
        return pd.DataFrame(
            columns=[
                "geo", "source_code", "targets", "value",
                "source_is_exact", "source_db", "source_year", "dataflow_id", "key_used",
                "population_scope",
            ]
        )
    emp = _standardize_fte_source(
        employment,
        target_industries=target_industries,
        value_name="employment",
        source_name="employment-by-industry",
    )
    hrs = _standardize_fte_source(
        hours_mean,
        target_industries=target_industries,
        value_name="hours",
        source_name="mean-hours-by-industry",
    )
    if emp.empty or hrs.empty:
        return pd.DataFrame(
            columns=[
                "geo", "source_code", "targets", "value",
                "source_is_exact", "source_db", "source_year", "dataflow_id", "key_used",
                "population_scope",
            ]
        )

    # Means are never summed. Conflicting values for the same area/source
    # code are rejected because they indicate incompatible sex/age categories.
    hrs_check = hrs.groupby(["geo", "source_code"])["hours"].nunique()
    if (hrs_check > 1).any():
        raise ValueError("Mean-hours input contains conflicting observations for one area/industry.")
    hrs = hrs.drop_duplicates(["geo", "source_code"])
    # Employment and hours often use different, but already mapped, industry
    # code systems.  Pair each employment block with the narrowest compatible
    # hours block that contains all of its FIGARO targets.  This preserves the
    # employment resolution while applying an observed parent mean to each
    # constituent child.  It also leaves overlapping parent rows available for
    # _resolve_fte_source_cells(), which subtracts known child labour volume.
    def _text(value) -> str:
        return "" if pd.isna(value) else str(value).strip().upper()

    def _same_year(emp_row, hours_row) -> bool:
        emp_year = pd.to_numeric(pd.Series([getattr(emp_row, "source_year", pd.NA)]), errors="coerce").iloc[0]
        hours_year = pd.to_numeric(pd.Series([getattr(hours_row, "source_year", pd.NA)]), errors="coerce").iloc[0]
        if allow_year_mismatch:
            known_years = [int(year) for year in (emp_year, hours_year) if pd.notna(year)]
            return not known_years or all(2021 <= year <= 2024 for year in known_years)
        return bool(pd.isna(emp_year) or pd.isna(hours_year) or emp_year == hours_year)

    def _years_differ(emp_row, hours_row) -> bool:
        emp_year = pd.to_numeric(pd.Series([getattr(emp_row, "source_year", pd.NA)]), errors="coerce").iloc[0]
        hours_year = pd.to_numeric(pd.Series([getattr(hours_row, "source_year", pd.NA)]), errors="coerce").iloc[0]
        return bool(pd.notna(emp_year) and pd.notna(hours_year) and emp_year != hours_year)

    def _population_compatible(emp_row, hours_row) -> bool:
        emp_flow = _text(getattr(emp_row, "dataflow_id", pd.NA))
        hours_flow = _text(getattr(hours_row, "dataflow_id", pd.NA))
        if not emp_flow and not hours_flow:
            return True
        # Both EMP_TEMP and modelled EMP_2EMP are counts of employed persons;
        # HOW_TEMP is mean hours actually worked per employed person.  The
        # modelled/observed distinction is retained in provenance, but is not
        # an employee-versus-employed population mismatch.
        return (
            ("EMP_TEMP" in emp_flow or "EMP_2EMP" in emp_flow or "OECD:" in emp_flow)
            and "HOW_TEMP" in hours_flow
        )

    pair_rows = []
    incompatible_rows = []
    for emp_row in emp.itertuples(index=False):
        emp_targets = tuple(getattr(emp_row, "targets", ()))
        if not emp_targets:
            continue
        hours_candidates = []
        for hours_row in hrs.loc[hrs["geo"].eq(getattr(emp_row, "geo"))].itertuples(index=False):
            hours_targets = tuple(getattr(hours_row, "targets", ()))
            if not set(emp_targets).issubset(hours_targets):
                continue
            if not _same_year(emp_row, hours_row) or not _population_compatible(emp_row, hours_row):
                incompatible_rows.append(
                    {
                        "geo": getattr(emp_row, "geo"),
                        "employment_source_code": getattr(emp_row, "original_source_code", getattr(emp_row, "source_code", pd.NA)),
                        "hours_source_code": getattr(hours_row, "original_source_code", getattr(hours_row, "source_code", pd.NA)),
                        "employment_classification": getattr(emp_row, "class_system", pd.NA),
                        "hours_classification": getattr(hours_row, "class_system", pd.NA),
                        "employment_year": getattr(emp_row, "source_year", pd.NA),
                        "hours_year": getattr(hours_row, "source_year", pd.NA),
                        "reason": "year or employed-person population mismatch",
                    }
                )
                continue
            hours_candidates.append(hours_row)
        if not hours_candidates:
            continue
        hours_candidates.sort(
            key=lambda row: (
                len(getattr(row, "targets", ())),
                not bool(getattr(row, "source_is_exact", False)),
                str(getattr(row, "source_code", "")),
            )
        )
        hours_row = hours_candidates[0]
        employment_value = float(getattr(emp_row, "employment"))
        hours_value = float(getattr(hours_row, "hours"))
        pair_rows.append(
            {
                "geo": getattr(emp_row, "geo"),
                "source_code": getattr(emp_row, "source_code"),
                "original_source_code": getattr(emp_row, "original_source_code", getattr(emp_row, "source_code", pd.NA)),
                "hours_source_code": getattr(hours_row, "original_source_code", getattr(hours_row, "source_code", pd.NA)),
                "targets": emp_targets,
                "value": employment_value * hours_value,
                "employment_observation": employment_value,
                "mean_weekly_hours": hours_value,
                "source_is_exact": bool(getattr(emp_row, "source_is_exact", False)),
                "source_db": getattr(emp_row, "source_db", pd.NA),
                "source_year": getattr(emp_row, "source_year", pd.NA),
                "dataflow_id": getattr(emp_row, "dataflow_id", pd.NA),
                "key_used": getattr(emp_row, "key_used", pd.NA),
                "population_scope": getattr(emp_row, "population_scope", pd.NA),
                "class_system": getattr(emp_row, "class_system", pd.NA),
                "hours_dataflow_id": getattr(hours_row, "dataflow_id", pd.NA),
                "hours_source_year": getattr(hours_row, "source_year", pd.NA),
                "hours_population_scope": getattr(hours_row, "population_scope", pd.NA),
                "compatibility_status": (
                    "compatible population and mapped FIGARO coverage; source years differ "
                    "and are used only for relative industrial weights"
                    if allow_year_mismatch and _years_differ(emp_row, hours_row)
                    else "compatible population, year, and mapped FIGARO coverage"
                ),
            }
        )

    # Several non-overlapping source codes can map to one FIGARO block. Sum
    # their employment-times-hours volumes only after each source row has
    # received the closest compatible hours observation.
    grouped = {}
    for row in pair_rows:
        key = (row["geo"], row["targets"])
        grouped.setdefault(key, []).append(row)
    rows = []
    for (geo, targets), group in grouped.items():
        employment_total = sum(row["employment_observation"] for row in group)
        first = group[0]
        rows.append(
            {
                "geo": geo,
                "source_code": ";".join(sorted({str(row["source_code"]) for row in group})),
                "original_source_code": ";".join(sorted({str(row["original_source_code"]) for row in group})),
                "hours_source_code": ";".join(sorted({str(row["hours_source_code"]) for row in group})),
                "targets": targets,
                "value": float(sum(row["value"] for row in group)),
                "employment_observation": float(employment_total),
                "mean_weekly_hours": float(
                    sum(row["value"] for row in group) / employment_total
                ) if employment_total > 0 else pd.NA,
                "source_is_exact": bool(all(row["source_is_exact"] for row in group)),
                "source_db": first["source_db"],
                "source_year": first["source_year"],
                "dataflow_id": first["dataflow_id"],
                "key_used": first["key_used"],
                "population_scope": first["population_scope"],
                "class_system": first["class_system"],
                "hours_dataflow_id": first["hours_dataflow_id"],
                "hours_source_year": first["hours_source_year"],
                "hours_population_scope": first["hours_population_scope"],
                "compatibility_status": first["compatibility_status"],
            }
        )
    result = pd.DataFrame(rows)
    result.attrs["compatible_pair_count"] = len(pair_rows)
    result.attrs["compatible_A64_covered"] = int(
        len(set().union(*result["targets"].map(set)))
    ) if not result.empty else 0
    if incompatible_rows:
        result.attrs["incompatible_pairs"] = pd.DataFrame(incompatible_rows)
    return result


def _standardize_output_frame(output, *, target_industries: set[str]) -> pd.DataFrame:
    if output is None:
        return pd.DataFrame(columns=["geo", "source_code", "targets", "output"])
    if isinstance(output, pd.Series):
        output = output.rename("output").reset_index()
    return _standardize_fte_source(
        output,
        target_industries=target_industries,
        value_name="output",
        source_name="FIGARO output",
    )


def _extract_fte_national_totals(
    frame: pd.DataFrame | None,
    *,
    value_name: str,
) -> tuple[dict[str, float], dict[str, dict], list[dict]]:
    """Extract national TOTAL observations without treating them as industries."""
    if frame is None or frame.empty:
        return {}, {}, []
    geo_col = _first_column(frame, ("geo", "REF_AREA", "country", "originArea", "destination_region"))
    industry_col = _first_column(
        frame,
        ("source_code", "nace_r2_raw", "nace_r2", "industry", "industry_code", "ECO", "EC2", "prd_use"),
    )
    value_col = _first_column(frame, (value_name, "OBS_VALUE", "obs_value", "value"))
    if geo_col is None or industry_col is None or value_col is None:
        return {}, {}, []

    values: dict[str, float] = {}
    metadata: dict[str, dict] = {}
    conflicts: list[dict] = []
    for _, row in frame.iterrows():
        raw_industry = row[industry_col]
        code = _clean_industry_code(raw_industry)
        if code != "TOTAL":
            continue
        geo = _fte40_area_to_iso2(row[geo_col]) or str(row[geo_col]).strip().upper()
        value = pd.to_numeric(pd.Series([row[value_col]]), errors="coerce").iloc[0]
        if pd.isna(value) or not np.isfinite(float(value)) or float(value) <= 0:
            continue
        value = float(value)
        if geo in values and not np.isclose(values[geo], value, rtol=0.0, atol=1e-9):
            conflicts.append(
                {
                    "geo": geo,
                    "source": value_name,
                    "rule": "conflicting national TOTAL observations",
                    "existing_total": values[geo],
                    "new_total": value,
                }
            )
            continue
        values[geo] = value
        metadata[geo] = {
            "source_db": row.get("source_db", "Eurostat"),
            "source_year": pd.to_numeric(
                pd.Series([row.get("year", row.get("source_year", pd.NA))])
                , errors="coerce",
            ).iloc[0],
            "dataflow_id": row.get("dataflow_id", pd.NA),
            "key_used": row.get("key_used", pd.NA),
        }
    return values, metadata, conflicts


def _resolve_fte_source_cells(
    source: pd.DataFrame,
    *,
    geo: str,
    target_industries: list[str],
    unresolved: set[str],
    value_name: str,
    direct_method: str,
    parent_method: str,
    fallback_weights: dict[str, float],
    fallback_weight_source: str | dict[str, str],
    fallback_weights_by_block: dict[tuple[str, tuple[str, ...]], dict[str, float]] | None = None,
    fallback_sources_by_block: dict[tuple[str, tuple[str, ...]], dict[str, str]] | None = None,
    conflict_log: list[dict],
    strict_parent_conflicts: bool,
) -> dict[str, dict]:
    """Resolve one source without adding parent and child observations.

    Direct observations are retained first.  A parent observation contributes
    only its residual after direct children have been removed, and that
    residual is allocated only to children still unresolved by higher-priority
    sources.  The returned values are allocation weights, not additional
    national totals.
    """
    if source.empty:
        return {}
    group = source.loc[source["geo"].eq(geo)].copy()
    if group.empty:
        return {}

    rows = []
    for row in group.itertuples(index=False):
        targets = tuple(
            target for target in getattr(row, "targets")
            if target in target_industries
        )
        value = float(getattr(row, value_name))
        if not targets or not np.isfinite(value) or value < 0:
            continue
        rows.append((len(targets), targets, value, bool(getattr(row, "source_is_exact", False)), row))

    # Several detailed source codes can be additive members of one broader
    # FIGARO block (for example, ISIC4 C10+C11+C12 -> C10T12).  Preserve an
    # exact one-code observation when it exists; otherwise sum the distinct
    # non-exact one-target observations before resolving parent residuals.
    direct_rows = []
    for target in target_industries:
        candidates = [
            item for item in rows
            if item[0] == 1 and item[1][0] == target
        ]
        if not candidates:
            continue
        exact_candidates = [item for item in candidates if item[3]]
        chosen = exact_candidates[0] if exact_candidates else candidates[0]
        if exact_candidates:
            direct_rows.append(chosen)
            continue
        _, target_tuple, _value, _source_is_exact, first_row = chosen
        direct_rows.append(
            (
                1,
                target_tuple,
                sum(item[2] for item in candidates),
                False,
                first_row,
                ";".join(sorted({str(getattr(item[4], "source_code", "")) for item in candidates})),
                ";".join(sorted({str(getattr(item[4], "original_source_code", getattr(item[4], "source_code", ""))) for item in candidates})),
            )
        )

    selected: dict[str, dict] = {}
    for direct_row in direct_rows:
        width, targets, value, source_is_exact, row = direct_row[:5]
        target = targets[0]
        current = selected.get(target)
        if current is None:
            selected[target] = {
                "value": value,
                "method": direct_method,
                "row": row,
                "source_is_exact": source_is_exact,
                "original_source_observation": value,
                "parent_source_observation": pd.NA,
                "parent_residual_observation": pd.NA,
                "within_parent_weight": 1.0,
                "source_codes": direct_row[5] if len(direct_row) > 5 else str(getattr(row, "source_code", "")),
                "original_source_codes": direct_row[6] if len(direct_row) > 6 else str(getattr(row, "original_source_code", getattr(row, "source_code", ""))),
                "allocation_raw_numerator": value,
                "allocation_raw_denominator": pd.NA,
                "allocation_raw_unit": {
                    "hours": "thousand hours/year",
                    "employment": "thousand employed persons",
                    "value": "thousand person-hours/week",
                    "gva": "million EUR",
                }.get(value_name, "source-specific"),
            }
        elif source_is_exact and not current["source_is_exact"]:
            selected[target] = {
                "value": value,
                "method": direct_method,
                "row": row,
                "source_is_exact": source_is_exact,
                "original_source_observation": value,
                "parent_source_observation": pd.NA,
                "parent_residual_observation": pd.NA,
                "within_parent_weight": 1.0,
                "source_codes": direct_row[5] if len(direct_row) > 5 else str(getattr(row, "source_code", "")),
                "original_source_codes": direct_row[6] if len(direct_row) > 6 else str(getattr(row, "original_source_code", getattr(row, "source_code", ""))),
                "allocation_raw_numerator": value,
                "allocation_raw_denominator": pd.NA,
                "allocation_raw_unit": {
                    "hours": "thousand hours/year",
                    "employment": "thousand employed persons",
                    "value": "thousand person-hours/week",
                    "gva": "million EUR",
                }.get(value_name, "source-specific"),
            }

    for width, targets, value, _source_is_exact, row in sorted(rows, key=lambda item: (item[0], item[1])):
        if width <= 1:
            continue
        missing = [
            target for target in targets
            if target in unresolved and target not in selected
        ]
        if not missing:
            continue
        observed_children = sum(
            selected.get(target, {}).get("value", 0.0)
            for target in targets
        )
        residual = value - observed_children
        if residual < -1e-9:
            conflict = {
                "geo": geo,
                "source": value_name,
                "parent_targets": ";".join(targets),
                "parent_value": value,
                "observed_children": observed_children,
                "residual": residual,
                "rule": "parent excluded; detailed observations retained",
            }
            conflict_log.append(conflict)
            if strict_parent_conflicts:
                raise ValueError(
                    f"Parent {targets} is smaller than its observed children for {geo}."
                )
            continue
        if residual <= 0:
            continue
        block_key = (
            str(getattr(row, "source_code", "")),
            tuple(sorted(targets)),
        )
        block_weights = (fallback_weights_by_block or {}).get(
            block_key, fallback_weights
        )
        block_sources = (fallback_sources_by_block or {}).get(block_key)
        allocation = {
            target: float(block_weights.get(target, 0.0))
            for target in missing
        }
        if sum(allocation.values()) <= 0:
            conflict_log.append(
                {
                    "geo": geo,
                    "source": value_name,
                    "parent_targets": ";".join(targets),
                    "parent_value": value,
                    "observed_children": observed_children,
                    "residual": residual,
                    "rule": "parent residual unresolved: no positive MRIO output",
                }
            )
            continue
        denominator = sum(allocation.values())
        output_targets = tuple(sorted(
            target
            for target in missing
            if (
                block_sources.get(target, "MRIO gross output")
                if block_sources is not None
                else (
                    fallback_weight_source.get(target, "MRIO gross output")
                    if isinstance(fallback_weight_source, dict)
                    else fallback_weight_source
                )
            ) == "MRIO gross output"
        ))
        for target in missing:
            allocation_source = (
                block_sources.get(target, "MRIO gross output")
                if block_sources is not None
                else (
                    fallback_weight_source.get(target, "MRIO gross output")
                    if isinstance(fallback_weight_source, dict)
                    else fallback_weight_source
                )
            )
            selected[target] = {
                "value": residual * allocation[target] / denominator,
                "method": parent_method,
                "row": row,
                "source_is_exact": False,
                "original_source_observation": value,
                "parent_source_observation": value,
                "parent_residual_observation": residual,
                "within_parent_weight": allocation[target] / denominator,
                "gva_allocation_block": pd.NA,
                "output_allocation_block": output_targets if target in output_targets else pd.NA,
                "output_allocation_share": (
                    allocation[target] / denominator
                    if target in output_targets else pd.NA
                ),
                "allocation_weight_source": allocation_source,
                "allocation_raw_numerator": allocation[target],
                "allocation_raw_denominator": denominator,
                "allocation_raw_unit": "million EUR" if target in output_targets else "source-specific relative weight",
            }

    resolved = {}
    for target, item in selected.items():
        # Retain zero-weight children created while splitting a valid parent
        # observation.  They remain part of that parent's coverage rather
        # than being misclassified as an unobserved GVA-only industry.
        if target not in unresolved or item["value"] < 0:
            continue
        row = item["row"]
        resolved[target] = {
            "value": float(item["value"]),
            "method": item["method"],
            "source_code": item.get("source_codes", str(getattr(row, "source_code", ""))),
            "original_source_code": item.get(
                "original_source_codes",
                getattr(row, "original_source_code", getattr(row, "source_code", pd.NA)),
            ),
            "source_is_exact": bool(getattr(row, "source_is_exact", False)),
            "source_db": getattr(row, "source_db", pd.NA),
            "source_year": getattr(row, "source_year", pd.NA),
            "dataflow_id": getattr(row, "dataflow_id", pd.NA),
            "key_used": getattr(row, "key_used", pd.NA),
            "population_scope": getattr(row, "population_scope", pd.NA),
            "source_coverage": ";".join(getattr(row, "targets", ())),
            "allocation_weight_source": item.get(
                "allocation_weight_source",
                (
                    "Direct observation"
                    if item["method"] == direct_method
                    else (
                        fallback_weight_source.get(target, "MRIO gross output")
                        if isinstance(fallback_weight_source, dict)
                        else fallback_weight_source
                    )
                ),
            ),
            "original_source_observation": item["original_source_observation"],
            "parent_source_observation": item["parent_source_observation"],
            "parent_residual_observation": item["parent_residual_observation"],
            "within_parent_weight": item["within_parent_weight"],
            "original_class_system": getattr(row, "original_class_system", getattr(row, "class_system", pd.NA)),
            "hours_source_code": getattr(row, "hours_source_code", pd.NA),
            "mean_weekly_hours": getattr(row, "mean_weekly_hours", pd.NA),
            "employment_observation": getattr(row, "employment_observation", pd.NA),
            "hours_dataflow_id": getattr(row, "hours_dataflow_id", pd.NA),
            "hours_source_year": getattr(row, "hours_source_year", pd.NA),
            "hours_population_scope": getattr(row, "hours_population_scope", pd.NA),
            "compatibility_status": getattr(row, "compatibility_status", pd.NA),
            "gva_allocation_block": item.get("gva_allocation_block", pd.NA),
            "output_allocation_block": item.get("output_allocation_block", pd.NA),
            "output_allocation_share": item.get("output_allocation_share", pd.NA),
            "allocation_raw_numerator": item.get("allocation_raw_numerator", pd.NA),
            "allocation_raw_denominator": item.get("allocation_raw_denominator", pd.NA),
            "allocation_raw_unit": item.get("allocation_raw_unit", pd.NA),
        }
    return resolved


def _fte_constraint_label(method: str, source_db) -> str:
    text = str(source_db).lower()
    if method in {"EU_H_DIRECT", "EU_H_PARENT"}:
        if "eurostat" in text:
            return "Eurostat annual hours"
        if "ilo" in text:
            return "ILOSTAT annual hours"
        return "annual hours"
    if method == "EMP_X_HOURS":
        return "employment x mean hours"
    if method == "EMP_NATIONAL_RESIDUAL_OUTPUT":
        return "ILOSTAT national employment total"
    if method in {"EMP_SHARE", "EMP_PARENT_OUTPUT"}:
        if "eurostat" in text:
            return "Eurostat employment"
        if "ilo" in text:
            return "ILOSTAT employment"
        return "employment"
    if method in {"NATIONAL_RESIDUAL_OUTPUT", "ZERO_OUTPUT_UNRESOLVED"}:
        return "FTE40 national anchor"
    return "MRIO gross output"


def _build_parent_labor_fallbacks(
    hours_source: pd.DataFrame,
    *,
    emp_hours_source: pd.DataFrame,
    employment_source: pd.DataFrame,
    base_weights: dict[str, float],
    geo: str,
) -> tuple[
    dict[tuple[str, tuple[str, ...]], dict[str, float]],
    dict[tuple[str, tuple[str, ...]], dict[str, str]],
    list[dict],
]:
    """Build coverage-aware weights for observed annual-hours parent blocks.

    A child employment partition is accepted only when all children are
    directly observed, or when an exactly matching observed parent identifies
    the residual.  With multiple unidentified children, the observed parent
    residual is retained as an aggregate and only those children use FIGARO
    output weights.
    """
    weight_maps: dict[tuple[str, tuple[str, ...]], dict[str, float]] = {}
    source_maps: dict[tuple[str, tuple[str, ...]], dict[str, str]] = {}
    audit_rows: list[dict] = []
    if hours_source is None or hours_source.empty:
        return weight_maps, source_maps, audit_rows

    hours_geo = hours_source.loc[hours_source["geo"].eq(geo)]
    emp_hours_geo = (
        emp_hours_source.loc[emp_hours_source["geo"].eq(geo)]
        if emp_hours_source is not None and not emp_hours_source.empty
        else pd.DataFrame()
    )
    employment_geo = (
        employment_source.loc[employment_source["geo"].eq(geo)]
        if employment_source is not None and not employment_source.empty
        else pd.DataFrame()
    )

    def _rows(frame: pd.DataFrame, value_name: str, block: tuple[str, ...]):
        direct: dict[str, list[tuple[float, bool]]] = {target: [] for target in block}
        parents = []
        block_set = set(block)
        if frame is None or frame.empty:
            return direct, parents
        for row in frame.itertuples(index=False):
            targets = tuple(
                sorted({target for target in getattr(row, "targets", ()) if target in block_set})
            )
            if not targets:
                continue
            value = pd.to_numeric(
                pd.Series([getattr(row, value_name, np.nan)]), errors="coerce"
            ).iloc[0]
            if pd.isna(value) or not np.isfinite(float(value)) or float(value) < 0:
                continue
            if len(targets) == 1:
                direct[targets[0]].append(
                    (float(value), bool(getattr(row, "source_is_exact", False)))
                )
            elif set(targets) == block_set:
                parents.append(row)
        return direct, parents

    def _candidate(
        frame: pd.DataFrame,
        *,
        value_name: str,
        label: str,
        block: tuple[str, ...],
        output_default: dict[str, float],
    ) -> tuple[dict[str, float] | None, dict[str, str] | None, str, dict]:
        direct, parents = _rows(frame, value_name, block)
        observed: dict[str, float] = {}
        for target, values in direct.items():
            if not values:
                continue
            exact = [value for value, is_exact in values if is_exact]
            observed[target] = exact[0] if exact else sum(value for value, _ in values)
        missing = [target for target in block if target not in observed]
        if not missing:
            return (
                observed,
                {target: label for target in block},
                "complete_direct",
                {"n_observed_children": len(block), "n_missing_children": 0},
            )
        if not parents:
            return None, None, "incomplete_rejected", {
                "n_observed_children": len(observed),
                "n_missing_children": len(missing),
                "parent_residual": pd.NA,
            }
        parent = sorted(
            parents,
            key=lambda row: (
                not bool(getattr(row, "source_is_exact", False)),
                str(getattr(row, "source_code", "")),
            ),
        )[0]
        parent_value = float(getattr(parent, value_name))
        residual = parent_value - sum(observed.values())
        tolerance = max(1e-9, 1e-9 * abs(parent_value))
        if residual < -tolerance:
            return None, None, "invalid_parent_residual", {
                "n_observed_children": len(observed),
                "n_missing_children": len(missing),
                "parent_residual": residual,
            }
        residual = max(0.0, residual)
        weights = dict(observed)
        sources = {target: label for target in observed}
        if len(missing) == 1:
            weights[missing[0]] = residual
            sources[missing[0]] = f"{label} parent residual"
            status = "identifiable_parent_residual"
        else:
            for target in missing:
                weights[target] = float(output_default.get(target, 0.0))
                sources[target] = "MRIO gross output"
            status = "parent_residual_with_unresolved_children"
        return weights, sources, status, {
            "n_observed_children": len(observed),
            "n_missing_children": len(missing),
            "parent_residual": residual,
            "parent_source_code": getattr(parent, "source_code", pd.NA),
        }

    seen_blocks = set()
    for row in hours_geo.itertuples(index=False):
        block = tuple(sorted(set(getattr(row, "targets", ()))))
        if len(block) <= 1:
            continue
        block_key = (str(getattr(row, "source_code", "")), block)
        if block_key in seen_blocks:
            continue
        seen_blocks.add(block_key)
        output_default = {
            target: float(base_weights.get(target, 0.0)) for target in block
        }
        selected = None
        attempts = []
        for frame, value_name, label in (
            (emp_hours_geo, "value", "Observed employment x mean weekly hours"),
            (employment_geo, "employment", "Observed industrial employment"),
        ):
            weights, sources, status, metadata = _candidate(
                frame,
                value_name=value_name,
                label=label,
                block=block,
                output_default=output_default,
            )
            attempts.append((status, label, metadata))
            if weights is not None and any(value > 0 for value in weights.values()):
                selected = (weights, sources, status, metadata, label)
                break
        if selected is None:
            weights = output_default
            sources = {target: "MRIO gross output" for target in block}
            status = "incomplete_rejected_output"
            metadata = {
                "n_observed_children": 0,
                "n_missing_children": len(block),
                "parent_residual": pd.NA,
            }
            selected_label = "FIGARO gross output"
        else:
            weights, sources, status, metadata, selected_label = selected
        weight_maps[block_key] = weights
        source_maps[block_key] = sources
        audit_rows.append(
            {
                "geo": geo,
                "hours_source_code": getattr(row, "source_code", pd.NA),
                "parent_targets": ";".join(block),
                "employment_weight_source": selected_label,
                "employment_coverage_status": status,
                "n_parent_children": len(block),
                "n_observed_employment_children": metadata.get("n_observed_children", 0),
                "n_missing_employment_children": metadata.get("n_missing_children", 0),
                "identifiable_employment_parent_residual": metadata.get("parent_residual", pd.NA),
                "employment_parent_source_code": metadata.get("parent_source_code", pd.NA),
                "incomplete_employment_rejected": status == "incomplete_rejected_output",
            }
        )
    return weight_maps, source_maps, audit_rows


def _resolve_fte40_country(
    geo: str,
    *,
    target_industries: list[str],
    national_fte_anchor: float | None = None,
    gva_source: pd.DataFrame,
    hours_source: pd.DataFrame,
    emp_hours_source: pd.DataFrame,
    employment_source: pd.DataFrame,
    output_source: pd.DataFrame,
    national_hours: dict[str, float],
    national_hours_metadata: dict[str, dict],
    national_hours_conflicts: list[dict],
    national_employment: dict[str, float],
    national_employment_metadata: dict[str, dict],
    national_employment_conflicts: list[dict],
    conflict_log: list[dict],
    strict_parent_conflicts: bool,
    employment_coverage_audit: list[dict] | None = None,
) -> tuple[dict[str, float], dict[str, dict]]:
    """Apply the single FTE40 priority hierarchy for one country."""
    if str(geo).strip().upper() == "ROW":
        # ROW is not an independently observed country.  Its A14/A64 values
        # are supplied later by apply_world_a14_row_residual(), using the
        # World FTE40 anchor less the explicit-country totals.  Keep the
        # constructor's row schema intact without applying an explicit-country
        # source-completeness validation or assigning the ROW anchor by output.
        pending = {}
        for target in target_industries:
            pending[target] = {
                "value": 0.0,
                "method": "ROW_PENDING_WORLD_A14",
                "source_code": pd.NA,
                "source_db": "Pending World-minus-explicit A14 residual",
                "source_year": pd.NA,
                "dataflow_id": pd.NA,
                "key_used": pd.NA,
                "population_scope": "Pending World FTE40/A14 residual",
                "source_coverage": pd.NA,
                "allocation_weight_source": "Pending World FTE40/A14 residual",
                "constraint_source": "Pending World FTE40/A14 residual",
                "original_source_observation": pd.NA,
                "parent_source_observation": pd.NA,
                "parent_residual_observation": pd.NA,
                "within_parent_weight": pd.NA,
                "original_source_code": pd.NA,
                "original_class_system": pd.NA,
                "hours_source_code": pd.NA,
                "mean_weekly_hours": pd.NA,
                "employment_observation": pd.NA,
                "hours_dataflow_id": pd.NA,
                "hours_source_year": pd.NA,
                "hours_population_scope": pd.NA,
                "compatibility_status": "pending World-minus-explicit residual",
                "output_allocation_block": pd.NA,
                "national_hours_total": pd.NA,
                "observed_hours_total": pd.NA,
                "remaining_hours_total": pd.NA,
                "national_hours_status": "pending World-minus-explicit residual",
                "national_hours_source_year": pd.NA,
                "national_employment_total": pd.NA,
                "national_employment_source_year": pd.NA,
                "employment_constraint_status": "not applicable",
            }
        return {target: 0.0 for target in target_industries}, pending
    target_set = set(target_industries)
    base = _output_weights_for_geo(
        output_source,
        geo=geo,
        target_industries=target_industries,
    )
    if not base:
        raise ValueError(
            f"No positive exact A64 MRIO output is available for {geo}; "
            "observed residual FTE cannot be allocated safely."
        )
    # Keep zero, positive, and missing FIGARO output distinct.  The positive
    # output weights above intentionally omit zero-output industries, so the
    # full map is needed for the residual audit below.
    output_values: dict[str, float] = {}
    for output_row in output_source.loc[output_source["geo"].eq(geo)].itertuples(index=False):
        targets = tuple(getattr(output_row, "targets", ()))
        if len(targets) != 1 or targets[0] not in target_set:
            continue
        value = pd.to_numeric(
            pd.Series([getattr(output_row, "output", np.nan)]), errors="coerce"
        ).iloc[0]
        if pd.notna(value) and np.isfinite(float(value)):
            output_values[targets[0]] = output_values.get(targets[0], 0.0) + float(value)

    def output_status(target: str) -> str:
        if target not in output_values:
            return "missing"
        return "zero" if output_values[target] <= 0 else "positive"
    emp_hours_weight_map = _source_weights(
        emp_hours_source,
        target_industries=target_industries,
        value_name="value",
        fallback_weights=base,
        conflict_log=conflict_log,
        strict_parent_conflicts=strict_parent_conflicts,
    )
    emp_hours_base = emp_hours_weight_map.get(geo, base)
    employment_weight_map = _source_weights(
        employment_source,
        target_industries=target_industries,
        value_name="employment",
        fallback_weights=emp_hours_base,
        conflict_log=conflict_log,
        strict_parent_conflicts=strict_parent_conflicts,
    )
    employment_base = employment_weight_map.get(geo, emp_hours_base)
    emp_hours_fallback_source = (
        "EMP_X_HOURS" if geo in emp_hours_weight_map else "MRIO gross output"
    )
    employment_detail = _resolve_fte_source_cells(
        employment_source,
        geo=geo,
        target_industries=target_industries,
        unresolved=set(target_industries),
        value_name="employment",
        direct_method="EMP_SHARE",
        parent_method="EMP_PARENT_OUTPUT",
        fallback_weights=base,
        fallback_weight_source="MRIO gross output",
        conflict_log=conflict_log,
        strict_parent_conflicts=strict_parent_conflicts,
    )
    # Build compatible employment-times-hours weights before resolving annual-
    # hours parent residuals. These are relative weights only; the observed
    # annual-hours parent remains the binding mass constraint.
    emp_hours_detail = _resolve_fte_source_cells(
        emp_hours_source,
        geo=geo,
        target_industries=target_industries,
        unresolved=set(target_industries),
        value_name="value",
        direct_method="EMP_X_HOURS",
        parent_method="EMP_X_HOURS",
        fallback_weights=base,
        fallback_weight_source="MRIO gross output",
        conflict_log=conflict_log,
        strict_parent_conflicts=strict_parent_conflicts,
    )
    annual_hours_fallback_weights = dict(base)
    annual_hours_fallback_sources = {
        target: "MRIO gross output" for target in target_industries
    }
    # Prefer compatible labour weights for the unresolved children of an
    # observed annual-hours parent. Direct employment-times-hours weights
    # outrank observed employment; observed employment from any compatible
    # source database is next; MRIO output remains the final fallback.
    for target, item in emp_hours_detail.items():
        if item["value"] <= 0 or item.get("allocation_weight_source") == "MRIO gross output":
            continue
        annual_hours_fallback_weights[target] = item["value"]
        annual_hours_fallback_sources[target] = "Observed employment x mean weekly hours"
    for target, item in employment_detail.items():
        if target in emp_hours_detail or item["value"] <= 0:
            continue
        if item["method"] != "EMP_SHARE":
            continue
        source_text = str(item.get("source_db", "")).lower()
        source_label = (
            "OECD industrial employment" if "oecd" in source_text
            else "Eurostat industrial employment" if "eurostat" in source_text
            else "ILOSTAT industrial employment"
        )
        annual_hours_fallback_weights[target] = item["value"]
        annual_hours_fallback_sources[target] = source_label
    (
        annual_hours_fallback_by_block,
        annual_hours_sources_by_block,
        coverage_rows,
    ) = _build_parent_labor_fallbacks(
        hours_source,
        emp_hours_source=emp_hours_source,
        employment_source=employment_source,
        base_weights=base,
        geo=geo,
    )
    if employment_coverage_audit is not None:
        employment_coverage_audit.extend(coverage_rows)
    # Preserve A14 employment constraints, but replace the within-parent GVA
    # split with employment x mean-hours only where the detailed source covers
    # every FIGARO child in that parent. Partial coverage is never normalised
    # as if it described the omitted industries.
    employment_fallback_weights = dict(base)
    employment_fallback_sources: dict[str, str] = {
        target: "MRIO gross output" for target in target_industries
    }
    employment_rows = employment_source.loc[employment_source["geo"].eq(geo)]
    for employment_row in employment_rows.itertuples(index=False):
        parent_targets = tuple(
            target for target in employment_row.targets if target in target_set
        )
        if not parent_targets or not all(target in emp_hours_detail for target in parent_targets):
            continue
        for target in parent_targets:
            detail = emp_hours_detail[target]
            employment_fallback_weights[target] = float(detail["value"])
            employment_fallback_sources[target] = (
                "ILOSTAT employment x mean weekly hours"
                if detail["allocation_weight_source"] == "Direct observation"
                else str(detail["allocation_weight_source"])
            )
    priority_sources = (
        (
            hours_source, "hours", "EU_H_DIRECT", "EU_H_PARENT",
            annual_hours_fallback_weights, annual_hours_fallback_sources,
        ),
        (
            emp_hours_source, "value", "EMP_X_HOURS", "EMP_X_HOURS",
            employment_base, "EMP_SHARE",
        ),
        (
            employment_source, "employment", "EMP_SHARE", "EMP_PARENT_OUTPUT",
            employment_fallback_weights, emp_hours_fallback_source,
        ),
    )

    def apply_relative_sources(weights, unresolved, details, sources):
        for (
            source, value_name, direct_method, parent_method,
            fallback_source_weights, fallback_weight_source,
        ) in sources:
            resolved = _resolve_fte_source_cells(
                source,
                geo=geo,
                target_industries=target_industries,
                unresolved=unresolved,
                value_name=value_name,
                direct_method=direct_method,
                parent_method=parent_method,
                fallback_weights=fallback_source_weights,
                fallback_weight_source=fallback_weight_source,
                conflict_log=conflict_log,
                strict_parent_conflicts=strict_parent_conflicts,
            )
            if not resolved:
                continue
            mass = sum(weights.get(target, 0.0) for target in resolved)
            total_value = sum(item["value"] for item in resolved.values())
            if total_value <= 0 or mass <= 0:
                continue
            for target, item in resolved.items():
                weights[target] = mass * item["value"] / total_value
                item = dict(item)
                item["constraint_source"] = _fte_constraint_label(
                    item["method"], item.get("source_db", pd.NA)
                )
                details[target] = item
            unresolved -= set(resolved)
        return weights, unresolved, details

    def national_residual_detail(target):
        output_rows = output_source.loc[
            output_source["geo"].eq(geo)
            & output_source["targets"].map(lambda values: target in values)
        ]
        output_row = output_rows.iloc[0] if not output_rows.empty else None
        return {
            "value": 0.0,
            "method": "NATIONAL_RESIDUAL_OUTPUT",
            "source_code": output_row["source_code"] if output_row is not None else "",
            "source_db": output_row["source_db"] if output_row is not None else "FIGARO output",
            "source_year": output_row["source_year"] if output_row is not None else pd.NA,
            "dataflow_id": output_row["dataflow_id"] if output_row is not None else pd.NA,
            "key_used": output_row["key_used"] if output_row is not None else pd.NA,
            "population_scope": "FTE40 national residual",
            "source_coverage": ";".join(output_row["targets"]) if output_row is not None else "",
            "allocation_weight_source": "MRIO gross output",
            "constraint_source": "FTE40 national anchor",
            "original_source_observation": pd.NA,
            "parent_source_observation": pd.NA,
            "parent_residual_observation": pd.NA,
            "within_parent_weight": pd.NA,
            "original_source_code": output_row["original_source_code"] if output_row is not None and "original_source_code" in output_row else pd.NA,
            "original_class_system": output_row["original_class_system"] if output_row is not None and "original_class_system" in output_row else "FIGARO A64",
            "hours_source_code": pd.NA,
            "mean_weekly_hours": pd.NA,
            "employment_observation": pd.NA,
            "hours_dataflow_id": pd.NA,
            "hours_source_year": pd.NA,
            "hours_population_scope": pd.NA,
            "compatibility_status": "not applicable",
            "output_allocation_block": pd.NA,
        }

    def zero_output_detail(target: str, *, status: str):
        detail = national_residual_detail(target)
        detail.update(
            {
                "method": "ZERO_OUTPUT_UNRESOLVED",
                "allocation_weight_source": "No positive MRIO output",
                "constraint_source": "No modeled production",
                "output_allocation_share": 0.0,
                "allocation_raw_numerator": 0.0,
                "allocation_raw_denominator": 0.0,
                "allocation_raw_unit": "million EUR",
                "output_residual_status": status,
                "zero_output_industry": True,
            }
        )
        return detail

    def national_employment_residual_detail(
        target: str,
        *,
        residual_employment: float,
        output_share: float,
        output_numerator: float,
        output_denominator: float,
    ):
        metadata = national_employment_metadata.get(geo, {})
        return {
            "value": residual_employment * output_share,
            "method": "EMP_NATIONAL_RESIDUAL_OUTPUT",
            "source_code": "TOTAL",
            "source_db": metadata.get("source_db", "ILOSTAT"),
            "source_year": metadata.get("source_year", pd.NA),
            "dataflow_id": metadata.get("dataflow_id", pd.NA),
            "key_used": metadata.get("key_used", pd.NA),
            "population_scope": "employment population (national TOTAL)",
            "source_coverage": "TOTAL",
            "allocation_weight_source": "MRIO gross output",
            "constraint_source": "ILOSTAT national employment total",
            "original_source_observation": employment_total,
            "parent_source_observation": employment_total,
            "parent_residual_observation": residual_employment,
            "within_parent_weight": output_share,
            "original_class_system": "ISIC4/ISIC3 national total",
            "hours_source_code": pd.NA,
            "mean_weekly_hours": pd.NA,
            "employment_observation": residual_employment * output_share,
            "hours_dataflow_id": pd.NA,
            "hours_source_year": pd.NA,
            "hours_population_scope": pd.NA,
            "compatibility_status": "compatible national employment total with observed detailed children",
            "output_allocation_block": tuple(sorted(unresolved)),
            "output_allocation_share": output_share,
            "allocation_raw_numerator": output_numerator,
            "allocation_raw_denominator": output_denominator,
            "allocation_raw_unit": "million EUR",
        }

    def finalize_weights(weights, details):
        """Normalize relative weights and retain the national discrepancy audit."""
        total = float(sum(weights.values()))
        if total <= 0:
            raise ValueError(
                f"No positive FTE40 allocation weights are available for {geo}; "
                "equal-split fallback is disabled."
            )
        anchor = pd.to_numeric(pd.Series([national_fte_anchor]), errors="coerce").iloc[0]
        anchor_value = (
            float(anchor) if pd.notna(anchor) and np.isfinite(float(anchor)) else np.nan
        )
        raw_fte_values = pd.to_numeric(
            pd.Series(
                [detail.get("raw_fte40_estimate", pd.NA) for detail in details.values()]
            ),
            errors="coerce",
        ).dropna()
        raw_fte_total = float(raw_fte_values.sum()) if not raw_fte_values.empty else np.nan
        pre_fte = (
            raw_fte_total
            if np.isfinite(raw_fte_total)
            else anchor_value * total
            if np.isfinite(anchor_value)
            else np.nan
        )
        discrepancy = anchor_value - pre_fte if np.isfinite(anchor_value) else np.nan
        normalization_factor = (
            anchor_value / pre_fte
            if np.isfinite(anchor_value) and np.isfinite(pre_fte) and pre_fte > 0
            else 1.0 / total
        )
        zero_excluded = any(
            detail.get("method") == "ZERO_OUTPUT_UNRESOLVED"
            for detail in details.values()
        )
        status = (
            "zero_output_residual_excluded"
            if zero_excluded
            else "national_anchor_normalization"
        )
        for target in target_industries:
            detail = details[target]
            detail["pre_normalization_share"] = float(weights.get(target, 0.0))
            detail_raw_fte = pd.to_numeric(
                pd.Series([detail.get("raw_fte40_estimate", pd.NA)]), errors="coerce"
            ).iloc[0]
            detail["pre_normalization_fte"] = (
                float(detail_raw_fte)
                if pd.notna(detail_raw_fte)
                else (
                    anchor_value * float(weights.get(target, 0.0))
                    if np.isfinite(anchor_value) else pd.NA
                )
            )
            detail["national_normalization_factor"] = normalization_factor
            detail["national_discrepancy_before_normalization"] = discrepancy
            detail["national_discrepancy_status"] = status
            state = output_status(target)
            detail["zero_output_industry"] = state == "zero"
            detail["output_status"] = state
            detail["fte_output_coefficient_status"] = (
                "undefined_zero_output" if state == "zero" else "defined"
            )
        return {
            target: float(value) / total
            for target, value in weights.items()
        }, details

    total_hours = national_hours.get(geo)
    if total_hours is not None and total_hours > 0:
        hour_source, hour_value_name, hour_direct_method, hour_parent_method, hour_fallback, hour_fallback_source = priority_sources[0]
        unresolved = set(target_industries)
        hour_resolved = _resolve_fte_source_cells(
            hour_source,
            geo=geo,
            target_industries=target_industries,
            unresolved=unresolved,
            value_name=hour_value_name,
            direct_method=hour_direct_method,
            parent_method=hour_parent_method,
            fallback_weights=hour_fallback,
            fallback_weight_source=hour_fallback_source,
            fallback_weights_by_block=annual_hours_fallback_by_block,
            fallback_sources_by_block=annual_hours_sources_by_block,
            conflict_log=conflict_log,
            strict_parent_conflicts=strict_parent_conflicts,
        )
        observed_hours = sum(item["value"] for item in hour_resolved.values())
        # Eurostat annual-hour cells are integer-valued.  A small excess of
        # one or two hours can therefore arise after resolving overlapping
        # parent/detail rows, and is treated as source rounding rather than
        # as an alternative denominator.
        national_hours_tolerance = max(2.0, 1e-6 * total_hours)
        if observed_hours <= total_hours + national_hours_tolerance:
            weights = {target: 0.0 for target in target_industries}
            details = {}
            for target, item in hour_resolved.items():
                item = dict(item)
                weights[target] = item["value"] / total_hours
                item["allocation_raw_numerator"] = item["value"]
                item["allocation_raw_denominator"] = total_hours
                item["allocation_raw_unit"] = "thousand hours/year"
                item["constraint_source"] = _fte_constraint_label(
                    item["method"], item.get("source_db", pd.NA)
                )
                details[target] = item
            unresolved -= set(hour_resolved)
            residual_share = max(0.0, 1.0 - observed_hours / total_hours)
            zero_output_unresolved = {
                target for target in unresolved if output_status(target) == "zero"
            }
            missing_output_unresolved = {
                target for target in unresolved if output_status(target) == "missing"
            }
            for target in zero_output_unresolved:
                details[target] = zero_output_detail(
                    target,
                    status=(
                        "zero_output_zero_residual"
                        if residual_share <= 1e-12
                        else "zero_output_residual_excluded"
                    ),
                )
                weights[target] = 0.0
            unresolved -= zero_output_unresolved
            if missing_output_unresolved:
                raise ValueError(
                    f"Annual-hours residual for {geo} has missing MRIO output for "
                    f"unresolved industries: {tuple(sorted(missing_output_unresolved))}."
                )
            residual_weights = {
                target: float(base.get(target, 0.0))
                for target in unresolved
            }
            denominator = sum(residual_weights.values())
            if unresolved and denominator <= 0 and residual_share > 0:
                conflict_log.append(
                    {
                        "geo": geo,
                        "source": "FTE40 allocation",
                        "parent_targets": ";".join(sorted(unresolved)),
                        "residual": residual_share,
                        "rule": "annual-hours residual unresolved: no positive MRIO output",
                    }
                )
                raise ValueError(
                    f"Annual-hours residual for {geo} has no positive MRIO output "
                    "for the unresolved industries."
                )
            if denominator > 0:
                residual_weights = {
                    target: value / denominator
                    for target, value in residual_weights.items()
                }
            residual_weights, unresolved, details = apply_relative_sources(
                residual_weights,
                unresolved,
                details,
                priority_sources[1:],
            )
            if unresolved:
                positive_output_unresolved = tuple(sorted(unresolved))
                conflict_log.append(
                    {
                        "geo": geo,
                        "source": "FTE40 allocation",
                        "parent_targets": ";".join(positive_output_unresolved),
                        "residual": residual_share,
                        "rule": "annual-hours residual unresolved without compatible industrial source",
                    }
                )
                raise ValueError(
                    f"Annual-hours allocation for {geo} leaves positive-output "
                    f"industries unresolved without a compatible industrial source "
                    f"or parent residual: {positive_output_unresolved}."
                )
            for target in target_industries:
                if target not in details:
                    continue
                details[target]["national_hours_total"] = total_hours
                details[target]["observed_hours_total"] = observed_hours
                details[target]["remaining_hours_total"] = total_hours - observed_hours
                details[target]["national_hours_status"] = (
                    "valid_rounding"
                    if observed_hours > total_hours
                    else "valid"
                )
                details[target]["national_hours_source_year"] = national_hours_metadata.get(geo, {}).get("source_year", pd.NA)
            return finalize_weights(weights, details)

        national_hours_conflicts.append(
            {
                "geo": geo,
                "source": "hours",
                "rule": "observed industry hours exceed national TOTAL; relative fallback used",
                "national_total": total_hours,
                "observed_hours": observed_hours,
                "residual": total_hours - observed_hours,
            }
        )

    # A complete annual-hours classification remains a valid relative
    # allocation when a national annual-hours TOTAL is unavailable.  Partial
    # hours observations, like partial employment observations, are never
    # normalised as though they covered the whole economy.
    relative_hours = _resolve_fte_source_cells(
        hours_source,
        geo=geo,
        target_industries=target_industries,
        unresolved=set(target_industries),
        value_name="hours",
        direct_method="EU_H_DIRECT",
        parent_method="EU_H_PARENT",
        fallback_weights=annual_hours_fallback_weights,
        fallback_weight_source=annual_hours_fallback_sources,
        fallback_weights_by_block=annual_hours_fallback_by_block,
        fallback_sources_by_block=annual_hours_sources_by_block,
        conflict_log=conflict_log,
        strict_parent_conflicts=strict_parent_conflicts,
    )
    relative_hours_mass = sum(item["value"] for item in relative_hours.values())
    if set(relative_hours) == set(target_industries) and relative_hours_mass > 0:
        weights = {target: 0.0 for target in target_industries}
        details = {}
        for target, item in relative_hours.items():
            detail = dict(item)
            weights[target] = detail["value"] / relative_hours_mass
            detail["allocation_raw_numerator"] = detail["value"]
            detail["allocation_raw_denominator"] = relative_hours_mass
            detail["allocation_raw_unit"] = "thousand hours/year"
            detail["constraint_source"] = _fte_constraint_label(
                detail["method"], detail.get("source_db", pd.NA)
            )
            detail["national_hours_total"] = total_hours if total_hours is not None else pd.NA
            detail["observed_hours_total"] = pd.NA
            detail["remaining_hours_total"] = pd.NA
            detail["national_hours_status"] = "relative_complete"
            detail["national_hours_source_year"] = national_hours_metadata.get(geo, {}).get("source_year", pd.NA)
            detail["national_employment_total"] = pd.NA
            detail["national_employment_source_year"] = pd.NA
            detail["employment_constraint_status"] = "not_used"
            details[target] = detail
        return finalize_weights(weights, details)

    # Outside the independently calibrated Eurostat annual-hours branch, a
    # complete compatible employment-times-hours extract is the preferred
    # relative allocation.  It is used only when every target industry is
    # covered; a partial extract cannot define national mass.
    emp_hours_resolved = _resolve_fte_source_cells(
        emp_hours_source,
        geo=geo,
        target_industries=target_industries,
        unresolved=set(target_industries),
        value_name="value",
        direct_method="EMP_X_HOURS",
        parent_method="EMP_X_HOURS",
        fallback_weights=base,
        fallback_weight_source="MRIO gross output",
        conflict_log=conflict_log,
        strict_parent_conflicts=strict_parent_conflicts,
    )
    emp_hours_mass = sum(item["value"] for item in emp_hours_resolved.values())
    if set(emp_hours_resolved) == set(target_industries) and emp_hours_mass > 0:
        weights = {target: 0.0 for target in target_industries}
        details = {}
        for target, item in emp_hours_resolved.items():
            detail = dict(item)
            weights[target] = detail["value"] / emp_hours_mass
            detail["allocation_raw_numerator"] = detail["value"]
            detail["allocation_raw_denominator"] = emp_hours_mass
            detail["allocation_raw_unit"] = "thousand person-hours/week"
            detail["constraint_source"] = _fte_constraint_label(
                detail["method"], detail.get("source_db", pd.NA)
            )
            detail["national_hours_total"] = total_hours if total_hours is not None else pd.NA
            detail["observed_hours_total"] = pd.NA
            detail["remaining_hours_total"] = pd.NA
            detail["national_hours_status"] = "relative_complete"
            detail["national_hours_source_year"] = national_hours_metadata.get(geo, {}).get("source_year", pd.NA)
            detail["national_employment_total"] = pd.NA
            detail["national_employment_source_year"] = pd.NA
            detail["employment_constraint_status"] = "not_used"
            detail["raw_fte40_estimate"] = detail["value"] / 40.0
            details[target] = detail
        return finalize_weights(weights, details)

    # An employment observation may determine national FTE mass only when
    # its source also supplies a compatible employment TOTAL, or when the
    # source resolves the complete non-overlapping FIGARO industry universe.
    # This prevents partial observations from being silently normalised
    # against GVA-defined mass.
    employment_resolved = _resolve_fte_source_cells(
        employment_source,
        geo=geo,
        target_industries=target_industries,
        unresolved=set(target_industries),
        value_name="employment",
        direct_method="EMP_SHARE",
        parent_method="EMP_PARENT_OUTPUT",
        fallback_weights=employment_fallback_weights,
        fallback_weight_source=employment_fallback_sources,
        conflict_log=conflict_log,
        strict_parent_conflicts=strict_parent_conflicts,
    )
    employment_mass = sum(item["value"] for item in employment_resolved.values())
    employment_total = national_employment.get(geo)
    employment_complete = set(employment_resolved) == set(target_industries)

    details: dict[str, dict] = {}
    weights = {target: 0.0 for target in target_industries}
    unresolved = set(target_industries)
    used_employment_constraint = False
    employment_share_tolerance = 1e-9

    if employment_resolved and employment_total is not None and employment_total > 0:
        coverage = employment_mass / employment_total
        # ILOSTAT reports the A14 components and TOTAL independently in
        # thousand persons.  Their published rounding can exceed 1e-9 of the
        # national total when fourteen components are summed.
        employment_tolerance = max(1e-3, 1e-6 * employment_total)
        employment_share_tolerance = max(
            1e-9, employment_tolerance / employment_total
        )
        if coverage <= 1.0 + employment_tolerance:
            coverage = min(coverage, 1.0)
            compatible_labour = {
                target: emp_hours_detail[target]
                for target in employment_resolved
                if target in emp_hours_detail
            }
            labour_mass = sum(item["value"] for item in compatible_labour.values())
            use_labour_weights = (
                set(compatible_labour) == set(employment_resolved)
                and labour_mass > 0
            )
            for target, item in employment_resolved.items():
                detail = (
                    dict(compatible_labour[target])
                    if use_labour_weights else dict(item)
                )
                weights[target] = (
                    coverage * detail["value"] / labour_mass
                    if use_labour_weights
                    else detail["value"] / employment_total
                )
                if use_labour_weights:
                    detail["method"] = "EMP_X_HOURS"
                    detail["allocation_weight_source"] = "ILOSTAT employment x mean weekly hours"
                    detail["constraint_source"] = "ILOSTAT industrial employment"
                    detail["allocation_raw_numerator"] = detail["value"]
                    detail["allocation_raw_denominator"] = labour_mass
                    detail["allocation_raw_unit"] = "thousand person-hours/week"
                else:
                    detail["allocation_raw_numerator"] = detail["value"]
                    detail["allocation_raw_denominator"] = employment_total
                    detail["allocation_raw_unit"] = "thousand employed persons"
                    detail["constraint_source"] = _fte_constraint_label(
                        detail["method"], detail.get("source_db", pd.NA)
                    )
                details[target] = detail
            unresolved -= set(employment_resolved)
            used_employment_constraint = True
        else:
            national_employment_conflicts.append(
                {
                    "geo": geo,
                    "source": "employment",
                    "rule": "industry employment exceeds compatible national TOTAL; constraint ignored",
                    "national_total": employment_total,
                    "observed_employment": employment_mass,
                    "residual": employment_total - employment_mass,
                }
            )
            raise ValueError(
                f"Observed industrial constraints exceed the national FTE40 anchor for {geo}: "
                f"observed={employment_mass:.12g}, national_total={employment_total:.12g}."
            )
    elif employment_resolved and employment_complete and employment_mass > 0:
        for target, item in employment_resolved.items():
            detail = dict(item)
            weights[target] = detail["value"] / employment_mass
            detail["constraint_source"] = _fte_constraint_label(
                detail["method"], detail.get("source_db", pd.NA)
            )
            details[target] = detail
        unresolved.clear()
        used_employment_constraint = True

    # The independent national anchor is applied only by final weight
    # normalisation.  It is not an industrial observation and must not be
    # converted into a residual for otherwise-unresolved positive-output
    # industries.  Such industries require a compatible source block (or a
    # valid parent residual) before they can receive a provisional weight.
    residual_share = 1.0 - sum(weights.values())
    if residual_share < -employment_share_tolerance:
        national_employment_conflicts.append(
            {
                "geo": geo,
                "source": "FTE40 allocation",
                "rule": "observed constraints exceed national FTE40 anchor",
                "national_total": 1.0,
                "observed_employment": sum(weights.values()),
                "residual": residual_share,
            }
        )
        raise ValueError(
            f"Observed industrial constraints exceed the national FTE40 anchor for {geo}."
        )
    residual_share = max(0.0, residual_share)
    if unresolved:
        zero_output_unresolved = {
            target for target in unresolved if output_status(target) == "zero"
        }
        missing_output_unresolved = {
            target for target in unresolved if output_status(target) == "missing"
        }
        for target in zero_output_unresolved:
            details[target] = zero_output_detail(
                target,
                status=(
                    "zero_output_zero_residual"
                    if residual_share <= 1e-12
                    else "zero_output_residual_excluded"
                ),
            )
            weights[target] = 0.0
        unresolved -= zero_output_unresolved
        if missing_output_unresolved and residual_share > 1e-12:
            raise ValueError(
                f"Unresolved FTE residual for {geo} has missing MRIO output for "
                f"industries: {tuple(sorted(missing_output_unresolved))}."
            )
        if unresolved:
            # A compatible source-observed national employment TOTAL is a
            # parent constraint, distinct from the independent FTE40 anchor.
            # Use only its residual after the resolved industrial observations
            # have been removed, and distribute that residual by MRIO output.
            # This never creates an employment mass from the FTE40 anchor.
            if used_employment_constraint and employment_total is not None and employment_total > 0:
                employment_residual = float(employment_total - employment_mass)
                employment_tolerance = max(1e-3, 1e-6 * float(employment_total))
                if employment_residual >= -employment_tolerance:
                    employment_residual = max(0.0, employment_residual)
                    residual_output = {
                        target: float(base.get(target, 0.0))
                        for target in unresolved
                    }
                    output_denominator = sum(residual_output.values())
                    if output_denominator > 0:
                        for target, output_numerator in residual_output.items():
                            output_share = output_numerator / output_denominator
                            detail = national_employment_residual_detail(
                                target,
                                residual_employment=employment_residual,
                                output_share=output_share,
                                output_numerator=output_numerator,
                                output_denominator=output_denominator,
                            )
                            details[target] = detail
                            weights[target] = residual_share * output_share
                        unresolved.clear()
            if not unresolved:
                pass
            else:
                positive_output_unresolved = tuple(sorted(unresolved))
                conflict_log.append(
                    {
                        "geo": geo,
                        "source": "FTE40 allocation",
                        "parent_targets": ";".join(positive_output_unresolved),
                        "residual": residual_share,
                        "residual_share": residual_share,
                        "output_values": {
                            target: output_values.get(target, np.nan)
                            for target in positive_output_unresolved
                        },
                        "rule": "positive-output industries unresolved without compatible industrial source",
                    }
                )
                raise ValueError(
                    f"Positive-output industries for {geo} remain unresolved without a "
                    f"compatible industrial source or parent residual: {positive_output_unresolved}. "
                    "The national FTE40 anchor is not used as an industrial residual."
                )

    status = "missing" if total_hours is None else "inconsistent"
    for target in target_industries:
        details[target]["national_hours_total"] = total_hours if total_hours is not None else pd.NA
        details[target]["observed_hours_total"] = pd.NA
        details[target]["remaining_hours_total"] = pd.NA
        details[target]["national_hours_status"] = status
        details[target]["national_hours_source_year"] = national_hours_metadata.get(geo, {}).get("source_year", pd.NA)
        details[target]["national_employment_total"] = employment_total if employment_total is not None else pd.NA
        details[target]["national_employment_source_year"] = national_employment_metadata.get(geo, {}).get("source_year", pd.NA)
        details[target]["employment_constraint_status"] = (
            "valid" if used_employment_constraint else "unavailable"
        )
    return finalize_weights(weights, details)


def _build_row_a14_employment_residual(
    global_employment: pd.DataFrame | None,
    explicit_employment: pd.DataFrame | None,
    *,
    explicit_geos: set[str],
    target_industries: set[str],
    row_geo: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build compatible World-minus-FIGARO-country A14 employment residuals.

    The result contains employment counts used exclusively as relative ROW
    industry weights.  The ROW FTE40 national anchor remains independent.
    A group is admitted only when every explicit FIGARO country has a
    compatible observation under the same ILOSTAT dataflow, year, population,
    and A14 source code as the World observation.
    """
    empty = pd.DataFrame(
        columns=[
            "geo", "source_code", "targets", "employment",
            "source_is_exact", "source_db", "source_year", "dataflow_id",
            "key_used", "population_scope", "class_system", "source_coverage",
        ]
    )
    diagnostics: dict[str, object] = {
        "status": "unavailable",
        "used_groups": "",
        "missing_groups": "",
        "reason": "global or explicit ILOSTAT employment was not supplied",
    }
    if global_employment is None or explicit_employment is None:
        return empty, diagnostics

    global_std = _standardize_fte_source(
        global_employment,
        target_industries=target_industries,
        value_name="employment",
        source_name="global ILOSTAT employment",
    )
    explicit_std = _standardize_fte_source(
        explicit_employment,
        target_industries=target_industries,
        value_name="employment",
        source_name="explicit ILOSTAT employment",
    )
    global_std = global_std.loc[global_std["geo"].eq(row_geo)].copy()
    explicit_std = explicit_std.loc[explicit_std["geo"].isin(explicit_geos)].copy()
    if global_std.empty or explicit_std.empty:
        diagnostics["reason"] = "no usable compatible A14 rows after standardisation"
        return empty, diagnostics

    def dataflow_family(value) -> str:
        text = str(value).strip().upper()
        text = re.sub(r"^DF_", "", text)
        return re.sub(r"_A$", "", text)

    rows = []
    unavailable = []
    for global_row in global_std.itertuples(index=False):
        source_code = str(global_row.source_code)
        if source_code not in ILO_GLOBAL_A14_CODES:
            continue
        candidates = explicit_std.loc[
            explicit_std["source_code"].eq(source_code)
            & explicit_std["source_year"].eq(global_row.source_year)
            & explicit_std["dataflow_id"].map(dataflow_family).eq(
                dataflow_family(global_row.dataflow_id)
            )
            & explicit_std["population_scope"].astype("string").eq(str(global_row.population_scope))
        ].copy()
        per_geo = candidates.groupby("geo", as_index=False)["employment"].sum()
        if set(per_geo["geo"]) != explicit_geos:
            unavailable.append(source_code)
            continue
        explicit_total = float(per_geo["employment"].sum())
        residual = float(global_row.employment) - explicit_total
        tolerance = max(1e-6, 1e-9 * abs(float(global_row.employment)))
        if residual < -tolerance:
            unavailable.append(source_code)
            continue
        rows.append(
            {
                "geo": row_geo,
                "source_code": source_code,
                "targets": global_row.targets,
                "employment": max(residual, 0.0),
                "source_is_exact": bool(global_row.source_is_exact),
                "source_db": "ILOSTAT World-minus-explicit residual",
                "source_year": global_row.source_year,
                "dataflow_id": global_row.dataflow_id,
                "key_used": global_row.key_used,
                "population_scope": global_row.population_scope,
                "class_system": global_row.class_system,
                "source_coverage": global_row.source_coverage,
            }
        )

    if not rows:
        diagnostics["reason"] = (
            "no A14 group has complete compatible explicit-country coverage"
        )
        diagnostics["missing_groups"] = ";".join(sorted(set(unavailable)))
        return empty, diagnostics
    result = pd.DataFrame(rows)
    expected = set(ILO_GLOBAL_A14_CODES)
    used = set(result["source_code"])
    diagnostics.update(
        {
            "status": "complete" if used == expected else "partial",
            "used_groups": ";".join(sorted(used)),
            "missing_groups": ";".join(sorted(expected - used)),
            "reason": "compatible World-minus-explicit ILOSTAT A14 residual",
        }
    )
    # A partial residual is not promoted to a national ROW constraint because
    # that would treat missing A14 groups as zero employment.
    if used != expected:
        return empty, diagnostics
    return result, diagnostics


def _unused_labor_information_audit(
    emp_hours: pd.DataFrame,
    allocated: pd.DataFrame,
    *,
    geos: set[str],
) -> pd.DataFrame:
    """Flag compatible employment-hours information that final allocation did not use.

    The diagnostic is deliberately based on the already validated paired
    source table.  It therefore distinguishes unavailable information from
    labour observations that survived source selection but were subsequently
    bypassed by the hierarchy resolver.
    """
    rows = []
    for geo in sorted(geos):
        available = emp_hours.loc[emp_hours["geo"].eq(geo)].copy()
        final = allocated.loc[allocated["geo"].eq(geo)].copy()
        available_targets = set()
        available_pairs = 0
        if not available.empty:
            available_pairs = sum(
                len([code for code in str(value).split(";") if code])
                for value in available.get("original_source_code", available["source_code"])
            )
            for targets in available["targets"]:
                available_targets.update(targets if isinstance(targets, tuple) else (targets,))
        used_targets = set(
            final.loc[
                final["allocation_method"].eq("EMP_X_HOURS"), "industry"
            ].astype(str)
        )
        unused_targets = available_targets - used_targets
        if not available_targets:
            status = "unavailable"
            reason = "no compatible employment-hours source pair"
        elif used_targets:
            status = "used"
            reason = (
                "compatible detailed labour information used; remaining mapped "
                "targets were constrained by higher-priority annual hours or unresolved"
            )
        else:
            status = "flagged"
            reason = "compatible employment-hours pairs exist but no final EMP_X_HOURS allocation"
        rows.append(
            {
                "geo": geo,
                "n_compatible_emp_hours_pairs": int(available_pairs),
                "n_figaro_industries_compatible_emp_hours": int(len(available_targets)),
                "n_figaro_industries_emp_hours_used": int(len(used_targets & available_targets)),
                "n_unused_labor_targets": int(len(unused_targets)),
                "unused_labor_information_status": status,
                "unused_labor_information_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def build_fte40_country_industry(
    fte40_totals: pd.DataFrame,
    mrio_industries,
    *,
    hours_total: pd.DataFrame | None = None,
    employment_industry: pd.DataFrame | None = None,
    hours_mean: pd.DataFrame | None = None,
    gva: pd.DataFrame | None = None,
    figaro_output=None,
    global_employment: pd.DataFrame | None = None,
    row_employment_industry: pd.DataFrame | None = None,
    source_availability_audit: pd.DataFrame | None = None,
    explicit_geos: set[str] | list[str] | tuple[str, ...] | None = None,
    world_geo: str = "X01",
    row_geo: str = "ROW",
    tolerance: float = 1,
    verbose: bool = False,
    strict_source_conflicts: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build calibrated FTE40 by country and FIGARO industry.

    FTE40 country totals are the only calibration anchors. Industrial sources
    provide shares in this order: observed total hours, employment multiplied
    by compatible mean hours, and observed employment. Parent and national
    residuals are split over unresolved children using MRIO gross output. The
    independent national anchor is applied only after provisional weights are
    complete; it is never treated as an industrial residual.
    """
    target_industries = [str(value).strip() for value in mrio_industries]
    target_industries = list(dict.fromkeys(target_industries))
    target_set = set(target_industries)
    totals = fte40_totals.copy()
    if totals.empty:
        raise ValueError("fte40_totals is empty.")
    if not {"geo", "OBS_VALUE"}.issubset(totals.columns):
        raise ValueError("fte40_totals must contain geo and OBS_VALUE columns.")
    totals["geo"] = totals["geo"].astype(str).str.strip().str.upper()
    totals["OBS_VALUE"] = pd.to_numeric(totals["OBS_VALUE"], errors="coerce")
    totals = totals.dropna(subset=["OBS_VALUE"])
    totals = totals.loc[~totals["geo"].isin({world_geo, row_geo})]
    excluded_fte_anchors: list[str] = []
    if explicit_geos is not None:
        allowed_geos = {str(value).strip().upper() for value in explicit_geos}
        excluded_fte_anchors = sorted(set(totals.loc[~totals["geo"].isin(allowed_geos), "geo"]))
        totals = totals.loc[totals["geo"].isin(allowed_geos)].copy()
    world_rows = fte40_totals.loc[fte40_totals["geo"].astype(str).str.strip().str.upper().eq(world_geo)].copy()
    world_rows["OBS_VALUE"] = pd.to_numeric(world_rows["OBS_VALUE"], errors="coerce")
    world_rows = world_rows.dropna(subset=["OBS_VALUE"])
    if len(world_rows) != 1:
        raise ValueError("fte40_totals must contain exactly one World (X01) anchor.")
    world_total = float(world_rows.iloc[0]["OBS_VALUE"])

    country_totals = totals.groupby("geo", as_index=False)["OBS_VALUE"].sum()
    explicit_geos = set(country_totals["geo"])
    explicit_total = float(country_totals["OBS_VALUE"].sum())
    row_total = world_total - explicit_total
    if row_total < -float(tolerance):
        raise ValueError(
            "World FTE40 is smaller than the explicit-country total: "
            f"residual={row_total:.12g}. Check country coverage and duplicates."
        )
    if abs(row_total) <= float(tolerance):
        row_total = 0.0
    country_totals = pd.concat(
        [country_totals, pd.DataFrame({"geo": [row_geo], "OBS_VALUE": [row_total]})],
        ignore_index=True,
    )

    gva_std = _standardize_fte_source(
        gva,
        target_industries=target_set,
        value_name="gva",
        source_name="FIGARO GVA",
    )
    output_std = _standardize_output_frame(
        figaro_output,
        target_industries=target_set,
    )
    if output_std.empty:
        raise ValueError(
            "FIGARO gross output is required for observed FTE residual allocation."
        )
    source_conflicts: list[dict] = []
    employment_coverage_audit: list[dict] = []
    gva_nonpositive_by_geo = {
        geo: sorted({target for row in group.itertuples(index=False) if float(row.gva) <= 0 for target in row.targets})
        for geo, group in gva_std.groupby("geo")
    }
    hours_std = _standardize_fte_source(
        hours_total,
        target_industries=target_set,
        value_name="hours",
        source_name="total-hours-by-industry",
    )
    national_hours, national_hours_metadata, national_hours_conflicts = _extract_fte_national_totals(
        hours_total,
        value_name="hours",
    )
    national_employment, national_employment_metadata, national_employment_conflicts = (
        _extract_fte_national_totals(
            employment_industry,
            value_name="employment",
        )
    )
    emp_hours = _mean_hours_employment_product(
        employment_industry,
        hours_mean,
        target_industries=target_set,
        allow_year_mismatch=True,
    )
    employment_std = _standardize_fte_source(
        employment_industry,
        target_industries=target_set,
        value_name="employment",
        source_name="employment-by-industry",
    )
    row_employment_std, row_employment_diagnostics = _build_row_a14_employment_residual(
        global_employment,
        row_employment_industry,
        explicit_geos=explicit_geos,
        target_industries=target_set,
        row_geo=row_geo,
    )

    records = []
    method_records = []
    for row in country_totals.itertuples(index=False):
        geo = str(row.geo)
        anchor = float(row.OBS_VALUE)
        employment_source_for_geo = (
            row_employment_std
            if geo == row_geo and not row_employment_std.empty
            else employment_std
        )
        weights, details = _resolve_fte40_country(
            geo,
            target_industries=target_industries,
            national_fte_anchor=anchor,
            gva_source=gva_std,
            hours_source=hours_std,
            emp_hours_source=emp_hours,
            employment_source=employment_source_for_geo,
            output_source=output_std,
            national_hours=national_hours,
            national_hours_metadata=national_hours_metadata,
            national_hours_conflicts=national_hours_conflicts,
            national_employment=national_employment,
            national_employment_metadata=national_employment_metadata,
            national_employment_conflicts=national_employment_conflicts,
            conflict_log=source_conflicts,
            strict_parent_conflicts=strict_source_conflicts,
            employment_coverage_audit=employment_coverage_audit,
        )
        methods = [details[target]["method"] for target in target_industries]
        country_method = "; ".join(
            f"{method}: {methods.count(method)}"
            for method in dict.fromkeys(methods)
        )
        for industry in target_industries:
            share = float(weights.get(industry, 0.0))
            detail = details[industry]
            records.append(
                {
                    "geo": geo,
                    "industry": industry,
                    "FTE_country_industry": anchor * share,
                    "allocation_method": detail["method"],
                    "allocation_method_key": detail["method"],
                    "allocation_share": share,
                    "FTE_country_total": anchor,
                    "constraint_source": detail["constraint_source"],
                    "allocation_weight_source": detail["allocation_weight_source"],
                    "allocation_operation": (
                        "Direct observation calibrated to national FTE40"
                        if detail["method"] in {"EU_H_DIRECT", "EMP_X_HOURS", "EMP_SHARE"}
                        else (
                            "Excluded from operational coefficient: no positive modeled output"
                            if detail["method"] == "ZERO_OUTPUT_UNRESOLVED"
                            else (
                                "National employment residual allocated using MRIO gross output"
                                if detail["method"] == "EMP_NATIONAL_RESIDUAL_OUTPUT"
                                else (
                                    "National residual allocated using MRIO gross output"
                                    if detail["method"] == "NATIONAL_RESIDUAL_OUTPUT"
                                    else (
                                        "Parent residual allocated using MRIO gross output"
                                        if detail["allocation_weight_source"] == "MRIO gross output"
                                        else "Parent residual allocated using employment x mean weekly hours"
                                    )
                                )
                            )
                        )
                    ),
                    "source_code": detail["source_code"],
                    "original_source_code": detail.get("original_source_code", pd.NA),
                    "source_observation": detail.get("original_source_observation", pd.NA),
                    "parent_source_observation": detail.get("parent_source_observation", pd.NA),
                    "parent_residual_observation": detail.get("parent_residual_observation", pd.NA),
                    "within_parent_weight": detail.get("within_parent_weight", pd.NA),
                    "estimated_child_labor_volume": detail.get("value", pd.NA),
                    "employment_observation": detail.get("employment_observation", pd.NA),
                    "mean_weekly_hours": detail.get("mean_weekly_hours", pd.NA),
                    "hours_source_code": detail.get("hours_source_code", pd.NA),
                    "hours_dataflow_id": detail.get("hours_dataflow_id", pd.NA),
                    "hours_source_year": detail.get("hours_source_year", pd.NA),
                    "hours_population_scope": detail.get("hours_population_scope", pd.NA),
                    "employment_hours_compatibility": detail.get("compatibility_status", pd.NA),
                    "original_class_system": detail.get("original_class_system", pd.NA),
                    "gva_allocation_block": detail.get("gva_allocation_block", pd.NA),
                    "gva_allocation_share": detail.get("gva_allocation_share", pd.NA),
                    "output_allocation_block": detail.get("output_allocation_block", pd.NA),
                    "output_allocation_share": detail.get("output_allocation_share", pd.NA),
                    "output_residual_status": detail.get("output_residual_status", pd.NA),
                    "allocation_raw_numerator": detail.get("allocation_raw_numerator", pd.NA),
                    "allocation_raw_denominator": detail.get("allocation_raw_denominator", pd.NA),
                    "allocation_raw_unit": detail.get("allocation_raw_unit", pd.NA),
                    "raw_fte40_estimate": detail.get("raw_fte40_estimate", pd.NA),
                    "pre_normalization_share": detail.get("pre_normalization_share", pd.NA),
                    "pre_normalization_fte": detail.get("pre_normalization_fte", pd.NA),
                    "national_normalization_factor": detail.get("national_normalization_factor", pd.NA),
                    "national_discrepancy_before_normalization": detail.get("national_discrepancy_before_normalization", pd.NA),
                    "national_discrepancy_status": detail.get("national_discrepancy_status", pd.NA),
                    "zero_output_industry": detail.get("zero_output_industry", False),
                    "output_status": detail.get("output_status", pd.NA),
                    "fte_output_coefficient_status": detail.get("fte_output_coefficient_status", pd.NA),
                    "source_db": detail["source_db"],
                    "source_year": detail["source_year"],
                    "dataflow_id": detail["dataflow_id"],
                    "key_used": detail["key_used"],
                    "population_scope": detail["population_scope"],
                    "source_coverage": detail["source_coverage"],
                    "national_hours_total": detail["national_hours_total"],
                    "observed_hours_total": detail["observed_hours_total"],
                    "remaining_hours_total": detail["remaining_hours_total"],
                    "national_hours_status": detail["national_hours_status"],
                    "national_hours_source_year": detail["national_hours_source_year"],
                    "national_employment_total": detail.get("national_employment_total", pd.NA),
                    "national_employment_source_year": detail.get("national_employment_source_year", pd.NA),
                    "employment_constraint_status": detail.get("employment_constraint_status", pd.NA),
                }
            )
        method_records.append(
            {
                "geo": geo,
                "FTE_country_total": anchor,
                "allocated_total": anchor * sum(weights.values()),
                "allocation_method": country_method,
            }
        )

    result = pd.DataFrame(records)
    audit = pd.DataFrame(method_records)
    # Preserve the underlying FIGARO GVA for every country-industry row,
    # independently of whether it was selected as an allocation weight.
    if not gva_std.empty:
        gva_rows = []
        for row in gva_std.itertuples(index=False):
            targets = row.targets if isinstance(row.targets, tuple) else (row.targets,)
            if len(targets) == 1:
                gva_rows.append(
                    {"geo": row.geo, "industry": targets[0], "FIGARO_GVA": row.gva}
                )
        if gva_rows:
            gva_values = (
                pd.DataFrame(gva_rows)
                .groupby(["geo", "industry"], as_index=False)["FIGARO_GVA"]
                .sum()
            )
            result = result.merge(gva_values, on=["geo", "industry"], how="left")
        else:
            result["FIGARO_GVA"] = pd.NA
    else:
        result["FIGARO_GVA"] = pd.NA

    if not output_std.empty:
        output_values = (
            output_std.groupby(["geo", "targets"], dropna=False)["output"]
            .sum()
            .reset_index()
        )
        output_rows = []
        for row in output_values.itertuples(index=False):
            targets = row.targets if isinstance(row.targets, tuple) else (row.targets,)
            if len(targets) == 1:
                output_rows.append({"geo": row.geo, "industry": targets[0], "FIGARO_output": row.output})
        if output_rows:
            result = result.merge(pd.DataFrame(output_rows), on=["geo", "industry"], how="left")
            result["FTE_intensity"] = (
                1000.0 * result["FTE_country_industry"]
                / result["FIGARO_output"].replace(0, np.nan)
            )
            zero_output = result["FIGARO_output"].eq(0)
            result.loc[zero_output, "FTE_intensity"] = 0.0
            result["zero_output_industry"] = zero_output
            result["fte_output_coefficient_status"] = np.where(
                zero_output, "undefined_zero_output", "defined"
            )

    result["allocation_block"] = result["output_allocation_block"].where(
        result["output_allocation_block"].notna(), result["gva_allocation_block"]
    )
    result["parent_output_total"] = pd.NA
    for geo, group in result.groupby("geo", sort=False):
        for block_key, block in group.groupby("output_allocation_block", dropna=True):
            targets = set(block_key) if isinstance(block_key, tuple) else set()
            denominator = group.loc[
                group["industry"].isin(targets), "FIGARO_output"
            ].sum(min_count=1)
            if pd.isna(denominator) or float(denominator) <= 0:
                raise ValueError(
                    f"Output denominator is non-positive for residual block {geo}:{block_key}."
                )
            index = block.index
            result.loc[index, "parent_output_total"] = denominator
            result.loc[index, "output_allocation_share"] = (
                result.loc[index, "FIGARO_output"] / denominator
            )
            result.loc[index, "allocation_raw_numerator"] = result.loc[index, "FIGARO_output"]
            result.loc[index, "allocation_raw_denominator"] = denominator
            result.loc[index, "allocation_raw_unit"] = "million EUR"

    result["GVA_used_as_weight"] = result["allocation_weight_source"].eq("FIGARO GVA")
    result["effective_allocation_weight"] = result["allocation_share"]
    result["weight_source"] = result["allocation_weight_source"]
    result["final_country_share"] = result["allocation_share"]
    result["constraint_level"] = np.select(
        [
            result["allocation_method"].isin({"EU_H_DIRECT", "EMP_X_HOURS", "EMP_SHARE"}),
            result["allocation_method"].isin({"EU_H_PARENT", "EMP_PARENT_OUTPUT"}),
            result["allocation_method"].eq("EMP_NATIONAL_RESIDUAL_OUTPUT"),
        ],
        ["individual industry", "parent/aggregate", "national employment residual"],
        default="residual",
    )
    result["observation_level"] = np.where(
        result["source_coverage"].astype(str).str.contains(";", na=False),
        "detailed mapped source or parent coverage",
        "exact FIGARO industry",
    )
    result["weight_role"] = np.select(
        [
            result["allocation_method"].isin({"EU_H_DIRECT", "EMP_X_HOURS", "EMP_SHARE"}),
            result["allocation_method"].isin({"EU_H_PARENT", "EMP_PARENT_OUTPUT"}),
            result["allocation_method"].eq("EMP_NATIONAL_RESIDUAL_OUTPUT"),
            result["allocation_method"].eq("NATIONAL_RESIDUAL_OUTPUT"),
            result["allocation_method"].eq("ZERO_OUTPUT_UNRESOLVED"),
        ],
        [
            "national allocation",
            "within-parent allocation",
            "national employment residual allocation",
            "national residual allocation",
            "zero-output excluded",
        ],
        default="residual allocation",
    )
    result["weight_coverage"] = np.where(
        result["allocation_block"].notna(), "unresolved subset", "complete or exact source"
    )
    result["gva_role"] = np.where(
        result["GVA_used_as_weight"],
        np.where(result["constraint_level"].eq("residual"), "unconstrained residual", "within-parent"),
        "unused",
    )
    _method = result["allocation_method"]
    _anchor = pd.to_numeric(result["FTE_country_total"], errors="coerce")
    _annual_total = pd.to_numeric(result["national_hours_total"], errors="coerce")
    _employment_total = pd.to_numeric(result["national_employment_total"], errors="coerce")
    _source_value = pd.to_numeric(result["source_observation"], errors="coerce")
    _parent_residual = pd.to_numeric(result["parent_residual_observation"], errors="coerce")
    result["stage1_raw_numerator"] = np.where(
        _method.eq("NATIONAL_RESIDUAL_OUTPUT"), _anchor,
        np.where(_method.eq("EMP_NATIONAL_RESIDUAL_OUTPUT"), _parent_residual, _source_value),
    )
    result["stage1_raw_denominator"] = np.select(
         [_method.isin({"EU_H_DIRECT", "EU_H_PARENT"}) & _annual_total.notna(),
          _method.isin({"EMP_X_HOURS", "EMP_SHARE", "EMP_PARENT_OUTPUT"}) & _employment_total.notna(),
          _method.eq("EMP_NATIONAL_RESIDUAL_OUTPUT") & _employment_total.notna(),
          _method.eq("NATIONAL_RESIDUAL_OUTPUT")],
         [_annual_total, _employment_total, _employment_total, _anchor],
        default=result["allocation_raw_denominator"],
    )
    result["stage1_weight_unit"] = np.select(
        [_method.isin({"EU_H_DIRECT", "EU_H_PARENT"}),
          _method.isin({"EMP_X_HOURS", "EMP_SHARE", "EMP_PARENT_OUTPUT"}),
          _method.eq("EMP_NATIONAL_RESIDUAL_OUTPUT"),
          _method.eq("NATIONAL_RESIDUAL_OUTPUT")],
        [
            "thousand hours/year",
            "thousand employed persons",
            "thousand employed persons",
            "thousand FTE",
        ],
        default="source-specific",
    )
    result["stage1_normalized_share"] = (
        pd.to_numeric(result["stage1_raw_numerator"], errors="coerce")
        / pd.to_numeric(result["stage1_raw_denominator"], errors="coerce").replace(0, np.nan)
    )
    result["stage1_resulting_fte"] = _anchor * result["stage1_normalized_share"]
    result["stage2_operation"] = np.where(
        result["allocation_block"].notna(),
        "residual child disaggregation using MRIO gross output",
        "not applicable",
    )
    result["stage2_allocation_block"] = result["allocation_block"]
    result["stage2_normalized_share"] = np.where(
        result["allocation_block"].notna(),
        result["output_allocation_share"],
        result["effective_allocation_weight"],
    )
    result["stage2_resulting_fte"] = np.where(
        result["allocation_block"].notna(), result["FTE_country_industry"], pd.NA
    )
    result["weight_unit"] = result["allocation_raw_unit"]
    result["parent_GVA_total"] = pd.NA
    result["GVA_allocation_share"] = result["gva_allocation_share"]
    result["effective_allocation_weight"] = (
        pd.to_numeric(result["allocation_raw_numerator"], errors="coerce")
        / pd.to_numeric(result["allocation_raw_denominator"], errors="coerce").replace(0, np.nan)
    )
    result["stage2_normalized_share"] = result["effective_allocation_weight"]

    check = result.groupby("geo", as_index=False)["FTE_country_industry"].sum()
    check = check.merge(country_totals, on="geo", suffixes=("_allocated", "_anchor"))
    check["reconciliation_error"] = check["FTE_country_industry"] - check["OBS_VALUE"]
    audit = audit.merge(check[["geo", "reconciliation_error"]], on="geo", how="left")
    method_map = {
        "EU_H_DIRECT": "observed_hours",
        "EU_H_PARENT": "eu_hours_parent",
        "EMP_X_HOURS": "employment_mean_hours",
        "EMP_SHARE": "employment_share",
        "EMP_PARENT_OUTPUT": "employment_parent_output",
        "EMP_NATIONAL_RESIDUAL_OUTPUT": "employment_national_residual_output",
        "NATIONAL_RESIDUAL_OUTPUT": "national_residual_output",
        "ZERO_OUTPUT_UNRESOLVED": "zero_output_unresolved",
    }
    summary_rows = []
    for geo, group in result.groupby("geo", sort=False):
        row = {"geo": geo}
        for key, label in method_map.items():
            subset = group.loc[group["allocation_method"].eq(key)]
            column = method_map[key]
            row[f"n_industries_{column}"] = int(len(subset))
            row[f"FTE_{column}"] = float(subset["FTE_country_industry"].sum())
        observed = group.loc[group["allocation_method"].isin({"EU_H_DIRECT", "EU_H_PARENT"})]
        row["n_industries_observed_hours"] = int(len(observed))
        row["FTE_observed_hours"] = float(observed["FTE_country_industry"].sum())
        row["share_observed_hours"] = (
            row["FTE_observed_hours"] / float(group["FTE_country_industry"].sum())
            if float(group["FTE_country_industry"].sum()) else np.nan
        )
        total = float(group["FTE_country_industry"].sum())
        for column in method_map.values():
            row[f"share_{column}"] = row[f"FTE_{column}"] / total if total else np.nan
        pre_fte = pd.to_numeric(group["pre_normalization_fte"], errors="coerce").sum(min_count=1)
        row["raw_industrial_FTE"] = float(pre_fte) if pd.notna(pre_fte) else np.nan
        anchor_value = float(group["FTE_country_total"].iloc[0])
        row["national_discrepancy"] = anchor_value - row["raw_industrial_FTE"] if pd.notna(row["raw_industrial_FTE"]) else np.nan
        normalization_values = pd.to_numeric(group["national_normalization_factor"], errors="coerce").dropna()
        row["national_normalization_factor"] = float(normalization_values.iloc[0]) if not normalization_values.empty else np.nan
        row["zero_output_industries"] = ";".join(
            group.loc[group["zero_output_industry"].fillna(False), "industry"].astype(str)
        )
        row["n_zero_output_industries"] = int(group["zero_output_industry"].fillna(False).sum())
        row["unresolved_positive_output_industries"] = ";".join(
            group.loc[
                group["allocation_method"].eq("NATIONAL_RESIDUAL_OUTPUT")
                & pd.to_numeric(group["FIGARO_output"], errors="coerce").gt(0),
                "industry",
            ].astype(str)
        )
        row["n_unresolved_positive_output_industries"] = int(
            (
                group["allocation_method"].eq("NATIONAL_RESIDUAL_OUTPUT")
                & pd.to_numeric(group["FIGARO_output"], errors="coerce").gt(0)
            ).sum()
        )
        discrepancy_status = group["national_discrepancy_status"].dropna().astype(str).unique().tolist()
        row["national_discrepancy_status"] = "; ".join(discrepancy_status)
        unresolved = group.loc[
            group["FTE_country_industry"].abs() <= float(tolerance), "industry"
        ].astype(str).tolist()
        row["unresolved_industries"] = ";".join(unresolved)
        row["n_unresolved_industries"] = len(unresolved)
        nonpositive_gva = gva_nonpositive_by_geo.get(geo, [])
        row["gva_nonpositive_industries"] = ";".join(nonpositive_gva)
        row["n_gva_nonpositive_industries"] = len(nonpositive_gva)
        row["negative_fte_rows"] = int((group["FTE_country_industry"] < -float(tolerance)).sum())
        status_values = group["national_hours_status"].dropna().astype(str).unique().tolist()
        row["national_hours_status"] = "; ".join(status_values)
        row["national_hours_total"] = group["national_hours_total"].dropna().iloc[0] if group["national_hours_total"].notna().any() else np.nan
        row["observed_hours_total"] = group["observed_hours_total"].dropna().iloc[0] if group["observed_hours_total"].notna().any() else np.nan
        row["remaining_hours_total"] = group["remaining_hours_total"].dropna().iloc[0] if group["remaining_hours_total"].notna().any() else np.nan
        row["national_hours_source_year"] = group["national_hours_source_year"].dropna().iloc[0] if group["national_hours_source_year"].notna().any() else np.nan
        row["national_employment_total"] = group["national_employment_total"].dropna().iloc[0] if group["national_employment_total"].notna().any() else np.nan
        row["national_employment_source_year"] = group["national_employment_source_year"].dropna().iloc[0] if group["national_employment_source_year"].notna().any() else np.nan
        employment_status = group["employment_constraint_status"].dropna().astype(str).unique().tolist()
        row["employment_constraint_status"] = "; ".join(employment_status)
        row["allocation_methods"] = "; ".join(
            f"{method}: {int((group['allocation_method'] == method).sum())}"
            for method in method_map
            if (group["allocation_method"] == method).any()
        )
        summary_rows.append(row)
    audit = audit.merge(pd.DataFrame(summary_rows), on="geo", how="left")
    unused_labor_audit = _unused_labor_information_audit(
        emp_hours,
        result,
        geos=set(country_totals["geo"].astype(str)),
    )
    audit = audit.merge(unused_labor_audit, on="geo", how="left")
    availability = pd.DataFrame()
    if source_availability_audit is not None and not source_availability_audit.empty:
        availability = source_availability_audit.copy()
        availability["geo"] = availability["geo"].astype(str).str.strip().str.upper()
        indicator_col = "indicator" if "indicator" in availability else "dataflow_id"
        code_count_col = (
            "usable_original_codes"
            if "usable_original_codes" in availability
            else "valid_source_observations"
        )
        availability_summary = (
            availability.groupby("geo", as_index=False)
            .agg(employment_source_candidates=(indicator_col, "size"))
        )
        selected_sources = (
            availability.loc[availability["selected"]]
            .groupby("geo")[indicator_col]
            .agg(lambda values: ";".join(sorted(values.astype(str).unique())))
            .rename("selected_employment_source")
            .reset_index()
        )
        discarded = (
            availability.loc[~availability["selected"]]
            .groupby("geo", as_index=False)[code_count_col]
            .sum()
            .rename(columns={code_count_col: "discarded_detailed_employment_codes"})
        )
        availability_summary = (
            availability_summary.merge(selected_sources, on="geo", how="left")
            .merge(discarded, on="geo", how="left")
        )
        audit = audit.merge(availability_summary, on="geo", how="left")
    conflict_df = pd.DataFrame(source_conflicts)
    if conflict_df.empty:
        conflict_summary = pd.DataFrame(columns=["geo", "source_conflict_count", "source_conflicts"])
    else:
        conflict_summary = (
            conflict_df.assign(
                description=lambda d: d["source"] + ": " + d["parent_targets"] +
                " residual=" + d["residual"].map(lambda value: f"{value:.6g}")
            )
            .groupby("geo", as_index=False)
            .agg(
                source_conflict_count=("description", "size"),
                source_conflicts=("description", "; ".join),
            )
        )
    audit = audit.merge(conflict_summary, on="geo", how="left")
    national_conflict_df = pd.DataFrame(national_hours_conflicts)
    if not national_conflict_df.empty:
        national_summary = (
            national_conflict_df.groupby("geo", as_index=False)
            .agg(
                national_hours_conflict_count=("rule", "size"),
                national_hours_conflicts=("rule", "; ".join),
            )
        )
        audit = audit.merge(national_summary, on="geo", how="left")
    else:
        audit["national_hours_conflict_count"] = 0
        audit["national_hours_conflicts"] = ""
    national_employment_conflict_df = pd.DataFrame(national_employment_conflicts)
    if not national_employment_conflict_df.empty:
        employment_summary = (
            national_employment_conflict_df.groupby("geo", as_index=False)
            .agg(
                national_employment_conflict_count=("rule", "size"),
                national_employment_conflicts=("rule", "; ".join),
            )
        )
        audit = audit.merge(employment_summary, on="geo", how="left")
    else:
        audit["national_employment_conflict_count"] = 0
        audit["national_employment_conflicts"] = ""
    audit["source_conflict_count"] = pd.to_numeric(
        audit["source_conflict_count"], errors="coerce"
    ).fillna(0).astype(int)
    audit["source_conflicts"] = audit["source_conflicts"].fillna("")
    audit["national_hours_conflict_count"] = pd.to_numeric(
        audit["national_hours_conflict_count"], errors="coerce"
    ).fillna(0).astype(int)
    audit["national_hours_conflicts"] = audit["national_hours_conflicts"].fillna("")
    audit["national_employment_conflict_count"] = pd.to_numeric(
        audit["national_employment_conflict_count"], errors="coerce"
    ).fillna(0).astype(int)
    audit["national_employment_conflicts"] = audit["national_employment_conflicts"].fillna("")
    audit["row_employment_residual_status"] = np.where(
        audit["geo"].eq(row_geo),
        row_employment_diagnostics["status"],
        pd.NA,
    )
    audit["row_employment_residual_groups"] = np.where(
        audit["geo"].eq(row_geo),
        row_employment_diagnostics["used_groups"],
        pd.NA,
    )
    audit["row_employment_residual_missing_groups"] = np.where(
        audit["geo"].eq(row_geo),
        row_employment_diagnostics["missing_groups"],
        pd.NA,
    )
    audit["row_employment_residual_reason"] = np.where(
        audit["geo"].eq(row_geo),
        row_employment_diagnostics["reason"],
        pd.NA,
    )
    audit["excluded_non_figaro_fte_anchors"] = ";".join(excluded_fte_anchors)
    audit["relative_reconciliation_error"] = audit["reconciliation_error"] / audit["FTE_country_total"].replace(0, np.nan)
    audit.attrs["source_availability_audit"] = availability
    coverage_df = pd.DataFrame(employment_coverage_audit)
    if coverage_df.empty:
        coverage_df = pd.DataFrame(
            columns=[
                "geo", "hours_source_code", "parent_targets",
                "employment_weight_source", "employment_coverage_status",
                "n_parent_children", "n_observed_employment_children",
                "n_missing_employment_children",
                "identifiable_employment_parent_residual",
                "employment_parent_source_code",
                "incomplete_employment_rejected",
            ]
        )
    else:
        coverage_summary = (
            coverage_df.assign(
                complete_employment_coverage=coverage_df["employment_coverage_status"].eq(
                    "complete_direct"
                ),
                incomplete_employment_coverage=~coverage_df["employment_coverage_status"].eq(
                    "complete_direct"
                ),
                identifiable_employment_parent_residual=coverage_df[
                    "employment_coverage_status"
                ].isin(
                    {
                        "identifiable_parent_residual",
                        "parent_residual_with_unresolved_children",
                    }
                ),
            )
            .groupby("geo", as_index=False)
            .agg(
                n_parent_blocks_complete_employment_coverage=(
                    "complete_employment_coverage", "sum"
                ),
                n_parent_blocks_incomplete_employment_coverage=(
                    "incomplete_employment_coverage", "sum"
                ),
                n_parent_blocks_identifiable_employment_parent_residual=(
                    "identifiable_employment_parent_residual", "sum"
                ),
                n_parent_blocks_incomplete_rejected_output=(
                    "incomplete_employment_rejected", "sum"
                ),
            )
        )
        audit = audit.merge(coverage_summary, on="geo", how="left")
    for column in (
        "n_parent_blocks_complete_employment_coverage",
        "n_parent_blocks_incomplete_employment_coverage",
        "n_parent_blocks_identifiable_employment_parent_residual",
        "n_parent_blocks_incomplete_rejected_output",
    ):
        if column not in audit:
            audit[column] = 0
        audit[column] = pd.to_numeric(audit[column], errors="coerce").fillna(0).astype(int)
    audit.attrs["employment_coverage_audit"] = coverage_df
    if (result["FTE_country_industry"] < -float(tolerance)).any():
        raise AssertionError("FTE40 allocation produced a negative country-industry value.")
    global_allocated = float(result["FTE_country_industry"].sum())
    row_pending = result["geo"].astype(str).str.upper().eq(str(row_geo).upper()) & result[
        "allocation_method"
    ].eq("ROW_PENDING_WORLD_A14")
    if row_pending.any():
        explicit_allocated = float(
            result.loc[~result["geo"].astype(str).str.upper().eq(str(row_geo).upper()), "FTE_country_industry"].sum()
        )
        if not np.isclose(explicit_allocated, explicit_total, rtol=0.0, atol=float(tolerance)):
            raise AssertionError(
                "Explicit-country FTE40 allocation failed reconciliation before ROW replacement: "
                f"allocated={explicit_allocated:.12g}, explicit_anchor={explicit_total:.12g}."
            )
    elif not np.isclose(global_allocated, world_total, rtol=0.0, atol=float(tolerance)):
        raise AssertionError(
            "FTE40 allocation failed the global reconciliation: "
            f"allocated={global_allocated:.12g}, world={world_total:.12g}."
        )
    if verbose:
        print(
            "FTE40 reconciliation: "
            f"explicit={explicit_total:.3f}, ROW={row_total:.3f}, "
            f"world={world_total:.3f}, allocated={global_allocated:.3f} "
            "(thousand FTE persons/year)"
        )
        print(
            audit.groupby("allocation_method", as_index=False)["FTE_country_total"]
            .sum()
            .to_string(index=False)
        )
        if source_conflicts:
            print("FTE40 source conflicts:")
            print(pd.DataFrame(source_conflicts).to_string(index=False))
    return result, audit


def build_world_fte40_blocks(
    fte40_totals: pd.DataFrame,
    world_employment: pd.DataFrame,
    *,
    target_industries: set[str] | list[str] | tuple[str, ...],
    world_geo: str = "X01",
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    """Allocate the existing World FTE40 anchor over observed A14 blocks.

    ``fte40_totals`` supplies the magnitude and is never reconstructed from
    employment or hours.  The World employment observations are used only as
    distribution weights.  The current FTE40 source has no industrial
    dimension, so the returned blocks are the complete non-overlapping A14
    partition selected from ``world_employment``.
    """
    if not {"geo", "OBS_VALUE"}.issubset(fte40_totals.columns):
        raise ValueError("fte40_totals must contain geo and OBS_VALUE columns.")
    target_set = {str(value).strip() for value in target_industries}
    anchor = fte40_totals.loc[
        fte40_totals["geo"].astype(str).str.strip().str.upper().eq(str(world_geo).upper())
    ].copy()
    anchor["OBS_VALUE"] = pd.to_numeric(anchor["OBS_VALUE"], errors="coerce")
    anchor = anchor.dropna(subset=["OBS_VALUE"])
    if len(anchor) != 1:
        raise ValueError("fte40_totals must contain exactly one World (X01) anchor.")
    world_total = float(anchor.iloc[0]["OBS_VALUE"])

    employment = _standardize_fte_source(
        world_employment,
        target_industries=target_set,
        value_name="employment",
        source_name="World industrial employment",
    ).loc[lambda frame: frame["geo"].eq("ROW")].copy()
    if employment.empty:
        raise ValueError("World employment has no usable industrial observations.")
    blocks = []
    for source_code, group in employment.groupby("source_code", sort=False):
        if group["targets"].map(tuple).nunique() != 1:
            raise ValueError(f"World employment block {source_code!r} has conflicting mappings.")
        targets = tuple(group.iloc[0]["targets"])
        value = float(group["employment"].sum())
        if value < -float(tolerance):
            raise ValueError(f"World employment block {source_code!r} is negative.")
        blocks.append(
            {
                "world_a14_block": str(source_code),
                "targets": targets,
                "world_employment": value,
                "source_year": group["source_year"].dropna().iloc[0] if group["source_year"].notna().any() else pd.NA,
                "dataflow_id": group["dataflow_id"].dropna().iloc[0] if group["dataflow_id"].notna().any() else pd.NA,
                "source_db": group["source_db"].dropna().iloc[0] if group["source_db"].notna().any() else pd.NA,
                "class_system": group["class_system"].dropna().iloc[0] if group["class_system"].notna().any() else pd.NA,
            }
        )
    result = pd.DataFrame(blocks)
    covered = set().union(*(set(values) for values in result["targets"]))
    if covered != target_set:
        raise ValueError(
            "World employment blocks do not cover the target industries: "
            f"missing={sorted(target_set - covered)}"
        )
    if len(covered) != sum(len(values) for values in result["targets"]):
        raise ValueError("World employment blocks overlap; cannot allocate without double counting.")
    employment_total = float(result["world_employment"].sum())
    if employment_total <= 0:
        raise ValueError("World employment weights have no positive total.")
    result["employment_weight"] = result["world_employment"] / employment_total
    result["FTE_world_block"] = world_total * result["employment_weight"]
    anchor_row = anchor.iloc[0]
    result["FTE40_anchor_dataflow"] = anchor_row.get("dataflow_id", pd.NA)
    result["FTE40_anchor_source_db"] = anchor_row.get("source_db", pd.NA)
    result["FTE40_anchor_year"] = anchor_row.get("TIME_PERIOD", pd.NA)
    result["FTE40_anchor_basis"] = anchor_row.get("JBF", ILO_FTE40_BASIS)
    if not np.isclose(result["FTE_world_block"].sum(), world_total, rtol=0.0, atol=float(tolerance)):
        raise AssertionError("World FTE40 block allocation failed to preserve the anchor total.")
    result.attrs["world_fte40_total"] = world_total
    result.attrs["world_fte40_resolution"] = "aggregate anchor allocated to non-overlapping ISIC4 A14 employment blocks"
    return result


def apply_world_a14_row_residual(
    fte_industry: pd.DataFrame,
    fte_audit: pd.DataFrame,
    world_fte_blocks: pd.DataFrame,
    *,
    target_industries: set[str] | list[str] | tuple[str, ...],
    world_geo: str = "X01",
    row_geo: str = "ROW",
    tolerance: float = 1e-9,
    allow_negative: bool = False,
    previous_row_fte: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Replace the ROW allocation with World-minus-FIGARO A14 residuals.

    World FTE40 is the mass anchor and the World industrial employment source
    supplies the non-overlapping A14 blocks.  Each block residual is split by
    ROW gross output weighted by median within-parent relative FTE/output
    intensities from eligible explicit-country observations.  Negative block
    residuals are rejected by default and may only be retained for an explicit
    diagnostic comparison with ``allow_negative=True``; they are never
    clipped or rebalanced.
    """
    required = {"geo", "industry", "FTE_country_industry"}
    if not required.issubset(fte_industry.columns):
        raise KeyError(f"fte_industry is missing required columns: {sorted(required - set(fte_industry.columns))}")
    block_required = {"world_a14_block", "targets", "FTE_world_block"}
    if not block_required.issubset(world_fte_blocks.columns):
        raise KeyError(
            "world_fte_blocks is missing required columns: "
            f"{sorted(block_required - set(world_fte_blocks.columns))}"
        )
    targets = [str(value).strip() for value in target_industries]
    targets = list(dict.fromkeys(targets))
    target_set = set(targets)

    explicit_rows = fte_industry.loc[
        ~fte_industry["geo"].astype(str).str.upper().isin({str(row_geo).upper(), str(world_geo).upper()})
        & fte_industry["industry"].astype(str).isin(target_set)
    ].copy()
    row_rows = fte_industry.loc[
        fte_industry["geo"].astype(str).str.upper().eq(str(row_geo).upper())
        & fte_industry["industry"].astype(str).isin(target_set)
    ].copy()
    if len(explicit_rows) != len(explicit_rows.drop_duplicates(["geo", "industry"])):
        raise ValueError("Explicit FTE40 input does not have unique country-industry rows.")
    if len(row_rows) != len(targets) or set(row_rows["industry"].astype(str)) != target_set:
        raise ValueError("ROW FTE40 input must contain one row for every target industry.")
    prior_row_rows = row_rows
    if previous_row_fte is not None and not previous_row_fte.empty:
        previous_required = {"geo", "industry", "FTE_country_industry"}
        if not previous_required.issubset(previous_row_fte.columns):
            raise KeyError(
                "previous_row_fte is missing required columns: "
                f"{sorted(previous_required - set(previous_row_fte.columns))}"
            )
        candidate = previous_row_fte.loc[
            previous_row_fte["geo"].astype(str).str.upper().eq(str(row_geo).upper())
            & previous_row_fte["industry"].astype(str).isin(target_set)
        ].copy()
        if (
            len(candidate) == len(targets)
            and len(candidate) == len(candidate.drop_duplicates("industry"))
            and set(candidate["industry"].astype(str)) == target_set
        ):
            prior_row_rows = candidate

    block_map: dict[str, set[str]] = {}
    world_values: dict[str, float] = {}
    for row in world_fte_blocks.itertuples(index=False):
        block = str(row.world_a14_block).strip()
        mapped = set(row.targets) & target_set
        if not mapped or block in {"TOTAL", "X"}:
            continue
        prior = block_map.get(block)
        if prior is not None and prior != mapped:
            raise ValueError(f"World employment source maps {block!r} to conflicting A64 targets.")
        block_map[block] = mapped
        world_values[block] = float(row.FTE_world_block)
    if not block_map:
        raise ValueError("World industrial employment source has no mapped A14 blocks.")
    covered = set().union(*block_map.values())
    if covered != target_set:
        raise ValueError(
            "World industrial employment source does not cover all target industries: "
            f"missing={sorted(target_set - covered)}"
        )
    target_blocks: dict[str, str] = {}
    for block, block_targets in block_map.items():
        for target in block_targets:
            if target in target_blocks and target_blocks[target] != block:
                raise ValueError(f"Target industry {target!r} belongs to multiple World A14 blocks.")
            target_blocks[target] = block

    explicit_rows["_fte"] = pd.to_numeric(
        explicit_rows["FTE_country_industry"], errors="coerce"
    )
    explicit_rows["_output"] = pd.to_numeric(
        explicit_rows.get("FIGARO_output", pd.Series(np.nan, index=explicit_rows.index)),
        errors="coerce",
    )
    explicit_rows["_block"] = explicit_rows["industry"].astype(str).map(target_blocks)
    if explicit_rows["_block"].isna().any():
        missing = sorted(explicit_rows.loc[explicit_rows["_block"].isna(), "industry"].astype(str).unique())
        raise ValueError(f"Explicit FTE40 rows lack a World A14 block mapping: {missing}")
    explicit_values = explicit_rows.groupby("industry", as_index=True)["_fte"].sum()

    # Parent intensity uses each country's complete final FTE/output block.
    # Donor eligibility is then applied to individual country-industries so
    # an output-imputed child cannot be treated as an observed-hours donor.
    parent_totals = (
        explicit_rows.groupby(["geo", "_block"], as_index=False)
        .agg(parent_fte=("_fte", "sum"), parent_output=("_output", "sum"))
    )
    donor_rows = explicit_rows.merge(parent_totals, on=["geo", "_block"], how="left")
    donor_method = donor_rows.get("allocation_method", pd.Series("", index=donor_rows.index)).astype(str)
    donor_weight = donor_rows.get(
        "allocation_weight_source", pd.Series("", index=donor_rows.index)
    ).astype(str).str.lower()
    eligible_method = donor_method.eq("EU_H_DIRECT")
    # ``EMP_X_HOURS`` is also used for a parent residual.  Its weight source
    # distinguishes a genuine employment-times-hours observation from a child
    # imputed by FIGARO output.
    eligible_method |= (
        donor_method.eq("EMP_X_HOURS")
        & ~donor_weight.str.contains("mrio gross output", regex=False, na=False)
    )
    eligible_method |= (
        donor_method.eq("EU_H_PARENT")
        & donor_weight.str.contains("mean weekly hours", regex=False, na=False)
        & ~donor_weight.str.contains("mrio gross output", regex=False, na=False)
    )
    donor_rows["_eligible_donor"] = (
        eligible_method
        & donor_rows["_output"].gt(0)
        & donor_rows["parent_output"].gt(0)
        & donor_rows["parent_fte"].gt(0)
    )
    donor_rows["_relative_intensity"] = np.nan
    valid_donors = donor_rows.loc[donor_rows["_eligible_donor"]].copy()
    valid_donors["_relative_intensity"] = (
        (valid_donors["_fte"] / valid_donors["_output"])
        / (valid_donors["parent_fte"] / valid_donors["parent_output"])
    )
    donor_summary = (
        valid_donors.groupby("industry", as_index=False)
        .agg(
            eligible_donor_countries=("geo", "nunique"),
            eligible_donor_observations=("geo", "size"),
            median_relative_intensity=("_relative_intensity", "median"),
        )
    )
    donor_lookup = donor_summary.set_index("industry")
    row_output = pd.to_numeric(
        row_rows.set_index("industry").get("FIGARO_output", pd.Series(dtype=float)).reindex(targets),
        errors="coerce",
    )
    if row_output.isna().any():
        missing = sorted(row_output.index[row_output.isna()].astype(str))
        raise ValueError(
            "ROW FIGARO gross output is required for the World A14 residual "
            f"disaggregation; missing={missing}."
        )

    residual_by_block: dict[str, float] = {}
    negative_blocks: list[tuple[str, float]] = []
    comparison_rows = []
    for block, block_targets in sorted(block_map.items()):
        world_total = float(world_values[block])
        explicit_total = float(explicit_values.reindex(sorted(block_targets)).fillna(0.0).sum())
        residual = world_total - explicit_total
        if residual < -float(tolerance):
            negative_blocks.append((block, residual))
        residual_by_block[block] = residual
        block_industries = sorted(block_targets)
        block_output = row_output.reindex(block_industries).fillna(0.0)
        block_q = pd.Series(
            {
                target: float(donor_lookup.at[target, "median_relative_intensity"])
                if target in donor_lookup.index
                and pd.notna(donor_lookup.at[target, "median_relative_intensity"])
                else 1.0
                for target in block_industries
            },
            dtype=float,
        )
        weighted_output = block_output * block_q.reindex(block_industries)
        denominator = float(weighted_output.sum())
        if abs(residual) > float(tolerance) and (not np.isfinite(denominator) or denominator <= 0):
            details = "; ".join(
                f"{target}: output={float(block_output.loc[target]):.12g}, "
                f"relative_intensity={float(block_q.loc[target]):.12g}"
                for target in block_industries
            )
            raise ValueError(
                "Cannot disaggregate nonzero ROW residual for "
                f"{block}: residual={residual:.12g}, weighted_output={denominator:.12g}; "
                + details
            )
        for target in block_industries:
            old_value = float(
                prior_row_rows.loc[
                    prior_row_rows["industry"].eq(target), "FTE_country_industry"
                ].iloc[0]
            )
            share = float(weighted_output.loc[target] / denominator) if denominator > 0 else 0.0
            new_value = residual * share
            comparison_rows.append(
                {
                    "industry": target,
                    "world_a14_block": block,
                    "old_row_fte": old_value,
                    "corrected_row_fte": new_value,
                    "change": new_value - old_value,
                    "world_block_fte": world_total,
                    "figaro49_block_fte": explicit_total,
                    "row_residual_block_fte": residual,
                    "row_output": float(block_output.loc[target]),
                    "median_relative_intensity": float(block_q.loc[target]),
                    "eligible_donor_countries": int(
                        donor_lookup.at[target, "eligible_donor_countries"]
                    ) if target in donor_lookup.index else 0,
                    "eligible_donor_observations": int(
                        donor_lookup.at[target, "eligible_donor_observations"]
                    ) if target in donor_lookup.index else 0,
                    "relative_intensity_fallback": target not in donor_lookup.index,
                    "row_output_relative_intensity_share": share,
                }
            )

    comparison = pd.DataFrame(comparison_rows).sort_values("industry").reset_index(drop=True)
    if set(comparison["industry"]) != target_set:
        raise AssertionError("World A14 residual comparison does not cover all target industries.")
    donor_coverage = comparison[
        [
            "industry", "world_a14_block", "eligible_donor_countries",
            "eligible_donor_observations", "median_relative_intensity",
            "relative_intensity_fallback",
        ]
    ].copy()
    row_a14_reconciliation = (
        comparison.groupby("world_a14_block", as_index=False)
        .agg(
            row_residual_fte=("row_residual_block_fte", "first"),
            allocated_row_fte=("corrected_row_fte", "sum"),
        )
    )
    row_a14_reconciliation["reconciliation_error"] = (
        row_a14_reconciliation["allocated_row_fte"]
        - row_a14_reconciliation["row_residual_fte"]
    )
    comparison.attrs["eligible_donor_coverage"] = donor_coverage
    comparison.attrs["industries_without_eligible_donors"] = donor_coverage.loc[
        donor_coverage["relative_intensity_fallback"], "industry"
    ].tolist()
    comparison.attrs["row_a14_reconciliation"] = row_a14_reconciliation
    comparison.attrs["row_total_before"] = float(
        pd.to_numeric(prior_row_rows["FTE_country_industry"], errors="coerce").sum()
    )
    comparison.attrs["row_total_after"] = float(comparison["corrected_row_fte"].sum())
    comparison.attrs["previous_row_allocation_available"] = prior_row_rows is not row_rows
    if negative_blocks and not allow_negative:
        details = "; ".join(
            f"{block}={residual:.12g}" for block, residual in negative_blocks
        )
        raise ValueError(
            "World-minus-FIGARO ROW residual has negative observed blocks; "
            "revised ROW coefficients were not applied: " + details
        )

    corrected = fte_industry.copy()
    corrected_index = corrected.index[
        corrected["geo"].astype(str).str.upper().eq(str(row_geo).upper())
        & corrected["industry"].astype(str).isin(target_set)
    ]
    new_by_industry = comparison.set_index("industry")["corrected_row_fte"]
    corrected.loc[corrected_index, "FTE_country_industry"] = corrected.loc[
        corrected_index, "industry"
    ].map(new_by_industry)
    corrected.loc[corrected_index, "allocation_method"] = "WORLD_A14_RESIDUAL_OUTPUT_RELATIVE_INTENSITY"
    corrected.loc[corrected_index, "allocation_method_key"] = "WORLD_A14_RESIDUAL_OUTPUT_RELATIVE_INTENSITY"
    corrected.loc[corrected_index, "constraint_source"] = "World FTE40/A14 employment residual"
    corrected.loc[corrected_index, "allocation_weight_source"] = "ROW gross output x median explicit-country relative FTE/output intensity"
    corrected.loc[corrected_index, "allocation_operation"] = "World A14 residual disaggregated with ROW gross output x median relative intensity"
    corrected.loc[corrected_index, "source_db"] = "ILOSTAT World FTE40/A14 residual"
    corrected.loc[corrected_index, "population_scope"] = "World FTE40 anchor and industrial employment"
    corrected.loc[corrected_index, "source_coverage"] = corrected.loc[
        corrected_index, "industry"
    ].map(comparison.set_index("industry")["world_a14_block"])
    corrected.loc[corrected_index, "allocation_share"] = corrected.loc[
        corrected_index, "FTE_country_industry"
    ] / pd.to_numeric(corrected.loc[corrected_index, "FTE_country_total"], errors="coerce")
    for column in ("allocation_share", "final_country_share", "effective_allocation_weight"):
        if column in corrected:
            corrected.loc[corrected_index, column] = corrected.loc[corrected_index, "allocation_share"]
    if "FTE_intensity" in corrected and "FIGARO_output" in corrected:
        output = pd.to_numeric(corrected.loc[corrected_index, "FIGARO_output"], errors="coerce")
        corrected.loc[corrected_index, "FTE_intensity"] = (
            1000.0 * corrected.loc[corrected_index, "FTE_country_industry"] / output.replace(0, np.nan)
        )
        corrected.loc[corrected_index[output.eq(0).to_numpy()], "FTE_intensity"] = 0.0

    updated_audit = fte_audit.copy()
    audit_mask = updated_audit["geo"].astype(str).str.upper().eq(str(row_geo).upper())
    if audit_mask.any():
        row_total = float(corrected.loc[corrected["geo"].astype(str).str.upper().eq(str(row_geo).upper()), "FTE_country_industry"].sum())
        updated_audit.loc[audit_mask, "allocated_total"] = row_total
        updated_audit.loc[audit_mask, "reconciliation_error"] = 0.0
        updated_audit.loc[audit_mask, "allocation_method"] = (
            f"WORLD_A14_RESIDUAL_OUTPUT_RELATIVE_INTENSITY: {len(targets)}"
        )
        updated_audit.loc[audit_mask, "n_industries_gva_fallback"] = 0
        updated_audit.loc[audit_mask, "FTE_gva_fallback"] = 0.0
        updated_audit.loc[audit_mask, "n_industries_world_a14_residual_output_relative_intensity"] = len(targets)
        updated_audit.loc[audit_mask, "FTE_world_a14_residual_output_relative_intensity"] = row_total
        updated_audit.loc[audit_mask, "world_a14_residual_status"] = (
            "complete_with_negative_blocks" if negative_blocks else "complete"
        )
        updated_audit.loc[audit_mask, "world_a14_negative_blocks"] = ";".join(
            block for block, _residual in negative_blocks
        )
        updated_audit.loc[audit_mask, "row_employment_residual_status"] = "not used: World A14 residual"

    final_row_values = pd.to_numeric(
        corrected.loc[
            corrected["geo"].astype(str).str.upper().eq(str(row_geo).upper()),
            "FTE_country_industry",
        ],
        errors="coerce",
    )
    final_explicit_values = pd.to_numeric(
        corrected.loc[
            ~corrected["geo"].astype(str).str.upper().isin(
                {str(row_geo).upper(), str(world_geo).upper()}
            ),
            "FTE_country_industry",
        ],
        errors="coerce",
    )
    final_world_total = float(pd.to_numeric(world_fte_blocks["FTE_world_block"], errors="coerce").sum())
    if final_row_values.lt(-float(tolerance)).any() or final_explicit_values.lt(-float(tolerance)).any():
        raise AssertionError("World-minus-explicit ROW replacement produced negative FTE values.")
    if not np.isclose(
        float(final_row_values.sum()) + float(final_explicit_values.sum()),
        final_world_total,
        rtol=0.0,
        atol=float(tolerance),
    ):
        raise AssertionError(
            "World-minus-explicit ROW replacement failed global reconciliation: "
            f"allocated={float(final_row_values.sum()) + float(final_explicit_values.sum()):.12g}, "
            f"world={final_world_total:.12g}."
        )

    return corrected, updated_audit, comparison





def assess_industrial_resolution_inventory(
    frame: pd.DataFrame,
    *,
    target_industries: set[str],
    source_db: str | None = None,
) -> pd.DataFrame:
    """Assess standardized Eurostat or ILOSTAT industrial observations."""
    target_set = {
        _clean_industry_code(code)
        for code in target_industries
        if _clean_industry_code(code) not in {"", "TOTAL", "X"}
    }
    missing_countries = sorted({
        "GR" if str(country).upper() == "EL" else str(country)
        for country in (frame.attrs.get("missing_countries", []) if frame is not None else [])
    })
    is_eurostat = (
        str(source_db or "").upper() == "EUROSTAT"
        or (
            source_db is None
            and frame is not None
            and not frame.empty
            and frame["source_db"].astype("string").str.upper().eq("EUROSTAT").all()
        )
    )
    columns = [
        "Country", "Year", "Classification", "Source profile",
        "Valid source industries", "Resolution",
    ]
    rows = []
    if frame is not None and not frame.empty:
        missing = set(_INDUSTRIAL_SOURCE_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(
                f"Industrial source is not standardized; missing {sorted(missing)}."
            )
        for key, group in frame.groupby(
            ["geo", "source_db", "dataflow_id", "year", "class_system"],
            dropna=False,
            sort=False,
        ):
            country, source_db, indicator, source_year, class_system = key
            country = "GR" if str(country).upper() == "EL" else str(country)
            valid = group.loc[
                group["OBS_VALUE"].notna()
                & ~group["source_code"].astype("string").str.upper().isin({"", "TOTAL", "X", "NAN"})
            ].drop_duplicates("source_code")
            valid_codes = int(valid["source_code"].nunique())
            metadata, _mapped = _effective_resolution_metadata(
                group,
                target_industries=target_set,
                value_name="OBS_VALUE",
                source_name=str(source_db),
            )
            rows.append({
                "Country": country,
                "Year": int(source_year) if pd.notna(source_year) else pd.NA,
                "Classification": class_system,
                "Source profile": (
                    pd.NA if is_eurostat else metadata["named_source_profile"]
                ),
                "Valid source industries": valid_codes,
                "Resolution": metadata["effective_resolution"],
            })

    for country in missing_countries:
        rows.append({
            "Country": country,
            "Year": pd.NA,
            "Classification": pd.NA,
            "Source profile": "unavailable: no compatible observations",
            "Valid source industries": 0,
            "Resolution": 0,
        })

    inventory = pd.DataFrame(rows, columns=columns)
    if not inventory.empty:
        inventory = inventory.sort_values(
            ["Country", "Year"],
            ascending=[True, False],
        ).reset_index(drop=True)
    inventory.attrs["missing_countries"] = missing_countries
    return inventory
