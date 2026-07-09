"""Employment and wage extension for the MRIO impact model."""

from __future__ import annotations

from pathlib import Path
import json
import pickle
import hashlib
import random
import re
import time
from io import StringIO
from functools import lru_cache

import numpy as np
import pandas as pd
import requests
import pycountry
import xml.etree.ElementTree as ET
from pandas.errors import EmptyDataError, ParserError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Bbox

from modules.config import HELPERS_DIR

from modules.country import COLOR_MAP
from modules.latex_style import (
    apply_house_style,
    save_and_close,
    set_house_style,
    tex_safe,
    get_text_color,
)

ind_to_prod = pd.read_excel(str(HELPERS_DIR / "ind_to_prd.xlsx"), sheet_name="mapping")
class_prd_ava = pd.read_excel(str(HELPERS_DIR / "class_prd_ava.xlsx"), sheet_name="parent_code")
class_ind_ava = pd.read_excel(str(HELPERS_DIR / "class_ind_ava.xlsx"), sheet_name="parent_code")
try:
    aggregate_coverage = pd.read_excel(str(HELPERS_DIR / "aggregate_coverage.xlsx"))
except FileNotFoundError:
    aggregate_coverage = pd.DataFrame(columns=["source_code", "covers_ind_ava_codes"])

if "coverage_profile" in aggregate_coverage.columns:
    aggregate_coverage["coverage_profile"] = (
        aggregate_coverage["coverage_profile"]
        .astype(str)
        .str.strip()
    )

CLASSIFICATION_CROSSWALK_DIR = HELPERS_DIR / "classification_crosswalks"
ISIC4_L2_TO_NACE_REV2_A64_PATH = CLASSIFICATION_CROSSWALK_DIR / "isic4_l2_to_nace_rev2_a64.csv"

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

    m = re.fullmatch(r"([A-Z]+)(\d{2})(?:[T_](\d{2}))?", s)
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

def _official_isic_coverage(code: str, class_system: str | None, mrio_prd_set, *, ind_ind: bool) -> set[str]:
    system = str(class_system or "").upper().replace(" ", "")
    if system != "ISIC4":
        return set()
    source = _clean_industry_code(code)
    if source in _aggregate_source_codes_raw() or source == "TOTAL":
        return set()

    nace_targets = _isic4_l2_to_nace_rev2_a64_map().get(source, set())
    if nace_targets:
        return _normalize_crosswalk_targets(nace_targets, tuple(map(str, mrio_prd_set)), ind_ind=ind_ind)

    # ISIC4 section-level rows such as A, B, C are handled directly from
    # their section/division coverage. No downloaded correspondence tables
    # are used here.
    div_to_target = _target_by_nace2_division(tuple(map(str, mrio_prd_set)), ind_ind=ind_ind)
    divs = _industry_code_divisions(source)
    return {div_to_target[d] for d in divs if d in div_to_target}

def _split_coverage_string(x) -> set[str]:
    if pd.isna(x) or str(x).strip() == "":
        return set()
    return {p.strip() for p in re.split(r"[;,]", str(x)) if p.strip()}

def _split_coverage_codes(x) -> list[str]:
    if pd.isna(x):
        return []
    return [p.strip() for p in re.split(r"[;,]", str(x)) if p and p.strip()]

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
        cpa = _add_cpa_prefix(raw)
        return canon_cpa.get(cpa, cpa) if canon_cpa is not None else cpa

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
            if e["profile_sources"].issubset(observed_codes)
        }
        if matching_profiles:
            best_size = max(len(v) for v in matching_profiles.values())
            best_profiles = {p for p, v in matching_profiles.items() if len(v) == best_size}
            entries = [e for e in entries if not e.get("profile") or e.get("profile") in best_profiles]
        else:
            entries = [e for e in entries if not e.get("profile")]

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

class EurostatUnavailable(RuntimeError):
    pass

def _chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def _session_with_retries():
    global _SESSION
    if _SESSION is not None: return _SESSION
    s=requests.Session()
    r=Retry(total=6,connect=6,read=6,status=6,backoff_factor=1.0,
            status_forcelist=(429,500,502,503,504,524),
            allowed_methods=frozenset(["GET"]),raise_on_status=False)
    a=HTTPAdapter(max_retries=r)
    s.mount("https://",a); s.mount("http://",a)
    _SESSION=s; return s

def _get_csv(url,params=None,timeout=180,retries=2,backoff=1.5,raise_on_non_csv=False):
    last_preview = ""
    for i in range(max(1, int(retries))):
        r=_session_with_retries().get(url,params=params,timeout=timeout)
        if r.status_code==524: return pd.DataFrame()
        r.raise_for_status()
        t=(r.text or "").strip()
        if not t: return pd.DataFrame()

        content_type = (r.headers.get("content-type") or "").lower()
        first = t[:200].lstrip().lower()
        is_html = (
            "text/html" in content_type
            or r.url.lower().startswith("https://sorry.ec.europa.eu")
            or first.startswith("<!doctype html")
            or first.startswith("<html")
        )
        if is_html:
            last_preview = " | ".join(t.splitlines()[:6])
            if i < max(1, int(retries)) - 1:
                time.sleep(backoff * (i + 1))
                continue
            msg = f"[Eurostat] non-CSV response after retries; returning empty data. Preview: {last_preview[:300]}"
            if raise_on_non_csv:
                raise EurostatUnavailable(msg)
            print(msg)
            return pd.DataFrame()

        try:
            return pd.read_csv(StringIO(t))
        except EmptyDataError:
            return pd.DataFrame()
        except ParserError:
            last_preview = " | ".join(t.splitlines()[:6])
            try:
                df = pd.read_csv(StringIO(t), engine="python", on_bad_lines="skip")
            except (EmptyDataError, ParserError):
                print(f"[Eurostat] could not parse CSV response; returning empty data. Preview: {last_preview[:300]}")
                return pd.DataFrame()
            expected = {"geo", "nace_r2", "TIME_PERIOD", "OBS_VALUE"}
            if expected.isdisjoint(set(map(str, df.columns))):
                print(f"[Eurostat] parsed response has no expected columns; returning empty data. Preview: {last_preview[:300]}")
                return pd.DataFrame()
            return df

    return pd.DataFrame()

def _pick_col(df,c):
    for x in c:
        if x in df.columns: return x
    raise KeyError(f"Missing columns {c}. Found={list(df.columns)}")

def _safe_div(n,d):
    return (n/d.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).fillna(0.0)

def _safe_mode(x,default="NA"):
    s=pd.Series(x).dropna()
    if s.empty: return default
    s=s.astype(str); m=s.mode()
    return m.iat[0] if len(m) else s.iloc[0]

def _groupby_apply_no_groupcols(grouped, func):
    try:
        # pandas >= 2.1
        return grouped.apply(func, include_groups=False)
    except TypeError:
        return grouped.apply(func)

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

def _prd_to_parent_letter(p):
    m=re.match(r"^CPA_([A-Z])",str(p).strip())
    return m.group(1) if m else np.nan

def _keep_latest(df,max_year_allowed=2023):
    if df.empty: return df
    d=df.copy()
    d["TIME_PERIOD"]=pd.to_numeric(d["TIME_PERIOD"],errors="coerce")
    d=d[d["TIME_PERIOD"].notna() & (d["TIME_PERIOD"]<=max_year_allowed)]
    if d.empty: return d
    y=d.groupby(["geo","nace_r2"])["TIME_PERIOD"].transform("max")
    return d[d["TIME_PERIOD"]==y]

def _mode_or_first(x,default="NA"):
    s=pd.Series(x).dropna()
    if s.empty: return default
    s=s.astype(str); m=s.mode()
    return m.iat[0] if len(m) else s.iloc[0]


def cpa_to_mrio_prd(cpa: str) -> str | None:
    if not isinstance(cpa, str) or not cpa:
        return None

    # direct hits
    if cpa in ind_to_prod:
        return cpa

    # roll-ups you need
    if cpa in {"CPA_C10","CPA_C11","CPA_C12"}: return "CPA_C10T12"
    if cpa in {"CPA_C13","CPA_C14","CPA_C15"}: return "CPA_C13T15"
    if cpa in {"CPA_C31","CPA_C32"}:          return "CPA_C31_32"
    if cpa in {"CPA_J59","CPA_J60"}:          return "CPA_J59_60"
    if cpa in {"CPA_J62","CPA_J63"}:          return "CPA_J62_63"
    if cpa in {"CPA_M69","CPA_M70"}:          return "CPA_M69_70"
    if cpa in {"CPA_M74","CPA_M75"}:          return "CPA_M74_75"
    if cpa in {"CPA_N80","CPA_N81","CPA_N82"}:return "CPA_N80T82"
    if cpa in {"CPA_Q87","CPA_Q88"}:          return "CPA_Q87_88"
    if cpa in {"CPA_R90","CPA_R91","CPA_R92"}:return "CPA_R90T92"
    if cpa in {"CPA_E37","CPA_E38","CPA_E39"}:return "CPA_E37T39"

    # coarsen where MRIO is coarser:
    if cpa.startswith("CPA_B"): return "CPA_B"
    if cpa.startswith("CPA_T"): return "CPA_T"
    if cpa.startswith("CPA_U"): return "CPA_U"

    return None

# ======================================================
# Eurostat EMP (latest-year picking)
# ======================================================

import pandas as pd

def load_emp_eurostat(CORE, EU_OFC, min_nace=70, years=range(2023, 2010, -1),
                      geo_list=None, require_unit="THS_PER", na_item="EMP_DC",
                      geo_chunk=8):
    DATAFLOW="nama_10_a64_e"
    BASE3="https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT"
    GEO_LIST = geo_list or [
        "EU27_2020","EA20","EA19","EA12",
        "BE","BG","CZ","DK","DE","EE","IE","EL","ES","FR","HR",
        "IT","CY","LV","LT","LU","HU","MT","NL","AT","PL","PT","RO","SI","SK","FI","SE",
        "IS","LI","NO","CH","UK","ME","MK","RS","XK"
    ]

    PARAMS_BASE = (
        "c[freq]=A"
        f"&c[unit]={require_unit}"
        "&c[nace_r2]=TOTAL,A,A01,A02,A03,B-E,B,C,C10-C12,C13-C15,C16-C18,C16,C17,C18,"
        "C19,C20,C21,C22_C23,C22,C23,C24_C25,C24,C25,C26,C27,C28,C29_C30,C29,C30,"
        "C31-C33,C31_C32,C33,D,D35,E,E36,E37-E39,F,G-I,G,G45,G46,G47,H,H49,H50,H51,H52,H53,"
        "I,J,J58-J60,J58,J59_J60,J61,J62_J63,K,K64,K65,K66,L,L68,L68A,M_N,M,M69-M71,M69_M70,M71,"
        "M72,M73-M75,M73,M74_M75,N,N77,N78,N79,N80-N82,O-Q,O,O84,P,P85,Q,Q86,Q87_Q88,"
        "R-U,R,R90-R92,R93,S,S94,S95,S96,T,U,U99"
        f"&c[na_item]={na_item}"
        "&compress=false&format=csvdata&formatVersion=1.0&lang=en"
    )

    def dl(year, geos):
        pending=list(geos)
        while pending:
            url=f"{BASE3}/{DATAFLOW}/1.0/*.*.*.*.*?{PARAMS_BASE}&c[geo]={','.join(pending)}&c[TIME_PERIOD]={year}"
            try:
                df=_get_csv(url, raise_on_non_csv=True)
                break
            except EurostatUnavailable as exc:
                print(str(exc))
                return None
            except requests.HTTPError as exc:
                body = exc.response.text if exc.response is not None else ""
                invalid_geo = sorted(set(re.findall(r"GEO=([A-Z0-9_]+)", body)))
                invalid_geo = [g for g in invalid_geo if g in pending]
                if not invalid_geo:
                    raise
                print(f"Eurostat rejected GEO code(s) {invalid_geo}; retrying without them.")
                pending=[g for g in pending if g not in invalid_geo]
        else:
            return pd.DataFrame()
        if df.empty: return df
        df=df.assign(
            geo=df["geo"].astype(str).str.strip(),
            nace_r2=df["nace_r2"].astype(str).str.strip(),
            TIME_PERIOD=df["TIME_PERIOD"].astype(str).str.strip(),
            OBS_VALUE=pd.to_numeric(df["OBS_VALUE"], errors="coerce"),
            source_db="Eurostat",
            dataflow_id=DATAFLOW,
            key_used=f"freq=A;unit={require_unit};na_item={na_item};year={year}",
        )
        return df

    accepted=[]; remaining=set(GEO_LIST)
    for y in years:
        if not remaining: break
        parts = []
        unavailable = False
        chunk_size = int(geo_chunk) if geo_chunk else len(remaining)
        for geos_chunk in _chunk(sorted(remaining), max(1, chunk_size)):
            d = dl(y, geos_chunk)
            if d is None:
                unavailable = True
                break
            if not d.empty:
                parts.append(d)
        if unavailable:
            break
        if not parts:
            continue
        d = pd.concat(parts, ignore_index=True)
        if d.empty: continue
        d_non_na = d[d["nace_r2"].ne("TOTAL") & d["OBS_VALUE"].notna()]
        cov = d_non_na.groupby("geo")["nace_r2"].nunique()
        ok=cov[cov>=min_nace].index.tolist()
        if ok:
            accepted.append(d[d["geo"].isin(ok)])
            remaining-=set(ok)

    if not accepted:
        return pd.DataFrame(columns=["geo","nace_r2","TIME_PERIOD","OBS_VALUE","source_db","dataflow_id","key_used"])

    emp=pd.concat(accepted, ignore_index=True)
    emp=emp[emp["nace_r2"].ne("TOTAL")].copy()

    # map NACE->CPA (replace nace_r2, keep unmapped log)
    m=(pd.read_excel(str(HELPERS_DIR/"ind_to_prd.xlsx"), sheet_name="mapping")
         .loc[:,["ind_ava_code","prd_ava_code"]]
         .assign(ind_ava_code=lambda d: d["ind_ava_code"].astype(str).str.strip(),
                 prd_ava_code=lambda d: d["prd_ava_code"].astype(str).str.strip())
         .set_index("ind_ava_code")["prd_ava_code"].to_dict())

    emp["nace_r2_raw"]=emp["nace_r2"]
    emp["nace_r2"]=emp["nace_r2_raw"].map(m)

    # Deduplicate exact clones created by NACE?CPA many-to-one mapping


    bad=emp[emp["nace_r2"].isna() & emp["nace_r2_raw"].notna()]
    if not bad.empty:
        print("UNMAPPED nace_r2 codes:")
        print(bad["nace_r2_raw"].value_counts().head(30).to_string())

    emp['geo'] = emp['geo'].replace({'UK': 'GB'})
    emp = emp.loc[emp['geo'].isin(set(CORE) | set(EU_OFC))].copy()

    # Deduplicate exact clones created by NACE?CPA many-to-one mapping
    emp = (
        emp.sort_values("OBS_VALUE", ascending=False)
        .drop_duplicates(subset=["geo", "TIME_PERIOD", "nace_r2"], keep="first")
    )
    emp = emp.sort_values(by=["geo", "nace_r2"], ascending=[True, True])

    return emp[["geo","nace_r2","TIME_PERIOD","OBS_VALUE","source_db","dataflow_id","key_used"]].copy()


def load_emp_eurostat_lfsa_egan22d(
    EU27,
    *,
    years=None,
    geo_list=None,
    require_unit="THS_PER",
    require_sex=None,
    require_age=None,
    freq="A",
    geo_chunk=4,
    year_chunk=5,
):
    """
    Download lfsa_egan22d (employment by sex, age, detailed economic activity)
    for EU27 geos into a single large DataFrame. Keeps industry codes as-is.
    """
    DATAFLOW = "lfsa_egan22d"
    BASE_STATS = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"

    geos = geo_list or sorted({str(x).strip() for x in EU27})

    def _jsonstat_to_long(js: dict) -> pd.DataFrame:
        ids = js.get("id", [])
        sizes = js.get("size", [])
        dim = js.get("dimension", {})
        if not ids or not sizes:
            return pd.DataFrame()

        code_lists = []
        for dim_id in ids:
            cat = dim.get(dim_id, {}).get("category", {})
            idx = cat.get("index", {})
            if isinstance(idx, dict):
                codes = sorted(idx.keys(), key=lambda k: idx[k])
            else:
                codes = list(idx)
            code_lists.append([str(c) for c in codes])

        total = int(np.prod(sizes))
        values = js.get("value", [])
        if isinstance(values, dict):
            arr = np.full(total, np.nan)
            for k, v in values.items():
                try:
                    arr[int(k)] = v
                except (ValueError, TypeError):
                    continue
            values = arr
        else:
            values = np.array(values, dtype=float) if values else np.array([])
            if len(values) < total:
                pad = np.full(total, np.nan)
                pad[: len(values)] = values
                values = pad

        idx = pd.MultiIndex.from_product(code_lists, names=ids)
        out = pd.DataFrame({"OBS_VALUE": values}, index=idx).reset_index()
        return out

    def _dl(geos_local, years_local=None):
        params = {"format": "JSON", "lang": "EN"}
        params["freq"] = freq
        if require_unit:
            params["unit"] = require_unit
        if require_age:
            params["age"] = require_age
        if require_sex:
            params["sex"] = require_sex
        params["geo"] = list(geos_local)
        if years_local:
            params["time"] = [str(y) for y in years_local]

        url = f"{BASE_STATS}/{DATAFLOW}"
        r = _session_with_retries().get(url, params=params, timeout=180)
        r.raise_for_status()
        js = r.json()
        if "error" in js:
            raise ValueError(js.get("error"))
        if "warning" in js and str(js["warning"].get("status")) == "413":
            raise ValueError("API returned asynchronous warning (413). Reduce chunk sizes.")

        df = _jsonstat_to_long(js)
        if df.empty:
            return df

        for c in ["freq", "unit", "age", "sex", "nace_r2", "geo", "time"]:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()
        df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")

        df["source_db"] = "Eurostat"
        df["dataflow_id"] = DATAFLOW
        key_bits = [f"freq={freq}"]
        if require_unit:
            key_bits.append(f"unit={require_unit}")
        if require_age:
            key_bits.append(f"age={require_age}")
        if require_sex:
            key_bits.append(f"sex={require_sex}")
        if years_local:
            key_bits.append(f"years={','.join(map(str, years_local))}")
        df["key_used"] = ";".join(key_bits)
        return df

    def _probe_years(sample_geo: str) -> list[str]:
        # Lightweight probe to list available years (use TOTAL/T to minimize payload)
        params = {
            "format": "JSON",
            "lang": "EN",
            "freq": freq,
            "unit": require_unit or "THS_PER",
            "age": "TOTAL",
            "sex": "T",
            "nace_r2": "TOTAL",
            "geo": sample_geo,
        }
        url = f"{BASE_STATS}/{DATAFLOW}"
        r = _session_with_retries().get(url, params=params, timeout=180)
        r.raise_for_status()
        js = r.json()
        dim = js.get("dimension", {})
        time_dim = dim.get("time", {}).get("category", {}).get("index", {})
        if isinstance(time_dim, dict):
            years_local = sorted(time_dim.keys(), key=lambda k: time_dim[k])
        else:
            years_local = list(time_dim)
        return [str(y) for y in years_local]

    def _chunks(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i : i + n]

    if years is None:
        # default to latest available year to keep downloads fast
        years_list = ["2024"]
    else:
        years_list = [str(y) for y in list(years)]

    pieces = []
    for gchunk in _chunks(geos, max(1, int(geo_chunk))):
        for ychunk in _chunks(years_list, max(1, int(year_chunk))):
            d = _dl(gchunk, ychunk)
            if not d.empty:
                pieces.append(d)

    if not pieces:
        return pd.DataFrame(
            columns=[
                "freq", "unit", "age", "sex", "nace_r2", "geo",
                "TIME_PERIOD", "OBS_VALUE", "source_db", "dataflow_id", "key_used"
            ]
        )

    emp = pd.concat(pieces, ignore_index=True)
    if "nace_r2" in emp.columns:
        emp = emp[emp["nace_r2"].ne("TOTAL")].copy()
    sort_cols = [c for c in ["geo", "nace_r2", "age", "sex", "time"] if c in emp.columns]
    if sort_cols:
        emp = emp.sort_values(by=sort_cols)
    return emp


def emp_total_by_geo_nace(
    emp: pd.DataFrame,
    *,
    year: str,
    geos=None,
    age_total="AUTO",
    sex_total="T",
    fill_sex_total=True,
    nace_filter=None,
):
    """
    Build totals by (geo, nace_r2) for a given year using age_total and sex_total.
    If sex_total is missing and fill_sex_total=True, uses F+M (same age_total).
    """
    df = emp.copy()
    df["time"] = df["time"].astype(str)
    df["sex"] = df["sex"].astype(str)
    df["age"] = df["age"].astype(str)
    df["geo"] = df["geo"].astype(str)
    df["nace_r2"] = df["nace_r2"].astype(str)

    df = df[df["time"] == str(year)].copy()
    if geos is not None:
        geo_set = {str(g) for g in geos}
        df = df[df["geo"].isin(geo_set)].copy()
    if nace_filter is not None:
        nace_set = {str(n) for n in nace_filter}
        df = df[df["nace_r2"].isin(nace_set)].copy()

    if df.empty:
        return pd.DataFrame(columns=["geo", "nace_r2", "OBS_VALUE"])

    # pick age_total automatically if requested
    if str(age_total).upper() == "AUTO":
        ages = sorted(df["age"].unique().tolist())
        if "TOTAL" in ages:
            age_total = "TOTAL"
        else:
            # choose age with max coverage after sex aggregation
            best_age, best_cov = None, -1
            for a in ages:
                dfa = df[df["age"] == a]
                piv = dfa.pivot_table(
                    index=["geo", "nace_r2"],
                    columns="sex",
                    values="OBS_VALUE",
                    aggfunc="first",
                )
                if piv.empty:
                    continue
                if sex_total in piv.columns:
                    total = piv[sex_total]
                else:
                    total = pd.Series(index=piv.index, dtype=float)
                    if fill_sex_total and "F" in piv.columns and "M" in piv.columns:
                        total = (piv["F"] + piv["M"])
                    elif not piv.empty:
                        total = piv[piv.notna().sum().idxmax()]
                cov = int(total.notna().sum())
                if cov > best_cov:
                    best_cov = cov
                    best_age = a
            age_total = best_age if best_age is not None else ages[0]

    df = df[df["age"] == str(age_total)].copy()
    if df.empty:
        return pd.DataFrame(columns=["geo", "nace_r2", "OBS_VALUE"])

    piv = df.pivot_table(
        index=["geo", "nace_r2"],
        columns="sex",
        values="OBS_VALUE",
        aggfunc="first",
    )

    if sex_total in piv.columns:
        total = piv[sex_total].copy()
    else:
        total = pd.Series(index=piv.index, dtype=float)
        if fill_sex_total and "F" in piv.columns and "M" in piv.columns:
            total = (piv["F"] + piv["M"])
        elif not piv.empty:
            total = piv[piv.notna().sum().idxmax()]

    out = total.to_frame("OBS_VALUE").reset_index()
    out["OBS_VALUE"] = pd.to_numeric(out["OBS_VALUE"], errors="coerce")
    return out


def emp_complete_geos(
    emp: pd.DataFrame,
    *,
    year: str,
    geos=None,
    drop_nace=("NRP", "UNK"),
    min_geo_share=0.9,
    age_total="AUTO",
    sex_total="T",
    fill_sex_total=True,
    nace_filter=None,
):
    """
    Return (complete_geos, expected_nace, coverage_by_nace) for given year.
    expected_nace = codes with non-NA coverage >= min_geo_share.
    """
    base = emp_total_by_geo_nace(
        emp,
        year=year,
        geos=geos,
        age_total=age_total,
        sex_total=sex_total,
        fill_sex_total=fill_sex_total,
        nace_filter=nace_filter,
    )
    if base.empty:
        return [], [], pd.Series(dtype=float)

    if drop_nace:
        drop_set = {str(x) for x in drop_nace}
        base = base[~base["nace_r2"].isin(drop_set)].copy()

    pivot = base.pivot_table(index="geo", columns="nace_r2", values="OBS_VALUE", aggfunc="first")
    coverage = pivot.notna().mean(axis=0).sort_values(ascending=False)

    if float(min_geo_share) <= 0.0:
        expected_nace = coverage[coverage > 0.0].index.tolist()
    else:
        expected_nace = coverage[coverage >= float(min_geo_share)].index.tolist()
    if not expected_nace:
        return [], [], coverage

    complete = pivot[expected_nace].notna().all(axis=1)
    complete_geos = complete[complete].index.tolist()
    return complete_geos, expected_nace, coverage
# ======================================================

# ---------- small net helpers ----------
def _get_xml(url: str, *, params=None, timeout=180, retries=6, backoff=1.8) -> ET.Element:
    """
    Retry SDMX requests on transient failures (503 etc.) with exponential backoff + jitter.
    Minimal change: same signature as before, with extra optional args.
    """
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers={"Accept": "application/xml"}  # helps some SDMX endpoints
            )
            # retry on transient HTTP codes
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"Transient HTTP {r.status_code}", response=r)

            r.raise_for_status()
            return ET.fromstring(r.content)

        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            last_err = e
            code = getattr(getattr(e, "response", None), "status_code", None)
            if i == retries - 1:
                raise
            sleep_s = (backoff ** i) + random.uniform(0, 0.5)
            print(f"[ILO SDMX] transient error {code} on try {i+1}/{retries} ? sleep {sleep_s:.2f}s")
            time.sleep(sleep_s)

    raise last_err

SDMX_BASE = "https://sdmx.ilo.org/rest"

NS = {
    "mes": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/message",
    "str": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure",
    "com": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common",
}

# Default preferred dataflows for ILO data
PREFERRED_DATAFLOWS = [
    "EMP_TEMP_SEX_EC2_A",
    "SAL_TYPE_OCU_NB_A",
    "EMP_2EMP_SEX_AGE_A",
]

def _dataflow_cache_path(cache_path: str | Path | None = None) -> Path:
    if cache_path is None:
        return Path(HELPERS_DIR) / "cache" / "ilo_dataflows.json"
    return Path(cache_path)

def _load_dataflow_cache(path: Path, *, ttl_days: float | None = 30.0, allow_stale: bool = True) -> dict | None:
    try:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        names = payload.get("names")
        if not isinstance(names, dict):
            return None
        fetched_at = float(payload.get("fetched_at", 0.0))
        if ttl_days is None or float(ttl_days) <= 0:
            return names
        age_days = (time.time() - fetched_at) / 86400.0 if fetched_at > 0 else None
        if age_days is None:
            return names if allow_stale else None
        if age_days <= float(ttl_days):
            return names
        return names if allow_stale else None
    except Exception:
        return None

def _save_dataflow_cache(path: Path, names: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"fetched_at": time.time(), "names": names}
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)
    except Exception:
        pass


def show_preferred_dataflows(
    flows=None,
    *,
    timeout: int = 30,
    retries: int = 2,
    backoff: float = 1.3,
    raise_on_error: bool = False,
    use_cache: bool = True,
    cache_path: str | Path | None = None,
    cache_ttl_days: float | None = 30.0,
    refresh_cache: bool = False,
    allow_stale_cache: bool = True,
):
    """
    Print preferred ILO SDMX dataflows and their names.
    Defaults are tuned to fail fast to avoid long waits on 5xx/524.
    """
    if flows is None:
        flows = PREFERRED_DATAFLOWS

    cache_file = _dataflow_cache_path(cache_path)
    names = None
    if use_cache and not refresh_cache:
        names = _load_dataflow_cache(cache_file, ttl_days=cache_ttl_days, allow_stale=allow_stale_cache)

    if names is None:
        try:
            root = _get_xml(
                f"{SDMX_BASE}/dataflow",
                timeout=timeout,
                retries=retries,
                backoff=backoff,
            )
            names = {d.get("id"): d.findtext(".//com:Name", "", namespaces=NS) for d in root.findall(".//str:Dataflow", NS)}
            if use_cache:
                _save_dataflow_cache(cache_file, names)
        except Exception as e:
            if raise_on_error:
                raise
            print(f"[ILO SDMX] failed to fetch dataflows ({type(e).__name__}: {e})")
            if use_cache:
                names = _load_dataflow_cache(cache_file, ttl_days=None, allow_stale=True)
            if names is None:
                names = {}
    print("\n=== Dataflows selected ===")
    [print(f"{f} -> {names.get(f,'NOT FOUND')}") for f in flows]

# ======================================================
# ILO SDMX
# ======================================================

# ---------- CL_AREA + ISO3 list ----------
def fetch_cl_area_iso3():
    root=_get_xml(f"{SDMX_BASE}/codelist/ILO/CL_AREA/1.0")
    return {c.get("id") for c in root.findall(".//str:Codelist[@id='CL_AREA']//str:Code",NS)}

def iso2_to_iso3(code2: str) -> str | None:
    c = pycountry.countries.get(alpha_2=str(code2).strip().upper())
    return c.alpha_3 if c else None

def eurostat_core_coverage(emp_eu: pd.DataFrame, core_iso2: set[str], *, min_nace=50):
    if emp_eu.empty:
        return set(), pd.Series(dtype=int)
    df = emp_eu.assign(
        geo=emp_eu["geo"].astype(str).str.strip().replace({"UK": "GB"}),
        nace_r2=emp_eu["nace_r2"].astype(str).str.strip(),
    )
    cov = df[df["nace_r2"].ne("TOTAL")].groupby("geo")["nace_r2"].nunique().sort_values(ascending=False)
    covered = set(cov[(cov >= min_nace) & cov.index.isin(core_iso2)].index)
    return covered, cov

def reduce_ilo_core_using_eurostat(emp_eu: pd.DataFrame, CORE: set[str], *, min_nace=50):
    core_iso2 = {c for c in CORE if c != "W2"}
    covered, cov = eurostat_core_coverage(emp_eu, core_iso2, min_nace=min_nace)
    ilo_needed_iso2 = sorted(core_iso2 - covered)

    ISO3_LIST, missing = [], []
    for c2 in ilo_needed_iso2:
        if c2.startswith(("X","_")):   # no-ISO aggregates: keep as-is (or mark missing)
            missing.append(c2); continue
        c3 = iso2_to_iso3(c2)
        (ISO3_LIST if c3 else missing).append(c3 if c3 else c2)

    return {"ISO3_LIST_REDUCED": ISO3_LIST, "missing_iso2": missing, "eurostat_cov": cov, "covered_by_eurostat_iso2": sorted(covered)}

# ---------- SDMX download (GenericData) ----------
def get_dimension_order(df_id, agency="ILO", version="1.0", timeout=180):
    root = _get_xml(f"{SDMX_BASE}/datastructure/{agency}/{df_id}/{version}", timeout=timeout)
    return [d.get("id") for d in root.findall(".//{*}DimensionList/{*}Dimension")]

def build_key_ref_area_and_freq(dims, iso3_list, freq="A"):
    iso = "+".join(iso3_list)
    return ".".join(iso if d=="REF_AREA" else (freq if d=="FREQ" else "") for d in dims)

def download_genericdata_to_df(df_id, year, iso3_list, agency="ILO", version="1.0", freq="A", timeout=180, chunk_size=4):
    root=_get_xml(f"{SDMX_BASE}/datastructure/{agency}/{df_id}/{version}", timeout=timeout)
    dims=[d.get("id") for d in root.findall(".//{*}DimensionList/{*}Dimension")]
    key_base=".".join(("+".join(iso3_list) if d=="REF_AREA" else (freq if d=="FREQ" else "")) for d in dims)

    data_url = f"{SDMX_BASE}/data/{agency},{df_id},{version}/{key_base}"
    params = {"startPeriod":str(year), "endPeriod":str(year), "dimensionAtObservation":"AllDimensions"}

    def _parse_genericdata(root_xml):
        rows=[]
        for obs in root_xml.findall(".//{*}Obs"):
            row={}
            for v in obs.findall("./{*}ObsKey/{*}Value"):
                k=v.get("concept") or v.get("id")
                val=v.get("value")
                if k and val is not None:
                    row[k]=val
            if "TIME_PERIOD" not in row:
                od=obs.find("./{*}ObsDimension")
                if od is not None and od.get("value"):
                    row["TIME_PERIOD"]=od.get("value")
            ov=obs.find("./{*}ObsValue")
            row["OBS_VALUE"]=ov.get("value") if ov is not None else None
            rows.append(row)
        return rows

    # --- try full request first ---
    try:
        root_data=_get_xml(data_url, params=params, timeout=timeout)
        df = pd.DataFrame(_parse_genericdata(root_data))
        return df, key_base

    except requests.HTTPError as e:
        code = getattr(getattr(e, "response", None), "status_code", None)
        if code not in (429, 500, 502, 503, 504):
            raise

    # --- fallback: chunk REF_AREA to smaller requests ---
    all_rows = []
    key_used = None

    for sub in _chunk(iso3_list, chunk_size):
        key_sub=".".join(("+".join(sub) if d=="REF_AREA" else (freq if d=="FREQ" else "")) for d in dims)
        url_sub = f"{SDMX_BASE}/data/{agency},{df_id},{version}/{key_sub}"
        root_sub = _get_xml(url_sub, params=params, timeout=timeout)  # uses retry
        all_rows.extend(_parse_genericdata(root_sub))
        key_used = key_sub  # last one (fine for logging; or store list if you want)
        time.sleep(0.2)     # politeness delay

    return pd.DataFrame(all_rows), f"CHUNKED({chunk_size})::{key_used}"


# ---------- deductive pull (keeps your FIX1/FIX2 + log) ----------
def _ilo_class_priority(df: pd.DataFrame) -> int:
    vals = []
    for col in ("EC2", "ECO"):
        if col in df.columns:
            vals.extend(df[col].dropna().astype(str).str.upper().tolist())
    text = " ".join(vals)
    if "ISIC4" in text:
        return 3
    if "ISIC3" in text:
        return 2
    if "AGGREGATE" in text or "SECTOR" in text:
        return 1
    return 0

def pull_flows_deductive(flows, iso3_list, *, year, freq="A", prefer_isic4_over_isic3=True):
    remaining=list(iso3_list); out=[]; log=[]
    candidates = {}
    for order, df_id in enumerate(flows, 1):
        if not remaining: break
        requested=list(remaining); reqset=set(requested)
        df,key_used=download_genericdata_to_df(df_id, year, requested, freq=freq)
        if not df.empty and "REF_AREA" in df: df=df[df["REF_AREA"].isin(reqset)].copy()  # FIX 1
        got=sorted(set(df["REF_AREA"].unique())) if ("REF_AREA" in df and not df.empty) else []
        chosen_now = []
        if not df.empty:
            df["source_db"]="ILOSTAT"; df["dataflow_id"]=df_id; df["key_used"]=key_used
            if prefer_isic4_over_isic3 and "REF_AREA" in df.columns:
                for area, g in df.groupby("REF_AREA", dropna=False):
                    area = str(area)
                    priority = _ilo_class_priority(g)
                    old = candidates.get(area)
                    if old is None or priority > old["priority"] or (priority == old["priority"] and order < old["order"]):
                        candidates[area] = {
                            "priority": priority,
                            "order": order,
                            "dataflow_id": df_id,
                            "df": g.copy(),
                        }
                        chosen_now.append(area)
            else:
                out.append(df)
                chosen_now = got
        if prefer_isic4_over_isic3:
            if order < len(flows):
                # Keep countries available for later flows unless an ISIC4
                # candidate has already been found. This lets an aggregate
                # ISIC4 flow replace a detailed ISIC3 flow for the same country.
                ready = {area for area, candidate in candidates.items() if candidate["priority"] >= 3}
                remaining=[c for c in remaining if c not in ready]
            else:
                remaining=[c for c in remaining if c not in candidates]
        else:
            remaining=[c for c in remaining if c not in set(got)]
        log.append({
            "order":order,
            "dataflow_id":df_id,
            "requested":len(requested),
            "got":len(got),
            "chosen_or_replaced":len(chosen_now),
            "remaining":len(remaining),
            "rows":len(df),
        })
    if prefer_isic4_over_isic3:
        selected = [
            candidate["df"].assign(selected_dataflow_id=candidate["dataflow_id"])
            for _, candidate in sorted(candidates.items())
        ]
        out = selected
        remaining = [c for c in iso3_list if c not in candidates]
    return (pd.concat(out, ignore_index=True) if out else pd.DataFrame(),
            pd.DataFrame(log),
            remaining)


def process_ilo_long(df, source_db="ILOSTAT_SDMX", drop_x=False):
    df=df.copy()
    df["OBS_VALUE"]=pd.to_numeric(df["OBS_VALUE"],errors="coerce").fillna(0.0)

    n=pd.Series(pd.NA,index=df.index,dtype="string")
    industry_col=pd.Series(pd.NA,index=df.index,dtype="string")
    class_system=pd.Series(pd.NA,index=df.index,dtype="string")
    for col,prio in (("EC2",True),("ECO",False)):
        if col in df:
            raw=df[col].astype("string")
            c=raw.str.replace(r"^(EC2|ECO)_(ISIC3|ISIC4)_","",regex=True).where(lambda x:x.notna())
            sys=raw.str.extract(r"^(?:EC2|ECO)_(ISIC3|ISIC4)_", expand=False)
            eco_agg=raw.str.extract(r"^ECO_(AGGREGATE|SECTOR|AGNAG)_", expand=False)
            sys=sys.mask(sys.isna() & eco_agg.notna(), eco_agg)
            take = c.notna() if prio else (n.isna() & c.notna())
            n=n.mask(take,c)
            industry_col=industry_col.mask(take,col)
            class_system=class_system.mask(take,sys)

    df=df.assign(
        nace_r2=n,
        employment_source_code=n,
        industry_col=industry_col,
        class_system=class_system,
    ).loc[lambda d:d["nace_r2"].notna()]

    s=df["nace_r2"].astype("string")
    agg = s.str.extract(r"^ECO_(?:AGGREGATE|SECTOR|AGNAG)_(.+)$", expand=False)
    s = s.mask(agg.notna(), agg)
    df["class_system"] = df["class_system"].mask(agg.notna(), "AGGREGATE")
    is_tot=s.str.contains("TOTAL",na=False)
    is_blk=s.str.match(r"^ECO_(AGGREGATE|AGNAG|SECTOR)_",na=False)&~is_tot
    is_det=s.str.startswith("ECO_DETAILS_",na=False)

    df["nace_r2"]=(s.mask(is_tot,"TOTAL")
                    .mask(is_blk,pd.NA)
                    .mask(is_det,s.str.replace(r"^ECO_DETAILS_","",regex=True))
                    .replace({"U99": "U"}))
    df["employment_source_code"] = (
        df["employment_source_code"].astype("string")
        .str.replace(r"^(EC2|ECO)_(ISIC3|ISIC4)_", "", regex=True)
        .str.replace(r"^ECO_(AGGREGATE|SECTOR|AGNAG)_", "", regex=True)
        .replace({"U99": "U"})
    )
    df=df.loc[df["nace_r2"].notna()]
    if drop_x: df=df.loc[df["nace_r2"].ne("X")]

    # TOTAL: no agregar ? dejar uno solo
    K=["REF_AREA","TIME_PERIOD","nace_r2","employment_source_code","industry_col","class_system","dataflow_id","key_used"]
    df=pd.concat([
        df.loc[~df["nace_r2"].eq("TOTAL")],
        df.loc[df["nace_r2"].eq("TOTAL")].drop_duplicates(K)
    ],ignore_index=True)

    if "SEX" in df:
        sex=df["SEX"].astype("string")
        t=df.loc[sex.eq("SEX_T")].groupby(K,dropna=False,as_index=False)["OBS_VALUE"].sum()
        nt=df.loc[~sex.eq("SEX_T")].groupby(K,dropna=False,as_index=False)["OBS_VALUE"].sum()
        if not t.empty and not nt.empty:
            nt=nt[~pd.MultiIndex.from_frame(nt[K]).isin(pd.MultiIndex.from_frame(t[K]))]
        df=pd.concat([t,nt],ignore_index=True)
    else:
        df=df.groupby(K,dropna=False,as_index=False)["OBS_VALUE"].sum()

    df["geo"]=df["REF_AREA"].map(lambda c3: pycountry.countries.get(alpha_3=str(c3).strip().upper()).alpha_2 if pycountry.countries.get(alpha_3=str(c3).strip().upper()) else None)
    df["source_db"]=source_db

    return df[["geo","nace_r2","employment_source_code","industry_col","class_system","TIME_PERIOD","OBS_VALUE","source_db","dataflow_id","key_used"]].copy()


def build_ISO3_LIST(emp_eu, CORE, min_nace=50):
    areas=fetch_cl_area_iso3()
    df=emp_eu.assign(geo=emp_eu["geo"].astype(str).str.strip().replace({"UK":"GB"}),
                     nace_r2=emp_eu["nace_r2"].astype(str).str.strip())
    cov=df[df["nace_r2"].ne("TOTAL")].groupby("geo")["nace_r2"].nunique()
    core=(CORE-{"W2"}); covered=set(cov[(cov>=min_nace) & cov.index.isin(core)].index)
    iso3=[]; missing=[]
    for c2 in sorted(core-covered):
        if c2.startswith(("X","_")): (iso3 if c2 in areas else missing).append(c2); continue
        c3=iso2_to_iso3(c2)
        (iso3 if (c3 and c3 in areas) else missing).append(c3 if (c3 and c3 in areas) else c2)
    return iso3, missing

def map_nace_to_cpa_inplace(
    df: pd.DataFrame,
    mrio_prd_set: pd.DataFrame,
    *,
    nace_col="nace_r2",
    ind_ind: bool | None = None,
) -> pd.DataFrame:
    ind_ind = _is_industry_matrix(mrio_prd_set, ind_ind)

    if ind_ind:
        out = df.copy()
        raw = f"{nace_col}_raw"
        source_col = "employment_source_code"
        coverage_col = "covers_ind_ava_codes"
        if coverage_col not in out.columns:
            out[coverage_col] = pd.NA
        if source_col not in out.columns:
            out[source_col] = out[nace_col]
        out[nace_col] = out[nace_col].astype(str).str.strip()
        out[source_col] = out[source_col].astype(str).str.strip().replace({"U99": "U"})
        out[raw] = out[nace_col]

        # Work in CPA_* space (for parent codes), then strip back at the end
        out[nace_col] = out[nace_col].map(_add_cpa_prefix)

        raw_np = out[raw].astype(str).str.replace(r"^CPA_", "", regex=True)
        is_agg_raw = raw_np.str.match(r"^[A-Z]$|^[A-Z]_[A-Z]$|^[A-Z]-[A-Z]$|^[A-Z]{2,3}$", na=False)
        stop = is_agg_raw & out[nace_col].astype(str).str.startswith("CPA_", na=False)
        official_has_coverage = pd.Series(False, index=out.index)

        if "class_system" in out.columns:
            for idx, row in out.iterrows():
                covers = _official_isic_coverage(
                    raw_np.loc[idx],
                    row.get("class_system"),
                    tuple(map(str, mrio_prd_set)),
                    ind_ind=True,
                )
                if not covers:
                    continue
                official_has_coverage.loc[idx] = True
                covers_plain = sorted({_strip_cpa_prefix(c) for c in covers})
                out.at[idx, coverage_col] = ";".join(covers_plain)
                if len(covers_plain) == 1:
                    out.at[idx, nace_col] = _add_cpa_prefix(covers_plain[0])
                else:
                    out.at[idx, nace_col] = _add_cpa_prefix(raw_np.loc[idx])
                    stop.loc[idx] = True

        allowed = {_add_cpa_prefix(c) for c in mrio_prd_set} | {"CPA_TOTAL"}
        canon = _canon_map(mrio_prd_set)
        canon_cpa = {_add_cpa_prefix(k): _add_cpa_prefix(v) for k, v in canon.items()}

        parent = class_prd_ava.set_index("prd_ava_code")["parent_code"].to_dict()
        parent = {canon_cpa.get(k, k): canon_cpa.get(v, v) for k, v in parent.items() if pd.notna(v)}

        s = out[nace_col].astype(str).map(lambda x: canon_cpa.get(x, x))
        liftable = s.str.startswith("CPA_", na=False) & ~stop
        for _ in range(12):
            mask = liftable & ~s.isin(allowed) & s.isin(parent)
            if not mask.any():
                break
            s = s.where(~mask, s.map(parent).fillna(s))
        out[nace_col] = s

        out[nace_col] = out[nace_col].map(_strip_cpa_prefix)
        def join_cov(s):
            vals = sorted(set().union(*(_split_coverage_string(x) for x in s)))
            return ";".join(vals) if vals else pd.NA
        def join_codes(s):
            vals = sorted({
                str(x).strip()
                for x in s
                if pd.notna(x) and str(x).strip() and str(x).strip().lower() != "nan"
            })
            return ";".join(vals) if vals else pd.NA
        gcols = ["geo", "TIME_PERIOD", "source_db", "dataflow_id", "key_used", nace_col]
        opt_cols = [c for c in ["industry_col", "class_system"] if c in out.columns]
        return (
            out.groupby(gcols + opt_cols, as_index=False, dropna=False)
            .agg(
                OBS_VALUE=("OBS_VALUE", "sum"),
                covers_ind_ava_codes=(coverage_col, join_cov),
                employment_source_code=(source_col, join_codes),
            )
        )

    m = (pd.read_excel(str(HELPERS_DIR/"ind_to_prd.xlsx"), sheet_name="mapping")
           .loc[:, ["ind_ava_code","prd_ava_code"]]
           .assign(ind_ava_code=lambda d: d["ind_ava_code"].astype(str).str.strip(),
                   prd_ava_code=lambda d: d["prd_ava_code"].astype(str).str.strip())
           .set_index("ind_ava_code")["prd_ava_code"].to_dict())

    out = df.copy()
    raw = f"{nace_col}_raw"
    source_col = "employment_source_code"
    coverage_col = "covers_ind_ava_codes"
    if coverage_col not in out.columns:
        out[coverage_col] = pd.NA
    if source_col not in out.columns:
        out[source_col] = out[nace_col]
    out[nace_col] = out[nace_col].astype(str).str.strip()
    out[source_col] = out[source_col].astype(str).str.strip().replace({"U99": "U"})
    out[raw] = out[nace_col]
    source_codes = _aggregate_source_codes_raw()
    is_eco_dataflow = out.get("dataflow_id", pd.Series("", index=out.index)).astype(str).str.contains("ECO", case=False, na=False)
    preserve_aggregate_source = is_eco_dataflow & out[raw].isin(source_codes)

    # Case: TOTAL
    out.loc[out[raw].eq("TOTAL"), nace_col] = "CPA_TOTAL"

    # Case: aggregated labels (preserve ILO aggregate source codes; CPA-replace ordinary detail codes)
    is_agg_raw = out[raw].str.match(r"^[A-Z]$|^[A-Z]_[A-Z]$|^[A-Z]-[A-Z]$|^[A-Z]{2,3}$", na=False)
    mapped = out[raw].map(m)
    is_mapped_cpa = mapped.astype(str).str.startswith("CPA_", na=False)

    base = out[nace_col].copy()
    base = base.where(~(out[raw].ne("TOTAL") & is_mapped_cpa & ~preserve_aggregate_source), mapped)
    out[nace_col] = base

    stop = preserve_aggregate_source | (is_agg_raw & out[nace_col].astype(str).str.startswith("CPA_", na=False))
    official_has_coverage = pd.Series(False, index=out.index)

    if "class_system" in out.columns:
        for idx, row in out.iterrows():
            if bool(preserve_aggregate_source.loc[idx]):
                continue
            covers = _official_isic_coverage(
                row[raw],
                row.get("class_system"),
                tuple(map(str, mrio_prd_set)),
                ind_ind=False,
            )
            if not covers:
                continue
            official_has_coverage.loc[idx] = True
            covers_out = sorted(covers)
            out.at[idx, coverage_col] = ";".join(covers_out)
            if len(covers_out) == 1:
                out.at[idx, nace_col] = covers_out[0]
            else:
                out.at[idx, nace_col] = row[raw]
                stop.loc[idx] = True

    # Case: well-classified (lift CPA codes only; never lift aggregates-stop)
    allowed = set(mrio_prd_set) | {"CPA_TOTAL"}
    canon = {c.replace("T","-"): c for c in mrio_prd_set if "T" in c}
    parent = class_prd_ava.set_index("prd_ava_code")["parent_code"].to_dict()
    parent = {k: canon.get(v, v) for k, v in parent.items() if pd.notna(v)}

    s = out[nace_col].astype(str).map(lambda x: canon.get(x, x))
    liftable = s.str.startswith("CPA_", na=False) & ~stop
    for _ in range(12):
        mask = liftable & ~s.isin(allowed) & s.isin(parent)
        if not mask.any(): break
        s = s.where(~mask, s.map(parent).fillna(s))
    out[nace_col] = s

    def join_cov(s):
        vals = sorted(set().union(*(_split_coverage_string(x) for x in s)))
        return ";".join(vals) if vals else pd.NA
    def join_codes(s):
        vals = sorted({
            str(x).strip()
            for x in s
            if pd.notna(x) and str(x).strip() and str(x).strip().lower() != "nan"
        })
        return ";".join(vals) if vals else pd.NA
    gcols = ["geo","TIME_PERIOD","source_db","dataflow_id","key_used",nace_col]
    opt_cols = [c for c in ["industry_col", "class_system"] if c in out.columns]
    return (
        out.groupby(gcols + opt_cols, as_index=False, dropna=False)
        .agg(
            OBS_VALUE=("OBS_VALUE", "sum"),
            covers_ind_ava_codes=(coverage_col, join_cov),
            employment_source_code=(source_col, join_codes),
        )
    )


import pandas as pd

def fit_emp_eu_to_mrio(
    emp_eu: pd.DataFrame,
    mrio_prd_set,
    *,
    nace_col="nace_r2",
    ind_ind: bool | None = None,
) -> pd.DataFrame:
    gcols = ["geo", "TIME_PERIOD", "source_db", "dataflow_id", "key_used"]
    emp_eu = _ensure_geo_col(emp_eu)
    MRIO = set(mrio_prd_set)
    ind_ind = _is_industry_matrix(mrio_prd_set, ind_ind)

    if ind_ind:
        total_code = "CPA_TOTAL"
        mrio_codes = {_add_cpa_prefix(c) for c in MRIO}
        canon = _canon_map(mrio_prd_set)
        canon_cpa = {_add_cpa_prefix(k): _add_cpa_prefix(v) for k, v in canon.items()}
        parent = (class_prd_ava.set_index("prd_ava_code")["parent_code"]
                  .dropna().astype(str).to_dict())
        parent = {canon_cpa.get(k, k): canon_cpa.get(v, v) for k, v in parent.items()}
    else:
        total_code = "CPA_TOTAL"
        mrio_codes = MRIO
        canon = _canon_map(mrio_prd_set)
        canon[total_code] = total_code
        parent = (class_prd_ava.set_index("prd_ava_code")["parent_code"]
                  .dropna().astype(str).to_dict())
        parent = {canon.get(k, k): canon.get(v, v) for k, v in parent.items()}
    parent_codes = {p for p in parent.values() if isinstance(p, str) and p}
    ch = {}
    for c, p in parent.items():
        ch.setdefault(p, []).append(c)

    def desc(c):
        s = {c}
        for k in ch.get(c, []):
            s |= desc(k)
        return s

    df = emp_eu.copy()
    coverage_col = "covers_ind_ava_codes"
    source_col = "employment_source_code"
    if coverage_col not in df.columns:
        df[coverage_col] = pd.NA
    if source_col not in df.columns:
        df[source_col] = df[nace_col]
    df[source_col] = df[source_col].astype(str).str.strip().replace({"U99": "U"})
    df[nace_col] = df[nace_col].astype(str).str.strip()
    if ind_ind:
        df[nace_col] = df[nace_col].map(_add_cpa_prefix)
        df[nace_col] = df[nace_col].map(lambda x: canon_cpa.get(x, x))
    else:
        df[nace_col] = df[nace_col].map(lambda x: canon.get(x, x))
    targets = list(mrio_codes)

    def observed_row_coverage(g: pd.DataFrame) -> dict[str, set[str]]:
        out = {}
        for _, row in g.iterrows():
            src = str(row[nace_col]).strip()
            covers = _split_coverage_string(row.get(coverage_col))
            if not covers:
                continue
            norm_covers = set()
            for c in covers:
                if ind_ind:
                    c_norm = canon_cpa.get(_add_cpa_prefix(c), _add_cpa_prefix(c))
                else:
                    c_norm = canon.get(c, c)
                norm_covers.add(c_norm)
            if norm_covers:
                out.setdefault(src, set()).update(norm_covers)
        return out

    def keep_codes(g: pd.DataFrame, *, geo=None) -> set[str]:
        if geo is None:
            if "geo" in g.columns:
                geo = g["geo"].iloc[0]
            elif isinstance(g.name, tuple):
                geo = g.name[0]
            else:
                geo = g.name
        present = set(g[nace_col])

        if present == {total_code}:
            return {total_code}

        keep = set()
        aggregate_cov = _aggregate_coverage_map(mrio_prd_set, ind_ind=ind_ind, observed_codes=present)
        for src, covers in observed_row_coverage(g).items():
            if src in present:
                aggregate_cov.setdefault(src, set()).update(covers)
        aggregate_expanded = {
            src: set().union(*(desc(c) for c in covers)) if covers else set()
            for src, covers in aggregate_cov.items()
        }
        for t in targets:
            x, seen = canon.get(t, t), set()
            while x not in present and x in parent and x not in seen:
                seen.add(x)
                if parent[x] == total_code:
                    break
                x = parent[x]
            if x in present and x != total_code:
                keep.add(x)
            else:
                for src, covered in aggregate_expanded.items():
                    if src in present and t in covered:
                        keep.add(src)
                        break
                else:
                    if total_code in present:
                        keep.add(total_code)
        return keep

    km_rows = []
    for name, g in df.groupby(gcols, observed=True, dropna=False, group_keys=False):
        geo = name[0] if isinstance(name, tuple) else name
        km_rows.append((*name, keep_codes(g, geo=geo)))
    km = pd.DataFrame(km_rows, columns=gcols + ["_keep"])

    df = df.merge(km, on=gcols, how="left")
    df = df[df.apply(lambda r: r[nace_col] in r["_keep"], axis=1)].drop(columns="_keep")

    # asegurar CPA_TOTAL (marginal, sin doble conteo)
    has_total = df[nace_col].eq(total_code)
    existing = set(map(tuple, df.loc[has_total, gcols].to_numpy()))
    leaf = ~df[nace_col].isin(parent_codes) & ~has_total

    tot = (df.loc[leaf]
             .groupby(gcols, as_index=False, dropna=False)["OBS_VALUE"].sum())
    tot[nace_col] = total_code
    tot[coverage_col] = pd.NA
    tot[source_col] = "TOTAL"
    tot = tot.loc[~tot[gcols].apply(tuple, axis=1).isin(existing)]

    out = pd.concat([df, tot], ignore_index=True)

    if ind_ind:
        out[nace_col] = out[nace_col].map(_strip_cpa_prefix)

    def join_cov(s):
        vals = sorted(set().union(*(_split_coverage_string(x) for x in s)))
        return ";".join(vals) if vals else pd.NA
    def join_codes(s):
        vals = sorted({
            str(x).strip()
            for x in s
            if pd.notna(x) and str(x).strip() and str(x).strip().lower() != "nan"
        })
        return ";".join(vals) if vals else pd.NA
    gcols_final = ["geo","TIME_PERIOD","source_db","dataflow_id","key_used",nace_col]
    opt_cols = [c for c in ["industry_col", "class_system"] if c in out.columns]
    return (
        out.groupby(gcols_final + opt_cols, as_index=False, dropna=False)
        .agg(
            OBS_VALUE=("OBS_VALUE", "sum"),
            covers_ind_ava_codes=(coverage_col, join_cov),
            employment_source_code=(source_col, join_codes),
        )
    )

def make_db_year_source_countrylist(emp_core: pd.DataFrame) -> pd.DataFrame:
    df = emp_core.copy()
    df["geo"] = df["geo"].astype(str).str.strip()
    df["source_db"] = df["source_db"].astype(str).str.strip()
    df["dataflow_id"] = df["dataflow_id"].astype(str).str.strip()
    df["TIME_PERIOD"] = pd.to_numeric(df["TIME_PERIOD"], errors="coerce").astype("Int64")

    out = (df.dropna(subset=["TIME_PERIOD"])
             .groupby(["dataflow_id","TIME_PERIOD","source_db"])["geo"]
             .apply(lambda s: ", ".join(sorted(set(s))))
             .rename("Countries")
             .reset_index()
             .rename(columns={"dataflow_id":"Database ID","TIME_PERIOD":"Year","source_db":"Source"}))

    return out.sort_values(["Source","Database ID","Year"]).reset_index(drop=True)


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
        key_src = f"rule=aggregate_coverage_xlsx_v15_l2_connected_components|ind={int(bool(ind_ind))}|total={total_code}|tc={tc}|emp={emp_rows}x{emp_cols}|mrio_prd={mrio_prd_len}|E={e_len}"
        key = hashlib.sha1(key_src.encode("utf-8")).hexdigest()[:10]
        fname = f"wages_{key}.pkl"
    else:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(cache_key)).strip("_") or "wages"
        fname = f"wages_{safe}.pkl"

    return Path(HELPERS_DIR) / "cache" / fname

def build_wages(
    emp_core,
    E_MRIO,
    mrio_prd_set,
    EU_OFC,
    *,
    total_countries=("CN","JP","CA"),
    total_code=None,
    ind_ind: bool | None = None,
    use_cache: bool = False,
    recompute: bool | None = None,
    cache_path: str | Path | None = None,
    cache_key: str | None = None,
):

    import pandas as pd, numpy as np
    from functools import lru_cache

    ind_ind = _is_industry_matrix(mrio_prd_set, ind_ind)
    if total_code is None:
        total_code = "CPA_TOTAL" if ind_ind else "CPA_TOTAL"

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
                if wages is not None and report is not None and empty is not None:
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

    def use_vals(g, ind):
        # 1. Direct exact match
        if (g, ind) in empd and (g, ind) in d1d:
            return d1d[(g, ind)], empd[(g, ind)]

        # 2. Explicit aggregate/reconstructed coverage, if available.
        # Many-to-many source rows are not assigned partially. Instead,
        # connected source-target blocks are reconstructed as:
        #   sum(D1 over target coverage) / sum(employment over observed sources).
        for sources, targets_cov in aggregate_emp_component_candidates(g, ind):
            e = sum(float(empd[(g, c)]) for c in sources if (g, c) in empd)
            d = sum(float(d1d.get((g, j), 0.0)) for j in targets_cov)
            if e > 0 and d > 0:
                return d, e

        # 3. Parent fallback
        c = ind
        while c in P:
            c = P[c]

            # 3a. If parent employment is explicitly available, use it
            if (g, c) in empd:
                e = empd[(g, c)]
                d = d1_sum_for_emp_code(g, c)
                if e > 0 and d > 0:
                    return d, e

            # 3b. If parent is not explicitly available, build parent wage
            #     from observed descendants under that parent.
            d, e = parent_proxy_vals(g, c)
            if d is not None and e is not None:
                return d, e

        return None, None

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

    eu_key = "EU-OFC"
    eu_in_mrio = eu_key in set(d1.index.get_level_values(0))

    all_geos = [x for x in w.index.get_level_values(0).unique().tolist() if x not in ("ROW", eu_key)]
    row_w = {i: bloc_wage(all_geos, i) for i in prds}
    if ("ROW", prds[0]) in w.index:
        for i in prds: w.loc[("ROW", i), "wage"] = row_w[i]

    if eu_in_mrio and (eu_key, prds[0]) in w.index:
        for i in prds:
            d = d1d.get((eu_key, i))
            e = empd.get((eu_key, i))
            if d is not None and pd.notna(e) and np.isclose(float(e), 0.0, atol=1e-12, rtol=0.0):
                w.loc[(eu_key, i), "wage"] = 0.0
            else:
                w.loc[(eu_key, i), "wage"] = (d/e) if (d is not None and pd.notna(e) and e > 0) else row_w[i]
    else:
        eu_w = {i: bloc_wage(list(EU_OFC), i) for i in prds}
        if (eu_key, prds[0]) in w.index:
            for i in prds: w.loc[(eu_key, i), "wage"] = eu_w[i]

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
    # For each industry, infer expected employment as D1/wage and use the
    # resulting compensation-to-employment ratio across explicit MRIO regions.
    row_mask = w.index.get_level_values("destination_region") == "ROW"
    if row_mask.any():
        explicit = w.loc[~row_mask].copy()
        explicit = explicit.loc[explicit.index.get_level_values("destination_region") != eu_key]
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

    # --- wages cache save ---
    if use_cache and cache_file is not None:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with cache_file.open("wb") as f:
                pickle.dump({
                    "wages": w["wage"],
                    "report": report,
                    "empty": empty,
                }, f)
        except Exception:
            pass

    return w["wage"], report, empty


def _aggregate_coverage_detail_sheet() -> pd.DataFrame:
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
    return out.loc[out["source_code"].notna()].sort_values(["coverage_profile", "source_code"])


def _aggregate_profile_source_sets() -> dict[str, set[str]]:
    details = _aggregate_coverage_detail_sheet()
    if details.empty or "coverage_profile" not in details.columns:
        return {}
    out = {}
    for profile, g in details.dropna(subset=["coverage_profile"]).groupby("coverage_profile", dropna=True):
        sources = {
            _strip_cpa_prefix(str(c).strip())
            for c in g["source_code"]
            if pd.notna(c) and str(c).strip()
        }
        if sources:
            out[str(profile)] = sources
    return out


def _detect_aggregate_profile(codes: set[str]) -> str | None:
    observed = {_strip_cpa_prefix(str(c).strip()) for c in codes if pd.notna(c)}
    observed.discard("TOTAL")
    profile_sources = _aggregate_profile_source_sets()
    matches = [
        (profile, sources)
        for profile, sources in profile_sources.items()
        if sources.issubset(observed)
    ]
    if not matches:
        return None
    matches.sort(key=lambda item: (-len(item[1]), str(item[0])))
    return matches[0][0]


def _infer_industry_col(profile: str | None, dataflows: set[str], source_dbs: set[str]) -> str:
    p = str(profile or "")
    if p.startswith("ECO_"):
        return "ECO"
    if p.startswith("EC2_"):
        return "EC2"
    text = " ".join(sorted(map(str, dataflows | source_dbs))).upper()
    if "EC2" in text:
        return "EC2"
    if "ECO" in text:
        return "ECO"
    if "EUROSTAT" in text or "NAMA_10_A64_E" in text:
        return "NACE_R2"
    return "UNKNOWN"


def _infer_class_system(profile: str | None, industry_col: str, codes: set[str], source_dbs: set[str]) -> str:
    p = str(profile or "")
    if "ISIC4" in p:
        return "ISIC4"
    if "ISIC3" in p:
        return "ISIC3"
    if "AGGREGATE" in p:
        return "AGGREGATE"
    if "SECTOR" in p:
        return "SECTOR"
    if industry_col == "NACE_R2":
        return "NACE_R2"
    if industry_col == "EC2":
        stripped = {_strip_cpa_prefix(str(c).strip()) for c in codes if pd.notna(c)}
        stripped.discard("TOTAL")
        return "ISIC3" if stripped and not ({"A02", "A03"} & stripped) else "ISIC4"
    if industry_col == "ECO":
        return "ISIC4"
    return "UNKNOWN"


def _detect_resolution_profile(codes: set[str], dataflows: set[str], source_dbs: set[str]) -> tuple[str, str, str]:
    stripped = {_strip_cpa_prefix(str(c).strip()) for c in codes if pd.notna(c)}
    non_total = stripped - {"TOTAL"}
    profile = _detect_aggregate_profile(stripped)
    industry_col = _infer_industry_col(profile, dataflows, source_dbs)

    if profile is None:
        class_system = _infer_class_system(None, industry_col, stripped, source_dbs)
        suffix = "TOTAL" if not non_total else "DETAIL"
        profile = f"{industry_col}_{class_system}_{suffix}"
    else:
        class_system = _infer_class_system(profile, industry_col, stripped, source_dbs)

    return profile, industry_col, class_system


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
    df["nace_r2"] = df["nace_r2"].astype(str).str.strip().map(_strip_cpa_prefix)
    if "source_db" not in df.columns:
        df["source_db"] = pd.NA
    if "dataflow_id" not in df.columns:
        df["dataflow_id"] = pd.NA
    if "TIME_PERIOD" not in df.columns:
        df["TIME_PERIOD"] = pd.NA

    country_rows = []
    for geo, g in df.groupby("geo", dropna=False):
        codes_all = {c for c in g["nace_r2"].dropna().astype(str) if c and c != "nan"}
        codes = codes_all - {"TOTAL"} or codes_all
        dataflows = {str(x).strip() for x in g["dataflow_id"].dropna().unique() if str(x).strip()}
        source_dbs = {str(x).strip() for x in g["source_db"].dropna().unique() if str(x).strip()}
        years = {str(x).strip() for x in g["TIME_PERIOD"].dropna().unique() if str(x).strip()}
        profile, industry_col, class_system = _detect_resolution_profile(codes_all, dataflows, source_dbs)
        profile_label = "NACE2" if profile == "NACE_R2_NACE_R2_DETAIL" else profile
        country_rows.append({
            "geo": str(geo),
            "coverage_profile": profile_label,
            "industry_col": industry_col,
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


def _wage_parent_lookup(mrio_prd_set, *, ind_ind: bool | None = None):
    ind_ind = _is_industry_matrix(mrio_prd_set, ind_ind)
    canon = _canon_map(mrio_prd_set)
    canon["CPA_TOTAL"] = "CPA_TOTAL"

    if ind_ind:
        allowed = {_add_cpa_prefix(c) for c in mrio_prd_set} | {"CPA_TOTAL"}
        canon_cpa = {_add_cpa_prefix(k): _add_cpa_prefix(v) for k, v in canon.items()}
        parent = (class_prd_ava.set_index("prd_ava_code")["parent_code"]
                  .dropna().astype(str).to_dict())
        parent = {canon_cpa.get(k, k): canon_cpa.get(v, v) for k, v in parent.items()}
    else:
        allowed = set(mrio_prd_set) | {"CPA_TOTAL"}
        parent = (class_prd_ava.set_index("prd_ava_code")["parent_code"]
                  .dropna().astype(str).to_dict())
        parent = {canon.get(k, k): canon.get(v, v) for k, v in parent.items()}

    return parent, allowed
def plot_wages_dispersion_boxplot_sorted(
    wages: pd.Series,
    *,
    eu_countries: set[str],
    filename="wages_dispersion_boxplot_sorted.pdf",
    save_figures=True,
    close_figures=True,
    figsize=(11, 5),
    dpi=150,
    stat="median",   # "median" o "p75"
):
    set_house_style()

    w = wages.copy()
    w.name = "wage"
    w = w.reset_index()
    w["destination_region"] = w["destination_region"].astype(str).str.strip()
    # Drop EU-OFC entirely
    w = w[~w["destination_region"].isin({"EU-OFC", "EU_OFC"})].copy()
    eu_countries = {str(c).strip() for c in eu_countries if str(c).strip() not in {"EU-OFC", "EU_OFC"}}
    w["wage"] = pd.to_numeric(w["wage"], errors="coerce")
    w = w[w["wage"] > 0].copy()
    w["wage"] = w["wage"] / 1_000.0

    # -------- robust statistic per region --------
    if stat == "median":
        score = w.groupby("destination_region")["wage"].median()
    elif stat == "p75":
        score = w.groupby("destination_region")["wage"].quantile(0.75)
    else:
        raise ValueError("stat must be 'median' or 'p75'")

    # -------- split EU vs ROW --------
    eu_regs  = score[score.index.isin(eu_countries)].sort_values(ascending=False)
    row_regs = score[~score.index.isin(eu_countries)].sort_values(ascending=False)

    reg_order = list(eu_regs.index) + list(row_regs.index)

    data = [
        w.loc[w["destination_region"] == r, "wage"].to_numpy()
        for r in reg_order
    ]

    # -------- plotting --------
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.boxplot(
        data,
        tick_labels=[tex_safe(r) for r in reg_order],
        showfliers=False
    )

    # visual separator EU / ROW
    if len(eu_regs) > 0:
        ax.axvline(len(eu_regs) + 0.5, color="black", linewidth=1)

    ax.set_ylabel(tex_safe("Compensations (kEUR/year)"), fontweight="bold")
    ax.set_xlabel(tex_safe("Destination region"), fontweight="bold")
    ax.tick_params(axis="x", rotation=45)

    apply_house_style(ax, grid=True)
    fig.tight_layout()

    return save_and_close(fig=fig, filename=filename, save=save_figures, close=close_figures)
