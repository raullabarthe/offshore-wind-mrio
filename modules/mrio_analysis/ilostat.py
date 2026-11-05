"""ILOSTAT source retrieval, parsing, normalization, and source selection.

The FTE40 allocation and reconciliation methodology remains in ``employment``.
This module owns ILOSTAT-specific SDMX/bulk requests, local source caches, and
source-format normalization.
"""

from __future__ import annotations

from io import BytesIO
from functools import lru_cache
from pathlib import Path
import json
import random
import re
import time
import unicodedata
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pycountry
import requests

from modules.config import EMPLOYMENT_DATA_DIR, HELPERS_DIR
from .source_utils import (
    _INDUSTRIAL_SOURCE_COLUMNS,
    _INDUSTRIAL_SOURCE_DTYPES,
    _chunk,
    _fte40_area_to_iso2,
    _fte40_explicit_countries,
    _get_csv,
    _session_with_retries,
)
from .employment import (
    _clean_industry_code,
    _configured_resolution_targets,
    _effective_resolution_metadata,
    _first_column,
    _identify_source_classification,
    ILO_GLOBAL_A14_CODES,
    ILO_FTE40_BASIS,
    _mean_hours_employment_product,
)
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
            if code not in (403, 429, 500, 502, 503, 504, 524):
                raise
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
        raise ValueError(
            "flows must be supplied explicitly; configure the selected ILO "
            "dataflows in main.ipynb."
        )
    flows = (flows,) if isinstance(flows, str) else tuple(str(flow).strip() for flow in flows)

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


ILO_CL_AREA_URL = f"{SDMX_BASE}/codelist/ILO/CL_AREA/1.0"
ILO_BULK_REF_AREA_TOC_URL = (
    "https://rplumber.ilo.org/metadata/toc/ref_area/"
)


def _xml_annotation_values(element: ET.Element) -> dict[str, str]:
    values = {}
    for annotation in element.findall(".//{*}Annotation"):
        key = annotation.findtext("./{*}AnnotationTitle") or annotation.get("type")
        value = annotation.findtext("./{*}AnnotationText")
        if key and value:
            values[str(key).strip()] = str(value).strip()
    return values


def fetch_cl_area_metadata(*, timeout: int = 180) -> pd.DataFrame:
    """Fetch the complete ILOSTAT CL_AREA classification."""
    root = _get_xml(ILO_CL_AREA_URL, timeout=timeout)
    rows = []
    for code in root.findall(".//{*}Codelist[@id='CL_AREA']//{*}Code"):
        code_id = str(code.get("id") or "").strip()
        if not code_id:
            continue
        names = code.findall("./{*}Name")
        name_en = ""
        for name in names:
            language = str(name.get("{http://www.w3.org/XML/1998/namespace}lang") or "").lower()
            if not language or language in {"en", "eng"}:
                name_en = str(name.text or "").strip()
                if language in {"en", "eng"}:
                    break
        annotations = _xml_annotation_values(code)
        iso2 = annotations.get("ISO2Code") or annotations.get("ISO2")
        m49 = annotations.get("M49Code") or annotations.get("M49")
        group_flag = str(annotations.get("Group", "")).strip().upper()
        is_aggregate = group_flag in {"Y", "YES", "TRUE"} or code_id.startswith("X")
        rows.append(
            {
                "area_code": code_id,
                "name_en": name_en,
                "iso3": None if is_aggregate else code_id,
                "iso2": iso2,
                "m49": m49,
                "is_aggregate": is_aggregate,
                "group_flag": group_flag,
                "identifier_source": (
                    "ILOSTAT CL_AREA annotations"
                    if iso2 or m49 or is_aggregate
                    else "ISO registry fallback"
                ),
                "source": "ILOSTAT CL_AREA",
                "source_url": ILO_CL_AREA_URL,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        result = pd.DataFrame(
            columns=[
                "geo", "source_code", "original_source_code", "hours_source_code",
                "targets", "value", "employment_observation", "mean_weekly_hours",
                "source_is_exact", "source_db", "source_year", "dataflow_id", "key_used",
                "population_scope", "class_system", "hours_dataflow_id", "hours_source_year",
                "hours_population_scope", "compatibility_status",
            ]
        )
    if result.empty:
        return result
    result["iso2"] = result["iso2"].replace({"": pd.NA})
    result["m49"] = result["m49"].replace({"": pd.NA})
    return result.sort_values("area_code").reset_index(drop=True)


def _fetch_ilo_bulk_area_metadata() -> pd.DataFrame:
    """Fallback official area metadata from the current ILO bulk TOC."""
    toc = _get_csv(ILO_BULK_REF_AREA_TOC_URL, raise_on_non_csv=True)
    if toc.empty or "ref_area" not in toc.columns:
        return pd.DataFrame()
    toc = toc.copy()
    toc["area_code"] = toc["ref_area"].astype(str).str.strip()
    toc = toc.drop_duplicates("area_code")
    aliases = {
        "wb_income_group": "group_income",
        "wb_income_group.label": "group_income.label",
        "ilo_region": "group_geo",
        "ilo_region.label": "group_geo.label",
    }
    toc = toc.rename(columns=aliases)
    rows = []
    for row in toc.itertuples(index=False):
        code = str(row.area_code)
        country = pycountry.countries.get(alpha_3=code)
        is_country = country is not None
        rows.append(
            {
                "area_code": code,
                "name_en": str(getattr(row, "ref_area_label", "") or ""),
                "iso3": code if is_country else None,
                "iso2": country.alpha_2 if country is not None else None,
                "m49": country.numeric if country is not None else None,
                "is_aggregate": not is_country,
                "group_flag": None,
                "identifier_source": "ISO registry" if is_country else "ILOSTAT area code",
                "source": "ILOSTAT bulk reference-area metadata",
                "source_url": ILO_BULK_REF_AREA_TOC_URL,
            }
        )
    return pd.DataFrame(rows)


def _complete_ilo_area_identifiers(areas: pd.DataFrame) -> pd.DataFrame:
    """Complete ISO identifiers when CL_AREA omits annotation values.

    ILOSTAT's CL_AREA annotations are not always populated consistently.  For
    genuine ISO alpha-3 country codes, pycountry supplies the corresponding
    ISO2 and M49 identifiers; non-country ILOSTAT aggregates are left
    untouched.
    """
    if areas.empty:
        return areas
    out = areas.copy()
    for column in ("iso2", "m49"):
        if column not in out.columns:
            out[column] = pd.NA
    if "iso3" not in out.columns:
        out["iso3"] = pd.NA
    if "is_aggregate" not in out.columns:
        out["is_aggregate"] = False
    if "identifier_source" not in out.columns:
        out["identifier_source"] = "ILOSTAT CL_AREA annotations"

    for index, row in out.iterrows():
        code = str(row.get("area_code") or "").strip().upper()
        if bool(row.get("is_aggregate")):
            continue
        country = pycountry.countries.get(alpha_3=code)
        if country is None:
            continue
        if pd.isna(row.get("iso3")) or not str(row.get("iso3")).strip():
            out.at[index, "iso3"] = country.alpha_3
        if pd.isna(row.get("iso2")) or not str(row.get("iso2")).strip():
            out.at[index, "iso2"] = country.alpha_2
        if pd.isna(row.get("m49")) or not str(row.get("m49")).strip():
            out.at[index, "m49"] = country.numeric
        if any(
            pd.isna(row.get(column)) or not str(row.get(column)).strip()
            for column in ("iso2", "m49")
        ):
            out.at[index, "identifier_source"] = "ISO registry fallback"
    return out


def load_ilo_area_metadata(
    *,
    cache_path: str | Path | None = None,
    use_cache: bool = True,
    refresh: bool = False,
    allow_stale: bool = True,
    strict: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load official ILO area metadata and preserve group memberships.

    Geographic and income memberships are returned separately because they
    overlap and must not be treated as mutually exclusive categories.
    """
    cache_file = Path(cache_path) if cache_path is not None else HELPERS_DIR / "cache" / "ilo_area_metadata.json"
    if use_cache and not refresh and cache_file.exists():
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            areas = pd.DataFrame(payload.get("areas", []))
            memberships = pd.DataFrame(payload.get("memberships", []))
            if not areas.empty:
                return areas, memberships
        except Exception:
            if not allow_stale:
                raise

    try:
        classification_source = "ILOSTAT CL_AREA"
        try:
            areas = fetch_cl_area_metadata()
        except Exception:
            areas = _fetch_ilo_bulk_area_metadata()
            classification_source = "ILOSTAT bulk reference-area metadata; CL_AREA unavailable"
        if areas.empty:
            raise RuntimeError("No ILO area metadata were returned.")
        areas = _complete_ilo_area_identifiers(areas)
        toc = _get_csv(ILO_BULK_REF_AREA_TOC_URL, raise_on_non_csv=True)
        if not toc.empty and "ref_area" in toc.columns:
            toc = toc.copy()
            toc["area_code"] = toc["ref_area"].astype(str).str.strip()
            toc = toc.rename(
                columns={
                    "wb_income_group": "group_income",
                    "wb_income_group.label": "group_income.label",
                    "ilo_region": "group_geo",
                    "ilo_region.label": "group_geo.label",
                }
            )
            keep = [
                column for column in [
                    "area_code", "ref_area.label", "group_geo", "group_geo.label",
                    "ilo_subregion_broad", "ilo_subregion_broad.label",
                    "ilo_subregion_detailed", "ilo_subregion_detailed.label",
                    "group_income", "group_income.label", "last.update",
                ] if column in toc.columns
            ]
            areas = areas.merge(toc[keep].drop_duplicates("area_code"), on="area_code", how="left")
        areas["classification_source"] = classification_source

        memberships_rows = []
        for row in areas.to_dict("records"):
            member = str(row.get("area_code"))
            for group_type, code_attr, label_attr in (
                ("geographic", "group_geo", "group_geo.label"),
                ("geographic_subregion", "ilo_subregion_broad", "ilo_subregion_broad.label"),
                ("geographic_subregion_detailed", "ilo_subregion_detailed", "ilo_subregion_detailed.label"),
                ("income", "group_income", "group_income.label"),
            ):
                code = row.get(code_attr)
                label = row.get(label_attr)
                if pd.notna(code) and str(code).strip():
                    memberships_rows.append(
                        {
                            "member_code": member,
                            "group_type": group_type,
                            "group_code": str(code).strip(),
                            "group_label": None if pd.isna(label) else str(label).strip(),
                        }
                    )
        memberships = pd.DataFrame(memberships_rows)
        retrieved = pd.Timestamp.utcnow().isoformat()
        areas["retrieved_at_utc"] = retrieved
        if not memberships.empty:
            memberships["retrieved_at_utc"] = retrieved
        payload = {
            "retrieved_at_utc": retrieved,
            "areas": areas.where(pd.notna(areas), None).to_dict("records"),
            "memberships": memberships.where(pd.notna(memberships), None).to_dict("records"),
        }
        if use_cache:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        return areas, memberships
    except Exception:
        if strict:
            raise
        print("[ILOSTAT] area metadata unavailable; no aggregate classification was substituted.")
        return pd.DataFrame(), pd.DataFrame()

def iso2_to_iso3(code2: str) -> str | None:
    c = pycountry.countries.get(alpha_2=str(code2).strip().upper())
    return c.alpha_3 if c else None

# ---------- SDMX download (GenericData) ----------


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
        if code == 404:
            return pd.DataFrame(), key_base
        if code not in (403, 429, 500, 502, 503, 504, 524):
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


# ======================================================
# FTE40 satellite account
# ======================================================

ILO_FTE40_DATAFLOW = "DF_EMP_2FTE_SEX_JBF_NB"
ILO_FTE40_BULK_INDICATOR = "EMP_2FTE_SEX_JBF_NB_A"
ILO_FTE40_BULK_BY_DATAFLOW = {
    ILO_FTE40_DATAFLOW: ILO_FTE40_BULK_INDICATOR,
}


def _normalise_fte40_raw(
    raw: pd.DataFrame,
    *,
    dataflow_id: str,
    key_used: str,
    basis: str = ILO_FTE40_BASIS,
) -> pd.DataFrame:
    """Standardise API and ILO bulk-indicator FTE40 extracts."""
    basis = str(basis).strip()
    if basis != ILO_FTE40_BASIS:
        raise ValueError(
            f"Unsupported FTE40 basis {basis!r}; required basis is {ILO_FTE40_BASIS!r}."
        )
    if raw is None or raw.empty:
        return pd.DataFrame(
            columns=[
                "geo", "OBS_VALUE", "TIME_PERIOD", "SEX", "JBF",
                "source_area", "source_db", "dataflow_id", "key_used", "unit",
            ]
        )

    df = raw.copy()
    area_col = _first_column(df, ("REF_AREA", "ref_area", "ref_area.label", "geo", "country"))
    value_col = _first_column(df, ("OBS_VALUE", "obs_value", "value"))
    year_col = _first_column(df, ("TIME_PERIOD", "time", "year"))
    sex_col = _first_column(df, ("SEX", "sex", "sex.label"))
    basis_col = _first_column(df, ("JBF", "classif1", "classif1.label", "basis"))
    if area_col is None or value_col is None or year_col is None:
        raise ValueError("The FTE40 source lacks area, year, or observation-value columns.")

    out = pd.DataFrame(index=df.index)
    out["source_area"] = df[area_col].astype("string").str.strip()
    out["geo"] = out["source_area"].map(_fte40_area_to_iso2)
    out["OBS_VALUE"] = pd.to_numeric(df[value_col], errors="coerce")
    out["TIME_PERIOD"] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
    out["SEX"] = df[sex_col].astype("string").str.strip() if sex_col else pd.NA
    out["JBF"] = df[basis_col].astype("string").str.strip() if basis_col else pd.NA
    out["source_db"] = "ILOSTAT_bulk" if "label" in str(area_col).lower() else "ILOSTAT_SDMX"
    out["dataflow_id"] = dataflow_id
    out["key_used"] = key_used
    out["unit"] = "THS_FTE"

    if sex_col:
        sex_values = out["SEX"].dropna().astype(str).str.upper()
        total_mask = sex_values.eq("SEX_T") | sex_values.eq("T") | sex_values.eq("TOTAL")
        if not total_mask.any():
            raise ValueError("The FTE40 extract does not contain a total-sex observation.")
        out = out.loc[total_mask.reindex(out.index, fill_value=False)].copy()

    if basis_col:
        basis_values = out["JBF"].fillna("").astype(str).str.upper()
        wanted = (
            basis_values.eq(basis.upper())
            | basis_values.str.contains(r"FTE40|40\s*HOURS", regex=True, na=False)
        )
        if not wanted.any():
            raise ValueError("The FTE40 extract does not contain JBF_FTE_FTE40 observations.")
        out = out.loc[wanted].copy()
    else:
        raise ValueError("The FTE40 extract lacks the JBF classification dimension.")

    out = out.loc[out["geo"].notna() & out["OBS_VALUE"].notna()].copy()
    out["geo"] = out["geo"].astype(str)
    return out.reset_index(drop=True)


def load_ilo_fte40(
    CORE,
    *,
    year: int,
    year_min: int | None = None,
    local_path: str | Path | None = None,
    dataflow_id: str = ILO_FTE40_DATAFLOW,
    basis: str = ILO_FTE40_BASIS,
    include_world: bool = True,
    strict: bool = True,
    freq: str = "A",
    include_area_metadata: bool = False,
    metadata_strict: bool = False,
) -> pd.DataFrame:
    """Load ILOSTAT FTE40 totals for FIGARO countries and World.

    The returned values are thousand FTE persons.  The source has no industry
    dimension; industry allocation is deliberately performed separately by
    ``build_fte40_country_industry``.  Statistical aggregates returned by the
    bulk file are ignored unless they are the World aggregate (X01).
    """
    dataflow_id = str(dataflow_id).strip()
    try:
        bulk_indicator = ILO_FTE40_BULK_BY_DATAFLOW[dataflow_id]
    except KeyError as exc:
        supported = ", ".join(sorted(ILO_FTE40_BULK_BY_DATAFLOW))
        raise ValueError(
            f"Unsupported FTE40 dataflow {dataflow_id!r}; supported dataflows: {supported}."
        ) from exc
    basis = str(basis).strip()
    if basis != ILO_FTE40_BASIS:
        raise ValueError(
            f"Unsupported FTE40 basis {basis!r}; required basis is {ILO_FTE40_BASIS!r}."
        )

    explicit = _fte40_explicit_countries(CORE)
    requested = set(explicit)
    if include_world:
        requested.add("X01")
    area_metadata = pd.DataFrame()
    area_memberships = pd.DataFrame()
    if include_area_metadata:
        area_metadata, area_memberships = load_ilo_area_metadata(strict=metadata_strict)

    path = Path(local_path) if local_path is not None else None
    if path is None:
        candidates = sorted(EMPLOYMENT_DATA_DIR.glob(f"{bulk_indicator}*.csv.gz"))
        if candidates:
            path = candidates[0]

    years = [int(year)]
    if year_min is not None:
        if int(year_min) > int(year):
            raise ValueError("year_min must be less than or equal to year")
        years.extend(range(int(year) - 1, int(year_min) - 1, -1))

    selected = None
    for query_year in years:
        if path is not None:
            raw = pd.read_csv(path, compression="infer")
            key_used = str(path)
        else:
            ilo_areas = fetch_cl_area_iso3()
            iso3 = [
                iso2_to_iso3(code)
                for code in explicit
                if iso2_to_iso3(code) and iso2_to_iso3(code) in ilo_areas
            ]
            if include_world:
                iso3.append("X01")
            raw, key_used = download_genericdata_to_df(
                dataflow_id,
                query_year,
                iso3,
                freq=freq,
            )
        normalised = _normalise_fte40_raw(
            raw,
            dataflow_id=dataflow_id,
            key_used=key_used,
            basis=basis,
        )
        normalised = normalised.loc[normalised["TIME_PERIOD"].eq(query_year)].copy()
        if normalised.empty:
            continue

        # A country-year must have one FTE40 total. Equal duplicate rows are
        # harmless; conflicting duplicates are an input-data error.
        key = ["geo", "TIME_PERIOD"]
        duplicate_counts = normalised.groupby(key, dropna=False)["OBS_VALUE"].nunique()
        if (duplicate_counts > 1).any():
            bad = duplicate_counts[duplicate_counts > 1].index.tolist()
            raise ValueError(f"Conflicting duplicate FTE40 observations for {bad}.")
        normalised = normalised.drop_duplicates(subset=key, keep="first")
        available = set(normalised["geo"])
        needed = requested & available
        if selected is None:
            selected = normalised.loc[normalised["geo"].isin(needed)].copy()
        else:
            already = set(selected["geo"])
            add = normalised.loc[
                normalised["geo"].isin(needed - already)
            ].copy()
            selected = pd.concat([selected, add], ignore_index=True)
        if requested.issubset(set(selected["geo"])):
            break

    if selected is None:
        selected = pd.DataFrame()

    missing = sorted(requested - set(selected.get("geo", pd.Series(dtype=str))))
    if missing and strict:
        raise ValueError(
            "FTE40 data are missing requested FIGARO area(s): "
            f"{missing}. No aggregate was substituted for a country."
        )
    if missing:
        print(f"[ILOSTAT] FTE40 areas unavailable: {missing}")

    if selected.empty:
        result = pd.DataFrame(
            columns=[
                "geo", "OBS_VALUE", "TIME_PERIOD", "SEX", "JBF",
                "source_area", "source_db", "dataflow_id", "key_used", "unit",
            ]
        )
        result.attrs["area_metadata"] = area_metadata
        result.attrs["area_memberships"] = area_memberships
        return result
    result = selected.sort_values(["geo", "TIME_PERIOD"]).reset_index(drop=True)
    result.attrs["area_metadata"] = area_metadata
    result.attrs["area_memberships"] = area_memberships
    return result


ILO_MEAN_HOURS_DATAFLOWS = (
    "DF_HOW_TEMP_SEX_EC2_NB",
    "DF_HOW_TEMP_SEX_ECO_NB",
)

ILO_MEAN_HOURS_BULK_BY_DATAFLOW = {
    "DF_HOW_TEMP_SEX_EC2_NB": "HOW_TEMP_SEX_EC2_NB_A",
    "DF_HOW_TEMP_SEX_ECO_NB": "HOW_TEMP_SEX_ECO_NB_A",
}
ILO_MEAN_HOURS_BULK_INDICATORS = tuple(ILO_MEAN_HOURS_BULK_BY_DATAFLOW.values())
ILO_EMPLOYMENT_DATAFLOWS = (
    "DF_EMP_2EMP_SEX_ECO_NB",
    "DF_EMP_TEMP_SEX_EC2_NB",
    "DF_EMP_TEMP_SEX_ECO_NB",
)
ILO_EMPLOYMENT_BULK_BY_DATAFLOW = {
    "DF_EMP_2EMP_SEX_ECO_NB": "EMP_2EMP_SEX_ECO_NB_A",
    "DF_EMP_TEMP_SEX_EC2_NB": "EMP_TEMP_SEX_EC2_NB_A",
    "DF_EMP_TEMP_SEX_ECO_NB": "EMP_TEMP_SEX_ECO_NB_A",
}
ILO_EMPLOYMENT_BULK_INDICATORS = tuple(ILO_EMPLOYMENT_BULK_BY_DATAFLOW.values())
ILO_BULK_INDICATOR_URL = "https://rplumber.ilo.org/data/indicator"


@lru_cache(maxsize=4)
def _download_ilo_bulk_indicator(indicator: str, timeout: int = 180) -> pd.DataFrame:
    """Download one official ILOSTAT annual bulk indicator in memory."""
    response = _session_with_retries().get(
        ILO_BULK_INDICATOR_URL,
        params={"format": ".csv.gz", "id": indicator},
        timeout=timeout,
    )
    response.raise_for_status()
    if not response.content:
        return pd.DataFrame()
    return pd.read_csv(
        BytesIO(response.content),
        compression="gzip",
        dtype={
            "obs_status": "string",
            "note_classif": "string",
            "note_indicator": "string",
            "note_source": "string",
        },
        low_memory=False,
    )


@lru_cache(maxsize=8)
def _ilo_bulk_sdmx_shape(indicator: str, loader_token: object | None = None) -> pd.DataFrame:
    """Read and reshape one bulk indicator once per Python session."""
    return _bulk_hours_to_sdmx_shape(
        _download_ilo_bulk_indicator(indicator),
        indicator,
    )


def _bulk_hours_to_sdmx_shape(frame: pd.DataFrame, indicator: str) -> pd.DataFrame:
    """Convert the ILO bulk HOW layout to the existing industry processor."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    required = {"ref_area", "sex", "classif1", "time", "obs_value"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    out = frame.rename(
        columns={
            "ref_area": "REF_AREA",
            "sex": "SEX",
            "classif1": "EC2" if "EC2" in indicator else "ECO",
            "time": "TIME_PERIOD",
            "obs_value": "OBS_VALUE",
        }
    ).copy()
    out["dataflow_id"] = indicator
    out["source_db"] = "ILOSTAT_bulk"
    out["key_used"] = f"bulk_indicator={indicator}"
    return out


def _select_ilo_bulk_classification(
    frame: pd.DataFrame,
    *,
    area_col: str = "REF_AREA",
    industry_col: str = "ECO",
    target_industries: set[str] | None = None,
    measure_name: str = "employment",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one classification family using effective FIGARO resolution."""
    if frame.empty or not {area_col, industry_col}.issubset(frame.columns):
        return frame, pd.DataFrame()
    work = frame.copy()
    work["class_system"] = work[industry_col].map(
        lambda code: _identify_source_classification(code=str(code))
    )
    candidates = []
    for area, area_group in work.groupby(area_col, sort=False):
        usable = area_group.loc[
            area_group["OBS_VALUE"].notna()
            & ~area_group[industry_col].astype(str).str.contains("TOTAL", case=False, na=False)
        ]
        if usable.empty:
            continue
        for class_system, group in usable.groupby("class_system", sort=False):
            try:
                processed = process_ilo_industry_measure(
                    group,
                    source_db="ILOSTAT_bulk",
                    measure_name=measure_name,
                )
            except ValueError:
                continue
            metadata, _mapped = _effective_resolution_metadata(
                processed,
                target_industries=target_industries,
                value_name="OBS_VALUE",
                source_name="ILOSTAT_bulk",
            )
            candidates.append(
                {
                    "area": area,
                    "class_system": class_system,
                    **metadata,
                    "resolution_count": metadata["effective_resolution"],
                    "priority": {
                        "ISIC4": 4, "ISIC3": 3,
                        "ECO_AGGREGATE": 1, "ECO_SECTOR": 1,
                    }.get(class_system, 0),
                    "frame": area_group.loc[area_group["class_system"].eq(class_system)].copy(),
                }
            )
    selected = []
    log = []
    for area in work[area_col].drop_duplicates():
        area_candidates = [candidate for candidate in candidates if candidate["area"] == area]
        if not area_candidates:
            continue
        best = max(
            area_candidates,
            key=lambda candidate: (
                candidate["effective_resolution"],
                candidate["exact_A64"],
                candidate["independent_blocks"],
                candidate["A64_covered"],
                -candidate["aggregate_blocks"],
                -candidate["n_unmapped"],
                candidate["priority"],
            ),
        )
        selected.append(best["frame"])
        log.append(
            {
                "geo": area,
                "class_system": best["class_system"],
                "resolution_count": best["resolution_count"],
                "resolution_profile": best["resolution_profile"],
                "named_source_profile": best["named_source_profile"],
                "effective_resolution": best["effective_resolution"],
                "resolution_completeness": best["resolution_completeness"],
                "valid_source_observations": best["valid_source_observations"],
                "mapped_source_observations": best["mapped_source_observations"],
                "independent_blocks": best["independent_blocks"],
                "exact_A64": best["exact_A64"],
                "aggregate_blocks": best["aggregate_blocks"],
                "A64_covered": best["A64_covered"],
                "target_count": best["target_count"],
                "unmapped_codes": ";".join(best["unmapped_codes"]),
                "ambiguous_codes": ";".join(best["ambiguous_codes"]),
                "n_unmapped": len(best["unmapped_codes"]),
                "n_ambiguous": len(best["ambiguous_codes"]),
            }
        )
    if not selected:
        return work.iloc[0:0].copy(), pd.DataFrame(log)
    return pd.concat(selected, ignore_index=True), pd.DataFrame(log)


@lru_cache(maxsize=64)
def _ilo_bulk_year_candidates(
    indicator: str,
    query_year: int,
    iso3: tuple[str, ...],
    target_codes: tuple[str, ...] | None,
    measure_name: str,
    loader_token: object | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cache one indicator-year classification evaluation.

    Both the ordinary-country and World-minus-ROW employment selections use
    this same candidate pool.  The callers still apply their own year and
    compatibility ranking after retrieving it.
    """
    raw = _ilo_bulk_sdmx_shape(indicator, loader_token)
    if raw.empty:
        empty = pd.DataFrame()
        return empty, empty, empty
    raw = raw.loc[
        raw["REF_AREA"].isin(set(iso3))
        & pd.to_numeric(raw["TIME_PERIOD"], errors="coerce").eq(int(query_year))
        & raw["SEX"].astype(str).str.upper().isin({"SEX_T", "T", "TOTAL"})
    ].copy()
    if raw.empty:
        empty = pd.DataFrame()
        return empty, empty, empty
    code_col = "EC2" if "EC2" in raw.columns else "ECO"
    target_set = set(target_codes) if target_codes is not None else None
    chosen, class_log = _select_ilo_bulk_classification(
        raw,
        area_col="REF_AREA",
        industry_col=code_col,
        target_industries=target_set,
        measure_name=measure_name,
    )
    processed = (
        process_ilo_industry_measure(
            chosen,
            source_db="ILOSTAT_bulk",
            measure_name=measure_name,
        )
        if not chosen.empty
        else pd.DataFrame()
    )
    return chosen, class_log, processed


def load_ilo_employment_by_industry(
    CORE,
    *,
    year: int,
    year_min: int | None = None,
    year_by_iso3: dict[str, int] | None = None,
    hours_mean: pd.DataFrame | None = None,
    indicators: tuple[str, ...] | None = None,
    flows: tuple[str, ...] = ILO_EMPLOYMENT_DATAFLOWS,
    target_industries: set[str] | None = None,
    include_world: bool = False,
) -> pd.DataFrame:
    """Load bulk ILO employment observations at the years used for HOW.

    ``year_by_iso3`` is used to enforce year compatibility when mean-hours
    resolution falls back from the requested year to an earlier bounded year.
    The function preserves missing observations and does not aggregate parent
    and detailed rows before FTE hierarchy resolution.
    """
    requested_flows = (flows,) if isinstance(flows, str) else tuple(str(flow).strip() for flow in flows)
    unsupported_flows = tuple(
        flow for flow in requested_flows if flow not in ILO_EMPLOYMENT_BULK_BY_DATAFLOW
    )
    if unsupported_flows:
        supported = ", ".join(sorted(ILO_EMPLOYMENT_BULK_BY_DATAFLOW))
        raise ValueError(
            f"Unsupported employment dataflow(s) {unsupported_flows}; supported dataflows: {supported}."
        )
    if indicators is None:
        indicators = tuple(
            ILO_EMPLOYMENT_BULK_BY_DATAFLOW[flow] for flow in requested_flows
        )
    iso2_countries = _fte40_explicit_countries(CORE, include_world=include_world)
    iso3 = {iso2_to_iso3(code) for code in iso2_countries if code != "X01"}
    if include_world and "X01" in iso2_countries:
        iso3.add("X01")
    iso3.discard(None)
    iso3_tuple = tuple(sorted(iso3))
    target_codes = (
        tuple(sorted(target_industries)) if target_industries is not None else None
    )
    requested_years = sorted(
        {int(year)}
        | {
            int(value)
            for value in (year_by_iso3 or {}).values()
            if pd.notna(value)
        }
    )
    candidates = {}
    availability_rows: list[dict] = []
    for indicator_order, indicator in enumerate(indicators):
        for query_year in requested_years:
            try:
                chosen, log, processed = _ilo_bulk_year_candidates(
                    indicator,
                    query_year,
                    iso3_tuple,
                    target_codes,
                    "employment",
                    _download_ilo_bulk_indicator,
                )
            except Exception:
                continue
            if chosen.empty or log.empty:
                continue
            code_col = "EC2" if "EC2" in chosen.columns else "ECO"
            for entry in log.to_dict("records"):
                area = entry["geo"]
                expected_year = int((year_by_iso3 or {}).get(area, year))
                if int(query_year) != expected_year:
                    continue
                candidate_frame = chosen.loc[chosen["REF_AREA"].eq(area)].copy()
                candidate_processed = processed.loc[
                    processed["geo"].eq(_fte40_area_to_iso2(area))
                ].copy()
                compatible_pairs = 0
                compatible_covered = 0
                if hours_mean is not None and not hours_mean.empty and not candidate_processed.empty:
                    hour_codes = hours_mean.loc[
                        hours_mean["geo"].eq(_fte40_area_to_iso2(area))
                    ].copy()
                    if not hour_codes.empty:
                        compatible = _mean_hours_employment_product(
                            candidate_processed,
                            hour_codes,
                            target_industries=target_industries or _configured_resolution_targets(),
                        )
                        compatible_pairs = int(
                            compatible.attrs.get("compatible_pair_count", 0)
                        )
                        compatible_covered = int(
                            compatible.attrs.get("compatible_A64_covered", 0)
                        )
                candidate = {
                    **entry,
                    "indicator": indicator,
                        "year": int(query_year),
                    "indicator_priority": 2 if "2EMP" in indicator else 1,
                    "order": indicator_order,
                    "compatible_hours_pairs": compatible_pairs,
                    "compatible_A64_covered": compatible_covered,
                    "processed": candidate_processed,
                }
                old = candidates.get(area)
                if target_industries is None:
                    candidate_key = (
                        candidate["compatible_hours_pairs"] > 0,
                        candidate["compatible_hours_pairs"],
                        candidate["resolution_count"],
                        candidate["class_system"] == "ISIC4",
                        candidate["indicator_priority"],
                        -candidate["order"],
                    )
                    old_key = (
                        old["compatible_hours_pairs"] > 0,
                        old["compatible_hours_pairs"],
                        old["resolution_count"],
                        old["class_system"] == "ISIC4",
                        old["indicator_priority"],
                        -old["order"],
                    ) if old is not None else None
                else:
                    candidate_key = (
                        candidate["effective_resolution"],
                        candidate["independent_blocks"],
                        candidate["exact_A64"],
                        candidate["A64_covered"],
                        candidate["compatible_A64_covered"],
                        candidate["compatible_hours_pairs"],
                        -candidate["aggregate_blocks"],
                        -candidate["n_unmapped"],
                        candidate["indicator_priority"],
                        -candidate["order"],
                    )
                    old_key = (
                        old["effective_resolution"],
                        old["independent_blocks"],
                        old["exact_A64"],
                        old["A64_covered"],
                        old["compatible_A64_covered"],
                        old["compatible_hours_pairs"],
                        -old["aggregate_blocks"],
                        -old["n_unmapped"],
                        old["indicator_priority"],
                        -old["order"],
                    ) if old is not None else None
                if old is None or candidate_key > old_key:
                    candidates[area] = candidate
                availability_rows.append(
                    {
                        "geo": _fte40_area_to_iso2(area),
                        "indicator": indicator,
                        "class_system": entry["class_system"],
                        "source_year": int(query_year),
                        "usable_original_codes": entry["resolution_count"],
                        "valid_source_observations": entry["valid_source_observations"],
                        "mapped_source_observations": entry["mapped_source_observations"],
                        "independent_blocks": entry["independent_blocks"],
                        "exact_A64": entry["exact_A64"],
                        "aggregate_blocks": entry["aggregate_blocks"],
                        "A64_covered": entry["A64_covered"],
                        "resolution_profile": entry["resolution_profile"],
                        "named_source_profile": entry["named_source_profile"],
                        "effective_resolution": entry["effective_resolution"],
                        "resolution_completeness": entry["resolution_completeness"],
                        "unmapped_codes": entry["unmapped_codes"],
                        "ambiguous_codes": entry["ambiguous_codes"],
                        "original_source_codes": ";".join(
                            sorted(
                                str(value).strip()
                                for value in candidate_frame[code_col].dropna().unique()
                                if "TOTAL" not in str(value).upper()
                            )
                        ),
                        "compatible_employment_hours_pairs": compatible_pairs,
                        "compatible_A64_covered": compatible_covered,
                        "selected": False,
                        "selection_reason": "candidate retained for resolution comparison",
                    }
                )
    missing_iso3 = sorted(set(iso3) - set(candidates))
    sdmx_result = pd.DataFrame()
    if missing_iso3 and flows:
        sdmx_raw, _sdmx_log, _remaining = pull_flows_deductive(
            list(flows),
            missing_iso3,
            year=int(year),
            year_min=year_min,
            freq="A",
            target_industries=target_industries,
        )
        if not sdmx_raw.empty:
            sdmx_result = process_ilo_long(
                sdmx_raw,
                drop_x=False,
                source_db="ILOSTAT_SDMX",
            )

    if not candidates and sdmx_result.empty:
        empty = pd.DataFrame(columns=_INDUSTRIAL_SOURCE_COLUMNS)
        empty.attrs["requested_countries"] = iso2_countries
        empty.attrs["missing_countries"] = iso2_countries
        return empty
    result = pd.concat(
        [candidate["processed"] for candidate in candidates.values()],
        ignore_index=True,
    ) if candidates else pd.DataFrame()
    if not sdmx_result.empty:
        result = pd.concat([result, sdmx_result], ignore_index=True)
    result.attrs["selection_log"] = pd.DataFrame(
        [
            {
                "geo": area,
                "year": candidate["year"],
                "indicator": candidate["indicator"],
                "class_system": candidate["class_system"],
                "resolution_count": candidate["resolution_count"],
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
                "unmapped_codes": candidate["unmapped_codes"],
                "ambiguous_codes": candidate["ambiguous_codes"],
                "n_unmapped": candidate["n_unmapped"],
                "n_ambiguous": candidate["n_ambiguous"],
                "compatible_hours_pairs": candidate["compatible_hours_pairs"],
                "compatible_A64_covered": candidate["compatible_A64_covered"],
                "population_scope": (
                    "all employed persons (modelled estimates)"
                    if "2EMP" in candidate["indicator"]
                    else "employment population"
                ),
            }
            for area, candidate in candidates.items()
        ]
    )
    availability = pd.DataFrame(availability_rows)
    if not availability.empty:
        selected_keys = {
            (
                _fte40_area_to_iso2(area),
                candidate["indicator"],
                candidate["class_system"],
                    candidate["year"],
            )
            for area, candidate in candidates.items()
        }
        availability["selected"] = availability.apply(
            lambda row: (
                row["geo"], row["indicator"], row["class_system"], row["source_year"]
            ) in selected_keys,
            axis=1,
        )
        availability["selection_reason"] = np.where(
            availability["selected"],
            "selected: highest effective industrial resolution; compatible-hours detail used as tie-break",
            "not selected: a candidate with higher effective industrial resolution was available",
        )
    result.attrs["source_availability_audit"] = availability
    result.attrs["requested_countries"] = iso2_countries
    result.attrs["missing_countries"] = sorted(
        set(iso2_countries) - set(result["geo"].dropna().astype(str).str.upper())
    )
    if not sdmx_result.empty:
        result.attrs["selection_log"] = pd.concat(
            [
                result.attrs["selection_log"],
                pd.DataFrame(
                    {
                        "geo": sorted(sdmx_result["geo"].dropna().unique()),
                        "year": pd.NA,
                        "indicator": "SDMX fallback",
                        "class_system": pd.NA,
                        "resolution_count": pd.NA,
                        "population_scope": "employment population",
                    }
                ),
            ],
            ignore_index=True,
        )
    # Keep TOTAL as a separate national-employment constraint.  The FTE
    # standardiser excludes it from industrial targets, while the allocator
    # reads it explicitly through _extract_fte_national_totals().
    return result.reset_index(drop=True)


def process_ilo_industry_measure(
    df: pd.DataFrame,
    *,
    source_db: str = "ILOSTAT_SDMX",
    measure_name: str = "value",
) -> pd.DataFrame:
    """Process one additive or mean ILO industry measure without zero filling.

    This is intentionally separate from ``process_ilo_long``.  The latter is
    retained for the historical headcount/wage workflow, whereas FTE40 source
    construction must preserve missing observations and must not add mean-hour
    observations across sex, age, or overlapping classifications.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=_INDUSTRIAL_SOURCE_COLUMNS)
    work = df.copy()
    if "SEX" in work.columns:
        sex = work["SEX"].astype("string").str.upper()
        total = sex.isin({"SEX_T", "T", "TOTAL"})
        if not total.any():
            raise ValueError("The ILO industry extract contains no total-sex observations.")
        work = work.loc[total].copy()
    industry_col = next((column for column in ("EC2", "ECO") if column in work.columns), None)
    if industry_col is None:
        raise ValueError("The ILO industry extract contains neither EC2 nor ECO.")
    raw_code = work[industry_col].astype("string").str.strip()
    work["original_source_code"] = raw_code
    work["source_code"] = (
        raw_code
        .str.replace(r"^(EC2|ECO)_(ISIC3|ISIC4)_", "", regex=True)
        .str.replace(r"^ECO_AGGREGATE_\d+(?:_AGG)?_", "", regex=True)
        .str.replace(r"^ECO_SECTOR_\d+_", "", regex=True)
        .str.replace(r"^ECO_AGNAG_(?:\d+_)?", "", regex=True)
        .str.replace(r"^ECO_(AGGREGATE|SECTOR|AGNAG)_", "", regex=True)
        .replace({"U99": "U"})
    )
    work["OBS_VALUE"] = pd.to_numeric(work["OBS_VALUE"], errors="coerce")
    work = work.loc[work["OBS_VALUE"].notna() & work["source_code"].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=_INDUSTRIAL_SOURCE_COLUMNS)

    keys = ["REF_AREA", "TIME_PERIOD", "source_code"]
    if not set(keys).issubset(work.columns):
        keys = ["TIME_PERIOD", "source_code"]
    duplicate_values = work.groupby(keys, dropna=False)["OBS_VALUE"].nunique()
    if (duplicate_values > 1).any():
        bad = duplicate_values[duplicate_values > 1].index.tolist()
        raise ValueError(f"Conflicting ILO industry observations for {bad}.")
    work = work.drop_duplicates(keys, keep="first")
    work["geo"] = work["REF_AREA"].map(_fte40_area_to_iso2) if "REF_AREA" in work else pd.NA
    work = work.loc[work["geo"].notna()].copy()
    work["source_db"] = source_db
    work["year"] = pd.to_numeric(work["TIME_PERIOD"], errors="coerce").astype("Int64")
    if "dataflow_id" not in work:
        work["dataflow_id"] = pd.NA
    if "key_used" not in work:
        work["key_used"] = pd.NA
    if "class_system" not in work:
        work["class_system"] = work["original_source_code"].map(
            lambda code: _identify_source_classification(code=str(code))
        )
    unit_col = _first_column(work, ("unit", "UNIT"))
    if unit_col is not None:
        work["unit"] = work[unit_col]
    else:
        work["unit"] = (
            "HOURS_PER_WEEK" if measure_name == "hours" else "THS_PER"
        )
    dataflow = work["dataflow_id"].astype("string").fillna("").str.upper()
    work["population_scope"] = np.select(
        [
            dataflow.str.contains("EMP_2EMP", na=False),
            dataflow.str.contains("EMP_TEMP", na=False),
            dataflow.str.contains("EES_TEES", na=False),
            dataflow.str.contains("HOW_", na=False),
        ],
        [
            "all employed persons (modelled estimates)",
            "employment population",
            "employees",
            "mean weekly hours actually worked per employed person",
        ],
        default=pd.NA,
    )
    return work[_INDUSTRIAL_SOURCE_COLUMNS].astype(
        _INDUSTRIAL_SOURCE_DTYPES
    ).reset_index(drop=True)


def load_ilo_mean_hours_by_industry(
    CORE,
    *,
    year: int,
    year_min: int | None = None,
    flows: tuple[str, ...] = ILO_MEAN_HOURS_DATAFLOWS,
    min_resolution_codes: int = 15,
    use_bulk: bool = True,
    target_industries: set[str] | None = None,
    include_world: bool = False,
) -> pd.DataFrame:
    """Load ILO mean weekly hours by economic activity for FTE allocation.

    The official bulk HOW files are preferred because they remain available
    when the SDMX data endpoint is rate-limited.  EC2 (ISIC Rev. 4) and ECO
    observations are selected country by country using the same bounded-year
    resolution rule as the existing deductive SDMX loader.
    """
    requested_flows = (flows,) if isinstance(flows, str) else tuple(str(flow).strip() for flow in flows)
    unsupported_flows = tuple(
        flow for flow in requested_flows if flow not in ILO_MEAN_HOURS_BULK_BY_DATAFLOW
    )
    if unsupported_flows:
        supported = ", ".join(sorted(ILO_MEAN_HOURS_BULK_BY_DATAFLOW))
        raise ValueError(
            f"Unsupported mean-hours dataflow(s) {unsupported_flows}; supported dataflows: {supported}."
        )
    requested_indicators = tuple(
        ILO_MEAN_HOURS_BULK_BY_DATAFLOW[flow] for flow in requested_flows
    )

    iso2_countries = _fte40_explicit_countries(CORE, include_world=include_world)
    iso3 = [iso2_to_iso3(code) for code in iso2_countries if code != "X01"]
    if include_world and "X01" in iso2_countries:
        iso3.append("X01")
    iso3 = [code for code in iso3 if code]
    iso3_tuple = tuple(sorted(iso3))
    target_codes = (
        tuple(sorted(target_industries)) if target_industries is not None else None
    )
    if use_bulk:
        query_years = [int(year)]
        lower_bound = int(year) - 1 if year_min is None else int(year_min)
        query_years.extend(
            candidate
            for candidate in range(int(year) - 1, lower_bound - 1, -1)
            if candidate not in query_years
        )
        candidates: dict[str, dict] = {}
        availability_rows = []
        for query_year in query_years:
            for flow_order, indicator in enumerate(requested_indicators, 1):
                try:
                    chosen, class_log, processed = _ilo_bulk_year_candidates(
                        indicator,
                        query_year,
                        iso3_tuple,
                        target_codes,
                        "hours",
                        _download_ilo_bulk_indicator,
                    )
                except Exception:
                    continue
                if chosen.empty or class_log.empty:
                    continue
                code_col = "EC2" if "EC2" in chosen.columns else "ECO"
                for entry in class_log.to_dict("records"):
                    area = entry["geo"]
                    group = chosen.loc[chosen["REF_AREA"].eq(area)].copy()
                    candidate = {
                        **entry,
                        "year": query_year,
                        "indicator": indicator,
                        "priority": 2 if "EC2" in indicator else 1,
                        "order": flow_order,
                        "processed": processed.loc[
                            processed["geo"].eq(_fte40_area_to_iso2(area))
                        ].copy(),
                    }
                    availability_rows.append(
                        {
                            "geo": _fte40_area_to_iso2(area),
                            "indicator": indicator,
                            "class_system": entry["class_system"],
                            "source_year": query_year,
                            "usable_original_codes": entry["resolution_count"],
                            "valid_source_observations": entry["valid_source_observations"],
                            "mapped_source_observations": entry["mapped_source_observations"],
                            "independent_blocks": entry["independent_blocks"],
                            "exact_A64": entry["exact_A64"],
                            "aggregate_blocks": entry["aggregate_blocks"],
                            "A64_covered": entry["A64_covered"],
                            "resolution_profile": entry["resolution_profile"],
                            "named_source_profile": entry["named_source_profile"],
                            "effective_resolution": entry["effective_resolution"],
                            "resolution_completeness": entry["resolution_completeness"],
                            "unmapped_codes": entry["unmapped_codes"],
                            "ambiguous_codes": entry["ambiguous_codes"],
                            "original_source_codes": ";".join(
                                sorted(
                                    str(value).strip()
                                    for value in group[code_col].dropna().unique()
                                    if "TOTAL" not in str(value).upper()
                                )
                            ),
                            "selected": False,
                        }
                    )
                    old = candidates.get(area)
                    if target_industries is None:
                        candidate_sufficient = candidate["resolution_count"] > int(min_resolution_codes)
                        old_sufficient = (
                            old is not None
                            and old["resolution_count"] > int(min_resolution_codes)
                        )
                        if old is None:
                            better = True
                        elif candidate_sufficient != old_sufficient:
                            better = candidate_sufficient
                        elif candidate_sufficient:
                            better = (
                                candidate["resolution_count"],
                                candidate["priority"],
                                -candidate["year"],
                                -candidate["order"],
                            ) > (
                                old["resolution_count"],
                                old["priority"],
                                -old["year"],
                                -old["order"],
                            )
                        else:
                            better = (
                                -candidate["year"],
                                candidate["resolution_count"],
                                candidate["priority"],
                                -candidate["order"],
                            ) > (
                                -old["year"],
                                old["resolution_count"],
                                old["priority"],
                                -old["order"],
                            )
                    else:
                        candidate_key = (
                            candidate["effective_resolution"],
                            candidate["exact_A64"],
                            candidate["independent_blocks"],
                            candidate["A64_covered"],
                            -candidate["aggregate_blocks"],
                            -candidate["n_unmapped"],
                            candidate["year"],
                            candidate["priority"],
                            -candidate["order"],
                        )
                        old_key = (
                            old["effective_resolution"],
                            old["exact_A64"],
                            old["independent_blocks"],
                            old["A64_covered"],
                            -old["aggregate_blocks"],
                            -old["n_unmapped"],
                            old["year"],
                            old["priority"],
                            -old["order"],
                        ) if old is not None else None
                        better = old is None or candidate_key > old_key
                    if better:
                        candidates[area] = candidate

        if candidates:
            result = pd.concat(
                [candidate["processed"] for candidate in candidates.values()],
                ignore_index=True,
            )
            result = result.loc[result["source_code"].ne("TOTAL")].copy()
            result.attrs["selection_log"] = pd.DataFrame(
                [
                    {
                        "geo": area,
                        "year": candidate["year"],
                        "indicator": candidate["indicator"],
                        "resolution_count": candidate["resolution_count"],
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
                        "unmapped_codes": candidate["unmapped_codes"],
                        "ambiguous_codes": candidate["ambiguous_codes"],
                    }
                    for area, candidate in candidates.items()
                ]
            )
            result.attrs["remaining_iso3"] = sorted(set(iso3) - set(candidates))
            result.attrs["requested_countries"] = iso2_countries
            result.attrs["missing_countries"] = sorted(
                set(iso2_countries) - set(result["geo"].dropna().astype(str).str.upper())
            )
            availability = pd.DataFrame(availability_rows)
            if not availability.empty:
                selected_keys = {
                    (str(_fte40_area_to_iso2(area)), candidate["indicator"], candidate["year"])
                    for area, candidate in candidates.items()
                }
                availability["selected"] = availability.apply(
                    lambda row: (str(row["geo"]), row["indicator"], row["source_year"])
                    in selected_keys,
                    axis=1,
                )
                availability["selection_reason"] = np.where(
                    availability["selected"],
                    "selected: maximum bounded-year industrial resolution",
                    "not selected: lower resolution or lower-priority candidate",
                )
            result.attrs["source_availability_audit"] = availability
            return result

    raw, log, remaining = pull_flows_deductive(
            list(requested_flows),
        iso3,
        year=year,
        year_min=year_min,
        min_resolution_codes=min_resolution_codes,
        target_industries=target_industries,
    )
    if raw.empty:
        empty = pd.DataFrame(columns=_INDUSTRIAL_SOURCE_COLUMNS)
        empty.attrs["requested_countries"] = iso2_countries
        empty.attrs["missing_countries"] = iso2_countries
        return empty
    result = process_ilo_industry_measure(raw, measure_name="hours")
    result.attrs["selection_log"] = log
    result.attrs["remaining_iso3"] = remaining
    result.attrs["source_availability_audit"] = pd.DataFrame(log)
    result.attrs["requested_countries"] = iso2_countries
    result.attrs["missing_countries"] = sorted(
        set(iso2_countries) - set(result["geo"].dropna().astype(str).str.upper())
    )
    return result

def load_ilo_global_employment(
    year: int,
    *,
    year_min: int | None = None,
    dataflow_id: str = "DF_EMP_2EMP_SEX_ECO_NB",
    freq: str = "A",
) -> pd.DataFrame:
    """Load global employment from ILOSTAT's X01 aggregate.

    The selected dataflow reports total employment in thousand persons.  The
    A14 industry rows are retained because they provide the resolution needed
    to allocate the global residual to the FIGARO industries.  ``year_min``
    bounds the backward search in the same way as the country-level pull.
    """
    dataflow_id = str(dataflow_id).strip()
    if dataflow_id not in ILO_EMPLOYMENT_BULK_BY_DATAFLOW:
        supported = ", ".join(sorted(ILO_EMPLOYMENT_BULK_BY_DATAFLOW))
        raise ValueError(
            f"Unsupported global-employment dataflow {dataflow_id!r}; "
            f"supported dataflows: {supported}."
        )

    first_year = int(year)
    last_year = first_year if year_min is None else int(year_min)
    if last_year > first_year:
        raise ValueError("year_min must be less than or equal to year")

    required = set(ILO_GLOBAL_A14_CODES)
    empty_columns = [
        "geo", "nace_r2", "TIME_PERIOD", "OBS_VALUE", "source_db",
        "dataflow_id", "key_used", "industry_col", "class_system",
    ]
    best = None

    for query_year in range(first_year, last_year - 1, -1):
        raw, key_used = download_genericdata_to_df(
            dataflow_id,
            query_year,
            ["X01"],
            freq=freq,
        )
        if raw.empty:
            continue

        df = raw.copy()
        if "SEX" in df.columns:
            total_sex = df["SEX"].astype(str).eq("SEX_T")
            if total_sex.any():
                df = df.loc[total_sex].copy()

        industry_col = next(
            (column for column in ("ECO", "EC2") if column in df.columns),
            None,
        )
        if industry_col is None or "OBS_VALUE" not in df.columns:
            continue

        codes = (
            df[industry_col]
            .astype(str)
            .str.extract(r"(?:ECO|EC2)_ISIC4_(.+)$", expand=False)
        )
        df = df.assign(nace_r2=codes)
        df = df.loc[df["nace_r2"].isin(required)].copy()
        if df.empty:
            continue

        df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
        df = (
            df.dropna(subset=["OBS_VALUE"])
            .groupby("nace_r2", as_index=False)["OBS_VALUE"]
            .sum()
        )
        coverage = len(set(df["nace_r2"]) & required)
        candidate = (coverage, query_year, df, key_used, industry_col)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
        if coverage == len(required):
            break

    if best is None:
        return pd.DataFrame(columns=empty_columns)

    coverage, selected_year, values, key_used, industry_col = best
    result = values.assign(
        geo="X01",
        TIME_PERIOD=int(selected_year),
        source_db="ILOSTAT_SDMX",
        dataflow_id=dataflow_id,
        key_used=key_used,
        industry_col=industry_col,
        class_system="ISIC4",
        population_scope="all employed persons (modelled estimates)",
    )
    result = result[
        [
            "geo", "nace_r2", "TIME_PERIOD", "OBS_VALUE", "source_db",
            "dataflow_id", "key_used", "industry_col", "class_system",
            "population_scope",
        ]
    ]
    if coverage < len(required):
        missing = sorted(required - set(result["nace_r2"]))
        print(
            "[ILOSTAT] global employment is missing A14 group(s): "
            f"{missing}; using the available groups."
        )
    return result.reset_index(drop=True)


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

def pull_flows_deductive(
    flows,
    iso3_list,
    *,
    year,
    freq="A",
    prefer_isic4_over_isic3=True,
    fallback_years=None,
    min_resolution_codes=15,
    year_min=None,
    target_industries: set[str] | None = None,
):
    """Download one preferred employment flow per country.

    The selected candidate maximizes usable industry-code resolution across
    the requested flows and bounded years. Classification priority is used
    only after resolution has been compared. If no candidate reaches the
    resolution threshold, the most recent available country-year is retained.
    """
    remaining=list(iso3_list); out=[]; log=[]
    query_years = [int(year)]
    if fallback_years is None:
        lower_bound = int(year) - 1 if year_min is None else int(year_min)
        if lower_bound > int(year):
            raise ValueError("year_min must be less than or equal to year")
        fallback_years = range(int(year) - 1, lower_bound - 1, -1)
    query_years.extend(
        int(candidate_year)
        for candidate_year in fallback_years
        if int(candidate_year) not in query_years
        and (year_min is None or int(candidate_year) >= int(year_min))
        and int(candidate_year) < int(year)
    )
    candidates = {}

    def _resolution_count(df):
        """Count usable codes using the same EC2-over-ECO precedence as processing."""
        if df.empty:
            return 0

        if "EC2" in df.columns:
            values = df["EC2"].copy()
            if "ECO" in df.columns:
                values = values.where(values.notna(), df["ECO"])
        elif "ECO" in df.columns:
            values = df["ECO"].copy()
        else:
            return 0

        values = values.dropna().astype(str).str.strip()
        values = values[values.str.match(r"^(?:EC2|ECO)_ISIC[34]_", na=False)]
        values = values[
            ~values.str.contains("TOTAL", case=False, na=False)
            & ~values.str.endswith("_X")
        ]
        return int(values.nunique())

    def _candidate_is_better(new, old):
        if old is None:
            return True
        if target_industries is not None:
            def _key(candidate):
                return (
                    candidate["effective_resolution"],
                    candidate["independent_blocks"],
                    candidate["exact_A64"],
                    candidate["A64_covered"],
                    -candidate["aggregate_blocks"],
                    -candidate["n_unmapped"],
                    candidate["year"],
                    candidate["priority"],
                    -candidate["order"],
                )
            return _key(new) > _key(old)
        if not prefer_isic4_over_isic3:
            return False

        new_sufficient = new["resolution_count"] > int(min_resolution_codes)
        old_sufficient = old["resolution_count"] > int(min_resolution_codes)
        if new_sufficient != old_sufficient:
            return new_sufficient
        # If neither candidate reaches the threshold, retain the most recent
        # available country-year rather than silently moving further back.
        if not new_sufficient and not old_sufficient:
            if new["year"] != old["year"]:
                return new["year"] > old["year"]
            if new["priority"] != old["priority"]:
                return new["priority"] > old["priority"]
            if new["resolution_count"] != old["resolution_count"]:
                return new["resolution_count"] > old["resolution_count"]
            return new["order"] < old["order"]
        if new["resolution_count"] != old["resolution_count"]:
            return new["resolution_count"] > old["resolution_count"]
        if new["priority"] != old["priority"]:
            return new["priority"] > old["priority"]
        return new["order"] < old["order"]

    def _record_candidates(df, *, df_id, key_used, query_year, order):
        chosen_now = []
        if df.empty or "REF_AREA" not in df.columns:
            return chosen_now
        df = df.copy()
        df["source_db"] = "ILOSTAT"
        df["dataflow_id"] = df_id
        df["key_used"] = key_used
        for area, group in df.groupby("REF_AREA", dropna=False):
            area = str(area)
            metadata = {}
            if target_industries is not None:
                processed = process_ilo_industry_measure(
                    group,
                    source_db="ILOSTAT",
                    measure_name="value",
                )
                metadata, _mapped = _effective_resolution_metadata(
                    processed,
                    target_industries=target_industries,
                    value_name="value",
                    source_name="ILOSTAT",
                )
            candidate = {
                "priority": _ilo_class_priority(group),
                "resolution_count": metadata.get(
                    "effective_resolution", _resolution_count(group)
                ),
                "order": order,
                "year": int(query_year),
                "dataflow_id": df_id,
                "df": group.copy(),
                **metadata,
            }
            if _candidate_is_better(candidate, candidates.get(area)):
                candidates[area] = candidate
                chosen_now.append(area)
        return chosen_now

    for order, df_id in enumerate(flows, 1):
        if prefer_isic4_over_isic3:
            requested = list(iso3_list)
        else:
            if not remaining:
                break
            requested = list(remaining)
        reqset=set(requested)
        pending = list(requested)
        keys_used = []
        years_used = []
        got_all = set()
        chosen_now = []
        for query_year in query_years:
            if not pending:
                break
            df_try, key_try = download_genericdata_to_df(
                df_id,
                query_year,
                pending,
                freq=freq,
            )
            if not df_try.empty and "REF_AREA" in df_try:
                df_try = df_try[df_try["REF_AREA"].isin(set(pending))].copy()
            got_try = (
                sorted(set(df_try["REF_AREA"].unique()))
                if ("REF_AREA" in df_try and not df_try.empty)
                else []
            )
            if not df_try.empty:
                keys_used.append(str(key_try))
                years_used.append(query_year)
            chosen_now.extend(
                _record_candidates(
                    df_try,
                    df_id=df_id,
                    key_used=key_try,
                    query_year=query_year,
                    order=order,
                )
            )
            got_all.update(got_try)
            if not prefer_isic4_over_isic3:
                ready = {
                    area
                    for area, candidate in candidates.items()
                    if candidate["resolution_count"] > int(min_resolution_codes)
                }
                pending = [country for country in pending if country not in ready]

        if not prefer_isic4_over_isic3:
            out.extend(
                candidate["df"]
                for area, candidate in candidates.items()
                if area in got_all
            )
        if not prefer_isic4_over_isic3:
            remaining = [c for c in remaining if c not in {
                area for area, candidate in candidates.items()
                if candidate["resolution_count"] > int(min_resolution_codes)
            }]
        log.append({
            "order":order,
            "dataflow_id":df_id,
            "requested":len(requested),
            "got":len(got_all),
            "chosen_or_replaced":len(chosen_now),
            "remaining":len(remaining),
            "rows":sum(
                len(candidate["df"])
                for area, candidate in candidates.items()
                if area in got_all and candidate["dataflow_id"] == df_id
            ),
            "years_used":";".join(map(str, dict.fromkeys(years_used))),
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
    original_source_code=pd.Series(pd.NA,index=df.index,dtype="string")
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
            original_source_code=original_source_code.mask(take,raw)
            industry_col=industry_col.mask(take,col)
            class_system=class_system.mask(take,sys)

    df=df.assign(
        nace_r2=n,
        employment_source_code=n,
        original_source_code=original_source_code,
        industry_col=industry_col,
        class_system=class_system,
    ).loc[lambda d:d["nace_r2"].notna()]

    s=df["nace_r2"].astype("string")
    aggregate_row = df["original_source_code"].astype("string").str.match(
        r"^ECO_(?:AGGREGATE|SECTOR|AGNAG)_", na=False
    )
    s = s.mask(
        aggregate_row,
        df["original_source_code"].map(_clean_industry_code),
    )
    df["class_system"] = df["class_system"].replace(
        {"AGGREGATE": "ECO_AGGREGATE", "AGNAG": "ECO_AGGREGATE", "SECTOR": "ECO_SECTOR"}
    )
    is_tot=s.str.contains("TOTAL",na=False)
    is_blk=s.str.match(r"^ECO_(AGGREGATE|AGNAG|SECTOR)_",na=False)&~is_tot
    is_det=s.str.startswith("ECO_DETAILS_",na=False)

    df["nace_r2"]=(s.mask(is_tot,"TOTAL")
                    .mask(is_blk,pd.NA)
                    .mask(is_det,s.str.replace(r"^ECO_DETAILS_","",regex=True))
                    .replace({"U99": "U"}))
    df["employment_source_code"] = df["original_source_code"].map(
        _clean_industry_code
    ).replace({"U99": "U"})
    df=df.loc[df["nace_r2"].notna()]
    if drop_x: df=df.loc[df["nace_r2"].ne("X")]

    # TOTAL: no agregar ? dejar uno solo
    K=["REF_AREA","TIME_PERIOD","nace_r2","employment_source_code","original_source_code","industry_col","class_system","dataflow_id","key_used"]
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
    df["year"] = pd.to_numeric(df["TIME_PERIOD"], errors="coerce").astype("Int64")
    df["source_code"] = df["employment_source_code"].astype("string").str.strip()
    df["unit"] = "THS_PER"
    dataflow = df["dataflow_id"].astype("string").fillna("").str.upper()
    df["population_scope"] = np.select(
        [
            dataflow.str.contains("EMP_2EMP", na=False),
            dataflow.str.contains("EMP_TEMP", na=False),
            dataflow.str.contains("EES_TEES", na=False),
        ],
        [
            "all employed persons (modelled estimates)",
            "employment population",
            "employees",
        ],
        default=pd.NA,
    )

    return df[_INDUSTRIAL_SOURCE_COLUMNS].astype(
        _INDUSTRIAL_SOURCE_DTYPES
    ).reset_index(drop=True)
