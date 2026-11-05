"""Eurostat employment and labour-input loaders.

This module owns Eurostat request construction, response parsing, and
Eurostat-specific source normalization. Shared HTTP helpers and industrial
source-schema definitions are in :mod:`source_utils`.
"""

from __future__ import annotations

import re

import pandas as pd
import requests

from modules.config import HELPERS_DIR
from .source_utils import (
    EurostatUnavailable,
    _INDUSTRIAL_SOURCE_COLUMNS,
    _INDUSTRIAL_SOURCE_DTYPES,
    _chunk,
    _fte40_explicit_countries,
    _get_csv,
)


def load_emp_eurostat(CORE, years=range(2023, 2010, -1),
                      geo_list=None, require_unit="THS_PER", na_item="EMP_DC",
                      geo_chunk=8, *, year: int | None = None,
                      preserve_hierarchy: bool = False,
                      include_total: bool = False):
    if year is not None:
        years = [int(year)]
    elif isinstance(years, int):
        years = [int(years)]

    DATAFLOW="nama_10_a64_e"
    BASE3="https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT"
    GEO_LIST = geo_list or [
        "EU27_2020","EA20","EA19","EA12",
        "BE","BG","CZ","DK","DE","EE","IE","EL","ES","FR","HR",
        "IT","CY","LV","LT","LU","HU","MT","NL","AT","PL","PT","RO","SI","SK","FI","SE",
        # The current nama_10_a64_e dissemination has no UK observations;
        # GB is covered by the ILO fallback later in the pipeline.
        "IS","LI","NO","CH","ME","MK","RS","XK"
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
        df = df.drop(columns=["DATAFLOW", "OBS_FLAG", "CONF_STATUS"], errors="ignore")
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
        available = d.loc[d["OBS_VALUE"].notna(), "geo"].drop_duplicates().tolist()
        if available:
            accepted.append(d[d["geo"].isin(available)])
            remaining-=set(available)

    missing_countries = sorted("GR" if geo == "EL" else geo for geo in remaining)

    if not accepted:
        columns = _INDUSTRIAL_SOURCE_COLUMNS if preserve_hierarchy else [
            "geo", "nace_r2", "TIME_PERIOD", "OBS_VALUE", "source_db",
            "dataflow_id", "key_used",
        ]
        empty = pd.DataFrame(columns=columns)
        empty.attrs["missing_countries"] = missing_countries
        return empty

    emp=pd.concat(accepted, ignore_index=True)
    emp["geo"] = emp["geo"].replace({"EL": "GR"})

    if preserve_hierarchy:
        # Keep the original Eurostat hierarchy.  The historical path below
        # maps to CPA and removes TOTAL rows for wage calculations, but FTE40
        # allocation must see both parent and detailed NACE observations
        # before resolving overlaps.
        emp["source_code"] = emp["nace_r2"].astype("string").str.strip()
        emp["original_source_code"] = emp["source_code"]
        if not include_total:
            emp = emp.loc[emp["source_code"].ne("TOTAL")].copy()
        emp["year"] = pd.to_numeric(emp["TIME_PERIOD"], errors="coerce").astype("Int64")
        emp["unit"] = require_unit
        emp["class_system"] = "NACE_R2"
        emp["population_scope"] = (
            "national-accounts employment"
            if require_unit == "THS_PER"
            else "national-accounts annual hours"
        )
        emp = emp.drop_duplicates().sort_values(
            by=["geo", "source_code", "year"],
            ascending=[True, True, True],
        )
        emp = emp[_INDUSTRIAL_SOURCE_COLUMNS].astype(
            _INDUSTRIAL_SOURCE_DTYPES
        ).reset_index(drop=True)
        emp.attrs["missing_countries"] = missing_countries
        return emp

    emp=emp[emp["nace_r2"].ne("TOTAL")].copy()

    # map NACE->CPA (replace nace_r2, keep unmapped log)
    m=(pd.read_csv(HELPERS_DIR/"ind_to_prd.csv")
         .loc[:,["ind_ava_code","prd_ava_code"]]
         .assign(ind_ava_code=lambda d: d["ind_ava_code"].astype(str).str.strip(),
                 prd_ava_code=lambda d: d["prd_ava_code"].astype(str).str.strip())
         .set_index("ind_ava_code")["prd_ava_code"].to_dict())

    emp["nace_r2_raw"]=emp["nace_r2"]
    emp["nace_r2"]=emp["nace_r2_raw"].map(m)

    bad=emp[emp["nace_r2"].isna() & emp["nace_r2_raw"].notna()]
    if not bad.empty:
        print("UNMAPPED nace_r2 codes:")
        print(bad["nace_r2_raw"].value_counts().head(30).to_string())

    legacy_uk = emp["geo"].astype(str).str.strip().eq("UK")
    if legacy_uk.any():
        raise ValueError(
            "Eurostat returned legacy GEO code 'UK'; expected canonical code 'GB'."
        )
    emp = emp.loc[emp["geo"].isin(set(CORE))].copy()

    # Deduplicate exact clones created by NACE?CPA many-to-one mapping
    emp = (
        emp.sort_values("OBS_VALUE", ascending=False)
        .drop_duplicates(subset=["geo", "TIME_PERIOD", "nace_r2"], keep="first")
    )
    emp = emp.sort_values(by=["geo", "nace_r2"], ascending=[True, True])

    emp = emp[["geo","nace_r2","TIME_PERIOD","OBS_VALUE","source_db","dataflow_id","key_used"]].copy()
    emp.attrs["missing_countries"] = missing_countries
    return emp


def load_emp_eurostat_bundle(
    CORE,
    *,
    year: int,
    geo_list=None,
    geo_chunk: int = 8,
) -> dict[str, pd.DataFrame]:
    """Load Eurostat employment and annual-hours inputs without flattening NACE.

    Both tables use ``EMP_DC`` and retain the national ``TOTAL`` row.  The
    returned ``employment`` and ``hours`` frames retain the original NACE
    codes so parent observations can be resolved against detailed children by
    the FTE40 allocator.  The conventional ``load_emp_eurostat`` defaults
    remain unchanged for existing wage consumers.
    """
    eurostat_geos = geo_list or sorted(_fte40_explicit_countries(CORE))
    eurostat_geos = ["EL" if str(geo).upper() == "GR" else geo for geo in eurostat_geos]
    common = {
        "CORE": CORE,
        "year": year,
        "geo_list": eurostat_geos,
        "geo_chunk": geo_chunk,
        "preserve_hierarchy": True,
        "include_total": True,
    }
    employment = load_emp_eurostat(
        require_unit="THS_PER",
        na_item="EMP_DC",
        **common,
    )
    hours = load_emp_eurostat(
        require_unit="THS_HW",
        na_item="EMP_DC",
        **common,
    )
    return {"employment": employment, "hours": hours}






