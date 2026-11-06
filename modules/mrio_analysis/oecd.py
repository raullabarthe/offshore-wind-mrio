"""OECD industrial labour-data download and source-resolution audit.

This module keeps OECD SDMX retrieval and provenance separate from the FTE40
resolver.  It downloads only bounded, explicit C2 overrides, preserves the
original dimensions, and reports whether a candidate offers additional FIGARO
A64 industrial detail.  The resolver remains responsible for the allocation
hierarchy and national FTE40 anchors.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import time
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import pycountry
import requests

from modules.config import EMPLOYMENT_DATA_DIR
from modules.mrio_analysis.source_utils import _fte40_area_to_iso2
from modules.mrio_analysis.employment import (
    _configured_resolution_targets,
    _effective_resolution_metadata,
)


OECD_SDMX_BASE = "https://sdmx.oecd.org/public/rest/v1"
OECD_CACHE_DIR = EMPLOYMENT_DATA_DIR / "oecd_cache"

# These are bounded, explicit production overrides.  The national FTE40
# anchor remains the independent ILOSTAT anchor; OECD data only provide
# industrial weights for the listed country/indicator/year combinations.
OECD_C2_ANNUAL_HOURS_SELECTION = {
    "AU": 2023, "BE": 2024, "CA": 2024, "DE": 2023, "EE": 2023,
    "FR": 2024, "LT": 2023, "LU": 2024, "NO": 2023, "SE": 2023,
}
OECD_C2_EMPLOYMENT_SELECTION = {
    "JP": 2021, "KR": 2023, "LU": 2024, "NO": 2023,
}
OECD_C2_SELECTION_FINGERPRINT = (
    "oecd-c2-annual-hours-v4-cross-source-relative-weights-complete-blocks:"
    "AU2023,BE2024,CA2024,DE2023,EE2023,FR2024,LT2023,LU2024,NO2023,SE2023;"
    "employment:JP2021,KR2023,LU2024,NO2023"
)

@dataclass(frozen=True)
class OECDDataset:
    key: str
    agency: str
    dataflow: str
    version: str = "latest"
    description: str = ""


OECD_DATASETS: tuple[OECDDataset, ...] = (
    OECDDataset(
        "table7",
        "OECD.SDD.NAD",
        "DSD_NAMAIN10@DF_TABLE7",
        description="Annual employment by detailed economic activity, domestic concept",
    ),
    OECDDataset(
        "stan2025",
        "OECD.STI.PIE",
        "DSD_STAN@DF_STAN_2025",
        description="STAN Database for Structural Analysis, 2025 edition",
    ),
    OECDDataset(
        "table3",
        "OECD.SDD.NAD",
        "DSD_NAMAIN10@DF_TABLE3_EMPDC",
        description="Annual employment by economic activity, domestic concept",
    ),
    OECDDataset(
        "productivity_isic4",
        "OECD.SDD.TPS",
        "DSD_PDB@DF_PDB_ISIC4_I4",
        version="1.0",
        description="Productivity by industry; currently registered as non-production external dataflow",
    ),
)


def _iso2(value) -> str | None:
    text = str(value).strip().upper()
    if text in {"ROW", "X01", "WORLD"}:
        return None
    if len(text) == 2 and pycountry.countries.get(alpha_2=text):
        return text
    country = pycountry.countries.get(alpha_3=text)
    return country.alpha_2 if country is not None else _fte40_area_to_iso2(text)


def _iso3(value) -> str | None:
    code = _iso2(value)
    country = pycountry.countries.get(alpha_2=code) if code else None
    return country.alpha_3 if country is not None else None


def oecd_country_scope(
    mrio_regions: Iterable[str],
    eu27: Iterable[str],
) -> pd.DataFrame:
    """Return the non-EU27 OECD query scope from the existing MRIO universe."""
    eu = {str(code).strip().upper() for code in eu27}
    rows = []
    for raw in sorted({str(code).strip().upper() for code in mrio_regions}):
        iso2 = _iso2(raw)
        iso3 = _iso3(raw)
        if iso2 is None or iso2 in eu or raw in {"ROW", "X01"}:
            continue
        rows.append({"geo": iso2, "iso3": iso3, "in_mrio": True})
    return pd.DataFrame(rows).drop_duplicates("geo").sort_values("geo").reset_index(drop=True)


def _oecd_requested_scope(
    mrio_regions: Iterable[str],
    eu27: Iterable[str],
    countries: Iterable[str] | None,
    *,
    include_eu27: bool = False,
) -> pd.DataFrame:
    """Build a bounded OECD scope without changing the MRIO universe."""
    if countries is None:
        return oecd_country_scope(mrio_regions, eu27)
    mrio = {str(code).strip().upper() for code in mrio_regions}
    eu = {str(code).strip().upper() for code in eu27}
    rows = []
    for raw in sorted({str(code).strip().upper() for code in countries}):
        iso2 = _iso2(raw)
        iso3 = _iso3(raw)
        if iso2 is None or iso2 not in mrio:
            continue
        if iso2 in eu and not include_eu27:
            continue
        rows.append({"geo": iso2, "iso3": iso3, "in_mrio": True})
    return pd.DataFrame(rows).drop_duplicates("geo").sort_values("geo").reset_index(drop=True)


class OECDSDMXClient:
    """Small cached OECD SDMX client using dataflow metadata for query keys."""

    def __init__(
        self,
        cache_dir: str | Path = OECD_CACHE_DIR,
        *,
        timeout: int = 120,
        retries: int = 3,
        min_interval: float = 0.75,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.timeout = int(timeout)
        self.retries = max(1, int(retries))
        self.min_interval = max(0.0, float(min_interval))
        self._last_request = 0.0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "FIGARO-FTE40-OECD-audit/1.0"})

    @staticmethod
    def _accept_structure() -> str:
        return "application/vnd.sdmx.structure+json;version=2.0"

    @staticmethod
    def _accept_data() -> str:
        return "text/csv"

    def _get(self, url: str, *, headers: Mapping[str, str], params: Mapping[str, str] | None = None):
        last_error = None
        for attempt in range(self.retries):
            elapsed = time.monotonic() - self._last_request
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            try:
                response = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )
                self._last_request = time.monotonic()
                if response.status_code in {429, 500, 502, 503, 504}:
                    retry_after = response.headers.get("Retry-After")
                    delay = min(
                        30.0,
                        float(retry_after) if retry_after and retry_after.isdigit() else 2.0 ** attempt,
                    )
                    time.sleep(delay)
                    continue
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                last_error = exc
                if attempt + 1 >= self.retries:
                    raise
                time.sleep(min(30.0, 2.0 ** attempt))
        raise RuntimeError(f"OECD request failed: {url}") from last_error

    @staticmethod
    def _cache_key(payload: Mapping) -> str:
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return sha256(encoded).hexdigest()[:24]

    def get_structure(self, dataset: OECDDataset) -> dict:
        payload = {
            "kind": "structure",
            "agency": dataset.agency,
            "dataflow": dataset.dataflow,
            "version": dataset.version,
        }
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"structure_{self._cache_key(payload)}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        url = (
            f"{OECD_SDMX_BASE}/dataflow/{dataset.agency}/"
            f"{dataset.dataflow}/{dataset.version}?references=all"
        )
        response = self._get(url, headers={"Accept": self._accept_structure()})
        structure = response.json()
        path.write_text(json.dumps(structure, ensure_ascii=False), encoding="utf-8")
        return structure

    @staticmethod
    def _data(structure: dict) -> dict:
        return structure.get("data", structure)

    def dimensions(self, dataset: OECDDataset) -> list[str]:
        data = self._data(self.get_structure(dataset))
        structures = data.get("dataStructures", [])
        if not structures:
            return []
        dimensions = structures[0].get("dataStructureComponents", {}).get(
            "dimensionList", {}
        ).get("dimensions", [])
        return [str(item["id"]) for item in sorted(dimensions, key=lambda x: x.get("position", 0))]

    def resolved_version(self, dataset: OECDDataset) -> str:
        """Return the concrete dataflow version advertised by the API."""
        dataflows = self._data(self.get_structure(dataset)).get("dataflows", [])
        if dataflows:
            return str(dataflows[0].get("version") or dataset.version)
        return dataset.version

    def dimension_codes(self, dataset: OECDDataset) -> dict[str, pd.DataFrame]:
        """Return metadata code lists for every dimension in the dataflow."""
        data = self._data(self.get_structure(dataset))
        codelists = {str(item.get("id")): item for item in data.get("codelists", [])}
        structures = data.get("dataStructures", [])
        if not structures:
            return {}
        dimensions = structures[0].get("dataStructureComponents", {}).get(
            "dimensionList", {}
        ).get("dimensions", [])
        out = {}
        for dimension in dimensions:
            enumeration = dimension.get("localRepresentation", {}).get("enumeration", "")
            match = re.search(r":([^:(]+)\([^)]*\)$", str(enumeration))
            codelist = codelists.get(match.group(1) if match else "")
            rows = []
            for code in (codelist or {}).get("codes", []):
                names = code.get("names", {})
                rows.append({"code": code.get("id"), "label": names.get("en", code.get("name", ""))})
            out[str(dimension["id"])] = pd.DataFrame(rows, columns=["code", "label"])
        return out

    def preferred_code(self, dataset: OECDDataset, dimension: str, label: str) -> str | None:
        """Resolve a query code from the published codelist label."""
        codes = self.dimension_codes(dataset).get(dimension, pd.DataFrame())
        if codes.empty:
            return None
        exact = codes.loc[codes["label"].astype(str).str.casefold().eq(label.casefold())]
        if not exact.empty:
            return str(exact.iloc[0]["code"])
        contains = codes.loc[codes["label"].astype(str).str.casefold().str.contains(label.casefold(), na=False)]
        return str(contains.iloc[0]["code"]) if not contains.empty else None

    def download(
        self,
        dataset: OECDDataset,
        *,
        countries: Iterable[str],
        year: int,
        filters: Mapping[str, str | Iterable[str]] | None = None,
    ) -> pd.DataFrame:
        """Download one bounded country-year slice while preserving raw fields."""
        dimensions = self.dimensions(dataset)
        if not dimensions:
            return pd.DataFrame()
        area_codes = sorted({code for code in (_iso3(value) for value in countries) if code})
        query_filters = dict(filters or {})
        query_filters.setdefault("FREQ", self.preferred_code(dataset, "FREQ", "Annual") or "A")
        query_filters["REF_AREA"] = area_codes
        key_parts = []
        for dimension in dimensions:
            value = query_filters.get(dimension, "")
            if isinstance(value, (tuple, list, set)):
                value = "+".join(sorted(str(item) for item in value))
            key_parts.append(str(value))
        key = ".".join(key_parts)
        data_version = self.resolved_version(dataset)
        payload = {
            "kind": "data",
            "agency": dataset.agency,
            "dataflow": dataset.dataflow,
            "version": data_version,
            "key": key,
            "startPeriod": int(year),
            "endPeriod": int(year),
        }
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        stem = f"data_{self._cache_key(payload)}"
        csv_path = self.cache_dir / f"{stem}.csv"
        meta_path = self.cache_dir / f"{stem}.json"
        if csv_path.exists():
            frame = pd.read_csv(csv_path)
            frame.attrs["request"] = payload
            return frame
        url = (
            f"{OECD_SDMX_BASE}/data/{dataset.agency},{dataset.dataflow},"
            f"{data_version}/{key}"
        )
        try:
            response = self._get(
                url,
                params={"startPeriod": str(year), "endPeriod": str(year)},
                headers={"Accept": self._accept_data()},
            )
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                frame = pd.DataFrame()
                frame.attrs["request"] = payload
                return frame
            raise
        if not response.content.strip():
            frame = pd.DataFrame()
        else:
            csv_path.write_bytes(response.content)
            frame = pd.read_csv(csv_path)
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        frame.attrs["request"] = payload
        return frame


def _year_column(frame: pd.DataFrame) -> str | None:
    return next((c for c in ("TIME_PERIOD", "time", "year") if c in frame.columns), None)


def _activity_column(frame: pd.DataFrame) -> str | None:
    return next((c for c in ("ACTIVITY", "activity", "SECTOR", "industry") if c in frame.columns), None)


def _status_summary(frame: pd.DataFrame) -> str:
    columns = [c for c in ("OBS_STATUS", "CONF_STATUS", "obs_status", "conf_status") if c in frame.columns]
    values = []
    for column in columns:
        values.extend(frame[column].dropna().astype(str).str.strip().unique().tolist())
    return ";".join(sorted(set(values)))


def _iso3_to_geo(frame: pd.DataFrame) -> pd.Series:
    area = frame.get("REF_AREA", pd.Series(index=frame.index, dtype="string"))
    return area.map(_iso2).astype("string")


def _semantic_rows(frame: pd.DataFrame, dataset_key: str) -> dict[str, pd.DataFrame]:
    """Separate total employment, employee-only, and annual-hours rows."""
    if frame.empty:
        return {"employment": frame, "employees": frame, "hours": frame}
    work = frame.copy()
    for column in work.columns:
        if work[column].dtype == object:
            work[column] = work[column].astype("string")
    if dataset_key in {"table7", "table3"}:
        transaction = work.get("TRANSACTION", pd.Series("", index=work.index)).astype(str).str.upper()
        unit = work.get("UNIT_MEASURE", pd.Series("", index=work.index)).astype(str).str.upper()
        hours = transaction.eq("EMP") & unit.isin({"H", "H_Y"})
        employment = transaction.eq("EMP") & unit.eq("PS")
        employees = transaction.eq("SAL") & ~hours
    elif dataset_key == "stan2025":
        measure = work.get("MEASURE", pd.Series("", index=work.index)).astype(str).str.upper()
        unit = work.get("UNIT_MEASURE", pd.Series("", index=work.index)).astype(str).str.upper()
        hours = measure.eq("EMP") & unit.isin({"H", "H_Y"})
        employment = measure.eq("EMP") & unit.eq("PS")
        employees = measure.eq("SAL") & unit.eq("PS")
    else:
        employment = pd.Series(False, index=work.index)
        employees = pd.Series(False, index=work.index)
        hours = pd.Series(False, index=work.index)
    return {
        "employment": work.loc[employment].copy(),
        "employees": work.loc[employees].copy(),
        "hours": work.loc[hours].copy(),
    }


def _compatibility(dataset_key: str, indicator: str, frame: pd.DataFrame) -> tuple[str, str, str]:
    if indicator == "employees":
        return "incompatible", "employees only", "Not total employment"
    if indicator == "hours":
        units = set(frame.get("UNIT_MEASURE", pd.Series(dtype=str)).dropna().astype(str).str.upper())
        population_column = "TRANSACTION" if "TRANSACTION" in frame else "MEASURE"
        population_values = set(
            frame.get(population_column, pd.Series(dtype=str))
            .dropna()
            .astype(str)
            .str.upper()
        )
        expected_unit = {"H", "H_Y"}
        if units and units <= expected_unit and population_values <= {"EMP"}:
            return "conditional", "total employment annual hours", "Annual-hours reference convention must be benchmarked to FTE40"
        return "incompatible", "not total annual hours", "Requires EMP and an additive annual total-hours unit"
    units = set(frame.get("UNIT_MEASURE", pd.Series(dtype=str)).dropna().astype(str).str.upper())
    population_column = "TRANSACTION" if "TRANSACTION" in frame else "MEASURE"
    population_values = set(frame.get(population_column, pd.Series(dtype=str)).dropna().astype(str).str.upper())
    if units and (units != {"PS"} or population_values != {"EMP"}):
        return "incompatible", "non-total employment", "Requires EMP and PS"
    return "conditional", "total employment persons", "Employment concept requires source review"


def _total_employment_query_filters(
    dataset: OECDDataset,
    dimension_catalog: Mapping[str, pd.DataFrame],
) -> dict[str, str]:
    """Restrict every OECD audit request to total employment (EMP)."""
    filters: dict[str, str] = {}
    if "TRANSACTION" in dimension_catalog:
        filters["TRANSACTION"] = "EMP"
    elif "MEASURE" in dimension_catalog:
        filters["MEASURE"] = "EMP"
    if "UNIT_MEASURE" in dimension_catalog:
        unit_codes = set(
            dimension_catalog["UNIT_MEASURE"].get("code", pd.Series(dtype=str))
            .dropna().astype(str).str.upper()
        )
        requested_units = sorted(unit_codes & {"H", "H_Y", "PS"})
        if requested_units:
            filters["UNIT_MEASURE"] = requested_units
    return filters


def audit_oecd_labor_sources(
    mrio_regions: Iterable[str],
    eu27: Iterable[str],
    *,
    years: Iterable[int] = (2024, 2023, 2022, 2021),
    datasets: Iterable[OECDDataset] = OECD_DATASETS,
    target_industries: Iterable[str] | None = None,
    client: OECDSDMXClient | None = None,
    countries: Iterable[str] | None = None,
    indicators: Iterable[str] | None = None,
    include_eu27: bool = False,
    query_units: Iterable[str] | None = None,
    country_years: Mapping[int, Iterable[str]] | None = None,
) -> dict[str, pd.DataFrame | dict]:
    """Download and audit bounded OECD candidates without touching production inputs."""
    client = client or OECDSDMXClient()
    scope = _oecd_requested_scope(
        mrio_regions,
        eu27,
        countries,
        include_eu27=include_eu27,
    )
    targets = set(target_industries or _configured_resolution_targets())
    requested_indicators = {
        str(value).strip().lower()
        for value in (indicators or ("employment", "hours"))
    }
    raw_frames: dict[tuple[str, int], pd.DataFrame] = {}
    inventory_rows = []
    coverage_rows = []
    diagnostic_rows = []
    failures = []

    for dataset in datasets:
        try:
            dimensions = client.dimensions(dataset)
            dimension_catalog = client.dimension_codes(dataset)
        except Exception as exc:
            failures.append({"dataset": dataset.key, "stage": "structure", "error": repr(exc)})
            dimensions = []
            dimension_catalog = {}
        for year in years:
            query_countries = set(scope["geo"].astype(str))
            if country_years is not None:
                query_countries &= {
                    str(value).strip().upper()
                    for value in country_years.get(int(year), ())
                }
            if not dimensions:
                raw = pd.DataFrame()
            else:
                try:
                    filters = _total_employment_query_filters(
                        dataset, dimension_catalog
                    )
                    if query_units is not None and "UNIT_MEASURE" in dimension_catalog:
                        available_units = set(
                            dimension_catalog["UNIT_MEASURE"].get("code", pd.Series(dtype=str))
                            .dropna().astype(str).str.upper()
                        )
                        units = sorted(available_units & {
                            str(value).strip().upper() for value in query_units
                        })
                        if not units:
                            raw = pd.DataFrame()
                            raw_frames[(dataset.key, int(year))] = raw
                            continue
                        filters["UNIT_MEASURE"] = units
                    raw = client.download(
                        dataset,
                        countries=query_countries,
                        year=int(year),
                        filters=filters,
                    )
                except Exception as exc:
                    failures.append({"dataset": dataset.key, "year": int(year), "stage": "data", "error": repr(exc)})
                    raw = pd.DataFrame()
            raw_frames[(dataset.key, int(year))] = raw
            semantics = _semantic_rows(raw, dataset.key)
            diagnostic_rows.extend(_annual_hours_diagnostics(dataset, year, semantics))
            for indicator, indicator_frame in semantics.items():
                if indicator not in requested_indicators:
                    continue
                if indicator_frame.empty and indicator not in {"employment", "hours"}:
                    continue
                if indicator_frame.empty:
                    for geo in scope["geo"]:
                        inventory_rows.append(_empty_inventory_row(dataset, year, geo, indicator, dimensions))
                    continue
                activity_col = _activity_column(indicator_frame)
                year_col = _year_column(indicator_frame)
                if activity_col is None:
                    continue
                work = indicator_frame.copy()
                work["geo"] = _iso3_to_geo(work)
                work = work.loc[work["geo"].isin(scope["geo"])].copy()
                if year_col is not None:
                    work["source_year"] = pd.to_numeric(work[year_col], errors="coerce")
                else:
                    work["source_year"] = int(year)
                work["source_code"] = work[activity_col].astype(str).str.strip()
                work["nace_r2"] = work["source_code"]
                work["class_system"] = "ISIC4"
                work["source_db"] = "OECD"
                work["dataflow_id"] = dataset.dataflow
                population_column = "TRANSACTION" if "TRANSACTION" in work else "MEASURE"
                population_values = set(
                    work.get(population_column, pd.Series(dtype=str))
                    .dropna()
                    .astype(str)
                    .str.upper()
                )
                if indicator == "employees":
                    population_scope = "employees only"
                elif indicator == "hours" and population_values <= {"SAL"}:
                    population_scope = "employees only"
                elif indicator == "hours" and population_values <= {"EMP"}:
                    population_scope = "total employed"
                elif dataset.key in {"table7", "table3"}:
                    population_scope = "national accounts domestic concept"
                elif indicator == "employment":
                    population_scope = "STAN total employment"
                else:
                    population_scope = pd.NA
                work["population_scope"] = population_scope
                value_column = "OBS_VALUE" if "OBS_VALUE" in work.columns else None
                valid = work.loc[
                    work["source_code"].str.upper().ne("_T")
                    & work["source_code"].str.upper().ne("TOTAL")
                    & work["source_code"].ne("")
                    & (work[value_column].notna() if value_column else False)
                ].copy()
                group_columns = ["geo"]
                for column in ("UNIT_MEASURE", "MEASURE"):
                    if column in valid.columns and column not in group_columns:
                        group_columns.append(column)
                for group_key, group in valid.groupby(group_columns, sort=True, dropna=False):
                    geo = group_key[0] if isinstance(group_key, tuple) else group_key
                    metadata, mapped = _effective_resolution_metadata(
                        group,
                        target_industries=targets,
                        value_name="OBS_VALUE",
                        source_name="OECD",
                    )
                    compatible, population, compatibility_note = _compatibility(dataset.key, indicator, group)
                    covered = set().union(
                        *(set(values) for values in mapped.get("targets", []) if isinstance(values, tuple))
                    ) if not mapped.empty else set()
                    covered &= targets
                    inventory_rows.append({
                        "geo": geo,
                        "year": int(year),
                        "dataset": dataset.key,
                        "agency": dataset.agency,
                        "dataflow_id": dataset.dataflow,
                        "indicator": indicator,
                        "description": dataset.description,
                        "valid_source_industries": int(group["source_code"].nunique()),
                        "nominal_classification": "ISIC Rev. 4",
                        "resolution_profile": metadata["resolution_profile"],
                        "exact_A64": metadata["exact_A64"],
                        "effective_A64_blocks": metadata["effective_resolution"],
                        "independent_blocks": metadata["independent_blocks"],
                        "A64_covered": int(len(covered)),
                        "A64_missing": ";".join(sorted(targets - covered)),
                        "available_parent_aggregates": ";".join(sorted(
                            mapped.loc[~mapped["source_is_exact"], "source_code"].astype(str).unique()
                        )) if not mapped.empty else "",
                        "original_codes": ";".join(sorted(group["source_code"].astype(str).unique())),
                        "observation_status": _status_summary(group),
                        "unit_measure": ";".join(sorted(
                            group.get("UNIT_MEASURE", pd.Series(dtype=str)).dropna().astype(str).unique()
                        )),
                        "compatible": compatible,
                        "population_scope": population,
                        "compatibility_note": compatibility_note,
                        "selected": False,
                    })
                    for code in sorted(group["source_code"].astype(str).unique()):
                        coverage_rows.append({
                            "geo": geo,
                            "year": int(year),
                            "dataset": dataset.key,
                            "indicator": indicator,
                            "original_industry_code": code,
                            "mapped_A64": ";".join(sorted(
                                mapped.loc[mapped["source_code"].eq(code), "targets"].explode().dropna().astype(str).unique()
                            )) if not mapped.empty else "",
                            "is_exact_A64": bool(
                                not mapped.empty and mapped.loc[mapped["source_code"].eq(code), "source_is_exact"].any()
                            ),
                            "observation_status": _status_summary(group.loc[group["source_code"].eq(code)]),
                        })

    inventory = pd.DataFrame(inventory_rows)
    coverage = pd.DataFrame(coverage_rows)
    selected_inventory = select_oecd_inventory_candidates(inventory, scope)
    if not inventory.empty:
        inventory["selected"] = False
        if not selected_inventory.empty:
            inventory.loc[selected_inventory.index, "selected"] = True
    return {
        "scope": scope,
        "inventory": inventory,
        "selected_inventory": selected_inventory,
        "coverage": coverage,
        "annual_hours_diagnostics": pd.DataFrame(diagnostic_rows),
        "failures": pd.DataFrame(failures),
        "raw_frames": raw_frames,
        "structures": {
            dataset.key: {
                "dimensions": client.dimensions(dataset),
                "dimension_codes": client.dimension_codes(dataset),
            }
            for dataset in datasets
            if dataset.key not in {row.get("dataset") for row in failures if row.get("stage") == "structure"}
        },
    }


def _empty_inventory_row(dataset: OECDDataset, year: int, geo: str, indicator: str, dimensions: list[str]) -> dict:
    compatible, population, note = _compatibility(dataset.key, indicator, pd.DataFrame())
    return {
        "geo": geo, "year": int(year), "dataset": dataset.key,
        "agency": dataset.agency, "dataflow_id": dataset.dataflow,
        "indicator": indicator, "description": dataset.description,
        "valid_source_industries": 0, "nominal_classification": "ISIC Rev. 4",
        "resolution_profile": None, "exact_A64": 0, "effective_A64_blocks": 0,
        "independent_blocks": 0, "A64_covered": 0, "A64_missing": "",
        "available_parent_aggregates": "", "original_codes": "",
        "observation_status": "", "compatible": "unavailable",
        "population_scope": population, "compatibility_note": note,
        "selected": False, "dimensions": ";".join(dimensions),
    }


def _annual_hours_diagnostics(
    dataset: OECDDataset,
    year: int,
    semantics: Mapping[str, pd.DataFrame],
) -> list[dict]:
    """Calculate annual-hours-per-employment diagnostics without FTE conversion."""
    employment = semantics.get("employment", pd.DataFrame()).copy()
    hours = semantics.get("hours", pd.DataFrame()).copy()
    if employment.empty or hours.empty or "ACTIVITY" not in employment.columns:
        return []
    employment = employment.loc[employment["UNIT_MEASURE"].astype(str).str.upper().isin({"PS", "JB"})].copy()
    allowed_hours = {"H", "H_Y"}
    hours = hours.loc[hours["UNIT_MEASURE"].astype(str).str.upper().isin(allowed_hours)].copy()
    population_column = "TRANSACTION" if "TRANSACTION" in hours.columns else "MEASURE"
    if population_column in hours.columns:
        hours = hours.loc[
            hours[population_column].astype(str).str.upper().eq("EMP")
        ].copy()
    if employment.empty or hours.empty:
        return []
    def scaled_values(frame: pd.DataFrame) -> pd.Series:
        multiplier = pd.to_numeric(
            frame.get("UNIT_MULT", pd.Series(0, index=frame.index)),
            errors="coerce",
        ).fillna(0.0)
        return pd.to_numeric(frame["OBS_VALUE"], errors="coerce") * np.power(10.0, multiplier)

    employment["_scaled_value"] = scaled_values(employment)
    hours["_scaled_value"] = scaled_values(hours)
    key = ["REF_AREA", "ACTIVITY"]
    emp = employment.groupby(key, as_index=False)["_scaled_value"].sum().rename(columns={"_scaled_value": "employment_value"})
    total_units = {"H", "H_Y"}
    hrs_total = hours.loc[hours["UNIT_MEASURE"].astype(str).str.upper().isin(total_units), key + ["_scaled_value"]]
    hrs_total = hrs_total.groupby(key, as_index=False)["_scaled_value"].sum().rename(columns={"_scaled_value": "annual_hours_total"})
    merged = emp.merge(hrs_total, on=key, how="inner")
    merged["annual_hours_per_person"] = merged["annual_hours_total"] / merged["employment_value"].replace(0, np.nan)
    rows = []
    for row in merged.dropna(subset=["annual_hours_per_person"]).itertuples(index=False):
        rows.append({
            "geo": _iso2(row.REF_AREA),
            "year": int(year),
            "dataset": dataset.key,
            "activity_code": row.ACTIVITY,
            "employment_value": float(row.employment_value),
            "annual_hours_per_person": float(row.annual_hours_per_person),
            "diagnostic_only": True,
            "fte40_conversion_applied": False,
            "note": "Diagnostic ratio only; no national FTE40 anchor or annual reference-hours conversion applied",
        })
    return rows

def select_oecd_inventory_candidates(
    inventory: pd.DataFrame,
    scope: pd.DataFrame,
    *,
    reference_year: int = 2024,
    prefer_latest_year: bool = False,
    include_incompatible: bool = False,
) -> pd.DataFrame:
    """Select one compatible maximum-resolution candidate per indicator."""
    if inventory is None or inventory.empty:
        return pd.DataFrame()
    status = {"conditional", "compatible"}
    if include_incompatible:
        status.add("incompatible")
    eligible = inventory.loc[
        inventory["compatible"].isin(status)
        & inventory["indicator"].isin({"hours", "employment"})
    ].copy()
    if include_incompatible and not eligible.empty:
        not_annual_hours = eligible["compatibility_note"].astype(str).eq(
            "Not annual hours actually worked per employed person"
        )
        eligible = eligible.loc[~not_annual_hours].copy()
    if eligible.empty:
        return eligible
    eligible["year_numeric"] = pd.to_numeric(eligible["year"], errors="coerce")
    eligible["effective_numeric"] = pd.to_numeric(
        eligible["effective_A64_blocks"], errors="coerce"
    ).fillna(0)
    eligible["year_distance"] = (reference_year - eligible["year_numeric"]).abs()
    eligible["compatibility_rank"] = eligible["compatible"].map(
        {"compatible": 0, "conditional": 1, "incompatible": 2}
    ).fillna(3)
    selected_indices = []
    for geo in scope["geo"].astype(str).str.upper():
        for indicator in ("hours", "employment"):
            candidates = eligible.loc[
                eligible["geo"].astype(str).str.upper().eq(geo)
                & eligible["indicator"].eq(indicator)
            ]
            if candidates.empty:
                continue
            if prefer_latest_year:
                sort_columns = [
                    "compatibility_rank",
                    "year_distance",
                    "effective_numeric",
                    "year_numeric",
                    "dataset",
                ]
                ascending = [True, True, False, False, True]
            else:
                sort_columns = [
                    "compatibility_rank",
                    "effective_numeric",
                    "year_distance",
                    "year_numeric",
                    "dataset",
                ]
                ascending = [True, False, True, False, True]
            chosen = candidates.sort_values(sort_columns, ascending=ascending).iloc[0]
            selected_indices.append(chosen.name)
    return eligible.loc[selected_indices].copy() if selected_indices else eligible.iloc[0:0].copy()


def build_oecd_resolution_tables(
    inventory: pd.DataFrame,
    scope: pd.DataFrame,
    *,
    reference_year: int = 2024,
    prefer_latest_year: bool = False,
    include_incompatible: bool = False,
) -> dict[str, pd.DataFrame]:
    """Build the compact country-level OECD inventory tables."""
    selected = select_oecd_inventory_candidates(
        inventory,
        scope,
        reference_year=reference_year,
        prefer_latest_year=prefer_latest_year,
        include_incompatible=include_incompatible,
    )
    labels = {
        "hours": "OECD \u2014 Annual total hours",
        "employment": "OECD \u2014 Industrial employment",
    }
    tables = {}
    for indicator, title in labels.items():
        rows = []
        selected_indicator = (
            selected.loc[selected["indicator"].eq(indicator)]
            if "indicator" in selected.columns
            else pd.DataFrame()
        )
        for geo in scope["geo"].astype(str).str.upper():
            candidate = (
                selected_indicator.loc[
                    selected_indicator["geo"].astype(str).str.upper().eq(geo)
                ]
                if "geo" in selected_indicator.columns
                else pd.DataFrame()
            )
            if candidate.empty:
                rows.append(
                    {
                        "Country": geo,
                        "Year": pd.NA,
                        "Classification": "Unavailable",
                        "Valid source industries": 0,
                        "Resolution": 0,
                    }
                )
                continue
            row = candidate.iloc[0]
            rows.append(
                {
                    "Country": geo,
                    "Year": int(row["year"]),
                    "Classification": row["nominal_classification"],
                    "Valid source industries": int(row["valid_source_industries"]),
                    "Resolution": int(row["effective_A64_blocks"]),
                }
            )
        tables[title] = pd.DataFrame(rows).sort_values("Country").reset_index(drop=True)
    return tables


def _oecd_production_frame(
    raw: pd.DataFrame,
    *,
    dataset: OECDDataset,
    indicator: str,
    geo: str,
    unit_measure: str,
    year: int,
) -> pd.DataFrame:
    """Convert one selected OECD EMP slice to the FTE40 input schema."""
    semantic = _semantic_rows(raw, dataset.key).get(indicator, pd.DataFrame()).copy()
    if semantic.empty:
        return pd.DataFrame()
    activity_column = _activity_column(semantic)
    if activity_column is None:
        return pd.DataFrame()
    semantic["geo"] = _iso3_to_geo(semantic)
    semantic = semantic.loc[
        semantic["geo"].eq(geo)
        & semantic["UNIT_MEASURE"].astype(str).str.upper().eq(unit_measure.upper())
    ].copy()
    if semantic.empty:
        return pd.DataFrame()
    values = pd.to_numeric(semantic.get("OBS_VALUE"), errors="coerce")
    multiplier = pd.to_numeric(
        semantic.get("UNIT_MULT", pd.Series(0, index=semantic.index)),
        errors="coerce",
    ).fillna(0.0)
    # The FTE40 source schema stores both employment and annual hours in
    # thousands. OECD ``PS`` with UNIT_MULT=3 is already thousand persons;
    # OECD ``H`` with UNIT_MULT=6 is million hours and therefore becomes
    # thousand hours after a 10**(6-3) conversion. Converting to absolute
    # units here would inflate the exported source observations by 1,000;
    # it does not belong in the relative industrial-weight calculation.
    semantic["OBS_VALUE"] = values * np.power(10.0, multiplier - 3.0)
    semantic = semantic.loc[semantic["OBS_VALUE"].notna()].copy()
    semantic["source_code"] = semantic[activity_column].astype(str).str.strip()
    semantic["source_code"] = semantic["source_code"].replace({"_T": "TOTAL", "_T_": "TOTAL"})
    semantic["original_source_code"] = semantic[activity_column].astype(str).str.strip()
    semantic["year"] = pd.to_numeric(
        semantic.get("TIME_PERIOD", pd.Series(year, index=semantic.index)),
        errors="coerce",
    ).fillna(year).astype("Int64")
    semantic["unit"] = "THS_HOURS_YEAR" if indicator == "hours" else "THS_PER"
    semantic["source_db"] = f"OECD {dataset.key}"
    semantic["dataflow_id"] = f"OECD:{dataset.dataflow}"
    semantic["class_system"] = "ISIC4"
    semantic["population_scope"] = "all employed persons (OECD EMP)"
    semantic["key_used"] = semantic.attrs.get("request", {}).get("key", pd.NA)
    columns = [
        "geo", "year", "source_code", "original_source_code", "OBS_VALUE",
        "unit", "source_db", "dataflow_id", "class_system",
        "population_scope", "key_used",
    ]
    return semantic.reindex(columns=columns).drop_duplicates(
        ["geo", "year", "source_code"], keep="first"
    )


def _build_oecd_selected_tables(selected: pd.DataFrame) -> dict[str, pd.DataFrame]:
    labels = {
        "hours": "OECD \u2014 Annual total hours",
        "employment": "OECD \u2014 Industrial employment",
    }
    tables = {}
    for indicator, title in labels.items():
        rows = []
        subset = (
            selected.loc[selected["indicator"].eq(indicator)].copy()
            if "indicator" in selected.columns
            else pd.DataFrame()
        )
        for row in subset.sort_values("geo").itertuples(index=False):
            rows.append(
                {
                    "Country": row.geo,
                    "Year": int(row.year),
                    "Classification": row.nominal_classification,
                    "Valid source industries": int(row.valid_source_industries),
                    "Resolution": int(row.effective_A64_blocks),
                }
            )
        tables[title] = pd.DataFrame(
            rows,
            columns=[
                "Country", "Year", "Classification",
                "Valid source industries", "Resolution",
            ],
        )
    return tables


def audit_oecd_c2_sources(
    mrio_regions: Iterable[str],
    eu27: Iterable[str],
    *,
    target_industries: Iterable[str] | None = None,
    client: OECDSDMXClient | None = None,
) -> dict[str, object]:
    """Audit and load only the explicit OECD C2 production overrides."""
    client = client or OECDSDMXClient()
    mrio = {str(value).strip().upper() for value in mrio_regions}
    selected_maps = {
        "hours": {
            geo: year for geo, year in OECD_C2_ANNUAL_HOURS_SELECTION.items()
            if geo in mrio
        },
        "employment": {
            geo: year for geo, year in OECD_C2_EMPLOYMENT_SELECTION.items()
            if geo in mrio
        },
    }
    countries = sorted(set().union(*(mapping for mapping in selected_maps.values())))
    scope = _oecd_requested_scope(
        mrio_regions, eu27, countries, include_eu27=True,
    )
    targets = set(target_industries or _configured_resolution_targets())
    audit_parts = {}
    for indicator, mapping in selected_maps.items():
        years = sorted(set(mapping.values()))
        country_years = {
            year: [geo for geo, selected_year in mapping.items() if selected_year == year]
            for year in years
        }
        audit_parts[indicator] = audit_oecd_labor_sources(
            mrio_regions,
            eu27,
            years=years,
            datasets=OECD_DATASETS[:3],
            target_industries=targets,
            client=client,
            countries=mapping,
            indicators=(indicator,),
            include_eu27=True,
            query_units={"H", "H_Y"} if indicator == "hours" else {"PS"},
            country_years=country_years,
        )

    inventory_parts = []
    for indicator, part in audit_parts.items():
        inventory = part["inventory"]
        if inventory.empty:
            continue
        mapping = selected_maps[indicator]
        inventory_parts.append(
            inventory.loc[
                inventory["indicator"].eq(indicator)
                & inventory["geo"].isin(mapping)
                & inventory.apply(
                    lambda row: int(row["year"]) == int(mapping.get(row["geo"], -1)),
                    axis=1,
                )
            ].copy()
        )
    inventory = (
        pd.concat(inventory_parts, ignore_index=True, sort=False)
        if inventory_parts else pd.DataFrame()
    )
    selected = select_oecd_inventory_candidates(inventory, scope)
    for frame in (inventory, selected):
        if frame.empty:
            continue
        frame["source"] = "OECD"
        frame["source_db"] = frame["dataset"].map(
            lambda value: f"OECD {value}"
        )
        frame["dataflow_id"] = frame["dataflow_id"].map(
            lambda value: value
            if str(value).upper().startswith("OECD:")
            else f"OECD:{value}"
        )
        frame["source_year"] = frame["year"]
        frame["class_system"] = frame["nominal_classification"]
        frame["measure"] = np.where(
            frame["indicator"].eq("hours"),
            "OECD annual total hours",
            "OECD employment",
        )
    if not inventory.empty:
        inventory["selected"] = False
        if not selected.empty:
            keys = set(
                zip(
                    selected["geo"], selected["indicator"],
                    selected["year"], selected["dataset"],
                )
            )
            inventory["selected"] = inventory.apply(
                lambda row: (
                    row["geo"], row["indicator"], row["year"], row["dataset"]
                ) in keys,
                axis=1,
            )
            selected["selected"] = True

    dataset_by_key = {dataset.key: dataset for dataset in OECD_DATASETS[:3]}
    production_inputs = {"hours": [], "employment": []}
    source_frames = {"hours": audit_parts["hours"]["raw_frames"],
                     "employment": audit_parts["employment"]["raw_frames"]}
    for row in selected.to_dict("records"):
        dataset = dataset_by_key[row["dataset"]]
        frame = _oecd_production_frame(
            source_frames[row["indicator"]][(dataset.key, int(row["year"]))],
            dataset=dataset,
            indicator=row["indicator"],
            geo=str(row["geo"]),
            unit_measure=str(row["unit_measure"]).split(";")[0],
            year=int(row["year"]),
        )
        if not frame.empty:
            production_inputs[row["indicator"]].append(frame)
    production_inputs = {
        indicator: (
            pd.concat(frames, ignore_index=True)
            if frames else pd.DataFrame()
        )
        for indicator, frames in production_inputs.items()
    }
    missing = []
    for indicator, mapping in selected_maps.items():
        selected_keys = (
            set(
                zip(
                    selected.loc[selected["indicator"].eq(indicator), "geo"],
                    selected.loc[selected["indicator"].eq(indicator), "year"],
                )
            )
            if {"indicator", "geo", "year"}.issubset(selected.columns)
            else set()
        )
        missing.extend(
            {"geo": geo, "indicator": indicator, "year": year}
            for geo, year in mapping.items()
            if (geo, year) not in selected_keys
        )
    audit = {
        "scope": scope,
        "inventory": inventory,
        "selected_inventory": selected,
        "coverage": pd.concat(
            [part["coverage"] for part in audit_parts.values()],
            ignore_index=True, sort=False,
        ),
        "failures": pd.concat(
            [part["failures"] for part in audit_parts.values()],
            ignore_index=True, sort=False,
        ),
        "missing_selections": pd.DataFrame(missing),
        "raw_frames": source_frames,
        "selection_fingerprint": OECD_C2_SELECTION_FINGERPRINT,
    }
    return {
        "audit": audit,
        "tables": _build_oecd_selected_tables(selected),
        "production_inputs": production_inputs,
    }


def apply_oecd_c2_production_overrides(
    *,
    eurostat_employment: pd.DataFrame | None,
    ilostat_employment: pd.DataFrame | None,
    eurostat_hours: pd.DataFrame | None,
    oecd_c2_audit: dict[str, object],
) -> dict[str, pd.DataFrame]:
    """Replace only the explicitly selected production countries with OECD data."""
    production = oecd_c2_audit["production_inputs"]
    employment_override = production.get("employment", pd.DataFrame())
    hours_override = production.get("hours", pd.DataFrame())
    employment_countries = set(
        employment_override.get("geo", pd.Series(dtype=str)).astype(str).str.upper()
    )
    hours_countries = set(
        hours_override.get("geo", pd.Series(dtype=str)).astype(str).str.upper()
    )

    def replace_countries(
        base: pd.DataFrame | None,
        override: pd.DataFrame,
        countries: set[str],
    ) -> pd.DataFrame:
        base = base.copy() if base is not None else pd.DataFrame()
        if "geo" in base.columns:
            base = base.loc[
                ~base["geo"].astype(str).str.upper().isin(countries)
            ].copy()
        if override is not None and not override.empty:
            base = pd.concat([base, override], ignore_index=True, sort=False)
        return base

    return {
        "eurostat_employment": replace_countries(
            eurostat_employment, pd.DataFrame(), employment_countries
        ),
        "ilostat_employment": replace_countries(
            ilostat_employment, employment_override, employment_countries
        ),
        "eurostat_hours": replace_countries(
            eurostat_hours, hours_override, hours_countries
        ),
    }

__all__ = [
    "OECD_DATASETS",
    "OECDDataset",
    "OECDSDMXClient",
    "OECD_C2_ANNUAL_HOURS_SELECTION",
    "OECD_C2_EMPLOYMENT_SELECTION",
    "OECD_C2_SELECTION_FINGERPRINT",
    "audit_oecd_labor_sources",
    "audit_oecd_c2_sources",
    "apply_oecd_c2_production_overrides",
    "select_oecd_inventory_candidates",
    "build_oecd_resolution_tables",
    "oecd_country_scope",
]
