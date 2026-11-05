"""Shared source-loading utilities for Eurostat and ILOSTAT.

This module intentionally contains no industrial-allocation methodology.  It
holds common HTTP/CSV handling, source-schema definitions, and country-list
normalization needed by independent source loaders.
"""

from __future__ import annotations

import time
import re
import unicodedata
from functools import lru_cache
from io import StringIO

import pandas as pd
import pycountry
import requests
from pandas.errors import EmptyDataError, ParserError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_SESSION = None

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

# Shared normalized industrial-source schema used by Eurostat and ILOSTAT.
_INDUSTRIAL_SOURCE_COLUMNS = [
    "geo", "year", "source_code", "original_source_code", "OBS_VALUE",
    "unit", "source_db", "dataflow_id", "class_system",
    "population_scope", "key_used",
]
_INDUSTRIAL_SOURCE_DTYPES = {
    "geo": "string", "year": "Int64", "source_code": "string",
    "original_source_code": "string", "OBS_VALUE": "Float64", "unit": "string",
    "source_db": "string", "dataflow_id": "string", "class_system": "string",
    "population_scope": "string", "key_used": "string",
}

# Shared country and column helpers used by source loaders and allocation.
def _fte40_explicit_countries(CORE, *, include_world: bool = False) -> list[str]:
    """Return country codes from CORE; statistical aggregates are excluded.

    ``X01`` is an ILOSTAT statistical aggregate rather than an ISO country.
    It is therefore opt-in so the existing national loaders keep their
    historical behaviour unless the World audit/allocation explicitly asks
    for it.
    """
    explicit = []
    for value in CORE:
        code = str(value).strip().upper()
        country = pycountry.countries.get(alpha_2=code)
        if country is not None:
            explicit.append(country.alpha_2)
    if include_world and "X01" in {str(value).strip().upper() for value in CORE}:
        explicit.append("X01")
    return sorted(set(explicit))

def _normalise_area_label(value) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Z0-9]+", " ", text.upper()).strip()


@lru_cache(maxsize=1)
def _country_label_to_iso2() -> dict[str, str]:
    """Build country-name aliases from pycountry without a manual country list."""
    aliases: dict[str, str] = {}
    for country in pycountry.countries:
        names = {
            getattr(country, "alpha_2", ""),
            getattr(country, "alpha_3", ""),
            getattr(country, "name", ""),
            getattr(country, "official_name", ""),
            getattr(country, "common_name", ""),
        }
        for name in names:
            if name:
                aliases[_normalise_area_label(name)] = country.alpha_2
                if "," in str(name):
                    left, right = str(name).split(",", 1)
                    aliases[_normalise_area_label(f"{right} {left}")] = country.alpha_2
    return aliases


@lru_cache(maxsize=4096)
def _fte40_area_to_iso2(value) -> str | None:
    """Convert ILO ISO3/country labels to FIGARO ISO2, retaining X01 as World."""
    raw = str(value).strip()
    if raw.upper() in {"X01", "WORLD", "WORLD TOTAL", "THE WORLD"}:
        return "X01"

    country = pycountry.countries.get(alpha_3=raw.upper())
    if country is not None:
        return country.alpha_2
    country = pycountry.countries.get(alpha_2=raw.upper())
    if country is not None:
        return country.alpha_2
    direct = _country_label_to_iso2().get(_normalise_area_label(raw))
    if direct is not None:
        return direct
    # Do not fuzzy-match statistical aggregates to a country.  The aliases
    # above cover the country names used by the ILO bulk indicator, including
    # comma-reversed names such as ``Republic of Korea``.
    return None
