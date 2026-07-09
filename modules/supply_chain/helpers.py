"""
Country utilities and constants.

Contains country name mappings, ISO codes, geographic data, and color schemes.
Transversal module used across supply chain and economic impact analysis.
"""

from __future__ import annotations

import math
import os.path as op
from collections import defaultdict

import pandas as pd
from matplotlib import colors as mcolors

from modules.config import SUPPLY_SOURCES
from .inputs import import_category, import_shares

SUPP_ROOT = str(SUPPLY_SOURCES)

# ============================================================================
# COUNTRY NAME TO ISO2 CODE MAPPING
# ============================================================================

NAME_TO_CODE = {
    "Australia": "AU", "Austria": "AT", "Åland Islands": "FI", "Belgium": "BE", "Bosnia & Herzegovina": "BA", 
    "Brazil": "BR", "Bulgaria": "BG", "Canada": "CA", "Croatia": "HR", "Cyprus": "CY", "Czech Republic": "CZ",
    "Denmark": "DK", "Estonia": "EE", "Finland": "FI", "France": "FR", "Germany": "DE", "Greece": "GR", "Guernsey": "GB", 
    "India": "IN", "Ireland": "IE", "Isle of Man": "GB", "Italy": "IT", "Japan": "JP", "Jersey": "GB", "Liechtenstein": "LI",
    "Lithuania": "LT", "Luxembourg": "LU", "Malaysia": "MY", "Malta": "MT", "Netherlands": "NL", "Norway": "NO", "Oman": "OM",
    "People's Republic of China": "CN", "Poland": "PL", "Portugal": "PT", "Republic of China (Taiwan)": "TW", "Reunion": "FR",
    "Romania": "RO", "Russia": "RU", "Serbia": "RS", "Singapore": "SG", "Slovakia": "SK", "Slovenia": "SI", "South Africa": "ZA",
    "South Korea": "KR", "Spain": "ES", "Sweden": "SE", "Switzerland": "CH", "Thailand": "TH", "Turkey": "TR", "Ukraine": "UA",
    "United Arab Emirates": "AE", "United Kingdom": "GB", "United States of America": "US"
}

# ============================================================================
# ISO2 CODE TO COUNTRY NAME MAPPING
# ============================================================================

CODE_TO_NAME = {v: k for k, v in NAME_TO_CODE.items()}

# Regional aliases (used in disambiguation)
REGION_ALIASES = {"Reunion", "Guernsey", "Åland Islands", "Jersey", "Isle of Man"}

# Build reverse lookup: code -> [names]
_names_by_code = defaultdict(list)
for name, code in NAME_TO_CODE.items():
    _names_by_code[code].append(name)


# ============================================================================
# GEOGRAPHIC DATA
# ============================================================================

# European ISO2 codes present in datasets
EUROPE_CODES = {
    "AT", "AX", "BE", "CH", "CY", "DE", "DK", "EE", "ES", "FI", "FR",
    "GG", "GR", "IE", "IM", "IT", "JE", "LI", "LT", "LU", "NL", "NO",
    "PL", "PT", "RO", "SE", "SI", "GB",
}

# Geographic coordinates for country centroids (used in distance calcs)
COUNTRY_COORDS = {
    "BE": (50.503887, 4.469936),
    "NL": (52.132633, 5.291266),
    "DK": (56.2639, 9.5018),
    "BR": (42.733883, 25.485830),
    "GB": (55.378051, -3.435973),
    "DE": (51.1657, 10.4515),
    "FR": (46.227638, 2.213749),
    "BG": (42.733883, 25.485830),
    "CZ": (49.817492, 15.472962),
    "PL": (51.919438, 19.145136),
    "ES": (40.463667, -3.749220),
    "IT": (41.871940, 12.567380),
    "GR": (39.074208, 21.824312),
    "CN": (35.861660, 104.195397),
    "PT": (39.399872, -8.224454),
    "RO": (45.943161, 24.966760),
    "EE": (58.595272, 25.013607),
    "SE": (60.128161, 18.643501),
    "FI": (61.924110, 25.748151),
    "NO": (60.472024, 8.468946),
    "AT": (47.516231, 14.550072),
    "RS": (44.016521, 21.005859),
    "TR": (38.963745, 30),
    "SK": (48.669026, 19.699024),
    "UA": (48.379433, 31.165580),
    "RU": (61.524010, 105.318756),
    "CH": (46.818188, 8.227512),
    "HR": (45.100000, 15.200000),
    "BA": (43.915886, 17.679076),
}

# Rest of World marker
ROW_REST_OF_WORLD = "RoW"


# ============================================================================
# COLOR PALETTES
# ============================================================================

# Country-specific colors (for maps, stacked charts, etc.)
COLOR_MAP = {
    "GB": "#760707",
    "DK": "#C85A17",
    "DE": "#9ED29E",
    "NL": "#2A9D8F",
    "BE": "#7A5C3A",
    "EU27": "#2F3E75",
    "ES": "#E3B505",
    "FR": "#17BECF",
    "NO": "#6E5A8A",
    "AT": "#7F3C8D",
    "CH": "#9C755F",
    "FI": "#00A087",
    "GR": "#8C6D31",
    "IT": "#7F7F7F",
    "PL": "#BCBD22",
    "SI": "#BEAED4",
    "SK": "#1B7837",
    "PT": "#A63603",
    "SE": "#4C7A4C",
    "RS": "#525252",
    "IE": "#1B9E77",
    "RO": "#E6550D",
    "CZ": "#4DAF4A",
    "EE": "#377EB8",
    "BG": "#984EA3",
    "SG": "#9467BD",
    "AE": "#8DA0CB",
    "TW": "#E78AC3",
    "CN": "#B56100",
    "ID": "#E1C007",
    "ZA": "#0B8EC2",
    "IN": "#894597",
    "TR": "#66D3008F",
    "CA": "#EB1A1A",
    "US": "#65C3E9",
    "KR": "#FA7FCB",
    "JP": "#CBCACA",
    "BR": "#5F9745",
    "RoW": "#BDBDBD",
}

FALLBACK_GREY = "#BDBDBD"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def canonical_name(code: str) -> str:
    """
    Get canonical (primary) country name for ISO2 code.
    
    Parameters
    ----------
    code : str
        ISO2 country code
        
    Returns
    -------
    str
        Primary country name, or code itself if not found
    """
    names = _names_by_code.get(code, [code])
    for name in names:
        if name not in REGION_ALIASES:
            return name
    return names[0] if names else code


def get_text_color(bg_color: str) -> str:
    """
    Determine text color (black or white) based on background brightness.
    
    Parameters
    ----------
    bg_color : str
        Background color (hex or named)
        
    Returns
    -------
    str
        "black" or "white" for best contrast
    """
    r, g, b, a = mcolors.to_rgba(bg_color)
    # Standard luminance formula
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if luminance > 0.6 else "white"

# --------------------------------------------------
# 0. Helpers: country codes and distances
# --------------------------------------------------

def to_iso2(country):
    if pd.isna(country):
        return None
    c = str(country).strip()
    if len(c) == 2 and c.isupper():
        return c
    return NAME_TO_CODE.get(c, c)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    return 2 * R * math.asin(math.sqrt(a))

def closest_country(base_country, candidate_countries, country_coords):
    if base_country not in country_coords:
        return None
    lat_b, lon_b = country_coords[base_country]
    best, best_dist = None, None
    for c in candidate_countries:
        if c not in country_coords:
            continue
        lat_c, lon_c = country_coords[c]
        d = haversine_km(lat_b, lon_b, lat_c, lon_c)
        if best is None or d < best_dist:
            best, best_dist = c, d
    return best

def pick_assembly_facility_country(base_country, oem, cg, facilities, oem_hq_country, country_coords):
    """Nearest OEM-specific facility for component group, fallback to OEM HQ."""
    cand = facilities[(facilities["Assembler"] == oem) & (facilities["component_group"] == cg)]
    if cand.empty:
        return oem_hq_country.get(oem)
    same = cand[cand["country_origin"] == base_country]
    if not same.empty:
        return base_country
    origin_country = closest_country(base_country, cand["country_origin"].unique(), country_coords)
    return origin_country or oem_hq_country.get(oem)

def pick_tower_assembly_country(base_country, facilities, country_coords):
    """
    For Towers: choose the assembly country as the country of the nearest
    tower facility in assembly_facilities, ignoring OEM.
    """
    fac = facilities[facilities["component_group"] == "Tower"]
    if fac.empty:
        return base_country

    candidate_countries = fac["country_origin"].dropna().unique()
    best = closest_country(base_country, candidate_countries, country_coords)
    return best or base_country


def tower_assembly_country_shares(facilities: pd.DataFrame) -> dict:
    """
    Return shares by country for tower assembly facilities (count-based).
    """
    fac = facilities[facilities["component_group"] == "Tower"]
    if fac.empty:
        return {}
    counts = fac["country_origin"].dropna().value_counts().astype(float)
    total = counts.sum()
    if total <= 0:
        return {}
    return (counts / total).to_dict()

def normalize_country_code(x):
    if pd.isna(x):
        return x
    return NAME_TO_CODE.get(x, x)

def get_import_category(component_group, sub_component):
    """
    Map (component_group, sub_component) → import category name.
    If not found, or category is 'Turbines', we treat it as having no import share.
    """
    key = (component_group, sub_component)
    cat = import_category.get(key)  # your mapping from (cg, sc) → category
    if cat is None or cat == "Turbines":
        return None
    return cat

def get_import_total_fraction(cg, sc):
    """
    Returns the import share as a fraction of TOTAL demand (BoM + Assembly).
    If no specific import category applies, returns 0.0.
    """
    cat = get_import_category(cg, sc)
    if cat is None or cat not in import_shares:
        return 0.0
    return import_shares[cat]["total"] / 100.0

def get_import_country_fractions(cg, sc):
    """
    Returns a dict {country_code: fraction_of_total_demand}.
    If no import category applies, returns {}.
    """
    cat = get_import_category(cg, sc)
    if cat is None or cat not in import_shares:
        return {}

    src = import_shares[cat]["sources"]
    # src values are in percentage of TOTAL demand (not just imports), so we /100 here
    return {NAME_TO_CODE.get(c, c): v / 100.0 for c, v in src.items()}

def load_excel(filename: str, sheet_name: str, supp_root: str = SUPP_ROOT) -> pd.DataFrame:
    """Load an Excel sheet from a file located in supp_root."""
    path = op.join(supp_root, filename)
    if not op.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    available_sheets = pd.ExcelFile(path).sheet_names
    if sheet_name not in available_sheets:
        raise ValueError(
            f"Sheet '{sheet_name}' not found in {path}. "
            f"Available sheets: {available_sheets}"
        )

    return pd.read_excel(path, sheet_name=sheet_name, header=0)
