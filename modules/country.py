"""
Country utilities and constants.

Contains country name mappings, ISO codes, geographic data, and color schemes.
Transversal module used across supply chain and economic impact analysis.
"""

from __future__ import annotations

from collections import defaultdict

# ============================================================================
# COUNTRY NAME TO ISO2 CODE MAPPING
# ============================================================================

NAME_TO_CODE = {
    "Australia": "AU", "Austria": "AT", "Åland Islands": "FI", "Belgium": "BE", "Bosnia & Herzegovina": "BA", 
    "Brazil": "BR", "Bulgaria": "BG", "Canada": "CA", "Croatia": "HR", "Cyprus": "CY", "Czech Republic": "CZ",
    "Denmark": "DK", "Estonia": "EE", "Finland": "FI", "France": "FR", "Germany": "DE", "Greece": "GR", "Guernsey": "GB", 
    "India": "IN", "Ireland": "IE", "Isle of Man": "GB", "Italy": "IT", "Japan": "JP", "Jersey": "GB", "Liechtenstein": "LI",
    "Lithuania": "LT", "Luxembourg": "LU", "Malaysia": "MY", "Malta": "MT", "Mexico": "MX", "Netherlands": "NL", "Norway": "NO", "Oman": "OM",
    "People's Republic of China": "CN", "Poland": "PL", "Portugal": "PT", "Republic of China (Taiwan)": "TW", "Reunion": "FR",
    "Romania": "RO", "Russia": "RU", "Serbia": "RS", "Singapore": "SG", "Slovakia": "SK", "Slovenia": "SI", "South Africa": "ZA",
    "South Korea": "KR", "Spain": "ES", "Sweden": "SE", "Switzerland": "CH", "Thailand": "TH", "Turkey": "TR", "Ukraine": "UA",
    "United Arab Emirates": "AE", "United Kingdom": "GB", "United States of America": "US"
    , "Indonesia": "ID"
}

# ============================================================================
# ISO2 CODE TO COUNTRY NAME MAPPING
# ============================================================================

# ============================================================================
# ISO2 CODE TO COUNTRY DISPLAY NAME
# ============================================================================

# Standardized country names for figures, tables, and legends.
# Independent of NAME_TO_CODE, which preserves supplier-source names.

CODE_TO_NAME = {
    "AE": "Arab Emirates",
    "AT": "Austria",
    "AU": "Australia",
    "BA": "Bosnia and Herzegovina",
    "BE": "Belgium",
    "BG": "Bulgaria",
    "BR": "Brazil",
    "CA": "Canada",
    "CH": "Switzerland",
    "CN": "China",
    "CY": "Cyprus",
    "CZ": "Czech Republic",
    "DE": "Germany",
    "DK": "Denmark",
    "EE": "Estonia",
    "ES": "Spain",
    "FI": "Finland",
    "FR": "France",
    "GB": "UK",
    "GR": "Greece",
    "HR": "Croatia",
    "ID": "Indonesia",
    "IE": "Ireland",
    "IN": "India",
    "IT": "Italy",
    "JP": "Japan",
    "KR": "South Korea",
    "LI": "Liechtenstein",
    "LT": "Lithuania",
    "LU": "Luxembourg",
    "LV": "Latvia",
    "MT": "Malta",
    "MX": "Mexico",
    "MY": "Malaysia",
    "NL": "Netherlands",
    "NO": "Norway",
    "OM": "Oman",
    "PL": "Poland",
    "PT": "Portugal",
    "RO": "Romania",
    "ROW": "RoW",
    "RS": "Serbia",
    "RU": "Russia",
    "SE": "Sweden",
    "SG": "Singapore",
    "SI": "Slovenia",
    "SK": "Slovakia",
    "TH": "Thailand",
    "TR": "Turkey",
    "TW": "Taiwan",
    "UA": "Ukraine",
    "US": "USA",
    "ZA": "South Africa",
}

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

# EU27 ISO2 codes present in datasets
EU27 = {"AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "ES", "FI", 
        "FR", "GR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT", 
        "NL", "PL", "PT", "RO", "SE", "SI", "SK"} 

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
    "GB": "#7C0000",
    "DK": "#C90000",
    "DE": "#025F51",
    "NL": "#2A9D8F",
    "BE": "#D5D044",
    "EU_dep": "#4F6BD1",
    "EU27": "#2F3E75",
    "ES": "#E3B505",
    "FR": "#17BECF",
    "NO": "#6E5A8A",
    "AT": "#7F3C8D",
    "CH": "#546E7A",
    "FI": "#A1DEA1",
    "GR": "#8C6D31",
    "IT": "#7F7F7F",
    "PL": "#532F00",
    "SI": "#BEAED4",
    "SK": "#1B7837",
    "PT": "#A63603",
    "SE": "#4C7A4C",
    "RS": "#525252",
    "IE": "#1B9E77",
    "RO": "#006C91",
    "CZ": "#4DAF4A",
    "EE": "#377EB8",
    "BG": "#984EA3",
    "SG": "#2CCDBD",
    "AE": "#8DA0CB",
    "TW": "#E78AC3",
    "CN": "#B56100",
    "ID": "#E1C007",
    "ZA": "#0B8EC2",
    "IN": "#88C1947A",
    "TR": "#3D2000",
    "CA": "#341700",
    "MX": "#135600",
    "US": "#BFE6F6",
    "KR": "#FA7FCB",
    "RU": "#10009C",
    "JP": "#F9CD5D",
    "BR": "#5F9745",
    "RoW": "#1A1818",
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




def normalize_country_code(code: str) -> str | None:
    """
    Normalize country code (name -> ISO2).
    
    Parameters
    ----------
    code : str
        Country name or code
        
    Returns
    -------
    str or None
        ISO2 code, or original if not found
    """
    return NAME_TO_CODE.get(code, code) if code else None
