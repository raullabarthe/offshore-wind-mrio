
"""Static sourcing assumptions and country mappings for supply-chain allocation."""


# Import shares by component in the EU
# Source: https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ%3AC_202503236
import_shares = {
    "Tower": {"total": 10, "sources": {"TR": 9, "RoW": 10 - 9}},
    "Permanent magnets": {"total": 99, "sources": {"CN": 93, "JP": 6, "RoW": 99 - (93 + 6)}},
    "Substations": {"total": 52, "sources": {"CN": 31, "TR": 3, "CH": 2, "RoW": 52 - (31 + 3 + 2)}},
    "Cables": {"total": 16, "sources": {"TR": 4, "CH": 3, "CN": 2, "RoW": 16 - (4 + 3 + 2)}},
    "Transformers": {"total": 22, "sources": {"TR": 9, "CN": 6, "CH": 2, "RoW": 22 - (9 + 6 + 2)}},
    "Switchgears": {"total": 20, "sources": {"NO": 5, "CN": 5, "TR": 5, "RoW": 20 - (5 + 5 + 5)}},
    "Circuit breakers": {"total": 25, "sources": {"CH": 5, "CN": 5, "GB": 5, "RoW": 25 - (5 + 5 + 5)}},
    "Insulators": {"total": 27, "sources": {"CN": 14, "CH": 3, "US": 2, "RoW": 27 - (14 + 3 + 2)}},
    "Disconnectors": {"total": 12, "sources": {"CH": 7, "KR": 2, "RoW": 12 - (7 + 2)}}
}

import_category = {
    # Drivetrain
    ("Drivetrain", "Generator - Magnets"): "Permanent magnets",

    # Nacelle
    ("Nacelle", "Controller"): "Switchgears",
    ("Nacelle", "Converter"): "Switchgears",

    # Tower
    ("Tower", "Structure"): "Tower",
    ("Tower", "Internal"): "Tower",
    ("Tower", "Secondary"): "Tower",
    ("Tower", "Cabling"): "Cables",
    ("Tower", "Transformer"): "Transformers",
}

# Connected offshore wind farm capacity by country
# Source: https://windeurope.org/data/products/european-offshore-wind-farms-map/
connected_country = {
    "GB": {"Power (MW)": 16577, "Turbines": 2921, "Projects": 47},
    "DE": {"Power (MW)": 9121,  "Turbines": 1639, "Projects": 33},
    "NL": {"Power (MW)": 4739,  "Turbines": 670,  "Projects": 10},
    "DK": {"Power (MW)": 2653,  "Turbines": 672,  "Projects": 17},
    "BE": {"Power (MW)": 2261,  "Turbines": 399,  "Projects": 11},
    "FR": {"Power (MW)": 1596,  "Turbines": 229,  "Projects": 6},
    "SE": {"Power (MW)": 195,   "Turbines": 80,   "Projects": 5},
    "NO": {"Power (MW)": 101,   "Turbines": 13,   "Projects": 3},
    "FI": {"Power (MW)": 71,    "Turbines": 19,   "Projects": 3},
    "IT": {"Power (MW)": 30,    "Turbines": 10,   "Projects": 1},
    "IE": {"Power (MW)": 25,    "Turbines": 7,    "Projects": 1},
    "PT": {"Power (MW)": 25,    "Turbines": 3,    "Projects": 1},
    "ES": {"Power (MW)": 7,     "Turbines": 2,    "Projects": 2},
}

power_by_country = {
    c: v["Power (MW)"]
    for c, v in connected_country.items()
}

# --------------------------------------------------
# OEM HQ mapping
# --------------------------------------------------
oem_hq_country = {
    "GE Vernova":     "FR",
    "Siemens Energy": "DK",
    "Vestas":         "DK",
}

# Canonical names for component_group (across all tables)
CG_STD_MAP = {
    "Blade":     "Blades",
    "Blades":    "Blades",
    "Hub":       "Hub",
    "Nacelle":   "Nacelle",
    "Nacelles":  "Nacelle",
    "Tower":     "Tower",
    "Towers":    "Tower",
    "Drivetrain":"Drivetrain",
}

# --------------------------------------------------
# 3. Ownership categories and structure mapping
# OEM in-house vs. outsourcing by component in Europe
# WindEurope. (2020). Wind energy and economic recovery in Europe [PDF]. WindEurope.https://proceedings.windeurope.org/biplatform/rails/active_storage/disk/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaDdDVG9JYTJWNVNTSWhaelJ1WmpCMll6a3pZM2RwZHpWMWNXTjFNemh0ZUd4aE5IaHBZd1k2QmtWVU9oQmthWE53YjNOcGRHbHZia2tpY1dsdWJHbHVaVHNnWm1sc1pXNWhiV1U5SWxkcGJtUkZkWEp2Y0dVdFJteGhaM05vYVhBdGNtVndiM0owTFRJd01qQXVjR1JtSWpzZ1ptbHNaVzVoYldVcVBWVlVSaTA0SnlkWGFXNWtSWFZ5YjNCbExVWnNZV2R6YUdsd0xYSmxjRzl5ZEMweU1ESXdMbkJrWmdZN0JsUTZFV052Ym5SbGJuUmZkSGx3WlVraUZHRndjR3hwWTJGMGFXOXVMM0JrWmdZN0JsUTZFWE5sY25acFkyVmZibUZ0WlRvS2JHOWpZV3c9IiwiZXhwIjoiMjAyNi0wMi0wNFQyMTo1MDowOC42NjdaIiwicHVyIjoiYmxvYl9rZXkifX0=--7466823ea81c7274667add8a62f5a2bc5a0c1cc9/WindEurope-Flagship-report-2020.pdf
# --------------------------------------------------

ownership_oem = {
    "Converters":  45, 
    "Castings":    0, # offshore castings not made in-house by OEMs
    "Forgings":    0, # offshore forgings not made in-house by OEMs
    "Tower":       0, # offshore towers not made in-house by OEMs
}

# Only map sub-components that can fall back to ownership + bom_facilities.
# Sub-components with explicit OEM suppliers (your supplier table) are NOT listed here.
structure = {
    "Hub": {
        "Structure": "Castings",
        "Secondary": "Castings",
    },
    "Nacelle": {
        "Cooling":    "Converters",
        "Converter":  "Converters",
        "Controller": "Converters",
        "Canopy":     "Castings",
        "Secondary":  "Castings",
    },
    "Tower": {
        "Structure": "Tower",
        "Internal":  "Tower",
        "Cabling":   "Tower",
        "Secondary": "Tower",
    },
    "Drivetrain": {
        "Bedplate":         "Castings",
        "Coupling & brake": "Forgings",
    },
}

bridge = {
    (comp, subcomp): cat_key
    for comp, subdict in structure.items()
    for subcomp, cat_key in subdict.items()
}

# UK local content by cost component
# source: https://www.offshorewindscotland.org.uk/media/11937/bvga-local-content-roadmap.pdf
GB_local = {
    "source": "https://www.offshorewindscotland.org.uk/media/11937/bvga-local-content-roadmap.pdf",
    "table": "Adjusted baseline Scottish and UK content for all UK projects",
    "unit": "percent",
    "data": {
        "DEVEX": 83,
        "Turbine": 7,
        "Substations": 19,
        "Foundations": 7,
        "Cables": 7,
        "Turbine and foundation installation": 5,
        "Cable installation": 7,
        "Installation Other": 75,
        "O&M": 81,
        "ABEX": 30,
        "Total": 48,
    },
}


GB_CATEGORY_MAP = {
    # Direct matches (CostType ignored)
    ("DEVEX", "DEVEX"): "DEVEX",
    ("Turbine", "Turbine"): "Turbine",
    ("O&M", "O&M"): "O&M",
    ("ABEX", "ABEX"): "ABEX",

    # BoP (supply)
    ("BoP", "OSS"): "Substations",
    ("BoP", "Other"): "Substations",
    ("BoP", "Foundation"): "Foundations",
    ("BoP", "Array cables"): "Cables",
    ("BoP", "Export cables"): "Cables",

    # Installation
    ("Installation", "Turbine"): "Turbine and foundation installation",
    ("Installation", "Foundation"): "Turbine and foundation installation",
    ("Installation", "OSS"): "Substations",
    ("Installation", "Array cables"): "Cable installation",
    ("Installation", "Export cables"): "Cable installation",
    ("Installation", "Other"): "Installation other",
}
