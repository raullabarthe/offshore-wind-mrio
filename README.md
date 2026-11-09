# Made in Europe for offshore wind? Value added, employment, and sourcing trade-offs

Computational research repository accompanying the preprint/submission-stage manuscript by **Raúl Labarthe, Ulrich Elmer Hansen, and Anastasia Ioannou** (DTU Wind and Energy Systems). No journal acceptance, software release, or DOI is claimed.

## Scope and research question

The research asks where offshore-wind procurement generates **gross value added (GVA) and employment** across global value chains; how much the EU27 captures beyond Tier-1 sourcing; and the economic and procurement-cost consequences of shifting manufacturing sourcing between EU27 and non-EU27 countries. It evaluates economic capture and exposure to offshoring as distinct industrial-policy objectives, rather than attempting to rank legal auction rules.

The model combines an engineering-based lifetime cost model for a **1 GW fixed-bottom offshore wind farm (100 × 10 MW turbines; 25-year lifetime)**, component/subcomponent sourcing estimates, the **FIGARO 2024 MRIO** framework, and country-industry GVA and employment coefficients. A representative EU27 sourcing profile is capacity-weighted using Belgium, Denmark, Germany and the Netherlands; the UK is a reference market but is excluded from these EU27 weights.

The baseline includes **Devex, Turbine, BoP, Installation and O&M**. Only **Turbine and balance of plant (BoP)** procurement is reallocated in the counterfactuals. The parameter `delta` (`δ`) shifts the **EU27 share of inferred equivalent production quantities**, in percentage points, between EU27 and non-EU27 suppliers; it is *not* a direct percentage change in spending. Supplier price differences are represented by `mu` (`μ`), with **0.3** as the central assumption and a sensitivity analysis. Reallocation uses baseline country GVA weights and donor-quantity constraints. These are hypothetical static sourcing reallocations.

## Main manuscript findings

| Measure | Rounded manuscript result |
| --- | ---: |
| Global GVA over the project lifetime | EUR 6.6 billion |
| Global employment supported | 90,000 FTE-years |
| EU27 share of global GVA | 71% |
| EU27 share of global employment | 41% |
| Full reshoring: change in EU27 GVA / employment | +8% / +9% |
| Full reshoring: change in project expenditure | +1.5% |
| Full offshoring: change in EU27 GVA / employment | −39% / −42% |
| Full offshoring: change in expenditure | approximately −6.7% |

The EU27 captures a larger fraction of GVA than of employment because global upstream production and labor intensities differ across countries. Existing EU27 procurement limits additional reshoring potential but creates substantial exposure to displacement through offshoring. Procurement savings are **not** welfare estimates; cumulative FTE-years are **not permanent jobs or net employment creation**.

These are rounded manuscript reporting values, not new computations. See [docs/VALIDATION.md](docs/VALIDATION.md) for archived checks and their limits.

## Code and outputs

| Path | Description |
| --- | --- |
| `main.ipynb` | Ordered paper computational workflow |
| `modules/config.py` | Project-relative paths and run options |
| `modules/cost_model/` | Engineering cost model |
| `modules/supply_chain/` | Country, OEM and component-level sourcing |
| `modules/mrio_analysis/` | FIGARO, industry mapping and employment methods |
| `modules/impacts/` | Economic impacts, sourcing scenarios and exporters |
| `results/images/` | Manuscript and supplementary figures |
| `results/tables/` | LaTeX manuscript tables |
| `results/supplementary/` | C2 employment audit and D detailed-results workbooks |
| `docs/` | Provenance, reproduction, validation, and release decisions |

Processed B1 wind-farm sourcing and B2 turbine sourcing inputs are under `modules/supply_chain/sources/`. The processed C1 cost-to-industry mapping is under `modules/mrio_analysis/helpers/`, alongside the prepared FIGARO 2024 matrix in `modules/mrio_analysis/datasets/mrio/2024/`. Raw TGS 4C Offshore supplier contracts are excluded. Full FIGARO source data and some Eurostat, ILOSTAT and OECD datasets must be accessed externally if their derived artifacts are rebuilt.

## Running the research workflow

With Python 3.10 or later, on Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab main.ipynb
```

Run the notebook in order. The prepared matrix and public processed sourcing inputs are included, but runtime employment and scenario caches are **not** tracked. Regeneration may depend on external datasets and takes substantial computation; exact byte-for-byte reproduction of historical runs from a fresh clone is not claimed. Review [the reproducibility guide](docs/REPRODUCIBILITY.md) before regenerating outputs.

See [SUPPLEMENTARY_MATERIALS.md](SUPPLEMENTARY_MATERIALS.md) for the asset map and filenames. Historical local workbook names may differ from tracked names; verify contents before treating them as interchangeable.

## Citation, rights, and status

Use [CITATION.cff](CITATION.cff) for the manuscript and software citation metadata. The repository is **already public**, following the merge of PR #1 into `main` on November 6, 2026. This does **not** mean a formal versioned GitHub/Zenodo release exists or that third-party data redistribution rights have been fully cleared. There is currently no repository-wide open-source license; do not assume permission to relicense processed third-party content. See [the data-rights audit](docs/DATA_LICENSE_AUDIT.md) and [release checklist](docs/RELEASE_CHECKLIST.md).
