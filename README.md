# Targeting Made-in-Europe rules in wind energy auctions

Computational research archive for the manuscript *Targeting Made-in-Europe rules in wind energy auctions: Which components should be covered?* by Raúl Labarthe, Ulrich Elmer Hansen, and Anastasia Ioannou. This is a submission version prepared for **Energy Economics**; it is not a published or released software version.

## Research overview

The model evaluates the global value-chain consequences of manufacturing procurement reallocations for a representative 1 GW offshore wind farm. It links detailed country-level production allocations for turbine and balance-of-plant components with the European Commission's FIGARO multiregional input-output (MRIO) accounts. The analysis reports gross value added (GVA), employment expressed as FTE-years, upstream trade effects, and procurement-price effects under EU27 resilience and offshoring counterfactuals.

The public workflow uses processed, redistributable sourcing inputs rather than raw supplier-contract data. It reconstructs geographic sourcing shares beyond direct contractors and then reallocates manufacturing procurement between EU27 and non-EU27 suppliers. It does not alter the underlying research assumptions or scenario definitions.

## Repository layout

```text
main.ipynb                         Ordered paper workflow
modules/config.py                  Project-relative paths and run controls
modules/cost_model/                Offshore-wind cost assumptions and model
modules/supply_chain/              Turbine and wind-farm sourcing allocations
modules/mrio_analysis/             FIGARO, employment, and classification code
modules/impacts/                   Economic impacts, scenarios, tables, figures
results/images/                    Final publication figures (tracked)
results/tables/                    Final publication LaTeX tables (tracked)
results/supplementary/             Final C2/D workbooks; runtime cache directories ignored
docs/                              Provenance, inventory, validation, workflow
```

## Public inputs and data availability

The following processed inputs are committed and are sufficient to execute the public sourcing and industrial-classification workflow:

| Input | Purpose | Status |
| --- | --- | --- |
| `Supplementary_Material_B1_Suppliers_Processing_Wind_Farm_4COffshore.xlsx` | Processed wind-farm sourcing shares and contract summaries | Public processed input |
| `Supplementary_Material_B2_Turbine_Supply_Chain_Sourcing_Inputs.xlsx` | OEM, supplier, facility, and import-share turbine inputs | Public processed input |
| `Supplementary_Material_C1_NACE2_Cost_Mapping.xlsx` | Cost-to-industry concordance | Public processed input |
| `MRIO_2024_CORE_ROW_ind_by_ind.parquet` | Prepared FIGARO 2024 core/ROW industry matrix | Derived public-data input |

The full FIGARO 2024 industry-by-industry flatfile is an external European Commission source and is deliberately not redistributed here. The code records its source URL and can read a locally downloaded flatfile when the prepared matrix must be rebuilt. Eurostat, ILOSTAT, and bounded OECD inputs are external sources used for the employment/FTE construction; their live APIs and release versions can change. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the exact distinction between included, external, and restricted data.

Raw TGS 4C Offshore supplier-contract files are **not included**. They remain subject to redistribution rights and are not needed by the notebook's public B1-based workflow. Do not add them to a public branch or release without written confirmation of redistribution rights.

## Installation and execution

Use Python 3.10 or later in a new environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab main.ipynb
```

Run `main.ipynb` from top to bottom. The default settings keep `RECOMPUTE`, `FIRST_MRIO`, and `FIRST_EMP` false. Final publication figures, LaTeX tables, and supplementary workbooks are version-controlled under `results/`. Runtime scenario and employment cache directories (`results/supplementary/cache*/`) remain local-only; a fresh public clone can therefore download external employment inputs and build missing cache artifacts. To rebuild the FIGARO matrix from its external flatfile, follow the controlled steps in the reproducibility guide rather than changing the published notebook defaults.

## Results and validation

The archived local source outputs were checked without recomputation. Rounded reference values are global GVA of EUR 6.6 billion, global employment of 90 thousand FTE-years, EU27 GVA capture of 71%, and EU27 employment capture of 41%. At the central price assumption, the table reports full resilience of about +8% EU27 GVA, +9% EU27 employment, and +1.5% project expenditure, versus full offshoring of about -39%, -42%, and -6.7%, respectively. These are reporting checks, not substitutes for the native-precision output files.

Detailed validation status and limitations are in [docs/VALIDATION.md](docs/VALIDATION.md). The checked final manuscript figures, tables, and C2/D supplementary workbooks are first-class publication assets in `results/`; runtime caches are deliberately not staged.

## Supplementary material, citation, and release status

The expected supplementary files and their roles are listed in [SUPPLEMENTARY_MATERIALS.md](SUPPLEMENTARY_MATERIALS.md). Cite this software using [CITATION.cff](CITATION.cff). A software license, public release version, tag, GitHub release, and Zenodo record have not yet been selected or created. See [docs/RELEASE_CHECKLIST.md](docs/RELEASE_CHECKLIST.md) before making the branch public.
