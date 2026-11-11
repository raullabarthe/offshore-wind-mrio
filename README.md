# Made in Europe for offshore wind?

Computational model accompanying the manuscript **“Made in Europe for offshore wind? Value added, employment, and cost trade-offs”** by Raul Labarthe, Ulrich Elmer Hansen, and Anastasia Ioannou (Technical University of Denmark), prepared for submission to *Energy Economics*.

## Purpose and scope

The model links offshore-wind farm cost and component sourcing allocations to a FIGARO 2024 industry-by-industry multiregional input-output (MRIO) framework. It estimates global and EU27 value added, employment, and procurement-cost effects for a representative 1 GW offshore wind farm under baseline, reshoring, offshoring, and component-level sourcing scenarios.

The implementation covers:

- offshore-wind farm cost, component, and subcomponent allocations;
- turbine and balance-of-plant sourcing scenarios;
- FIGARO 2024 MRIO integration with corrected input-output normalization (`A_ij = Z_ij / X_j`);
- baseline GVA and employment calculations using Eurostat and ILOSTAT sources, including an ILOSTAT World anchor and World-minus-FIGARO residual ROW treatment;
- supplier price assumptions, the finalized lower-cost supplier classification, and the central `mu = 0.30` sensitivity with its delta scenarios;
- generation of manuscript figures, tables, and supplementary workbooks.

## Repository layout

```text
main.ipynb                         Executable end-to-end workflow
modules/config.py                  Project-relative paths and run settings
modules/cost_model/                Offshore-wind cost model
modules/supply_chain/              Supplier and sourcing allocations
modules/mrio_analysis/             FIGARO, employment, and industrial analysis
modules/impacts/                   Scenario, table, and figure calculations
docs/                              Reproducibility and data documentation
requirements.txt                   Python dependencies
setup_env.ps1                      Windows environment setup helper
```

Generated figures, tables, supplementary result workbooks, and runtime caches are intentionally excluded from GitHub. The processed B1/B2 sourcing workbooks are different: they are version-controlled model inputs and are included below their original paths.

## Setup and execution

The supported environment is Python **3.13**, recorded in `.python-version`.

On Windows:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_env.ps1
```

Or install manually:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab main.ipynb
```

The notebook uses paths derived from `modules/config.py`, so it must be run from a clone of this repository or opened with the repository as the working directory. It does not rely on the original Windows project path.

## Model inputs

The repository includes the processed B1 and B2 sourcing workbooks, industry classifications, exchange-rate inputs, employment-source adapters, and the prepared FIGARO 2024 core/ROW matrix. Additional employment source data can be obtained through the documented source APIs. Generated data and runtime caches are excluded.

## Validation reference

The validated local results used as the comparison reference are approximately EUR 6.6 billion global GVA, 90,000 global FTE-years, 71% EU27 GVA capture, and 41% EU27 employment capture. Full reshoring is approximately +8% EU27 GVA, +3,300 FTE-years, and +1.3% project expenditure; full offshoring is approximately −39% EU27 GVA, −16,000 FTE-years, and −6.2% project expenditure. These are rounded reporting values, not hard-coded targets. See [docs/VALIDATION.md](docs/VALIDATION.md) for the validation record and limitations.

## Citation

See `CITATION.cff` for the manuscript and model citation.
