# Reproducibility

This repository contains the computational implementation accompanying **“Made in Europe for offshore wind? Value added, employment, and cost trade-offs”**, prepared for submission to *Energy Economics*. The authoritative local publication archive also retains generated results and caches; those generated artifacts are deliberately excluded from GitHub.

## Supported environment

- Python 3.13, as specified by `.python-version` and enforced by `setup_env.ps1`.
- Windows setup: run `Set-ExecutionPolicy -Scope Process Bypass` followed by `.\setup_env.ps1`.
- Manual setup: create a virtual environment, install `requirements.txt`, and open `main.ipynb` with JupyterLab.
- The notebook and modules resolve paths from `modules/config.py` relative to the project root. No path depends on the original local Windows directory.

## Computational workflow

Run the notebook cells in order. The workflow:

1. loads cost assumptions and exchange-rate inputs for the representative offshore wind farm;
2. constructs component and subcomponent sourcing allocations for turbine and balance-of-plant items;
3. loads the processed B1 wind-farm and B2 turbine sourcing workbooks;
4. maps cost items to the included industry classifications and FIGARO A64 industries;
5. loads the prepared FIGARO 2024 26ed industry-by-industry core/ROW matrix;
6. applies the corrected input-output normalization `A_ij = Z_ij / X_j`, where each intermediate-input column is divided by the output of the purchasing industry;
7. builds industrial employment coefficients from Eurostat and ILOSTAT inputs, using the ILOSTAT World anchor and a World-minus-explicit-FIGARO-country residual for ROW;
8. calculates baseline global and EU27 GVA and employment impacts;
9. evaluates turbine, balance-of-plant, full reshoring, full offshoring, supplier-price, and component-level sourcing scenarios;
10. runs the central `mu = 0.30` case, the main sensitivity range, and the complete delta scenario calculations; and
11. writes figures, tables, and supplementary workbooks to `results/` when the notebook export settings are enabled.

## Model inputs

| Input class | Repository location or source | Reproduction status |
| --- | --- | --- |
| Processed wind-farm sourcing | `modules/supply_chain/sources/wind_farm/Supplementary_Material_B1_Suppliers_Processing_Wind_Farm_4COffshore.xlsx` | Included |
| Processed turbine sourcing | `modules/supply_chain/sources/turbine/Supplementary_Material_B2_Turbine_Supply_Chain_Sourcing_Inputs.xlsx` | Included |
| Cost-to-industry mapping | `modules/mrio_analysis/helpers/Supplementary_Material_C1_NACE2_Cost_Mapping.xlsx` | Included |
| Classifications and crosswalks | `modules/mrio_analysis/helpers/` | Included |
| FIGARO 2024 prepared matrix | `modules/mrio_analysis/datasets/mrio/2024/MRIO_2024_CORE_ROW_ind_by_ind.parquet` | Included processed input |
| FIGARO 2024 flatfile | External Eurostat/CIRCABC FIGARO 26ed release | External input for rebuilding; place the downloaded industry-by-industry flatfile under `modules/mrio_analysis/datasets/mrio/2024/` when rebuilding the prepared matrix |
| Employment sources | Eurostat, ILOSTAT, and bounded OECD candidates | Obtain through the documented public downloads/APIs; local downloads and caches are ignored because source terms and responses can change |

B1, B2, and C1 are included in the repository and loaded by the public notebook. Rebuilding employment coefficients may require downloading data from the documented external sources. Runtime caches are regenerated locally and are not tracked.

## Outputs and Git policy

The notebook writes generated artifacts below `results/images/`, `results/tables/`, and `results/supplementary/`. Scenario, MRIO, employment, and source-audit caches are also generated locally. These paths are ignored and are not part of the clean GitHub publication. The complete local publication copy and its backup retain the validated figures, tables, workbooks, and caches for reference.

The notebook committed to GitHub is code-only: execution outputs and embedded figures are cleared before publication. Run it in a disposable or ignored output tree when validating a clone so that archived local results are not overwritten.

## Validation and limitations

The rounded manuscript reference values are approximately:

- global GVA: EUR 6.6 billion;
- global employment: 90,000 FTE-years;
- EU27 capture: 71% of GVA and 41% of employment;
- full reshoring: +8% EU27 GVA, +3,300 FTE-years, and +1.3% project expenditure;
- full offshoring: −39% EU27 GVA, −16,000 FTE-years, and −6.2% project expenditure;
- turbine reshoring procurement increase: EUR 76 million;
- balance-of-plant reshoring procurement increase: EUR 10 million.

These are rounded reporting values. Validation compares the actual local results with numerical tolerances and does not alter economic assumptions to force rounded agreement. Exact byte-for-byte reproduction is not claimed when external datasets, API responses, or excluded runtime caches differ from the validated local archive.
