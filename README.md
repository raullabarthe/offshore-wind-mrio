# Targeting Made-in-Europe rules in wind energy auctions

Computational model for the article:

> Targeting Made-in-Europe rules in wind energy auctions: Which components should be covered?

Raul Labarthe, Ulrich Elmer Hansen, and Anastasia Ioannou  
Department of Wind and Energy Systems, Technical University of Denmark  
Journal target: Energy Policy

## Overview

The model links offshore-wind component sourcing with a FIGARO multiregional input-output framework. It estimates how alternative Tier-1 sourcing rules affect EU27 gross value added, employment, and upstream trade-linkage effects for a representative 1 GW offshore wind farm.

The workflow is implemented in `main.ipynb` and organized around the following main components:

```text
modules/config.py        Paths and run settings
modules/cost_model/      Offshore-wind cost structure
modules/supply_chain/    Supplier, OEM, facility, and sourcing allocation
modules/mrio_analysis/   FIGARO MRIO construction and employment extension
modules/results/         Scenario calculations, tables, and figures
outputs/                 Generated figures, tables, and scenario caches
```

## Reproducibility

The public workflow runs from the supplementary workbooks submitted with the article:

- B1 wind-farm sourcing starts from `sourcing_long`.
- B2 turbine sourcing reconstructs the allocation process from OEM, supplier, facility, and import-share inputs.
- C1 industrial classification starts from `mapping_long`.

Raw TGS 4C Offshore supplier-contract records are the source for the B1 supplier calculations reported in the supplementary material; the public run uses the processed B1 workbook.

FIGARO MRIO flatfiles are not committed because the source CSV files are large. If automatic download from CIRCABC fails, place the FIGARO 2025 EU inter-country input-output flatfiles in `modules/mrio_analysis/datasets/` using the filenames expected by `modules/mrio_analysis/mrio.py`.

Use Python 3.13.x. The notebook checks the active interpreter before installing
or importing the analysis dependencies.

Install the Python requirements and run the notebook:

```bash
py -3.13 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab main.ipynb
```

On Windows, the helper script `setup_env.ps1` creates the virtual environment,
installs the requirements, and registers the Jupyter kernel:

```powershell
.\setup_env.ps1
```

Generated files are written under `outputs/`: figures in `outputs/figures`,
tables in `outputs/tables`, and scenario caches in `outputs/cache/results`.

## Supplementary Materials

The methodological description, source documentation, spreadsheet audit trail, and detailed results are provided in the article supplementary material. Release assets should include the supplementary PDF and workbooks listed in `SUPPLEMENTARY_MATERIALS.md`.

## Citation

Citation metadata is provided in `CITATION.cff`. The repository release can be archived in Zenodo once the article submission package is final.

## License

A code and data license should be selected before public release.
