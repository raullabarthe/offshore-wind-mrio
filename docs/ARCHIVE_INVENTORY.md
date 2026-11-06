# Archive inventory

**Historical inventory:** This describes the selection used when creating the now-merged branch; refer to the root README and release checklist for the current manuscript and repository status.


This inventory records the conservative selection used to create the Energy
Economics submission branch. It is a publication archive, not a replacement
for the original working directory.

## Retained

| Category | Retained material | Reason |
| --- | --- | --- |
| Workflow | `main.ipynb` and all current `modules/**/*.py` | Ordered paper execution and its implementation |
| Cost inputs | Exchange-rate XML files | Required by cost-model initialization |
| Sourcing | Processed B1 and B2 workbooks | Public notebook entry points for wind-farm and turbine allocations |
| Classification | C1 workbook, classification CSVs, and metadata XML | Required by industrial mapping and FIGARO parsing |
| MRIO | Prepared 2024 industry-by-industry core/ROW FIGARO matrix | Default cache-first matrix consumed by the notebook |
| Publication results | `results/images/`, `results/tables/`, and final C2/D workbooks in `results/supplementary/` | Final scientific outputs retained with source-relative paths and staged for publication |
| Documentation | README, supplementary map, citation, Zenodo draft, provenance and release records | Reproducibility and release review |

## Excluded

| Category | Excluded material | Reason |
| --- | --- | --- |
| Restricted inputs | Raw TGS 4C Offshore contracts and internal raw supplier/allocation spreadsheets | Redistribution rights have not been verified; B1/B2 provide the public processed workflow |
| Runtime caches | Scenario, employment, ILO, OECD, and historical validation caches under `results/supplementary/cache*/` | Preserved in this complete local archive for provenance, but excluded from Git and public staging |
| Historical MRIO | 2023 data and full 2024 FIGARO flatfile/intermediate parquet | Not used by the default 2024 cache-first workflow; full source is externally available |
| Development artifacts | Python bytecode, old `modules/results`, old `outputs/`, superseded Excel classification helpers | Not used by the current module architecture |

The excluded categories are protected by `.gitignore`. Before release, use
`git check-ignore -v` and `git ls-files` to confirm they have not been
added or retained from any future change.
