# Reproducibility workflow

## Scope

`main.ipynb` is the executable paper workflow. It retains the finalized model
logic and uses paths defined only through `modules.config`; it does not depend
on the original local project location.

## Ordered workflow

1. Configure project cost assumptions and exchange rates.
2. Load B2 and construct turbine OEM/supplier/facility sourcing allocations.
3. Load B1 processed sourcing shares to construct other wind-farm component
   allocations; raw 4C Offshore data are not read by the public notebook.
4. Load C1 and classifications to map component costs to FIGARO industries.
5. Load the included 2024 prepared FIGARO core/ROW industry matrix. Rebuilding
   it requires the externally downloaded FIGARO industry-by-industry flatfile.
6. Construct employment/FTE coefficients from Eurostat, ILOSTAT, and bounded
   OECD source data when no local cache is present.
7. Calculate baseline GVA and employment impacts.
8. Load or calculate EU27 resilience/offshoring counterfactuals and procurement
   price sensitivities.
9. Generate manuscript tables, figures, and supplementary result workbooks in
   `results/` when the relevant save/export settings are enabled.

## Input classes

| Class | Inputs | Reproduction implication |
| --- | --- | --- |
| Included public processed inputs | B1, B2, C1, exchange rates, classifications, prepared FIGARO core matrix | Available in a fresh clone |
| External public sources | Full FIGARO flatfile, Eurostat, ILOSTAT, OECD | Download or query only when rebuilding their derived artifacts; APIs/releases can change |
| Restricted source material | Raw TGS 4C Offshore supplier-contract files and internal raw allocation sheets | Not included and not needed for the public B1/B2 workflow |
| Final publication outputs | `results/images/`, `results/tables/`, C2/D workbooks | Version-controlled scientific outputs, retained at their original relative paths |
| Runtime caches | `results/supplementary/cache*/` | Retained in the complete local archive but ignored by Git; rebuilt locally when absent |

## Reproducibility boundary

The archive supports execution of the documented public workflow from processed
B1/B2/C1 inputs. Exact byte-for-byte regeneration of the historical employment
and scenario outputs is not claimed from a bare clone, because external source
releases/API responses and excluded runtime caches are not versioned here.
The verified C2 and D output workbooks, figures, and tables are retained in
this repository; their runtime caches are not staged. Never substitute a newly
recomputed cache for an archived manuscript output without a documented
comparison.
