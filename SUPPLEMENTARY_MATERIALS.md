# Supplementary-material map

This repository contains the executable processed inputs B1, B2, and C1, plus the final C2 and D output workbooks under `results/supplementary/`. The remaining manuscript supplementary materials should be checked against the final submitted manuscript before distribution.

| Material | Role | Repository status |
| --- | --- | --- |
| `Supplementary_Material.pdf` | Methodological description and source documentation | Release asset; not currently committed |
| `Supplementary_Material_A_Cost_Model.xlsx` | Cost-model audit workbook | Release asset; not required by notebook |
| `Supplementary_Material_B1_Suppliers_Processing_Wind_Farm_4COffshore.xlsx` | Processed wind-farm sourcing input (`sourcing_long`) | Committed public input |
| `Supplementary_Material_B2_Turbine_Supply_Chain_Sourcing_Inputs.xlsx` | Turbine sourcing input sheets | Committed public input |
| `Supplementary_Material_C1_NACE2_Cost_Mapping.xlsx` | Industrial cost mapping (`mapping_long`) | Committed public input |
| `results/supplementary/Supplementary_Material_C2_Wages_Calculation.xlsx` | Employment/FTE calculation and audit | Committed final publication output; external-source dependent |
| `results/supplementary/Supplementary_Material_D_Detailed_Results.xlsx` | Detailed model results | Committed final publication output; not a runtime input |

The final figures and LaTeX tables are likewise committed under `results/images/` and `results/tables/`. Runtime cache directories under `results/supplementary/cache*/` remain excluded from Git. The raw TGS 4C Offshore supplier-contract source files used in an internal preprocessing pipeline are not supplementary materials and are not redistributable through this repository. B1 is the public processed replacement used by `main.ipynb`.
