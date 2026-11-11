# Model validation

## Reference results

The following values are rounded results of the 1 GW offshore wind farm model presented in **“Made in Europe for offshore wind? Value added, employment, and cost trade-offs”**.

| Quantity | Reference value |
| --- | ---: |
| Global lifetime GVA | approximately EUR 6.6 billion |
| Global lifetime employment | approximately 90,000 FTE-years |
| EU27 share of GVA | 71% |
| EU27 share of employment | 41% |
| Full reshoring: EU27 GVA | +8% |
| Full reshoring: employment | +3,300 FTE-years |
| Full reshoring: total project expenditure | +1.3% |
| Full offshoring: EU27 GVA | −39% |
| Full offshoring: employment | −16,000 FTE-years |
| Full offshoring: total project expenditure | −6.2% |
| Turbine reshoring: additional procurement expenditure | EUR 76 million |
| BoP reshoring: additional procurement expenditure | EUR 10 million |

The reference values are rounded for reporting. Numerical comparisons should use the model's full-precision outputs and appropriate tolerances.

## Checks performed

- Python modules compiled successfully.
- The Jupyter notebook passed JSON validation and contains executable source cells without embedded execution outputs.
- The processed B1 and B2 input workbooks loaded successfully: B1 contains 1,050 sourcing rows across four components; B2 contains 62 OEM orders, 62 known suppliers, 35 assembly facilities, and 696 general-facility rows.
- The processed C1 industry-mapping workbook and prepared FIGARO 2024 core/ROW matrix are present.
- Project-relative imports and notebook input paths were checked.
- Available local output tables were inspected against the rounded reference results above.

A complete independent run from a newly cloned repository has not yet been documented. Employment-source data obtained through external services may vary across retrieval dates; the repository describes the required inputs in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
