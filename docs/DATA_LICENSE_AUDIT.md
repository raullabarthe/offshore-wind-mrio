# Data and licence audit

**Current status:** the files discussed here may already be publicly visible in the merged `main` tree. Unverified rights remain unverified, despite the public push. This is a record of release and remediation needs, not confirmation of a license.


**Status:** publication decision record, not legal advice.
**Scope:** files committed in the now-merged public repository. Runtime caches and raw TGS 4C Offshore files are excluded. A transformed
or aggregated file is not presumed redistributable unless the applicable source
terms or written permission allow it.

## Status key

- **Verified general terms:** an official provider page states reusable terms,
  subject to stated conditions and any dataset-specific notice.
- **Conditional:** general terms exist, but the particular edition, source data,
  attribution, third-party content, or derivative redistribution still needs
  confirmation.
- **Unverified / do not publish until cleared:** this archive has no written
  redistribution permission adequate for the public file.

## Inputs and processed datasets

| Public file or group | Original provider / provenance | Terms evidenced in this audit | Status and publication condition |
| --- | --- | --- | --- |
| B1: Supplementary_Material_B1_Suppliers_Processing_Wind_Farm_4COffshore.xlsx | Processed sourcing workbook based on the **TGS 4C Offshore Intelligence Platform**. Its sources sheet cites the platform and related BVG/academic benchmark material. | No TGS licence, subscription agreement, or written redistribution permission is stored here. | **Unverified / do not publish until cleared.** Written TGS permission must explicitly cover the processed/aggregated workbook and any residual supplier, contract, project, or facility information. |
| B2: Supplementary_Material_B2_Turbine_Supply_Chain_Sourcing_Inputs.xlsx | Author-compiled workbook using WindEurope offshore-wind-farms and supply-chain-map products, EC/JRC wind evidence, a Commission notice, and a WindEurope report, as listed in its sources sheet. | No WindEurope product licence or permission is stored. Public Commission material may have separate reuse terms, but that does not grant rights in WindEurope products. | **Unverified / do not publish until cleared.** Confirm the terms for every WindEurope product and whether selected/derived records may be redistributed. |
| C1: Supplementary_Material_C1_NACE2_Cost_Mapping.xlsx and classification CSVs | Author-created cost-to-NACE/FIGARO bridge. C1 cites BVG cost benchmark material; supporting classifications include Eurostat/FIGARO and ILO/ISIC mappings. | No permission for quoted/extracted BVG benchmark data is stored. Eurostat material has general reuse terms; provenance and licence of each imported classification extract are not fully recorded. | **Conditional.** Document exact sources/versions and confirm that reproduced benchmark values and classification extracts may be redistributed. |
| Prepared FIGARO matrix: MRIO_2024_CORE_ROW_ind_by_ind.parquet | Archive-derived aggregation of the 2024 FIGARO industry-by-industry table, produced by Eurostat and the JRC. | Eurostat authorises reuse of published statistical data and metadata with source acknowledgement, subject to individual notices and third-party exceptions. FIGARO is an official Eurostat statistical product. | **Conditional.** Record the exact FIGARO release/download URL, edition, access date, attribution, and any edition-specific notice before distributing this derived parquet. Do not assume general Eurostat terms clear every underlying non-EU source. |
| ECB exchange-rate XML files | European Central Bank reference exchange-rate data, identified by source URLs in modules/cost_model/inputs.py. | Provider/source URLs are recorded, but no licence notice or retrieval record is stored in the archive. | **Conditional.** Verify the ECB reuse notice for the exact files and add attribution/version information. |
| Employment/FTE source inputs used to construct C2 | Eurostat, ILOSTAT/ILO, and OECD data queried by the employment workflow; raw downloads and caches are not staged. | Eurostat: public statistical data are generally reusable with acknowledgement, subject to exceptions and individual notices. ILO: datasets published from 3 May 2023 are CC BY 4.0 unless restricted. OECD: data may generally be reused with attribution, but users must check dataset-specific terms and third-party ownership. | **Conditional.** Record dataset identifiers, retrieval dates, licence/notice at access, and OECD source-tab ownership for every series used. Do not infer that a derived coefficient removes attribution or third-party conditions. |
| C2: results/supplementary/Supplementary_Material_C2_Wages_Calculation.xlsx | Author-generated employment/FTE calculation and source-resolution workbook using the employment inputs above and model transformations. | No standalone rights statement for the workbook; permissibility follows the employment and FIGARO inputs. | **Conditional.** Public distribution requires the preceding Eurostat/ILO/OECD/FIGARO checks and appropriate attribution/disclaimer where required. |
| D: results/supplementary/Supplementary_Material_D_Detailed_Results.xlsx, figures, and LaTeX tables | Author-generated model outputs using B1/B2/C1, FIGARO, and employment coefficients. | No standalone data licence is present. The files are outputs, but may embody protected or contract-restricted source data. | **Conditional.** Coauthors must confirm that outputs do not disclose restricted TGS/WindEurope content and that upstream terms permit this level of redistribution. |
| Supplementary Material A and PDF | Referenced release assets; not currently tracked. | Not audited as public repository files. | **Out of scope until added.** Audit before staging. |

## Verified provider terms to apply, not to overextend

- [Eurostat copyright notice and free re-use of data](https://ec.europa.eu/eurostat/help/copyright-notice):
  published statistical data and metadata may generally be reused with source
  acknowledgement, but individual notices and third-party/non-EU exceptions can
  apply. Adaptations must be identified.
- [Eurostat FIGARO information](https://ec.europa.eu/eurostat/web/esa-supply-use-input-tables/information-data)
  identifies FIGARO as the EU inter-country supply, use, and input-output
  statistical product. This establishes provider/provenance, not a substitute
  for the specific release notice.
- [ILO rights and permissions](https://www.ilo.org/rights-and-permissions):
  databases/datasets and referential metadata published from 3 May 2023 are
  CC BY 4.0 unless restricted; earlier data require the exact work notice.
- [OECD terms and conditions](https://www.oecd.org/en/about/terms-conditions.html):
  data are reusable with credit unless further restrictions apply, but users
  must verify metadata and third-party ownership for the particular dataset.
- No comparable written permission was found in this archive for TGS 4C
  Offshore, WindEurope products, or the cited BVG benchmark material.

## Recommended licensing architecture

Do **not** apply a single blanket licence to all repository contents.

1. **Original code:** after coauthor approval, use a permissive software licence
   such as Apache-2.0 (recommended where a patent grant is desired) or MIT.
   The chosen licence must cover only code the authors own.
2. **Original documentation and authored narrative:** after approval, CC BY 4.0
   is suitable where broad reuse with attribution is intended.
3. **Data and processed inputs:** use a per-file rights statement and attribution
   manifest. License only author-owned data for which upstream rights are clear;
   retain restricted files outside the public repository. Do not relicense
   Eurostat, ILO, OECD, TGS, WindEurope, FIGARO, or BVG material as if owned by
   the authors.
4. **Derived outputs:** license figures/tables/C2/D only after confirming that
   source-data agreements permit public redistribution. If not, publish an
   approved aggregate output or a request-access procedure instead.

## Required evidence before publication

- Written TGS decision on B1 and a record of the exact permitted use.
- WindEurope decision on B2 and confirmation for each relevant product.
- Exact FIGARO release citation, reuse notice, and attribution statement.
- Employment-source manifest: Eurostat dataset codes/DOIs, ILO publication
  dates/licences, OECD dataset metadata/source ownership, and retrieval dates.
- BVG benchmark permission or a documented decision to remove/reconstruct
  affected C1 values.
- Coauthor approval of the selected code/document/data licence arrangement.
