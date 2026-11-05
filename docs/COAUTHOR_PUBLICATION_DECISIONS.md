# Coauthor publication decisions

Please decide the following before any public push or release.

1. **B1 / TGS 4C Offshore:** Do we have written permission to publish the
   processed sourcing workbook, including its aggregation level and all derived
   supplier/contract information? If not, remove it from the public package or
   replace it with an approved public reconstruction.
2. **B2 / WindEurope:** Are the referenced WindEurope map products licensed for
   redistribution through the processed turbine-input workbook? Confirm the
   exact products and permitted derivative use.
3. **C1 / BVG and classifications:** May the bridge reproduce the cited
   benchmark values and classification extracts? Approve either the documented
   attribution/permission route or a revised public-only bridge.
4. **FIGARO and employment provenance:** Approve the edition-specific FIGARO
   citation and the Eurostat/ILOSTAT/OECD dataset-attribution manifest. Confirm
   that the C2/D outputs may be published at their present detail.
5. **Licensing:** Select separate licences for author-owned code and
   documentation; approve a per-file data-rights approach rather than a blanket
   repository data licence.
6. **Large result workbook:** Decide whether the 68.42 MiB D workbook remains
   in ordinary Git, moves to Git LFS, or is a checked release asset.

No public commit, push, merge, tag, release, or Zenodo deposit should occur
until these decisions are recorded.
