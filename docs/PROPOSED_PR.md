# Proposed pull request: Energy Economics submission archive

**Base:** `main`

**Head:** `paper/energy-economics`

## Summary

This pull request updates the repository from the earlier Energy Policy/Tier-1
snapshot to the current Energy Economics submission workflow. It adds the
current modules and notebook, processed public B1/B2/C1 inputs, the prepared
2024 FIGARO core matrix, final figures/tables/C2/D results, project-relative
configuration, and publication-facing documentation. It removes the obsolete
`modules/results` and `outputs` structure plus superseded Excel classification
helpers.

## Data and release safeguards

- Raw TGS 4C Offshore data and internal raw allocation files are not staged.
- Runtime scenario/employment caches and external raw FIGARO files are ignored;
  final publication outputs under `results/` are staged.
- No license, release version, release date, DOI, GitHub Release, Zenodo
  deposit, push, merge, or tag is included.

## Validation

- Python modules compile; all requirements resolve.
- Notebook and Zenodo JSON validate; notebook source cells are unchanged while
  stale saved outputs were cleared.
- B1, B2, and C1 public workbook entry points load successfully.
- Six critical source/input hashes match; the 2024 MRIO cache loads without a
  rebuild.
- Read-only source-cache/table checks match the manuscript reporting
  benchmarks. No scenario was rerun and no production cache was changed.
- High-confidence credential, restricted-path, and public-history scans found
  no matches.

## Reviewer attention

The prepared FIGARO matrix is 15.60 MiB (below GitHub's 50 MiB warning
threshold). The imported current source includes pre-existing trailing
whitespace in code/classification data; it was intentionally left unchanged to
avoid unrelated source changes. Confirm licensing and redistribution terms,
coauthor approval, and final supplementary release assets before merging.
