# Public repository and formal release checklist

**Current state:** The repository is public. PR #1 (`paper/energy-economics`) was merged into `main` on November 6, 2026. This merge is not a tagged software release, DOI deposit, or journal acceptance.

Before a formal tagged release or Zenodo deposit:

1. Confirm manuscript title, authors, data availability statement, and citation metadata.
2. Check exact model/data vintages and reconcile tracked figures, tables, and C2/D workbooks with the manuscript; do not replace scientific outputs with unverified reruns.
3. Resolve the current **public exposure** of processed B1/TGS and B2/WindEurope materials whose redistribution rights remain unverified; do not assume public visibility establishes a license.
4. Check C1/BVG, FIGARO, Eurostat, ILOSTAT, OECD, source-specific attribution, and derivative-distribution permissions. See [DATA_LICENSE_AUDIT.md](DATA_LICENSE_AUDIT.md).
5. Record coauthor approval; select a license for author-owned code and separately document data/documentation rights. Do not place an indiscriminate license on third-party material.
6. Inspect repository history and tracked files for secrets or restricted information.
7. Decide whether the large D workbook belongs in Git, a release asset, or an external research-data archive.
8. Document validated run commands, environment versions, and any reproducibility boundary for the exact release commit.

After approval, a version tag (e.g. `v1.0.0`) can identify the frozen implementation. A GitHub Release and Zenodo deposit are later, separate actions. Do not invent a DOI, publication date, journal acceptance, or release record. Protect `main` from deletion and force pushes where feasible.
