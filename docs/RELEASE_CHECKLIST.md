# Release checklist and safe Git workflow

This local branch is prepared for review only. It must not be pushed, merged,
tagged, released, or archived until the following checks receive explicit
approval.

## Pre-push review

1. Review `git diff --cached --stat` and `git diff --cached`.
2. Run secret/credential scans and inspect any finding manually.
3. Confirm `git ls-files` contains no raw TGS 4C Offshore data, internal
   allocation files, generated caches, or manuscript-private material.
4. Check prior Git history for restricted files. Do not rewrite history without
   explicit approval and an agreed remediation plan.
5. Confirm the full FIGARO redistribution terms and obtain coauthor/legal
   review of the selected software/data license.
6. Verify CFF and Zenodo metadata against the submitted manuscript; do not add
   a DOI, publication date, release version, or acceptance claim prematurely.
7. Re-run the structural and native-precision validation appropriate to the
   intended release asset set.

## Proposed pull request

- Base: `main`
- Compare branch: `paper/energy-economics`
- Purpose: replace the previous Tier-1 snapshot with the audited
  current **Made in Europe for offshore wind? Value added, employment, and cost
  trade-offs** workflow and reproducibility documentation,
  reproducibility documentation, and safe ignore rules.
- Do not open or publish the pull request until the staged diff, restricted-data
  audit, and coauthor review are approved.

## Versioning and archival recommendation

A branch is a line of development. Merging the reviewed branch into `main`
makes the approved source the default repository state. After the manuscript
submission package is frozen, create an annotated tag such as
`v1.0.0-energy-economics-submission` only if the authors approve that naming.
A GitHub Release is a published distribution built from that tag. Zenodo
archival should occur only after release review, producing a DOI that can then
be added to the final citation metadata. None of these actions has been taken
for this archive.
