# Validation record

**Manuscript:** What is at stake for the EU in offshore wind value chains? Economic capture and manufacturing trade-offs. The results and checks below are historical reporting audits, not newly executed model runs.


Validation is deliberately read-only with respect to the finalized source
project and its production caches.

## Reference outputs inspected

The existing native LaTeX outputs report these rounded manuscript checks:

| Quantity | Reference value |
| --- | --- |
| Global baseline GVA | EUR 6.568 billion |
| Global baseline employment | 90 thousand FTE-years |
| EU27 baseline GVA capture | 71% |
| EU27 baseline employment capture | 41% |
| Full resilience, central price assumption | EU27 GVA +7.6%, employment +3.3 thousand FTE-years, procurement +1.3% |
| Full offshoring, central price assumption | EU27 GVA -39%, employment -16 thousand FTE-years, procurement -6.2% |

The figures are read from the archived source tables; no scenarios were rerun,
no production cache was written, and no result was changed to match these
benchmarks.

## Structural checks required for this branch

- Notebook JSON is valid and the public B1/B2/C1 paths resolve.
- Python modules compile without syntax errors.
- Critical-file SHA-256 values equal the pre-copy record in
  `PROVENANCE.md`.
- The included MRIO cache loads as the 2024 industry-by-industry core/ROW
  matrix without rebuilding a flatfile.
- `.gitignore` excludes generated caches, external raw flatfiles, and known
  restricted input names.

Results of the checks performed on this archive are added after the commands
are run. A full scenario execution is intentionally out of scope: it could
regenerate caches and requires external employment sources.

## Completed archive checks

- `python -m compileall -q modules`: passed.
- `main.ipynb` JSON parse: passed; all 103 cell sources match the source
  notebook exactly after stale saved outputs were cleared.
- B1 required sheets (`sourcing_long`, `contracts_summary`), B2 required input
  sheets, and C1 `mapping_long`: present; the B2 public-input loader passed.
- Critical file hashes: 6/6 matched `PROVENANCE.md`.
- Cache read: the included matrix loaded without download or cache rebuild as
  shape `(3206, 3450)`, 51 regions, total `300072865`.
- The source central v10 scenario cache was inspected read-only: its
  `scenarios_delta.pkl` SHA-256 is
  `691239d8560b7d1f5a8e7fc99b4b11cb3f9337f76f14ff771261274cbd3c8db2`
  and its `scenarios_result.pkl` SHA-256 is
  `f287d91339d9cfd26703a0f8b528e945003cca8416cc9fc7354aa15603128549`.
- Source output tables contain the expected native reporting values for the
  central resilience/offshoring GVA, employment, and procurement rows, and the
  6568/90/71%/41% baseline benchmarks.
- Complete-results transfer: all source `results/` files, including final C2
  and D workbooks, remain in the complete local publication archive; they are
  ignored and are not part of the clean GitHub staging set.
- Ignore checks: generated scenario cache paths, the known restricted raw B1
  supplier file, full FIGARO flatfile paths, and generated result paths are
  ignored. Final figures, tables, and C2/D publication outputs remain local
  and are intentionally not staged.

The staged diff has 976 whitespace diagnostics, all in imported current source
code or classification data; none is in the archive documentation. They are
left unchanged to preserve the current implementation and input bytes.

No scenario, employment, or MRIO rebuild was run, and no production output or
cache in the original project was changed.
