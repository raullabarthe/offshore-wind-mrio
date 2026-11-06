# Provenance and integrity record

## Source relationship

- Publication branch: `paper/energy-economics`
- Base repository: `https://github.com/raullabarthe/offshore-wind-mrio`
- Base commit: `aa047763f641e5e716935bb293f592cdd7fa71af`
- Local source working tree: a pre-existing private research working tree,
  preserved read-only; its machine-specific location is intentionally omitted.
- Local source Git HEAD observed during audit: `978ecfa` on `main`
- The source worktree was dirty; no source file, Git configuration, cache, or
  result was modified while preparing this copy.

The base repository is an older public snapshot. This branch layers the audited
current implementation over that history; it is not a history rewrite.

## Critical-file SHA-256 record

The following values were computed from the source before copying. All
non-notebook values were checked again in this archive after copying.

```text
b40bcae98628a1a7f97783c23a52a72b1be1cee7689ccee343cd219e35ed21b0  main.ipynb
b28a9d9c831a1f295aefe25691e6897c231b7b6d0affbd9d463c3f2822310c3a  modules/config.py
829499bc470f8ce4681e1e06aed8bc5511da87733b02188755bca7a8299fc1bf  modules/supply_chain/sources/wind_farm/Supplementary_Material_B1_Suppliers_Processing_Wind_Farm_4COffshore.xlsx
1f4d6940adac09ef0186b30bfff5e24af7c84d71ad13c584fc57024beb17a68f  modules/supply_chain/sources/turbine/Supplementary_Material_B2_Turbine_Supply_Chain_Sourcing_Inputs.xlsx
cb1d3c50d5d3e9192150591692ac5d8bb7c943e94d09f0e94a5e6390ae69a2a6  modules/mrio_analysis/helpers/Supplementary_Material_C1_NACE2_Cost_Mapping.xlsx
0f8275b02da4d60f59584dc45516bcf2fee526015e66da39a48f73ab4211d626  modules/mrio_analysis/datasets/mrio/2024/MRIO_2024_CORE_ROW_ind_by_ind.parquet
```

The source notebook's full-file hash is retained above as a source record. Its
saved outputs were then cleared in this archive to remove stale absolute paths
and historical result output. The resulting archive file SHA-256 is
`7ab1d341c782c43fe863af3a0e27a4b2e908dfdc174dd860162b947335b2fd15`.
The concatenated notebook cell sources are unchanged between source and archive
and have SHA-256
`3ea9aea51c5162140b1e3e5840e08eaa7f5faacfdebc5a61507dd90a3fcb9fd2`
(103 cells). The hashes otherwise cover path configuration, public processed
inputs, and the prepared FIGARO matrix. Documentation, ignore rules, and
dependency declaration are intentional archive-specific changes.
