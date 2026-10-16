# Data provenance, access and redistribution

## Input table
`data/dataset_json_enriched.csv` — 43,398 rows × 156 columns, one row per PDP device record.
SHA-256 is recorded in every `results/*/meta.json`.

| Block | Columns | Source |
|---|---|---|
| Identifiers / target | `Ref_ID`, `Ref_DOI_number`, `final_pce` (= PDP `JV_default_PCE`, %), `Cell_architecture`, `ETL/HTL/Perovskite` stack strings | Perovskite Database Project export (Jacobsson et al., Nat. Energy 7, 107 (2022); https://www.perovskite-database.org), CC BY 4.0 |
| Database ("t1") | deposition methods, thicknesses, anneal temperature/time, device flags, composition fractions/counts, geometry flags | PDP columns and composition parsing (`a_*`, `b_*`, `halide_*`, `num_*`, `is_*`) |
| Material properties ("t2" adds 49 base columns) | `{ETL,HTL,Perovskite}_{band_gap, LUMO, HOMO, mobilities, dielectric_constant, atomic mass/radius/electronegativity min/max/avg}` + tolerance, octahedral, tau, cation-radius-ratio | Curated catalogue for materials occurring ≥100 times (literature values, Materials Project where applicable); first non-missing value per layer family; **material-level reference values, not per-device measurements** |
| LLM literature features ("t3" adds 32 base columns) | `has_*` material flags, `used_*` process flags, thicknesses, anneal, spin speed, concentration, `json_numeric_values_found`, … | Qwen-based extraction from full-text Markdown, paper-level, joined by DOI |

`Ref_DOI_number` contains inconsistencies (case, BOM, URL prefixes). `psc.data.normalize_doi` canonicalises them; this reduces 7,371 raw strings to 7,326 publications (+1 explicit no-DOI block of 232 records).

## Excluded from predictors
`Ref_ID`, DOI, `final_pce`, composition short-form string, stack strings, free-text, `Stability_measured`, `EQE_measured`, `JV_hysteresis_index`, `Cell_area_measured_cm2`, `Stabilised_performance_measured` (outcome-adjacent measurements), and any column >80 % missing in the training partition.

## Audit facts (from `results/feature_audit.json`)
* 36.2 % of records (2,699 normalised DOIs) have LLM-derived features; every literature feature has identical coverage, i.e. extraction is all-or-none per paper and constant within a paper.
* 69 % of the variance of PCE lies between publications (one-way ICC by DOI = 0.69).
* 31,591 records belong to duplicated t3 predictor configurations (7,733 groups); only 162 groups span more than one DOI.
* 17 records (6 DOIs) have PCE > 27 %.

## Not redistributed / not in this package
* Publisher PDFs and their Markdown conversions (copyright; DOI list is derivable from the PDP).
* Raw LLM JSON outputs (`../json/`, 2,835 files) are derived from copyrighted full text; the paper-level feature table is included as columns of the CSV.
* Materials Project API key (never stored; see `.gitignore`).
Reviewer access: deposit this directory (code, CSV, results, tests) in a private Zenodo record or anonymised GitHub repository before submission; Digital Discovery requires DOIs for code and data in the Data Availability Statement.

## Known gaps
* The retrieval route for the 3,299 PDFs and the exact Qwen model checkpoint/version are not recorded anywhere in the repository; they must be supplied by the authors.
* No hand-annotated gold set exists, so extraction precision/recall cannot be reported (see TODO register).
