# Audit report: M1 / M2 / M3 data, joins, definitions, leakage (experiment m3opt_20261019)

Scope: read-only audit of `ml_paper/` code and data, plus the LLM JSON directory (`UGP/json`, 2,835 files)
and the markdown directory (`UGP/sandbox/markdown`, 22 files). Nothing outside `experiments/m3opt_20261019/`
was modified. Git state at start: HEAD `b2efd1f`, branch `main`, with pre-existing uncommitted changes
(`snapshot/git_status.txt`, `snapshot/git_diff_stat.txt`). These were left untouched.
The machine-readable results behind this report are in `audit/integrity_audit.json`, `features/paper_status.csv`
and `features/record_json_state.csv`, all produced by `scripts/build_canonical.py`.

## 1. Tier definitions (as implemented; unchanged)

| Tier | Construction | Raw columns | After train-fitted preprocessing (full dev) |
|---|---|---|---|
| M1 (`t1`) | `psc.features.base_frame(df,'t1')`: device descriptors (stack, composition, process, architecture). `TIER_DROP['t1']` removes material-property and JSON-derived columns | 55 | 77 (with `_na` flags) |
| M2 (`t2`) | M1 + 49 computed material-property columns (ETL/HTL/perovskite band gap, HOMO/LUMO, mobilities, radii, electronegativity...) | 104 | 179 |
| M3 (`t3`) | M2 + 36 binary recipe flags `rc_*` derived from `data/lit_v4.json` by regex over LLM-extracted JSON (`psc.litfeat4.paper_recipe`). NaN for publications without JSON. CatBoost `first_feature_use_penalties=0.5` on `rc_*` | 140 | 215 |

Exact column lists: `configs/tiers.json`. Per-feature definitions, sources and missing-value meaning:
`feature_provenance.csv`. Representation and states: `feature_schema.json`.
`TrainFitPreprocessor` is fit on the training part of each fold only. It drops columns with more than 80% missing values,
imputes the median and adds `_na` indicators, so it is leak-free. `FORBIDDEN` (19 columns: Voc/Jsc/FF, stability outcomes, etc.)
are never features. Target: `JV_default_PCE`, population PCE ≥ 1 (41,107 records, 7,327 publications).

## 2. Source integrity

* `sandbox/pdb.csv` and `ml_paper` `dataset_json_enriched.csv` have 43,398 rows each and agree on every Ref_ID, DOI, PCE
  and architecture (0 mismatches). There are 1,002 missing targets, 1,289 with PCE < 1 (excluded by `load_dataset`) and 13 with PCE > 30
  (kept by the production loader; flagged here as implausible single-junction values).
* DOI normalisation (strip BOM/whitespace, lower-case, drop resolver prefix and `doi:`) maps 7,499 raw DOIs to 7,455 groups.
  The 286 records without a DOI form one group `__NO_DOI__`. That group keeps them on one side of every split, which is conservative.
* `lit_v4.json` reproduces exactly from the JSON directory with `litfeat4.paper_recipe` (`v4_flags_reproduce_all: true`).
* The recipe table is unique on `doi_group`, so the join is many-to-one. No rows are duplicated and none are dropped.

## 3. Join problems found (production M3 affected)

1. **IOP/multi-slash DOIs unjoined.** File stems encode `/` as `_`. Production restores only the first `_`, so 53 files never
   join (e.g. `10.1088/0957-4484_26_49_494002`). Of these, 49 are recoverable by replacing every `_` and 47 via the file's `source` field.
   Corrected ("fixed") join: 2,763 DOIs / 15,174 records. Production: 2,731 DOIs / 15,027 records.
2. **Empty JSON counted as covered.** 18 files have no materials; in production, 17 matched publications (59 records) get
   all-zero flags (`lx_covered = 1`) instead of missing.
3. **`source` field disagrees with the filename in 249 files.** 196 are template placeholders, 47 are underscore-only differences,
   2 are typos, 1 is an IOP DOI, and 3 point to a genuinely different paper: 10.1002/adma.201401685 vs 10.1039/C4EE01138D;
   10.1002/aenm.201602121 vs 10.1021/acsenergylett.6b00719; 10.1063/1.4960698 vs 10.1063/1.4944735.
   For those 3 the extraction may belong to another paper. They are kept by filename, as in production, and flagged.
4. A junk file `test_article.json` exists in the JSON directory and does not join.

## 4. Feature-construction problems (production flags)

1. **Outcome-key filter incomplete.** `litfeat.OUTCOME_KEY` misses 2,455 outcome-like keys (`stability*`, `pce [%]`, `voc (v)`,
   `j_sc`...). When their text enters the flag search string, stability-test conditions trigger process flags.
   A strict filter changes flags in 112 publications: ambient 93, n2_box 22, undoped 5, vacuum 2, hx_acid 1.
   No numeric outcome value reaches any production feature, so this is label-adjacent noise, not target leakage.
2. **Prompt-template artefacts.** About 492 publications contain placeholder records (`<chemical formula>`,
   `<human-readable name>`, "...: all extracted numerical properties"). 140 publications (774 dev+final records) are template-only:
   production treats them as covered with all-zero flags. The `clean` variant treats them as missing.
3. **Coverage is year-confounded.** JSON coverage is 16–44% for 2013–2019, 9% for 2020 and 0% from 2021 on. "Covered" is therefore partly a
   publication-era indicator. The covered-vs-uncovered PCE difference is small (12.33 vs 12.47) but the
   `covflag`/`CTL_cov` controls were run for this reason.
4. **Redundancy.** `htl_tbp` and `htl_litfsi` have φ = 0.86; `cs` and `inorganic` 0.63; `pea_family` has |r| = 0.83 with an M2 column.
   `hot_cast` occurs in a single publication and `htl_borate` in 9 (`feature_statistics.csv`, `feature_correlations.csv`).
5. **Flag accuracy.** The prior 17-paper pilot (`audit/recipe_flag_pilot`) gave precision 0.74 and recall 0.44. It was model-assisted, not human-adjudicated,
   so flag noise is substantial and recall-limited.
6. M1 already contains `Perovskite_band_gap_measured` (24% missing). Literature-JSON band gaps (`jprop`) are therefore largely redundant.

## 5. Leakage checks

* No v4final publication was loaded by any development script (`run_dev.py` filters to `v4dev` before anything else).
* All folds are DOI-grouped. The ledger records `doi_overlap = 0` for every fold of every run.
* Fold-internal text blocks (TF-IDF vocabulary, SVD, kNN and token target encodings) are fit only on the fold's training and validation
  publications. Target-encoding blocks use only training labels. DOI-permuted versions of every block are run as controls.
* The TF-IDF vocabulary of the raw extraction contains evaluative words ("best", "champion", "stability", "degradation"), which could
  correlate with the target. SVD components 1 and 4 of the raw text load on template and extraction-style artefacts. The `clean`
  text variant removes template records and an evaluative-word list; `chem` keeps only identity fields.
* **The reserved partition is not pristine.** v4final (35% of publications, salt `psc-v4`) was scored by `scripts/v4_final.py`
  (t2/t3 default and tuned, permutation controls) and by an earlier single-feature-drop experiment (`sub.py`) that inspected
  v4final. In this experiment v4final is used once, for the configuration frozen in `configs/F_final.json` before any final scoring.
  It is labelled a diagnostic, not a confirmatory test.
* The manuscript repeated-holdout splits (`make_splits(...,'doi',5,0.2,0.15,42)`) overlap the v4dev publications. Any choice made
  on v4dev is therefore advantaged on those splits by selection.

## 6. Repository discrepancies

* The README references scripts that do not exist in the tree: `v4_tune.py`, `v4_save_bundle.py`, `v4_t1_baseline.py`,
  `verify_manuscript_numbers.py`, `validate_extraction.py`, `verify_t3_numbers.py`. `results/v4/tune.json` exists, but its
  producing script does not.
* The working tree has uncommitted changes to `ml.tex`, figures, `results/candidates/`, `src/psc/candidates.py` and
  `audit/recipe_flag_pilot`. Manuscript numbers may come from uncommitted code.

## 7. Canonical JSON→CSV representation

`scripts/build_canonical.py` deterministically flattens all 2,835 JSON files into `features/json_long.csv.gz`, one row per
(publication, material index, key). Columns: `doi_group, file, mat_idx, role, canon_name, key, key_class, value_kind,
value_raw, value_num`. Key classes are identity / process / outcome / characterisation / property / other. Per-publication
status is in `features/paper_status.csv` and per-record JSON state (covered / empty / uncovered) in `features/record_json_state.csv`.
Every feature variant in `scripts/m3feat.py` is derived from these inputs with no LLM calls.
