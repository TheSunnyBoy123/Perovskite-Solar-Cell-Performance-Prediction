# PSC PCE prediction with DOI-grouped evaluation — reproducible package

Manuscript: `psc_manuscript/ml.tex` (build: `bash scripts/build_paper.sh`).
Everything reported in the manuscript is produced by the scripts below from `data/dataset_json_enriched.csv`.

## Layout
| Path | Content |
|---|---|
| `src/psc/` | DOI normalisation, tier definitions, train-fitted preprocessing, splits, models, metrics, feature groups |
| `configs/` | Experiment configs (`ablation.json`, `ablation_groups.json`, `baselines.json`, `model_dev.json`) |
| `scripts/` | CLI entry points (below) |
| `tests/` | pytest suite (splits, leakage, preprocessing, tiers, smoke test) |
| `results/<experiment>/` | `runs.jsonl` (one record per fit: sizes, DOI overlaps, metrics, runtime, hyper-parameter info), `preds/*.npz` (test indices, targets, predictions, train/val indices), `meta.json` (config hash, data SHA-256, git HEAD, library versions) |
| `docs/` | `FEATURES.md`, data provenance, audit reports |
| `legacy/` | Original scripts/results/figures from before the audit. **Not used for any reported number** (they early-stopped on the test fold and used un-normalised DOIs). |

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # tested with Python 3.13, catboost 1.2.10, xgboost 3.2, scikit-learn 1.6, torch 2.10
```

## Full-text retrieval (legal routes only; optional, not needed to reproduce the modelling results)
```bash
UNPAYWALL_EMAIL=you@institution.edu python scripts/retrieve_fulltext.py --dois dois.txt --out retrieval/
# Unpaywall -> OpenAlex -> Crossref open links -> retrieval/manual_queue.csv (download through your own legal access)
```

## Reproduce (from `ml_paper/`)
```bash
python -m pytest tests -q                                   # unit/leakage/smoke tests
python scripts/run_experiment.py --config configs/ablation.json --smoke   # ~3 min smoke run
python scripts/run_experiment.py --config configs/ablation.json           # t1/t2/t3 x {mean,ridge,CatBoost} x {DOI,random} x {pooled,pin,nip} x 5 splits (270 fits, ~1.5 h on 11 cores)
python scripts/run_experiment.py --config configs/ablation_groups.json    # feature-group controls
python scripts/run_experiment.py --config configs/baselines.json          # XGBoost-default, RF, MLP, subsample-GPR
python scripts/run_model_dev.py  --config configs/model_dev.json          # validation-tuned CatBoost / bag / XGBoost / ensemble (separate from the ablation)
python scripts/run_experiment.py --config configs/ablation_random_cap6000.json   # record-level CatBoost arm with 6,000-iteration cap (used by tables/figures)
python scripts/run_experiment.py --config configs/random_convergence.json    # pooled record-level CatBoost, 20,000-iteration cap (converged; used by tables)
python scripts/run_experiment.py --config configs/strict_validation.json     # lead-author-grouped and forward-in-time protocols
python scripts/validate_extraction.py                                          # proxy validation of LLM features vs PDP curation
python scripts/aggregate.py results/ablation_v1
python scripts/paired_bootstrap.py results/ablation_v1 catboost doi
python scripts/shap_analysis.py overall pin nip
python scripts/audit_features.py && python scripts/export_feature_lists.py
python scripts/make_tables.py && python scripts/make_figures.py && python scripts/make_si.py && python scripts/build_workflow_figure.py
python scripts/repro_check.py                                  # re-runs 3 stored cells; must match to 4 decimals
python scripts/code_manifest.py
# full clean-environment reproduction (new venv from requirements.txt):
#   python scripts/run_experiment.py --config configs/repro_cleanenv.json && python scripts/compare_runs.py results/repro_cleanenv results/ablation_v1
bash scripts/build_paper.sh
```
All experiments are resumable (completed cells are skipped).

## Evaluation protocol (summary)
* DOI key = lower-cased DOI with BOM/whitespace/`doi.org`/`doi:` prefixes removed (`psc.data.normalize_doi`); 7,371 raw strings collapse to 7,326 publications. Records without a DOI form **one** explicit group (`__NO_DOI__`) that is never split.
* Five repeated holdouts (`GroupShuffleSplit`, test 20 % of DOIs, seed 42); within each training partition a DOI-disjoint validation set (15 % of training DOIs) is used for early stopping / model selection. The test partition is never seen before final scoring. Repeated holdout test sets **overlap**; this is not k-fold CV.
* Record-level comparison: `ShuffleSplit` with identical fractions and the same models/budgets.
* All imputation statistics, the >80 %-missing column filter and the `_na` indicator set are learned on the training partition only.
* Tiers are nested by construction (`psc.data.TIER_DROP`, asserted in tests): t1 = database + composition (77 columns incl. `_na`), t2 = t1 + 102 material-property columns, t3 = t2 + 44 LLM literature columns.

## Data
`data/dataset_json_enriched.csv` (43,398 PDP records x 156 columns; `data/pdp_ref_metadata.csv` adds lead author and publication date from the PDP export) is the merged table of the PDP export (Jacobsson et al., Nat. Energy 2022), material-property catalogue and LLM-extracted paper-level features. See `docs/DATA.md` for provenance, redistribution limits and what is not included (PDFs, Markdown, raw LLM JSON).

## Reports
`docs/AUDIT_REPORT.md` (scientific audit), `docs/REPRODUCIBILITY_REPORT.md`, `docs/TODO_REGISTER.md`, `docs/JOURNAL_CHECKLIST.md`, `docs/LIT_VERIFICATION.md`, `docs/EXTRACTION.md`, `audit/` (literature and adversarial reviews).

## lx4 / t3 model (LLM recipe features)
See `docs/V4_REPORT.md` for results and limits. Reproduce: `python scripts/build_lit_v4.py` (needs the cached extraction JSONs, `--json-dir`), `python scripts/v4_dev.py --tiers t2,t2+rec --params '{"pen_lit":0.5}' --out results/v4/x.csv`, `python scripts/v4_tune.py` (~3 h), `python scripts/v4_final.py` (one-shot reserved evaluation, do not rerun for selection), `python scripts/v4_save_bundle.py`. Predict: `python scripts/predict.py --input devices.csv --out preds.csv`; the input needs the `data/dataset_json_enriched.csv` columns (PDP + curated material properties) and `Ref_DOI_number`; recipe features come from `data/lit_v4.json` by DOI or from `rc_*` columns, else are missing.

## Three-tier (t1/t2/t3) common-split comparison
`python scripts/v4_final.py` (t2/t3 arms, controls; `results/v4/final/`) and `python scripts/v4_t1_baseline.py` (adds t1 on the identical reserved publications and writes `results/v4/final_t1/metrics_3tier.csv`, `comparisons_3tier.json`). `python scripts/verify_manuscript_numbers.py` checks the manuscript numbers against these files. Excluded large artifacts: `docs/ARTIFACT_CHECKSUMS.md`; versioning rules: `docs/DEVELOPMENT_WORKFLOW.md`. Tier names: t1 = PDP features, t2 = + material properties, t3 = + 36 LLM recipe flags; the earlier 44-predictor literature block is *not* t3 (run files still call it `t3`).
