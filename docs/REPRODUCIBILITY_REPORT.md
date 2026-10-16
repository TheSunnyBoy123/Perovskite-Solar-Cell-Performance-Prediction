# Reproducibility report

## Environments
* Development: Python 3.13.5, pandas 2.2.3, scikit-learn 1.6.1, catboost 1.2.10, xgboost 3.2.0, torch 2.10, 11 CPU cores (macOS).
* Clean virtualenv built **only** from `requirements.txt`: pandas 3.0.6, scikit-learn 1.9.1, catboost 1.2.10, torch 2.10.

## Commands run and outcomes
| Stage | Command | Outcome |
|---|---|---|
| Unit/leakage/smoke tests | `python -m pytest tests -q` | 29 passed in both environments (after fixing a pandas-3 incompatibility found in the clean venv) |
| Smoke run | `run_experiment.py --config configs/ablation.json --smoke` | completed; zero DOI overlap |
| Full ablation | `run_experiment.py --config configs/ablation.json` | 270/270 fits (an initial ridge divergence, R² ≈ −33,000 on t3, was diagnosed — extreme feature values — fixed by train-percentile winsorising, and the ridge cells were rerun) |
| Random-arm rerun | `run_experiment.py --config configs/ablation_random_cap6000.json` | 45/45 fits; CatBoost iteration cap raised 1,500 → 6,000 because 31/45 record-level fits had been truncated; ≈ half still reach the cap |
| Feature-group controls | `configs/ablation_groups.json` | 45/45 |
| Reference models | `configs/baselines.json` | 40/40 (XGBoost-default, RF, MLP, subsample GPR; t1 and t3) |
| Model development | `run_model_dev.py --config configs/model_dev.json` | 45/45 cells |
| Paired DOI-cluster bootstrap | `paired_bootstrap.py results/ablation_v1 catboost doi` | 45 contrasts × splits |
| SHAP | `shap_analysis.py` | 3 models |
| Tables, figures, SI, workflow figure | `make_tables.py`, `make_figures.py`, `make_si.py`, `build_workflow_figure.py` | regenerated from `results/` |
| LaTeX | `bash scripts/build_paper.sh` | `ml.pdf` 21 pages, no undefined references/citations; minor overfull boxes in two older Methods paragraphs |
| Strict validation | `configs/strict_validation.json` | 216/216 fits (lead-author-grouped ×5 splits, forward-in-time ×3 windows); zero overlap |
| Record-level convergence | `configs/random_convergence.json` | 15/15 fits converged (best iteration 7,207–11,702, cap 20,000) |
| Proxy extraction validation | `scripts/validate_extraction.py` | `results/extraction_validation.json` |
| Full clean-environment reproduction | `configs/repro_cleanenv.json` + `scripts/compare_runs.py` | 270/270 matched, max abs ΔR² 1.1e-4 |
| Cross-environment spot check | `python scripts/repro_check.py` | three stored cells reproduced to 4 decimals (diff +0.0000) in both environments |

## What was *not* done
* The full matched ablation (270 fits, `configs/repro_cleanenv.json`) was **re-run end-to-end in the clean venv** (pandas 3.0.6, scikit-learn 1.9.1, catboost 1.2.10): all 270 cells matched the original run; max |ΔR²| = 1.1×10⁻⁴, max |ΔMAE| = 2.8×10⁻⁴ (ridge numerics; CatBoost bit-identical). See `results/repro_cleanenv/comparison.json`. The other experiments (feature-group, baselines, model development, strict validation, convergence) were run once; only their code paths are covered by the clean-venv tests.
* `ml_paper/` is not a git commit (the directory is untracked in the parent repository); `results/code_manifest.json` records SHA-256 hashes of all code/config files and of the data table instead. `meta.json` files record `git_dirty: true` accordingly.
* The LaTeX build disables microtype font expansion (minimal TeX Live lacking scalable fonts); `elsarticle.cls` and a few packages were installed into `~/Library/texmf` from CTAN/TeX Live 2025 archives.
* `standalone.cls` was unavailable, so the workflow figure is compiled via a wrapper (`scripts/build_workflow_figure.py`).

## Known limitations of reproducibility
* PDFs, Markdown and raw LLM JSON are not included; the LLM extraction stage cannot be reproduced from this package, only the downstream stages (starting at `data/dataset_json_enriched.csv`).
* Exact Qwen checkpoint is not recorded.
