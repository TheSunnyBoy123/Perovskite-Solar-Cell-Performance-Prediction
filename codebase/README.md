# Perovskite Solar Cell PCE Prediction — Reproducible Codebase

Peer-reviewer package. Walks from the raw merged dataset to trained M1/M2/M3 models in four scripts.

## Data notice

**Provided:** `data/dataset_json_enriched.csv` (43,398 device records, git-ignored for size; SHA-256 in each run's `meta.json`), `data/lit_v4.json` (36 LLM-derived recipe flags per DOI), `data/pdp_ref_metadata.csv`.

**Not provided:** Publisher PDFs, their Markdown conversions, and raw LLM JSON cannot be redistributed (copyright). The derived recipe flags in `lit_v4.json` are sufficient to reproduce all modelling results without re-running extraction.

Source database: Jacobsson et al., *Nat. Energy* 2022. Material properties: Materials Project API.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pandas pyarrow scikit-learn catboost shap
```

## Pipeline

| Script | What it does |
|--------|-------------|
| `scripts/01_prepare_data.py` | Load CSV, normalise DOI keys, join metadata and recipe flags → `data/prepared.parquet` |
| `scripts/02_train.py` | 5-fold DOI-grouped cross-validation with CatBoost for M1/M2/M3 → `results/runs_<tier>.jsonl` |
| `scripts/03_evaluate.py` | Summarise RMSE/MAE/R² across splits; optional SHAP plot |
| `scripts/04_predict.py` | Score new devices with the saved M3 bundle → `preds.csv` |

```bash
python scripts/01_prepare_data.py
python scripts/02_train.py --tier t2          # M2; use --smoke for a 2-split test run
python scripts/02_train.py --tier t3          # M3 (requires lit_v4.json)
python scripts/03_evaluate.py
python scripts/04_predict.py --input my_devices.csv --out preds.csv
```

## Feature tiers

| Tier | Model | Features |
|------|-------|---------|
| `t1` | M1 | PDP device / composition columns only |
| `t2` | M2 | M1 + curated material-property columns (band gap, HOMO/LUMO, mobility, …) |
| `t3` | M3 | M2 + 36 LLM recipe flags from `data/lit_v4.json` |

Tiers are nested: M3 ⊃ M2 ⊃ M1.

## Evaluation protocol

DOI-grouped `GroupShuffleSplit`: test = 20 % of publications, val = 15 % of remaining publications (for early stopping), 5 repeated holdouts. Records sharing a DOI always land in the same fold. Records without a DOI form one explicit `__NO_DOI__` group.
