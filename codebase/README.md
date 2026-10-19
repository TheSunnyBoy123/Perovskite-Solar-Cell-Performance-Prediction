# Perovskite Solar Cell PCE Prediction — Reproducible Codebase

This directory documents the full pipeline from raw data to trained models as shared with peer reviewers.

## Data availability notice

**Supplied in this repository:**
- `data/dataset_json_enriched.csv` — merged table (43,398 PDP device records × 156 columns); SHA-256 is recorded in each experiment's `meta.json`. Included in the review package but git-ignored for file size.
- `data/lit_v4.json` — per-DOI LLM-derived recipe flags and numeric aggregates (36 features used by M3).
- `data/lit_v2.csv`, `data/lit_v3.csv` — earlier literature-feature versions (used in development experiments).
- `data/pdp_ref_metadata.csv` — lead author and publication date from the PDP export.

**Not supplied (redistribution restrictions):**
- Publisher PDFs used during full-text retrieval.
- Markdown conversions of those PDFs.
- Raw LLM JSON responses produced during extraction.
- The standalone material-property catalogue CSV (intermediate; properties are already merged into `dataset_json_enriched.csv`).
- The recipe-flag human-annotation worksheet (contains spans of raw LLM output).

The PDP source database is publicly available: Jacobsson et al., *Nat. Energy* 2022. Material properties were retrieved from the Materials Project API (mp-api ≥ 0.41).

---

## Pipeline overview

```
data/dataset_json_enriched.csv          ← merged PDP + material properties + LLM recipe flags
        │
        ▼
Step 1: Load & split            (src/psc/data.py, src/psc/splits.py)
        │    DOI-grouped GroupShuffleSplit, 5 repeated holdouts
        │    train → val (15 % of train DOIs) + test (20 % of DOIs)
        ▼
Step 2: Feature engineering     (src/psc/features.py)
        │    Tier selection (M1/M2/M3), _na indicators, barrier descriptors
        │    TrainFitPreprocessor: imputation statistics learned on train only
        ▼
Step 3: Model training          (src/psc/models.py, scripts/run_experiment.py)
        │    Mean baseline, Ridge, CatBoost (early-stop on val), XGBoost, RF
        │    Results written to results/<experiment>/runs.jsonl
        ▼
Step 4: Evaluation & analysis   (src/psc/evaluate.py, scripts/aggregate.py,
        │                        scripts/paired_bootstrap.py, scripts/shap_analysis.py)
        ▼
Step 5: Final bundle & predict  (scripts/v4_final.py, scripts/v4_save_bundle.py,
                                 scripts/predict.py)
```

---

## Quickstart

```bash
# 0. Install dependencies (Python 3.10+)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Run unit tests (splits, leakage, preprocessing, tiers, smoke)
python -m pytest tests -q

# 2. Smoke run — verifies the full pipeline in ~3 min (2 splits, 300 iterations)
python scripts/run_experiment.py --config configs/ablation.json --smoke

# 3. Full ablation (M1/M2/M3 × DOI-grouped/record-level × pooled/pin/nip × 5 splits)
python scripts/run_experiment.py --config configs/ablation.json

# 4. Aggregate and compare
python scripts/aggregate.py results/ablation_v1
python scripts/paired_bootstrap.py results/ablation_v1 catboost doi

# 5. SHAP analysis
python scripts/shap_analysis.py overall pin nip

# 6. Reproduce manuscript numbers
python scripts/repro_check.py
```

---

## Step-by-step walkthrough

### Step 1 — Data loading (`src/psc/data.py`)

`dataset_json_enriched.csv` is the single input. It contains:
- **PDP columns** (device architecture, layer stacks, deposition methods, composition, J-V metrics)
- **Material-property columns** (band gap, HOMO/LUMO, mobility, tolerance factor … per ETL/HTL/Perovskite layer)
- **LLM recipe flags** (36 binary/numeric paper-level features from `lit_v4.json`)
- `Ref_DOI_number` — the grouping key after DOI normalisation

```python
from src.psc.data import load_dataset
df = load_dataset("data/dataset_json_enriched.csv", arch="overall")
# df has 43,398 rows; target column is "final_pce"
```

DOI normalisation (`psc.data.normalize_doi`) strips BOM, whitespace and `doi.org/` prefixes and lowercases. The 7,371 raw strings collapse to 7,326 publication groups; records without a DOI form one `__NO_DOI__` group.

### Step 2 — Splits (`src/psc/splits.py`)

```python
from src.psc.splits import make_splits
splits = list(make_splits(df["doi_group"].values, protocol="doi",
                          n_splits=5, test_size=0.2, val_size=0.15, seed=42))
# each element: (train_idx, val_idx, test_idx)
```

Five repeated holdouts with `GroupShuffleSplit` (test = 20 % of DOIs, seed 42). Within each training partition a DOI-disjoint validation set (15 % of training DOIs) is used for early stopping / model selection. The test partition is never seen before final scoring.

### Step 3 — Feature engineering (`src/psc/features.py`)

Three nested feature tiers mirror the three model families reported in the manuscript:

| Tier key | Model | Features |
|----------|-------|---------|
| `t1`     | M1    | 77 PDP database + composition columns (incl. `_na` indicators) |
| `t2`     | M2    | M1 + 102 curated material-property columns (MAT49 per layer) |
| `t3`     | M3    | M2 + 36 LLM recipe flags from `data/lit_v4.json` |

```python
from src.psc.features import base_frame, TrainFitPreprocessor

Xraw = base_frame(df, tier="t2")                     # raw (unimputed) frame
pre  = TrainFitPreprocessor().fit(Xraw.iloc[train_idx])  # learn on train only
Xtr  = pre.transform(Xraw.iloc[train_idx])
Xva  = pre.transform(Xraw.iloc[val_idx])
Xte  = pre.transform(Xraw.iloc[test_idx])
```

`TrainFitPreprocessor` drops columns with >80 % missing (computed on train), adds `_na` binary indicators, and median-imputes. All statistics are learned on the training partition only.

### Step 4 — Model training (`src/psc/models.py`)

```python
from src.psc.models import fit_predict

preds, info = fit_predict("catboost", Xtr, ytr, Xva, yva, Xte)
# info contains best_iteration, validation RMSE, hyperparameters
```

Available model names: `"mean"` (intercept baseline), `"ridge"`, `"catboost"`, `"xgb_default"`, `"rf"`, `"mlp"`.

CatBoost hyperparameters used throughout (matched across protocols):
```python
iterations=1500, learning_rate=0.05, depth=8, l2_leaf_reg=3.0,
subsample=0.8, colsample_bylevel=0.7, min_data_in_leaf=5
```
Early stopping at 100 rounds on the validation set.

### Step 5 — Evaluation (`src/psc/evaluate.py`)

```python
from src.psc.evaluate import metrics
m = metrics(y_true, y_pred)
# returns dict with RMSE, MAE, R², Pearson r, and coverage/interval stats
```

### Step 6 — Final model bundle and prediction (`scripts/v4_final.py`, `scripts/predict.py`)

The final M3 model is a bagged ensemble of five seeds saved to `models/v4_t3_bundle.pkl`.

```bash
# Reproduce the final reserved-split evaluation (run once only)
python scripts/v4_final.py

# Save the bundle
python scripts/v4_save_bundle.py

# Predict on new devices (same column schema as dataset_json_enriched.csv)
python scripts/predict.py --model models/v4_t3_bundle.pkl \
                          --input my_devices.csv --out preds.csv
```

`predict.py` looks up recipe features from `data/lit_v4.json` by DOI; devices without a matching DOI are predicted with the recipe block missing (M2-equivalent).

---

## Rebuilding `data/dataset_json_enriched.csv` from scratch

The merged CSV is built in three independent stages. All stages require only open data sources.

### Stage A — PDP export + material properties

The PDP device table (Jacobsson et al., *Nat. Energy* 2022) is the base. Material properties for each unique ETL/HTL/Perovskite composition were fetched from the Materials Project API and merged by composition string. The merged file is `dataset_json_enriched.csv`.

### Stage B — LLM recipe-flag extraction (optional, needs publisher PDFs)

> **Note:** The publisher PDFs that were processed during full-text extraction **cannot be redistributed**. The derived per-DOI features are provided in `data/lit_v4.json` so that no re-extraction is necessary to reproduce the modelling results. The steps below are documented for transparency only.

```bash
# 1. Retrieve open-access full text (Unpaywall → OpenAlex → Crossref)
UNPAYWALL_EMAIL=you@institution.edu \
  python scripts/retrieve_fulltext.py --dois dois.txt --out retrieval/
# Non-OA papers land in retrieval/manual_queue.csv (institutional access required)

# 2. Build the derived feature table from the cached extraction JSONs
python scripts/build_lit_v4.py --json-dir path/to/extraction_cache/
# Writes data/lit_v4.json
```

### Stage C — Merge

Once `lit_v4.json` is available, the recipe flags are merged into the main CSV automatically when `load_dataset` is called (via `psc.litfeat4.attach`).

---

## Repository layout

```
src/psc/          core library (data, splits, features, models, evaluate, results)
configs/          experiment JSON configs
scripts/          CLI entry points (run_experiment, aggregate, predict, …)
tests/            pytest suite
data/             input data (dataset_json_enriched.csv git-ignored; lit_v4.json tracked)
models/           saved model bundles (*.pkl git-ignored; regenerate with v4_save_bundle.py)
results/          experiment outputs (runs.jsonl tracked; *.npz git-ignored)
docs/             FEATURES.md, DATA.md, audit reports, reproducibility report
legacy/           pre-audit scripts — NOT used for any reported number
```

See the top-level `README.md` for the full list of experiment commands and manuscript build instructions.
