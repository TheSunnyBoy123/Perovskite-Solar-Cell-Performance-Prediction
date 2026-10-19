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
| `scripts/05_retrieve_pdfs.py` | Fetch open-access PDFs (Unpaywall → OpenAlex → Crossref); outputs manual queue |
| `scripts/06_manual_queue_app.py` | Streamlit app — upload manually downloaded PDFs, convert and extract with LLM |
| `scripts/07_extract_features.py` | Build `data/lit_v4.json` recipe flags from per-paper extraction JSONs |

```bash
# Modelling (requires data/dataset_json_enriched.csv)
python scripts/01_prepare_data.py
python scripts/02_train.py --tier t2          # M2; use --smoke for a quick 2-split test
python scripts/02_train.py --tier t3          # M3 (requires data/lit_v4.json)
python scripts/03_evaluate.py
python scripts/04_predict.py --input my_devices.csv --out preds.csv

# Full-text retrieval + LLM extraction (to reproduce data/lit_v4.json from scratch)
UNPAYWALL_EMAIL=you@inst.edu python scripts/05_retrieve_pdfs.py --dois data/dois.txt --out retrieval/
streamlit run scripts/06_manual_queue_app.py   # for papers needing institutional access
python scripts/07_extract_features.py --extractions extractions/ --out data/lit_v4.json
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
