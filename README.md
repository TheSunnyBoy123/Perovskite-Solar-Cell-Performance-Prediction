# Perovskite Solar Cell PCE Prediction

## Data

Place `data/dataset_json_enriched.csv` (43,398 device records) and `data/pdp_ref_metadata.csv` before running any script. Source: Jacobsson et al., *Nat. Energy* 2022 (PDP export) merged with Materials Project properties.

Publisher PDFs and LLM extraction outputs cannot be redistributed.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Scripts

| Script | What it does |
|--------|-------------|
| `01_prepare_data.py` | Load and clean dataset → `data/prepared.parquet` |
| `02_train.py` | 5-fold DOI-grouped cross-validation (M1/M2/M3) → `results/` |
| `03_evaluate.py` | Summarise results; optional SHAP plot |
| `04_predict.py` | Score new devices with a saved model bundle |
| `05_retrieve_pdfs.py` | Fetch open-access PDFs (Unpaywall → OpenAlex → Crossref) |
| `06_manual_queue_app.py` | Streamlit app to upload, convert and extract from paywalled PDFs |
| `07_extract_features.py` | Build recipe-flag JSON from per-paper extraction outputs |

```bash
python scripts/01_prepare_data.py
python scripts/02_train.py --tier t2
python scripts/03_evaluate.py
python scripts/04_predict.py --input devices.csv --out preds.csv

UNPAYWALL_EMAIL=you@inst.edu python scripts/05_retrieve_pdfs.py --dois data/dois.txt --out retrieval/
streamlit run scripts/06_manual_queue_app.py
python scripts/07_extract_features.py --extractions extractions/ --out data/lit_v4.json
```

## Feature tiers

| Tier | Model | Features |
|------|-------|---------|
| `t1` | M1 | PDP device / composition columns |
| `t2` | M2 | M1 + material properties |
| `t3` | M3 | M2 + LLM recipe flags |

Splits: DOI-grouped `GroupShuffleSplit`, test = 20 % of publications, val = 15 %, 5 repeated holdouts.
