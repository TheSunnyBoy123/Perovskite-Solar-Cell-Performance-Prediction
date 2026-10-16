# Artifacts excluded from Git

Large or raw files are listed in `.gitignore` and are **not versioned**. Their SHA-256 checksums (computed when the baseline snapshot was made) and how to obtain them again:

| path | size | sha256 | how to regenerate / obtain |
|---|---|---|---|
| `data/dataset_json_enriched.csv` |  31M | `600efce587a1bd81c2a192e36db6534d8d78a3edfadb635dd40e685c9314ffdc` | processed modelling table (PDP export + material properties + earlier literature block); built by the data pipeline in `src/psc/data.py` from the PDP export; the PDP is CC BY. Needed by every experiment script. |
| `models/v4_t2_bundle.pkl` |  40M | `73521390d4f5735dc4b0d66bdb87683fe8f831a80fc4c062ead0d3ea02d796b0` | `scripts/v4_save_bundle.py` (retrains the five seed models; needs the data table above). |
| `models/v4_t3_bundle.pkl` |  34M | `00670d33eb7d38841a5197c1590af0b8a710e8f24f4ff1b4e262866f3b48a7c6` | `scripts/v4_save_bundle.py` (retrains the five seed models; needs the data table above). |
| `results/**/*.npz`, `results/repro_cleanenv/`, `results/v4/*.npy` | ~150 MB | n/a | per-fold prediction/intermediate files re-created by the corresponding `scripts/*.py`; the summary CSV/JSON/JSONL files and `results/v4/final*/` predictions are tracked. |
| `legacy/models_original/` | ~50 MB | n/a | superseded model files of the first manuscript version. |

Not in the project at all (and never to be committed): publisher PDFs, their Markdown conversions and the raw LLM output files (copyright), and Hugging Face caches. The extraction outputs are consumed through `data/lit_v4.json` (tracked, 6 MB).

Provenance note: `legacy/NOTES_retrieval_history.md` documents that earlier retrieval scripts used routes beyond the legal-only released `scripts/retrieve_fulltext.py`; the manuscript (Section 2.2.1) deliberately makes no claim that the historical corpus was retrieved exclusively through legal routes. This is an open decision for the authors.
