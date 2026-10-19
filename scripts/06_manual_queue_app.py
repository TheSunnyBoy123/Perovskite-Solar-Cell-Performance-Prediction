"""
Step 6 — Streamlit app to process the manual download queue.

Shows papers that couldn't be retrieved automatically, lets you upload
the PDF, converts to Markdown, and sends to the LLM for extraction.

Usage:
    streamlit run scripts/06_manual_queue_app.py

Requirements:
    pip install streamlit pymupdf4llm requests python-dotenv
    Set LLM_ENDPOINT and LLM_MODEL in .env (OpenAI-compatible API).
"""
import json
import os
import re
import tempfile
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).parents[1]
load_dotenv(ROOT / ".env", override=False)

RETRIEVAL_DIR   = ROOT / "retrieval"
MANUAL_QUEUE    = RETRIEVAL_DIR / "manual_queue.csv"
EXTRACTIONS_DIR = ROOT / "extractions"
EXTRACTIONS_DIR.mkdir(exist_ok=True)

LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://localhost:8000/v1")
LLM_MODEL    = os.getenv("LLM_MODEL", "")

SYSTEM_PROMPT = """\
You are a materials-science data-extraction assistant.

Given a Markdown document converted from a published perovskite solar cell paper,
extract every material with its measured or reported properties.

Return ONLY a single JSON object:
{
  "source": "DOI: <doi>",
  "materials": [
    {
      "formula": "<chemical formula>",
      "common_name": "<human-readable name>",
      "type": "<perovskite | ETL | HTL | metal contact | additive | solvent | ...>",
      ... <all properties: deposition_method, spin_speed_rpm, anneal_temp_C,
           anneal_time_min, thickness_nm, concentration, antisolvent, ...>
    }
  ]
}

Rules:
- Include units when available. Do NOT invent data.
- Drop stability / EQE measurements; keep PCE/Voc/Jsc/FF if reported.
- Return raw JSON only — no markdown fences, no commentary.
"""


def load_queue():
    if not MANUAL_QUEUE.exists():
        return []
    import csv
    return list(csv.DictReader(open(MANUAL_QUEUE)))


def pdf_to_markdown(pdf_bytes: bytes) -> str:
    import pymupdf4llm, pymupdf
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_bytes)
        tmp = f.name
    doc = pymupdf.open(tmp)
    return pymupdf4llm.to_markdown(doc)


def call_llm(markdown: str) -> str:
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": markdown},
        ],
        "temperature": 0.1,
        "max_tokens": 16384,
    }
    r = requests.post(f"{LLM_ENDPOINT}/chat/completions", json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def clean_json(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$",          "", raw)
    return raw.strip()


def extraction_path(doi: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9.\-]+", "_", doi.strip())
    return EXTRACTIONS_DIR / f"{safe}.json"


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Manual PDF Queue", layout="wide")
st.title("Manual PDF Queue")
st.caption(f"LLM: `{LLM_ENDPOINT}`  •  model: `{LLM_MODEL or 'unset — set LLM_MODEL in .env'}`")

queue = load_queue()
done  = {p.stem.replace("_", "/", 1) for p in EXTRACTIONS_DIR.glob("*.json")}

if not queue:
    st.info("No manual queue found. Run `05_retrieve_pdfs.py` first.")
    st.stop()

remaining = [r for r in queue if r["doi"] not in done]
st.markdown(f"**{len(remaining)} remaining** of {len(queue)} total  •  {len(done)} already extracted")

if not remaining:
    st.success("All papers extracted! Run `07_extract_features.py` next.")
    st.stop()

row = st.selectbox("Select paper", remaining, format_func=lambda r: r["doi"])
doi = row["doi"]
st.markdown(f"**DOI:** `{doi}`  •  expected file: `{row['expected_filename']}`")
st.markdown(f"[Open on DOI resolver ↗](https://doi.org/{doi})")

uploaded = st.file_uploader("Upload PDF (downloaded via institutional access)", type="pdf")

if uploaded:
    with st.spinner("Converting PDF to Markdown…"):
        try:
            md = pdf_to_markdown(uploaded.read())
            st.success(f"Converted — {len(md):,} characters")
        except Exception as e:
            st.error(f"PDF conversion failed: {e}")
            st.stop()

    with st.expander("Preview first 2 000 chars"):
        st.code(md[:2000])

    if st.button("Extract with LLM", type="primary"):
        if not LLM_MODEL:
            st.error("Set LLM_MODEL in .env before extracting.")
            st.stop()
        with st.spinner("Extracting… (1–3 min)"):
            try:
                raw  = call_llm(md)
                data = json.loads(clean_json(raw))
                data.setdefault("source", f"DOI: {doi}")
                out  = extraction_path(doi)
                out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
                st.success(f"Saved → `{out.name}`  ({len(data.get('materials', []))} materials)")
                st.rerun()
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot reach LLM at `{LLM_ENDPOINT}`. Is it running?")
            except json.JSONDecodeError as e:
                st.error(f"LLM returned invalid JSON: {e}")
                st.code(raw[:2000])
