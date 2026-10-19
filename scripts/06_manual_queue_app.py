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
Extract structured material data from a Markdown paper on perovskite solar cells.
Return ONLY a single JSON object:
{
  "source": "DOI: <doi>",
  "materials": [
    {
      "formula": "...", "common_name": "...", "type": "perovskite|ETL|HTL|contact|additive|solvent|...",
      <all properties: deposition_method, spin_speed_rpm, anneal_temp_C, anneal_time_min,
       thickness_nm, concentration, antisolvent, PCE, Voc, Jsc, FF, ...>
    }
  ]
}
Do NOT invent data. Drop stability/EQE. Raw JSON only, no fences.
"""


def load_queue():
    if not MANUAL_QUEUE.exists():
        return []
    import csv
    return list(csv.DictReader(open(MANUAL_QUEUE)))


def pdf_to_markdown(pdf_bytes):
    import pymupdf4llm, pymupdf
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_bytes)
        tmp = f.name
    return pymupdf4llm.to_markdown(pymupdf.open(tmp))


def call_llm(markdown):
    r = requests.post(
        f"{LLM_ENDPOINT}/chat/completions",
        json={"model": LLM_MODEL,
              "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                           {"role": "user",   "content": markdown}],
              "temperature": 0.1, "max_tokens": 16384},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def clean_json(raw):
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    return re.sub(r"\s*```$", "", raw).strip()


def extraction_path(doi):
    return EXTRACTIONS_DIR / (re.sub(r"[^A-Za-z0-9.\-]+", "_", doi.strip()) + ".json")


st.set_page_config(page_title="Manual PDF Queue", layout="wide")
st.title("Manual PDF Queue")
st.caption(f"`{LLM_ENDPOINT}`  •  `{LLM_MODEL or 'set LLM_MODEL in .env'}`")

queue     = load_queue()
done      = {p.stem.replace("_", "/", 1) for p in EXTRACTIONS_DIR.glob("*.json")}
remaining = [r for r in queue if r["doi"] not in done]

if not queue:
    st.info("Run 05_retrieve_pdfs.py first.")
    st.stop()

st.markdown(f"**{len(remaining)} remaining** of {len(queue)}  •  {len(done)} done")

if not remaining:
    st.success("All done. Run 07_extract_features.py next.")
    st.stop()

row      = st.selectbox("Paper", remaining, format_func=lambda r: r["doi"])
doi      = row["doi"]
uploaded = st.file_uploader(f"Upload PDF for `{doi}`", type="pdf")

if uploaded:
    with st.spinner("Converting…"):
        try:
            md = pdf_to_markdown(uploaded.read())
            st.success(f"{len(md):,} chars")
        except Exception as e:
            st.error(f"Conversion failed: {e}")
            st.stop()

    with st.expander("Preview"):
        st.code(md[:2000])

    if st.button("Extract", type="primary"):
        if not LLM_MODEL:
            st.error("Set LLM_MODEL in .env")
            st.stop()
        with st.spinner("Extracting…"):
            try:
                data = json.loads(clean_json(call_llm(md)))
                data.setdefault("source", f"DOI: {doi}")
                out = extraction_path(doi)
                out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
                st.success(f"Saved → {out.name}  ({len(data.get('materials', []))} materials)")
                st.rerun()
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot reach `{LLM_ENDPOINT}`")
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
