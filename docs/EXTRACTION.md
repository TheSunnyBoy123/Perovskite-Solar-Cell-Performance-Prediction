# LLM extraction: what the repository actually records

Extraction settings recovered from `sandbox/extract_app.py`.

## Model call
* OpenAI-compatible (vLLM) chat-completions endpoint; model alias `qwen35` (env `LLM_MODEL`; default in code).
* `temperature = 0.1`, `max_tokens = 16384`, one request per Markdown document.
* **Exact checkpoint / version / quantisation is not recorded in the repository** and must be stated by the authors (manuscript says "Qwen3.5-9B").
* The code in `extract_app.py` sends only the system prompt below and the document; no few-shot examples are present in this file (the manuscript's "multi-shot prompting" claim must be checked against the version actually run for the 2,835 JSON files).

## System prompt (verbatim)
```
You are a materials-science data-extraction assistant.

Given a Markdown document converted from a published paper, extract every
material reported with its measured or computed properties.

Return ONLY a single JSON object (no markdown fences, no commentary) with this
top-level schema:

{
  "source": "DOI: <doi-from-paper>",
  "materials": [
    {
      "formula": "<chemical formula>",
      "common_name": "<human-readable name>",
      "type": "<e.g. perovskite, ETL, HTL, metal contact>",
      ... <all properties you can find: band_gap, PV_performance, deposition_method, etc.>
    }
  ]
}

Rules:
- Always include units and measurement method when available.
- Include error bars / standard deviations if reported.
- For PV performance, separate champion vs average and HTM-free vs with-HTM.
- If the DOI is embedded in the document, use it for the "source" field.
- Do NOT invent data. Only extract what is explicitly stated.
- Return raw JSON only – no surrounding text, no code fences.
```

## PDF retrieval (released workflow)
`scripts/retrieve_fulltext.py` retrieves full texts for a DOI list using legal routes only, in this order: Unpaywall open-access PDF locations (set `UNPAYWALL_EMAIL`), OpenAlex open-access locations, publisher-deposited open full-text links from Crossref; DOIs not obtained go to `manual_queue.csv` for download through the user's own institutional or other legal access. Each DOI gets a row in `retrieval_log.csv` (route, URL, status, bytes), so the corpus share per route can be reported. PDFs, Markdown and raw LLM JSON are not redistributed.
PDF -> Markdown: OpenDataLoader PDF.
