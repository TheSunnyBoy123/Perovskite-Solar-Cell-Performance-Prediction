# Journal-readiness checklist (Digital Discovery, RSC)

The manuscript text contains no placeholders. The items below are **submission-stage actions** that cannot be performed from this workspace (they need accounts, credentials or the authors' own decisions).
| Item | Status |
|---|---|
| Code deposited in persistent repository (Zenodo) with DOI | Action at acceptance: publish a Zenodo release of `ml_paper/` (private/anonymous link for reviewers); add the DOIs to the Data Availability Statement (text already says this) |
| Data availability statement | Written; DOI to be added at acceptance |
| Processed data table shared | Yes (`data/dataset_json_enriched.csv`, 31 MB, SHA-256 in manifests); PDP is CC BY |
| Full texts / Markdown / raw LLM JSON | Not shareable (copyright) — stated |
| Retrieval route, Qwen version, prompt | Prompt in SI; released retrieval workflow is legal-only with a provenance log; **authors must make sure the full-text corpus behind the reported features is consistent with the described workflow (re-run it, or restrict to documents obtained legally) before submission**; checkpoint stated as unrecorded |
| AI-tool disclosure | Written; verify against RSC's current policy page at submission (the page returned HTTP 403 to automated fetch) |
| Citations verified | 13 original DOIs verified by Crossref; 5 errors fixed; 10 entries added (4 author lists Crossref-verified); Fukasawa/Roberts checked against PDFs. Qwen reference still to add |
| LaTeX compiles without undefined refs | Yes |
| Supplementary information | `psc_manuscript/si.pdf` (feature lists, per-split results, split integrity, prompt, grids) |
| Tests / reproduction | 29 passing; full 270-fit ablation reproduced in a clean venv (max ΔR² 1.1e-4) |
| Secrets | none in `src/ scripts/ configs/ docs/ tests/ ml.tex` (scanned) |
| Remaining red TODOs in manuscript | 0 |
