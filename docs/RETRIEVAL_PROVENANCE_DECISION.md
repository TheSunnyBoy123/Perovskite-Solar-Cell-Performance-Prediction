# Retrieval provenance: facts and open decision

**Recorded facts** (from `legacy/NOTES_retrieval_history.md`, recovered from the project sandbox during the audit)
* `simple_download.py`: tries OpenAlex open-access locations first, then Sci-Hub (`sci-hub.ru`, `sci-hub.st`).
* `doi2pdf_fixed.py`, `download_pdfs.py`: Sci-Hub-based downloaders; the sandbox README also describes manual download into `pdfs/`.
* PDF to Markdown: OpenDataLoader PDF / marker endpoint.
* The released `scripts/retrieve_fulltext.py` is legal-route only (Unpaywall/OpenAlex open-access, Crossref publisher links, manual institutional download) and logs the route per DOI. It was written for release and is **not** the script that produced the historical corpus.
* The route by which each of the 3,299 historical PDFs was obtained is **not logged**; therefore nothing in the repository proves the corpus was obtained legally, or the opposite for any specific PDF.

**What the manuscript says** (`ml.tex`, Section 2.2.1): describes the released script as legal-only, states that the historical corpus was collected with earlier recovered scripts (OpenAlex first, then Sci-Hub mirrors, plus manual downloads) with no per-PDF route log, and makes no claim of exclusively legal retrieval. PDFs, Markdown and raw LLM JSON are not redistributed; only derived paper-level features are.

**Decision needed from the authors**
1. Final wording of the provenance statement (keep the explicit disclosure, or replace with a different accurate description).
2. Whether the derived features may be released/used under publisher terms and journal policy.
3. Whether to regenerate the extraction from a corpus retrieved through documented legal routes (not done here; no data were deleted or regenerated).
