# Retrieval history (internal note, not part of the release)

Recovered from sandbox/ during the audit. The recovered scripts included shadow-library routes, which the authors decided not to use in the released pipeline or describe as the method.

## PDF retrieval route (as implemented)
* `simple_download.py`: "tries OpenAlex OA first, then SciHub" (domains `sci-hub.ru`, `sci-hub.st`).
* `doi2pdf_fixed.py`, `download_pdfs.py`: Sci-Hub based downloaders; `sandbox/README.md` additionally describes manual downloading into `pdfs/`.
* PDF -> Markdown: OpenDataLoader PDF / marker endpoint (`watcher.py`, GPU marker endpoint).

**Consequence for the manuscript.** The retrieval statement cannot say "open-access" only. How the retrieval route is described, and whether the full-text-derived features may be released, is a decision for the authors (publisher terms / copyright / journal policy). The manuscript keeps an explicit TODO for it. Derived paper-level numeric features are redistributable at the authors' discretion; PDFs, Markdown and raw JSON are not included in the package.
