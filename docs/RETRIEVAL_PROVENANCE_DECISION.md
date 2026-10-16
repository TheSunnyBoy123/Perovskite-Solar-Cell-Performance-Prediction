# Retrieval provenance note (supersedes the earlier "open decision" version)

**Author-provided account (used in the manuscript).** Papers were identified by DOI. Full texts were retrieved over the institutional network using the institution's subscriptions and open-access availability. When automated retrieval failed, some papers were obtained manually through institutional access. Sci-Hub was dropped and was not used to obtain the study corpus.

**Why the stale statement existed.** `legacy/NOTES_retrieval_history.md` records that scripts recovered from `sandbox/` (`simple_download.py`, `doi2pdf_fixed.py`, `download_pdfs.py`) contain OpenAlex-then-Sci-Hub fallbacks. The audit read their presence as evidence of how the corpus was obtained. Those files and notes are preserved unchanged as project history.

**What is and is not established.** The route for each of the 3,299 PDFs was not logged, so neither account can be verified per document from the repository. The manuscript therefore states the author account and explicitly says that per-PDF routes were not logged. It makes no legal assurance.

**Actions that remain with the authors (outside the results).**
1. Because the recovered scripts do include Sci-Hub fallbacks, confirm that none of the 3,299 PDFs came through them (for example by checking download logs or file provenance) or amend the Methods.
2. Institutional network access is not by itself proof of text-and-data-mining permission. Confirm with the library or the publishers whether LLM-based extraction of these texts is covered, and whether journal policy requires a statement.
3. PDFs, Markdown and raw LLM JSON are not redistributed; only derived artifacts are (see the Data and code availability statement).
