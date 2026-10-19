"""
Step 5 — Retrieve open-access PDFs for a list of DOIs.

Attempts three legal open-access routes in order, then writes a
manual_queue.csv for papers that need institutional access.

Usage:
    UNPAYWALL_EMAIL=you@institution.edu \
      python scripts/05_retrieve_pdfs.py --dois data/dois.txt --out retrieval/

Output:
    retrieval/pdfs/              downloaded PDFs (named by DOI)
    retrieval/retrieval_log.csv  per-DOI route and status
    retrieval/manual_queue.csv   DOIs that need manual download

Note: No shadow libraries are used. The manual queue covers papers only
obtainable through institutional subscriptions.
"""
import argparse
import csv
import os
import re
import time
from collections import Counter
from pathlib import Path

import requests

EMAIL = os.environ.get("UNPAYWALL_EMAIL", "")
UA = f"psc-ml-paper-retrieval/1.0 (mailto:{EMAIL or 'unset'})"
SESSION = requests.Session()
SESSION.headers["User-Agent"] = UA


def safe_name(doi: str) -> str:
    return re.sub(r"[^A-Za-z0-9.\-]+", "_", doi.strip()) + ".pdf"


def _get_json(url, **kw):
    try:
        r = SESSION.get(url, timeout=25, **kw)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def _download_pdf(url: str, dest: Path) -> int:
    """Download url if it returns a valid PDF. Returns file size or 0."""
    try:
        r = SESSION.get(url, timeout=60, allow_redirects=True)
        if r.status_code == 200 and r.content[:5] == b"%PDF-" and len(r.content) > 20_000:
            dest.write_bytes(r.content)
            return len(r.content)
    except Exception:
        pass
    return 0


def open_access_urls(doi: str):
    """Yield (route, url) pairs from Unpaywall → OpenAlex → Crossref."""
    if EMAIL:
        data = _get_json(f"https://api.unpaywall.org/v2/{doi}", params={"email": EMAIL})
        if data:
            for loc in ([data.get("best_oa_location")] + (data.get("oa_locations") or [])):
                if loc and loc.get("url_for_pdf"):
                    yield "unpaywall", loc["url_for_pdf"]

    params = {"mailto": EMAIL} if EMAIL else None
    data = _get_json(f"https://api.openalex.org/works/https://doi.org/{doi}", params=params)
    if data:
        for loc in ([data.get("best_oa_location")] + (data.get("locations") or [])):
            if loc and loc.get("pdf_url") and loc.get("is_oa", True):
                yield "openalex", loc["pdf_url"]

    data = _get_json(f"https://api.crossref.org/works/{doi}")
    if data:
        for link in data["message"].get("link", []):
            if link.get("content-type") in ("application/pdf", "unspecified") and link.get("URL"):
                yield "crossref", link["URL"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dois",  required=True, help="text file, one DOI per line")
    ap.add_argument("--out",   required=True, help="output directory")
    ap.add_argument("--limit", type=int,      help="process at most N DOIs (for testing)")
    a = ap.parse_args()

    out = Path(a.out)
    (out / "pdfs").mkdir(parents=True, exist_ok=True)
    log_path = out / "retrieval_log.csv"

    done = {}
    if log_path.exists():
        done = {r["doi"]: r for r in csv.DictReader(open(log_path))}

    dois = [d.strip() for d in open(a.dois) if d.strip()]
    if a.limit:
        dois = dois[: a.limit]

    is_new = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["doi", "route", "url", "status", "bytes"])

        for i, doi in enumerate(dois, 1):
            if doi in done:
                continue
            dest = out / "pdfs" / safe_name(doi)
            if dest.exists():
                writer.writerow([doi, "existing_file", "", "ok", dest.stat().st_size])
                f.flush()
                continue

            row = [doi, "manual_queue", "", "needs_manual", 0]
            for route, url in open_access_urls(doi):
                n = _download_pdf(url, dest)
                if n:
                    row = [doi, route, url, "ok", n]
                    break
                time.sleep(0.5)

            writer.writerow(row)
            f.flush()
            time.sleep(1.0)
            print(f"[{i}/{len(dois)}] {doi}  →  {row[1]}", flush=True)

    rows = list(csv.DictReader(open(log_path)))
    with open(out / "manual_queue.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["doi", "expected_filename"])
        for r in rows:
            if r["status"] != "ok":
                writer.writerow([r["doi"], safe_name(r["doi"])])

    routes = Counter(r["route"] for r in rows)
    n_ok   = sum(1 for r in rows if r["status"] == "ok")
    print(f"\n{n_ok}/{len(rows)} retrieved automatically. Routes: {dict(routes)}")
    print(f"Manual queue ({len(rows) - n_ok} DOIs) → {out / 'manual_queue.csv'}")
    print("Place manually downloaded PDFs in retrieval/pdfs/ using the expected filename.")
    print("Then run: streamlit run scripts/06_manual_queue_app.py")


if __name__ == "__main__":
    main()
