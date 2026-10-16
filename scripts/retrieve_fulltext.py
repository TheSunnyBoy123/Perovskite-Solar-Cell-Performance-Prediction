#!/usr/bin/env python3
"""Legal full-text retrieval for a DOI list, with a per-DOI provenance log.

Order of attempts (stops at the first that returns a real PDF):
  1. Unpaywall      best open-access PDF location (needs UNPAYWALL_EMAIL)
  2. OpenAlex       open-access locations / pdf_url (repositories, PMC, arXiv, publisher OA)
  3. Crossref       publisher-deposited full-text links that are openly accessible (text-mining links)
  4. manual queue   DOIs not obtained are written to <out>/manual_queue.csv for download through your
                    own institutional/legal access; place the PDFs in <out>/pdfs/ named by `safe_name`.
No shadow libraries are used. Every DOI gets one row in <out>/retrieval_log.csv (doi, route, url, status, bytes),
so the share of the corpus per route can be reported.

Usage: UNPAYWALL_EMAIL=you@inst.edu python scripts/retrieve_fulltext.py --dois dois.txt --out retrieval/ [--limit N]
Polite use: identifies itself with a mailto, sleeps between requests, resumes from the log.
"""
import argparse, csv, os, re, sys, time
from pathlib import Path
import requests

EMAIL = os.environ.get("UNPAYWALL_EMAIL", "")
UA = f"psc-ml-paper-retrieval/1.0 (mailto:{EMAIL or 'unset'})"
S = requests.Session(); S.headers["User-Agent"] = UA


def safe_name(doi):
    return re.sub(r"[^A-Za-z0-9.\-]+", "_", doi.strip()) + ".pdf"


def get_json(url, **kw):
    try:
        r = S.get(url, timeout=25, **kw)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def try_pdf(url, dest):
    """Download url; accept only genuine PDFs (magic bytes). Returns number of bytes or 0."""
    try:
        r = S.get(url, timeout=60, allow_redirects=True)
        if r.status_code == 200 and r.content[:5] == b"%PDF-" and len(r.content) > 20_000:
            dest.write_bytes(r.content); return len(r.content)
    except Exception:
        pass
    return 0


def candidates(doi):
    if EMAIL:
        u = get_json(f"https://api.unpaywall.org/v2/{doi}", params={"email": EMAIL})
        if u:
            for loc in ([u.get("best_oa_location")] + (u.get("oa_locations") or [])):
                if loc and loc.get("url_for_pdf"):
                    yield "unpaywall", loc["url_for_pdf"]
    w = get_json(f"https://api.openalex.org/works/https://doi.org/{doi}", params={"mailto": EMAIL} if EMAIL else None)
    if w:
        locs = [w.get("best_oa_location")] + (w.get("locations") or [])
        for loc in locs:
            if loc and loc.get("pdf_url") and loc.get("is_oa", True):
                yield "openalex", loc["pdf_url"]
    c = get_json(f"https://api.crossref.org/works/{doi}")
    if c:
        for l in c["message"].get("link", []):
            if l.get("content-type") in ("application/pdf", "unspecified") and l.get("URL"):
                yield "crossref", l["URL"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dois", required=True, help="text file, one DOI per line")
    ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    out = Path(a.out); (out / "pdfs").mkdir(parents=True, exist_ok=True)
    log_f = out / "retrieval_log.csv"; done = {}
    if log_f.exists():
        done = {r["doi"]: r for r in csv.DictReader(open(log_f))}
    dois = [d.strip() for d in open(a.dois) if d.strip()][: a.limit]
    new = not log_f.exists()
    with open(log_f, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["doi", "route", "url", "status", "bytes"])
        for i, doi in enumerate(dois, 1):
            if doi in done: continue
            dest = out / "pdfs" / safe_name(doi)
            if dest.exists():
                w.writerow([doi, "existing_file", "", "ok", dest.stat().st_size]); f.flush(); continue
            row = [doi, "manual_queue", "", "needs_manual", 0]
            for route, url in candidates(doi):
                n = try_pdf(url, dest)
                if n: row = [doi, route, url, "ok", n]; break
                time.sleep(0.5)
            w.writerow(row); f.flush(); time.sleep(1.0)
            print(f"[{i}/{len(dois)}] {doi} -> {row[1]}", flush=True)
    rows = list(csv.DictReader(open(log_f)))
    with open(out / "manual_queue.csv", "w", newline="") as f:
        mw = csv.writer(f); mw.writerow(["doi", "expected_filename"])
        for r in rows:
            if r["status"] != "ok": mw.writerow([r["doi"], safe_name(r["doi"])])
    from collections import Counter
    print("routes:", dict(Counter(r["route"] for r in rows)))


if __name__ == "__main__":
    main()
