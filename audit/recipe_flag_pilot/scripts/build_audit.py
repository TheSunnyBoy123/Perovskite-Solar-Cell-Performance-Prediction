#!/usr/bin/env python3
"""Build the recipe-flag source audit tables for the 17-paper pilot.

Writes (into audit/recipe_flag_pilot/):
  sample_manifest.csv        frame papers, source availability, exclusions
  recipe_flag_audit.csv      one row per paper x flag (612 rows)
  human_review_required.csv  unresolved, borderline and disagreeing decisions
  frame_check.json           reproduction of data/lit_v4.json flags from the combined JSON

Exits non-zero if a decision is missing for a flag with source hits, or if an anchor
phrase cannot be found in the source text (so no evidence string is ever typed by hand).

Run from anywhere:  python3 scripts/build_audit.py
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_common import (FLAGS, LIT_V4, OUT, PDF_DIR, REASONS, flatten, find_anchor,  # noqa: E402
                          frame, heading_before, json_alias_present, json_matches,
                          json_flags, json_text_lower, load_combined, regex_hits,
                          source_text, window, COMBINED_JSON, MARKDOWN_DIR)
from source_decisions import DECISIONS, decision_for  # noqa: E402

AUDIT_COLS = [
    "doi", "markdown_file", "flag", "json_flag", "json_record", "json_role", "json_matched_text",
    "json_alias_present", "regex_source_hits", "source_label", "source_status", "reason_code",
    "evidence_verbatim", "evidence_location", "decision_note", "borderline", "comparison",
    "discrepancy_locus", "discrepancy_category", "label_provenance", "needs_human_review",
    "review_priority",
]
PROVENANCE = "model-assisted (Claude reading of cached full text; not human-verified)"


def comparison(pred, label):
    if label == "U":
        return "UNRESOLVED"
    if pred and label == "P":
        return "TP"
    if pred and label == "N":
        return "FP"
    if not pred and label == "P":
        return "FN"
    return "TN"


def locus_and_category(cmp, reason, alias_present):
    """Where the disagreement arises: the extraction JSON or the deterministic JSON-to-flag conversion."""
    if cmp == "FP":
        if reason == "ARTIFACT":
            return "conversion", "regex over-breadth (non-target string matched)"
        if reason == "NO_SOURCE_TERM":
            return "extraction", "JSON record flagged with no source term in the full text"
        return "extraction", f"JSON record triggers flag in a non-device context ({reason})"
    if cmp == "FN":
        if alias_present:
            return "conversion", "material named in the JSON under an alias the regex does not match"
        return "extraction", "device use absent from the JSON (omission)"
    return "", ""


# Specific, verified causes of disagreement (checked against the JSON text in-session).
CAUSE_NOTES = {
    ("10.1002/adfm.201604733", "pea_family"): "JSON formula '(PEA)2(MA)...' is not matched: the regex needs PEAI/phenethyl forms, not the bare PEA abbreviation",
    ("10.1039/c9ta01893j", "cs"): "JSON formula 'Cs0.05(...)' is not matched: the regex needs CsI/CsBr/CsPb/cesium",
    ("10.1016/j.nanoen.2018.10.011", "cs"): "JSON formula with Cs0.05/CsI text is not matched by the cs regex",
    ("10.1039/c7se00100b", "etl_dopant"): "JSON 'Sn-doped TiO2' photo-anode is mapped to role 'contact' by role(), which is outside the etl_dopant scope",
    ("10.1021/acs.jpcc.5b06269", "pea_family"): "methylammonium iodide is matched by the 'ammonium' term",
    ("10.1088/1361-6463/ab1626", "carbon"): "JSON names PC71BM; the regex 'pcbm' does not match the PC71BM spelling",
    ("10.1007/s10118-017-1891-z", "carbon"): "JSON names PC71BM; the regex 'pcbm' does not match the PC71BM spelling",
    ("10.1007/s10118-017-1891-z", "lewis_base"): "JSON names DMSO; the lewis_base regex matches 'sulfoxide' only, not the DMSO abbreviation",
}


def main():
    rows_frame, excluded = frame()
    J = load_combined()
    lit = json.loads(LIT_V4.read_text())["papers"]

    # ---- reproduction check: combined JSON flags vs the stored lit_v4 flags ----
    shared = [d for d in J if d in lit or d.replace("/", "/") in lit]
    from psc.data import normalize_doi  # noqa: E402
    Jn = {normalize_doi(k): v for k, v in J.items()}
    diff = 0
    for d in set(Jn) & set(lit):
        f, _ = json_flags(d, Jn)
        stored = lit[d]["recipe"]
        if any(int(f[k]) != int(stored.get(k, 0) == 1.0) for k in FLAGS):
            diff += 1
    frame_check = {"combined_json_records": len(J), "lit_v4_papers": len(lit),
                   "shared_dois": len(set(Jn) & set(lit)), "shared_dois_with_any_flag_difference": diff,
                   "flags_compared": len(FLAGS)}
    (OUT / "frame_check.json").write_text(json.dumps(frame_check, indent=2))

    # ---- decisions must cover every flag with source hits; anchors must match ----
    errors, audit_rows, manifest_rows = [], [], []
    per_paper = {}
    for r in rows_frame:
        doi, md = r["doi"], r["md_path"]
        raw = source_text(md)
        flat = flatten(raw)
        hits = regex_hits(flat)
        jflags, _ = json_flags(doi, Jn)
        jm = json_matches(doi, Jn)
        jtext = json_text_lower(Jn, doi)
        pdf_ok = (PDF_DIR / f"{r['stem']}.pdf").exists() or (PDF_DIR / r["stem"]).exists()
        per_paper[doi] = {"stem": r["stem"]}
        manifest_rows.append({
            "doi": doi, "markdown_file": md.name, "pdf_file_present": pdf_ok,
            "markdown_chars": len(raw), "extraction_record_present": True,
            "json_flags_predicted": sum(jflags.values()),
            "fully_auditable": True,
            "sampling_stratum": "census of the auditable frame (all 17 with full text + extraction record)",
            "selection_rationale": "source availability; selection did not use flag values or audit outcomes",
        })
        for k in FLAGS:
            dec = decision_for(doi, k)
            if dec is None:
                if hits[k]["n"] > 0:
                    errors.append(f"{doi} {k}: {hits[k]['n']} source hits but no decision")
                    continue
                dec = {"label": "N", "reason": "NO_SOURCE_TERM", "note": "", "borderline": False, "anchor": None}
            label = dec["label"]
            evidence, loc = "", ""
            if label == "P":
                idx = find_anchor(flat, dec["anchor"])
                if idx < 0:
                    errors.append(f"{doi} {k}: anchor not found: {dec['anchor']!r}")
                    continue
                evidence = window(flat, idx, len(dec["anchor"]))
                loc = f"char {idx}; section: {heading_before(flat, idx) or 'n/a'}"
            else:
                anchor = dec.get("anchor")
                idx = find_anchor(flat, anchor) if anchor else -1
                if anchor and idx < 0:
                    errors.append(f"{doi} {k}: anchor not found: {anchor!r}")
                    continue
                if idx < 0 and hits[k]["n"] > 0:
                    idx = hits[k]["first"]
                if idx >= 0:
                    evidence = window(flat, idx, 0)
                    loc = f"char {idx}; section: {heading_before(flat, idx) or 'n/a'}"
                else:
                    evidence = "no regex hit in full text"
            pred = jflags[k]
            cmp = comparison(pred, label)
            alias = json_alias_present(k, jtext)
            locus, cat = locus_and_category(cmp, dec.get("reason", ""), alias)
            if (doi, k) in CAUSE_NOTES and cmp in ("FP", "FN"):
                cat = CAUSE_NOTES[(doi, k)]
            m = jm.get(k, {})
            if label == "P":
                status = "SUPPORTED_POSITIVE"
            elif label == "N":
                status = "SUPPORTED_NEGATIVE"
            else:
                status = "UNRESOLVED"
            human = (label == "U") or cmp in ("FP", "FN") or bool(dec.get("borderline"))
            prio = 1 if label == "U" else (2 if cmp in ("FP", "FN") or dec.get("borderline") else 3)
            audit_rows.append({
                "doi": doi, "markdown_file": md.name, "flag": k, "json_flag": pred,
                "json_record": m.get("record", ""), "json_role": m.get("role", ""),
                "json_matched_text": m.get("matched", ""), "json_alias_present": int(alias),
                "regex_source_hits": hits[k]["n"], "source_label": label, "source_status": status,
                "reason_code": dec.get("reason", "POSITIVE" if label == "P" else ""),
                "evidence_verbatim": evidence, "evidence_location": loc,
                "decision_note": dec.get("note", ""), "borderline": int(bool(dec.get("borderline"))),
                "comparison": cmp, "discrepancy_locus": locus, "discrepancy_category": cat,
                "label_provenance": PROVENANCE, "needs_human_review": int(human),
                "review_priority": prio,
            })
    if errors:
        print("\n".join(errors), file=sys.stderr)
        sys.exit(f"{len(errors)} audit errors; nothing written")

    # ---- write tables ----
    with open(OUT / "recipe_flag_audit.csv", "w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=AUDIT_COLS)
        w.writeheader()
        w.writerows(audit_rows)

    with open(OUT / "sample_manifest.csv", "w", newline="", encoding="utf8") as fh:
        cols = ["doi", "markdown_file", "pdf_file_present", "markdown_chars", "extraction_record_present",
                "json_flags_predicted", "fully_auditable", "sampling_stratum", "selection_rationale"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(manifest_rows)
        for e in excluded:
            stem = e["stem"]
            w.writerow({"doi": stem.replace("_", "/", 1), "markdown_file": Path(e["md_path"]).name,
                        "pdf_file_present": (PDF_DIR / f"{stem}.pdf").exists() or (PDF_DIR / stem).exists(),
                        "markdown_chars": "", "extraction_record_present": False,
                        "json_flags_predicted": "", "fully_auditable": False,
                        "sampling_stratum": "excluded",
                        "selection_rationale": "EXCLUDED: no extraction record in the combined JSON (not auditable against flags)"})

    review = [r for r in audit_rows if r["needs_human_review"]]
    review.sort(key=lambda r: (r["review_priority"], r["comparison"], r["doi"], r["flag"]))
    with open(OUT / "human_review_required.csv", "w", newline="", encoding="utf8") as fh:
        cols = ["review_priority", "doi", "flag", "comparison", "source_label", "reason_code", "json_flag",
                "json_matched_text", "discrepancy_locus", "discrepancy_category", "borderline",
                "evidence_verbatim", "decision_note", "human_decision", "human_reviewer", "human_notes"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in review:
            w.writerow({**{c: r.get(c, "") for c in cols if c not in ("human_decision", "human_reviewer", "human_notes")},
                        "human_decision": "", "human_reviewer": "", "human_notes": ""})

    cmp_counts = Counter(r["comparison"] for r in audit_rows)
    print(f"papers={len(rows_frame)} excluded={len(excluded)} decisions={len(audit_rows)} "
          f"review_rows={len(review)} comparisons={dict(cmp_counts)}")


if __name__ == "__main__":
    main()
