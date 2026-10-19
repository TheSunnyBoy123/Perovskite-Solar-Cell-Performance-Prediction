#!/usr/bin/env python3
"""Per-flag and overall diagnostic metrics from recipe_flag_audit.csv.

Only resolved decisions (source label P or N) enter the confusion counts; unresolved
decisions are counted separately and never treated as correct or incorrect.

Writes audit_summary.csv and metrics_overall.json into audit/recipe_flag_pilot/.
Exact (Clopper-Pearson) 95% intervals are used for precision and recall. Precision and
recall are reported as NA when their denominator is zero.

Caveat: the pilot is a census of the auditable frame, not a random sample, so these are
descriptive for this set of 17 papers and are not corpus-wide estimates.
"""
import csv
import json
import math
import sys
from pathlib import Path

from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_common import FLAGS, OUT  # noqa: E402

MIN_SUPPORT = 3  # reviewable cases needed before a per-flag estimate is reported as adequately supported


def clopper_pearson(k, n, alpha=0.05):
    if n == 0:
        return (math.nan, math.nan)
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def ratio(num, den):
    return math.nan if den == 0 else num / den


def f1(tp, fp, fn):
    den = 2 * tp + fp + fn
    return math.nan if den == 0 else 2 * tp / den


def flag_row(flag, rows):
    rs = [r for r in rows if r["flag"] == flag]
    c = {k: sum(1 for r in rs if r["comparison"] == k) for k in ("TP", "FP", "TN", "FN", "UNRESOLVED")}
    resolved = c["TP"] + c["FP"] + c["TN"] + c["FN"]
    pos_src = c["TP"] + c["FN"]        # reviewable positives (source-supported)
    neg_src = c["TN"] + c["FP"]        # reviewable negatives
    pred_pos = c["TP"] + c["FP"]
    prec = ratio(c["TP"], pred_pos)
    rec = ratio(c["TP"], pos_src)
    spec = ratio(c["TN"], neg_src)
    p_lo, p_hi = clopper_pearson(c["TP"], pred_pos)
    r_lo, r_hi = clopper_pearson(c["TP"], pos_src)
    if pos_src >= MIN_SUPPORT and pred_pos >= MIN_SUPPORT:
        support = "adequate"
    elif pos_src == 0 and pred_pos == 0:
        support = "no reviewable positives or predicted positives"
    elif pos_src == 0:
        support = "no source-supported positives (recall undefined)"
    else:
        support = "low (fewer than 3 positive or predicted-positive cases)"
    return {
        "flag": flag, "decisions": len(rs), "resolved": resolved, "unresolved": c["UNRESOLVED"],
        "reviewable_source_positive": pos_src, "reviewable_source_negative": neg_src,
        "predicted_positive": pred_pos, "TP": c["TP"], "FP": c["FP"], "TN": c["TN"], "FN": c["FN"],
        "precision": prec, "precision_ci95_low": p_lo, "precision_ci95_high": p_hi,
        "recall": rec, "recall_ci95_low": r_lo, "recall_ci95_high": r_hi,
        "specificity": spec, "F1": f1(c["TP"], c["FP"], c["FN"]),
        "support": support,
        "borderline_decisions": sum(1 for r in rs if r["borderline"] == "1"),
        "human_review_rows": sum(1 for r in rs if r["needs_human_review"] == "1"),
    }


def fmt(x):
    if isinstance(x, float):
        return "NA" if math.isnan(x) else f"{x:.3f}"
    return x


def main():
    rows = list(csv.DictReader(open(OUT / "recipe_flag_audit.csv", encoding="utf8")))
    per_flag = [flag_row(f, rows) for f in FLAGS]
    cols = list(per_flag[0].keys())
    with open(OUT / "audit_summary.csv", "w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in per_flag:
            w.writerow({k: (fmt(v) if isinstance(v, float) else v) for k, v in r.items()})

    tot = {k: sum(r[k] for r in per_flag) for k in ("TP", "FP", "TN", "FN", "unresolved")}
    tot["UNRESOLVED"] = tot.pop("unresolved")
    resolved = tot["TP"] + tot["FP"] + tot["TN"] + tot["FN"]
    pooled_prec = ratio(tot["TP"], tot["TP"] + tot["FP"])
    pooled_rec = ratio(tot["TP"], tot["TP"] + tot["FN"])
    precs = [r["precision"] for r in per_flag if not math.isnan(r["precision"])]
    recs = [r["recall"] for r in per_flag if not math.isnan(r["recall"])]
    papers = sorted({r["doi"] for r in rows})
    fully = {d: all(x["source_label"] in ("P", "N") for x in rows if x["doi"] == d) for d in papers}
    paper_unres = {d: sum(1 for x in rows if x["doi"] == d and x["comparison"] == "UNRESOLVED") for d in papers}
    overall = {
        "papers_audited": len(papers), "paper_flag_decisions": len(rows), "flags": len(FLAGS),
        "resolved_decisions": resolved, "unresolved_decisions": tot["UNRESOLVED"],
        "unresolved_fraction": tot["UNRESOLVED"] / len(rows),
        "source_positive_decisions": tot["TP"] + tot["FN"], "source_negative_decisions": tot["TN"] + tot["FP"],
        "pooled_TP": tot["TP"], "pooled_FP": tot["FP"], "pooled_TN": tot["TN"], "pooled_FN": tot["FN"],
        "overall_agreement_resolved": (tot["TP"] + tot["TN"]) / resolved,
        "micro_precision": pooled_prec, "micro_recall": pooled_rec,
        "micro_precision_ci95": clopper_pearson(tot["TP"], tot["TP"] + tot["FP"]),
        "micro_recall_ci95": clopper_pearson(tot["TP"], tot["TP"] + tot["FN"]),
        "macro_precision_over_defined_flags": sum(precs) / len(precs), "macro_precision_defined_flags": len(precs),
        "macro_recall_over_defined_flags": sum(recs) / len(recs), "macro_recall_defined_flags": len(recs),
        "flags_with_adequate_support": sum(1 for r in per_flag if r["support"] == "adequate"),
        "flags_with_any_source_positive": sum(1 for r in per_flag if r["reviewable_source_positive"] > 0),
        "flags_with_no_source_positive": [r["flag"] for r in per_flag if r["reviewable_source_positive"] == 0],
        "papers_fully_resolved": sum(1 for d in papers if fully[d]),
        "papers_with_unresolved_decisions": {d: n for d, n in paper_unres.items() if n},
        "discrepancy_locus": {},
        "discrepancy_category": {},
        "paper_flag_pairs_predicted_positive": sum(1 for r in rows if r["json_flag"] == "1"),
        "paper_flag_pairs_source_positive": tot["TP"] + tot["FN"],
        "target_papers": 40, "sampling": "census of the auditable frame (not random)",
    }
    from collections import Counter
    overall["discrepancy_locus"] = dict(Counter(r["discrepancy_locus"] for r in rows if r["comparison"] in ("FP", "FN")))
    overall["discrepancy_category"] = dict(Counter(r["discrepancy_category"] for r in rows if r["comparison"] in ("FP", "FN")))
    (OUT / "metrics_overall.json").write_text(json.dumps(overall, indent=2, default=lambda x: None))
    print(json.dumps({k: overall[k] for k in ("papers_audited", "paper_flag_decisions", "resolved_decisions",
                                              "unresolved_decisions", "pooled_TP", "pooled_FP", "pooled_TN",
                                              "pooled_FN", "micro_precision", "micro_recall",
                                              "overall_agreement_resolved", "flags_with_adequate_support",
                                              "flags_with_any_source_positive", "discrepancy_locus")},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
