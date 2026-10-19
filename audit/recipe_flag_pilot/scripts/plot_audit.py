#!/usr/bin/env python3
"""Figures for the recipe-flag source audit. Reads only the saved audit tables.

Writes PDF and PNG (200 dpi) into audit/recipe_flag_pilot/figures/:
  fig1_flag_precision_recall   per-flag precision and recall with exact 95% CIs
  fig2_confusion_heatmap       TP/FP/TN/FN counts per flag; unresolved kept in a separate column
  fig3_coverage_errors         resolved vs unresolved decisions, and discrepancy categories
  fig4_evidence_panel          verbatim evidence for representative confirmed disagreements

Style follows scripts/make_figures.py in the manuscript repository (font 10, no top/right
spines, Agg backend, PNG + PDF).
"""
import csv
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_common import OUT  # noqa: E402

FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
C_ADEQ = "#2471a3"   # t2 blue from the manuscript palette
C_LOW = "#9aa5b1"    # t1 grey
C_FP = "#c0392b"
C_FN = "#8e44ad"    # t3 purple
C_UNRES = "#d5d8dc"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_summary():
    rows = list(csv.DictReader(open(OUT / "audit_summary.csv", encoding="utf8")))
    for r in rows:
        for k in ("precision", "recall", "precision_ci95_low", "precision_ci95_high",
                  "recall_ci95_low", "recall_ci95_high", "specificity", "F1"):
            r[k] = float(r[k]) if r[k] not in ("NA", "") else math.nan
        for k in ("TP", "FP", "TN", "FN", "unresolved", "reviewable_source_positive",
                  "reviewable_source_negative", "predicted_positive", "decisions"):
            r[k] = int(r[k])
    return rows


def fig1(rows):
    sel = [r for r in rows if r["reviewable_source_positive"] > 0 or r["predicted_positive"] > 0]
    sel.sort(key=lambda r: (-r["reviewable_source_positive"], r["flag"]))
    y = np.arange(len(sel))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 0.28 * len(sel) + 1.4), sharey=True)
    for ax, metric in zip(axes, ("precision", "recall")):
        for yi, r in zip(y, sel):
            val = r[metric]
            lo, hi = r[f"{metric}_ci95_low"], r[f"{metric}_ci95_high"]
            adeq = r["support"] == "adequate"
            col = C_ADEQ if adeq else C_LOW
            if math.isnan(val):
                ax.text(0.02, yi, "undefined", va="center", fontsize=7, color="#555555")
                continue
            ax.plot([lo, hi], [yi, yi], color=col, lw=1.2, zorder=1)
            ax.scatter([val], [yi], s=26, color=col if adeq else "white",
                       edgecolor=col, linewidth=1.2, zorder=2)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel(metric.capitalize() + " (exact 95% CI)")
        ax.grid(axis="x", color="#eeeeee", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r["flag"] for r in sel], fontsize=7.5)
    for yi, r in zip(y, sel):
        axes[1].text(1.03, yi, f"P={r['reviewable_source_positive']}  pred+={r['predicted_positive']}",
                     va="center", fontsize=6.5, color="#333333", transform=axes[1].get_yaxis_transform())
    fig.suptitle("Per-flag precision and recall, 17-paper pilot (filled blue = adequate support; "
                 "open grey = low support)", fontsize=8.5, y=1.02)
    fig.tight_layout()
    save(fig, "fig1_flag_precision_recall")


def fig2(rows):
    cols = ["TP", "FP", "TN", "FN"]
    data = np.array([[r[c] for c in cols] + [r["unresolved"]] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(4.8, 0.22 * len(rows) + 1.2))
    shown = np.log1p(data[:, :4])
    im = ax.imshow(shown, aspect="auto", cmap="Blues", vmin=0, vmax=max(1e-9, shown.max()))
    ax.imshow(np.zeros_like(data[:, 4:]), aspect="auto", cmap="Greys", vmin=0, vmax=1,
              extent=(3.5, 4.5, len(rows) - 0.5, -0.5))
    for i in range(len(rows)):
        for j in range(4):
            ax.text(j, i, int(data[i, j]), ha="center", va="center", fontsize=6.5,
                    color="white" if shown[i, j] > shown.max() * 0.6 else "#222222")
        ax.text(4, i, int(data[i, 4]), ha="center", va="center", fontsize=6.5, color="#333333")
    ax.set_xticks(list(range(5)))
    ax.set_xticklabels(cols + ["UNRES."], fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["flag"] for r in rows], fontsize=6.8)
    ax.axvline(3.5, color="#555555", lw=0.8)
    ax.set_title("Decision counts per flag (colour: log count; unresolved shown separately)", fontsize=8)
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="log(1+count)")
    fig.tight_layout()
    save(fig, "fig2_confusion_heatmap")


def fig3():
    rows = list(csv.DictReader(open(OUT / "recipe_flag_audit.csv", encoding="utf8")))
    cmp_order = ["TP", "TN", "FP", "FN", "UNRESOLVED"]
    cnt = Counter(r["comparison"] for r in rows)
    colours = {"TP": C_ADEQ, "TN": "#aed6f1", "FP": C_FP, "FN": C_FN, "UNRESOLVED": C_UNRES}
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [1, 1.25]})
    left = 0
    total = sum(cnt[c] for c in cmp_order)
    for c in cmp_order:
        a.barh([0], [cnt[c]], left=left, color=colours[c], edgecolor="white", label=f"{c} ({cnt[c]})")
        if cnt[c] > 25:
            a.text(left + cnt[c] / 2, 0, f"{cnt[c]}", ha="center", va="center", fontsize=7.5)
        left += cnt[c]
    a.set_xlim(0, total)
    a.set_yticks([])
    a.set_xlabel(f"Paper-flag decisions (n = {total})")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=3, fontsize=6.5, frameon=False)
    a.set_title("Decision outcomes", fontsize=8.5)
    def bucket(c):
        if "omission" in c:
            return "device use omitted from JSON (extraction)"
        if "over-breadth" in c or "regex" in c or "not matched" in c or "does not match" in c or "alias" in c or "mapped to role" in c or "matched by" in c:
            return "regex or role mapping misses/overreaches (conversion)"
        if "non-device" in c or "no source term" in c:
            return "JSON record without device support (extraction)"
        return "other"
    cat = Counter(bucket(r["discrepancy_category"]) for r in rows if r["comparison"] in ("FP", "FN"))
    items = sorted(cat.items(), key=lambda kv: -kv[1])
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    yy = np.arange(len(items))[::-1]
    b.barh(yy, vals, color=[C_FN if "omitted" in lab else (C_FP if "conversion" in lab else C_LOW)
                             for lab in labels])
    b.set_yticks(yy)
    b.set_yticklabels(labels, fontsize=6.5)
    for yi, v in zip(yy, vals):
        b.text(v + 0.3, yi, str(v), va="center", fontsize=7)
    b.set_xlabel("Discrepant decisions (FP + FN)")
    b.set_title("Discrepancy categories", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig3_coverage_errors")


def _display(text, anchor_hint, width=118, max_lines=3):
    """Display-only cleanup: strip HTML tags and markdown emphasis; the CSV keeps the verbatim text."""
    import re
    import textwrap
    clean = re.sub(r"<[^>]+>", "", text).replace("***", "").replace("*", "").replace("$", "")
    clean = re.sub(r"\s+", " ", clean).strip()
    lines = textwrap.wrap(clean, width=width)[:max_lines]
    return "\n".join(lines)


def fig4():
    import textwrap
    rows = list(csv.DictReader(open(OUT / "recipe_flag_audit.csv", encoding="utf8")))
    pick = [
        ("10.1038/ncomms13503", "htl_tbp"),
        ("10.1002/adfm.201604733", "pea_family"),
        ("10.1039/c7se00100b", "etl_dopant"),
        ("10.1088/1361-6463/ab1626", "carbon"),
    ]
    sel = [next(x for x in rows if x["doi"] == doi and x["flag"] == flag) for doi, flag in pick]
    fig, ax = plt.subplots(figsize=(7.6, 2.1 * len(sel) + 0.6))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(sel))
    for i, r in enumerate(sel):
        top = len(sel) - i
        head = (f"{r['flag']}   JSON flag = {r['json_flag']}   source = {r['source_label']} "
                f"({'supported positive' if r['source_label'] == 'P' else 'supported negative'})")
        ax.text(0.0, top - 0.08, head, fontsize=8, weight="bold", va="top")
        ev = _display(r["evidence_verbatim"], None, width=125, max_lines=2)
        ax.text(0.0, top - 0.32, textwrap.fill("Evidence: \u201c" + ev.replace("\n", " ") + "\u201d", 125),
                fontsize=6.8, va="top", linespacing=1.3)
        cat = f"Category: {r['discrepancy_category']} ({r['discrepancy_locus']})   Paper: {r['doi']}"
        ax.text(0.0, top - 0.78, textwrap.fill(cat, 125), fontsize=6.4, color="#444444", va="top")
    ax.set_title("Representative confirmed disagreements (evidence extracted verbatim from the source; "
                 "markup removed for display)", fontsize=8)
    save(fig, "fig4_evidence_panel")


def main():
    rows = load_summary()
    fig1(rows)
    fig2(rows)
    fig3()
    fig4()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
