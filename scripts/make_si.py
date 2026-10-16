#!/usr/bin/env python3
"""Generate psc_manuscript/si.tex (Supplementary Information) from results. Run from ml_paper/."""
import json, re
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "results"
esc = lambda x: x.replace("_", "\\_")
fl = json.load(open(R / "feature_lists.json")); audit = json.load(open(R / "feature_audit.json"))
A = pd.DataFrame([json.loads(l) for l in open(R / "ablation_v1" / "runs.jsonl")]).drop(columns=["info"])
meta = json.load(open(R / "ablation_v1" / "meta.json"))
prompt = re.search(r"System prompt \(verbatim\)\n```\n(.*?)```", (ROOT / "docs" / "EXTRACTION.md").read_text(), re.S).group(1)
o = [r"""\documentclass[10pt]{article}
\usepackage[T1]{fontenc}\usepackage[margin=2cm]{geometry}\usepackage{booktabs,longtable,lmodern,url,xcolor}
\usepackage{microtype}\usepackage{hyperref}
\title{Supplementary Information: Perovskite Solar Cell Performance Prediction with LLM-Derived Feature Augmentation under Leakage-Aware DOI Evaluation}
\date{}\begin{document}\maketitle\small
\section*{Table S1. Predictors per tier}
Column sets shown for a preprocessor fitted on the full filtered data (documentation); in the experiments the \texttt{\_na} indicators and the $>$80\% missing filter are re-derived from each training partition. Tiers are nested: $t_1\subset t_2\subset t_3$.\par"""]
prev = set()
for t in ("t1", "t2", "t3"):
    cols = fl[t]; new = [c for c in cols if c not in prev]
    o.append(f"\\subsection*{{{t}: {len(cols)} predictors ({len(new)} added relative to the previous tier)}}")
    for g in ("process", "device", "composition", "material", "literature"):
        gc = [c for c in new if cols[c] == g]
        if gc: o.append(f"\\textbf{{{g}}} ({len(gc)}): " + ", ".join(f"\\texttt{{{esc(c)}}}" for c in gc) + "\\par\\smallskip")
    prev = set(cols)
o.append(r"\section*{Table S2. Per-split results of the matched CatBoost ablation (DOI-grouped)}\begin{longtable}{lllrrrr}\toprule Subset & Tier & Split & $n_\mathrm{test}$ & $R^2$ & MAE & RMSE\\\midrule\endhead")
x = A[(A.protocol == "doi") & (A.model == "catboost")].sort_values(["arch", "tier", "split"])
for _, r in x.iterrows():
    o.append(f"{r.arch} & {r.tier} & {r.split} & {r.n_test} & {r.r2:.4f} & {r.mae:.3f} & {r.rmse:.3f}\\\\")
o.append(r"\bottomrule\end{longtable}")
o.append(r"\section*{Table S3. Split integrity (all 270 recorded fits)}")
d = A[A.protocol == "doi"]
o.append(f"DOI-grouped fits: {len(d)}; maximum DOI overlap between test and train: {d.doi_overlap_test_train.max()}, validation and train: {d.doi_overlap_val_train.max()}, test and validation: {d.doi_overlap_test_val.max()}.\\par")
o.append(f"Records without DOI: {audit['n_records_no_doi']} (one block). Publications (normalised DOIs): {audit['n_doi_groups_incl_no_doi']-1}.\\par")
o.append(f"Data SHA-256: \\texttt{{{meta['data_sha256']}}}; config SHA-256: \\texttt{{{meta['config_sha256'][:16]}}}; git HEAD at run time: \\texttt{{{meta['git_head'][:12]}}} (working tree dirty: {meta['git_dirty']}); CatBoost {meta['catboost']}, scikit-learn {meta['sklearn']}, Python {meta['python']}.\\par")
o.append(r"\section*{Table S4. Literature-feature coverage and constancy within publication}\begin{tabular}{lrr}\toprule Feature & coverage & constant within DOI\\\midrule")
for c in sorted(audit["lit_coverage_per_feature"]):
    cv = audit["lit_within_doi_constant_fraction"].get(c)
    o.append(f"\\texttt{{{esc(c)}}} & {audit['lit_coverage_per_feature'][c]:.3f} & {'' if cv is None else f'{cv:.3f}'}\\\\")
o.append(r"\bottomrule\end{tabular}\par\smallskip Coverage values below the median belong to sparsely reported process parameters; all other columns are non-missing exactly for the 14,896 records of publications with an extraction record.")
o.append(r"\section*{Extraction system prompt (verbatim from the recovered script)}\begin{verbatim}" + prompt + r"\end{verbatim}")
o.append(r"\section*{Model-development grid}CatBoost depth/L2 pairs: (6,3), (8,3), (10,3), (8,10), (10,10), (8,1); learning rate 0.05; up to 3,000 iterations; early stopping 150 rounds on the validation set; three seeds (42, 7, 123) for the selected configuration; XGBoost: depth 8, learning rate 0.05, 1,500 trees, subsample 0.8, colsample 0.7, $\lambda=3$, early stopping 100 rounds.")
o.append(r"\end{document}")
(ROOT / "psc_manuscript" / "si.tex").write_text("\n".join(o))
print("si.tex written")
