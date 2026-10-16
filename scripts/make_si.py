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
EARLY = "t2 + earlier lit.\\ block"   # tier label 't3' in the old run files
o = [r"""\documentclass[10pt]{article}
\usepackage[T1]{fontenc}\usepackage[margin=2cm]{geometry}\usepackage{booktabs,longtable,lmodern,url,xcolor,graphicx}
\usepackage{microtype}\usepackage{hyperref}
\title{Supplementary Information: Perovskite Solar Cell Performance Prediction with LLM-Derived Feature Augmentation under Leakage-Aware DOI Evaluation}
\date{}\begin{document}\maketitle\small
\noindent\textbf{Tier naming.} Throughout the main paper and this document, $t_1$ = PDP-derived device, processing and composition features; $t_2$ = $t_1$ + curated material-property descriptors; $t_3$ = $t_2$ + the 36 LLM-derived recipe flags. An earlier, superseded 44-predictor literature representation (the ``earlier literature block'') is \emph{not} $t_3$; where its results are shown they are labelled ``$t_2$ + earlier lit.\ block''. In the underlying run files and result tables of the repository this block carries the tier name \texttt{t3}; this document relabels it and never mixes its numbers with the recipe-flag results.\par
\section*{Table S1. Predictors per tier}
Column sets shown for a preprocessor fitted on the full filtered data (documentation); in the experiments the \texttt{\_na} indicators and the $>$80\% missing filter are re-derived from each training partition. Tiers are nested: $t_1\subset t_2\subset t_3$ (77, 179 and 215 predictor columns in the common-split models).\par"""]
prev = set()
for t in ("t1", "t2"):
    cols = fl[t]; new = [c for c in cols if c not in prev]
    o.append(f"\\subsection*{{{t}: {len(cols)} predictors ({len(new)} added relative to the previous tier)}}")
    for g in ("process", "device", "composition", "material", "literature"):
        gc = [c for c in new if cols[c] == g]
        if gc: o.append(f"\\textbf{{{g}}} ({len(gc)}): " + ", ".join(f"\\texttt{{{esc(c)}}}" for c in gc) + "\\par\\smallskip")
    prev = set(cols)
import sys; sys.path.insert(0, str(ROOT / "src"))
from psc import v4
T = v4.paper_table("rec"); prev_rc = T.mean()
o.append(f"\\subsection*{{t3: 215 predictors (36 recipe flags added to $t_2$)}}")
o.append("Flags are paper-level, binary, and missing (native NaN, no indicator) for publications without an extraction record. Share positive among the 2,835 extracted papers in parentheses: " +
         ", ".join(f"\\texttt{{{esc(c)}}} ({100*prev_rc[c]:.1f}\\%)" for c in T.columns) + ".\\par\\smallskip")
o.append(f"\\subsection*{{Earlier literature block (44 predictors; not part of $t_3$)}}")
lit = [c for c in fl["t3"] if fl["t3"][c] == "literature"]
o.append("\\textbf{literature} (" + str(len(lit)) + "): " + ", ".join(f"\\texttt{{{esc(c)}}}" for c in lit) + "\\par\\smallskip")
o.append(r"\section*{Table S2. Per-split results of the matched CatBoost five-holdout experiment (DOI-grouped; $t_3$ in the run files = earlier literature block)}\begin{longtable}{lllrrrr}\toprule Subset & Tier & Split & $n_\mathrm{test}$ & $R^2$ & MAE & RMSE\\\midrule\endhead")
x = A[(A.protocol == "doi") & (A.model == "catboost")].sort_values(["arch", "tier", "split"])
for _, r in x.iterrows():
    o.append(f"{r.arch} & {r.tier if r.tier != 't3' else EARLY} & {r.split} & {r.n_test} & {r.r2:.4f} & {r.mae:.3f} & {r.rmse:.3f}\\\\")
o.append(r"\bottomrule\end{longtable}")
o.append(r"\section*{Table S3. Split integrity (all 270 recorded fits)}")
d = A[A.protocol == "doi"]
o.append(f"DOI-grouped fits: {len(d)}; maximum DOI overlap between test and train: {d.doi_overlap_test_train.max()}, validation and train: {d.doi_overlap_val_train.max()}, test and validation: {d.doi_overlap_test_val.max()}.\\par")
o.append(f"Records without DOI: {audit['n_records_no_doi']} (one block). Publications (normalised DOIs): {audit['n_doi_groups_incl_no_doi']-1}.\\par")
o.append(f"Data SHA-256: \\texttt{{{meta['data_sha256']}}}; config SHA-256: \\texttt{{{meta['config_sha256'][:16]}}}; git HEAD at run time: \\texttt{{{meta['git_head'][:12]}}} (working tree dirty: {meta['git_dirty']}); CatBoost {meta['catboost']}, scikit-learn {meta['sklearn']}, Python {meta['python']}.\\par")
o.append(r"\section*{Table S4. Earlier literature block: coverage and constancy within publication}\begin{tabular}{lrr}\toprule Feature & coverage & constant within DOI\\\midrule")
for c in sorted(audit["lit_coverage_per_feature"]):
    cv = audit["lit_within_doi_constant_fraction"].get(c)
    o.append(f"\\texttt{{{esc(c)}}} & {audit['lit_coverage_per_feature'][c]:.3f} & {'' if cv is None else f'{cv:.3f}'}\\\\")
o.append(r"\bottomrule\end{tabular}\par\smallskip Coverage values below the median belong to sparsely reported process parameters; all other columns are non-missing exactly for the 14,896 records of publications with an extraction record.")

# ---- S5: earlier literature block (five-holdout ablation) ----
import numpy as np
from psc.results import load_ablation, read
AB = load_ablation(); D = AB[(AB.protocol == "doi") & (AB.model == "catboost")]
nm = {"t1": "$t_1$", "t2": "$t_2$", "t3": "$t_2$ + earlier lit.\\ block"}
o.append(r"\section*{Table S5. Earlier 44-predictor literature block: five-holdout ablation (matched CatBoost, DOI-grouped; mean $\pm$ s.d. over five splits)}")
o.append(r"This experiment evaluated the superseded literature block, not $t_3$. $\Delta R^2$ is the mean per-split difference to $t_2$.\par\smallskip\begin{tabular}{llcccc}\toprule Subset & Tier & $R^2$ & MAE & RMSE & $\Delta R^2$ vs $t_2$\\\midrule")
for a, an in (("overall", "Pooled"), ("pin", "p-i-n"), ("nip", "n-i-p")):
    g = D[D.arch == a]; r2 = {t: g[g.tier == t].sort_values("split").r2.values for t in ("t1", "t2", "t3")}
    for t in ("t1", "t2", "t3"):
        x = g[g.tier == t]; d = "" if t == "t2" else f"{(r2[t] - r2['t2']).mean():+.3f}"
        o.append(f"{an} & {nm[t]} & ${x.r2.mean():.3f}\\pm{x.r2.std():.3f}$ & ${x.mae.mean():.3f}\\pm{x.mae.std():.3f}$ & ${x.rmse.mean():.3f}\\pm{x.rmse.std():.3f}$ & {d}\\\\")
    o.append(r"\midrule")
o.append(r"\bottomrule\end{tabular}\par\medskip")
G = read("feature_groups_v1")
nmg = {"t1_lit": "$t_1$ + earlier lit.\\ block (no material properties)", "t3_nometa": "$t_2$ + earlier block minus 2 completeness flags", "t2_noise": "$t_2$ + 32 noise columns"}
o.append(r"\noindent Feature-group controls (same protocol):\par\smallskip\begin{tabular}{llcccc}\toprule Subset & Variant & Feat. & $R^2$ & MAE & RMSE\\\midrule")
for a, an in (("overall", "Pooled"), ("pin", "p-i-n"), ("nip", "n-i-p")):
    for t in ("t1_lit", "t3_nometa", "t2_noise"):
        x = G[(G.arch == a) & (G.tier == t)]
        o.append(f"{an} & {nmg[t]} & {x.n_features.mean():.0f} & ${x.r2.mean():.3f}\\pm{x.r2.std():.3f}$ & ${x.mae.mean():.3f}\\pm{x.mae.std():.3f}$ & ${x.rmse.mean():.3f}\\pm{x.rmse.std():.3f}$\\\\")
o.append(r"\bottomrule\end{tabular}\par\medskip")
# ---- S6: reference and model-development rows of the earlier block, and strict protocols ----
B = read("baselines_v1"); M = read("model_dev_v1"); S = read("strict_validation_v1")
bn = {"gpr_sub": "GPR (3,000-record subsample)", "mlp": "MLP", "rf": "Random forest", "xgb_default": "XGBoost (defaults)"}
o.append(r"\section*{Table S6. Earlier-block rows of the reference-model, model-development and stricter-protocol experiments}")
o.append(r"\noindent Reference models, pooled, five DOI-grouped splits (mean $\pm$ s.d.):\par\smallskip\begin{tabular}{llccc}\toprule Model & Tier & $R^2$ & MAE & RMSE\\\midrule")
for m in ("xgb_default", "rf", "mlp", "gpr_sub"):
    x = B[(B.model == m) & (B.tier == "t3")]
    o.append(f"{bn[m]} & {nm['t3']} & ${x.r2.mean():.3f}\\pm{x.r2.std():.3f}$ & ${x.mae.mean():.3f}\\pm{x.mae.std():.3f}$ & ${x.rmse.mean():.3f}\\pm{x.rmse.std():.3f}$\\\\")
o.append(r"\bottomrule\end{tabular}\par\medskip")
o.append(r"\noindent Model-development experiment ($R^2$, mean $\pm$ s.d. over five splits):\par\smallskip\begin{tabular}{llcccc}\toprule Subset & Tier & Tuned & Bag & XGB & Ens\\\midrule")
for a, an in (("overall", "Pooled"), ("pin", "p-i-n"), ("nip", "n-i-p")):
    x = M[(M.arch == a) & (M.tier == "t3")]
    o.append(f"{an} & {nm['t3']} & " + " & ".join(f"${x[c].mean():.3f}\\pm{x[c].std():.3f}$" for c in ("cb_tuned_r2", "cb_bag_r2", "xgb_r2", "ens_r2")) + "\\\\")
o.append(r"\bottomrule\end{tabular}\par\medskip")
o.append(r"\noindent Stricter protocols, pooled CatBoost $R^2$ (mean $\pm$ s.d.):\par\smallskip\begin{tabular}{llcc}\toprule Protocol & Tier & $R^2$ & MAE\\\midrule")
for pr, pn in (("author", "Lead-author-grouped"), ("temporal", "Forward in time")):
    x = S[(S.protocol == pr) & (S.arch == "overall") & (S.model == "catboost") & (S.tier == "t3")]
    o.append(f"{pn} & {nm['t3']} & ${x.r2.mean():.3f}\\pm{x.r2.std():.3f}$ & {x.mae.mean():.2f}\\\\")
o.append(r"\bottomrule\end{tabular}\par")
# ---- S7: SHAP (earlier block) ----
o.append(r"\section*{Figs.\ S1 and S2. SHAP analysis (models trained on $t_2$ + the earlier 44-predictor literature block; not $t_3$)}")
o.append(r"\begin{figure}[h]\centering\includegraphics[width=0.75\textwidth]{figures/fig_shap_groups.png}\caption*{Fig.\ S1. Share of total feature attribution by feature group for matched CatBoost models trained on $t_2$ plus the earlier 44-predictor literature block (first DOI-grouped split, 1,500 test records). Attribution is not predictive benefit, and no SHAP analysis was run for the recipe-flag model $t_3$.}\end{figure}")
o.append(r"\begin{figure}[h]\centering\includegraphics[width=\textwidth]{figures/shap.png}\caption*{Fig.\ S2. Twelve highest-ranked features by mean $|$SHAP$|$ (PCE percentage points) for the (a) pooled, (b) p-i-n and (c) n-i-p models of Fig.\ S1; \texttt{na} denotes a missingness indicator.}\end{figure}")
# ---- S8: common-split details ----
CMP = json.load(open(R / "v4" / "final" / "comparisons.json")); C3 = json.load(open(R / "v4" / "final_t1" / "comparisons_3tier.json"))
o.append(r"\section*{Table S7. Common-split comparison: per-seed paired $\Delta R^2$}")
o.append(r"Reserved publications, matched default CatBoost; each seed model scored separately (the main-text values use the mean of the five seed models).\par\smallskip\begin{tabular}{llccccc}\toprule Population & Comparison & seed 0 & seed 1 & seed 2 & seed 3 & seed 4\\\midrule")
for sc, sn in (("covered", "LLM-covered"), ("full", "All reserved")):
    for k, kn in (("t2_minus_t1", "$t_2-t_1$"), ("t3_minus_t2", "$t_3-t_2$")):
        o.append(f"{sn} & {kn} & " + " & ".join(f"{x:+.4f}" for x in C3[sc][k]["per_seed"]) + "\\\\")
o.append(r"\bottomrule\end{tabular}\par\medskip")
o.append(r"Negative controls (default settings; $\Delta R^2$ relative to $t_2$ under the same settings): ")
for sc, sn in (("covered", "LLM-covered"), ("full", "all reserved")):
    pv = [v["dR2"] for k, v in CMP[sc].items() if k.startswith("ctl_perm")]; cf = CMP[sc]["ctl_coverflag_default vs t2_default"]["dR2"]
    o.append(f"{sn}: five DOI-permuted flag blocks {min(pv):+.4f} to {max(pv):+.4f}; coverage indicator only {cf:+.4f}." + (" " if sc == "covered" else ""))
o.append(r"\par")
o.append(r"\section*{Extraction system prompt (verbatim from the recovered script)}\begin{verbatim}" + prompt + r"\end{verbatim}")
o.append(r"\section*{Model-development grid}CatBoost depth/L2 pairs: (6,3), (8,3), (10,3), (8,10), (10,10), (8,1); learning rate 0.05; up to 3,000 iterations; early stopping 150 rounds on the validation set; three seeds (42, 7, 123) for the selected configuration; XGBoost: depth 8, learning rate 0.05, 1,500 trees, subsample 0.8, colsample 0.7, $\lambda=3$, early stopping 100 rounds.")
o.append(r"\end{document}")
(ROOT / "psc_manuscript" / "si.tex").write_text("\n".join(l.rstrip() for l in "\n".join(o).split("\n")) + "\n")
print("si.tex written")
