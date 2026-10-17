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
\title{Supplementary Information: Publication-Level Generalization in Perovskite Solar Cell Prediction: DOI-Grouped Evaluation with Literature-Derived Recipe Features}
\date{}\begin{document}\maketitle\small
\noindent\textbf{Tier naming.} Throughout the main paper and this document, $\mathcal{M}_1$ = PDP-derived device, processing and composition features; $\mathcal{M}_2$ = $\mathcal{M}_1$ + curated material-property descriptors; $\mathcal{M}_3$ = $\mathcal{M}_2$ + the 36 LLM-derived recipe flags. An earlier, superseded 44-predictor literature representation (the ``earlier literature block'') is \emph{not} $\mathcal{M}_3$; where its results are shown they are labelled ``$\mathcal{M}_2$ + earlier lit.\ block''. In the underlying run files and result tables of the repository this block carries the tier name \texttt{t3}; this document relabels it and never mixes its numbers with the recipe-flag results.\par
\section*{Table S1. Predictors per tier}
Column sets shown for a preprocessor fitted on the full filtered data (documentation); in the experiments the \texttt{\_na} indicators and the $>$80\% missing filter are re-derived from each training partition. Tiers are nested: $\mathcal{M}_1\subset \mathcal{M}_2\subset \mathcal{M}_3$ (77, 179 and 215 predictor columns in the common-split models).\par"""]
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
o.append(f"\\subsection*{{t3: 215 predictors (36 recipe flags added to $\mathcal{M}_2$)}}")
o.append("Flags are paper-level, binary, and missing (native NaN, no indicator) for publications without an extraction record. Share positive among the 2,835 extracted papers in parentheses: " +
         ", ".join(f"\\texttt{{{esc(c)}}} ({100*prev_rc[c]:.1f}\\%)" for c in T.columns) + ".\\par\\smallskip")
o.append(f"\\subsection*{{Earlier literature block (44 predictors; not part of $\mathcal{M}_3$)}}")
lit = [c for c in fl["t3"] if fl["t3"][c] == "literature"]
o.append("\\textbf{literature} (" + str(len(lit)) + "): " + ", ".join(f"\\texttt{{{esc(c)}}}" for c in lit) + "\\par\\smallskip")
o.append(r"\section*{Table S2. Per-split results of the matched CatBoost five-holdout experiment (DOI-grouped; $\mathcal{M}_3$ in the run files = earlier literature block)}\begin{longtable}{lllrrrr}\toprule Subset & Tier & Split & $n_\mathrm{test}$ & $R^2$ & MAE & RMSE\\\midrule\endhead")
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
nm = {"t1": "$\mathcal{M}_1$", "t2": "$\mathcal{M}_2$", "t3": "$\mathcal{M}_2$ + earlier lit.\\ block"}
o.append(r"\section*{Table S5. Earlier 44-predictor literature block: five-holdout ablation (matched CatBoost, DOI-grouped; mean $\pm$ s.d. over five splits)}")
o.append(r"This experiment evaluated the superseded literature block, not $\mathcal{M}_3$. $\Delta R^2$ is the mean per-split difference to $\mathcal{M}_2$.\par\smallskip\begin{tabular}{llcccc}\toprule Subset & Tier & $R^2$ & MAE & RMSE & $\Delta R^2$ vs $\mathcal{M}_2$\\\midrule")
for a, an in (("overall", "Pooled"), ("pin", "p-i-n"), ("nip", "n-i-p")):
    g = D[D.arch == a]; r2 = {t: g[g.tier == t].sort_values("split").r2.values for t in ("t1", "t2", "t3")}
    for t in ("t1", "t2", "t3"):
        x = g[g.tier == t]; d = "" if t == "t2" else f"{(r2[t] - r2['t2']).mean():+.3f}"
        o.append(f"{an} & {nm[t]} & ${x.r2.mean():.3f}\\pm{x.r2.std():.3f}$ & ${x.mae.mean():.3f}\\pm{x.mae.std():.3f}$ & ${x.rmse.mean():.3f}\\pm{x.rmse.std():.3f}$ & {d}\\\\")
    o.append(r"\midrule")
o.append(r"\bottomrule\end{tabular}\par\medskip")
G = read("feature_groups_v1")
nmg = {"t1_lit": "$\mathcal{M}_1$ + earlier lit.\\ block (no material properties)", "t3_nometa": "$\mathcal{M}_2$ + earlier block minus 2 completeness flags", "t2_noise": "$\mathcal{M}_2$ + 32 noise columns"}
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
o.append(r"\section*{Figs.\ S1 and S2. SHAP analysis (models trained on $\mathcal{M}_2$ + the earlier 44-predictor literature block; not $\mathcal{M}_3$)}")
o.append(r"\begin{figure}[h]\centering\includegraphics[width=0.75\textwidth]{figures/fig_shap_groups.png}\caption*{Fig.\ S1. Share of total feature attribution by feature group for matched CatBoost models trained on $\mathcal{M}_2$ plus the earlier 44-predictor literature block (first DOI-grouped split, 1,500 test records). Attribution is not predictive benefit, and no SHAP analysis was run for the recipe-flag model $\mathcal{M}_3$.}\end{figure}")
o.append(r"\begin{figure}[h]\centering\includegraphics[width=\textwidth]{figures/shap.png}\caption*{Fig.\ S2. Twelve highest-ranked features by mean $|$SHAP$|$ (PCE percentage points) for the (a) pooled, (b) p-i-n and (c) n-i-p models of Fig.\ S1; \texttt{na} denotes a missingness indicator.}\end{figure}")
# ---- S8: common-split details ----
CMP = json.load(open(R / "v4" / "final" / "comparisons.json")); C3 = json.load(open(R / "v4" / "final_t1" / "comparisons_3tier.json"))
o.append(r"\section*{Table S7. Common-split comparison: per-seed paired $\Delta R^2$}")
o.append(r"Reserved publications, matched default CatBoost; each seed model scored separately (the main-text values use the mean of the five seed models).\par\smallskip\begin{tabular}{llccccc}\toprule Population & Comparison & seed 0 & seed 1 & seed 2 & seed 3 & seed 4\\\midrule")
for sc, sn in (("covered", "LLM-covered"), ("full", "All reserved")):
    for k, kn in (("t2_minus_t1", "$\mathcal{M}_2-\mathcal{M}_1$"), ("t3_minus_t2", "$\mathcal{M}_3-\mathcal{M}_2$")):
        o.append(f"{sn} & {kn} & " + " & ".join(f"{x:+.4f}" for x in C3[sc][k]["per_seed"]) + "\\\\")
o.append(r"\bottomrule\end{tabular}\par\medskip")
o.append(r"Negative controls (default settings; $\Delta R^2$ relative to $\mathcal{M}_2$ under the same settings): ")
for sc, sn in (("covered", "LLM-covered"), ("full", "all reserved")):
    pv = [v["dR2"] for k, v in CMP[sc].items() if k.startswith("ctl_perm")]; cf = CMP[sc]["ctl_coverflag_default vs t2_default"]["dR2"]
    o.append(f"{sn}: five DOI-permuted flag blocks {min(pv):+.4f} to {max(pv):+.4f}; coverage indicator only {cf:+.4f}." + (" " if sc == "covered" else ""))
o.append(r"\par")

# ---- S8-S14: revision analyses (results/revision/*.json, produced by scripts/revision_analyses.py and v4_penalty0_reserved.py) ----
def tt(x):
    m = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "_": r"\_", "#": r"\#", "%": r"\%", "&": r"\&", "$": r"\$", "^": r"\^{}", "~": r"\~{}"}
    return "".join(m.get(ch, ch) for ch in x).replace("|", r"\textbar{}\allowbreak ")
RV = R / "revision"
OUTC = ", ".join("\\texttt{"+esc(c)+"}" for c in ("Stability_measured", "EQE_measured", "JV_hysteresis_index", "Cell_area_measured_cm2", "Stabilised_performance_measured"))
AU = json.load(open(RV / "leakage_audit.json")); IL = json.load(open(RV / "illumination_sensitivity.json"))
PW = json.load(open(RV / "publication_weighted.json")); CB = json.load(open(RV / "coverage_bias.json")); P0 = json.load(open(RV / "penalty0" / "penalty0_comparison.json"))
import collections
o.append(r"\section*{Table S8. Predictor and target-leakage audit}")
o.append(f"Target: PDP field \\texttt{{JV\\_default\\_PCE}} (renamed \\texttt{{final\\_pce}}), documented in the PDP column metadata as the selected power conversion efficiency of the J--V section; identical to the PDP field in all {AU['target']['records_filtered']:,} filtered rows (and in all 43,398 exported rows). The full per-predictor audit (predictor, tier, source field, derivation, handling, leakage risk) is in \\texttt{{results/revision/leakage\\_audit.json}}.\\par")
cnt = collections.Counter((a["tier"], a["leakage_risk"]) for a in AU["predictors"])
o.append(r"\begin{tabular}{lrrr}\toprule Tier (added predictors) & none & low & moderate\\\midrule")
for t in ("t1", "t2", "t3"):
    o.append(f"{t} & {cnt[(t,'none')]} & {cnt[(t,'low')]} & {cnt[(t,'moderate')]}\\\\")
o.append(r"\bottomrule\end{tabular}\par\smallskip")
low = sorted({a["predictor"] for a in AU["predictors"] if a["leakage_risk"] == "low" and a["tier"] == "t1" and not a["predictor"].endswith("_na")})
o.append("Predictors of $\mathcal{M}_1$ rated low risk and retained: " + ", ".join(f"\\texttt{{{esc(c)}}}" for c in low) + ". These are device-context or film-characterisation fields (the absorber band gap, the photoluminescence maximum, encapsulation, an additional front layer, back-contact thickness) and not J--V, EQE, stability or stabilised-performance results. Material-property predictors ($\mathcal{M}_2$) are catalogue reference values and the recipe flags ($\mathcal{M}_3$) are paper-level identity flags; $\mathcal{M}_3$ flags are rated low because they are constant within a publication and could encode publication identity.\\par")
o.append(f"Checks: no predictor of the three tiers matches the outcome pattern \\texttt{{{tt(AU['banned_pattern'])}}} (hits: {len(AU['predictors_matching_banned_outcome_pattern'])}); the processed table contains five outcome-adjacent columns ({OUTC}), all excluded in code; {len(AU['pdp_outcome_or_stability_fields_not_in_enriched_table'])} fields of the full PDP export whose names match this pattern (J--V, EQE, stability and stabilised-performance fields among them) were never carried into the processed table. The largest absolute Spearman correlation of any numeric $\mathcal{M}_2$ predictor with the target is {abs(AU['max_abs_spearman_with_target'][0]['rho']):.2f} (\\texttt{{{esc(AU['max_abs_spearman_with_target'][0]['predictor'])}}}), so no predictor is a near-copy of the target.\\par")
o.append(r"\section*{Table S9. Illumination sensitivity on the reserved publications (test-set exclusion, no retraining)}")
o.append(f"The full PDP export records the light spectrum and intensity. All {IL['records_pce_gt_27']} filtered records with PCE $>$ 27\\% (six publications) have intensity $\\leq$ 0.31 (standard illumination is 100 on this scale); {IL['pce_gt_27_light_spectra'].get('Indoor light',0)} are labelled indoor light and {IL['pce_gt_27_light_spectra'].get('missing',0)} have no spectrum label. In the filtered data {IL['filtered_records_non_AM15_spectrum']} records have a spectrum other than AM1.5 and {IL['filtered_records_intensity_outside_90_110']} an intensity outside 90--110. Matched default CatBoost, mean of five seed models.\\par\\smallskip")
o.append(r"\begin{tabular}{llrcccc}\toprule Population & Test records & $n$ & $R^2$ $\mathcal{M}_1$ & $R^2$ $\mathcal{M}_2$ & $R^2$ $\mathcal{M}_3$ & MAE $\mathcal{M}_3$\\\midrule")
lab = {"all": "all", "exclude_pce_gt_27": "PCE $\\leq$ 27\\%", "exclude_nonstandard_illumination": "AM1.5, 90--110 only"}
for sc, sn in (("covered", "LLM-covered"), ("full", "All reserved")):
    for k, kn in lab.items():
        v = IL["reserved"][sc][k]
        o.append(f"{sn} & {kn} & {v['t3']['n']:,} & {v['t1']['r2']:.4f} & {v['t2']['r2']:.4f} & {v['t3']['r2']:.4f} & {v['t3']['mae']:.3f}\\\\")
o.append(r"\bottomrule\end{tabular}\par")
o.append(r"\section*{Table S10. Publication-weighted sensitivity metrics (reserved publications)}")
o.append(r"Each reserved record receives weight $1/n_p$, where $n_p$ is the number of reserved records of its publication within the population; weighted $R^2=1-\sum w_ie_i^2/\sum w_i(y_i-\bar y_w)^2$ with the weighted mean $\bar y_w$; weighted MAE and RMSE are weighted means. This is equivalent to averaging the error within each publication and then across publications. Paired intervals: publication-cluster bootstrap (2000 resamples, rng seed 0).\par\smallskip")
o.append(r"\begin{tabular}{llcccc}\toprule Population & Tier & $R^2$ record-wtd & $R^2$ publication-wtd & MAE pub.-wtd & RMSE pub.-wtd\\\midrule")
for sc, sn in (("covered", "LLM-covered"), ("full", "All reserved")):
    for t in ("t1", "t2", "t3"):
        a, b = PW[sc]["record_weighted"][t], PW[sc]["publication_weighted"][t]
        o.append(f"{sn} & ${t[0]}_{t[1]}$ & {a['r2']:.3f} & {b['r2']:.3f} & {b['mae']:.3f} & {b['rmse']:.3f}\\\\")
o.append(r"\midrule")
o.append(r"\end{tabular}\par\smallskip")
o.append(r"\begin{tabular}{llc}\toprule Population & Publication-weighted paired $\Delta R^2$ & value [95\% interval]\\\midrule")
for sc, sn in (("covered", "LLM-covered"), ("full", "All reserved")):
    for k, kn in (("t2_minus_t1", "$\mathcal{M}_2-\mathcal{M}_1$"), ("t3_minus_t2", "$\mathcal{M}_3-\mathcal{M}_2$")):
        v = PW[sc]["paired_publication_weighted"][k]
        o.append(f"{sn} & {kn} & ${v['dR2']:+.4f}$ [${v['lo']:+.4f}$, ${v['hi']:+.4f}$]\\\\")
o.append(r"\bottomrule\end{tabular}\par")
o.append(r"\section*{Table S11. Extracted versus non-extracted publications (all 41,107 filtered records)}")
o.append("`Covered' means that the publication has one of the 2,835 extraction records. Year is the PDP \\texttt{Ref\\_publication\\_date} field. Descriptive only.\\par\\smallskip")
o.append(r"\begin{tabular}{lrrrrrrrr}\toprule Group & records & publ. & PCE mean & median & p10 & p90 & share $\geq$20\% & n-i-p share\\\midrule")
for r_ in CB["summary"]:
    o.append(f"{r_['group']} & {r_['n_records']:,} & {r_['n_publications']:,} & {r_['pce_mean']:.2f} & {r_['pce_median']:.2f} & {r_['pce_p10']:.2f} & {r_['pce_p90']:.2f} & {100*r_['frac_pce_ge_20']:.1f}\\% & {100*r_['frac_nip']:.1f}\\%\\\\")
o.append(r"\bottomrule\end{tabular}\par\smallskip")
o.append(r"\begin{tabular}{lrr}\toprule Year & records & covered share\\\midrule")
for r_ in CB["coverage_by_year"]:
    if r_["n_records"] >= 100: o.append(f"{int(r_['year'])} & {r_['n_records']:,} & {100*r_['covered_fraction']:.1f}\\%\\\\")
o.append(r"\bottomrule\end{tabular}\hspace{1.5cm}\begin{tabular}{lrr}\toprule PCE bin (\%) & records & covered share\\\midrule")
for r_ in CB["coverage_by_pce_bin"]:
    o.append(f"{esc(r_['final_pce'])} & {r_['n_records']:,} & {100*r_['covered_fraction']:.1f}\\%\\\\")
o.append(r"\bottomrule\end{tabular}\par\smallskip")
o.append(f"Years with fewer than 100 records are omitted from the table. A two-sample Kolmogorov--Smirnov test of the PCE distributions of covered and uncovered records gives $D = {CB['ks_pce_covered_vs_not']['statistic']:.3f}$ ($p = {CB['ks_pce_covered_vs_not']['p']:.1e}$), a small difference that is detectable only because of the sample size. Reserved partition: mean PCE {CB['reserved_split']['covered_mean_pce']:.2f} (covered) versus {CB['reserved_split']['all_mean_pce']:.2f} (all).\\par")
o.append(r"\section*{Table S12. Recipe flags without the first-feature-use penalty (reserved publications, matched default CatBoost)}")
o.append(r"$\mathcal{M}_3$ refitted once with the penalty on the flag columns set to 0 instead of 0.5 (same development records, seeds, validation splits and configuration); scored on the same reserved records. The reserved publications had been examined before this run, so it is a robustness check and not independent confirmation. On development folds the penalties 0, 0.5 and 2 gave mean $\Delta R^2$ of $+0.0052$, $+0.0070$ and $+0.0038$ (12, 12 and 11 of 15 folds positive).\par\smallskip")
o.append(r"\begin{tabular}{llcccc}\toprule Population & Variant & $R^2$ & MAE & RMSE & $\Delta R^2$ vs $\mathcal{M}_2$ [95\% interval]\\\midrule")
for sc, sn in (("covered", "LLM-covered"), ("full", "All reserved")):
    o.append(f"{sn} & $\mathcal{M}_2$ & {P0[sc]['t2']['r2']:.4f} & {P0[sc]['t2']['mae']:.3f} & {P0[sc]['t2']['rmse']:.3f} & \\\\")
    for k, kn in (("t3_pen0.5", "$\mathcal{M}_3$, penalty 0.5 (main)"), ("t3_pen0", "$\mathcal{M}_3$, no penalty")):
        v = P0[sc][k]; d = P0[sc][k + "_minus_t2"]
        o.append(f"{sn} & {kn} & {v['r2']:.4f} & {v['mae']:.3f} & {v['rmse']:.3f} & ${d['dR2']:+.4f}$ [${d['lo']:+.4f}$, ${d['hi']:+.4f}$]\\\\")
o.append(r"\bottomrule\end{tabular}\par")
o.append(r"\section*{Table S13. Definitions of the 36 recipe flags}")
from psc.litfeat4 import RECIPE
o.append(r"Each flag is 1 if the (case-insensitive) pattern occurs in the non-outcome string content of the publication's extracted material records (HTL and ETL flags: only records of that role, additive or other), and 0 otherwise. Patterns are broad lexical matches: for example \texttt{rc\_pea\_family} also matches ``ammonium'' and hence methylammonium compounds (only 104 of its 1,723 positive publications contain a spacer-specific term). Source: \texttt{src/psc/litfeat4.py}.\par\smallskip")
o.append(r"{\footnotesize\begin{longtable}{p{3cm}p{2.6cm}p{10.5cm}}\toprule Flag & Group & Pattern\\\midrule\endhead")
GN = {"P": "additive/precursor", "H": "HTL dopant", "E": "ETL treatment", "R": "process route", "C": "device context", "U": "undoped/interface"}
for g, cs in v4.REC_GROUPS.items():
    for c in cs:
        o.append(f"\\texttt{{rc\\_{esc(c)}}} & {GN[g]} & \\texttt{{{tt(RECIPE[c])}}}\\\\")
o.append(r"\bottomrule\end{longtable}}")
o.append(r"\section*{Material-descriptor implementation notes}")
o.append(r"\noindent\textbf{Perovskite geometry descriptors.} Fixed effective radii (pm): A-site FA 253, MA 217, Cs 188, Rb 172; B-site Pb 119, Sn 110, Bi 103; X-site I 220, Br 196, Cl 181, F 133. Site radius = fraction-weighted mean over the listed species of that site, divided by the sum of their fractions; $\mu=r_B/r_X$, $t=(r_A+r_X)/[\sqrt2(r_B+r_X)]$, $r_A/r_X$, $\tau=r_X/r_B-(1-(r_A/r_B)\ln(r_A/r_B))$ (script \texttt{opus/evaluate\_tolerance\_factor.py}; the stored octahedral and tolerance factors were reproduced to floating-point precision from the ion-fraction columns). \textbf{Layer averages.} \texttt{*\_avg\_radius}, \texttt{*\_avg\_atomic\_mass} and \texttt{*\_en\_avg} are atom-count-weighted means of covalent radii (pm), atomic masses and Pauling electronegativities over the elements of a hand-entered formula (script \texttt{opus/fill\_atomic\_data.py}); \texttt{min}/\texttt{max} are the extremes over the elements. For MAPbI$_3$ (C$_1$H$_6$N$_1$Pb$_1$I$_3$) the average radius is $(77+6\cdot31+71+146+3\cdot139)/12=74.75$ pm, matching the stored value.\par")
o.append(r"\section*{Extraction system prompt (verbatim from the recovered script)}\begin{verbatim}" + prompt + r"\end{verbatim}")
o.append(r"\section*{Model-development grid}CatBoost depth/L2 pairs: (6,3), (8,3), (10,3), (8,10), (10,10), (8,1); learning rate 0.05; up to 3,000 iterations; early stopping 150 rounds on the validation set; three seeds (42, 7, 123) for the selected configuration; XGBoost: depth 8, learning rate 0.05, 1,500 trees, subsample 0.8, colsample 0.7, $\lambda=3$, early stopping 100 rounds.")
o.append(r"\end{document}")
(ROOT / "psc_manuscript" / "si.tex").write_text("\n".join(l.rstrip() for l in "\n".join(o).split("\n")) + "\n")
print("si.tex written")
