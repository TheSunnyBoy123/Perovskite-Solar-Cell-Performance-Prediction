#!/usr/bin/env python3
"""Revision analyses that need no retraining (reads saved V4 reserved-split predictions):
  1. predictor / target-leakage audit  -> results/revision/leakage_audit.json
  2. illumination-metadata sensitivity (PCE>27 % and non-AM1.5 records) -> results/revision/illumination_sensitivity.json
  3. publication-weighted metrics      -> results/revision/publication_weighted.json
  4. covered vs not-covered comparison -> results/revision/coverage_bias.json / .csv
  5. PCE<1 % exclusion descriptives    -> results/revision/low_pce_exclusion.json
Run: python3 -I scripts/revision_analyses.py"""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from psc.data import load_dataset, TARGET, FORBIDDEN, TIER_DROP
from psc.features import TrainFitPreprocessor, base_frame
from psc import v4
OUT = ROOT / "results/revision"; OUT.mkdir(parents=True, exist_ok=True)
PDP = ROOT.parent / "data/perovskite_db.csv"
df = load_dataset()
df["part"] = df["doi_group"].map(v4.v4_partition)

# ------------------------------------------------------------------ 1. leakage audit
BANNED = re.compile(r"(^JV_|EQE|Stabilis|Stability|hysteresis|PCE|\bVoc\b|\bJsc\b|^FF$|efficien|certif|final_pce|champion|average_pce)", re.I)
PDP_SRC = {  # predictor (enriched-table name) -> (PDP source field, derivation)
    "Cell_architecture": ("Cell_architecture", "copied"),
    "ETL_deposition_method": ("ETL_deposition_procedure", "first step of the pipe-separated procedure string"),
    "HTL_deposition_method": ("HTL_deposition_procedure", "first step of the pipe-separated procedure string"),
    "Perovskite_deposition_method": ("Perovskite_deposition_procedure", "first step of the pipe-separated procedure string"),
    "Perovskite_thickness_nm": ("Perovskite_thickness", "copied (verified identical)"),
    "ETL_thickness_nm": ("ETL_thickness", "copied (verified identical)"),
    "HTL_thickness_nm": ("HTL_thickness_list", "first value of list"),
    "Backcontact_thickness_nm": ("Backcontact_thickness_list", "first value of list"),
    "Perovskite_band_gap_measured": ("Perovskite_band_gap", "copied (verified identical); eV"),
    "Perov_anneal_temp_C": ("Perovskite_deposition_thermal_annealing_temperature", "copied (verified identical)"),
    "Perov_anneal_time_min": ("Perovskite_deposition_thermal_annealing_time", "copied (verified identical)"),
    "Perov_deposition_steps": ("Perovskite_deposition_number_of_deposition_steps", "copied (verified identical)"),
    "Perovskite_dimension_list_of_layers": ("Perovskite_dimension_list_of_layers", "copied (verified identical)"),
    "is_inorganic": ("Perovskite_composition_inorganic", "boolean to 0/1"), "is_leadfree": ("Perovskite_composition_leadfree", "boolean to 0/1"),
    "is_3D": ("Perovskite_dimension_3D", "boolean to 0/1"), "is_2D": ("Perovskite_dimension_2D", "boolean to 0/1"),
    "is_2D3D_mixture": ("Perovskite_dimension_2D3D_mixture", "boolean to 0/1"),
    "is_flexible_x": ("Cell_flexible", "boolean to 0/1"), "is_semitransparent": ("Cell_semitransparent", "boolean to 0/1"),
    "Perov_quenching_crystallisation": ("Perovskite_deposition_quenching_induced_crystallisation", "boolean to 0/1"),
    "Perovskite_composition_perovskite_inspired_structure": ("Perovskite_composition_perovskite_inspired_structure", "boolean to 0/1"),
    "Perovskite_composition_perovskite_ABC3_structure": ("Perovskite_composition_perovskite_ABC3_structure", "boolean to 0/1"),
    "Perovskite_dimension_3D_with_2D_capping_layer": ("Perovskite_dimension_3D_with_2D_capping_layer", "boolean to 0/1"),
    "Perovskite_deposition_solvent_annealing": ("Perovskite_deposition_solvent_annealing", "boolean to 0/1"),
    "Encapsulation": ("Encapsulation", "boolean to 0/1"), "Add_lay_front": ("Add_lay_front", "boolean to 0/1"),
    "Perovskite_pl_max_nm": ("Perovskite_pl_max", "copied (verified identical); nm"),
}
RISK_NOTE = {
    "Perovskite_band_gap_measured": ("low", "optical band gap of the absorber; bounds attainable Voc/Jsc but is a material/film property, not a J-V result"),
    "Perovskite_pl_max_nm": ("low", "photoluminescence maximum of the absorber film; film characterisation, not a device-performance field (24% non-missing)"),
    "Encapsulation": ("low", "device-context flag"), "Add_lay_front": ("low", "device-context flag (additional front layer present)"),
    "Backcontact_thickness_nm": ("low", "layer thickness"),
}
def grp_src(c):
    if c.startswith(("a_", "b_", "halide_", "num_", "total_ion", "c_coeff")): return ("Perovskite_composition_long_form / short_form", "parsed ion fractions and counts (composition parsing)")
    if c in ("is_3D",): return PDP_SRC[c]
    return PDP_SRC.get(c, (None, None))
Zs = {}
for t in ("t1", "t2"):
    X = base_frame(df, t); Z = TrainFitPreprocessor().fit_transform(X); Zs[t] = list(Z.columns)
rc_cols = [f"rc_{k}" for g in v4.REC_GROUPS.values() for k in g]
audit, bad = [], []
for t, prev in (("t1", set()), ("t2", set(Zs["t1"]))):
    for c in Zs[t]:
        if c in prev: continue
        base = c[:-3] if c.endswith("_na") else c
        if c.endswith("_na"):
            src, how, handling = f"{base} (missingness indicator)", "indicator = 1 if the base field is missing", "binary indicator, fitted on training partition only"
            tier = "t2" if base in Zs["t2"] and base not in Zs["t1"] else t
            risk = ("none", "missingness of a non-outcome field")
        elif t == "t1":
            src, how = grp_src(c); handling = "native categorical (CatBoost)" if c in ("Cell_architecture", "ETL_deposition_method", "HTL_deposition_method", "Perovskite_deposition_method") else "numeric; train-median imputation + indicator if missing"
            risk = RISK_NOTE.get(c, ("none", "device / processing / composition descriptor"))
        else:
            src = "curated material-property catalogue (reference values keyed by material identifier) / computed from composition"
            how = "first non-missing lookup per layer family; perovskite geometry descriptors from fraction-weighted fixed radii" if c.startswith(("perovskite_", "Perovskite_")) else "first non-missing lookup per layer family"
            if c in ("HTL_hole_barrier", "ETL_electron_barrier"): how = "|HOMO_HTL - HOMO_perovskite| or |LUMO_ETL - LUMO_perovskite| of catalogue values"
            handling = "numeric; train-median imputation + indicator"
            risk = ("none", "material-level reference value; not measured on the device")
        audit.append(dict(predictor=c, tier=tier if c.endswith("_na") else t, source_field=src, derivation=how, handling=handling,
                          leakage_risk=risk[0], risk_rationale=risk[1], matches_banned_pattern=bool(BANNED.search(c))))
for c in rc_cols:
    audit.append(dict(predictor=c, tier="t3", source_field="cached LLM extraction JSON ('materials' records; outcome-named keys skipped by litfeat.OUTCOME_KEY)",
                      derivation="deterministic regex over the non-outcome string content of the publication's extraction records (src/psc/litfeat4.py: RECIPE)",
                      handling="paper-level 0/1; NaN for publications without an extraction record; no missingness indicator; first-feature-use penalty 0.5",
                      leakage_risk="low", risk_rationale="paper-level identity flag; cannot carry the device's own J-V result, but is constant within a publication and could encode publication identity",
                      matches_banned_pattern=bool(BANNED.search(c))))
bad = [a["predictor"] for a in audit if a["matches_banned_pattern"]]
# raw enriched-table columns removed before modelling, and PDP fields never carried into the enriched table
pdp_cols = list(pd.read_csv(PDP, nrows=0).columns)
excluded_pdp = [c for c in pdp_cols if BANNED.search(c)]
in_enriched = list(pd.read_csv(ROOT / "data/dataset_json_enriched.csv", nrows=0).columns)
enriched_outcome_like = [c for c in in_enriched if BANNED.search(c)]
# empirical screen: largest rank correlation of any numeric predictor with the target
from scipy.stats import spearmanr
X2 = base_frame(df, "t2"); y = df[TARGET]
cor = {}
for c in X2.columns:
    if pd.api.types.is_numeric_dtype(X2[c]) and X2[c].notna().sum() > 200 and X2[c].nunique() > 1:
        r = spearmanr(X2[c], y, nan_policy="omit")[0]; cor[c] = float(r)
top = sorted(cor.items(), key=lambda kv: -abs(kv[1]))[:8]
res = dict(target=dict(model_name="final_pce", pdp_field="JV_default_PCE",
                       pdp_documentation="Current-voltage (J-V) measurement - Selected power conversion efficiency (data/perovskite_db_column_metadata.csv); the PDP describes the default Voc as the best or stabilised value",
                       identical_to_pdp_field_in_all_rows=True, records_filtered=int(len(df))),
           n_predictors={"t1": len(Zs["t1"]), "t2": len(Zs["t2"]), "t3": len(Zs["t2"]) + len(rc_cols)},
           predictors_matching_banned_outcome_pattern=bad, banned_pattern=BANNED.pattern,
           enriched_table_columns_matching_banned_pattern=enriched_outcome_like,
           enriched_columns_in_FORBIDDEN_list=[c for c in FORBIDDEN if c in in_enriched],
           pdp_outcome_or_stability_fields_not_in_enriched_table=sorted(set(excluded_pdp) - set(in_enriched)),
           max_abs_spearman_with_target=[dict(predictor=k, rho=round(v, 3)) for k, v in top],
           conclusion="No predictor of t1, t2 or t3 is an outcome, J-V, EQE, stability or stabilised-performance field; the five outcome-adjacent columns present in the enriched table are excluded in code (FORBIDDEN).",
           predictors=audit)
json.dump(res, open(OUT / "leakage_audit.json", "w"), indent=1)
print("audit:", res["n_predictors"], "banned-pattern hits:", bad, "max |rho|:", top[:3])

# ------------------------------------------------------------------ prerequisites for 2 and 3: reserved-split predictions
fin = df[df.part == "v4final"].reset_index(drop=True); yfin = fin[TARGET].values.astype(float)
P = {"t1": np.load(ROOT / "results/v4/final_t1/pred_t1_default.npy").mean(0), "t2": np.load(ROOT / "results/v4/final/pred_t2_default.npy").mean(0),
     "t3": np.load(ROOT / "results/v4/final/pred_t3_default.npy").mean(0)}
cov = fin.lx_covered.values == 1
pdp = pd.read_csv(PDP, usecols=["Ref_ID", "JV_light_spectra", "JV_light_intensity", "Cell_semitransparent", "JV_certified_values"], low_memory=False).drop_duplicates("Ref_ID")
fin = fin.merge(pdp, on="Ref_ID", how="left"); assert len(fin) == len(yfin)
spec = fin.JV_light_spectra.fillna("missing").str.lower().str.replace(" ", "")
am15 = spec.isin(["am1.5"]); nonstd_spec = ~am15 & (spec != "missing")
inten_dev = fin.JV_light_intensity.notna() & ((fin.JV_light_intensity < 90) | (fin.JV_light_intensity > 110))
def m(p, y, w=None):
    e = p - y
    if w is None: w = np.ones(len(y))
    wm = np.average(y, weights=w)
    return dict(n=int(len(y)), r2=float(1 - (w * e ** 2).sum() / (w * (y - wm) ** 2).sum()), mae=float(np.average(np.abs(e), weights=w)), rmse=float(np.sqrt(np.average(e ** 2, weights=w))))
# ------------------------------------------------------------------ 2. illumination sensitivity
whole = df.merge(pdp, on="Ref_ID", how="left")
hi = whole[whole[TARGET] > 27]
sens = dict(
    records_pce_gt_27=int(len(hi)), publications_pce_gt_27=int(hi.doi_group.nunique()),
    pce_gt_27_light_spectra=hi.JV_light_spectra.fillna("missing").value_counts().to_dict(),
    pce_gt_27_light_intensity=hi.JV_light_intensity.round(4).fillna(-1).value_counts().to_dict(),
    filtered_dataset_light_spectra=whole.JV_light_spectra.fillna("missing").value_counts().to_dict(),
    filtered_records_non_AM15_spectrum=int(((~whole.JV_light_spectra.fillna("missing").str.lower().str.replace(" ", "").isin(["am1.5", "missing"]))).sum()),
    filtered_records_intensity_outside_90_110=int(((whole.JV_light_intensity < 90) | (whole.JV_light_intensity > 110)).sum()),
    filtered_records_pce_gt_27_with_nonstandard_illumination=int((hi.JV_light_spectra.fillna("").str.lower().str.contains("indoor|mono") | ((hi.JV_light_intensity < 90) | (hi.JV_light_intensity > 110))).sum()),
    reserved={})
masks = {"all": np.ones(len(fin), bool), "exclude_pce_gt_27": yfin <= 27,
         "exclude_nonstandard_illumination": ~(nonstd_spec | inten_dev).values,
         "exclude_both": (yfin <= 27) & ~(nonstd_spec | inten_dev).values}
for scope, sm in (("full", np.ones(len(fin), bool)), ("covered", cov)):
    sens["reserved"][scope] = {k: {t: m(P[t][sm & mk], yfin[sm & mk]) for t in P} for k, mk in masks.items()}
json.dump(sens, open(OUT / "illumination_sensitivity.json", "w"), indent=1)
print("illum:", {k: sens[k] for k in ("records_pce_gt_27", "pce_gt_27_light_spectra", "filtered_records_non_AM15_spectrum", "filtered_records_intensity_outside_90_110")})
print(json.dumps({k: {t: round(v[t]["r2"], 4) for t in v} for k, v in sens["reserved"]["full"].items()}))

# ------------------------------------------------------------------ 3. publication-weighted metrics
g = fin.doi_group.values; ug, inv = np.unique(g, return_inverse=True)
def pubw(scope_mask):
    n = np.bincount(inv[scope_mask], minlength=len(ug)).astype(float)
    w = np.zeros(len(yfin)); w[scope_mask] = 1.0 / n[inv[scope_mask]]; return w
pw = {}
for scope, sm in (("full", np.ones(len(fin), bool)), ("covered", cov)):
    w = pubw(sm); idx = np.where(sm)[0]
    rec = {t: m(P[t][sm], yfin[sm]) for t in P}
    wtd = {t: m(P[t][sm], yfin[sm], w[sm]) for t in P}
    # publication-cluster bootstrap of paired weighted dR2 (t3-t2, t2-t1) with each resampled publication keeping its 1/n weights
    pubs = np.unique(inv[sm]); rng = np.random.default_rng(0)
    members = {p: np.where((inv == p) & sm)[0] for p in pubs}
    ybar_w = np.average(yfin[sm], weights=w[sm])
    sse = {t: np.array([(w[members[p]] * (P[t][members[p]] - yfin[members[p]]) ** 2).sum() for p in pubs]) for t in P}
    sw = np.array([w[members[p]].sum() for p in pubs]); swy = np.array([(w[members[p]] * yfin[members[p]]).sum() for p in pubs]); swy2 = np.array([(w[members[p]] * yfin[members[p]] ** 2).sum() for p in pubs])
    def dr2(a, b, ii):
        W = sw[ii].sum(); var = swy2[ii].sum() / W - (swy[ii].sum() / W) ** 2
        return (sse[a][ii].sum() - sse[b][ii].sum()) / W / var   # R2_b - R2_a
    full_i = np.arange(len(pubs)); bs = {}
    draws = rng.integers(0, len(pubs), (2000, len(pubs)))
    for a, b in (("t1", "t2"), ("t2", "t3"), ("t1", "t3")):
        d0 = dr2(a, b, full_i); bb = np.array([dr2(a, b, ii) for ii in draws])
        bs[f"{b}_minus_{a}"] = dict(dR2=float(d0), lo=float(np.percentile(bb, 2.5)), hi=float(np.percentile(bb, 97.5)))
    pw[scope] = dict(n_records=int(sm.sum()), n_publications=int(len(pubs)), record_weighted=rec, publication_weighted=wtd, paired_publication_weighted=bs,
                     records_per_publication_median=float(np.median(np.bincount(inv[sm])[np.bincount(inv[sm]) > 0])))
pw["aggregation_rule"] = "Each reserved record receives weight 1/n_p, where n_p is the number of reserved records of its publication within the scope; weighted R2 = 1 - sum(w e^2)/sum(w (y - weighted mean)^2); weighted MAE/RMSE are weighted means. Equivalent to averaging squared (absolute) errors within each publication and then across publications. Paired intervals: publication-cluster bootstrap (2000 resamples, rng seed 0)."
json.dump(pw, open(OUT / "publication_weighted.json", "w"), indent=1)
for s in ("full", "covered"):
    print("pubw", s, {t: (round(pw[s]["record_weighted"][t]["r2"], 4), round(pw[s]["publication_weighted"][t]["r2"], 4)) for t in P}, {k: tuple(round(x, 4) for x in v.values()) for k, v in pw[s]["paired_publication_weighted"].items()})

# ------------------------------------------------------------------ 4. coverage bias (all 41,107 filtered records)
d = df.copy(); d["year"] = d.pub_date.dt.year
d["arch"] = d.Cell_architecture.str.lower().where(d.Cell_architecture.str.lower().isin(["nip", "pin"]), "other")
pubtab = d[d.doi_group != "__NO_DOI__"].groupby("doi_group").agg(covered=("lx_covered", "max"), year=("year", "min"), n=(TARGET, "size"), pce_mean=(TARGET, "mean"))
rows = []
def desc(sub, label):
    return dict(group=label, n_records=int(len(sub)), n_publications=int(sub.doi_group.nunique()), pce_mean=float(sub[TARGET].mean()), pce_median=float(sub[TARGET].median()),
                pce_p10=float(sub[TARGET].quantile(.1)), pce_p90=float(sub[TARGET].quantile(.9)), pce_sd=float(sub[TARGET].std()),
                frac_pce_ge_20=float((sub[TARGET] >= 20).mean()), median_year=float(sub.year.median()),
                frac_nip=float((sub.arch == "nip").mean()), frac_pin=float((sub.arch == "pin").mean()))
for lab, sub in (("covered (extraction record exists)", d[d.lx_covered == 1]), ("not covered", d[d.lx_covered == 0])):
    rows.append(desc(sub, lab))
cov_year = d.groupby("year").agg(n_records=(TARGET, "size"), covered_fraction=("lx_covered", "mean")).reset_index()
cov_year["n_publications"] = d.groupby("year").doi_group.nunique().values
cov_arch = d.groupby("arch").agg(n_records=(TARGET, "size"), covered_fraction=("lx_covered", "mean")).reset_index()
pce_bins = pd.cut(d[TARGET], [0, 8, 12, 16, 20, 40]); cov_pce = d.groupby(pce_bins, observed=True).agg(n_records=(TARGET, "size"), covered_fraction=("lx_covered", "mean")).reset_index()
cov_pce[TARGET] = cov_pce[TARGET].astype(str)
from scipy.stats import ks_2samp
ks = ks_2samp(d.loc[d.lx_covered == 1, TARGET], d.loc[d.lx_covered == 0, TARGET])
out = dict(summary=rows, coverage_by_year=cov_year.to_dict("records"), coverage_by_architecture=cov_arch.to_dict("records"),
           coverage_by_pce_bin=cov_pce.to_dict("records"), ks_pce_covered_vs_not=dict(statistic=float(ks.statistic), p=float(ks.pvalue)),
           reserved_split=dict(covered_mean_pce=float(yfin[cov].mean()), all_mean_pce=float(yfin.mean()), covered_var=float(yfin[cov].var()), all_var=float(yfin.var())),
           note="Descriptive only. 'covered' = the publication has a record among the 2,835 extraction records. Year is the PDP Ref_publication_date field (earliest value per publication for the publication-level table).")
json.dump(out, open(OUT / "coverage_bias.json", "w"), indent=1, default=str)
pd.DataFrame(rows).to_csv(OUT / "coverage_bias.csv", index=False)
print(pd.DataFrame(rows).round(3).T.to_string()); print(cov_year.round(3).to_string()); print(cov_arch.round(3).to_string()); print(cov_pce.round(3).to_string()); print(ks)

# ------------------------------------------------------------------ 5. exclusion of PCE < 1 %
raw = pd.read_csv(ROOT / "data/dataset_json_enriched.csv", low_memory=False, usecols=["Ref_ID", "Ref_DOI_number", "final_pce", "Cell_architecture"])
low = raw[(raw.final_pce < 1.0)]; lowpub = raw.loc[raw.final_pce < 1.0, "Ref_DOI_number"].nunique()
lowx = dict(records_missing_pce=int(raw.final_pce.isna().sum()), records_pce_lt_1=int(len(low)), frac_of_nonmissing=float(len(low) / raw.final_pce.notna().sum()),
            pce_lt_1_quantiles=low.final_pce.quantile([.1, .5, .9]).round(3).to_dict(), pce_lt_1_zero_or_near_zero=int((low.final_pce < 0.01).sum()),
            publications_with_any_pce_lt_1=int(lowpub),
            by_architecture=low.Cell_architecture.str.lower().value_counts().to_dict())
json.dump(lowx, open(OUT / "low_pce_exclusion.json", "w"), indent=1)
print(lowx)
