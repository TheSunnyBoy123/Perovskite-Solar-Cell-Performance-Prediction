#!/usr/bin/env python3
"""Feature-level diagnostics of the 36 LLM recipe flags (dev partition only; v4final never loaded).

Writes to experiments/m3opt_20261019/:
  feature_statistics.csv    per flag: prevalence (papers / covered records), coverage by architecture, raw and
                            M2-residual association (dev OOF residuals of the E1 M2 arm, seed 42), sign stability
                            across the 5 dev folds, redundancy (max |phi| with other flags, max |r| with M1/M2 columns),
                            held-out grouped permutation importance (dMSE on covered test records, M3_strict, seed 42)
  feature_correlations.csv  flag x flag phi matrix (covered papers)
  feature_provenance.csv    per feature: source, definition, transformation, missing-value meaning
  feature_schema.json       canonical representation, states, tier definitions
  configs/tiers.json        exact raw M1 / M2 / M3 column lists (before train-fitted filtering) and a full-dev
                            post-filter list
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
import m3feat as F  # noqa: E402
from run_dev import dev_frame, CB_DEFAULT  # noqa: E402
from psc.data import TARGET, CAT_ALL, FORBIDDEN, TIER_DROP  # noqa: E402
from psc.features import base_frame, TrainFitPreprocessor  # noqa: E402
from psc.splits import make_splits  # noqa: E402
from psc import litfeat4 as L4  # noqa: E402


def main():
    df = dev_frame(); y = df[TARGET].values.astype(float); g = df.doi_group.values
    arch = df.Cell_architecture.str.lower().values
    T = {v: F.flags(v) for v in ("v4", "strict")}
    R = F.attach(g, T["strict"]); cov = R.notna().any(axis=1).values
    # M2 OOF residuals from E1 (seed 42)
    run = EXP / "runs" / "E1_screen"
    p = np.full(len(y), np.nan); fold = np.zeros(len(y), int)
    for k in range(1, 6):
        z = np.load(run / "preds" / f"M2_s42_f{k}.npz"); p[z["test_idx"]] = z["pred"]; fold[z["test_idx"]] = k
    res = y - p
    X2 = base_frame(df, "t2")
    num2 = X2.select_dtypes("number")
    paper_cov = T["strict"].dropna(how="all")
    rows = []
    phi = paper_cov.corr()
    for c in F.FLAG_NAMES:
        x = R[c].values
        grp = next(k for k, v in F.GROUPS.items() if c in v)
        m1 = cov & (x == 1); m0 = cov & (x == 0)
        fold_sign = []
        for k in range(1, 6):
            a, b = m1 & (fold == k), m0 & (fold == k)
            fold_sign.append(np.sign(res[a].mean() - res[b].mean()) if a.sum() >= 10 and b.sum() >= 10 else 0)
        xc = pd.Series(x[cov])
        rmax = num2[cov].corrwith(xc.set_axis(num2[cov].index)).abs()
        rows.append(dict(
            flag=c, group=grp, regex=L4.RECIPE[c],
            papers_pos=int((paper_cov[c] == 1).sum()), papers_neg=int((paper_cov[c] == 0).sum()),
            paper_prevalence=float(paper_cov[c].mean()),
            v4_paper_prevalence=float(T["v4"][c].mean()),
            dev_records_pos=int(m1.sum()), dev_records_cov=int(cov.sum()),
            prev_pin=float(np.nanmean(x[cov & (arch == "pin")])), prev_nip=float(np.nanmean(x[cov & (arch == "nip")])),
            dPCE_raw=float(y[m1].mean() - y[m0].mean()) if m1.any() and m0.any() else np.nan,
            dPCE_paperlevel=float(pd.Series(y[m1]).groupby(g[m1]).mean().mean() - pd.Series(y[m0]).groupby(g[m0]).mean().mean())
            if m1.any() and m0.any() else np.nan,
            dResid_M2=float(res[m1].mean() - res[m0].mean()) if m1.any() and m0.any() else np.nan,
            resid_sign_folds=" ".join(f"{int(s):+d}" for s in fold_sign),
            resid_sign_consistent=int(abs(sum(fold_sign)) == 5),
            max_phi_other_flag=float(phi[c].drop(c).abs().max()), max_phi_with=phi[c].drop(c).abs().idxmax(),
            max_abs_r_M2_column=float(rmax.max()) if rmax.notna().any() else np.nan,
            max_r_with=rmax.idxmax() if rmax.notna().any() else None))
    st = pd.DataFrame(rows)

    # held-out grouped permutation importance (M3_strict, seed 42 folds, covered test records)
    from catboost import CatBoostRegressor
    rng = np.random.default_rng(0)
    imp = {c: [] for c in F.FLAG_NAMES}; gimp = {k: [] for k in F.GROUPS}; allimp = []
    Rr = R.add_prefix("rc_")
    for k, (tr, va, te) in enumerate(make_splits(g, "kfold", 5, 0.2, 0.15, 42), 1):
        pre = TrainFitPreprocessor().fit(X2.iloc[tr]); Z = pd.concat([pre.transform(X2).reset_index(drop=True), Rr], axis=1)
        prm = {**CB_DEFAULT, "random_seed": 4200, "first_feature_use_penalties": [0.5 if c.startswith("rc_") else 0.0 for c in Z.columns],
               "penalties_coefficient": 1.0}
        m = CatBoostRegressor(**prm, verbose=0, thread_count=-1)
        m.fit(Z.iloc[tr], y[tr], cat_features=[i for i, c in enumerate(Z.columns) if c in CAT_ALL],
              eval_set=(Z.iloc[va], y[va]), early_stopping_rounds=100, verbose=False)
        tc = te[cov[te]]
        Zt = Z.iloc[tc].copy(); base = np.mean((m.predict(Zt) - y[tc]) ** 2)
        # permute among covered test publications at the publication level (flags are paper-level)
        gt = g[tc]; up = np.unique(gt)

        def perm_cols(cols, reps=5):
            out = []
            for _ in range(reps):
                Zp = Zt.copy()
                pp = pd.DataFrame(Zt[cols].values, index=gt, columns=cols)
                paper = pp[~pp.index.duplicated()].loc[up]
                sh = paper.iloc[rng.permutation(len(up))].set_axis(up)
                Zp[cols] = sh.loc[gt].values
                out.append(np.mean((m.predict(Zp) - y[tc]) ** 2) - base)
            return float(np.mean(out))
        for c in F.FLAG_NAMES:
            imp[c].append(perm_cols(["rc_" + c]))
        for gk, cs in F.GROUPS.items():
            gimp[gk].append(perm_cols(["rc_" + c for c in cs]))
        allimp.append(perm_cols(["rc_" + c for c in F.FLAG_NAMES]))
        print("fold", k, "base mse", round(base, 3), "all-flags perm dMSE", round(allimp[-1], 4), flush=True)
    st["perm_dMSE_mean"] = [np.mean(imp[c]) for c in F.FLAG_NAMES]
    st["perm_dMSE_folds_pos"] = [int(sum(v > 0 for v in imp[c])) for c in F.FLAG_NAMES]
    st.to_csv(EXP / "feature_statistics.csv", index=False)
    gi = pd.DataFrame({"group": list(F.GROUPS) + ["ALL36"],
                       "perm_dMSE_mean": [np.mean(gimp[k]) for k in F.GROUPS] + [np.mean(allimp)],
                       "folds_pos": [int(sum(v > 0 for v in gimp[k])) for k in F.GROUPS] + [int(sum(v > 0 for v in allimp))],
                       "per_fold": [" ".join(f"{v:+.4f}" for v in gimp[k]) for k in F.GROUPS] + [" ".join(f"{v:+.4f}" for v in allimp)]})
    gi.to_csv(EXP / "feature_group_permutation.csv", index=False)
    phi.round(4).to_csv(EXP / "feature_correlations.csv")

    # provenance + schema + tiers
    prov = [dict(feature="rc_" + c, tier="M3", source="raw LLM JSON (../json) -> litfeat4.paper_recipe regex over non-outcome "
                 "key names and string values of all material records (role-scoped for htl_/etl_)", regex=L4.RECIPE[c],
                 level="publication", values="1 = mentioned in extraction; 0 = not mentioned (NOT a verified negative); "
                 "NaN = no joined JSON or empty extraction", group=next(k for k, v in F.GROUPS.items() if c in v))
            for c in F.FLAG_NAMES]
    raw1, raw2 = base_frame(df, "t1"), base_frame(df, "t2")
    for c in raw2.columns:
        prov.append(dict(feature=c, tier="M1" if c in raw1.columns else "M2",
                         source="PDP export / composition parsing" if c in raw1.columns else "material-property catalogue",
                         regex="", level="device record", values="numeric/categorical; NaN -> train-median + _na flag", group=""))
    pd.DataFrame(prov).to_csv(EXP / "feature_provenance.csv", index=False)
    pre = TrainFitPreprocessor().fit(raw2)
    tiers = {"note": "raw columns = psc.features.base_frame(df, tier) (row-wise, learns nothing); the train-fitted "
                     "TrainFitPreprocessor then drops >80%-missing columns and adds *_na flags per training fold.",
             "M1_raw": list(raw1.columns), "M2_raw": list(raw2.columns),
             "M2_minus_M1": [c for c in raw2.columns if c not in raw1.columns],
             "M3_raw": list(raw2.columns) + ["rc_" + c for c in F.FLAG_NAMES],
             "M3_flags": ["rc_" + c for c in F.FLAG_NAMES],
             "M2_post_filter_full_dev": list(pre.transform(raw2.head(5)).columns),
             "categorical": [c for c in CAT_ALL if c in raw2.columns], "forbidden": FORBIDDEN,
             "tier_drop_sets": {k: sorted(v) for k, v in TIER_DROP.items() if k in ("t1", "t2", "t3")}}
    (EXP / "configs" / "tiers.json").write_text(json.dumps(tiers, indent=1))
    schema = {
        "canonical_long_table": "features/json_long.csv.gz: doi_group, file, mat_idx, role, canon_name, key, key_class "
                                "(identity/outcome/characterisation/property/process/other), value_kind, value_raw, value_num",
        "paper_states": {"covered": "joined JSON with >= 1 material record", "empty": "JSON parsed but 0 material records "
                         "(17 dev+final papers) -> all M3 features NaN (production v4 set them to 0)",
                         "uncovered": "no JSON for the publication -> NaN", "unjoined": "JSON whose DOI matches no PDP publication"},
        "not_recoverable": ["evidence spans", "extraction confidence", "explicit negatives", "device-level assignment",
                            "inapplicable vs absent"],
        "doi_resolution_rules": ["R1 filename first '_' -> '/' (production)", "R2 filename all '_' -> '/' (IOP)",
                                 "R3 JSON 'source' field when it is a DOI", "else unjoined"],
        "flag_variants": {"v4": "stored data/lit_v4.json (production M3)", "fixed": "R1-R3 resolution + empty -> NaN",
                          "strict": "fixed + extended outcome-key filter (stability / PCE [%] / Voc (V) ... keys removed "
                                    "before the regex search)"},
        "tiers": {"M1": "base_frame(t1)", "M2": "base_frame(t2)", "M3": "M2 + 36 rc_* flags (CatBoost first-feature-use penalty 0.5)"}}
    (EXP / "feature_schema.json").write_text(json.dumps(schema, indent=1))
    pd.set_option("display.width", 250)
    print(st[["flag", "papers_pos", "paper_prevalence", "dPCE_raw", "dResid_M2", "resid_sign_folds", "max_phi_other_flag",
              "max_abs_r_M2_column", "perm_dMSE_mean", "perm_dMSE_folds_pos"]].round(3).to_string(index=False))
    print(gi.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
