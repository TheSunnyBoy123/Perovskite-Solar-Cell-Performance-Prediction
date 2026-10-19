#!/usr/bin/env python3
"""Covered-population development selection of the literature representations C1-C3 (C4 if >= 2 pass).

Population: development partition (v4dev), records with lx_covered == 1 only. Training and test sets are
both drawn from this population. Folds: DOI-grouped 5-fold, seeds 42, 43, 44 (15 paired folds). M2 (the
t2 base) and every candidate are fitted on the same training fold with the same CatBoost configuration and
preprocessing (TrainFitPreprocessor fitted on the training fold). Candidate columns carry a first-feature-use
penalty of 0.5; M2 has none.

Selection is computed by the pre-registered rule below and written to selection.json. Nothing is tuned on
the reserved partition or the repeated holdouts, and the thresholds are fixed in the code before any run.

Usage:  python scripts/select_covered.py [--smoke]
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from psc import candidates as cd  # noqa: E402
from psc import v4  # noqa: E402
from psc.data import CAT_ALL, TARGET, load_dataset  # noqa: E402
from psc.features import TrainFitPreprocessor, base_frame  # noqa: E402
from psc.splits import make_splits  # noqa: E402

OUT = ROOT / "results" / "candidates" / "covered_dev"
SEEDS = (42, 43, 44)
N_FOLDS = 5
N_CTRL = 5

# Pre-registered selection rule (fixed before any result is examined).
RULE = dict(min_mean_dR2=0.003, min_positive_folds=11, min_excess_over_controls=0.005,
            min_subgroup_dR2=-0.005, max_dMAE=0.02, max_dRMSE=0.02)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_head():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def fit_predict(Z, y, tr, va, te, cat, pen=None):
    p = dict(iterations=1500, learning_rate=0.05, depth=8, l2_leaf_reg=3.0, subsample=0.8, colsample_bylevel=0.7,
             min_data_in_leaf=5, random_seed=42, loss_function="RMSE")
    if pen is not None:
        p["first_feature_use_penalties"] = pen
        p["penalties_coefficient"] = 1.0
    ci = [i for i, c in enumerate(Z.columns) if c in cat]
    m = CatBoostRegressor(**p, verbose=0, thread_count=-1)
    m.fit(Z.iloc[tr], y[tr], cat_features=ci, eval_set=(Z.iloc[va], y[va]), early_stopping_rounds=100, verbose=False)
    return m.predict(Z.iloc[te])


def metrics(y, p):
    e = p - y
    return dict(r2=float(1 - (e ** 2).mean() / y.var()), mae=float(np.abs(e).mean()), rmse=float(np.sqrt((e ** 2).mean())))


def control_tables(tables):
    """Paper-level permutation controls, built once. Control j of candidate k permutes k's table (and REC for C3)."""
    ctl = {}
    for ci, k in enumerate(("C1", "C2", "C3")):
        for j in range(N_CTRL):
            seed = 1000 + 10 * ci + j
            t = dict(tables)
            t[k] = cd.permute_block(tables[k], seed)
            if k == "C3":
                t["REC"] = cd.permute_block(tables["REC"], seed)
            ctl[(k, j)] = t
    return ctl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="first fold of the first seed, one control")
    a = ap.parse_args()
    global N_CTRL
    seeds = SEEDS[:1] if a.smoke else SEEDS
    n_folds = 1 if a.smoke else N_FOLDS
    N_CTRL = 1 if a.smoke else N_CTRL
    OUT.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    df["v4part"] = df["doi_group"].map(v4.v4_partition)
    dev = df[(df.v4part == "v4dev") & (df.lx_covered == 1)].reset_index(drop=True)
    y = dev[TARGET].values.astype(float)
    g = dev["doi_group"].values
    arch = dev["Cell_architecture"].str.lower().values
    X2 = base_frame(dev, "t2")
    print(f"covered dev n={len(dev)} dois={len(set(g))} base features={X2.shape[1]}", flush=True)

    tables = cd.paper_tables()
    ctl = control_tables(tables)

    arms = [("M2", [], None), ("C1", ["C1"], None), ("C2", ["C2"], None), ("C3", ["C3"], None)]
    arms += [(f"{k}_ctl{j}", [k], j) for k in ("C1", "C2", "C3") for j in range(N_CTRL)]

    def blocks(kinds, j, tr):
        out = []
        for k in kinds:
            t = tables if j is None else ctl[(k, j)]
            out.append(cd.build(k, t, dev, tr))
        return out

    rows, feats = [], {}
    folds = [(seed, k, tr, va, te) for seed in seeds
             for k, (tr, va, te) in enumerate(make_splits(g, "kfold", N_FOLDS, 0.2, 0.15, seed), 1) if k <= n_folds]
    t0 = time.time()
    for seed, fk, tr, va, te in folds:
        assert not set(g[tr]) & set(g[te]) and not set(g[va]) & set(g[te]) and not set(g[tr]) & set(g[va]), "DOI overlap"
        pre = TrainFitPreprocessor().fit(X2.iloc[tr])
        Z2 = pre.transform(X2).reset_index(drop=True)
        preds = {}
        for name, kinds, j in arms:
            bl = blocks(kinds, j, tr)
            if bl:
                cand = pd.concat(bl, axis=1)
                cd.assert_prefixes(cand.columns)
                Z = pd.concat([Z2, cand], axis=1)
                pen = cd.penalty_vector(Z.columns, cand.columns)
                if sum(pen) == 0:
                    raise AssertionError("candidate arm without penalties")
                cov = float(v4.attach(dev, tables[kinds[0]] if j is None else ctl[(kinds[0], j)][kinds[0]]).iloc[te].notna().any(axis=1).mean())
                feats[(name, seed, fk)] = list(cand.columns)
            else:
                Z, pen, cov = Z2, None, 1.0
                feats[(name, seed, fk)] = list(Z2.columns)
            pr = fit_predict(Z, y, tr, va, te, set(CAT_ALL), pen)
            preds[name] = pr
            m = metrics(y[te], pr)
            ma = te[arch[te] == "pin"]; mn = te[arch[te] == "nip"]
            mp = metrics(y[ma], pr[arch[te] == "pin"]); mq = metrics(y[mn], pr[arch[te] == "nip"])
            rows.append(dict(arm=name, seed=seed, fold=fk, n=len(te), ndoi=len(set(g[te])), nfeat=Z.shape[1], cov_frac=cov,
                             **m, r2_pin=mp["r2"], mae_pin=mp["mae"], rmse_pin=mp["rmse"], n_pin=len(ma),
                             r2_nip=mq["r2"], mae_nip=mq["mae"], rmse_nip=mq["rmse"], n_nip=len(mn)))
        np.savez_compressed(OUT / f"preds_s{seed}_f{fk}.npz", test_idx=te, y=y[te], **preds)
        print(f"seed {seed} fold {fk} done ({time.time() - t0:.0f}s)", flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(OUT / "folds.csv", index=False)
    sel = select(R, tables, ctl, folds, X2, dev, y, g, arch, feats)
    (OUT / "selection.json").write_text(json.dumps(sel, indent=1))
    print(json.dumps({k: v for k, v in sel.items() if k != "feature_counts"}, indent=1)[:3000])


def summarise(R, cand, M2):
    """Paired per-fold differences of candidate `cand` against M2 and against its own controls."""
    c = R[R.arm == cand].sort_values(["seed", "fold"]).set_index(["seed", "fold"])
    b = M2.set_index(["seed", "fold"]).loc[c.index]
    d = c.r2 - b.r2
    ctl = R[R.arm.str.startswith(cand + "_ctl")].copy()
    ctl = ctl.merge(M2[["seed", "fold", "r2"]].rename(columns={"r2": "r2_m2"}), on=["seed", "fold"])
    ctl["d"] = ctl.r2 - ctl.r2_m2
    ctl_fold = ctl.groupby(["seed", "fold"]).d.mean()
    out = dict(
        mean_dR2=float(d.mean()), sd_dR2=float(d.std(ddof=1)), positive_folds=int((d > 0).sum()), n_folds=int(len(d)),
        fold_dR2=[round(float(x), 5) for x in d.values],
        control_mean_dR2=float(ctl_fold.mean()),
        control_fold_dR2=[round(float(x), 5) for x in ctl_fold.reindex(c.index).values],
        excess_over_controls=float(d.mean() - ctl_fold.mean()),
        mean_dR2_pin=float((c.r2_pin - b.r2_pin).mean()), mean_dR2_nip=float((c.r2_nip - b.r2_nip).mean()),
        mean_dMAE=float((c.mae - b.mae).mean()), mean_dRMSE=float((c.rmse - b.rmse).mean()),
        mean_nfeat=float(c.nfeat.mean()), mean_cov_frac=float(c.cov_frac.mean()),
    )
    out["checks"] = dict(
        dR2=out["mean_dR2"] >= RULE["min_mean_dR2"],
        positive=out["positive_folds"] >= RULE["min_positive_folds"],
        controls=out["excess_over_controls"] >= RULE["min_excess_over_controls"],
        subgroups=(out["mean_dR2_pin"] >= RULE["min_subgroup_dR2"]) and (out["mean_dR2_nip"] >= RULE["min_subgroup_dR2"]),
        error=(out["mean_dMAE"] <= RULE["max_dMAE"]) and (out["mean_dRMSE"] <= RULE["max_dRMSE"]),
    )
    out["pass"] = all(out["checks"].values())
    return out


def select(R, tables, ctl, folds, X2, dev, y, g, arch, feats):
    M2 = R[R.arm == "M2"]
    res = {k: summarise(R, k, M2) for k in ("C1", "C2", "C3")}
    passing = [k for k in ("C1", "C2", "C3") if res[k]["pass"]]
    c4_run = len(passing) >= 2
    if c4_run:
        # C4 = union of the passing candidates; its controls permute each passing block with its own control seed.
        R4 = run_c4(passing, tables, ctl, folds, X2, dev, y, g, arch, feats, M2)
        R = pd.concat([R, R4], ignore_index=True)
        R.to_csv(OUT / "folds.csv", index=False)
        res["C4"] = summarise(R, "C4", M2)
    eligible = [k for k in res if res[k]["pass"]]
    selected = max(eligible, key=lambda k: res[k]["mean_dR2"]) if eligible else None
    return dict(
        rule=RULE, population="covered, development partition (v4dev, lx_covered == 1)",
        folds="DOI-grouped 5-fold kfold, seeds " + ",".join(map(str, sorted(set(R.seed)))),
        n_controls=N_CTRL, c4_evaluated=c4_run, passing_C1_C3=passing, candidates=res,
        selected=selected,
        selected_rule="among candidates passing every criterion, the highest mean paired dR2 (none if none pass)",
        data_sha256=sha(ROOT / "data" / "dataset_json_enriched.csv"), lit_v4_sha256=sha(v4.LIT_V4),
        code_git_head=git_head(), feature_counts={f"{k[0]}_s{k[1]}_f{k[2]}": len(v) for k, v in feats.items()},
    )


def run_c4(passing, tables, ctl, folds, X2, dev, y, g, arch, feats, M2):
    rows = []
    for seed, fk, tr, va, te in folds:
        pre = TrainFitPreprocessor().fit(X2.iloc[tr])
        Z2 = pre.transform(X2).reset_index(drop=True)
        for name, j in [("C4", None)] + [(f"C4_ctl{j}", j) for j in range(N_CTRL)]:
            bl = []
            for k in passing:
                t = tables if j is None else ctl[(k, j)]
                bl.append(cd.build(k, t, dev, tr))
            cand = pd.concat(bl, axis=1)
            cd.assert_prefixes(cand.columns)
            Z = pd.concat([Z2, cand], axis=1)
            pen = cd.penalty_vector(Z.columns, cand.columns)
            pr = fit_predict(Z, y, tr, va, te, set(CAT_ALL), pen)
            m = metrics(y[te], pr)
            ma = arch[te] == "pin"; mn = arch[te] == "nip"
            mp = metrics(y[te][ma], pr[ma]); mq = metrics(y[te][mn], pr[mn])
            rows.append(dict(arm=name, seed=seed, fold=fk, n=len(te), ndoi=len(set(g[te])), nfeat=Z.shape[1],
                             cov_frac=1.0, **m, r2_pin=mp["r2"], mae_pin=mp["mae"], rmse_pin=mp["rmse"], n_pin=int(ma.sum()),
                             r2_nip=mq["r2"], mae_nip=mq["mae"], rmse_nip=mq["rmse"], n_nip=int(mn.sum())))
            feats[(name, seed, fk)] = list(cand.columns)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
