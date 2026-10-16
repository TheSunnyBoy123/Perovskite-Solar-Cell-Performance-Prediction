#!/usr/bin/env python3
"""One-shot evaluation on the reserved v4final publications (DOI-disjoint from v4dev; never used for any choice).

Arms: t2 / t3 with matched default CatBoost config (Question A: same estimator, same budget) and with each tier's own
tuned config from results/v4/tune.json (Question B); t2 + 5 DOI-permuted recipe blocks and a coverage-flag-only block as
negative controls. Predictions are the mean of --n-seeds seed models. Paired cluster bootstrap over publications.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from psc.data import load_dataset, TARGET
from psc import v4
from psc.v4model import PSCModel

ap = argparse.ArgumentParser(); ap.add_argument("--n-seeds", type=int, default=5); ap.add_argument("--out", default="results/v4/final")
ap.add_argument("--no-controls", action="store_true"); a = ap.parse_args()
out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
tune = json.loads((ROOT / "results/v4/tune.json").read_text())
best = {t: min(tune[t], key=lambda r: r["mse"])["cfg"] for t in ("t2", "t3")}
json.dump(best, open(out / "tuned_cfg.json", "w"), indent=1)

df = load_dataset(); df["part"] = df["doi_group"].map(v4.v4_partition)
dev, fin = df[df.part == "v4dev"].reset_index(drop=True), df[df.part == "v4final"].reset_index(drop=True)
ydev, yfin = dev[TARGET].values.astype(float), fin[TARGET].values.astype(float)
print(f"dev n={len(dev)} dois={dev.doi_group.nunique()} | final n={len(fin)} dois={fin.doi_group.nunique()} covered={int(fin.lx_covered.sum())}", flush=True)

class Ctl(PSCModel):   # t2 + a control block in place of the recipe block
    def __init__(self, T, **k): super().__init__("t3", **k); self.T = T
def run(name, tier, cfg, ctl=None):
    f = out / f"pred_{name}.npy"
    if f.exists(): return np.load(f)
    m = PSCModel(tier, cfg, seeds=range(a.n_seeds))
    if ctl is not None:
        import psc.v4model as vm
        orig = vm.rec_columns
        vm.rec_columns = lambda d: v4.attach(d, ctl)
        try: m.fit(dev, ydev, dev.doi_group.values); P = m.predict(fin, per_seed=True)
        finally: vm.rec_columns = orig
    else:
        m.fit(dev, ydev, dev.doi_group.values); P = m.predict(fin, per_seed=True)
    np.save(f, P); print("done", name, flush=True); return P

arms = {"t2_default": ("t2", {}, None), "t3_default": ("t3", {}, None),
        "t2_tuned": ("t2", best["t2"], None), "t3_tuned": ("t3", best["t3"], None),
        "t3cfg_on_t2": ("t2", best["t3"], None), "t2cfg_on_t3": ("t3", best["t2"], None)}
if not a.no_controls:
    Trec = v4.paper_table("rec")
    for j in range(5):
        rng = np.random.default_rng(100 + j)
        arms[f"ctl_perm{j}_default"] = ("t3", {}, pd.DataFrame(Trec.values[rng.permutation(len(Trec))], index=Trec.index, columns=Trec.columns))
    arms["ctl_coverflag_default"] = ("t3", {}, pd.DataFrame({"rc_covered": 1.0}, index=Trec.index))
preds = {k: run(k, *v) for k, v in arms.items()}

cov = fin.lx_covered.values == 1; g = fin.doi_group.values
ug, inv = np.unique(g, return_inverse=True)
def stats(P, mask):
    e = P.mean(0) - yfin
    return dict(r2=1 - (e[mask] ** 2).mean() / yfin[mask].var(), mae=float(np.abs(e[mask]).mean()), rmse=float(np.sqrt((e[mask] ** 2).mean())))
def paired(A, B, mask, nb=2000):
    """B vs A: delta R2 (positive = B better) with publication-cluster bootstrap CI, plus per-seed deltas."""
    d = ((A.mean(0) - yfin) ** 2 - (B.mean(0) - yfin) ** 2)
    s = np.bincount(inv[mask], weights=d[mask], minlength=len(ug)); n = np.bincount(inv[mask], minlength=len(ug)); ok = n > 0
    s, n = s[ok], n[ok]; rng = np.random.default_rng(0); idx = rng.integers(0, len(s), (nb, len(s)))
    bs = s[idx].sum(1) / n[idx].sum(1) / yfin[mask].var()
    per_seed = [float((((A[i] - yfin) ** 2 - (B[i] - yfin) ** 2)[mask]).mean() / yfin[mask].var()) for i in range(len(A))]
    return dict(dR2=float(d[mask].mean() / yfin[mask].var()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                seeds_pos=int(sum(x > 0 for x in per_seed)), n_seeds=len(per_seed), per_seed=per_seed)
rows = []
for scope, mask in (("full", np.ones(len(fin), bool)), ("covered", cov)):
    r = {"scope": scope, "n": int(mask.sum()), "dois": int(len(set(g[mask])))}
    for k, P in preds.items(): rows.append({**r, "arm": k, **stats(P, mask)})
pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
comp = {}
for scope, mask in (("full", np.ones(len(fin), bool)), ("covered", cov)):
    comp[scope] = {
        "A_matched_default (t3 vs t2)": paired(preds["t2_default"], preds["t3_default"], mask),
        "A_matched_tuned_t3cfg (t3 vs t2 under t3's config)": paired(preds["t3cfg_on_t2"], preds["t3_tuned"], mask),
        "A_matched_tuned_t2cfg (t3 vs t2 under t2's config)": paired(preds["t2_tuned"], preds["t2cfg_on_t3"], mask),
        "B_each_tuned (t3_tuned vs t2_tuned)": paired(preds["t2_tuned"], preds["t3_tuned"], mask),
        "tuning_effect_t2 (t2_tuned vs t2_default)": paired(preds["t2_default"], preds["t2_tuned"], mask),
        "tuning_effect_t3 (t3_tuned vs t3_default)": paired(preds["t3_default"], preds["t3_tuned"], mask)}
    for k in preds:
        if k.startswith("ctl_"): comp[scope][f"{k} vs t2_default"] = paired(preds["t2_default"], preds[k], mask)
json.dump(comp, open(out / "comparisons.json", "w"), indent=1)
print(pd.DataFrame(rows).round(4).to_string())
for sc in comp:
    print("==", sc)
    for k, v in comp[sc].items(): print(f"{k:60s} dR2={v['dR2']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}] seeds>0 {v['seeds_pos']}/{v['n_seeds']}")
