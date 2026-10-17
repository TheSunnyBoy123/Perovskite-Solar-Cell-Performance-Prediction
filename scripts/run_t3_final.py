#!/usr/bin/env python3
"""Final-t3 (t2 + 36 V4 recipe flags) arms for the protocols that were run for t1/t2 only.

Families (python scripts/run_t3_final.py --config configs/t3final_<family>.json):
  matched    CatBoost + ridge, DOI-grouped / record-level / lead-author / forward-in-time (same splits as ablation_v1,
             strict_validation_v1, random_convergence_cap20000)
  baselines  XGBoost (defaults), random forest, MLP, subsample GPR on t2 and t3f (same splits as baselines_v1)
  model_dev  validation-selected CatBoost grid, 3-seed bag, XGBoost, mean ensemble for t3f (same splits/grid as model_dev_v1)
Resumable. The t3f test indices are asserted equal to those of the stored t2 arm of the same protocol.
"""
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from psc.data import CAT, DEFAULT_DATA, TARGET, load_dataset  # noqa: E402
from psc.evaluate import metrics  # noqa: E402
from psc.features import base_frame  # noqa: E402
from psc.models import fit_predict  # noqa: E402
from psc.splits import make_splits  # noqa: E402
from psc import t3final  # noqa: E402


def sh(cmd):
    try:
        return subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def ref_test_idx(refs, fname):
    for d in refs:
        f = ROOT / "results" / d / "preds" / fname
        if f.exists():
            return np.load(f)["test_idx"]
    return None


def prep(cfg, a):
    name = cfg["name"] + ("_smoke" if a.smoke else "")
    out = ROOT / "results" / name; (out / "preds").mkdir(parents=True, exist_ok=True)
    data_path = ROOT / cfg.get("data", str(DEFAULT_DATA.relative_to(ROOT)))
    meta = {"config": cfg, "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
            "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "lit_v4_sha256": hashlib.sha256((ROOT / "data/lit_v4.json").read_bytes()).hexdigest(),
            "git_head": sh(["git", "rev-parse", "HEAD"]), "first_feature_use_penalty": t3final.PEN}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return out, data_path


def done_set(f, keys):
    s = set()
    if f.exists():
        for l in f.read_text().splitlines():
            r = json.loads(l); s.add(tuple(r[k] for k in keys))
    return s


def run_matched(cfg, a):
    out, data_path = prep(cfg, a); runs = out / "runs.jsonl"
    keys = ("protocol", "arch", "tier", "model", "split"); done = done_set(runs, keys)
    for arch in cfg["archs"]:
        df = load_dataset(data_path, arch); y = df[TARGET].values.astype(float); groups = df["doi_group"].values
        Xraw = base_frame(df, "t2"); R = t3final.rec_block(df)
        for protocol in cfg["protocols"]:
            if protocol == "random" and arch not in cfg.get("random_archs", cfg["archs"]):
                continue
            g_split = df["author_group"].values if protocol == "author" else groups
            n_s = 3 if protocol == "temporal" else cfg["n_splits"]
            if a.smoke: n_s = min(n_s, 2)
            splits = list(make_splits(g_split, protocol, n_s, cfg["test_size"], cfg["val_size"], cfg["seed"],
                                      dates=df["pub_date"].values if "pub_date" in df else None))
            for model in cfg["models"]:
                params = dict(cfg.get("catboost", {}))
                if protocol == "random" and model == "catboost" and arch == "overall":
                    params.update(cfg.get("catboost_random_overall", {}))   # converged record-level fits (20,000-iteration cap)
                if a.smoke: params["iterations"] = 300
                for k, (tr, va, te) in enumerate(splits, 1):
                    key = (protocol, arch, "t3f", model, k)
                    if key in done: continue
                    fn = f"{protocol}_{arch}_t2_{model}_{k}.npz"
                    ref = ref_test_idx(cfg["ref_dirs"] if not (protocol == "random" and model == "catboost" and arch == "overall")
                                       else ["random_convergence_cap20000"] + cfg["ref_dirs"], fn)
                    if ref is not None:
                        assert np.array_equal(ref, te), f"test indices differ from stored t2 arm: {fn}"
                    Xtr, Xva, Xte, pen, nf = t3final.matrices(df, Xraw, tr, va, te, "t3f", impute=(model != "catboost"), R=R)
                    p_ = dict(params)
                    if model == "catboost":
                        p_["first_feature_use_penalties"] = pen; p_["penalties_coefficient"] = 1.0
                    t0 = time.time()
                    p, info = fit_predict(model, Xtr, y[tr], Xva, y[va], Xte, p_)
                    info.pop("params", None)
                    m = metrics(y[te], p)
                    rec = dict(protocol=protocol, arch=arch, tier="t3f", model=model, split=k, n_train=len(tr), n_val=len(va),
                               n_test=len(te), n_doi_test=len(set(groups[te])),
                               doi_overlap_test_train=len(set(groups[te]) & set(groups[tr])),
                               doi_overlap_val_train=len(set(groups[va]) & set(groups[tr])),
                               doi_overlap_test_val=len(set(groups[te]) & set(groups[va])),
                               n_features=Xtr.shape[1], n_flags=nf, penalty=t3final.PEN if model == "catboost" else None,
                               test_ref_checked=ref is not None, runtime_s=round(time.time() - t0, 1), info=info, **m)
                    np.savez_compressed(out / "preds" / f"{protocol}_{arch}_t3f_{model}_{k}.npz",
                                        test_idx=te, y=y[te], pred=p, train_idx=tr, val_idx=va)
                    with open(runs, "a") as f: f.write(json.dumps(rec) + "\n")
                    print(f"{protocol:8s} {arch:7s} t3f {model:8s} s{k} R2={m['r2']:.4f} MAE={m['mae']:.3f} nf={Xtr.shape[1]} [{rec['runtime_s']}s]", flush=True)


def run_baselines(cfg, a):
    out, data_path = prep(cfg, a); runs = out / "runs.jsonl"
    keys = ("protocol", "arch", "tier", "model", "split"); done = done_set(runs, keys)
    for arch in cfg["archs"]:
        df = load_dataset(data_path, arch); y = df[TARGET].values.astype(float); groups = df["doi_group"].values
        Xraw = base_frame(df, "t2"); R = t3final.rec_block(df)
        splits = list(make_splits(groups, "doi", 2 if a.smoke else cfg["n_splits"], cfg["test_size"], cfg["val_size"], cfg["seed"]))
        for tier in cfg["tiers"]:
            for model in cfg["models"]:
                for k, (tr, va, te) in enumerate(splits, 1):
                    key = ("doi", arch, tier, model, k)
                    if key in done: continue
                    ref = ref_test_idx(["ablation_v1"], f"doi_{arch}_t2_catboost_{k}.npz")
                    if ref is not None: assert np.array_equal(ref, te)
                    Xtr, Xva, Xte, _, nf = t3final.matrices(df, Xraw, tr, va, te, tier, impute=True, R=R)
                    t0 = time.time()
                    p, info = fit_predict(model, Xtr, y[tr], Xva, y[va], Xte, {})
                    info.pop("params", None); m = metrics(y[te], p)
                    rec = dict(protocol="doi", arch=arch, tier=tier, model=model, split=k, n_train=len(tr), n_val=len(va), n_test=len(te),
                               n_features=Xtr.shape[1], n_flags=nf, runtime_s=round(time.time() - t0, 1), info=info, **m)
                    np.savez_compressed(out / "preds" / f"doi_{arch}_{tier}_{model}_{k}.npz", test_idx=te, y=y[te], pred=p, train_idx=tr, val_idx=va)
                    with open(runs, "a") as f: f.write(json.dumps(rec) + "\n")
                    print(f"doi {arch} {tier} {model:11s} s{k} R2={m['r2']:.4f} MAE={m['mae']:.3f} nf={Xtr.shape[1]} [{rec['runtime_s']}s]", flush=True)


def run_model_dev(cfg, a):
    import run_model_dev as md
    from catboost import CatBoostRegressor
    from xgboost import XGBRegressor
    out, data_path = prep(cfg, a); runs = out / "runs.jsonl"
    done = done_set(runs, ("arch", "tier", "split"))
    base = dict(md.BASE); grid = md.GRID[:2] if a.smoke else md.GRID
    if a.smoke: base["iterations"] = 200
    for arch in cfg["archs"]:
        df = load_dataset(data_path, arch); y = df[TARGET].values.astype(float); g = df["doi_group"].values
        Xraw = base_frame(df, "t2"); R = t3final.rec_block(df)
        splits = list(make_splits(g, "doi", 2 if a.smoke else cfg["n_splits"], cfg["test_size"], cfg["val_size"], cfg["seed"]))
        for k, (tr, va, te) in enumerate(splits, 1):
            if (arch, "t3f", k) in done: continue
            ref = ref_test_idx(["model_dev_v1"], f"{arch}_t2_{k}.npz")
            if ref is not None: assert np.array_equal(ref, te)
            t0 = time.time()
            Xtr, Xva, Xte, pen, nf = t3final.matrices(df, Xraw, tr, va, te, "t3f", impute=False, R=R)
            ci = [i for i, c in enumerate(Xtr.columns) if c in CAT]
            kw = dict(first_feature_use_penalties=pen, penalties_coefficient=1.0)
            best = None
            for gp in grid:
                m = CatBoostRegressor(**base, **gp, **kw, random_seed=42, verbose=0, thread_count=-1)
                m.fit(Xtr, y[tr], cat_features=ci, eval_set=(Xva, y[va]), early_stopping_rounds=150, verbose=False)
                rm = float(np.sqrt(np.mean((m.predict(Xva) - y[va]) ** 2)))
                if best is None or rm < best[0]: best = (rm, gp, m)
            _, gp, m0 = best; p_cb = m0.predict(Xte); bag = [p_cb]
            for s in md.SEEDS[1:]:
                m = CatBoostRegressor(**base, **gp, **kw, random_seed=s, verbose=0, thread_count=-1)
                m.fit(Xtr, y[tr], cat_features=ci, eval_set=(Xva, y[va]), early_stopping_rounds=150, verbose=False)
                bag.append(m.predict(Xte))
            p_bag = np.mean(bag, axis=0)
            cats = {c: {v: i for i, v in enumerate(sorted(Xtr[c].unique()))} for c in CAT}
            def enc(X):
                Z = X.copy()
                for c in CAT: Z[c] = Z[c].map(cats[c]).fillna(-1)
                return Z.values.astype(np.float32)
            gx = XGBRegressor(n_estimators=1500 if not a.smoke else 100, max_depth=8, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.7, reg_lambda=3.0, tree_method="hist", early_stopping_rounds=100,
                              random_state=42, n_jobs=-1)
            gx.fit(enc(Xtr), y[tr], eval_set=[(enc(Xva), y[va])], verbose=False)   # recipe flags: native missing values
            p_x = gx.predict(enc(Xte)); preds = {"cb_tuned": p_cb, "cb_bag": p_bag, "xgb": p_x, "ens": (p_bag + p_x) / 2}
            rec = dict(arch=arch, tier="t3f", split=k, n_train=len(tr), n_val=len(va), n_test=len(te), selected=gp,
                       n_features=Xtr.shape[1], n_flags=nf, penalty=t3final.PEN, runtime_s=round(time.time() - t0, 1),
                       doi_overlap=len(set(g[te]) & (set(g[tr]) | set(g[va]))),
                       **{f"{n}_{mk}": v for n, p in preds.items() for mk, v in metrics(y[te], p).items()})
            np.savez_compressed(out / "preds" / f"{arch}_t3f_{k}.npz", test_idx=te, y=y[te], **preds)
            with open(runs, "a") as f: f.write(json.dumps(rec) + "\n")
            print(arch, "t3f", k, gp, {n: round(rec[f"{n}_r2"], 4) for n in preds}, f"[{rec['runtime_s']}s]", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); cfg = json.loads(Path(a.config).read_text())
    {"matched": run_matched, "baselines": run_baselines, "model_dev": run_model_dev}[cfg["family"]](cfg, a)
