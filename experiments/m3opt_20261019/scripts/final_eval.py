#!/usr/bin/env python3
"""Frozen-choice evaluation (run once after the development choices are fixed in configs/F_final.json).

Protocol R (reserved diagnostic): train on all v4dev publications, score the v4final publications once.
  Mirrors scripts/v4_final.py exactly: per model seed s in 0..4, a DOI-disjoint 15% validation split of v4dev
  (make_splits(groups, 'doi', 1, 0.001, 0.15, 1000 + s), the 0.1% sliver returned to training), preprocessing fit
  on that seed's training part, CatBoost default config with random_seed = s; prediction = mean of the 5 seed models.
  v4final was scored before (v4_final.py, and the single-feature sub.py experiment), so it is NOT pristine: the
  result is labelled a diagnostic, not a confirmatory test.
Protocol H (manuscript repeated DOI holdouts): full data, overall / pin / nip, make_splits(groups, 'doi', 5, 0.2, 0.15, 42)
  with CatBoost random_seed 42 (as ablation_v1 / t3final_matched_v1). Test indices are asserted equal to the stored
  ablation_v1 t2 predictions. The v4dev development publications overlap these test sets, so any arm selected on v4dev
  is advantaged here by selection; H is reported for comparability with the manuscript table only.

  python final_eval.py --config ../configs/F_final.json --protocol R|H
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(HERE))
from psc.data import load_dataset, TARGET, DEFAULT_DATA  # noqa: E402
from psc.features import base_frame, TrainFitPreprocessor  # noqa: E402
from psc.splits import make_splits  # noqa: E402
from psc import v4  # noqa: E402
import m3feat as F  # noqa: E402
from run_dev import build_block, fit_predict  # noqa: E402


def matrix(arm, df, y, g, tr, va, X):
    pre = TrainFitPreprocessor().fit(X[arm.get("base", "t2")].iloc[tr])
    parts = [pre.transform(X[arm.get("base", "t2")]).reset_index(drop=True)]
    ctxs = {}
    for spec in arm.get("blocks", []):
        ctx = ctxs.setdefault(f"{spec.get('perm')}|{spec.get('text', 'v1')}", {})
        parts.append(build_block(spec, df, y, g, tr, va, ctx).reset_index(drop=True))
    return pd.concat(parts, axis=1), len(parts) > 1


def protocol_R(cfg, out):
    full = load_dataset(); F.set_known(set(full.doi_group))
    full["part"] = full.doi_group.map(v4.v4_partition)
    dev = full[full.part == "v4dev"]; fin = full[full.part == "v4final"]
    df = pd.concat([dev, fin]).reset_index(drop=True)          # dev rows first, then final rows
    nd = len(dev); y = df[TARGET].values.astype(float); g = df.doi_group.values
    te = np.arange(nd, len(df))
    X = {t: base_frame(df, t) for t in ("t1", "t2")}
    np.save(out / "y_final.npy", y[te]); df.iloc[te][["doi_group", "Cell_architecture"]].to_csv(out / "rows_final.csv", index=False)
    for arm in cfg["arms"]:
        f = out / f"pred_{arm['name']}.npy"
        if f.exists():
            continue
        P = []; t0 = time.time()
        for s in range(cfg.get("n_seeds", 5)):
            tr, va, sl = next(make_splits(g[:nd], "doi", 1, 0.001, 0.15, 1000 + s))
            tr = np.concatenate([tr, sl])
            Z, has_lit = matrix(arm, df, y, g, tr, va, X)
            params = {**cfg.get("params", {}), **arm.get("params", {})}
            p, _ = fit_predict(Z, y, tr, va, te, params, arm.get("pen", 0.5) if has_lit else 0.0,
                               arm.get("model", "catboost"), s + arm.get("seed_offset", 0))
            P.append(p)
        np.save(f, np.array(P)); print("R", arm["name"], f"{time.time() - t0:.0f}s", flush=True)


def protocol_H(cfg, out):
    from catboost import CatBoostRegressor  # noqa: F401
    for archname in cfg.get("archs", ["overall", "pin", "nip"]):
        df = load_dataset(arch=archname); F.set_known(set(load_dataset().doi_group))
        y = df[TARGET].values.astype(float); g = df.doi_group.values
        X = {t: base_frame(df, t) for t in ("t1", "t2")}
        for k, (tr, va, te) in enumerate(make_splits(g, "doi", 5, 0.2, 0.15, 42), 1):
            ref = ROOT / "results" / "ablation_v1" / "preds" / f"doi_{archname}_t2_catboost_{k}.npz"
            assert np.array_equal(np.load(ref)["test_idx"], te), "test split differs from manuscript"
            for arm in cfg["arms_H"]:
                f = out / f"H_{archname}_{arm['name']}_{k}.npz"
                if f.exists():
                    continue
                t0 = time.time()
                Z, has_lit = matrix(arm, df, y, g, tr, va, X)
                params = {**cfg.get("params", {}), **arm.get("params", {})}
                P = [fit_predict(Z, y, tr, va, te, params, arm.get("pen", 0.5) if has_lit else 0.0, arm.get("model", "catboost"),
                                 42 + b + arm.get("seed_offset", 0))[0] for b in range(arm.get("bag", 1))]
                p = np.mean(P, axis=0)
                np.savez_compressed(f, test_idx=te, y=y[te], pred=p, doi=g[te])
                e = p - y[te]
                print("H", archname, arm["name"], k, f"R2={1 - (e ** 2).mean() / y[te].var():.4f}", f"{time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--protocol", required=True)
    a = ap.parse_args(); cfg = json.loads(Path(a.config).read_text())
    out = EXP / "runs" / f"{cfg['name']}_{a.protocol}"; out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps({"config": cfg, "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
                                               "data_sha256": hashlib.sha256(DEFAULT_DATA.read_bytes()).hexdigest(),
                                               "started": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
    {"R": protocol_R, "H": protocol_H}[a.protocol](cfg, out)


if __name__ == "__main__":
    main()
