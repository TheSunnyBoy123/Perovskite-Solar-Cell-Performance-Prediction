#!/usr/bin/env python3
"""Matched t1/t2/t3 ablation under DOI-grouped and record-level protocols.

Run from ml_paper/:  python scripts/run_experiment.py --config configs/ablation.json
Resumable: completed (protocol, arch, tier, model, split) cells are skipped.
Outputs: results/<name>/runs.jsonl, results/<name>/preds/*.npz, results/<name>/meta.json
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from psc.data import DEFAULT_DATA, TARGET, load_dataset  # noqa: E402
from psc.evaluate import metrics  # noqa: E402
from psc.features import TrainFitPreprocessor, base_frame  # noqa: E402
from psc.models import fit_predict  # noqa: E402
from psc.splits import make_splits  # noqa: E402


def sh(cmd):
    try:
        return subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true", help="2 splits, 300 iterations")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    name = cfg["name"] + ("_smoke" if a.smoke else "")
    out = ROOT / "results" / name
    (out / "preds").mkdir(parents=True, exist_ok=True)
    runs_f = out / "runs.jsonl"
    done = set()
    if runs_f.exists():
        for line in runs_f.read_text().splitlines():
            r = json.loads(line)
            done.add((r["protocol"], r["arch"], r["tier"], r["model"], r["split"]))
    n_splits = 2 if a.smoke else cfg["n_splits"]  # temporal protocol always uses its 3 forward windows
    data_path = ROOT / cfg.get("data", str(DEFAULT_DATA.relative_to(ROOT)))
    meta = {"config": cfg, "config_sha256": hashlib.sha256(
                json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
            "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "git_head": sh(["git", "rev-parse", "HEAD"]),
            "git_dirty": bool(sh(["git", "status", "--porcelain", "--", "."])),
            "python": sys.version.split()[0]}
    import catboost, sklearn
    meta.update(catboost=catboost.__version__, sklearn=sklearn.__version__)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    for arch in cfg["archs"]:
        df = load_dataset(data_path, arch, subset=cfg.get("subset"), partition=cfg.get("partition"))
        y = df[TARGET].values.astype(float)
        groups = df["doi_group"].values
        print(f"[{arch}] n={len(df)} n_doi_groups={len(set(groups))}", flush=True)
        for protocol in cfg["protocols"]:
            g_split = df["author_group"].values if protocol == "author" else groups
            seed_p = cfg["seed"]
            n_s = 3 if protocol == "temporal" else n_splits
            if a.smoke:
                n_s = min(n_s, 2)
            splits = list(make_splits(g_split, protocol, n_s, cfg["test_size"], cfg["val_size"],
                                      cfg["seed"], dates=df["pub_date"].values if "pub_date" in df else None))
            for tier in cfg["tiers"]:
                Xraw = base_frame(df, tier)
                for k, (tr, va, te) in enumerate(splits, 1):
                    pre = TrainFitPreprocessor().fit(Xraw.iloc[tr])
                    Xtr, Xva, Xte = (pre.transform(Xraw.iloc[i]) for i in (tr, va, te))
                    for model in cfg["models"]:
                        key = (protocol, arch, tier, model, k)
                        if key in done:
                            continue
                        params = dict(cfg.get("catboost", {}))
                        if a.smoke:
                            params["iterations"] = 300
                        t0 = time.time()
                        p, info = fit_predict(model, Xtr, y[tr], Xva, y[va], Xte, params)
                        m = metrics(y[te], p)
                        rec = dict(protocol=protocol, arch=arch, tier=tier, model=model,
                                   split=k, n_train=len(tr), n_val=len(va), n_test=len(te),
                                   n_doi_train=len(set(groups[tr])), n_doi_val=len(set(groups[va])),
                                   n_doi_test=len(set(groups[te])),
                                   doi_overlap_test_train=len(set(groups[te]) & set(groups[tr])),
                                   doi_overlap_val_train=len(set(groups[va]) & set(groups[tr])),
                                   doi_overlap_test_val=len(set(groups[te]) & set(groups[va])),
                                   author_overlap_test_train=(len(set(df["author_group"].values[te]) & set(df["author_group"].values[tr]))
                                                              if "author_group" in df else None),
                                   test_date_min=(str(df["pub_date"].iloc[te].min().date()) if "pub_date" in df else None),
                                   train_date_max=(str(df["pub_date"].iloc[tr].max().date()) if "pub_date" in df else None),
                                   n_features=Xtr.shape[1], runtime_s=round(time.time() - t0, 1),
                                   info=info, **m)
                        np.savez_compressed(out / "preds" / f"{protocol}_{arch}_{tier}_{model}_{k}.npz",
                                            test_idx=te, y=y[te], pred=p, train_idx=tr, val_idx=va)
                        with open(runs_f, "a") as f:
                            f.write(json.dumps(rec) + "\n")
                        print(f"{protocol:6s} {arch:7s} {tier} {model:8s} s{k} "
                              f"R2={m['r2']:.4f} MAE={m['mae']:.3f} nf={Xtr.shape[1]} "
                              f"[{rec['runtime_s']}s]", flush=True)


if __name__ == "__main__":
    main()
