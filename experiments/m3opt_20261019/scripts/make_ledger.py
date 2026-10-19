#!/usr/bin/env python3
"""Collect every fit of every run into experiment_ledger.csv (one row per arm x seed x fold, incl. unsuccessful arms).

Columns: experiment_id, run, hypothesis, arm, arm_spec (JSON), seed, fold, split_id, config_sha256, data_sha256, git_head,
n_train/n_val/n_test, n_features, n_lit, best_iter, r2/mae/rmse (all test records of the fold), runtime_s, pred_path.
"""
import json
from pathlib import Path
import pandas as pd

EXP = Path(__file__).resolve().parents[1]


def main():
    rows = []
    for d in sorted((EXP / "runs").iterdir()):
        led, meta = d / "ledger.jsonl", d / "meta.json"
        if not led.exists() or not meta.exists() or d.name.endswith("_smoke"):
            continue
        m = json.loads(meta.read_text()); cfg = m["config"]
        spec = {a["name"]: a for a in cfg.get("arms", [])}
        for l in open(led):
            r = json.loads(l)
            rows.append(dict(experiment_id=f"{d.name}:{r['arm']}", run=d.name, hypothesis=cfg.get("hypothesis", ""),
                             arm=r["arm"], arm_spec=json.dumps(spec.get(r["arm"], {}), sort_keys=True),
                             seed=r["seed"], fold=r["fold"],
                             split_id=f"{cfg.get('partition', 'v4dev')}|kfold{cfg.get('folds', 5)}|seed{r['seed']}|fold{r['fold']}",
                             config_sha256=m["config_sha256"], data_sha256=m["data_sha256"], git_head=m.get("git_head"),
                             n_train=r["n_train"], n_val=r["n_val"], n_test=r["n_test"], n_features=r["n_features"],
                             n_lit=r["n_lit"], best_iter=json.dumps(r["best_iter"]), r2=r["r2"], mae=r["mae"], rmse=r["rmse"],
                             runtime_s=r["runtime_s"], pred_path=f"runs/{d.name}/preds/{r['arm']}_s{r['seed']}_f{r['fold']}.npz"))
    df = pd.DataFrame(rows).drop_duplicates(["run", "arm", "seed", "fold"], keep="last")
    df.to_csv(EXP / "experiment_ledger.csv", index=False)
    print(len(df), "fits;", df.groupby("run").arm.nunique().to_dict())


if __name__ == "__main__":
    main()
