#!/usr/bin/env python3
"""OOF calibration plots (self-contained; run from ml_paper/).

Reads predictions/oof_{arch}.csv (pooled held-out predictions, every record
predicted exactly once) and writes figures/oof_{arch}.png:
  (a) predicted vs experimental PCE (pooled, all 5 folds)
  (b) residual vs experimental PCE
  (c) residual histogram
Plus high-PCE-regime error table on stdout.

Usage: python3 pipeline/plot_oof.py {overall|pin|nip}
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCH = sys.argv[1].lower()
assert ARCH in ("overall", "pin", "nip")

d = pd.read_csv(os.path.join(ROOT, "predictions", f"oof_{ARCH}.csv"))
y, p = d["true_pce"].values.astype(float), d["pred_pce"].values.astype(float)
r = p - y
print(f"[{ARCH}] pooled OOF n={len(d)} folds={sorted(d.fold.unique())}")
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
pooled = {"r2": float(r2_score(y, p)), "mae": float(mean_absolute_error(y, p)),
          "rmse": float(np.sqrt(mean_squared_error(y, p)))}
print(f"[{ARCH}] pooled R2={pooled['r2']:.4f} MAE={pooled['mae']:.3f} "
      f"RMSE={pooled['rmse']:.3f} | unique_records={d.ref_id.nunique()} "
      f"(test sets overlap across runs by design)")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 4.2))
a1.scatter(y, p, s=4, alpha=0.25)
lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
a1.plot([lo, hi], [lo, hi], "k--", lw=1)
a1.set_xlabel("Experimental PCE (%)")
a1.set_ylabel("Predicted PCE (%)")
a2.scatter(y, r, s=4, alpha=0.25)
a2.axhline(0, color="k", lw=1)
a2.set_xlabel("Experimental PCE (%)")
a2.set_ylabel("Residual = Pred − Exp (%)")
a3.hist(r, bins=60, color="#4c72b0", edgecolor="white", linewidth=0.4)
a3.axvline(0, color="k", ls="--", lw=1)
a3.set_xlabel("Residual (%)")
a3.set_ylabel("Count")
for ax in (a1, a2, a3):
    ax.grid(alpha=0.3)
from sklearn.metrics import mean_absolute_error, r2_score
fig.suptitle(f"{ARCH} CatBoost pooled OOF — {len(d):,} held-out predictions "
             f"(5 DOI-grouped runs, test sets overlap) "
             f"R²={pooled['r2']:.3f} MAE={pooled['mae']:.2f}%", fontsize=10)
fig.tight_layout()
path = os.path.join(ROOT, "figures", f"oof_{ARCH}.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print("saved", path)
for lo_, hi_, name in [(0, 10, "<10"), (10, 15, "10–15"), (15, 20, "15–20"),
                       (20, 100, "≥20")]:
    m_ = (y >= lo_) & (y < hi_)
    if m_.sum():
        print(f"  exp {name}: n={m_.sum()} MAE={np.abs(r[m_]).mean():.2f} "
              f"mean_resid={r[m_].mean():+.2f}")
