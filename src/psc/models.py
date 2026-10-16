"""Estimators with validation-only model selection. Test data is never passed in."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import pandas as pd

from .data import CAT_ALL as CAT


class Winsorizer:
    """Clip numeric columns to train-set [0.5, 99.5] percentiles (linear baseline only)."""
    def fit(self, X):
        num = [c for c in X.columns if c not in CAT]
        self.lo_ = X[num].quantile(0.005); self.hi_ = X[num].quantile(0.995)
        return self

    def transform(self, X):
        Z = X.copy()
        for c in self.lo_.index:
            Z[c] = Z[c].clip(self.lo_[c], self.hi_[c])
        return Z

CATBOOST_MATCHED = dict(iterations=1500, learning_rate=0.05, depth=8, l2_leaf_reg=3.0,
                        subsample=0.8, colsample_bylevel=0.7, min_data_in_leaf=5,
                        random_seed=42, loss_function="RMSE")
ES_ROUNDS = 100
RIDGE_ALPHAS = [0.1, 1, 10, 100, 1000]


def fit_predict(name, Xtr, ytr, Xva, yva, Xte, params=None):
    """Fit on train, select/early-stop on val, predict test. Returns (pred, info)."""
    if name == "mean":
        return np.full(len(Xte), ytr.mean()), {}
    if name == "ridge":
        ct = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), [c for c in CAT if c in Xtr.columns])],
                               remainder=StandardScaler())
        w = Winsorizer().fit(Xtr)
        Xtr, Xva, Xte = w.transform(Xtr), w.transform(Xva), w.transform(Xte)
        best = None
        for a in RIDGE_ALPHAS:
            m = make_pipeline(ct, Ridge(alpha=a, solver="auto")).fit(Xtr, ytr)
            rm = float(np.sqrt(np.mean((m.predict(Xva) - yva) ** 2)))
            if best is None or rm < best[0]:
                best = (rm, a, m)
        return best[2].predict(Xte), {"alpha": best[1], "val_rmse": best[0]}
    if name == "catboost":
        from catboost import CatBoostRegressor
        p = {**CATBOOST_MATCHED, **(params or {})}
        es = p.pop("early_stopping_rounds", ES_ROUNDS)
        ci = [i for i, c in enumerate(Xtr.columns) if c in CAT]
        m = CatBoostRegressor(**p, verbose=0, thread_count=-1)
        m.fit(Xtr, ytr, cat_features=ci, eval_set=(Xva, yva),
              early_stopping_rounds=es, verbose=False)
        return m.predict(Xte), {"best_iteration": int(m.get_best_iteration()), "params": p}
    if name in ("xgb_default", "rf"):
        cats = {c: {v: i for i, v in enumerate(sorted(Xtr[c].unique()))} for c in CAT if c in Xtr.columns}

        def enc(X):
            Z = X.copy()
            for c in cats:
                Z[c] = Z[c].map(cats[c]).fillna(-1)
            return Z.values.astype(np.float32)
        if name == "xgb_default":   # library defaults (the setting used by Roberts-style pipelines); no tuning/ES
            from xgboost import XGBRegressor
            m = XGBRegressor(tree_method="hist", random_state=42, n_jobs=-1).fit(enc(Xtr), ytr)
        else:                        # Fukasawa-style random forest, 300 trees, sklearn defaults otherwise
            from sklearn.ensemble import RandomForestRegressor
            m = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                                      random_state=42).fit(enc(Xtr), ytr)
        return m.predict(enc(Xte)), {}
    if name in ("mlp", "gpr_sub"):
        w = Winsorizer().fit(Xtr)
        ct = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [c for c in CAT if c in Xtr.columns])],
                               remainder=StandardScaler())
        A = ct.fit_transform(w.transform(Xtr)); V = ct.transform(w.transform(Xva)); T = ct.transform(w.transform(Xte))
        mu, sd = ytr.mean(), ytr.std()
        if name == "gpr_sub":    # exact GPR is O(n^3): fit on a fixed random subsample of the training records
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
            n = min(3000, len(A)); i = np.random.default_rng(0).choice(len(A), n, replace=False)
            k = ConstantKernel(1.0) * RBF(length_scale=np.sqrt(A.shape[1])) + WhiteKernel(0.5)
            g = GaussianProcessRegressor(k, normalize_y=True, random_state=0).fit(A[i], ytr[i])
            return g.predict(T), {"n_fit": int(n)}
        import torch
        torch.manual_seed(0)
        f = lambda a: torch.tensor(a, dtype=torch.float32)
        At, Vt, Tt = f(A), f(V), f(T); yt = f((ytr - mu) / sd)
        net = torch.nn.Sequential(torch.nn.Linear(A.shape[1], 256), torch.nn.ReLU(), torch.nn.Dropout(0.2),
                                  torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Dropout(0.2),
                                  torch.nn.Linear(128, 1))
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
        best, bad, state = 1e9, 0, None
        for ep in range(200):
            net.train(); perm = torch.randperm(len(At))
            for b in range(0, len(At), 256):
                j = perm[b:b + 256]; opt.zero_grad()
                torch.nn.functional.mse_loss(net(At[j]).squeeze(1), yt[j]).backward(); opt.step()
            net.eval()
            with torch.no_grad():
                rm = float(((net(Vt).squeeze(1).numpy() * sd + mu - yva) ** 2).mean() ** 0.5)
            if rm < best - 1e-4:
                best, bad, state = rm, 0, {k_: v.clone() for k_, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= 15:
                    break
        net.load_state_dict(state); net.eval()
        with torch.no_grad():
            return net(Tt).squeeze(1).numpy() * sd + mu, {"epochs": ep + 1, "val_rmse": best}
    raise ValueError(name)
