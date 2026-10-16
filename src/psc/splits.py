"""Record-level and DOI-grouped train/val/test splits."""
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit


def make_splits(groups, protocol, n_splits=5, test_size=0.2, val_size=0.15,
                seed=42, dates=None):
    """Yield (train, val, test) integer index arrays.

    protocol='doi'   : test and val are DOI-disjoint from each other and from
                       train (groups = normalised DOI; missing DOIs form one
                       explicit block).
    protocol='random': record-level shuffle with identical sizes.
    Repeated holdouts: test sets of different repeats may overlap.
    """
    n = len(groups)
    idx = np.arange(n)
    if protocol == "author":  # same mechanics as "doi" but groups are lead authors (pass author keys as `groups`)
        protocol = "doi"
    if protocol == "temporal":
        yield from _temporal(groups, dates, n_splits, val_size)
        return
    if protocol == "kfold":   # disjoint grouped K-fold: the n_splits test folds partition the publications
        rng = np.random.default_rng(seed)
        ug = np.unique(np.asarray(groups)); rng.shuffle(ug)
        fold_of = {g: i % n_splits for i, g in enumerate(ug)}
        f = np.array([fold_of[g] for g in np.asarray(groups)])
        for k in range(n_splits):
            te = np.where(f == k)[0]; tr = np.where(f != k)[0]
            inner = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed + k)
            a, b = next(inner.split(tr, groups=np.asarray(groups)[tr]))
            yield tr[a], tr[b], te
        return
    if protocol == "doi":
        outer = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
        for k, (tr, te) in enumerate(outer.split(idx, groups=groups)):
            inner = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed + k)
            a, b = next(inner.split(tr, groups=np.asarray(groups)[tr]))
            yield tr[a], tr[b], te
    elif protocol == "random":
        outer = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
        for k, (tr, te) in enumerate(outer.split(idx)):
            inner = ShuffleSplit(n_splits=1, test_size=val_size, random_state=seed + k)
            a, b = next(inner.split(tr))
            yield tr[a], tr[b], te
    else:
        raise ValueError(f"unknown protocol {protocol!r}")


TEMPORAL_WINDOWS = [(0.55, 0.70), (0.70, 0.85), (0.85, 1.00)]


def _temporal(groups, dates, n_splits, val_size):
    """Forward-in-time splits. Publications (DOI groups) are dated by their earliest record.
    Split k tests on publications whose date quantile lies in TEMPORAL_WINDOWS[k]; training uses
    strictly earlier publications; the validation set is the most recent `val_size` of those
    earlier publications (so validation is also later than training). All three sets are
    DOI-disjoint by construction. Records without a DOI are dated individually and kept together."""
    import pandas as pd
    g = pd.Series(np.asarray(groups)); d = pd.Series(pd.to_datetime(np.asarray(dates)))
    gd = d.groupby(g.values).min()                    # publication date per group
    order = gd.sort_values(kind="stable")
    n = len(order); ranks = pd.Series(np.arange(n) / max(n - 1, 1), index=order.index)
    gr = g.map(ranks).values
    for lo, hi in TEMPORAL_WINDOWS[:n_splits]:
        te = np.where((gr > lo) & (gr <= hi))[0] if lo > 0 else np.where(gr <= hi)[0]
        pre = np.where(gr <= lo)[0]
        cut = np.quantile(gr[pre], 1 - val_size)
        yield pre[gr[pre] <= cut], pre[gr[pre] > cut], te
