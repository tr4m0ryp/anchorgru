"""Benchmark on PhysioNet/CinC Challenge 2012: in-hospital mortality from 48 h of ICU measurements.

Prior: a logistic fit of the outcome on the SAPS-I severity score (plus a missing flag), out of fold for the
training pool, fitted on the whole pool for the test set. Sequence: hourly grid of 36 measured variables (NaN =
not measured), every 2nd hour. Train pool = sets A + B (8,000 stays), test = set C (4,000 stays). Learning curve:
training subsamples of 500 and 2,000 stays, 3 repeats each (repeat r = subsample r and seed r).
Stays are independent, so the paired SE is sd / sqrt(n) over test stays (per-example differences averaged over the
3 repeats first).

Run: PYTHONPATH=. python bench/physionet2012.py
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from bench.common.arms import NET_DESC
from bench.common.runner import run
from bench.common.tabular import window_summaries
from bench.data.physionet import load

EPOCHS = {500: 100, 2000: 30}
BATCH = 128


class Bench:
    name = "physionet2012"
    description = ("in-hospital mortality; prior = logistic(SAPS-I), out of fold; sequence = 48 h hourly grid of "
                   "36 variables, stride 2; pool A+B, test C")
    se = "iid"
    sizes = (500, 2000)
    reps = 3

    def __init__(self):
        d = load()
        self.X, self.y, saps, sets = d["X"], d["y"].astype(int), d["saps"], d["set"]
        self.pool, self.test = np.flatnonzero(sets != "c"), np.flatnonzero(sets == "c")
        f = np.column_stack([np.where(saps < 0, np.median(saps[saps >= 0]), saps), saps < 0])
        lr = LogisticRegression(max_iter=1000)
        self.q = np.empty(len(self.y))
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        self.q[self.pool] = cross_val_predict(lr, f[self.pool], self.y[self.pool], cv=cv,
                                              method="predict_proba")[:, 1]
        self.q[self.test] = lr.fit(f[self.pool], self.y[self.pool]).predict_proba(f[self.test])[:, 1]

    def params(self, size):
        return dict(stride=2, epochs=EPOCHS[size], batch_size=BATCH)

    def settings(self):
        return {"epochs_by_size": EPOCHS, "batch_size": BATCH, "stride": 2, "reps": self.reps, "n_pool": int(len(self.pool)),
                "n_test": int(len(self.test)), "test_positive_rate": float(self.y[self.test].mean()),
                "network": NET_DESC}

    def features(self, X):
        return window_summaries(X)

    def splits(self):
        out = []
        for size in self.sizes:
            for r in range(self.reps):
                rng = np.random.default_rng(1000 * r + size)
                tr = np.sort(rng.choice(self.pool, size, replace=False)) if size < len(self.pool) else self.pool
                perm = rng.permutation(len(tr))
                cut = int(0.8 * len(tr))
                out.append(dict(name=f"n{size}_r{r}", size=size, seed=r, tr=tr, te=self.test, q_tr=self.q[tr],
                                q_te=self.q[self.test], inner=(np.sort(perm[:cut]), np.sort(perm[cut:])),
                                keys_te=self.test))
        return out


if __name__ == "__main__":
    import torch
    torch.set_num_threads(2)
    run(Bench())
