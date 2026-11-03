"""Benchmark on the synthetic known-answer task (bench/data/synthetic.py).

20,000 examples: a fixed test set of 5,000, a pool of 15,000. Prior: logistic on the static features, out of fold
on the pool, fitted on the whole pool for the test set. Learning curve n in {250, 1,000, 4,000}, 3 repeats (repeat
r = subsample r and seed r). Examples are independent: paired SE = sd / sqrt(n) over test examples (per-example
differences averaged over repeats first). The true probability is known, so the Bayes log-loss is reported too.

Run: PYTHONPATH=. python bench/synthetic.py
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from bench.common.metrics import summary
from bench.common.arms import NET_DESC
from bench.common.runner import RESULTS, run
from bench.common.tabular import flat_lags
from bench.data.synthetic import make

EPOCHS = {250: 150, 1000: 60, 4000: 20}
BATCH = 128


class Bench:
    name = "synthetic"
    description = ("known answer: logit = static . w + 0.5 * (5-step change of channel 0); prior = logistic on the "
                   "static features; 3 random-walk channels, 24 steps, 20 % missing")
    se = "iid"
    sizes = (250, 1000, 4000)
    reps = 3

    def __init__(self):
        d = make()
        self.X, self.y, self.p_true = d["X"], d["y"], d["p_true"]
        self.test, self.pool = np.arange(15000, 20000), np.arange(15000)
        lr = LogisticRegression()
        self.q = np.empty(len(self.y))
        self.q[self.pool] = cross_val_predict(lr, d["static"][self.pool], self.y[self.pool], cv=5,
                                              method="predict_proba")[:, 1]
        self.q[self.test] = lr.fit(d["static"][self.pool], self.y[self.pool]).predict_proba(
            d["static"][self.test])[:, 1]

    def params(self, size):
        return dict(epochs=EPOCHS[size], batch_size=BATCH)

    def settings(self):
        return {"epochs_by_size": EPOCHS, "batch_size": BATCH, "reps": self.reps, "n_test": 5000,
                "bayes": summary(self.y[self.test], self.p_true[self.test]),
                "network": NET_DESC}

    def features(self, X):
        return flat_lags(X, X.shape[1])

    def splits(self):
        out = []
        for size in self.sizes:
            for r in range(self.reps):
                rng = np.random.default_rng(1000 * r + size)
                tr = np.sort(rng.choice(self.pool, size, replace=False))
                perm = rng.permutation(size)
                cut = int(0.8 * size)
                out.append(dict(name=f"n{size}_r{r}", size=size, seed=r, tr=tr, te=self.test, q_tr=self.q[tr],
                                q_te=self.q[self.test], inner=(np.sort(perm[:cut]), np.sort(perm[cut:])),
                                keys_te=self.test))
        return out


if __name__ == "__main__":
    import torch
    torch.set_num_threads(2)
    run(Bench())
