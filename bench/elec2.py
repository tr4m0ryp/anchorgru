"""Benchmark on Electricity (elec2): does the half-hourly price end above its 24 h average.

Prior: a logistic fit on the current row only (day, period and the five numeric columns, quantile-transformed),
refitted for every one-week block on the 2,000 rows that end one week before that block, for training and test
rows alike (an upstream model refreshed weekly; the series drifts, and a longer history, raw-scaled inputs or a prior
frozen over the test fold gave a worse prior). Sequence: the last 48 half-hours of the five numeric columns, every 2nd step. Forward chaining: three
test folds of 4,000 rows starting at rows 20,000 / 28,000 / 36,000, each preceded by a one-week (336-row) embargo;
the training rows are the n most recent rows before the embargo, n in {500, 2,000}; fold k uses seed k.
Rows are serially dependent: the paired SE is a moving-block bootstrap over one-week blocks (336 rows), 2,000 draws.

Run: PYTHONPATH=. python bench/elec2.py
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer

from bench.common.arms import NET_DESC
from bench.common.runner import run
from bench.common.tabular import flat_lags
from bench.data.elec2 import load

WEEK, L = 336, 48
EPOCHS = {500: 100, 2000: 30}
BATCH = 128
PRIOR_ROWS = 2000


class Bench:
    name = "elec2"
    description = ("price above its 24 h average; prior = logistic on the current row; sequence = last 48 "
                   "half-hours of 5 columns, stride 2; forward chaining, 3 test folds of 4,000 rows")
    se = ("block", WEEK)
    sizes = (500, 2000)
    starts = (20000, 28000, 36000)
    test_len = 4000

    def __init__(self):
        d = load()
        self.rows, self.y = d["rows"], d["y"]
        W = sliding_window_view(d["seq"], L, axis=0).transpose(0, 2, 1)       # (n - L + 1, L, 5)
        self.X = np.full((len(self.y), L, W.shape[2]), np.nan)
        self.X[L - 1:] = W

    def params(self, size):
        return dict(stride=2, epochs=EPOCHS[size], batch_size=BATCH)

    def settings(self):
        return {"epochs_by_size": EPOCHS, "batch_size": BATCH, "stride": 2, "window": L, "test_starts": self.starts,
                "test_len": self.test_len, "embargo_rows": WEEK, "prior_rows": PRIOR_ROWS,
                "network": NET_DESC}

    def features(self, X):
        return flat_lags(X[:, ::-1][:, ::2][:, ::-1], L // 2)        # the network's own 24 steps, flattened

    def _fit_prior(self, end):
        rows = np.arange(end - PRIOR_ROWS, end)
        lr = make_pipeline(QuantileTransformer(n_quantiles=200), LogisticRegression(max_iter=1000))
        return lr.fit(self.rows[rows], self.y[rows])

    def _prior(self, rows):
        """Per one-week block, a fit on the 2,000 rows ending one week before the block (out of sample in time,
        the same rule for training and test rows: an upstream model refreshed weekly)."""
        q = np.full(len(self.y), np.nan)
        for w in np.unique(rows // WEEK):
            r = rows[rows // WEEK == w]
            q[r] = self._fit_prior(w * WEEK - WEEK).predict_proba(self.rows[r])[:, 1]
        return q

    def splits(self):
        out = []
        for k, start in enumerate(self.starts):
            te = np.arange(start, start + self.test_len)
            cut = start - WEEK
            q = self._prior(np.concatenate([np.arange(cut - max(self.sizes), cut), te]))
            for size in self.sizes:
                tr = np.arange(cut - size, cut)
                m = int(0.8 * size)
                inner = (np.arange(0, m - L), np.arange(m, size))
                out.append(dict(name=f"n{size}_f{k}", size=size, seed=k, tr=tr, te=te, q_tr=q[tr], q_te=q[te],
                                inner=inner, keys_te=te))
        return out


if __name__ == "__main__":
    import torch
    torch.set_num_threads(2)
    run(Bench())
