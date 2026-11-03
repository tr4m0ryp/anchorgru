"""Variant of the PhysioNet 2012 benchmark: the prior is the strongest tabular model available, and the network
may only add what the hour-by-hour path holds beyond per-variable summaries.

Prior: LightGBM on the static admission values, the SAPS-I score and summaries (last, mean, min, max, count per
variable) of all 48 hours, out of fold on the pool (5 folds) and fitted on the whole pool for the test set.
Sequence: the full 48 h grid, every 2nd hour. Sizes 500 and 2,000, 3 repeats, no ablations, iid paired SE.

Run: PYTHONPATH=. python bench/physionet2012_gbmprior.py
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedKFold

from bench.common.runner import run
from bench.common.tabular import window_summaries
from bench.data.physionet import load
from bench.physionet2012 import Bench as Base


def gbm_prior(F, y, pool, test, q):
    """Out-of-fold LightGBM probabilities on the pool, a whole-pool fit for the test rows (fills q in place)."""
    kw = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.5, n_jobs=2, random_state=0, verbose=-1)
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(pool, y[pool]):
        q[pool[b]] = lgb.LGBMClassifier(**kw).fit(F[pool[a]], y[pool[a]]).predict_proba(F[pool[b]])[:, 1]
    q[test] = lgb.LGBMClassifier(**kw).fit(F[pool], y[pool]).predict_proba(F[test])[:, 1]
    return q


class Bench(Base):
    name = "physionet2012_gbmprior"
    description = ("in-hospital mortality; prior = LightGBM on admission values, SAPS-I and 48 h summaries (out of "
                   "fold); sequence = 48 h hourly grid, stride 2; pool A+B, test C")

    def __init__(self):
        super().__init__()
        d = load()
        F = np.column_stack([d["static"], np.where(d["saps"] < 0, np.nan, d["saps"]), window_summaries(self.X)])
        gbm_prior(F, self.y, self.pool, self.test, self.q)

    def settings(self):
        s = super().settings()
        s.update(ablations=False)
        return s


if __name__ == "__main__":
    import torch
    torch.set_num_threads(2)
    run(Bench(), ablations=False)
