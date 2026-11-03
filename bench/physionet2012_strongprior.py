"""Variant of the PhysioNet 2012 benchmark built to meet the method's precondition: a strong prior that is blind to
the sequence.

Prior: LightGBM on the static admission values, the SAPS-I score and summaries (last, mean, min, max, count per
variable) of the first 24 hours, out of fold on the pool (5 folds) and fitted on the whole pool for the test set.
Sequence: hours 24-47 only (every 2nd hour), which the prior never sees. Everything else as in
physionet2012.py (pool A + B, test C, 3 repeats, iid paired SE), except the sizes (500, 2,000) and no ablations.

Run: PYTHONPATH=. python bench/physionet2012_strongprior.py
"""
from __future__ import annotations

import numpy as np

from bench.common.runner import run
from bench.common.tabular import window_summaries
from bench.physionet2012 import Bench as Base
from bench.physionet2012_gbmprior import gbm_prior

EPOCHS = {500: 100, 2000: 30}


class Bench(Base):
    name = "physionet2012_strongprior"
    description = ("in-hospital mortality; prior = LightGBM on admission values, SAPS-I and first-24 h summaries "
                   "(out of fold); sequence = hours 24-47 only; pool A+B, test C")
    sizes = (500, 2000)

    def __init__(self):
        super().__init__()
        from bench.data.physionet import load
        d = load()
        F = np.column_stack([d["static"], np.where(d["saps"] < 0, np.nan, d["saps"]),
                             window_summaries(self.X[:, :24])])
        self.X = self.X[:, 24:]
        gbm_prior(F, self.y, self.pool, self.test, self.q)

    def params(self, size):
        return dict(stride=2, epochs=EPOCHS[size], batch_size=128)

    def settings(self):
        s = super().settings()
        s.update(epochs_by_size=EPOCHS, stride=2, sequence_hours="24-47", ablations=False)
        return s


if __name__ == "__main__":
    import torch
    torch.set_num_threads(2)
    b = Bench()
    run(b, ablations=False)
