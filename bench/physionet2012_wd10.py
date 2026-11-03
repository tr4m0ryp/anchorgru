"""Variant of the PhysioNet 2012 benchmark: every network (anchored and unanchored) trains with ten times the
benchmark network's weight decay (2.0 instead of 0.2), because the main run's weight-decay ablation was its best network.
Prior, sequence, sizes and repeats as in physionet2012.py; no ablations.

Run: PYTHONPATH=. python bench/physionet2012_wd10.py
"""
from __future__ import annotations

from bench.common.runner import run
from bench.physionet2012 import Bench as Base


class Bench(Base):
    name = "physionet2012_wd10"
    description = "as physionet2012, every network with weight decay 2.0 (10x the benchmark network's 0.2)"

    def params(self, size):
        return dict(super().params(size), weight_decay=2.0)

    def settings(self):
        s = super().settings()
        s.update(weight_decay=2.0, ablations=False)
        return s


if __name__ == "__main__":
    import torch
    torch.set_num_threads(2)
    run(Bench(), ablations=False)
