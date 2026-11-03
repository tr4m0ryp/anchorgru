"""Synthetic task with a known answer: a prior that sees static features, a sequence that holds the rest.

logit P(y = 1) = static . (1, -0.5, 0.3, 0) + beta * (x_T - x_{T-5}) of channel 0, with three random-walk channels
and 20 % of the values missing at random. The prior is a logistic fit on the static features only.
"""
from __future__ import annotations

import numpy as np


def make(n: int = 20000, T: int = 24, C: int = 3, beta: float = 0.5, missing: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, T, C)).cumsum(axis=1)
    trend = X[:, -1, 0] - X[:, -6, 0]
    X[rng.random((n, T, C)) < missing] = np.nan
    static = rng.normal(size=(n, 4))
    z = static @ np.array([1.0, -0.5, 0.3, 0.0]) + beta * trend
    p_true = 1 / (1 + np.exp(-z))
    y = (rng.random(n) < p_true).astype(int)
    return {"X": X, "static": static, "y": y, "p_true": p_true}
