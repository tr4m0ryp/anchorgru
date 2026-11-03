import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)


def make_data(n=240, T=12, C=2, missing=0.2, signal=0.0, seed=0):
    """Random-walk windows, a prior that sees a static score only, and labels that may depend on the window."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, T, C)).cumsum(axis=1)
    slope = X[:, -1, 0] - X[:, -4, 0]
    X[rng.random(X.shape) < missing] = np.nan
    z = rng.normal(size=n)
    prior = 1.0 / (1.0 + np.exp(-z))
    logit = z + signal * slope
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return X, y, prior


@pytest.fixture
def data():
    return make_data()
