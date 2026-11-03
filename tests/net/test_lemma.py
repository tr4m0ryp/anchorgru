"""Pointwise shrinkage lemma, checked numerically.

For one input with anchor b, conditional mean mu and lambda' > 0, f(r) = E BCE(y, sigmoid(b + r)) + lambda' r^2 is
strictly convex with minimiser r* solving sigmoid(b + r*) + 2 lambda' r* = mu. Then (i) r* = 0 iff sigmoid(b) = mu,
(ii) sign(r*) = sign(logit(mu) - b), (iii) |r*| < |logit(mu) - b|. Training on a sample with one shared anchor and
no usable window signal must land on that stationary point, with lambda' = lambda * n / n_valid.
"""
import numpy as np
import pytest
from scipy.optimize import brentq

from anchorgru.net import TrainConfig, bag_residuals, train_net


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def lg(p):
    return np.log(p / (1.0 - p))


def r_star(b, mu, lam):
    return brentq(lambda r: sig(b + r) - mu + 2 * lam * r, -60, 60, xtol=1e-14)


GRID = [(b, mu, lam) for b in (-3.0, -0.5, 0.0, 1.2, 4.0) for mu in (0.02, 0.3, 0.5, 0.8, 0.99)
        for lam in (1e-3, 0.1, 1.0, 10.0)]


@pytest.mark.parametrize("b,mu,lam", GRID)
def test_minimiser_sign_and_no_overshoot(b, mu, lam):
    r = r_star(b, mu, lam)
    gap = lg(mu) - b
    f = lambda t: -(mu * np.log(sig(b + t)) + (1 - mu) * np.log(1 - sig(b + t))) + lam * t * t  # noqa: E731
    assert f(r) <= min(f(r - 1e-3), f(r + 1e-3))
    if gap == 0:
        assert r == pytest.approx(0.0, abs=1e-12)
    else:
        assert np.sign(r) == np.sign(gap)
        assert abs(r) < abs(gap)


@pytest.mark.parametrize("b,lam", [(-2.0, 0.5), (0.0, 1.0), (1.5, 0.01)])
def test_minimiser_is_zero_iff_the_prior_is_calibrated(b, lam):
    assert r_star(b, sig(b), lam) == pytest.approx(0.0, abs=1e-12)
    assert abs(r_star(b, sig(b) * 0.9, lam)) > 1e-4


def test_objective_is_strictly_convex():
    b, mu, lam = 0.4, 0.7, 0.2
    r = np.linspace(-8, 8, 2001)
    f = -(mu * np.log(sig(b + r)) + (1 - mu) * np.log(1 - sig(b + r))) + lam * r * r
    assert np.all(np.diff(f, 2) > 0)


def test_shrinkage_vanishes_as_lambda_goes_to_zero():
    b, mu = -1.0, 0.6
    assert r_star(b, mu, 1e-9) == pytest.approx(lg(mu) - b, abs=1e-6)


@pytest.mark.parametrize("lam,invalid_share", [(0.5, 0.0), (0.5, 0.4), (2.0, 0.25)])
def test_training_reaches_the_stationary_point(lam, invalid_share):
    rng = np.random.default_rng(1)
    n, b = 400, -0.4
    seq = np.zeros((n, 3, 2), np.float32)
    valid = (rng.random(n) >= invalid_share).astype(float)
    y = (rng.random(n) < 0.7).astype(float)
    cfg = TrainConfig(hidden=2, cap=float("inf"), epochs=1500, lr=0.03, weight_decay=0.0,
                      batch_size=None, penalty=lam)
    net = train_net(seq, np.full(n, b), valid, y, cfg, seed=0)
    r = bag_residuals([net], seq, valid)[:, 0]
    assert np.all(r[valid == 0] == 0)
    rv = r[valid > 0]
    assert np.ptp(rv) < 1e-6
    lam_eff = lam * n / valid.sum()
    want = r_star(b, y[valid > 0].mean(), lam_eff)
    assert rv[0] == pytest.approx(want, abs=2e-3)
    assert abs(rv[0]) < abs(lg(y[valid > 0].mean()) - b)
