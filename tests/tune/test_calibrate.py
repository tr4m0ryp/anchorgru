"""Calibration: the maps, and the cross-fitted choice with its isotonic margin."""
import numpy as np

from anchorgru import Calibrator, choose_calibration


def _oof(n=600, S=3, slope=1.0, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n) * 1.5
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    P = 1 / (1 + np.exp(-(z[:, None] * slope + rng.normal(scale=0.05, size=(n, S)))))
    return P, y, np.arange(n) % 5


def test_none_is_identity_and_maps_keep_shape():
    P, y, _ = _oof()
    np.testing.assert_array_equal(Calibrator("none").fit(P, np.repeat(y, 3)).apply(P), P)
    for kind in ("platt", "isotonic"):
        c = Calibrator(kind).fit(P.ravel(), np.repeat(y, 3))
        assert c.apply(P).shape == P.shape and c.apply(P[:, 0]).shape == (len(y),)


def test_overconfident_input_gets_platt_with_slope_below_one():
    P, y, f = _oof(slope=3.0)
    cal, rep = choose_calibration(P, y, f)
    assert rep["chosen"] == "platt"
    assert cal.params()["slope"] < 0.6
    assert rep["cv_logloss"]["platt"] < rep["cv_logloss"]["none"]


def test_isotonic_needs_to_beat_platt_by_the_margin():
    P, y, f = _oof(slope=3.0)
    _, rep = choose_calibration(P, y, f, margin=10.0, kinds=("platt", "isotonic"))
    assert rep["chosen"] == "platt"


def test_rows_never_held_out_are_ignored():
    P, y, f = _oof()
    P[:10] = np.nan
    cal, rep = choose_calibration(P, y, f)
    assert np.isfinite(list(rep["cv_logloss"].values())).all()
