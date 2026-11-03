"""AnchoredGRUClassifier: start at the prior, shrinkage, learning, routing, persistence, determinism."""
import pickle

import numpy as np
import pytest
from sklearn.base import clone

from anchorgru import AnchoredGRUClassifier, load, save
from anchorgru._prob import log_loss
from tests.conftest import make_data

FAST = dict(hidden=4, epochs=3, n_seeds=2, n_bags=2, batch_size=64)


def test_untrained_model_is_the_prior_exactly(data):
    X, y, q = data
    clf = AnchoredGRUClassifier(**{**FAST, "epochs": 0}).fit(X, y, prior=q)
    np.testing.assert_allclose(clf.predict_proba(X, prior=q)[:, 1], q, rtol=1e-12, atol=0)
    assert np.all(clf.residual(X, prior=q) == 0)


def test_invalid_windows_return_the_prior(data):
    X, y, q = data
    X = X.copy()
    X[:20] = np.nan
    clf = AnchoredGRUClassifier(**FAST).fit(X, y, prior=q)
    assert not clf.valid(X)[:20].any()
    np.testing.assert_allclose(clf.predict_proba(X[:20], prior=q[:20])[:, 1], q[:20], rtol=1e-12)


def test_no_prior_means_the_base_rate(data):
    X, y, _ = data
    clf = AnchoredGRUClassifier(**{**FAST, "epochs": 0}).fit(X, y)
    np.testing.assert_allclose(clf.predict_proba(X)[:, 1], y.mean(), rtol=1e-9)


def test_probabilities_labels_and_classes(data):
    X, y, q = data
    labels = np.where(y == 1, "yes", "no")
    clf = AnchoredGRUClassifier(**FAST).fit(X, labels, prior=q)
    P = clf.predict_proba(X, prior=q)
    assert P.shape == (len(y), 2) and np.allclose(P.sum(1), 1)
    assert list(clf.classes_) == ["no", "yes"]
    assert set(clf.predict(X, prior=q)) <= {"no", "yes"}
    with pytest.raises(ValueError):
        clf.predict_proba(X[:, :, :1], prior=q)
    with pytest.raises(ValueError):
        clf.predict_proba(X, prior=q[:5])


def test_penalty_shrinks_the_offset_where_the_window_has_no_signal():
    X, y, q = make_data(n=300, signal=0.0, seed=2)
    kw = dict(hidden=8, epochs=60, n_seeds=2, n_bags=1, weight_decay=0.0, lr=0.02, batch_size=None)
    free = AnchoredGRUClassifier(penalty=0.0, **kw).fit(X, y, prior=q)
    shrunk = AnchoredGRUClassifier(penalty=3.0, **kw).fit(X, y, prior=q)
    assert shrunk.diagnostics_["residual_rms"] < 0.5 * free.diagnostics_["residual_rms"]
    assert shrunk.diagnostics_["seed_spread"] < free.diagnostics_["seed_spread"]


def test_moderate_penalty_still_learns_a_strong_signal():
    X, y, q = make_data(n=800, signal=1.5, missing=0.05, seed=3)
    tr, te = slice(0, 600), slice(600, None)
    clf = AnchoredGRUClassifier(hidden=8, epochs=40, n_seeds=2, n_bags=1, penalty=0.1, lr=0.02,
                                weight_decay=0.0).fit(X[tr], y[tr], prior=q[tr])
    p = clf.predict_proba(X[te], prior=q[te])[:, 1]
    assert log_loss(p, y[te]) < log_loss(q[te], y[te]) - 0.02


def test_same_seed_same_model_and_parallel_seeds_match(data):
    X, y, q = data
    a = AnchoredGRUClassifier(**FAST).fit(X, y, prior=q).predict_proba(X, prior=q)
    b = AnchoredGRUClassifier(**FAST, n_jobs=2).fit(X, y, prior=q).predict_proba(X, prior=q)
    c = AnchoredGRUClassifier(**FAST, random_state=5).fit(X, y, prior=q).predict_proba(X, prior=q)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_pickle_and_sha256_round_trip(tmp_path, data):
    X, y, q = data
    clf = AnchoredGRUClassifier(**FAST, dropout=0.3).fit(X, y, prior=q)
    p = clf.predict_proba(X, prior=q)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(clf)).predict_proba(X, prior=q), p)
    digest = save(clf, tmp_path / "m.pkl")
    np.testing.assert_array_equal(load(tmp_path / "m.pkl").predict_proba(X, prior=q), p)
    np.testing.assert_array_equal(load(tmp_path / "m.pkl", sha256=digest).predict_proba(X, prior=q), p)
    with pytest.raises(ValueError):
        load(tmp_path / "m.pkl", sha256="0" * 64)


@pytest.mark.parametrize("kind", ["platt", "isotonic", "auto"])
def test_calibration_modes(kind, data):
    X, y, q = data
    clf = AnchoredGRUClassifier(**FAST, calibration=kind).fit(X, y, prior=q)
    p = clf.predict_proba(X, prior=q)[:, 1]
    assert np.all((p > 0) & (p < 1))
    assert clf.calibration_report_["chosen"] in ("none", "platt", "isotonic")
    if kind != "auto":
        assert clf.calibration_report_["chosen"] == kind


def test_diagnostics_keys_and_ranges(data):
    X, y, q = data
    d = AnchoredGRUClassifier(**FAST, cap=0.5).fit(X, y, prior=q).diagnostics_
    assert set(d) == {"n", "valid_share", "residual_rms", "mean_offset", "share_at_cap", "seed_spread",
                      "seed_spread_offset"}
    assert 0 <= d["share_at_cap"] <= 1 and d["residual_rms"] <= 0.5 and d["n"] == len(y)


def test_sklearn_clone_params_and_errors(data):
    X, y, q = data
    clf = AnchoredGRUClassifier(penalty=0.3, hidden=4)
    assert clone(clf).get_params() == clf.get_params()
    with pytest.raises(ValueError):
        AnchoredGRUClassifier(penalty=-1, **FAST).fit(X, y, prior=q)
    with pytest.raises(ValueError):
        AnchoredGRUClassifier(**FAST).fit(X, np.arange(len(y)) % 3, prior=q)
    with pytest.raises(ValueError):
        AnchoredGRUClassifier(**FAST).fit(X, y, prior=q * 3)
