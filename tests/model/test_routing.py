"""The prior through scikit-learn tooling: routed into fit, or carried inside X for scoring."""
import numpy as np
import pytest
import sklearn
from sklearn.metrics import log_loss, make_scorer
from sklearn.model_selection import KFold, cross_val_score, cross_validate

from anchorgru import AnchoredGRUClassifier

FAST = dict(hidden=4, epochs=2, n_seeds=1, n_bags=1)


def _with_prior_channel(X, q):
    extra = np.full(X.shape[:2] + (1,), np.nan)
    extra[:, -1, 0] = q
    return np.concatenate([X, extra], axis=2)


def test_fit_prior_is_routed_by_cross_validate(data):
    X, y, q = data
    cv = KFold(3)
    with sklearn.config_context(enable_metadata_routing=True):
        clf = AnchoredGRUClassifier(**FAST).set_fit_request(prior=True)
        res = cross_validate(clf, X, y, cv=cv, params={"prior": q}, return_estimator=True,
                             scoring=make_scorer(log_loss, response_method="predict_proba"))
    for est, (tr, te) in zip(res["estimator"], cv.split(X)):
        ref = AnchoredGRUClassifier(**FAST).fit(X[tr], y[tr], prior=q[tr])
        np.testing.assert_array_equal(est.predict_proba(X[te], prior=q[te]), ref.predict_proba(X[te], prior=q[te]))


def test_prior_channel_equals_an_explicit_prior(data):
    X, y, q = data
    Xq = _with_prior_channel(X, q)
    a = AnchoredGRUClassifier(**FAST, prior_channel=2).fit(Xq, y)
    b = AnchoredGRUClassifier(**FAST).fit(Xq, y, prior=q)
    np.testing.assert_array_equal(a.predict_proba(Xq), b.predict_proba(Xq, prior=q))


def test_prior_channel_scores_the_prior_in_cross_val_score(data):
    X, y, q = data
    Xq = _with_prior_channel(X, q)
    cv = KFold(3)
    clf = AnchoredGRUClassifier(**{**FAST, "epochs": 0}, prior_channel=2)
    got = cross_val_score(clf, Xq, y, cv=cv, scoring="neg_log_loss")
    want = [-log_loss(y[te], q[te]) for _, te in cv.split(Xq)]
    np.testing.assert_allclose(got, want, rtol=1e-10)


def test_prior_channel_rejects_non_probabilities(data):
    X, y, q = data
    with pytest.raises(ValueError):
        AnchoredGRUClassifier(**FAST, prior_channel=0).fit(X, y)
