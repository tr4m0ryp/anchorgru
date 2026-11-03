"""Out-of-fold per-seed probabilities of an anchored classifier."""
from __future__ import annotations

import numpy as np
from sklearn.base import clone


def oof_raw(estimator, X, y, prior=None, cv=None, groups=None):
    """Fit a clone of ``estimator`` (calibration off) on each training fold and predict its held-out fold.

    Returns ``(P, fold_ids)``: P (n, n_seeds) uncalibrated probabilities, NaN for rows never held out (e.g. the
    first block under forward chaining); fold_ids (n,) the fold a row was held out in, -1 if never.
    """
    from sklearn.model_selection import KFold

    X, y = np.asarray(X, dtype=np.float64), np.asarray(y)
    cv = KFold(5) if cv is None else cv
    base = clone(estimator).set_params(calibration="none")
    P = np.full((len(y), base.n_seeds), np.nan)
    fold_ids = np.full(len(y), -1)
    for k, (tr, va) in enumerate(cv.split(X, y, groups)):
        m = clone(base).fit(X[tr], y[tr], prior=None if prior is None else np.asarray(prior)[tr])
        P[va] = m.predict_proba_per_seed(X[va], prior=None if prior is None else np.asarray(prior)[va],
                                         calibrated=False)
        fold_ids[va] = k
    return P, fold_ids
