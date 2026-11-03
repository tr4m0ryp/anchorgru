"""Post-hoc calibration of out-of-fold probabilities: none, Platt or isotonic, chosen by cross-fitting.

Calibrators are fitted on out-of-fold predictions only, never on the training rows a network has seen.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ._prob import log_loss, logit

KINDS = ("none", "platt", "isotonic")


class Calibrator:
    """One calibration map applied column by column to an (n, S) array of per-seed probabilities."""

    def __init__(self, kind: str = "none"):
        if kind not in KINDS:
            raise ValueError(f"calibration must be one of {KINDS}, got {kind!r}")
        self.kind, self.model = kind, None

    def fit(self, p, y) -> "Calibrator":
        p, y = np.asarray(p, dtype=np.float64).ravel(), np.asarray(y).ravel()
        if self.kind == "platt":
            self.model = LogisticRegression(C=1e4, max_iter=1000).fit(logit(p, 1e-4)[:, None], y)
        elif self.kind == "isotonic":
            self.model = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip").fit(p, y)
        return self

    def _one(self, p: np.ndarray) -> np.ndarray:
        if self.kind == "platt":
            return self.model.predict_proba(logit(p, 1e-4)[:, None])[:, 1]
        if self.kind == "isotonic":
            return self.model.predict(p)
        return p

    def apply(self, P) -> np.ndarray:
        """Calibrate every column of an (n, S) array (or a 1-d array)."""
        P = np.asarray(P, dtype=np.float64)
        if P.ndim == 1:
            return self._one(P)
        return np.column_stack([self._one(P[:, j]) for j in range(P.shape[1])])

    def params(self) -> dict:
        if self.kind == "platt":
            return {"kind": "platt", "slope": float(self.model.coef_[0, 0]),
                    "intercept": float(self.model.intercept_[0])}
        return {"kind": self.kind}


def choose_calibration(oof, y, fold_ids, margin: float = 0.002, kinds=KINDS):
    """Pick and fit a calibrator from out-of-fold probabilities.

    Parameters
    ----------
    oof : (n, S) out-of-fold probabilities, one column per seed (NaN rows are ignored).
    y : (n,) 0 / 1 labels.
    fold_ids : (n,) the fold each row was held out in. Each candidate is scored by leave-one-fold-out log-loss
        of the seed-averaged calibrated probability (a calibrator is fitted on the pooled seeds).
    margin : isotonic is a step function; it is kept only if it beats Platt by more than ``margin`` nats.

    Returns (fitted Calibrator, report dict with the cross-fitted log-loss of each candidate and the choice).
    """
    oof = np.asarray(oof, dtype=np.float64)
    oof = oof[:, None] if oof.ndim == 1 else oof
    y, fold_ids = np.asarray(y), np.asarray(fold_ids)
    ok = np.isfinite(oof).all(axis=1)
    oof, y, fold_ids = oof[ok], y[ok], fold_ids[ok]
    S = oof.shape[1]
    scores = {}
    for kind in kinds:
        p = np.zeros(len(y))
        for f in np.unique(fold_ids):
            tr, va = fold_ids != f, fold_ids == f
            c = Calibrator(kind).fit(oof[tr].ravel(), np.repeat(y[tr], S))
            p[va] = c.apply(oof[va]).mean(axis=1)
        scores[kind] = log_loss(p, y)
    best = min(scores, key=scores.get)
    if best == "isotonic" and "platt" in scores and scores["platt"] - scores["isotonic"] < margin:
        best = "platt"
    return Calibrator(best).fit(oof.ravel(), np.repeat(y, S)), {"cv_logloss": scores, "chosen": best}
