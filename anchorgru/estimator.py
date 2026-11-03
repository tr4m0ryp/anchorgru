"""AnchoredGRUClassifier: a recurrent network that nudges a prior probability, scikit-learn style."""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

from ._prob import expit, logit
from .calibrate import Calibrator, choose_calibration
from .diagnostics import residual_diagnostics
from .net import TrainConfig, bag_residuals, fit_bag
from .windows.encode import as_windows, encode, fit_scale, last_observed


def _fit_seed(seq, anchor, valid, y, cfg, seed, n_bags, threads):
    import torch
    torch.set_num_threads(threads)
    return fit_bag(seq, anchor, valid, y, cfg, seed, n_bags)


class AnchoredGRUClassifier(ClassifierMixin, BaseEstimator):
    """Binary classifier that corrects a prior probability with a small recurrent network.

    For example i with prior q_i and window x_i the prediction is ``sigmoid(logit(q_i) + r_i)`` with
    ``r_i = cap * tanh(head(GRU(x_i))) * valid_i``. The head starts at zero, so an untrained model returns the
    prior exactly. Training minimises cross-entropy plus ``penalty * sum(r^2) / n_valid``, a shrinkage of every
    offset toward zero (that is, of every prediction toward its prior) that keeps the network from drifting where
    the window carries no signal. ``penalty=0`` is the plain unpenalised residual.

    X is an array (n, T, C) of windows, oldest step first, NaN = missing (a 2-d (n, T) array is one channel).
    ``prior`` (n,) is the probability of the positive class (``classes_[1]``) from any upstream source; None uses
    the training base rate (then the model is a plain recurrent classifier with a bounded head).

    The prior can also travel inside X: with ``prior_channel=k`` and no ``prior`` argument, the prior of each
    example is the last observed value of channel k (for instance a channel that is NaN except for the prior at
    the last step). That keeps the prior and the window on the same point-in-time row and makes the estimator
    work unchanged in ``cross_val_score``, ``GridSearchCV`` and pipelines, which do not route predict-time
    metadata. ``prior`` as a fit parameter is also routable (``set_fit_request(prior=True)``).

    Parameters
    ----------
    hidden, cell, layers, dropout : recurrent encoder (cell "gru" or "lstm").
    cap : bound on |r| in log-odds; ``float("inf")`` removes the tanh.
    penalty : output-space shrinkage strength lambda (>= 0).
    epochs, lr, weight_decay, batch_size, label_smoothing : AdamW training (batch_size None = full batch).
    n_bags : networks per seed, each on its own bootstrap resample when > 1.
    n_seeds : independent seeds; the prediction averages them.
    window, stride : use the last ``window`` steps (None = all), every ``stride``-th step back from the last.
    min_observed : an example with a smaller share of observed steps is invalid and returns its prior.
    relative : re-express each channel relative to its last observed value.
    standardize : divide each value channel by its training standard deviation.
    zero_init : start the head at zero (False is an ablation).
    calibration : "none", "platt", "isotonic" or "auto" (chosen by cross-fitting out-of-fold predictions).
    cv : splitter for the out-of-fold predictions behind calibration (default 5-fold, unshuffled).
    isotonic_margin : "auto" keeps isotonic only if it beats Platt by this many nats.
    prior_channel : channel of X whose last observed value is the prior when ``prior`` is not given (None = off;
        an example whose channel was never observed falls back to the training base rate).
    prior_eps : the prior is clipped to [prior_eps, 1 - prior_eps] before the logit.
    random_state : base seed; seed s uses ``random_state * 1000 + s``.
    n_jobs : parallel seeds (joblib).
    torch_threads : torch intra-op threads per worker.
    """

    def __init__(self, hidden=16, cell="gru", layers=1, cap=2.0, penalty=1.0, epochs=100, lr=1e-2,
                 weight_decay=0.1, batch_size=64, dropout=0.0, label_smoothing=0.0, n_bags=3, n_seeds=5,
                 window=None, stride=1, min_observed=0.25, relative=True, standardize=True, zero_init=True,
                 calibration="none", cv=None, isotonic_margin=0.002, prior_channel=None, prior_eps=1e-6, random_state=0,
                 n_jobs=None, torch_threads=1):
        self.hidden, self.cell, self.layers, self.cap, self.penalty = hidden, cell, layers, cap, penalty
        self.epochs, self.lr, self.weight_decay, self.batch_size = epochs, lr, weight_decay, batch_size
        self.dropout, self.label_smoothing, self.n_bags, self.n_seeds = dropout, label_smoothing, n_bags, n_seeds
        self.window, self.stride, self.min_observed, self.relative = window, stride, min_observed, relative
        self.standardize, self.zero_init, self.calibration, self.cv = standardize, zero_init, calibration, cv
        self.isotonic_margin, self.prior_channel, self.prior_eps = isotonic_margin, prior_channel, prior_eps
        self.random_state = random_state
        self.n_jobs, self.torch_threads = n_jobs, torch_threads

    # ----- inputs -------------------------------------------------------------------------------------------
    def _config(self) -> TrainConfig:
        return TrainConfig(hidden=self.hidden, cell=self.cell, layers=self.layers, dropout=self.dropout,
                           cap=self.cap, zero_init=self.zero_init, epochs=self.epochs, lr=self.lr,
                           weight_decay=self.weight_decay, batch_size=self.batch_size,
                           label_smoothing=self.label_smoothing, penalty=self.penalty)

    def _anchor(self, prior, n: int, X=None) -> np.ndarray:
        if prior is None and self.prior_channel is not None and X is not None:
            q = last_observed(X, self.prior_channel)
            if np.any(np.isfinite(q) & ((q < 0) | (q > 1))):
                raise ValueError("the prior channel must hold probabilities in [0, 1]")
            return np.where(np.isfinite(q), logit(np.nan_to_num(q, nan=0.5), self.prior_eps), self.base_logit_)
        if prior is None:
            return np.full(n, self.base_logit_)
        prior = np.asarray(prior, dtype=np.float64).ravel()
        if prior.shape != (n,):
            raise ValueError(f"prior must have shape ({n},), got {prior.shape}")
        if not np.all((prior >= 0) & (prior <= 1)):
            raise ValueError("prior must hold probabilities in [0, 1]")
        return logit(prior, self.prior_eps)

    def _encode(self, X):
        X = as_windows(X)
        if X.shape[2] != self.n_channels_:
            raise ValueError(f"X has {X.shape[2]} channels, the model was fitted on {self.n_channels_}")
        return encode(X, self.window, self.stride, self.relative, self.min_observed, self.scale_)

    def _seed(self, s: int) -> int:
        return int(self.random_state or 0) * 1000 + s

    # ----- fitting ------------------------------------------------------------------------------------------
    def fit(self, X, y, prior=None, groups=None):
        """Fit ``n_seeds`` bags of networks. ``groups`` is passed to ``cv`` when calibration needs it."""
        if self.penalty < 0:
            raise ValueError("penalty must be >= 0")
        if self.calibration not in ("none", "platt", "isotonic", "auto"):
            raise ValueError(f"unknown calibration {self.calibration!r}")
        X = as_windows(X)
        le = LabelEncoder().fit(y)
        self.classes_ = le.classes_
        if len(self.classes_) != 2:
            raise ValueError("AnchoredGRUClassifier is a binary classifier")
        yb = le.transform(y).astype(np.float64)
        if len(yb) != len(X):
            raise ValueError("X and y have different lengths")
        self.n_channels_ = X.shape[2]
        self.base_logit_ = float(logit(yb.mean(), self.prior_eps))
        anchor = self._anchor(prior, len(yb), X)
        self.scale_ = None
        seq, valid = self._encode(X)
        if self.standardize:
            self.scale_ = fit_scale(seq, valid)
            seq, valid = self._encode(X)
        cfg = self._config()
        bags = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_seed)(seq, anchor, valid, yb, cfg, self._seed(s), self.n_bags, self.torch_threads)
            for s in range(self.n_seeds))
        self.nets_ = list(bags)
        self.calibrator_, self.calibration_report_ = Calibrator("none"), {"chosen": "none"}
        if self.calibration != "none":
            self._fit_calibration(X, yb, prior, groups)
        R, P = self._per_seed(seq, anchor, valid)
        self.diagnostics_ = residual_diagnostics(R, self.calibrator_.apply(P), valid, self.cap)
        return self

    def _fit_calibration(self, X, yb, prior, groups):
        from .tune.oof import oof_raw
        oof, fold_ids = oof_raw(self, X, yb, prior, self.cv, groups)
        if self.calibration == "auto":
            self.calibrator_, self.calibration_report_ = choose_calibration(oof, yb, fold_ids,
                                                                            self.isotonic_margin)
        else:
            ok = np.isfinite(oof).all(axis=1)
            self.calibrator_ = Calibrator(self.calibration).fit(oof[ok].ravel(), np.repeat(yb[ok], oof.shape[1]))
            self.calibration_report_ = {"chosen": self.calibration}

    # ----- prediction ---------------------------------------------------------------------------------------
    def _per_seed(self, seq, anchor, valid):
        """(R, P): mean offset per seed and uncalibrated probability per seed, both (n, n_seeds)."""
        Rs, Ps = [], []
        for nets in self.nets_:
            r = bag_residuals(nets, seq, valid)
            Rs.append(r.mean(axis=1))
            Ps.append(expit(anchor[:, None] + r).mean(axis=1))
        return np.column_stack(Rs), np.column_stack(Ps)

    def predict_proba_per_seed(self, X, prior=None, calibrated: bool = True) -> np.ndarray:
        """(n, n_seeds) probability of ``classes_[1]`` from each seed's bag."""
        check_is_fitted(self, "nets_")
        seq, valid = self._encode(X)
        _, P = self._per_seed(seq, self._anchor(prior, len(seq), X), valid)
        return self.calibrator_.apply(P) if calibrated else P

    def predict_proba(self, X, prior=None) -> np.ndarray:
        """(n, 2) class probabilities; column 1 is the seed mean of the (calibrated) bag probabilities."""
        p = self.predict_proba_per_seed(X, prior).mean(axis=1)
        return np.column_stack([1.0 - p, p])

    def predict(self, X, prior=None) -> np.ndarray:
        """Class labels at the 0.5 threshold."""
        return self.classes_[(self.predict_proba(X, prior)[:, 1] > 0.5).astype(int)]

    def residual(self, X, prior=None) -> np.ndarray:
        """(n,) log-odds offsets r averaged over seeds and bags (0 where the window is invalid)."""
        check_is_fitted(self, "nets_")
        seq, valid = self._encode(X)
        R, _ = self._per_seed(seq, self._anchor(prior, len(seq), X), valid)
        return R.mean(axis=1)

    def valid(self, X) -> np.ndarray:
        """(n,) True where the window has enough observed steps for the network to act."""
        check_is_fitted(self, "nets_")
        return self._encode(X)[1] > 0
