"""Tabular summaries of windows for the boosting arm."""
from __future__ import annotations

import warnings

import numpy as np


def window_summaries(X: np.ndarray) -> np.ndarray:
    """Per channel: last observed value, mean, min, max and number of observed steps -> (n, 5 C)."""
    from anchorgru.windows.encode import last_observed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        parts = [last_observed(X), np.nanmean(X, axis=1), np.nanmin(X, axis=1), np.nanmax(X, axis=1),
                 np.isfinite(X).sum(axis=1).astype(float)]
    return np.concatenate(parts, axis=1)


def flat_lags(X: np.ndarray, k: int) -> np.ndarray:
    """The last k steps of every channel, flattened -> (n, k C)."""
    return X[:, -k:, :].reshape(len(X), -1)
