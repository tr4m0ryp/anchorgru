"""Turn raw multichannel windows (NaN = missing) into network inputs.

For a window x[1..T] with C channels the encoding is

* forward fill: a missing value carries the last observed one forward;
* relative values: each channel minus its last observed value, so the final step is 0 and the network sees the
  path that led to the cutoff rather than its level (steps before a channel's first observation are 0);
* observation masks: one 0 / 1 channel per value channel marking the steps that were observed;
* stride: every ``stride``-th step counted back from the last one;
* validity: an example whose share of observed steps (any channel) is below ``min_observed`` is invalid and
  gets a zero offset, so the model returns its prior there.

The result has 2 C channels: C values followed by C masks.
"""
from __future__ import annotations

import numpy as np


def as_windows(X) -> np.ndarray:
    """Coerce to a float64 array of shape (n, T, C); a 2-d input is read as one channel."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 2:
        X = X[:, :, None]
    if X.ndim != 3:
        raise ValueError(f"expected windows of shape (n, T, C) or (n, T), got {X.shape}")
    if X.shape[1] < 1 or X.shape[2] < 1:
        raise ValueError("windows need at least one step and one channel")
    return X


def forward_fill(X: np.ndarray) -> np.ndarray:
    """Forward fill along the step axis of an (n, T, C) array; leading NaNs stay NaN."""
    n, T, C = X.shape
    pos = np.where(np.isfinite(X), np.arange(T)[None, :, None], -1)
    np.maximum.accumulate(pos, axis=1, out=pos)
    filled = np.take_along_axis(X, np.maximum(pos, 0), axis=1)
    return np.where(pos >= 0, filled, np.nan)


def last_observed(X, channel: int | None = None) -> np.ndarray:
    """Last finite value per example (and channel); NaN where a channel was never observed.

    Use it to take the prior from a channel that itself holds a probability, so that the prior and the window
    come from the same point-in-time row.
    """
    last = forward_fill(as_windows(X))[:, -1, :]
    return last if channel is None else last[:, channel]


def step_indices(T: int, stride: int) -> np.ndarray:
    """Indices of the kept steps, oldest first, counted back from the last step T - 1."""
    if stride < 1:
        raise ValueError("stride must be >= 1")
    return np.arange(T - 1, -1, -stride)[::-1]


def encode(X, window: int | None = None, stride: int = 1, relative: bool = True,
           min_observed: float = 0.25, scale: np.ndarray | None = None):
    """Encode windows for the network.

    Returns ``(seq, valid)``: ``seq`` float32 of shape (n, T', 2 C) with T' = ceil(T / stride) (T = ``window``
    if given, the last ``window`` steps), and ``valid`` float64 of shape (n,) in {0, 1}. ``scale`` (C,) divides
    the value channels (see :func:`fit_scale`).
    """
    X = as_windows(X)
    if window is not None:
        if window < 1:
            raise ValueError("window must be >= 1")
        X = X[:, -window:]
    obs = np.isfinite(X)
    F = forward_fill(X)
    if relative:
        F = F - F[:, -1:, :]
    vals = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    if scale is not None:
        vals = vals / np.asarray(scale, dtype=np.float64)[None, None, :]
    valid = (obs.any(axis=2).mean(axis=1) >= min_observed) & obs.any(axis=(1, 2))
    idx = step_indices(X.shape[1], stride)
    seq = np.concatenate([vals[:, idx], obs[:, idx].astype(np.float64)], axis=2).astype(np.float32)
    return seq, valid.astype(np.float64)


def fit_scale(seq: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-channel standard deviation of the encoded values over observed steps of valid examples (1 if 0)."""
    C = seq.shape[2] // 2
    v, m = seq[valid > 0, :, :C].astype(np.float64), seq[valid > 0, :, C:] > 0
    out = np.ones(C)
    for c in range(C):
        x = v[:, :, c][m[:, :, c]]
        s = float(np.std(x)) if x.size > 1 else 0.0
        out[c] = s if np.isfinite(s) and s > 1e-12 else 1.0
    return out
