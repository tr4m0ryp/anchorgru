"""Small probability helpers shared across the package."""
from __future__ import annotations

import numpy as np


def logit(p, eps: float = 1e-6) -> np.ndarray:
    """log(p / (1 - p)) after clipping p to [eps, 1 - eps]."""
    q = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(q) - np.log1p(-q)


def expit(z) -> np.ndarray:
    """The logistic function, computed stably in float64."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def log_loss(p, y, eps: float = 1e-12) -> float:
    """Mean binary cross-entropy of probabilities p against 0 / 1 labels y."""
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    y = np.asarray(y, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))
