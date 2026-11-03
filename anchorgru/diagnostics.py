"""Summary statistics of the learned offsets: how far, how often and how consistently the model moves its prior."""
from __future__ import annotations

import math

import numpy as np


def residual_diagnostics(R, P, valid, cap: float, at_cap: float = 0.95) -> dict:
    """Diagnostics from per-seed offsets and probabilities.

    Parameters
    ----------
    R : (n, S) mean log-odds offset of each seed's bag.
    P : (n, S) each seed's probability.
    valid : (n,) 1 where the network acts.
    cap : the model's bound on |r|.
    at_cap : an offset counts as saturated when |r| >= at_cap * cap.

    Returns a dict: ``n``, ``valid_share``, ``residual_rms`` and ``mean_offset`` (seed-mean offset over valid
    rows), ``share_at_cap`` (share of valid rows and seeds with a saturated offset; NaN for an unbounded head),
    ``seed_spread`` (mean over rows of the standard deviation of p across seeds) and ``seed_spread_offset``.
    """
    R, P = np.atleast_2d(np.asarray(R, dtype=np.float64)), np.atleast_2d(np.asarray(P, dtype=np.float64))
    v = np.asarray(valid) > 0
    r = R[v].mean(axis=1) if v.any() else np.zeros(0)
    single = R.shape[1] < 2
    return {
        "n": int(len(v)),
        "valid_share": float(v.mean()) if len(v) else float("nan"),
        "residual_rms": float(np.sqrt(np.mean(r ** 2))) if r.size else 0.0,
        "mean_offset": float(r.mean()) if r.size else 0.0,
        "share_at_cap": (float("nan") if math.isinf(cap) or not v.any()
                         else float(np.mean(np.abs(R[v]) >= at_cap * cap))),
        "seed_spread": 0.0 if single else float(np.std(P, axis=1).mean()),
        "seed_spread_offset": 0.0 if single else float(np.std(R, axis=1).mean()),
    }
