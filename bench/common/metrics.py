"""Per-arm metrics and paired per-example differences with their standard errors."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

EPS = 1e-12


def example_loss(y, p):
    """Per-example log-loss (nats) and squared error."""
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    y = np.asarray(y, dtype=np.float64)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p)), (p - y) ** 2


def ece(y, p, bins: int = 15) -> float:
    """Expected calibration error over equal-mass bins."""
    order = np.argsort(p, kind="stable")
    err = 0.0
    for part in np.array_split(order, bins):
        if part.size:
            err += part.size * abs(float(np.mean(p[part]) - np.mean(y[part])))
    return err / len(p)


def summary(y, p) -> dict:
    ll, br = example_loss(y, p)
    auc = roc_auc_score(y, p) if 0 < y.mean() < 1 else float("nan")
    return {"log_loss": float(ll.mean()), "brier": float(br.mean()), "ece": ece(y, p), "auc": float(auc)}


def iid_se(d) -> dict:
    """Mean and SE of independent per-example differences."""
    d = np.asarray(d, dtype=np.float64)
    return {"mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))), "n": int(len(d)),
            "se_method": "iid per example: sd / sqrt(n)"}


def block_bootstrap_se(d, block_len: int, draws: int = 2000, seed: int = 0) -> dict:
    """Moving-block bootstrap of the mean of a time-ordered series of differences."""
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    b = max(1, min(block_len, n))
    k = int(np.ceil(n / b))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - b + 1, size=(draws, k))
    csum = np.concatenate([[0.0], np.cumsum(d)])
    sums = (csum[starts + b] - csum[starts]).sum(axis=1)
    means = sums / (k * b)
    return {"mean": float(d.mean()), "se": float(means.std(ddof=1)), "n": n,
            "q05": float(np.quantile(means, 0.05)), "q95": float(np.quantile(means, 0.95)),
            "p_le_0": float(np.mean(means <= 0)),
            "se_method": f"moving-block bootstrap, block {b} rows, {draws} draws"}
