"""Two-stage random search with seed-averaged out-of-fold scoring, resumable from a JSON-lines log.

Stage A scores the default configuration and ``n_trials`` random ones with one seed. Stage B re-scores the
``top_k`` best of A (and the default) with several seeds, optionally on a second splitter, and ranks them by the
log-loss of the seed-averaged out-of-fold probability: the quantity a seed ensemble actually delivers.
Every finished trial is appended to the log; a rerun skips what is already there.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone

from .._prob import log_loss
from ..estimator import AnchoredGRUClassifier
from .oof import oof_raw
from .space import sample


def _score(p, y) -> float:
    ok = np.isfinite(p)
    return log_loss(p[ok], y[ok])


def _trial(est, params, seed, cv, X, y, prior, groups, stage, i):
    t = time.time()
    m = clone(est).set_params(**params, n_seeds=1, random_state=seed, calibration="none", n_jobs=1)
    p = oof_raw(m, X, y, prior, cv, groups)[0][:, 0]
    return {"stage": stage, "i": i, "seed": seed, "params": params,
            "p": [None if not np.isfinite(v) else float(v) for v in p], "seconds": round(time.time() - t, 2)}


def _read(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _arr(rec) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in rec["p"]], dtype=np.float64)


def search(X, y, prior=None, cv=None, cv_b=None, groups=None, n_trials: int = 70, top_k: int = 6,
           seeds=(1, 2, 3), log=None, estimator=None, fixed: dict | None = None,
           space: Callable = sample, random_state: int = 0, n_jobs: int | None = 1) -> dict:
    """Search AnchoredGRUClassifier parameters.

    Parameters
    ----------
    X, y, prior, groups : training data, as for ``fit``; ``groups`` goes to the splitters.
    cv, cv_b : splitters for stage A and stage B (``cv_b`` defaults to ``cv``; a differently seeded partition
        reduces the selection bias of reusing the stage A folds).
    log : path of the JSON-lines log (None = in memory only, not resumable).
    estimator : template whose parameters are the defaults (default ``AnchoredGRUClassifier()``).
    fixed : parameters merged into every configuration (e.g. ``{"penalty": 0.0}`` for an ablation arm).
    space : ``space(rng, n_steps) -> dict`` drawing one configuration.

    Returns a dict with ``best_params`` (to pass to ``set_params``), ``best_index``, ``stage_a`` (log-loss per
    configuration index), ``stage_b`` (per index: seed-averaged and single-seed log-loss, mean seed spread) and
    ``configs``.
    """
    X, y = np.asarray(X, dtype=np.float64), np.asarray(y)
    est = AnchoredGRUClassifier() if estimator is None else estimator
    fixed = dict(fixed or {})
    rng = np.random.default_rng(random_state)
    n_steps = X.shape[1]
    cfgs = [dict(fixed)] + [{**space(rng, n_steps), **fixed} for _ in range(n_trials)]
    path = None if log is None else Path(log)
    recs = _read(path)
    have = {(r["stage"], r["i"], r["seed"]) for r in recs}
    yb = (y == np.unique(y)[-1]).astype(float)

    def run(tasks, splitter):
        todo = [t for t in tasks if t not in have]
        out = Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(_trial)(est, cfgs[i], s, splitter, X, y, prior, groups, st, i) for st, i, s in todo)
        fh = None if path is None else path.open("a")
        try:
            for r in out:
                if fh is not None:
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()
                recs.append(r)
                have.add((r["stage"], r["i"], r["seed"]))
        finally:
            if fh is not None:
                fh.close()

    run([("A", i, 0) for i in range(len(cfgs))], cv)
    sc_a = {r["i"]: _score(_arr(r), yb) for r in recs if r["stage"] == "A" and r["i"] < len(cfgs)}
    top = sorted((i for i in sc_a if i >= 1), key=sc_a.get)[:top_k]
    run([("B", i, s) for i in [0, *top] for s in seeds], cv if cv_b is None else cv_b)
    by_i: dict[int, list[np.ndarray]] = {}
    for r in recs:
        if r["stage"] == "B" and r["i"] in [0, *top]:
            by_i.setdefault(r["i"], []).append(_arr(r))
    sc_b = {}
    for i, ps in by_i.items():
        P = np.column_stack(ps)
        sc_b[i] = {"seedavg_logloss": _score(P.mean(axis=1), yb),
                   "single_logloss": float(np.mean([_score(p, yb) for p in ps])),
                   "seed_spread": float(np.nanmean(P.std(axis=1))) if P.shape[1] > 1 else 0.0,
                   "n_seeds": P.shape[1]}
    best = min(sc_b, key=lambda i: sc_b[i]["seedavg_logloss"])
    return {"best_params": cfgs[best], "best_index": best, "stage_a": sc_a, "stage_b": sc_b,
            "configs": {i: cfgs[i] for i in sc_b}}
