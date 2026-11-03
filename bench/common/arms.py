"""Fit every arm on one train / test split and return its test probabilities.

Arms (same windows, same rows, same seed):
  prior         the prior alone
  anch_l<lam>   the anchored network with penalty lam (the sweep; lam = 0 is the unpenalised residual)
  ours          the anchored network with the penalty chosen on an inner holdout of the training rows
  unanchored    the same recurrent network without the anchor (base-rate start, no cap, no penalty), the prior's
                log-odds given as an extra input channel
  gbm_init      LightGBM boosted from the prior (init_score = prior log-odds) on tabular window summaries
  abl_*         ablations of ours: no cap, random head init, no gate, and weight decay x10 with penalty 0
"""
from __future__ import annotations

import time
import warnings

import numpy as np

SWEEP = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0)


def _logit(q, eps=1e-6):
    q = np.clip(q, eps, 1 - eps)
    return np.log(q) - np.log1p(-q)


# The network every benchmark arm uses (the recorded results were produced with exactly these settings).
NET = dict(hidden=8, cell="gru", layers=1, cap=2.5, lr=8e-3, weight_decay=0.2, dropout=0.0)
NET_DESC = "benchmark network (hidden 8, gru, cap 2.5, lr 8e-3, wd 0.2) with batch 128, 1 seed, 1 bag"


def _gru(params: dict, **over):
    from anchorgru import AnchoredGRUClassifier
    kw = dict(NET, n_seeds=1, n_bags=1, calibration="none", n_jobs=None, torch_threads=1)
    kw.update(params)
    kw.update(over)
    return AnchoredGRUClassifier(**kw)


def _ll(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def _gbm(F_tr, y_tr, b_tr, F_te, b_te, inner, seed):
    import lightgbm as lgb
    kw = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_child_samples=20, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=1, random_state=seed, verbose=-1)
    a, v = inner
    m = lgb.LGBMClassifier(n_estimators=1000, **kw)
    m.fit(F_tr[a], y_tr[a], init_score=b_tr[a], eval_set=[(F_tr[v], y_tr[v])], eval_init_score=[b_tr[v]],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    best = max(1, int(m.best_iteration_ or 1))
    m = lgb.LGBMClassifier(n_estimators=best, **kw).fit(F_tr, y_tr, init_score=b_tr)
    z = m.predict_proba(F_te, raw_score=True) + b_te
    return 1 / (1 + np.exp(-z)), best


def run_split(s: dict, params: dict, ablations: bool = True) -> dict:
    """Fit every arm on split ``s`` (see runner) and return {"p": {arm: p_te}, "info": {...}}."""
    import torch
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*eval_set.*")
    t0 = time.time()
    seed = int(s["seed"])
    X_tr, y_tr, q_tr, X_te, q_te = s["X_tr"], s["y_tr"], s["q_tr"], s["X_te"], s["q_te"]
    a, v = s["inner"]
    P, info = {"prior": q_te.astype(np.float64)}, {"seed": seed, "n_train": int(len(y_tr))}
    fits, inner_ll = {}, {}
    for lam in SWEEP:
        m = _gru(params, penalty=lam, random_state=seed).fit(X_tr[a], y_tr[a], prior=q_tr[a])
        inner_ll[lam] = _ll(y_tr[v], m.predict_proba(X_tr[v], prior=q_tr[v])[:, 1])
        m = _gru(params, penalty=lam, random_state=seed).fit(X_tr, y_tr, prior=q_tr)
        P[f"anch_l{lam:g}"] = m.predict_proba(X_te, prior=q_te)[:, 1]
        fits[lam] = m
        info[f"diag_anch_l{lam:g}"] = m.diagnostics_
    lam_star = min(SWEEP, key=lambda k: (inner_ll[k], -k))
    P["ours"] = P[f"anch_l{lam_star:g}"]
    info.update(lam_star=lam_star, inner_ll={f"{k:g}": x for k, x in inner_ll.items()},
                diag_ours=fits[lam_star].diagnostics_,
                ours_offset_test_rms=float(np.sqrt(np.mean(fits[lam_star].residual(X_te, prior=q_te) ** 2))))
    if ablations:
        for name, over in (("abl_nocap", dict(cap=float("inf"))), ("abl_randinit", dict(zero_init=False)),
                           ("abl_nogate", dict(min_observed=0.0)),
                           ("abl_wd10_l0", dict(weight_decay=10 * params.get("weight_decay", NET["weight_decay"]), penalty=0.0))):
            kw = dict(penalty=lam_star, random_state=seed)
            kw.update(over)
            m = _gru(params, **kw).fit(X_tr, y_tr, prior=q_tr)
            P[name] = m.predict_proba(X_te, prior=q_te)[:, 1]
            info[f"diag_{name}"] = m.diagnostics_
    m = _gru(params, penalty=0.0, cap=float("inf"), relative=False, random_state=seed)
    m.fit(s["Xu_tr"], y_tr)
    P["unanchored"] = m.predict_proba(s["Xu_te"])[:, 1]
    P["gbm_init"], info["gbm_rounds"] = _gbm(s["F_tr"], y_tr, _logit(q_tr), s["F_te"], _logit(q_te), (a, v), seed)
    info["seconds"] = time.time() - t0
    return {"p": {k: np.asarray(x, dtype=np.float64) for k, x in P.items()}, "info": info}


def unanchored_inputs(X, q, mean=None, sd=None):
    """Centre and scale each channel by training statistics and append the prior log-odds as a channel."""
    if mean is None:
        mean, sd = np.nanmean(X, axis=(0, 1)), np.nanstd(X, axis=(0, 1))
        mean, sd = np.nan_to_num(mean), np.where(np.nan_to_num(sd) > 1e-12, np.nan_to_num(sd), 1.0)
    Z = (X - mean) / sd
    b = np.broadcast_to(_logit(q)[:, None, None], (len(X), X.shape[1], 1))
    return np.concatenate([Z, b], axis=2), mean, sd
