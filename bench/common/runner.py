"""Run every split of a benchmark (resumable, two worker processes) and write the results JSON.

A benchmark module provides ``Bench`` with: ``name``, ``se`` ("iid" or ("block", rows)), ``sizes``,
``splits()`` -> list of split dicts (name, size, seed, tr, te, q_tr, q_te, inner, keys_te), ``X``, ``y``,
``features(X)`` (tabular summaries for the boosting arm) and ``params(size)`` (network settings).
Per split predictions are cached in ``bench/results/raw/<name>/`` (kept out of git).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from .arms import _gru, run_split, unanchored_inputs
from .metrics import block_bootstrap_se, example_loss, iid_se, summary

RESULTS = Path(__file__).resolve().parents[1] / "results"
BASELINES = ("prior", "unanchored", "gbm_init", "anch_l0")


def _materialise(b, s):
    X, y = b.X, b.y
    tr, te = s["tr"], s["te"]
    Xu_tr, mu, sd = unanchored_inputs(X[tr], s["q_tr"])
    Xu_te, _, _ = unanchored_inputs(X[te], s["q_te"], mu, sd)
    return dict(seed=s["seed"], X_tr=X[tr], y_tr=y[tr], q_tr=s["q_tr"], X_te=X[te], q_te=s["q_te"],
                inner=s["inner"], Xu_tr=Xu_tr, Xu_te=Xu_te, F_tr=b.features(X[tr]), F_te=b.features(X[te]))


def _job(b, s, path: Path, ablations: bool):
    os.environ["OMP_NUM_THREADS"] = "1"
    out = run_split(_materialise(b, s), b.params(s["size"]), ablations)
    np.savez_compressed(path, keys=s["keys_te"], y=b.y[s["te"]], **{f"p_{k}": v for k, v in out["p"].items()})
    path.with_suffix(".json").write_text(json.dumps(out["info"], default=float))
    return s["name"], out["info"]["seconds"]


def _spread(b, s, lam_star: float, n_seeds: int = 5) -> dict:
    """Seed-to-seed spread of p at one split: penalty 0, the default 1 and the chosen penalty."""
    m = _materialise(b, s)
    out = {}
    for lam in sorted({0.0, 1.0, float(lam_star)}):
        f = _gru(b.params(s["size"]), penalty=lam, random_state=s["seed"], n_seeds=n_seeds)
        f.fit(m["X_tr"], m["y_tr"], prior=m["q_tr"])
        P = f.predict_proba_per_seed(m["X_te"], prior=m["q_te"])
        out[f"l{lam:g}"] = {"seed_spread_p_test": float(P.std(axis=1).mean()), "n_seeds": n_seeds,
                            "log_loss_ensemble": summary(b.y[s["te"]],
                                                         P.mean(axis=1))["log_loss"],
                            "share_at_cap_train": f.diagnostics_["share_at_cap"]}
    return out


def run(b, ablations: bool = True, spread: bool = True, n_jobs: int = 2) -> dict:
    raw = RESULTS / "raw" / b.name
    raw.mkdir(parents=True, exist_ok=True)
    splits = b.splits()
    todo = [s for s in splits if not (raw / f"{s['name']}.npz").exists()]
    t0 = time.time()
    done = Parallel(n_jobs=n_jobs, verbose=0)(delayed(_job)(b, s, raw / f"{s['name']}.npz", ablations)
                                              for s in sorted(todo, key=lambda s: -s["size"]))
    for name, sec in done:
        print(f"[{b.name}] {name}: {sec:.0f} s", flush=True)
    res = {"dataset": b.name, "description": b.description, "sizes": {}, "settings": b.settings(),
           "wall_seconds_this_call": None}
    for size in b.sizes:
        res["sizes"][str(size)] = _aggregate(b, [s for s in splits if s["size"] == size], raw)
    if spread:
        s0 = [s for s in splits if s["size"] == min(b.sizes)][0]
        lam = json.loads((raw / f"{s0['name']}.json").read_text())["lam_star"]
        sp = raw / "spread.json"
        if not sp.exists():
            sp.write_text(json.dumps({"split": s0["name"], **_spread(b, s0, lam)}))
        res["seed_spread"] = json.loads(sp.read_text())
    res["wall_seconds_this_call"] = time.time() - t0
    out = RESULTS / f"{b.name}.json"
    out.write_text(json.dumps(res, indent=1, default=float))
    return res


def _aggregate(b, splits, raw) -> dict:
    loads = [np.load(raw / f"{s['name']}.npz") for s in splits]
    infos = [json.loads((raw / f"{s['name']}.json").read_text()) for s in splits]
    arms = [k[2:] for k in loads[0].files if k.startswith("p_")]
    per_arm = {}
    for arm in arms:
        ms = [summary(z["y"], z[f"p_{arm}"]) for z in loads]
        per_arm[arm] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
        per_arm[arm]["per_split_log_loss"] = [m["log_loss"] for m in ms]
    keys = np.unique(np.concatenate([z["keys"] for z in loads]))
    diffs = {}
    for arm in arms:
        if arm == "ours":
            continue
        tot, cnt = np.zeros(len(keys)), np.zeros(len(keys))
        for z in loads:
            pos = np.searchsorted(keys, z["keys"])
            d = example_loss(z["y"], z["p_ours"])[0] - example_loss(z["y"], z[f"p_{arm}"])[0]
            np.add.at(tot, pos, d)
            np.add.at(cnt, pos, 1)
        d = tot / cnt
        st = iid_se(d) if b.se == "iid" else block_bootstrap_se(d, b.se[1])
        st["beats_by_2se"] = bool(st["mean"] < -2 * st["se"])
        st["loses_by_2se"] = bool(st["mean"] > 2 * st["se"])
        diffs[arm] = st
    keep = all(diffs[a]["beats_by_2se"] for a in BASELINES if a in diffs)
    return {"n_splits": len(splits), "n_test_unique": int(len(keys)), "metrics": per_arm,
            "diff_log_loss_ours_minus_arm": diffs, "keep_rule_ours_beats_all_baselines_by_2se": keep,
            "lam_star_per_split": [i["lam_star"] for i in infos],
            "inner_ll_per_split": [i["inner_ll"] for i in infos],
            "ours_offset_rms_test": [i["ours_offset_test_rms"] for i in infos],
            "diagnostics_train": {k[5:]: [i[k] for i in infos] for k in infos[0] if k.startswith("diag_")},
            "gbm_rounds": [i["gbm_rounds"] for i in infos], "seconds": [i["seconds"] for i in infos]}
