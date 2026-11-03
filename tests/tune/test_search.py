"""Random search: deterministic draws, both stages, a resumable log, fixed overrides."""
import json

import numpy as np

from anchorgru import AnchoredGRUClassifier, BlockedTimeSplit
from anchorgru.tune import sample, search
from tests.conftest import make_data


def test_space_is_deterministic_and_respects_step_bounds():
    a = [sample(np.random.default_rng(4), 48) for _ in range(20)]
    b = [sample(np.random.default_rng(4), 48) for _ in range(20)]
    assert a == b
    for c in a:
        w = c["window"] or 48
        assert 4 <= -(-w // c["stride"]) <= 90
        AnchoredGRUClassifier().set_params(**c)


def test_search_runs_both_stages_and_resumes(tmp_path):
    X, y, q = make_data(n=120, T=8)
    est = AnchoredGRUClassifier(hidden=4, epochs=2, n_bags=1)
    small = lambda rng, T: {**sample(rng, T), "epochs": 2, "hidden": 4, "layers": 1}  # noqa: E731
    log = tmp_path / "s.jsonl"
    kw = dict(prior=q, cv=BlockedTimeSplit(3), n_trials=4, top_k=2, seeds=(1, 2), log=log, estimator=est,
              space=small, fixed={"penalty": 0.5})
    res = search(X, y, **kw)
    recs = [json.loads(line) for line in log.read_text().splitlines()]
    assert sum(r["stage"] == "A" for r in recs) == 5
    assert sum(r["stage"] == "B" for r in recs) == 3 * 2
    top = sorted((i for i in res["stage_a"] if i >= 1), key=res["stage_a"].get)[:2]
    assert set(res["stage_b"]) == {0, *top}
    assert res["best_params"]["penalty"] == 0.5
    assert all(v["n_seeds"] == 2 for v in res["stage_b"].values())
    again = search(X, y, **kw)
    assert len(log.read_text().splitlines()) == len(recs)
    assert again["best_index"] == res["best_index"]
    AnchoredGRUClassifier().set_params(**res["best_params"])
