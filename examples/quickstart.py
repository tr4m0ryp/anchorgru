"""Quickstart on synthetic data: a prior that sees only static features, plus a sequence that holds the rest.

Run: python examples/quickstart.py
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_predict, train_test_split

from anchorgru import AnchoredGRUClassifier

rng = np.random.default_rng(0)
n, T, C = 2000, 24, 3
X = rng.normal(size=(n, T, C)).cumsum(axis=1)                    # random-walk channels
trend = X[:, -1, 0] - X[:, -6, 0]                                 # recent movement of channel 0
X[rng.random((n, T, C)) < 0.2] = np.nan                           # 20 % missing
static = rng.normal(size=(n, 4))
logit = static @ np.array([1.0, -0.5, 0.3, 0.0]) + 0.5 * trend
y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

tr, te = train_test_split(np.arange(n), test_size=0.3, random_state=0)
# The prior sees only the static features. Its training probabilities are out of fold, so the network does not
# learn to undo the prior's own overfit.
lr = LogisticRegression()
prior = np.empty(n)
prior[tr] = cross_val_predict(lr, static[tr], y[tr], cv=5, method="predict_proba")[:, 1]
prior[te] = lr.fit(static[tr], y[tr]).predict_proba(static[te])[:, 1]

results = {"prior alone": prior[te]}
# penalty sets how far the network may move the prior: 0 is the unpenalised residual, larger values shrink every
# offset toward zero. Tune it by cross-validation on your data (anchorgru.tune.search does this).
for penalty in (0.0, 0.1, 1.0):
    name = f"anchored, penalty {penalty:g}"
    clf = AnchoredGRUClassifier(penalty=penalty, epochs=40, n_seeds=3, n_bags=1, n_jobs=2)
    clf.fit(X[tr], y[tr], prior=prior[tr])
    results[name] = clf.predict_proba(X[te], prior=prior[te])[:, 1]
    d = clf.diagnostics_
    print(f"{name}: offset rms {d['residual_rms']:.3f}, seed spread of p {d['seed_spread']:.4f}")

for name, p in results.items():
    print(f"{name:22s} held-out log-loss {log_loss(y[te], p):.4f}")
