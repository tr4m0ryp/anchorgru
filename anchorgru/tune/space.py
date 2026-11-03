"""The random search space over AnchoredGRUClassifier parameters."""
from __future__ import annotations

import numpy as np

MAX_STEPS = 64
MIN_STEPS = 4


def sample(rng: np.random.Generator, n_steps: int) -> dict:
    """One random configuration for windows of ``n_steps`` steps.

    Window length and stride are drawn so the network sees between 4 and 64 steps (fewer only when the windows
    are shorter than that), which keeps a trial cheap on a CPU.
    """
    windows = sorted({max(1, int(round(n_steps * f))) for f in (0.125, 0.25, 0.5, 1.0)})
    pairs = [(w, s) for w in windows for s in (1, 2, 3, 4, 6)
             if MIN_STEPS <= -(-w // s) <= MAX_STEPS]
    if not pairs:
        pairs = [(n_steps, 1)]
    w, s = pairs[int(rng.integers(len(pairs)))]
    return {
        "window": None if w == n_steps else int(w),
        "stride": int(s),
        "hidden": int(rng.choice([4, 8, 16, 32])),
        "dropout": float(rng.choice([0.0, 0.1, 0.3])),
        "weight_decay": float(10 ** rng.uniform(-3.0, 0.0)),
        "epochs": int(rng.choice([25, 50, 100, 200])),
        "lr": float(10 ** rng.uniform(-3.0, -1.5)),
        "cap": float(rng.choice([0.5, 1.0, 2.0, 3.0])),
        "n_bags": int(rng.choice([1, 3])),
        "cell": str(rng.choice(["gru", "lstm"])),
        "layers": int(rng.choice([1, 2])),
        "batch_size": [None, 32, 64, 128][int(rng.integers(4))],
        "label_smoothing": float(rng.choice([0.0, 0.05])),
        "penalty": float(rng.choice([0.1, 0.3, 1.0, 3.0, 10.0])),
    }
