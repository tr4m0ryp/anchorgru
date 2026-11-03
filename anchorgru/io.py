"""Save and load fitted models with a sha256 check."""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save(model, path) -> str:
    """Pickle ``model`` to ``path``, write its sha256 to ``path + '.sha256'`` and return the digest."""
    path = Path(path)
    with path.open("wb") as fh:
        pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)
    digest = _sha256(path)
    Path(str(path) + ".sha256").write_text(digest + "\n")
    return digest


def load(path, sha256: str | None = None):
    """Load a model saved by :func:`save`.

    The file's sha256 must equal ``sha256`` when given, else the sidecar ``.sha256`` file when it exists.
    Raises ValueError on a mismatch. Only load files you trust: unpickling runs code.
    """
    path = Path(path)
    want = sha256
    side = Path(str(path) + ".sha256")
    if want is None and side.exists():
        want = side.read_text().strip()
    if want is not None:
        got = _sha256(path)
        if got != want.lower():
            raise ValueError(f"sha256 mismatch for {path}: expected {want}, got {got}")
    with path.open("rb") as fh:
        return pickle.load(fh)
