"""Training loop, bootstrap bags and bag prediction for :class:`ResidualRNN`."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .._prob import expit
from .residual import ResidualRNN, penalised_loss


@dataclass(frozen=True)
class TrainConfig:
    """Architecture and optimisation settings of one network."""

    hidden: int = 16
    cell: str = "gru"
    layers: int = 1
    dropout: float = 0.0
    cap: float = 2.0
    zero_init: bool = True
    epochs: int = 100
    lr: float = 1e-2
    weight_decay: float = 0.1
    batch_size: int | None = 64
    label_smoothing: float = 0.0
    penalty: float = 1.0


def _as_tensors(seq, anchor, valid, y=None):
    t = [torch.as_tensor(np.asarray(seq), dtype=torch.float32),
         torch.as_tensor(np.asarray(anchor), dtype=torch.float32),
         torch.as_tensor(np.asarray(valid), dtype=torch.float32)]
    if y is not None:
        t.append(torch.as_tensor(np.asarray(y), dtype=torch.float32))
    return t


def train_net(seq, anchor, valid, y, cfg: TrainConfig, seed: int, bootstrap: bool = False,
              on_epoch: Callable | None = None) -> ResidualRNN:
    """Train one network with AdamW on every given row (or a bootstrap resample when ``bootstrap``).

    ``on_epoch(epoch, net, bce)`` is called after each epoch with the last batch's cross-entropy; it may read
    the network (for learning curves) but must not modify it. No early stopping: epochs is a hyper-parameter.
    """
    seq, anchor, valid, y = _as_tensors(seq, anchor, valid, y)
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    n = len(y)
    idx = torch.randint(0, n, (n,), generator=g) if bootstrap else torch.arange(n)
    seq, anchor, valid, y = seq[idx], anchor[idx], valid[idx], y[idx]
    y = y * (1 - cfg.label_smoothing) + 0.5 * cfg.label_smoothing
    net = ResidualRNN(seq.shape[2], cfg.hidden, cfg.cell, cfg.layers, cfg.dropout, cfg.cap, cfg.zero_init)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    bs = min(cfg.batch_size or n, n)
    for e in range(cfg.epochs):
        net.train()
        perm = torch.randperm(n, generator=g) if bs < n else torch.arange(n)
        for a in range(0, n, bs):
            j = perm[a:a + bs]
            opt.zero_grad()
            loss, bce = penalised_loss(net, seq[j], anchor[j], valid[j], y[j], float(cfg.penalty))
            loss.backward()
            opt.step()
        if on_epoch is not None:
            net.eval()
            with torch.no_grad():
                on_epoch(e + 1, net, float(bce.detach()))
    return net.eval()


def fit_bag(seq, anchor, valid, y, cfg: TrainConfig, seed: int, n_bags: int = 1) -> list[ResidualRNN]:
    """``n_bags`` networks; with more than one bag each trains on its own bootstrap resample."""
    return [train_net(seq, anchor, valid, y, cfg, seed * 100 + b, bootstrap=n_bags > 1) for b in range(n_bags)]


def bag_residuals(nets: list[ResidualRNN], seq, valid) -> np.ndarray:
    """(n, n_bags) offsets r of every network in the bag, float64."""
    s, _, v = _as_tensors(seq, np.zeros(len(seq)), valid)
    with torch.no_grad():
        return np.column_stack([n.residual(s, v).double().numpy() for n in nets])


def predict_bag(nets: list[ResidualRNN], seq, anchor, valid) -> np.ndarray:
    """Bag mean of sigmoid(anchor + r), computed in float64 (exactly the prior when every r is 0)."""
    r = bag_residuals(nets, seq, valid)
    return expit(np.asarray(anchor, dtype=np.float64)[:, None] + r).mean(axis=1)
