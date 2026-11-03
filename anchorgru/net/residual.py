"""The residual recurrent network and its penalised training loss.

The network reads an encoded sequence and returns a bounded offset r on the log-odds scale:

    r = cap * tanh(w . h_T + beta) * valid

where h_T is the last hidden state of a GRU (or LSTM). The prediction is sigmoid(anchor + r), the anchor being the
log-odds of the prior. The head (w, beta) starts at zero, so an untrained network returns the prior exactly.
"""
from __future__ import annotations

import math

import torch
from torch import nn

CELLS = {"gru": nn.GRU, "lstm": nn.LSTM}


class ResidualRNN(nn.Module):
    """Recurrent encoder plus a linear head whose output is squashed into (-cap, cap).

    Parameters
    ----------
    n_inputs : number of input channels per step (values and observation masks).
    hidden : hidden state size.
    cell : "gru" or "lstm".
    layers : stacked recurrent layers.
    dropout : dropout before the head (and between stacked layers).
    cap : bound on |r|. ``math.inf`` removes the tanh (an unbounded linear head, an ablation).
    zero_init : start the head at zero (the network then equals the prior). False uses PyTorch's default init.
    """

    def __init__(self, n_inputs: int, hidden: int = 16, cell: str = "gru", layers: int = 1,
                 dropout: float = 0.0, cap: float = 2.0, zero_init: bool = True):
        super().__init__()
        if cell not in CELLS:
            raise ValueError(f"cell must be one of {sorted(CELLS)}, got {cell!r}")
        if not cap > 0:
            raise ValueError("cap must be positive")
        self.rnn = CELLS[cell](n_inputs, hidden, num_layers=layers, batch_first=True,
                               dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)
        self.cap = float(cap)
        if zero_init:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def residual(self, seq: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Offset r per example, shape (n,); exactly 0 where ``valid`` is 0."""
        h = self.rnn(seq)[0][:, -1]
        z = self.head(self.drop(h))[:, 0]
        r = z if math.isinf(self.cap) else self.cap * torch.tanh(z)
        return r * valid

    def forward(self, seq: torch.Tensor, anchor: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Log-odds of the prediction: anchor + r."""
        return anchor + self.residual(seq, valid)


def penalised_loss(net: ResidualRNN, seq, anchor, valid, y, penalty: float):
    """Return (training loss, its cross-entropy part).

    loss = mean BCE(y, sigmoid(anchor + r)) + penalty * sum(r^2) / sum(valid)

    The penalty acts on the output offset itself, so it shrinks every example's prediction toward its prior,
    which a weight-space penalty does not do under Adam. With ``penalty == 0`` the returned loss is the plain
    cross-entropy tensor, so training is identical bit for bit to an unpenalised loop.
    """
    r = net.residual(seq, valid)
    bce = nn.functional.binary_cross_entropy_with_logits(anchor + r, y)
    if not penalty:
        return bce, bce
    return bce + penalty * (r ** 2).sum() / valid.sum().clamp(min=1.0), bce
