"""The network: start equivalence, cap, invalid rows, the loss formula and lambda = 0 bit for bit."""
import math

import numpy as np
import pytest
import torch

from anchorgru.net import ResidualRNN, TrainConfig, fit_bag, penalised_loss, predict_bag, train_net
from anchorgru.windows import encode


def _inputs(n=64, T=10, C=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    seq = torch.randn(n, T, 2 * C, generator=g)
    anchor = torch.randn(n, generator=g)
    valid = (torch.rand(n, generator=g) > 0.25).float()
    y = (torch.rand(n, generator=g) > 0.5).float()
    return seq, anchor, valid, y


def test_untrained_network_returns_the_anchor_exactly():
    seq, anchor, valid, _ = _inputs()
    net = ResidualRNN(seq.shape[2]).eval()
    assert torch.equal(net(seq, anchor, valid), anchor)
    assert torch.count_nonzero(net.residual(seq, valid)) == 0


def test_residual_is_bounded_by_the_cap_and_zero_on_invalid_rows():
    seq, anchor, valid, _ = _inputs()
    net = ResidualRNN(seq.shape[2], cap=0.7, zero_init=False).eval()
    with torch.no_grad():
        net.head.weight.mul_(50.0)
        r = net.residual(seq, valid)
    assert torch.all(r.abs() <= 0.7 + 1e-6)
    assert torch.all(r[valid == 0] == 0)
    assert torch.allclose(net(seq, anchor, valid) - anchor, r)
    assert r.abs().max() > 0.6


def test_infinite_cap_is_the_unbounded_linear_head():
    seq, _, valid, _ = _inputs()
    net = ResidualRNN(seq.shape[2], cap=math.inf, zero_init=False).eval()
    with torch.no_grad():
        h = net.rnn(seq)[0][:, -1]
        expect = net.head(h)[:, 0] * valid
        assert torch.allclose(net.residual(seq, valid), expect)


@pytest.mark.parametrize("lam", [0.0, 0.3, 4.0])
def test_loss_is_bce_plus_lambda_sum_r2_over_valid_rows(lam):
    seq, anchor, valid, y = _inputs()
    net = ResidualRNN(seq.shape[2], zero_init=False).eval()
    loss, bce = penalised_loss(net, seq, anchor, valid, y, lam)
    with torch.no_grad():
        r = net.residual(seq, valid).double()
        z = anchor.double() + r
        p = torch.sigmoid(z)
        want_bce = -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()
        want = want_bce + lam * (r ** 2).sum() / valid.sum()
    assert float(bce.detach()) == pytest.approx(float(want_bce), rel=1e-5)
    assert float(loss.detach()) == pytest.approx(float(want), rel=1e-5)


def _plain_loop(seq, anchor, valid, y, cfg, seed):
    """The unpenalised reference: same seeding, same optimiser, plain cross-entropy."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    n = len(y)
    net = ResidualRNN(seq.shape[2], cfg.hidden, cfg.cell, cfg.layers, cfg.dropout, cfg.cap, cfg.zero_init)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    bs = min(cfg.batch_size or n, n)
    for _ in range(cfg.epochs):
        net.train()
        perm = torch.randperm(n, generator=g) if bs < n else torch.arange(n)
        for a in range(0, n, bs):
            j = perm[a:a + bs]
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(anchor[j] + net.residual(seq[j], valid[j]), y[j])
            loss.backward()
            opt.step()
    return net.eval()


@pytest.mark.parametrize("batch_size,dropout,cell", [(None, 0.0, "gru"), (16, 0.3, "gru"), (16, 0.0, "lstm")])
def test_penalty_zero_is_the_plain_loop_bit_for_bit(batch_size, dropout, cell):
    seq, anchor, valid, y = _inputs()
    cfg = TrainConfig(epochs=4, batch_size=batch_size, dropout=dropout, cell=cell, penalty=0.0)
    a = train_net(seq, anchor, valid, y, cfg, seed=7)
    b = _plain_loop(seq, anchor, valid, y, cfg, seed=7)
    for (k, v), (_, w) in zip(a.state_dict().items(), b.state_dict().items()):
        assert torch.equal(v, w), k


def test_penalty_zero_bagging_is_deterministic_and_bags_differ():
    seq, anchor, valid, y = _inputs()
    cfg = TrainConfig(epochs=3, penalty=0.0, batch_size=None)
    a, b = fit_bag(seq, anchor, valid, y, cfg, 3, n_bags=2), fit_bag(seq, anchor, valid, y, cfg, 3, n_bags=2)
    for na, nb in zip(a, b):
        assert all(torch.equal(v, w) for v, w in zip(na.state_dict().values(), nb.state_dict().values()))
    assert not torch.equal(a[0].head.weight, a[1].head.weight)


def test_epoch_callback_sees_every_epoch_and_does_not_change_training():
    seq, anchor, valid, y = _inputs()
    cfg = TrainConfig(epochs=5, dropout=0.2, batch_size=16)
    seen = []

    def cb(e, net, bce):
        net(seq, anchor, valid)
        seen.append((e, bce))

    a = train_net(seq, anchor, valid, y, cfg, seed=1, on_epoch=cb)
    b = train_net(seq, anchor, valid, y, cfg, seed=1)
    assert [e for e, _ in seen] == [1, 2, 3, 4, 5]
    assert all(np.isfinite(v) for _, v in seen)
    assert all(torch.equal(v, w) for v, w in zip(a.state_dict().values(), b.state_dict().values()))


def test_predict_bag_of_untrained_nets_is_the_prior_in_float64():
    rng = np.random.default_rng(0)
    q = rng.uniform(0.01, 0.99, 50)
    seq, valid = encode(rng.normal(size=(50, 8, 1)))
    nets = [ResidualRNN(seq.shape[2]).eval() for _ in range(3)]
    p = predict_bag(nets, seq, np.log(q / (1 - q)), valid)
    np.testing.assert_allclose(p, q, rtol=1e-12, atol=0)
