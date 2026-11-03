"""Figures for the paper, from the benchmark result files (vector PDF, Okabe-Ito colours).

Fig. 1 (fig1_paired.pdf)   ours - arm held-out log-loss with 95 % CI (+-1.96 reported paired SE), per dataset.
Fig. 2 (fig2_sweep.pdf)    the mechanism: log-loss minus the prior's against the penalty strength.
Fig. 3 (fig3_reliability.pdf)  reliability of prior, penalty 0 and ours at the smallest n, ECE in the legend.

Run: python3 figures/make.py [--src DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "results"))
from paired import load_splits, reliability  # noqa: E402

SRC = Path("/home/tr4m0ryp/projects/oss/anchorgru/bench/results")
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "blue": "#0072B2",
      "verm": "#D55E00", "purple": "#CC79A7", "yellow": "#F0E442"}
PANELS = [("synthetic", "1000", "(a) Synthetic, $n$ = 1,000"), ("elec2", "2000", "(b) elec2, $n$ = 2,000"),
          ("physionet2012", "2000", "(c) PhysioNet, SAPS-I prior, $n$ = 2,000"),
          ("physionet2012_gbmprior", "2000", "(d) PhysioNet, strong prior, $n$ = 2,000")]
ARMS = [("prior", "prior alone"), ("anch_l0", "penalty 0"), ("unanchored", "unanchored GRU"),
        ("gbm_init", "LightGBM from prior")]
SWEEP = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False})


def fig_paired(res, out):
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.6))
    axes = axes.ravel()
    for ax, (ds, size, title) in zip(axes, PANELS):
        diffs = res[ds]["sizes"][size]["diff_log_loss_ours_minus_arm"]
        for i, (arm, _) in enumerate(ARMS):
            d = diffs[arm]
            m, se = d["mean"], d["se"]
            col = OI["blue"] if m + 1.96 * se < 0 else (OI["verm"] if m - 1.96 * se > 0 else OI["black"])
            ax.errorbar(m, i, xerr=1.96 * se, fmt="o", ms=3.5, color=col, capsize=2, lw=1.2)
            if se == 0 and m == 0:
                ax.annotate("same model", (m, i), xytext=(-4, 0), textcoords="offset points", ha="right",
                            va="center", fontsize=6.5, color="0.3")
        ax.axvline(0, color="0.5", lw=0.7, ls="--")
        ax.set_yticks(range(len(ARMS)))
        ax.set_yticklabels([a[1] for a in ARMS] if ax in (axes[0], axes[2]) else [])
        ax.set_ylim(len(ARMS) - 0.5, -0.5)
        ax.set_title(title)
        if ax in (axes[2], axes[3]):
            ax.set_xlabel("ours $-$ arm, held-out log-loss (nats)")
        ax.ticklabel_format(axis="x", style="plain")
        ax.locator_params(axis="x", nbins=4)
    fig.tight_layout(w_pad=0.8, h_pad=0.8)
    fig.savefig(out)
    plt.close(fig)


def fig_sweep(res, out):
    sets = [("synthetic", "(a) Synthetic"), ("elec2", "(b) elec2"), ("physionet2012", "(c) PhysioNet 2012, SAPS-I prior"),
            ("physionet2012_gbmprior", "(d) PhysioNet 2012, strong prior")]
    cols = [OI["orange"], OI["blue"], OI["green"]]
    xs = np.arange(len(SWEEP))
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.9))
    axes = axes.ravel()
    for ax, (ds, title) in zip(axes, sets):
        for c, (size, s) in zip(cols, res[ds]["sizes"].items()):
            met = s["metrics"]
            base = met["prior"]["log_loss"]
            y = [met[f"anch_l{lam:g}"]["log_loss"] - base for lam in SWEEP]
            ax.plot(xs, y, "-o", ms=2.5, color=c, lw=1.1, label=f"$n$ = {int(size):,}")
            ours = met["ours"]["log_loss"] - base
            ax.plot([xs[-1] + 0.9], [ours], marker="*", ms=6, color=c, ls="none")
        ax.axhline(0, color="0.5", lw=0.7, ls="--")
        ax.set_xticks(list(xs) + [xs[-1] + 0.9])
        ax.set_xticklabels([f"{v:g}" for v in SWEEP] + ["ours"], rotation=0)
        ax.set_title(title)
        if ax in (axes[2], axes[3]):
            ax.set_xlabel(r"penalty $\lambda$ (fixed), or ours (chosen per split)")
        if ax in (axes[0], axes[2]):
            ax.set_ylabel("log-loss $-$ prior's (nats)")
        ax.legend(frameon=False, loc="best", handlelength=1.2)
    fig.tight_layout(w_pad=0.8, h_pad=0.8)
    fig.savefig(out)
    plt.close(fig)


def fig_reliability(src, out):
    sets = [("synthetic", 250, "Synthetic, $n$ = 250"), ("physionet2012", 500, "PhysioNet 2012, $n$ = 500")]
    arms = [("prior", "prior", OI["black"], "o"), ("anch_l0", "penalty 0", OI["verm"], "s"),
            ("ours", "ours", OI["blue"], "^")]
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.3))
    for ax, (ds, size, title) in zip(axes, sets):
        sp = load_splits(src / "raw" / ds, size)
        y = np.concatenate([z["y"] for z in sp]).astype(float)
        hi = 0.0
        for arm, lab, c, mk in arms:
            p = np.concatenate([z[f"p_{arm}"] for z in sp])
            rows, e = reliability(y, p, bins=15)
            mp, my = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
            hi = max(hi, mp.max(), my.max())
            ax.plot(mp, my, marker=mk, ms=2.8, lw=1.0, color=c, label=f"{lab} (ECE {e:.3f})")
        hi = min(1.0, hi * 1.05)
        ax.plot([0, hi], [0, hi], color="0.6", lw=0.7, ls="--")
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("mean predicted probability")
        if ax is axes[0]:
            ax.set_ylabel("observed frequency")
        ax.legend(frameon=False, loc="upper left", handlelength=1.2)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    a = ap.parse_args()
    names = ["synthetic", "elec2", "physionet2012", "physionet2012_gbmprior"]
    res = {n: json.loads((a.src / f"{n}.json").read_text()) for n in names}
    fig_paired(res, HERE / "fig1_paired.pdf")
    fig_sweep(res, HERE / "fig2_sweep.pdf")
    fig_reliability(a.src, HERE / "fig3_reliability.pdf")
    print("wrote", *(p.name for p in sorted(HERE.glob("fig*.pdf"))))


if __name__ == "__main__":
    main()
