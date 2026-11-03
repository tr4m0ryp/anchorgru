"""Print a compact table per dataset and size from bench/results/<dataset>.json.

Run: PYTHONPATH=. python bench/report.py [dataset ...]
"""
from __future__ import annotations

import json
import sys

from bench.common.runner import RESULTS

ORDER = ("prior", "anch_l0", "anch_l0.1", "anch_l0.3", "anch_l1", "anch_l3", "anch_l10", "ours", "unanchored",
         "gbm_init", "abl_nocap", "abl_randinit", "abl_nogate", "abl_wd10_l0")


def show(name: str) -> None:
    r = json.loads((RESULTS / f"{name}.json").read_text())
    print(f"## {name}: {r['description']}")
    for size, s in r["sizes"].items():
        print(f"\nn = {size}  (splits {s['n_splits']}, test rows {s['n_test_unique']}, chosen penalty "
              f"{s['lam_star_per_split']}, keep rule met: {s['keep_rule_ours_beats_all_baselines_by_2se']})")
        print(f"{'arm':14s} {'logloss':>8s} {'brier':>7s} {'ece':>6s} {'auc':>6s} {'ours-arm':>9s} {'SE':>7s}")
        for arm in [a for a in ORDER if a in s["metrics"]]:
            m, d = s["metrics"][arm], s["diff_log_loss_ours_minus_arm"].get(arm)
            dd = f"{d['mean']:+9.4f} {d['se']:7.4f}" if d else f"{'':9s} {'':7s}"
            print(f"{arm:14s} {m['log_loss']:8.4f} {m['brier']:7.4f} {m['ece']:6.3f} {m['auc']:6.3f} {dd}")
    if "seed_spread" in r:
        print("\nseed spread (5 seeds, smallest n):", json.dumps(r["seed_spread"]))
    print()


if __name__ == "__main__":
    for n in sys.argv[1:] or ("physionet2012", "elec2", "synthetic"):
        if (RESULTS / f"{n}.json").exists():
            show(n)
