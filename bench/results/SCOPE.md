# Benchmark scope, cuts and variant log

All numbers are held-out log-loss in nats. "ours - arm" is the paired per-example difference (negative = ours
better) with its SE; "wins" means more than 2 SE. Ours = the anchored network with the penalty chosen on an inner
20 % holdout of its own training rows from {0, 0.1, 0.3, 1, 3, 10}. Full tables: `PYTHONPATH=. python bench/report.py`.

## What ran

| file | task | prior | sequence | sizes | repeats | SE |
|---|---|---|---|---|---|---|
| `physionet2012.json` | ICU mortality, PhysioNet/CinC 2012 | logistic on SAPS-I, out of fold | 48 h hourly grid, 36 variables, stride 2 | 500, 2,000 | 3 subsamples x seed | iid over 4,000 test stays |
| `elec2.json` | elec2, price above its 24 h average | logistic on the current row, refitted weekly on the previous 2,000 rows | last 48 half-hours of 5 columns, stride 2 | 500, 2,000 | 3 forward-chaining folds (seed = fold) | moving-block bootstrap, one-week blocks (336 rows), 2,000 draws, 12,000 test rows |
| `synthetic.json` | known-answer synthetic | logistic on static features | 3 random-walk channels, 24 steps, 20 % missing | 250, 1,000, 4,000 | 3 | iid over 5,000 test rows |
| `physionet2012_strongprior.json` | variant V1 | LightGBM on admission values, SAPS-I, first-24 h summaries | hours 24-47 only | 500, 2,000 | 3 | iid |
| `physionet2012_gbmprior.json` | variant V2 | LightGBM on admission values, SAPS-I, 48 h summaries | 48 h | 500, 2,000 | 3 | iid |
| `physionet2012_wd10.json` | variant V3 | as physionet2012 | as physionet2012 | 500, 2,000 | 3 | iid |

Arms per split: prior; anchored with penalty 0, 0.1, 0.3, 1, 3, 10; ours; unanchored (same GRU, base-rate start,
no cap, no penalty, the prior's log-odds as an input channel, inputs centred); LightGBM boosted from the prior
(`init_score` = prior log-odds, rounds by early stopping on the same inner holdout; features: per-variable last /
mean / min / max / count for PhysioNet, the network's own flattened steps for elec2 and the synthetic task);
ablations of ours (no tanh cap, random head init, no validity gate, penalty 0 with weight decay x10). Seed spread:
5 seeds of penalty 0, 1 and the chosen penalty on the first split of the smallest size.

Wall time on 2 worker processes of 1 thread each, on a shared 8-core machine (load 12-20): physionet2012 15.7 min,
elec2 7.1 min, synthetic 15.5 min, V1 6.0 min, V2 7.5 min, V3 7.3 min; **59 min in total** (`bench/run_all.sh`).

## Cuts (all deliberate, none silent)

- **weatherAUS dropped.** It needs a Kaggle login; this run has none. The synthetic task takes its slot.
- **GRU-D (PyPOTS) arm dropped.** PyPOTS is not installed in the shared environment and pulls a large dependency
  tree. The unanchored GRU with the prior as input is the sequence-model competitor.
- **Full-size training (8,000 PhysioNet stays, 8,000 elec2 rows) dropped.** A first attempt at it was too slow for
  the budget (about 20 min per split under the machine's load), so the learning curves stop at 2,000 (4,000 on the
  synthetic task). Claim (b) of the plan concerns small n, which is kept.
- **One network per arm and split** (`n_seeds=1, n_bags=1`, the package default is 5 x 3), batch 128 instead of 64,
  epochs by size (PhysioNet / elec2: 100 at 500, 30 at 2,000; synthetic: 150 / 60 / 20). Seed variation enters
  through the 3 repeats; seed spread is measured separately with 5 seeds.
- **Penalty tuned on one inner 80 / 20 holdout** (not inner CV); the sweep is reported in full so the effect of the
  choice is visible. The 70-trial random search (`anchorgru.tune.search`) was not run.
- **Repeats: 3** (the plan: 5 folds x 3). PhysioNet and the synthetic task use one fixed test set and pool the 3
  repeats per test example before the SE, so the SE covers test-sample noise, not training-subsample noise; the
  per-split log-losses are in each JSON (`per_split_log_loss`).
- **No "no mask channel" ablation**: the encoder has no switch for it; adding one would change the package.
- **elec2, LightGBM features:** a first elec2 run gave LightGBM only the last 12 raw steps, which hides the 24 h
  average that defines the label. It was rerun with the network's own 24 steps (results above are the rerun; the
  first run's LightGBM log-loss was 0.563 / 0.527, the rerun's 0.503 / 0.457, every other arm unchanged).

## Results (main datasets)

| dataset, n | ours | prior | ours - prior | ours - penalty 0 | ours - unanchored | ours - LightGBM from prior |
|---|---|---|---|---|---|---|
| PhysioNet, 500 | 0.3916 | 0.3960 | **-0.0045 (0.0007)** | **-0.1148 (0.0083)** | **-0.3074 (0.0170)** | +0.0471 (0.0030) lost |
| PhysioNet, 2,000 | 0.3811 | 0.3960 | **-0.0150 (0.0017)** | **-0.0913 (0.0076)** | **-0.0585 (0.0105)** | +0.0726 (0.0047) lost |
| elec2, 500 | 0.3291 | 0.5598 | **-0.2307 (0.0263)** | -0.0173 (0.0087) | **-0.9692 (0.0918)** | **-0.1738 (0.0269)** |
| elec2, 2,000 | 0.2151 | 0.5598 | **-0.3447 (0.0205)** | 0 (same model) | **-0.3301 (0.0343)** | **-0.2416 (0.0353)** |
| synthetic, 250 | 0.5940 | 0.5954 | -0.0015 (0.0029) tie | **-0.2341 (0.0079)** | **-0.3308 (0.0098)** | -0.0062 (0.0030) |
| synthetic, 1,000 | 0.5514 | 0.5954 | **-0.0440 (0.0042)** | **-0.0080 (0.0024)** | **-0.0695 (0.0056)** | **-0.0294 (0.0035)** |
| synthetic, 4,000 | 0.5251 | 0.5954 | **-0.0703 (0.0047)** | 0 (same model) | -0.0038 (0.0019) tie | **-0.0264 (0.0030)** |

Bayes log-loss of the synthetic task: 0.5044. The pre-registered keep rule (ours beats prior, penalty 0, unanchored
and LightGBM-from-prior, each by more than 2 SE) is met on **synthetic n = 1,000 only**. On elec2 it misses only
against penalty 0 (n = 500: 2.0 SE; n = 2,000: the inner holdout chose penalty 0 itself).

**Penalty 0 versus the penalty (claim a, no harm):** the unpenalised residual saturates its cap on 44-90 % of
training rows at the smallest n and its seed spread of p is 6-10x the penalised one (PhysioNet 0.099 vs 0.014,
elec2 0.035 vs 0.006, synthetic 0.157 vs 0.020, V2 0.092 vs 0.010). On PhysioNet and the synthetic task penalty 0
is worse than the prior itself (PhysioNet +0.11 / +0.08 nats); where the sequence is very informative (elec2,
synthetic n = 4,000) penalty 0 is the best setting and the inner holdout picks it.

**Ablations of ours:** no cap, no gate and random head init change log-loss by at most 0.0015 on PhysioNet and at
most 0.006 on the synthetic task; on elec2 the cap matters (no cap worse by 0.15 / 0.03) and random init is mixed
(better by 0.013 at 500, worse by 0.016 at 2,000). Penalty 0 with weight decay x10 is the best network on 4 of 7
rows (PhysioNet 2,000, elec2 500, synthetic 250 and 1,000), which motivated V3.

## Variant log (section 6: what would have to be true for a win over every baseline)

The PhysioNet loss is to LightGBM boosted from the prior; the GRU reads the irregular hourly grid worse than
per-variable summaries do. Three principled variants, each run on held-out data, plus the two sweeps built into
every run.

| variant | hypothesis | result | kept |
|---|---|---|---|
| sweep: penalty in {0 .. 10} | the default 1.0 is not the right strength | the best penalty moves with n and signal: 0.3 at PhysioNet 2,000, 0 on elec2, 0.1 at synthetic 1,000. Default 1.0 is too strong wherever the sequence has signal | the inner-holdout choice is ours; no fixed value |
| small n (250, 500) | an anchored model wins where an unanchored one overfits | yes against the unanchored GRU everywhere (-0.31 .. -0.97 nats at the smallest n); against the prior a tie at synthetic 250 (-0.0015, SE 0.0029) because a 50-row holdout picks the penalty badly (fixed 1.0 would give -0.0096 vs the prior) | reported, not kept as a rule |
| V1 prior strong but blind to hours 24-47 | strong prior + unseen sequence = the method's regime | prior 0.3090 (AUC 0.849); ours ties it at 2,000 (-0.0000, SE 0.0001) and is slightly worse at 500 (+0.0017, SE 0.0008); LightGBM-from-prior gains 0.004-0.006 over ours | no: lost a comparison |
| V2 prior = LightGBM on all 48 h summaries | the residual adds path information beyond summaries | ours ties the prior at both sizes (-0.0002 / +0.0002, SE 0.0002) and ties LightGBM-from-prior; penalty 0 is +0.165 / +0.120 worse than ours | no gain; the no-harm claim holds |
| V3 weight decay x10 for every network | the gap to LightGBM is overfitting, not representation | ours improves (2,000: 0.3811 -> 0.3571; vs prior -0.0389, SE 0.0042) but the unanchored GRU improves more (0.3277, ours +0.0294 worse) and LightGBM-from-prior stays ahead (+0.0486) | no: lost a comparison |

Class of the PhysioNet outcome: **lost a comparison** to LightGBM boosted from the prior, reopenable. What would
have to be true: an encoder that reads irregular clinical series at least as well as per-variable summaries (for
example summaries as extra static inputs to the head, or a time-aware cell), or a task whose label depends on the
order of events rather than their levels. With a weak prior and n = 2,000 the anchored model also loses to an
unanchored GRU once both are regularised (V3), as the method notes predict: the cap and the penalty hold it near a
poor anchor.
