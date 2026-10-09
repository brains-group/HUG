# Spec 05b results: fusion probe (idea A100)

**Run:** idea cluster, A100 80GB, commit `49f3502` (clean, `CODE_COMPAT_VERSION = 1`).
KuaiRand-1K, N4 pre-tuning defaults, seed 42, 3 epochs, validation only. All four runs reached
epoch 3 (exit 0), at 1192–1213 s/epoch and 38.3 GB peak each. **Best epoch was 1 for every run.**
Reported 2026-10-09.

| Run | Ep | Val AUC | Holdout AUC | BCE | aux/bce | co-act | row Jaccard | never fired on val (sh_G / sh_S / pr_G / pr_S) | dead in train window (shared / G / S) |
|---|---|---|---|---|---|---|---|---|---|
| P0 concat | 1 | 0.7730 | 0.7889 | 0.5591 | 0 | – | – | – | – |
| | 2 | 0.7709 | 0.7964 | 0.5209 | 0 | – | – | – | – |
| | 3 | 0.7598 | 0.7968 | 0.5016 | 0 | – | – | – | – |
| P1 sparse | 1 | 0.7511 | 0.7853 | 0.5569 | 0.422 | 0.239 | 0.713 | .734 / .480 / .480 / .363 | 0 / 0 / .0039 |
| | 2 | 0.7466 | 0.7949 | 0.5229 | 0.366 | 0.235 | 0.697 | .551 / .422 / .398 / .363 | .0078 / 0 / .0039 |
| | 3 | 0.7479 | 0.8004 | 0.5051 | 0.345 | 0.321 | 0.547 | .258 / .414 / .109 / .156 | .0039 / .0039 / .0078 |
| P2 sparse + dense skip | 1 | 0.7710 | 0.7906 | 0.5554 | 0.422 | 0.225 | 0.684 | .656 / .516 / .457 / .363 | 0 / 0 / .0078 |
| | 2 | 0.7670 | 0.7966 | 0.5201 | 0.368 | 0.206 | 0.702 | .500 / .520 / .309 / .332 | .0039 / .0078 / .0039 |
| | 3 | 0.7605 | 0.7996 | 0.5015 | 0.343 | 0.237 | 0.581 | .070 / .422 / .062 / .152 | 0 / .0039 / 0 |
| P3 evgate | 1 | 0.7731 | 0.7916 | 0.5498 | 0 | – | – | – | – |
| | 2 | 0.7672 | 0.7981 | 0.5160 | 0 | – | – | – | – |
| | 3 | 0.7548 | 0.7985 | 0.4930 | 0 | – | – | – | – |

k_pr ranges were identical in all six sparse epochs (P1 and P2, epochs 1–3):
- by n_G quintile: 4–9 (mean 5.2), 11–18 (14.4), 19–26 (22.5), 26–36 (30.8), 36–48 (43.2);
- by n_S quintile: 4–27 (15.9), 27–31 (27.0), 31–35 (32.2), 35–41 (37.5), 41–48 (45.0).

## Decision (architect, 2026-10-09)

- Rule as written (epoch 3): P2 0.7605 > P1 0.7479; P2 − max(P0, P3) = 0.7605 − 0.7598 = **+0.0007**,
  within 0.003.
- At the best epoch (1), the reading that matters for early-stopped models: P2 0.7710 vs
  max(P0, P3) 0.7731 = **−0.0021**, also within 0.003.
- **Stage 1b goes ahead with `--fusion-dense-skip` on** for the `sparse` and `misa` heads.
- Without the skip path, sparse codes cost ~0.022 AUC at epoch 1 (P1). The codes alone don't
  carry the signal, so the paper can't claim the sparse code is what predicts.
- Stage 1b on KuaiRand runs sparse, evgate and misa (plus the sparse ablations), matching the
  MIND/ZhihuRec plans. `gate` and `moe` are dropped for compute.

## Platform note

On the idea machine (AMD EPYC 7413), `test_fusion.py::test_concat_bitwise_matches_pre_spec05` fails:
the reference hash `60d53e57…` reproduces on brains but not there (`ace1a30a…`). The hash is the
same at `2224069`, `92b28d5` and `49f3502`, so Spec 05b didn't change concat. The reference is
CPU-specific, and the failure is waived on that machine.
