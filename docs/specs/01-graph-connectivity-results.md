# Spec 01 results: structural graph connectivity

**Date:** 2026-10-06
**Code:** R0 at `cabf8c9`; R0-noedge at `4e8fdcc`; R1 at `a4ab807`; R2 at `27b9218`; R3 and
R3-noedge at `18e7a6e`
**Protocol:**
- real KuaiRand-1K, snapshot protocol (daily snapshots, causal session prefixes);
- 1 epoch, seed 42, default flags;
- **validation only**: no run in this table scored test. R0's original checkpoint run did score
  test (before `--eval-test` existed); its validation numbers here come from an `--eval-only`
  rescore of that checkpoint.

Artefacts: `runs/spec01/<run>/` (`final_metrics.json`, `history.json`, checkpoint,
`train.log`); `runs/spec01/global_cvr_floor.json`.

## Results (validation)

| Run | Config | AUC | AP | LogLoss | nDCG@10 | AUC cold | AUC warm |
|---|---|---|---|---|---|---|---|
| floor | `global_cvr` alone (point-in-time) | 0.5894 | — | — | — | 0.5118 | 0.6327 |
| R0 | one-way relations, 200k cap, zero author/category input | 0.6821 | 0.5519 | 0.6181 | 0.5809 | 0.6689 | 0.6785 |
| **R0-noedge** | R0, no edges (features only) | **0.6942** | 0.5656 | **0.6098** | 0.6070 | **0.6787** | 0.6944 |
| R1 | + reverse relations (capped) | 0.6609 | 0.5195 | 0.6324 | 0.5733 | 0.6512 | 0.6498 |
| R2 | + author/category ID embeddings (capped) | 0.6691 | 0.5355 | 0.6232 | 0.5923 | 0.6590 | 0.6600 |
| R3 | full spec: + uncapped time-masked edges, 24 h warmup | 0.6779 | 0.5447 | 0.6443 | 0.5980 | 0.6665 | 0.6681 |
| **R3-noedge** | R3, no edges (features only) | 0.6920 | **0.5680** | 0.6128 | **0.6214** | 0.6729 | **0.6966** |

Cold = the candidate video had 0–1 earlier interactions in its snapshot: 679,645 of 1,169,625
validation rows (58%). The bucket is identical in every run.

## Cost

| Run | Relation edges | Encode s/snapshot (mean) | Peak GPU (GB) | Params with gradients |
|---|---|---|---|---|
| R0 | ≤1.4M (sampled) | 0.34 | 34.8 | 83,073 |
| R0-noedge | 0 | 0.33 | 34.7 | 83,073 |
| R1 | ≤2.8M (sampled) | 0.57 | 34.9 | 83,073 |
| R2 | ≤2.8M (sampled) | 0.66 | 35.6 | 83,073 |
| R3 | 28,412,450 (master; masked per snapshot) | 0.71 | 37.1 | 83,073 |
| R3-noedge | 0 | 0.56 | 35.4 | 83,073 |

- Most of the ~35 GB peak is node-level tensors over 4.36M videos, not edges.
- R3's uncapped index fits on one 95 GB GPU with plenty of headroom.
- R0's numbers come from the eval-only rescore. It has the same architecture as R0-noedge, so
  the same 83,073 parameters receive gradients (the alignment module and head only; the
  encoders are frozen).

## Reading

- **The features-only controls agree.** R0-noedge 0.6942 vs R3-noedge 0.6920, as the spec
  predicted. With no edges the author/category embeddings are unreachable, so the remaining
  gap is the 24 h warmup difference plus seed noise.
- **With frozen random encoders, the graph hurts.** Every graph run is below its features-only
  control:
  - R0 vs R0-noedge: −0.012 AUC;
  - R3 vs R3-noedge: −0.014 AUC, plus worse calibration (LogLoss 0.644 vs 0.613).
  - The deficit is present for both cold and warm items. Warm items lose most under R3
    (0.668 vs 0.697).
- **The spec's changes move in the expected direction, but don't reach the control:**
  - Reverse relations alone (R1) *lower* AUC. Users now receive messages, but through a random
    5% edge sample and identical author/category vectors.
  - Author/category identity (R2) recovers part of that.
  - Removing the cap (R3) recovers more (0.661 → 0.669 → 0.678).
- **This is outcome 3 in the spec ("R3 ≈ R0"), in a stronger form:** a connected graph with
  random weights is *worse* than features alone. The most likely mechanism is that frozen
  random message passing mixes each node's informative features (`global_cvr`, counts, user
  profile) with randomly projected neighbour averages, and LayerNorm then renormalises the
  blend. The head reads a diluted version of the same features. This is consistent with the
  original L=1 > L=2 > L=3 pattern.
- **Implication for the next spec:** the graph has to be *learned* (end-to-end or pretrained)
  before it can add anything. The paper must also report the features-only control: without
  it, a trained-graph result can't be told apart from feature extraction.

## Caveats

- One epoch, one seed. Differences of about 0.01 AUC need the multi-seed harness (F13) before
  they're claimed.
- R3 and R3-noedge differ from R0, R1 and R2 in warmup (24 h vs 0), per D1. Compare R3 with
  R3-noedge, and R0 with R0-noedge.

## Other spec items

- Real-data test suite: **28 passed**.
  - `test_real_model_forward_backward` now exercises the training path: `build_model` →
    `encode_graph` → `forward_from_embeddings` → backward. Previously it called the legacy
    per-session readout, which took hours on CPU.
- Synthetic suite: **63 passed** (6 new spec-01 tests). The invariance test's logic is
  unchanged; only its `SnapshotStore` construction follows the new `rel_master` argument.
- Cache bug found and fixed (D2): the previous cache-validity check compared the edge count with
  `len(log_combined)`. 1,075 log rows involve videos absent from the video metadata, so the
  check never passed and the HKG was rebuilt on every run. That was harmless for correctness;
  it's now replaced by `HKG_BUILD_VERSION` plus node-count checks, with atomic writes.
