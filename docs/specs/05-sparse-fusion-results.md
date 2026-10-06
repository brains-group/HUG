# Spec 05 results: fusion smoke (§G)

**Status:** smoke item 1 done (one epoch of every fusion head). Item 2 (Stage 1b queue dry
run + aggregation) and the controlled overhead measurement are **pending GPU time** (see §5).
**Date:** 2026-10-06. Code: `1adc1b9` (spec 05 amendments) on
`docs/technical-report-and-revision-plan`; runs in `~/HUG/runs/dev/fusion/` (`chain.sh`).
**Protocol:** KuaiRand-1K, N4 pre-tuning defaults, seed 42, one epoch, validation only (no
`--eval-test`). One epoch per head is a sanity check, not a comparison: tuning comes in Stage 1b.

## 1. Amendments implemented before the smoke

| Note | Change | Test |
|---|---|---|
| 1 | `metadata_evidence`: a missing neighbour contributes 0 (no pooled placeholder group) | test 2 (+ missing-neighbour test) |
| 2 | `evgate` gates on `s_v = clip(n_v / n_ref,v, 0, 1)`, same `n_ref` buffer as `sparse` | `test_evgate_uses_normalised_evidence` |
| 3 | `n_ref_set` flag buffer; training-mode forward raises until `set_n_ref`; `n_ref` round-trips `best.pt`, `last.pt`, `--resume` | `test_training_without_n_ref_raises`, `test_n_ref_round_trips_checkpoint_and_resume` |
| 4 | dead-atom resampling zeroes Adam `exp_avg`/`exp_avg_sq` of the reset dictionary and encoder rows | test 6 |
| 5 | `n_S`'s history term = `log1p(hist_end − hist_start)`, the encoder's own (click-only, truncated to `max_seq_len`) history. Before it was the untruncated click count: same definition, different cap | `test_sequence_evidence_uses_encoder_history` |

Concat stays bitwise identical (test 7); `CODE_COMPAT_VERSION` unchanged.

## 2. One-epoch results

| head | val AUC | holdout AUC | val LogLoss | train BCE | train aux | dense params |
|---|---|---|---|---|---|---|
| concat (= N4) | 0.7725 | 0.7884 | 0.5511 | 0.5605 | — | 244,831 |
| gate | **0.7729** | 0.7915 | 0.5492 | 0.5517 | — | 208,415 |
| evgate | 0.7718 | **0.7918** | 0.5504 | 0.5498 | — | 212,703 |
| moe | 0.7714 | 0.7913 | 0.5506 | 0.5507 | 0.010 | 447,618 |
| misa | 0.7189 | 0.7203 | 0.5918 | 0.5957 | 3.54 | 339,935 |
| sparse (adaptive) | 0.7418 | 0.7820 | 0.5767 | 0.5576 | 6.66 | 496,095 |
| sparse (fixed, k = 27) | 0.7435 | 0.7821 | 0.5852 | 0.5593 | 6.45 | 496,095 |

(Concat reproduces the spec 04 N4 dev run, 0.7720.) Peak GPU memory is 38.4 GB for every
head: the graph view dominates, the head adds nothing measurable.

### Reading

1. **The projection to `fusion_dim` = 64 is not a bottleneck**: `gate`, `evgate` and `moe` match
   concat after one epoch (0.7714–0.7729; `moe`'s load-balancing aux is 0.01).
2. **The heads with auxiliary losses lose 0.03–0.05 val AUC**, and their auxiliary loss is
   6–12× the BCE. `L_rec` is a squared error summed over the 64 LayerNorm dimensions
   (≈ 64 at initialisation), weighted 1.0. In `sparse`, the encoder `E_v` is shared by the task
   path and the reconstruction path, so the reconstruction gradient dominates the codes the
   head reads. MISA has the same scale problem through its decoder/encoders.
3. **The sparse gap is not evidence drift.** `sparse` trains *better* than concat
   (BCE 0.558 vs 0.561) and is close on the in-period holdout (0.782 vs 0.788), but much worse
   on val (the next period). I first suspected the evidence budget (counts grow over time, so
   val rows get larger `k_pr`: mean G 21.7 → 23.1, S 33.1 → 34.4), but the fixed schedule
   loses the same (0.7435). So the generalisation gap comes from the sparse codes themselves.
4. **No low-evidence advantage**: sparse loses ≈ 0.03 in every `video_train_count`,
   `coldwarm` and `history_len` bucket except users with no history (0.927 vs 0.907 concat).

Suggested before tuning (architect's call): put `L_rec` on a per-dimension mean (÷ d') or
scale `w_rec` to ≈ 0.01–0.02 so `aux` is comparable to the BCE, and include `w_rec ∈
{0.01, 0.05}` in the tuning grid (today {0.1, 1.0}). Same for MISA.

## 3. Sparse diagnostics (spec thresholds)

**Budgets vary as designed.** Mean active private codes per `n_G` quintile (val):
G-private 5.2 / 14.4 / 22.4 / 28.9 / 32.3; S-private grows with `n_S` likewise; shared blocks
stay at k_sh = 16. Shared-mass ratio ρ: 0.27 mean, falling with evidence (n_G quintiles
0.34 → 0.23; cold 0.31 vs warm 0.22). Fixed schedule: ρ = 0.18, k = 27 for every row.

**Co-activation: low (spec: report before tuning).**

| | adaptive | fixed |
|---|---|---|
| shared atoms that fire for both views on ≥ 1% of their firings | **4.2%** | 7.8% |
| mean per-row Jaccard of the two views' active shared sets | 0.083 | 0.112 |

The shared dictionary has split into view-specific atoms: the "split dictionary" failure the
spec warns about. `L_align` (InfoNCE on the shared codes, weight 0.1) is not enough to keep it
shared after one epoch.

**Dead atoms.** The in-training tracker reports 0% dead, but that number is meaningless at this
scale: an atom is dead after 10,000 idle steps and checked every 1,000, while one epoch is
≈ 913 steps at batch 8,192 (a 20-epoch run is ≈ 18k steps; resampling could start only after
≈ 11 epochs). Measured instead on the 200k sampled validation rows (`val_codes.npz`):

| never fired on 200k val rows | adaptive | fixed |
|---|---|---|
| shared, graph view | **48.0%** | 32.8% |
| shared, sequence view | **40.6%** | 32.0% |
| shared, either view | 17.6% | 10.5% |
| private G | 26.2% | 31.6% |
| private S | 27.7% | 32.4% |

Several dictionaries are above the spec's 30% line. **The dead-atom window should be set in
rows or epochs, not steps** (e.g. dead = no firing in the last ~10M training rows ≈ 1.3
epochs; check every ~100 steps). Spec decision.

## 4. Overhead

The per-epoch seconds are **not usable**: the GPUs are shared with other users' jobs and the
architect's dev chain, so `gate` (1711 s) and `sparse` (1567 s) ran *faster* than concat
(2195 s). `scripts/fusion_overhead.py` times identical batches for every head in one process,
interleaved over 3 rounds; it needs ~10 min on one GPU (§5).

## 5. Pending (GPU)

- Controlled overhead (`python scripts/fusion_overhead.py --device cuda:X --out runs/dev/fusion/overhead.json`), target < 25% over concat.
- Stage 1b dry run: `python scripts/run_queue.py experiments/heavy_run.yaml --dry-run --max-steps 50 --only 'tune_fusion_*' --only 'hug_sparse*' --only 'hug_gate*' --only 'hug_evgate*' --only 'hug_moe*' --only 'hug_misa*' --only 'final_eval_hug_sparse*' …`, then `python scripts/aggregate.py --runs-root runs/heavy_dryrun`. Each KuaiRand HUG job needs ~38 GB of GPU memory even at 50 steps (the full graph), so it can't share a GPU with a Stage 1a job.
