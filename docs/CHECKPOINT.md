# HUG — Revision Checkpoint

**Last updated:** 2026-10-06
**Branch:** `docs/technical-report-and-revision-plan` (changes below are uncommitted at time of writing)
**Purpose:** current state of the revision for whoever acts as architect / reviewer. Read this
first, then `docs/TECHNICAL_REPORT.md` (findings F1–F14) and `docs/REVISION_WORKFLOW.md`.

---

## Your role (architect / reviewer)

Review code changes and experimental results for correctness, data leakage, and whether the
evidence supports the paper's claims. Be adversarial about data snooping, and don't accept a
number until you know which protocol produced it.

## Project

- Paper: *HUG: A Heterogeneous Unified Graph Framework for CVR Prediction*. Rejected at RecSys
  2026 (#353); being reworked for The Web Conference.
- Inputs: `Reviews.txt`, `Rebuttal.txt`, `RecSys_2026_paper_353 (1).pdf`,
  `docs/TECHNICAL_REPORT.md`, `docs/REVISION_WORKFLOW.md`. The workflow doc predates the
  decisions below. **The user is not yet happy with the revision plan; methodology is still
  being redesigned.**
- Environment:
  - conda env `hkg-env` (`/home/dasgua3/miniconda3/envs/hkg-env/bin/python`)
  - torch 2.5.1, PyG, numpy 1.26, fuxictr 2.3.9
  - hardware: 2× 95 GB GPUs
- Data: KuaiRand-1K at `/home/dasgua3/HUG/KuaiRand-1K/data` (git-ignored).

## Model as implemented

**Graph (HKG).**
- Nodes: user, video, author, category, session.
- Edges:
  - user→video feedback: clicked / liked / hated / commented / forwarded, plus `interacted`;
  - user→author `follows`;
  - video→author `made_by`;
  - video→category `tagged_as`;
  - video→video `next_in_session`.

**HUG-Dual.**
- `StructuralGNN`: R-GCN, 7 relations, at most 200k randomly sampled edges per relation.
- `SequentialGNN`: GGNN, followed by an attention readout and a recency gate.
- `AlignmentModule`, then the MLP head. The module's "cross-attention" takes a softmax over
  heads with a single key, so it's degenerate (F6).

**HUG-Unified.**
- `SingleHGT` over the full graph, then concat(user, video), then the MLP head. It ignores
  session embeddings.
- Agreed framing: a **controlled ablation baseline, not a contribution**. To be fair it needs
  the same fusion/head and inputs as Dual, and a matched parameter count.

**The GNN encoders are never trained.** `encode_graph` runs under `@torch.no_grad()`, so only
the alignment module and head learn; the GNNs are random-init feature extractors. Conceptually,
HUG today is random-projection SIGN/SGC-style features fed to an MLP. The user is deciding
between three directions:
1. parameter-free aggregation;
2. end-to-end training;
3. per-view self-supervised pretraining (link prediction / next-item), then fusion.

**Other facts.**
- Author and category nodes have all-zero input features, so the KG side contributes almost
  nothing at L=1.
- Agreed to remove the KGA gate (never executes, F4), IPS (no-op, F5) and the recency gate.
  **None of these is removed from the code yet**: the user is still designing HUG.

## Decisions made with the user

- **Baselines:** DIN, BST and KGAT are removed. The new baselines are **TransAct** (KDD'23) and
  **WuKong** (Meta, 2024), both from FuxiCTR's `model_zoo` (pinned commit `b7dff73`). The
  models Reviewer 2 named (InterFormer, OneTrans, HyFormer) have no official code, only
  unofficial ports.
- **Leakage rule (user):** *"If we predict at time t, we can only learn from data up to t-1."*
  As implemented:
  - every **input** for a row at time t comes from interactions with `time_ms < t`;
  - model **weights** are fit on the full training split, all of which comes before
    validation and test (standard offline training).
- **Datasets to add** (recommended, not yet downloaded or integrated): Taobao Display Ad
  (Tianchi/Alimama) and MIND.

## Changes in this checkpoint

### Removed
- `Baselines/KGAT/`, `Baselines/evaluate.py`, `Baselines/evaluate_standalone.py`,
  `Baselines/BASELINE_STEPS.md`
- `runs/{BST,DIN,1k_kgat_*}`, `runs/archive/{BST,DIN}*.log`
- KGAT sections of `run_experiments.sh`; DIN/BST/KGAT entries in `plot_results.py` and
  `Plots/view_results.py`
- `training_window()`, superseded by snapshots

### `Framework/data_loader.py`
- `video_features_statistic_*.csv` is no longer read: it aggregates the whole period,
  including test.
- `chronological_cutoffs` / `assign_split`: a global time-based 70/10/20 split, where tied
  timestamps never straddle a cutoff. Shared with the baselines.
- `compute_video_statistics(log)`: per-window popularity statistics.
- `KuaiRandData.n_sessions`.

### `Framework/hkg_constructor.py`
- Every behavioural edge carries `edge_time`. For `next_in_session` it is the later item's
  time.
- The builders are vectorised. Per-session truncation of transitions was removed, so the
  graph keeps all consecutive transitions.
- `video_feature_matrix()`.
- `snapshot_bundle(bundle, cutoff, video_x)`: drops timed edges with `edge_time >= cutoff` and
  swaps in statistics computed from that window.

### `Framework/temporal.py` (new)
- `build_interactions`: rows sorted by time, plus a causal session prefix for each row (at most
  `--max-seq-len` items, same session, strictly earlier time).
- `snapshot_boundaries` / `assign_snapshots`: a daily grid by default (`--snapshot-hours`), with
  `t_val` and `t_test` inserted. A row at time t uses the snapshot at a boundary `b_k <= t`.
- `SnapshotStore`: per-snapshot video features and sampled relation edges, computed once.
- `PrefixBatcher`.

### `Framework/gnn_encoders.py`
- `SequentialGNN.item_hidden()`: GGNN item states, without the O(sessions × E) readout loop.
- `SequentialGNN.prefix_readout()`: the same attention and recency-gate math as
  `_session_readout`, applied per row to its prefix.
- `SingleHGT.forward(with_sessions=False)`.

### `Framework/models.py`
- `encode_graph` returns `s_user`, `s_video`, `q_video` and `q_hidden`.
- `forward_from_embeddings` takes `prefix_items [B, L]` (-1 = padding). The session embedding
  is pooled per row under `no_grad`, so the encoders stay frozen exactly as before.

### `Framework/main.py`
- The timed HKG is built once and cached as `hkg_bundle_timed.pkl`.
- Train, val and test are processed snapshot by snapshot; for train, both the snapshot order
  and the row order are shuffled.
- Checkpoint selection uses validation AUC only. Test is evaluated once, from the
  val-selected checkpoint.
- `final_metrics.json` goes in the run-specific directory (F12). Checkpoints are loaded only
  with `--eval-only`.
- Flags:
  - new: `--val-ratio`, `--snapshot-hours`, `--warmup-hours`;
  - removed: `--chunk-size`, along with the 27K chunked and dual-GPU encode paths;
  - `--multi-gpu` still works for the dual model via `split_across_gpus`.
- The encoders always run in eval mode while encoding.

### `Framework/tests.py`
- `TestCausality` checks:
  - snapshot edges are older than the cutoff, and snapshot pairs are a subset of the past;
  - per-snapshot video features equal statistics computed from `log < cutoff`;
  - prefixes contain only strictly earlier items;
  - no snapshot period mixes splits.
- **End-to-end invariance test.** Every row at or after time T is rewritten (all feedback
  flipped, videos shuffled), the whole pipeline is rebuilt, and predictions for rows before T
  must be bit-identical. The test also asserts that rows after T *do* change, so it can't pass
  vacuously.
- Real-data test: 1,000 test-only (user, video) pairs must be absent from the graph.

### `Baselines/`
- **`preprocess.py`**
  - Uses HUG's loader, filters and split, so the rows are identical to HUG's.
  - Inputs per row:
    - behaviour sequence: the last 50 **clicked** videos strictly before the row;
    - item statistics: point-in-time, from the video's strictly earlier rows;
    - context: tab and hour;
    - user and video categorical features.
- **`train.py`**
  - FuxiCTR training with early stopping on validation, then the best weights are reloaded.
  - Val/test are scored with HUG's `compute_metrics` (AUC, AP, LogLoss, per-user nDCG@10).
  - Writes `runs/<Model>[_seed<n>]/final_metrics.json`.
  - Flags: `--model`, `--gpu`, `--seeds`, `--epochs`, `--runs-dir`.
- `setup_fuxictr.sh` (pinned clone plus dependencies) and `run_all.sh`.
- `config/dataset_config.yaml`: two dataset IDs over the same CSVs —
  - `kuairand_1k_seq`: raw sequence, for TransAct;
  - `kuairand_1k`: mean-pooled sequence, for WuKong.
- `config/model_config.yaml`: `embedding_dim` 64, monitoring validation AUC.

## Split (identical for HUG and baselines, verified)

| Split | Rows | Click rate |
|---|---|---|
| train | 8,187,361 | 0.379 |
| val | 1,169,625 | 0.380 |
| test | 2,339,250 | 0.371 |

`t_val = 1651305018060`, `t_test = 1651544133374`. HUG's `main.py` and `Baselines/preprocess.py`
log identical cutoffs and row counts.

## Verification status

| Check | Status |
|---|---|
| Synthetic test suite (incl. `TestCausality`, invariance test) | 57 passed, 28 skipped (real-data tests need `--data-dir`) |
| `main.py` dual + single, synthetic data | runs end to end |
| HUG and baseline splits match on real data | yes (table above) |
| TransAct, 1 epoch, real data | runs; see note below |
| WuKong, 1 epoch, real data | runs; val AUC 0.643 / LogLoss 0.735, test AUC 0.632 / LogLoss 0.672; train loss 0.328 (197M params) |
| Real-data test suite (vectorised build) | 27 passed in 90 s, incl. 1,000 test-only pairs absent from the graph. `test_real_model_forward_backward` excluded: it calls the legacy end-to-end `forward()` with the per-session Python loop (F11) and runs for hours on CPU; the training pipeline no longer uses that path |
| HUG-Dual, 1 epoch, real data, snapshot protocol | runs; AUC train 0.686 / val 0.682 / test 0.682, LogLoss 0.612 / 0.618 / 0.615. HKG build 42 s, 34 daily snapshots prepared in 2 min, 212 s/epoch, 55 GB GPU reserved |

**TransAct note.**
- With whole-window item statistics: validation AUC 0.546, LogLoss 3.7.
- Cause: each row's own label was inside its statistics. `global_cvr` alone scored AUC 0.79 on
  train but 0.57 on validation. Switching to point-in-time statistics fixed it.
- After the fix: validation AUC 0.666, test AUC 0.651, but LogLoss about 1.7, so the model is
  badly calibrated. The likely cause:
  - 224M parameters, almost all ID embeddings;
  - only 33% of validation rows involve a video seen at least twice in training.
- This needs regularisation and capacity tuning on validation. The checks so far don't
  indicate leakage.

**Baselines vs HUG — train/eval gap.** Both baselines reach a training loss of about 0.33 in one
epoch, but validation LogLoss is 0.74 (WuKong) and 1.7 (TransAct). HUG's train and validation
losses match (0.612 / 0.618). The baselines' inputs are causal. The gap comes from shuffled
training of ID embeddings: a video's embedding learns that video's later click rate, which helps
predict its earlier training rows but doesn't transfer to the 67% of validation rows whose
videos are rarely or never seen in training. HUG has no ID embeddings. Whether baselines should
be trained in time order (one pass, no shuffling) to follow the "learn only from t-1" rule
literally is an open decision (see open issue 8).

**HUG note.** Train and validation agree (no train-only shortcut) and LogLoss is calibrated,
unlike TransAct before its statistics fix. Encoders are still frozen at random init.

**These are 1-epoch smoke numbers, not results.** Every number in the paper and under `runs/`
from before this checkpoint was produced under the leaky protocol and is void.

## Open issues to review or decide

1. **HUG methodology:** frozen random encoders, trained, or pretrained (the user's open
   question). The causal plumbing is independent of this choice.
2. **Capacity parity:** TransAct has 224M parameters (mostly ID embeddings); HUG has no ID
   embeddings. Sweep `embedding_dim` and regularisation on validation (workflow 1.3).
3. **Information asymmetry:** the baselines see click history up to the moment of
   prediction. HUG sees the graph as of the start of its period (up to `--snapshot-hours`
   stale) plus the in-session prefix. Both are causal; decide whether the gap is acceptable
   or the period should shrink.
4. **Residual full-period information** (not label leakage):
   - the min-interactions user filter counts the whole log;
   - `user_features_1k.csv` is a single profile snapshot of unknown date;
   - video upload metadata is static.
5. **Known defects still present:**
   - F6 (degenerate cross-attention);
   - F8 (both encoders share depth);
   - F10 (HGT softmax normalisation);
   - F13 (single seed; no multi-seed harness yet);
   - random 200k-edge sampling per relation;
   - HUG-Unified is not yet a fair control (no session input, different head).
6. **Behaviour changes to confirm:**
   - the transition graph is no longer truncated to the last 50 items per session;
   - the prefix readout keeps duplicate items (the old readout deduplicated session nodes);
   - with `--warmup-hours 0` (the default), first-day training rows are encoded from
     near-empty graphs.
7. Stale artefacts: the old `hkg_bundle.pkl` caches are ignored, and the old checkpoints in
   `runs/` are void.
8. **Training order:** inputs are causal everywhere, but training rows are shuffled, so weights
   (especially ID embeddings) learn from training rows later than the row being fit. That is
   standard offline training and cannot affect val/test (all training data precedes them).
   Time-ordered single-pass training would satisfy "learn only from t-1" literally. User to
   decide.

## How to run

```bash
# HUG (dual; add --model-type single for HUG-Unified)
python Framework/main.py --data-dir KuaiRand-1K/data --cache-dir Framework/cache/1K \
    --output-dir runs --device cuda:0

# Baselines
bash Baselines/setup_fuxictr.sh
python Baselines/preprocess.py --data-dir KuaiRand-1K/data
python Baselines/train.py --model all --gpu 0

# Tests
cd Framework && python -m pytest tests.py -q                        # synthetic
cd Framework && python -m pytest tests.py -q -k TestRealData \
    --data-dir ../KuaiRand-1K/data                                   # real data
```
