# Spec 04: Pipeline readiness for the heavy run

**Status:** ready for implementation
**Author:** architect session, 2026-10-06
**Builds on:** Spec 01 (landed, `984bb01`). **Implements Spec 03's model design.** Spec 03
(`03-modern-encoders.md`) stays the design reference for the new model. Its "Experiment arms"
section is **deferred to the heavy run**; don't run it as part of this spec.

**Goal:** bring HUG, the baselines, the tests and the experiment harness to the point where one
command launches the full multi-day run (HUG arms, tuned baselines, seeds), and it finishes
unattended with resumable, comparable, validation-selected results.

**This spec runs smoke tests and diagnostics only.** It produces no results tables and no
final ablations. Test-set numbers aren't produced anywhere in this spec.

---

## Contents

- W0 Housekeeping and protocol decisions
- W1 HUG model (Spec 03 implementation, with amendments)
- W2 HUG-Unified: retire from the main comparison
- W3 Baselines: fix the strawman comparison
- W4 Experiment harness (queue, resume, aggregation, test-set discipline)
- W5 Tests
- W6 Smoke validation and run-time estimate
- W7 Docs and the heavy-run runbook
- Heavy-run plan (prepared and dry-run here, executed later)
- Out of scope; Done when

Commit each workstream separately. Order: W0 → W1 and W3 in parallel → W4 → W5 throughout →
W6 → W7.

---

## W0. Housekeeping and protocol decisions

1. **Environment pin.** Update `environment.yml` with the versions actually in use: torch
   2.5.1, PyG and its sparse extensions, numpy 1.26, pandas, scikit-learn, and fuxictr 2.3.9
   with the pinned model-zoo commit `b7dff73`, as `setup_fuxictr.sh` does. Log the versions at
   the start of every run.
2. **Determinism.** Add `set_determinism(seed)`, used by HUG and the baseline wrapper: seeds
   for python, numpy and torch (CPU and CUDA), plus `torch.backends.cudnn.benchmark = False`.
   Full CUDA determinism isn't required; on GPU, seeds only need to be reproducible to within
   noise. The CPU tests in W5 rely on bitwise determinism.
3. **User filter (open issue 4).** `min_interactions` currently counts the whole log.
   - Measure how many users it drops on 1K.
   - If none: document that in `CHECKPOINT.md` and leave the code alone.
   - If some: count only rows before `t_val`, with cutoffs computed on the unfiltered log, then
     re-verify that the HUG and baseline splits still match (row counts and cutoffs).

   Either way, do this **before** W3's preprocessing is regenerated, so it doesn't have to be
   redone.
4. **Dataset isolation (prep for a second dataset, which isn't added here).** Everything
   KuaiRand-specific stays behind the loader and the shared feature functions (W3.2). The model,
   harness and tests must not hard-code KuaiRand column names outside those modules. One
   grep-style test (W5) enforces this for `Framework/hug.py`.
5. **Artefact size.** Model checkpoints stay git-ignored. Keep only `best.pt` plus `last.pt`
   (for resume) per run. Delete `last.pt` when a run completes. Log checkpoint size.
6. **Quiet logs.** When stdout isn't a TTY, or with `--quiet`, turn tqdm off and log one line
   per N steps instead. Spec 01's `train.log` files are 5,000 lines of progress bars.

## W1. HUG model: implement Spec 03, with these amendments

Implement Spec 03 sections A–E exactly as written: `Framework/hug.py` with `InputEncoder`,
`RelLightGCN` plus the XSimGCL term, `HistoryTransformer` and `HUGModel`;
`--model-type hug` as the default; legacy models untouched. The amendments below take
precedence over Spec 03 where they differ.

1. **Per-row as-of video statistics (information parity).** The baselines get *per-row*
   point-in-time video statistics (`asof_video_statistics`, computed from strictly earlier
   rows). HUG only has daily snapshot statistics on video nodes, which can be up to 24 h stale.
   - Add the same per-row as-of statistics to HUG's `ctx` vector, computed by the **shared**
     function (W3.2).
   - Keep the snapshot statistics on the graph nodes.

   This closes most of open issue 3. The per-row history plus per-row statistics are fresh; only
   graph structure lags.
2. **One shared history function.** As in Spec 03 §C, the function lives in `temporal.py` and
   returns index arrays. `Baselines/preprocess.py` calls the same function and turns the arrays
   into strings for FuxiCTR.
3. **Arm switches as flags,** so every heavy-run arm is one config (W4):
   - `--no-graph`: drops the graph view from `z` and skips the encoder entirely;
   - `--no-seq`: drops `s_seq` and skips the transformer;
   - `--freeze-graph`: encoder under `no_grad`, uniform `a`, `λ_cl` forced to 0;
   - `--cl-weight`;
   - `--graph-tokens`: optional G-tok arm;
   - `--structural-encoder {rellightgcn,rgcn}`: optional M1 arm.

   `--no-graph --no-seq` is arm N0.
4. **Resume.** Write `last.pt` at the end of every epoch, containing model, optimizer,
   scheduler, RNG states, epoch, best val AUC, early-stopping counter and history.
   `--resume` continues from it. A resumed run must match an uninterrupted one bitwise on the CPU
   synthetic test (W5).
5. **Early stopping and scheduler.** Early-stop on val AUC with patience 2, max 20 epochs, then
   restore the best weights. Replace the cosine `T_max=epochs` scheduler, which is wrong under
   early stopping, with a constant LR plus `ReduceLROnPlateau` on val AUC (factor 0.5,
   patience 1).
6. **`--snapshot-hours` cost probe.** In W6, measure the per-epoch cost at 24 h and 6 h. The
   heavy run uses 24 h unless 6 h costs under 1.5× as much. Report both.
7. **Bucket metrics.** HUG and the baselines compute the bucket metrics through one function in
   a shared module, `Framework/metrics.py`. Move `compute_metrics` and `_per_user_ndcg_at_k`
   there as well.

   | Bucket | Values | Source |
   |---|---|---|
   | candidate video's training-window count | `0`, `1–4`, `≥5` | train window |
   | Spec 01 cold/warm | — | as Spec 01 |
   | user history length at the row | `0`, `1–9`, `10–49`, `50` | per row |

   The per-row bucket keys come from the shared feature functions, so they're identical for
   every model.

## W2. HUG-Unified: retire from the main comparison

In the new design, the decomposition question is answered by HUG's own arms:

| Arm | What it is |
|---|---|
| N1 | sequence view only |
| N2 | graph view only (the graph includes `next_in_session` transitions, so N2 is the single-graph model over all relations) |
| N4 | both |

The HGT-based `single` model is no longer a fair or needed control (F10, no session input, a
different head).

- Remove `single` from the heavy-run plan and from `run_experiments.sh`. Keep the code so
  older runs can be reproduced.
- Note in `CHECKPOINT.md` that N2 replaces HUG-Unified as the single-graph control.
- *(Architect's note: the user should confirm this. If they want HGT as an external
  heterogeneous-GNN baseline, it would use PyG's `HGTConv` in the baseline harness, not the
  custom layer. That isn't in this spec.)*

## W3. Baselines: fix the strawman comparison

Problem: untuned TransAct (val AUC 0.666, LogLoss 1.70) and WuKong (0.643, 0.735) are below
HUG's features-only control (0.694). Their training loss is about 0.33, against validation
LogLoss of 0.74 and 1.7. As it stands, the comparison would be rejected as a strawman.

### W3.1 Leak-or-drift diagnostic (run in W6; results in the W6 report)

- Carve a fixed **training holdout**: a random 5% of training rows, seed 0, saved as a row-id
  list, and excluded from fitting **for every model** (HUG and baselines alike). It's *in-period*:
  same users, same videos, same time window as training.
- In W6, score the 1-epoch TransAct and WuKong models and HUG N0 and N4 (smoke) on the
  holdout. How to read the result:

  | Holdout result | Reading |
  |---|---|
  | holdout LogLoss ≈ training loss (~0.33) ≪ val | the gap is temporal drift plus cold items. Legitimate; tuning addresses it |
  | holdout ≫ training loss | something lets training rows see their own labels. **Stop** and find it before continuing W3 |

- Extra checks to include in the report:
  - the share of (user, video) pairs that repeat in training;
  - label purity per video in training: the fraction of videos whose training clicks are all 0
    or all 1, weighted by rows;
  - one-feature AUCs on the holdout and on val, for `video_id`-only (a training-window target
    encoding) and `global_cvr` as-of.
- The heavy run keeps the holdout excluded, so validation stays the only selection signal and the
  holdout stays a clean in-period reference. Report it as a diagnostic column, never as a
  headline metric.

### W3.2 Shared feature functions (single source of truth)

Create `Framework/features.py`, imported by `Baselines/preprocess.py`, with:

- `click_history` (moved from `preprocess.py`, returning index arrays; see W1.2);
- `asof_video_statistics` (moved from `preprocess.py`);
- `hour_of_day` (Beijing-time formula), `tab`, and the video categorical encodings
  (`VIDEO_CAT_COLS`);
- the bucket keys for W1.7;
- `holdout_rows(train_rows, frac, seed)`.

After this, `preprocess.py` contains no feature logic of its own; it formats outputs for
FuxiCTR.

### W3.3 Tuning harness (built and smoke-tested here; the sweep runs in the heavy run)

- `Baselines/tune.py`: a fixed grid or random search over a declared space, a **fixed trial
  budget per model**, selection on val AUC only, holdout excluded.
- It writes `runs/tune/<Model>/trial_<i>/` and `runs/tune/<Model>/best.yaml`.
- Search space, the same size for every baseline:

  | Parameter | Values |
  |---|---|
  | `embedding_dim` | {16, 32, 64} |
  | `min_categr_count` | {2, 5, 10, 20} |
  | `embedding_regularizer` | {0, 1e-6, 1e-5, 1e-4} |
  | `net_dropout` | {0, 0.1, 0.2, 0.3} |
  | `learning_rate` | {1e-3, 5e-4} |
  | `epochs` | up to 20 with early stopping (patience 2), plus a forced **one-epoch** variant per trial (the "one-epoch phenomenon" common in CTR with ID embeddings) |

- Budget: **16 trials per model** (random search, fixed seed). The same budget applies to HUG
  N4 and N1 through `Framework/tune_hug.py`, over Spec 03's tuning space: `emb-dim`,
  `min-id-count`, `graph-layers`, `cl-weight` ∈ {0.05, 0.1, 0.2}, `seq-layers` ∈ {1, 2}, and
  dropout. Equal budgets go in the paper.
- **Check the vocabulary.** `min_categr_count` thresholds must be computed on training rows
  only. Verify FuxiCTR's `FeatureProcessor` fits on `train_data` only, and add a test (W5).

### W3.4 Baseline roster for the heavy run

| Baseline | Role | Status |
|---|---|---|
| TransAct (KDD'23) | modern sequential CTR, the most direct comparison for the sequence view | exists; tune |
| WuKong (2024) | modern scaling CTR (feature interactions) | exists; tune |
| **FiGNN** (CIKM'19) | GNN-based CTR, filling the graph-model slot KGAT left | **add**: in the pinned FuxiCTR model zoo; dataset config `kuairand_1k` (no sequence) |
| *DCNv2* (WWW'21) | widely known feature-interaction reference; cheap | **add, optional**: config only; include if the W6 time estimate allows |

- The external graph-recommender option (official XSimGCL/LightGCN trained two-stage, then a
  CTR head) is **not** in this spec. N2 (our trained `RelLightGCN` + XSimGCL, graph only, with
  the same inputs) fills that role, disclosed as such. The user may still ask for an external
  run later.
- Every baseline uses **the same row inputs** as HUG: identical history, as-of statistics,
  `tab`/`hour`, and user and video categorical features, all from the shared functions.
  FiGNN and DCNv2 get no sequence features (they can't use them); TransAct gets the raw
  sequence; WuKong the mean-pooled sequence (as now).
- `train.py` must write HUG's metric schema, including the W1.7 buckets and the holdout
  diagnostic, via `Framework/metrics.py`.

### W3.5 Baseline robustness

- `train.py --resume` (or just idempotent): rerunning a finished trial is a no-op (W4
  registry).
- Delete FuxiCTR checkpoints except the best one. Write each trial's resolved config into its
  run directory.

## W4. Experiment harness

### W4.1 Run registry and queue

- **`experiments/heavy_run.yaml`:** the complete heavy-run plan (below), as a list of jobs. Each
  job has a name, a kind (`hug` | `baseline` | `tune_hug` | `tune_baseline` | `final_eval`),
  args, a seed, GPU memory class, and dependencies (e.g. `final_eval` depends on its tuning
  job).
- **`scripts/run_queue.py`:**
  - runs the jobs across 2 GPUs as a queue, respecting dependencies;
  - one job per GPU by default; `--pack` allows two light jobs per GPU when the memory classes
    allow it;
  - **skips** jobs whose output directory has a `final_metrics.json` with a matching
    **config hash**: the hash of the resolved args, the git commit of `Framework/`/`Baselines/`,
    and the data fingerprint (cutoffs plus row counts);
  - resumes interrupted HUG jobs with `--resume`; restarts baseline trials;
  - writes per-job stdout/stderr logs and appends status lines to `runs/heavy/queue.log`;
  - retries a failed job once, then marks it failed and continues;
  - writes `runs/heavy/status.json` (jobs done/running/failed/pending, and an ETA from the W6
    per-epoch timings);
  - `--dry-run --max-steps 50`: runs every job for 50 training steps plus a val pass on 2
    snapshots, to prove every config starts, trains, evaluates and writes outputs. W6 uses this.
- Everything goes under `runs/heavy/<job>/`.

### W4.2 Test-set discipline (enforced, not by convention)

- **Stage 1** jobs (tuning, arms, seeds) run without `--eval-test`.
- **Stage 2** `final_eval` jobs load the val-selected checkpoint of each reported config and
  score test once. They're the only jobs allowed to pass `--eval-test`.
- `--eval-test` refuses to run unless the job comes from a `final_eval` entry. The launcher
  passes a token; manual use needs `--i-know-this-touches-test`.
- Every test score is appended to `runs/heavy/test_access.log`: job, config hash, timestamp.

### W4.3 Aggregation

`scripts/aggregate.py` turns `runs/heavy/**/final_metrics.json` into:

- `runs/heavy/summary/val_table.md` and `.csv`: per arm/baseline, mean ± std over seeds of AUC,
  AP, LogLoss and nDCG@10, plus the holdout diagnostic column;
- `bucket_tables.md`: AUC per bucket (W1.7), per model;
- `significance.md`: the paired difference of each model vs. N4 (and N4 vs. N1, N2, N4-frozen).
  It uses a **per-user paired bootstrap of AUC** (1,000 resamples over users, the same
  resamples for both models; scores read from the saved per-row predictions), plus a per-seed
  paired t-test where there are ≥3 seeds;
- `test_table.md`: **only** from Stage 2 jobs, never mixed with val tables.

Every run saves its per-row val predictions (`val_preds.npy`: row id, score), so significance
testing doesn't need reruns. Stage 2 saves `test_preds.npy` the same way.

## W5. Tests

The synthetic suite must stay under 5 minutes on CPU, with the real-data suite separate. All
Spec 01 tests and the existing invariance test must keep passing.

**From Spec 03 (all required):**

1. History parity with the baselines: 10,000 real rows match item for item; every history
   item's time is strictly before the row's time (synthetic).
2. Gradient routing (N4 vs. N4-frozen).
3. `RelLightGCN` matches a hand computation, and destination-sized outputs equal full-size
   ones.
4. Snapshot degrees come from masked edges only.
5. The vocabulary comes from the training window only (OOV for videos seen only in val/test).
6. **Weights don't depend on val/test:** rewrite every row at or after `t_val`, retrain N4 for
   2 epochs (CPU, deterministic), and all parameters are bit-identical.
7. Prediction invariance: `test_predictions_invariant_to_future` extended to `hug` with fixed
   weights.
8. The contrastive term uses only batch nodes; `λ_cl=0` leaves exactly BCE + L2.
9. Eval is deterministic.

**New in this spec:**

10. **As-of statistics are strictly earlier, including ties.** For synthetic rows, including
    several rows of the same video at the same timestamp, each row's as-of statistics equal a
    brute-force count over rows with `time < t`. Rows sharing a timestamp don't see each other.
11. **Feature parity, HUG vs. baselines.** For 10,000 real rows, the values HUG feeds in (`tab`,
    `hour`, video categorical codes, as-of statistics, history) equal the corresponding columns
    in the baseline CSVs.
12. **The holdout is disjoint and excluded.** Holdout rows ⊂ training rows, the set is
    deterministic given the seed, and they never appear in a HUG training batch or in the
    baseline training CSV.
13. **Baseline vocabulary fits on training only.** FuxiCTR's feature processor sees only the
    training file. A synthetic category value that appears only in val maps to OOV.
14. **Bucket keys are shared.** HUG and the baseline wrapper produce identical bucket keys
    for the same rows (both come from `features.py`).
15. **Resume is exact.** On CPU synthetic data, training 3 epochs straight gives the same
    parameters, history and selected epoch, bitwise, as 1 epoch + kill + `--resume` for 2 more.
16. **The run queue works.**
    - Config hashes are stable across runs and change when args, code or data change.
    - A finished job is skipped; an interrupted job is resumed.
    - Dependencies are respected.
    - A failed job is retried once, then marked failed without stopping the queue.

    Use stub jobs that write fake outputs.
17. **The test-set guard holds.** `--eval-test` outside a `final_eval` job raises an error, and
    every allowed access is logged.
18. **Aggregation is correct.** On synthetic `final_metrics` and prediction files, the means
    and standard deviations are correct. The paired bootstrap returns about 0 difference for
    identical predictions and detects a planted difference.
19. **Dataset isolation.** `Framework/hug.py` contains no KuaiRand-specific column names
    (grep-based check against a list from the loader).
20. **Arm flags.** For each arm flag combination, the model builds, runs forward and backward
    on synthetic data, and skips the parameters it should (e.g. `--no-graph` builds no
    encoder and computes no snapshot matrices).

## W6. Smoke validation and run-time estimate (the only runs in this spec)

Real 1K data, validation only, no `--eval-test` anywhere.

1. **Diagnostics (W3.1):** 1-epoch TransAct and WuKong with current configs, plus HUG N0 and N4
   (1 epoch each), all trained with the holdout excluded. Report:
   - holdout vs. training vs. val LogLoss/AUC;
   - the extra checks (repeat pairs, label purity, one-feature AUCs).

   **If the result says "stop", stop the spec there and report.**
2. **One epoch of every HUG arm** in the heavy-run plan (N0, N1, N2, N3, N4, N4-frozen;
   optional arms if implemented). Report per arm: seconds per epoch, peak GPU memory,
   parameter count, val AUC after 1 epoch. These are **sanity numbers, not results**. Check
   that N4 trains (loss falls), that its `softmax(a_l)` table isn't stuck at uniform, and that
   the contrastive loss decreases.
3. **The `--snapshot-hours` probe** (W1.6): N4 one epoch at 6 h vs. 24 h; time and memory only.
4. **One tuning trial per baseline** (TransAct, WuKong, FiGNN, plus DCNv2 if added), and one
   `tune_hug` trial, all through the queue.
5. **Full dry run of the queue:** `scripts/run_queue.py experiments/heavy_run.yaml --dry-run
   --max-steps 50`. Every job must start, train, evaluate and write outputs. Then run
   aggregation over the dry-run outputs to prove the tables and significance code work end to
   end.
6. **Run-time estimate:** from the measured times, estimate the wall clock of the full plan on
   2 GPUs, assuming early stopping at a median of about 6 epochs (state the assumption).
   - If it's over **72 h**, propose cuts in the report, e.g. fewer tuning trials, dropping
     DCNv2, or 2 seeds for non-key arms.
   - Don't apply the cuts without the user's sign-off.

## W7. Docs

- `docs/specs/04-heavy-run-readiness-results.md`: the W0.3 user-filter finding, W6.1
  diagnostics, W6.2–W6.4 smoke tables, the dry-run outcome, the run-time estimate, and any
  proposed cuts.
- `docs/HEAVY_RUN.md` (runbook):
  - the exact launch command;
  - how to monitor (`status.json`, `queue.log`);
  - how to resume after a crash or reboot;
  - where outputs go;
  - expected duration and disk use;
  - how to run Stage 2 and aggregation;
  - what not to do (manual `--eval-test`; editing code mid-run changes config hashes).
- Update `docs/CHECKPOINT.md`: the new model, flags, harness, the W2 retirement of the HGT
  control, the verification table, and resolved or remaining open issues (3, 4, 5, 6, 8).
- Update the header of Spec 03 to say its design is implemented by Spec 04 and its experiment
  section is deferred to the heavy run.

## Heavy-run plan (`experiments/heavy_run.yaml`): prepared and dry-run, not executed

**Stage 1 (val only):**

1. **Tuning:**
   - `tune_hug` for N4 and N1: 16 trials each;
   - `tune_baseline` for TransAct, WuKong and FiGNN (plus DCNv2 if included): 16 trials each.
2. **HUG arms** at the tuned N4 settings (arm switches applied on top of them):
   - N1, N3 and N4 with seeds 42/43/44;
   - N0, N2 and N4-frozen with seed 42.

   N1 runs at *its own* tuned settings as well (as the TransAct-like comparison).
3. **Baselines** at their best tuned settings: 3 seeds each.

**Stage 2:** `final_eval` for each reported configuration, with every seed scored once on
test.

**Aggregation:** val tables, bucket tables, significance, then the test table.

## Out of scope

- Executing the heavy run (Stages 1 and 2) or writing any results tables.
- Fusion research (shared/private sparse codes); that's the next method spec, built on N4.
- A second dataset (Taobao Display Ad or MIND). W0.4 keeps the code ready for it; the loader
  is a separate spec.
- An external XSimGCL/LightGCN baseline, and an external HGT baseline (W2/W3.4 notes).
- Removing legacy model code.

## Done when

- [ ] W0: environment pinned, determinism helper, user-filter finding resolved or documented,
      quiet logs, checkpoint policy
- [ ] W1: `Framework/hug.py` per Spec 03 + amendments 1–7; arm flags; resume;
      `Framework/metrics.py`
- [ ] W2: `single` removed from the plan and scripts; documented
- [ ] W3: holdout; `Framework/features.py` used by both pipelines; `tune.py` / `tune_hug.py`;
      FiGNN (and optionally DCNv2) configs; metric schema parity
- [ ] W4: `heavy_run.yaml`, `run_queue.py` (skip/resume/retry/dry-run/status), test-set guard,
      `aggregate.py`, per-row prediction dumps
- [ ] W5: tests 1–20 pass; synthetic suite < 5 min; real-data suite passes
- [ ] W6: diagnostics reported (and nothing points to a leak); every arm smoke-trained; dry run
      of the whole queue green; aggregation works on the dry-run outputs; run-time estimate
      within 72 h, or cuts proposed
- [ ] W7: results file, `HEAVY_RUN.md`, `CHECKPOINT.md` and the Spec 03 header updated
- [ ] After this, launching the heavy run takes one command:
      `python scripts/run_queue.py experiments/heavy_run.yaml`
