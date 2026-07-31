# HUG — Revision Workflow

Execution plan for the next submission cycle. Companion to `docs/TECHNICAL_REPORT.md`, which contains the diagnosis and finding IDs (F1–F14) referenced throughout.

**How to use this document.** Phases 0–2 are sequential and mandatory; nothing downstream is interpretable until they land. Phase 3 onward is partly parallel. Each task lists the files to touch and an acceptance criterion — a task is not done until its criterion is demonstrably met. Two decision gates (after Phase 2, after Phase 4) determine venue and framing; do not pre-commit to a narrative before them.

**Guiding rule for the whole cycle:** every number currently in the paper is void. Do not attempt to preserve, recover, or defend any existing result. Rebuild the evidence base, then see what it says.

---

## Phase 0 — Foundations

Nothing measured before these land is trustworthy. Small, mechanical, unblocks everything.

### 0.1 Three-way global chronological split
**Files:** `Framework/main.py` (`temporal_split`, ~L259-306)
Return train / validation / test by global `time_ms` order. Suggested 70 / 10 / 20. Thread `val_loader` through the epoch loop.
**Acceptance:** `max(time_ms)` in train < `min(time_ms)` in val < `min(time_ms)` in test, asserted in code.

### 0.2 Move all model selection to validation (F9)
**Files:** `Framework/main.py:1402` (checkpoint criterion), sweep scripts
Checkpoint on `val_m.auc`. Test evaluated once, from the val-selected checkpoint. No hyperparameter may be chosen by inspecting a test metric.
**Acceptance:** grep confirms no `test_*` metric feeds a selection or checkpoint decision.
> Supersedes the rebuttal's R2-Q3 commitment. Promoting `L=1` on the strength of test-set sweeps must not reach the resubmission.

### 0.3 Fix result-file collisions (F12)
**Files:** `Framework/main.py:1472`, `Framework/main.py:1416-1417`
Delete the `out_dir` reassignment before `final_metrics.json`; use `make_run_name()` throughout. Load the final-eval checkpoint from `out_dir`, matching where `save_checkpoint` writes.
**Acceptance:** two ablations run back-to-back produce two distinct `final_metrics.json` files; checkpoint load logs the run-specific path.

### 0.4 Multi-seed harness (F13)
**Files:** `run_experiments.sh`, `Framework/main.py` (arg plumbing), `Plots/view_results.py`
Every configuration runs ≥5 seeds. Aggregate to mean ± std. Add paired t-tests / bootstrap CIs between configurations sharing seeds.
**Acceptance:** results tables report `mean ± std (n=5)`; a `significance.json` per comparison.
> Directly answers R2's clarification Q4.

### 0.5 Vectorise the session readout (F11)
**Files:** `Framework/gnn_encoders.py:279-330` (`_session_readout`), and the identical copy in `SingleHGT:523-559`
Replace the Python loop with segment operations: `pyg_softmax` (already imported, L29) for per-session attention, `scatter` for pooling.
**Acceptance:** outputs match the loop to ≤1e-5 on a fixture; encode wall-time drops ≥10× on KuaiRand-1K.
> Precondition for Phase 2's end-to-end arm — this is the bottleneck that likely motivated freezing.

### 0.6 Housekeeping
- Delete the misleading docstring at `models.py:409-411` (F14)
- Fix the `F` / `F_local` NameError at `main.py:794` (F11)
- Reconcile `README.md` numbers with `runs/`, or remove the table pending Phase 1 (F12)
- Route `is_follow → followed` edges into `REL_MAP` and the structural subgraph, or delete them (F14)
- Apply `min_interactions` to both policy logs (`data_loader.py:185`) (F14)
- Replace `session_idx.clamp` at `main.py:968` with an assertion (F14)

---

## Phase 1 — Integrity

The number reset. Expect every metric to move, likely downward. That is the point.

### 1.1 Leak-free HKG construction (F1)
**Files:** `Framework/hkg_constructor.py`, `Framework/main.py` (build order)
Pass a training-window cutoff into `HKGConstructor`. Three channels to close:

- **(a) Label edges.** Restrict all feedback edge types — critically `("user","clicked","video")`, relation 0 at `main.py:313` — to interactions strictly before the cutoff.
- **(b) Session edges.** Restrict `next_in_session` (`hkg_constructor.py:351-355`) to training-window sessions. Sessions straddling the boundary are truncated at the cutoff, not dropped.
- **(c) Item statistics.** Recompute `global_cvr` and all `*_cnt_log` features (`data_loader.py:364-372`) from training-window interactions rather than the shipped corpus-wide statistics file. This means deriving them from the logs, not reading `video_features_statistic_*.csv`.

**Acceptance:** an automated leakage test — for a random sample of 1000 test pairs `(u,v)`, assert no edge of any feedback type exists between them in the encoding graph, and assert every node feature is reproducible from training-window rows alone. Add to `Framework/tests.py`.

> Split validity is separate and already correct: `main.py:294-299` does a proper global chronological split. §4.1 of the paper misdescribes it as per-user — **fix the paper text**, and note the correction explicitly in the resubmission rather than relitigating R2's weakness #1.

### 1.2 Rebuild the baselines (F2)
**Files:** `Baselines/preprocess.py` (L36-38, L57, L143-150)
- Load the same three logs as `data_loader.py`, not only `log_standard_4_08_to_4_21_1k.csv`.
- Sort by `time_ms` **only**. The current `["user_id","time_ms"]` sort followed by positional slicing produces a user-disjoint split, testing DIN/BST cold-start.
- Use the identical cutoffs as `temporal_split` so all models see byte-identical splits.

**Acceptance:** row counts, positive rates, and user/item coverage of the baseline test set match HUG's test set exactly. Assert this in a shared fixture consumed by both pipelines.

### 1.3 Capacity parity (R3, unaddressed)
**Files:** `Baselines/config/model_config.yaml`
Sweep DIN/BST `embedding_dim` over {16, 32, 64, 128}, selecting on validation. Report baselines at their *best* setting, not their default.
**Acceptance:** Table 2 shows comparable parameter counts across all models; baseline capacity was selected on val.

### 1.4 Re-run everything
Full sweep, 5 seeds, leak-free, on the repaired splits.
**Acceptance:** a single results artefact superseding Table 3 entirely.

> **Expect the headline gap to shrink or vanish.** If HUG no longer beats a properly-trained, capacity-matched BST, that is the true state of the work and must be known now, not after another review cycle.

---

## Phase 2 — The decisive experiment

One question, three arms, on Phase 1 data. Determines the paper's identity.

### 2.1 Training-regime comparison (F3)
| Arm | Encoders | Notes |
|---|---|---|
| **A** Random frozen | untrained, `@torch.no_grad()` | current published behaviour |
| **B** SSL-pretrained frozen | link prediction (structural) + next-item (session), then frozen | natural home for the "feature extractor" framing |
| **C** End-to-end | gradients through both encoders | needs 0.5; add `NeighborLoader` (imported unused, `main.py:79`) if memory-bound |

Arm C requires removing `@torch.no_grad()` at `models.py:359` and `Baselines/KGAT/kgat_model.py:232`.
**Acceptance:** all three arms, 5 seeds, with significance tests. Run the same three arms for KGAT so the comparison is controlled.

### 2.2 Untrained-GNN control as a first-class baseline
Add "random graph features + MLP head" to the permanent baseline set. Most graph-recommendation papers omit this control; including it is defensible regardless of outcome and pre-empts R2's "trivial consequence of richer features" line.

### ⛔ Decision Gate 1
- **C ≫ A:** → **Path A (architecture paper)**. Unfreeze permanently; inductive-bias matching becomes supportable. Proceed to Phase 3 in full.
- **A ≈ C, or A > C:** → **Path B (analysis / reproducibility paper)**. The finding is that learned message passing contributes little to CVR here. Prioritise Phase 4 mechanism evidence and Phase 5 breadth over Phase 3 architecture work.

Record the outcome and date here when known:
```
Gate 1 outcome: ___________  date: __________
```

---

## Phase 3 — Architecture

Weighted toward Path A, but 3.1 and 3.2 are worth doing under either.

### 3.1 Decoupled propagation depth (F8) — *do this first*
**Files:** `Framework/main.py:1072` (split `--gnn-layers` into `--struct-layers` / `--seq-layers`), `build_model`
Sweep `L_str × L_seq ∈ {1,2,3}²`, selecting on validation.
**Hypothesis:** interior optimum with `L_str > L_seq`.
**Why it matters:** R2 read the `L=1` optimum as proof the architecture cannot exploit high-order topology. If the true cause is a forced shared depth, this converts the paper's most damaging result into its central supporting evidence. Nine runs on working code — highest return per unit effort in the whole plan.

### 3.2 Make the sequential branch sequential (F7) — *precondition for everything else*
**Files:** `Framework/hkg_constructor.py:327-369`, `Framework/gnn_encoders.py:239-277`
Currently all `next_in_session` edges from all sessions live in one global graph and `self.ggnn(h, edge_index)` (`gnn_encoders.py:267`) propagates across the whole thing — session identity enters only at readout. This is a global item-transition encoder, not a session encoder, so **both branches are consuming corpus-level structure and the dual-view thesis is not instantiated**.
Batch per-session subgraphs (true SR-GNN) or use a within-session sequence encoder.
**Acceptance:** a test asserting no message path exists between nodes in different sessions.
> Until this lands, any measured difference between the two branches is confounded. Treat it as blocking for 3.4 and 3.5.

### 3.3 Two-timescale formulation (§7.3 of the report) — *strongest reframing*
Recast the thesis from "two subgraphs" to "two timescales": a slow user state (EMA or low-LR across time windows) and a fast session state recomputed per session. Frame in graph-signal terms — structural branch as low-pass over the interaction graph, session branch as order-sensitive and band-limited.
Gives the paper a theoretical spine, makes depth principled rather than a hyperparameter, and predicts 3.1's asymmetry analytically. Also generalises past KuaiRand, speaking to the metareview's single-dataset concern.

### 3.4 Complementarity objective (F6) — *strongest novelty candidate*
Replace the degenerate gate (softmax over heads with a single key, `models.py:142`) with an objective that makes decomposition **measurable**: align the shared component across views while decorrelating view-specific components (Barlow Twins / VICReg covariance penalty, or InfoNCE over matched `(user,item)` pairs).
**Deliverable:** a shared-vs-view-specific information decomposition. High redundancy is a publishable negative result; complementarity is the mechanism evidence the paper lacks. Either outcome is reportable — the property the current ablation table lacks entirely.

### 3.5 Identifiable fusion
Replace concatenate-then-MLP with one of: MoE routing with load balancing (routing weights are directly interpretable); product-of-experts in logit space (`logit_str + logit_seq`, each branch with a standalone AUC); or gated residual publishing the distribution of `g`.
**Acceptance:** the fusion mechanism produces a reportable statistic, not just a scalar delta.

### 3.6 KGA gate: implement or remove (F4)
Currently `kg_relation` is never constructed and never passed (`main.py:976-983`), so Equation 2's `Φ_k(R_kg)` term does not execute. Either implement genuine relation-path features `r_uv` between user and candidate and thread them through, or delete the component and every associated claim.
**Do not** carry forward §5.2's explanation that sparsity caused the null result — the cause was non-execution.

### 3.7 Repair the HGT baseline (F10)
Per-destination-node softmax with correct normalisation across edge types (`gnn_encoders.py:647-659`); add the missing residual (L662); fix the `W_Q` initialiser (L589-590). HUG-Unified underpins the "+4.24% from elevating metadata to topology" claim and must be correct.

---

## Phase 4 — Evidence

**This phase, more than any other, determines acceptance.** Aggregate deltas of 0.005 will not survive review; mechanism evidence will.

### 4.1 Stratified evaluation
Report every headline comparison broken down by:

| Stratum | Bins | Expected signal |
|---|---|---|
| User history length | cold-start → heavy | structural branch dominates cold end |
| Session length | 1–2 items → long | sequential branch dominates long end |
| Item popularity | long-tail → head | KG paths matter most in the tail |
| Time-to-prediction | early → late in test window | degradation under distribution shift |

**Target claim:** *"the structural branch carries cold-start users, the sequential branch carries long sessions, and the fusion recovers both."* This is defensible even if aggregate AUC is flat, and it is the paper this project should be aiming at.

### 4.2 Conflation diagnostic
Make the paper's founding assumption falsifiable. Probe unified-encoder representations for whether long-term preference and current-session intent remain linearly recoverable, versus the dual encoder. CKA between branch representations, or linear probes against held-out proxies (user's long-run category distribution vs. session category distribution).
**Deliverable:** direct evidence that conflation occurs in a unified encoder — currently asserted in §1 and §3.3, never measured.

### 4.3 Ranking metrics appropriate to the venue
Add GAUC alongside AUC. Keep per-user macro-averaged NDCG@10 (already correct — `_per_user_ndcg_at_k` filters users with <2 interactions or no positives; the 1.0 in `README.md` is a stale pre-fix archived run).

### 4.4 Rolling-origin evaluation
Multiple sequential time folds rather than one split. Directly answers R2's demand for "a strict, global Time-Dependent Split (TDS) or chronological rollout protocol," and demonstrates stability across periods.

### 4.5 Cost accounting
Training and inference cost versus KGAT and the transformer baselines — R3's only substantive question in the Review field, never answered.

### ⛔ Decision Gate 2
Do the stratified results show a **differential** effect — each branch winning where the thesis predicts?
- **Yes:** strong submission on the mechanism story regardless of aggregate deltas.
- **No:** the decomposition does not pay off on this data. Pivot to Path B and report it as a negative result with the conflation diagnostic as the contribution.

```
Gate 2 outcome: ___________  date: __________
```

---

## Phase 5 — Breadth

Directly addresses the metareview: *"extend the empirical evaluation with more datasets, stronger baselines."*

### 5.1 Datasets
| Dataset | Why |
|---|---|
| **KuaiRand-Pure** | Large random-exposure fraction — the only way to make IPS meaningful (F5). Answers R1's key clarification directly |
| **KuaiRand-27K** | Scale claim, currently unsupported (F11). Fix the NameError and the float32 index precision issue (`main.py:298-299`) first |
| **Tenrec** or **Taobao UserBehavior** | Second domain, rich side information |
| **Diginetica** / **Yoochoose** | Session-based standard; isolates the sequential branch against SR-GNN's home turf |

Minimum for resubmission: **two datasets beyond KuaiRand-1K**, one of which is KuaiRand-Pure.

### 5.2 IPS: fix or cut (F5)
On KuaiRand-Pure, estimate real propensities and use genuine `1/P(exposure)` weights. If not pursued, remove IPS and every associated claim from abstract, §1, §4.3 and Table 3. The current scheme — constants 1.0 and 0.9963, then mean-normalised (`main.py:288-289`, `models.py:241`) — cannot be defended as inverse propensity scoring.

### 5.3 Baselines
- **Sequence/CTR:** DIN, BST (both repaired), SASRec, BERT4Rec, DCNv2, FiBiNET, MaskNet
- **Graph:** KGAT (repaired + trained), LightGCN, NGCF, SR-GNN
- **Control:** untrained-GNN + MLP (Phase 2.2)
- **Modern:** InterFormer, OneTrans, HyFormer — attempt each

> **On the availability dispute:** R2 called the closed-source claim "fabricating a claim" and "a breach of academic rigor." Do not re-litigate. Delete the closed-source assertion from §4.2 entirely and add whichever of these can actually be run. If one genuinely cannot be obtained, state only that an implementation could not be sourced, with the date checked — never assert that none exists.

---

## Phase 6 — Writing

### 6.1 Corrections to carry explicitly
1. §4.1 — describe the *implemented* global chronological split (paper currently says per-user; code is global)
2. §5.3 — replace the over-smoothing explanation of the `L` sweep with whatever Phase 3.1 establishes
3. §5.2 — remove the sparsity explanation of the KGA null result; the gate did not execute
4. §4.2 — remove the closed-source assertion
5. Algorithm 1/2 — align with whatever training regime Gate 1 selects
6. Abstract — headline must be the *recommended* configuration selected on validation, not a sweep maximum

### 6.2 Positioning
Per Gate 1. **Path A:** architecture paper on the repaired dual-view design, mechanism evidence from Phase 4 as the core. **Path B:** analysis paper on learned message passing versus structured feature aggregation in CVR, with the untrained-GNN control and conflation diagnostic as contributions.

Under either path, lead with the mechanism/stratified results, not the aggregate table.

### 6.3 Reproducibility statement
The repository was scored 4/5 for reproducibility by all three reviewers while containing F1, F2 and F4. Before resubmission: leakage tests green, baseline parity fixture green, `README.md` numbers matching the paper, and a `RESULTS.md` mapping every table cell to a run directory and seed set.

---

## Pre-submission checklist

Mapped to the concerns that produced the reject. Every line must be checkable by a reviewer with repository access.

**Integrity**
- [ ] HKG built from training-window data only; leakage test in CI (F1)
- [ ] No feedback edge between any test pair exists in the encoding graph (F1a)
- [ ] Session edges do not cross the split boundary (F1b)
- [ ] Item statistics recomputed from the training window (F1c)
- [ ] Baselines use identical data and identical splits; parity fixture green (F2)
- [ ] Baseline capacity selected on validation (R3)
- [ ] Validation split exists; no test metric feeds any selection (F9)
- [ ] ≥5 seeds with variance and significance tests on every comparison (F13, R2-Q4)

**Claims**
- [ ] Every component in Table 3 executes at runtime — verified by assertion, not inspection (F4)
- [ ] IPS either uses estimated propensities or is removed entirely (F5)
- [ ] Attention modules attend over a real key set, or are renamed (F6)
- [ ] Sequential branch propagates within session boundaries (F7)
- [ ] Encoder depths configured independently (F8)
- [ ] Training regime in Algorithms 1/2 matches the code exactly (F3)

**Evidence**
- [ ] ≥3 datasets, including KuaiRand-Pure (metareview)
- [ ] ≥2 baselines from 2024 or later (R2 §2)
- [ ] Untrained-GNN control reported (R2 §3)
- [ ] Stratified results by history length, session length, popularity (Phase 4.1)
- [ ] Conflation diagnostic reported (Phase 4.2)
- [ ] GAUC alongside AUC (Phase 4.3)
- [ ] Rolling-origin evaluation (R2-Q2)
- [ ] Training/inference cost table (R3)

**Text**
- [ ] §4.1 split description matches implementation
- [ ] Closed-source assertion removed (R2 §2)
- [ ] Headline configuration selected on validation (R2 §5)
- [ ] `README.md` reconciled with paper tables (F12)
- [ ] Corrections from the prior submission stated openly

---

## Sequencing

```
Phase 0  Foundations ────────────┐
                                 ├──> Phase 2 ──> ⛔ Gate 1 ──> Phase 3 ──┐
Phase 1  Integrity   ────────────┘                    │                   ├──> ⛔ Gate 2 ──> Phase 6
                                                      └──> Phase 4 ───────┤
                                          Phase 5 (parallel from Gate 1) ─┘
```

Phases 0 and 1 are strictly blocking. Phase 5's dataset work can begin during Phase 3 since it is largely independent engineering. Phase 4 is where acceptance is won and should not be compressed.

**Order of first three actions, if picking this up cold:**
1. Phase 1.2 — the baseline preprocessing fix. Smallest change, largest correction to the record, and it is the most exposed defect in the public repository.
2. Phase 0.1 + 0.2 — validation split and selection discipline. Everything downstream needs it.
3. Phase 1.1 — leak-free HKG. Largest single piece of work in Phases 0–1; start once the split plumbing exists.
