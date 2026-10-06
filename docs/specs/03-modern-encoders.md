# Spec 03: Modern, trained encoders (graph view + sequence view)

**Status:** design reference, implemented by `04-heavy-run-readiness.md` (W1). Its "Experiment arms" section is deferred to the heavy run; supersedes Spec 02.
**Author:** architect session, 2026-10-06
**Builds on:** Spec 01 (reverse relations, master edge index with time mask, author/category
embeddings, `--no-edges`, `--eval-test`, the cold/warm bucket metric).

---

## Why

Spec 01 found that with frozen random encoders, every graph run is below its features-only
control: R3 0.678 vs R3-noedge 0.692 val AUC. The graph has to be learned before it can add
anything.

This spec does three things:

1. **Trains the encoders end to end.**
2. **Modernises them:**
   - structural view: a relation-aware LightGCN with an XSimGCL-style contrastive
     regulariser. That's the current standard for graph collaborative filtering, and it's known
     to help long-tail items;
   - sequence view: a light causal transformer with target attention, the standard
     history encoder in modern CTR models.
3. **Puts HUG on the same raw inputs as the baselines,** so the comparison measures
   architecture, not information.

The paper's framing becomes "graph view + sequence view", one encoder per modality, rather
than "dual GNN".

## Design

All new model code goes in a new module, `Framework/hug.py`. Keep `KuaiCVRModel`,
`SingleGNNModel`, `StructuralGNN`, `SequentialGNN` and `SingleHGT` as they are, so Spec 01's
runs stay reproducible. Add `--model-type hug` and make it the default.

### A. Shared input layer: `InputEncoder`

Every node starts from `h0` at dimension `d` (`--emb-dim`, default 64):

| Node | h0 |
|---|---|
| user | `user_id_emb + Linear(user cont. features ‖ onehot embeddings)` |
| video | `video_id_emb[vocab(v)] + Linear(snapshot video stats) + Σ cat_emb(music_id, music_type, upload_type)` |
| author / category | ID embedding (moved here from Spec 01) |

- **ID vocabularies come from the training window only** (`time_ms < t_val`). A video or
  `music_id` with ≥ `--min-id-count` (default 5) training interactions gets its own row; all
  others map to a shared OOV row.
  - Never build a vocabulary from the full log.
  - Log the vocabulary sizes and the OOV share of training and validation rows.
- The video categorical columns must be the **same columns, with the same encoding, as
  `Baselines/preprocess.py`** (`VIDEO_CAT_COLS`). `author_id` and `primary_tag` reach HUG through
  the graph, as author and category nodes.
- Init `normal(std=0.1)`. Apply L2 (`--emb-l2`, default 1e-6) only to the embedding rows the
  batch touches.

### B. Graph view: `RelLightGCN` + contrastive regulariser

```
h{l+1}[i] = Σ_r  softmax(a_l)[r] · Σ_{j ∈ N_r(i)}  h_l[j] / sqrt(deg_r(i) · deg_r(j))
g[i]      = mean_{l=0..L} h_l[i]          (LightGCN layer mean)
```

- Relations: the 14 from Spec 01 (7 forward plus their reverses). `next_in_session` becomes a
  15th relation, kept one-directional: transitions are structural co-occurrence (F7). Derive
  the count from `REL_MAP`; don't hard-code it.
- `a_l` is a learned per-relation logit for each layer. Depth comes from `--graph-layers`
  (default 2).
- Degrees come **only from the snapshot's masked edges.**
- Implementation:
  - each time a snapshot is entered, build one sparse matrix per relation from the Spec 01
    master index (`rel_time < b_k`); don't store per-snapshot copies;
  - size each relation's output to its **destination** node type. Relations into users
    produce `[n_users, d]`, not `[N_all, d]`. That keeps backward memory around 15 GB at L=2
    rather than ~30 GB.
- **Contrastive term (XSimGCL style),** structural view only:
  1. In training, add noise to every layer: `h_l ← h_l + eps · sign(h_l) ⊙ normalize(U(0,1))`,
     with `eps` = 0.1.
  2. Take the final embedding and the embedding at layer `--cl-layer` (default 1) as two views.
  3. Compute InfoNCE (temperature 0.2) over the **distinct users and distinct videos in the
     batch**, separately for each node type.
  4. Add `λ_cl · (L_user + L_video)` to the loss, with `--cl-weight` default 0.1. Setting it to 0
     disables the term.
- In eval, no noise is added.

### C. Sequence view: `HistoryTransformer`

**History definition: the same rows as the baselines.** For each row, use the user's last
`--max-seq-len` (50) **clicked** videos strictly before the row's time.

- This is the `click_history` rule in `Baselines/preprocess.py`, including how ties are
  handled: rows that share a (user, timestamp) all see the same history.
- **Move that logic into one shared function** (`temporal.py` or `data_loader.py`) and call it
  from both `preprocess.py` and HUG. Store histories as index arrays (like `PrefixBatcher`'s
  `pre_start`/`pre_end` design), not strings.
- The old same-session prefix is retired for this model. The per-token same-session flag below
  keeps the short-term-intent signal.

**Tokens:**
`h0(video) + pos_emb(recency rank) + emb(log-bucketed time gap to the row) + emb(same-session flag)`.
Use `h0` from the shared input layer, not graph outputs (but see the optional arm G-tok). Every
token feature must be causal: the gap and the flag use only the row's own time and the history
item's time.

**Encoder:** `--seq-layers` (default 2) transformer layers, 2 heads, width `d`, dropout 0.1, pre-LN,
and a padding mask. Causal masking isn't needed: every token is already in the past.

**Readout:** target attention, with the candidate video's `h0` as the query over the encoder
outputs. Concatenate that output with the mean over valid tokens to get `s_seq` (`2d`). Rows
with an empty history get a learned "no history" vector.

### D. Fusion head (deliberately plain)

```
z = [ g_user ‖ g_video ‖ g_user ⊙ g_video ‖ s_seq ‖ ctx ]
ctx = emb(tab) ‖ emb(hour)          (hour = Beijing time, same formula as preprocess.py)
logit = MLP(z)                      hidden [256, 128], LayerNorm + ReLU + dropout 0.1
```

- Use LayerNorm, not BatchNorm. Training batches are grouped by day, so batch statistics shift
  between snapshots.
- Loss: BCE, plus the contrastive term, plus embedding L2.
- **No IPS, no KGA, no recency gate, no `AlignmentModule`** in the new model. Better fusion
  comes in a later spec, and this head is its reference point.

### E. Training loop

- `HUGModel.encode(snapshot, grad: bool)`:
  - **training:** the full-graph `RelLightGCN` forward runs **every step** with gradients, on
    the current snapshot's matrices. No cached embeddings.
  - **validation/test:** one forward per snapshot, without gradients.
- Keep the snapshot-contiguous loop: snapshot order shuffled, rows shuffled within each one.
  `--batch-size` default 8192; Adam, lr 1e-3, weight decay 1e-5 on dense parameters.
- **Early stopping** on val AUC, `--patience 2`, max 20 epochs, restoring the best weights.
- **Training order:** keep shuffled offline training. Every training row precedes validation
  and test, so this can't affect val/test (open issue 8). Treat a time-ordered single-pass
  variant as a possible later robustness check, not the protocol.
- **Device:** one GPU by default. Add a placement option (`--graph-gpu`, `--head-gpu`) only if
  memory requires it.
- **Compute target:** ≤ 30 min per training epoch on 1K. If it's well over, report a per-step
  breakdown (graph forward/backward, transformer, head, contrastive) **before** optimising.
  Don't switch to neighbour sampling or stale embeddings without asking.

## Experiment arms

Real 1K data, snapshot protocol, **validation only**, early stopping as above.

| Arm | Graph view | Sequence view | Notes / question |
|---|---|---|---|
| N0 | — | — | `h0` of user and video, plus ctx, into the same head. **Features + IDs floor** |
| N1 | — | ✓ | sequence only: a TransAct-like model in our harness |
| N2 | ✓ (trained, λ_cl=0.1) | — | graph only |
| N3 | ✓ (trained, λ_cl=0) | ✓ | full model without the contrastive term |
| **N4** | ✓ (trained, λ_cl=0.1) | ✓ | **proposed** |
| N4-frozen | ✓ (**no grad**, uniform `a`, λ_cl=0) | ✓ | is training the graph worth it? |
| G-tok (optional) | ✓ | ✓, tokens = graph outputs `g` instead of `h0` | graph-enhanced history; only if N4 > N1 |
| M1 (optional) | trained R-GCN (14+1 relations) in place of `RelLightGCN` | ✓ | encoder ablation; report OOM if infeasible |

**Seeds:** run N1, N3 and N4 with 3 seeds (42, 43, 44); the rest with 1.

**Small tuning budget, on val, the same for N1 and N4:**
- `--emb-dim` ∈ {32, 64, 128};
- `--min-id-count` ∈ {2, 5, 10};
- `--graph-layers` ∈ {1, 2, 3} (N4 only).

Report what was tried. The baselines will get an equivalent budget (separate spec).

## Measurements to report

For each arm:

- val AUC, AP, LogLoss, nDCG@10;
- best epoch;
- AUC bucketed by the candidate video's training-window interaction count (`0`, `1–4`, `≥5`),
  plus the Spec 01 cold/warm bucket;
- AUC bucketed by the user's history length at the row (`0`, `1–9`, `10–49`, `50`);
- seconds per epoch, peak memory, parameter count (embedding vs. dense);
- for N2–N4: the learned `softmax(a_l)` table (relations × layers).

Also report the vocabulary sizes and OOV shares.

## Tests (`Framework/tests.py`)

1. **History parity.** For 10,000 random real rows, HUG's history equals the
   `hist_video_ids` produced by `Baselines/preprocess.py`, item for item and in order. On
   synthetic data, every history item's time is strictly before the row's time.
2. **Gradient routing.**
   - N4: one backward gives nonzero grads on all `a_l`, the input projections, the transformer
     and the head, and on the ID rows the batch touched.
   - N4-frozen: the graph parameters have no grad.
3. **RelLightGCN correctness.** On a hand-built graph with 6 nodes and 2 relations, one layer
   and the layer mean match a hand computation. Relations sized to their destination type give
   the same result as a full-size computation.
4. **Snapshot degrees.** For each relation, the degrees used for snapshot k equal those
   computed from edges with `rel_time < b_k`.
5. **Vocabulary from the training window.** A video whose interactions are all at or after
   `t_val` maps to OOV, even if its full-log count is ≥ m.
6. **Weights don't depend on val/test.** On synthetic data, CPU, deterministic:
   - train N4 for 2 epochs;
   - rewrite every row at or after `t_val` (flip all feedback, permute the videos), rebuild,
     and retrain with the same seed;
   - all parameters must be bit-identical.
7. **Prediction invariance still holds.** Run
   `TestCausality::test_predictions_invariant_to_future` for the `hug` model with fixed
   weights, in addition to the legacy models.
8. **The contrastive term uses only batch nodes.** The InfoNCE sets equal the distinct
   users and videos in the batch. With `λ_cl=0`, the loss equals BCE + L2 exactly.
9. **Eval is deterministic.** Two eval passes give identical scores: no noise and no dropout in
   eval.

## Decision this feeds

What counts as "clearly" depends on the 3-seed spread; a gap smaller than about 2 std
doesn't count.

| Outcome | Reading |
|---|---|
| N4 > N1 clearly, gain concentrated on cold videos, and N4 > N4-frozen | the trained graph view is a real contribution on top of a strong sequence model. Proceed to fusion work and external baselines |
| N4 ≈ N1 | the graph view adds nothing over a TransAct-like model. Stop and rethink before any more engineering |
| N3 ≈ N4 | the contrastive term isn't pulling its weight; drop it |
| N0 ≥ N1 | the sequence encoder or the history is broken. Debug before reading anything else |

## Not in this spec

- Baseline tuning, new external baselines, history parity on the baseline side beyond the
  shared function (separate spec, next).
- Fusion research (shared/private sparse codes).
- HUG-Unified redesign. Recommendation stands: the same `RelLightGCN` over the merged graph
  plus the same history and head. Next spec after this one.
- Removing the legacy model code.
- Other datasets.

## Done when

- [ ] `Framework/hug.py` (`InputEncoder`, `RelLightGCN`, `HistoryTransformer`, `HUGModel`);
      `--model-type hug`; shared history function used by both HUG and `preprocess.py`
- [ ] Tests 1–9 pass; the full synthetic suite and the real-data suite pass
- [ ] Arms N0–N4 and N4-frozen reported (N1/N3/N4 with 3 seeds), validation only; optional
      arms reported or skipped with a reason
- [ ] `docs/CHECKPOINT.md` and a `docs/specs/03-modern-encoders-results.md` write-up
