# Spec 02: Trainable encoders

**Status:** SUPERSEDED by `03-modern-encoders.md`; do not implement. Kept for the rationale.
**Author:** architect session, 2026-10-06
**Scope:** make both HUG-Dual encoders trainable end to end under the existing snapshot
protocol, and run the experiment that decides whether training is worth it.

---

## Why

Today `encode_graph` runs under `@torch.no_grad()`, so only the alignment module and head
learn. The head only sees the 64-d vectors the encoders hand it, and it can't recover what they
threw away. Three things happen inside the encoder that the head can't undo:

1. **How neighbours are weighted.** A video's embedding is a fixed random mix of its
   `clicked`, `liked` and `hated` neighbourhoods. After that, the head sees only the sum.
2. **What survives the compression.** Two random nonlinear layers squeeze each neighbourhood
   down to 64-d. Which directions survive is arbitrary.
3. **Similarity.** Random ID vectors are all roughly equidistant. Two authors with the same
   audience look unrelated, so the cold-item path (cold video → its author → similar users)
   can't generalise. Only training creates that collaborative similarity.

The 1-epoch R0 run is underfitting, not overfitting (train AUC 0.686 vs val 0.682), which fits
the picture of frozen features as the bottleneck.

**Counter-hypothesis to test, not assume.** Frozen propagation can be competitive when it's
*parameter-free and linear over meaningful features* (SGC/SIGN). The current encoder is
neither: it uses random weights and nonlinear layers. This spec therefore includes a frozen
**linear** arm, so the comparison is fair to the frozen side.

## Design

### A. New structural encoder: `RelLightGCN` (`Framework/gnn_encoders.py`)

A relation-aware LightGCN: no per-layer feature transforms and no nonlinearities.

```
h0_i      = input embedding of node i                        (section C)
h{l+1}_i  = Σ_r  softmax(a_l)_r · Σ_{j ∈ N_r(i)}  h_l_j / sqrt(deg_r(i) · deg_r(j))
out_i     = mean_l (h_l_i)             for l = 0..L            (LightGCN layer mean)
```

- `a_l ∈ R^{num_relations}` is a learned logit per relation per layer, with
  `num_relations = 2 · len(REL_MAP)` (forward + reverse, from Spec 01).
- `deg_r` is computed **from the snapshot's masked edges only**.
- Implement it with `torch.sparse` / `torch_sparse` SpMM, one sparse matrix per relation, built
  once each time a snapshot is entered: mask the master index with `rel_time < b_k`, compute
  the degrees, then the values. Don't store 34 copies; rebuild on entry. It's a few hundred ms
  on the GPU.
- Return `{"user": out[users], "video": out[videos]}` at `out_dim`, with no output projection.
- Keep `StructuralGNN` (the R-GCN) in the code for arm M1 and the Spec 01 reference.

**Why this encoder:**
- It's cheap enough to run on the full graph every step. The cost is ~28M-nnz SpMMs at d=64;
  there are no dense matmuls over 5M nodes and no stored activations per relation.
- It's the standard strong collaborative-filtering propagation.
- The learned `softmax(a_l)` gives a per-relation weight table (like vs. hate vs.
  author/category) that the paper can report.

### B. Sequential encoder: make the existing one trainable

- Keep the `SequentialGNN` GGNN and `prefix_readout` as they are, but run them with gradients
  enabled in training.
- Add `--seq-hidden` (default 64 when training). The GGNN over ~4.4M video nodes stores GRU
  activations, so halving the width matters.
- Put the sequential branch on GPU 1 and the structural branch plus head on GPU 0
  (`split_across_gpus` already exists; it just needs to work with gradients).
- **F8 fix:** replace `--gnn-layers` with `--struct-layers` (default 2) and `--seq-layers`
  (default 1).

### C. Shared input layer: IDs plus features

Add one `InputEncoder`, shared by both branches, that produces `h0` at `out_dim` for every node
type:

| Node | h0 |
|---|---|
| user | `user_id_emb + Linear(user features + onehot embeddings)` |
| video | `video_id_emb[vocab(v)] + Linear(snapshot video features)` |
| author / category | ID embedding (Spec 01's tables move here) |

- **Video ID vocabulary is built from the training window only** (`time_ms < t_val`). A video
  with ≥ `--min-video-count` (default 5) training interactions gets its own row. Every other
  video maps to one shared OOV row, plus its features, plus whatever reaches it from its
  author/category through propagation. Log the vocabulary size and the OOV share of val rows.
- Never build the vocabulary from the full log. That's leakage: test-only videos would get
  their own rows, and the vocabulary would encode future popularity.
- Init: `normal(std=0.1)`.
- Regularisation: L2 on the embedding rows touched in the batch (`--emb-l2`, default 1e-6),
  plus the existing weight decay on dense parameters.

### D. Training loop (`main.py`, `models.py`)

- Replace the no-grad `encode_graph` with `encode(snapshot, grad: bool)`, taking:
  - `grad=False` for eval and for the frozen arms;
  - `grad=True` in training for the trainable arms.
- Per training step, inside a snapshot:
  1. Run the full encoder forward on that snapshot.
  2. Gather the batch's user/video/prefix rows.
  3. Run the head and backward.
  4. Take one optimizer step.

  Recompute the encoder every step. No cached embeddings, no stale weights.
- Keep the existing snapshot-contiguous loop (snapshot order shuffled, rows shuffled within).
  Default `--batch-size 8192`; scale the LR if the batch size changes.
- **Validation:** encode once per snapshot with `grad=False` (as now). Early-stop on val AUC
  with `--patience 2`, max 20 epochs, and restore the best weights.
- Set the encoders' train/eval mode explicitly, so dropout is active in training forwards and
  off in validation.
- Run every arm with `--no-ips --no-recency-gate --kg-alignment 0`. Those components are being
  removed, and the head must be the same across arms.
- **Compute target:** under 30 min per training epoch on 1K. If it's well over, report the
  per-step breakdown (structural forward/backward, sequential forward/backward, head) before
  optimising anything. **Don't** fall back to neighbour sampling or cached embeddings without
  checking with the architect.

## Experiment arms

All arms use the same input layer (section C, IDs switched as listed), the same head and the
same data, and are reported on **validation only**:

| Arm | Structural | Sequential | Grad through encoders | ID embeddings | Question |
|---|---|---|---|---|---|
| F | R-GCN (Spec 01 R3) | GGNN | no | author/category only (random) | reference |
| E1 | RelLightGCN, uniform `a` | GGNN | **no** | none (features only) | frozen *linear* propagation: the fair frozen arm |
| E2 | RelLightGCN | GGNN | **yes** | none (features only) | does training the encoders help, holding inputs fixed? |
| E3 | RelLightGCN | GGNN | **yes** | user, author, category, video (≥ m) | **proposed model** |
| E3-noedge | none (h0 only) | none (h0 only) | yes | as E3 | is it the graph or just the ID embeddings? |
| M1 (optional) | R-GCN | GGNN | yes | as E3 | training vs. architecture; skip if it OOMs, and say so |

Why each comparison matters:

- **E1 → E2:** what training the encoders adds. This is the direct answer to "is training the
  GNN wasted compute?"
- **E2 → E3:** what learned identities add.
- **E3-noedge → E3:** whether message passing earns its place. This is the comparison
  reviewers will ask for. E3-noedge is essentially a feature + ID-embedding MLP: if it matches
  E3, the graph isn't the contribution.

**Seeds:** run E1 and E3 with 3 seeds (42, 43, 44); the other arms with 1 seed. Report the
mean ± std for the 3-seed arms.

## Measurements to report

For each arm:

- val AUC, AP, LogLoss, nDCG@10;
- best epoch and epochs run;
- val AUC bucketed by the candidate video's training-window interaction count:
  `0`, `1–4`, `≥5` (the last bucket matches the ID-vocabulary cut);
- per-epoch wall time and peak GPU memory on each GPU;
- trainable parameter count, split into embedding vs. dense;
- for E2/E3: the learned `softmax(a_l)` table (relations × layers).

Also report the ID vocabulary size and the OOV share of training and validation rows.

## Tests (add to `Framework/tests.py`)

1. **Gradient routing.**
   - E2/E3: after one backward, every encoder parameter, and every ID-embedding row touched by
     the batch, has a nonzero grad.
   - E1/F: encoder parameters have no grad.
2. **RelLightGCN correctness.** On a 6-node, 2-relation hand-built graph, one layer matches a
   hand-computed symmetric-normalised aggregation, and the layer mean matches too.
3. **Snapshot degrees.** The degrees used for snapshot k equal the degrees computed from edges
   with `rel_time < b_k` (reverse edges included).
4. **Vocabulary from the training window.** A video whose interactions are all at or after
   `t_val` maps to OOV, even when its full-log count is ≥ m.
5. **Weights don't depend on val/test.** On synthetic data, CPU, deterministic:
   - train E3 for 2 epochs;
   - rewrite every row at or after `t_val` (flip all feedback, permute the videos), rebuild,
     and retrain with the same seed;
   - all parameters must be bit-identical.

   This is the trained-model counterpart of the prediction-invariance test.
6. **Prediction invariance still holds.** Extend
   `TestCausality::test_predictions_invariant_to_future` to run with a (fixed-weight) E3 model
   as well as the current dual model.
7. **Eval is deterministic.** Two eval passes over the same snapshot produce identical scores.

## Decision this feeds

| Outcome | Reading | Next |
|---|---|---|
| E3 > E2 > E1, and E3 > E3-noedge clearly | trained graph encoders are the contribution | proceed to fusion work (sparse shared/private codes) on top of E3 |
| E2 ≈ E1, but E3 > E3-noedge | training helps only through identities; propagation could stay parameter-free | consider a SIGN-style precomputed variant for efficiency, and frame accordingly |
| E3 ≈ E3-noedge | the graph adds nothing over IDs + features | stop and rethink the thesis before any more engineering |

What counts as "clearly" depends on the seed spread from the 3-seed arms. A gap smaller than
about 2 std doesn't count.

## Not in this spec

- HUG-Unified retraining. **Recommendation for a later spec:** rebuild Unified as the *same*
  `RelLightGCN` over the merged graph, with `next_in_session` edges included, one encoder and
  the same head. That makes it a fair single-encoder control and removes the HGT bugs (F10)
  from the critical path.
- Feedback features on prefix tokens; candidate-query (target) attention in the readout.
- Removing the IPS/KGA/recency-gate code (the arms just switch them off).
- Replacing the alignment module (F6).
- `interacted` edges.

## Done when

- [ ] `RelLightGCN`, `InputEncoder` with a training-window vocabulary, trainable sequential
      branch, split depths, two-GPU placement
- [ ] Tests 1–7 pass; the full synthetic suite passes
- [ ] Arms F, E1 (3 seeds), E2, E3 (3 seeds) and E3-noedge reported with the measurements
      above, validation only; M1 reported or marked infeasible with the reason
- [ ] `docs/CHECKPOINT.md` updated
