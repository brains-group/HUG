# Spec 01: Structural graph connectivity

**Status:** ready for implementation
**Author:** architect session, 2026-10-06
**Depends on:** the uncommitted checkpoint described in `docs/CHECKPOINT.md`
**Scope:** encoder inputs and message flow only. The encoders stay **frozen**. Whether to train
them is decided *after* this spec's measurements (see "What this is for" below).

---

## Problem

The structural R-GCN can't carry collaborative or knowledge-graph signal to the nodes the head
reads.

1. **Every relation is one-directional.** `REL_MAP` (`Framework/main.py:242`) only has edges
   `user→video` and `video→author|category`. `RGCNConv` aggregates at the destination, so:
   - **user** nodes have no incoming edges. `struct_emb["user"]` is a transform of the user's
     own profile features, at every depth;
   - **video** nodes only receive from the users who clicked, liked, etc. them;
   - **author/category** nodes receive from videos but are never read out. The knowledge graph
     never reaches users or videos.

   `SingleHGT.EDGE_TYPES` (`Framework/gnn_encoders.py:412`) has the same problem.
2. **Authors and categories have no identity.** Their input is `torch.zeros(n, 1)`
   (`hkg_constructor.py`), so after `author_proj` / `category_proj` every author is the same
   vector, and so is every category. Fixing (1) alone would just push a constant onto videos.
3. **Most edges are thrown away.** `--max-edges 200_000` randomly keeps at most 200k edges *per
   relation*. That's roughly 5% of the click edges, resampled independently for each snapshot.
4. **Snapshot storage doesn't scale without the cap.** `SnapshotStore` caches a full relation
   edge index per snapshot (about 31 copies). Uncapped, that's tens of GB of CPU RAM.

## Changes

### A. Reverse relations (structural R-GCN and HGT)

- Add reverse relations **when the relation edge index is built**, not in `HKGConstructor`. This
  keeps `hkg_bundle_timed.pkl` valid and needs no rebuild.
- For each forward relation `r` with edges `(src, dst)`, add relation `r + R` with edges
  `(dst, src)`, where `R = len(REL_MAP)`. That gives 7 → 14 relations for the dual model.
- Pass `num_relations = 2 * len(REL_MAP)` from `build_model`. Derive it from `REL_MAP`; don't
  hard-code `7` or `14`.
- **A reverse edge must inherit the forward edge's `edge_time`.** This is the leakage-critical
  part of the spec; see C and the tests.
- HGT (`build_full_relation_edge_index`, `SingleHGT`): do the same. The Unified model is the
  controlled baseline, so it must see the same connectivity as Dual. `_HGTLayer` sizes its
  per-type weights from `len(EDGE_TYPES)`, so make the type count a constructor argument and
  pass the doubled count from the builder. Leave `next_in_session` one-directional in both
  models; it's ordered by definition.
- Don't touch the GGNN in `SequentialGNN`.

### B. Author and category ID embeddings

- `StructuralGNN` and `SingleHGT`:
  - replace `author_proj` / `category_proj` with `nn.Embedding(n_authors, hidden_dim)` and
    `nn.Embedding(n_categories, hidden_dim)`;
  - take `n_authors` and `n_categories` as constructor arguments (from `bundle.n_authors`,
    `bundle.n_categories`);
  - drop the `author_feat_dim` / `category_feat_dim` arguments.
- Initialise with `nn.init.normal_(std=hidden_dim ** -0.5)`, so the embeddings are on the same
  scale as the projected user/video inputs. The R-GCN's LayerNorm handles the rest.
- While the encoders are frozen, these act as **fixed random identity vectors**. That's
  intended: each author is now distinct, which is all message passing needs. They become
  learned once the encoders are trained (a later spec).
- Leave `graph["author"].x` / `graph["category"].x` in the HKG (the cache depends on them), but
  stop reading them.
- **Out of scope:** user and video ID embeddings. See "Not in this spec".

### C. Uncapped edges, built once with a per-snapshot mask

- Build one **master** relation index per model, once, from the full timed bundle, as three
  tensors:
  - `rel_ei [2, E]`;
  - `rel_t [E]` (relation id);
  - `rel_time [E]`: `edge_time` for behavioural edges, and `-inf` (or `int64 min`) for static
    edges (`made_by`, `tagged_as`), so the cutoff test always keeps them.
- Reverse edges get the same `rel_time` as their forward edge.
- In `SnapshotStore`, keep only the master index on CPU. Compute snapshot k's edges as
  `keep = rel_time < boundaries[k]`. Either precompute and store the per-snapshot boolean masks
  (cheap: 1 byte per edge per snapshot) or compute them on demand in `encode_snapshot`. Don't
  store per-snapshot copies of the edge index.
- Remove `max_edges_per_type` sampling from both builders, and drop `--max-edges`. A random
  subsample was never part of the method.
- On 1K, the uncapped dual-model index is roughly 28M edges after reversal: about 4.4M clicks,
  4.4M `made_by`, the tags and smaller relations, all doubled. `RGCNConv` runs per relation
  under `no_grad`, so this should fit on one 95 GB GPU. **Log peak GPU memory and encode time
  per snapshot.** If it doesn't fit, report back; don't reintroduce sampling.

### D. Small fixes bundled here

1. **`--warmup-hours`:** default it to `--snapshot-hours`. Snapshot 0's graph is empty, so
   training on its rows teaches the head that "empty graph" is a normal input.
2. **HKG cache safety** (`main.py:107-136`):
   - add `HKG_BUILD_VERSION = 2` to `hkg_constructor.py`, store it on the bundle, and require
     it to match in `load_hkg`. Bump it whenever constructor output changes;
   - write the pickle to a temp file in the same directory, then `os.replace` it into place.
3. Delete the unused GAT refinement in `StructuralGNN` (`self.gat`, plus the
   `video_video_edge_index` argument). It never runs, and it inflates the parameter count.

## Tests (add to `Framework/tests.py`)

Synthetic fixtures are enough for all of these. Every existing test, especially
`TestCausality::test_predictions_invariant_to_future`, must still pass **unchanged**.

1. **Reverse completeness.** For each forward relation `r`, the edges of `r + R` are exactly the
   flipped edges of `r`, with equal counts and the same multiset of `(src, dst)` after flipping.
2. **Reverse edges are causal.** For every snapshot k, every edge kept by the mask, including
   reverse edges, has `rel_time < boundaries[k]`. Reverse edges whose forward edge is dropped
   are dropped too.
3. **Mask equivalence.** For a few k, the masked master index equals, as a multiset of
   `(src, dst, rel)`, the index you get by building from `snapshot_bundle(bundle, b_k)` with
   reversal applied. This proves the one-time build matches the per-view build it replaces.
4. **Users now receive graph signal.** On a fixed synthetic graph with frozen weights and
   `eval()`:
   - remove one user's clicked edges; that user's `struct_emb["user"]` must change;
   - with L=1, a user with no edges must equal the output for the same features with an empty
     edge set (sanity check that nothing else changed).
5. **The KG reaches videos and users.** With L=2, perturb one author's embedding row. The
   embeddings of that author's videos must change, and so must the embeddings of users who
   clicked those videos. Users and videos with no path to the author must not change.
6. **HGT parity.** Repeat test 1 for `build_full_relation_edge_index`.

## Measurements to report

Report **validation only**: one epoch each, seed 42, otherwise default flags, on real 1K data,
snapshot protocol. Don't evaluate or report test.

| Run | Config |
|---|---|
| R0 | current code (one-way, 200k cap, zero author/category inputs). **Already done:** the checkpoint run (`/tmp/claude-163583/-home-dasgua3-HUG/a46881cb-c62c-421c-a2ee-22836d6a1f73/scratchpad/hug_runs/1k_dual_kg0_ips1_rg1`, all defaults; scratchpad is temporary, so copy it into `runs/` before it is cleaned up) is R0: val AUC 0.6821, LogLoss 0.6181. Don't rerun it. |
| R0-noedge | **New control, required.** R0 code with an **empty** structural edge set and an empty `next_in_session` edge set, so each encoder only sees node features (self/root transform). It answers whether the graph contributes anything over the features. Run it on the current code, before applying the spec. |
| R1 | A only: reverse relations, still capped, still zero inputs |
| R2 | A + B: plus author/category embeddings, still capped |
| R3 | A + B + C: full spec |
| R3-noedge | R3 code with empty edge sets: the features-only control for the new code (author/category embeddings are unreachable without edges, so it should match R0-noedge closely) |

For each run, report:
- val AUC, AP, LogLoss, nDCG@10;
- val AUC split into two buckets: candidate video with 0–1 vs. ≥2 earlier interactions in its
  snapshot (the cold vs. warm items from the TransAct note);
- per-snapshot encode time and peak GPU memory;
- trainable parameter count, counting only parameters that receive gradients.

R1 and R2 can be skipped if time is tight, but **R0-noedge, R3 and R3-noedge are required**.
Also report the val AUC of `global_cvr` alone (point-in-time, from the snapshot features) as a
one-feature floor.

**Test-set discipline.** `main.py` currently scores test at the end of every run. During
development, add a `--eval-test` flag (default off) so test is only scored when a configuration
is being reported. None of these runs should touch it. Also run the full
real-data test suite once (`-k TestRealData --data-dir …`), which is still pending from the
checkpoint.

## What this is for

The point is to see whether the graph carries signal once it's actually connected. The
frozen-vs-trained decision depends on it:

- **R0 ≈ R0-noedge:** the current graph adds nothing, as the one-way wiring predicts, and the
  0.682 comes entirely from the features.
- **R3 ≫ R0, especially on cold videos:** connectivity was the bottleneck, and the knowledge
  graph carries signal to cold items. Training the encoders is likely to add more on top.
- **R3 ≈ R0:** even a connected graph with random weights adds little over the features. The
  next step is trained or pretrained encoders, and the paper has to show what training adds.

Either way, the number is needed before the next spec.

## Not in this spec

Don't do these; each has its own spec coming:

- training or pretraining the encoders;
- user or video ID embeddings;
- feedback features on prefix tokens;
- including `interacted` (non-click exposure) edges in `REL_MAP`;
- removing IPS, KGA or the recency gate;
- F6 / F8 / F10;
- the multi-seed harness;
- head / normalisation changes.

## Done when

- [ ] A–D implemented; `--max-edges` removed; no per-snapshot copies of the edge index
- [ ] New tests 1–6 pass; full synthetic suite passes; the invariance test is unchanged and
      passing
- [ ] Real-data test suite run, with results reported
- [ ] R0-noedge, R3, R3-noedge (and R1/R2 if run) reported with the metrics above,
      validation only; `--eval-test` flag added
- [ ] `docs/CHECKPOINT.md` updated: model description, flags, verification table
