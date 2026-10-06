# Spec 05: Evidence-adaptive shared/private sparse fusion

**Status:** ready for implementation. **§0 should go to the coder now, before the heavy run
launches;** the rest can start as soon as Spec 04 lands.
**Author:** architect session, 2026-10-06
**Builds on:** Spec 04: `Framework/hug.py` (`HUGModel`), `hug_train.py`, `features.py`,
`metrics.py`, `scripts/run_queue.py`, `experiments/heavy_run.yaml`.
**Paper:** `paper/main.tex` §"Evidence-adaptive sparse fusion" now matches this spec (per-view
evidence, top-k as the primary mechanism). **Literature:** `docs/literature/novelty_check.md`
narrows the novelty to the evidence-conditioned, private-atom-only budget; `evgate` (§C) is the
baseline that protects that claim.

---

## Why

This is the paper's contribution. HUG keeps the graph view and the sequence view separate
through encoding. The current head (`HUGModel.head`) then concatenates
`[u ‖ v ‖ u⊙v ‖ s_seq ‖ ctx]` into an MLP. The claim we're testing:

1. The two views overlap. Concatenation counts the shared evidence twice and has no pressure
   to separate it from complementary evidence.
2. How reliable each view is depends on *that view's* support for the prediction.
3. A fusion that decomposes each view into **shared** and **view-private** sparse codes, with
   each view's private capacity scaled to its own evidence, should beat concatenation. The
   gains should concentrate in low-evidence buckets, and the decomposition should give a
   measurable overlap statistic.

**Why the budget is per view.** If low evidence shrank *every* private code, a cold video
would also lose the graph-specific signal that's supposed to help it: its author and category
paths. So each view's private budget depends on the evidence available **to that view**:

| View | Evidence `n_v` (per row, causal) | Meaning |
|---|---|---|
| graph (G) | `log1p(candidate item prior interactions) + metadata_evidence(candidate)` (author on KuaiRand; see below) | structural support: a cold video by an active author still has a strong graph view |
| sequence (S) | `log1p(user history length) + log1p(candidate item prior interactions)` | sequence support: the target attention needs both a history and a candidate it knows |

"User history length" means the as-of length of **the same history the sequence encoder
attends over** (e.g. prior clicks if the encoder's history is click-only), not a separate
count. Use the encoder's history definition so the two can't drift apart.

Every count is **as of strictly before the row's time**, from the shared feature functions
(`features.py`; ties excluded, as with `asof_video_statistics`).

**Keep the "metadata neighbour" generic.** The graph-view term uses the candidate's
*metadata neighbour* count. That's the author on KuaiRand, but MIND has no authors, and
ZhihuRec has both authors and questions. Implement it as a dataset-provided function in
`features.py`:

```
metadata_evidence(rows) -> log1p of the as-of interaction count of the item's primary metadata neighbour(s)
```

| Dataset | Primary metadata neighbour(s) |
|---|---|
| KuaiRand | author |
| MIND | max over the item's subcategory and linked entities |
| ZhihuRec | author and question (max) |

**Missing neighbours contribute 0.** An item with no metadata neighbour (NaN author, etc.)
gets `metadata_evidence = 0`. Never factorize missing values into a shared placeholder group:
that pools every author-less item into one fake "author" with huge as-of counts.

`fusion.py` must not hard-code "author". The per-dataset choice is declared once, in the
dataset adapter (dataset spec, coming next).

---

## §0. Harness amendment (send to the coder now)

**Problem.** `scripts/run_queue.py`'s config hash includes the git tree hashes of `Framework/`
and `Baselines/`. If the heavy run starts after Spec 04 and Spec 05's code lands mid-run, every
completed job's hash changes, and the queue reruns everything.

**Change:**
- The config hash = resolved args + `--params` content + data fingerprint +
  **`CODE_COMPAT_VERSION`**: an integer constant in `Framework/runtime.py`, bumped *by hand*
  only when a change alters the results of existing configurations.
- The full git commit and dirty-diff hash are still **recorded** in each job's outputs, but
  they don't enter the hash.
- Spec 05 must not change the results of any existing arm (test 7 below enforces this), so it
  doesn't bump the version.
- Update the W5 queue test (#16) to cover this: the hash ignores a code change and changes when
  `CODE_COMPAT_VERSION` changes.

**Consequence:** the heavy run can launch as soon as Spec 04 is done. Spec 05's fusion jobs are
appended to `heavy_run.yaml` later, and the queue picks them up without rerunning finished jobs.

---

## A. Module layout

- New file `Framework/fusion.py`. `HUGModel` takes `fusion: FusionHead` in place of its inline
  `self.head`.
- `ConcatFusion` reproduces today's head **exactly**: same layers, same init order, same
  outputs.
- New flag `--fusion {concat,gate,moe,misa,sparse}` (default `concat`), plus the variant flags
  below. Every variant gets the **same** encoder outputs:
  - graph part `e_G = [u ‖ v ‖ u⊙v]` (3d);
  - sequence part `e_S = s_seq` (2d);
  - `ctx` (categorical embeddings ‖ numeric).
- Each `FusionHead.forward(e_G, e_S, ctx, evidence)` returns
  `{"logit", "aux_loss", "stats"}`. `aux_loss` is added to the training loss; `stats` holds
  per-batch diagnostics (below) for logging.
- When `--no-graph` or `--no-seq` is set, only `concat` is allowed (fusion needs both views).
  Raise an error otherwise. Check **both** flags. `ConcatFusion` must accept a missing
  `e_G` as well as a missing `e_S`, and drop it from the concatenation exactly as today's head
  does.

## B. The proposed head: `SparseSPFusion` (`--fusion sparse`)

### B1. Projections
`p_v = LayerNorm(W_v e_v)` for `v ∈ {G, S}`, at dimension `d'` (`--fusion-dim`, default 64).

### B2. Dictionaries
Three dictionaries of `d'`-dimensional atoms:

| Dictionary | Size flag | Default |
|---|---|---|
| shared `D_sh` | `--m-shared` | 256 |
| graph-private `D_G` | `--m-private` | 256 |
| sequence-private `D_S` | `--m-private` | 256 |

- Atoms are renormalised to unit L2 norm after every optimiser step. Do it in a hook in
  `hug_train.py`, not inside `forward`.
- Init: random unit vectors.

### B3. Encoders and codes
For each view, `a_v = ReLU(E_v p_v + b_v)`, of size `m_shared + m_private` (the view's own
private block). Split it into `a_v^sh` (the first `m_shared` entries) and `a_v^pr` (the rest).

### B4. Sparsity: top-k with a per-view evidence schedule (primary)

- **Shared block:** keep the top `k_sh` entries (`--k-shared`, default 16), fixed for every row.
- **Private block:** keep the top `k_pr,v(n_v)` entries, where

  ```
  s_v      = clip( n_v / n_ref,v , 0, 1 )            # n_ref,v = 95th percentile of n_v over TRAINING rows
  k_pr,v   = k_min + round( (k_max - k_min) * s_v )  # --k-private-min (4), --k-private-max (48)
  ```

- `n_ref,v` is computed from training rows only and stored in the checkpoint. The module
  must refuse to run `forward` in training mode until `n_ref` has been set explicitly (for
  example a `n_ref_set` flag buffer). A forgotten `n_ref` would silently push every row to
  `k_max`.
- Implement a per-row variable k by sorting each private block once and masking positions
  `≥ k_pr,v[row]` (vectorised; no Python loop over rows). Gradients flow through the kept
  entries only, as in standard top-k sparse autoencoders.
- `--k-schedule fixed` sets `k_pr,v = k_fixed` (`--k-private-fixed`, default 24) for every row.
  This is the "no evidence adaptation" ablation, at a comparable mean budget. Log the realised
  mean `k_pr` for the adaptive variant, and set `k_fixed` to that mean in the ablation run.

### B5. Alternative mechanism (optional; implement after everything else passes)

`--sparsity hardconcrete`: hard-concrete gates on each code (Louizos et al.), with an
expected-L0 penalty on the private block weighted by `λ_v(n_v) = λ0 / (1 + n_v)`. The shared
block uses a fixed `λ_sh`. Validation decides between this and top-k; the paper reports the
winner and puts the other in the appendix.

### B6. Losses
All losses are computed per batch and returned as `aux_loss`:

```
L_rec   = Σ_v || stopgrad(p_v) − (D_shᵀ a_v^sh + D_vᵀ a_v^pr) ||²            weight --w-rec   (1.0)
L_align = InfoNCE( norm(a_G^sh), norm(a_S^sh) ), positives = same row, τ=0.2  weight --w-align (0.1)
L_dec   = || Cov_batch(a_G^pr, a_S^pr) ||_F² / (m_private²)                  weight --w-dec   (0.1)
```

- **The reconstruction target is detached.** That's what stops `p_v` collapsing towards
  whatever is easiest to reconstruct. `p_v` learns from the task, alignment and
  decorrelation losses.
- `L_dec` is linear-kernel HSIC, i.e. the batch cross-covariance. It's cheap and needs no kernel
  bandwidth. Call it that in the paper.
- **Overhead:** reconstruction runs the encoder a second time on the detached `p_v`, so that
  `L_rec` can't reach `W_v` (test 4). That's expected. Count it in the §D overhead budget.
- **Dead atoms:** track how often each atom fires (EMA). Every 1,000 steps, any atom that hasn't
  fired in the last 10,000 steps is re-initialised towards the `p_v` residual of a random
  high-loss batch row, with its encoder row reset to match. Log the dead fraction per
  dictionary. If more than 30% are dead at the end of epoch 1 in smoke tests, report it before
  tuning. When an atom is reset, zero the optimiser state (Adam moments) for its dictionary
  and encoder rows too.

### B7. Head input

```
z = [ ½(a_G^sh + a_S^sh)  ‖  a_G^pr  ‖  a_S^pr  ‖  ctx ]        (sparse, m_shared + 2·m_private + ctx dims)
logit = MLP(z)        hidden [256, 128], LayerNorm + ReLU + dropout 0.1   (same as concat)
```

### B8. Statistics (logged per epoch; saved for validation rows)

- **Per row:** active counts per block, and the shared-mass ratio
  `ρ = (‖a_G^sh‖₁ + ‖a_S^sh‖₁) / (‖a_G‖₁ + ‖a_S‖₁)`.
- **Validation:** mean ρ overall and per evidence bucket (the Spec 04 W1.7 buckets plus `n_v`
  quintiles).
- **Shared-atom co-activation:** the fraction of shared atoms that, over validation, fire for
  *both* views on the same row at least 1% of the times they fire at all. Also report the mean
  per-row Jaccard overlap of the two views' active shared sets. This catches shared atoms
  degenerating into de facto private ones (the "split dictionary" failure in Kaushik et al.
  2026). If co-activation is low, report it before tuning.
- **Saved for analysis:** a fixed random sample of 200k validation rows (seed 0), with each
  row's nonzero code indices and values for all blocks and the row ids, written to
  `val_codes.npz` in the run directory. That's enough for atom-interpretability analysis
  without rerunning.

## C. Fusion baselines (RQ2)

Same encoder outputs, same head MLP, same training loop:

| Flag | Design | Extra hyperparameters |
|---|---|---|
| `concat` | today's head (= arm N4) | — |
| `gate` | `g = σ(W_g[p_G ‖ p_S])`; `z = [g⊙p_G + (1−g)⊙p_S ‖ ctx]` | — |
| `evgate` | **evidence-conditioned gate** (GateSID-style): `g = σ(MLP([p_G ‖ p_S ‖ s_G ‖ s_S]))` (one hidden layer, 64 units), where `s_v = clip(n_v / n_ref,v, 0, 1)` is the **same normalised evidence** `sparse` uses (same `n_ref`, from training rows); `z = [g⊙p_G + (1−g)⊙p_S ‖ ctx]`. It uses the same causal `n_G`, `n_S` as `sparse` | — |
| `moe` | 4 expert MLPs over `[p_G ‖ p_S ‖ ctx]`, softmax router, load-balancing loss (weight 0.01); logit = Σ router·expert | `--moe-experts` |
| `misa` | **dense** shared/private (MISA-style): one shared encoder applied to both `p_v`, a private encoder per view; losses: InfoNCE on shared (as `L_align`), orthogonality `‖H_shᵀ H_pr‖_F²` per view, reconstruction (detached target, as `L_rec`); head on `[½(h_G^sh+h_S^sh) ‖ h_G^pr ‖ h_S^pr ‖ ctx]` | same loss weights as sparse |

There are two key controls:

- **`misa`:** identical losses with no dictionaries and no sparsity. It isolates what sparsity,
  and then evidence adaptivity, add.
- **`evgate`:** evidence adaptivity *without* shared/private codes. It's the cheapest
  alternative explanation for gains on low-evidence items, and the literature check
  (`docs/literature/novelty_check.md` §5) flags it as the baseline reviewers will ask for. If
  `evgate` ≈ `sparse`-adaptive, the dictionary machinery isn't earning its place.

## D. Training and integration

- Train jointly with the encoders from scratch (default).
- `--init-from <checkpoint>` warm-starts the input layer and encoders from an N4 checkpoint
  (optional; used only if from-scratch training is too slow, and disclosed if so).
- Use the encoder hyperparameters from N4's tuned config. Only fusion hyperparameters are tuned
  for fusion jobs (§F).
- Must work with `--resume`, the test-set guard, metric schema parity and prediction dumps,
  exactly like the other HUG jobs.
- **Overhead target:** under 25% extra seconds per epoch over `concat`. Report it.

## E. Tests (`Framework/tests.py` or `test_fusion.py`; keep the synthetic suite under 5 min)

1. **Top-k is exact.** The mask keeps exactly `k_sh` shared positions and exactly
   `k_pr,v(n_v)` private positions per view, so each row has **at most** that many nonzero
   codes. After ReLU a row can have fewer positive entries than k, so test the mask rather than
   the nonzero count, and also check that no entry outside the mask is nonzero. `k_pr` is monotone non-decreasing in `n_v`
   and stays within `[k_min, k_max]`. The fixed schedule gives the same k for every row.
2. **Evidence is causal.** On synthetic data with ties, `n_G` and `n_S` equal brute-force
   counts over rows with `time < t` (item, author, history). Items with a missing metadata
   neighbour get `metadata_evidence = 0`. `n_ref` is computed from training rows only and is
   unchanged when val/test rows are rewritten. Training `sparse` or `evgate` without setting
   `n_ref` raises an error, and `n_ref` round-trips through the checkpoint.
3. **Atoms stay unit-norm** after an optimiser step, for all three dictionaries.
4. **Gradient routing.** `L_rec` gives no gradient to `W_v` or the encoders upstream of `p_v`
   (detached target). `L_align` gives gradient only through the shared blocks; `L_dec` only
   through the private blocks.
5. **Losses behave sensibly.** On toy data, `L_align` is lower for aligned views than for
   shuffled ones. `L_dec` is about 0 for independent codes and > 0 for correlated ones.
6. **Dead-atom resampling** fires on an atom forced to be dead and leaves live atoms untouched.
7. **No regression.** On the synthetic deterministic CPU setup, `--fusion concat` reproduces
   pre-Spec-05 N4 outputs bitwise: parameters after 2 epochs and validation predictions. This is
   what keeps `CODE_COMPAT_VERSION` unchanged (§0).
8. **Every variant runs.** `concat`, `gate`, `evgate`, `moe`, `misa`, `sparse` (adaptive and fixed), and
   `hardconcrete` if implemented: build, forward and backward on synthetic data, log stats, and
   with all aux weights set to 0 the loss equals BCE + L2. `--fusion X` combined with
   `--no-graph`/`--no-seq` raises an error unless X = concat.
9. **Prediction invariance** (`test_predictions_invariant_to_future`) holds for
   `--fusion sparse` with fixed weights.
10. **Weights don't depend on val/test:** rewrite every row at or after `t_val`, retrain
    `--fusion sparse` for 2 epochs, and all parameters are bit-identical (CPU, deterministic).
11. **Analysis outputs are valid.** ρ ∈ [0, 1]. `val_codes.npz` reloads, and its row ids are a
    subset of the validation rows with a stable sample (same rows every run).
12. **Hash semantics (§0).** A code change doesn't invalidate a finished job; bumping
    `CODE_COMPAT_VERSION` does.

## F. Heavy-run additions (prepared and dry-run here, executed in the heavy run)

Append to `experiments/heavy_run.yaml` as **Stage 1b**, which depends on `tune_hug` N4
completing:

1. **`tune_fusion` for `sparse` (adaptive):** 12 trials, random search, validation AUC only.

   | Parameter | Values |
   |---|---|
   | `m_shared = m_private` | {128, 256, 512} |
   | `k_shared` | {8, 16, 32} |
   | `(k_private_min, k_private_max)` | {(4, 32), (8, 48), (8, 64)} |
   | `w_align` | {0.05, 0.1, 0.2} |
   | `w_dec` | {0.01, 0.1} |
   | `w_rec` | {0.1, 1.0} |

2. **Tuning for the fusion baselines,** in proportion to how many hyperparameters each has:
   - `gate`: 4 trials (dropout, `fusion_dim`);
   - `evgate`: 6 trials (dropout, `fusion_dim`, gate hidden size {32, 64, 128});
   - `moe`: 6 trials (`experts` {2, 4, 8}, dropout);
   - `misa`: 8 trials (the same loss-weight grid as sparse, without the k parameters).

   Report every budget in the paper.
3. **Seeds:**
   - 3 seeds (42/43/44) for `sparse`-adaptive, `sparse`-fixed (with `k_fixed` = the adaptive
     run's realised mean `k_pr`), `evgate`, `misa`, `moe` and `gate`; `concat` reuses N4's three seeds;
   - 1 seed each for the loss ablations: `sparse`-adaptive with `w_align=0`, `w_dec=0` and
     `w_rec=0`.
4. **Stage 2** `final_eval` covers `sparse`-adaptive and every RQ2 variant reported in the
   paper's fusion table.

Reading order for the results, which feeds the paper's RQ2–RQ4:

| Comparison | Question |
|---|---|
| `concat` → `evgate` | does evidence-conditioned weighting alone help? |
| `evgate` → `sparse`-adaptive | do sparse shared/private codes add anything over an evidence gate? (**the novelty-critical comparison**) |
| `concat` → `misa` | does explicit shared/private decomposition help at all? |
| `misa` → `sparse`-fixed | does sparsity help? |
| `sparse`-fixed → `sparse`-adaptive | does evidence adaptivity help, and is the gain concentrated in low-`n_v` buckets? |
| loss ablations | which terms carry the effect |

## G. Smoke validation (the only runs in this spec)

Real 1K data, validation only, no `--eval-test`. Use N4's current (pre-tuning) defaults if
`tune_hug` hasn't finished.

1. One epoch of each fusion variant. Report:
   - seconds per epoch vs. `concat` (overhead);
   - peak memory;
   - val AUC (sanity check only);
   - for `sparse`: the dead-atom fraction per dictionary, mean active counts per block per
     evidence bucket (confirm `k_pr,v` varies as designed), and mean ρ.
2. A 50-step dry run of the queue over the new Stage 1b jobs, then aggregation over the dry-run
   outputs.
3. Write `docs/specs/05-sparse-fusion-results.md` with the above, and update `CHECKPOINT.md`.

## Not in this spec

- Changing the encoders (Spec 03/04).
- Atom-level bilinear crosses in the head (possible later extension).
- The interpretability *analysis*. Only the `val_codes.npz` dump is in scope; the analysis
  script comes with the results.
- Datasets other than KuaiRand-1K. The fusion code must stay dataset-agnostic: evidence counts
  come from `features.py` functions, not KuaiRand column names.

## Done when

- [ ] §0 harness amendment merged **before the heavy run starts**
- [ ] `Framework/fusion.py` with `ConcatFusion` (bitwise regression-safe), `GateFusion`,
      `EvidenceGateFusion`, `MoEFusion`, `MISAFusion` and `SparseSPFusion` (top-k adaptive/fixed; hard-concrete
      optional); causal as-of author counts in `features.py`; evidence features wired through
      the batcher
- [ ] Tests 1–12 pass; full synthetic and real-data suites pass
- [ ] Stage 1b jobs in `heavy_run.yaml`; dry run and aggregation green
- [ ] Smoke results and overhead reported; `CHECKPOINT.md` updated
