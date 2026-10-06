# Spec 05b: fixes to the sparse fusion head before Stage 1b

**Status:** ready for implementation. **It blocks Stage 1b fusion tuning** (`tune_fusion_*`
and everything downstream of it). Stage 1a (HUG N4/N1 and baseline tuning) is unaffected and
already running from `~/HUG-heavy`.
**Author:** architect session, 2026-10-06
**Input:** `docs/specs/05-sparse-fusion-results.md` (one-epoch smoke of every head).

## Why

The smoke run found four problems with `sparse`, and the same loss-scale problem in `misa`:

1. **The auxiliary loss swamps the task.** `aux` is 6–12× the BCE. `L_rec` sums squared error
   over the 64 LayerNorm dimensions (≈ 64 at init), and `L_align` is an InfoNCE over the whole
   batch of 8,192 rows (≈ log 8192 ≈ 9 at chance). Both are on scales unrelated to the BCE.
2. **The shared dictionary splits by view.** Only 4.2% of shared atoms co-fire across views
   (row Jaccard 0.08). Each view has its own encoder for the shared block, so nothing makes
   atom *j* mean the same thing in both views. InfoNCE at weight 0.1 doesn't fix that.
3. **Sparse codes generalise worse across time.** `sparse` trains better than concat (BCE
   0.558 vs 0.561) and is close on the in-period holdout, but loses 0.03 val AUC on the next
   period. The fixed-k schedule loses the same, so this isn't the evidence schedule. Top-k is a
   hard selection, so small shifts in `p_v` between periods change which atoms fire.
4. **The dead-atom window is in steps, and it's too long.** At 10,000 steps, resampling can't
   start before epoch 11. Measured on validation, 26–48% of atoms never fire.

## Changes

### 1. Loss scales (`sparse` and `misa`)
- `L_rec` = mean over dimensions of the squared error (divide by `d'`), averaged over rows,
  summed over the two views. That's ≈ 2 at init.
- `L_align` = InfoNCE over a random subset of **1,024 rows** of the batch (or the whole batch if
  it's smaller), **divided by log(n)**, so chance = 1.0. Use the same subset size in `misa`.
- `L_dec` unchanged.
- New defaults: `--w-rec 0.1`, `--w-align 0.3`, `--w-dec 0.1`. Log `aux / bce` per epoch.

### 2. One shared encoder for both views (`sparse`)
- The shared block is `a_v^sh = ReLU(E_sh p_v + b_sh)` with **one** `E_sh` (m_shared × d')
  for both views, as in `misa`. Each private block keeps its own `E_v`.
- This makes `misa` → `sparse` differ **only** in dictionaries and sparsity, which is the
  comparison the paper needs.
- Resampling a shared atom resets its row of `E_sh`, plus the Adam state.

### 3. Optional dense skip path (`--fusion-dense-skip`, default off; the probe decides)
- With the flag, the head input becomes
  `z = [p_G ‖ p_S ‖ ½(a_G^sh + a_S^sh) ‖ a_G^pr ‖ a_S^pr ‖ ctx]`.
- It applies to `sparse` and `misa` alike, so the pair stays matched.

### 4. Dead atoms, measured in rows
- An atom is dead if it hasn't fired in the last **one epoch's worth of training rows**
  (`dead_after_rows = n_train_rows`, converted to steps with the batch size).
- Check every **100 steps**, and start checking once `dead_after_rows` rows have been seen.
- Log the dead fraction per dictionary at every epoch end, and on validation, as in the
  results doc.

### 5. Tuning grid (`tune_hug.py` `LOSS_GRID`, used by `sparse` and `misa`)
`w_rec ∈ {0.03, 0.1, 0.3}`, `w_align ∈ {0.1, 0.3, 1.0}`, `w_dec ∈ {0.01, 0.1}`. Keep the
m/k grid and the trial budgets from Spec 05 §F. The dense-skip setting is fixed by the probe
(§Probe), not tuned.

## Tests (add to `test_fusion.py`)
- `L_rec` is ≈ 2 and `L_align` is ≈ 1 at init on random data. `aux / bce` is under 2 at init
  with the default weights.
- Shared atoms: the same `E_sh` weights serve both views (the parameter is identical), and a
  gradient from either view reaches it.
- Gradient routing (old test 4) still holds with the shared encoder.
- Dead-atom window: an atom forced idle for one epoch's worth of rows is resampled at the next
  100-step check, and an atom idle for less isn't.
- `--fusion-dense-skip`: the head input width is `2d' + m_shared + 2·m_private + ctx`; without
  the flag it's unchanged.
- Test 7 (concat bitwise) still passes. **No `CODE_COMPAT_VERSION` bump**: no fusion job has
  run in the heavy run.

## Probe (the only runs in this spec)
KuaiRand-1K, N4 pre-tuning defaults, seed 42, **3 epochs**, validation only.
**Where:** the idea A100 once its port check passes (it's free and has KuaiRand), or a brains
GPU if one frees up. **Never** inside `~/HUG-heavy`, and never on a brains GPU that the
Stage 1a queue is using.

| Run | Head |
|---|---|
| P0 | `concat` (3-epoch reference) |
| P1 | `sparse` with §1, §2 and §4 (codes only) |
| P2 | P1 + `--fusion-dense-skip` |
| P3 | `evgate` (3-epoch reference) |

For each epoch, report: val/holdout AUC, BCE, `aux/bce`, co-activation and row Jaccard of the
shared sets, dead fraction per dictionary, and the k_pr range per evidence quintile.

**Decision rule** (report the numbers to the architect; don't proceed on your own):
- Choose the better of P1 and P2 by epoch-3 val AUC.
- If it's within **0.003** of max(P0, P3), Stage 1b goes ahead with that setting.
- Otherwise, stop. The paper's framing changes, and the architect decides what to do with
  Stage 1b.

## Done when
- [ ] §1–§4 implemented with flags. The tests above pass, as does the full synthetic suite.
- [ ] The `LOSS_GRID` update is committed, and Stage 1b jobs in `heavy_run.yaml` carry the
      probe's dense-skip choice.
- [ ] Probe results are in `docs/specs/05b-fusion-fixes-results.md`.
