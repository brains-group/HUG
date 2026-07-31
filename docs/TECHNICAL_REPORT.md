# HUG — Technical Report and Post-Mortem

**Subject:** *HUG: A Heterogeneous Unified Graph Framework for Conversion Rate Prediction*
**Venue:** ACM RecSys 2026, submission #353
**Outcome:** Reject (scores: +1 weak accept, −2 reject, −2 reject). Metareview recommends resubmission after extending evaluation.
**Report date:** 2026-07-30
**Scope of audit:** `Framework/` (all modules), `Baselines/` (FuxiCTR harness + KGAT), `runs/` (13 completed sweeps), `Reviews.txt`, `Rebuttal.txt`, submitted PDF.

---

## 1. Executive summary

The reviewers converged on a correct verdict via largely incorrect reasoning. The paper's core idea — decomposing static relational structure from transient session intent and matching an encoder's inductive bias to each — was praised by all three reviewers and remains defensible. What failed is the chain of evidence connecting that idea to the reported numbers.

Three independent problems break that chain, in descending order of severity:

1. **Look-ahead leakage in graph construction.** The HKG is built over the entire interaction log, including the test window. The prediction target (`is_click`) is materialised as a first-class graph edge. Reviewer 2 asserted leakage via a mechanism that does not exist in the code (per-user splitting), and the rebuttal correctly refuted that mechanism — while the actual leakage went unmentioned by both sides.

2. **The baseline comparison is not a comparison.** DIN and BST are preprocessed from a different, smaller subset of the dataset and evaluated under a *user-disjoint* split, meaning they are tested cold-start on users never seen in training. HUG is evaluated transductively on the full corpus. The headline "+8.81% AUC over BST" measures this discrepancy, not architecture. No reviewer found this.

3. **The architecture cannot support the claim being made of it.** The GNN encoders are frozen at random initialisation by design (documented in Algorithm 2). "Matching the inductive bias of each encoder to the temporal character of its modality" is a claim about what an encoder *learns*; it cannot be demonstrated with encoders that never learn. Every named contribution — KGA gate, recency gate, IPS weighting — is empirically inert, and two of the three are inert for mechanical reasons that have nothing to do with the dataset properties cited in the rebuttal.

The good news is that the intellectual asset survives all three. The decomposition thesis is untested, not refuted. Section 6 of the companion workflow document lays out how to test it properly, and §7 of this report proposes architectural directions that strengthen modality separation rather than abandoning it.

**The single most important strategic point:** the rebuttal won a narrow argument about split methodology while the substantive version of that criticism remains unaddressed and is verifiable from the public repository. Resubmitting on the current framing — "we already demonstrated there is no look-ahead bias" — invites a far worse outcome than a soft reject.

---

## 2. Review triangulation

### 2.1 What the reviewers got right

| Claim | Reviewer | Verdict |
|---|---|---|
| KGA gate has negligible effect | R1, R2, R3 | **Correct**, and more strongly than stated — the gate never executes at all (F4) |
| IPS weighting is mathematically inert | R2 §4, R3 §3 | **Correct**. R2's phrase "mathematical theater" is accurate (F5) |
| Recency gate degrades performance | R1, R2 §3 | **Correct**, reproduced in `runs/` (rg0 = 0.7670 vs rg1 = 0.7625) |
| Headline AUC 0.7683 is cherry-picked from a sweep | R2 §5 | **Correct**, and worse than stated — `L=1` was selected on the test set (F9) |
| Gains stem from richer features, not algorithmic innovation | R2 §3 | **Correct**, and understated — the feature registry is *randomly projected* (F3) |
| Capacity confound: DIN/BST at dim 16 vs HUG at 128/64 | R3 §2 | **Correct and unaddressed.** The rebuttal answers only for KGAT |
| Single-dataset evaluation limits validation | R1, R3 | **Correct** |
| No variance / significance testing | R2 Q4 | **Correct**. Single seed (42) throughout; all ablation deltas sit inside ±0.005 |

### 2.2 What Reviewer 2 got wrong — and why winning that point was dangerous

R2's weakness #1 and clarification Q2 assert *per-user* temporal partitioning, producing look-ahead bias across users on different timelines.

**The code does not do this.** `Framework/main.py:294-299` sorts all interactions globally by `time_ms` and cuts at `int(len(df) * (1 - test_ratio))`. This is a strict global chronological split. The rebuttal (R2-Q2) is factually correct.

**But the paper says otherwise.** §4.1 of the submitted PDF states the split assigns "the earliest 80% of *each user's* interactions to training and the most recent 20% to test." R2 read the paper accurately and criticised what it described. The rebuttal then contradicted the paper without acknowledging that the paper's own methods section was the error.

From the area chair's position this reads as a flat denial of a documented reviewer concern, which is a plausible contributor to "the rebuttal clarified several details, [but] it did not address all reviewer concerns."

**Action:** correct §4.1 to describe the implemented global split, and in any resubmission explicitly flag the correction rather than relitigating.

### 2.3 What everyone missed

Two findings in §3 (F1 leakage, F2 baseline preprocessing) appear in neither the reviews nor the rebuttal. Both are verifiable in under ten minutes by anyone who clones the repository — which the paper advertises, and for which all three reviewers assigned a reproducibility score of 4/5. These are the primary resubmission risks.

---

## 3. Findings

Severity: **S1** invalidates reported results · **S2** undermines a stated contribution · **S3** correctness or hygiene defect.

---

### F1 — Look-ahead leakage in HKG construction · S1

The HKG is built once from the complete interaction log with no temporal filtering (`hkg_constructor.py`, `build()` consumes `data.log_combined`). Three distinct leakage channels result:

**(a) The label is an edge.** `Framework/main.py:313` registers `("user", "clicked", "video")` as relation id 0 in `REL_MAP`. A test-set pair `(u, v)` with `is_click = 1` therefore has that exact positive label present in the graph the encoder reads. Message passing over relation 0 propagates test-window click structure into both user and video embeddings.

**(b) Session edges span the split boundary.** `hkg_constructor.py:351-355` builds `next_in_session` edges over the full log. A test-window session contributes edges linking its own items to each other, so the session embedding used to predict an interaction encodes what the user actually consumed next.

**(c) Item statistics are computed over the full corpus.** `data_loader.py:364-372` derives `global_cvr` (`valid_play_cnt / show_cnt`) and seven `*_cnt_log` popularity features from the dataset-wide statistics file. These aggregate the test period into every video's node features.

Channels (a) and (b) are transductive leakage of the target. Channel (c) is temporal leakage of item popularity. All three inflate every HUG and KGAT number in Table 3.

> **Note on interaction with F3.** Frozen random encoders do not neutralise this. A random projection of a neighbourhood aggregate still carries the aggregate's information; the trained MLP head learns to read it. Freezing changes the encoding, not the leak.

**Remediation:** build the HKG from training-window interactions only; recompute item statistics from the training window; drop or time-mask `next_in_session` edges crossing the boundary. See workflow Phase 1.

---

### F2 — DIN/BST baselines use a different dataset and an invalid split · S1

`Baselines/preprocess.py` produces the FuxiCTR train/valid/test files. Two defects:

**(a) Wrong data subset.** Lines 36-38 read a single file:
```python
log = pd.read_csv(
    os.path.join(RAW_DIR, "log_standard_4_08_to_4_21_1k.csv"),
```
Only the early standard-policy log is loaded. HUG's loader (`data_loader.py`, `FILES_1K`) consumes both standard logs *and* the random-policy log — the full 11.7M interactions cited in the paper. The baselines see a fraction of that, from an earlier and non-overlapping time window.

**(b) User-disjoint split presented as chronological.** Line 57 sorts by `["user_id", "time_ms"]`; lines 148-150 then split positionally:
```python
train = log.iloc[:t1]
valid = log.iloc[t1:t2]
test  = log.iloc[t2:]
```
Because the primary sort key is `user_id`, this partitions by *user*, not by time. DIN and BST are trained on roughly the first 80% of user IDs and evaluated on users with no training history whatsoever — a pure cold-start regime — while HUG is evaluated transductively on users whose interactions saturate the graph.

The comment on line 143 ("chronological train/valid/test split") and the paper's §4.1 both describe a split the code does not perform.

**Consequence:** the paper's central quantitative claim ("8.81% relative improvement in AUC and 19.60% in AP over the best sequence baseline") is not measuring architecture. Combined with R3's unaddressed capacity confound (embedding dim 16 vs 128/64), the baseline row of Table 3 cannot be defended.

**Remediation:** regenerate baseline data from the identical loader and identical global chronological split used by HUG; re-tune baseline capacity to parity. Workflow Phase 1.

---

### F3 — Frozen encoders cannot support the inductive-bias claim · S2

This is a **design choice, not a defect**, and it is documented: paper §3.2 ("two GNNs serve as *structured feature extractors*: their role is to encode topological and temporal context into fixed embedding matrices"), and Algorithm 2 ("**Require:** Frozen GNN parameters Θ_str, Θ_seq"). The implementation matches — `models.py:359` decorates `encode_graph` with `@torch.no_grad()`, `main.py:530-532` detaches the output to CPU, and `forward_from_embeddings` (`models.py:420-423`) indexes into detached tensors. Adam therefore steps only `AlignmentModule` and `CVRHead`. The KGAT baseline is frozen identically (`Baselines/KGAT/kgat_model.py:232`), so that particular comparison is at least controlled.

The problem is the mismatch between this design and the claims built on it:

- §1 promises "we match the inductive bias of each encoder to the temporal character of its modality." An untrained R-GCN and an untrained GGNN are two arbitrary nonlinear functions of a neighbourhood aggregate. Inductive-bias matching is a statement about learned representations and is not evidenced here.
- §5.3 attributes `L=1 > L=2 > L=3` to over-smoothing. With frozen random weights, the parsimonious explanation is that each additional untrained layer compounds random mixing and destroys signal. The observed monotone degradation (0.7683 → 0.7625 → 0.7549) fits that reading cleanly.
- R2 inferred the conclusion without repository access: "the framework reduces to a standard concatenation of established GNN models [whose] performance advantages... are a trivial consequence of supplying HUG with a richer, graph-structured multi-modal feature registry." That is accurate, and the code makes it stronger than R2 realised — the registry is *random*.
- The paper never justifies freezing (no efficiency argument, no overfitting argument) and never reports the trained comparison.

The docstring at `models.py:409-411` compounds the confusion by asserting that "the GNN weights are updated implicitly because encode_graph is called with the live model weights at the start of each epoch." That is not how autograd works and contradicts Algorithm 2. It should be deleted regardless of which direction the project takes.

**This finding is not a reason to abandon the frozen design.** It is a reason to *test* it — see F3-followup in the workflow (Phase 2), where frozen / self-supervised-pretrained / end-to-end becomes the decisive experiment.

---

### F4 — The KG alignment gate never executes · S2

`AlignmentModule.forward` applies the gate only when `kg_relation is not None` (`models.py:137-140`). The training loop never constructs or passes it — `main.py:976-983` calls `forward_from_embeddings` without the argument, so it defaults to `None`. `--kg-alignment` allocates `self.kg_proj` (`main.py:412`) and the parameter is never used in a forward pass.

Paper Equation 2 defines the gate as an additive per-head bias `Φ_k(R_kg)` on the attention logits; that term is absent at runtime.

This explains the ablation exactly: `kg0 = 0.7626` vs `kg64 = 0.7625` are two runs of an identical model differing only by an unused parameter tensor. §5.2's interpretation — "the alignment gate remains architecturally motivated and may provide clearer gains on graphs with denser or more varied relational paths" — attributes to graph sparsity an outcome caused by the feature never being invoked. The same applies to R2's §3 explanation (the gate vector "approaches zero for the vast majority of pairs"); plausible, but not what happens.

**Remediation:** implement `r_uv` (relation-path features between user and candidate) and pass it through the loop, or remove the component and all associated claims. Workflow Phase 3.

---

### F5 — IPS weighting is a no-op · S2

`main.py:288-289` assigns weights of `1.0` to random-policy rows and `0.9963` to standard-policy rows. `CVRHead.ips_bce_loss` then clips and mean-normalises (`models.py:241`), mapping the two values to ≈1.0004 and ≈0.9967. The gradient contribution difference is ~0.04%.

More fundamentally this is not an inverse propensity score. An IPS weight is `1 / P(exposure | context)`; this is a global constant keyed on a binary policy flag, with no propensity model anywhere in the codebase. Paper §4.3 is candid that it is "a pragmatic normalization mechanism... without requiring propensity estimation," but the abstract and §1 frame the work in terms of exposure-bias correction, and the component is presented as a contribution in Table 3.

R2's assessment is correct. The rebuttal's defence — that the framing "is intended to motivate the mechanism for datasets with larger intervention fractions" — is unpersuasive without a dataset that has one.

**Remediation:** either (a) evaluate on KuaiRand-Pure, whose random-exposure fraction is large enough for IPS to be meaningful, and estimate real propensities; or (b) remove IPS and its claims. Option (a) directly answers R1's key clarification. Workflow Phase 5.

---

### F6 — The cross-attention module is degenerate · S2

`models.py:127` computes `attn_s` with shape `[B, H]` — one scalar per attention head — then `models.py:142` applies `torch.softmax(attn_s, dim=-1)`, normalising **across heads**, not across keys.

Standard attention normalises over a key set. Here each query attends to exactly one key (the session embedding for direction A, the structural video embedding for direction B), so a correct softmax over keys would return 1.0 identically. What the code computes instead is a learned convex weighting over heads applied to a single value vector — a gated linear reweighting, not attention.

Paper Equation 2 writes `α_k = softmax((W_Q h)ᵀ(W_K h)/√(d/H) + Φ_k(R_kg))` with subscript `k` indexing heads, so the equation and the code agree — but neither implements the "dual cross-attention mechanism" that §3.4 describes in prose, because there is no set to attend over.

**Remediation:** either attend over a genuine key set (session item sequence, or multi-hop KG neighbours of the candidate), or rename the component honestly as gated fusion and drop the attention framing. Workflow Phase 3.

---

### F7 — The "sequential" encoder models global transitions, not sessions · S2

`hkg_constructor.py:351-355` writes all `next_in_session` edges from all sessions into a single `("video", "next_in_session", "video")` edge index, retaining `session_id` as a per-edge attribute. `SequentialGNN.forward` then runs one pass over the entire structure (`gnn_encoders.py:267`: `self.ggnn(h, edge_index)`).

Message passing therefore mixes transitions from every session in the corpus into a global item-transition graph. Session identity enters only at readout (`_session_readout`), which pools node embeddings that have already been contaminated by every other session's transitions.

This is not SR-GNN. SR-GNN constructs a *per-session* graph and propagates within it; that locality is the entire point of the architecture. The current encoder computes something closer to a global co-occurrence embedding — which is a legitimate signal, but it is a *structural* signal, not a session-dynamics signal. The paper's decomposition claim is undermined by its own implementation: both branches are consuming corpus-level structure, differing mainly in which edges they see.

This may be the single most important architectural finding in this report, because it means the dual-view thesis has never actually been instantiated.

**Remediation:** batch per-session subgraphs (true SR-GNN), or replace the branch with a sequence encoder operating within session boundaries. Workflow Phase 3.

---

### F8 — Both encoders are forced to share message-passing depth · S2

`main.py:1072` defines a single `--gnn-layers` flag, passed to both `StructuralGNN` and `SequentialGNN` in `build_model` (`main.py:394-411`). There is no way to configure `L_str ≠ L_seq`.

This matters because it is precisely the compromise the paper's own thesis predicts should be harmful. Structural collaborative signal typically benefits from `L ≥ 2` (second-order user–item–user paths); session-order signal is short-range and degrades under repeated smoothing. Forcing a shared `L` guarantees at least one branch is mis-specified, and the sweep then reports the best *compromise* value rather than the best configuration.

R2 read the resulting `L=1` optimum as a "structural paradox" showing the architecture "cannot support deep graph propagation or exploit high-order relational topologies." Under decoupled depth, that same observation becomes *evidence for* the decomposition thesis rather than against it. Adding two flags is trivial and could produce the paper's central result.

---

### F9 — No validation split; hyperparameters selected on test · S1

There is no validation set anywhere in `Framework/main.py`. `temporal_split` returns exactly two arrays. Consequences:

- `main.py:1402` checkpoints on `test_m.auc`. (Mitigating: Table 3 reports final-epoch metrics, so the reported values are not max-over-epochs.)
- The `L`, `h`, `d` sweep in §5.3 is evaluated entirely on test. Selecting `L=1` as the headline configuration is model selection on the test set.
- The rebuttal (R2-Q3) commits to designating `L=1` as the recommended setting in revision. **Doing so without a validation split converts a presentation complaint into a methodology violation.** This must not go into the resubmission as promised.

**Remediation:** three-way global chronological split (train / val / test); all selection on val; test touched once per reported configuration. Workflow Phase 0.

---

### F10 — HGT attention normalisation is mathematically invalid · S3

`_HGTLayer.forward` accumulates a softmax denominator per destination node across edge types. Line 647 exponentiates with a **per-edge-type** maximum subtracted:
```python
exp_t = torch.exp(attn_t - attn_t.max())
```
Line 650 then scatter-adds these into a shared denominator, and the second pass normalises with it. Because each edge type is scaled by a different constant `exp(-max_t)`, the resulting weights do not form a valid softmax for any node receiving edges of more than one type — i.e. essentially every node.

Additional defects in the same class: the "residual update" comment at line 661 precedes a non-residual update (line 662 has no `+ h` term); the `W_Q` initialiser at lines 589-590 is a garbled chain of `view`/`unsqueeze`/`squeeze`/`reshape` calls whose fan-in semantics differ from those applied to `W_K`/`W_V`.

Affects HUG-Unified, which is the stepping-stone baseline underpinning the "+4.24% AUC from elevating metadata to topology" claim in §5.1.

---

### F11 — Scalability claims are unsupported; 27K path does not execute · S3

`main.py:794` calls `F.relu(...)` inside `encode_epoch_chunked`. The module never imports `F`; `torch.nn.functional` is imported as `F_local` five lines later at line 799. Any `--scale 27k` run raises `NameError` on entry to the first chunked encode. Consistent with `runs/` containing 1K results exclusively.

Independently, `_session_readout` (`gnn_encoders.py:298`) loops in Python over every session, and each iteration evaluates `session_id == sid` across the full edge tensor — O(n_sessions × E) work per encode. This is the likely original motivation for freezing the encoders (F3), since end-to-end training would require this in the backward path every step.

Both are avoidable with machinery already imported and unused: `pyg_softmax` (`gnn_encoders.py:29`) and `NeighborLoader` (`main.py:79`). Vectorising the readout with `scatter` + segment softmax removes the bottleneck and makes end-to-end training tractable.

---

### F12 — Result-file collisions and stale reporting · S3

`main.py:1472` reassigns `out_dir` immediately before writing `final_metrics.json`:
```python
out_dir = Path(args.output_dir) / f"kuairand_{args.scale}_{args.model_type}"
```
This discards the run-specific name from `make_run_name()` used for `history.json`. Every `dual` ablation therefore writes `final_metrics.json` to the same path, each overwriting the last. Only `history.json` is per-configuration — which is why the surviving per-run evidence in `runs/` is history files.

Related: `main.py:1417` loads the "best checkpoint" for final evaluation from `cache_dir`, but `save_checkpoint` writes to `out_dir` (`main.py:1404`). The final-evaluation block generally re-evaluates last-epoch weights, or stale checkpoints from an unrelated run sharing the cache directory.

`README.md` reports an older archived sweep (0.7398 / 0.7344 / 0.7306) that contradicts both Table 3 and the current `runs/` directory (0.7625 / 0.7626 / 0.7306). A reviewer comparing repository to paper will see two different result sets.

---

### F13 — Single-seed experiments · S1 for the ablation claims

`main.py:1051` defaults `--seed 42` and `run_experiments.sh` never varies it. Every number in Table 3 is one run.

The ablation deltas the paper interprets are: KGA ±0.0001, IPS ±0.0003, RG ±0.0045. Without variance estimates none of these are interpretable, and R2's Q4 request for paired t-tests across initialisations is entirely reasonable. Given F1's leakage and F2's baseline invalidity, the honest current status of Table 3 is that it supports no comparative conclusion at all.

---

### F14 — Minor defects · S3

| Ref | Issue |
|---|---|
| `main.py:298-299` | Interaction arrays cast to `float32`; indices above 2²⁴ lose precision. Safe at 1K, breaks silently at 27K scale (32M videos) |
| `main.py:968` | `session_idx.clamp(0, n_sess-1)` silently masks out-of-range session indices rather than surfacing them |
| `data_loader.py:185` | `_filter_active_users` applied to `log_std` only; random-policy log bypasses the `min_interactions` filter, so the two policies have different user populations |
| `hkg_constructor.py:43` + `main.py:312-320` | `is_follow → "followed"` edges are constructed but appear in neither `REL_MAP` nor `_extract_structural_subgraph`'s type list, and are silently discarded |
| `models.py:409-411` | Docstring contradicts Algorithm 2 and misstates autograd semantics |

---

## 4. What the current numbers actually measure

Assembling F1, F2, F3 and F7, the quantity reported as HUG-Dual's test AUC is:

> the performance of a **trained MLP** reading **randomly-projected** neighbourhood aggregates drawn from a graph that **contains the test labels as edges** and **corpus-wide item popularity**, compared against sequence baselines trained on a **different data subset** and evaluated **cold-start**.

Each clause independently invalidates the comparison. This is not a statement about the dual-view architecture, in either direction — the architecture has not yet been measured.

The ablation table is more informative than the headline, because ablations are internally controlled (same leakage, same seed, same data). Read that way it says:

| Comparison | Δ AUC | Mechanical explanation |
|---|---|---|
| KGA on vs off | −0.0001 | Gate never executes (F4) |
| IPS on vs off | +0.0003 | Weights differ by 0.37% then mean-normalised (F5) |
| RG on vs off | −0.0045 | Gate operates on globally-contaminated session embeddings (F7) |
| L=1 vs L=2 vs L=3 | 0.7683 / 0.7625 / 0.7549 | Compounding random mixing in untrained layers (F3), plus forced shared depth (F8) |

Every ablation outcome has a mechanical explanation upstream of the architecture. None currently supports or refutes the decomposition thesis.

---

## 5. Resubmission risk register

| Risk | Likelihood if unaddressed | Impact |
|---|---|---|
| Reviewer clones repo and finds F2 (baseline split) | High — repo is advertised, reproducibility scored 4/5 | Fatal. Reads as manufactured baselines |
| Reviewer finds F1 (leakage) after the rebuttal denied leakage | High | Fatal. Prior denial becomes an integrity question |
| Reviewer discovers F4 (dead KGA code) against Equation 2 | Moderate | Severe. Equation describes unexecuted code |
| `L=1` promoted to recommended config per rebuttal, still no val split | Certain if the rebuttal commitment is honoured | Severe. Explicit tuning on test |
| InterFormer / OneTrans / HyFormer availability dispute reopened | Moderate | Severe. R2 used "fabricating a claim" |
| Ablation deltas still reported single-seed | Certain if unaddressed | Moderate. Direct unmet reviewer request |

On the last-but-one row: this report takes no position on whether those three repositories are public, and the claim should not be re-litigated. The defensible move is to delete the closed-source assertion entirely and add whichever modern baselines can actually be run.

---

## 6. What survives

Worth stating plainly, because the finding list is long:

- **The decomposition thesis is intact.** All three reviewers independently endorsed the intuition; R1 called it "conceptually intuitive and well-reasoned," R2 (the harshest) called the motivation "well-motivated by the differing temporal properties of these signals." It has not been tested, which is different from having been refuted.
- **The HKG substrate is genuinely useful.** Five entity types and twelve interaction edges over KuaiRand is real engineering, reusable, and the KGAT adaptation onto the same substrate was singled out as "engineering diligence."
- **The transparency was noticed and credited.** R1 explicitly praised honest reporting of components that fail. That credibility is an asset; the revision should lean into it by reporting the corrections in this document openly rather than quietly fixing them.
- **The infrastructure is sound.** Caching, run naming, sweep orchestration, and the FuxiCTR harness all work. The repairs in the workflow document are mostly surgical, not rewrites.

---

## 7. Architectural directions

The brief for the next iteration is explicit: strengthen the separation between the two data modalities rather than collapsing them into a single unified sequence-plus-features transformer. Seven directions, ordered by expected return relative to effort. Full task breakdowns are in `REVISION_WORKFLOW.md` §Phase 3.

### 7.1 Decoupled propagation depth — *highest return, lowest effort*

Give each branch its own depth (`L_str`, `L_seq`) and sweep them independently (F8). The thesis predicts an interior optimum with `L_str > L_seq`: collaborative structure needs multi-hop diffusion, session order does not survive repeated smoothing.

This converts the paper's most damaging empirical result into its central supporting evidence. R2 read `L=1` as proof the architecture cannot exploit high-order topology; the decoupled sweep tests whether the true reading is that a *shared* depth forced a bad compromise. A 2-D grid over `L_str × L_seq ∈ {1,2,3}²` is nine runs on already-working code.

### 7.2 Repair the sequential branch so it is actually sequential

Address F7 by constructing per-session subgraphs and propagating within session boundaries, as SR-GNN specifies. Until this is done the two branches are both consuming corpus-level structure and the "dual-view" framing is not instantiated.

This is a precondition for every other item in this section: without it, any measured difference between branches is confounded.

### 7.3 Two-timescale state instead of two-subgraph views — *strongest reframing*

The deepest version of the thesis is not "two subgraphs" but **two timescales**. Structural preference is slow-varying; session intent is fast-varying. Make that explicit in the state, not just the topology:

- Maintain a **slow user state** updated with an EMA or a low learning rate across time windows, and a **fast session state** recomputed from scratch each session.
- Frame it in graph-signal terms: the structural branch is a low-pass filter over the interaction graph (which is what stacked GCN layers compute), the session branch is order-sensitive and band-limited. Depth then has a principled interpretation rather than being a hyperparameter, and §7.1's predicted asymmetry follows analytically.

This gives the paper a theoretical spine it currently lacks, and it explains the `L` sweep rather than apologising for it. It also generalises beyond KuaiRand, which speaks directly to the metareview's single-dataset concern.

### 7.4 Complementarity objective instead of attention fusion — *strongest novelty candidate*

The current "alignment" is a degenerate gate (F6). Replace it with an objective that makes the decomposition **measurable**:

- Add a cross-view term that aligns the shared component of the two representations while **decorrelating** the view-specific components — a Barlow Twins / VICReg-style covariance penalty, or an InfoNCE term over matched `(user, item)` pairs across views.
- This yields a number the paper currently cannot produce: *how much information is shared versus view-specific*. If the views are 90% redundant, that is a publishable negative result about decomposition. If they are complementary, it is the mechanism evidence the paper needs.

Either outcome is reportable, which is exactly the property the current ablation table lacks.

### 7.5 Identifiable fusion: routing or product-of-experts

Replace concatenation-then-MLP with a fusion whose behaviour can be read off:

- **Mixture-of-experts routing** with load balancing: a router chooses per `(user, item, context)` whether structural or sequential evidence dominates. Routing weights are directly interpretable — one can *show* that cold-start users route structural and long-session users route sequential.
- **Product of experts in logit space:** `logit = logit_str + logit_seq`, each branch calibrated independently. Ablation becomes meaningful by construction, since each expert has a standalone AUC.
- **Gated residual with reported gate statistics:** `z = h_str + g ⊙ (h_seq − h_str)`, publishing the distribution of `g`. If `g ≈ 0` everywhere, the sequential view contributes nothing and that is a finding.

All three make the fusion falsifiable, which the current module is not.

### 7.6 Resolve the frozen-versus-trained question properly

Rather than defending or abandoning the frozen design (F3), make it an axis:

1. **Random frozen** (current behaviour)
2. **Self-supervised pretrained, frozen** — link prediction on the structural graph, next-item prediction on session graphs
3. **End-to-end trained** — requires the vectorised readout and neighbour sampling from F11

Run all three on leak-free data. This is the decisive experiment: it determines whether the paper's contribution is architectural or a demonstration that structured feature aggregation suffices. Row 2 is independently interesting and is the natural home for the "feature extractor" framing the paper already uses.

### 7.7 Differential evidence instead of aggregate deltas

Aggregate AUC differences of 0.005 will never survive review. Mechanism evidence will. Stratify every reported comparison by:

- **User history length** (cold-start → heavy) — structural branch should dominate the cold end
- **Session length** (1–2 items → long) — sequential branch should dominate the long end
- **Item popularity** (long-tail → head) — where KG paths should matter most
- **Time-to-prediction** within the test window — tests degradation under distribution shift

"The structural branch carries cold-start users while the sequential branch carries long sessions, and the fusion recovers both" is a defensible RecSys contribution even if the aggregate delta is flat. That is the paper this project should be aiming at.

---

## 8. Two viable venue paths

The frozen-versus-trained result (§7.6) determines which of these to pursue. Do not commit before that experiment reports.

**Path A — Architecture paper.** If trained encoders win on leak-free data: unfreeze, restore inductive-bias matching as a supported claim, land §7.1 + §7.2 + §7.5, add a second dataset and repaired baselines. Conventional full-paper submission. Highest ceiling, most work.

**Path B — Analysis / reproducibility paper.** If frozen matches or beats trained: reframe around "how much of graph-based CVR performance comes from learned message passing versus structured feature aggregation?" The existing evidence already points this way — shallow beats deep, every learned gate inert. Requires the frozen/trained comparison across *all* baselines to carry it, plus honest reporting of the defects in this document. RecSys has a reproducibility track; this converts every current weakness into the finding.

Path B is not a consolation prize. A rigorous negative result about learned message passing in CVR, backed by a clean leak-free protocol and an untrained-GNN control that most papers in this area omit, is more useful to the field than another 0.005 AUC gain — and it is substantially more likely to survive review than Path A on a compressed timeline.

---

*Companion document: `docs/REVISION_WORKFLOW.md` — phased execution plan, acceptance criteria, and pre-submission checklist.*
