# Spec 06: Second and third datasets (MIND, ZhihuRec)

**Status:** ready for implementation after Spec 04 lands. It can run in parallel with Spec 05:
the two touch different modules, except for `features.py`, which both extend.
**Author:** architect session, 2026-10-06
**Inputs:**
- `docs/datasets/MIND_profile.md` and `docs/datasets/ZhihuRec_profile.md` (all numbers below come
  from them);
- data in `/home/dasgua3/HUG/MIND/` (MINDlarge train+dev, extracted) and
  `/home/dasgua3/HUG/ZhihuRec/raw/`, both git-ignored;
- the profiling scripts in `/tmp/claude-163583/-home-dasgua3-HUG/{mind_profile,zhihu_profile}/`.
  They're temporary, so copy anything reused into `Framework/datasets/`.

**Goal:** HUG and every baseline run on MIND and ZhihuRec under the same causal protocol as
KuaiRand-1K, through a dataset adapter, and the heavy-run harness gets a dataset dimension. This
spec runs smoke tests only.

---

## Why these two

| | KuaiRand-1K | MIND (large, train+dev) | ZhihuRec |
|---|---|---|---|
| Domain | short video | news | Q&A answers |
| Rows (full) | 11.7M | 97.6M candidate rows | 100.0M impressions |
| Click rate | 0.38 | 0.041 | 0.27 |
| Main cold-start axis | items (58% of val rows) | **items**: 53% of test rows involve news unseen in train (3–4k new articles a day) | **users**: 67% of test rows come from users unseen in train |
| Graph side info | author, tags | category, subcategory, 44k Wikidata entities | author, question, topics |

Together they test the thesis on three domains and on both cold-start axes.

---

## A. Dataset adapter contract (`Framework/datasets/`)

Create `Framework/datasets/{__init__,base,kuairand,mind,zhihurec}.py`. `base.py` defines the
adapter interface. Everything downstream (`temporal.py`, `features.py`, `hkg_constructor.py`,
`hug.py`, `Baselines/preprocess.py`) consumes only this interface; no dataset column names leak
past the adapters. Spec 04 W0.4 and test 19 are extended to every module.

An adapter returns a `DatasetBundle` with these fields:

| Field | Contents |
|---|---|
| `interactions` | DataFrame, one row per impression candidate: `user, item, time, label, label_time, ctx_*`. `time` is the impression time. `label_time` is **the earliest time the label is known** (see B). |
| `nodes` | dict of node type → static feature frame (user, item, plus metadata types) |
| `static_edges` | list of `(src_type, rel, dst_type, src_idx, dst_idx)` for metadata links with no timestamp (item→author, item→category, …) |
| `pre_window_history` | optional dict user → ordered item list of clicks *before the logging window* (MIND only) |
| `metadata_evidence_types` | node types used by Spec 05's `metadata_evidence` (KuaiRand: `[author]`; MIND: `[subcategory, entity]`; Zhihu: `[author, question]`) |
| `split_rule` | `"quantile_70_10_20"` (KuaiRand, unchanged) or `"day_snap"` (MIND, Zhihu; see C) |
| `timezone` | used by `hour_of_day` and day snapping (KuaiRand/Zhihu: Asia/Shanghai; MIND: the timestamps' local clock as given) |
| `user_sample` | `(fraction, seed)`, applied before anything else (see D) |
| `fingerprint()` | the data fingerprint for the queue hash |

**KuaiRand behaviour must not change.** Refactor the current loader into `datasets/kuairand.py`.
Every KuaiRand output must be bitwise identical: split, histories, features, graph and
predictions (test A1). This keeps `CODE_COMPAT_VERSION` unchanged, and the KuaiRand heavy run
unaffected.

## B. Label timing (new, required by ZhihuRec)

Zhihu logs clicks after their impressions: median delay 9 s, p95 247 s, p99 841 s, max about 6.7 h.
The profile found that **18.4% of rows have an earlier same-user impression whose click lands at
or after t.** So "use rows with `time < t`" isn't enough. **Any input derived from a label** has
to use rows with `label_time < t`. That covers click edges in graph snapshots, clicked-item
histories, click-based as-of statistics (click counts, CTR), and the evidence counts in Spec 05.

`label_time` definitions:

| Dataset | Positive (clicked) | Negative (not clicked) |
|---|---|---|
| KuaiRand | `time` (no click timestamp; assumption kept and documented) | `time` |
| MIND | `time` (no click timestamp; assumption documented) | `time` |
| ZhihuRec | `max(click_ts, impression_ts)`. 0.22% of clicks are logged before their impression; clamp those | `impression_ts + W`, with `W` = 900 s, just above the p99 click delay. A non-click isn't "known" until then |

- **Exposure-only** inputs (impression counts, `show_cnt`, an "impressed" edge) use `time`;
  **label-derived** inputs use `label_time`. Rename or comment the `features.py` functions so the
  difference is obvious.
- For KuaiRand and MIND, `label_time == time`, so nothing changes (test A1).
- Graph snapshots: a feedback edge's `edge_time` becomes its `label_time`, and impression/exposure
  edges keep `time`.
- **Rows that share a timestamp:** 53.6% of Zhihu rows share an exact (user, timestamp) with
  another row. They can't see each other (strictly before t). This is already the tie rule; the
  tests must cover it on Zhihu-shaped fixtures.

## C. Splits

**MIND and Zhihu use `day_snap`:** take the 70% and 90% row-quantile timestamps, then move each to
the **nearest local midnight**. Assign rows by impression `time`, as before.

| Dataset | Train | Val | Test | Notes |
|---|---|---|---|---|
| MIND | 11-09 → 11-13 | 11-14 | 11-15 (= the official dev day) | the profile's 70/90 quantiles fall at 11-14 04:48 and 11-14 12:59, which snap to these midnights. Day-aligned splits avoid the profile's one-morning validation window |
| Zhihu | to be confirmed | to be confirmed | to be confirmed | daily volume grows sharply (0.45M on day 1 → 27.6M on 05-12), so snapping may give an odd val/test share |

**Guard:** if snapping leaves train < 50% or val < 5% of rows, fall back to the exact 70/10/20
quantile cut, and log which rule was used. Report the final cutoffs and shares for both datasets
in the W-results file. Decide nothing silently.

KuaiRand keeps its verified exact 70/10/20 cut, so its heavy run stays valid. The paper says
"global chronological 70/10/20 split; for MIND and ZhihuRec the cut points are snapped to the
nearest day boundary".

## D. Subsampling (before splitting and before computing any statistics)

| Dataset | Rule | Expected size |
|---|---|---|
| MIND | random 20% of users, seed 0, keeping all their impressions | about 19.5M candidate rows; verify |
| Zhihu | random 12.5% of users, seed 0 (the profile's recommendation; **not** the README's "first N lines" — the file is sorted by user, so a prefix is a biased user range) | 12,504,877 rows, 99,761 users, click rate 0.2699 |

- All popularity statistics, vocabularies and graphs are computed **within the sample**, the
  same for every model.
- Record the sample's user list hash in the fingerprint.
- Make the fractions flags (`--user-frac`), so the heavy run can scale them if the run-time
  estimate allows.

## E. Graph mapping (per adapter → HUG relation set)

Every feedback relation gets reverses, as in Spec 01, and all edges are time-masked per snapshot.

**MIND**
- user –clicked→ news. `edge_time` = `label_time`. Pre-window history clicks get
  `edge_time` = window start − 1, so they're visible from the first snapshot (they're genuinely
  in the past).
- user –skipped→ news (impressed but not clicked). This mirrors KuaiRand's non-click
  information. **Include it behind a flag (`--skip-edges`, default on), and log the edge count:
  it's about 24× the click edges.** If it breaks the memory or time budget, default it off and
  report.
- Static edges:
  - news –in_category→ category;
  - news –in_subcategory→ subcategory;
  - subcategory –part_of→ category;
  - news –mentions→ entity (title + abstract entities, `Confidence` ≥ 0.5; log how many are
    dropped).
- Transitions: consecutive clicked items of the same user within a 30-minute gap, by
  `label_time` (generic transition builder, the same rule as KuaiRand's `next_in_session`).
- Entities: optional static input features from the provided TransE `entity_embedding.vec`
  (100-d, frozen projection; `--entity-init transe|id`, default `id`). Missing embeddings fall
  back to ID.

**ZhihuRec**
- user –clicked→ answer (`edge_time` = `label_time`); user –skipped→ answer (flag, as with MIND).
- Static edges: answer –written_by→ author (non-anonymous only; anonymous answers simply have
  no author edge); answer –answers→ question; answer –has_topic→ topic;
  question –has_topic→ topic.
- Transitions: generic builder, as above.
- **Not used:** `topics_followed` and `questions_followed` lists, and every counter column (user,
  answer, question and author counters). They're snapshots without timestamps, taken at
  collection time (the profile's recommendation), so including them would leak post-window
  information. Use them only in an *optional* "with snapshot features" sensitivity run, never
  in the main protocol.
- `inter_query.csv` (search queries): not used.

**Node features**

| Dataset | Node | Features |
|---|---|---|
| MIND | news | category, subcategory (categorical); age of the news at t isn't available, since news has no creation time |
| MIND | users | ID only (MIND has no user features) |
| Zhihu | users | the static categoricals and flags (gender, login_freq, register_type, register_platform, platform, device_model, device_brand, province, city, the `from_*` flags), plus "account age at t" from `register_ts` (causal; negative values clamped to 0 and flagged) |
| Zhihu | answers | anonymous, high_value, editor_rec, has_pic, has_video, and "answer age at t" from `create_ts` (causal). Treat `create_ts` = 0 as missing, and flag the 0.10% created after their first impression |
| Zhihu | authors | excellent_author, excellent_answerer (flags; assumed static, documented) |

- Text (MIND titles and abstracts, Zhihu token lists) isn't used by any model, HUG or
  baselines. Note this as a limitation.
- **Context features:** `hour_of_day` and `day_of_week` (from `time` in the adapter's timezone).
  MIND has no `tab` equivalent; Zhihu has none either.

## F. Histories

- The shared `click_history` function (Spec 04 W3.2) gains two arguments:
  - `label_time`: history items are clicks with `label_time < t`;
  - `pre_window`: optional per-user list, prepended in order.
- Pre-window items get the special token flags `gap_bucket = PRE_WINDOW` and
  `same_session = 0`.
- **MIND:** the provided `history` field is the same for every impression of a user (a frozen
  pre-window snapshot, verified at 0.0% variation). It's the `pre_window` list. In-window
  clicks are appended from earlier impressions by `label_time`.
- **Zhihu:** in-window clicks only. The median is 14 prior clicks and 5.5% of rows have none.
- `--max-seq-len` stays at 50 for all datasets, and the most recent items are kept.
- Baselines get the same histories through the same function (parity test A5).

## G. Baselines on the new datasets

- `Baselines/preprocess.py` takes `--dataset {kuairand,mind,zhihurec}` and builds the FuxiCTR
  CSVs from the adapter: the same rows, histories, as-of statistics (label-timed), and
  categorical and context features.
- Write dataset configs for `kuairand`, `mind` and `zhihurec`, and their `_seq` variants.
- FiGNN and DCNv2 don't use sequences; TransAct gets the raw sequence; WuKong the mean-pooled one.
- MIND entity IDs are graph-only, and aren't fed to the baselines as features (nor to HUG's
  head). Category and subcategory are fed to everyone.

## H. Heavy-run integration

- `heavy_run.yaml` gains a `datasets:` list, and every job gets a `--dataset` argument. Hashes
  include the dataset fingerprint.
- **Default plan per new dataset** (cheaper than KuaiRand, pending the W-results run-time
  estimate):
  - tuning: 8 trials per model (HUG N4, N1 and every baseline, equal budgets);
  - HUG arms: N0, N1, N2 and N4, plus `sparse`-adaptive, `evgate` and `misa` once Spec 05 lands;
  - 3 seeds for N1, N4, `sparse`-adaptive and the best baseline, 1 seed for the rest.

  **Get the user's sign-off before launching;** this spec only adds the entries and dry-runs
  them.
- Spec 05's fusion jobs get a dataset dimension the same way.

## Tests

- **A1.** KuaiRand refactor regression: split, histories, features, snapshot edges, and N4
  predictions after 2 synthetic epochs are bitwise identical to before the refactor.
- **A2.** Adapter contract: each adapter, on a small synthetic fixture in its native file
  format, returns a `DatasetBundle` that passes schema checks (dtypes, no nulls in keys,
  `label_time ≥ time`, static edges reference valid nodes). The MIND fixture can reuse the
  profiling agent's `make_synth.py`; write a Zhihu one.
- **A3. Label timing.** On a Zhihu-shaped fixture, an earlier impression whose click has
  `click_ts ≥ t` contributes to **none** of the row's: click edges in its snapshot, history,
  click-based as-of statistics or evidence counts. A non-click is invisible until
  `impression_ts + W`. Exposure-based counts still see the earlier impression.
- **A4. Prediction invariance with delayed labels.** Extend
  `test_predictions_invariant_to_future`:
  - rewriting the **label** of an earlier impression whose `label_time ≥ T` leaves every
    prediction at time < T unchanged;
  - rewriting all rows with `time ≥ T` does too;
  - the rewritten rows themselves do change.
- **A5.** History and feature parity, HUG vs. baselines, on 5,000 rows of each new dataset
  (as in Spec 04 tests 1/11).
- **A6.** `day_snap`: cutoffs fall on local midnights; the guard falls back correctly on a
  fixture built to fail it; the MIND fixture splits as 11-09–13 / 11-14 / 11-15.
- **A7.** The subsample is deterministic given the seed. Users are either fully in or fully out.
  Statistics are computed after sampling.
- **A8.** Pre-window history: tokens come first, carry the `PRE_WINDOW` gap bucket, and the
  corresponding edges exist in every snapshot from snapshot 0. In-window clicks are appended
  only once `label_time < t`.
- **A9.** No snapshot features: Zhihu counters and follow lists are absent from every node
  feature table and from the baseline CSVs unless the sensitivity flag is set.
- **A10.** Dataset isolation: grep-test (Spec 04 test 19) over `hug.py`, `fusion.py`,
  `temporal.py`, `features.py` and `hkg_constructor.py`.
- **A11.** The full synthetic suite still runs in under 5 min. Real-data tests for the new
  datasets are separate and marked.

## Smoke validation (the only runs in this spec)

For each new dataset:

1. Adapter stats vs. profile: row, user and item counts, and click rate after subsampling.
   Report them next to the profile's numbers, and explain any difference.
2. Final split cutoffs and shares (`day_snap` result or fallback); cold-item and cold-user
   shares on val/test, in the sampled data.
3. Graph sizes per relation, including the skip-edge count and memory, and per-snapshot encode
   time.
4. One epoch of N0 and N4, and 1 epoch of TransAct, all validation-only. Report time per epoch,
   peak memory and val AUC as a sanity check, and confirm N4 trains.
5. A 50-step dry run of the queue over the new dataset entries, then aggregation.
6. A run-time estimate for the per-dataset plan in H. If the total heavy run (all datasets)
   exceeds what's needed to have results by **2026-10-21**, propose cuts: user fractions, trials,
   seeds.

Write `docs/specs/06-datasets-results.md`, and update `CHECKPOINT.md`.

## Not in this spec

- Running the heavy run on the new datasets.
- Text features of any kind.
- Taobao: dropped, because the user preferred not to register with Alibaba.
- Changing the KuaiRand split or pipeline (A1 forbids it).

## Done when

- [ ] `Framework/datasets/` adapters (KuaiRand refactor bitwise-safe; MIND; ZhihuRec);
      `label_time` throughout the label-derived inputs; `day_snap` with guard; user subsampling
- [ ] Graph mappings, histories (with pre-window support) and baseline preprocessing for both
      datasets
- [ ] Tests A1–A11 pass; synthetic suite < 5 min
- [ ] Smoke results, cutoffs, sizes and run-time estimate reported; heavy-run entries added and
      dry-run green
