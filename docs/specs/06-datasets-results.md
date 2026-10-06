# Spec 06 results: MIND and ZhihuRec through the dataset adapter

**Status:** implementation and tests done; CPU-side smoke (items 1–3) done on the real data.
GPU smoke (item 4), the queue dry run (item 5) and the run-time estimate (item 6) are
**pending GPU time** (see "Not yet run").
**Branch:** `spec06-datasets` (worktree `~/HUG-spec06`), rebased on
`docs/technical-report-and-revision-plan` at `5565d99`.
**Date:** 2026-10-06.

---

## 1. What was built

| Spec item | Where | Notes |
|---|---|---|
| A. Adapter contract | `Framework/datasets/base.py` (`DatasetBundle`, `GraphSpec`, `GraphBuilder`), `datasets/__init__.py` (`load_dataset`, bundle cache) | Node layout `user, video (= item), <metadata types>`. MIND/ZhihuRec bundles are pickled under `<cache-dir>/datasets/`, keyed by every argument that changes them. |
| KuaiRand refactor | `datasets/kuairand.py` (+ `kuairand_columns.py`, `kuairand_hkg.py`) | Wraps the original loader/HKG; **bitwise identical** (test A1). |
| MIND adapter | `datasets/mind.py` | MINDlarge train + dev, 20% users (seed 0), pre-window history, entities (Confidence ≥ 0.5), optional TransE (`--entity-init transe`, train-release `.vec`, missing rows → ID only). |
| ZhihuRec adapter | `datasets/zhihurec.py` | 12.5% users (seed 0), `label_time`, causal account/answer ages, snapshot counters only with `--snapshot-features`. |
| B. Label timing | `features.click_history(label_time=…)`, `features.asof_item_label_stats`, `item_label_stats_before`, click/skip edge times | Exposure counts use `time`; histories, click counts, CTR, click/skip edges use `label_time`. KuaiRand/MIND: `label_time == time`. |
| C. Splits | `datasets/base.split_cutoffs` | `day_snap` with the guard (train ≥ 50%, val ≥ 5%, non-empty test) and a logged fallback. |
| D. Subsampling | `datasets/base.sample_users` | `default_rng(seed).choice(np.unique(users), round(frac·n))`, the profiler's exact recipe; `--user-frac`. |
| E. Graph mapping | adapters + `GraphBuilder` | Bidirectional relations get reverses (ids `r`, `r+B`), transitions are one-directional and last, like KuaiRand's layout. |
| F. Histories | `features.click_history(pre_window=…)`, `hug_train` token table | Pre-window tokens come first, gap bucket `PRE_WINDOW_GAP = 31`, never same-session. |
| G. Baselines | `Baselines/preprocess.py --dataset`, `train.py/tune.py --dataset`, configs `mind{,_seq,_noseq}`, `zhihurec{…}`, `<Model>_{mind,zhihurec}` | CSVs keep KuaiRand's role names (`user_id`, `video_id` = item, `is_click`, `hist_video_ids`) with integer node ids. |
| H. Heavy run | `scripts/run_queue.py`, `experiments/heavy_run.yaml`, `Framework/fingerprint.py --dataset`, `scripts/aggregate.py --dataset` | KuaiRand jobs get no `--dataset` flag, so **their config hashes are unchanged** (test). 59 jobs per new dataset; not launched. |

### Generic-code changes that touch KuaiRand

- `InputEncoder` takes the metadata node types from the `GraphSpec`; for KuaiRand they are
  registered as `author`, `category` in the same order, so parameter names and init are unchanged.
- `metadata_evidence` takes `(item, neighbour)` link sets (several neighbours per item, the
  max over them); KuaiRand's video→author links give the same counts.
- `hkg_constructor._snapshot_graph` copies every per-edge attribute instead of a fixed key list
  (same output on KuaiRand's stores).
- `features.hour_of_day` gained a timezone offset (default +8, KuaiRand's).

## 2. Tests

`pytest tests.py test_fusion.py test_harness.py test_guard.py test_datasets.py --device cpu`:
**179 passed, 34 skipped, in 18 s** (A11: under 5 min). The skips are the KuaiRand real-data
tests (`--data-dir`) and the two `HUG_REAL_DATA=1` tests.

| Test | Result |
|---|---|
| A1 KuaiRand bitwise | data-pipeline fingerprint (split, histories as the batcher sees them, context, evidence, buckets, vocabularies, node data, snapshot relation index, per-snapshot video features, baseline CSV frame) = pre-refactor hash `f54615d0…`; N4 2-epoch model fingerprint = `60d53e57…` (unchanged since spec 05). |
| A2 contract | MIND, ZhihuRec and KuaiRand fixtures: dtypes, no null keys, sorted time, `label_time ≥ time`, every edge within its declared node types, links in range. |
| A3 label timing | ZhihuRec fixture: `label_time` = max(click, impression) / impression + 900 s; history = brute-force known clicks for every row (the fixture has naive-visible clicks that are not yet known: `leaked > 0`); item show/click/CTR stats brute-forced; click/skip edges in a snapshot = rows with `label_time < boundary`; same-timestamp rows never see each other. |
| A4 invariance | ZhihuRec fixture, HUG N4 with fixed weights: flipping the labels of 6 impressions with t < T ≤ label_time leaves every prediction before T unchanged; rewriting every row at or after T (labels, click times, answers) too; the rewritten rows' predictions change. |
| A5 parity | HUG vs baseline frame on both fixtures, every row: history items, context, numeric features, labels, split, buckets. **Real data** (`HUG_REAL_DATA=1`): 5,000 random rows of MIND and of ZhihuRec, histories and every numeric feature identical: passed (7.7 min). |
| A6 day_snap | MIND fixture: cutoffs at UTC midnights, splits exactly 11-09…13 / 11-14 / 11-15; Beijing-midnight snapping; Zhihu-shaped fallback. |
| A7 subsample | deterministic, users fully in, stats computed within the sample (both adapters). |
| A8 pre-window | tokens first, `PRE_WINDOW_GAP`, not same-session; in-window clicks appended once known; pre-window click edges present in snapshot 0. |
| A9 no snapshot features | no counter or follow-list column in ZhihuRec's per-row features, item/user categoricals or baseline CSV columns unless `--snapshot-features`. |
| A10 isolation | no dataset column name in `hug.py`, `fusion.py`, `temporal.py`, `features.py`, `hkg_constructor.py`; no dataset-name string literal there. KuaiRand registries moved to `datasets/kuairand_columns.py`, the KuaiRand HKG builder to `datasets/kuairand_hkg.py` (both re-exported for old imports). |
| Queue (H) | KuaiRand hashes unchanged with the new plan; MIND jobs get `--dataset mind` and their own data fingerprint; cross-dataset `{best:}` refused; plan has N0/N1/N2/N4/sparse/evgate/misa groups per dataset. |

## 3. Smoke items 1–3 (real data, CPU)

### 3.1 Adapter output vs. the profiles

| | MIND (20% users) | profile | ZhihuRec (12.5% users) | profile |
|---|---|---|---|---|
| rows | 19,502,700 | ≈ 19.5M (20% of 97.6M) | **12,504,877** | 12,504,877 |
| users | 150,087 | 20% of 750,434 | **99,761** | 99,761 |
| items (nodes) | 72,497 (25,776 shown as candidates; the rest only in pre-window histories) | — | 288,160 answers | 288,160 |
| CTR | 0.0406 | 0.0406 | 0.26989 | 0.26989 |
| impressions | 522,304 | — | — | — |
| other nodes | 18 categories, 274 subcategories, 34,902 entities | — | 145,798 authors, 88,541 questions, 25,258 topics | 145,798 / 88,541 / — |
| adapter build / peak RSS | 136 s / 10.0 GB | — | 86 s / 7.1 GB | — |
| cached bundle load | 1.7 s (2.4 GB pickle) | — | 1.0 s (1.7 GB pickle) | — |

ZhihuRec matches the profile's subsample exactly. MIND's entity filter drops nothing: every
entity in MINDlarge has Confidence ≥ 0.901, so `Confidence ≥ 0.5` is a no-op (logged count 0).
7,347 ZhihuRec clicks (0.22%) are logged before their impression and clamped; 60 rows score an
answer created after the impression (age flagged unknown).

### 3.2 Splits (final)

| | rule used | t_val | t_test | train / val / test |
|---|---|---|---|---|
| MIND | **day_snap** | 2019-11-14 00:00 | 2019-11-15 00:00 | 67.7% / 17.8% / 14.5% (11-09…13 / 11-14 / 11-15) |
| ZhihuRec | **quantile_fallback** | 2018-05-11 21:53:44 CST | 2018-05-12 11:24:26 CST | 70.0% / 10.0% / 20.0% |

ZhihuRec: the 70% and 90% quantiles (05-11 21:53 and 05-12 11:24 CST) both snap to 05-12
00:00 CST, so val would be empty (train 72.5%, val 0%, test 27.5%). The guard falls back to the
exact cut, which is the profile's split. Logged, and recorded in the data fingerprint
(`split_rule_used`). The paper's split sentence needs "for MIND the cut points are snapped to
day boundaries; ZhihuRec keeps the exact cut because its last two days hold 45% of the rows".

### 3.3 Cold shares in val/test (sampled data)

| | MIND val | MIND test | ZhihuRec val | ZhihuRec test |
|---|---|---|---|---|
| rows | 3,463,936 | 2,832,473 | 1,250,489 | 2,500,999 |
| CTR | 0.0387 | 0.0409 | 0.2769 | 0.2589 |
| **item unseen in train rows** | 15.6% | **77.1%** | 6.3% | 15.4% |
| **user unseen in train rows** | 16.0% | 17.5% | 37.6% | **67.1%** |
| item with 0 prior impressions (causal) | 0.10% | 0.08% | 1.47% | 1.20% |
| item with 1–4 prior impressions | 0.36% | 0.29% | 3.48% | 3.28% |
| user with 0 prior impressions | 12.6% | 9.6% | 0.79% | 0.45% |

**MIND item cold start is stronger under the day-aligned split than in the profile** (77% of
test rows vs. 53%), because test is now only 11-15 (the dev day). Val sees 15.6%, so val-based
model selection still under-represents the item-cold regime (profile caveat 4). Report test
results split by seen/unseen item (the `video_train_count` bucket already does this).

### 3.4 Graph sizes

| relation (forward; reverse has the same count) | MIND | ZhihuRec |
|---|---|---|
| clicked (label_time) | 3,536,819 (incl. 2,744,777 pre-window) | 3,374,930 |
| skipped (label_time) | **18,710,658** (5.3× clicks) | 9,129,947 (2.7×) |
| metadata | in_category 72,497 · in_subcategory 72,497 · part_of 295 · mentions 173,552 | written_by 249,631 · answers 278,755 · has_topic 846,773 · question_has_topic 236,481 |
| next_click (one-directional) | 304,855 | 2,871,769 |
| **master index** | **45.4M edges, 1.45 GB** (int64) | **31.1M edges, 1.0 GB** |

KuaiRand-1K's master index is 28.4M edges. Skip edges dominate MIND's graph (82%); they are on
by default (`--skip-edges`), and the GPU smoke will show whether they fit the time budget.

### 3.5 Baseline CSVs

`preprocess.py --dataset mind`: 4 min, 18.5 GB peak RSS, 4.6 GB of CSVs (train 12,545,976 /
holdout 660,315 / valid 3,463,936 / test 2,832,473 rows). `--dataset zhihurec`: 3.7 min,
19.9 GB, 2.8 GB (8,315,720 / 437,669 / 1,250,489 / 2,500,999). Both from the cached bundles.

## 4. Decisions made while implementing

1. **ZhihuRec split falls back to the exact cut** (3.2); MIND uses day_snap.
2. **Item vocabulary includes pre-window clicks** (MIND): an item's ID embedding is learned if it
   has ≥ `min_id_count` training-window rows *or* pre-window clicks (both precede `t_val`).
   KuaiRand has no pre-window, so nothing changes there.
3. **MIND users** have no features: `user_x` is a constant zero column, no categoricals.
4. **ZhihuRec author flags** (excellent_author/answerer) are item categoricals of the answer
   (authors themselves are ID-only graph nodes). Baselines also get `author_id` and
   `question_id` as categoricals (KuaiRand's baselines get `author_id` the same way). Topics are
   graph-only.
5. **ZhihuRec per-row context**: hour, day-of-week; numeric: the 3 label-timed item stats plus
   `log1p(account age h)`, unknown flag, `log1p(answer age h)`, unknown flag (create_ts = 0 or
   after the impression, or a registration after the impression).
6. **Generic item statistics** are named `item_shows_log`, `item_clicks_log`, `item_ctr_known`
   (not KuaiRand's `show_cnt_log`…), so no KuaiRand column name appears in `features.py`.
7. **Heavy-run plan per new dataset**: tuning 8 trials each (HUG N4, N1, every baseline; fusion
   sparse 8, evgate 6, misa 8), arms N0/N2 (1 seed), N1 at its own tuned settings and N4
   (3 seeds), sparse (3 seeds), evgate/misa (1 seed), baselines seed 42; baseline seeds 43/44 are
   `optional` and should be run for the best baseline only (`--only` after tuning). Groups are
   `<arm>@<dataset>`; `aggregate.py --dataset` strips the suffix.
8. **Bundle cache** (`<cache-dir>/datasets/<name>_<key>.pkl`): the queue's fingerprint command
   builds it once, before any job.

## 5. Not yet run (needs GPU time)

- **Item 4**: 1 epoch of N0, N4 and TransAct per dataset (validation only): time/epoch, peak
  memory, val AUC, per-snapshot encode time. Script ready: `runs/dev/spec06/chain.sh <gpu> mind
  zhihurec` (from `~/HUG-spec06`).
- **Item 5**: 50-step queue dry run over the MIND/ZhihuRec entries, then aggregation.
- **Item 6**: run-time estimate for the per-dataset plan and proposed cuts against the
  2026-10-21 results deadline.
