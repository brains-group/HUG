# MIND (Microsoft News) dataset profile

Profiled 2026-10-06. Source: Hugging Face `yjw1029/MIND` (gated mirror of the official release, Microsoft Research License).
Files: `MINDlarge_{train,dev}.zip` and `MINDsmall_{train,dev}.zip`. The large test set was not downloaded because its labels are not public.
Raw zips are in `MIND/raw/` (685 MB) and the extracted files in `MIND/extracted/<name>/` (2.0 GB). Both are git-ignored via `MIND/`.

Scripts, all run with `python -I` from the `hkg-env` conda env, are in `/tmp/claude-163583/-home-dasgua3-HUG/mind_profile/`:
- `run_all.sh`: downloads, verifies and extracts the zips, then runs the profiler.
- `profile_mind.py`: computes all statistics and writes `MINDlarge.json` / `MINDsmall.json`. The `*.time` files hold the `/usr/bin/time -v` output.
- `spotcheck.py`: independent re-check of history constancy and item first-show times.
- `make_synth.py`: builds a synthetic test dataset.

Unless stated otherwise, "train" and "dev" mean the original MIND files, and "combined" means train and dev concatenated.

## Summary

- **MINDlarge (train + dev): 2.61 M impressions and 97.6 M candidate rows.** There are 750 k users. 29.3 k distinct news items appear as candidates, out of 104 k in `news.tsv`. CTR on candidate rows is 4.06%, with a median of 24 candidates per impression.
- **Time range:** 2019-11-09 00:00:00 to 2019-11-15 23:59:43. Train covers 6 days and dev covers 1 day (2019-11-15). **Dev is strictly after train.**
- **Clean data:** no malformed rows, bad tokens or unknown news IDs, and no duplicate impression IDs.
- **Every impression has at least one click.** Zero-click impressions are absent, so they were filtered out upstream.
- **`history` never changes for a user.** Across all impressions, in both train and dev, 0.0% of users show a varying history. It is a frozen snapshot from before the logging window, so it never includes clicks made inside the window.
- **Graph side info:** 18 categories, 285 subcategories and 44 k Wikidata entities. 88% of news have at least one entity. `entity_embedding.vec` (TransE, 100-d) ships with every split.
- **Global causal 70/10/20 split (MINDlarge):**

  | split | time range | impressions | candidate rows | CTR |
  |---|---|---|---|---|
  | train | up to 2019-11-14 04:48:16 | 1.83 M | 67.3 M | 4.09% |
  | val | 2019-11-14 04:48:17 to 12:59:47 | 0.26 M | 10.7 M | 3.86% |
  | test | 2019-11-14 12:59:48 to 2019-11-15 23:59:43 | 0.52 M | 19.6 M | 4.04% |

  The test split contains all of the original dev set plus the last 11 hours of the original train set.
- **Cold items depend on what counts as prior information.**
  - Against the whole log before each row, almost nothing is cold. Only 0.02% of test rows have zero earlier impressions and 0.59% have zero earlier clicks.
  - Against the train split only, **53% of test candidate rows (54% of distinct test news) were never seen**. For val it is 0.8% of rows (24% of distinct news).
  - The reason is item churn: about 2–4.5 k new items start being shown each day.
  - So a model whose item IDs are frozen at the train cutoff will face mostly unseen items in test. A model that updates online will not.
- **MINDsmall** has the same structure at 230 k impressions and 8.6 M candidate rows (exact figures in the tables below). Its train and dev files sample users independently, so under the global split 85% of val/test users are unseen in train.
- **No blockers.** The full MINDlarge profile took 9 min 20 s with a peak RSS of 59 GiB, using naive pandas with object strings.

## 1. Sizes

"Candidate rows" means one row per (impression, candidate news) pair after exploding the impressions field.

| | MINDlarge train | MINDlarge dev | **MINDlarge combined** | MINDsmall train | MINDsmall dev | **MINDsmall combined** |
|---|---|---|---|---|---|---|
| impressions (rows) | 2,232,748 | 376,471 | **2,609,219** | 156,965 | 73,152 | **230,117** |
| candidate rows | 83,507,374 | 14,085,557 | **97,592,931** | 5,843,444 | 2,740,998 | **8,584,442** |
| users | 711,222 | 255,990 | **750,434** | 50,000 | 50,000 | **94,057** |
| distinct candidate news | 27,046 | 6,997 | **29,309** | 20,288 | 5,369 | **22,771** |
| news in news.tsv (deduplicated) | | | **104,151** | | | **65,238** |
| CTR on candidate rows | 4.052% | 4.081% | **4.056%** | 4.045% | 4.064% | **4.051%** |
| candidates/impression: mean | 37.40 | 37.41 | **37.40** | 37.23 | 37.47 | **37.30** |
| candidates/impression: median | 25 | 23 | **24** | 24 | 23 | **24** |
| candidates/impression: p95 | 118 | 119 | **119** | 118 | 119 | **118** |
| candidates/impression: max | 300 | 299 | **300** | 299 | 295 | **299** |

When a news ID appeared in both the train and dev `news.tsv`, the category, subcategory and title always matched (0 conflicts).

## 2. Time

| | min timestamp | max timestamp |
|---|---|---|
| MINDlarge train | 2019-11-09 00:00:00 | 2019-11-14 23:59:59 |
| MINDlarge dev | 2019-11-15 00:00:00 | 2019-11-15 23:59:43 |
| MINDsmall train | 2019-11-09 00:00:19 | 2019-11-14 23:59:13 |
| MINDsmall dev | 2019-11-15 00:00:01 | 2019-11-15 23:58:03 |

Dev is strictly after train in both versions: no dev row is at or before the train maximum.

Impressions per day:

| date (2019) | weekday | MINDlarge | MINDsmall |
|---|---|---|---|
| 11-09 | Sat | 192,552 | 13,570 |
| 11-10 | Sun | 212,343 | 15,048 |
| 11-11 | Mon | 464,467 | 32,799 |
| 11-12 | Tue | 478,375 | 33,654 |
| 11-13 | Wed | 453,494 | 31,624 |
| 11-14 | Thu | 431,517 | 30,270 |
| 11-15 (dev) | Fri | 376,471 | 73,152 |

Timestamps have one-second resolution. All candidates of an impression share its timestamp.

## 3. Graph side info (news.tsv, train and dev combined)

Entity counts per news item are distinct WikidataIds across title and abstract entities.

| | MINDlarge | MINDsmall |
|---|---|---|
| categories | 18 | 18 |
| subcategories | 285 | 270 |
| distinct (category, subcategory) pairs | 308 | 291 |
| distinct WikidataIds (title + abstract) | 44,077 | 32,472 |
| entities per news: mean | 2.46 | 2.38 |
| entities per news: median / p95 / max | 2 / 6 / 30 | 2 / 6 / 30 |
| share of news with at least 1 entity | 88.3% | 87.5% |
| share of news with empty abstract | 5.4% | 5.2% |
| `entity_embedding.vec` present | yes, in train and dev | yes, in train and dev |
| `relation_embedding.vec` present | yes | yes |
| rows in the train-split `entity_embedding.vec` | 42,007 | 26,904 |
| share of WikidataIds covered by the train `.vec` | 95.3% | 82.9% |

The dev `.vec` file is different from the train one. I did not compute coverage for the union of the two files.

## 4. Histories

| | MINDlarge | MINDsmall |
|---|---|---|
| history length: mean | 32.9 | 32.5 |
| history length: median / p95 / max | 19 / 110 / 801 | 19 / 108 / 558 |
| share of impressions with empty history | 2.2% | 2.4% |
| users with more than 1 impression (combined) | 533,415 | 47,827 |
| **share of those users whose history string varies (combined)** | **0.0%** | **0.0%** |
| ... within train only (users with more than 1 impression) | 0.0% (478,577) | 0.0% (33,617) |
| ... within dev only | 0.0% (76,607) | 0.0% (14,826) |
| distinct news appearing in histories | 79,937 | 44,908 |
| share of history news that never appear as a candidate | 93.6% | 94.6% |

The history field is an exact constant per user. It is identical even between a user's train-week and dev-day impressions. `spotcheck.py` confirmed this independently on a 2% user sample (15,057 users, 0 varying). Most history items are older articles that are never shown as candidates.

## 5. Causal-split feasibility (global chronological 70/10/20)

Method:
- The split runs over combined train + dev impression rows, sorted by timestamp.
- The train cutoff is the first timestamp at which the cumulative share of impression rows reaches 70%; the val cutoff is where it reaches 80%.
- All rows sharing a timestamp go to the same side of a cutoff.
- Fractions are measured on impressions. On candidate rows, MINDlarge comes out at 68.9 / 11.0 / 20.1.

**MINDlarge**

| split | timestamp range | impressions | candidate rows | CTR | users | origin of rows | users unseen in train split |
|---|---|---|---|---|---|---|---|
| train | 2019-11-09 00:00:00 to 2019-11-14 04:48:16 | 1,826,459 | 67,250,734 | 4.094% | 658,077 | all original train | |
| val | 2019-11-14 04:48:17 to 2019-11-14 12:59:47 | 260,920 | 10,741,367 | 3.857% | 188,156 | all original train | 16.5% |
| test | 2019-11-14 12:59:48 to 2019-11-15 23:59:43 | 521,840 | 19,600,830 | 4.036% | 325,753 | 376,471 original dev + 145,369 original train | 17.5% |

**MINDsmall**

| split | timestamp range | impressions | candidate rows | CTR | users | origin of rows | users unseen in train split |
|---|---|---|---|---|---|---|---|
| train | 2019-11-09 00:00:19 to 2019-11-15 04:07:05 | 161,082 | 5,988,906 | 4.054% | 53,248 | 156,965 original train + 4,117 original dev | |
| val | 2019-11-15 04:07:09 to 08:22:54 | 23,012 | 845,284 | 4.137% | 19,561 | original dev | 85.5% |
| test | 2019-11-15 08:22:55 to 23:58:03 | 46,023 | 1,750,252 | 3.998% | 34,222 | original dev | 85.5% |

MINDsmall's cutoffs fall later than MINDlarge's because its dev file is relatively larger: 32% of its impressions, against 14% for MINDlarge.

## 6. Cold items (val/test candidate rows)

"Prior" means candidate rows of the same news item at a strictly earlier timestamp anywhere in the combined log, with any label. A row's own timestamp is excluded.

"Unseen in train split" means the news item never appears as a candidate or in any history within train-split rows.

| | MINDlarge val | MINDlarge test | MINDsmall val | MINDsmall test |
|---|---|---|---|---|
| candidate rows | 10,741,367 | 19,600,830 | 845,284 | 1,750,252 |
| 0 prior impressions | 0.015% | 0.020% | 0.096% | 0.077% |
| 1–4 prior impressions | 0.068% | 0.076% | 0.351% | 0.281% |
| 5 or more prior impressions | 99.916% | 99.904% | 99.553% | 99.642% |
| **0 prior clicks** | **0.55%** | **0.59%** | **2.44%** | **1.86%** |
| CTR of rows with 0 prior impressions | 4.27% | 4.73% | 5.55% | 5.17% |
| **rows whose news is unseen in train split** | **0.83%** | **53.2%** | **1.78%** | **12.4%** |
| distinct candidate news | 6,973 | 8,945 | 3,377 | 4,429 |
| share of distinct news unseen in train split | 23.7% | 54.4% | 23.7% | 43.7% |
| rows unseen in train split and in all histories | 0.83% | 53.2% | 1.78% | 12.4% |

Adding histories to the "seen" set barely changes anything. New candidate items are almost never in anyone's frozen history.

New items keep arriving. Items first shown per day in MINDlarge (from `spotcheck.py`):

| 11-09 | 11-10 | 11-11 | 11-12 | 11-13 | 11-14 | 11-15 |
|---|---|---|---|---|---|---|
| 6,988 | 3,111 | 4,544 | 4,205 | 4,443 | 3,755 | 2,263 |

The 11-09 figure is inflated because items already live when the log starts are counted there. 3,823 items are first shown inside the MINDlarge test window.

A new item collects at least 5 impressions within seconds of its first show. That is why "0 prior impressions" is tiny even though "unseen in train split" is large.

## 7. Duplicates and oddities

| | MINDlarge | MINDsmall |
|---|---|---|
| malformed behaviors rows (field count or timestamp) | 0 | 0 |
| malformed impression tokens (not `N<digits>-[01]`) | 0 | 0 |
| candidate or history news missing from news.tsv | 0 | 0 |
| duplicate impression IDs (within train or within dev) | 0 | 0 |
| candidate rows repeating an earlier (user, news) pair | 14,084,528 (14.4%) | 1,146,486 (13.4%) |
| distinct (user, news) pairs shown more than once | 10,760,556 | 890,387 |
| ... of which the label differs between repeats | 758,709 | 62,828 |
| same news twice within one impression | 0 | 0 |
| **impressions with zero clicks** | **0** | **0** |
| impressions with 2 or more clicks | 727,045 (27.9%) | 64,162 (27.9%) |
| clicks per impression: mean / median / p95 / max | 1.52 / 1 / 4 / 51 | 1.51 / 1 / 4 / 35 |
| impressions with a single candidate | 0 | 0 |
| candidate rows whose news is already in that row's history | 219,155 | 15,831 |

## 8. Loading cost

The machine has 503 GB RAM. The profiler uses a naive pandas pipeline that holds all IDs as Python strings before factorizing them.

| | MINDlarge | MINDsmall |
|---|---|---|
| total wall time (`/usr/bin/time`) | 9 min 20 s | 49 s |
| ... TSV parse of behaviors + news | 15 s | 2 s |
| ... explode impressions into candidate rows | 211 s | 20 s |
| ... prior-impression/click counting | 98 s | 8 s |
| peak RSS | 59.3 GiB | 5.7 GiB |

Most of the memory goes on the exploded candidate rows: 97.6 M rows for MINDlarge. A training loader should stream the TSV, map IDs to int32 on the fly and store the candidates as numpy arrays. Exploded int32/int8 columns plus an int64 timestamp need about 2 GB.

## Caveats for a strictly causal protocol

The protocol: each impression at time t may only use information from before t.

1. **`history` is a frozen pre-window snapshot.** It is legal at any t because it predates the whole log. But it never updates: even dev-day histories exclude the user's 11-09 to 11-14 clicks. Any history from inside the window has to be built causally from earlier impression labels, i.e. clicked candidates at timestamps before t. Treat the static history and the in-window click sequence as separate inputs.
2. **Labels within one impression are simultaneous.** All candidates share one timestamp and 28% of impressions have 2 or more clicks. Neither a candidate's own label nor its siblings' labels may be used as features. "Strictly before t" must exclude same-timestamp rows. This matters for item counters too, since popular items get many rows in the same second.
3. **The data is conditioned on at least one click per impression.** The 4.06% CTR is therefore conditional, not a natural CTR. Both the metrics and any calibration analysis should state this.
4. **The 53% unseen-item rate in MINDlarge test is the main design issue.**
   - Test spans about 35 hours after a train cutoff of 2019-11-14 04:48, and about 3.8 k items are born inside it.
   - Models that learn item ID embeddings only up to the train cutoff will be scoring mostly unseen items.
   - Options:
     - (a) Content/entity/category-based item representations, which `news.tsv` supports.
     - (b) Online updates of counters or graph edges from test-period events before t, as long as only labels revealed before t are used.
     - (c) Report results split by seen versus unseen items.
   - Val has only 0.8% unseen rows, so it under-represents this problem compared with test. Model selection on val may favour ID-heavy models.
5. **The val window is a single Thursday morning** (04:48 to 12:59). Its CTR is lower (3.86% against 4.04–4.09%) and it samples only one part of the day, while test covers 1.5 days including a full Friday. Consider an alternative val window, or check that conclusions survive another one.
6. **Test mixes the original train and dev files.** Published MIND numbers on dev are not comparable to this split. If comparability matters, an alternative is train = the original train file (with val cut from its last part) and test = the original dev file. That split is also causal.
7. **Re-exposure is common.**
   - 14% of candidate rows repeat an earlier (user, news) pair, and 7% of repeated pairs change label.
   - 219 k rows show a candidate already in the user's history.
   - Prior-exposure features for the same user and item are legal when computed only from earlier timestamps.
   - Deduplicating pairs would change the task. If it is done, it must be done causally (keep the first occurrence).
8. **News metadata has no timestamp.** `news.tsv` has no publish time, so first-impression time is the only proxy. Title, abstract and entities are treated as known at first show.
9. **`entity_embedding.vec` was trained outside the log.** It is pretrained TransE on WikiKG, not on the impression log, so it adds no label leakage. However, the train and dev releases ship different `.vec` files. Use a fixed union, or the train file only; avoid picking embeddings per split in a way that depends on test items.
10. **MINDsmall is a poor fit for user-level causal evaluation.** Its train and dev files sample 50 k users each, independently, and only about 5.9 k users overlap. Under the global split, 85.5% of val/test users never appear in train. MINDlarge (16–18% unseen users) is the right version for this protocol. MINDsmall is fine for smoke tests.
11. **Weekend volume is lower.** 11-09 and 11-10 have about 42–46% of weekday volume. Rolling-window features should normalise for this.
12. **Neither version has user features or a full user log.** Users and impressions are a sample of MSN traffic. The only user signals are the static history and in-window impressions.
