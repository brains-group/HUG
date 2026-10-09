# ZhihuRec-100M: dataset profile for a causal CTR protocol

Source: THUIR + Zhihu, public Seafile share https://cloud.tsinghua.edu.cn/d/d6c045c55aa14bb39ebc/ (research use only; cite Hao et al. 2021, arXiv:2106.06467).
Local copy: `ZhihuRec/raw/` (git-ignored). Downloaded 2026-10-06. Every file matched the API-listed size and passed `gzip -t`.
`info_token.csv.gz` (64-d word2vec token vectors, 172 MB) was **not** downloaded. Only the token-ID columns of the other files refer to it, and the profile does not use those columns.

Scripts (outside the repo, run with `python -I` from the hkg-env interpreter):
- `/tmp/claude-163583/-home-dasgua3-HUG/zhihu_profile/profile_zhihurec.py <raw_dir> <out_json> [subsample_frac=0.125] [seed=0]`
- Raw results: `/tmp/claude-163583/-home-dasgua3-HUG/zhihu_profile/full.json`, plus a log in `full.log` in the same directory.
- Note: in `full.json` the key `delay.p99` actually holds p99.9 (a key collision in the quantile helper, now fixed in the script). Corrected delay quantiles came from `delay_q.py` in the same directory.

All numbers below come from that run. Shares are fractions of rows unless the text says otherwise.

## Summary

- **Size**: 99,978,523 impressions, 798,086 users, 554,976 answers, 165,012 questions, 240,956 authors and 72,318 topics. There are 26,981,583 clicks, so CTR = **0.2699**. These match the README exactly.
- **Time**: Unix seconds. The window runs from 2018-05-02 16:01:38 UTC to 2018-05-13 10:59:27 UTC, which is 2018-05-03 00:01 to 2018-05-13 18:59 Beijing time (UTC+8). The day boundaries and the late-evening peak both point to Beijing time. Volume grows sharply over the window: 0.45M impressions on day 1 and 27.6M on 05-12.
- **Per-user histories are truncated**: each user has between 10 and 160 impressions (median 146, p95 160). It looks like the providers kept up to the last 160 impressions per user, which would also explain the volume growth (this is an inference). Early-window history is therefore systematically missing.
- **Click delay**: median 9 s, p95 247 s, p99 841 s, p99.9 2,496 s, max 24,095 s (about 6.7 h). 0.22% of clicks have click_ts < impression_ts (min -6,446 s), and 3.3% have delay 0.
- **Chronological 70/10/20 split** (ties kept together): val starts at 1526046994 (2018-05-11 21:56:34 CST) and test at 1526095709 (2018-05-12 11:28:29 CST). Rows: train 69,984,767 / val 9,997,867 / test 19,995,889. CTR: 0.2720 / 0.2777 / 0.2585. The val and test periods cover only about 1.3 days.
- **Cold items** (counted causally over the full data) are rare. In test, 0.22% of rows have an answer with 0 prior impressions, 0.65% have 1 to 4, and 99.1% have 5 or more. 1.3% have an answer with 0 known prior clicks. Among rows with an author, 0.09% have a cold author; 10.5% of rows are anonymous and have no author. Among rows with a question, 0.05% have a cold question. Cold **users** are not rare at the split level: 67% of test rows come from users who never appear in train.
- **Label-timing leak**: 18.4% of all rows have at least one earlier impression from the same user (impression_ts < t) whose click happens at or after t. Using earlier rows' labels naively therefore changes the click history of about 1 in 5 rows. In aggregate, 1.18% of naively visible past clicks are not yet observable at t.
- **Same-timestamp batches**: 53.6% of rows share their (user, impression_ts) with at least one other row. Under the rule "strictly before t", none of those rows sees the others.
- **Static side-feature counters** (#likes, #followers, #answers, …) carry no timestamp. They are probably snapshots taken at collection time, which is a leakage risk (see Caveats).
- **Downsampling**: there are no separate 1M or 20M files. The README says to take the top N lines of each file. The impression file is sorted by user_id, so top-N gives the first 7,974 users (1M) or the first 159,642 users (20M) with their full histories. Recommendation: a random **12.5% of users (seed 0)**, which gives 99,761 users, 12,504,877 rows and CTR 0.2699. Its split is 8,753,389 / 1,250,489 / 2,500,999.
- **Cost**: the full impression file loads in 32 s (pandas C engine, int32/int64) and takes about 4.8 GB RSS. The side tables load in 12 s. The full causal-count pass took about 21.5 min, and peak RSS for the whole script was 25 GB (`/usr/bin/time` reported 26.2 GB).
- **Blocking problems**: none. Everything requested could be computed.

## 0. Files and schema (from README, columns verified by inspection)

All files are header-less CSVs. Null values are empty strings, and list fields are space-separated IDs. IDs are dense, 0-based integers (`id == row index` holds for the answer, user, question, author and topic tables).

| File | gz size | Rows | Columns |
|---|---:|---:|---|
| inter_impression.csv | 672 MB | 99,978,523 | user_id, answer_id, impression_ts, click_ts (0 = no click) |
| inter_query.csv | 30 MB | 3,899,553 | user_id, query token IDs, query_ts (501,893 users; ts from 2018-05-02 16:02 to 05-13 15:59 UTC) |
| info_user.csv | 56 MB | 798,086 | 27 cols (see §3) |
| info_answer.csv | 376 MB | 554,976 | 18 cols (see §3) |
| info_question.csv | 6.5 MB | 165,012 | 8 cols (see §3) |
| info_author.csv | 0.9 MB | 240,956 | author_id, excellent_author, n_followers, excellent_answerer |
| info_topic.csv | 0.15 MB | 72,318 | topic_id only |
| info_token.csv | 172 MB | not downloaded | token_id, 64-d word2vec vector |

The raw file is sorted by user_id, and by impression_ts within each user. There are 8,715 exact duplicate rows and 1,927,699 repeated (user, answer) pairs, meaning the same answer was shown to the same user more than once.

## 1. Sizes

| Quantity | Value |
|---|---:|
| Impression rows | 99,978,523 |
| Users | 798,086 |
| Answers (impressed) | 554,976 |
| Questions (via answers) | 165,012 |
| Authors (via answers) | 240,956 |
| Topics (info_topic) | 72,318 |
| Clicks | 26,981,583 |
| CTR | 0.26987 |
| Impressions/user: mean / median / p95 / min / max | 125.27 / 146 / 160 / 10 / 160 |

## 2. Time

Timestamps are Unix seconds and appear to be Beijing time (UTC+8) when converted to local days.

| | Value |
|---|---|
| Min impression_ts | 1525276898 = 2018-05-02 16:01:38 UTC (05-03 00:01 CST) |
| Max impression_ts | 1526209167 = 2018-05-13 10:59:27 UTC (05-13 18:59 CST) |
| Max click_ts | 1526209178 (1 click lands after the last impression_ts) |

Impressions per CST day:

| Day (CST) | Rows | CTR |
|---|---:|---:|
| 05-03 | 453,356 | 0.297 |
| 05-04 | 1,967,318 | 0.267 |
| 05-05 | 5,233,298 | 0.249 |
| 05-06 | 8,844,947 | 0.256 |
| 05-07 | 8,258,566 | 0.274 |
| 05-08 | 9,229,069 | 0.289 |
| 05-09 | 9,492,125 | 0.285 |
| 05-10 | 11,784,194 | 0.274 |
| 05-11 | 17,098,271 | 0.269 |
| 05-12 | 27,606,876 | 0.265 |
| 05-13 (partial, to 18:59) | 10,503 | 0.128 |

By hour of day (CST), volume is lowest at 04:00 (0.52M) and highest at 23:00 (7.7M).

Click delay (click_ts minus impression_ts) over the 26,981,583 clicks:

| Stat | Value |
|---|---:|
| p1 / p10 / p25 / p50 | 0 / 2 / 4 / 9 s |
| p75 / p90 / p95 / p99 / p99.5 / p99.9 | 41 / 128 / 247 / 841 / 1,216 / 2,496 s |
| mean / max | 59.3 s / 24,095 s |
| delay > 1 h | 0.036% |
| delay > 1 day | 0 |
| delay = 0 | 891,224 (3.30%) |
| **delay < 0 (anomaly)** | **58,379 (0.216%)**, min -6,446 s |

## 3. Graph side info

Link coverage:

| Link | Share of impressed answers | Share of impression rows |
|---|---:|---:|
| answer present in info_answer | 100% | 100% |
| answer → question (non-null) | 96.84% | 98.91% |
| answer → author (non-null; null means anonymous) | 88.37% | 89.12% |
| linked question present in info_question | 100% | |
| linked author present in info_author | 100% | |
| answer has ≥1 answer-topic | 82.54% | |
| impressed question has ≥1 question-topic | 80.97% (of impressed questions) | |
| user present in info_user | 100% of users | |

Topic counts:

| | mean | share 0 | median | p95 | max |
|---|---:|---:|---:|---:|---:|
| topics per question | 2.66 | 19.0% | 3 | 5 | 16 |
| topics per answer | 3.03 | 17.5% | 3 | 5 | 16 |
| topics followed per user | 24.5 | 19.5% | 12 | 88 | 11,023 |

Feature columns. Every column is integer-coded, and only the ID-list columns contain nulls:

- **User (27 columns)**:
  - register_ts. Max 1526139856; 2.4% of users registered after the window started.
  - Categoricals: gender (3), login_freq (5), register_type (6), register_platform (4), platform (4), device_model (2,957), device_brand (372), province (330), city (1,230).
  - Binary flags: from_android, from_iphone, from_ipad, from_pc, from_mobile_web.
  - Counters (snapshot, no timestamp): n_followers, n_topics_followed, n_questions_followed, n_answers, n_questions, n_comments, n_thanks_recv, n_comments_recv, n_likes_recv, n_dislikes_recv.
  - topics_followed list: 19.5% null.
- **Answer (18 columns)**:
  - question_id: 3.16% null.
  - anonymous flag; author_id: 11.63% null, the same as the anonymous share.
  - Flags: high_value, editor_rec, has_pic, has_video.
  - create_ts. Min is 0, an anomaly. 50.0% of impressed answers were created inside the window, and 0.10% were created after their own first impression (anomaly).
  - Counters (snapshot): n_thanks, n_likes, n_comments, n_collections, n_dislikes, n_reports, n_helpless.
  - token list (not loaded); topics list: 17.5% null.
- **Question (8 columns)**: create_ts, plus counters n_answers, n_followers, n_invitations and n_comments (snapshot). Also a token list and a topics list (19.0% null).
- **Author (4 columns)**: excellent_author, excellent_answerer and n_followers (snapshot).
- **Topic**: ID only, with no features.

## 4. User click histories at impression time t

| History definition | share = 0 | mean | median | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| Known clicks: click_ts < t (causal) | 5.52% | 17.6 | 14 | 39 | 48 | 68 |
| Naive: clicked impressions with impression_ts < t | 5.05% | 17.8 | 14 | 39 | 49 | 69 |
| Prior impressions (impression_ts < t, any label) | 1.12% | 68.4 | 64 | 133 | 144 | 155 |

Known clicks by split (median / p95 / share 0): train 12 / 45 / 6.6%, val 15 / 49 / 3.9%, test 19 / 56 / 2.4%. These counts include the user's activity in all earlier splits, which is allowed causally. Histories start at the beginning of the dataset window; anything before 2018-05-03 is not available.

## 5. Global chronological split 70/10/20

A row goes to train if ts < c_val, to val if c_val ≤ ts < c_test, and to test otherwise. Each cutoff is the timestamp at the 70% or 80% position, so all rows with an equal timestamp land on the same side.

| Split | ts range (UTC) | Rows | Share | CTR | Users | Answers |
|---|---|---:|---:|---:|---:|---:|
| train | 05-02 16:01:38 → 05-11 13:56:33 | 69,984,767 | 0.7000 | 0.2720 | 679,205 | 482,038 |
| val | 05-11 13:56:34 → 05-12 03:28:28 | 9,997,867 | 0.1000 | 0.2777 | 227,442 | 126,090 |
| test | 05-12 03:28:29 → 05-13 10:59:27 | 19,995,889 | 0.2000 | 0.2585 | 269,447 | 173,755 |

The cutoffs are c_val = 1526046994 and c_test = 1526095709. Val covers about 13.5 h and test about 31.5 h, so the two splits sit on different times of day. Test CTR is about 1.9 points lower than val CTR.

## 6. Cold entities in val/test (full data, counted strictly before t over all earlier rows)

| | train | val | test |
|---|---:|---:|---:|
| answer, 0 prior impressions | 0.69% | 0.28% | 0.22% |
| answer, 1–4 prior impressions | 1.65% | 0.81% | 0.65% |
| answer, ≥5 prior impressions | 97.66% | 98.91% | 99.13% |
| answer, 0 known prior clicks (click_ts < t) | 2.74% | 1.57% | 1.31% |
| author, 0 prior impressions (among rows with an author) | 0.35% | 0.12% | 0.09% |
| anonymous answer (no author) | 11.10% | 10.17% | 10.49% |
| question, 0 prior impressions (among rows with a question) | 0.22% | 0.07% | 0.05% |
| answer has no question | 1.20% | 0.63% | 0.92% |
| user, 0 prior impressions | 1.35% | 0.79% | 0.45% |
| CTR of rows whose answer has 0 prior impressions | 0.145 | 0.150 | 0.150 |
| answer never seen in train (split-level) | | 5.06% | 13.52% |
| user never seen in train (split-level) | | 37.6% | 67.0% |

Item cold-start is a minor factor at full scale. User cold-start at the split level is large: most test users first appear after the train cutoff. The truncation at 160 impressions per user is the likely cause. Even so, within the causal stream 99.5% of test rows have at least one prior impression for their user.

## 7. Label-timing leak

For a row at time t, an earlier row r from the same user (impression_ts_r < t) has an observable label at t only if r was not clicked or click_ts_r < t. A row counts as leaking if impression_ts_r < t ≤ click_ts_r. Treating r's label as known at t would then reveal a future click, and the causally correct state at t is "no click observed yet".

| Metric | Value |
|---|---:|
| Rows with ≥1 leaking earlier row | **18.43%** |
| Rows with ≥1 earlier clicked row (naive view) | 94.95% |
| Leaking pairs / naive earlier-clicked pairs | 1.18% |
| Leaking pairs / all earlier same-user impression pairs | 0.31% |
| Leaking earlier rows per row: mean / p90 / p99 | 0.21 / 1 / 2 |
| Rows where the naive history has ≥1 click but the causal history has 0 | 0.475% |

The leak is small in volume but present in nearly a fifth of rows. It comes mostly from the last few impressions before t, which are also the most informative ones. "Not yet clicked" must be represented as unknown or negative-so-far, not as the final label. Same-timestamp batches add to this: 53.6% of rows sit in a (user, ts) group of size ≥2 (group size p95 = 3, p99 = 4).

## 8. Downsampling

- **There are no separate 1M or 20M files in the share.** The README says to take the top N lines of all eight files. Because the impression file is sorted by user_id and IDs are dense, top-N impressions means a prefix of users with their full histories:
  - Top 999,970 lines: users 0–7,973 (7,974 users), CTR 0.2687.
  - Top 19,999,857 lines: users 0–159,641 (159,642 users), CTR 0.2701.

  The README says these were "randomly sampled", which suggests user IDs were assigned randomly. The script did not verify this.
- **Proposed subsample**: draw users uniformly at random with `numpy.random.default_rng(0).choice(unique_users, round(0.125 * 798,086), replace=False)` and keep each selected user's full history.

| Subsample (12.5% users, seed 0) | Value |
|---|---:|
| Users / rows | 99,761 / 12,504,877 |
| Answers / questions / authors | 288,160 / 88,541 / 145,798 |
| Clicks / CTR | 3,374,930 / 0.26989 |
| Impressions/user: mean / median / p95 | 125.3 / 146 / 160 |
| Split (own cutoffs 1526046824 / 1526095466) | train 8,753,389 (CTR 0.2720) / val 1,250,489 (0.2769) / test 2,500,999 (0.2589) |
| Split under full-data cutoffs | 8,759,010 / 1,255,392 / 2,490,475 |
| Test cold answer (0 / 1–4 / ≥5 prior imps, within subsample) | 1.20% / 3.28% / 95.52% |
| Test answer with 0 known prior clicks | 4.50% |
| Test cold author (rows with an author) / cold question | 0.57% / 0.28% |
| Test user unseen in train | 67.1% |
| Rows with ≥1 leaking earlier label | 18.43% |

Cold shares rise in the subsample because item popularity is counted only from the sampled users. Leak and history statistics are unchanged. An alternative with direct comparability to published ZhihuRec-20M results is the official top-19,999,857-line prefix. Profiling it would require rerunning the script on that prefix.

## 9. Load time and memory

| Step | Time | Peak RSS after the step |
|---|---:|---:|
| Load inter_impression.csv.gz (pandas C engine, int32 IDs + int64 ts) | 32 s | 4.8 GB |
| Load side tables (answer without tokens, user, question, author, topic, query) | 12 s | 11.9 GB |
| Causal counts on full data (sorted-key searchsorted for users, answers, authors, questions) | 1,292 s | 25.0 GB |
| Whole script (full data + 12.5% subsample) | 1,488 s | 25.0 GB (26.2 GB per /usr/bin/time) |

## Caveats for a strictly causal protocol ("an impression at time t may only use information from before t")

1. **Labels arrive late.** Use the click events with click_ts < t, not the labels of rows with impression_ts < t. 18% of rows would otherwise see at least one future click. Delays are short (median 9 s) but have a long tail (up to 6.7 h).
2. **Negative delays** (0.22% of clicks, down to -107 min) cannot be causal. Either drop those rows, or clamp click_ts to impression_ts and treat the click as observable at max(impression_ts, click_ts). The script uses the max rule for "known" labels.
3. **Same-timestamp batches** (53.6% of rows) should not see each other's labels, or the fact that they were impressed together, unless the protocol explicitly allows within-batch impression context without labels. "Strictly before" excludes them.
4. **Static counters leak.** Answer, question, author and user counters (#likes, #collections, #followers, #answers, …) have no timestamps and were probably taken at or after collection time, so they may encode the outcome of the window. A strictly causal protocol should drop them, or at least flag them and ablate them. The same applies to topics_followed. Timestamped fields (create_ts, register_ts) are safe once compared against t. 0.10% of answers have create_ts after their first impression, and some create_ts values are 0; treat both as unknown.
5. **Graph edges** (answer→author, answer→question, question/answer→topic) are static metadata and do not reveal labels. Adding an answer node at time t when create_ts > t is a minor anomaly (0.1%). Edges derived from interactions (user→answer clicks) must use click_ts < t.
6. **Truncated histories.** Users have 10–160 impressions, which appears to be a per-user cap. The window has no warm-up period, and early-window events are underrepresented. The global split puts 67% of test rows on users unseen in train. Any "user is new" behaviour is partly an artifact of how the dataset was built.
7. **Short evaluation windows** (val about 13.5 h, test about 31.5 h) differ in time-of-day mix and CTR (0.278 vs 0.259). Expect calibration and logloss differences between val and test that have nothing to do with the model.
8. **Repeated exposures**: 1.93M (user, answer) pairs repeat, and 8,715 rows are exact duplicates. Decide whether a re-impression counts as history for the later one. Causally it is allowed if the earlier one is strictly before t.
9. **Queries** (inter_query) are timestamped and could be causal side information, but they were not profiled beyond their size and time range.
10. **Not computed**: the published 20M prefix was not profiled. Token vectors were not downloaded. Whether user IDs were randomly assigned was not checked.
