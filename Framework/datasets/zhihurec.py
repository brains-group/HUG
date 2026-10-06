"""
ZhihuRec-100M adapter (spec 06)
--------------------------------
Rows: one per impression (inter_impression.csv), after a random user sample
(--user-frac, default 0.125, seed 0; the file is sorted by user, so a prefix
would be a biased user range).  Times are Unix seconds in Beijing time.

Label timing (spec 06 B): a click is known at max(click_ts, impression_ts)
(0.22% of clicks are logged before their impression; clamped); a non-click is
known at impression_ts + W (--label-delay-s, default 900 s, just above the p99
click delay).  Every label-derived input uses label_time.

Graph: user –clicked→ answer, user –skipped→ answer (--skip-edges),
answer → author (non-anonymous), answer → question, answer → topic,
question → topic, next_click transitions.

Not used (spec 06 E): follow lists and every counter column — untimed snapshots
taken at collection time.  --snapshot-features adds the counters for a
sensitivity run only.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from datasets.base import (
    BaselineTables, Const, DatasetBundle, GraphBuilder, LabelStatItemFeatures, data_fingerprint,
    make_interactions, sample_users, split_cutoffs, transitions,
)
from features import (
    ITEM_STAT_COLS, asof_item_label_stats, day_of_week, derive_sessions, hour_of_day,
)

logger = logging.getLogger(__name__)

NAME = "zhihurec"
TZ_OFFSET_HOURS = 8
SPLIT_RULE = "day_snap"
METADATA_EVIDENCE_TYPES = ["author", "question"]
ADAPTER_VERSION = 1

IMPRESSION_COLS = ["user_id", "answer_id", "impression_ts", "click_ts"]
ANSWER_COLS = ["answer_id", "question_id", "anonymous", "author_id", "high_value", "editor_rec",
               "create_ts", "has_pic", "has_video", "n_thanks", "n_likes", "n_comments",
               "n_collections", "n_dislikes", "n_reports", "n_helpless", "tokens", "topics"]
USER_COLS = ["user_id", "register_ts", "gender", "login_freq", "n_followers", "n_topics_followed",
             "n_questions_followed", "n_answers", "n_questions", "n_comments", "n_thanks_recv",
             "n_comments_recv", "n_likes_recv", "n_dislikes_recv", "register_type",
             "register_platform", "from_android", "from_iphone", "from_ipad", "from_pc",
             "from_mobile_web", "device_model", "device_brand", "platform", "province", "city",
             "topics_followed"]
QUESTION_COLS = ["question_id", "create_ts", "n_answers", "n_followers", "n_invitations",
                 "n_comments", "tokens", "topics"]
AUTHOR_COLS = ["author_id", "excellent_author", "n_followers", "excellent_answerer"]

USER_CAT_COLS = ["gender", "login_freq", "register_type", "register_platform", "platform",
                 "device_model", "device_brand", "province", "city"]
USER_FLAG_COLS = ["from_android", "from_iphone", "from_ipad", "from_pc", "from_mobile_web"]
ANSWER_FLAG_COLS = ["anonymous", "high_value", "editor_rec", "has_pic", "has_video"]
AUTHOR_FLAG_COLS = ["excellent_author", "excellent_answerer"]
# Untimed snapshot counters (never in the main protocol; --snapshot-features only)
SNAPSHOT_USER_COLS = ["n_followers", "n_topics_followed", "n_questions_followed", "n_answers",
                      "n_questions", "n_comments", "n_thanks_recv", "n_comments_recv",
                      "n_likes_recv", "n_dislikes_recv"]
SNAPSHOT_ANSWER_COLS = ["n_thanks", "n_likes", "n_comments", "n_collections", "n_dislikes",
                        "n_reports", "n_helpless"]
SNAPSHOT_QUESTION_COLS = ["n_answers", "n_followers", "n_invitations", "n_comments"]
SNAPSHOT_AUTHOR_COLS = ["n_followers"]
AGE_COLS = ["account_age_log", "account_age_unknown", "answer_age_log", "answer_age_unknown"]


def _read(path: Path, names: list[str], usecols: list[str], dtype=None) -> pd.DataFrame:
    return pd.read_csv(path, header=None, names=names, usecols=usecols, dtype=dtype,
                       keep_default_na=False, na_values=[""], engine="c")


def _topic_pairs(ids: np.ndarray, topics: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """(owner, topic_raw) pairs from space-separated topic lists (NaN = none)."""
    s = topics.fillna("").astype(str).str.split()
    lens = s.str.len().to_numpy()
    flat = np.concatenate(s.to_numpy()) if lens.sum() else np.zeros(0, str)
    return np.repeat(ids, lens), flat.astype(np.int64)


def _age_hours(t_s: np.ndarray, ref_s: np.ndarray, unknown: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """log1p(hours since ref) at t; negative ages are clamped to 0 and flagged unknown."""
    age = (t_s - ref_s) / 3600.0
    bad = unknown | (age < 0)
    return np.log1p(np.where(bad, 0.0, age)).astype(np.float32), bad.astype(np.float32)


def load(args) -> DatasetBundle:
    raw = Path(args.data_dir)
    W = int(args.label_delay_s)
    imp = _read(raw / "inter_impression.csv.gz", IMPRESSION_COLS, IMPRESSION_COLS,
                {"user_id": np.int64, "answer_id": np.int64, "impression_ts": np.int64,
                 "click_ts": np.int64})
    stats = {"rows_full": len(imp), "users_full": int(imp["user_id"].nunique())}

    # 1. users first (before any statistic)
    frac = float(args.user_frac)
    keep = sample_users(imp["user_id"].to_numpy(), frac, seed=0)
    imp = imp[np.isin(imp["user_id"].to_numpy(), keep)].reset_index(drop=True)
    imp = imp.iloc[np.argsort(imp["impression_ts"].to_numpy(), kind="stable")].reset_index(drop=True)

    t_s = imp["impression_ts"].to_numpy(np.int64)
    c_s = imp["click_ts"].to_numpy(np.int64)
    clicked = c_s > 0
    lt_s = np.where(clicked, np.maximum(c_s, t_s), t_s + W)
    stats["clicks_before_impression_clamped"] = int((clicked & (c_s < t_s)).sum())
    time, label_time = t_s * 1000, lt_s * 1000
    label = clicked.astype(np.int8)

    users_raw = np.unique(imp["user_id"].to_numpy())
    items_raw = np.unique(imp["answer_id"].to_numpy())
    user = np.searchsorted(users_raw, imp["user_id"].to_numpy())
    item = np.searchsorted(items_raw, imp["answer_id"].to_numpy())
    n_users, n_items = len(users_raw), len(items_raw)

    # 2. side tables restricted to the sample
    snap = bool(getattr(args, "snapshot_features", False))
    ans = _read(raw / "info_answer.csv.gz", ANSWER_COLS, [c for c in ANSWER_COLS if c != "tokens"],
                {"topics": str})
    ans = ans.set_index("answer_id").reindex(items_raw)
    usr = _read(raw / "info_user.csv.gz", USER_COLS,
                [c for c in USER_COLS if c != "topics_followed"])
    usr = usr.set_index("user_id").reindex(users_raw)
    qcols = ["question_id", "create_ts", "topics"] + (SNAPSHOT_QUESTION_COLS if snap else [])
    qs = _read(raw / "info_question.csv.gz", QUESTION_COLS, qcols, {"topics": str}).set_index("question_id")
    au = _read(raw / "info_author.csv.gz", AUTHOR_COLS, AUTHOR_COLS).set_index("author_id")

    # 3. metadata nodes
    has_auth = ans["author_id"].notna().to_numpy()
    has_q = ans["question_id"].notna().to_numpy()
    authors_raw = np.unique(ans["author_id"].dropna().astype(np.int64).to_numpy())
    questions_raw = np.unique(ans["question_id"].dropna().astype(np.int64).to_numpy())
    a_item = np.flatnonzero(has_auth)
    a_code = np.searchsorted(authors_raw, ans["author_id"].to_numpy()[has_auth].astype(np.int64))
    q_item = np.flatnonzero(has_q)
    q_code = np.searchsorted(questions_raw, ans["question_id"].to_numpy()[has_q].astype(np.int64))
    qsub = qs.reindex(questions_raw)
    at_owner, at_topic = _topic_pairs(np.arange(n_items), ans["topics"])
    qt_owner, qt_topic = _topic_pairs(np.arange(len(questions_raw)), qsub["topics"])
    topics_raw = np.unique(np.r_[at_topic, qt_topic])
    counts = {"user": n_users, "video": n_items, "author": len(authors_raw),
              "question": len(questions_raw), "topic": len(topics_raw)}

    # 4. split (day-snapped, with fallback) and interactions
    t_val, t_test, split_info = split_cutoffs(time, SPLIT_RULE, TZ_OFFSET_HOURS,
                                              args.val_ratio, args.test_ratio)
    inter = make_interactions(user, item, time, label, label_time, t_val, t_test)

    # 5. per-row features: context, label-timed item stats, causal ages
    session = derive_sessions(user, time)
    ctx_cat = np.stack([hour_of_day(time, TZ_OFFSET_HOURS), day_of_week(time, TZ_OFFSET_HOURS)], 1)
    reg = usr["register_ts"].to_numpy(np.float64)[user]
    acc_log, acc_bad = _age_hours(t_s, np.nan_to_num(reg), np.isnan(reg))
    cre = ans["create_ts"].to_numpy(np.float64)[item]
    cre_bad = np.isnan(cre) | (cre <= 0)
    ans_log, ans_bad = _age_hours(t_s, np.nan_to_num(cre), cre_bad)
    num = [asof_item_label_stats(item, time, label, label_time),
           np.stack([acc_log, acc_bad, ans_log, ans_bad], 1)]
    num_names = list(ITEM_STAT_COLS) + AGE_COLS
    stats["answers_created_after_impression_rows"] = int(((cre > t_s) & ~cre_bad).sum())
    if snap:
        logger.warning("ZhihuRec: --snapshot-features adds untimed counters (sensitivity run only)")
        q_of_item = np.full(n_items, -1)
        q_of_item[q_item] = q_code
        a_of_item = np.full(n_items, -1)
        a_of_item[a_item] = a_code
        snap_cols = []
        for c in SNAPSHOT_USER_COLS:
            snap_cols.append(("user_" + c, usr[c].to_numpy(np.float64)[user]))
        for c in SNAPSHOT_ANSWER_COLS:
            snap_cols.append(("answer_" + c, ans[c].to_numpy(np.float64)[item]))
        qv = qsub.reset_index(drop=True)
        for c in SNAPSHOT_QUESTION_COLS:
            v = np.r_[qv[c].to_numpy(np.float64), np.nan][q_of_item[item]]
            snap_cols.append(("question_" + c, v))
        av = au.reindex(authors_raw).reset_index(drop=True)
        for c in SNAPSHOT_AUTHOR_COLS:
            v = np.r_[av[c].to_numpy(np.float64), np.nan][a_of_item[item]]
            snap_cols.append(("author_" + c, v))
        num.append(np.stack([np.log1p(np.nan_to_num(np.maximum(v, 0))) for _, v in snap_cols], 1))
        num_names += [f"snap_{n}" for n, _ in snap_cols]
    ctx_num = np.concatenate(num, axis=1).astype(np.float32)

    # 6. node features
    user_onehot = np.stack([pd.factorize(usr[c].fillna(-1).astype(np.int64), sort=True)[0]
                            for c in USER_CAT_COLS], 1).astype(np.int64)
    user_x = usr[USER_FLAG_COLS].fillna(0).to_numpy(np.float32)
    auth_flags = au.reindex(ans["author_id"].to_numpy())[AUTHOR_FLAG_COLS].reset_index(drop=True)
    item_cat = pd.concat([ans[ANSWER_FLAG_COLS].reset_index(drop=True), auth_flags], axis=1)
    item_cat = item_cat.apply(lambda c: c.map(lambda x: "" if pd.isna(x) else str(int(x))))
    item_cat_cols = ANSWER_FLAG_COLS + AUTHOR_FLAG_COLS

    # 7. graph
    gb = GraphBuilder(["user", "video", "author", "question", "topic"], counts)
    clk = label == 1
    gb.add("clicked", "user", "video", user[clk], item[clk], label_time[clk])
    if args.skip_edges:
        gb.add("skipped", "user", "video", user[~clk], item[~clk], label_time[~clk])
    gb.add("written_by", "video", "author", a_item, a_code)
    gb.add("answers", "video", "question", q_item, q_code)
    gb.add("has_topic", "video", "topic", at_owner, np.searchsorted(topics_raw, at_topic))
    gb.add("question_has_topic", "question", "topic", qt_owner, np.searchsorted(topics_raw, qt_topic))
    tr_s, tr_d, tr_t = transitions(user, item, label_time, time, clk)
    gb.add("next_click", "video", "video", tr_s, tr_d, tr_t, reverse=False)
    graph = gb.build()
    stats["edges"] = graph.edge_counts()

    uc = usr[USER_CAT_COLS + USER_FLAG_COLS].reset_index(drop=True)
    uc = uc.apply(lambda c: c.map(lambda x: "" if pd.isna(x) else str(int(x))))
    ic = item_cat.copy()
    ic["author_id"] = ans["author_id"].map(lambda x: "" if pd.isna(x) else str(int(x))).to_numpy()
    ic["question_id"] = ans["question_id"].map(lambda x: "" if pd.isna(x) else str(int(x))).to_numpy()
    baseline = Const(BaselineTables(user_ids=users_raw, item_ids=items_raw, user_cats=uc, item_cats=ic))

    frac_hash = hashlib.sha256(np.ascontiguousarray(users_raw).tobytes()).hexdigest()[:16]
    fp = data_fingerprint(inter, {"dataset": NAME, "user_frac": frac, "user_sample": frac_hash,
                                  "split_rule_used": split_info["rule_used"], "label_delay_s": W,
                                  "skip_edges": bool(args.skip_edges), "snapshot_features": snap,
                                  "adapter_version": ADAPTER_VERSION})
    stats.update({"rows": len(time), "users": n_users, "items": n_items,
                  "authors": len(authors_raw), "questions": len(questions_raw),
                  "topics": len(topics_raw), "ctr": float(label.mean())})
    return DatasetBundle(
        name=NAME, inter=inter, session=session,
        ctx_cat=ctx_cat, ctx_cat_names=["hour", "dow"],
        ctx_num=ctx_num, ctx_num_names=num_names,
        n_users=n_users, n_items=n_items, user_x=user_x, user_onehot=user_onehot,
        item_cat=item_cat, item_cat_cols=item_cat_cols, graph=graph,
        item_features=LabelStatItemFeatures(item, time, label, label_time, n_items),
        metadata_links=[(a_item, a_code), (q_item, q_code)],
        metadata_evidence_types=METADATA_EVIDENCE_TYPES,
        split_rule=SPLIT_RULE, split_info=split_info,
        timezone="Asia/Shanghai", tz_offset_hours=TZ_OFFSET_HOURS,
        fingerprint=fp, user_sample=(frac, 0), baseline_fn=baseline, stats=stats,
    )
