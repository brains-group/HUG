"""
MIND (MINDlarge train + dev) adapter (spec 06)
-----------------------------------------------
Rows: one per (impression, candidate news), from behaviors.tsv of the train and
dev releases concatenated (dev is strictly later).  Users are subsampled first
(--user-frac, default 0.2, seed 0).  Labels are the candidate click flags;
there is no click timestamp, so label_time == time (documented assumption).

History: the provided `history` field is a frozen pre-window snapshot (the same
for every impression of a user), used as the pre-window part of the click
history; in-window clicks are appended causally.

Graph: user –clicked→ news (pre-window clicks at window start − 1 ms),
user –skipped→ news (--skip-edges), news → category / subcategory / entity
(title + abstract entities with Confidence ≥ 0.5), subcategory → category, and
next_click transitions.  Timestamps are the log's local clock as given (tz 0).
"""

from __future__ import annotations

import json
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

NAME = "mind"
TZ_OFFSET_HOURS = 0
SPLIT_RULE = "day_snap"
ENTITY_MIN_CONFIDENCE = 0.5
METADATA_EVIDENCE_TYPES = ["subcategory", "entity"]
ADAPTER_VERSION = 1
# Raw column names that must never appear in the generic modules (spec 06 A10)
COLUMN_NAMES = ["news_id", "subcategory", "title_entities", "abstract_entities", "impression_id",
                "impressions", "WikidataId", "Confidence"]

BEHAVIOR_COLS = ["impression_id", "user", "time", "history", "impressions"]
NEWS_COLS = ["news_id", "category", "subcategory", "title", "abstract", "url",
             "title_entities", "abstract_entities"]


def _split_dirs(root: Path, variant: str) -> list[Path]:
    return [root / f"MIND{variant}_train", root / f"MIND{variant}_dev"]


def read_behaviors(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", header=None, names=BEHAVIOR_COLS, dtype=str,
                       quoting=3, na_filter=False)


def read_news(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", header=None, names=NEWS_COLS, dtype=str, quoting=3,
                       na_filter=False, usecols=["news_id", "category", "subcategory",
                                                 "title_entities", "abstract_entities"])


def parse_time_ms(s: pd.Series) -> np.ndarray:
    t = pd.to_datetime(s, format="%m/%d/%Y %I:%M:%S %p")
    return (t.to_numpy().astype("datetime64[ms]").astype(np.int64))


def entities(news: pd.DataFrame, min_conf: float) -> tuple[list[tuple[str, str]], dict]:
    """(news_id, WikidataId) pairs from title + abstract entities with Confidence >= min_conf."""
    pairs, kept_any, dropped_only = set(), 0, 0
    low = set()
    for nid, te, ae in zip(news["news_id"], news["title_entities"], news["abstract_entities"]):
        for field_ in (te, ae):
            if not field_ or field_ == "[]":
                continue
            for e in json.loads(field_):
                q = e.get("WikidataId")
                if not q:
                    continue
                if float(e.get("Confidence", 1.0)) >= min_conf:
                    pairs.add((nid, q))
                else:
                    low.add((nid, q))
    dropped = low - pairs
    stats = {"entity_pairs_kept": len(pairs), "entity_pairs_dropped_low_confidence": len(dropped)}
    return sorted(pairs), stats


def load(args) -> DatasetBundle:
    root = Path(args.data_dir)
    variant = getattr(args, "mind_variant", "large")
    dirs = _split_dirs(root, variant)
    beh = pd.concat([read_behaviors(d / "behaviors.tsv") for d in dirs], ignore_index=True)
    news = pd.concat([read_news(d / "news.tsv") for d in dirs], ignore_index=True)
    news = news.drop_duplicates("news_id", keep="first").reset_index(drop=True)
    stats = {"impressions_full": len(beh), "users_full": int(beh["user"].nunique())}

    # 1. users first (before any statistic)
    frac = float(args.user_frac)
    keep = sample_users(beh["user"].to_numpy(), frac, seed=0)
    beh = beh[beh["user"].isin(keep)].reset_index(drop=True)
    beh_time = parse_time_ms(beh["time"])

    # 2. candidate rows
    cand = beh["impressions"].str.split(" ")
    lens = cand.str.len().to_numpy()
    tok = pd.Series(np.concatenate(cand.to_numpy()))
    cand_news = tok.str[:-2].to_numpy()
    label = (tok.str[-1] == "1").to_numpy().astype(np.int8)
    imp = np.repeat(np.arange(len(beh)), lens)

    # 3. pre-window histories: one frozen list per user (from the user's earliest impression)
    first = pd.DataFrame({"user": beh["user"], "t": beh_time, "h": beh["history"]}) \
        .sort_values("t", kind="stable").drop_duplicates("user")
    hist = first.set_index("user")["h"]

    # 4. node ids
    users = np.sort(beh["user"].unique())
    umap = pd.Series(np.arange(len(users)), index=users)
    hist_lists = {u: (h.split(" ") if h else []) for u, h in hist.items()}
    hist_news = np.concatenate([np.asarray(v, dtype=object) for v in hist_lists.values()]) \
        if hist_lists else np.zeros(0, dtype=object)
    items = np.unique(np.r_[cand_news.astype(object), hist_news].astype(str))
    imap = pd.Series(np.arange(len(items)), index=items)
    meta = news.set_index("news_id").reindex(items)
    missing = int(meta["category"].isna().sum())
    if missing:
        logger.warning("MIND: %d items have no news.tsv row", missing)
    meta = meta.fillna("")

    user = umap[beh["user"].to_numpy()].to_numpy()[imp]
    item = imap[cand_news].to_numpy()
    time = beh_time[imp]
    order = np.argsort(time, kind="stable")
    user, item, time, label, imp = user[order], item[order], time[order], label[order], imp[order]
    label_time = time.copy()                     # no click timestamps in MIND

    pw_user = np.concatenate([np.full(len(hist_lists[u]), umap[u], np.int64) for u in users]) \
        if len(users) else np.zeros(0, np.int64)
    pw_item = np.concatenate([imap[hist_lists[u]].to_numpy() if hist_lists[u] else np.zeros(0, np.int64)
                              for u in users]).astype(np.int64) if len(users) else np.zeros(0, np.int64)

    # 5. split (day-snapped) and interactions
    t_val, t_test, split_info = split_cutoffs(time, SPLIT_RULE, TZ_OFFSET_HOURS,
                                              args.val_ratio, args.test_ratio)
    frame = pd.DataFrame({"impression_id": beh["impression_id"].to_numpy()[imp]})
    inter = make_interactions(user, item, time, label, label_time, t_val, t_test, frame=frame)

    # 6. per-row features (shared with the baselines)
    session = derive_sessions(user, time)
    ctx_cat = np.stack([hour_of_day(time, TZ_OFFSET_HOURS), day_of_week(time, TZ_OFFSET_HOURS)], 1)
    ctx_num = asof_item_label_stats(item, time, label, label_time)

    # 7. metadata nodes and links
    n_items = len(items)
    cat_names, cat_code = np.unique(meta["category"].to_numpy(), return_inverse=True)
    sub_names, sub_code = np.unique(meta["subcategory"].to_numpy(), return_inverse=True)
    part = np.unique(np.stack([sub_code, cat_code], 1), axis=0)
    ent_pairs, ent_stats = entities(pd.DataFrame({
        "news_id": items, "title_entities": meta["title_entities"].to_numpy(),
        "abstract_entities": meta["abstract_entities"].to_numpy()}), ENTITY_MIN_CONFIDENCE)
    ent_item = imap[[p[0] for p in ent_pairs]].to_numpy() if ent_pairs else np.zeros(0, np.int64)
    ent_names, ent_code = (np.unique([p[1] for p in ent_pairs], return_inverse=True)
                           if ent_pairs else (np.zeros(0, str), np.zeros(0, np.int64)))
    stats.update(ent_stats)

    counts = {"user": len(users), "video": n_items, "category": len(cat_names),
              "subcategory": len(sub_names), "entity": len(ent_names)}
    gb = GraphBuilder(["user", "video", "category", "subcategory", "entity"], counts)
    clk = label == 1
    window_start = int(time.min())
    gb.add("clicked", "user", "video", np.r_[user[clk], pw_user], np.r_[item[clk], pw_item],
           np.r_[label_time[clk], np.full(len(pw_user), window_start - 1, np.int64)])
    if args.skip_edges:
        gb.add("skipped", "user", "video", user[~clk], item[~clk], label_time[~clk])
    gb.add("in_category", "video", "category", np.arange(n_items), cat_code)
    gb.add("in_subcategory", "video", "subcategory", np.arange(n_items), sub_code)
    gb.add("part_of", "subcategory", "category", part[:, 0], part[:, 1])
    gb.add("mentions", "video", "entity", ent_item, ent_code)
    tr_s, tr_d, tr_t = transitions(user, item, label_time, time, clk)
    gb.add("next_click", "video", "video", tr_s, tr_d, tr_t, reverse=False)
    graph = gb.build()
    stats["edges"] = graph.edge_counts()
    stats["pre_window_clicks"] = int(len(pw_user))

    meta_features = {}
    if getattr(args, "entity_init", "id") == "transe" and len(ent_names):
        meta_features["entity"] = transe_matrix(dirs[0] / "entity_embedding.vec", ent_names)

    item_cat = pd.DataFrame({"category": meta["category"].to_numpy(),
                             "subcategory": meta["subcategory"].to_numpy()})

    baseline = Const(BaselineTables(user_ids=users, item_ids=items,
                                    user_cats=pd.DataFrame(index=range(len(users))),
                                    item_cats=item_cat.copy()))

    fp = data_fingerprint(inter, {"dataset": NAME, "user_frac": frac, "user_sample": _str_hash(users),
        "split_rule_used": split_info["rule_used"],
        "skip_edges": bool(args.skip_edges), "adapter_version": ADAPTER_VERSION})
    stats.update({"rows": len(time), "users": len(users), "items": n_items,
                  "candidate_items": int(len(np.unique(item))), "ctr": float(label.mean()),
                  "impressions": int(len(beh))})
    return DatasetBundle(
        name=NAME, inter=inter, session=session,
        ctx_cat=ctx_cat, ctx_cat_names=["hour", "dow"],
        ctx_num=ctx_num, ctx_num_names=list(ITEM_STAT_COLS),
        n_users=len(users), n_items=n_items,
        user_x=np.zeros((len(users), 1), np.float32), user_onehot=np.zeros((len(users), 0), np.int64),
        item_cat=item_cat, item_cat_cols=["category", "subcategory"],
        graph=graph,
        item_features=LabelStatItemFeatures(item, time, label, label_time, n_items),
        metadata_links=[(np.arange(n_items), sub_code), (ent_item, ent_code)],
        metadata_evidence_types=METADATA_EVIDENCE_TYPES,
        split_rule=SPLIT_RULE, split_info=split_info,
        timezone="log clock (UTC offset 0)", tz_offset_hours=TZ_OFFSET_HOURS,
        fingerprint=fp, pre_window=(pw_user, pw_item), user_sample=(frac, 0),
        baseline_fn=baseline, stats=stats, meta_features=meta_features,
    )


def _str_hash(values: np.ndarray) -> str:
    import hashlib
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()[:16]


def transe_matrix(path: Path, names: np.ndarray) -> np.ndarray:
    """Frozen TransE vectors (train release) for `names`; missing rows are 0 (ID-only)."""
    vec = pd.read_csv(path, sep="\t", header=None, index_col=0, quoting=3)
    vec = vec.dropna(axis=1, how="all")
    out = vec.reindex(names).fillna(0.0).to_numpy(np.float32)
    logger.info("TransE: %d / %d entities covered", int(vec.index.isin(names).sum()), len(names))
    return out
