"""
KuaiRand-1K adapter (spec 06 A)
--------------------------------
Wraps the original KuaiRand pipeline (data_loader.KuaiRandLoader,
temporal.build_interactions, hkg_constructor.HKGConstructor) and exposes it as
a DatasetBundle.  Every output is bitwise identical to the pre-adapter HUG
pipeline (spec 06 test A1): exact 70/10/20 split, label_time == time, the
14 structural relations + next_in_session, [tab, hour] context and the
as-of video statistics.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from data_loader import KuaiRandData, KuaiRandLoader, compute_video_statistics
from datasets.base import BaselineTables, DatasetBundle, GraphSpec, data_fingerprint
from features import (
    HUG_VIDEO_CAT_COLS, USER_CAT_COLS, VIDEO_CAT_COLS, VIDEO_STAT_COLS, asof_video_statistics, hour_of_day,
    user_categoricals, video_categoricals,
)
from datasets.kuairand_hkg import HKGConstructor, video_feature_matrix
from hkg_constructor import HKGBundle
from data_loader import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL, assign_split, chronological_cutoffs
from features import FEEDBACK_COLS
from temporal import Interactions

logger = logging.getLogger(__name__)

NAME = "kuairand"
TZ_OFFSET_HOURS = 8                                  # Asia/Shanghai
METADATA_EVIDENCE_TYPES = ["author"]
# Raw column names that must never appear in the generic modules (spec 06 A10)
COLUMN_NAMES = sorted(
    set(KuaiRandLoader.LOG_USECOLS) | set(USER_CAT_COLS) | set(VIDEO_CAT_COLS) | set(VIDEO_STAT_COLS)
    | set(FEEDBACK_COLS) | {"session_id", "tag_list", "play_ratio", "duration_s", "aspect_ratio"})


def hug_relations():
    """(forward_ids, reverse_ids, [(src_type, dst_type)] by relation id): the 14 structural
    relations (spec 01) plus the one-directional next_in_session."""
    from main import REL_MAP, REVERSE_REL_MAP
    seq_et = ("video", "next_in_session", "video")
    forward = dict(REL_MAP)
    forward[seq_et] = 2 * len(REL_MAP)                    # one-directional
    types = [None] * (2 * len(REL_MAP) + 1)
    names = [None] * (2 * len(REL_MAP) + 1)
    for et, rid in forward.items():
        types[rid], names[rid] = (et[0], et[2]), et[1]
    for et, rid in REVERSE_REL_MAP.items():
        types[rid], names[rid] = (et[2], et[0]), f"rev_{et[1]}"
    return forward, REVERSE_REL_MAP, types, names


def build_interactions(
    data:       KuaiRandData,
    val_ratio:  float = 0.1,
    test_ratio: float = 0.2,
    max_prefix: int   = 50,
) -> Interactions:
    """
    Global chronological train/val/test split (data_loader.chronological_cutoffs,
    shared with Baselines/preprocess.py) plus, for every row, the index range
    of its session prefix: up to `max_prefix` items of the same session with a
    strictly earlier timestamp.
    """
    sm = data.session_map
    df = sm[sm["user_id"].isin(data.user_id_map) & sm["video_id"].isin(data.video_id_map)]
    df = df[["user_id", "video_id", "session_id", "time_ms", "is_rand", "tab"] + FEEDBACK_COLS]
    df = df.sort_values("time_ms", kind="stable").reset_index(drop=True)

    user  = df["user_id"].map(data.user_id_map).to_numpy(np.int64)
    video = df["video_id"].map(data.video_id_map).to_numpy(np.int64)
    time  = df["time_ms"].to_numpy(np.int64)
    sess  = df["session_id"].to_numpy(np.int64)
    ips   = np.where(df["is_rand"].to_numpy() == 1, 1.0, 0.9963).astype(np.float32)
    label = df["is_click"].to_numpy(np.float32)

    t_val, t_test = chronological_cutoffs(time, val_ratio, test_ratio)
    split = assign_split(time, t_val, t_test)
    assert time[split == SPLIT_TRAIN].max() < time[split == SPLIT_VAL].min()
    assert time[split == SPLIT_VAL].max()   < time[split == SPLIT_TEST].min()

    # Session-ordered item sequence; ties broken by global row order
    n     = len(df)
    order = np.lexsort((np.arange(n), time, sess))
    s_o, t_o = sess[order], time[order]
    pos   = np.arange(n)
    new_sess  = np.r_[True, s_o[1:] != s_o[:-1]]
    new_block = new_sess | np.r_[True, t_o[1:] != t_o[:-1]]
    sess_start  = np.maximum.accumulate(np.where(new_sess,  pos, 0))
    block_start = np.maximum.accumulate(np.where(new_block, pos, 0))

    pre_end   = np.empty(n, dtype=np.int64)
    pre_start = np.empty(n, dtype=np.int64)
    pre_end[order]   = block_start                              # excludes same-time items
    pre_start[order] = np.maximum(sess_start, block_start - max_prefix)

    logger.info("Chronological split  train=%s  val=%s  test=%s  (t_val=%d  t_test=%d)",
                f"{(split == SPLIT_TRAIN).sum():,}", f"{(split == SPLIT_VAL).sum():,}",
                f"{(split == SPLIT_TEST).sum():,}", t_val, t_test)
    return Interactions(user, video, time, ips, label, split,
                        pre_start, pre_end, video[order], t_val, t_test, max_prefix,
                        frame=df)


class KuaiRandItemFeatures:
    """Video node features (static metadata + window statistics) as of a cutoff."""

    LOG_COLS = ["video_id", "time_ms", "long_view", "is_click", "is_like", "is_follow",
                "is_forward", "is_comment"]

    def __init__(self, data: KuaiRandData) -> None:
        vb = data.video_basic[data.video_basic["video_id"].isin(data.video_id_map)].copy()
        vb["node_idx"] = vb["video_id"].map(data.video_id_map)
        self.vb = vb.sort_values("node_idx")
        self.log = data.log_combined[self.LOG_COLS]

    def at(self, cutoff: int) -> np.ndarray:
        stats = compute_video_statistics(self.log[self.log["time_ms"] < int(cutoff)])
        return video_feature_matrix(self.vb, stats)


def load_hkg_bundle(args, data: KuaiRandData) -> HKGBundle:
    """The timed HKG, from the cache when it matches the data."""
    from main import load_hkg, resolve_cache_dir, save_hkg
    cache_dir = resolve_cache_dir(args)
    bundle = load_hkg(cache_dir, data) if cache_dir else None
    if bundle is None:
        bundle = HKGConstructor(data, device="cpu").build()
        if cache_dir:
            save_hkg(bundle, cache_dir)
    return bundle


def load(args, data: KuaiRandData | None = None, hkg: HKGBundle | None = None) -> DatasetBundle:
    if data is None:
        data = KuaiRandLoader(args.data_dir, min_interactions=args.min_interactions,
                              filter_ads=True).load()
    inter = build_interactions(data, args.val_ratio, args.test_ratio, max_prefix=args.max_seq_len)
    inter.label_time = inter.time
    frame = inter.frame
    session = frame["session_id"].to_numpy(np.int64)

    tab = frame["tab"].fillna(0).to_numpy(np.int64)
    ctx_cat = np.stack([tab, hour_of_day(inter.time, TZ_OFFSET_HOURS)], axis=1)
    ctx_num = asof_video_statistics(frame)

    if hkg is None:
        hkg = load_hkg_bundle(args, data)

    def master():
        from main import _merged_edge_index
        fwd, rev, _, _ = hug_relations()
        m = _merged_edge_index(hkg.full_graph, fwd, rev)
        logger.info("HUG master relation index: %s edges", f"{m[0].shape[1]:,}")
        return m

    _, _, types, names = hug_relations()
    graph = GraphSpec(
        node_types=["user", "video", "author", "category"],
        counts={"user": hkg.n_users, "video": hkg.n_videos, "author": hkg.n_authors,
                "category": hkg.n_categories},
        relations=types, relation_names=names, master_fn=master)

    vc = video_categoricals(data.video_basic)
    vc = vc[vc["video_id"].isin(data.video_id_map)]
    vc = vc.assign(node=vc["video_id"].map(data.video_id_map)).sort_values("node")

    # metadata evidence: video → author (videos with a missing author have no link)
    vb = data.video_basic[data.video_basic["video_id"].isin(data.video_id_map)]
    has = vb["author_id"].notna()
    a_codes = pd.factorize(vb.loc[has, "author_id"])[0]
    links = [(vb.loc[has, "video_id"].map(data.video_id_map).to_numpy(np.int64), a_codes)]

    ug = hkg.full_graph["user"]

    def baseline():
        users = sorted(data.user_id_map, key=data.user_id_map.get)
        videos = sorted(data.video_id_map, key=data.video_id_map.get)
        uc = user_categoricals(data.user_features).set_index("user_id").reindex(users)
        return BaselineTables(
            user_ids=np.asarray(users), item_ids=np.asarray(videos),
            user_cats=uc.reset_index(drop=True),
            item_cats=vc.drop(columns=["video_id", "node"]).reset_index(drop=True))

    return DatasetBundle(
        name=NAME, inter=inter, session=session,
        ctx_cat=ctx_cat, ctx_cat_names=["tab", "hour"],
        ctx_num=ctx_num, ctx_num_names=list(VIDEO_STAT_COLS),
        n_users=len(data.user_id_map), n_items=len(data.video_id_map),
        user_x=ug.x.float().cpu().numpy(), user_onehot=ug.onehot.long().cpu().numpy(),
        item_cat=vc.reset_index(drop=True), item_cat_cols=list(HUG_VIDEO_CAT_COLS),
        graph=graph, item_features=KuaiRandItemFeatures(data),
        metadata_links=links, metadata_evidence_types=METADATA_EVIDENCE_TYPES,
        split_rule="quantile_70_10_20",
        split_info={"rule": "quantile_70_10_20", "rule_used": "quantile",
                    "cutoffs": [int(inter.t_val), int(inter.t_test)]},
        timezone="Asia/Shanghai", tz_offset_hours=TZ_OFFSET_HOURS,
        fingerprint=data_fingerprint(inter), baseline_fn=baseline,
    )
