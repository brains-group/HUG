"""
Shared per-row features — the single source of truth for HUG and the baselines
-------------------------------------------------------------------------------
Everything here is computed on the canonical interaction table
(temporal.build_interactions: one row per interaction, globally time-sorted;
a row's *row id* is its position in that table).  Framework/hug.py and
Baselines/preprocess.py both call these functions, so the two pipelines feed
identical values for every row.

Causality: every per-row feature uses only interactions with a strictly
earlier timestamp.  Rows sharing a timestamp never see each other.  Inputs
derived from a *label* (click histories, click counts, CTRs) use the row's
`label_time` instead: the earliest time the label is known (spec 06 B).
Exposure-only inputs (impression counts) use `time`.  For KuaiRand and MIND
label_time == time.

KuaiRand-specific column names live here and in data_loader.py only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Column registries ─────────────────────────────────────────────────────────

USER_CAT_COLS = [
    "user_active_degree", "is_lowactive_period", "is_live_streamer",
    "is_video_author", "follow_user_num_range", "fans_user_num_range",
    "friend_user_num_range", "register_days_range",
] + [f"onehot_feat{i}" for i in range(18)]

# Video categorical columns, same encoding everywhere.  HUG embeds the
# HUG_VIDEO_CAT_COLS subset directly; author and primary tag reach it as graph
# nodes.
VIDEO_CAT_COLS     = ["author_id", "music_id", "music_type", "upload_type", "primary_tag"]
HUG_VIDEO_CAT_COLS = ["music_id", "music_type", "upload_type"]

# Per-row point-in-time video statistics (same names as the snapshot stats)
VIDEO_STAT_COLS = [
    "global_cvr", "show_cnt_log", "play_cnt_log", "like_cnt_log",
    "follow_cnt_log", "share_cnt_log", "comment_cnt_log",
]
# count column → feedback flag it sums (show_cnt counts rows)
STAT_SOURCES = {
    "valid_play_cnt": "long_view", "play_cnt": "is_click", "like_cnt": "is_like",
    "follow_cnt": "is_follow", "share_cnt": "is_forward", "comment_cnt": "is_comment",
}
FEEDBACK_COLS = sorted(set(STAT_SOURCES.values()))

HOUR_MS = 3_600_000


# ── Context ───────────────────────────────────────────────────────────────────

def hour_of_day(time_ms: np.ndarray, tz_offset_hours: int = 8) -> np.ndarray:
    """Hour of day in the dataset's local time (default UTC+8, KuaiRand/ZhihuRec)."""
    return ((np.asarray(time_ms) // HOUR_MS + tz_offset_hours) % 24).astype(np.int64)


def day_of_week(time_ms: np.ndarray, tz_offset_hours: int = 8) -> np.ndarray:
    """Day of week in local time, 0 = Monday (1970-01-01 was a Thursday)."""
    days = (np.asarray(time_ms, np.int64) + tz_offset_hours * HOUR_MS) // (24 * HOUR_MS)
    return ((days + 3) % 7).astype(np.int64)


def video_categoricals(video_basic: pd.DataFrame) -> pd.DataFrame:
    """video_id plus VIDEO_CAT_COLS as strings ('' for missing)."""
    vb = video_basic[["video_id", "author_id", "music_id", "music_type",
                      "upload_type", "tag_list"]].copy()
    vb["primary_tag"] = vb["tag_list"].map(lambda t: t[0] if len(t) else -1)
    out = vb[["video_id"] + VIDEO_CAT_COLS].copy()
    for c in VIDEO_CAT_COLS:
        out[c] = out[c].map(_cat_str)
    return out


def _cat_str(x) -> str:
    """Canonical string for a categorical value (ints and integral floats agree)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    if isinstance(x, (float, np.floating)) and float(x).is_integer():
        return str(int(x))
    return str(x)


def user_categoricals(user_features: pd.DataFrame) -> pd.DataFrame:
    """user_id plus USER_CAT_COLS as strings."""
    out = user_features[["user_id"] + USER_CAT_COLS].copy()
    for c in USER_CAT_COLS:
        out[c] = out[c].map(_cat_str)
    return out


# ── Behaviour history ─────────────────────────────────────────────────────────

def click_history(
    user:       np.ndarray,
    time:       np.ndarray,
    item:       np.ndarray,
    click:      np.ndarray,
    max_len:    int,
    label_time: np.ndarray | None = None,
    pre_window: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For every row, the user's last `max_len` known clicks, oldest first: clicked
    rows whose label is known strictly before the row's time (label_time < t;
    label_time defaults to time).

    `pre_window` = (user, item) clicks from before the logging window, oldest first
    per user (MIND).  They precede every in-window click and are always visible.
    Their `item` values must be in the same id space as `item`.

    Returns (seq, start, end): row i's history is seq[start[i]:end[i]].
    `seq` holds `item` values, per user: pre-window items, then clicked rows
    ordered by (label_time, time, row).
    """
    user, time = np.asarray(user, np.int64), np.asarray(time, np.int64)
    lt = time if label_time is None else np.asarray(label_time, np.int64)
    clicked = np.flatnonzero(np.asarray(click) == 1)
    order   = clicked[np.lexsort((clicked, time[clicked], lt[clicked], user[clicked]))]
    item    = np.asarray(item)

    t0 = int(min(time.min(), lt.min())) - 1                # pre-window tokens sit at t0
    span = int(max(time.max(), lt.max()) - t0) + 1
    tok_user, tok_t, seq = user[order], lt[order], item[order]
    if pre_window is not None and len(pre_window[0]):
        pu = np.asarray(pre_window[0], np.int64)
        pos = np.r_[np.arange(len(pu)), np.arange(len(order))]
        tok_user = np.r_[pu, tok_user]
        tok_t = np.r_[np.full(len(pu), t0, np.int64), tok_t]
        seq = np.r_[np.asarray(pre_window[1], dtype=item.dtype), seq]
        o = np.lexsort((pos, tok_t, tok_user))             # stable within user
        tok_user, tok_t, seq = tok_user[o], tok_t[o], seq[o]
    seq_key = tok_user * span + (tok_t - t0)               # (user, label_time) lexicographic
    end     = np.searchsorted(seq_key, user * span + (time - t0), side="left")  # known < t
    seg     = np.searchsorted(seq_key, user * span, side="left")              # user's first
    start   = np.maximum(seg, end - max_len)
    return seq, start.astype(np.int64), end.astype(np.int64)


def count_events_before(ev_group: np.ndarray, ev_time: np.ndarray,
                        q_group: np.ndarray, q_time: np.ndarray) -> np.ndarray:
    """For each query: events of the same group with ev_time < q_time."""
    ev_group, ev_time = np.asarray(ev_group, np.int64), np.asarray(ev_time, np.int64)
    q_group, q_time = np.asarray(q_group, np.int64), np.asarray(q_time, np.int64)
    if len(ev_group) == 0:
        return np.zeros(len(q_group), np.int64)
    t0 = int(min(ev_time.min(), q_time.min()))
    span = int(max(ev_time.max(), q_time.max()) - t0) + 1
    keys = np.sort(ev_group * span + (ev_time - t0))
    return (np.searchsorted(keys, q_group * span + (q_time - t0), "left")
            - np.searchsorted(keys, q_group * span, "left")).astype(np.int64)


def derive_sessions(user: np.ndarray, time: np.ndarray, gap_ms: int = 30 * 60 * 1000) -> np.ndarray:
    """Session id per row: a user's activity split where consecutive rows are > gap_ms apart."""
    user, time = np.asarray(user, np.int64), np.asarray(time, np.int64)
    o = np.lexsort((np.arange(len(user)), time, user))
    u, t = user[o], time[o]
    new = np.r_[True, (u[1:] != u[:-1]) | (t[1:] - t[:-1] > gap_ms)]
    out = np.empty(len(user), np.int64)
    out[o] = np.cumsum(new) - 1
    return out


# ── Label-timed item statistics (MIND, ZhihuRec) ──────────────────────────────

ITEM_STAT_COLS = ["show_cnt_log", "click_cnt_log", "ctr_known"]


def asof_item_label_stats(item: np.ndarray, time: np.ndarray, label: np.ndarray,
                          label_time: np.ndarray) -> np.ndarray:
    """
    Per-row ITEM_STAT_COLS, float32 [N, 3]:
      show_cnt_log   log1p(impressions of the item with time < t)            (exposure)
      click_cnt_log  log1p(clicks of the item with label_time < t)            (label-derived)
      ctr_known      clicks / labelled impressions, both with label_time < t (0 if none)
    """
    item, time = np.asarray(item, np.int64), np.asarray(time, np.int64)
    lt, clicked = np.asarray(label_time, np.int64), np.asarray(label) == 1
    shows = asof_group_count(item, time)
    known = count_events_before(item, lt, item, time)
    clicks = count_events_before(item[clicked], lt[clicked], item, time)
    ctr = np.where(known > 0, clicks / np.maximum(known, 1), 0.0)
    return np.stack([np.log1p(shows), np.log1p(clicks), ctr], axis=1).astype(np.float32)


def item_label_stats_before(item: np.ndarray, time: np.ndarray, label: np.ndarray,
                            label_time: np.ndarray, n_items: int, cutoff: int) -> np.ndarray:
    """Snapshot version of ITEM_STAT_COLS per item node: rows with (label_)time < cutoff."""
    item = np.asarray(item, np.int64)
    shows = np.bincount(item[np.asarray(time) < cutoff], minlength=n_items)
    k = np.asarray(label_time) < cutoff
    known = np.bincount(item[k], minlength=n_items)
    clicks = np.bincount(item[k & (np.asarray(label) == 1)], minlength=n_items)
    ctr = np.where(known > 0, clicks / np.maximum(known, 1), 0.0)
    return np.stack([np.log1p(shows), np.log1p(clicks), ctr], axis=1).astype(np.float32)


# ── Point-in-time video statistics ────────────────────────────────────────────

def asof_video_statistics(frame: pd.DataFrame) -> np.ndarray:
    """
    Per-row VIDEO_STAT_COLS from the same video's interactions with a strictly
    earlier timestamp.  `frame` needs video_id, time_ms and FEEDBACK_COLS.
    Returns float32 [N, len(VIDEO_STAT_COLS)] aligned with `frame` rows.
    """
    df = frame[["video_id", "time_ms"] + FEEDBACK_COLS]
    g  = df.groupby(["video_id", "time_ms"], sort=True)
    agg = g.size().rename("show_cnt").to_frame()
    for out, src in STAT_SOURCES.items():
        agg[out] = g[src].sum()
    counts = (agg.groupby(level="video_id").cumsum() - agg).reset_index()   # exclusive

    stats = pd.DataFrame({"video_id": counts["video_id"], "time_ms": counts["time_ms"]})
    stats["global_cvr"] = (
        counts["valid_play_cnt"] / counts["show_cnt"].replace(0, np.nan)
    ).clip(0, 1).fillna(0)
    for col in ["show_cnt", "play_cnt", "like_cnt", "follow_cnt", "share_cnt", "comment_cnt"]:
        stats[f"{col}_log"] = np.log1p(counts[col])

    merged = df[["video_id", "time_ms"]].merge(stats, on=["video_id", "time_ms"], how="left")
    return merged[VIDEO_STAT_COLS].to_numpy(np.float32)


# ── Bucket keys (identical for every model) ───────────────────────────────────

def _count_before(video: np.ndarray, time: np.ndarray, cutoff: np.ndarray) -> np.ndarray:
    """For each row: interactions of the same video with time < cutoff[row]."""
    video, time = np.asarray(video, np.int64), np.asarray(time, np.int64)
    t0   = int(min(time.min(), np.min(cutoff)))
    span = int(max(time.max(), np.max(cutoff)) - t0) + 1
    keys = np.sort(video * span + (time - t0))
    return (np.searchsorted(keys, video * span + (np.asarray(cutoff, np.int64) - t0), "left")
            - np.searchsorted(keys, video * span, "left"))


def bucket_keys(
    video:        np.ndarray,
    time:         np.ndarray,
    hist_len:     np.ndarray,
    t_val:        int,
    snapshot_start: np.ndarray,
    max_len:      int,
) -> dict[str, np.ndarray]:
    """
    Per-row bucket labels:
      video_train_count : the video's training-window interactions (time < t_val) — 0 / 1-4 / >=5
      coldwarm          : Spec 01 — video interactions before the row's snapshot <= 1 ('cold')
      history_len       : user history length at the row — 0 / 1-9 / 10-49 / max_len
    """
    tc = _count_before(video, time, np.full(len(time), t_val))
    sc = _count_before(video, time, snapshot_start)
    hl = np.asarray(hist_len)
    return {
        "video_train_count": np.select([tc == 0, tc <= 4], ["0", "1-4"], ">=5"),
        "coldwarm":          np.where(sc <= 1, "cold", "warm"),
        "history_len":       np.select([hl == 0, hl <= 9, hl <= 49],
                                       ["0", "1-9", "10-49"], str(max_len)),
    }


# ── Training holdout ──────────────────────────────────────────────────────────

def holdout_rows(train_rows: np.ndarray, frac: float = 0.05, seed: int = 0) -> np.ndarray:
    """Deterministic in-period holdout: a random `frac` of training rows (sorted ids)."""
    train_rows = np.asarray(train_rows, np.int64)
    k = int(round(frac * len(train_rows)))
    return np.sort(np.random.default_rng(seed).choice(train_rows, size=k, replace=False))


# ── Evidence counts for fusion (spec 05) ──────────────────────────────────────

# The metadata neighbour types behind metadata_evidence are declared per dataset
# by the adapter (DatasetBundle.metadata_evidence_types, Framework/datasets/).


def asof_group_count(group: np.ndarray, time: np.ndarray) -> np.ndarray:
    """For each row: rows of the same group with a strictly earlier timestamp."""
    group, time = np.asarray(group, np.int64), np.asarray(time, np.int64)
    t0, span = int(time.min()), int(time.max() - time.min()) + 1
    keys = np.sort(group * span + (time - t0))
    return (np.searchsorted(keys, group * span + (time - t0), "left")
            - np.searchsorted(keys, group * span, "left")).astype(np.int64)


def metadata_evidence(item: np.ndarray, time: np.ndarray,
                      links: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    log1p of the as-of interaction count of each row's item's metadata neighbours,
    max over all neighbours in `links`.  Each link set is (item_idx, neighbour_idx)
    pairs, one per (item, neighbour); an item may have several neighbours (MIND
    entities) or none (anonymous Zhihu answer, NaN KuaiRand author).  A neighbour's
    count is the number of rows with a strictly earlier timestamp whose item links
    to it.  Items without a neighbour get 0: missing values never form a group.
    """
    item, time = np.asarray(item, np.int64), np.asarray(time, np.int64)
    n_items = int(item.max()) + 1 if len(item) else 0
    out = np.zeros(len(item), dtype=np.int64)
    for li, ln in links:
        li, ln = np.asarray(li, np.int64), np.asarray(ln, np.int64)
        keep = li < n_items
        li, ln = li[keep], ln[keep]
        if not len(li):
            continue
        o = np.argsort(li, kind="stable")
        li, ln = li[o], ln[o]
        deg = np.bincount(li, minlength=n_items)
        first = np.r_[0, np.cumsum(deg)[:-1]]
        rows = np.flatnonzero(deg[item] > 0)
        rdeg = deg[item[rows]]
        rep = np.repeat(rows, rdeg)                                  # sorted by row
        offs = np.arange(len(rep)) - np.repeat(np.cumsum(rdeg) - rdeg, rdeg)
        nb = ln[first[item[rep]] + offs]
        cnt = asof_group_count(nb, time[rep])
        seg = np.r_[0, np.cumsum(rdeg)[:-1]]
        out[rows] = np.maximum(out[rows], np.maximum.reduceat(cnt, seg))
    return np.log1p(out).astype(np.float32)


def view_evidence(item: np.ndarray, time: np.ndarray, hist_len: np.ndarray,
                  links: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    Per-row causal evidence for each view, float32 [N, 2]:
      n_G = log1p(item prior interactions) + metadata_evidence
      n_S = log1p(hist_len)                + log1p(item prior interactions)
    `hist_len` is the length of the history the sequence encoder attends over
    (end − start from click_history), so the two cannot drift apart.  Interaction
    counts are exposure counts over rows with a strictly earlier timestamp.
    """
    item_prior = np.log1p(asof_group_count(item, time))
    n_g = item_prior + metadata_evidence(item, time, links)
    n_s = np.log1p(np.asarray(hist_len, np.int64)) + item_prior
    return np.stack([n_g, n_s], axis=1).astype(np.float32)
