"""
Shared per-row features — the single source of truth for HUG and the baselines
-------------------------------------------------------------------------------
Everything here is computed on the canonical interaction table
(temporal.build_interactions: one row per interaction, globally time-sorted;
a row's *row id* is its position in that table).  Framework/hug.py and
Baselines/preprocess.py both call these functions, so the two pipelines feed
identical values for every row.

Causality: every per-row feature uses only interactions with a strictly
earlier timestamp.  Rows sharing a timestamp never see each other.

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

def hour_of_day(time_ms: np.ndarray) -> np.ndarray:
    """Hour of day in Beijing time (UTC+8)."""
    return ((np.asarray(time_ms) // HOUR_MS + 8) % 24).astype(np.int64)


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
    user:    np.ndarray,
    time:    np.ndarray,
    item:    np.ndarray,
    click:   np.ndarray,
    max_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For every row, the user's last `max_len` clicked items with a strictly
    earlier timestamp, oldest first.

    Returns (seq, start, end): row i's history is seq[start[i]:end[i]].
    `seq` holds `item` values of clicked rows ordered by (user, time, row).
    """
    user, time = np.asarray(user, np.int64), np.asarray(time, np.int64)
    clicked = np.flatnonzero(np.asarray(click) == 1)
    order   = clicked[np.lexsort((clicked, time[clicked], user[clicked]))]
    seq     = np.asarray(item)[order]

    t0, span = int(time.min()), int(time.max() - time.min()) + 1
    key      = user * span + (time - t0)                  # (user, time) lexicographic
    seq_key  = key[order]
    end      = np.searchsorted(seq_key, key, side="left")            # clicks with time < t
    seg      = np.searchsorted(seq_key, user * span, side="left")    # user's first click
    start    = np.maximum(seg, end - max_len)
    return seq, start.astype(np.int64), end.astype(np.int64)


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

# Primary metadata neighbour(s) of an item, declared per dataset.  The graph
# view's evidence uses the as-of interaction count of these neighbours (max
# over the list).  KuaiRand: the author.
METADATA_NEIGHBOURS = ["author_id"]


def asof_group_count(group: np.ndarray, time: np.ndarray) -> np.ndarray:
    """For each row: rows of the same group with a strictly earlier timestamp."""
    group, time = np.asarray(group, np.int64), np.asarray(time, np.int64)
    t0, span = int(time.min()), int(time.max() - time.min()) + 1
    keys = np.sort(group * span + (time - t0))
    return (np.searchsorted(keys, group * span + (time - t0), "left")
            - np.searchsorted(keys, group * span, "left")).astype(np.int64)


def metadata_evidence(frame: pd.DataFrame, item_meta: pd.DataFrame) -> np.ndarray:
    """
    log1p of the as-of interaction count of each row's item's primary metadata
    neighbour(s) (METADATA_NEIGHBOURS; max over them).  `item_meta` maps
    video_id to the neighbour columns.
    """
    m = frame[["video_id"]].merge(item_meta[["video_id"] + METADATA_NEIGHBOURS],
                                  on="video_id", how="left")
    t = frame["time_ms"].to_numpy(np.int64)
    out = np.zeros(len(frame), dtype=np.int64)
    for col in METADATA_NEIGHBOURS:
        codes = pd.factorize(m[col].fillna(-1))[0]
        out = np.maximum(out, asof_group_count(codes, t))
    return np.log1p(out).astype(np.float32)


def view_evidence(frame: pd.DataFrame, item_meta: pd.DataFrame, user: np.ndarray,
                  click: np.ndarray) -> np.ndarray:
    """
    Per-row causal evidence for each view, float32 [N, 2]:
      n_G = log1p(item prior interactions) + metadata_evidence
      n_S = log1p(user prior clicks)       + log1p(item prior interactions)
    Every count uses rows with a strictly earlier timestamp (ties excluded).
    """
    t = frame["time_ms"].to_numpy(np.int64)
    item_prior = np.log1p(asof_group_count(pd.factorize(frame["video_id"])[0], t))
    clicked = np.asarray(click) == 1
    # user prior clicks: as-of count over clicked rows only
    user = np.asarray(user, np.int64)
    span = int(t.max() - t.min()) + 1
    ck = np.sort(user[clicked] * span + (t[clicked] - t.min()))
    user_prior = (np.searchsorted(ck, user * span + (t - t.min()), "left")
                  - np.searchsorted(ck, user * span, "left"))
    n_g = item_prior + metadata_evidence(frame, item_meta)
    n_s = np.log1p(user_prior) + item_prior
    return np.stack([n_g, n_s], axis=1).astype(np.float32)
