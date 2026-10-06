"""
KuaiRand column registries and KuaiRand-only per-row features (spec 06 A10).
Moved out of features.py so the generic modules carry no dataset column names;
features.py re-exports these names for the KuaiRand pipeline and the baselines.
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


# ── Categorical tables ────────────────────────────────────────────────────────

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
