"""
preprocess.py
=============
Builds the FuxiCTR train / valid / test files for the TransAct and WuKong
baselines from KuaiRand-1K.

Rows, split and item statistics are produced by the same code HUG uses
(Framework/data_loader.py), so every model is trained and evaluated on
identical interactions:

  * KuaiRandLoader with the same filters as Framework/main.py
    (both standard logs + the random-policy log, AD videos removed,
    min_interactions=10)
  * global chronological split via chronological_cutoffs / assign_split
    (train < t_val <= valid < t_test <= test)
  * per-video popularity statistics computed *as of each row* — only from
    that video's interactions with a strictly earlier timestamp — never from
    video_features_statistic_*.csv.  (Aggregating over the whole training
    window would put each training row's own click into its features.)

Behaviour sequence: for each interaction, the user's most recently *clicked*
videos with a strictly earlier timestamp (max MAX_SEQ_LEN, oldest first).
Only past behaviour is used, so the sequence never contains the row's own
label or anything after it.

Usage:
    python preprocess.py --data-dir ../KuaiRand-1K/data

Output (./data/processed/kuairand_1k/):
    train.csv  valid.csv  test.csv  split_stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "Framework"))

from data_loader import (  # noqa: E402
    SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL,
    KuaiRandLoader, assign_split, chronological_cutoffs,
)

MAX_SEQ_LEN = 50
SPLIT_NAMES = {SPLIT_TRAIN: "train", SPLIT_VAL: "valid", SPLIT_TEST: "test"}

USER_CAT_COLS = [
    "user_active_degree", "is_lowactive_period", "is_live_streamer",
    "is_video_author", "follow_user_num_range", "fans_user_num_range",
    "friend_user_num_range", "register_days_range",
] + [f"onehot_feat{i}" for i in range(18)]

VIDEO_CAT_COLS = ["author_id", "music_id", "music_type", "upload_type", "primary_tag"]

# Same item statistics HUG's video nodes carry, but point-in-time per row
VIDEO_STAT_COLS = [
    "global_cvr", "show_cnt_log", "play_cnt_log", "like_cnt_log",
    "follow_cnt_log", "share_cnt_log", "comment_cnt_log",
]


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def click_history(df: pd.DataFrame, max_len: int) -> list[str]:
    """
    Space-joined ids of the user's last `max_len` clicked videos strictly
    before each row.  `df` must be sorted by (user_id, time_ms).
    """
    users  = df["user_id"].to_numpy()
    times  = df["time_ms"].to_numpy()
    vids   = df["video_id"].astype(str).to_numpy()
    clicks = df["is_click"].to_numpy()

    out = [""] * len(df)
    hist: deque[str] = deque(maxlen=max_len)
    i, n = 0, len(df)
    while i < n:
        if i == 0 or users[i] != users[i - 1]:
            hist.clear()
        # Rows sharing (user, timestamp) all see the same history; their own
        # clicks are appended only after the whole block is emitted.
        j = i
        while j < n and users[j] == users[i] and times[j] == times[i]:
            j += 1
        h = " ".join(hist)
        for k in range(i, j):
            out[k] = h
        for k in range(i, j):
            if clicks[k] == 1:
                hist.append(vids[k])
        i = j
    return out


# count column → source flag in the log (show_cnt counts rows)
STAT_SOURCES = {
    "valid_play_cnt": "long_view", "play_cnt": "is_click", "like_cnt": "is_like",
    "follow_cnt": "is_follow", "share_cnt": "is_forward", "comment_cnt": "is_comment",
}


def asof_video_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-row video statistics from interactions with the same video and a
    strictly earlier timestamp (rows sharing a timestamp do not see each other).
    Same definitions as data_loader.compute_video_statistics.
    """
    g = df.groupby(["video_id", "time_ms"], sort=True)
    agg = g.size().rename("show_cnt").to_frame()
    for out, src in STAT_SOURCES.items():
        agg[out] = g[src].sum()
    # exclusive cumulative sum within each video
    counts = agg.groupby(level="video_id").cumsum() - agg
    counts = counts.reset_index()

    counts["global_cvr"] = (
        counts["valid_play_cnt"] / counts["show_cnt"].replace(0, np.nan)
    ).clip(0, 1).fillna(0).astype(np.float32)
    for col in ["show_cnt", "play_cnt", "like_cnt", "follow_cnt", "share_cnt", "comment_cnt"]:
        counts[f"{col}_log"] = np.log1p(counts[col]).astype(np.float32)
    return counts[["video_id", "time_ms"] + VIDEO_STAT_COLS]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",   default=str(ROOT.parent / "KuaiRand-1K" / "data"))
    p.add_argument("--out-dir",    default=str(ROOT / "data" / "processed" / "kuairand_1k"))
    p.add_argument("--min-interactions", type=int,   default=10)
    p.add_argument("--val-ratio",        type=float, default=0.1)
    p.add_argument("--test-ratio",       type=float, default=0.2)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log("Loading KuaiRand with the HUG loader …")
    data = KuaiRandLoader(args.data_dir, min_interactions=args.min_interactions,
                          filter_ads=True).load()

    # Same row set as Framework/main.py:temporal_split
    df = data.session_map
    df = df[df["user_id"].isin(data.user_id_map) & df["video_id"].isin(data.video_id_map)]
    df = df[["user_id", "video_id", "time_ms", "tab", "is_click"]
            + [c for c in STAT_SOURCES.values() if c != "is_click"]].copy()

    t_val, t_test = chronological_cutoffs(df["time_ms"].to_numpy(),
                                          args.val_ratio, args.test_ratio)
    df["split"] = assign_split(df["time_ms"].to_numpy(), t_val, t_test)
    log(f"Cutoffs  t_val={t_val}  t_test={t_test}")

    log(f"Building click histories (max_len={MAX_SEQ_LEN}) …")
    df = df.sort_values(["user_id", "time_ms"], kind="stable").reset_index(drop=True)
    df["hist_video_ids"] = click_history(df, MAX_SEQ_LEN)

    # Context
    df["hour"] = ((df["time_ms"] // 3_600_000 + 8) % 24).astype(int)   # Beijing time

    log("Joining user features …")
    uf = data.user_features[["user_id"] + USER_CAT_COLS]
    df = df.merge(uf, on="user_id", how="left")

    log("Computing point-in-time video statistics …")
    df = df.merge(asof_video_statistics(df), on=["video_id", "time_ms"], how="left")

    log("Joining video features …")
    vb = data.video_basic[["video_id", "author_id", "music_id", "music_type",
                           "upload_type", "tag_list", "duration_s"]].copy()
    vb["primary_tag"] = vb["tag_list"].map(lambda t: t[0] if len(t) else -1)
    vb = vb.drop(columns="tag_list")
    df = df.merge(vb, on="video_id", how="left")

    cols = (["user_id", "video_id", "is_click", "hist_video_ids", "tab", "hour"]
            + USER_CAT_COLS + VIDEO_CAT_COLS + ["duration_s"] + VIDEO_STAT_COLS)

    stats_out: dict = {"t_val": t_val, "t_test": t_test,
                       "max_seq_len": MAX_SEQ_LEN, "splits": {}}
    for code, name in SPLIT_NAMES.items():
        part = df[df["split"] == code].sort_values("time_ms", kind="stable")[cols]
        path = out_dir / f"{name}.csv"
        part.to_csv(path, index=False)
        stats_out["splits"][name] = {
            "rows":     int(len(part)),
            "pos_rate": float(part["is_click"].mean()),
            "users":    int(part["user_id"].nunique()),
            "videos":   int(part["video_id"].nunique()),
        }
        log(f"  {name:5s}: {len(part):>10,} rows  pos_rate={part['is_click'].mean():.4f}  → {path}")

    (out_dir / "split_stats.json").write_text(json.dumps(stats_out, indent=2))
    log(f"Split summary → {out_dir / 'split_stats.json'}")


if __name__ == "__main__":
    main()
