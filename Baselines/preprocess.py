"""
preprocess.py
=============
Builds the FuxiCTR train / valid / test / holdout files for the baselines
from KuaiRand-1K.  It contains no feature logic of its own: every value comes
from Framework/features.py over the canonical interaction table
(temporal.build_interactions), exactly as HUG computes it.

  * rows, split and cutoffs: build_interactions (shared with HUG)
  * behaviour sequence:      features.click_history (last 50 clicks, strictly earlier)
  * item statistics:         features.asof_video_statistics (strictly earlier rows)
  * context / categoricals:  features.hour_of_day, tab, video/user_categoricals
  * bucket keys:             features.bucket_keys (identical to HUG's)
  * training holdout:        features.holdout_rows (5 %, seed 0) — written to
                             holdout.csv and excluded from train.csv

Every file carries `row_id` (position in the canonical table) and the bucket
columns; FuxiCTR ignores columns that are not in feature_cols.

Usage:
    python preprocess.py --data-dir ../KuaiRand-1K/data

Output (./data/processed/kuairand_1k_csv/):
    train.csv  valid.csv  test.csv  holdout.csv  split_stats.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "Framework"))

from data_loader import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL, KuaiRandLoader  # noqa: E402
from features import (  # noqa: E402
    USER_CAT_COLS, VIDEO_CAT_COLS, VIDEO_STAT_COLS, asof_video_statistics, bucket_keys,
    click_history, holdout_rows, hour_of_day, user_categoricals, video_categoricals,
)
from temporal import assign_snapshots, build_interactions, snapshot_boundaries  # noqa: E402

MAX_SEQ_LEN    = 50
SNAPSHOT_HOURS = 24.0
HOLDOUT_FRAC   = 0.05
BUCKET_COLS    = ["bucket_video_train_count", "bucket_coldwarm", "bucket_history_len"]
# FuxiCTR dataset dirs built from these CSVs (deleted so they are rebuilt)
FUXICTR_DATASET_PREFIX = "kuairand_1k"


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def baseline_frame(data, inter, max_len: int = MAX_SEQ_LEN,
                   snapshot_hours: float = SNAPSHOT_HOURS) -> pd.DataFrame:
    """All rows in canonical order with every baseline column."""
    fr = inter.frame
    seq, hs, he = click_history(inter.user, inter.time, fr["video_id"].to_numpy(),
                                inter.label, max_len)
    seq_str = seq.astype(str)
    hist = [" ".join(seq_str[a:b]) for a, b in zip(hs, he)]

    boundaries = snapshot_boundaries(inter.time, inter.t_val, inter.t_test, snapshot_hours)
    snap_start = boundaries[assign_snapshots(inter.time, boundaries)]
    bk = bucket_keys(inter.video, inter.time, he - hs, inter.t_val, snap_start, max_len)

    df = pd.DataFrame({
        "row_id":   np.arange(len(fr)),
        "user_id":  fr["user_id"].to_numpy(),
        "video_id": fr["video_id"].to_numpy(),
        "is_click": inter.label.astype(int),
        "hist_video_ids": hist,
        "tab":  fr["tab"].fillna(0).astype(int).to_numpy(),
        "hour": hour_of_day(inter.time),
    })
    df = df.merge(user_categoricals(data.user_features), on="user_id", how="left")
    vb = video_categoricals(data.video_basic).merge(
        data.video_basic[["video_id", "duration_s"]], on="video_id", how="left")
    df = df.merge(vb, on="video_id", how="left")
    df[VIDEO_STAT_COLS] = asof_video_statistics(fr)
    df["bucket_video_train_count"] = bk["video_train_count"]
    df["bucket_coldwarm"]          = bk["coldwarm"]
    df["bucket_history_len"]       = bk["history_len"]
    df["split"] = inter.split
    return df.sort_values("row_id").reset_index(drop=True)


COLUMNS = (["row_id", "user_id", "video_id", "is_click", "hist_video_ids", "tab", "hour"]
           + USER_CAT_COLS + VIDEO_CAT_COLS + ["duration_s"] + VIDEO_STAT_COLS + BUCKET_COLS)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",   default=str(ROOT.parent / "KuaiRand-1K" / "data"))
    p.add_argument("--out-dir",    default=str(ROOT / "data" / "processed" / "kuairand_1k_csv"))
    p.add_argument("--min-interactions", type=int,   default=10)
    p.add_argument("--val-ratio",        type=float, default=0.1)
    p.add_argument("--test-ratio",       type=float, default=0.2)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    processed_root = out_dir.parent
    # FuxiCTR caches transformed data per dataset id; stale caches would ignore new CSVs
    for d in processed_root.glob(f"{FUXICTR_DATASET_PREFIX}*"):
        if d.is_dir() and d != out_dir:
            shutil.rmtree(d)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    log("Loading KuaiRand with the HUG loader …")
    data = KuaiRandLoader(args.data_dir, min_interactions=args.min_interactions,
                          filter_ads=True).load()
    inter = build_interactions(data, args.val_ratio, args.test_ratio, max_prefix=MAX_SEQ_LEN)
    log(f"Cutoffs  t_val={inter.t_val}  t_test={inter.t_test}")

    log("Building shared per-row features …")
    df = baseline_frame(data, inter)

    holdout = holdout_rows(inter.rows(SPLIT_TRAIN), HOLDOUT_FRAC, seed=0)
    is_hold = np.zeros(len(df), dtype=bool)
    is_hold[holdout] = True

    parts = {
        "train":   (df["split"] == SPLIT_TRAIN) & ~is_hold,
        "holdout": is_hold,
        "valid":   df["split"] == SPLIT_VAL,
        "test":    df["split"] == SPLIT_TEST,
    }
    stats = {"t_val": int(inter.t_val), "t_test": int(inter.t_test), "max_seq_len": MAX_SEQ_LEN,
             "holdout_frac": HOLDOUT_FRAC, "splits": {}}
    for name, mask in parts.items():
        part = df.loc[mask, COLUMNS]
        path = out_dir / f"{name}.csv"
        part.to_csv(path, index=False)
        stats["splits"][name] = {"rows": int(len(part)), "pos_rate": float(part["is_click"].mean()),
                                 "users": int(part["user_id"].nunique()),
                                 "videos": int(part["video_id"].nunique())}
        log(f"  {name:7s}: {len(part):>10,} rows  pos_rate={part['is_click'].mean():.4f}  → {path}")

    (out_dir / "split_stats.json").write_text(json.dumps(stats, indent=2))
    log(f"Split summary → {out_dir / 'split_stats.json'}")


if __name__ == "__main__":
    main()
