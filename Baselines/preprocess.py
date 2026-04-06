"""
step1_preprocess.py
====================
Prepares KuaiRand-1K for FuxiCTR DIN / BST.

Expected raw files (place in ./data/raw/):
  - log_standard_4_08_to_4_21_1k.csv   (interaction log)
  - user_features_1k.csv                      (user side features)
  - video_features_basic_1k.csv               (item side features)

Output (written to ./data/processed/):
  - train.csv
  - valid.csv
  - test.csv
  - dataset_stats.txt   (label distribution summary)
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

# ── paths ──────────────────────────────────────────────────────────────────
RAW_DIR  = "../KuaiRand-1K/data"
OUT_DIR  = "./data/processed/kuairand_1k/"
os.makedirs(OUT_DIR, exist_ok=True)

# Max history length fed to DIN / BST
MAX_SEQ_LEN = 50

# Chronological split ratios
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1   # remaining 0.1 → test

print(f"[{datetime.now():%H:%M:%S}] Loading interaction log …")
log = pd.read_csv(
    os.path.join(RAW_DIR, "log_standard_4_08_to_4_21_1k.csv"),
    dtype={
        "user_id":   str,
        "video_id":  str,
        "is_click":  float,
        "is_like":   float,
        "is_follow": float,
        "is_forward":float,
        "is_hate":   float,
        "long_view": float,
        "play_time_ms":   float,
        "duration_ms":    float,
        "time_ms":      float,
        "is_random":      float,
    },
    low_memory=False,
)

# ── sort chronologically within each user ──────────────────────────────────
print(f"[{datetime.now():%H:%M:%S}] Sorting by user × time_ms …")
log = log.sort_values(["user_id", "time_ms"]).reset_index(drop=True)

# ── build history sequence column ─────────────────────────────────────────
print(f"[{datetime.now():%H:%M:%S}] Building behaviour sequences (max_len={MAX_SEQ_LEN}) …")

def build_hist(group):
    vids = group["video_id"].tolist()
    hists = []
    for i in range(len(vids)):
        window = vids[max(0, i - MAX_SEQ_LEN): i]
        hists.append(" ".join(window) if window else "0")   # "0" = empty pad token
    group = group.copy()
    group["hist_video_ids"] = hists
    return group

log = log.groupby("user_id", group_keys=False).apply(build_hist)

# Drop rows where history is empty (first interaction per user)
log = log[log["hist_video_ids"] != "0"].copy()

# ── merge side features ───────────────────────────────────────────────────
print(f"[{datetime.now():%H:%M:%S}] Merging user features …")
user_feat = pd.read_csv(
    os.path.join(RAW_DIR, "user_features_1k.csv"),
    dtype=str, low_memory=False,
)
# normalise column name if needed
if "user_id" not in user_feat.columns and "user_id " in user_feat.columns:
    user_feat.rename(columns={"user_id ": "user_id"}, inplace=True)

log = log.merge(user_feat, on="user_id", how="left")

print(f"[{datetime.now():%H:%M:%S}] Merging video features …")
item_feat = pd.read_csv(
    os.path.join(RAW_DIR, "video_features_basic_1k.csv"),
    dtype=str, low_memory=False,
)
if "video_id" not in item_feat.columns and "video_id " in item_feat.columns:
    item_feat.rename(columns={"video_id ": "video_id"}, inplace=True)

# Keep numeric duration separately before dtype=str merge
if "duration_ms" in item_feat.columns:
    item_feat["duration_ms"] = pd.to_numeric(item_feat["duration_ms"], errors="coerce")

log = log.merge(item_feat, on="video_id", how="left", suffixes=("", "_item"))

# ── fill missing categoricals ─────────────────────────────────────────────
cat_cols = [
    "user_active_degree", "follow_user_num_range", "fans_user_num_range",
    "friend_user_num_range", "register_days_range",
    "author_id", "music_id", "upload_type", "server_width", "server_height",
]
for c in cat_cols:
    if c in log.columns:
        log[c] = log[c].fillna("unknown")

# Numeric: fill duration_ms with median
if "duration_ms" in log.columns:
    log["duration_ms"] = pd.to_numeric(log["duration_ms"], errors="coerce")
    log["duration_ms"].fillna(log["duration_ms"].median(), inplace=True)
else:
    log["duration_ms"] = 0.0

# ── normalise time_ms to [0,1] ──────────────────────────────────────────
ts = pd.to_numeric(log["time_ms"], errors="coerce")
ts_min, ts_max = ts.min(), ts.max()
log["timestamp_norm"] = ((ts - ts_min) / (ts_max - ts_min + 1e-9)).fillna(0.0)

# ── select final columns ──────────────────────────────────────────────────
keep = [
    # identifiers / target
    "user_id", "video_id", "is_click",
    # behaviour sequence
    "hist_video_ids",
    # context
    "timestamp_norm",
    # user features
    "user_active_degree", "follow_user_num_range", "fans_user_num_range",
    "friend_user_num_range", "register_days_range",
    # item features
    "author_id", "music_id", "upload_type", "duration_ms",
]
keep = [c for c in keep if c in log.columns]
log = log[keep].copy()
log["is_click"] = pd.to_numeric(log["is_click"], errors="coerce").fillna(0).astype(int)

# ── chronological train/valid/test split ──────────────────────────────────
n = len(log)
t1 = int(n * TRAIN_RATIO)
t2 = int(n * (TRAIN_RATIO + VALID_RATIO))

train = log.iloc[:t1]
valid = log.iloc[t1:t2]
test  = log.iloc[t2:]

print(f"[{datetime.now():%H:%M:%S}] Writing splits …")
train.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
valid.to_csv(os.path.join(OUT_DIR, "valid.csv"), index=False)
test.to_csv( os.path.join(OUT_DIR, "test.csv"),  index=False)

# ── write stats ───────────────────────────────────────────────────────────
def split_stats(df, name):
    total = len(df)
    pos   = int(df["is_click"].sum())
    neg   = total - pos
    ctr   = pos / total if total else 0
    return (f"{name:>6s}: total={total:>8,d}  pos={pos:>7,d}  "
            f"neg={neg:>7,d}  CTR={ctr:.4f}")

stats_lines = [
    "=== KuaiRand-1K split statistics ===",
    split_stats(train, "train"),
    split_stats(valid, "valid"),
    split_stats(test,  "test"),
    split_stats(log,   "ALL"),
    f"\nColumns: {list(log.columns)}",
]
stats_path = os.path.join(OUT_DIR, "dataset_stats.txt")
with open(stats_path, "w") as f:
    f.write("\n".join(stats_lines) + "\n")

for line in stats_lines:
    print(line)

print(f"\n[{datetime.now():%H:%M:%S}] Done. Processed files saved to: {OUT_DIR}/")
