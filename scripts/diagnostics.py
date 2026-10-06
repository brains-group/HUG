"""
diagnostics.py — leak-or-drift checks (spec 04 W3.1)
-----------------------------------------------------
    python scripts/diagnostics.py --out runs/dev/diagnostics.json

Data-only checks on the canonical interaction table (no model needed):
  * share of (user, video) pairs that repeat within the training rows
  * label purity per video in training (row-weighted share of rows whose video's
    training clicks are all 0 or all 1)
  * one-feature AUCs on the training holdout and on val:
      - video_id target encoding fitted on training rows minus the holdout
        (unseen videos get the global click rate)
      - point-in-time global_cvr (features.asof_video_statistics)
The model-based part (holdout vs train vs val LogLoss) comes from each run's
final_metrics.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Framework"))

from data_loader import SPLIT_TRAIN, SPLIT_VAL, KuaiRandLoader  # noqa: E402
from features import VIDEO_STAT_COLS, asof_video_statistics, holdout_rows  # noqa: E402
from temporal import build_interactions  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(ROOT / "KuaiRand-1K" / "data"))
    p.add_argument("--out", required=True)
    args = p.parse_args()

    data  = KuaiRandLoader(args.data_dir, min_interactions=10).load()
    inter = build_interactions(data)
    train_all = inter.rows(SPLIT_TRAIN)
    hold = holdout_rows(train_all, 0.05, seed=0)
    fit  = np.setdiff1d(train_all, hold, assume_unique=True)
    val  = inter.rows(SPLIT_VAL)
    y = inter.label

    # repeat pairs among training rows
    n_v = int(inter.video.max()) + 1
    pair = inter.user[train_all] * n_v + inter.video[train_all]
    _, counts = np.unique(pair, return_counts=True)
    repeat_rows = float(counts[counts > 1].sum() / len(train_all))

    # label purity per video (fit rows)
    clicks = np.bincount(inter.video[fit], weights=y[fit], minlength=n_v)
    shows  = np.bincount(inter.video[fit], minlength=n_v)
    pure = (shows > 0) & ((clicks == 0) | (clicks == shows))
    purity_rows = float(shows[pure].sum() / shows.sum())
    purity_rows_multi = float(shows[pure & (shows > 1)].sum() / shows[shows > 1].sum())

    # one-feature AUCs
    gmean = float(y[fit].mean())
    te = np.where(shows > 0, clicks / np.maximum(shows, 1), gmean)
    cvr = asof_video_statistics(inter.frame)[:, VIDEO_STAT_COLS.index("global_cvr")]
    auc = lambda rows, s: float(roc_auc_score(y[rows], s[rows]))
    out = {
        "rows": {"train_fit": int(len(fit)), "holdout": int(len(hold)), "val": int(len(val))},
        "repeat_pair_row_share_train": repeat_rows,
        "label_pure_video_row_share_train": purity_rows,
        "label_pure_video_row_share_train_videos_with_2plus_rows": purity_rows_multi,
        "auc_video_target_encoding": {"holdout": auc(hold, te[inter.video]),
                                      "val": auc(val, te[inter.video])},
        "auc_global_cvr_asof": {"holdout": auc(hold, cvr), "val": auc(val, cvr)},
        "holdout_videos_seen_in_fit": float((shows[inter.video[hold]] > 0).mean()),
        "val_videos_seen_in_fit": float((shows[inter.video[val]] > 0).mean()),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
