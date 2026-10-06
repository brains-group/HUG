"""
Causal (point-in-time) inputs for HUG
--------------------------------------
Every interaction at time t is scored from inputs built only from interactions
with time_ms < t:

  * Graph snapshots.  The timeline is cut at regular boundaries (plus the
    val/test cutoffs).  An interaction in [b_k, b_{k+1}) is encoded from
    snapshot k: the HKG restricted to edges with edge_time < b_k, with video
    statistics computed from the same window.  b_k <= t, so the snapshot only
    contains the past.

  * Session prefixes.  The session embedding of an interaction pools only the
    items the user watched earlier in the same session (strictly smaller
    time_ms), never the target item or anything after it.

Model weights are still fit on the whole training split — all of which
precedes validation and test — as in any offline evaluation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from data_loader import (
    SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL,
    KuaiRandData, assign_split, chronological_cutoffs, compute_video_statistics,
)
from hkg_constructor import (
    VIDEO_FEATURE_COLS, HKGBundle, snapshot_bundle, video_feature_matrix,
)

logger = logging.getLogger(__name__)

HOUR_MS = 3_600_000


# ── Interaction table ─────────────────────────────────────────────────────────

@dataclass
class Interactions:
    """All interactions in global time order, with causal session prefixes."""

    user:      np.ndarray   # [N] user node index
    video:     np.ndarray   # [N] video node index
    time:      np.ndarray   # [N] time_ms
    ips:       np.ndarray   # [N] ips weight
    label:     np.ndarray   # [N] is_click
    split:     np.ndarray   # [N] SPLIT_TRAIN / SPLIT_VAL / SPLIT_TEST
    pre_start: np.ndarray   # [N] prefix = seq_items[pre_start:pre_end]
    pre_end:   np.ndarray   # [N]
    seq_items: np.ndarray   # [N] video idx ordered by (session, time)
    t_val:     int
    t_test:    int
    max_prefix: int

    def rows(self, split: int) -> np.ndarray:
        return np.flatnonzero(self.split == split)


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
    df = df[["user_id", "video_id", "session_id", "time_ms", "is_rand", "is_click"]]
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
                        pre_start, pre_end, video[order], t_val, t_test, max_prefix)


# ── Snapshot schedule ─────────────────────────────────────────────────────────

def snapshot_boundaries(time: np.ndarray, t_val: int, t_test: int,
                        period_hours: float) -> np.ndarray:
    """
    Boundaries b_0 < b_1 < … : a regular grid from the first interaction,
    with t_val and t_test inserted so no snapshot period spans two splits.
    """
    t0, t1 = int(time.min()), int(time.max())
    step   = int(period_hours * HOUR_MS)
    grid   = np.arange(t0, t1 + 1, step, dtype=np.int64)
    return np.unique(np.r_[grid, t_val, t_test]).astype(np.int64)


def assign_snapshots(time: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Snapshot k for each time t with boundaries[k] <= t < boundaries[k+1]."""
    return np.searchsorted(boundaries, time, side="right") - 1


class SnapshotStore:
    """
    Per-snapshot inputs, precomputed once: video node features from the
    snapshot's window and the (sampled) relation edge index for the encoder.
    Graph views are cut from the timed bundle on demand.
    """

    def __init__(
        self,
        bundle:      HKGBundle,
        data:        KuaiRandData,
        boundaries:  np.ndarray,
        rel_builder: Callable[[HKGBundle], tuple[Tensor, Tensor]],
        needed:      set[int] | None = None,
        drop_edges:  bool = False,
    ) -> None:
        self.bundle     = bundle
        self.boundaries = boundaries
        self.drop_edges = drop_edges        # features-only control: views carry no edges
        self.encode_log: list[dict] = []
        self.video_x:   dict[int, Tensor] = {}
        self.rel:       dict[int, tuple[Tensor, Tensor]] = {}

        vb = data.video_basic[data.video_basic["video_id"].isin(data.video_id_map)].copy()
        vb["node_idx"] = vb["video_id"].map(data.video_id_map)
        vb = vb.sort_values("node_idx")

        log  = data.log_combined[["video_id", "time_ms", "long_view", "is_click",
                                  "is_like", "is_follow", "is_forward", "is_comment"]]
        keys = sorted(needed) if needed is not None else range(len(boundaries))
        for k in keys:
            cutoff = int(boundaries[k])
            stats  = compute_video_statistics(log[log["time_ms"] < cutoff])
            self.video_x[k] = torch.from_numpy(video_feature_matrix(vb, stats))
            self.rel[k] = rel_builder(self.view(k))
        logger.info("Prepared %d graph snapshots", len(self.video_x))

    def view(self, k: int) -> HKGBundle:
        """Bundle restricted to behaviour before boundaries[k]."""
        v = snapshot_bundle(self.bundle, int(self.boundaries[k]), self.video_x.get(k))
        if self.drop_edges:
            for g in (v.full_graph, v.structural_graph, v.sequential_graph):
                for et in g.edge_types:
                    g[et].edge_index = g[et].edge_index[:, :0]
        return v

    _SHOW_COL = VIDEO_FEATURE_COLS.index("show_cnt_log")
    _COLD_MAX = float(np.log1p(1.0))                 # show_cnt <= 1

    def is_cold(self, k: int, video_idx: Tensor) -> np.ndarray:
        """True where the video had 0–1 interactions before snapshot k."""
        return (self.video_x[k][video_idx, self._SHOW_COL] <= self._COLD_MAX + 1e-6).numpy()


# ── Batching ──────────────────────────────────────────────────────────────────

class PrefixBatcher:
    """Turns row indices into model inputs, including padded session prefixes."""

    def __init__(self, inter: Interactions) -> None:
        self.user      = torch.from_numpy(inter.user)
        self.video     = torch.from_numpy(inter.video)
        self.ips       = torch.from_numpy(inter.ips)
        self.label     = torch.from_numpy(inter.label)
        self.pre_start = torch.from_numpy(inter.pre_start)
        self.pre_end   = torch.from_numpy(inter.pre_end)
        self.seq_items = torch.from_numpy(inter.seq_items)
        self.offsets   = torch.arange(-inter.max_prefix, 0)       # [L]

    def __call__(self, rows: np.ndarray) -> dict[str, Tensor]:
        idx  = torch.from_numpy(np.asarray(rows, dtype=np.int64))
        pos  = self.pre_end[idx, None] + self.offsets               # [B, L] right-aligned
        keep = pos >= self.pre_start[idx, None]
        items = self.seq_items[pos.clamp(min=0)]
        items[~keep] = -1
        return {
            "user_idx":     self.user[idx],
            "video_idx":    self.video[idx],
            "prefix_items": items,
            "ips_weight":   self.ips[idx],
            "label":        self.label[idx],
        }
