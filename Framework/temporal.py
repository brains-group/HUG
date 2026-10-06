"""
Causal (point-in-time) inputs for HUG
--------------------------------------
Every interaction at time t is scored from inputs built only from interactions
with time < t:

  * Graph snapshots.  The timeline is cut at regular boundaries (plus the
    val/test cutoffs).  An interaction in [b_k, b_{k+1}) is encoded from
    snapshot k: the HKG restricted to edges with edge_time < b_k, with video
    statistics computed from the same window.  b_k <= t, so the snapshot only
    contains the past.

  * Session prefixes.  The session embedding of an interaction pools only the
    items the user watched earlier in the same session (strictly smaller
    time), never the target item or anything after it.

Model weights are still fit on the whole training split — all of which
precedes validation and test — as in any offline evaluation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from data_loader import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL  # noqa: F401  (re-exported)
from hkg_constructor import HKGBundle, snapshot_bundle

logger = logging.getLogger(__name__)

HOUR_MS = 3_600_000


# ── Interaction table ─────────────────────────────────────────────────────────

@dataclass
class Interactions:
    """All interactions in global time order, with causal session prefixes."""

    user:      np.ndarray   # [N] user node index
    video:     np.ndarray   # [N] video node index
    time:      np.ndarray   # [N] time (ms)
    ips:       np.ndarray   # [N] ips weight
    label:     np.ndarray   # [N] click label
    split:     np.ndarray   # [N] SPLIT_TRAIN / SPLIT_VAL / SPLIT_TEST
    pre_start: np.ndarray   # [N] prefix = seq_items[pre_start:pre_end]
    pre_end:   np.ndarray   # [N]
    seq_items: np.ndarray   # [N] video idx ordered by (session, time)
    t_val:     int
    t_test:    int
    max_prefix: int
    # Raw per-row columns in the same row order (row id = position), from the
    # adapter (dataset-specific; generic code reads only the arrays above)
    frame:     pd.DataFrame | None = None
    # Earliest time each row's label is known (spec 06 B); None = time
    label_time: np.ndarray | None = None

    def rows(self, split: int) -> np.ndarray:
        return np.flatnonzero(self.split == split)


def build_interactions(data, val_ratio: float = 0.1, test_ratio: float = 0.2,
                       max_prefix: int = 50) -> Interactions:
    """KuaiRand's canonical table (pre-adapter entry point; see datasets/kuairand.py)."""
    from datasets.kuairand import build_interactions as build
    return build(data, val_ratio, test_ratio, max_prefix)


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
    Per-snapshot inputs.  Video node features from each snapshot's window are
    precomputed once.  The encoder's relation edges are one master index
    (built once, uncapped) and a snapshot is the mask edge_time < boundary;
    no per-snapshot copies are stored.  Graph views are cut on demand.
    """

    def __init__(
        self,
        bundle:      HKGBundle,
        data,                                 # legacy: the raw data container
        boundaries:  np.ndarray,
        rel_master:  tuple[Tensor, Tensor, Tensor] | None,
        needed:      set[int] | None = None,
        drop_edges:  bool = False,
        item_features=None,
    ) -> None:
        self.bundle     = bundle
        self.boundaries = boundaries
        self.drop_edges = drop_edges        # features-only control: views carry no edges
        self.encode_log: list[dict] = []
        self.video_x:   dict[int, Tensor] = {}
        if rel_master is None:                    # no edges at all
            rel_master = (torch.zeros(2, 0, dtype=torch.long),
                          torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))
        self.rel_master = rel_master
        self._rel_dev: dict[torch.device, tuple[Tensor, Tensor, Tensor]] = {}

        if item_features is None:                 # legacy KuaiRand callers
            from datasets.kuairand import KuaiRandItemFeatures
            item_features = KuaiRandItemFeatures(data)
        keys = sorted(needed) if needed is not None else range(len(boundaries))
        for k in keys:
            self.video_x[k] = torch.from_numpy(item_features.at(int(boundaries[k])))
        logger.info("Prepared %d graph snapshots", len(self.video_x))

    def rel(self, k: int, device: torch.device | str = "cpu") -> tuple[Tensor, Tensor]:
        """Relation edges visible in snapshot k: master edges with edge_time < boundary."""
        device = torch.device(device)
        if device not in self._rel_dev:                       # one copy per device
            self._rel_dev[device] = tuple(t.to(device) for t in self.rel_master)
        ei, et, etime = self._rel_dev[device]
        keep = etime < int(self.boundaries[k])
        return ei[:, keep], et[keep]

    def view(self, k: int) -> HKGBundle:
        """Bundle restricted to behaviour before boundaries[k]."""
        v = snapshot_bundle(self.bundle, int(self.boundaries[k]), self.video_x.get(k))
        if self.drop_edges:
            for g in (v.full_graph, v.structural_graph, v.sequential_graph):
                for et in g.edge_types:
                    g[et].edge_index = g[et].edge_index[:, :0]
        return v

    _COLD_MAX = float(np.log1p(1.0))                 # show count <= 1

    def is_cold(self, k: int, video_idx: Tensor) -> np.ndarray:
        """True where the video had 0–1 interactions before snapshot k."""
        from datasets.kuairand_hkg import SHOW_COL       # legacy models (KuaiRand features)
        return (self.video_x[k][video_idx, SHOW_COL] <= self._COLD_MAX + 1e-6).numpy()


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
