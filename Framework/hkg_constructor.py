"""
Heterogeneous Knowledge Graph (HKG) bundle and time-filtered views
------------------------------------------------------------------
HKGBundle holds a timed heterogeneous graph (PyG HeteroData) plus two
filtered views.  Every behavioural edge carries an `edge_time`; a graph
built over the whole log must only reach a model through `snapshot_bundle`,
which keeps edges with edge_time < cutoff and swaps in item statistics
computed from the same window (see temporal.py).  Static metadata edges
carry no time and are always present.

The KuaiRand HKG builder (HKGConstructor, video_feature_matrix, …) lives in
datasets/kuairand_hkg.py and is re-exported here for existing callers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)

# Bump whenever HKGConstructor output changes; cached bundles must match
HKG_BUILD_VERSION = 2


@dataclass
class HKGBundle:
    """All graph views produced by HKGConstructor."""

    full_graph:       HeteroData   # complete HKG (all node/edge types)
    structural_graph: HeteroData   # subgraph for structural GNN (R-GCN / HAN)
    sequential_graph: HeteroData   # subgraph for sequential GNN (SR-GNN)

    # Convenient metadata
    n_users:      int = 0
    n_videos:     int = 0
    n_sessions:   int = 0
    n_authors:    int = 0
    n_categories: int = 0
    # Snapshot views: behaviour restricted to edge_time < cutoff_ms (None = timed
    # graph over the whole log, never fed to a model directly)
    cutoff_ms:    int | None = None
    build_version: int = HKG_BUILD_VERSION

    def summary(self) -> str:  # pragma: no cover
        lines = [
            "HKGBundle",
            f"  users      : {self.n_users}",
            f"  videos     : {self.n_videos}",
            f"  sessions   : {self.n_sessions}",
            f"  authors    : {self.n_authors}",
            f"  categories : {self.n_categories}",
            f"  full graph node types : {self.full_graph.node_types}",
            f"  full graph edge types : {self.full_graph.edge_types}",
        ]
        return "\n".join(lines)


# ── Time-filtered views ───────────────────────────────────────────────────────

def _snapshot_graph(g: HeteroData, cutoff_ms: int, video_x: Tensor | None) -> HeteroData:
    out = HeteroData()
    for nt in g.node_types:
        for key, val in g[nt].items():
            out[nt][key] = val
        if nt == "video" and video_x is not None:
            out[nt].x = video_x
    for et in g.edge_types:
        store = g[et]
        if "edge_time" in store:
            keep = store.edge_time < cutoff_ms
            n = len(keep)
            for key, val in store.items():         # every per-edge attribute
                if key == "edge_index":
                    out[et][key] = val[:, keep]
                elif torch.is_tensor(val) and val.dim() >= 1 and val.shape[0] == n:
                    out[et][key] = val[keep]
        else:                                    # static metadata edge
            for key, val in store.items():
                out[et][key] = val
    return out


def snapshot_bundle(bundle: HKGBundle, cutoff_ms: int,
                    video_x: Tensor | None = None) -> HKGBundle:
    """
    View of `bundle` containing only behaviour observed before `cutoff_ms`:
    every timed edge with edge_time >= cutoff_ms is dropped, and video node
    features are replaced by `video_x` (statistics from the same window).
    Node sets and ids are unchanged.
    """
    return HKGBundle(
        full_graph       = _snapshot_graph(bundle.full_graph,       cutoff_ms, video_x),
        structural_graph = _snapshot_graph(bundle.structural_graph, cutoff_ms, video_x),
        sequential_graph = _snapshot_graph(bundle.sequential_graph, cutoff_ms, video_x),
        n_users          = bundle.n_users,
        n_videos         = bundle.n_videos,
        n_sessions       = bundle.n_sessions,
        n_authors        = bundle.n_authors,
        n_categories     = bundle.n_categories,
        cutoff_ms        = int(cutoff_ms),
    )


_MOVED = ("HKGConstructor", "FEEDBACK_EDGE_TYPES", "RANDOM_POLICY_RATE", "IPS_STANDARD_WEIGHT",
          "N_ONEHOT_USER_FEATS", "VIDEO_FEATURE_COLS", "video_feature_matrix")


def __getattr__(name):
    # KuaiRand's builder, re-exported lazily (it imports this module for HKGBundle)
    if name in _MOVED:
        from datasets import kuairand_hkg
        return getattr(kuairand_hkg, name)
    raise AttributeError(name)
