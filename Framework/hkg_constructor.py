"""
Heterogeneous Knowledge Graph (HKG) Constructor
-------------------------------------------------
Takes the cleaned KuaiRandData container and builds a PyTorch Geometric
HeteroData object representing the full HKG, plus two filtered views:
  - structural_graph  : User / Video / Author / Category nodes + static edges
  - sequential_graph  : Session / Video nodes + ordered next_in_session edges

Node features are assembled from raw DataFrame columns and normalised.
Edge attributes carry IPS weights derived from the is_rand flag.

Every behavioural edge carries an `edge_time` (time_ms of the interaction it
came from; for next_in_session, the time of the later item).  A graph built
over the whole log must only reach a model through `snapshot_bundle`, which
keeps edges with edge_time < cutoff and swaps in item statistics computed
from the same window — see temporal.py.  Static metadata edges (made_by,
tagged_as) carry no time and are always present.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from data_loader import KuaiRandData

logger = logging.getLogger(__name__)

# ── IPS weight for random-policy edges ────────────────────────────────────────
# Edges from the random policy are unbiased → weight 1.0.
# Standard-policy edges are downweighted by a fixed propensity estimate.
# In practice this should be estimated empirically; 0.37 % replacement rate
# is the KuaiRand-reported figure.
RANDOM_POLICY_RATE = 0.0037
IPS_STANDARD_WEIGHT = float(1.0 - RANDOM_POLICY_RATE)

# Number of onehot encrypted user features
N_ONEHOT_USER_FEATS = 18

# Structural edge relation names (maps is_* columns → relation string)
FEEDBACK_EDGE_TYPES: list[tuple[str, str]] = [
    ("is_click",   "clicked"),
    ("is_like",    "liked"),
    ("is_follow",  "followed"),   # User → Video (author link handled separately)
    ("is_comment", "commented"),
    ("is_forward", "forwarded"),
    ("is_hate",    "hated"),
]


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
    # Snapshot views: behaviour restricted to time_ms < cutoff_ms (None = timed
    # graph over the whole log, never fed to a model directly)
    cutoff_ms:    int | None = None

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


class HKGConstructor:
    """
    Constructs a PyG HeteroData HKG from a KuaiRandData container.

    Parameters
    ----------
    data : KuaiRandData
        Populated data container from KuaiRandLoader.
    device : str
        Torch device to place tensors on ('cpu' or 'cuda').
    """

    def __init__(
        self,
        data: KuaiRandData,
        device: str = "cpu",
    ) -> None:
        self.data = data
        self.device = torch.device(device)

    # ── Public API ─────────────────────────────────────────────────────────────

    def build(self) -> HKGBundle:
        logger.info("Building HKG …")
        full = HeteroData()
        self._add_user_nodes(full)
        self._add_video_nodes(full)
        self._add_author_nodes(full)
        self._add_category_nodes(full)
        self._add_session_nodes(full)

        self._add_user_video_edges(full)
        self._add_user_author_edges(full)
        self._add_video_author_edges(full)
        self._add_video_category_edges(full)
        self._add_session_video_edges(full)

        structural = self._extract_structural_subgraph(full)
        sequential = self._extract_sequential_subgraph(full)

        bundle = HKGBundle(
            full_graph       = full,
            structural_graph = structural,
            sequential_graph = sequential,
            n_users          = full["user"].num_nodes,
            n_videos         = full["video"].num_nodes,
            n_sessions       = full["session"].num_nodes,
            n_authors        = full["author"].num_nodes,
            n_categories     = full["category"].num_nodes,
        )
        logger.info("HKG built:\n%s", bundle.summary())
        return bundle

    # ── Node builders ──────────────────────────────────────────────────────────

    def _add_user_nodes(self, graph: HeteroData) -> None:
        uid_map = self.data.user_id_map
        uf = self.data.user_features.copy()
        uf = uf[uf["user_id"].isin(uid_map)].copy()
        uf["node_idx"] = uf["user_id"].map(uid_map)
        uf = uf.sort_values("node_idx")

        # Continuous features
        cont_cols = [
            "activity_level",
            "follow_user_num_log", "fans_user_num_log",
            "friend_user_num_log", "register_days_log",
            "is_lowactive_period", "is_live_streamer", "is_video_author",
        ]
        cont_cols = [c for c in cont_cols if c in uf.columns]
        cont = uf[cont_cols].fillna(0).values.astype(np.float32)

        # Encrypted categorical features — embed dimensions set later in GNN
        onehot_cols = [f"onehot_feat{i}" for i in range(N_ONEHOT_USER_FEATS) if f"onehot_feat{i}" in uf.columns]
        onehot = uf[onehot_cols].fillna(0).values.astype(np.int64)

        graph["user"].x          = torch.tensor(cont, dtype=torch.float32).to(self.device)
        graph["user"].onehot     = torch.tensor(onehot, dtype=torch.long).to(self.device)
        graph["user"].num_nodes  = len(uf)
        logger.debug("User nodes: %d, feat_dim=%d", len(uf), cont.shape[1])

    def _add_video_nodes(self, graph: HeteroData) -> None:
        vid_map  = self.data.video_id_map
        vb = self.data.video_basic.copy()
        vb = vb[vb["video_id"].isin(vid_map)].copy()
        vb["node_idx"] = vb["video_id"].map(vid_map)
        vb = vb.sort_values("node_idx")

        cont = video_feature_matrix(vb, self.data.video_statistic)

        graph["video"].x         = torch.tensor(cont, dtype=torch.float32).to(self.device)
        graph["video"].video_ids = torch.tensor(
            [vid_map[v] for v in vb["video_id"]], dtype=torch.long
        ).to(self.device)
        graph["video"].num_nodes = len(vb)
        logger.debug("Video nodes: %d, feat_dim=%d", len(vb), cont.shape[1])

    def _add_author_nodes(self, graph: HeteroData) -> None:
        n = len(self.data.author_id_map)
        # Authors have no raw feature file → learned embeddings only
        graph["author"].num_nodes = n
        graph["author"].x = torch.zeros(n, 1, dtype=torch.float32).to(self.device)
        logger.debug("Author nodes: %d", n)

    def _add_category_nodes(self, graph: HeteroData) -> None:
        n = len(self.data.category_id_map)
        graph["category"].num_nodes = n
        graph["category"].x = torch.zeros(n, 1, dtype=torch.float32).to(self.device)
        logger.debug("Category nodes: %d", n)

    def _add_session_nodes(self, graph: HeteroData) -> None:
        n = self.data.n_sessions
        if n == 0 and not self.data.session_map.empty:
            n = int(self.data.session_map["session_id"].max()) + 1
        if n == 0:
            graph["session"].num_nodes = 0
            return
        graph["session"].num_nodes = n
        graph["session"].x = torch.zeros(n, 1, dtype=torch.float32).to(self.device)
        logger.debug("Session nodes: %d", n)

    # ── Edge builders ──────────────────────────────────────────────────────────

    def _add_user_video_edges(self, graph: HeteroData) -> None:
        uid_map = self.data.user_id_map
        vid_map = self.data.video_id_map
        log     = self.data.log_combined

        # Keep only rows where both user and video are in the id maps
        mask = log["user_id"].isin(uid_map) & log["video_id"].isin(vid_map)
        log  = log[mask].copy()

        u_idx = log["user_id"].map(uid_map).values.astype(np.int64)
        v_idx = log["video_id"].map(vid_map).values.astype(np.int64)
        edge_index = torch.tensor(np.stack([u_idx, v_idx]), dtype=torch.long).to(self.device)

        # IPS weights
        ips = np.where(log["is_rand"].values == 1, 1.0, IPS_STANDARD_WEIGHT).astype(np.float32)

        # Edge attribute vector: [play_ratio, long_view, tab, ips_weight]
        tab   = log["tab"].fillna(0).values.astype(np.float32) / 14.0  # normalise 0–1
        attrs = np.stack([
            log["play_ratio"].values,
            log["long_view"].fillna(0).values.astype(np.float32),
            tab,
            ips,
        ], axis=1)
        edge_attr = torch.tensor(attrs, dtype=torch.float32).to(self.device)
        edge_time = torch.tensor(log["time_ms"].values, dtype=torch.long).to(self.device)

        # Single combined interaction edge (is_rand flag preserved as attr)
        graph["user", "interacted", "video"].edge_index = edge_index
        graph["user", "interacted", "video"].edge_attr  = edge_attr
        graph["user", "interacted", "video"].edge_time  = edge_time
        graph["user", "interacted", "video"].is_rand    = torch.tensor(
            log["is_rand"].values, dtype=torch.bool
        ).to(self.device)

        # Typed edges per feedback signal
        for col, rel_name in FEEDBACK_EDGE_TYPES:
            if col not in log.columns:
                continue
            mask_rel  = log[col].values == 1
            ei_rel    = edge_index[:, mask_rel]
            ea_rel    = edge_attr[mask_rel]
            graph["user", rel_name, "video"].edge_index = ei_rel
            graph["user", rel_name, "video"].edge_attr  = ea_rel
            graph["user", rel_name, "video"].edge_time  = edge_time[mask_rel]

        logger.debug(
            "User→Video edges: %d total, %d clicked",
            edge_index.shape[1],
            int(log["is_click"].sum()),
        )

    def _add_user_author_edges(self, graph: HeteroData) -> None:
        """User → Author edges derived from is_follow interactions."""
        uid_map = self.data.user_id_map
        aut_map = self.data.author_id_map
        vid_map = self.data.video_id_map

        log = self.data.log_combined
        follow_mask = (
            (log["is_follow"] == 1) &
            log["user_id"].isin(uid_map) &
            log["video_id"].isin(vid_map)
        )
        follow_log = log[follow_mask].copy()

        # Map video → author
        v2a = self.data.video_basic.set_index("video_id")["author_id"].to_dict()
        follow_log["author_id"] = follow_log["video_id"].map(v2a)
        follow_log = follow_log.dropna(subset=["author_id"])
        follow_log = follow_log[follow_log["author_id"].isin(aut_map)]

        u_idx = follow_log["user_id"].map(uid_map).values.astype(np.int64)
        a_idx = follow_log["author_id"].map(aut_map).values.astype(np.int64)

        graph["user", "follows", "author"].edge_index = torch.tensor(
            np.stack([u_idx, a_idx]), dtype=torch.long
        ).to(self.device)
        graph["user", "follows", "author"].edge_time = torch.tensor(
            follow_log["time_ms"].values, dtype=torch.long
        ).to(self.device)

    def _add_video_author_edges(self, graph: HeteroData) -> None:
        vid_map = self.data.video_id_map
        aut_map = self.data.author_id_map

        vb = self.data.video_basic.copy()
        vb = vb[vb["video_id"].isin(vid_map) & vb["author_id"].isin(aut_map)]

        v_idx = vb["video_id"].map(vid_map).values.astype(np.int64)
        a_idx = vb["author_id"].map(aut_map).values.astype(np.int64)

        graph["video", "made_by", "author"].edge_index = torch.tensor(
            np.stack([v_idx, a_idx]), dtype=torch.long
        ).to(self.device)

    def _add_video_category_edges(self, graph: HeteroData) -> None:
        vid_map = self.data.video_id_map
        cat_map = self.data.category_id_map

        vt = self.data.video_basic[["video_id", "tag_list"]]
        vt = vt[vt["video_id"].isin(vid_map)].explode("tag_list").dropna()
        vt = vt[vt["tag_list"].isin(cat_map)]

        if len(vt):
            rows_v = vt["video_id"].map(vid_map).values.astype(np.int64)
            rows_c = vt["tag_list"].map(cat_map).values.astype(np.int64)
            graph["video", "tagged_as", "category"].edge_index = torch.tensor(
                np.stack([rows_v, rows_c]), dtype=torch.long
            ).to(self.device)

    def _add_session_video_edges(self, graph: HeteroData) -> None:
        """
        Directed next_in_session edges between consecutive items of each
        session.  edge_time is the time of the later item, so an edge only
        appears in snapshots where both of its interactions are in the past.
        """
        if self.data.session_map.empty:
            return

        vid_map = self.data.video_id_map
        sm = self.data.session_map
        sm = sm[sm["video_id"].isin(vid_map)][["session_id", "time_ms", "video_id", "tab"]]
        sm = sm.sort_values(["session_id", "time_ms"], kind="stable")

        sess = sm["session_id"].values
        same = sess[1:] == sess[:-1]
        if not same.any():
            return

        v_idx = sm["video_id"].map(vid_map).values.astype(np.int64)
        tab   = sm["tab"].fillna(0).values.astype(np.float32) / 14.0
        t     = sm["time_ms"].values

        ei = torch.tensor(np.stack([v_idx[:-1][same], v_idx[1:][same]]),
                          dtype=torch.long).to(self.device)
        ea = torch.tensor(tab[:-1][same], dtype=torch.float32).unsqueeze(1).to(self.device)

        et = ("video", "next_in_session", "video")
        graph[et].edge_index = ei
        graph[et].edge_attr  = ea
        graph[et].edge_time  = torch.tensor(t[1:][same], dtype=torch.long).to(self.device)
        graph[et].session_id = torch.tensor(sess[:-1][same], dtype=torch.long).to(self.device)

        logger.debug("Session→Video next_in_session edges: %d", ei.shape[1])

    # ── Subgraph extraction ────────────────────────────────────────────────────

    def _extract_structural_subgraph(self, full: HeteroData) -> HeteroData:
        """Filter the full HKG to structural edge types only."""
        structural = HeteroData()

        # Copy all node types (structural GNN needs user/video/author/category)
        for nt in ["user", "video", "author", "category"]:
            if nt in full.node_types:
                structural[nt].x         = full[nt].x
                structural[nt].num_nodes = full[nt].num_nodes
                if hasattr(full[nt], "onehot"):
                    structural[nt].onehot = full[nt].onehot

        # Copy structural edge types
        structural_edge_types = [
            ("user", "interacted",  "video"),
            ("user", "clicked",     "video"),
            ("user", "liked",       "video"),
            ("user", "follows",     "author"),
            ("user", "hated",       "video"),
            ("user", "commented",   "video"),
            ("user", "forwarded",   "video"),
            ("video", "made_by",    "author"),
            ("video", "tagged_as",  "category"),
        ]
        for et in structural_edge_types:
            if et in full.edge_types:
                structural[et].edge_index = full[et].edge_index
                if hasattr(full[et], "edge_attr"):
                    structural[et].edge_attr = full[et].edge_attr
                if hasattr(full[et], "is_rand"):
                    structural[et].is_rand = full[et].is_rand
                if hasattr(full[et], "edge_time"):
                    structural[et].edge_time = full[et].edge_time

        return structural

    def _extract_sequential_subgraph(self, full: HeteroData) -> HeteroData:
        """Filter the full HKG to sequential edge types only."""
        sequential = HeteroData()

        # Sequential GNN only needs video nodes (session is implicit in edges)
        for nt in ["video", "session"]:
            if nt in full.node_types:
                sequential[nt].x         = full[nt].x
                sequential[nt].num_nodes = full[nt].num_nodes

        seq_et = ("video", "next_in_session", "video")
        if seq_et in full.edge_types:
            sequential[seq_et].edge_index  = full[seq_et].edge_index
            sequential[seq_et].edge_attr   = full[seq_et].edge_attr
            sequential[seq_et].session_id  = full[seq_et].session_id
            sequential[seq_et].edge_time   = full[seq_et].edge_time

        return sequential


# ── Video node features ───────────────────────────────────────────────────────

VIDEO_FEATURE_COLS = [
    "duration_s", "aspect_ratio",
    "global_cvr",
    "show_cnt_log", "play_cnt_log", "like_cnt_log",
    "follow_cnt_log", "share_cnt_log", "comment_cnt_log",
]


def video_feature_matrix(video_basic: pd.DataFrame,
                         video_statistic: pd.DataFrame) -> np.ndarray:
    """
    Video node feature matrix, one row per video in `video_basic` order.
    Static metadata plus popularity statistics; videos absent from
    `video_statistic` (no interactions in its window) get zero statistics.
    """
    vb = video_basic[["video_id", "duration_s", "aspect_ratio"]].copy()
    if not video_statistic.empty:
        vb = vb.merge(video_statistic, on="video_id", how="left")
    for c in VIDEO_FEATURE_COLS:
        if c not in vb.columns:
            vb[c] = 0.0
    return vb[VIDEO_FEATURE_COLS].fillna(0).values.astype(np.float32)


# ── Time-filtered views ───────────────────────────────────────────────────────

_EDGE_ATTRS = ("edge_index", "edge_attr", "edge_time", "session_id", "is_rand")


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
            for key in _EDGE_ATTRS:
                if key in store:
                    val = store[key]
                    out[et][key] = val[:, keep] if key == "edge_index" else val[keep]
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
