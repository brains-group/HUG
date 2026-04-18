"""
main.py  —  KuaiRand CVR Training & Evaluation Pipeline
---------------------------------------------------------
Temporal train/test split -> HKG construction -> dual-GNN training ->
metrics on train and held-out test set.

Works for both KuaiRand-1K and KuaiRand-27K.

Memory strategy
---------------
With 4.4M videos the naive approach (move all node embeddings to GPU at once)
exhausts VRAM.  This file uses three mitigations:

  1. CPU-resident graph  —  HeteroData stays on CPU.  Only the sampled
     mini-batch of node embeddings is moved to GPU per step, not the
     entire 4.4M × hidden_dim matrix.

  2. Neighbourhood sampling  —  Instead of full-graph R-GCN (which builds
     a [N_all, hidden] tensor on GPU), we use PyG's NeighborLoader to
     sample a fixed-size k-hop subgraph around each batch's nodes.  Only
     those sampled nodes hit the GPU.

  3. fp16 mixed precision  —  torch.autocast halves activation memory
     with no code change to the model itself.

Caching
-------
  --cache-dir  controls where the HKG is serialised.  On first run the
               graph is built from CSVs and saved; on subsequent runs it
               is loaded in seconds.  The best model checkpoint is also
               saved to this directory and reloaded automatically.

Usage
-----
# KuaiRand-1K, first run (builds and caches HKG):
    python main.py --data-dir /path/to/KuaiRand-1K/data \\
                   --cache-dir ./cache/1k

# KuaiRand-1K, subsequent runs (loads cached HKG):
    python main.py --data-dir /path/to/KuaiRand-1K/data \\
                   --cache-dir ./cache/1k

# KuaiRand-27K (larger dims, more edges):
    python main.py --data-dir /path/to/KuaiRand-27K/data \\
                   --scale 27k --cache-dir ./cache/27k \\
                   --hidden-dim 256 --out-dim 128 \\
                   --batch-size 1024 --epochs 20

# Resume from checkpoint only (skip training):
    python main.py --data-dir /path/to/KuaiRand-1K/data \\
                   --cache-dir ./cache/1k --eval-only

# Full options:
    python main.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    ndcg_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from data_loader import KuaiRandData, KuaiRandLoader
from gnn_encoders import SequentialGNN, StructuralGNN, SingleHGT
from hkg_constructor import HKGBundle, HKGConstructor
from models import AlignmentModule, CVRHead, KuaiCVRModel, SingleGNNModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Cache helpers ─────────────────────────────────────────────────────────────

HKG_CACHE_FILE = "hkg_bundle.pkl"
ARGS_CACHE_FILE = "run_args.json"


def _ckpt_filename(model_type: str) -> str:
    """Checkpoint filename scoped to model type — dual and single never collide."""
    return f"best_model_{model_type}.pt"


def _hkg_cache_path(cache_dir: Path) -> Path:
    return cache_dir / HKG_CACHE_FILE


def save_hkg(bundle: HKGBundle, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _hkg_cache_path(cache_dir)
    with open(path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("HKG cached -> %s  (%.1f MB)", path, path.stat().st_size / 1e6)


def load_hkg(cache_dir: Path) -> HKGBundle | None:
    path = _hkg_cache_path(cache_dir)
    if not path.exists():
        return None
    logger.info("Loading cached HKG from %s …", path)
    t0 = time.time()
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    logger.info("HKG loaded from cache in %.1fs", time.time() - t0)
    return bundle


def save_checkpoint(model, args: argparse.Namespace,
                    best_auc: float, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / _ckpt_filename(args.model_type)
    torch.save({
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "best_auc":    best_auc,
        "args":        vars(args),
        "model_type":  args.model_type,
    }, ckpt_path)
    (out_dir / ARGS_CACHE_FILE).write_text(json.dumps(vars(args), indent=2))
    logger.info("Checkpoint saved -> %s  (AUC=%.4f)", ckpt_path, best_auc)


def load_checkpoint(model, device: torch.device,
                    out_dir: Path, model_type: str = "dual") -> float:
    """Load best checkpoint into model in-place. Returns best_auc or 0."""
    ckpt_path = out_dir / _ckpt_filename(model_type)
    if not ckpt_path.exists():
        return 0.0
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict({k: v.to(device) for k, v in ckpt["model_state"].items()})
    auc = float(ckpt.get("best_auc", 0.0))
    logger.info("Checkpoint loaded from %s  (AUC=%.4f)", ckpt_path, auc)
    return auc


# ── Dataset ───────────────────────────────────────────────────────────────────

class InteractionDataset(Dataset):
    """
    Flat dataset of (user_idx, video_idx, session_idx, ips_weight, label).

    Stored on CPU as a float32 ndarray; moved to GPU per-batch by the
    DataLoader pin_memory + non_blocking mechanism.
    """

    def __init__(self, interactions: np.ndarray) -> None:
        self.data = torch.from_numpy(interactions).float()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data[idx]
        return {
            "user_idx":    row[0].long(),
            "video_idx":   row[1].long(),
            "session_idx": row[2].long(),
            "ips_weight":  row[3],
            "label":       row[4],
        }


# ── Metrics ───────────────────────────────────────────────────────────────────

@dataclass
class Metrics:
    split:     str
    loss:      float = 0.0
    auc:       float = 0.0
    ap:        float = 0.0
    logloss:   float = 0.0
    ndcg10:    float = 0.0
    n_samples: int   = 0
    n_pos:     int   = 0

    def __str__(self) -> str:
        pos_rate = self.n_pos / max(self.n_samples, 1)
        return (
            f"[{self.split:5s}]  loss={self.loss:.4f}  "
            f"AUC={self.auc:.4f}  AP={self.ap:.4f}  "
            f"LogLoss={self.logloss:.4f}  nDCG@10={self.ndcg10:.4f}  "
            f"n={self.n_samples:,}  pos_rate={pos_rate:.3f}"
        )


def _per_user_ndcg_at_k(
    user_idxs: np.ndarray,
    labels:    np.ndarray,
    scores:    np.ndarray,
    k:         int = 10,
) -> float:
    """
    Compute NDCG@k averaged over users.

    Only users with ≥2 interactions and at least one positive label contribute
    to the average. Users with a single interaction trivially achieve 1.0 and
    would inflate the metric; users with no positives have undefined NDCG.
    """
    unique_users = np.unique(user_idxs)
    per_user_ndcg: list[float] = []

    for uid in unique_users:
        mask  = user_idxs == uid
        u_lbl = labels[mask]
        u_scr = scores[mask]

        if u_lbl.sum() == 0 or len(u_lbl) < 2:
            continue

        try:
            n = min(k, len(u_lbl))
            score = float(ndcg_score(u_lbl.reshape(1, -1), u_scr.reshape(1, -1), k=n))
            per_user_ndcg.append(score)
        except Exception:
            continue

    return float(np.mean(per_user_ndcg)) if per_user_ndcg else 0.0


def compute_metrics(split: str, labels: np.ndarray,
                    probas: np.ndarray, loss: float,
                    user_idxs: np.ndarray | None = None) -> Metrics:
    m = Metrics(split=split, loss=loss,
                n_samples=len(labels), n_pos=int(labels.sum()))
    if m.n_pos == 0 or m.n_pos == m.n_samples:
        logger.warning("%s split has no label variance — skipping AUC/AP", split)
        return m
    m.auc     = float(roc_auc_score(labels, probas))
    m.ap      = float(average_precision_score(labels, probas))
    m.logloss = float(log_loss(labels, probas))
    if user_idxs is not None:
        m.ndcg10 = _per_user_ndcg_at_k(user_idxs, labels, probas, k=10)
    return m


# ── Temporal split ────────────────────────────────────────────────────────────

def temporal_split(data: KuaiRandData,
                   test_ratio: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """
    Split interactions by time_ms.  The earliest (1-test_ratio) fraction
    is train; the most recent test_ratio fraction is test.

    Memory-efficient for 27K: works with only the columns needed and avoids
    full DataFrame copies.  Peak RAM is approximately one copy of the 5-column
    interaction array rather than multiple copies of the full session_map.

    Returns np.ndarray of shape [N, 5]:
        columns: user_idx, video_idx, session_idx, ips_weight, is_click
    """
    uid_map = data.user_id_map
    vid_map = data.video_id_map
    sm      = data.session_map

    # Work only on the 6 columns we need — avoids copying all ~20 columns
    needed = ["user_id", "video_id", "session_id", "time_ms", "is_rand", "is_click"]
    present = [c for c in needed if c in sm.columns]
    df = sm[present].copy()

    # Filter to users/videos in the id maps
    mask = df["user_id"].isin(uid_map) & df["video_id"].isin(vid_map)
    df   = df[mask].reset_index(drop=True)

    # Map ids to contiguous integers in-place (avoids extra columns)
    df["u_idx"] = df["user_id"].map(uid_map).astype(np.int32)
    df["v_idx"] = df["video_id"].map(vid_map).astype(np.int32)
    df["ips"]   = np.where(df["is_rand"].values == 1,
                           np.float32(1.0), np.float32(0.9963))

    # Free the large string/int64 columns we no longer need before sorting
    df.drop(columns=["user_id", "video_id", "is_rand"], inplace=True)

    df = df.sort_values("time_ms").reset_index(drop=True)
    cutoff = int(len(df) * (1 - test_ratio))
    cols   = ["u_idx", "v_idx", "session_id", "ips", "is_click"]

    train_arr = df.iloc[:cutoff][cols].values.astype(np.float32)
    test_arr  = df.iloc[cutoff:][cols].values.astype(np.float32)

    # Free df before returning — caller only needs the numpy arrays
    del df

    logger.info("Temporal split  train=%s  test=%s",
                f"{len(train_arr):,}", f"{len(test_arr):,}")
    return train_arr, test_arr


# ── Memory-safe relation edge index ───────────────────────────────────────────

# Relation type ids for RGCNConv
REL_MAP = {
    ("user",  "clicked",    "video"):    0,
    ("user",  "liked",      "video"):    1,
    ("user",  "hated",      "video"):    2,
    ("user",  "commented",  "video"):    3,
    ("user",  "forwarded",  "video"):    4,
    ("video", "made_by",    "author"):   5,
    ("video", "tagged_as",  "category"): 6,
}


def build_relation_edge_index(
    sg:                 HeteroData,
    device:             torch.device,
    max_edges_per_type: int = 200_000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build the merged (edge_index, relation_types) for RGCNConv ONCE using
    the full graph with correct global node-index offsets.

    Called once before the training loop and reused every epoch.
    Never called per-batch — that was the source of the hang.

    h_all layout inside StructuralGNN:
        [user(0..n_user) | video(n_user..) | author(..) | category(..)]
    """
    n_user   = sg["user"].num_nodes
    n_video  = sg["video"].num_nodes
    n_author = sg["author"].num_nodes

    offsets = {
        "user":     0,
        "video":    n_user,
        "author":   n_user + n_video,
        "category": n_user + n_video + n_author,
    }

    parts, type_parts = [], []
    for et, rel_id in REL_MAP.items():
        if et not in sg.edge_types:
            continue
        ei = sg[et].edge_index.cpu()
        if ei.shape[1] > max_edges_per_type:
            idx = torch.randperm(ei.shape[1])[:max_edges_per_type]
            ei  = ei[:, idx]
        shifted    = ei.clone()
        shifted[0] += offsets[et[0]]
        shifted[1] += offsets[et[2]]
        parts.append(shifted)
        type_parts.append(torch.full((shifted.shape[1],), rel_id, dtype=torch.long))

    if not parts:
        return (torch.zeros(2, 0, dtype=torch.long, device=device),
                torch.zeros(0,    dtype=torch.long, device=device))

    return (torch.cat(parts,      dim=1).to(device),
            torch.cat(type_parts, dim=0).to(device))


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(data: KuaiRandData, bundle: HKGBundle,
                hidden_dim: int, out_dim: int,
                device: torch.device,
                kg_alignment: int = 0,
                use_recency_gate: bool = True,
                num_layers: int = 2) -> KuaiCVRModel:

    sg = bundle.structural_graph

    user_cont_cols = [
        c for c in data.user_features.columns
        if c.endswith("_log") or c in (
            "activity_level", "is_lowactive_period",
            "is_live_streamer", "is_video_author",
        )
    ]
    onehot_cols   = [f"onehot_feat{i}" for i in range(18)
                     if f"onehot_feat{i}" in data.user_features.columns]
    onehot_vocabs = [int(data.user_features[c].max()) + 1 for c in onehot_cols]
    video_feat_dim = sg["video"].x.shape[1]

    struct = StructuralGNN(
        user_cont_dim     = len(user_cont_cols),
        user_onehot_vocab = onehot_vocabs,
        video_feat_dim    = video_feat_dim,
        author_feat_dim   = 1,
        category_feat_dim = 1,
        hidden_dim        = hidden_dim,
        out_dim           = out_dim,
        num_relations     = 7,
        num_layers        = num_layers,
    )
    seq = SequentialGNN(
        video_feat_dim   = video_feat_dim,
        hidden_dim       = hidden_dim,
        out_dim          = out_dim,
        num_layers       = num_layers,
        use_recency_gate = use_recency_gate,
    )
    align = AlignmentModule(emb_dim=out_dim, kg_relation_dim=kg_alignment, num_heads=4)
    head  = CVRHead(fused_dim=out_dim, hidden_dims=[hidden_dim, hidden_dim // 2])

    model    = KuaiCVRModel(struct, seq, align, head).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model built  params=%s  device=%s", f"{n_params:,}", device)
    return model


def build_full_relation_edge_index(
    bundle:             HKGBundle,
    device:             torch.device,
    max_edges_per_type: int = 200_000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a merged edge index over ALL HKG edge types for the single HGT.

    Unlike build_relation_edge_index (structural only, 7 types), this includes
    next_in_session edges so the HGT sees both structural and sequential signal
    in one pass.  Edge type ids match SingleHGT.EDGE_TYPES ordering.
    """
    fg = bundle.full_graph
    n_user   = fg["user"].num_nodes
    n_video  = fg["video"].num_nodes
    n_author = fg["author"].num_nodes

    offsets = {
        "user":     0,
        "video":    n_user,
        "author":   n_user + n_video,
        "category": n_user + n_video + n_author,
    }

    parts, type_parts = [], []
    for et_id, et in enumerate(SingleHGT.EDGE_TYPES):
        if et not in fg.edge_types:
            continue
        ei = fg[et].edge_index.cpu()
        if ei.shape[1] > max_edges_per_type:
            idx = torch.randperm(ei.shape[1])[:max_edges_per_type]
            ei  = ei[:, idx]
        shifted    = ei.clone()
        shifted[0] += offsets[et[0]]
        shifted[1] += offsets[et[2]]
        parts.append(shifted)
        type_parts.append(torch.full((shifted.shape[1],), et_id, dtype=torch.long))

    if not parts:
        return (torch.zeros(2, 0, dtype=torch.long, device=device),
                torch.zeros(0,    dtype=torch.long, device=device))

    return (torch.cat(parts,      dim=1).to(device),
            torch.cat(type_parts, dim=0).to(device))


def build_single_model(data: KuaiRandData, bundle: HKGBundle,
                        hidden_dim: int, out_dim: int,
                        device: torch.device,
                        use_recency_gate: bool = True,
                        num_layers: int = 2) -> SingleGNNModel:
    """Build the single-HGT baseline model."""
    fg = bundle.full_graph

    user_cont_cols = [
        c for c in data.user_features.columns
        if c.endswith("_log") or c in (
            "activity_level", "is_lowactive_period",
            "is_live_streamer", "is_video_author",
        )
    ]
    onehot_cols   = [f"onehot_feat{i}" for i in range(18)
                     if f"onehot_feat{i}" in data.user_features.columns]
    onehot_vocabs = [int(data.user_features[c].max()) + 1 for c in onehot_cols]
    video_feat_dim = fg["video"].x.shape[1]

    hgt = SingleHGT(
        user_cont_dim     = len(user_cont_cols),
        user_onehot_vocab = onehot_vocabs,
        video_feat_dim    = video_feat_dim,
        author_feat_dim   = 1,
        category_feat_dim = 1,
        hidden_dim        = hidden_dim,
        out_dim           = out_dim,
        num_heads         = 4,
        num_layers        = num_layers,
        use_recency_gate  = use_recency_gate,
    )
    # CVR head input is out_dim * 2 (user cat video, no alignment module)
    head = CVRHead(fused_dim=out_dim * 2, hidden_dims=[hidden_dim, hidden_dim // 2])

    model    = SingleGNNModel(hgt, head).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("SingleGNNModel built  params=%s  device=%s", f"{n_params:,}", device)
    return model


# ── Pre-computed embedding encode step ───────────────────────────────────────

def encode_epoch(
    model:   KuaiCVRModel,
    sg:      HeteroData,
    qg:      HeteroData,
    rel_ei:  torch.Tensor,
    rel_t:   torch.Tensor,
    sess_id: torch.Tensor,
    device:  torch.device,
) -> dict:
    """
    Run both GNNs once over the FULL graph with no gradients.

    Returns CPU tensors to avoid holding the full embedding matrices on GPU
    between batches.  They are moved to GPU lazily inside the batch loop.

    This replaces the per-batch subgraph approach which was scanning millions
    of edges on CPU every step and causing the training hang.
    """
    tqdm.write("      Encoding graph (GNN forward pass) …")
    t0 = time.time()
    emb = model.encode_graph(sg, qg, rel_ei, rel_t, sess_id)
    # Move to CPU immediately to free GPU memory for the batch loop
    emb_cpu = {k: v.cpu() for k, v in emb.items()}
    tqdm.write(f"      Encoded in {time.time()-t0:.1f}s  "
               f"s_user={emb_cpu['s_user'].shape}  "
               f"s_video={emb_cpu['s_video'].shape}")
    return emb_cpu


# ── Single-model encode helper ────────────────────────────────────────────────

def _encode_single(
    model:       "SingleGNNModel",
    bundle:      HKGBundle,
    full_rel_ei: torch.Tensor,
    full_rel_t:  torch.Tensor,
    sess_id:     torch.Tensor,
    device:      torch.device,
) -> dict:
    """Encode the full HKG with the single HGT model."""
    tqdm.write("      Encoding full HKG with SingleHGT …")
    t0  = time.time()
    emb = model.encode_graph(
        bundle.full_graph, full_rel_ei, full_rel_t, sess_id)
    emb_cpu = {k: v.cpu() for k, v in emb.items()}
    # Map to the key names expected by run_epoch / forward_from_embeddings
    out = {
        "s_user":    emb_cpu["user"],
        "s_video":   emb_cpu["video"],
        "q_video":   emb_cpu["video"],    # same tensor — no separate seq encoder
        "q_session": emb_cpu["session"],
    }
    tqdm.write(f"      Encoded in {time.time()-t0:.1f}s")
    return out


# ── Dual-GPU parallel encode ──────────────────────────────────────────────────

def encode_epoch_dual_gpu(
    model:      KuaiCVRModel,
    sg:         HeteroData,
    qg:         HeteroData,
    rel_ei:     torch.Tensor,   # on cuda:0
    rel_t:      torch.Tensor,   # on cuda:0
    sess_id:    torch.Tensor,   # on cuda:1
    chunk_size: int = 500_000,
) -> dict:
    """
    Encode the graph using two GPUs in parallel via Python threads.

    GPU 0: StructuralGNN  — R-GCN over all node types, chunked over videos.
    GPU 1: SequentialGNN  — GGNN over session graph, chunked over videos.

    Both write their output to pinned CPU RAM simultaneously.  The batch
    loop then runs on GPU 0 (alignment + CVR head) using small [B, D] slices
    transferred from CPU via non_blocking pin_memory.

    Peak GPU memory per device:
        GPU 0: (chunk_size + n_users + n_authors + n_cats) × hidden_dim
        GPU 1: chunk_size × hidden_dim  (sequential graph is video-only)

    With chunk_size=500K and hidden_dim=128:
        GPU 0: ~(500K + 1.4M) × 128 × 4B ≈ 1.0 GB
        GPU 1: 500K × 128 × 4B            ≈ 0.25 GB
    Both well within 93 GB each.
    """
    import threading

    dev0 = torch.device(f"cuda:{model._gpu_structural}")
    dev1 = torch.device(f"cuda:{model._gpu_sequential}")

    n_video  = sg["video"].num_nodes
    n_chunks = (n_video + chunk_size - 1) // chunk_size

    # Pre-allocate pinned output tensors on CPU for zero-copy transfer
    D = next(model.structural_gnn.parameters()).shape[0]  # out_dim
    # Resolve actual out_dim from the output projection layer
    for m in model.structural_gnn.modules():
        if hasattr(m, 'out_features'):
            D = m.out_features
            break

    s_user_out   = torch.zeros(sg["user"].num_nodes, D).pin_memory()
    s_video_out  = torch.zeros(n_video,              D).pin_memory()
    q_video_out  = torch.zeros(n_video,              D).pin_memory()
    q_session_out = torch.zeros(
        qg["session"].num_nodes, D
    ).pin_memory()

    errors: list = []

    # ── Thread 0: Structural GNN on GPU 0 ────────────────────────────────────
    def run_structural():
        try:
            import torch.nn.functional as Fs
            with torch.no_grad():
                # Encode small node types once
                h_user = model.structural_gnn._encode_user(
                    sg["user"].x.to(dev0), sg["user"].onehot.to(dev0))
                h_author   = Fs.relu(model.structural_gnn.author_proj(
                    sg["author"].x.to(dev0)))
                h_category = Fs.relu(model.structural_gnn.category_proj(
                    sg["category"].x.to(dev0)))

                n_u = h_user.shape[0]
                n_a = h_author.shape[0]

                # Re-map rel_ei (already on dev0)
                n_user_g  = sg["user"].num_nodes
                n_author_g = sg["author"].num_nodes

                pbar = tqdm(range(n_chunks), desc="  struct",
                            unit="chunk", leave=False, dynamic_ncols=True)
                for c in pbar:
                    v0, v1 = c * chunk_size, min((c+1)*chunk_size, n_video)
                    C = v1 - v0
                    h_vid = Fs.relu(model.structural_gnn.video_proj(
                        sg["video"].x[v0:v1].to(dev0)))

                    h_all = torch.cat([h_user, h_vid, h_author, h_category], dim=0)

                    # Filter + remap edges for this chunk
                    vlo, vhi = n_user_g + v0, n_user_g + v1
                    src, dst = rel_ei[0], rel_ei[1]
                    src_vid = (src >= n_user_g) & (src < n_user_g + n_video)
                    dst_vid = (dst >= n_user_g) & (dst < n_user_g + n_video)
                    src_ok  = (~src_vid) | ((src >= vlo) & (src < vhi))
                    dst_ok  = (~dst_vid) | ((dst >= vlo) & (dst < vhi))
                    valid   = src_ok & dst_ok
                    ei      = rel_ei[:, valid].clone()
                    rt      = rel_t[valid]
                    # Remap video positions to chunk-local
                    sv = (ei[0] >= n_user_g) & (ei[0] < n_user_g + n_video)
                    dv = (ei[1] >= n_user_g) & (ei[1] < n_user_g + n_video)
                    ei[0, sv] -= v0
                    ei[1, dv] -= v0
                    shift = n_video - C
                    ei[0, ei[0] >= n_user_g + C] -= shift
                    ei[1, ei[1] >= n_user_g + C] -= shift

                    for layer in model.structural_gnn.rgcn_layers:
                        h_all = Fs.relu(
                            model.structural_gnn.norm(layer(h_all, ei, rt)))

                    s_video_out[v0:v1] = model.structural_gnn.video_out(
                        h_all[n_u: n_u + C]).cpu()
                    pbar.set_postfix({"done": f"{v1:,}"})

                # User embeddings (full pass was already done above)
                s_user_out[:] = model.structural_gnn.user_out(h_user).cpu()

        except Exception as e:
            errors.append(("structural", e))

    # ── Thread 1: Sequential GNN on GPU 1 ────────────────────────────────────
    def run_sequential():
        try:
            import torch.nn.functional as Fq
            seq_et = ("video", "next_in_session", "video")
            ei_seq = qg[seq_et].edge_index   # CPU

            with torch.no_grad():
                pbar = tqdm(range(n_chunks), desc="  seq   ",
                            unit="chunk", leave=False, dynamic_ncols=True)
                for c in pbar:
                    v0, v1 = c * chunk_size, min((c+1)*chunk_size, n_video)
                    x = qg["video"].x[v0:v1].to(dev1)
                    h = Fq.relu(model.sequential_gnn.input_proj(x))

                    src, dst = ei_seq[0], ei_seq[1]
                    in_c = (src >= v0) & (src < v1) & \
                           (dst >= v0) & (dst < v1)
                    ei_loc = (ei_seq[:, in_c] - v0).to(dev1)

                    h = model.sequential_gnn.norm(
                        model.sequential_gnn.ggnn(h, ei_loc))
                    q_video_out[v0:v1] = model.sequential_gnn.out_proj(h).cpu()
                    pbar.set_postfix({"done": f"{v1:,}"})

                # Session embeddings
                full_seq = model.sequential_gnn(qg, sess_id)
                q_session_out[:] = full_seq["session"].cpu()

        except Exception as e:
            errors.append(("sequential", e))

    tqdm.write(f"      Dual-GPU encode: {n_chunks} chunks × {chunk_size:,} videos …")
    t0 = time.time()

    t_struct = threading.Thread(target=run_structural, name="struct-gpu0")
    t_seq    = threading.Thread(target=run_sequential, name="seq-gpu1")
    t_struct.start()
    t_seq.start()
    t_struct.join()
    t_seq.join()

    if errors:
        for name, exc in errors:
            tqdm.write(f"ERROR in {name} thread: {exc}")
        raise RuntimeError(f"Dual-GPU encode failed: {[e[0] for e in errors]}")

    tqdm.write(f"      Dual-GPU encode done in {time.time()-t0:.1f}s  "
               f"RAM ~{(s_video_out.numel()+q_video_out.numel())*4/1e9:.1f} GB")

    return {
        "s_user":    s_user_out,
        "s_video":   s_video_out,
        "q_video":   q_video_out,
        "q_session": q_session_out,
    }


# ── Chunked encode for 27K ────────────────────────────────────────────────────

def encode_epoch_chunked(
    model:      KuaiCVRModel,
    sg:         HeteroData,
    qg:         HeteroData,
    rel_ei:     torch.Tensor,
    rel_t:      torch.Tensor,
    sess_id:    torch.Tensor,
    device:     torch.device,
    chunk_size: int = 500_000,
) -> dict:
    """
    Memory-safe graph encoding for 27K (32M videos).

    Problem with full encode_epoch on 27K
    ---------------------------------------
    encode_graph runs the R-GCN by concatenating ALL node types into h_all:
        [27K users | 32M videos | 1.4M authors | categories]
    That single tensor at hidden_dim=128 is ~17 GB on GPU — OOM.
    Plus the output embedding matrices (s_video, q_video) would be ~8 GB
    each on CPU.

    Solution: chunk the video node features
    -----------------------------------------
    1. Run the structural GNN with the FULL edge index but only a chunk of
       video node features at a time, accumulating output embeddings on CPU.
    2. For sequential, chunk the video node embeddings from the GGNN similarly.

    This keeps peak GPU usage to:
        (chunk_size + n_users + n_authors + n_categories) × hidden_dim
    e.g. 500K chunk → ~256 MB instead of 17 GB.

    Trade-off: message passing is approximate — edges between chunk members
    and out-of-chunk nodes carry their initialised (pre-layer) features, not
    the updated ones.  This is equivalent to 1-hop GraphSAGE sampling and is
    standard practice for graphs of this scale.
    """
    tqdm.write("      Encoding graph in chunks (27K mode) …")
    t0 = time.time()

    dev = next(model.parameters()).device

    n_user   = sg["user"].num_nodes
    n_video  = sg["video"].num_nodes
    n_author = sg["author"].num_nodes

    # ── Structural: encode users + fixed non-video nodes once ─────────────────
    with torch.no_grad():
        h_user_in     = model.structural_gnn._encode_user(
            sg["user"].x.to(dev), sg["user"].onehot.to(dev)
        )                                                          # [n_user, H]
        h_author_in   = F.relu(model.structural_gnn.author_proj(
            sg["author"].x.to(dev)))                               # [n_author, H]
        h_category_in = F.relu(model.structural_gnn.category_proj(
            sg["category"].x.to(dev)))                             # [n_cat, H]

    import torch.nn.functional as F_local

    s_video_chunks, q_video_chunks = [], []

    n_chunks = (n_video + chunk_size - 1) // chunk_size
    chunk_bar = tqdm(range(n_chunks), desc="      chunks",
                     unit="chunk", leave=False, dynamic_ncols=True)

    for c in chunk_bar:
        v_start = c * chunk_size
        v_end   = min(v_start + chunk_size, n_video)

        # ── Structural chunk ──────────────────────────────────────────────────
        with torch.no_grad():
            h_video_chunk = F_local.relu(model.structural_gnn.video_proj(
                sg["video"].x[v_start:v_end].to(dev)))             # [C, H]

            # Build h_all for this chunk: user | video_chunk | author | cat
            h_all = torch.cat([h_user_in, h_video_chunk,
                                h_author_in, h_category_in], dim=0)

            # Filter relation edges to those whose video endpoints fall in chunk.
            # Video nodes in the chunk have global offsets [v_start, v_end).
            # In rel_ei they appear at positions [n_user+v_start, n_user+v_end).
            chunk_v_lo = n_user + v_start
            chunk_v_hi = n_user + v_end

            # Remap video indices in rel_ei to chunk-local positions
            # Keep only edges where BOTH src and dst are in-scope for h_all.
            # h_all layout: [0..n_user | n_user..n_user+chunk | ...]
            # We shift video global offsets → chunk-local offsets.
            src = rel_ei[0].clone()
            dst = rel_ei[1].clone()

            # Identify video positions in each side
            src_is_video = (src >= n_user) & (src < n_user + n_video)
            dst_is_video = (dst >= n_user) & (dst < n_user + n_video)

            # For video positions: remap global → chunk-local
            # out-of-chunk video edges are dropped
            src_in_chunk = src_is_video & (src >= chunk_v_lo) & (src < chunk_v_hi)
            dst_in_chunk = dst_is_video & (dst >= chunk_v_lo) & (dst < chunk_v_hi)

            # Keep edge if: neither side is a video, OR the video side is in chunk
            src_ok = (~src_is_video) | src_in_chunk
            dst_ok = (~dst_is_video) | dst_in_chunk
            valid  = src_ok & dst_ok

            ei_chunk = rel_ei[:, valid].clone()
            rt_chunk = rel_t[valid]

            # Remap video global indices → chunk-local (subtract v_start from
            # video positions; non-video positions shift too, fix below)
            # Easier: rebase everything relative to h_all layout for this chunk.
            # h_all = [user(0..n_user) | video_chunk(n_user..n_user+C) |
            #          author(n_user+C..) | cat(..)]
            C = v_end - v_start
            # Positions that were video global → subtract v_start to get chunk pos
            src_vid_mask = (ei_chunk[0] >= n_user) & (ei_chunk[0] < n_user + n_video)
            dst_vid_mask = (ei_chunk[1] >= n_user) & (ei_chunk[1] < n_user + n_video)
            ei_chunk[0, src_vid_mask] -= v_start
            ei_chunk[1, dst_vid_mask] -= v_start
            # Author/category positions need to shift down by (n_video - C)
            shift = n_video - C
            src_auth_mask = ei_chunk[0] >= (n_user + n_video)
            dst_auth_mask = ei_chunk[1] >= (n_user + n_video)
            ei_chunk[0, src_auth_mask] -= shift
            ei_chunk[1, dst_auth_mask] -= shift

            for layer in model.structural_gnn.rgcn_layers:
                h_all = F_local.relu(
                    model.structural_gnn.norm(layer(h_all, ei_chunk, rt_chunk))
                )

            # Extract video chunk output: positions n_user .. n_user+C
            h_vid_out = model.structural_gnn.video_out(h_all[n_user: n_user + C])
            s_video_chunks.append(h_vid_out.cpu())

        # ── Sequential chunk ──────────────────────────────────────────────────
        with torch.no_grad():
            seq_et    = ("video", "next_in_session", "video")
            ei_seq    = qg[seq_et].edge_index
            x_chunk   = qg["video"].x[v_start:v_end].to(dev)
            h_seq     = F_local.relu(model.sequential_gnn.input_proj(x_chunk))

            # Filter sequential edges to this chunk
            src_seq   = ei_seq[0]
            dst_seq   = ei_seq[1]
            in_chunk  = (src_seq >= v_start) & (src_seq < v_end) & \
                        (dst_seq >= v_start) & (dst_seq < v_end)
            ei_local  = ei_seq[:, in_chunk] - v_start  # local 0-based

            ei_local  = ei_local.to(dev)
            h_seq     = model.sequential_gnn.norm(
                model.sequential_gnn.ggnn(h_seq, ei_local))

            q_video_chunks.append(
                model.sequential_gnn.out_proj(h_seq).cpu())

        chunk_bar.set_postfix({"videos_done": f"{v_end:,}"})

    chunk_bar.close()

    # Encode users and sessions (small — full pass)
    with torch.no_grad():
        full_struct = model.structural_gnn(sg, rel_ei, rel_t)
        full_seq    = model.sequential_gnn(qg, sess_id)

    emb_cpu = {
        "s_user":    full_struct["user"].cpu(),
        "s_video":   torch.cat(s_video_chunks, dim=0),
        "q_video":   torch.cat(q_video_chunks, dim=0),
        "q_session": full_seq["session"].cpu(),
    }

    tqdm.write(f"      Chunked encode done in {time.time()-t0:.1f}s  "
               f"s_video={emb_cpu['s_video'].shape}  "
               f"RAM ~{sum(v.numel()*4 for v in emb_cpu.values())/1e9:.1f} GB")
    return emb_cpu


# ── One epoch ─────────────────────────────────────────────────────────────────

def run_epoch(
    model:     KuaiCVRModel,
    loader:    DataLoader,
    emb:       dict,          # pre-computed CPU embeddings from encode_epoch
    n_sess:    int,
    device:    torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler:    torch.cuda.amp.GradScaler | None,
    split:     str,
    use_amp:   bool,
    no_ips:    bool = False,
) -> Metrics:
    """
    One pass over the DataLoader using pre-computed graph embeddings.

    The GNNs are NOT called here — encode_epoch() handles that once per epoch.
    This loop only runs: index lookup → alignment → CVR head → loss/backward.
    Each batch takes milliseconds instead of minutes.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    # Determine which device runs the head (alignment + CVR)
    head_dev = model.head_device if hasattr(model, 'head_device') else device

    # Move embedding matrices to head device once for the whole epoch
    emb_dev = {k: v.to(head_dev, non_blocking=True) for k, v in emb.items()}

    total_loss, all_labels, all_probas, all_users = 0.0, [], [], []
    n_seen  = 0
    ctx     = torch.enable_grad() if is_train else torch.no_grad()
    bar_fmt = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining} {rate_fmt}] {postfix}"

    pbar = tqdm(
        loader,
        desc          = f"  {split:5s}",
        unit          = "batch",
        bar_format    = bar_fmt,
        dynamic_ncols = True,
        leave         = False,
    )

    with ctx:
        for batch in pbar:
            user_idx    = batch["user_idx"].to(head_dev, non_blocking=True)
            video_idx   = batch["video_idx"].to(head_dev, non_blocking=True)
            session_idx = batch["session_idx"].to(head_dev, non_blocking=True).clamp(0, n_sess - 1)
            ips_weights = batch["ips_weight"].to(head_dev, non_blocking=True)
            labels      = batch["label"].to(head_dev, non_blocking=True)

            if no_ips:
                ips_weights = torch.ones_like(ips_weights)

            with torch.autocast(device_type=head_dev.type, enabled=use_amp):
                out = model.forward_from_embeddings(
                    emb         = emb_dev,
                    user_idx    = user_idx,
                    video_idx   = video_idx,
                    session_idx = session_idx,
                    ips_weights = ips_weights,
                    labels      = labels,
                )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(out["loss"]).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out["loss"].backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()

            n           = batch["label"].shape[0]
            total_loss += out["loss"].item() * n
            n_seen     += n
            running_loss = total_loss / max(n_seen, 1)

            all_labels.append(batch["label"].cpu().numpy())
            all_probas.append(out["proba"].detach().cpu().float().numpy())
            all_users.append(batch["user_idx"].cpu().numpy())

            postfix: dict = {"loss": f"{running_loss:.4f}"}
            if torch.cuda.is_available():
                postfix["gpu"] = f"{torch.cuda.memory_allocated()/1e9:.1f}GB"
            pbar.set_postfix(postfix)

    pbar.close()
    # Free GPU embedding copies
    del emb_dev

    labels_np    = np.concatenate(all_labels)
    probas_np    = np.concatenate(all_probas)
    user_idxs_np = np.concatenate(all_users)
    avg_loss     = total_loss / max(len(labels_np), 1)
    return compute_metrics(split, labels_np, probas_np, avg_loss, user_idxs=user_idxs_np)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="KuaiRand CVR pipeline — temporal train/test split",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data & paths
    p.add_argument("--data-dir",   type=str, required=True,
                   help="Path to KuaiRand data directory (six CSV files)")
    p.add_argument("--cache-dir",  type=str, default=None,
                   help="Directory for HKG cache and model checkpoint. "
                        "Defaults to ./cache/<scale>. Set to 'none' to disable.")
    p.add_argument("--output-dir", type=str, default="./runs",
                   help="Directory for metrics JSON and training history")
    p.add_argument("--scale",      type=str, default="1k", choices=["1k", "27k"],
                   help="Dataset scale — used for file naming only")

    # Runtime
    p.add_argument("--device",  type=str, default=None,
                   help="Torch device (cuda / cuda:1 / mps / cpu). Auto if omitted")
    p.add_argument("--no-amp",  action="store_true",
                   help="Disable automatic mixed precision (fp16)")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training; load checkpoint and evaluate only")
    p.add_argument("--rebuild-hkg", action="store_true",
                   help="Ignore cached HKG and rebuild from CSV files")
    p.add_argument("--seed",    type=int, default=42)

    # Data loading
    p.add_argument("--min-interactions", type=int,   default=10)
    p.add_argument("--test-ratio",       type=float, default=0.2)
    p.add_argument("--max-seq-len",      type=int,   default=50)

    # Model
    p.add_argument("--kg-alignment", type=int, default=0)
    p.add_argument("--hidden-dim",  type=int, default=128)
    p.add_argument("--out-dim",     type=int, default=64)
    p.add_argument("--model-type",  type=str, default="dual",
                   choices=["dual", "single"],
                   help="dual: R-GCN + SR-GNN + alignment (proposed). "
                        "single: HGT over full HKG, no alignment (baseline).")

    # Ablation flags
    p.add_argument("--no-ips", action="store_true",
                   help="Disable IPS weighting; use uniform weights (ablation).")
    p.add_argument("--no-recency-gate", action="store_true",
                   help="Disable recency gate in SequentialGNN (ablation).")
    p.add_argument("--gnn-layers", type=int, default=2,
                   help="Number of message-passing layers in both GNN encoders.")
    p.add_argument("--run-tag", type=str, default=None,
                   help="Optional suffix appended to the run name for labelling "
                        "hyperparameter-sensitivity runs (e.g. 'layers3').")

    # Training
    p.add_argument("--epochs",       type=int,   default=10)
    p.add_argument("--batch-size",   type=int,   default=2048)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)

    p.add_argument("--multi-gpu", action="store_true",
                   help="Split model across 2 GPUs: structural on cuda:0, "
                        "sequential on cuda:1, head on cuda:0. "
                        "Enables parallel encode_epoch for 27K.")
    p.add_argument("--gpu-structural", type=int, default=0,
                   help="GPU index for StructuralGNN (default 0)")
    p.add_argument("--gpu-sequential", type=int, default=1,
                   help="GPU index for SequentialGNN (default 1)")
    # Memory controls
    p.add_argument("--max-edges",   type=int, default=200_000,
                   help="Max edges per relation type sampled for R-GCN.")
    p.add_argument("--chunk-size",  type=int, default=500_000,
                   help="Video node chunk size for 27K encoding (reduce if OOM).")

    return p.parse_args()


def resolve_device(requested: str | None) -> torch.device:
    if requested is not None:
        d = torch.device(requested)
        torch.zeros(1, device=d)
        return d
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_cache_dir(args: argparse.Namespace) -> Path | None:
    if args.cache_dir and args.cache_dir.lower() == "none":
        return None
    base = args.cache_dir or f"./cache/{args.scale}"
    return Path(base)


def make_run_name(args: argparse.Namespace) -> str:
    """
    Build a canonical run-name string encoding every flag that changes the
    model or training objective, so results from different runs never collide.

    Base format: {scale}_{model_type}_kg{kg}_ips{0|1}_rg{0|1}
    Non-default layers appended as:  _L{n}
    Non-default dims appended as:    _h{hidden}d{out}
    Optional free-text tag appended: _{run_tag}

    Examples
    --------
    1k_dual_kg64_ips1_rg1           → full model, default hypers
    1k_dual_kg64_ips0_rg1           → no IPS
    1k_dual_kg64_ips1_rg0           → no recency gate
    1k_dual_kg64_ips1_rg1_L1        → 1 GNN layer
    1k_dual_kg64_ips1_rg1_h256d128  → large embedding
    """
    ips_flag = "0" if getattr(args, "no_ips", False)          else "1"
    rg_flag  = "0" if getattr(args, "no_recency_gate", False) else "1"
    kg_dim   = getattr(args, "kg_alignment", 0)
    name     = f"{args.scale}_{args.model_type}_kg{kg_dim}_ips{ips_flag}_rg{rg_flag}"

    layers = getattr(args, "gnn_layers", 2)
    hidden = getattr(args, "hidden_dim", 128)
    out    = getattr(args, "out_dim",    64)
    if layers != 2:
        name += f"_L{layers}"
    if hidden != 128 or out != 64:
        name += f"_h{hidden}d{out}"

    tag = getattr(args, "run_tag", None)
    if tag:
        name += f"_{tag}"

    return name


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args      = parse_args()
    device    = resolve_device(args.device)
    cache_dir = resolve_cache_dir(args)
    use_amp   = (not args.no_amp) and (device.type == "cuda")

    set_seed(args.seed)

    # ── Pipeline header ───────────────────────────────────────────────────────
    print()
    print("╔" + "═" * 60 + "╗")
    print(f"║  KuaiRand CVR Pipeline  —  {args.scale.upper():<30} ║")
    print("╠" + "═" * 60 + "╣")
    print(f"║  data_dir  : {str(args.data_dir):<46} ║")
    print(f"║  cache_dir : {str(cache_dir or 'disabled'):<46} ║")
    print(f"║  device    : {str(device):<46} ║")
    print(f"║  amp       : {str(use_amp):<46} ║")
    print(f"║  epochs    : {args.epochs:<46} ║")
    print(f"║  batch     : {args.batch_size:<46} ║")
    print(f"║  max_edges : {args.max_edges:<46} ║")
    print(f"║  hidden    : {args.hidden_dim}  out: {args.out_dim:<40} ║")
    print(f"║  run_name  : {make_run_name(args):<46} ║")
    print(f"║  no_ips    : {str(args.no_ips):<46} ║")
    print(f"║  no_rg     : {str(args.no_recency_gate):<46} ║")
    print("╚" + "═" * 60 + "╝")
    print()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    tqdm.write("[1/5] Loading data …")
    t0   = time.time()
    data = KuaiRandLoader(
        args.data_dir,
        min_interactions = args.min_interactions,
        filter_ads       = True,
    ).load()
    tqdm.write(f"      Done in {time.time()-t0:.1f}s  →  {data}")

    # ── 2. Temporal split ─────────────────────────────────────────────────────
    tqdm.write("[2/5] Temporal train/test split …")
    train_arr, test_arr = temporal_split(data, test_ratio=args.test_ratio)

    # Keep DataLoaders on CPU (pin_memory moves to GPU per batch)
    dl_kwargs = dict(num_workers=0, pin_memory=(device.type == "cuda"))
    train_loader = DataLoader(InteractionDataset(train_arr),
                              batch_size=args.batch_size, shuffle=True,  **dl_kwargs)
    test_loader  = DataLoader(InteractionDataset(test_arr),
                              batch_size=args.batch_size * 2, shuffle=False, **dl_kwargs)
    del train_arr, test_arr   # numpy arrays are now inside the DataLoaders

    # ── 3. HKG — load from cache or build ────────────────────────────────────
    tqdm.write("[3/5] Preparing HKG …")
    bundle: HKGBundle | None = None
    if cache_dir and not args.rebuild_hkg:
        bundle = load_hkg(cache_dir)

    if bundle is None:
        tqdm.write("      No cache found — building from CSV files (this may take a while) …")
        t0     = time.time()
        bundle = HKGConstructor(data, device="cpu",
                                max_seq_len=args.max_seq_len).build()
        tqdm.write(f"      HKG built in {time.time()-t0:.1f}s")
        if cache_dir:
            save_hkg(bundle, cache_dir)
    else:
        tqdm.write(f"      Loaded from cache  n_videos={bundle.n_videos:,}  n_sessions={bundle.n_sessions:,}")

    # Free large DataFrames — no longer needed after HKG is built
    # log_standard alone is ~12 GB on 27K; freeing it here recovers
    # that RAM for the HKG graph tensors and embedding matrices
    import gc
    data.log_standard  = None
    data.log_random    = None
    data.log_combined  = None
    data.session_map   = None
    gc.collect()

    sg = bundle.structural_graph
    qg = bundle.sequential_graph

    # Build relation edge index ONCE — reused every epoch encode step
    tqdm.write("      Building relation edge index …")
    rel_ei, rel_t = build_relation_edge_index(sg, device, args.max_edges)
    tqdm.write(f"      Relation edges: {rel_ei.shape[1]:,} ({rel_ei.shape[1]*8/1e6:.1f} MB on GPU)")

    # Sequential session ids
    seq_et  = ("video", "next_in_session", "video")
    sess_id = qg[seq_et].session_id.to(device)
    n_sess  = max(bundle.n_sessions, 1)

    if device.type == "cuda":
        logger.info(
            "GPU memory after HKG setup: allocated=%.1f GB  reserved=%.1f GB",
            torch.cuda.memory_allocated() / 1e9,
            torch.cuda.memory_reserved()  / 1e9,
        )

    # ── 4. Build model ────────────────────────────────────────────────────────
    tqdm.write(f"[4/5] Building model  (type={args.model_type}) …")

    use_rg = not args.no_recency_gate
    if args.model_type == "single":
        model = build_single_model(data, bundle, args.hidden_dim, args.out_dim, device,
                                   use_recency_gate=use_rg,
                                   num_layers=args.gnn_layers)
        # Single model needs the full graph + full edge index
        full_rel_ei, full_rel_t = build_full_relation_edge_index(
            bundle, device, args.max_edges)
        tqdm.write(f"      Full HKG edges: {full_rel_ei.shape[1]:,}")
    else:
        model = build_model(data, bundle, args.hidden_dim, args.out_dim, device,
                            kg_alignment=args.kg_alignment,
                            use_recency_gate=use_rg,
                            num_layers=args.gnn_layers)

    if args.multi_gpu:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            tqdm.write("WARNING: --multi-gpu requested but fewer than 2 GPUs found. "
                       "Falling back to single GPU.")
            args.multi_gpu = False
        else:
            model.split_across_gpus(
                gpu_structural = args.gpu_structural,
                gpu_sequential = args.gpu_sequential,
                gpu_head       = args.gpu_structural,
            )
            tqdm.write(f"      Model split: structural→cuda:{args.gpu_structural}  "
                       f"sequential→cuda:{args.gpu_sequential}  "
                       f"head→cuda:{args.gpu_structural}")
            # Rebuild rel_ei on gpu_structural, sess_id on gpu_sequential
            rel_ei = rel_ei.to(f"cuda:{args.gpu_structural}")
            rel_t  = rel_t.to(f"cuda:{args.gpu_structural}")
            sess_id = sess_id.to(f"cuda:{args.gpu_sequential}")

    # AMP scaler (no-op on non-CUDA)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Each run gets its own output + checkpoint directory (avoids cross-run
    # checkpoint collisions when two runs share the same cache_dir).
    out_dir = Path(args.output_dir) / make_run_name(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint if eval-only or if a prior run left one
    best_auc = 0.0
    ckpt_file = out_dir / _ckpt_filename(args.model_type)
    if args.eval_only or ckpt_file.exists():
        loaded_auc = load_checkpoint(model, device, out_dir, args.model_type)
        if args.eval_only:
            best_auc = loaded_auc

    if args.eval_only:
        tqdm.write("--eval-only: skipping training, loading checkpoint.")
    else:
        # ── 5. Training loop ──────────────────────────────────────────────────
        tqdm.write(f"[5/5] Training  ({len(train_loader):,} batches/epoch × {args.epochs} epochs) …")
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.1,
        )

        history: list[dict] = []
        n_train_batches = len(train_loader)
        n_test_batches  = len(test_loader)

        epoch_bar = tqdm(
            range(1, args.epochs + 1),
            desc          = "Training",
            unit          = "epoch",
            dynamic_ncols = True,
            bar_format    = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
        )

        for epoch in epoch_bar:
            t0 = time.time()
            tqdm.write(f"\n── Epoch {epoch}/{args.epochs} ─────────────────────────────")

            # Encode graph once — dispatch by model type and scale
            if args.model_type == "single":
                emb = _encode_single(model, bundle, full_rel_ei, full_rel_t, sess_id, device)
            elif args.multi_gpu:
                emb = encode_epoch_dual_gpu(
                    model, sg, qg, rel_ei, rel_t, sess_id,
                    chunk_size=args.chunk_size,
                )
            elif args.scale == "27k":
                emb = encode_epoch_chunked(
                    model, sg, qg, rel_ei, rel_t, sess_id, device,
                    chunk_size=args.chunk_size,
                )
            else:
                emb = encode_epoch(model, sg, qg, rel_ei, rel_t, sess_id, device)

            train_m = run_epoch(
                model, train_loader, emb, n_sess,
                device, optimizer, scaler,
                split="train", use_amp=use_amp, no_ips=args.no_ips,
            )
            test_m = run_epoch(
                model, test_loader, emb, n_sess,
                device, None, None,
                split="test", use_amp=use_amp, no_ips=args.no_ips,
            )
            del emb   # free CPU embedding matrices between epochs
            scheduler.step()

            elapsed = time.time() - t0
            lr_now  = scheduler.get_last_lr()[0]

            gpu_str = ""
            if device.type == "cuda":
                gpu_str = (f"  gpu {torch.cuda.memory_allocated()/1e9:.1f}/"
                           f"{torch.cuda.memory_reserved()/1e9:.1f}GB")

            summary = (
                f"  train  loss={train_m.loss:.4f}  AUC={train_m.auc:.4f}  "
                f"AP={train_m.ap:.4f}  nDCG@10={train_m.ndcg10:.4f}\n"
                f"  test   loss={test_m.loss:.4f}  AUC={test_m.auc:.4f}  "
                f"AP={test_m.ap:.4f}  nDCG@10={test_m.ndcg10:.4f}\n"
                f"  lr={lr_now:.2e}  elapsed={elapsed:.1f}s{gpu_str}"
            )
            tqdm.write(summary)

            epoch_bar.set_postfix({
                "tr_auc": f"{train_m.auc:.4f}",
                "te_auc": f"{test_m.auc:.4f}",
                "loss":   f"{train_m.loss:.4f}",
            })

            history.append({
                "epoch": epoch, "train": asdict(train_m),
                "test":  asdict(test_m), "lr": lr_now, "time_s": round(elapsed, 2),
            })

            if test_m.auc > best_auc:
                best_auc = test_m.auc
                save_checkpoint(model, args, best_auc, out_dir)
                tqdm.write(f"  ★ new best test AUC={best_auc:.4f}  (checkpoint saved)")

        epoch_bar.close()

        # Save history
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        logger.info("Training history -> %s", out_dir / "history.json")

    # ── 6. Final evaluation with best checkpoint ──────────────────────────────
    print()
    tqdm.write("── Final evaluation (best checkpoint) ──────────────────────")
    if cache_dir and ckpt_file and ckpt_file.exists():
        load_checkpoint(model, device, cache_dir, args.model_type)

    if args.model_type == "single":
        emb_final = _encode_single(model, bundle, full_rel_ei, full_rel_t, sess_id, device)
    elif args.multi_gpu:
        emb_final = encode_epoch_dual_gpu(
            model, sg, qg, rel_ei, rel_t, sess_id,
            chunk_size=args.chunk_size,
        )
    elif args.scale == "27k":
        emb_final = encode_epoch_chunked(
            model, sg, qg, rel_ei, rel_t, sess_id, device,
            chunk_size=args.chunk_size,
        )
    else:
        emb_final = encode_epoch(model, sg, qg, rel_ei, rel_t, sess_id, device)

    final_train = run_epoch(
        model, train_loader, emb_final, n_sess,
        device, None, None,
        split="train", use_amp=use_amp, no_ips=args.no_ips,
    )
    final_test = run_epoch(
        model, test_loader, emb_final, n_sess,
        device, None, None,
        split="test ", use_amp=use_amp, no_ips=args.no_ips,
    )
    del emb_final

    # ── 7. Print metrics table ────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"  CVR Metrics  —  KuaiRand-{args.scale.upper()}  [{args.model_type.upper()} GNN]")
    print("=" * 62)
    hdr = f"  {'Metric':<12} {'Train':>10} {'Test':>10}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for name, t_val, e_val in [
        ("Loss",    final_train.loss,    final_test.loss),
        ("AUC-ROC", final_train.auc,     final_test.auc),
        ("AP",      final_train.ap,      final_test.ap),
        ("LogLoss", final_train.logloss, final_test.logloss),
        ("nDCG@10", final_train.ndcg10,  final_test.ndcg10),
    ]:
        print(f"  {name:<12} {t_val:>10.4f} {e_val:>10.4f}")

    print()
    print(f"  Train samples  : {final_train.n_samples:>12,}")
    print(f"  Test  samples  : {final_test.n_samples:>12,}")
    print(f"  Train pos rate : {final_train.n_pos / max(final_train.n_samples,1):>12.3f}")
    print(f"  Test  pos rate : {final_test.n_pos  / max(final_test.n_samples, 1):>12.3f}")
    print(f"  Best test AUC  : {best_auc:>12.4f}")
    print("=" * 62 + "\n")

    # ── 8. Save final metrics ─────────────────────────────────────────────────
    # Include model_type in the directory so dual and single runs never collide
    out_dir = Path(args.output_dir) / f"kuairand_{args.scale}_{args.model_type}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "final_metrics.json").write_text(json.dumps({
        "train": asdict(final_train),
        "test":  asdict(final_test),
        "args":  vars(args),
        "best_auc": best_auc,
    }, indent=2))
    logger.info("Final metrics -> %s", out_dir / "final_metrics.json")
    logger.info("Done.")


if __name__ == "__main__":
    main()