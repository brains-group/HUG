"""
main.py  —  KuaiRand CVR Training & Evaluation Pipeline
---------------------------------------------------------
Global chronological train/val/test split -> timed HKG -> dual-GNN training
with checkpoint selection on validation -> one evaluation of the selected
checkpoint on test.

Every interaction at time t is scored from inputs built from interactions
before t only (temporal.py): it is encoded from the graph snapshot taken at the
start of its period, and its session embedding pools only the items watched
earlier in the same session.

Memory strategy
---------------
  1. CPU-resident graph  —  the timed HeteroData stays on CPU; each snapshot
     view is moved to the encoder device only while it is being encoded.
  2. Relation edge sampling  —  at most --max-edges edges per relation type
     feed the R-GCN / HGT, sampled once per snapshot.
  3. fp16 mixed precision  —  torch.autocast on the batch loop.

Caching
-------
  --cache-dir  controls where the timed HKG is serialised.  On first run the
               graph is built from CSVs and saved; on subsequent runs it
               is loaded in seconds.  Checkpoints go to the run directory
               under --output-dir.

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
from torch_geometric.data import HeteroData
from tqdm import tqdm

from data_loader import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL, KuaiRandData, KuaiRandLoader
from gnn_encoders import SequentialGNN, StructuralGNN, SingleHGT
from hkg_constructor import HKGBundle, HKGConstructor
from models import AlignmentModule, CVRHead, KuaiCVRModel, SingleGNNModel
from temporal import (
    Interactions, PrefixBatcher, SnapshotStore,
    assign_snapshots, build_interactions, snapshot_boundaries,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Cache helpers ─────────────────────────────────────────────────────────────

HKG_CACHE_FILE = "hkg_bundle_timed.pkl"
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


def load_hkg(cache_dir: Path, data: KuaiRandData) -> HKGBundle | None:
    """Load the cached timed HKG, or None if absent or built for other data."""
    path = _hkg_cache_path(cache_dir)
    if not path.exists():
        return None
    logger.info("Loading cached HKG from %s …", path)
    t0 = time.time()
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    et = ("user", "interacted", "video")
    ok = (
        et in bundle.full_graph.edge_types
        and "edge_time" in bundle.full_graph[et]
        and bundle.n_videos   == len(data.video_id_map)
        and bundle.n_sessions == data.n_sessions
        and bundle.full_graph[et].edge_index.shape[1] == len(data.log_combined)
    )
    if not ok:
        logger.warning("Cached HKG does not match this data — rebuilding.")
        return None
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


# ── Snapshot encode + split runner ────────────────────────────────────────────

def encode_snapshot(model, store: SnapshotStore, k: int, model_type: str) -> dict:
    """
    Run the (frozen) encoders over snapshot k, i.e. the graph of behaviour
    before store.boundaries[k].  Encoders run in eval mode so embeddings are
    deterministic.  Returns embeddings on the head device.
    """
    view = store.view(k)
    rel_ei, rel_t = store.rel[k]
    if model_type == "single":
        enc = model.hgt
    else:
        enc = model.structural_gnn
    enc_dev = next(enc.parameters()).device

    was_training = model.training
    model.eval()
    if model_type == "single":
        emb = model.encode_graph(view.full_graph, rel_ei.to(enc_dev), rel_t.to(enc_dev))
    else:
        emb = model.encode_graph(view.structural_graph, view.sequential_graph,
                                 rel_ei.to(enc_dev), rel_t.to(enc_dev))
    model.train(was_training)
    head_dev = model.head_device
    return {k_: v.to(head_dev) for k_, v in emb.items()}


def run_split(
    model,
    rows:       np.ndarray,          # row indices into the Interactions table
    snap_of:    np.ndarray,          # [N] snapshot index per row
    store:      SnapshotStore,
    batcher:    PrefixBatcher,
    model_type: str,
    batch_size: int,
    optimizer:  torch.optim.Optimizer | None,
    scaler:     torch.cuda.amp.GradScaler | None,
    split:      str,
    use_amp:    bool,
    no_ips:     bool = False,
) -> Metrics:
    """
    One pass over `rows`, snapshot by snapshot.  For each snapshot the
    encoders run once and its rows are scored in mini-batches (train: shuffled
    snapshot order and shuffled rows within each snapshot).
    """
    is_train = optimizer is not None
    head_dev = model.head_device

    snaps = np.unique(snap_of[rows])
    if is_train:
        snaps = np.random.permutation(snaps)

    total_loss, all_labels, all_probas, all_users = 0.0, [], [], []
    pbar = tqdm(total=len(rows), desc=f"  {split:5s}", unit="row",
                dynamic_ncols=True, leave=False)

    for k in snaps:
        sel = rows[snap_of[rows] == k]
        if is_train:
            sel = np.random.permutation(sel)
        emb = encode_snapshot(model, store, int(k), model_type)
        model.train() if is_train else model.eval()
        ctx = torch.enable_grad() if is_train else torch.no_grad()

        with ctx:
            for b0 in range(0, len(sel), batch_size):
                batch = batcher(sel[b0:b0 + batch_size])
                batch = {key: v.to(head_dev, non_blocking=True) for key, v in batch.items()}
                ips_weights = torch.ones_like(batch["ips_weight"]) if no_ips else batch["ips_weight"]

                with torch.autocast(device_type=head_dev.type, enabled=use_amp):
                    out = model.forward_from_embeddings(
                        emb          = emb,
                        user_idx     = batch["user_idx"],
                        video_idx    = batch["video_idx"],
                        prefix_items = batch["prefix_items"],
                        ips_weights  = ips_weights,
                        labels       = batch["label"],
                    )

                if is_train:
                    optimizer.zero_grad(set_to_none=True)
                    if scaler is not None and use_amp:
                        scaler.scale(out["loss"]).backward()
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        out["loss"].backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        optimizer.step()

                n = batch["label"].shape[0]
                total_loss += out["loss"].item() * n
                all_labels.append(batch["label"].cpu().numpy())
                all_probas.append(out["proba"].detach().cpu().float().numpy())
                all_users.append(batch["user_idx"].cpu().numpy())
                pbar.update(n)
                pbar.set_postfix({"loss": f"{total_loss / max(pbar.n, 1):.4f}"})
        del emb

    pbar.close()
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
    p.add_argument("--val-ratio",        type=float, default=0.1)
    p.add_argument("--test-ratio",       type=float, default=0.2)
    p.add_argument("--max-seq-len",      type=int,   default=50,
                   help="Max earlier items of the same session pooled per interaction")
    p.add_argument("--snapshot-hours",   type=float, default=24.0,
                   help="Graph snapshot period: interactions are encoded from the "
                        "graph as of the start of their period")
    p.add_argument("--warmup-hours",     type=float, default=0.0,
                   help="Training interactions this close to the start of the log "
                        "are used only as history, not as training targets")

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
                        "sequential on cuda:1, head on cuda:0.")
    p.add_argument("--gpu-structural", type=int, default=0,
                   help="GPU index for StructuralGNN (default 0)")
    p.add_argument("--gpu-sequential", type=int, default=1,
                   help="GPU index for SequentialGNN (default 1)")
    # Memory controls
    p.add_argument("--max-edges",   type=int, default=200_000,
                   help="Max edges per relation type sampled for R-GCN.")

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

    # ── 2. Chronological split + causal session prefixes ─────────────────────
    tqdm.write("[2/5] Chronological train/val/test split …")
    inter = build_interactions(data, args.val_ratio, args.test_ratio,
                               max_prefix=args.max_seq_len)
    t_val, t_test = inter.t_val, inter.t_test

    boundaries = snapshot_boundaries(inter.time, t_val, t_test, args.snapshot_hours)
    snap_of    = assign_snapshots(inter.time, boundaries)

    train_rows = inter.rows(SPLIT_TRAIN)
    if args.warmup_hours > 0:
        warm_end   = inter.time.min() + int(args.warmup_hours * 3_600_000)
        train_rows = train_rows[inter.time[train_rows] >= warm_end]
    val_rows   = inter.rows(SPLIT_VAL)
    test_rows  = inter.rows(SPLIT_TEST)
    tqdm.write(f"      {len(boundaries)} snapshot boundaries every {args.snapshot_hours:g}h  "
               f"train rows={len(train_rows):,}")

    # ── 3. Timed HKG — load from cache or build ──────────────────────────────
    # Built once over the whole log with a timestamp on every behavioural
    # edge; the model only ever sees time-filtered snapshots of it.
    tqdm.write("[3/5] Preparing HKG …")
    bundle: HKGBundle | None = None
    if cache_dir and not args.rebuild_hkg:
        bundle = load_hkg(cache_dir, data)

    if bundle is None:
        tqdm.write("      No valid cache — building from CSV files …")
        t0     = time.time()
        bundle = HKGConstructor(data, device="cpu").build()
        tqdm.write(f"      HKG built in {time.time()-t0:.1f}s")
        if cache_dir:
            save_hkg(bundle, cache_dir)
    else:
        tqdm.write(f"      Loaded from cache  n_videos={bundle.n_videos:,}  n_sessions={bundle.n_sessions:,}")

    if args.model_type == "single":
        rel_builder = lambda b: build_full_relation_edge_index(b, torch.device("cpu"), args.max_edges)
    else:
        rel_builder = lambda b: build_relation_edge_index(b.structural_graph, torch.device("cpu"), args.max_edges)
    needed = set(np.unique(snap_of[np.r_[train_rows, val_rows, test_rows]]).tolist())
    t0 = time.time()
    store = SnapshotStore(bundle, data, boundaries, rel_builder, needed=needed)
    tqdm.write(f"      {len(needed)} snapshots prepared in {time.time()-t0:.1f}s")
    batcher = PrefixBatcher(inter)

    # Free large DataFrames — no longer needed once snapshots are prepared
    import gc
    data.log_standard  = None
    data.log_random    = None
    data.log_combined  = None
    data.session_map   = None
    gc.collect()

    # ── 4. Build model ────────────────────────────────────────────────────────
    tqdm.write(f"[4/5] Building model  (type={args.model_type}) …")

    use_rg = not args.no_recency_gate
    if args.model_type == "single":
        model = build_single_model(data, bundle, args.hidden_dim, args.out_dim, device,
                                   use_recency_gate=use_rg,
                                   num_layers=args.gnn_layers)
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
        elif args.model_type == "single":
            tqdm.write("WARNING: --multi-gpu applies to the dual model only; ignoring.")
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

    split_kw = dict(snap_of=snap_of, store=store, batcher=batcher,
                    model_type=args.model_type, use_amp=use_amp, no_ips=args.no_ips)

    # AMP scaler (no-op on non-CUDA)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Each run gets its own output + checkpoint directory (avoids cross-run
    # checkpoint collisions when two runs share the same cache_dir).
    out_dir = Path(args.output_dir) / make_run_name(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoints are only loaded for --eval-only; training always starts from
    # fresh weights so a stale checkpoint from an earlier run cannot leak in.
    best_auc = 0.0
    ckpt_file = out_dir / _ckpt_filename(args.model_type)
    if args.eval_only:
        best_auc = load_checkpoint(model, device, out_dir, args.model_type)
        tqdm.write("--eval-only: skipping training, loading checkpoint.")
    else:
        # ── 5. Training loop ──────────────────────────────────────────────────
        tqdm.write(f"[5/5] Training  ({len(train_rows):,} rows/epoch × {args.epochs} epochs) …")
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.1,
        )

        history: list[dict] = []
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

            train_m = run_split(model, train_rows, batch_size=args.batch_size,
                                optimizer=optimizer, scaler=scaler, split="train", **split_kw)
            val_m   = run_split(model, val_rows, batch_size=args.batch_size * 2,
                                optimizer=None, scaler=None, split="val", **split_kw)
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
                f"  val    loss={val_m.loss:.4f}  AUC={val_m.auc:.4f}  "
                f"AP={val_m.ap:.4f}  nDCG@10={val_m.ndcg10:.4f}\n"
                f"  lr={lr_now:.2e}  elapsed={elapsed:.1f}s{gpu_str}"
            )
            tqdm.write(summary)

            epoch_bar.set_postfix({
                "tr_auc": f"{train_m.auc:.4f}",
                "va_auc": f"{val_m.auc:.4f}",
                "loss":   f"{train_m.loss:.4f}",
            })

            history.append({
                "epoch": epoch, "train": asdict(train_m),
                "val":   asdict(val_m), "lr": lr_now, "time_s": round(elapsed, 2),
            })

            # Model selection on validation only — test is never consulted here
            if val_m.auc > best_auc:
                best_auc = val_m.auc
                save_checkpoint(model, args, best_auc, out_dir)
                tqdm.write(f"  ★ new best val AUC={best_auc:.4f}  (checkpoint saved)")

        epoch_bar.close()

        # Save history
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        logger.info("Training history -> %s", out_dir / "history.json")

    # ── 6. Final evaluation with the val-selected checkpoint ─────────────────
    print()
    tqdm.write("── Final evaluation (val-selected checkpoint) ──────────────")
    if ckpt_file.exists():
        load_checkpoint(model, device, out_dir, args.model_type)

    final_train = run_split(model, train_rows, batch_size=args.batch_size * 2,
                            optimizer=None, scaler=None, split="train", **split_kw)
    final_val   = run_split(model, val_rows, batch_size=args.batch_size * 2,
                            optimizer=None, scaler=None, split="val", **split_kw)
    final_test  = run_split(model, test_rows, batch_size=args.batch_size * 2,
                            optimizer=None, scaler=None, split="test", **split_kw)

    # ── 7. Print metrics table ────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"  CVR Metrics  —  KuaiRand-{args.scale.upper()}  [{args.model_type.upper()} GNN]")
    print("=" * 62)
    hdr = f"  {'Metric':<12} {'Train':>10} {'Val':>10} {'Test':>10}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for name, tr, va, te in [
        ("Loss",    final_train.loss,    final_val.loss,    final_test.loss),
        ("AUC-ROC", final_train.auc,     final_val.auc,     final_test.auc),
        ("AP",      final_train.ap,      final_val.ap,      final_test.ap),
        ("LogLoss", final_train.logloss, final_val.logloss, final_test.logloss),
        ("nDCG@10", final_train.ndcg10,  final_val.ndcg10,  final_test.ndcg10),
    ]:
        print(f"  {name:<12} {tr:>10.4f} {va:>10.4f} {te:>10.4f}")

    print()
    print(f"  Train samples  : {final_train.n_samples:>12,}")
    print(f"  Val   samples  : {final_val.n_samples:>12,}")
    print(f"  Test  samples  : {final_test.n_samples:>12,}")
    print(f"  Train pos rate : {final_train.n_pos / max(final_train.n_samples,1):>12.3f}")
    print(f"  Val   pos rate : {final_val.n_pos   / max(final_val.n_samples, 1):>12.3f}")
    print(f"  Test  pos rate : {final_test.n_pos  / max(final_test.n_samples, 1):>12.3f}")
    print(f"  Best val AUC   : {best_auc:>12.4f}")
    print("=" * 62 + "\n")

    # ── 8. Save final metrics ─────────────────────────────────────────────────
    # Same run-specific directory as history.json and the checkpoint
    (out_dir / "final_metrics.json").write_text(json.dumps({
        "train": asdict(final_train),
        "val":   asdict(final_val),
        "test":  asdict(final_test),
        "args":  vars(args),
        "best_val_auc": best_auc,
        "t_val":  t_val,
        "t_test": t_test,
    }, indent=2))
    logger.info("Final metrics -> %s", out_dir / "final_metrics.json")
    logger.info("Done.")


if __name__ == "__main__":
    main()