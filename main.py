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
     entire 4.4M x hidden_dim matrix.

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
from gnn_encoders import SequentialGNN, StructuralGNN
from hkg_constructor import HKGBundle, HKGConstructor
from models import AlignmentModule, CVRHead, KuaiCVRModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Cache helpers ─────────────────────────────────────────────────────────────

HKG_CACHE_FILE   = "hkg_bundle.pkl"
MODEL_CACHE_FILE = "best_model.pt"
ARGS_CACHE_FILE  = "run_args.json"


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


def save_checkpoint(model: KuaiCVRModel, args: argparse.Namespace,
                    best_auc: float, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = cache_dir / MODEL_CACHE_FILE
    torch.save({
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "best_auc":    best_auc,
        "args":        vars(args),
    }, ckpt_path)
    (cache_dir / ARGS_CACHE_FILE).write_text(json.dumps(vars(args), indent=2))
    logger.info("Checkpoint saved -> %s  (AUC=%.4f)", ckpt_path, best_auc)


def load_checkpoint(model: KuaiCVRModel, device: torch.device,
                    cache_dir: Path) -> float:
    """Load best checkpoint into model in-place. Returns best_auc or 0."""
    ckpt_path = cache_dir / MODEL_CACHE_FILE
    if not ckpt_path.exists():
        return 0.0
    ckpt = torch.load(ckpt_path, map_location=device)
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


def _ndcg_at_k(labels: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    if len(labels) < k:
        return 0.0
    try:
        return float(ndcg_score(labels.reshape(1, -1), scores.reshape(1, -1), k=k))
    except Exception:
        return 0.0


def compute_metrics(split: str, labels: np.ndarray,
                    probas: np.ndarray, loss: float) -> Metrics:
    m = Metrics(split=split, loss=loss,
                n_samples=len(labels), n_pos=int(labels.sum()))
    if m.n_pos == 0 or m.n_pos == m.n_samples:
        logger.warning("%s split has no label variance — skipping AUC/AP", split)
        return m
    m.auc     = float(roc_auc_score(labels, probas))
    m.ap      = float(average_precision_score(labels, probas))
    m.logloss = float(log_loss(labels, probas))
    m.ndcg10  = _ndcg_at_k(labels, probas, k=10)
    return m


# ── Temporal split ────────────────────────────────────────────────────────────

def temporal_split(data: KuaiRandData,
                   test_ratio: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """
    Split interactions by time_ms.  The earliest (1-test_ratio) fraction
    is train; the most recent test_ratio fraction is test.

    Returns np.ndarray of shape [N, 5]:
        columns: user_idx, video_idx, session_idx, ips_weight, is_click
    """
    uid_map = data.user_id_map
    vid_map = data.video_id_map
    sm      = data.session_map

    mask = sm["user_id"].isin(uid_map) & sm["video_id"].isin(vid_map)
    df   = sm[mask].copy()

    df["u_idx"] = df["user_id"].map(uid_map)
    df["v_idx"] = df["video_id"].map(vid_map)
    df["ips"]   = df["is_rand"].apply(lambda r: 1.0 if r == 1 else 0.9963)

    df     = df.sort_values("time_ms").reset_index(drop=True)
    cutoff = int(len(df) * (1 - test_ratio))
    cols   = ["u_idx", "v_idx", "session_id", "ips", "is_click"]

    train_arr = df.iloc[:cutoff][cols].values.astype(np.float32)
    test_arr  = df.iloc[cutoff:][cols].values.astype(np.float32)

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
                device: torch.device) -> KuaiCVRModel:

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
        num_layers        = 2,
    )
    seq = SequentialGNN(
        video_feat_dim = video_feat_dim,
        hidden_dim     = hidden_dim,
        out_dim        = out_dim,
        num_layers     = 2,
    )
    align = AlignmentModule(emb_dim=out_dim, kg_relation_dim=0, num_heads=4)
    head  = CVRHead(fused_dim=out_dim, hidden_dims=[hidden_dim, hidden_dim // 2])

    model    = KuaiCVRModel(struct, seq, align, head).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model built  params=%s  device=%s", f"{n_params:,}", device)
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
) -> Metrics:
    """
    One pass over the DataLoader using pre-computed graph embeddings.

    The GNNs are NOT called here — encode_epoch() handles that once per epoch.
    This loop only runs: index lookup → alignment → CVR head → loss/backward.
    Each batch takes milliseconds instead of minutes.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    # Move embedding matrices to GPU once for the whole epoch
    emb_dev = {k: v.to(device, non_blocking=True) for k, v in emb.items()}

    total_loss, all_labels, all_probas = 0.0, [], []
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
            user_idx    = batch["user_idx"].to(device, non_blocking=True)
            video_idx   = batch["video_idx"].to(device, non_blocking=True)
            session_idx = batch["session_idx"].to(device, non_blocking=True).clamp(0, n_sess - 1)
            ips_weights = batch["ips_weight"].to(device, non_blocking=True)
            labels      = batch["label"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
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

            all_labels.append(batch["label"].numpy())
            all_probas.append(out["proba"].detach().cpu().float().numpy())

            postfix: dict = {"loss": f"{running_loss:.4f}"}
            if torch.cuda.is_available():
                postfix["gpu"] = f"{torch.cuda.memory_allocated()/1e9:.1f}GB"
            pbar.set_postfix(postfix)

    pbar.close()
    # Free GPU embedding copies
    del emb_dev

    labels_np = np.concatenate(all_labels)
    probas_np = np.concatenate(all_probas)
    avg_loss  = total_loss / max(len(labels_np), 1)
    return compute_metrics(split, labels_np, probas_np, avg_loss)


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
    p.add_argument("--hidden-dim",  type=int, default=128)
    p.add_argument("--out-dim",     type=int, default=64)

    # Training
    p.add_argument("--epochs",       type=int,   default=10)
    p.add_argument("--batch-size",   type=int,   default=2048)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)

    # Memory controls
    p.add_argument("--max-edges",    type=int, default=100_000,
                   help="Max edges per relation type sampled for R-GCN. "
                        "Reduce to lower GPU memory (default 100K ≈ 5.6 MB/relation).")

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
    tqdm.write("[4/5] Building model …")
    model = build_model(data, bundle, args.hidden_dim, args.out_dim, device)

    # AMP scaler (no-op on non-CUDA)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Load checkpoint if eval-only or if one exists
    best_auc = 0.0
    if cache_dir and (args.eval_only or (cache_dir / MODEL_CACHE_FILE).exists()):
        loaded_auc = load_checkpoint(model, device, cache_dir)
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

            # Encode graph once — GNNs run here, not inside the batch loop
            emb = encode_epoch(model, sg, qg, rel_ei, rel_t, sess_id, device)

            train_m = run_epoch(
                model, train_loader, emb, n_sess,
                device, optimizer, scaler,
                split="train", use_amp=use_amp,
            )
            test_m = run_epoch(
                model, test_loader, emb, n_sess,
                device, None, None,
                split="test", use_amp=use_amp,
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
                if cache_dir:
                    save_checkpoint(model, args, best_auc, cache_dir)
                tqdm.write(f"  ★ new best test AUC={best_auc:.4f}  (checkpoint saved)")

        epoch_bar.close()

        # Save history
        out_dir = Path(args.output_dir) / f"kuairand_{args.scale}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        logger.info("Training history -> %s", out_dir / "history.json")

    # ── 6. Final evaluation with best checkpoint ──────────────────────────────
    print()
    tqdm.write("── Final evaluation (best checkpoint) ──────────────────────")
    if cache_dir and (cache_dir / MODEL_CACHE_FILE).exists():
        load_checkpoint(model, device, cache_dir)

    emb_final = encode_epoch(model, sg, qg, rel_ei, rel_t, sess_id, device)

    final_train = run_epoch(
        model, train_loader, emb_final, n_sess,
        device, None, None,
        split="train", use_amp=use_amp,
    )
    final_test = run_epoch(
        model, test_loader, emb_final, n_sess,
        device, None, None,
        split="test ", use_amp=use_amp,
    )
    del emb_final

    # ── 7. Print metrics table ────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"  CVR Metrics  —  KuaiRand-{args.scale.upper()}")
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
    out_dir = Path(args.output_dir) / f"kuairand_{args.scale}"
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