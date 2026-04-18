"""
train_kgat.py
=============
Training script for the KGAT baseline on KuaiRand-1K.

Reuses the Framework pipeline (data loading, HKG construction, temporal split,
InteractionDataset, run_epoch, compute_metrics) from main.py unchanged.
Only the model construction and encode step differ from the HUG runs.

Architecture
------------
KGAT propagates over the structural subgraph (same 7-relation edge index as
StructuralGNN) using TransR-style relational attention, then predicts CVR via
concat(user_emb, video_emb) → MLP — no sequential encoder, no alignment module.

Run name convention
-------------------
1k_kgat_ips{0|1}_L{n}_h{H}d{D}[_{tag}]

Examples
    1k_kgat_ips1_L2_h128d64          default
    1k_kgat_ips0_L2_h128d64          --no-ips
    1k_kgat_ips1_L3_h128d64          --n-layers 3
    1k_kgat_ips1_L2_h256d128_large   --hidden-dim 256 --out-dim 128 --run-tag large

Usage
-----
    python Baselines/KGAT/train_kgat.py \\
        --data-dir /path/to/KuaiRand-1K/data \\
        --cache-dir ./cache/1k \\
        --out-dir   ./runs
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Resolve Framework on sys.path ─────────────────────────────────────────────
_HERE      = Path(__file__).resolve().parent          # Baselines/KGAT/
_FRAMEWORK = _HERE.parent.parent / "Framework"        # Framework/
if str(_FRAMEWORK) not in sys.path:
    sys.path.insert(0, str(_FRAMEWORK))

from data_loader import KuaiRandLoader
from hkg_constructor import HKGConstructor
import main as fw                                       # Framework/main.py
from main import (
    InteractionDataset,
    Metrics,
    build_relation_edge_index,
    compute_metrics,
    load_hkg,
    run_epoch,
    save_hkg,
    set_seed,
    temporal_split,
)

from kgat_model import KGATModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="KGAT baseline — KuaiRand CVR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",   type=str, required=True)
    p.add_argument("--cache-dir",  type=str, default=None,
                   help="HKG cache directory. Defaults to ./cache/1k. "
                        "Set 'none' to disable.")
    p.add_argument("--out-dir",    type=str, default="./runs")
    p.add_argument("--scale",      type=str, default="1k")

    p.add_argument("--device",    type=str, default=None)
    p.add_argument("--no-amp",    action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--rebuild-hkg", action="store_true")
    p.add_argument("--seed",      type=int, default=42)

    p.add_argument("--min-interactions", type=int,   default=10)
    p.add_argument("--test-ratio",       type=float, default=0.2)
    p.add_argument("--max-seq-len",      type=int,   default=50)
    p.add_argument("--max-edges",        type=int,   default=200_000)

    # KGAT-specific model hypers
    p.add_argument("--hidden-dim", type=int,   default=128)
    p.add_argument("--out-dim",    type=int,   default=64)
    p.add_argument("--n-layers",   type=int,   default=2,
                   help="Number of KGAT message-passing layers.")
    p.add_argument("--dropout",    type=float, default=0.2)

    # Ablation / sensitivity tags
    p.add_argument("--no-ips",  action="store_true",
                   help="Disable IPS weighting (use uniform weights).")
    p.add_argument("--run-tag", type=str, default=None,
                   help="Optional suffix appended to the run directory name.")

    # Training
    p.add_argument("--epochs",       type=int,   default=20)
    p.add_argument("--batch-size",   type=int,   default=2048)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)

    return p.parse_args()


def make_run_name(args: argparse.Namespace) -> str:
    ips_flag = "0" if args.no_ips else "1"
    name     = f"{args.scale}_kgat_ips{ips_flag}_L{args.n_layers}_h{args.hidden_dim}d{args.out_dim}"
    if args.run_tag:
        name += f"_{args.run_tag}"
    return name


def resolve_device(requested: str | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_cache_dir(args: argparse.Namespace) -> Path | None:
    if args.cache_dir and args.cache_dir.lower() == "none":
        return None
    return Path(args.cache_dir or f"./cache/{args.scale}")


# ── Model factory ─────────────────────────────────────────────────────────────

def build_kgat_model(
    data,
    bundle,
    hidden_dim: int,
    out_dim:    int,
    n_layers:   int,
    dropout:    float,
    device:     torch.device,
) -> KGATModel:
    """
    Construct KGATModel with dimensions derived from HKG node features.

    Feature dimensions must exactly match what StructuralGNN uses in
    Framework/main.py::build_model so the same graph tensors are read.
    """
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

    video_feat_dim    = sg["video"].x.shape[1]
    author_feat_dim   = sg["author"].x.shape[1]
    category_feat_dim = sg["category"].x.shape[1]

    model = KGATModel(
        user_cont_dim     = len(user_cont_cols),
        user_onehot_vocab = onehot_vocabs,
        video_feat_dim    = video_feat_dim,
        author_feat_dim   = author_feat_dim,
        category_feat_dim = category_feat_dim,
        hidden_dim        = hidden_dim,
        out_dim           = out_dim,
        n_relations       = 7,
        n_layers          = n_layers,
        dropout           = dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("KGATModel built  params=%s  device=%s  layers=%d  H=%d  D=%d",
                f"{n_params:,}", device, n_layers, hidden_dim, out_dim)
    return model


# ── Encode step (KGAT-specific) ───────────────────────────────────────────────

def encode_kgat(
    model:     KGATModel,
    sg,                           # structural_graph HeteroData (CPU)
    rel_ei:    torch.Tensor,      # [2, E] on device
    rel_t:     torch.Tensor,      # [E]    on device
) -> dict:
    """
    Run KGAT propagation over the full structural subgraph with no gradients.

    Returns CPU tensors {'s_user': [N_u, D], 's_video': [N_v, D]}.
    """
    tqdm.write("      Encoding graph with KGAT …")
    t0  = time.time()
    emb = model.encode_graph(sg, rel_ei, rel_t)
    emb_cpu = {k: v.cpu() for k, v in emb.items()}
    tqdm.write(
        f"      Encoded in {time.time()-t0:.1f}s  "
        f"s_user={emb_cpu['s_user'].shape}  s_video={emb_cpu['s_video'].shape}"
    )
    return emb_cpu


# ── Checkpoint helpers ────────────────────────────────────────────────────────

_CKPT_NAME = "best_model_kgat.pt"


def save_ckpt(model: KGATModel, out_dir: Path, best_auc: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": {k: v.cpu() for k, v in model.state_dict().items()},
         "best_auc": best_auc},
        out_dir / _CKPT_NAME,
    )
    logger.info("Checkpoint saved  AUC=%.4f", best_auc)


def load_ckpt(model: KGATModel, out_dir: Path, device: torch.device) -> float:
    path = out_dir / _CKPT_NAME
    if not path.exists():
        return 0.0
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict({k: v.to(device) for k, v in ckpt["model_state"].items()})
    auc = float(ckpt.get("best_auc", 0.0))
    logger.info("Checkpoint loaded  AUC=%.4f", auc)
    return auc


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args      = parse_args()
    device    = resolve_device(args.device)
    cache_dir = resolve_cache_dir(args)
    use_amp   = (not args.no_amp) and (device.type == "cuda")
    run_name  = make_run_name(args)
    out_dir   = Path(args.out_dir) / run_name

    set_seed(args.seed)

    print()
    print("╔" + "═" * 60 + "╗")
    print(f"║  KGAT Baseline  —  {args.scale.upper():<38} ║")
    print("╠" + "═" * 60 + "╣")
    print(f"║  data_dir  : {str(args.data_dir):<46} ║")
    print(f"║  device    : {str(device):<46} ║")
    print(f"║  layers    : {args.n_layers:<46} ║")
    print(f"║  hidden    : {args.hidden_dim}  out: {args.out_dim:<40} ║")
    print(f"║  run_name  : {run_name:<46} ║")
    print(f"║  no_ips    : {str(args.no_ips):<46} ║")
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

    dl_kwargs    = dict(num_workers=0, pin_memory=(device.type == "cuda"))
    train_loader = DataLoader(InteractionDataset(train_arr),
                              batch_size=args.batch_size, shuffle=True, **dl_kwargs)
    test_loader  = DataLoader(InteractionDataset(test_arr),
                              batch_size=args.batch_size * 2, shuffle=False, **dl_kwargs)
    del train_arr, test_arr

    # ── 3. HKG — load from cache or build ────────────────────────────────────
    tqdm.write("[3/5] Preparing HKG …")
    bundle = None
    if cache_dir and not args.rebuild_hkg:
        bundle = load_hkg(cache_dir)

    if bundle is None:
        tqdm.write("      No cache found — building from CSV files …")
        t0     = time.time()
        bundle = HKGConstructor(data, device="cpu",
                                max_seq_len=args.max_seq_len).build()
        tqdm.write(f"      HKG built in {time.time()-t0:.1f}s")
        if cache_dir:
            save_hkg(bundle, cache_dir)
    else:
        tqdm.write(f"      Loaded from cache  n_videos={bundle.n_videos:,}  "
                   f"n_sessions={bundle.n_sessions:,}")

    import gc
    data.log_standard = None
    data.log_random   = None
    data.log_combined = None
    data.session_map  = None
    gc.collect()

    sg = bundle.structural_graph

    tqdm.write("      Building relation edge index …")
    rel_ei, rel_t = build_relation_edge_index(sg, device, args.max_edges)
    tqdm.write(f"      Relation edges: {rel_ei.shape[1]:,}")

    # KGAT has no session encoder; pass n_sess≥1 to satisfy run_epoch's clamp
    n_sess = max(bundle.n_sessions, 1)

    # ── 4. Build model ────────────────────────────────────────────────────────
    tqdm.write("[4/5] Building KGATModel …")
    model = build_kgat_model(
        data, bundle,
        hidden_dim = args.hidden_dim,
        out_dim    = args.out_dim,
        n_layers   = args.n_layers,
        dropout    = args.dropout,
        device     = device,
    )

    best_auc = 0.0
    if args.eval_only:
        best_auc = load_ckpt(model, out_dir, device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    ) if not args.eval_only else None

    scaler = (
        torch.cuda.amp.GradScaler()
        if use_amp and not args.eval_only
        else None
    )

    # ── 5. Training loop ──────────────────────────────────────────────────────
    tqdm.write(f"[5/5] Training for {args.epochs} epochs …")
    out_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []

    if args.eval_only:
        emb = encode_kgat(model, sg, rel_ei, rel_t)
        test_m = run_epoch(model, test_loader, emb, n_sess,
                           device, None, None, "test", use_amp, no_ips=args.no_ips)
        print(test_m)
        (out_dir / "final_metrics.json").write_text(
            json.dumps({"test": asdict(test_m)}, indent=2))
        return

    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        tqdm.write(f"\n── Epoch {epoch}/{args.epochs} ──────────────────────")

        # Encode graph (no grad) — returns CPU tensors
        emb = encode_kgat(model, sg, rel_ei, rel_t)

        train_m = run_epoch(model, train_loader, emb, n_sess,
                            device, optimizer, scaler, "train", use_amp,
                            no_ips=args.no_ips)
        test_m  = run_epoch(model, test_loader,  emb, n_sess,
                            device, None, None, "test", use_amp,
                            no_ips=args.no_ips)

        del emb   # free embedding matrices before next encode

        epoch_time = time.time() - t_ep
        print(f"  Epoch {epoch:3d}  {epoch_time:.0f}s")
        print(f"  {train_m}")
        print(f"  {test_m}")

        history.append({
            "epoch": epoch,
            "train": asdict(train_m),
            "test":  asdict(test_m),
        })

        if test_m.auc > best_auc:
            best_auc = test_m.auc
            save_ckpt(model, out_dir, best_auc)

        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    # ── Save final metrics ────────────────────────────────────────────────────
    # Reload best checkpoint and re-evaluate
    load_ckpt(model, out_dir, device)
    emb = encode_kgat(model, sg, rel_ei, rel_t)
    train_final = run_epoch(model, train_loader, emb, n_sess,
                            device, None, None, "train", use_amp,
                            no_ips=args.no_ips)
    test_final  = run_epoch(model, test_loader,  emb, n_sess,
                            device, None, None, "test", use_amp,
                            no_ips=args.no_ips)
    del emb

    final = {"train": asdict(train_final), "test": asdict(test_final)}
    (out_dir / "final_metrics.json").write_text(json.dumps(final, indent=2))

    print()
    print("═" * 62)
    print(f"  KGAT run complete  →  {out_dir}")
    print(f"  Best AUC : {best_auc:.4f}")
    print(f"  {train_final}")
    print(f"  {test_final}")
    print("═" * 62)


if __name__ == "__main__":
    main()
