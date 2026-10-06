"""
HUG training pipeline (spec 03 model, spec 04 harness integration)
-------------------------------------------------------------------
Data preparation from the shared feature functions, the snapshot-contiguous
training loop (early stopping on val AUC, ReduceLROnPlateau, resume), and
evaluation with bucket metrics and per-row prediction dumps.

Entry point: run(args), called by main.py for --model-type hug.

Run directory layout (--run-dir):
    final_metrics.json   val / holdout / (test) metrics, config, history
    val_preds.npz        row_id, user, label, score
    test_preds.npz       final_eval only
    best.pt              best-on-val weights
    last.pt              resume point (deleted on completion)
    hug.log
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from data_loader import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL, KuaiRandData, KuaiRandLoader
from features import (
    HUG_VIDEO_CAT_COLS, asof_video_statistics, bucket_keys, click_history,
    holdout_rows, hour_of_day, video_categoricals,
)
from hkg_constructor import HKG_BUILD_VERSION, HKGBundle, HKGConstructor
from hug import HistoryTransformer, HUGModel, InputEncoder, RelLightGCN
from metrics import Metrics, compute_metrics
from runtime import Progress, git_state, is_quiet, log_versions, set_determinism
from temporal import (
    Interactions, SnapshotStore, assign_snapshots, build_interactions, snapshot_boundaries,
)
from test_guard import authorize_test_access

logger = logging.getLogger(__name__)


# ── Relations: the 14 structural ones (spec 01) + next_in_session ─────────────

def hug_relations():
    """(forward_ids, reverse_ids, [(src_type, dst_type)] by relation id)."""
    from main import REL_MAP, REVERSE_REL_MAP
    seq_et = ("video", "next_in_session", "video")
    forward = dict(REL_MAP)
    forward[seq_et] = 2 * len(REL_MAP)                    # one-directional
    types = [None] * (2 * len(REL_MAP) + 1)
    for et, rid in forward.items():
        types[rid] = (et[0], et[2])
    for et, rid in REVERSE_REL_MAP.items():
        types[rid] = (et[2], et[0])
    return forward, REVERSE_REL_MAP, types


def arm_name(args) -> str:
    if args.no_graph and args.no_seq:
        return "N0"
    if args.no_graph:
        return "N1"
    if args.no_seq:
        return "N2"
    if args.freeze_graph:
        return "N4-frozen"
    if args.cl_weight == 0:
        return "N3"
    return "N4-gtok" if args.graph_tokens else "N4"


# ── Data ──────────────────────────────────────────────────────────────────────

@dataclass
class HugData:
    inter:      Interactions
    boundaries: np.ndarray
    snap:       np.ndarray              # [N] snapshot per row
    train_rows: np.ndarray              # rows used as training targets
    holdout:    np.ndarray
    val_rows:   np.ndarray
    test_rows:  np.ndarray
    hist_seq:   np.ndarray              # history token rows (row ids), see click_history
    hist_start: np.ndarray
    hist_end:   np.ndarray
    session:    np.ndarray              # [N]
    ctx_cat:    np.ndarray              # [N, C] long
    ctx_num:    np.ndarray              # [N, K] float32
    buckets:    dict
    store:      SnapshotStore | None
    bundle:     HKGBundle
    node_data:  dict                    # tensors for InputEncoder.set_node_data
    vocab:      dict                    # sizes and OOV shares
    fingerprint: dict
    video_cat_raw: object = None        # shared categorical strings per video (node order)


def _vocab(counts: np.ndarray, m: int) -> np.ndarray:
    """Item index → vocabulary row (0 = OOV) for items with >= m training-window interactions."""
    keep = counts >= m
    out = np.zeros(len(counts), dtype=np.int64)
    out[keep] = np.arange(1, keep.sum() + 1)
    return out


def data_fingerprint(inter: Interactions) -> dict:
    return {
        "t_val": int(inter.t_val), "t_test": int(inter.t_test),
        "rows": {name: int((inter.split == code).sum())
                 for name, code in (("train", SPLIT_TRAIN), ("val", SPLIT_VAL), ("test", SPLIT_TEST))},
        "n_users": int(inter.user.max()) + 1,
    }


def prepare(args, data: KuaiRandData | None = None, bundle: HKGBundle | None = None) -> HugData:
    if data is None:
        data = KuaiRandLoader(args.data_dir, min_interactions=args.min_interactions,
                              filter_ads=True).load()
    inter = build_interactions(data, args.val_ratio, args.test_ratio, max_prefix=args.max_seq_len)
    N, t_val = len(inter.time), inter.t_val

    boundaries = snapshot_boundaries(inter.time, t_val, inter.t_test, args.snapshot_hours)
    snap = assign_snapshots(inter.time, boundaries)

    train_all = inter.rows(SPLIT_TRAIN)
    holdout = holdout_rows(train_all, args.holdout_frac, seed=0)
    train_rows = np.setdiff1d(train_all, holdout, assume_unique=True)
    warmup = args.snapshot_hours if args.warmup_hours is None else args.warmup_hours
    if warmup > 0:
        train_rows = train_rows[inter.time[train_rows] >= inter.time.min() + int(warmup * 3_600_000)]

    # History: last max_seq_len clicks strictly before each row (shared with the baselines)
    hist_seq, hs, he = click_history(inter.user, inter.time, np.arange(N), inter.label,
                                     args.max_seq_len)
    frame   = inter.frame
    session = frame["session_id"].to_numpy(np.int64)

    # Context: tab, hour, per-row as-of video statistics
    tab  = frame["tab"].fillna(0).to_numpy(np.int64)
    hour = hour_of_day(inter.time)
    ctx_cat = np.stack([tab, hour], axis=1)
    ctx_num = asof_video_statistics(frame)

    # Vocabularies from the training window only (time < t_val)
    tw = inter.time < t_val
    n_users  = len(data.user_id_map)
    n_videos = len(data.video_id_map)
    user_cnt  = np.bincount(inter.user[tw],  minlength=n_users)
    video_cnt = np.bincount(inter.video[tw], minlength=n_videos)
    user_vocab  = _vocab(user_cnt,  args.min_id_count)
    video_vocab = _vocab(video_cnt, args.min_id_count)

    vc = video_categoricals(data.video_basic)
    vc = vc[vc["video_id"].isin(data.video_id_map)]
    vc = vc.assign(node=vc["video_id"].map(data.video_id_map)).sort_values("node")
    cat_idx, cat_sizes = [], []
    for col in HUG_VIDEO_CAT_COLS:
        codes, uniq = pd_factorize(vc[col].to_numpy())
        code_cnt = np.bincount(codes, weights=video_cnt[vc["node"].to_numpy()], minlength=len(uniq))
        voc = _vocab(code_cnt, args.min_id_count)
        cat_idx.append(voc[codes])
        cat_sizes.append(int(voc.max()) + 1)
    video_cat = np.stack(cat_idx, axis=1)

    vocab = {
        "users": int(user_vocab.max()), "videos": int(video_vocab.max()),
        "categoricals": dict(zip(HUG_VIDEO_CAT_COLS, [s - 1 for s in cat_sizes])),
        "oov_share_train_rows": float((video_vocab[inter.video[train_rows]] == 0).mean()),
        "oov_share_val_rows":   float((video_vocab[inter.video[inter.rows(SPLIT_VAL)]] == 0).mean()),
    }
    logger.info("Vocabulary (min_id_count=%d): %s", args.min_id_count, vocab)

    # Graph: timed HKG + master relation index over the HUG relations
    if bundle is None:
        from main import load_hkg, resolve_cache_dir, save_hkg
        cache_dir = resolve_cache_dir(args)
        bundle = load_hkg(cache_dir, data) if cache_dir else None
        if bundle is None:
            bundle = HKGConstructor(data, device="cpu").build()
            if cache_dir:
                save_hkg(bundle, cache_dir)
    store = None
    real_test = args.eval_test and not args.max_steps
    rows_used = np.r_[train_rows, holdout, inter.rows(SPLIT_VAL),
                      inter.rows(SPLIT_TEST) if real_test else np.zeros(0, np.int64)]
    needed = set(np.unique(snap[rows_used]).tolist())
    if not args.no_graph:
        from main import _merged_edge_index
        fwd, rev, _ = hug_relations()
        master = _merged_edge_index(bundle.full_graph, fwd, rev)
        logger.info("HUG master relation index: %s edges", f"{master[0].shape[1]:,}")
        store = SnapshotStore(bundle, data, boundaries, master, needed=needed)
    else:
        store = SnapshotStore(bundle, data, boundaries, None, needed=needed)

    ug = bundle.full_graph["user"]
    node_data = {
        "user_vocab": torch.from_numpy(user_vocab),
        "user_x": ug.x.float().cpu(), "user_onehot": ug.onehot.long().cpu(),
        "video_vocab": torch.from_numpy(video_vocab),
        "video_cat": torch.from_numpy(video_cat),
        "cat_sizes": cat_sizes,
    }

    return HugData(
        inter=inter, boundaries=boundaries, snap=snap, train_rows=train_rows, holdout=holdout,
        val_rows=inter.rows(SPLIT_VAL), test_rows=inter.rows(SPLIT_TEST),
        hist_seq=hist_seq, hist_start=hs, hist_end=he, session=session,
        ctx_cat=ctx_cat, ctx_num=ctx_num,
        buckets=bucket_keys(inter.video, inter.time, he - hs, t_val, boundaries[snap], args.max_seq_len),
        store=store, bundle=bundle, node_data=node_data, vocab=vocab,
        fingerprint=data_fingerprint(inter),
        video_cat_raw=vc.reset_index(drop=True),
    )


def pd_factorize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd
    codes, uniq = pd.factorize(values, sort=True)
    return codes.astype(np.int64), uniq


class HugBatcher:
    """Row ids → model inputs (history right-aligned; rank 0 = most recent)."""

    def __init__(self, d: HugData, max_len: int) -> None:
        it = d.inter
        self.user, self.video = torch.from_numpy(it.user), torch.from_numpy(it.video)
        self.time  = torch.from_numpy(it.time)
        self.label = torch.from_numpy(it.label)
        self.sess  = torch.from_numpy(d.session)
        self.hs, self.he = torch.from_numpy(d.hist_start), torch.from_numpy(d.hist_end)
        self.hrow  = torch.from_numpy(d.hist_seq)
        self.ctx_cat = torch.from_numpy(d.ctx_cat)
        self.ctx_num = torch.from_numpy(d.ctx_num)
        self.offsets = torch.arange(-max_len, 0)
        self.rank = torch.arange(max_len - 1, -1, -1)

    def __call__(self, rows: np.ndarray) -> dict[str, torch.Tensor]:
        r   = torch.from_numpy(np.asarray(rows, dtype=np.int64))
        pos = self.he[r, None] + self.offsets
        ok  = pos >= self.hs[r, None]
        hr  = self.hrow[pos.clamp(min=0)]                         # history token row ids
        hist = torch.where(ok, self.video[hr], torch.full_like(hr, -1))
        gap  = HistoryTransformer.gap_bucket(self.time[r, None] - self.time[hr])
        same = (self.sess[hr] == self.sess[r, None]).long()
        return {
            "row": r, "user": self.user[r], "video": self.video[r], "label": self.label[r],
            "hist": hist, "hist_rank": self.rank.expand_as(hist),
            "hist_gap": torch.where(ok, gap, torch.zeros_like(gap)),
            "hist_sess": torch.where(ok, same, torch.zeros_like(same)),
            "ctx_cat": self.ctx_cat[r], "ctx_num": self.ctx_num[r],
        }


# ── Model ─────────────────────────────────────────────────────────────────────

def build(args, d: HugData) -> HUGModel:
    nd = d.node_data
    onehot_vocab = [int(nd["user_onehot"][:, i].max()) + 1 for i in range(nd["user_onehot"].shape[1])]
    video_feat_dim = next(iter(d.store.video_x.values())).shape[1]
    inp = InputEncoder(
        args.emb_dim, int(nd["user_vocab"].max()) + 1, nd["user_x"].shape[1], onehot_vocab,
        int(nd["video_vocab"].max()) + 1, video_feat_dim, nd["cat_sizes"],
        d.bundle.n_authors, d.bundle.n_categories,
    )
    inp.set_node_data(nd["user_vocab"], nd["user_x"], nd["user_onehot"],
                      nd["video_vocab"], nd["video_cat"])
    gcn = None
    if not args.no_graph:
        _, _, types = hug_relations()
        gcn = RelLightGCN(types, args.graph_layers, eps=args.cl_eps, cl_layer=args.cl_layer)
    seq = None if args.no_seq else HistoryTransformer(args.emb_dim, args.seq_layers,
                                                      args.max_seq_len, dropout=args.dropout)
    ctx_vocab = [int(d.ctx_cat[:, i].max()) + 1 for i in range(d.ctx_cat.shape[1])]
    return HUGModel(inp, gcn, seq, ctx_vocab, d.ctx_num.shape[1], args.emb_dim,
                    cl_weight=args.cl_weight, cl_temp=args.cl_temp, emb_l2=args.emb_l2,
                    freeze_graph=args.freeze_graph, graph_tokens=args.graph_tokens,
                    dropout=args.dropout)


def param_counts(model: HUGModel) -> dict:
    emb = sum(p.numel() for n, p in model.named_parameters() if isinstance(
        model.get_submodule(n.rsplit(".", 1)[0]), torch.nn.Embedding))
    total = sum(p.numel() for p in model.parameters())
    return {"total": total, "embedding": emb, "dense": total - emb}


# ── Snapshot handling ─────────────────────────────────────────────────────────

class Snapshots:
    """Enters snapshot k: per-relation edges into the GCN, video features to device."""

    def __init__(self, d: HugData, model: HUGModel, device: torch.device) -> None:
        self.d, self.model, self.device = d, model, device
        b = d.bundle
        self.offsets = {"user": 0, "video": b.n_users, "author": b.n_users + b.n_videos,
                        "category": b.n_users + b.n_videos + b.n_authors}
        self.counts = {"user": b.n_users, "video": b.n_videos,
                       "author": b.n_authors, "category": b.n_categories}
        self.current = None
        self.video_x = None

    def enter(self, k: int) -> torch.Tensor:
        if self.current != k:
            self.video_x = self.d.store.video_x[k].to(self.device)
            gcn = self.model.gcn
            if gcn is not None:
                ei, et = self.d.store.rel(k, self.device)
                edges = []
                for r, (st, dt) in enumerate(gcn.relations):
                    m = et == r
                    edges.append((ei[0, m] - self.offsets[st], ei[1, m] - self.offsets[dt]))
                gcn.set_graph(edges, self.counts)
            self.current = k
        return self.video_x


# ── Train / eval ──────────────────────────────────────────────────────────────

def _to(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def predict(model: HUGModel, d: HugData, rows: np.ndarray, snaps: Snapshots,
            batcher: HugBatcher, batch_size: int, device, quiet: bool,
            desc: str, max_snapshots: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Scores for `rows` (returned in the order of the returned row array)."""
    model.eval()
    out_rows, out_scores = [], []
    ks = np.unique(d.snap[rows])
    if max_snapshots is not None:
        ks = ks[:max_snapshots]
    prog = Progress(int(np.isin(d.snap[rows], ks).sum()), desc, quiet)
    for k in ks:
        sel = rows[d.snap[rows] == k]
        vx = snaps.enter(int(k))
        tables = model.graph_tables(vx, train=False) if model.gcn is not None else None
        for b0 in range(0, len(sel), batch_size):
            b = _to(batcher(sel[b0:b0 + batch_size]), device)
            b.pop("label")
            out_scores.append(model(b, vx, tables)["proba"].float().cpu().numpy())
            out_rows.append(sel[b0:b0 + batch_size])
            prog.update(len(out_rows[-1]))
    prog.close()
    return np.concatenate(out_rows), np.concatenate(out_scores)


def evaluate(split, model, d, rows, snaps, batcher, args, device, max_snapshots=None):
    r, s = predict(model, d, rows, snaps, batcher, args.batch_size * 2, device,
                   is_quiet(args.quiet), split, max_snapshots)
    labels = d.inter.label[r]
    eps = 1e-7
    loss = float(-np.mean(labels * np.log(np.clip(s, eps, 1)) + (1 - labels) * np.log(np.clip(1 - s, eps, 1))))
    m = compute_metrics(split, labels, s, loss, user_idxs=d.inter.user[r],
                        buckets={k: v[r] for k, v in d.buckets.items()})
    return m, {"row_id": r, "user": d.inter.user[r], "label": labels.astype(np.float32),
               "score": s.astype(np.float32)}


def _rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _set_rng_state(s: dict) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def make_optimizer(model: HUGModel, args) -> torch.optim.Optimizer:
    emb, dense = [], []
    for name, p in model.named_parameters():
        mod = model.get_submodule(name.rsplit(".", 1)[0])
        (emb if isinstance(mod, torch.nn.Embedding) else dense).append(p)
    return torch.optim.Adam([
        {"params": emb,   "weight_decay": 0.0},
        {"params": dense, "weight_decay": args.weight_decay},
    ], lr=args.lr)


def train(args, d: HugData, model: HUGModel, device, run_dir: Path) -> dict:
    """Snapshot-contiguous training with early stopping; returns training summary."""
    quiet = is_quiet(args.quiet)
    batcher = HugBatcher(d, args.max_seq_len)
    snaps = Snapshots(d, model, device)
    opt = make_optimizer(model, args)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)

    state = {"epoch": 0, "best_auc": -1.0, "best_epoch": None, "bad_epochs": 0,
             "history": [], "steps": 0}
    last = run_dir / "last.pt"
    if args.resume and last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        state = ck["state"]
        _set_rng_state(ck["rng"])
        logger.info("Resumed from %s at epoch %d", last, state["epoch"])

    train_snaps = np.unique(d.snap[d.train_rows])
    stop = False
    while state["epoch"] < args.max_epochs and not stop:
        epoch = state["epoch"] + 1
        t0 = time.time()
        model.train()
        tot, n_seen, cl_tot = 0.0, 0, 0.0
        prog = Progress(len(d.train_rows), f"train e{epoch}", quiet)
        for k in np.random.permutation(train_snaps):
            sel = np.random.permutation(d.train_rows[d.snap[d.train_rows] == k])
            vx = snaps.enter(int(k))
            for b0 in range(0, len(sel), args.batch_size):
                b = _to(batcher(sel[b0:b0 + args.batch_size]), device)
                tables = model.graph_tables(vx, train=True) if model.gcn is not None else None
                out = model(b, vx, tables, train=True)
                opt.zero_grad(set_to_none=True)
                out["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                n = len(b["user"])
                tot += out["bce"].item() * n
                cl_tot += out["cl"].item() * n
                n_seen += n
                state["steps"] += 1
                prog.update(n)
                prog.set_postfix({"bce": f"{tot / n_seen:.4f}", "cl": f"{cl_tot / n_seen:.4f}"})
                if args.max_steps and state["steps"] >= args.max_steps:
                    stop = True
                    break
            if stop:
                break
        prog.close()

        max_snaps = 2 if args.max_steps else None
        val_m, _ = evaluate("val", model, d, d.val_rows, snaps, batcher, args, device, max_snaps)
        sched.step(val_m.auc)
        rec = {"epoch": epoch, "train_bce": tot / max(n_seen, 1), "train_cl": cl_tot / max(n_seen, 1),
               "val": asdict(val_m), "lr": opt.param_groups[0]["lr"],
               "seconds": round(time.time() - t0, 1)}
        if model.gcn is not None and not model.freeze_graph:
            rec["relation_weights"] = model.gcn.relation_weights().cpu().tolist()
        state["history"].append(rec)
        logger.info("epoch %d  train_bce=%.4f  cl=%.4f  %s  (%.0fs)", epoch, rec["train_bce"],
                    rec["train_cl"], val_m, rec["seconds"])

        if val_m.auc > state["best_auc"]:
            state.update(best_auc=val_m.auc, best_epoch=epoch, bad_epochs=0)
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            state["bad_epochs"] += 1
            if state["bad_epochs"] >= args.patience:
                stop = True
        state["epoch"] = epoch
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "state": state, "rng": _rng_state()}, last)

    return state


def run(args) -> None:
    """Full HUG job: prepare → (train) → evaluate best on val/holdout (and test) → outputs."""
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(run_dir / "hug.log")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(fh)

    if args.eval_test:
        authorize_test_access(run_dir.parent, args.test_access_token,
                              args.i_know_this_touches_test, run_dir.name, args.config_hash)

    versions = log_versions()
    set_determinism(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    d = prepare(args)
    model = build(args, d).to(device)
    counts = param_counts(model)
    logger.info("HUG %s  params=%s", arm_name(args), counts)

    state = None
    if args.eval_only:
        ck = Path(args.checkpoint) if args.checkpoint else run_dir / "best.pt"
        model.load_state_dict(torch.load(ck, map_location=device, weights_only=False))
    else:
        state = train(args, d, model, device, run_dir)
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=False))

    batcher = HugBatcher(d, args.max_seq_len)
    snaps = Snapshots(d, model, device)
    max_snaps = 2 if args.max_steps else None
    val_m, val_p = evaluate("val", model, d, d.val_rows, snaps, batcher, args, device, max_snaps)
    np.savez(run_dir / "val_preds.npz", **val_p)
    hold_m, _ = evaluate("holdout", model, d, d.holdout, snaps, batcher, args, device, max_snaps)
    test_m = None
    dry_substitute = bool(args.eval_test and args.max_steps)
    if args.eval_test:
        # Dry runs exercise the final_eval path on validation rows: no test label is read
        rows = d.val_rows if dry_substitute else d.test_rows
        test_m, test_p = evaluate("test", model, d, rows, snaps, batcher, args, device,
                                  max_snaps if dry_substitute else None)
        np.savez(run_dir / "test_preds.npz", **test_p)
    logger.info("%s\n%s", val_m, hold_m)

    ckpt = run_dir / "best.pt"
    out = {
        "job": run_dir.name, "kind": "hug", "model": "HUG", "arm": arm_name(args),
        "seed": args.seed, "config_hash": args.config_hash,
        "val": asdict(val_m), "holdout": asdict(hold_m),
        "test": asdict(test_m) if test_m is not None else None,
        "test_is_dry_run_substitute": dry_substitute,
        "best_epoch": state["best_epoch"] if state else None,
        "history": state["history"] if state else None,
        "params": {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, type(None)))},
        "n_params": counts, "vocab": d.vocab, "fingerprint": d.fingerprint,
        "encode": None, "checkpoint_mb": round(ckpt.stat().st_size / 1e6, 1) if ckpt.exists() else None,
        "environment": versions, "hkg_build_version": HKG_BUILD_VERSION,
        "git": git_state(),
    }
    if model.gcn is not None:
        out["relation_weights"] = model.gcn.relation_weights().cpu().tolist()
    (run_dir / "final_metrics.json").write_text(json.dumps(out, indent=2, default=float))
    if not args.eval_only and (run_dir / "last.pt").exists():
        (run_dir / "last.pt").unlink()
    logger.info("Done → %s", run_dir)
    logging.getLogger().removeHandler(fh)
