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

from data_loader import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL, KuaiRandData
from datasets import DatasetBundle, GraphSpec, load_dataset
from features import bucket_keys, click_history, holdout_rows, view_evidence
from hkg_constructor import HKG_BUILD_VERSION, HKGBundle
from hug import HistoryTransformer, HUGModel, InputEncoder, RelLightGCN
from metrics import Metrics, compute_metrics
from runtime import Progress, git_state, is_quiet, log_versions, set_determinism
from temporal import Interactions, SnapshotStore, assign_snapshots, snapshot_boundaries
from test_guard import authorize_test_access

logger = logging.getLogger(__name__)


# Pre-window history tokens (MIND) carry this gap bucket; real gaps never reach it
PRE_WINDOW_GAP = HistoryTransformer.N_GAP_BUCKETS - 1


def hug_relations():
    """KuaiRand's HUG relation set (kept for callers of the pre-adapter API)."""
    from datasets.kuairand import hug_relations as rel
    fwd, rev, types, _ = rel()
    return fwd, rev, types


def arm_name(args) -> str:
    fusion = getattr(args, "fusion", "concat")
    if fusion != "concat":
        suffix = "-fixed" if fusion == "sparse" and args.k_schedule == "fixed" else ""
        return f"N4-{fusion}{suffix}"
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
    hist_seq:   np.ndarray              # history token ids (row ids, then pre-window tokens)
    hist_start: np.ndarray
    hist_end:   np.ndarray
    session:    np.ndarray              # [N]
    ctx_cat:    np.ndarray              # [N, C] long
    ctx_num:    np.ndarray              # [N, K] float32
    buckets:    dict
    store:      SnapshotStore | None
    graph:      GraphSpec
    node_data:  dict                    # tensors for InputEncoder.set_node_data
    vocab:      dict                    # sizes and OOV shares
    fingerprint: dict
    video_cat_raw: object = None        # shared categorical strings per video (node order)
    evidence:   np.ndarray | None = None  # [N, 2] causal per-view evidence (n_G, n_S)
    # history token table: token id → item, time, session, pre-window flag
    tok_item:   np.ndarray | None = None
    tok_time:   np.ndarray | None = None
    tok_sess:   np.ndarray | None = None
    tok_pre:    np.ndarray | None = None
    dataset:    str = "kuairand"


def _vocab(counts: np.ndarray, m: int) -> np.ndarray:
    """Item index → vocabulary row (0 = OOV) for items with >= m training-window interactions."""
    keep = counts >= m
    out = np.zeros(len(counts), dtype=np.int64)
    out[keep] = np.arange(1, keep.sum() + 1)
    return out


def data_fingerprint(inter: Interactions) -> dict:
    from datasets.base import data_fingerprint as fp
    return fp(inter)


def prepare(args, data: KuaiRandData | None = None, bundle: HKGBundle | None = None,
            ds: DatasetBundle | None = None) -> HugData:
    """HugData from a dataset adapter (spec 06).  `data`/`bundle` preload KuaiRand."""
    if ds is None:
        ds = load_dataset(args, data=data, hkg=bundle)
    inter = ds.inter
    N, t_val = len(inter.time), inter.t_val

    boundaries = snapshot_boundaries(inter.time, t_val, inter.t_test, args.snapshot_hours)
    snap = assign_snapshots(inter.time, boundaries)

    train_all = inter.rows(SPLIT_TRAIN)
    holdout = holdout_rows(train_all, args.holdout_frac, seed=0)
    train_rows = np.setdiff1d(train_all, holdout, assume_unique=True)
    warmup = args.snapshot_hours if args.warmup_hours is None else args.warmup_hours
    if warmup > 0:
        train_rows = train_rows[inter.time[train_rows] >= inter.time.min() + int(warmup * 3_600_000)]

    # History: last max_len known clicks (label_time < t) plus any pre-window clicks,
    # shared with the baselines.  Token ids: rows 0..N-1, then pre-window tokens.
    session = np.asarray(ds.session, np.int64)
    pw_item = np.zeros(0, np.int64)
    pre = None
    if ds.pre_window is not None and len(ds.pre_window[0]):
        pw_user, pw_item = (np.asarray(a, np.int64) for a in ds.pre_window)
        pre = (pw_user, N + np.arange(len(pw_user)))
    hist_seq, hs, he = click_history(inter.user, inter.time, np.arange(N), inter.label,
                                     args.max_seq_len, label_time=ds.label_time, pre_window=pre)
    n_pre = len(pw_item)
    tok_item = np.r_[inter.video, pw_item]
    tok_time = np.r_[inter.time, np.zeros(n_pre, np.int64)]
    tok_sess = np.r_[session, np.full(n_pre, -1, np.int64)]
    tok_pre  = np.r_[np.zeros(N, bool), np.ones(n_pre, bool)]

    ctx_cat, ctx_num = ds.ctx_cat, ds.ctx_num
    # Per-view evidence for fusion heads (spec 05); n_S's history term is the length of
    # the encoder's own history (hist_seq[hs:he])
    evidence = view_evidence(inter.video, inter.time, he - hs, ds.metadata_links)

    # Vocabularies from the training window only (time < t_val; pre-window clicks precede it)
    tw = inter.time < t_val
    n_users, n_videos = ds.n_users, ds.n_items
    user_cnt  = np.bincount(inter.user[tw],  minlength=n_users)
    video_cnt = np.bincount(inter.video[tw], minlength=n_videos)
    if n_pre:
        video_cnt = video_cnt + np.bincount(pw_item, minlength=n_videos)
    user_vocab  = _vocab(user_cnt,  args.min_id_count)
    video_vocab = _vocab(video_cnt, args.min_id_count)

    vc = ds.item_cat
    nodes = vc["node"].to_numpy() if "node" in vc else np.arange(len(vc))
    cat_idx, cat_sizes = [], []
    for col in ds.item_cat_cols:
        codes, uniq = pd_factorize(vc[col].to_numpy())
        code_cnt = np.bincount(codes, weights=video_cnt[nodes], minlength=len(uniq))
        voc = _vocab(code_cnt, args.min_id_count)
        cat_idx.append(voc[codes])
        cat_sizes.append(int(voc.max()) + 1)
    video_cat = (np.stack(cat_idx, axis=1) if cat_idx
                 else np.zeros((n_videos, 0), np.int64))

    vocab = {
        "users": int(user_vocab.max()), "videos": int(video_vocab.max()),
        "categoricals": dict(zip(ds.item_cat_cols, [s - 1 for s in cat_sizes])),
        "oov_share_train_rows": float((video_vocab[inter.video[train_rows]] == 0).mean()),
        "oov_share_val_rows":   float((video_vocab[inter.video[inter.rows(SPLIT_VAL)]] == 0).mean()),
    }
    logger.info("Vocabulary (min_id_count=%d): %s", args.min_id_count, vocab)

    # Graph: master relation index (timed) + per-snapshot item features
    real_test = args.eval_test and not args.max_steps
    rows_used = np.r_[train_rows, holdout, inter.rows(SPLIT_VAL),
                      inter.rows(SPLIT_TEST) if real_test else np.zeros(0, np.int64)]
    needed = set(np.unique(snap[rows_used]).tolist())
    master = None if args.no_graph else ds.graph.master()
    store = SnapshotStore(None, None, boundaries, master, needed=needed,
                          item_features=ds.item_features)

    node_data = {
        "user_vocab": torch.from_numpy(user_vocab),
        "user_x": torch.from_numpy(np.asarray(ds.user_x, np.float32)),
        "user_onehot": torch.from_numpy(np.asarray(ds.user_onehot, np.int64)),
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
        store=store, graph=ds.graph, node_data=node_data, vocab=vocab,
        fingerprint=ds.fingerprint,
        video_cat_raw=vc.reset_index(drop=True),
        evidence=evidence,
        tok_item=tok_item, tok_time=tok_time, tok_sess=tok_sess, tok_pre=tok_pre,
        dataset=ds.name,
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
        # history tokens: rows, then pre-window clicks
        self.tok_item = torch.from_numpy(d.tok_item if d.tok_item is not None else it.video)
        self.tok_time = torch.from_numpy(d.tok_time if d.tok_time is not None else it.time)
        self.tok_sess = torch.from_numpy(d.tok_sess if d.tok_sess is not None else d.session)
        self.tok_pre  = torch.from_numpy(d.tok_pre if d.tok_pre is not None
                                         else np.zeros(len(it.time), bool))
        self.hs, self.he = torch.from_numpy(d.hist_start), torch.from_numpy(d.hist_end)
        self.hrow  = torch.from_numpy(d.hist_seq)
        self.ctx_cat = torch.from_numpy(d.ctx_cat)
        self.ctx_num = torch.from_numpy(d.ctx_num)
        self.evidence = torch.from_numpy(d.evidence) if d.evidence is not None else None
        self.offsets = torch.arange(-max_len, 0)
        self.rank = torch.arange(max_len - 1, -1, -1)

    def __call__(self, rows: np.ndarray) -> dict[str, torch.Tensor]:
        r   = torch.from_numpy(np.asarray(rows, dtype=np.int64))
        pos = self.he[r, None] + self.offsets
        ok  = pos >= self.hs[r, None]
        hr  = self.hrow[pos.clamp(min=0)]                         # history token ids
        hist = torch.where(ok, self.tok_item[hr], torch.full_like(hr, -1))
        pre  = self.tok_pre[hr]
        gap  = HistoryTransformer.gap_bucket(self.time[r, None] - self.tok_time[hr])
        gap  = torch.where(pre, torch.full_like(gap, PRE_WINDOW_GAP), gap)
        same = ((self.tok_sess[hr] == self.sess[r, None]) & ~pre).long()
        return {
            "row": r, "user": self.user[r], "video": self.video[r], "label": self.label[r],
            "hist": hist, "hist_rank": self.rank.expand_as(hist),
            "hist_gap": torch.where(ok, gap, torch.zeros_like(gap)),
            "hist_sess": torch.where(ok, same, torch.zeros_like(same)),
            "ctx_cat": self.ctx_cat[r], "ctx_num": self.ctx_num[r],
            **({"evidence": self.evidence[r]} if self.evidence is not None else {}),
        }


# ── Model ─────────────────────────────────────────────────────────────────────

def build(args, d: HugData) -> HUGModel:
    nd = d.node_data
    onehot_vocab = [int(nd["user_onehot"][:, i].max()) + 1 for i in range(nd["user_onehot"].shape[1])]
    video_feat_dim = next(iter(d.store.video_x.values())).shape[1]
    inp = InputEncoder(
        args.emb_dim, int(nd["user_vocab"].max()) + 1, nd["user_x"].shape[1], onehot_vocab,
        int(nd["video_vocab"].max()) + 1, video_feat_dim, nd["cat_sizes"],
        {t: d.graph.counts[t] for t in d.graph.node_types[2:]},
    )
    inp.set_node_data(nd["user_vocab"], nd["user_x"], nd["user_onehot"],
                      nd["video_vocab"], nd["video_cat"])
    gcn = None
    if not args.no_graph:
        gcn = RelLightGCN(d.graph.relations, args.graph_layers, eps=args.cl_eps,
                          cl_layer=args.cl_layer)
    seq = None if args.no_seq else HistoryTransformer(args.emb_dim, args.seq_layers,
                                                      args.max_seq_len, dropout=args.dropout)
    ctx_vocab = [int(d.ctx_cat[:, i].max()) + 1 for i in range(d.ctx_cat.shape[1])]
    model = HUGModel(inp, gcn, seq, ctx_vocab, d.ctx_num.shape[1], args.emb_dim,
                     cl_weight=args.cl_weight, cl_temp=args.cl_temp, emb_l2=args.emb_l2,
                     freeze_graph=args.freeze_graph, graph_tokens=args.graph_tokens,
                     dropout=args.dropout, fusion=args.fusion, fusion_args=args)
    if hasattr(model.head, "n_ref"):
        # 95th percentile of each view's evidence over training rows only
        model.head.set_n_ref(np.percentile(d.evidence[d.train_rows], 95, axis=0))
    return model


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
        self.offsets = d.graph.offsets
        self.counts = dict(d.graph.counts)
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


# ── Fusion statistics (spec 05 B8) ────────────────────────────────────────────

VAL_CODE_SAMPLE = 200_000


class FusionStats:
    """
    Accumulates sparse-fusion diagnostics over an eval pass: per-row ρ and active
    counts, shared-atom co-activation, per-row Jaccard of the views' shared sets,
    and the codes of a fixed row sample (seed 0) for val_codes.npz.
    """

    def __init__(self, m_shared: int, sample_rows: np.ndarray) -> None:
        self.rows, self.rho, self.active = [], [], {k: [] for k in
                                                     ("sh_G", "sh_S", "pr_G", "pr_S")}
        self.fire_any = np.zeros(m_shared, dtype=np.int64)
        self.fire_both = np.zeros(m_shared, dtype=np.int64)
        self.jaccard = []
        self.sample = set(np.asarray(sample_rows).tolist())
        self.codes = {k: [] for k in ("row_id", "sh_G", "sh_S", "pr_G", "pr_S")}

    def add(self, rows: np.ndarray, stats: dict, codes: dict) -> None:
        self.rows.append(rows)
        self.rho.append(stats["rho"].cpu().numpy())
        for k in self.active:
            self.active[k].append(stats[f"active_{k}"].cpu().numpy())
        g, s_ = codes["sh_G"] > 0, codes["sh_S"] > 0
        self.fire_any += (g | s_).sum(0).cpu().numpy()
        self.fire_both += (g & s_).sum(0).cpu().numpy()
        inter, union = (g & s_).sum(-1).float(), (g | s_).sum(-1).float()
        self.jaccard.append((inter / union.clamp_min(1)).cpu().numpy())
        keep = np.array([r in self.sample for r in rows])
        if keep.any():
            self.codes["row_id"].append(rows[keep])
            for k in ("sh_G", "sh_S", "pr_G", "pr_S"):
                self.codes[k].append(codes[k][torch.from_numpy(keep).to(codes[k].device)].cpu())

    def summary(self, d: HugData) -> dict:
        rows = np.concatenate(self.rows)
        rho = np.concatenate(self.rho)
        out = {"rho_mean": float(rho.mean())}
        buckets = dict(d.buckets)
        for i, v in enumerate(("G", "S")):
            q = np.quantile(d.evidence[d.train_rows, i], [0.2, 0.4, 0.6, 0.8])
            buckets[f"n_{v}_quintile"] = np.searchsorted(q, d.evidence[:, i], side="right").astype(str)
        out["rho_by_bucket"] = {name: {k: float(rho[keys[rows] == k].mean())
                                       for k in np.unique(keys[rows])}
                                for name, keys in buckets.items()}
        act = {k: np.concatenate(v) for k, v in self.active.items()}
        out["active_by_evidence_quintile"] = {
            name: {k: {b: float(a[keys[rows] == k].mean()) for b, a in act.items()}
                   for k in np.unique(keys[rows])}
            for name, keys in buckets.items() if name.startswith("n_")}
        out["active_mean"] = {b: float(a.mean()) for b, a in act.items()}
        fired = self.fire_any > 0
        co = np.zeros_like(self.fire_any, dtype=float)
        co[fired] = self.fire_both[fired] / self.fire_any[fired]
        out["shared_coactivation_fraction"] = float((co >= 0.01)[fired].mean()) if fired.any() else 0.0
        out["shared_never_fired_fraction"] = float((~fired).mean())
        out["shared_jaccard_mean"] = float(np.concatenate(self.jaccard).mean())
        return out

    def save_codes(self, path: Path) -> None:
        """Nonzero indices and values per block for the sampled rows."""
        if not self.codes["row_id"]:
            return
        out = {"row_id": np.concatenate(self.codes["row_id"])}
        for k in ("sh_G", "sh_S", "pr_G", "pr_S"):
            c = torch.cat(self.codes[k])
            kmax = int((c > 0).sum(-1).max())
            vals, idx = c.topk(max(kmax, 1), dim=-1)
            out[f"{k}_idx"] = torch.where(vals > 0, idx, torch.full_like(idx, -1)).numpy()
            out[f"{k}_val"] = vals.numpy().astype(np.float32)
        np.savez(path, **out)


# ── Train / eval ──────────────────────────────────────────────────────────────

def _to(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def predict(model: HUGModel, d: HugData, rows: np.ndarray, snaps: Snapshots,
            batcher: HugBatcher, batch_size: int, device, quiet: bool,
            desc: str, max_snapshots: int | None = None,
            fstats: FusionStats | None = None) -> tuple[np.ndarray, np.ndarray]:
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
            out = model(b, vx, tables)
            out_scores.append(out["proba"].float().cpu().numpy())
            out_rows.append(sel[b0:b0 + batch_size])
            if fstats is not None:
                fstats.add(out_rows[-1], out["stats"], model.head._last["codes"])
            prog.update(len(out_rows[-1]))
    prog.close()
    return np.concatenate(out_rows), np.concatenate(out_scores)


def evaluate(split, model, d, rows, snaps, batcher, args, device, max_snapshots=None,
             fstats: FusionStats | None = None):
    r, s = predict(model, d, rows, snaps, batcher, args.batch_size * 2, device,
                   is_quiet(args.quiet), split, max_snapshots, fstats)
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
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        tot, n_seen, cl_tot, aux_tot = 0.0, 0, 0.0, 0.0
        dead, kp_sum = {}, {"G": 0.0, "S": 0.0}
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
                if hasattr(model.head, "renormalize"):
                    model.head.renormalize()
                    dead = model.head.track_and_resample(state["steps"] + 1, optimizer=opt)
                n = len(b["user"])
                tot += out["bce"].item() * n
                cl_tot += out["cl"].item() * n
                aux_tot += out["aux"].item() * n
                if "active_pr_G" in out["stats"]:
                    kp_sum["G"] += out["stats"]["active_pr_G"].float().sum().item()
                    kp_sum["S"] += out["stats"]["active_pr_S"].float().sum().item()
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
               "train_aux": aux_tot / max(n_seen, 1),
               "val": asdict(val_m), "lr": opt.param_groups[0]["lr"],
               "seconds": round(time.time() - t0, 1)}
        if device.type == "cuda":
            rec["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 2)
        if model.gcn is not None and not model.freeze_graph:
            rec["relation_weights"] = model.gcn.relation_weights().cpu().tolist()
        if hasattr(model.head, "renormalize"):
            rec["dead_atom_fraction"] = dead
            rec["mean_k_private"] = {v: kp_sum[v] / max(n_seen, 1) for v in kp_sum}
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


def realised_k_private(run_dir: Path) -> int:
    """Mean private k (over both views) at the best epoch of a finished sparse-adaptive run."""
    m = json.loads((run_dir / "final_metrics.json").read_text())
    best = next(h for h in m["history"] if h["epoch"] == m["best_epoch"])
    k = best["mean_k_private"]
    return int(round((k["G"] + k["S"]) / 2))


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

    if getattr(args, "k_private_fixed_from", None):
        args.k_private_fixed = realised_k_private(Path(args.k_private_fixed_from))
        logger.info("k_private_fixed = %d (realised mean of %s)", args.k_private_fixed,
                    args.k_private_fixed_from)
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
    fstats = None
    if getattr(model, "fusion", "concat") == "sparse":
        sample = np.sort(np.random.default_rng(0).choice(
            d.val_rows, size=min(VAL_CODE_SAMPLE, len(d.val_rows)), replace=False))
        fstats = FusionStats(model.head.m_sh, sample)
    val_m, val_p = evaluate("val", model, d, d.val_rows, snaps, batcher, args, device, max_snaps,
                            fstats)
    np.savez(run_dir / "val_preds.npz", **val_p)
    fusion_stats = None
    if fstats is not None:
        fusion_stats = fstats.summary(d)
        fstats.save_codes(run_dir / "val_codes.npz")
        logger.info("Fusion stats: %s", {k: v for k, v in fusion_stats.items() if not isinstance(v, dict)})
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
    if fusion_stats is not None:
        out["fusion_stats"] = fusion_stats
    (run_dir / "final_metrics.json").write_text(json.dumps(out, indent=2, default=float))
    if not args.eval_only and (run_dir / "last.pt").exists():
        (run_dir / "last.pt").unlink()
    logger.info("Done → %s", run_dir)
    logging.getLogger().removeHandler(fh)
