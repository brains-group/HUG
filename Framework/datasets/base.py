"""
Dataset adapter contract (spec 06 A)
-------------------------------------
Every dataset reaches HUG and the baselines as a `DatasetBundle`.  Downstream
code (hug_train.py, features.py, temporal.py, hug.py, Baselines/preprocess.py)
sees only this interface: integer node indices, per-row arrays and feature
matrices.  Dataset column names stay inside the adapters.

Canonical interaction table: one row per impression candidate, sorted by
impression time (stable, file order breaks ties); a row's *row id* is its
position.  `time` is the impression time and `label_time` the earliest time
the row's label is known (spec 06 B).  Times are int64 milliseconds.

Node layout: GraphSpec.node_types[0] is "user" and [1] is "video" (the item
type, whatever the dataset calls its items), then the metadata node types.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
import pandas as pd
import torch

from data_loader import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL, assign_split, chronological_cutoffs
from temporal import Interactions

logger = logging.getLogger(__name__)

HOUR_MS = 3_600_000
DAY_MS = 24 * HOUR_MS
STATIC_EDGE_TIME = torch.iinfo(torch.long).min       # metadata edges: always visible
ITEM_TYPE = "video"                                  # internal name of the item node type


# ── Graph ─────────────────────────────────────────────────────────────────────

@dataclass
class GraphSpec:
    """
    Typed graph over the dataset's nodes.  `relations[r]` = (src_type, dst_type) of
    relation id r.  The master index is every relation's edges in global node ids
    with their edge_time (STATIC_EDGE_TIME for metadata edges); a snapshot is the
    mask edge_time < boundary (temporal.SnapshotStore).
    """

    node_types:     list[str]
    counts:         dict[str, int]
    relations:      list[tuple[str, str]]
    relation_names: list[str]
    parts:          list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=list)
    master_fn:      Callable | None = None           # KuaiRand: built from the cached HKG

    @property
    def offsets(self) -> dict[str, int]:
        out, acc = {}, 0
        for t in self.node_types:
            out[t] = acc
            acc += self.counts[t]
        return out

    def master(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.master_fn is not None:
            return self.master_fn()
        if not self.parts:
            return (torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, dtype=torch.long),
                    torch.zeros(0, dtype=torch.long))
        ei = torch.from_numpy(np.concatenate([np.stack([s, d]) for _, s, d, _ in self.parts], axis=1))
        et = torch.from_numpy(np.concatenate([np.full(len(s), r, np.int64) for r, s, _, _ in self.parts]))
        tt = torch.from_numpy(np.concatenate([t for *_, t in self.parts]))
        return ei.long(), et, tt.long()

    def edge_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r, s, _, _ in self.parts:
            out[self.relation_names[r]] = out.get(self.relation_names[r], 0) + len(s)
        return out


class GraphBuilder:
    """
    Collects relations, then lays them out like KuaiRand's HUG relation set: every
    bidirectional relation r gets id r and its reverse id r + B (B = number of
    bidirectional relations); one-directional relations (transitions) come last.
    """

    def __init__(self, node_types: list[str], counts: dict[str, int]) -> None:
        assert node_types[0] == "user" and node_types[1] == ITEM_TYPE
        self.node_types, self.counts = list(node_types), dict(counts)
        self._rels: list[tuple] = []

    def add(self, name: str, src_type: str, dst_type: str, src: np.ndarray, dst: np.ndarray,
            time: np.ndarray | None = None, reverse: bool = True) -> None:
        src, dst = np.asarray(src, np.int64), np.asarray(dst, np.int64)
        assert len(src) == len(dst)
        assert len(src) == 0 or (src.min() >= 0 and src.max() < self.counts[src_type]), name
        assert len(dst) == 0 or (dst.min() >= 0 and dst.max() < self.counts[dst_type]), name
        t = (np.full(len(src), STATIC_EDGE_TIME, np.int64) if time is None
             else np.asarray(time, np.int64))
        self._rels.append((name, src_type, dst_type, src, dst, t, reverse))

    def build(self) -> GraphSpec:
        spec = GraphSpec(self.node_types, self.counts, [], [])
        off = spec.offsets
        bi = [r for r in self._rels if r[6]]
        one = [r for r in self._rels if not r[6]]
        B = len(bi)
        spec.relations = [None] * (2 * B + len(one))
        spec.relation_names = [None] * (2 * B + len(one))
        for i, (name, st, dt, s, d, t, _) in enumerate(bi):
            spec.relations[i], spec.relation_names[i] = (st, dt), name
            spec.relations[i + B], spec.relation_names[i + B] = (dt, st), f"rev_{name}"
            gs, gd = s + off[st], d + off[dt]
            spec.parts.append((i, gs, gd, t))
            spec.parts.append((i + B, gd, gs, t))
        for j, (name, st, dt, s, d, t, _) in enumerate(one):
            r = 2 * B + j
            spec.relations[r], spec.relation_names[r] = (st, dt), name
            spec.parts.append((r, s + off[st], d + off[dt], t))
        return spec


# ── Item node features per snapshot ───────────────────────────────────────────

class ItemFeatures(Protocol):
    def at(self, cutoff: int) -> np.ndarray:
        """float32 [n_items, F]: item features from behaviour strictly before `cutoff`."""


class LabelStatItemFeatures:
    """features.ITEM_STAT_COLS per item from rows before the cutoff, ‖ static numeric features."""

    def __init__(self, item, time, label, label_time, n_items: int,
                 static: np.ndarray | None = None) -> None:
        self.item, self.time, self.label, self.label_time = item, time, label, label_time
        self.n_items, self.static = n_items, static

    def at(self, cutoff: int) -> np.ndarray:
        from features import item_label_stats_before
        x = item_label_stats_before(self.item, self.time, self.label, self.label_time,
                                    self.n_items, int(cutoff))
        return x if self.static is None else np.concatenate([x, self.static], axis=1).astype(np.float32)


# ── Baseline tables ───────────────────────────────────────────────────────────

@dataclass
class BaselineTables:
    """What Baselines/preprocess.py needs beyond the shared per-row arrays."""

    user_ids:   np.ndarray                 # raw id per user node (written to the CSVs)
    item_ids:   np.ndarray                 # raw id per item node
    user_cats:  pd.DataFrame               # user node order, string categorical columns
    item_cats:  pd.DataFrame               # item node order, string categorical columns
    item_num:   pd.DataFrame | None = None # item node order, static numeric columns


# ── The bundle ────────────────────────────────────────────────────────────────

@dataclass
class DatasetBundle:
    name:          str
    inter:         Interactions            # canonical rows; inter.video = item node index
    session:       np.ndarray              # [N] session id (same-session history flag)
    ctx_cat:       np.ndarray              # [N, C] int64 context categoricals
    ctx_cat_names: list[str]
    ctx_num:       np.ndarray              # [N, K] float32 per-row numeric context (causal)
    ctx_num_names: list[str]
    n_users:       int
    n_items:       int
    user_x:        np.ndarray              # [n_users, c] float32 static user features
    user_onehot:   np.ndarray              # [n_users, k] int64 static user categoricals
    item_cat:      pd.DataFrame            # item node order; HUG item categoricals (raw values)
    item_cat_cols: list[str]
    graph:         GraphSpec
    item_features: ItemFeatures
    metadata_links: list[tuple[np.ndarray, np.ndarray]]    # (item, neighbour) per type
    metadata_evidence_types: list[str]
    split_rule:    str
    split_info:    dict
    timezone:      str
    tz_offset_hours: int
    fingerprint:   dict
    pre_window:    tuple[np.ndarray, np.ndarray] | None = None   # (user, item), oldest first
    user_sample:   tuple[float, int] | None = None
    baseline_fn:   Callable[[], BaselineTables] | None = None   # built on demand (preprocess.py)
    stats:         dict = field(default_factory=dict)            # adapter-side counts for reports

    def baseline_tables(self) -> BaselineTables:
        if self.baseline_fn is None:
            raise ValueError(f"{self.name}: adapter provides no baseline tables")
        return self.baseline_fn()

    @property
    def label_time(self) -> np.ndarray:
        return self.inter.label_time if self.inter.label_time is not None else self.inter.time

    @property
    def interactions(self) -> pd.DataFrame:
        """The contract view: user, item, time, label, label_time, ctx_* (row id = position)."""
        df = pd.DataFrame({"user": self.inter.user, "item": self.inter.video, "time": self.inter.time,
                           "label": self.inter.label, "label_time": self.label_time})
        for i, n in enumerate(self.ctx_cat_names):
            df[f"ctx_{n}"] = self.ctx_cat[:, i]
        return df


# ── Splits ────────────────────────────────────────────────────────────────────

def nearest_midnight(t_ms: int, tz_offset_hours: int) -> int:
    """The local midnight nearest to t_ms (ties round up), as a UTC ms timestamp."""
    off = tz_offset_hours * HOUR_MS
    return int(((t_ms + off + DAY_MS // 2) // DAY_MS) * DAY_MS - off)


def split_cutoffs(time: np.ndarray, rule: str, tz_offset_hours: int = 0,
                  val_ratio: float = 0.1, test_ratio: float = 0.2,
                  min_train: float = 0.5, min_val: float = 0.05) -> tuple[int, int, dict]:
    """
    (t_val, t_test, info).  'quantile_70_10_20': the exact row-quantile cut
    (data_loader.chronological_cutoffs).  'day_snap': the same quantiles moved to the
    nearest local midnight; if that leaves train < min_train or val < min_val of the
    rows (or an empty test), fall back to the exact cut.  info records the rule used.
    """
    time = np.asarray(time, np.int64)
    q_val, q_test = chronological_cutoffs(time, val_ratio, test_ratio)
    info = {"rule": rule, "quantile_cutoffs": [q_val, q_test]}
    if rule.startswith("quantile"):
        t_val, t_test, info["rule_used"] = q_val, q_test, "quantile"
    elif rule == "day_snap":
        s_val, s_test = nearest_midnight(q_val, tz_offset_hours), nearest_midnight(q_test, tz_offset_hours)
        n = len(time)
        tr, va, te = (time < s_val).sum() / n, ((time >= s_val) & (time < s_test)).sum() / n, (time >= s_test).sum() / n
        info["snapped_cutoffs"] = [s_val, s_test]
        info["snapped_shares"] = [float(tr), float(va), float(te)]
        if s_val < s_test and tr >= min_train and va >= min_val and te > 0:
            t_val, t_test, info["rule_used"] = s_val, s_test, "day_snap"
        else:
            t_val, t_test, info["rule_used"] = q_val, q_test, "quantile_fallback"
            logger.warning("day_snap rejected (train=%.3f val=%.3f test=%.3f): falling back to the "
                           "exact %d/%d/%d quantile cut", tr, va, te,
                           round(100 * (1 - val_ratio - test_ratio)), round(100 * val_ratio),
                           round(100 * test_ratio))
    else:
        raise ValueError(f"unknown split rule {rule!r}")
    split = assign_split(time, t_val, t_test)
    info["cutoffs"] = [int(t_val), int(t_test)]
    info["shares"] = [float((split == s).mean()) for s in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)]
    logger.info("Split rule %s → %s  shares=%s", rule, info["rule_used"],
                [round(x, 4) for x in info["shares"]])
    return int(t_val), int(t_test), info


# ── Subsampling ───────────────────────────────────────────────────────────────

def sample_users(users: np.ndarray, frac: float, seed: int = 0) -> np.ndarray:
    """A uniform random `frac` of the distinct users (sorted), drawn with default_rng(seed)."""
    uniq = np.unique(np.asarray(users))
    if frac >= 1.0:
        return uniq
    pick = np.random.default_rng(seed).choice(uniq, int(round(frac * len(uniq))), replace=False)
    return np.sort(pick)


def id_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.sort(np.asarray(values).astype(np.int64))).tobytes()
                          ).hexdigest()[:16]


# ── Canonical table helpers ───────────────────────────────────────────────────

def make_interactions(user, item, time, label, label_time, t_val: int, t_test: int,
                      frame: pd.DataFrame | None = None) -> Interactions:
    """Interactions for a canonical table already sorted by time."""
    user, item = np.asarray(user, np.int64), np.asarray(item, np.int64)
    time, label_time = np.asarray(time, np.int64), np.asarray(label_time, np.int64)
    assert (np.diff(time) >= 0).all(), "canonical table must be sorted by time"
    assert (label_time >= time).all(), "label_time must be >= time"
    split = assign_split(time, t_val, t_test)
    n = len(time)
    z = np.zeros(n, np.int64)
    return Interactions(user, item, time, np.ones(n, np.float32), np.asarray(label, np.float32),
                        split, z, z, item, int(t_val), int(t_test), 0, frame=frame,
                        label_time=label_time)


def data_fingerprint(inter: Interactions, extra: dict | None = None) -> dict:
    fp = {
        "t_val": int(inter.t_val), "t_test": int(inter.t_test),
        "rows": {name: int((inter.split == code).sum())
                 for name, code in (("train", SPLIT_TRAIN), ("val", SPLIT_VAL), ("test", SPLIT_TEST))},
        "n_users": int(inter.user.max()) + 1,
    }
    if extra:
        fp.update(extra)
    return fp


def fingerprint_hash(fp: dict) -> str:
    return hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()


def transitions(user: np.ndarray, item: np.ndarray, label_time: np.ndarray, time: np.ndarray,
                clicked: np.ndarray, gap_ms: int = 30 * 60 * 1000):
    """
    Consecutive known clicks of the same user (ordered by label_time, time, row) at most
    gap_ms apart: (prev_item, next_item, edge_time = later click's label_time).
    """
    rows = np.flatnonzero(np.asarray(clicked))
    o = rows[np.lexsort((rows, time[rows], label_time[rows], user[rows]))]
    u, it, lt = user[o], item[o], label_time[o]
    ok = (u[1:] == u[:-1]) & (lt[1:] - lt[:-1] <= gap_ms)
    return it[:-1][ok], it[1:][ok], lt[1:][ok]
