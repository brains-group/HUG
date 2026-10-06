"""
Shared evaluation metrics for HUG and the baselines.

AUC, AP, LogLoss, per-user nDCG@10, and AUC per bucket (features.bucket_keys).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, ndcg_score, roc_auc_score

logger = logging.getLogger(__name__)


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
    # Spec 01 cold/warm (kept as top-level fields for continuity)
    auc_cold:  float = float("nan")
    auc_warm:  float = float("nan")
    n_cold:    int   = 0
    # {bucket name: {key: {"auc": float, "n": int}}}
    buckets:   dict  = field(default_factory=dict)

    def __str__(self) -> str:
        pos_rate = self.n_pos / max(self.n_samples, 1)
        return (
            f"[{self.split:5s}]  loss={self.loss:.4f}  "
            f"AUC={self.auc:.4f}  AP={self.ap:.4f}  "
            f"LogLoss={self.logloss:.4f}  nDCG@10={self.ndcg10:.4f}  "
            f"n={self.n_samples:,}  pos_rate={pos_rate:.3f}  "
            f"AUC cold/warm={self.auc_cold:.4f}/{self.auc_warm:.4f} (n_cold={self.n_cold:,})"
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


def _auc(labels: np.ndarray, probas: np.ndarray) -> float:
    if len(labels) == 0 or labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, probas))


def compute_metrics(split: str, labels: np.ndarray,
                    probas: np.ndarray, loss: float,
                    user_idxs: np.ndarray | None = None,
                    buckets: dict[str, np.ndarray] | None = None) -> Metrics:
    labels = np.asarray(labels, dtype=np.float64)
    probas = np.asarray(probas, dtype=np.float64)
    m = Metrics(split=split, loss=loss,
                n_samples=len(labels), n_pos=int(labels.sum()))
    if m.n_pos == 0 or m.n_pos == m.n_samples:
        logger.warning("%s split has no label variance — skipping AUC/AP", split)
        return m
    m.auc     = float(roc_auc_score(labels, probas))
    m.ap      = float(average_precision_score(labels, probas))
    m.logloss = float(log_loss(labels, np.clip(probas, 1e-7, 1 - 1e-7)))
    if user_idxs is not None:
        m.ndcg10 = _per_user_ndcg_at_k(np.asarray(user_idxs), labels, probas, k=10)
    for name, keys in (buckets or {}).items():
        keys = np.asarray(keys)
        m.buckets[name] = {
            str(k): {"auc": _auc(labels[keys == k], probas[keys == k]), "n": int((keys == k).sum())}
            for k in np.unique(keys)
        }
    cw = m.buckets.get("coldwarm")
    if cw:
        m.auc_cold = cw.get("cold", {}).get("auc", float("nan"))
        m.auc_warm = cw.get("warm", {}).get("auc", float("nan"))
        m.n_cold   = cw.get("cold", {}).get("n", 0)
    return m
