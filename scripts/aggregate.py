"""
Aggregate heavy-run outputs into summary tables (Spec 04, W4.3).

    python scripts/aggregate.py [--runs-root runs/heavy] [--n-boot 1000] [--reference N4]

Reads ``<runs_root>/<job>/final_metrics.json`` (plus ``job.json`` written by the queue and
``val_preds.npz``) and writes into ``<runs_root>/summary/``:

- ``val_table.md`` / ``val_table.csv``: per group (arm or baseline), mean ± std over seeds
  of val AUC, AP, LogLoss, nDCG@10, plus the holdout AUC/LogLoss diagnostic. Stage-1 jobs only.
- ``bucket_tables.md``: val AUC per bucket key per group (mean over seeds).
- ``significance.md``: per-user paired bootstrap of the val AUC difference of each group vs.
  the reference (and reference vs. N1, N2, N4-frozen), plus a paired t-test over seeds.
- ``test_table.md``: test metrics, only from ``final_eval`` jobs.

Groups come from ``job.json``'s ``group`` (set in the plan, e.g. "N1-own"); without it, the
arm (HUG) or model name from ``final_metrics.json``. Std is the sample std (ddof=1).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("aggregate")

VAL_METRICS = ("auc", "ap", "logloss", "ndcg10")
HOLDOUT_METRICS = ("auc", "logloss")
TEST_METRICS = ("auc", "ap", "logloss", "ndcg10", "auc_cold", "auc_warm")
TUNE_KINDS = ("tune_hug", "tune_baseline")
ABLATIONS_VS_REFERENCE = ("N1", "N2", "N4-frozen")


# ── Loading ────────────────────────────────────────────────────────────────────


@dataclass
class Run:
    job: str
    kind: str
    group: str
    model: str | None
    arm: str | None
    seed: int | None
    run_dir: Path
    metrics: dict[str, Any] = field(default_factory=dict)

    def split(self, name: str) -> dict[str, Any]:
        return self.metrics.get(name) or {}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def load_runs(runs_root: Path) -> list[Run]:
    """One Run per ``<runs_root>/<job>/final_metrics.json``, sorted by job name."""
    runs = []
    for path in sorted(runs_root.glob("*/final_metrics.json")):
        if path.parent.name == "summary":
            continue
        metrics = _read_json(path)
        if not metrics:
            logger.warning("unreadable %s", path)
            continue
        meta = _read_json(path.parent / "job.json")
        kind = meta.get("kind") or metrics.get("kind") or "unknown"
        if kind in TUNE_KINDS:
            continue
        model = meta.get("model") or metrics.get("model")
        arm = meta.get("arm") or metrics.get("arm")
        group = meta.get("group") or arm or model or path.parent.name
        seed = metrics.get("seed", meta.get("seed"))
        runs.append(Run(job=metrics.get("job") or path.parent.name, kind=kind, group=str(group),
                        model=model, arm=arm, seed=None if seed is None else int(seed),
                        run_dir=path.parent, metrics=metrics))
    return runs


def by_group(runs: list[Run]) -> dict[str, list[Run]]:
    groups: dict[str, list[Run]] = {}
    for r in runs:
        groups.setdefault(r.group, []).append(r)
    return {g: sorted(rs, key=lambda r: (r.seed is None, r.seed or 0, r.job))
            for g, rs in sorted(groups.items())}


# ── Summary statistics ─────────────────────────────────────────────────────────


def _num(x: Any) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return None if math.isnan(x) else float(x)


def mean_std(values: list[Any]) -> tuple[float, float, int]:
    """Mean, sample std (ddof=1; nan for n<2) and n over the numeric, non-nan values."""
    vals = [v for v in (_num(x) for x in values) if v is not None]
    if not vals:
        return float("nan"), float("nan"), 0
    arr = np.asarray(vals, dtype=np.float64)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")
    return float(arr.mean()), std, len(arr)


def fmt_ms(mean: float, std: float, digits: int = 4) -> str:
    if math.isnan(mean):
        return "–"
    if math.isnan(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def _seeds(runs: list[Run]) -> str:
    return ",".join(str(r.seed) for r in runs if r.seed is not None)


def metric_table(groups: dict[str, list[Run]], split: str, metrics: tuple[str, ...],
                 extra_split: str | None = None,
                 extra_metrics: tuple[str, ...] = ()) -> tuple[list[str], list[list[Any]]]:
    """Rows of (group, model, arm, n, seeds, then mean/std per metric) for one split."""
    header = ["group", "model", "arm", "n_seeds", "seeds"]
    for m in metrics:
        header += [f"{split}_{m}_mean", f"{split}_{m}_std"]
    for m in extra_metrics:
        header += [f"{extra_split}_{m}_mean", f"{extra_split}_{m}_std"]
    rows = []
    for g, rs in groups.items():
        row: list[Any] = [g, rs[0].model or "", rs[0].arm or "", len(rs), _seeds(rs)]
        for m in metrics:
            mu, sd, _ = mean_std([r.split(split).get(m) for r in rs])
            row += [mu, sd]
        for m in extra_metrics:
            mu, sd, _ = mean_std([r.split(extra_split).get(m) for r in rs])
            row += [mu, sd]
        rows.append(row)
    return header, rows


def _md_from_metric_rows(header: list[str], rows: list[list[Any]]) -> str:
    md_header = header[:5] + [h[:-5] for h in header[5::2]]
    md_rows = []
    for row in rows:
        cells = [str(c) for c in row[:5]]
        cells += [fmt_ms(row[i], row[i + 1]) for i in range(5, len(row), 2)]
        md_rows.append(cells)
    return _md_table(md_header, md_rows)


def write_val_table(groups: dict[str, list[Run]], out_dir: Path) -> None:
    header, rows = metric_table(groups, "val", VAL_METRICS, "holdout", HOLDOUT_METRICS)
    with open(out_dir / "val_table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for row in rows:
            w.writerow([f"{c:.6f}" if isinstance(c, float) else c for c in row])
    text = ("# Validation results (Stage 1)\n\n"
            "Mean ± sample std over seeds. Holdout columns are a diagnostic (training-period "
            "rows excluded from training), not a selection criterion. No test numbers here.\n\n")
    (out_dir / "val_table.md").write_text(text + _md_from_metric_rows(header, rows))


def write_bucket_tables(groups: dict[str, list[Run]], out_dir: Path) -> None:
    """One table per bucket name: rows = groups, columns = bucket keys, cells = mean val AUC."""
    names = sorted({b for rs in groups.values() for r in rs
                    for b in (r.split("val").get("buckets") or {})})
    parts = ["# Val AUC per bucket (mean over seeds)\n\n"
             "Cells: mean val AUC over seeds (mean bucket size n in the last row).\n"]
    if not names:
        parts.append("\nNo bucket metrics found.\n")
    for name in names:
        keys = sorted({str(k) for rs in groups.values() for r in rs
                       for k in ((r.split("val").get("buckets") or {}).get(name) or {})})
        rows = []
        n_by_key: dict[str, list[Any]] = {k: [] for k in keys}
        for g, rs in groups.items():
            cells = [g]
            for k in keys:
                entries = [((r.split("val").get("buckets") or {}).get(name) or {}).get(k) or {}
                           for r in rs]
                mu, _, _ = mean_std([e.get("auc") for e in entries])
                n_by_key[k] += [e.get("n") for e in entries]
                cells.append("–" if math.isnan(mu) else f"{mu:.4f}")
            rows.append(cells)
        n_row = ["mean n"] + [("–" if math.isnan(mean_std(n_by_key[k])[0])
                               else f"{mean_std(n_by_key[k])[0]:.0f}") for k in keys]
        parts.append(f"\n## {name}\n\n" + _md_table(["group"] + keys, rows + [n_row]))
    (out_dir / "bucket_tables.md").write_text("".join(parts))


def write_test_table(test_groups: dict[str, list[Run]], out_dir: Path) -> None:
    text = ("# Test results (Stage 2 final_eval jobs only)\n\n"
            "Each stage-1 config/seed scored once on test from its val-selected checkpoint. "
            "Mean ± sample std over seeds.\n\n")
    if not test_groups:
        text += "No final_eval results found.\n"
    else:
        header, rows = metric_table(test_groups, "test", TEST_METRICS)
        text += _md_from_metric_rows(header, rows)
    (out_dir / "test_table.md").write_text(text)


# ── Significance ───────────────────────────────────────────────────────────────


def load_preds(run: Run, name: str = "val_preds.npz") -> dict[str, np.ndarray] | None:
    path = run.run_dir / name
    if not path.is_file():
        return None
    with np.load(path) as z:
        return {k: z[k] for k in ("row_id", "user", "label", "score")}


class WeightedAUC:
    """
    AUC with integer weights per group of rows (bootstrap multiplicities per user), O(n) per
    call after a one-off sort. Ties count one half, matching sklearn's roc_auc_score.
    ``groups[i]`` is row i's group code (0..n_groups-1); by default each row is its own group.
    """

    def __init__(self, labels: np.ndarray, scores: np.ndarray, groups: np.ndarray | None = None):
        order = np.argsort(scores, kind="mergesort")
        s = scores[order]
        self.groups = (order if groups is None else groups[order]).astype(np.intp)
        self.pos = labels[order].astype(np.float64)
        self.neg = 1.0 - self.pos
        self.starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])   # tie-group starts

    def __call__(self, weights: np.ndarray) -> float:
        w = np.asarray(weights, dtype=np.float64)[self.groups]
        pos = np.add.reduceat(w * self.pos, self.starts)
        neg = np.add.reduceat(w * self.neg, self.starts)
        n_pos, n_neg = pos.sum(), neg.sum()
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        neg_below = np.cumsum(neg) - neg
        return float((pos * (neg_below + 0.5 * neg)).sum() / (n_pos * n_neg))


@dataclass
class PairedPreds:
    """Two groups' predictions joined on row_id: per pair of runs, aligned score arrays."""
    labels: np.ndarray
    user_codes: np.ndarray          # 0..n_users-1
    n_users: int
    pairs: list[tuple[np.ndarray, np.ndarray]]
    mode: str                       # "seeds 42,43,44" or "seed-averaged"


def _common_row_ids(preds: list[dict[str, np.ndarray]]) -> np.ndarray:
    ids = preds[0]["row_id"]
    for p in preds[1:]:
        ids = np.intersect1d(ids, p["row_id"])
    return np.unique(ids)


def _aligned(p: dict[str, np.ndarray], ids: np.ndarray, key: str) -> np.ndarray:
    """Values of ``key`` for the rows ``ids`` (all present in ``p``), in the order of ``ids``."""
    order = np.argsort(p["row_id"], kind="mergesort")
    idx = np.searchsorted(p["row_id"], ids, sorter=order)
    return p[key][order[idx]]


def pair_predictions(a: list[Run], b: list[Run]) -> PairedPreds | None:
    """
    Join two groups' val predictions on row_id. With common seeds, keep one (A, B) pair per
    common seed; otherwise compare the seed-averaged score of each group.
    """
    pa = {r.seed: p for r in a if (p := load_preds(r)) is not None}
    pb = {r.seed: p for r in b if (p := load_preds(r)) is not None}
    if not pa or not pb:
        return None
    common = sorted(s for s in pa if s in pb and s is not None)
    used = [pa[s] for s in common] + [pb[s] for s in common] if common else \
        list(pa.values()) + list(pb.values())
    ids = _common_row_ids(used)
    if len(ids) == 0:
        return None
    ref = used[0]
    labels = _aligned(ref, ids, "label").astype(np.float64)
    users = _aligned(ref, ids, "user")
    _, user_codes = np.unique(users, return_inverse=True)
    if common:
        pairs = [(_aligned(pa[s], ids, "score"), _aligned(pb[s], ids, "score")) for s in common]
        mode = "seeds " + ",".join(str(s) for s in common)
    else:
        mean_a = np.mean([_aligned(p, ids, "score") for p in pa.values()], axis=0)
        mean_b = np.mean([_aligned(p, ids, "score") for p in pb.values()], axis=0)
        pairs = [(mean_a, mean_b)]
        mode = "seed-averaged"
    return PairedPreds(labels, user_codes, int(user_codes.max()) + 1, pairs, mode)


def paired_bootstrap(pp: PairedPreds, n_boot: int = 1000, seed: int = 0) -> dict[str, float]:
    """
    Per-user paired bootstrap of AUC(A) - AUC(B), averaged over seed pairs. Users are
    resampled with replacement; each resample is shared by both models and all seed pairs.
    """
    rng = np.random.default_rng(seed)
    aucs = [(WeightedAUC(pp.labels, sa, pp.user_codes), WeightedAUC(pp.labels, sb, pp.user_codes))
            for sa, sb in pp.pairs]
    ones = np.ones(pp.n_users)
    observed = float(np.mean([fa(ones) - fb(ones) for fa, fb in aucs]))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        draw = rng.integers(0, pp.n_users, size=pp.n_users)
        user_w = np.bincount(draw, minlength=pp.n_users)
        diffs[i] = np.mean([fa(user_w) - fb(user_w) for fa, fb in aucs])
    diffs = diffs[~np.isnan(diffs)]
    lo, hi = np.percentile(diffs, [2.5, 97.5]) if len(diffs) else (float("nan"),) * 2
    return {"observed": observed, "mean": float(diffs.mean()) if len(diffs) else float("nan"),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "frac_le_0": float((diffs <= 0).mean()) if len(diffs) else float("nan"),
            "n_boot": int(len(diffs)), "n_rows": int(len(pp.labels)), "n_users": pp.n_users}


def paired_ttest(x: list[float], y: list[float]) -> dict[str, Any]:
    """Two-sided paired t-test (scipy if installed, else t statistic with a normal approx)."""
    d = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    n = len(d)
    sd = d.std(ddof=1)
    t = float(d.mean() / (sd / math.sqrt(n))) if sd > 0 else (
        0.0 if d.mean() == 0 else math.copysign(math.inf, d.mean()))
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), df=n - 1))
        method = "scipy t"
    except ImportError:
        p = math.erfc(abs(t) / math.sqrt(2))
        method = "normal approx"
    return {"t": t, "p": p, "n": n, "method": method}


def compare(a: list[Run], b: list[Run], n_boot: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    pp = pair_predictions(a, b)
    out["boot"] = paired_bootstrap(pp, n_boot) if pp else None
    out["mode"] = pp.mode if pp else None
    va = {r.seed: _num(r.split("val").get("auc")) for r in a}
    vb = {r.seed: _num(r.split("val").get("auc")) for r in b}
    common = sorted(s for s in va if s in vb and s is not None
                    and va[s] is not None and vb[s] is not None)
    out["ttest"] = (paired_ttest([va[s] for s in common], [vb[s] for s in common])
                    if len(common) >= 3 else None)
    return out


def _sig_rows(pairs: list[tuple[str, str]], groups: dict[str, list[Run]],
              n_boot: int) -> list[list[str]]:
    rows = []
    for ga, gb in pairs:
        res = compare(groups[ga], groups[gb], n_boot)
        b = res["boot"]
        t = res["ttest"]
        rows.append([
            ga, gb, res["mode"] or "no predictions",
            f"{b['observed']:+.4f}" if b else "–",
            f"{b['mean']:+.4f}" if b else "–",
            f"[{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}]" if b else "–",
            f"{b['frac_le_0']:.3f}" if b else "–",
            f"{b['n_users']}" if b else "–",
            f"t={t['t']:.2f}, p={t['p']:.3g} (n={t['n']}, {t['method']})" if t else "n<3 seeds",
        ])
    return rows


def write_significance(groups: dict[str, list[Run]], reference: str, n_boot: int,
                       out_dir: Path) -> None:
    header = ["A", "B", "paired on", "AUC diff (full)", "boot mean", "95% CI", "P(diff ≤ 0)",
              "users", "paired t-test (val AUC over seeds)"]
    text = (f"# Significance (val AUC, per-user paired bootstrap, {n_boot} resamples, seed 0)\n\n"
            "Diff = AUC(A) − AUC(B), averaged over common seeds when the groups share seeds "
            "(else on seed-averaged scores). Rows joined on row_id. P(diff ≤ 0) is the "
            "fraction of bootstrap resamples where A is not better than B.\n\n")
    if reference not in groups:
        text += f"Reference group {reference!r} not found; groups: {sorted(groups)}\n"
        (out_dir / "significance.md").write_text(text)
        return
    vs_ref = [(g, reference) for g in groups if g != reference]
    text += f"## Each group vs. {reference}\n\n" + _md_table(header, _sig_rows(vs_ref, groups, n_boot))
    ablations = [(reference, g) for g in ABLATIONS_VS_REFERENCE if g in groups and g != reference]
    if ablations:
        text += (f"\n## {reference} vs. ablations\n\n"
                 + _md_table(header, _sig_rows(ablations, groups, n_boot)))
    (out_dir / "significance.md").write_text(text)


# ── Entry point ────────────────────────────────────────────────────────────────


def aggregate(runs_root: Path, n_boot: int = 1000, reference: str = "N4") -> Path:
    """Write all summary files; returns the summary directory."""
    runs = load_runs(runs_root)
    val_groups = by_group([r for r in runs if r.kind != "final_eval"])
    test_groups = by_group([r for r in runs if r.kind == "final_eval"])
    out_dir = runs_root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_val_table(val_groups, out_dir)
    write_bucket_tables(val_groups, out_dir)
    write_significance(val_groups, reference, n_boot, out_dir)
    write_test_table(test_groups, out_dir)
    logger.info("%d stage-1 runs in %d groups, %d final_eval runs -> %s",
                sum(map(len, val_groups.values())), len(val_groups),
                sum(map(len, test_groups.values())), out_dir)
    return out_dir


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--runs-root", type=Path, default=Path("runs/heavy"))
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--reference", default="N4", help="reference group for significance")
    args = p.parse_args(argv)
    aggregate(args.runs_root, args.n_boot, args.reference)


if __name__ == "__main__":
    main()
