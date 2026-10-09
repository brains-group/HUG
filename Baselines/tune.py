"""
tune.py — fixed-budget random search for one baseline (spec 04 W3.3)
---------------------------------------------------------------------
    python Baselines/tune.py --model TransAct --trials 16 --seed 0 --out runs/heavy/tune_baseline_TransAct --gpu 0

Every baseline gets the same declared space and the same trial budget.  Each
trial is a separate train.py job (`<out>/trial_<i>/`, holdout excluded,
early stopping on val with patience 2, max 20 epochs); selection is on val AUC
only.  Writes `<out>/best.yaml` (model-config overrides for train.py --params)
and `<out>/trials.json`.  Finished trials are skipped on rerun, but only when they were
built on the processed dataset this trial would use now (same dataset id, so e.g. trials from
before the history-vocabulary fix, see train.py, are retrained, never silently reused).

The spec's forced one-epoch variant is covered by early stopping: FuxiCTR
evaluates val after every epoch and keeps the best one, so a trial whose best
epoch is 1 *is* the one-epoch model (same seed, same first epoch).  trials.json
records each trial's epoch-1 and best val AUC.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent

SPACE = {
    "embedding_dim":         [16, 32, 64],
    "min_categr_count":      [2, 5, 10, 20],
    "embedding_regularizer": [0, 1.0e-6, 1.0e-5, 1.0e-4],
    "net_dropout":           [0.0, 0.1, 0.2, 0.3],
    "learning_rate":         [1.0e-3, 5.0e-4],
}
# FiGNN has no dropout parameter; its trials vary the other four
UNSUPPORTED = {"FiGNN": {"net_dropout"}}


def sample_trials(model: str, n: int, seed: int) -> list[dict]:
    keys = [k for k in SPACE if k not in UNSUPPORTED.get(model, set())]
    grid = list(itertools.product(*(SPACE[k] for k in keys)))
    pick = np.random.default_rng(seed).choice(len(grid), size=min(n, len(grid)), replace=False)
    return [dict(zip(keys, grid[i])) for i in pick]


def _epoch_aucs(log: Path) -> list[float]:
    """Validation AUC after each epoch, from a FuxiCTR training log."""
    if not log.exists():
        return []
    return [float(x) for x in re.findall(r"\[Metrics\].*?AUC: ([0-9.]+)", log.read_text())]


def reusable_trial(metrics: dict, expected_dataset_id: str) -> bool:
    """A finished trial counts only if it ran on the processed dataset this trial needs."""
    return (metrics.get("params") or {}).get("dataset_id") == expected_dataset_id


def set_aside_stale_trials(out: Path, expected: dict[int, str]) -> Path | None:
    """
    Move finished trials built on another processed dataset (e.g. before the history-vocabulary
    fix), plus the trials.json/best.yaml selected from them, into <out>/superseded_<UTC time>/
    so they are kept for reference but neither reused nor overwritten.
    """
    stale = []
    for i, dataset_id in expected.items():
        metrics = out / f"trial_{i:02d}" / "final_metrics.json"
        if metrics.is_file() and not reusable_trial(json.loads(metrics.read_text()), dataset_id):
            stale.append(metrics.parent)
    if not stale:
        return None
    dest = out / f"superseded_{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}"
    dest.mkdir(parents=True)
    for path in [*stale, out / "trials.json", out / "best.yaml"]:
        if path.exists():
            shutil.move(str(path), str(dest / path.name))
    print(f"[tune] {len(stale)} trial(s) ran on another processed dataset: moved to {dest}",
          flush=True)
    return dest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--trials", type=int, default=16)
    p.add_argument("--seed", type=int, default=0, help="search seed (trials train with seed 2024)")
    p.add_argument("--out", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--config-hash", default=None)
    p.add_argument("--dataset", default="kuairand")
    p.add_argument("--vocab-from-history", type=int, choices=(0, 1), default=0,
                   help="passed to train.py (0 = item vocabulary from the target-item column only)")
    args, _ = p.parse_known_args()
    from train import resolve_dataset_params

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trials = sample_trials(args.model, args.trials, args.seed)
    set_aside_stale_trials(out, {
        i: resolve_dataset_params(args.model, args.dataset, t, bool(args.vocab_from_history))["dataset_id"]
        for i, t in enumerate(trials)})
    results = []
    for i, trial in enumerate(trials):
        tdir = out / f"trial_{i:02d}"
        tdir.mkdir(exist_ok=True)
        (tdir / "params.yaml").write_text(yaml.safe_dump(trial))
        metrics = tdir / "final_metrics.json"
        if not metrics.exists():
            cmd = [sys.executable, str(ROOT / "train.py"), "--model", args.model,
                   "--run-dir", str(tdir), "--gpu", str(args.gpu), "--seed", "2024",
                   "--params", str(tdir / "params.yaml"), "--dataset", args.dataset,
                   "--vocab-from-history", str(args.vocab_from_history), "--quiet"]
            if args.max_steps:
                cmd += ["--max-steps", str(args.max_steps)]
            print(f"[tune] {args.model} trial {i}: {trial}", flush=True)
            rc = subprocess.call(cmd, cwd=ROOT.parent)
            if rc != 0 or not metrics.exists():
                print(f"[tune] trial {i} failed (exit {rc})", flush=True)
                results.append({"trial": i, "params": trial, "val_auc": None, "failed": True})
                continue
        m = json.loads(metrics.read_text())
        aucs = (_epoch_aucs(tdir / f"{args.model}.log")
                or _epoch_aucs(tdir / m["params"]["dataset_id"] / f"{args.model}.log"))
        results.append({"trial": i, "params": trial, "val_auc": m["val"]["auc"],
                        "epoch1_val_auc": aucs[0] if aucs else None,
                        "best_epoch": int(np.argmax(aucs)) + 1 if aucs else None})

    ok = [r for r in results if r.get("val_auc") is not None]
    (out / "trials.json").write_text(json.dumps(results, indent=2))
    if not ok:
        sys.exit("[tune] every trial failed")
    best = max(ok, key=lambda r: r["val_auc"])
    (out / "best.yaml").write_text(yaml.safe_dump(best["params"]))
    print(f"[tune] {args.model} best trial {best['trial']}: val AUC {best['val_auc']:.4f}  {best['params']}")


if __name__ == "__main__":
    main()
