"""
tune_hug.py — fixed-budget random search for a HUG arm (spec 04 W3.3)
-----------------------------------------------------------------------
    python Framework/tune_hug.py --arm N4 --trials 16 --seed 0 --out runs/heavy/tune_hug_N4 --gpu 0

Samples `--trials` distinct configurations from the declared space (seeded),
trains each as a separate main.py job (`<out>/trial_<i>/`, holdout excluded,
early stopping on val), selects on val AUC only, and writes `<out>/best.yaml`
(argument overrides for `main.py --params`) plus `<out>/trials.json`.
Finished trials are skipped on rerun.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

# Spec 03 tuning space + spec 04 additions; graph keys apply to graph arms only
SPACE = {
    "emb_dim":      [32, 64, 128],
    "min_id_count": [2, 5, 10],
    "seq_layers":   [1, 2],
    "dropout":      [0.0, 0.1, 0.2],
    "graph_layers": [1, 2, 3],
    "cl_weight":    [0.05, 0.1, 0.2],
}
GRAPH_KEYS = {"graph_layers", "cl_weight"}
ARM_FLAGS = {"N4": [], "N1": ["--no-graph"]}


def sample_trials(arm: str, n: int, seed: int) -> list[dict]:
    keys = [k for k in SPACE if arm != "N1" or k not in GRAPH_KEYS]
    grid = list(itertools.product(*(SPACE[k] for k in keys)))
    pick = np.random.default_rng(seed).choice(len(grid), size=min(n, len(grid)), replace=False)
    return [dict(zip(keys, grid[i])) for i in pick]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=sorted(ARM_FLAGS), required=True)
    p.add_argument("--trials", type=int, default=16)
    p.add_argument("--seed", type=int, default=0, help="search seed (trials train with seed 42)")
    p.add_argument("--out", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--config-hash", default=None)
    args, passthrough = p.parse_known_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for i, trial in enumerate(sample_trials(args.arm, args.trials, args.seed)):
        tdir = out / f"trial_{i:02d}"
        tdir.mkdir(exist_ok=True)
        (tdir / "params.yaml").write_text(yaml.safe_dump(trial))
        metrics = tdir / "final_metrics.json"
        if not metrics.exists():
            cmd = [sys.executable, str(ROOT / "Framework" / "main.py"), "--model-type", "hug",
                   "--run-dir", str(tdir), "--device", f"cuda:{args.gpu}", "--seed", "42",
                   "--params", str(tdir / "params.yaml"), "--quiet", *ARM_FLAGS[args.arm], *passthrough]
            if args.max_steps:
                cmd += ["--max-steps", str(args.max_steps)]
            print(f"[tune_hug] trial {i}: {trial}", flush=True)
            rc = subprocess.call(cmd, cwd=ROOT)
            if rc != 0 or not metrics.exists():
                print(f"[tune_hug] trial {i} failed (exit {rc})", flush=True)
                results.append({"trial": i, "params": trial, "val_auc": None, "failed": True})
                continue
        m = json.loads(metrics.read_text())
        results.append({"trial": i, "params": trial, "val_auc": m["val"]["auc"],
                        "best_epoch": m.get("best_epoch")})

    ok = [r for r in results if r.get("val_auc") is not None]
    (out / "trials.json").write_text(json.dumps(results, indent=2))
    if not ok:
        sys.exit("[tune_hug] every trial failed")
    best = max(ok, key=lambda r: r["val_auc"])
    (out / "best.yaml").write_text(yaml.safe_dump(best["params"]))
    print(f"[tune_hug] best trial {best['trial']}: val AUC {best['val_auc']:.4f}  {best['params']}")


if __name__ == "__main__":
    main()
