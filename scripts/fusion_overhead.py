"""
Spec 05 §D overhead benchmark: seconds per training step and peak GPU memory of each
fusion head relative to concat, on identical batches in one process (so other jobs
on the GPU affect every head alike; heads are interleaved over several rounds).

    python scripts/fusion_overhead.py --device cuda:0 [--steps 60] [--rounds 3]

Uses the HUG defaults on KuaiRand-1K (cached HKG).  Writes JSON to --out.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Framework"))

import hug_train  # noqa: E402
import main as pipeline  # noqa: E402
from runtime import set_determinism  # noqa: E402

VARIANTS = [("concat", []), ("gate", []), ("evgate", []), ("moe", []), ("misa", []),
            ("sparse", []), ("sparse-fixed", ["--k-schedule", "fixed"])]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--cache-dir", default=str(ROOT / "Framework" / "cache"))
    p.add_argument("--out", default=None)
    a = p.parse_args()
    device = torch.device(a.device)

    base = ["--model-type", "hug", "--cache-dir", a.cache_dir, "--quiet"]
    args0 = pipeline.parse_args(base)
    d = hug_train.prepare(args0)
    # one fixed sequence of training batches from the busiest training snapshot
    ks, cnt = np.unique(d.snap[d.train_rows], return_counts=True)
    k = int(ks[np.argmax(cnt)])
    rows = d.train_rows[d.snap[d.train_rows] == k]
    rng = np.random.default_rng(0)
    batches = [rng.choice(rows, args0.batch_size, replace=False) for _ in range(a.warmup + a.steps)]

    times = {n: [] for n, _ in VARIANTS}
    peak = {}
    for _ in range(a.rounds):
        for name, extra in VARIANTS:
            fusion = name.split("-")[0]
            args = pipeline.parse_args(base + ["--fusion", fusion] + extra)
            set_determinism(0)
            model = hug_train.build(args, d).to(device)
            opt = hug_train.make_optimizer(model, args)
            snaps = hug_train.Snapshots(d, model, device)
            batcher = hug_train.HugBatcher(d, args.max_seq_len)
            vx = snaps.enter(k)
            model.train()
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            for i, r in enumerate(batches):
                if i == a.warmup:
                    torch.cuda.synchronize(device)
                    t0 = time.perf_counter()
                b = hug_train._to(batcher(r), device)
                tables = model.graph_tables(vx, train=True) if model.gcn is not None else None
                out = model(b, vx, tables, train=True)
                opt.zero_grad(set_to_none=True)
                out["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                if hasattr(model.head, "renormalize"):
                    model.head.renormalize()
                    model.head.track_and_resample(i + 1, optimizer=opt)
            torch.cuda.synchronize(device)
            times[name].append((time.perf_counter() - t0) / a.steps)
            peak[name] = max(peak.get(name, 0.0), torch.cuda.max_memory_allocated(device) / 1e9)
            del model, opt, snaps
            torch.cuda.empty_cache()

    base_t = float(np.median(times["concat"]))
    res = {n: {"sec_per_step_median": float(np.median(t)), "sec_per_step_all": t,
               "overhead_vs_concat": float(np.median(t)) / base_t - 1.0,
               "peak_gpu_gb": round(peak[n], 2)} for n, t in times.items()}
    res["_setup"] = {"snapshot": k, "batch_size": args0.batch_size, "steps": a.steps,
                     "warmup": a.warmup, "rounds": a.rounds, "device": torch.cuda.get_device_name(device)}
    text = json.dumps(res, indent=2)
    print(text)
    if a.out:
        Path(a.out).write_text(text)


if __name__ == "__main__":
    main()
