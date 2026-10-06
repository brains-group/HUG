"""
Data fingerprint for config hashing (scripts/run_queue.py).

    python Framework/fingerprint.py --data-dir KuaiRand-1K/data --out runs/heavy/data_fingerprint.json
    python Framework/fingerprint.py --dataset mind --out runs/heavy/data_fingerprint_mind.json

For MIND/ZhihuRec the fingerprint comes from the dataset adapter (cutoffs, row
counts, user sample hash, split rule used, adapter version); building it also
fills the adapter's bundle cache that the jobs then load.

Writes the split cutoffs and row counts of the canonical interaction table —
the same values every HUG and baseline job records.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import KuaiRandLoader  # noqa: E402
from hug_train import data_fingerprint  # noqa: E402
from temporal import build_interactions  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[1] / "KuaiRand-1K" / "data"))
    p.add_argument("--min-interactions", type=int, default=10)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--out", required=True)
    p.add_argument("--dataset", default="kuairand", choices=["kuairand", "mind", "zhihurec"])
    p.add_argument("--user-frac", type=float, default=None)
    args = p.parse_args()
    if args.dataset != "kuairand":
        import main as pipeline
        from datasets import load_dataset
        argv = ["--dataset", args.dataset, "--val-ratio", str(args.val_ratio),
                "--test-ratio", str(args.test_ratio)]
        argv += ["--user-frac", str(args.user_frac)] if args.user_frac is not None else []
        fp = load_dataset(pipeline.parse_args(argv)).fingerprint
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(fp, indent=2))
        print(json.dumps(fp))
        return
    data = KuaiRandLoader(args.data_dir, min_interactions=args.min_interactions).load()
    fp = data_fingerprint(build_interactions(data, args.val_ratio, args.test_ratio))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(fp, indent=2))
    print(json.dumps(fp))


if __name__ == "__main__":
    main()
