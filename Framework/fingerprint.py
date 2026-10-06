"""
Data fingerprint for config hashing (scripts/run_queue.py).

    python Framework/fingerprint.py --data-dir KuaiRand-1K/data --out runs/heavy/data_fingerprint.json

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
    args = p.parse_args()
    data = KuaiRandLoader(args.data_dir, min_interactions=args.min_interactions).load()
    fp = data_fingerprint(build_interactions(data, args.val_ratio, args.test_ratio))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(fp, indent=2))
    print(json.dumps(fp))


if __name__ == "__main__":
    main()
