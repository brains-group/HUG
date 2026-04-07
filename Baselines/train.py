"""
train.py
========
Trains DIN and/or BST using FuxiCTR on KuaiRand-1K.

Usage:
    python train.py              # train both
    python train.py --model DIN  # train DIN only
    python train.py --gpu 0      # select GPU
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import shutil

# ── FuxiCTR paths ───────────────────────────────────────────
FUXICTR_ROOT = os.environ.get("FUXICTR_ROOT", "./FuxiCTR")
MODEL_ZOO_PATHS = {
    "DIN": os.path.join(FUXICTR_ROOT, "model_zoo", "DIN"),
    "BST": os.path.join(FUXICTR_ROOT, "model_zoo", "BST"),
}
EXPID_MAP = {
    "DIN": "DIN_kuairand_1k",
    "BST": "BST_kuairand_1k",
}
CONFIG_DIR = Path(__file__).resolve().parent / "config"

# ── Train functions ────────────────────────────────────────
def train_din(gpu: int):
    import subprocess
    din_dir = MODEL_ZOO_PATHS["DIN"]
    expid = EXPID_MAP["DIN"]

    # copy configs into the location run_expid.py expects
    cfg_dest = Path(din_dir) / "config"
    cfg_dest.mkdir(exist_ok=True)
    for cfg in ("dataset_config.yaml", "model_config.yaml"):
        shutil.copy2(CONFIG_DIR / cfg, cfg_dest / cfg)

    print(f"\n{'='*60}")
    print(f"Training DIN (gpu={gpu})")
    print(f"Started at {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable,
        os.path.join(din_dir, "run_expid.py"),
        "--expid", expid,
        "--gpu", str(gpu),
        "--config", str(CONFIG_DIR),
    ]
    subprocess.run(cmd, check=True)

    print(f"\n[{datetime.now():%H:%M:%S}] DIN training complete.")

def train_bst(gpu: int):
    import subprocess
    bst_dir = MODEL_ZOO_PATHS["BST"]
    expid = EXPID_MAP["BST"]

    # copy configs
    cfg_dest = Path(bst_dir) / "config"
    cfg_dest.mkdir(exist_ok=True)
    for cfg in ("dataset_config.yaml", "model_config.yaml"):
        shutil.copy2(CONFIG_DIR / cfg, cfg_dest / cfg)

    cmd = [
        sys.executable,
        os.path.join(bst_dir, "run_expid.py"),
        "--expid", expid,
        "--gpu", str(gpu),
        "--config", str(CONFIG_DIR),
    ]
    subprocess.run(cmd, check=True)

# ── Main ───────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["DIN", "BST", "both"], default="both")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    models = ["DIN", "BST"] if args.model == "both" else [args.model]

    for m in models:
        if m == "DIN":
            train_din(args.gpu)
        elif m == "BST":
            train_bst(args.gpu)

    print(f"\n{'='*60}")
    print("All training jobs complete.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()