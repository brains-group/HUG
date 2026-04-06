"""
step2_train.py
==============
Trains DIN and / or BST using FuxiCTR on KuaiRand-1K.

Usage:
    # Train both models (default)
    python step2_train.py

    # Train a single model
    python step2_train.py --model DIN
    python step2_train.py --model BST

    # GPU selection (default: CPU / gpu 0)
    python step2_train.py --gpu 0

This script drives FuxiCTR's internal pipeline:
  1. Loads dataset_config.yaml + model_config.yaml
  2. Preprocesses CSV → HDF5 feature store
  3. Trains the model with early stopping
  4. Evaluates on validation set each epoch
  5. Saves the best checkpoint to ./checkpoints/

The best model weights are reused by step3_evaluate.py.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


# ── resolve FuxiCTR model_zoo paths ───────────────────────────────────────
FUXICTR_ROOT = os.environ.get("FUXICTR_ROOT", "./FuxiCTR")

MODEL_ZOO_PATHS = {
    "DIN": os.path.join(FUXICTR_ROOT, "model_zoo", "DIN"),
    "BST": os.path.join(FUXICTR_ROOT, "model_zoo", "BST"),
}

EXPID_MAP = {
    "DIN": "DIN_kuairand_1k",
    "BST": "BST_kuairand_1k",
}

# Absolute path to our configs
CONFIG_DIR = Path(__file__).resolve().parent / "config"


def train_model(model_name: str, gpu: int):
    model_dir = MODEL_ZOO_PATHS[model_name]
    expid     = EXPID_MAP[model_name]

    if not os.path.isdir(model_dir):
        print(
            f"\n[ERROR] FuxiCTR model_zoo path not found:\n  {model_dir}\n"
            f"  → Clone FuxiCTR with:\n"
            f"    git clone https://github.com/reczoo/FuxiCTR.git {FUXICTR_ROOT}\n"
        )
        sys.exit(1)

    # Copy our configs into the model directory so run_expid.py can find them
    config_dest = Path(model_dir) / "config"
    config_dest.mkdir(exist_ok=True)

    for cfg in ("dataset_config.yaml", "model_config.yaml"):
        src = CONFIG_DIR / cfg
        dst = config_dest / cfg
        import shutil
        shutil.copy2(src, dst)

    cmd = [
        sys.executable, "run_expid.py",
        "--expid", expid,
        "--gpu", str(gpu),
        "--config", "config",   # relative to model_dir
    ]

    print(f"\n{'='*60}")
    print(f"  Training {model_name}  (expid={expid})")
    print(f"  Working dir : {model_dir}")
    print(f"  Command     : {' '.join(cmd)}")
    print(f"  Started     : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'='*60}\n")

    cmd = [
    sys.executable,
    os.path.join(model_dir, "run_expid.py"),
    "--expid", expid,
    "--gpu", str(gpu),
    "--config", str(CONFIG_DIR),
    ]
    result = subprocess.run(cmd)  # no cwd= argument

    if result.returncode != 0:
        print(f"\n[ERROR] {model_name} training exited with code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n[{datetime.now():%H:%M:%S}] {model_name} training complete.")


def main():
    parser = argparse.ArgumentParser(description="Train DIN / BST on KuaiRand-1K")
    parser.add_argument(
        "--model", choices=["DIN", "BST", "both"], default="both",
        help="Which model to train (default: both)",
    )
    parser.add_argument(
        "--gpu", type=int, default=-1,
        help="GPU index (-1 for CPU, default: -1)",
    )
    args = parser.parse_args()

    models = ["DIN", "BST"] if args.model == "both" else [args.model]

    for m in models:
        train_model(m, args.gpu)

    print(f"\n{'='*60}")
    print("  All training jobs complete.")
    print(f"  Checkpoints in: ./FuxiCTR/checkpoints/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
