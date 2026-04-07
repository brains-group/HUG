"""
step2_train.py
==============
Trains DIN and/or BST using FuxiCTR on KuaiRand-1K.

Usage:
    python step2_train.py              # train both
    python step2_train.py --model DIN  # train DIN only
    python step2_train.py --gpu 0      # select GPU
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

# ── DIN patch ───────────────────────────────────────────────
def patch_din_forward(DIN_class):
    import torch

    old_forward = DIN_class.forward

    def new_forward(self, X):
        # expand target embedding to match sequence length
        sequence_fields = self.sequence_feature_list
        for idx, field in enumerate(sequence_fields):
            seq_emb = X[field]  # (batch, seq_len, embed_dim)
            target_field = self.target_field_list[idx]
            tgt_emb = X[target_field]  # (batch, embed_dim)

            # expand target to seq_len
            X[target_field] = tgt_emb.unsqueeze(1).expand(-1, seq_emb.size(1), -1)
        return old_forward(self, X)

    DIN_class.forward = new_forward
    return DIN_class

# ── Train functions ────────────────────────────────────────
def train_din(gpu: int):
    sys.path.insert(0, str(MODEL_ZOO_PATHS["DIN"]))
    from src.DIN import DIN as LocalDIN
    from fuxictr.pytorch.data.dataset import Dataset
    from fuxictr.pytorch.trainer import Trainer

    # apply patch
    LocalDIN = patch_din_forward(LocalDIN)

    # load configs
    import yaml
    dataset_config = yaml.safe_load(open(CONFIG_DIR / "dataset_config.yaml"))
    model_config   = yaml.safe_load(open(CONFIG_DIR / "model_config.yaml"))

    # initialize datasets
    train_gen = Dataset(dataset_config, "train")
    valid_gen = Dataset(dataset_config, "valid")

    # initialize model
    model = LocalDIN(model_config, dataset_config)

    device = f"cuda:{gpu}" if gpu >= 0 else "cpu"
    model.to(device)

    print(f"\n{'='*60}")
    print(f"Training DIN on {device}")
    print(f"Started at {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'='*60}\n")

    # trainer params
    training_params = dict(
        batch_size=model_config.get("batch_size", 128),
        epochs=model_config.get("epochs", 10),
        learning_rate=model_config.get("learning_rate", 1e-3),
    )

    model.fit(train_gen, validation_data=valid_gen, **training_params)

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