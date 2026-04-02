"""
step3_evaluate.py
=================
Evaluates trained DIN and BST models on KuaiRand-1K and reports:

    ┌─────────────────────────────────────────────────────────┐
    │  Metric            │  Train  │  Test                    │
    │  ──────────────────┼─────────┼────────                  │
    │  Average Precision │         │                          │
    │  Log Loss          │         │                          │
    │  AUC-ROC           │         │                          │
    │  Samples (total)   │         │                          │
    │  Positive Samples  │         │                          │
    └─────────────────────────────────────────────────────────┘

Usage:
    python step3_evaluate.py                 # evaluate both models
    python step3_evaluate.py --model DIN
    python step3_evaluate.py --model BST
    python step3_evaluate.py --gpu 0         # GPU inference

Prerequisites:
    • step1_preprocess.py  must have run  → ./data/processed/*.csv
    • step2_train.py       must have run  → checkpoints saved by FuxiCTR

How it works
────────────
FuxiCTR does not expose a simple "load model + predict on arbitrary CSV"
public API at the module level, so this script re-uses the *exact same*
FuxiCTR internals that run_expid.py uses:

    1.  sys.path is patched to include the model_zoo model directory so
        that the model-local `src/<Model>.py` is importable.
    2.  fuxictr.config   → load_config(), setup_logger()
    3.  fuxictr.datasets → FeatureProcessor
    4.  model class      → loaded from the model_zoo src/
    5.  Model.load_weights()  then  Model.predict() on DataLoader

All five requested metrics are computed with scikit-learn after collecting
predictions, so no FuxiCTR metric internals are relied on.
"""

import os
import sys
import glob
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    roc_auc_score,
)


# ── configuration ──────────────────────────────────────────────────────────
FUXICTR_ROOT = os.environ.get("FUXICTR_ROOT", "./FuxiCTR")
PROCESSED_DIR = "./data/processed"
CHECKPOINT_ROOT = os.path.join(FUXICTR_ROOT, "checkpoints")

MODEL_ZOO_PATHS = {
    "DIN": os.path.join(FUXICTR_ROOT, "model_zoo", "DIN", "DIN_torch"),
    "BST": os.path.join(FUXICTR_ROOT, "model_zoo", "BST", "BST_torch"),
}
EXPID_MAP = {
    "DIN": "DIN_kuairand_1k",
    "BST": "BST_kuairand_1k",
}
CONFIG_DIR = Path(__file__).resolve().parent / "config"


# ── helpers ────────────────────────────────────────────────────────────────
def patch_sys_path(model_name: str):
    """Add FuxiCTR root and model_zoo src/ to sys.path."""
    model_dir = MODEL_ZOO_PATHS[model_name]
    for p in [FUXICTR_ROOT, model_dir, os.path.join(model_dir, "src")]:
        if p not in sys.path:
            sys.path.insert(0, p)


def find_best_checkpoint(model_name: str, expid: str) -> str:
    """
    FuxiCTR saves checkpoints under:
        {checkpoint_root}/{expid}_{hash}/{expid}_{hash}_model.ckpt
    or occasionally:
        {checkpoint_root}/{expid}/{expid}_model.ckpt
    Return the most recently modified matching file.
    """
    patterns = [
        os.path.join(CHECKPOINT_ROOT, f"{expid}*", f"{expid}*_model.ckpt"),
        os.path.join(CHECKPOINT_ROOT, f"{expid}*", "model.ckpt"),
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found for expid='{expid}'.\n"
            f"  Searched: {patterns}\n"
            f"  Run step2_train.py first."
        )
    # pick the most recently touched file
    best = max(candidates, key=os.path.getmtime)
    return best


def load_fuxictr_model(model_name: str, expid: str, gpu: int):
    """
    Reconstruct the FuxiCTR model object from config + checkpoint.
    Returns (model, feature_map) ready for inference.
    """
    patch_sys_path(model_name)

    import fuxictr
    from fuxictr import datasets
    from fuxictr.utils import load_config, set_logger, print_to_json

    config_dir = str(CONFIG_DIR)
    params = load_config(config_dir, expid)
    params["gpu"] = gpu

    set_logger(params)
    logging.info(f"Loaded config for expid={expid}")

    # Build feature encoder (reuses cached HDF5 if available)
    feature_encoder = datasets.FeatureProcessor(**params)
    feature_encoder.fit_transform(rebuild_dataset=False)

    # Import the model class dynamically from model_zoo src/
    import importlib
    module = importlib.import_module(model_name)   # e.g. src/DIN.py → DIN module
    ModelClass = getattr(module, model_name)

    model = ModelClass(feature_encoder.feature_map, **params)

    ckpt_path = find_best_checkpoint(model_name, expid)
    logging.info(f"Loading weights from: {ckpt_path}")
    model.load_weights(ckpt_path)
    model.eval()

    return model, feature_encoder


def get_predictions(model, feature_encoder, csv_path: str, batch_size: int = 4096):
    """
    Run inference on a CSV split.
    Returns (y_true, y_pred) as numpy arrays.
    """
    import torch
    from torch.utils.data import DataLoader
    from fuxictr.datasets import DataGenerator

    dg = DataGenerator(feature_encoder, data_path=csv_path, batch_size=batch_size)
    loader = dg.make_iterator()

    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in loader:
            y_pred = model.predict(batch)           # → tensor of shape (B,)
            y_true = batch[feature_encoder.feature_map.label_name]
            all_preds.append(y_pred.cpu().numpy())
            all_labels.append(y_true.cpu().numpy())

    return np.concatenate(all_labels), np.concatenate(all_preds)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute the five requested metrics."""
    total    = len(y_true)
    positive = int(y_true.sum())

    ap  = average_precision_score(y_true, y_pred)
    ll  = log_loss(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    return {
        "Average Precision": ap,
        "Log Loss":          ll,
        "AUC-ROC":           auc,
        "Samples":           total,
        "Positive Samples":  positive,
    }


def print_results_table(model_name: str, results: dict):
    """Pretty-print a side-by-side train / test table."""
    sep = "─" * 62
    header = f"\n{'='*62}\n  {model_name} — Evaluation Results\n{'='*62}"
    print(header)
    print(f"  {'Metric':<22}  {'Train':>12}  {'Test':>12}")
    print(f"  {sep}")

    for metric in ["Average Precision", "Log Loss", "AUC-ROC",
                   "Samples", "Positive Samples"]:
        tr_val = results["train"][metric]
        te_val = results["test"][metric]

        if isinstance(tr_val, float):
            tr_str = f"{tr_val:.6f}"
            te_str = f"{te_val:.6f}"
        else:
            tr_str = f"{tr_val:>12,}"
            te_str = f"{te_val:>12,}"

        print(f"  {metric:<22}  {tr_str:>12}  {te_str:>12}")

    print(f"  {sep}\n")


def save_results_csv(all_results: dict, out_path: str):
    """Save all model results to a single CSV for downstream analysis."""
    rows = []
    for model_name, splits in all_results.items():
        for split_name, metrics in splits.items():
            row = {"model": model_name, "split": split_name}
            row.update(metrics)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Results saved to: {out_path}")
    return df


# ── main ───────────────────────────────────────────────────────────────────
def evaluate_model(model_name: str, gpu: int) -> dict:
    expid = EXPID_MAP[model_name]
    print(f"\n[{datetime.now():%H:%M:%S}] Evaluating {model_name} (expid={expid}) …")

    model, feature_encoder = load_fuxictr_model(model_name, expid, gpu)

    splits = {
        "train": os.path.join(PROCESSED_DIR, "train.csv"),
        "test":  os.path.join(PROCESSED_DIR, "test.csv"),
    }

    results = {}
    for split_name, csv_path in splits.items():
        print(f"  Running inference on {split_name} ({csv_path}) …")
        y_true, y_pred = get_predictions(model, feature_encoder, csv_path)
        results[split_name] = compute_metrics(y_true, y_pred)
        print(f"  {split_name} done — {len(y_true):,} samples.")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DIN / BST on KuaiRand-1K"
    )
    parser.add_argument(
        "--model", choices=["DIN", "BST", "both"], default="both",
        help="Which model(s) to evaluate (default: both)",
    )
    parser.add_argument(
        "--gpu", type=int, default=-1,
        help="GPU index for inference (-1 = CPU, default: -1)",
    )
    parser.add_argument(
        "--out_csv", default="./results/evaluation_results.csv",
        help="Path to save the combined results CSV",
    )
    args = parser.parse_args()

    models = ["DIN", "BST"] if args.model == "both" else [args.model]
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    all_results = {}
    for m in models:
        all_results[m] = evaluate_model(m, args.gpu)
        print_results_table(m, all_results[m])

    # Side-by-side comparison if both models were evaluated
    if len(models) == 2:
        print(f"\n{'='*72}")
        print("  COMBINED COMPARISON")
        print(f"{'='*72}")
        print(f"  {'Metric':<22}  {'DIN-Train':>11}  {'DIN-Test':>10}  "
              f"{'BST-Train':>11}  {'BST-Test':>10}")
        print(f"  {'─'*70}")
        for metric in ["Average Precision", "Log Loss", "AUC-ROC",
                       "Samples", "Positive Samples"]:
            vals = []
            for m in ["DIN", "BST"]:
                for split in ["train", "test"]:
                    v = all_results[m][split][metric]
                    vals.append(f"{v:.6f}" if isinstance(v, float) else f"{v:,}")
            print(f"  {metric:<22}  {vals[0]:>11}  {vals[1]:>10}  "
                  f"{vals[2]:>11}  {vals[3]:>10}")
        print(f"  {'─'*70}\n")

    save_results_csv(all_results, args.out_csv)
    print(f"\n[{datetime.now():%H:%M:%S}] Evaluation complete.")


if __name__ == "__main__":
    main()