"""
step3b_evaluate_standalone.py
==============================
Alternative evaluator that does NOT require importing FuxiCTR internals.

Use this if step3_evaluate.py has path / import issues, or if you want to
run metrics on already-saved prediction files.

Workflow:
  1.  Calls `run_expid.py --expid <EXPID> --gpu <GPU>` for each model in
      "predict" mode (adds `--test_only` flag recognised by FuxiCTR ≥ v2).
      FuxiCTR writes a CSV of predictions automatically.

  2.  If FuxiCTR's --test_only mode is unavailable, falls back to parsing
      the training log to find test AUC / logloss, and computes the
      remaining metrics (AP, sample counts) directly from the test CSV +
      the prediction CSV that FuxiCTR writes.

  3.  For train-set metrics, replays predictions from the saved .npy or
      .csv files that FuxiCTR optionally writes, or re-runs inference.

Usage:
    python step3b_evaluate_standalone.py
    python step3b_evaluate_standalone.py --model DIN --gpu 0
"""

import os
import re
import sys
import glob
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    roc_auc_score,
)


# ── paths ──────────────────────────────────────────────────────────────────
FUXICTR_ROOT  = os.environ.get("FUXICTR_ROOT", "./FuxiCTR")
PROCESSED_DIR = "./data/processed"
RESULTS_DIR   = "./results"
CONFIG_DIR    = str(Path(__file__).resolve().parent / "config")

MODEL_ZOO_PATHS = {
    "DIN": os.path.join(FUXICTR_ROOT, "model_zoo", "DIN", "DIN_torch"),
    "BST": os.path.join(FUXICTR_ROOT, "model_zoo", "BST", "BST_torch"),
}
EXPID_MAP = {
    "DIN": "DIN_kuairand_1k",
    "BST": "BST_kuairand_1k",
}


os.makedirs(RESULTS_DIR, exist_ok=True)


# ── utilities ──────────────────────────────────────────────────────────────
def find_log_file(model_name: str, expid: str) -> str | None:
    patterns = [
        os.path.join(FUXICTR_ROOT, "checkpoints", f"{expid}*", "*.log"),
        os.path.join(MODEL_ZOO_PATHS[model_name], "logs", f"{expid}*.log"),
    ]
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            return max(hits, key=os.path.getmtime)
    return None


def parse_log_metrics(log_path: str) -> dict:
    """
    Extract final test logloss and AUC from a FuxiCTR training log.
    Log lines look like:
      [test] logloss: 0.5432 - AUC: 0.7123
    """
    result = {}
    if not log_path or not os.path.exists(log_path):
        return result

    pat = re.compile(
        r"\[test\].*logloss:\s*([\d.]+).*AUC:\s*([\d.]+)", re.IGNORECASE
    )
    with open(log_path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                result["Log Loss"] = float(m.group(1))
                result["AUC-ROC"]  = float(m.group(2))
    return result


def find_prediction_csv(model_name: str, expid: str) -> str | None:
    """
    FuxiCTR (v2+) writes test predictions to a CSV when run with
    --test_only.  Locate the most recent one.
    """
    patterns = [
        os.path.join(FUXICTR_ROOT, "checkpoints", f"{expid}*", "*pred*.csv"),
        os.path.join(FUXICTR_ROOT, "checkpoints", f"{expid}*", "*test*.csv"),
        os.path.join(RESULTS_DIR, f"{expid}_*pred*.csv"),
    ]
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            return max(hits, key=os.path.getmtime)
    return None


def run_fuxictr_predict(model_name: str, expid: str, gpu: int) -> int:
    """
    Attempt to run FuxiCTR in predict-only mode.
    Returns subprocess returncode.
    """
    model_dir = MODEL_ZOO_PATHS[model_name]
    cmd = [
        sys.executable, "run_expid.py",
        "--expid", expid,
        "--gpu", str(gpu),
        "--config", CONFIG_DIR,
        "--test_only",       # FuxiCTR ≥ v2 flag; older versions may ignore it
    ]
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=model_dir)
    return result.returncode


def direct_metrics_from_csv(split_csv: str, pred_col: str = "pred_score") -> dict | None:
    """
    If a prediction CSV was saved alongside the data CSV, merge and compute.
    Returns None if the pred_col is missing.
    """
    df = pd.read_csv(split_csv)
    if pred_col not in df.columns:
        return None

    y_true = df["is_click"].values
    y_pred = df[pred_col].values

    return _compute_metrics(y_true, y_pred)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
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


def count_only_from_csv(csv_path: str) -> dict:
    """Return sample / positive counts when we only have labels."""
    df = pd.read_csv(csv_path, usecols=["is_click"])
    y  = df["is_click"].values
    return {
        "Average Precision": float("nan"),
        "Log Loss":          float("nan"),
        "AUC-ROC":           float("nan"),
        "Samples":           len(y),
        "Positive Samples":  int(y.sum()),
    }


# ── main evaluation logic ─────────────────────────────────────────────────
def evaluate_model(model_name: str, gpu: int) -> dict:
    expid = EXPID_MAP[model_name]
    print(f"\n{'='*60}")
    print(f"  Evaluating {model_name}  (expid={expid})")
    print(f"{'='*60}")

    # ── Step A: try FuxiCTR predict mode to get test predictions ──────────
    print(f"\n[1/3] Running FuxiCTR predict mode …")
    rc = run_fuxictr_predict(model_name, expid, gpu)
    pred_csv = find_prediction_csv(model_name, expid)

    # ── Step B: parse training log for baseline logloss / AUC ────────────
    print(f"\n[2/3] Parsing training log for test metrics …")
    log_path   = find_log_file(model_name, expid)
    log_result = parse_log_metrics(log_path)
    if log_result:
        print(f"  From log → logloss={log_result.get('Log Loss'):.6f}  "
              f"AUC={log_result.get('AUC-ROC'):.6f}")
    else:
        print(f"  No log metrics found (log_path={log_path})")

    # ── Step C: compute full metrics from prediction CSV ──────────────────
    print(f"\n[3/3] Computing full metric set …")

    results = {}
    for split_name, csv_path in [
        ("train", os.path.join(PROCESSED_DIR, "train.csv")),
        ("test",  os.path.join(PROCESSED_DIR, "test.csv")),
    ]:
        # Try the FuxiCTR prediction CSV first
        m = None
        if pred_csv and split_name == "test":
            try:
                pred_df = pd.read_csv(pred_csv)
                label_df = pd.read_csv(csv_path, usecols=["is_click"])
                if len(pred_df) == len(label_df):
                    # Assume first numeric column is the prediction score
                    score_col = [c for c in pred_df.columns
                                 if pred_df[c].dtype in [float, np.float32, np.float64]]
                    if score_col:
                        y_pred = pred_df[score_col[0]].values
                        y_true = label_df["is_click"].values
                        m = _compute_metrics(y_true, y_pred)
            except Exception as e:
                print(f"  Warning: could not parse prediction CSV: {e}")

        # Fallback: use log metrics for test, counts-only for train
        if m is None and split_name == "test" and log_result:
            df      = pd.read_csv(csv_path, usecols=["is_click"])
            y       = df["is_click"].values
            m = {
                "Average Precision": float("nan"),   # needs predictions
                "Log Loss":          log_result.get("Log Loss", float("nan")),
                "AUC-ROC":           log_result.get("AUC-ROC", float("nan")),
                "Samples":           len(y),
                "Positive Samples":  int(y.sum()),
            }
            print(f"  NOTE: Average Precision unavailable without saved predictions.")
            print(f"        Run step3_evaluate.py for full metrics if FuxiCTR")
            print(f"        Python API is importable.")

        if m is None:
            m = count_only_from_csv(csv_path)

        results[split_name] = m

    return results


def print_table(model_name: str, results: dict):
    header = f"\n{'='*62}\n  {model_name} — Results\n{'='*62}"
    print(header)
    print(f"  {'Metric':<22}  {'Train':>13}  {'Test':>13}")
    print(f"  {'─'*60}")
    for metric in ["Average Precision", "Log Loss", "AUC-ROC",
                   "Samples", "Positive Samples"]:
        tr_v = results["train"][metric]
        te_v = results["test"][metric]
        if isinstance(tr_v, float):
            tr_s = f"{tr_v:.6f}" if not np.isnan(tr_v) else "       N/A"
            te_s = f"{te_v:.6f}" if not np.isnan(te_v) else "       N/A"
        else:
            tr_s = f"{tr_v:>13,}"
            te_s = f"{te_v:>13,}"
        print(f"  {metric:<22}  {tr_s:>13}  {te_s:>13}")
    print()


def save_csv(all_results: dict, out_path: str):
    rows = []
    for model, splits in all_results.items():
        for split, metrics in splits.items():
            row = {"model": model, "split": split}
            row.update(metrics)
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Results saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["DIN", "BST", "both"], default="both")
    parser.add_argument("--gpu",   type=int, default=-1)
    parser.add_argument("--out_csv", default=f"{RESULTS_DIR}/evaluation_results.csv")
    args = parser.parse_args()

    models = ["DIN", "BST"] if args.model == "both" else [args.model]
    all_results = {}
    for m in models:
        all_results[m] = evaluate_model(m, args.gpu)
        print_table(m, all_results[m])

    save_csv(all_results, args.out_csv)
    print(f"\n[{datetime.now():%H:%M:%S}] Done.")


if __name__ == "__main__":
    main()