"""
train.py
========
Trains and evaluates one FuxiCTR baseline (TransAct, WuKong, FiGNN, DCNv2) on
KuaiRand-1K as a single job (spec 04 job contract).

Training follows FuxiCTR's run_expid.py: early stopping and checkpoint
selection on the validation split, best weights reloaded at the end.  The
selected model then scores val and the training holdout (and test, only for a
queue-issued final_eval), and metrics are computed with Framework/metrics.py —
the same code, schema and bucket keys as HUG.

Usage:
    python train.py --model TransAct --gpu 0 --run-dir ../runs/dev/TransAct
    python train.py --model WuKong --params best.yaml --seed 43 --run-dir ...

Prerequisites:
    python preprocess.py          → ./data/processed/kuairand_1k_csv/*.csv
    bash setup_fuxictr.sh         → ./FuxiCTR (model_zoo sources, pinned)

Run directory:
    final_metrics.json, val_preds.npz (row_id, user, label, score),
    test_preds.npz (final_eval only), best.pt, resolved_config.yaml, <model>.log
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import gc
import glob
import itertools
import json
import logging
import os
import pickle
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT         = Path(__file__).resolve().parent
CONFIG_DIR   = ROOT / "config"
FUXICTR_ROOT = Path(os.environ.get("FUXICTR_ROOT", ROOT / "FuxiCTR"))
CSV_DIR      = ROOT / "data" / "processed" / "kuairand_1k_csv"
sys.path.insert(0, str(ROOT.parent / "Framework"))

from metrics import compute_metrics  # noqa: E402
from runtime import environment_versions, git_state, is_quiet, set_determinism  # noqa: E402
from test_guard import authorize_test_access  # noqa: E402

MODELS = {
    "TransAct": "TransAct_kuairand_1k",
    "WuKong":   "WuKong_kuairand_1k",
    "FiGNN":    "FiGNN_kuairand_1k",
    "DCNv2":    "DCNv2_kuairand_1k",
}
BUCKETS = {"video_train_count": "bucket_video_train_count", "coldwarm": "bucket_coldwarm",
           "history_len": "bucket_history_len"}
DRY_RUN_EVAL_BATCHES = 20


class _Limited:
    """First `n` batches of a FuxiCTR data generator (dry runs)."""

    def __init__(self, gen, n: int) -> None:
        self.gen, self.n = gen, n

    def __iter__(self):
        return itertools.islice(iter(self.gen), self.n)

    def __len__(self) -> int:
        return min(self.n, len(self.gen))


def _import_model_zoo(model_name: str):
    zoo_dir = FUXICTR_ROOT / "model_zoo" / model_name
    if not zoo_dir.is_dir():
        raise FileNotFoundError(f"{zoo_dir} not found — run setup_fuxictr.sh first")
    for mod in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
        del sys.modules[mod]
    sys.path.insert(0, str(zoo_dir))
    import src  # noqa: E402
    return src


@contextlib.contextmanager
def _dataset_lock(data_dir: Path):
    """
    Exclusive lock on a processed-data directory.  Concurrent jobs (two GPUs, --pack,
    tuning trials) share these directories; FuxiCTR writes feature_map.json before the
    parquet files, so without the lock a second job could read a half-built dataset.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / ".build.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _build_dataset_once(feature_encoder, data_dir: Path, params: dict) -> tuple:
    """build_dataset under the lock; a directory without the .built marker is rebuilt."""
    from fuxictr.preprocess import build_dataset
    with _dataset_lock(data_dir):
        marker = data_dir / ".built"
        if not marker.exists():
            # Interrupted or pre-marker build: drop what FuxiCTR would otherwise reuse
            for stale in ("feature_map.json", "holdout.parquet"):
                (data_dir / stale).unlink(missing_ok=True)
        paths = build_dataset(feature_encoder, **params)
        marker.touch()
    return paths


def _transform_extra(data_dir: Path, name: str, csv: Path) -> str:
    """Transform an extra CSV (the holdout) with the fitted feature processor."""
    from fuxictr.preprocess.build_dataset import transform
    out = data_dir / f"{name}.parquet"
    with _dataset_lock(data_dir):
        if not out.exists():
            with open(data_dir / "feature_processor.pkl", "rb") as f:
                fe = pickle.load(f)
            ddf = fe.read_data(str(csv), data_format="csv")
            transform(fe, fe.preprocess(ddf), name)
    return str(data_dir / name)


def _score(model, gen, ref: pd.DataFrame, split: str):
    y_pred = model.predict(gen)
    ref = ref.iloc[:len(y_pred)]
    if len(y_pred) != len(ref):
        raise RuntimeError(f"{split}: {len(y_pred)} predictions for {len(ref)} rows")
    y = ref["is_click"].to_numpy(dtype=np.float64)
    p = np.clip(y_pred, 1e-7, 1 - 1e-7)
    loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    m = compute_metrics(split, y, y_pred, loss, user_idxs=ref["user_id"].to_numpy(),
                        buckets={k: ref[c].astype(str).to_numpy() for k, c in BUCKETS.items()})
    preds = {"row_id": ref["row_id"].to_numpy(np.int64), "user": ref["user_id"].to_numpy(np.int64),
             "label": y.astype(np.float32), "score": y_pred.astype(np.float32)}
    return m, preds


def run(args) -> dict:
    run_dir = Path(args.run_dir).resolve()
    done = run_dir / "final_metrics.json"
    if done.exists() and args.config_hash and not args.eval_only:
        if json.loads(done.read_text()).get("config_hash") == args.config_hash:
            print(f"[train] {run_dir} already complete for this config — skipping")
            return json.loads(done.read_text())
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.eval_test:
        authorize_test_access(run_dir.parent, args.test_access_token,
                              args.i_know_this_touches_test, run_dir.name, args.config_hash)

    from fuxictr.features import FeatureMap
    from fuxictr.preprocess import FeatureProcessor
    from fuxictr.pytorch.dataloaders import RankDataLoader
    from fuxictr.pytorch.torch_utils import seed_everything
    from fuxictr.utils import load_config, print_to_json, set_logger

    src = _import_model_zoo(args.model)
    params = load_config(str(CONFIG_DIR), MODELS[args.model])
    overrides = yaml.safe_load(Path(args.params).read_text()) if args.params else {}
    params.update(overrides or {})
    # dataset_config.yaml paths are relative to Baselines/ (the repo can live anywhere)
    for key in ("data_root", "train_data", "valid_data", "test_data"):
        if params.get(key) and not os.path.isabs(params[key]):
            params[key] = str(ROOT / params[key])
    params["gpu"] = args.gpu
    params["seed"] = args.seed
    if args.epochs is not None:
        params["epochs"] = args.epochs
    if args.max_steps:
        params["epochs"] = 1
    params["verbose"] = 0 if is_quiet(args.quiet) else 1
    # One processed-data cache per vocabulary threshold (fitted on train.csv only)
    params["dataset_id"] = f"{params['dataset_id']}_m{params['min_categr_count']}"
    params["model_root"] = str(run_dir)
    params["model_id"] = args.model
    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(
        {k: v for k, v in params.items() if isinstance(v, (int, float, str, list, dict, bool, type(None)))}))
    set_determinism(args.seed)
    seed_everything(seed=args.seed)

    data_dir = Path(params["data_root"]) / params["dataset_id"]
    feature_encoder = FeatureProcessor(**params)
    params["train_data"], params["valid_data"], params["test_data"] = _build_dataset_once(
        feature_encoder, data_dir, params)
    feature_map = FeatureMap(params["dataset_id"], str(data_dir))
    feature_map.load(str(data_dir / "feature_map.json"), params)

    model = getattr(src, params["model"])(feature_map, **params)
    n_params = int(sum(p.numel() for p in model.parameters()))
    ckpt = run_dir / "best.pt"

    valid_gen = None
    if args.eval_only:
        model.load_weights(args.checkpoint or str(ckpt))
    else:
        train_gen, valid_gen = RankDataLoader(feature_map, stage="train", **params).make_iterator()
        if args.max_steps:
            train_gen = _Limited(train_gen, args.max_steps)
            valid_gen = _Limited(valid_gen, DRY_RUN_EVAL_BATCHES)
        model.fit(train_gen, validation_data=valid_gen, **params)   # reloads best-on-valid weights
        del train_gen
        gc.collect()
        shutil.move(model.checkpoint, ckpt)                          # keep only the best weights
        for f in glob.glob(str(run_dir / params["dataset_id"] / "*.model")):
            os.remove(f)

    if valid_gen is None:
        _, valid_gen = RankDataLoader(feature_map, stage="train", **params).make_iterator()
        if args.max_steps:
            valid_gen = _Limited(valid_gen, DRY_RUN_EVAL_BATCHES)

    ref_cols = ["row_id", "user_id", "is_click", *BUCKETS.values()]
    val_m, val_p = _score(model, valid_gen, pd.read_csv(CSV_DIR / "valid.csv", usecols=ref_cols), "val")
    np.savez(run_dir / "val_preds.npz", **val_p)

    hold_path = _transform_extra(data_dir, "holdout", CSV_DIR / "holdout.csv")
    hold_gen = RankDataLoader(feature_map, stage="test", **{**params, "test_data": hold_path}).make_iterator()
    if args.max_steps:
        hold_gen = _Limited(hold_gen, DRY_RUN_EVAL_BATCHES)
    hold_m, _ = _score(model, hold_gen, pd.read_csv(CSV_DIR / "holdout.csv", usecols=ref_cols), "holdout")

    test_m = None
    dry_substitute = bool(args.eval_test and args.max_steps)
    if args.eval_test:
        if dry_substitute:       # dry runs exercise the final_eval path on val rows: no test label is read
            test_gen = _Limited(RankDataLoader(feature_map, stage="train", **params).make_iterator()[1],
                                DRY_RUN_EVAL_BATCHES)
            test_ref = pd.read_csv(CSV_DIR / "valid.csv", usecols=ref_cols)
        else:
            test_gen = RankDataLoader(feature_map, stage="test", **params).make_iterator()
            test_ref = pd.read_csv(CSV_DIR / "test.csv", usecols=ref_cols)
        test_m, test_p = _score(model, test_gen, test_ref, "test")
        np.savez(run_dir / "test_preds.npz", **test_p)

    logging.info(str(val_m))
    logging.info(str(hold_m))
    out = {
        "job": run_dir.name, "kind": "baseline", "model": args.model, "arm": None,
        "seed": args.seed, "config_hash": args.config_hash,
        "val": asdict(val_m), "holdout": asdict(hold_m),
        "test": asdict(test_m) if test_m is not None else None,
        "test_is_dry_run_substitute": dry_substitute,
        "best_epoch": None, "params": {k: v for k, v in params.items()
                                       if isinstance(v, (int, float, str, bool, type(None)))},
        "tuned": overrides, "n_params": n_params,
        "checkpoint_mb": round(ckpt.stat().st_size / 1e6, 1) if ckpt.exists() else None,
        "environment": environment_versions(),
        "git": git_state(),
    }
    done.write_text(json.dumps(out, indent=2, default=float))
    logging.info("Results → %s", run_dir)
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--gpu", type=int, default=0, help="GPU index, -1 for CPU")
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--params", default=None, help="YAML of model-config overrides (tuned settings)")
    p.add_argument("--epochs", type=int, default=None, help="Override the config epochs")
    p.add_argument("--max-steps", type=int, default=0, help="dry run: N training batches")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--eval-test", action="store_true")
    p.add_argument("--test-access-token", default=None)
    p.add_argument("--i-know-this-touches-test", action="store_true")
    p.add_argument("--config-hash", default=None)
    p.add_argument("--data-dir", default=None, help="accepted for CLI symmetry; CSVs come from preprocess.py")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
