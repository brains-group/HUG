"""
train.py
========
Trains and evaluates the FuxiCTR baselines (TransAct, WuKong) on KuaiRand-1K.

Training follows FuxiCTR's run_expid.py: early stopping and checkpoint
selection on the validation split, best weights reloaded at the end.  The
selected model then scores valid and test once, and metrics are computed with
HUG's own compute_metrics (AUC, AP, LogLoss, per-user nDCG@10) so baseline
and HUG numbers are directly comparable.

Usage:
    python train.py --model TransAct --gpu 0
    python train.py --model all --gpu 0 --seeds 2024 2025 2026

Prerequisites:
    python preprocess.py          → ./data/processed/kuairand_1k/*.csv
    bash setup_fuxictr.sh         → ./FuxiCTR (model_zoo sources, pinned)

Output:
    ../runs/<Model>[_seed<n>]/final_metrics.json   (same layout as HUG runs)
    ../runs/<Model>[_seed<n>]/<expid>.log          (FuxiCTR training log)
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT         = Path(__file__).resolve().parent
CONFIG_DIR   = ROOT / "config"
FUXICTR_ROOT = Path(os.environ.get("FUXICTR_ROOT", ROOT / "FuxiCTR"))
RUNS_DIR     = ROOT.parent / "runs"

MODELS = {
    "TransAct": "TransAct_kuairand_1k",
    "WuKong":   "WuKong_kuairand_1k",
}


def run(model_name: str, gpu: int, seed: int | None,
        epochs: int | None = None, runs_dir: Path = RUNS_DIR) -> dict:
    from fuxictr.features import FeatureMap
    from fuxictr.preprocess import FeatureProcessor, build_dataset
    from fuxictr.pytorch.dataloaders import RankDataLoader
    from fuxictr.pytorch.torch_utils import seed_everything
    from fuxictr.utils import load_config, print_to_json, set_logger

    expid = MODELS[model_name]
    zoo_dir = FUXICTR_ROOT / "model_zoo" / model_name
    if not zoo_dir.is_dir():
        raise FileNotFoundError(f"{zoo_dir} not found — run setup_fuxictr.sh first")

    # Each model_zoo entry ships its own `src` package; make this one importable
    for mod in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
        del sys.modules[mod]
    sys.path.insert(0, str(zoo_dir))
    import src  # noqa: E402

    params = load_config(str(CONFIG_DIR), expid)
    params["gpu"] = gpu
    if epochs is not None:
        params["epochs"] = epochs
    if seed is not None:
        params["seed"] = seed
        params["model_id"] = f"{expid}_seed{seed}"
    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params["seed"])

    data_dir = os.path.join(params["data_root"], params["dataset_id"])
    feature_encoder = FeatureProcessor(**params)
    params["train_data"], params["valid_data"], params["test_data"] = \
        build_dataset(feature_encoder, **params)
    feature_map = FeatureMap(params["dataset_id"], data_dir)
    feature_map.load(os.path.join(data_dir, "feature_map.json"), params)

    model = getattr(src, params["model"])(feature_map, **params)
    model.count_parameters()

    train_gen, valid_gen = RankDataLoader(feature_map, stage="train", **params).make_iterator()
    model.fit(train_gen, validation_data=valid_gen, **params)   # reloads best-on-valid weights
    del train_gen
    gc.collect()

    # HUG's metric implementation, imported lazily so its logging setup does
    # not pre-empt FuxiCTR's
    sys.path.insert(0, str(ROOT.parent / "Framework"))
    from main import compute_metrics

    results = {}
    processed = Path(params["train_data"]).parent
    for split, gen in [
        ("val",  valid_gen),
        ("test", RankDataLoader(feature_map, stage="test", **params).make_iterator()),
    ]:
        csv = processed.parent / "kuairand_1k" / ("valid.csv" if split == "val" else "test.csv")
        ref = pd.read_csv(csv, usecols=["user_id", "is_click"])
        y_pred = model.predict(gen)
        if len(y_pred) != len(ref):
            raise RuntimeError(f"{split}: {len(y_pred)} predictions for {len(ref)} rows")
        y_true = ref["is_click"].to_numpy(dtype=np.float64)
        loss = float(-np.mean(y_true * np.log(np.clip(y_pred, 1e-7, 1))
                              + (1 - y_true) * np.log(np.clip(1 - y_pred, 1e-7, 1))))
        m = compute_metrics(split, y_true, y_pred, loss,
                            user_idxs=ref["user_id"].to_numpy())
        logging.info(str(m))
        results[split] = asdict(m)

    run_name = model_name if seed is None else f"{model_name}_seed{seed}"
    out_dir = runs_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "final_metrics.json").write_text(json.dumps({
        **results,
        "params": {k: v for k, v in params.items() if isinstance(v, (int, float, str, list, bool, type(None)))},
        "n_params": int(sum(p.numel() for p in model.parameters())),
    }, indent=2, default=str))
    log_file = Path(params["model_root"]) / params["dataset_id"] / f"{params['model_id']}.log"
    if log_file.exists():
        shutil.copy(log_file, out_dir / log_file.name)
    logging.info("Results → %s", out_dir)
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="all", choices=list(MODELS) + ["all"])
    p.add_argument("--gpu",   type=int, default=0, help="GPU index, -1 for CPU")
    p.add_argument("--seeds", type=int, nargs="*", default=None,
                   help="Override the config seed; one run per seed")
    p.add_argument("--epochs", type=int, default=None, help="Override the config epochs")
    p.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    args = p.parse_args()

    names = list(MODELS) if args.model == "all" else [args.model]
    seeds = args.seeds or [None]
    for name in names:
        for seed in seeds:
            run(name, args.gpu, seed, args.epochs, args.runs_dir)


if __name__ == "__main__":
    main()
