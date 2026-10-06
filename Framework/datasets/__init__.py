"""
Dataset adapters (spec 06).  `load_dataset(args)` returns the DatasetBundle for
args.dataset; see base.py for the contract.  MIND and ZhihuRec bundles are
cached (pickle) under <cache-dir>/datasets/, keyed by every argument that
changes them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DATASETS = ("kuairand", "mind", "zhihurec")
# Arguments that change an adapter's output (part of the cache key)
CACHE_KEY_ARGS = ("data_dir", "user_frac", "skip_edges", "entity_init", "label_delay_s",
                  "snapshot_features", "val_ratio", "test_ratio")


def _adapter(name: str):
    if name == "mind":
        from datasets import mind
        return mind
    if name == "zhihurec":
        from datasets import zhihurec
        return zhihurec
    raise ValueError(f"unknown dataset {name!r}; choose from {DATASETS}")


def cache_path(args) -> Path | None:
    from main import resolve_cache_dir
    root = resolve_cache_dir(args)
    if root is None:
        return None
    mod = _adapter(args.dataset)
    key = {k: (str(Path(getattr(args, k)).resolve()) if k == "data_dir" else getattr(args, k, None))
           for k in CACHE_KEY_ARGS}
    key["adapter_version"] = mod.ADAPTER_VERSION
    h = hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return root / "datasets" / f"{args.dataset}_{h}.pkl"


def load_dataset(args, **kw) -> "DatasetBundle":
    name = getattr(args, "dataset", "kuairand")
    if name == "kuairand":
        from datasets import kuairand
        return kuairand.load(args, **kw)
    path = cache_path(args)
    if path is not None and path.exists() and not getattr(args, "rebuild_hkg", False):
        t0 = time.time()
        with open(path, "rb") as f:
            ds = pickle.load(f)
        logger.info("%s bundle loaded from %s in %.1fs", name, path, time.time() - t0)
        return ds
    t0 = time.time()
    ds = _adapter(name).load(args)
    logger.info("%s adapter: %s rows, %.0fs", name, f"{len(ds.inter.time):,}", time.time() - t0)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(ds, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
        except BaseException:
            os.unlink(tmp)
            raise
        logger.info("%s bundle cached → %s (%.0f MB)", name, path, path.stat().st_size / 1e6)
    return ds


def __getattr__(name):
    # lazy, so features.py can import datasets.kuairand_columns without a cycle
    if name in ("DatasetBundle", "GraphSpec"):
        from datasets import base
        return getattr(base, name)
    raise AttributeError(name)


__all__ = ["DATASETS", "DatasetBundle", "GraphSpec", "load_dataset"]
