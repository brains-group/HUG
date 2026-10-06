"""
Dataset adapters (spec 06).  `load_dataset(args)` returns the DatasetBundle for
args.dataset; see base.py for the contract.
"""

from __future__ import annotations

from datasets.base import DatasetBundle, GraphSpec

DATASETS = ("kuairand", "mind", "zhihurec")


def load_dataset(args, **kw) -> DatasetBundle:
    name = getattr(args, "dataset", "kuairand")
    if name == "kuairand":
        from datasets import kuairand
        return kuairand.load(args, **kw)
    if name == "mind":
        from datasets import mind
        return mind.load(args)
    if name == "zhihurec":
        from datasets import zhihurec
        return zhihurec.load(args)
    raise ValueError(f"unknown dataset {name!r}; choose from {DATASETS}")


__all__ = ["DATASETS", "DatasetBundle", "GraphSpec", "load_dataset"]
