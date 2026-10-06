"""
Run-level helpers shared by HUG and the baseline wrapper: determinism,
environment logging, and quiet progress output.
"""

from __future__ import annotations

import logging
import os
import random
import sys

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_determinism(seed: int) -> None:
    """
    Seed python, numpy and torch (CPU + CUDA) and disable cuDNN autotuning.
    CPU runs are bitwise reproducible; GPU runs are reproducible to within noise.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def environment_versions() -> dict[str, str]:
    import pandas
    import sklearn
    import torch_geometric
    out = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "torch_geometric": torch_geometric.__version__,
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "sklearn": sklearn.__version__,
    }
    try:
        import fuxictr
        out["fuxictr"] = fuxictr.__version__
    except ImportError:
        pass
    return out


def log_versions() -> dict[str, str]:
    v = environment_versions()
    logger.info("Environment: %s", "  ".join(f"{k}={x}" for k, x in v.items()))
    return v


def is_quiet(flag: bool = False) -> bool:
    """Quiet output when asked to, or when stdout is not a terminal."""
    return flag or not sys.stdout.isatty()


class Progress:
    """
    tqdm on a terminal; otherwise one log line every `every` updates.
    Mirrors the subset of the tqdm API the training loops use.
    """

    def __init__(self, total: int, desc: str, quiet: bool, every: int = 200, unit: str = "row"):
        self.total, self.desc, self.quiet, self.every = total, desc, quiet, every
        self.n, self._updates, self._postfix = 0, 0, {}
        self._bar = None
        if not quiet:
            from tqdm import tqdm
            self._bar = tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True, leave=False)

    def update(self, n: int) -> None:
        self.n += n
        self._updates += 1
        if self._bar is not None:
            self._bar.update(n)
        elif self._updates % self.every == 0:
            post = "  ".join(f"{k}={v}" for k, v in self._postfix.items())
            logger.info("%s  %d/%d  %s", self.desc, self.n, self.total, post)

    def set_postfix(self, d: dict) -> None:
        self._postfix = d
        if self._bar is not None:
            self._bar.set_postfix(d)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
