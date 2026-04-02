"""
conftest.py
-----------
Pytest configuration for the KuaiRand CVR test suite.

CLI options
-----------
--data-dir  Path to the KuaiRand-1K data directory (six CSV files).
            Required for TestRealData; all real-data tests are skipped
            automatically when omitted.

--device    Torch device for HKG construction and model tests.
            Accepts any valid torch.device string: cpu, cuda, cuda:0,
            cuda:1, mps, etc.  Defaults to 'cuda' if a GPU is available,
            then 'mps' (Apple Silicon), then 'cpu'.

Usage
-----
# Synthetic data, CPU (default):
    pytest tests.py -v

# Real KuaiRand-1K data, auto device:
    pytest tests.py -v --data-dir /path/to/KuaiRand-1K/data

# Real data, explicit GPU:
    pytest tests.py -v --data-dir /path/to/KuaiRand-1K/data --device cuda

# Real data, specific GPU index:
    pytest tests.py -v --data-dir /path/to/KuaiRand-1K/data --device cuda:1

# Real data, Apple Silicon:
    pytest tests.py -v --data-dir /path/to/KuaiRand-1K/data --device mps

# Real data, force CPU:
    pytest tests.py -v --data-dir /path/to/KuaiRand-1K/data --device cpu
"""

from pathlib import Path

import torch
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--data-dir",
        action="store",
        default=None,
        help="Path to the KuaiRand-1K data directory containing the six CSV files.",
    )
    parser.addoption(
        "--device",
        action="store",
        default=None,
        help=(
            "Torch device for HKG and model tests "
            "(e.g. cpu, cuda, cuda:0, cuda:1, mps). "
            "Defaults to 'cuda' if available, otherwise 'cpu'."
        ),
    )


@pytest.fixture(scope="session")
def real_data_dir(request) -> Path | None:
    """Returns the Path supplied via --data-dir, or None if not provided."""
    raw = request.config.getoption("--data-dir")
    if raw is None:
        return None
    p = Path(raw)
    if not p.is_dir():
        pytest.fail(f"--data-dir does not exist or is not a directory: {p}")
    return p


@pytest.fixture(scope="session")
def device(request) -> torch.device:
    """
    Returns the torch.device to use for all HKG construction and model tests.

    Resolution order:
      1. --device CLI flag if provided (always honoured; fails fast if invalid)
      2. 'cuda' if torch.cuda.is_available()
      3. 'mps'  if torch.backends.mps.is_available()   (Apple Silicon)
      4. 'cpu'  fallback
    """
    raw = request.config.getoption("--device")

    if raw is not None:
        try:
            d = torch.device(raw)
            torch.zeros(1, device=d)   # confirm device is actually usable
        except (RuntimeError, AssertionError) as exc:
            pytest.fail(f"--device '{raw}' is not available: {exc}")
        return d

    # Auto-detect best available device
    if torch.cuda.is_available():
        d = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        d = torch.device("mps")
    else:
        d = torch.device("cpu")

    return d