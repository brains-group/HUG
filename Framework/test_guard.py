"""
Test-set guard (spec 04 W4.2)
-----------------------------
Scoring the test split requires either a one-time token issued by the run
queue for a `final_eval` job (a file `<runs_root>/.test_tokens/<token>`, consumed
on use) or the explicit manual override `--i-know-this-touches-test`.
Every allowed access is appended to `<runs_root>/test_access.log`.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path


class TestAccessDenied(PermissionError):
    pass


def authorize_test_access(
    runs_root:   Path,
    token:       str | None,
    override:    bool,
    job:         str,
    config_hash: str | None,
) -> None:
    runs_root = Path(runs_root)
    if override:
        how = "manual-override"
    else:
        if not token:
            raise TestAccessDenied(
                "--eval-test needs a --test-access-token issued by scripts/run_queue.py for a "
                "final_eval job (or --i-know-this-touches-test for deliberate manual use)."
            )
        tok = runs_root / ".test_tokens" / token
        if not tok.is_file():
            raise TestAccessDenied(f"test access token {token!r} is not valid for {runs_root}")
        tok.unlink()
        how = "token"
    runs_root.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().isoformat(timespec="seconds")
    with open(runs_root / "test_access.log", "a") as f:
        f.write(f"{stamp}\t{job}\t{config_hash or '-'}\t{how}\n")
