"""
Heavy-run job queue (Spec 04, W4.1/W4.2).

Reads a plan (``experiments/heavy_run.yaml``) and runs its jobs over the GPUs as a
dependency-respecting queue:

    python scripts/run_queue.py experiments/heavy_run.yaml [--gpus 0,1] [--pack]
        [--dry-run --max-steps 50] [--only <glob>] [--skip-optional]
        [--runs-root runs/heavy]

Behaviour
---------
- Every job runs as a subprocess from the repo root with its own run directory
  ``<runs_root>/<job>/`` (stdout.log / stderr.log go there).
- Dependencies: explicit ``depends_on``, plus the tuning job behind every
  ``{best:<tune_job>}`` placeholder, every ``{run:<job>}`` placeholder (expands to that
  job's run directory) and the stage-1 job named by a final_eval's ``of``.
- Config hash: sha256 over the job's resolved args (minus GPU/run-dir/logging/token
  flags), the content of any ``--params`` file, the data fingerprint and
  ``CODE_COMPAT_VERSION`` (``Framework/runtime.py``, bumped by hand only when a change
  alters existing configurations' results — spec 05 §0). The git commit and dirty-diff
  hash of ``Framework/`` and ``Baselines/`` are recorded in each job's ``job.json`` but do
  not enter the hash, so code landing mid-run does not rerun finished jobs. A job whose
  outputs carry a matching hash is skipped.
- HUG jobs that left ``last.pt`` behind (with the same hash) are resumed with ``--resume``;
  baseline and tuning jobs are restarted.
- A failed job is retried once, then marked failed; its dependents are marked blocked and
  everything else keeps running.
- Test-set discipline: only ``final_eval`` jobs get ``--eval-test``, each with a fresh
  one-time token in ``<runs_root>/.test_tokens/``. A plan that puts ``--eval-test`` in any
  other job is refused.
- ``<runs_root>/status.json`` and ``<runs_root>/queue.log`` track progress and an ETA.
- ``--dry-run`` appends ``--max-steps N`` (default 50) plus the plan's per-kind
  ``dry_run_args`` to every job and writes under ``<runs_root>_dryrun/``.

Testing
-------
``JobQueue`` takes a ``launcher`` callable, so tests can replace subprocesses with stubs:

    launcher(job, cmd, run_dir, stdout_path, stderr_path, cwd) -> process-like

where the returned object has ``poll() -> int | None`` (and optionally ``terminate()``).
The default launcher is ``popen_launcher`` (``subprocess.Popen``).
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import fnmatch
import hashlib
import json
import logging
import os
import re
import secrets
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import yaml

logger = logging.getLogger("run_queue")

REPO_ROOT = Path(__file__).resolve().parents[1]

KINDS = ("hug", "baseline", "tune_hug", "tune_baseline", "final_eval")
TUNE_KINDS = ("tune_hug", "tune_baseline")
STAGE1_EVAL_KINDS = ("hug", "baseline")
MEMORY_CLASSES = ("heavy", "light")
MAX_ATTEMPTS = 2  # first try + one retry

# Flags that never enter the config hash (they change where/how a job runs, not what it computes).
# Value-taking flags drop the following token too.
HASH_EXCLUDED_VALUE_FLAGS = ("--gpu", "--device", "--run-dir", "--out", "--test-access-token",
                             "--config-hash")
HASH_EXCLUDED_BARE_FLAGS = ("--quiet", "--resume")

# Flags only a final_eval job may carry.
TEST_FLAGS = ("--eval-test", "--test-access-token", "--i-know-this-touches-test")

BEST_PREFIX = "{best:"
RUN_PREFIX = "{run:"

TERMINAL_STATES = ("done", "skipped", "failed", "blocked")


class PlanError(ValueError):
    """The plan file is malformed or violates test-set discipline."""


# ── Plan ────────────────────────────────────────────────────────────────────────


@dataclass
class Job:
    name: str
    kind: str
    args: list[str]
    seed: int | None = None
    memory: str = "heavy"
    depends_on: list[str] = field(default_factory=list)
    of: str | None = None
    optional: bool = False
    group: str | None = None          # aggregation label (e.g. "N1-own", "TransAct")
    model: str | None = None
    arm: str | None = None

    # Runtime state
    state: str = "pending"
    gpu: int | None = None
    attempts: int = 0
    start: float | None = None
    end: float | None = None
    config_hash: str | None = None
    reason: str | None = None
    resumed: bool = False

    def best_refs(self) -> list[str]:
        """Tune jobs referenced by ``{best:<name>}`` placeholders in this job's own args."""
        return [a[len(BEST_PREFIX):-1] for a in self.args
                if a.startswith(BEST_PREFIX) and a.endswith("}")]

    def run_refs(self) -> list[str]:
        """Jobs referenced by ``{run:<name>}`` placeholders (their run directory)."""
        return [a[len(RUN_PREFIX):-1] for a in self.args
                if a.startswith(RUN_PREFIX) and a.endswith("}")]

    def metadata(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "args": self.args, "seed": self.seed,
                "memory": self.memory, "depends_on": self.depends_on, "of": self.of,
                "optional": self.optional, "group": self.group, "model": self.model,
                "arm": self.arm}


@dataclass
class Plan:
    defaults: dict[str, Any]
    jobs: dict[str, Job]               # insertion order = plan order

    def all_deps(self, job: Job) -> list[str]:
        """Explicit plus implicit dependencies, in a stable order without duplicates."""
        deps = list(job.depends_on) + job.best_refs() + job.run_refs()
        if job.of:
            deps.append(job.of)
        return list(dict.fromkeys(deps))


def _as_arg_list(raw: Any, job_name: str) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return shlex.split(raw)
    if isinstance(raw, list):
        return [str(a) for a in raw]
    raise PlanError(f"job {job_name}: args must be a string or a list, got {type(raw).__name__}")


def parse_plan(data: dict[str, Any]) -> Plan:
    """Build and validate a Plan from the parsed YAML mapping."""
    if not isinstance(data, dict) or "jobs" not in data:
        raise PlanError("plan must be a mapping with a 'jobs' list")
    defaults = dict(data.get("defaults") or {})
    jobs: dict[str, Job] = {}
    for raw in data["jobs"]:
        name = raw.get("name")
        if not name:
            raise PlanError(f"job without a name: {raw}")
        if name in jobs:
            raise PlanError(f"duplicate job name: {name}")
        kind = raw.get("kind")
        if kind not in KINDS:
            raise PlanError(f"job {name}: unknown kind {kind!r} (expected one of {KINDS})")
        default_mem = (defaults.get("memory") or {}).get(kind, "heavy")
        depends = raw.get("depends_on") or []
        if isinstance(depends, str):
            depends = [depends]
        jobs[name] = Job(
            name=name,
            kind=kind,
            args=_as_arg_list(raw.get("args"), name),
            seed=raw.get("seed"),
            memory=raw.get("memory", default_mem),
            depends_on=[str(d) for d in depends],
            of=raw.get("of"),
            optional=bool(raw.get("optional", False)),
            group=raw.get("group"),
            model=raw.get("model"),
            arm=raw.get("arm"),
        )
    plan = Plan(defaults=defaults, jobs=jobs)
    validate_plan(plan)
    return plan


def load_plan(path: Path) -> Plan:
    with open(path) as fh:
        return parse_plan(yaml.safe_load(fh))


def validate_plan(plan: Plan) -> None:
    """Check kinds, references, memory classes, test-set discipline and acyclicity."""
    for job in plan.jobs.values():
        if job.memory not in MEMORY_CLASSES:
            raise PlanError(f"job {job.name}: memory must be one of {MEMORY_CLASSES}")
        bad = [a for a in job.args if a.split("=")[0] in TEST_FLAGS]
        if bad:
            # final_eval jobs get --eval-test from the queue itself, never from the plan.
            raise PlanError(f"job {job.name}: test-set flags {bad} are not allowed in plan args "
                            f"(only the queue adds --eval-test, and only to final_eval jobs)")
        if job.kind == "final_eval":
            if not job.of or job.of not in plan.jobs:
                raise PlanError(f"final_eval job {job.name}: 'of' must name a plan job")
            if plan.jobs[job.of].kind not in STAGE1_EVAL_KINDS:
                raise PlanError(f"final_eval job {job.name}: 'of' must be a hug or baseline job")
        elif job.of:
            raise PlanError(f"job {job.name}: only final_eval jobs may set 'of'")
        if job.kind in ("hug", "baseline") and job.seed is None:
            raise PlanError(f"job {job.name}: {job.kind} jobs need a seed")
        for ref in job.best_refs():
            if ref not in plan.jobs or plan.jobs[ref].kind not in TUNE_KINDS:
                raise PlanError(f"job {job.name}: {{best:{ref}}} must name a tuning job")
        for dep in plan.all_deps(job):
            if dep not in plan.jobs:
                raise PlanError(f"job {job.name}: unknown dependency {dep}")
    _check_acyclic(plan)


def _check_acyclic(plan: Plan) -> None:
    visiting: set[str] = set()
    done: set[str] = set()

    def visit(name: str, path: list[str]) -> None:
        if name in done:
            return
        if name in visiting:
            raise PlanError("dependency cycle: " + " -> ".join(path + [name]))
        visiting.add(name)
        for dep in plan.all_deps(plan.jobs[name]):
            visit(dep, path + [name])
        visiting.discard(name)
        done.add(name)

    for name in plan.jobs:
        visit(name, [])


def select_jobs(plan: Plan, patterns: list[str] | None, skip_optional: bool = False) -> list[str]:
    """
    Jobs matching any glob in ``patterns`` (all jobs if none) plus their transitive
    dependencies. With ``skip_optional``, optional jobs and everything depending on them
    are dropped.
    """
    names = list(plan.jobs)
    if skip_optional:
        dropped = {n for n in names if plan.jobs[n].optional}
        changed = True
        while changed:
            extra = {n for n in names if n not in dropped
                     and any(d in dropped for d in plan.all_deps(plan.jobs[n]))}
            changed = bool(extra)
            dropped |= extra
        names = [n for n in names if n not in dropped]
    if not patterns:
        return names
    wanted: set[str] = set()
    stack = [n for n in names if any(fnmatch.fnmatch(n, p) for p in patterns)]
    while stack:
        name = stack.pop()
        if name not in wanted:
            wanted.add(name)
            stack.extend(plan.all_deps(plan.jobs[name]))
    return [n for n in plan.jobs if n in wanted]


# ── Fingerprints and config hash ───────────────────────────────────────────────


def _sha256(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


def _git(repo_root: Path, *args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=repo_root, check=True,
                          capture_output=True).stdout


def code_fingerprint(repo_root: Path, dirs: tuple[str, ...] = ("Framework", "Baselines")) -> str:
    """
    Hash of the committed tree of each code dir plus any uncommitted changes in them
    (tracked diffs and untracked, non-ignored files).
    """
    parts = []
    for d in dirs:
        try:
            parts.append(f"{d}:{_git(repo_root, 'rev-parse', f'HEAD:{d}').decode().strip()}")
        except subprocess.CalledProcessError:
            parts.append(f"{d}:absent")
    diff = _git(repo_root, "diff", "--binary", "HEAD", "--", *dirs)
    parts.append(f"diff:{_sha256(diff) if diff else 'clean'}")
    untracked = _git(repo_root, "ls-files", "--others", "--exclude-standard", "--", *dirs)
    for rel in sorted(untracked.decode().splitlines()):
        path = repo_root / rel
        if path.is_file():
            parts.append(f"untracked:{rel}:{_sha256(path.read_bytes())}")
    return _sha256("\n".join(parts))


def code_compat_key(repo_root: Path) -> str:
    """Hash key for code: CODE_COMPAT_VERSION from Framework/runtime.py (read as text)."""
    text = (repo_root / "Framework" / "runtime.py").read_text()
    m = re.search(r"^CODE_COMPAT_VERSION\s*=\s*(\d+)", text, re.MULTILINE)
    if not m:
        raise SystemExit("CODE_COMPAT_VERSION not found in Framework/runtime.py")
    return f"compat:{m.group(1)}"


def data_fingerprint_from_file(path: Path) -> str:
    """Canonical hash of the data fingerprint JSON (cutoffs plus row counts)."""
    with open(path) as fh:
        content = json.load(fh)
    return _sha256(json.dumps(content, sort_keys=True))


def strip_hash_excluded(args: list[str]) -> list[str]:
    """Drop GPU/run-dir/logging/token flags (and their values) from a command's args."""
    out: list[str] = []
    skip_next = False
    for a in args:
        if skip_next:
            skip_next = False
            continue
        flag = a.split("=")[0]
        if flag in HASH_EXCLUDED_BARE_FLAGS:
            continue
        if flag in HASH_EXCLUDED_VALUE_FLAGS:
            skip_next = "=" not in a
            continue
        out.append(a)
    return out


def config_hash(hash_args: list[str], code_fp: str, data_fp: str,
                extra: dict[str, str] | None = None) -> str:
    """sha256 over the hashed args, code fingerprint, data fingerprint and extras."""
    payload = {"args": hash_args, "code": code_fp, "data": data_fp, "extra": extra or {}}
    return _sha256(json.dumps(payload, sort_keys=True))


# ── Launchers ──────────────────────────────────────────────────────────────────


class ProcessLike(Protocol):
    def poll(self) -> int | None: ...


Launcher = Callable[[Job, list[str], Path, Path, Path, Path], ProcessLike]


class _PopenWithLogs:
    """Popen wrapper that closes the log files once the process has exited."""

    def __init__(self, cmd: list[str], cwd: Path, stdout_path: Path, stderr_path: Path):
        self._out = open(stdout_path, "ab")
        self._err = open(stderr_path, "ab")
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        self._proc = subprocess.Popen(cmd, cwd=cwd, stdout=self._out, stderr=self._err,
                                      stdin=subprocess.DEVNULL, env=env)

    def poll(self) -> int | None:
        rc = self._proc.poll()
        if rc is not None and not self._out.closed:
            self._out.close()
            self._err.close()
        return rc

    def terminate(self) -> None:
        self._proc.terminate()
        try:
            self._proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        self.poll()


def popen_launcher(job: Job, cmd: list[str], run_dir: Path, stdout_path: Path,
                   stderr_path: Path, cwd: Path) -> ProcessLike:
    """Default launcher: run ``cmd`` as a subprocess with logs in the run dir."""
    return _PopenWithLogs(cmd, cwd, stdout_path, stderr_path)


# ── Queue ──────────────────────────────────────────────────────────────────────


def _now_iso(ts: float | None = None) -> str | None:
    if ts is None:
        return None
    return dt.datetime.fromtimestamp(ts).isoformat(timespec="seconds")


class JobQueue:
    """Dependency-aware GPU job queue. See the module docstring."""

    def __init__(
        self,
        plan: Plan,
        runs_root: Path,
        gpus: list[int],
        code_fp: str,
        data_fp: str,
        *,
        git_fp: str | None = None,
        repo_root: Path = REPO_ROOT,
        launcher: Launcher = popen_launcher,
        pack: bool = False,
        max_steps: int | None = None,
        dry_run: bool = False,
        only: list[str] | None = None,
        skip_optional: bool = False,
        python: str = sys.executable,
        poll_seconds: float = 10.0,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not gpus:
            raise ValueError("need at least one GPU")
        self.plan = plan
        self.runs_root = Path(runs_root)
        self.gpus = list(gpus)
        self.code_fp = code_fp            # CODE_COMPAT_VERSION key (hashed)
        self.git_fp = git_fp              # git tree + dirty diff (recorded only)
        self.data_fp = data_fp
        self.repo_root = Path(repo_root)
        self.launcher = launcher
        self.pack = pack
        self.max_steps = max_steps
        self.dry_run = dry_run
        self.python = python
        self.poll_seconds = poll_seconds
        self.clock = clock
        self.sleep = sleep

        self.selected = select_jobs(plan, only, skip_optional)
        # Copies, so runtime state never leaks back into the plan.
        self.jobs = {n: copy.deepcopy(plan.jobs[n]) for n in self.selected}
        self.procs: dict[str, ProcessLike] = {}
        self.gpu_jobs: dict[int, list[str]] = {g: [] for g in self.gpus}
        self.started_at = clock()
        self.runs_root.mkdir(parents=True, exist_ok=True)

    # ── Paths ──

    def run_dir(self, name: str) -> Path:
        return self.runs_root / name

    def _rel(self, path: Path) -> str:
        """Path as passed on the command line: relative to the repo root when inside it."""
        path = Path(os.path.abspath(path))
        try:
            return str(path.relative_to(self.repo_root.resolve()))
        except ValueError:
            return str(path)

    # ── Command building ──

    def resolve_args(self, args: list[str]) -> list[str]:
        """Expand ``{best:<tune_job>}`` into ``--params <runs_root>/<tune_job>/best.yaml``."""
        out: list[str] = []
        for a in args:
            if a.startswith(BEST_PREFIX) and a.endswith("}"):
                ref = a[len(BEST_PREFIX):-1]
                out += ["--params", self._rel(self.run_dir(ref) / "best.yaml")]
            elif a.startswith(RUN_PREFIX) and a.endswith("}"):
                out.append(self._rel(self.run_dir(a[len(RUN_PREFIX):-1])))
            else:
                out.append(a)
        return out

    def _base_command(self, job: Job) -> list[str]:
        """The command without per-launch flags (device, hash, token, resume, quiet)."""
        d = self._rel(self.run_dir(job.name))
        if job.kind == "hug":
            return ["Framework/main.py", "--model-type", "hug", "--run-dir", d,
                    "--seed", str(job.seed), *self.resolve_args(job.args)]
        if job.kind == "baseline":
            return ["Baselines/train.py", "--run-dir", d, "--seed", str(job.seed),
                    *self.resolve_args(job.args)]
        if job.kind == "tune_hug":
            return ["Framework/tune_hug.py", "--out", d, *self.resolve_args(job.args)]
        if job.kind == "tune_baseline":
            return ["Baselines/tune.py", "--out", d, *self.resolve_args(job.args)]
        # final_eval: re-run the stage-1 job's eval on test from its val-selected checkpoint.
        stage1 = self.plan.jobs[job.of]
        ckpt = self._rel(self.run_dir(stage1.name) / "best.pt")
        head = (["Framework/main.py", "--model-type", "hug"] if stage1.kind == "hug"
                else ["Baselines/train.py"])
        return [*head, "--run-dir", d, "--eval-only", "--checkpoint", ckpt, "--eval-test",
                "--seed", str(stage1.seed), *self.resolve_args(stage1.args),
                *self.resolve_args(job.args)]

    def _uses_device_flag(self, job: Job) -> bool:
        if job.kind == "final_eval":
            return self.plan.jobs[job.of].kind == "hug"
        return job.kind == "hug"

    def _dry_run_args(self, job: Job) -> list[str]:
        if not self.dry_run:
            return []
        out = ["--max-steps", str(self.max_steps)] if self.max_steps else []
        extra = (self.plan.defaults.get("dry_run_args") or {}).get(job.kind)
        return out + _as_arg_list(extra, job.name)

    def hashed_args(self, job: Job) -> list[str]:
        return [job.kind, *strip_hash_excluded(self._base_command(job) + self._dry_run_args(job))]

    def compute_hash(self, job: Job) -> str:
        """Config hash; includes the content of any --params file and the stage-1 job's hash."""
        cmd = self._base_command(job)
        extra: dict[str, str] = {}
        for i, a in enumerate(cmd[:-1]):
            if a == "--params":
                p = self.repo_root / cmd[i + 1] if not os.path.isabs(cmd[i + 1]) else Path(cmd[i + 1])
                extra[f"params:{cmd[i + 1]}"] = _sha256(p.read_bytes()) if p.is_file() else "missing"
        if job.kind == "final_eval":
            stage1 = self.jobs[job.of]
            extra["of"] = stage1.config_hash or self.compute_hash(stage1)
        return config_hash(self.hashed_args(job), self.code_fp, self.data_fp, extra)

    def build_command(self, job: Job, gpu: int, chash: str, *, resume: bool = False,
                      token: str | None = None) -> list[str]:
        cmd = [self.python, *self._base_command(job)]
        if job.kind != "final_eval" and "--eval-test" in cmd:
            raise PlanError(f"refusing to pass --eval-test to non-final_eval job {job.name}")
        cmd += ["--device", f"cuda:{gpu}"] if self._uses_device_flag(job) else ["--gpu", str(gpu)]
        cmd += ["--config-hash", chash, "--quiet", *self._dry_run_args(job)]
        if resume:
            cmd.append("--resume")
        if token is not None:
            cmd += ["--test-access-token", token]
        return cmd

    # ── Completion checks ──

    def is_complete(self, job: Job, chash: str) -> bool:
        d = self.run_dir(job.name)
        if job.kind in TUNE_KINDS:
            marker = d / ".config_hash"
            return ((d / "best.yaml").is_file() and marker.is_file()
                    and marker.read_text().strip() == chash)
        metrics = d / "final_metrics.json"
        if not metrics.is_file():
            return False
        try:
            with open(metrics) as fh:
                return json.load(fh).get("config_hash") == chash
        except (OSError, json.JSONDecodeError):
            return False

    def _can_resume(self, job: Job, chash: str) -> bool:
        if job.kind != "hug":
            return False
        d = self.run_dir(job.name)
        if not (d / "last.pt").is_file():
            return False
        launch = d / ".launch_hash"
        if launch.is_file() and launch.read_text().strip() == chash:
            return True
        self.log_event(job.name, "stale_last_pt", "last.pt has a different config hash; restarting")
        return False

    # ── Logging / status ──

    def log_event(self, name: str, event: str, detail: str = "") -> None:
        line = f"{_now_iso(self.clock())}\t{name}\t{event}" + (f"\t{detail}" if detail else "")
        logger.info(line.replace("\t", "  "))
        with open(self.runs_root / "queue.log", "a") as fh:
            fh.write(line + "\n")

    def _expected_minutes(self, job: Job) -> float | None:
        """Observed mean duration of finished jobs of this kind, else the plan default."""
        observed = [(j.end - j.start) / 60 for j in self.jobs.values()
                    if j.kind == job.kind and j.state == "done" and j.start and j.end]
        if observed:
            return sum(observed) / len(observed)
        val = (self.plan.defaults.get("expected_minutes") or {}).get(job.kind)
        return float(val) if isinstance(val, (int, float)) else None

    def eta(self) -> dict[str, Any]:
        """Remaining work in GPU-minutes divided by the number of GPUs (a coarse estimate)."""
        remaining = 0.0
        unknown = 0
        now = self.clock()
        for job in self.jobs.values():
            if job.state not in ("pending", "running"):
                continue
            exp = self._expected_minutes(job)
            if exp is None:
                unknown += 1
                continue
            if job.state == "running" and job.start:
                exp = max(0.0, exp - (now - job.start) / 60)
            remaining += exp
        minutes = remaining / len(self.gpus)
        return {"remaining_gpu_minutes": round(remaining, 1),
                "eta_minutes": round(minutes, 1),
                "eta_at": _now_iso(now + minutes * 60),
                "jobs_without_estimate": unknown}

    def write_status(self) -> None:
        counts = {s: 0 for s in ("pending", "running", "done", "skipped", "failed", "blocked")}
        per_job = {}
        for job in self.jobs.values():
            counts[job.state] += 1
            per_job[job.name] = {
                "kind": job.kind, "state": job.state, "gpu": job.gpu, "attempts": job.attempts,
                "start": _now_iso(job.start), "end": _now_iso(job.end),
                "duration_min": (round((job.end - job.start) / 60, 2)
                                 if job.start and job.end else None),
                "config_hash": job.config_hash, "resumed": job.resumed, "reason": job.reason,
            }
        status = {"updated": _now_iso(self.clock()), "started": _now_iso(self.started_at),
                  "dry_run": self.dry_run, "runs_root": str(self.runs_root), "gpus": self.gpus,
                  "counts": counts, "eta": self.eta(), "jobs": per_job}
        tmp = self.runs_root / "status.json.tmp"
        tmp.write_text(json.dumps(status, indent=2))
        tmp.replace(self.runs_root / "status.json")

    # ── Scheduling ──

    def _deps_state(self, job: Job) -> str:
        """'ready', 'waiting' or 'blocked' given the states of the job's dependencies."""
        states = [self.jobs[d].state for d in self.plan.all_deps(job)]
        if any(s in ("failed", "blocked") for s in states):
            return "blocked"
        if all(s in ("done", "skipped") for s in states):
            return "ready"
        return "waiting"

    def _free_gpu(self, job: Job) -> int | None:
        """Least-loaded GPU that can take this job (two light jobs per GPU with --pack)."""
        for gpu in sorted(self.gpus, key=lambda g: len(self.gpu_jobs[g])):
            running = self.gpu_jobs[gpu]
            if not running:
                return gpu
            if (self.pack and job.memory == "light" and len(running) == 1
                    and self.jobs[running[0]].memory == "light"):
                return gpu
        return None

    def _launch(self, job: Job, gpu: int) -> None:
        d = self.run_dir(job.name)
        d.mkdir(parents=True, exist_ok=True)
        resume = self._can_resume(job, job.config_hash)
        token = None
        if job.kind == "final_eval":
            token = secrets.token_hex(16)
            token_dir = self.runs_root / ".test_tokens"
            token_dir.mkdir(parents=True, exist_ok=True)
            (token_dir / token).write_text(json.dumps(
                {"job": job.name, "config_hash": job.config_hash, "created": _now_iso(self.clock())}))
        cmd = self.build_command(job, gpu, job.config_hash, resume=resume, token=token)
        (d / ".launch_hash").write_text(job.config_hash + "\n")
        meta = job.metadata() | {"config_hash": job.config_hash, "dry_run": self.dry_run,
                                 "command": cmd, "code_compat": self.code_fp,
                                 "git_code_fingerprint": self.git_fp}
        if job.kind == "final_eval":
            stage1 = self.plan.jobs[job.of]
            meta.update(group=job.group or stage1.group, model=job.model or stage1.model,
                        arm=job.arm or stage1.arm, seed=stage1.seed,
                        command=[a if a != token else "<token>" for a in cmd])
        (d / "job.json").write_text(json.dumps(meta, indent=2))

        job.attempts += 1
        job.state = "running"
        job.gpu = gpu
        job.start = self.clock()
        job.end = None
        job.resumed = resume
        self.gpu_jobs[gpu].append(job.name)
        self.procs[job.name] = self.launcher(job, cmd, d, d / "stdout.log", d / "stderr.log",
                                             self.repo_root)
        self.log_event(job.name, "start", f"gpu={gpu} attempt={job.attempts}"
                       + (" resume" if resume else ""))

    def _finish(self, job: Job, rc: int) -> None:
        job.end = self.clock()
        self.gpu_jobs[job.gpu].remove(job.name)
        del self.procs[job.name]
        d = self.run_dir(job.name)
        if rc == 0 and job.kind in TUNE_KINDS and (d / "best.yaml").is_file():
            (d / ".config_hash").write_text(job.config_hash + "\n")
        ok = rc == 0 and self.is_complete(job, job.config_hash)
        if ok:
            if job.kind not in TUNE_KINDS:
                (d / ".config_hash").write_text(job.config_hash + "\n")
            job.state = "done"
            job.reason = None
            self.log_event(job.name, "done", f"{(job.end - job.start) / 60:.1f} min")
            return
        job.reason = (f"exit code {rc}" if rc != 0
                      else "exited 0 but outputs missing or config_hash mismatch")
        if job.attempts < MAX_ATTEMPTS:
            job.state = "pending"
            self.log_event(job.name, "retry", job.reason)
        else:
            job.state = "failed"
            self.log_event(job.name, "failed", job.reason)

    def _reap(self) -> bool:
        changed = False
        for name in list(self.procs):
            rc = self.procs[name].poll()
            if rc is not None:
                self._finish(self.jobs[name], rc)
                changed = True
        return changed

    def _schedule(self) -> bool:
        """Mark blocked/skipped jobs and launch ready ones; True if anything changed."""
        changed = False
        for job in self.jobs.values():
            if job.state != "pending":
                continue
            deps = self._deps_state(job)
            if deps == "blocked":
                job.state = "blocked"
                job.reason = "a dependency failed or is blocked"
                self.log_event(job.name, "blocked")
                changed = True
                continue
            if deps != "ready":
                continue
            if job.config_hash is None:
                job.config_hash = self.compute_hash(job)
                if self.is_complete(job, job.config_hash):
                    job.state = "skipped"
                    self.log_event(job.name, "skip", "outputs match config hash")
                    changed = True
                    continue
            gpu = self._free_gpu(job)
            if gpu is not None:
                self._launch(job, gpu)
                changed = True
        return changed

    def run(self) -> dict[str, str]:
        """Run the queue to completion. Returns {job name: final state}."""
        self.log_event("-", "queue_start",
                       f"jobs={len(self.jobs)} gpus={self.gpus} pack={self.pack} "
                       f"dry_run={self.dry_run}")
        try:
            while True:
                changed = self._reap()
                # Repeat until stable so skips/blocks cascade within one tick.
                while self._schedule():
                    changed = True
                if changed:
                    self.write_status()
                if not self.procs:
                    pending = [j for j in self.jobs.values() if j.state == "pending"]
                    if not pending:
                        break
                    # Nothing running and nothing launchable: should be impossible for a valid plan.
                    for j in pending:
                        j.state = "blocked"
                        j.reason = "unschedulable"
                        self.log_event(j.name, "blocked", "unschedulable")
                    self.write_status()
                    break
                self.sleep(self.poll_seconds)
        except KeyboardInterrupt:
            self.log_event("-", "interrupted", "terminating running jobs")
            for name, proc in list(self.procs.items()):
                if hasattr(proc, "terminate"):
                    proc.terminate()
                job = self.jobs[name]
                job.state, job.end = "pending", self.clock()
                self.gpu_jobs[job.gpu].remove(name)
                del self.procs[name]
            self.write_status()
            raise
        self.write_status()
        counts = {s: sum(j.state == s for j in self.jobs.values()) for s in TERMINAL_STATES}
        self.log_event("-", "queue_end", " ".join(f"{k}={v}" for k, v in counts.items()))
        return {n: j.state for n, j in self.jobs.items()}


# ── CLI ────────────────────────────────────────────────────────────────────────


def dry_run_root(runs_root: Path) -> Path:
    """Dry-run outputs go to a sibling ``<runs_root>_dryrun`` so they never mix with real ones."""
    return runs_root.with_name(runs_root.name + "_dryrun")


def ensure_data_fingerprint(plan: Plan, path: Path, repo_root: Path, python: str) -> str:
    """Return the data fingerprint, running the plan's fingerprint command if the file is missing."""
    if not path.is_file():
        template = plan.defaults.get("fingerprint_cmd")
        if not template:
            raise SystemExit(f"{path} is missing and the plan has no defaults.fingerprint_cmd")
        cmd = shlex.split(template.format(python=python, out=path,
                                          data_dir=plan.defaults.get("data_dir", "")))
        logger.info("computing data fingerprint: %s", " ".join(cmd))
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, cwd=repo_root, check=True)
    return data_fingerprint_from_file(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("plan", type=Path, help="plan YAML, e.g. experiments/heavy_run.yaml")
    p.add_argument("--gpus", default="0,1", help="comma-separated GPU indices (default 0,1)")
    p.add_argument("--pack", action="store_true", help="allow two light jobs per GPU")
    p.add_argument("--dry-run", action="store_true",
                   help="short runs of every job, written under <runs-root>_dryrun/")
    p.add_argument("--max-steps", type=int, default=None,
                   help="optimizer steps per job in a dry run (default 50); requires --dry-run")
    p.add_argument("--only", action="append", default=None, metavar="GLOB",
                   help="run only jobs matching GLOB (plus their dependencies); repeatable")
    p.add_argument("--skip-optional", action="store_true",
                   help="drop jobs marked optional (e.g. DCNv2) and their dependents")
    p.add_argument("--runs-root", type=Path, default=Path("runs/heavy"))
    p.add_argument("--fingerprint-file", type=Path, default=None,
                   help="data fingerprint JSON (default <runs-root>/data_fingerprint.json)")
    p.add_argument("--poll-seconds", type=float, default=10.0)
    p.add_argument("--python", default=sys.executable, help="interpreter for the jobs")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    args = parse_args(argv)
    if args.max_steps is not None and not args.dry_run:
        raise SystemExit("--max-steps is only allowed with --dry-run")
    max_steps = (args.max_steps or 50) if args.dry_run else None

    plan = load_plan(args.plan)
    runs_root = args.runs_root if args.runs_root.is_absolute() else REPO_ROOT / args.runs_root
    if args.dry_run:
        runs_root = dry_run_root(runs_root)
    fp_file = args.fingerprint_file or runs_root / "data_fingerprint.json"
    data_fp = ensure_data_fingerprint(plan, fp_file, REPO_ROOT, args.python)
    code_fp = code_compat_key(REPO_ROOT)          # enters the hash
    git_fp = code_fingerprint(REPO_ROOT)          # recorded only

    queue = JobQueue(plan, runs_root, [int(g) for g in args.gpus.split(",") if g.strip()],
                     code_fp, data_fp, git_fp=git_fp, pack=args.pack, max_steps=max_steps,
                     dry_run=args.dry_run, only=args.only,
                     skip_optional=args.skip_optional, python=args.python,
                     poll_seconds=args.poll_seconds)
    states = queue.run()
    return 1 if any(s in ("failed", "blocked") for s in states.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
