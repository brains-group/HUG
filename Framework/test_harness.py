"""
Tests for the heavy-run harness (Spec 04, W5 items 16 and 18): ``scripts/run_queue.py`` and
``scripts/aggregate.py``. CPU only, no real jobs: the queue runs stub jobs through an injected
launcher (plus one end-to-end test through the real subprocess launcher with stub scripts).

    cd Framework && python -m pytest test_harness.py -q
"""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rq = _load_script("run_queue")
agg = _load_script("aggregate")


# ── Stub jobs ──────────────────────────────────────────────────────────────────


def _flag(cmd: list[str], name: str) -> str | None:
    return cmd[cmd.index(name) + 1] if name in cmd else None


class FakeProc:
    """Process-like object that exits after ``polls`` calls to poll()."""

    def __init__(self, rc: int, polls: int = 1):
        self.rc = rc
        self.polls = polls

    def poll(self) -> int | None:
        self.polls -= 1
        return self.rc if self.polls <= 0 else None


class StubLauncher:
    """
    Injected launcher. Records every launch and fakes the job's outputs according to
    ``behaviour[name]``: "ok" (default), "fail" (always exit 1), "fail_once" (exit 1 on the
    first attempt). Checks that a job's inputs exist when it starts.
    """

    def __init__(self, cwd: Path, behaviour: dict[str, str] | None = None, polls: int = 1):
        self.cwd = cwd
        self.behaviour = behaviour or {}
        self.polls = polls
        self.launches: list[tuple[str, list[str]]] = []
        self.running: set[str] = set()
        self.max_per_gpu: dict[str, int] = {}

    def __call__(self, job, cmd, run_dir, stdout_path, stderr_path, cwd):
        assert cwd == self.cwd
        self.launches.append((job.name, cmd))
        # Inputs must already exist: tuned params and (final_eval) the stage-1 checkpoint.
        for flag in ("--params", "--checkpoint"):
            if flag in cmd:
                assert (cwd / _flag(cmd, flag)).is_file(), f"{job.name}: {flag} missing at launch"
        attempts = sum(1 for n, _ in self.launches if n == job.name)
        mode = self.behaviour.get(job.name, "ok")
        if mode == "fail" or (mode == "fail_once" and attempts == 1):
            return FakeProc(1, self.polls)
        out = cwd / (_flag(cmd, "--run-dir") or _flag(cmd, "--out"))
        out.mkdir(parents=True, exist_ok=True)
        chash = _flag(cmd, "--config-hash")
        if job.kind in ("tune_hug", "tune_baseline"):
            (out / "best.yaml").write_text("lr: 0.001\n")
        else:
            (out / "best.pt").write_text("weights")
            (out / "final_metrics.json").write_text(json.dumps(
                {"job": job.name, "kind": job.kind, "config_hash": chash, "seed": job.seed,
                 "val": {"auc": 0.7}}))
            (out / "last.pt").unlink(missing_ok=True)
        return FakeProc(0, self.polls)

    def names(self) -> list[str]:
        return [n for n, _ in self.launches]

    def cmd(self, name: str, attempt: int = -1) -> list[str]:
        return [c for n, c in self.launches if n == name][attempt]


PLAN = {
    "defaults": {
        "expected_minutes": {"tune_hug": 30, "hug": 10, "baseline": 5, "tune_baseline": 20,
                             "final_eval": 1},
        "memory": {"hug": "heavy", "tune_hug": "heavy", "baseline": "light",
                   "tune_baseline": "light"},
        "dry_run_args": {"tune_hug": "--trials 1", "tune_baseline": "--trials 1"},
    },
    "jobs": [
        {"name": "tune_hug_N4", "kind": "tune_hug", "args": "--arm N4 --trials 16 --seed 0"},
        {"name": "hug_N4_s42", "kind": "hug", "args": "{best:tune_hug_N4}", "seed": 42,
         "group": "N4"},
        {"name": "hug_N1_s42", "kind": "hug", "args": "{best:tune_hug_N4} --no-graph",
         "seed": 42, "group": "N1"},
        {"name": "tune_baseline_TransAct", "kind": "tune_baseline",
         "args": "--model TransAct --trials 16 --seed 0"},
        {"name": "baseline_TransAct_s42", "kind": "baseline",
         "args": "--model TransAct {best:tune_baseline_TransAct}", "seed": 42,
         "group": "TransAct"},
        {"name": "report", "kind": "baseline", "args": "--model X", "seed": 1,
         "depends_on": ["hug_N1_s42"]},
        {"name": "final_eval_hug_N4_s42", "kind": "final_eval", "of": "hug_N4_s42",
         "memory": "heavy"},
        {"name": "final_eval_baseline_TransAct_s42", "kind": "final_eval",
         "of": "baseline_TransAct_s42", "memory": "light"},
    ],
}


def make_queue(tmp_path: Path, launcher, plan: dict | None = None, *, code_fp="code-a",
               data_fp="data-a", runs_root: Path | None = None, **kw):
    import copy
    return rq.JobQueue(rq.parse_plan(copy.deepcopy(plan or PLAN)),
                       runs_root or tmp_path / "runs" / "heavy", [0, 1], code_fp, data_fp,
                       repo_root=tmp_path, launcher=launcher, python="python",
                       poll_seconds=0, sleep=lambda s: None, **kw)


# ── 16. Queue ──────────────────────────────────────────────────────────────────


class TestConfigHash:

    def test_stable_across_invocations(self, tmp_path):
        a = make_queue(tmp_path, StubLauncher(tmp_path))
        b = make_queue(tmp_path, StubLauncher(tmp_path))
        for name in a.jobs:
            assert a.compute_hash(a.jobs[name]) == b.compute_hash(b.jobs[name])

    def test_changes_with_args_code_and_data(self, tmp_path):
        base = make_queue(tmp_path, StubLauncher(tmp_path))
        h0 = base.compute_hash(base.jobs["hug_N4_s42"])
        plan = json.loads(json.dumps(PLAN))
        plan["jobs"][1]["args"] += " --lr 0.01"
        changed_args = make_queue(tmp_path, StubLauncher(tmp_path), plan)
        assert changed_args.compute_hash(changed_args.jobs["hug_N4_s42"]) != h0
        changed_code = make_queue(tmp_path, StubLauncher(tmp_path), code_fp="code-b")
        assert changed_code.compute_hash(changed_code.jobs["hug_N4_s42"]) != h0
        changed_data = make_queue(tmp_path, StubLauncher(tmp_path), data_fp="data-b")
        assert changed_data.compute_hash(changed_data.jobs["hug_N4_s42"]) != h0
        # Different tuned params (best.yaml content) change the hash of jobs using them.
        best = tmp_path / "runs" / "heavy" / "tune_hug_N4" / "best.yaml"
        best.parent.mkdir(parents=True)
        best.write_text("lr: 0.1\n")
        h1 = base.compute_hash(base.jobs["hug_N4_s42"])
        best.write_text("lr: 0.2\n")
        assert base.compute_hash(base.jobs["hug_N4_s42"]) not in (h0, h1)

    def test_runtime_flags_excluded(self):
        args = ["main.py", "--run-dir", "x", "--seed", "1", "--gpu", "0", "--device=cuda:1",
                "--quiet", "--resume", "--test-access-token", "tok", "--config-hash", "h"]
        assert rq.strip_hash_excluded(args) == ["main.py", "--seed", "1"]

    def test_code_fingerprint_tracks_commits_diffs_and_untracked(self, tmp_path):
        def git(*a):
            subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a],
                           cwd=tmp_path, check=True, capture_output=True)
        (tmp_path / "Framework").mkdir()
        (tmp_path / "Baselines").mkdir()
        (tmp_path / "Framework" / "m.py").write_text("a = 1\n")
        (tmp_path / "Baselines" / "t.py").write_text("b = 1\n")
        (tmp_path / "other.txt").write_text("x")
        git("init", "-q")
        git("add", ".")
        git("commit", "-qm", "init")
        fp0 = rq.code_fingerprint(tmp_path)
        assert rq.code_fingerprint(tmp_path) == fp0
        (tmp_path / "other.txt").write_text("y")            # outside the code dirs
        assert rq.code_fingerprint(tmp_path) == fp0
        (tmp_path / "Framework" / "m.py").write_text("a = 2\n")
        fp_diff = rq.code_fingerprint(tmp_path)
        assert fp_diff != fp0
        (tmp_path / "Baselines" / "new.py").write_text("c = 1\n")
        fp_untracked = rq.code_fingerprint(tmp_path)
        assert fp_untracked not in (fp0, fp_diff)
        git("add", ".")
        git("commit", "-qm", "two")
        assert rq.code_fingerprint(tmp_path) not in (fp0, fp_diff, fp_untracked)


class TestQueue:

    def test_runs_everything_in_dependency_order(self, tmp_path):
        launcher = StubLauncher(tmp_path)
        q = make_queue(tmp_path, launcher)
        states = q.run()
        assert set(states.values()) == {"done"}
        order = launcher.names()
        assert len(order) == len(PLAN["jobs"])
        for name, job in q.jobs.items():
            for dep in q.plan.all_deps(job):
                assert order.index(dep) < order.index(name), (dep, name)
        status = json.loads((q.runs_root / "status.json").read_text())
        assert status["counts"]["done"] == len(PLAN["jobs"])
        assert status["eta"]["remaining_gpu_minutes"] == 0
        log = (q.runs_root / "queue.log").read_text()
        assert "hug_N4_s42\tdone" in log and "queue_end" in log
        # Device flags: hug gets --device, others --gpu; tuned params resolved.
        hug_cmd = launcher.cmd("hug_N4_s42")
        assert _flag(hug_cmd, "--device").startswith("cuda:") and "--gpu" not in hug_cmd
        assert _flag(hug_cmd, "--params") == "runs/heavy/tune_hug_N4/best.yaml"
        assert hug_cmd[:5] == ["python", "Framework/main.py", "--model-type", "hug", "--run-dir"]
        assert "--gpu" in launcher.cmd("baseline_TransAct_s42")
        assert _flag(launcher.cmd("tune_hug_N4"), "--out") == "runs/heavy/tune_hug_N4"

    def test_finished_jobs_skipped_and_changes_rerun(self, tmp_path):
        make_queue(tmp_path, StubLauncher(tmp_path)).run()
        launcher = StubLauncher(tmp_path)
        q = make_queue(tmp_path, launcher)
        assert set(q.run().values()) == {"skipped"}
        assert launcher.launches == []
        # Changing one job's args reruns it and its final_eval, nothing else.
        plan = json.loads(json.dumps(PLAN))
        plan["jobs"][1]["args"] += " --lr 0.01"
        launcher = StubLauncher(tmp_path)
        states = make_queue(tmp_path, launcher, plan).run()
        assert sorted(launcher.names()) == ["final_eval_hug_N4_s42", "hug_N4_s42"]
        assert states["hug_N1_s42"] == "skipped"
        # A new code fingerprint reruns everything.
        launcher = StubLauncher(tmp_path)
        make_queue(tmp_path, launcher, plan, code_fp="code-b").run()
        assert len(launcher.launches) == len(PLAN["jobs"])

    def test_interrupted_hug_job_resumed_others_restarted(self, tmp_path):
        q = make_queue(tmp_path, StubLauncher(tmp_path))
        root = q.runs_root
        for tune in ("tune_hug_N4", "tune_baseline_TransAct"):     # tuning already finished
            d = root / tune
            d.mkdir(parents=True)
            (d / "best.yaml").write_text("lr: 0.001\n")
            (d / ".config_hash").write_text(q.compute_hash(q.jobs[tune]))
        for name in ("hug_N4_s42", "hug_N1_s42", "baseline_TransAct_s42"):
            d = root / name
            d.mkdir(parents=True)
            (d / "last.pt").write_text("partial")
            (d / ".launch_hash").write_text(q.compute_hash(q.jobs[name]))
        (root / "hug_N1_s42" / ".launch_hash").write_text("stale")   # different config
        launcher = StubLauncher(tmp_path)
        states = make_queue(tmp_path, launcher).run()
        assert states["tune_hug_N4"] == "skipped"
        assert "tune_hug_N4" not in launcher.names()
        assert "--resume" in launcher.cmd("hug_N4_s42")
        assert "--resume" not in launcher.cmd("hug_N1_s42")
        assert "--resume" not in launcher.cmd("baseline_TransAct_s42")
        assert states["hug_N4_s42"] == "done"
        assert not (root / "hug_N4_s42" / "last.pt").exists()

    def test_failed_job_retried_once_then_dependents_blocked(self, tmp_path):
        launcher = StubLauncher(tmp_path, {"tune_hug_N4": "fail",
                                           "baseline_TransAct_s42": "fail_once"})
        q = make_queue(tmp_path, launcher)
        states = q.run()
        assert launcher.names().count("tune_hug_N4") == 2
        assert states["tune_hug_N4"] == "failed"
        for name in ("hug_N4_s42", "hug_N1_s42", "report", "final_eval_hug_N4_s42"):
            assert states[name] == "blocked"
            assert name not in launcher.names()
        # The unrelated chain still runs; a single failure is recovered by the retry.
        assert launcher.names().count("baseline_TransAct_s42") == 2
        assert states["baseline_TransAct_s42"] == "done"
        assert states["final_eval_baseline_TransAct_s42"] == "done"
        status = json.loads((q.runs_root / "status.json").read_text())
        assert status["counts"]["failed"] == 1 and status["counts"]["blocked"] == 4
        assert status["jobs"]["tune_hug_N4"]["attempts"] == 2
        log = (q.runs_root / "queue.log").read_text()
        assert "tune_hug_N4\tretry" in log and "tune_hug_N4\tfailed" in log

    def test_exit_zero_without_outputs_is_a_failure(self, tmp_path):
        def launcher(job, cmd, run_dir, out, err, cwd):
            return FakeProc(0)
        plan = {"jobs": [{"name": "a", "kind": "hug", "seed": 1, "args": ""}]}
        q = make_queue(tmp_path, launcher, plan)
        assert q.run() == {"a": "failed"}

    def test_dry_run_redirects_and_shortens(self, tmp_path):
        assert rq.dry_run_root(Path("runs/heavy")) == Path("runs/heavy_dryrun")
        launcher = StubLauncher(tmp_path)
        root = rq.dry_run_root(tmp_path / "runs" / "heavy")
        q = make_queue(tmp_path, launcher, runs_root=root, dry_run=True, max_steps=50)
        assert set(q.run().values()) == {"done"}
        assert not (tmp_path / "runs" / "heavy").exists()
        assert (root / "hug_N4_s42" / "final_metrics.json").is_file()
        for name, cmd in launcher.launches:
            assert _flag(cmd, "--max-steps") == "50"
            assert "heavy_dryrun" in (_flag(cmd, "--run-dir") or _flag(cmd, "--out"))
        # The plan's "--trials 16" is followed by the dry-run "--trials 1" (last one wins).
        tune_cmd = launcher.cmd("tune_hug_N4")
        assert [tune_cmd[i + 1] for i, a in enumerate(tune_cmd) if a == "--trials"] == ["16", "1"]

    def test_pack_allows_two_light_jobs_per_gpu(self, tmp_path):
        plan = {"jobs": [{"name": f"l{i}", "kind": "baseline", "seed": i, "args": "--model X",
                          "memory": "light"} for i in range(4)]
                + [{"name": "h", "kind": "hug", "seed": 0, "args": "", "memory": "heavy"}]}

        def peak_per_gpu(pack: bool) -> int:
            q = make_queue(tmp_path / str(pack), StubLauncher(tmp_path / str(pack), polls=3),
                           plan, pack=pack)
            q.gpus, q.gpu_jobs = [0], {0: []}
            peak = 0
            original = q._launch

            def spy(job, gpu):
                nonlocal peak
                original(job, gpu)
                kinds = [q.jobs[n].memory for n in q.gpu_jobs[gpu]]
                assert kinds == ["heavy"] or all(k == "light" for k in kinds)
                peak = max(peak, len(kinds))
            q._launch = spy
            assert set(q.run().values()) == {"done"}
            return peak

        (tmp_path / "True").mkdir()
        (tmp_path / "False").mkdir()
        assert peak_per_gpu(False) == 1
        assert peak_per_gpu(True) == 2

    def test_only_selects_jobs_and_their_dependencies(self, tmp_path):
        launcher = StubLauncher(tmp_path)
        q = make_queue(tmp_path, launcher, only=["final_eval_hug_*"])
        q.run()
        assert sorted(launcher.names()) == ["final_eval_hug_N4_s42", "hug_N4_s42", "tune_hug_N4"]

    def test_skip_optional_drops_optional_jobs_and_dependents(self):
        plan = rq.parse_plan({"jobs": [
            {"name": "t", "kind": "tune_baseline", "args": "--model DCNv2", "optional": True},
            {"name": "b", "kind": "baseline", "seed": 1, "args": "--model DCNv2 {best:t}"},
            {"name": "c", "kind": "baseline", "seed": 1, "args": "--model X"}]})
        assert rq.select_jobs(plan, None, skip_optional=True) == ["c"]
        assert rq.select_jobs(plan, None) == ["t", "b", "c"]


class TestTestSetDiscipline:

    @pytest.mark.parametrize("bad", ["--eval-test", "--test-access-token abc",
                                     "--i-know-this-touches-test"])
    @pytest.mark.parametrize("kind", ["hug", "baseline", "tune_hug", "tune_baseline"])
    def test_plan_with_test_flag_refused(self, kind, bad):
        with pytest.raises(rq.PlanError):
            rq.parse_plan({"jobs": [{"name": "x", "kind": kind, "seed": 1, "args": bad}]})

    def test_final_eval_args_cannot_carry_test_flags_either(self):
        with pytest.raises(rq.PlanError):
            rq.parse_plan({"jobs": [
                {"name": "a", "kind": "hug", "seed": 1, "args": ""},
                {"name": "f", "kind": "final_eval", "of": "a", "args": "--eval-test"}]})

    def test_final_eval_must_point_at_a_stage1_job(self):
        with pytest.raises(rq.PlanError):
            rq.parse_plan({"jobs": [{"name": "t", "kind": "tune_hug", "args": ""},
                                    {"name": "f", "kind": "final_eval", "of": "t"}]})

    def test_only_final_eval_gets_eval_test_and_a_token(self, tmp_path):
        launcher = StubLauncher(tmp_path, {"final_eval_hug_N4_s42": "fail_once"})
        q = make_queue(tmp_path, launcher)
        q.run()
        tokens = []
        for name, cmd in launcher.launches:
            if q.jobs[name].kind != "final_eval":
                assert "--eval-test" not in cmd and "--test-access-token" not in cmd
                continue
            assert "--eval-test" in cmd and "--eval-only" in cmd
            token = _flag(cmd, "--test-access-token")
            record = json.loads((q.runs_root / ".test_tokens" / token).read_text())
            assert record["job"] == name and record["config_hash"] == q.jobs[name].config_hash
            tokens.append(token)
        assert len(tokens) == 3 and len(set(tokens)) == 3      # one-time: the retry gets a new one
        hug_eval = launcher.cmd("final_eval_hug_N4_s42")
        assert _flag(hug_eval, "--checkpoint") == "runs/heavy/hug_N4_s42/best.pt"
        assert _flag(hug_eval, "--seed") == "42" and "--device" in hug_eval
        assert _flag(hug_eval, "--params") == "runs/heavy/tune_hug_N4/best.yaml"
        base_eval = launcher.cmd("final_eval_baseline_TransAct_s42")
        assert base_eval[1] == "Baselines/train.py" and "--gpu" in base_eval
        meta = json.loads((q.runs_root / "final_eval_hug_N4_s42" / "job.json").read_text())
        assert meta["group"] == "N4" and "<token>" in meta["command"]

    def test_heavy_run_plan_is_valid(self):
        plan = rq.load_plan(REPO_ROOT / "experiments" / "heavy_run.yaml")
        stage1 = [j for j in plan.jobs.values() if j.kind in ("hug", "baseline")]
        evals = {j.of for j in plan.jobs.values() if j.kind == "final_eval"}
        assert evals == {j.name for j in stage1}
        groups = {j.group for j in stage1}
        assert {"N0", "N1", "N2", "N3", "N4", "N4-frozen", "N1-own"} <= groups


def test_popen_launcher_end_to_end(tmp_path):
    """The real subprocess launcher with a stub Framework/main.py: logs and outputs land."""
    (tmp_path / "Framework").mkdir()
    (tmp_path / "Framework" / "main.py").write_text(
        "import json, pathlib, sys\n"
        "a = sys.argv[1:]\n"
        "d = pathlib.Path(a[a.index('--run-dir') + 1]); d.mkdir(parents=True, exist_ok=True)\n"
        "print('training', a[a.index('--seed') + 1]); print('warn', file=sys.stderr)\n"
        "json.dump({'config_hash': a[a.index('--config-hash') + 1]},\n"
        "          open(d / 'final_metrics.json', 'w'))\n")
    plan = rq.parse_plan({"jobs": [{"name": "a", "kind": "hug", "seed": 7, "args": ""},
                                   {"name": "b", "kind": "hug", "seed": 8, "args": "",
                                    "depends_on": ["a"]}]})
    q = rq.JobQueue(plan, tmp_path / "runs" / "heavy", [0], "c", "d", repo_root=tmp_path,
                    python=sys.executable, poll_seconds=0.02)
    assert q.run() == {"a": "done", "b": "done"}
    assert (q.runs_root / "a" / "stdout.log").read_text().strip() == "training 7"
    assert (q.runs_root / "b" / "stderr.log").read_text().strip() == "warn"


# ── 18. Aggregation ────────────────────────────────────────────────────────────


def _metrics(auc: float, logloss: float, test: dict | None = None) -> dict:
    return {"val": {"auc": auc, "ap": auc / 2, "logloss": logloss, "ndcg10": 0.5,
                    "buckets": {"freq": {"low": {"auc": auc - 0.1, "n": 10},
                                         "high": {"auc": auc + 0.1, "n": 30}}}},
            "holdout": {"auc": auc + 0.05, "logloss": logloss - 0.1},
            "test": test}


def _write_run(root: Path, job: str, kind: str, group: str, seed: int, metrics: dict,
               preds: dict | None = None, model: str = "HUG", arm: str | None = None) -> None:
    d = root / job
    d.mkdir(parents=True)
    (d / "final_metrics.json").write_text(json.dumps(
        {"job": job, "kind": kind, "model": model, "arm": arm, "seed": seed,
         "config_hash": "h", **metrics}))
    (d / "job.json").write_text(json.dumps({"kind": kind, "group": group, "model": model,
                                            "arm": arm, "seed": seed}))
    if preds is not None:
        np.savez(d / "val_preds.npz", **preds)


def _preds(labels, users, scores, row_id=None) -> dict:
    n = len(labels)
    return {"row_id": np.arange(n, dtype=np.int64) if row_id is None else row_id,
            "user": users.astype(np.int64), "label": labels.astype(np.float32),
            "score": scores.astype(np.float32)}


@pytest.fixture
def synthetic_runs(tmp_path):
    """N4 and N1 (3 seeds), TransAct (2 seeds), final_eval runs of N4; planted signal in N4."""
    rng = np.random.default_rng(1)
    n_users, per_user = 60, 30
    users = np.repeat(np.arange(n_users), per_user) + 1000
    labels = (rng.random(len(users)) < 0.3).astype(np.float32)
    root = tmp_path / "heavy"
    aucs = {"N4": [0.80, 0.82, 0.81], "N1": [0.75, 0.74, 0.79], "TransAct": [0.70, 0.72]}
    for group, vals in aucs.items():
        for i, auc in enumerate(vals):
            seed = 42 + i
            if group == "N4":     # informative scores
                scores = labels + rng.normal(0, 0.8, len(labels))
            else:                  # noise
                scores = rng.normal(0, 1, len(labels))
            perm = rng.permutation(len(labels))   # files need not be row-aligned
            p = _preds(labels[perm], users[perm], scores[perm], np.arange(len(labels))[perm])
            _write_run(root, f"{group}_s{seed}", "hug" if group != "TransAct" else "baseline",
                       group, seed, _metrics(auc, 0.5 + i / 100, test={"auc": 0.123456}), p,
                       model="HUG" if group != "TransAct" else "TransAct",
                       arm=group if group != "TransAct" else None)
    for i, t in enumerate([0.61, 0.63, 0.65]):
        _write_run(root, f"final_eval_N4_s{42 + i}", "final_eval", "N4", 42 + i,
                   {"val": {"auc": 0.999}, "test": {"auc": t, "ap": 0.3, "logloss": 0.4,
                                                    "ndcg10": 0.6}})
    # A tuning dir and a trial inside it must be ignored.
    _write_run(root, "tune_hug_N4", "tune_hug", "tune", 0, _metrics(0.99, 0.1))
    return root, aucs


class TestAggregation:

    def test_weighted_auc_matches_sklearn(self):
        rng = np.random.default_rng(0)
        labels = (rng.random(500) < 0.4).astype(float)
        scores = np.round(rng.normal(size=500), 1)          # many ties
        weights = rng.integers(0, 4, size=500)
        f = agg.WeightedAUC(labels, scores)
        assert f(np.ones(500)) == pytest.approx(roc_auc_score(labels, scores), abs=1e-12)
        assert f(weights) == pytest.approx(
            roc_auc_score(labels, scores, sample_weight=weights), abs=1e-12)

    def test_val_table_means_and_stds(self, synthetic_runs):
        root, aucs = synthetic_runs
        out = agg.aggregate(root, n_boot=50)
        with open(out / "val_table.csv") as fh:
            rows = {r["group"]: r for r in csv.DictReader(fh)}
        assert set(rows) == {"N1", "N4", "TransAct"}     # no final_eval, no tuning rows
        for g, vals in aucs.items():
            assert float(rows[g]["val_auc_mean"]) == pytest.approx(np.mean(vals), abs=1e-6)
            assert float(rows[g]["val_auc_std"]) == pytest.approx(np.std(vals, ddof=1), abs=1e-6)
            assert float(rows[g]["holdout_auc_mean"]) == pytest.approx(np.mean(vals) + 0.05,
                                                                       abs=1e-6)
            assert int(rows[g]["n_seeds"]) == len(vals)
        assert float(rows["N4"]["val_logloss_mean"]) == pytest.approx(0.51, abs=1e-6)
        md = (out / "val_table.md").read_text()
        assert "0.8100 ± 0.0100" in md
        assert "0.999" not in md and "0.1234" not in md
        buckets = (out / "bucket_tables.md").read_text()
        assert "## freq" in buckets and "0.9100" in buckets      # N4 high bucket: 0.81 + 0.1

    def test_test_table_only_from_final_eval(self, synthetic_runs):
        root, _ = synthetic_runs
        out = agg.aggregate(root, n_boot=50)
        text = (out / "test_table.md").read_text()
        assert "0.6300 ± 0.0200" in text
        assert "0.1234" not in text                              # stage-1 "test" fields ignored
        assert "N1" not in text and "TransAct" not in text

    def test_bootstrap_zero_for_identical_predictions(self, synthetic_runs):
        root, _ = synthetic_runs
        runs = agg.by_group(agg.load_runs(root))
        pp = agg.pair_predictions(runs["N1"], runs["N1"])
        res = agg.paired_bootstrap(pp, n_boot=200)
        assert pp.mode == "seeds 42,43,44"
        assert res["observed"] == 0 and res["mean"] == 0
        assert res["ci_lo"] == 0 and res["ci_hi"] == 0 and res["frac_le_0"] == 1.0

    def test_bootstrap_detects_planted_difference(self, synthetic_runs):
        root, _ = synthetic_runs
        runs = agg.by_group(agg.load_runs(root))
        res = agg.compare(runs["N1"], runs["N4"], n_boot=200)   # noise vs informative
        b = res["boot"]
        assert b["observed"] < -0.2 and b["ci_hi"] < 0 and b["frac_le_0"] == 1.0
        assert b["n_users"] == 60 and b["n_rows"] == 1800
        assert res["ttest"]["n"] == 3 and res["ttest"]["t"] < 0
        # Without common seeds the groups are compared on seed-averaged scores.
        assert agg.pair_predictions(runs["TransAct"][:1],
                                    [r for r in runs["N4"] if r.seed == 44]).mode == "seed-averaged"

    def test_bootstrap_is_deterministic_and_matches_full_auc(self, synthetic_runs):
        root, _ = synthetic_runs
        runs = agg.by_group(agg.load_runs(root))
        pp = agg.pair_predictions(runs["N4"], runs["N1"])
        r1 = agg.paired_bootstrap(pp, n_boot=100)
        r2 = agg.paired_bootstrap(pp, n_boot=100)
        assert r1 == r2
        expected = np.mean([roc_auc_score(pp.labels, a) - roc_auc_score(pp.labels, b)
                            for a, b in pp.pairs])
        assert r1["observed"] == pytest.approx(expected, abs=1e-9)

    def test_paired_ttest_matches_formula(self):
        x, y = [0.80, 0.82, 0.81], [0.75, 0.74, 0.79]
        d = np.subtract(x, y)
        res = agg.paired_ttest(x, y)
        assert res["t"] == pytest.approx(d.mean() / (d.std(ddof=1) / np.sqrt(3)))
        assert 0 < res["p"] < 1

    def test_significance_file(self, synthetic_runs):
        root, _ = synthetic_runs
        out = agg.aggregate(root, n_boot=50)
        text = (out / "significance.md").read_text()
        assert "Each group vs. N4" in text and "N4 vs. ablations" in text
        assert "| N1 | N4 |" in text and "| TransAct | N4 |" in text and "| N4 | N1 |" in text

    def test_robust_to_missing_fields(self, tmp_path):
        root = tmp_path / "heavy"
        _write_run(root, "a", "hug", "N4", 42, {"val": {"auc": 0.7}})   # no preds, no holdout
        (root / "broken").mkdir()
        (root / "broken" / "final_metrics.json").write_text("{not json")
        out = agg.aggregate(root, n_boot=10)
        assert "0.7000" in (out / "val_table.md").read_text()
        assert "No final_eval results" in (out / "test_table.md").read_text()



class TestCodeCompatHash:
    """Spec 05 §0: code changes don't invalidate jobs; CODE_COMPAT_VERSION does."""

    def _repo(self, tmp_path, version: int, extra: str = "") -> Path:
        (tmp_path / "Framework").mkdir(parents=True, exist_ok=True)
        (tmp_path / "Framework" / "runtime.py").write_text(
            f"import os\n\nCODE_COMPAT_VERSION = {version}\n{extra}")
        (tmp_path / "Framework" / "hug.py").write_text("x = 1\n" + extra)
        return tmp_path

    def test_hash_ignores_code_change_and_tracks_compat_version(self, tmp_path):
        args = ["--model-type", "hug", "--seed", "42"]
        h = lambda root: rq.config_hash(args, rq.code_compat_key(root), "data-a")
        base = h(self._repo(tmp_path, 1))
        assert h(self._repo(tmp_path, 1, extra="y = 2  # new fusion head\n")) == base
        assert h(self._repo(tmp_path, 2)) != base

    def test_finished_job_survives_code_change(self, tmp_path):
        repo = self._repo(tmp_path / "repo", 1)
        launcher = StubLauncher(tmp_path)
        make_queue(tmp_path, launcher, code_fp=rq.code_compat_key(repo)).run()
        n_first = len(launcher.launches)
        self._repo(tmp_path / "repo", 1, extra="z = 3\n")             # code edit, same version
        make_queue(tmp_path, launcher, code_fp=rq.code_compat_key(repo)).run()
        assert len(launcher.launches) == n_first                        # nothing reran
        self._repo(tmp_path / "repo", 2)                                # bump
        make_queue(tmp_path, launcher, code_fp=rq.code_compat_key(repo)).run()
        assert len(launcher.launches) > n_first
