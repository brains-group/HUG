#!/usr/bin/env bash
# KuaiRand Stage 1a launcher: readiness checks, then the tuning jobs of experiments/heavy_run.yaml.
#
#   scripts/launch_stage1a.sh            # check, wait for GPUs, dry run, launch
#   scripts/launch_stage1a.sh --check    # checks only (no GPU wait, no dry run, no launch)
#
# Run it from a pinned checkout (a git worktree nobody edits), inside tmux:
#
#   git worktree add --detach ~/HUG-heavy <commit>
#   ln -s ~/HUG/KuaiRand-1K ~/HUG-heavy/KuaiRand-1K
#   tmux new -s heavy 'cd ~/HUG-heavy && scripts/launch_stage1a.sh; exec bash'
#
# Stage 1a = tune_hug_N4, tune_hug_N1 and the four baseline tunings (12 trials each, val only).
# Every step logs to runs/heavy/launcher.log; any failed check stops before the queue starts.
# Re-running is safe: finished jobs and trials are skipped by config hash (docs/HEAVY_RUN.md).
#
# Env overrides: PY (interpreter), GPUS (default 0,1), MIN_FREE_MB (free memory each GPU needs
# before launch, default 60000), SKIP_DRYRUN=1, SKIP_TESTS=1.

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PY=${PY:-/home/dasgua3/miniconda3/envs/hkg-env/bin/python}
GPUS=${GPUS:-0,1}
MIN_FREE_MB=${MIN_FREE_MB:-60000}
PLAN=experiments/heavy_run.yaml
RUNS=runs/heavy
# Exact names: since Spec 06 the plan also holds *_mind / *_zhihurec jobs that globs would catch
STAGE1A=(tune_hug_N4 tune_hug_N1 tune_baseline_TransAct tune_baseline_WuKong tune_baseline_FiGNN
         tune_baseline_DCNv2)
ONLY=(); for j in "${STAGE1A[@]}"; do ONLY+=(--only "$j"); done
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

mkdir -p "$RUNS"
LOG=$RUNS/launcher.log
log()  { echo "$(date '+%F %T')  $*" | tee -a "$LOG"; }
fail() { log "NOT READY: $*"; exit 1; }

log "── Stage 1a launcher in $ROOT (commit $(git rev-parse --short HEAD)) ──"

# 1. Pinned, clean code ---------------------------------------------------------
dirty=$(git status --porcelain -- Framework Baselines scripts experiments | grep -v '^??' || true)
[ -z "$dirty" ] || fail "uncommitted changes in code dirs:\n$dirty"
[ "$ROOT" != "/home/dasgua3/HUG" ] || log "WARNING: running from the main tree the coder edits; prefer a pinned worktree"
log "ok  code clean at $(git rev-parse HEAD)"

# 2. Environment ----------------------------------------------------------------
"$PY" - <<'EOF' || fail "environment check failed (see above)"
import sys, torch, torch_geometric, numpy, pandas, sklearn, fuxictr
want = {"torch": "2.5.1", "torch_geometric": "2.6.1", "numpy": "1.26.4", "pandas": "2.3.3",
        "sklearn": "1.7.2", "fuxictr": "2.3.9"}
have = {"torch": torch.__version__.split("+")[0], "torch_geometric": torch_geometric.__version__,
        "numpy": numpy.__version__, "pandas": pandas.__version__, "sklearn": sklearn.__version__,
        "fuxictr": fuxictr.__version__}
bad = {k: (have[k], v) for k, v in want.items() if have[k] != v}
if bad or not torch.cuda.is_available():
    sys.exit(f"version mismatch {bad} / cuda={torch.cuda.is_available()}")
print("env", have, "gpus", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
EOF
log "ok  environment"

# 3. Raw data (checksums recorded on brains, 2026-10-06) ---------------------------
[ -d KuaiRand-1K/data ] || fail "KuaiRand-1K/data missing (symlink or download it)"
sha256sum -c --quiet <<'EOF' || fail "KuaiRand-1K checksum mismatch"
e98841eabb3b078c812c4646bf0c5e3f0c947a4183dd950cb091e32377b9a849  KuaiRand-1K/data/log_random_4_22_to_5_08_1k.csv
355355897a84baa4df26b78b0271bb9a27b127a39cd7aa4a898cf86db0bc1810  KuaiRand-1K/data/log_standard_4_08_to_4_21_1k.csv
548daf771e54e2b73086cc8e7c6f56787d44421f5026c2100421810e47ae9dd4  KuaiRand-1K/data/log_standard_4_22_to_5_08_1k.csv
07813068ac9ca1071dd456c6d84bd20d0dd81a8ff6596a22e1bbe2b12dc5ea6d  KuaiRand-1K/data/user_features_1k.csv
18cb8b9133635a5e53d4e33b1644137b8673b615cc1aaa500b6d3d09695efae3  KuaiRand-1K/data/video_features_basic_1k.csv
5951175389697705dec4a4f992f891f0df3d32584b024f63e96191dc9f0939c5  KuaiRand-1K/data/video_features_statistic_1k.csv
EOF
log "ok  KuaiRand-1K checksums"

# 4. FuxiCTR model zoo (pinned commit) --------------------------------------------
if [ ! -d Baselines/FuxiCTR/model_zoo/DCNv2 ]; then
    log "..  FuxiCTR sources missing: running Baselines/setup_fuxictr.sh"
    bash Baselines/setup_fuxictr.sh >>"$LOG" 2>&1 || fail "setup_fuxictr.sh failed"
fi
for m in TransAct WuKong FiGNN DCNv2; do
    [ -d "Baselines/FuxiCTR/model_zoo/$m/src" ] || fail "FuxiCTR model_zoo/$m missing"
done
log "ok  FuxiCTR model zoo"

# 5. Baseline CSVs (built by this checkout's preprocess.py) --------------------------
CSV=Baselines/data/processed/kuairand_1k_csv
if ! ls "$CSV"/{train,valid,test,holdout}.csv "$CSV"/split_stats.json >/dev/null 2>&1; then
    log "..  baseline CSVs missing: running Baselines/preprocess.py (~7 min)"
    "$PY" Baselines/preprocess.py >>"$LOG" 2>&1 || fail "preprocess.py failed"
fi
"$PY" - "$CSV/split_stats.json" <<'EOF' || fail "baseline CSV cutoffs differ from the canonical split"
import json, sys
s = json.load(open(sys.argv[1]))
assert (s["t_val"], s["t_test"]) == (1651305018060, 1651544133374), (s["t_val"], s["t_test"])
assert s["splits"]["valid"]["rows"] == 1169625, s["splits"]["valid"]["rows"]
EOF
log "ok  baseline CSVs (cutoffs and val rows match)"

# 6. Synthetic tests --------------------------------------------------------------
if [ "${SKIP_TESTS:-0}" != 1 ]; then
    (cd Framework && "$PY" -m pytest tests.py test_fusion.py test_harness.py -x -q -p no:cacheprovider) \
        >"$RUNS/launcher_pytest.log" 2>&1 || fail "synthetic tests failed: $RUNS/launcher_pytest.log"
    log "ok  synthetic tests ($(tail -1 "$RUNS/launcher_pytest.log"))"
fi

# 7. Plan sanity: Stage 1a selection, no test access --------------------------------
"$PY" - "$PLAN" "${STAGE1A[@]}" <<'EOF' || fail "plan check failed"
import sys, fnmatch, yaml
jobs = yaml.safe_load(open(sys.argv[1]))["jobs"]
sel = [j for j in jobs if any(fnmatch.fnmatch(j["name"], g) for g in sys.argv[2:])]
names = sorted(j["name"] for j in sel)
assert names == sorted(["tune_hug_N4", "tune_hug_N1", "tune_baseline_TransAct", "tune_baseline_WuKong",
                        "tune_baseline_FiGNN", "tune_baseline_DCNv2"]), names
assert all("--eval-test" not in str(j.get("args", "")) for j in sel)
assert all(j.get("dataset") in (None, "kuairand") for j in sel), "non-KuaiRand job selected"
assert all("--trials 12" in j["args"] for j in sel), [j["args"] for j in sel]
print("stage 1a jobs:", ", ".join(names))
EOF
log "ok  plan selects exactly the six Stage 1a tuning jobs"

[ "$CHECK_ONLY" = 1 ] && { log "checks passed (--check: stopping here)"; exit 0; }

# 8. Wait for GPUs: dev chains finished and enough free memory on every GPU ----------
gpu_ready() {
    pgrep -u "$(id -u)" -f 'runs/dev/.*/chain\.sh' >/dev/null && return 1
    local g free
    for g in ${GPUS//,/ }; do
        free=$(nvidia-smi -i "$g" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
        [ "$free" -ge "$MIN_FREE_MB" ] || return 1
    done
}
waited=0
until gpu_ready && sleep 60 && gpu_ready; do      # must hold for a minute
    if [ $((waited % 600)) -eq 0 ]; then
        log "..  waiting for GPUs (dev chains running or < ${MIN_FREE_MB} MB free): $(nvidia-smi \
            --query-gpu=index,memory.free --format=csv,noheader | tr '\n' ' ')"
    fi
    sleep 60; waited=$((waited + 60))
done
log "ok  GPUs free: $(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | tr '\n' ' ')"

# 9. Dry run (50 steps, 1 trial per tuning job; writes runs/heavy_dryrun/) ------------
if [ "${SKIP_DRYRUN:-0}" != 1 ]; then
    log "..  dry run of Stage 1a"
    "$PY" scripts/run_queue.py "$PLAN" --gpus "$GPUS" --pack --dry-run --max-steps 50 "${ONLY[@]}" \
        >"$RUNS/launcher_dryrun.log" 2>&1 || fail "dry run failed: $RUNS/launcher_dryrun.log and runs/heavy_dryrun/"
    [ ! -s runs/heavy_dryrun/test_access.log ] || fail "dry run touched the test set: runs/heavy_dryrun/test_access.log"
    log "ok  dry run green"
fi

# 10. Launch -----------------------------------------------------------------------
log ">>  launching Stage 1a: run_queue.py --gpus $GPUS --pack ${ONLY[*]}"
set +e
"$PY" scripts/run_queue.py "$PLAN" --gpus "$GPUS" --pack "${ONLY[@]}" >>"$RUNS/queue.out" 2>&1
rc=$?
set -e
log "<<  queue exited with $rc; status in $RUNS/status.json (failed/blocked jobs make rc=1)"
exit $rc
