# Heavy run — runbook

**Plan:** `experiments/heavy_run.yaml` (spec 04 "Heavy-run plan"). **Queue:** `scripts/run_queue.py`.
**Aggregation:** `scripts/aggregate.py`. All commands run from the repo root with the `hkg-env`
interpreter (`/home/dasgua3/miniconda3/envs/hkg-env/bin/python`, written `python` below).

## Before launching

1. Commit everything. Config hashes include the git tree of `Framework/` and `Baselines/` *and*
   any uncommitted diff, so editing code mid-run invalidates every job that hasn't started yet
   (they rerun under the new hash).
2. Baseline CSVs exist: `python Baselines/preprocess.py` (≈7 min). FuxiCTR sources:
   `bash Baselines/setup_fuxictr.sh`.
3. Check free GPU memory: this machine is shared. A HUG graph job peaks at roughly
   ⟨W6 value⟩ GB; `--pack` only co-locates `light` jobs.

## Launch

Stage 1 (tuning, HUG arms, baselines — validation only):

```bash
nohup python scripts/run_queue.py experiments/heavy_run.yaml --gpus 0,1 \
    --only 'tune_*' --only 'hug_*' --only 'baseline_*' > runs/heavy/queue.out 2>&1 &
```

Add `--skip-optional` to drop DCNv2 (and its final_eval). Stage 2 (test, once per reported
configuration), after Stage 1 is reviewed:

```bash
nohup python scripts/run_queue.py experiments/heavy_run.yaml --gpus 0,1 > runs/heavy/queue.out 2>&1 &
```

The second command skips every finished Stage 1 job (matching config hash) and runs the
`final_eval` jobs, the only jobs the queue ever gives a test-access token.

## Monitor

- `runs/heavy/status.json` — per-job state (pending / running / done / skipped / failed /
  blocked), start and end times, ETA.
- `runs/heavy/queue.log` — one line per event.
- `runs/heavy/<job>/stdout.log`, `stderr.log`; HUG jobs also write `hug.log`.
- `runs/heavy/test_access.log` — every test-set access (job, config hash, time). It must contain
  only `final_eval` jobs.

## Resume after a crash or reboot

Run the same command again. Finished jobs are skipped by config hash, interrupted HUG jobs
continue from their `last.pt` (`--resume`, exact to the epoch boundary), and baseline/tuning
jobs restart (finished tuning trials are skipped). A job that fails twice is marked `failed`
and its dependents `blocked`; fix the cause and rerun the command.

## Outputs

- `runs/heavy/<job>/final_metrics.json`, `val_preds.npz` (+ `test_preds.npz` for `final_eval`),
  `best.pt` (git-ignored; deleted `last.pt` on completion).
- `runs/heavy/tune_*/best.yaml`, `trials.json`, `trial_<i>/`.

## Aggregate

```bash
python scripts/aggregate.py --runs-root runs/heavy
```

Writes `runs/heavy/summary/`: `val_table.md/.csv` (mean ± std over seeds, holdout diagnostic
column), `bucket_tables.md`, `significance.md` (per-user paired bootstrap vs N4 and N4 vs
N1/N2/N4-frozen; per-seed t-tests), and `test_table.md` — built **only** from `final_eval` jobs.
`significance.md` takes roughly 30–40 minutes on the full plan.

## Expected duration and disk

See `docs/specs/04-heavy-run-readiness-results.md` §"Run-time estimate".

## Do not

- Pass `--eval-test` by hand. The guard refuses it without a queue token; the manual override
  `--i-know-this-touches-test` is logged in `test_access.log` and must not be used for reported
  numbers.
- Edit `Framework/` or `Baselines/` while the queue runs (config hashes change; see above).
- Delete `last.pt` files of running or interrupted jobs.
- Report the holdout column as a headline metric: it is a diagnostic, and it is confounded for
  history-consuming models (see the spec 04 results, W6.1).
