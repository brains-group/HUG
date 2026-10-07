# Architect state: where the WWW'27 work stands

Read this first when resuming. Update it at the end of each session.

## As of 2026-10-06 22:15 EDT

**Deadlines:** abstract 2026-10-18, paper 2026-10-25 (WWW'27, User Modeling track; draft in
`paper/`).

### Sessions and machines
| Who | Where | Role |
|---|---|---|
| Architect (Claude) | brains | specs, review, decisions. Doesn't write method code |
| Coding agent (Claude) | brains, `~/HUG` + `~/HUG-spec06` | implements specs; dev runs on cuda:1 only |
| Idea agent (Claude) | idea cluster, A100 80GB | ZhihuRec site + Spec 05b probe. Prompt: `docs/idea-agent-prompt.md` |

Repo: `git@github.com:brains-group/HUG.git` (formerly GCIC), branch
`docs/technical-report-and-revision-plan`, HEAD `49f3502` + this file.

### Running
- **KuaiRand Stage 1a**: tmux `heavy` on brains, pinned worktree `~/HUG-heavy` (commit
  `3d6142b`), started 19:47. Status: `~/HUG-heavy/runs/heavy/status.json`, `queue.log`,
  `launcher.log`. If the machine reboots: `tmux new -s heavy -c ~/HUG-heavy
  'scripts/launch_stage1a.sh; exec bash'` (finished jobs and trials are skipped).
  - `tune_hug_N1` done: best val AUC **0.7812** (trial 03: emb 32, min_id_count 10, seq_layers 2,
    dropout 0.1).
  - `tune_hug_N4`: 12 trials at ~1.5 h each (30 min/epoch, best epoch is always 1, patience 2);
    ETA about **2026-10-07 15:00**. Trial 00: 0.7735.
  - Baseline tunings (TransAct, WuKong, FiGNN, DCNv2) are running on GPU1 (packed).
- **Spec 05b probe**: idea A100, 4 runs × 3 epochs (P1 sparse, P2 sparse + dense skip,
  P0 concat, P3 evgate), ~6–8 h. Output: `runs/probe05b/summary.txt` on idea. The decision
  rule is in `docs/specs/05b-fusion-fixes.md`.
- **Coding agent**: MIND dry run (Spec 06 item 5), then D1 (does `--eval-history-shuffle` hurt
  tuned N1 on KuaiRand?) and D2 (MIND 1-epoch N0/N1; N4 is done: 0.7262, 23 min/epoch,
  7.9 GB). Results go into `docs/specs/06-datasets-results.md`.
- **Idea agent, after the probe**: ZhihuRec smoke, including 1-epoch N0/N1/N4 with cold/warm
  buckets.

### The open problem (top priority)
On KuaiRand-1K (1-epoch dev arms, `~/HUG/runs/dev/arms`): N0 (IDs + context) 0.7781 = N1
(+ sequence) 0.7780 > N2 (graph only) 0.7735 > N3 0.7723 ≈ N4 (dual view) 0.7720 ≫ N4-frozen
0.7261. The graph costs about 0.006 everywhere, **including cold items**. Every model peaks at
epoch 1. Spec 03's pre-registered rules fire: "N4 ≈ N1 → stop and rethink" and
"N0 ≥ N1 → debug the sequence encoder/history". Spec 05's fusion smoke: sparse 0.742 vs
concat 0.7725 (fixed in 05b; the probe is pending).

**Stage 1b (fusion tuning, ~100 GPU-h) is on hold** until the dual-view premise is checked.

### Decision due ~2026-10-07 afternoon
Inputs:
- tuned N4 vs tuned N1 on KuaiRand;
- D1 (is the history used at all?);
- MIND and ZhihuRec N0/N1/N4 at 1 epoch (does the graph help where cold start is severe:
  MIND 77% unseen test items, ZhihuRec 67% unseen test users?);
- the 05b probe.

Possible outcomes to weigh then:
1. The graph helps on MIND/ZhihuRec but not KuaiRand. Keep the dual view and say where it
   helps; fusion has to guarantee ≥ the best single view.
2. The graph helps nowhere. The dual-view + fusion framing fails; pick a new framing before
   Oct 18 (e.g. the causal protocol / evidence-adaptive fusion as an analysis contribution).
3. D1 shows the history is unused. That's a bug, so fix it before reading anything else.

Then set the Stage 1b budgets (KuaiRand) and decide where MIND runs (it's light: 7.9 GB, so it
can share a GPU).

### Budgets and compute notes
- Each KuaiRand N4-sized job ≈ 30 min/epoch on a shared H100, 38.4 GB peak. Since every model
  peaks at epoch 1, patience 2 spends 2 of every 3 epochs confirming the stop. Consider
  patience 1 for Stage 1b, for all models.
- MIND N4: 23 min/epoch, 7.9 GB. ZhihuRec: unknown (idea).
- Stage 1a dropped emb_dim 128 (OOM) and uses 12 trials per model (`0155b4d`).

### Specs
01–04 done. 05 done, plus 05b (fixes) implemented with the probe pending. 06 implemented; its
GPU smoke is partly done. Results docs: `docs/specs/*-results.md`.
