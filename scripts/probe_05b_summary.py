"""
Spec 05b probe summary: per-epoch table for P0–P3 from their final_metrics.json histories.

    python scripts/probe_05b_summary.py runs/probe05b          # expects P0 P1 P2 P3 subdirs

Prints, per run and epoch: val/holdout AUC, train BCE, aux/bce, and for sparse runs the
shared co-activation, row Jaccard of the shared sets, never-fired fraction per dictionary
(validation), in-training dead fraction, and the k_pr range per evidence quintile; then the
decision-rule inputs (epoch-3 val AUC of the better of P1/P2 vs max(P0, P3)).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RUNS = {"P0": "concat", "P1": "sparse", "P2": "sparse + dense skip", "P3": "evgate"}


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/probe05b")
    last = {}
    for run, label in RUNS.items():
        f = root / run / "final_metrics.json"
        if not f.exists():
            print(f"{run} ({label}): missing {f}")
            continue
        m = json.loads(f.read_text())
        print(f"\n## {run}: {label}  (best epoch {m.get('best_epoch')})")
        for h in m["history"]:
            hold = h.get("holdout", {}).get("auc")
            line = (f"epoch {h['epoch']}: val AUC {h['val']['auc']:.4f}  holdout AUC "
                    f"{hold:.4f}  " if hold is not None else
                    f"epoch {h['epoch']}: val AUC {h['val']['auc']:.4f}  holdout –  ")
            line += f"BCE {h['train_bce']:.4f}  aux/bce {h.get('aux_over_bce', 0):.3f}  {h['seconds']:.0f}s"
            print(line)
            fs = h.get("val_fusion_stats")
            if fs:
                print(f"    co-activation {fs['shared_coactivation_fraction']:.3f}  "
                      f"row Jaccard {fs['shared_jaccard_mean']:.3f}  rho {fs['rho_mean']:.3f}")
                print(f"    never fired (val) {json.dumps({k: round(v, 3) for k, v in fs['never_fired_fraction'].items()})}"
                      f"  dead (train window) {h.get('dead_atom_fraction')}")
                for q, per in fs.get("k_pr_by_evidence_quintile", {}).items():
                    print(f"    k_pr by {q}: " + "  ".join(
                        f"{k}:{v['min']}-{v['max']} (mean {v['mean']:.1f})" for k, v in per.items()))
            last[run] = h["val"]["auc"]
    if all(r in last for r in RUNS):
        best_sparse = max(("P1", "P2"), key=lambda r: last[r])
        ref = max(last["P0"], last["P3"])
        gap = ref - last[best_sparse]
        print(f"\nDecision inputs (last epoch): {best_sparse} val AUC {last[best_sparse]:.4f}; "
              f"max(P0, P3) {ref:.4f}; gap {gap:+.4f} (threshold 0.003). "
              "Report to the architect; do not proceed on this alone.")


if __name__ == "__main__":
    main()
