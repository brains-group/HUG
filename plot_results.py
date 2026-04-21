"""
plot_results.py
===============
Read history.json / .log files from ./runs/ and produce publication-ready figures.

Main figures (AUC, AP, NDCG@10 — final test epoch):
  figures/fig1_baselines_progression.pdf
  figures/fig2_ablations.pdf
  figures/fig3_sensitivity.pdf
  figures/fig4_ndcg_over_epochs.pdf

Appendix figures (full convergence curves):
  figures/app1_convergence_baselines.pdf   — AUC, AP, NDCG, BCE over epochs
  figures/app2_convergence_ablations.pdf   — AUC, AP, NDCG, BCE over epochs
  figures/app3_convergence_sensitivity.pdf — AUC, AP, NDCG, BCE over epochs

Usage
-----
    python plot_results.py [--runs-dir ./runs] [--out-dir ./figures]
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Publication styling ────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.size":        22,
    "axes.titlesize":   24,
    "axes.labelsize":   22,
    "xtick.labelsize":  20,
    "ytick.labelsize":  20,
    "legend.fontsize":  18,
    "font.family":      "serif",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
})

# ── Metrics shown in bar charts ───────────────────────────────────────────────

METRICS = [
    ("auc",     "AUC",      True),
    ("ap",      "AP",       True),
    ("ndcg10",  "NDCG@10",  True),
    ("logloss", "BCE Loss", False),
]

# All metrics used in convergence/appendix plots
ALL_METRICS = [
    ("auc",     "AUC",      True),
    ("ap",      "AP",       True),
    ("ndcg10",  "NDCG@10",  True),
    ("logloss", "BCE Loss", False),
]

# ── Run groups ────────────────────────────────────────────────────────────────

GROUPS: dict[str, list[tuple[str, str]]] = {
    "baselines_progression": [
        ("DIN",                               "DIN"),
        ("BST",                               "BST"),
        ("1k_kgat_ips0_L2_h128d64",          "KGAT\n(no IPS)"),
        ("1k_kgat_ips1_L2_h128d64",          "KGAT\n(IPS)"),
        ("1k_single_kg0_ips1_rg1",           "HUG-Unified"),
        ("1k_dual_kg0_ips1_rg1",             "HUG-Dual\n(KGA off)"),
        ("1k_dual_kg64_ips1_rg1",            "HUG-Dual\n(full model)"),
    ],
    "ablations": [
        ("1k_dual_kg64_ips1_rg1",            "Full model"),
        ("1k_dual_kg64_ips0_rg1",            "No IPS"),
        ("1k_dual_kg64_ips1_rg0",            "No recency\ngate"),
        ("1k_dual_kg64_ips0_rg0",            "No IPS +\nno RG"),
    ],
    "sensitivity": [
        ("1k_dual_kg64_ips1_rg1",            "Default\n(L=2, h=128)"),
        ("1k_dual_kg64_ips1_rg1_L1",         "L=1"),
        ("1k_dual_kg64_ips1_rg1_L3",         "L=3"),
        ("1k_dual_kg32_ips1_rg1_h64d32",     "h=64, d=32"),
        ("1k_dual_kg128_ips1_rg1_h256d128",  "h=256, d=128"),
    ],
}

GROUP_TITLES = {
    "baselines_progression": "Baselines & Model Progression",
    "ablations":             "Ablation Study",
    "sensitivity":           "Hyperparameter Sensitivity",
}

OUTPUT_FILES = {
    "baselines_progression": "fig1_baselines_progression.pdf",
    "ablations":             "fig2_ablations.pdf",
    "sensitivity":           "fig3_sensitivity.pdf",
}

# Grey for external/KGAT baselines; blue→green progression for HUG variants
RUN_COLOURS: dict[str, str] = {
    "DIN":                               "#8C8C8C",
    "BST":                               "#8C8C8C",
    "1k_kgat_ips0_L2_h128d64":          "#AAAAAA",
    "1k_kgat_ips1_L2_h128d64":          "#4D4D4D",
    "1k_single_kg0_ips0_rg1":           "#4C72B0",
    "1k_single_kg0_ips1_rg1":           "#4C72B0",
    "1k_dual_kg0_ips1_rg1":             "#55A868",
    "1k_dual_kg64_ips1_rg1":            "#2CA02C",
    "1k_dual_kg64_ips0_rg1":            "#4C72B0",
    "1k_dual_kg64_ips1_rg0":            "#55A868",
    "1k_dual_kg64_ips0_rg0":            "#DD8452",
    "1k_dual_kg64_ips1_rg1_L1":         "#4C72B0",
    "1k_dual_kg64_ips1_rg1_L3":         "#DD8452",
    "1k_dual_kg32_ips1_rg1_h64d32":     "#C44E52",
    "1k_dual_kg128_ips1_rg1_h256d128":  "#8172B2",
}
COLOUR_DEFAULT = "#4C72B0"
COLOUR_MISSING = "#CCCCCC"

# ── Data loading ──────────────────────────────────────────────────────────────

_LOG_METRICS_RE = re.compile(
    r"logloss:\s*([\d.]+).*?AUC:\s*([\d.]+).*?AP:\s*([\d.]+).*?NDCG\(10\):\s*([\d.]+)"
)

def _parse_log(log_path: Path) -> dict | None:
    """Extract test metrics from a FuxiCTR .log file (best checkpoint)."""
    text = log_path.read_text(errors="replace")
    marker = "******** Test evaluation ********"
    idx = text.rfind(marker)
    if idx == -1:
        return None
    m = _LOG_METRICS_RE.search(text, idx)
    if not m:
        return None
    return {
        "logloss": float(m.group(1)),
        "auc":     float(m.group(2)),
        "ap":      float(m.group(3)),
        "ndcg10":  float(m.group(4)),
    }


def load_results(runs_dir: Path) -> dict[str, dict]:
    """Return {run_name: metrics} using the final (last) test epoch."""
    results = {}

    for path in sorted(runs_dir.glob("*/history.json")):
        run_name = path.parent.name
        try:
            history = json.loads(path.read_text())
            test = [ep for ep in history if "test" in ep][-1]["test"]
            results[run_name] = {
                "auc":     test.get("auc"),
                "ap":      test.get("ap"),
                "logloss": test.get("logloss"),
                "ndcg10":  test.get("ndcg10"),
            }
        except Exception as e:
            print(f"  WARNING: could not parse {path}: {e}")

    for log_path in sorted(runs_dir.glob("*/*.log")):
        run_name = log_path.parent.name
        if run_name in results:
            continue
        try:
            metrics = _parse_log(log_path)
            if metrics:
                results[run_name] = metrics
        except Exception as e:
            print(f"  WARNING: could not parse {log_path}: {e}")

    return results


def load_histories(runs_dir: Path) -> dict[str, list[dict]]:
    """Return {run_name: [epoch_record, ...]} for history.json runs."""
    histories = {}
    for path in sorted(runs_dir.glob("*/history.json")):
        run_name = path.parent.name
        try:
            histories[run_name] = json.loads(path.read_text())
        except Exception as e:
            print(f"  WARNING: could not parse {path}: {e}")
    return histories


# ── Helpers ───────────────────────────────────────────────────────────────────

def zoom_ylim(ax, values, missing):
    valid = [v for v, m in zip(values, missing) if not m]
    if not valid:
        return
    lo, hi = min(valid), max(valid)
    pad = max((hi - lo) * 0.4, 1e-4)
    ax.set_ylim(max(0.0, lo - pad), hi + pad * 2.5)


def save_fig(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ── Single-metric bar chart ───────────────────────────────────────────────────

def plot_single_metric(
    metric_key: str,
    metric_label: str,
    higher_better: bool,
    runs: list[tuple[str, str]],
    results: dict[str, dict],
    title: str,
    out_path: Path,
) -> None:
    n_bars   = len(runs)
    x        = np.arange(n_bars)
    labels   = [lbl for _, lbl in runs]
    flat_labels = [lbl.replace("\n", "") for lbl in labels]
    rotation = 30 if any(len(l) > 5 for l in flat_labels) else 0

    fig, ax = plt.subplots(figsize=(max(9, n_bars * 1.4), 6), constrained_layout=True)

    values, colours, missing = [], [], []
    for run_name, _ in runs:
        val = results.get(run_name, {}).get(metric_key)
        if val is None:
            values.append(0.0); colours.append(COLOUR_MISSING); missing.append(True)
        else:
            values.append(float(val))
            colours.append(RUN_COLOURS.get(run_name, COLOUR_DEFAULT))
            missing.append(False)

    bars = ax.bar(x, values, color=colours, width=0.6, edgecolor="white", zorder=3)

    for bar, val, miss in zip(bars, values, missing):
        if miss:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                    "n/a", ha="center", va="center", fontsize=16, color="#666666")
        else:
            ax.annotate(f"{val:.4f}",
                        xy=(bar.get_x() + bar.get_width() / 2, val),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", va="bottom", fontsize=16, fontweight="bold")

    direction = "↑ higher better" if higher_better else "↓ lower better"
    ax.set_ylabel(f"{metric_label}  ({direction})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=18,
                       ha="right" if rotation else "center", rotation=rotation)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    zoom_ylim(ax, values, missing)

    save_fig(fig, out_path)


def plot_group(
    group_key: str,
    runs: list[tuple[str, str]],
    results: dict[str, dict],
    out_dir: Path,
    prefix: str,
) -> None:
    """Save one figure per metric for the given group."""
    suffixes = {
        "auc":     "auc",
        "ap":      "ap",
        "ndcg10":  "ndcg",
        "logloss": "bce",
    }
    for metric_key, metric_label, higher_better in METRICS:
        fname = f"{prefix}_{suffixes[metric_key]}.pdf"
        plot_single_metric(
            metric_key=metric_key,
            metric_label=metric_label,
            higher_better=higher_better,
            runs=runs,
            results=results,
            title=f"{GROUP_TITLES[group_key]} — {metric_label}",
            out_path=out_dir / fname,
        )


# ── NDCG over training epochs ─────────────────────────────────────────────────

def plot_ndcg_over_epochs(
    histories: dict[str, list[dict]],
    runs: list[tuple[str, str]],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)

    for run_name, label in runs:
        history = histories.get(run_name)
        if not history:
            continue
        epochs = [ep["epoch"]          for ep in history if "test" in ep]
        ndcg   = [ep["test"]["ndcg10"] for ep in history if "test" in ep]
        ax.plot(epochs, ndcg, marker="o", markersize=6, linewidth=2.5,
                color=RUN_COLOURS.get(run_name, COLOUR_DEFAULT),
                label=label.replace("\n", " "))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test NDCG@10")
    ax.legend(loc="lower right", frameon=True)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    save_fig(fig, out_path)


# ── Appendix: full convergence curves ────────────────────────────────────────

def plot_convergence_curves(
    title: str,
    histories: dict[str, list[dict]],
    runs: list[tuple[str, str]],
    out_path: Path,
) -> None:
    """
    2×2 grid of line charts (AUC, AP, NDCG, BCE Loss) over training epochs.
    One line per run in the group. Useful for showing convergence in the appendix.
    """
    fig, axes = plt.subplots(2, 2, figsize=(22, 14), constrained_layout=True)
    axes_flat = axes.flatten()

    for ax, (metric_key, metric_label, higher_better) in zip(axes_flat, ALL_METRICS):
        for run_name, label in runs:
            history = histories.get(run_name)
            if not history:
                continue
            epochs = [ep["epoch"]                    for ep in history if "test" in ep]
            vals   = [ep["test"].get(metric_key, None) for ep in history if "test" in ep]
            if any(v is None for v in vals):
                continue
            ax.plot(epochs, vals, marker="o", markersize=5, linewidth=2.2,
                    color=RUN_COLOURS.get(run_name, COLOUR_DEFAULT),
                    label=label.replace("\n", " "))

        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"Test {metric_label}")
        ax.legend(loc="lower right" if higher_better else "upper right",
                  frameon=True, fontsize=15)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    save_fig(fig, out_path)


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary_table(results: dict[str, dict]) -> None:
    col_w  = 38
    header = f"{'Run':<{col_w}}  {'AUC':>7}  {'AP':>7}  {'NDCG@10':>8}  {'BCE':>7}"
    print()
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    for group_key, runs in GROUPS.items():
        print(f"\n-- {GROUP_TITLES[group_key]} --")
        for run_name, _ in runs:
            r = results.get(run_name)
            if r is None:
                print(f"{'  ' + run_name:<{col_w}}  {'n/a':>7}  {'n/a':>7}  {'n/a':>8}  {'n/a':>7}")
            else:
                auc = f"{r['auc']:.4f}"     if r["auc"]     is not None else "n/a"
                ap  = f"{r['ap']:.4f}"      if r["ap"]      is not None else "n/a"
                nd  = f"{r['ndcg10']:.4f}"  if r["ndcg10"]  is not None else "n/a"
                ll  = f"{r['logloss']:.4f}" if r["logloss"] is not None else "n/a"
                print(f"{'  ' + run_name:<{col_w}}  {auc:>7}  {ap:>7}  {nd:>8}  {ll:>7}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot HUG experiment results")
    p.add_argument("--runs-dir", type=str, default="./runs")
    p.add_argument("--out-dir",  type=str, default="./figures")
    return p.parse_args()


def main() -> None:
    args      = parse_args()
    runs_dir  = Path(args.runs_dir)
    out_dir   = Path(args.out_dir)

    print(f"Scanning {runs_dir} …")
    results   = load_results(runs_dir)
    histories = load_histories(runs_dir)

    found = [k for k in results if any(k == rn for g in GROUPS.values() for rn, _ in g)]
    print(f"Found {len(results)} run(s), {len(found)} match expected names.")
    print_summary_table(results)

    prefixes = {
        "baselines_progression": "fig1_baselines",
        "ablations":             "fig2_ablations",
        "sensitivity":           "fig3_sensitivity",
    }

    print("Generating main figures …")
    for group_key, runs in GROUPS.items():
        plot_group(group_key=group_key, runs=runs, results=results,
                   out_dir=out_dir, prefix=prefixes[group_key])

    plot_ndcg_over_epochs(
        histories=histories,
        runs=GROUPS["baselines_progression"],
        out_path=out_dir / "fig4_ndcg_over_epochs.pdf",
    )

    print("Generating appendix figures …")
    plot_convergence_curves(
        title="Training Convergence — Baselines & Model Progression",
        histories=histories,
        runs=GROUPS["baselines_progression"],
        out_path=out_dir / "app1_convergence_baselines.pdf",
    )
    plot_convergence_curves(
        title="Training Convergence — Ablation Study",
        histories=histories,
        runs=GROUPS["ablations"],
        out_path=out_dir / "app2_convergence_ablations.pdf",
    )
    plot_convergence_curves(
        title="Training Convergence — Hyperparameter Sensitivity",
        histories=histories,
        runs=GROUPS["sensitivity"],
        out_path=out_dir / "app3_convergence_sensitivity.pdf",
    )

    print(f"\nDone. Figures in {out_dir}/")


if __name__ == "__main__":
    main()
