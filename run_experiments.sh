#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh  —  Full experiment sweep
# =============================================================================
#
# USAGE
#   bash run_experiments.sh --data-dir /path/to/KuaiRand-1K/data
#
# All results land in ./runs/<run_name>/  where run_name encodes every flag.
#
# Experiments
# -----------
#   [A] KGAT baseline        — knowledge-graph attention network (Wang KDD 2019)
#   [B] Main model progression — re-run with fixed per-user NDCG@10
#   [C] IPS ablation           — full model with IPS disabled
#   [D] Recency-gate ablation  — full model with recency gate disabled
#   [E] Combined ablation      — both IPS and recency gate disabled
#   [F] Hyperparameter sensitivity (full model):
#         F1  Shallow:  1 GNN layer
#         F2  Deep:     3 GNN layers
#         F3  Small:    hidden=64, out=32
#         F4  Large:    hidden=256, out=128
#
# DIN + BST baselines are handled separately via Baselines/train.py.
#
# The HKG is built once and cached in ./cache/1k.
# Subsequent runs load from cache (fast).
#
# Set EPOCHS, BATCH, DEVICE below to override defaults.
# =============================================================================

set -euo pipefail

# ── User-configurable ────────────────────────────────────────────────────────
DATA_DIR=""
EPOCHS=20
BATCH=2048
DEVICE=""          # leave empty to auto-detect (cuda > mps > cpu)
CACHE_DIR="Framework/cache/1K"
OUT_DIR="./runs"
SEED=42
KG_DIM=64          # kg_relation_dim used when KGA is ON (must match out_dim)
GROUP=""           # "" = run all; "0" = group-0 runs only; "1" = group-1 runs only
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/Framework"
KGAT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/Baselines/KGAT"

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)   DATA_DIR="$2";  shift 2 ;;
        --epochs)     EPOCHS="$2";   shift 2 ;;
        --batch-size) BATCH="$2";    shift 2 ;;
        --device)     DEVICE="$2";   shift 2 ;;
        --cache-dir)  CACHE_DIR="$2"; shift 2 ;;
        --out-dir)    OUT_DIR="$2";  shift 2 ;;
        --group)      GROUP="$2";    shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$DATA_DIR" ]]; then
    echo "ERROR: --data-dir is required."
    echo ""
    echo "  Single node (all runs sequentially):"
    echo "    bash run_experiments.sh --data-dir /path/to/data"
    echo ""
    echo "  Two nodes in parallel (run each on its own node):"
    echo "    node-A:  bash run_experiments.sh --data-dir /path/to/data --device cuda:0 --group 0"
    echo "    node-B:  bash run_experiments.sh --data-dir /path/to/data --device cuda:0 --group 1"
    echo ""
    echo "    Group 0: A1 A2 B1 B2 B3  (KGAT baseline + model progression)"
    echo "    Group 1: C1 D1 E1 F1 F2 F3 F4  (ablations + sensitivity)"
    echo ""
    echo "  NOTE: --cache-dir should point to a shared filesystem path so both"
    echo "        nodes load the same cached HKG rather than rebuilding it."
    exit 1
fi

if [[ -n "$GROUP" && "$GROUP" != "0" && "$GROUP" != "1" ]]; then
    echo "ERROR: --group must be 0 or 1"
    exit 1
fi

# Convenience: returns 0 (true) if the given group should run on this node.
# Usage: in_group 0 && run ...   or   in_group 1 && run ...
in_group() {
    [[ -z "$GROUP" || "$GROUP" == "$1" ]]
}

DEVICE_ARG=""
if [[ -n "$DEVICE" ]]; then
    DEVICE_ARG="--device $DEVICE"
fi

# ── Helpers ───────────────────────────────────────────────────────────────────

# run: launch a Framework/main.py experiment
run() {
    local label="$1"; shift
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  STARTING: $label"
    echo "════════════════════════════════════════════════════════════"
    python "$SCRIPT_DIR/main.py" \
        --data-dir   "$DATA_DIR" \
        --cache-dir  "$CACHE_DIR" \
        --output-dir "$OUT_DIR" \
        --epochs     "$EPOCHS" \
        --batch-size "$BATCH" \
        --seed       "$SEED" \
        $DEVICE_ARG \
        "$@"
    echo "  DONE: $label"
}

# run_kgat: launch a Baselines/KGAT/train_kgat.py experiment
# run_kgat() {
#     local label="$1"; shift
#     echo ""
#     echo "════════════════════════════════════════════════════════════"
#     echo "  STARTING: $label"
#     echo "════════════════════════════════════════════════════════════"
#     python "$KGAT_DIR/train_kgat.py" \
#         --data-dir  "$DATA_DIR" \
#         --cache-dir "$CACHE_DIR" \
#         --out-dir   "$OUT_DIR" \
#         --epochs    "$EPOCHS" \
#         --batch-size "$BATCH" \
#         --seed      "$SEED" \
#         $DEVICE_ARG \
#         "$@"
#     echo "  DONE: $label"
# }

# =============================================================================
# [A] KGAT baseline  (structural KG attention, no sequential encoder)
# =============================================================================

# A1 — KGAT with IPS weighting on (fair comparison to HUG full model)
#      run_name: 1k_kgat_ips1_L2_h128d64
# in_group 0 && run_kgat "A1  KGAT baseline  (IPS on)" \
#     --hidden-dim 128 \
#     --out-dim    64  \
#     --n-layers   2

# # A2 — KGAT with IPS off  (matches the DIN/BST training objective)
# #      run_name: 1k_kgat_ips0_L2_h128d64
# in_group 0 && run_kgat "A2  KGAT baseline  (IPS off)" \
#     --hidden-dim 128 \
#     --out-dim    64  \
#     --n-layers   2   \
#     --no-ips

# =============================================================================
# [B] Main model progression  (all use IPS=on, recency-gate=on)
# =============================================================================

# B1 — HUG-Unified: single HGT over full HKG, no KG alignment
#      run_name: 1k_single_kg0_ips1_rg1
# in_group 0 && run "B1  HUG-Unified  (single HGT, no KGA)" \
#     --model-type single \
#     --kg-alignment 0

# B2 — HUG-Dual (KGA Off): dual GNN + cross-attention, no KG gate
#      run_name: 1k_dual_kg0_ips1_rg1
# in_group 0 && run "B2  HUG-Dual (KGA Off)" \
#     --model-type dual \
#     --kg-alignment 0

# # B3 — HUG-Dual (KGA On): full model
# #      run_name: 1k_dual_kg64_ips1_rg1
# in_group 0 && run "B3  HUG-Dual (KGA On)  [FULL MODEL]" \
#     --model-type dual \
#     --kg-alignment $KG_DIM

# =============================================================================
# [C] IPS ablation  (full model, IPS disabled)
# =============================================================================

# C1 — Full model, IPS off
#      run_name: 1k_dual_kg64_ips0_rg1
# in_group 1 && run "C1  Ablation: no IPS  (full model)" \
#     --model-type dual \
#     --kg-alignment $KG_DIM \
#     --no-ips
 in_group 0 && run "B2  HUG-Unified  (single HGT, no KGA)" \
     --model-type single \
     --kg-alignment 0 \
     --no-ips
# =============================================================================
# [D] Recency-gate ablation  (full model, recency gate disabled)
# =============================================================================

# D1 — Full model, recency gate off
#      run_name: 1k_dual_kg64_ips1_rg0
# in_group 1 && run "D1  Ablation: no recency gate  (full model)" \
#     --model-type dual \
#     --kg-alignment $KG_DIM \
#     --no-recency-gate

# # =============================================================================
# # [E] Combined ablation  (IPS off + recency gate off)
# # =============================================================================

# # E1 — Full model, no IPS, no recency gate
# #      run_name: 1k_dual_kg64_ips0_rg0
# in_group 1 && run "E1  Ablation: no IPS + no recency gate" \
#     --model-type dual \
#     --kg-alignment $KG_DIM \
#     --no-ips \
#     --no-recency-gate

# # =============================================================================
# # [F] Hyperparameter sensitivity  (full model: B3 config with varied hypers)
# # =============================================================================

# # F1 — Shallow: 1 GNN layer
# #      run_name: 1k_dual_kg64_ips1_rg1_L1
# in_group 1 && run "F1  Sensitivity: 1 GNN layer" \
#     --model-type  dual \
#     --kg-alignment $KG_DIM \
#     --gnn-layers  1

# # F2 — Deep: 3 GNN layers
# #      run_name: 1k_dual_kg64_ips1_rg1_L3
# in_group 1 && run "F2  Sensitivity: 3 GNN layers" \
#     --model-type  dual \
#     --kg-alignment $KG_DIM \
#     --gnn-layers  3

# # F3 — Small embeddings: hidden=64, out=32
# #      run_name: 1k_dual_kg32_ips1_rg1_h64d32
# #      Note: kg-alignment must equal out-dim for the alignment module to work
# in_group 1 && run "F3  Sensitivity: small dims  (hidden=64, out=32)" \
#     --model-type   dual \
#     --kg-alignment 32 \
#     --hidden-dim   64 \
#     --out-dim      32

# # F4 — Large embeddings: hidden=256, out=128
# #      run_name: 1k_dual_kg128_ips1_rg1_h256d128
# in_group 1 && run "F4  Sensitivity: large dims  (hidden=256, out=128)" \
#     --model-type   dual \
#     --kg-alignment 128 \
#     --hidden-dim   256 \
#     --out-dim      128

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ALL RUNS COMPLETE"
echo "  Results are in: $OUT_DIR/"
echo ""
echo "  KGAT baselines:"
echo "    $OUT_DIR/1k_kgat_ips1_L2_h128d64/      → A1  KGAT (IPS on)"
echo "    $OUT_DIR/1k_kgat_ips0_L2_h128d64/      → A2  KGAT (IPS off)"
echo ""
echo "  Model progression:"
echo "    $OUT_DIR/1k_single_kg0_ips1_rg1/       → B1  HUG-Unified"
echo "    $OUT_DIR/1k_dual_kg0_ips1_rg1/         → B2  HUG-Dual (KGA Off)"
echo "    $OUT_DIR/1k_dual_kg${KG_DIM}_ips1_rg1/        → B3  HUG-Dual (KGA On)  FULL MODEL"
echo ""
echo "  Ablations:"
echo "    $OUT_DIR/1k_dual_kg${KG_DIM}_ips0_rg1/        → C1  No IPS"
echo "    $OUT_DIR/1k_dual_kg${KG_DIM}_ips1_rg0/        → D1  No recency gate"
echo "    $OUT_DIR/1k_dual_kg${KG_DIM}_ips0_rg0/        → E1  No IPS + no recency gate"
echo ""
echo "  Hyperparameter sensitivity:"
echo "    $OUT_DIR/1k_dual_kg${KG_DIM}_ips1_rg1_L1/     → F1  1 layer"
echo "    $OUT_DIR/1k_dual_kg${KG_DIM}_ips1_rg1_L3/     → F2  3 layers"
echo "    $OUT_DIR/1k_dual_kg32_ips1_rg1_h64d32/        → F3  small dims"
echo "    $OUT_DIR/1k_dual_kg128_ips1_rg1_h256d128/     → F4  large dims"
echo ""
echo "  Each directory contains:"
echo "    final_metrics.json   — AUC, AP, LogLoss, per-user NDCG@10 (train + test)"
echo "    history.json         — per-epoch metrics"
echo "════════════════════════════════════════════════════════════"
