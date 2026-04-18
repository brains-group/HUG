#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh  —  Full experiment sweep for the HUG paper
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
CACHE_DIR="./cache/1k"
OUT_DIR="./runs"
SEED=42
KG_DIM=64          # kg_relation_dim used when KGA is ON (must match out_dim)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/Framework"
KGAT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/Baselines/KGAT"

# ── Parse --data-dir argument ────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)   DATA_DIR="$2";  shift 2 ;;
        --epochs)     EPOCHS="$2";   shift 2 ;;
        --batch-size) BATCH="$2";    shift 2 ;;
        --device)     DEVICE="$2";   shift 2 ;;
        --cache-dir)  CACHE_DIR="$2"; shift 2 ;;
        --out-dir)    OUT_DIR="$2";  shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$DATA_DIR" ]]; then
    echo "ERROR: --data-dir is required."
    echo "Usage: bash run_experiments.sh --data-dir /path/to/KuaiRand-1K/data"
    exit 1
fi

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
run_kgat() {
    local label="$1"; shift
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  STARTING: $label"
    echo "════════════════════════════════════════════════════════════"
    python "$KGAT_DIR/train_kgat.py" \
        --data-dir  "$DATA_DIR" \
        --cache-dir "$CACHE_DIR" \
        --out-dir   "$OUT_DIR" \
        --epochs    "$EPOCHS" \
        --batch-size "$BATCH" \
        --seed      "$SEED" \
        $DEVICE_ARG \
        "$@"
    echo "  DONE: $label"
}

# =============================================================================
# [A] KGAT baseline  (structural KG attention, no sequential encoder)
# =============================================================================

# A1 — KGAT with IPS weighting on (fair comparison to HUG full model)
#      run_name: 1k_kgat_ips1_L2_h128d64
run_kgat "A1  KGAT baseline  (IPS on)" \
    --hidden-dim 128 \
    --out-dim    64  \
    --n-layers   2

# A2 — KGAT with IPS off  (matches the DIN/BST training objective)
#      run_name: 1k_kgat_ips0_L2_h128d64
run_kgat "A2  KGAT baseline  (IPS off)" \
    --hidden-dim 128 \
    --out-dim    64  \
    --n-layers   2   \
    --no-ips

# =============================================================================
# [B] Main model progression  (all use IPS=on, recency-gate=on)
# =============================================================================

# B1 — HUG-Unified: single HGT over full HKG, no KG alignment
#      run_name: 1k_single_kg0_ips1_rg1
run "B1  HUG-Unified  (single HGT, no KGA)" \
    --model-type single \
    --kg-alignment 0

# B2 — HUG-Dual (KGA Off): dual GNN + cross-attention, no KG gate
#      run_name: 1k_dual_kg0_ips1_rg1
run "B2  HUG-Dual (KGA Off)" \
    --model-type dual \
    --kg-alignment 0

# B3 — HUG-Dual (KGA On): full model
#      run_name: 1k_dual_kg64_ips1_rg1
run "B3  HUG-Dual (KGA On)  [FULL MODEL]" \
    --model-type dual \
    --kg-alignment $KG_DIM

# =============================================================================
# [C] IPS ablation  (full model, IPS disabled)
# =============================================================================

# C1 — Full model, IPS off
#      run_name: 1k_dual_kg64_ips0_rg1
run "C1  Ablation: no IPS  (full model)" \
    --model-type dual \
    --kg-alignment $KG_DIM \
    --no-ips

# =============================================================================
# [D] Recency-gate ablation  (full model, recency gate disabled)
# =============================================================================

# D1 — Full model, recency gate off
#      run_name: 1k_dual_kg64_ips1_rg0
run "D1  Ablation: no recency gate  (full model)" \
    --model-type dual \
    --kg-alignment $KG_DIM \
    --no-recency-gate

# =============================================================================
# [E] Combined ablation  (IPS off + recency gate off)
# =============================================================================

# E1 — Full model, no IPS, no recency gate
#      run_name: 1k_dual_kg64_ips0_rg0
run "E1  Ablation: no IPS + no recency gate" \
    --model-type dual \
    --kg-alignment $KG_DIM \
    --no-ips \
    --no-recency-gate

# =============================================================================
# [F] Hyperparameter sensitivity  (full model: B3 config with varied hypers)
# =============================================================================

# F1 — Shallow: 1 GNN layer
#      run_name: 1k_dual_kg64_ips1_rg1_L1
run "F1  Sensitivity: 1 GNN layer" \
    --model-type  dual \
    --kg-alignment $KG_DIM \
    --gnn-layers  1

# F2 — Deep: 3 GNN layers
#      run_name: 1k_dual_kg64_ips1_rg1_L3
run "F2  Sensitivity: 3 GNN layers" \
    --model-type  dual \
    --kg-alignment $KG_DIM \
    --gnn-layers  3

# F3 — Small embeddings: hidden=64, out=32
#      run_name: 1k_dual_kg32_ips1_rg1_h64d32
#      Note: kg-alignment must equal out-dim for the alignment module to work
run "F3  Sensitivity: small dims  (hidden=64, out=32)" \
    --model-type   dual \
    --kg-alignment 32 \
    --hidden-dim   64 \
    --out-dim      32

# F4 — Large embeddings: hidden=256, out=128
#      run_name: 1k_dual_kg128_ips1_rg1_h256d128
run "F4  Sensitivity: large dims  (hidden=256, out=128)" \
    --model-type   dual \
    --kg-alignment 128 \
    --hidden-dim   256 \
    --out-dim      128

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
