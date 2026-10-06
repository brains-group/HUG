#!/usr/bin/env bash
# run_all.sh — end-to-end pipeline for the TransAct + WuKong baselines
# ─────────────────────────────────────────────────────────────────
# Usage:
#   bash run_all.sh                  # GPU 0, both models
#   bash run_all.sh 1                # GPU 1, both models
#   bash run_all.sh 0 TransAct       # GPU 0, one model
#   DATA_DIR=/path/to/KuaiRand-1K/data bash run_all.sh
# ─────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

GPU=${1:-0}
MODEL=${2:-all}
DATA_DIR=${DATA_DIR:-../KuaiRand-1K/data}

echo "[1/3] Fetching FuxiCTR …"
bash setup_fuxictr.sh

echo "[2/3] Preprocessing KuaiRand-1K (shared chronological split) …"
python preprocess.py --data-dir "$DATA_DIR"

echo "[3/3] Training + evaluating ${MODEL} on GPU ${GPU} …"
python train.py --model "$MODEL" --gpu "$GPU"

echo "Done. Results: ../runs/<Model>/final_metrics.json"
