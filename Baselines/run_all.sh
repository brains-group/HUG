#!/usr/bin/env bash
# run_all.sh  — end-to-end pipeline for DIN + BST on KuaiRand-1K
# ─────────────────────────────────────────────────────────────────
# Usage:
#   bash run_all.sh            # CPU, both models
#   bash run_all.sh 0          # GPU 0, both models
#   bash run_all.sh 0 DIN      # GPU 0, DIN only
# ─────────────────────────────────────────────────────────────────

GPU=${1:--1}
MODEL=${2:-both}

set -euo pipefail

echo ""
echo "========================================================"
echo "  KuaiRand-1K  |  DIN + BST  |  FuxiCTR pipeline"
echo "========================================================"
echo ""

# 0. Clone FuxiCTR if not already present
if [ ! -d "./FuxiCTR" ]; then
    echo "[0/3] Cloning FuxiCTR …"
    git clone https://github.com/reczoo/FuxiCTR.git
fi

# Install python requirements
pip install -q fuxictr scikit-learn pandas numpy pyyaml torch

# 1. Preprocess
echo ""
echo "[1/3] Preprocessing KuaiRand-1K …"
python preprocess.py

# 2. Train
echo ""
echo "[2/3] Training model(s): ${MODEL} on GPU=${GPU} …"
python train.py --model "${MODEL}" --gpu "${GPU}"

# 3. Evaluate
echo ""
echo "[3/3] Evaluating …"
python evaluate.py --model "${MODEL}" --gpu "${GPU}" \
    --out_csv ../runs/baselines/evaluation_results.csv \
    || python step3b_evaluate_standalone.py \
           --model "${MODEL}" --gpu "${GPU}" \
           --out_csv ../runs/baselines/evaluation_results.csv

echo ""
echo "========================================================"
echo "  Pipeline complete.  Results: ./results/"
echo "========================================================"