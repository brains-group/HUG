#!/usr/bin/env bash
# setup_fuxictr.sh — fetch the FuxiCTR sources the baselines run from.
# ─────────────────────────────────────────────────────────────────
# TransAct and WuKong live in FuxiCTR's model_zoo (not in the pip package),
# so the repository is cloned next to this script at a pinned commit.
# ─────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

FUXICTR_COMMIT=b7dff736885fdb8f59387d82d08219ad2e4cae50   # 2025-06-17, fuxictr 2.3.9
FUXICTR_VERSION=2.3.9

if [ ! -d FuxiCTR ]; then
    git clone https://github.com/reczoo/FuxiCTR.git
fi
git -C FuxiCTR fetch --quiet origin
git -C FuxiCTR checkout --quiet "$FUXICTR_COMMIT"

for m in TransAct WuKong; do
    [ -d "FuxiCTR/model_zoo/$m/src" ] || { echo "missing FuxiCTR/model_zoo/$m"; exit 1; }
done

pip install -q "fuxictr==${FUXICTR_VERSION}" "numpy<2" polars pyarrow scikit-learn pyyaml
echo "FuxiCTR ready at $(git -C FuxiCTR rev-parse --short HEAD)"
