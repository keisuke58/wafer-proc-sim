#!/usr/bin/env bash
# Reproduce the core paper results (figures + GP LOO + dicing pipeline).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> [1/3] Paper figures (paper/sic_gp_quality/gen_figures.py)"
python paper/sic_gp_quality/gen_figures.py

echo "==> [2/3] Heteroscedastic GP LOO — Grade A/B (ml/train_from_experimental.py)"
python ml/train_from_experimental.py --loo --quality AB

echo "==> [3/3] Full dicing pipeline (pipeline/sic_dicing_pipeline.py)"
python pipeline/sic_dicing_pipeline.py --no-plots

echo "==> Done. Figures in paper/sic_gp_quality/figures/ and results/"