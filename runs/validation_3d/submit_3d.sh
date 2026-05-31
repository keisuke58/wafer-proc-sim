#!/bin/bash
# Submit 3D validation FEM jobs (after 2D Gc recalibration completes)
# Purpose: validate 2D surrogate trends with coarse 3D models at key conditions
#
# Job 1: d150_bw23  — peak del_frac in 2D  (~2.5h)
# Job 2: d300_bw23  — deeper cut comparison (~3.0h)
#
# Usage: bash submit_3d.sh [d150|d300|all]

ABQ=/home/nishioka/DassaultSystemes/SIMULIA/Commands/abaqus
SCRIPT=$(cd "$(dirname "$0")/../.." && pwd)/fem/dicing_blade_3d.py
DIR="$(cd "$(dirname "$0")" && pwd)"

run_job() {
  local config="$1"
  local job_name
  job_name=$(python3 -c "import json; d=json.load(open('$config')); print(d.get('job_name','dicing_3d'))")

  STA="${DIR}/${job_name}_run.sta"
  if [ -f "$STA" ] && grep -q "HAS COMPLETED SUCCESSFULLY" "$STA"; then
    echo "[SKIP] $job_name (already completed)"
    return
  fi

  echo "[->] Submitting 3D job: $job_name ..."
  cp "$config" "${DIR}/run_config_3d.json"
  cd "$DIR"
  $ABQ cae noGUI="$SCRIPT" -- 2>&1 | tee "${job_name}.log"
  echo "[OK] $job_name"
}

case "${1:-all}" in
  d150) run_job "${DIR}/run_config_3d_d150.json" ;;
  d300) run_job "${DIR}/run_config_3d_d300.json" ;;
  all)
    run_job "${DIR}/run_config_3d_d150.json"
    run_job "${DIR}/run_config_3d_d300.json"
    ;;
  *) echo "Usage: $0 [d150|d300|all]"; exit 1 ;;
esac
echo "[ALL DONE]"
