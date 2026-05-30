#!/bin/bash
# Submit 5 extended FEM jobs sequentially (1 CPU each)
set -e
ABQ=/home/nishioka/DassaultSystemes/SIMULIA/Commands/abaqus
DIR="$(cd "$(dirname "$0")" && pwd)"

JOBS=(
  dicing_4HSiC_d080_bw23
  dicing_4HSiC_d150_bw23
  dicing_4HSiC_d220_bw23
  dicing_4HSiC_d290_bw23
  dicing_4HSiC_d360_bw23
)

for JOB in "${JOBS[@]}"; do
  echo "[->] Submitting $JOB ..."
  cd "$DIR"
  $ABQ job="${JOB}_run" input="${JOB}.inp" cpus=1 interactive ask_delete=off
  echo "[OK] $JOB done"
done
echo "[ALL DONE]"
