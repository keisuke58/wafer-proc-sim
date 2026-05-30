#!/bin/bash
# Submit extended FEM jobs sequentially (1 CPU each, SPOS+center fix applied)
# Skip jobs whose _run.sta already shows "HAS COMPLETED SUCCESSFULLY"
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
  STA="${DIR}/${JOB}_run.sta"
  if [ -f "$STA" ] && grep -q "HAS COMPLETED SUCCESSFULLY" "$STA"; then
    echo "[SKIP] $JOB (already completed)"
    continue
  fi
  echo "[->] Submitting $JOB ..."
  cd "$DIR"
  $ABQ job="${JOB}_run" input="${JOB}.inp" cpus=1 interactive ask_delete=off
  echo "[OK] $JOB"
done
echo "[ALL DONE]"
