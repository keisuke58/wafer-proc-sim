#!/usr/bin/env bash
# DISCO interview demo — paper spine + software stack (+ optional Streamlit).
#
# Usage:
#   ./scripts/demo_disco.sh              # quick (~3 min): figures + LOO + SW stack
#   ./scripts/demo_disco.sh --full       # full paper reproduction + SW stack
#   ./scripts/demo_disco.sh --streamlit  # quick + launch demos/paper_app.py
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="quick"
STREAMLIT=false
for arg in "$@"; do
  case "$arg" in
    --full) MODE="full" ;;
    --streamlit) STREAMLIT=true ;;
    --help|-h)
      sed -n '2,7p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg (try --help)" >&2
      exit 1
      ;;
  esac
done

banner() {
  echo ""
  echo "════════════════════════════════════════════════════════════"
  echo "  $1"
  echo "════════════════════════════════════════════════════════════"
}

banner "DISCO Interview Demo — wafer-proc-sim"

if [[ "$MODE" == "full" ]]; then
  banner "[1/4] Full paper reproduction"
  ./scripts/reproduce_paper.sh
else
  banner "[1/4] Paper figures"
  python paper/sic_gp_quality/gen_figures.py

  banner "[2/4] Heteroscedastic GP LOO (Grade A/B)"
  python ml/train_from_experimental.py --loo --quality AB
fi

step_sw="[3/4]"
step_note="[4/4]"
if [[ "$MODE" == "full" ]]; then
  step_sw="[2/4]"
  step_note="[3/4]"
fi

banner "$step_sw DISCO machine control software stack"
python -c "from pipeline.disco_sw_stack import demo; demo()"

banner "$step_note Process optimization preview (sic_dicing_pipeline)"
python pipeline/sic_dicing_pipeline.py --no-plots

banner "Demo complete"
echo ""
echo "  Figures : paper/sic_gp_quality/figures/"
echo "  Portfolio map : docs/PORTFOLIO.md"
echo "  Full semiconductor chain : python pipeline/run_full_pipeline.py --process stealth"
if [[ "$STREAMLIT" == true ]]; then
  echo ""
  echo "  Launching Streamlit paper demo …"
  exec streamlit run demos/paper_app.py
else
  echo "  Streamlit UI : streamlit run demos/paper_app.py"
fi