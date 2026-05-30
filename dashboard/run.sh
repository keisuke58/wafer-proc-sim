#!/usr/bin/env bash
# Launch FastAPI backend + Streamlit frontend for the dicing dashboard.
#
# Usage:
#   bash dashboard/run.sh            # default ports 8000 / 8501
#   API_PORT=9000 bash dashboard/run.sh

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT"

API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-8501}"

# ── dependency check ──────────────────────────────────────────────────────────
for pkg in uvicorn fastapi streamlit plotly requests; do
    python -c "import $pkg" 2>/dev/null || {
        echo "[!] Missing: $pkg — run: pip install -r requirements-dashboard.txt"
        exit 1
    }
done

cleanup() {
    echo ""
    echo "[*] Shutting down..."
    kill "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── start API ─────────────────────────────────────────────────────────────────
echo "[*] FastAPI  → http://localhost:${API_PORT}/api/history"
uvicorn dashboard.api:app \
    --port "$API_PORT" \
    --log-level warning \
    --no-access-log &
API_PID=$!

# give uvicorn a moment to bind
sleep 1

# ── start Streamlit ───────────────────────────────────────────────────────────
echo "[*] Streamlit → http://localhost:${UI_PORT}"
DASHBOARD_API="http://localhost:${API_PORT}" \
streamlit run "$ROOT/dashboard/app.py" \
    --server.port "$UI_PORT" \
    --server.headless true \
    --browser.gatherUsageStats false
