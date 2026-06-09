// telemetry/main.go
//
// DISCO dicing saw digital twin — telemetry server
//
// Runs DiscoMachine via a Python subprocess and exposes its state over
// HTTP / WebSocket so an HMI dashboard can read real-time data.
//
// Endpoints:
//   GET  /health             → {"status":"ok"}
//   POST /simulate           → JSON body {"n_steps":N}  → telemetry
//   GET  /ws                 → WebSocket stream (JSON frames at ~100 Hz)
//
// Usage:
//   go run ./telemetry        (requires PYTHON env var or 'python' on PATH)

package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"os/exec"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

// ── Telemetry frame ───────────────────────────────────────────────────────────

type TelemetryFrame struct {
	TimestampMs  int64   `json:"ts_ms"`
	StepIndex    int     `json:"step"`
	SpeedRPM     float64 `json:"speed_rpm"`
	TorqueNm     float64 `json:"torque_Nm"`
	XMM          float64 `json:"x_mm"`
	YMM          float64 `json:"y_mm"`
	ZMM          float64 `json:"z_mm"`
	EStop        bool    `json:"e_stop"`
	Mode         string  `json:"mode"`
}

// ── Machine state (shared, guarded by mutex) ──────────────────────────────────

type MachineState struct {
	mu      sync.RWMutex
	latest  TelemetryFrame
	running bool
}

var state = &MachineState{}

// ── Python bridge — call DiscoMachine.simulate() via subprocess ───────────────

const pythonScript = `
import sys, json
sys.path.insert(0, '.')
from machine import _disco_machine as _dm
import math

n = int(sys.argv[1])
m = _dm.DiscoMachine(_dm.MachineParams())
m.set_target(50.0, 25.0, -0.2)
r = m.simulate(n)
s = m.get_state()

out = {
    "speed_rpm":  s["omega_rad_s"] * 60 / (2 * math.pi),
    "torque_Nm":  s["T_e_Nm"],
    "x_mm":       s["x_mm"],
    "y_mm":       s["y_mm"],
    "z_mm":       s["z_mm"],
    "mode":       s["mode"],
    "e_stop":     s["e_stop"],
    "omega_series": list(r["omega_rad_s"])[:200],
}
print(json.dumps(out))
`

func runSimulation(nSteps int) (map[string]interface{}, error) {
	py := os.Getenv("PYTHON")
	if py == "" {
		py = "python"
	}

	// Write temp script
	f, err := os.CreateTemp("", "disco_sim_*.py")
	if err != nil {
		return nil, err
	}
	defer os.Remove(f.Name())
	if _, err = f.WriteString(pythonScript); err != nil {
		return nil, err
	}
	f.Close()

	out, err := exec.Command(py, f.Name(), fmt.Sprintf("%d", nSteps)).Output()
	if err != nil {
		return nil, fmt.Errorf("python error: %w", err)
	}

	var result map[string]interface{}
	if err = json.Unmarshal(out, &result); err != nil {
		return nil, fmt.Errorf("json parse: %w", err)
	}
	return result, nil
}

// ── Handlers ──────────────────────────────────────────────────────────────────

func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func simulateHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		NSteps int `json:"n_steps"`
	}
	req.NSteps = 5000
	json.NewDecoder(r.Body).Decode(&req)

	result, err := runSimulation(req.NSteps)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

// ── WebSocket broadcaster ─────────────────────────────────────────────────────

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

// Synthetic real-time stream (runs DiscoMachine in Go-side loop, 100 ms ticks)
func streamLoop() {
	step := 0
	t0   := time.Now()
	omega := 0.0
	x     := 0.0

	for {
		time.Sleep(100 * time.Millisecond)
		// Advance synthetic spindle (lightweight approximation for stream demo)
		dt := 0.1
		omegaRef := 30000.0 * 2 * math.Pi / 60.0
		omega += (omegaRef-omega) * dt * 2.0
		x += (50.0 - x) * dt * 0.5

		f := TelemetryFrame{
			TimestampMs: time.Since(t0).Milliseconds(),
			StepIndex:   step % 12,
			SpeedRPM:    omega * 60 / (2 * math.Pi),
			TorqueNm:    0.1 * omega / omegaRef,
			XMM:         x,
			YMM:         0.0,
			ZMM:         -0.2 * (1 - math.Exp(-float64(step)*0.01)),
			EStop:       false,
			Mode:        "RUNNING",
		}
		step++

		state.mu.Lock()
		state.latest  = f
		state.running = true
		state.mu.Unlock()
	}
}

func wsHandler(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Println("ws upgrade:", err)
		return
	}
	defer conn.Close()

	for {
		time.Sleep(100 * time.Millisecond)
		state.mu.RLock()
		f := state.latest
		state.mu.RUnlock()

		if err := conn.WriteJSON(f); err != nil {
			break
		}
	}
}

// ── main ──────────────────────────────────────────────────────────────────────

func main() {
	go streamLoop()

	http.HandleFunc("/health",   healthHandler)
	http.HandleFunc("/simulate", simulateHandler)
	http.HandleFunc("/ws",       wsHandler)

	addr := ":8080"
	log.Printf("DISCO telemetry server listening on %s", addr)
	log.Printf("  GET  /health      — liveness probe")
	log.Printf("  POST /simulate    — run simulation, returns JSON")
	log.Printf("  GET  /ws          — WebSocket real-time stream")
	if err := http.ListenAndServe(addr, nil); err != nil {
		log.Fatal(err)
	}
}
