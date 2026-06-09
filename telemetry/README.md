# Telemetry server — wafer-proc-sim

Go HTTP/WebSocket server that streams `DiscoMachine` digital twin output
to any HMI dashboard (browser, Grafana, custom client).

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET  | `/health`   | Liveness probe → `{"status":"ok"}` |
| POST | `/simulate` | Body `{"n_steps":5000}` → JSON telemetry from Python kernel |
| GET  | `/ws`       | WebSocket — JSON frames at 10 Hz, `TelemetryFrame` schema |

## TelemetryFrame schema

```json
{
  "ts_ms":      12345,
  "step":       3,
  "speed_rpm":  29814.2,
  "torque_Nm":  0.043,
  "x_mm":       12.5,
  "y_mm":       0.0,
  "z_mm":       -0.18,
  "e_stop":     false,
  "mode":       "RUNNING"
}
```

## Run

```bash
# Install Go >= 1.21
cd telemetry
go mod tidy
go run .

# Test
curl http://localhost:8080/health
curl -X POST http://localhost:8080/simulate -d '{"n_steps":5000}'
```

## Architecture

```
Browser / Grafana
      │  WebSocket / HTTP
      ▼
  Go server  (gorilla/websocket)
      │  os/exec + JSON
      ▼
  Python subprocess  (machine/_disco_machine.so)
      │  pybind11 FFI
      ▼
  C++ DiscoMachine kernel
```

The Go server is intentionally thin: all physics stay in the C++ kernel,
Go handles concurrency (goroutines, channels) and I/O — the same split
used in real DISCO machine software (C++ control plane, separate HMI
process communicating over TCP).
