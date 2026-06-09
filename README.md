# wafer-proc-sim

**DISCO semiconductor dicing saw — digital twin & APC simulation suite**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-279%20passing-brightgreen)](#quick-start)
[![Languages](https://img.shields.io/badge/languages-Python%20%7C%20C%2B%2B%20%7C%20CUDA%20%7C%20Rust%20%7C%20Go%20%7C%20IEC%2061131--3-blue)](#language-coverage)

Physics-accurate simulation of a DISCO DFL7160-class dicing saw.
Hot loops in **C++ / CUDA** (pybind11), orchestration in **Python**,
PLC logic in **IEC 61131-3 Structured Text**, telemetry in **Go**, safety-critical kernels in **Rust**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Python Orchestration Layer                       │
│   notebooks/  ·  ml/  ·  analysis/  ·  pipeline/  ·  optimization/    │
└────────────────────────────┬───────────────────────────────────────────┘
                             │  pybind11 / PyO3 FFI
        ┌────────────────────┼────────────────────────┐
        │                   │                        │
        ▼                   ▼                        ▼
 ┌─────────────┐   ┌─────────────────┐   ┌─────────────────────┐
 │  C++ Kernels│   │  CUDA Kernel    │   │  Rust Kernels       │
 │  (pybind11) │   │  (nvcc sm_89)   │   │  (PyO3 / maturin)   │
 │             │   │                 │   │                     │
 │ Spindle FOC │   │  2D heat PDE    │   │  SpindleKernel      │
 │ SVPWM inv.  │   │  TILE=16 shmem  │   │  nondominated_sort  │
 │ Quad encoder│   │  Neumann BC     │   │  rbf_kernel_matrix  │
 │ Frame FEM   │   │  + source field │   └─────────────────────┘
 │ 5-axis traj │   │  (blade heat)   │
 │ Recipe seq. │   └─────────────────┘   ┌─────────────────────┐
 │ Interlock   │                         │  IEC 61131-3 ST     │
 │ EnKF (APC)  │   ┌─────────────────┐   │  (OpenPLC / CODESYS)│
 │ GP inference│   │  Go Telemetry   │   │                     │
 │ MPC optim.  │   │  HTTP :8080     │   │  SpindleFB          │
 │ NSGA-II     │   │  /health        │   │  InterlockFB        │
 └──────┬──────┘   │  /simulate      │   │  RecipeSeqFB        │
        │          │  /ws  (10 Hz)   │   │  DicingController   │
        ▼          └─────────────────┘   └─────────────────────┘
 ┌─────────────┐
 │DiscoMachine │  ← unified C++ class, one tick() per 10 µs
 │             │
 │ SpindleState│  PMSM dq-axis + PI-FOC
 │ InverterState  SVPWM + dead-time compensation
 │ MotionState │  P-ctrl XYZθ + 4× quadrature encoder
 │ E-STOP latch│  blade depth / overspeed / door interlock
 └─────────────┘
```

---

## Module Map

| Directory | Language | Description |
|-----------|----------|-------------|
| `fem/` | C++ + CUDA | Spindle FOC, SVPWM inverter, ABZ encoder, Euler-Bernoulli frame FEM, **2D wafer heat diffusion** |
| `ml/` | C++ | EnKF blade-wear APC, GP inference (RBF kernel, EI, UCB acquisition) |
| `pipeline/` | C++ | 5-axis dicing trajectory, recipe sequencer, interlock monitor, **GP-surrogate MPC** |
| `optimization/` | C++ | NSGA-II multi-objective (quality × throughput × wear Pareto front) |
| `machine/` | C++ | `DiscoMachine` — all subsystems in one deterministic tick loop |
| `analysis/` | Python | Tolerance stack-up (worst-case / RSS / Monte Carlo, Cpk) |
| `plc/` | IEC 61131-3 | `SpindleFB`, `InterlockFB`, `RecipeSeqFB`, `DicingController` |
| `rust/` | Rust / PyO3 | `SpindleKernel`, `nondominated_sort`, `rbf_kernel_matrix` — memory-safe, IEC 61508 compatible |
| `telemetry/` | Go | HTTP + WebSocket server streaming `DiscoMachine` state in real time |
| `tests/` | Python | **279 tests** — physics invariants + C++/Python parity + integration |

---

## Kernel Benchmark  (C++ / CUDA vs Python / NumPy)

| Kernel | Python | C++ | Speedup |
|--------|--------|-----|:-------:|
| NSGA-II nondom sort (N=300, obj=3) | 623 ms | 0.75 ms | **827×** |
| Quadrature encoder decode (N=100 k) | 33 ms | 0.15 ms | **215×** |
| EnKF analysis (NE=200, Nx=3) | 0.40 ms | 0.004 ms | **89×** |
| SVPWM batch (N=10 000) | 130 ms | 8.5 ms | **15×** |
| GP RBF kernel matrix (N=300, D=8) | 10.9 ms | 0.83 ms | **13×** |
| GP predict mean (M=500, N=200, D=6) | 5.3 ms | 1.7 ms | **3×** |
| **CUDA** 2D heat diffusion 256×256 | 12 ms/step ¹ | **0.08 ms/step** ² | **~150×** |

> ¹ NumPy `np.roll` FD on i9-13900K.  ² RTX 4090, sm_89 (Vancouver node).  
> Run `python benchmark_all_kernels.py` or `python fem/test_heat_diffusion.py` to reproduce.

---

## DISCO 職種カバレッジ

| 職種 | Required Skills | Covered by |
|------|----------------|------------|
| **ソフトウェア開発** | C++, embedded, real-time, Python | `machine/`, `fem/`, `pipeline/` 14 C++ kernels |
| **電気・制御** | PMSM, FOC, SVPWM, PLC (IEC 61131-3) | `SpindleFB.st`, `_spindle_kernel`, `_servo_inverter_kernel` |
| **プロセス技術** | APC, GP regression, Bayesian opt | `_enkf_kernel`, `_gp_inference_kernel`, `_mpc_kernel` |
| **生産技術** | Tolerance, yield, multi-objective opt | `analysis/tolerance_stack.py`, `_nsga2_kernel` |
| **機械設計** | FEM, stiffness, thermal PDE | `_frame_stiffness_kernel`, `_heat_diffusion_kernel.cu` |

---

## Language Coverage

```
Python          ████████████████████  orchestration, ML, analysis, notebooks
C++ (pybind11)  █████████████████     14 compiled hot-loop kernels
IEC 61131-3     ████████              PLC function blocks (OpenPLC / CODESYS)
Rust (PyO3)     ████                  memory-safe kernel alternatives
Go              ████                  HTTP/WebSocket telemetry server
CUDA (nvcc)     ████                  GPU 2D heat diffusion (RTX 4090, sm_89)
```

---

## Quick Start

### Build C++ kernels

```bash
pip install pybind11
bash build_all_kernels.sh        # builds all 15 C++ kernels
```

### Run tests

```bash
python -m pytest tests/ -q       # 279 tests, ~60 s
```

### Run digital twin

```python
from machine import _disco_machine as m

sim = m.DiscoMachine(m.MachineParams())
sim.set_target(50.0, 25.0, -0.2)   # x, y, z target [mm]
r   = sim.simulate(5000)            # 5000 × 10 µs = 50 ms

s = sim.get_state()
print(f"Speed:  {s['omega_rpm']:,.0f} rpm")
print(f"X pos:  {s['x_mm']:.3f} mm")
print(f"Mode:   {s['mode']}")
```

### Wafer thermal simulation

```python
from fem.dicing_heat_sim import DicingHeatSim

sim = DicingHeatSim(nx=256, ny=256, dx=20e-6)   # SiC defaults
sim.run_lane(y_mm=2.56, feed_mms=80.0, Q_W_per_m2=5e7)
print(f"Peak wafer temp: {sim.peak_temp:.1f} °C")  # build with CUDA for GPU
```

### MPC / APC optimizer

```python
import numpy as np
from pipeline import _mpc_kernel as mpc

p = mpc.MPCParams()
p.horizon = 10;  p.w_wear = 0.8    # penalise wear more

# x0 = [blade_wear_um, material_hardness, depth_um]
x0 = np.array([15.0, 7.5, 20.0])
# Provide GP training data (see ml/ examples for full workflow)
result = mpc.mpc_optimize(x0, Xtr, alpha_q, Xtr, alpha_w, ls, sf,
                           u_prev_feed=80.0, u_prev_rpm=30000.0, params=p)
print(f"Optimal feed: {result['feed_rate_mms']:.1f} mm/s  "
      f"rpm: {result['spindle_rpm']:.0f}")

pareto = mpc.mpc_pareto_front(x0, Xtr, alpha_q, Xtr, alpha_w, ls, sf, p)
print(f"Pareto front: {len(pareto)} non-dominated actions")
```

### Build CUDA kernel  (Vancouver RTX 4090)

```bash
CUDA_ARCH=sm_89 bash build_all_kernels.sh heat_diffusion
python fem/test_heat_diffusion.py   # reports GPU name + ~150× speedup
```

### Build Rust kernel

```bash
pip install maturin
cd rust && maturin develop --release
python -c "from wafer_proc_sim import SpindleKernel; k=SpindleKernel(); print(k.tick(30000,10))"
```

### Run Go telemetry server

```bash
cd telemetry && go mod tidy && go run .
curl http://localhost:8080/health
curl -X POST http://localhost:8080/simulate -d '{"n_steps":5000}'
# WebSocket: ws://localhost:8080/ws  → JSON frames at 10 Hz
```

### Compile PLC programs

```
OpenPLC Editor (free):
  1. Create new project → Add POU
  2. Import plc/SpindleFB.st, InterlockFB.st, RecipeSeqFB.st, DicingController.st
  3. Build (F5) → Simulate → monitor variables via Modbus TCP
```

---

## Physical Parameters

| Parameter | Value |
|-----------|-------|
| Max spindle speed | 35 000 rpm |
| Spindle motor | PMSM, p=4 pole pairs, ψ=0.08 Wb |
| Wafer material | SiC (α = 84×10⁻⁶ m²/s, ρc = 2.5×10⁶ J/m³K) |
| Control cycle | 10 µs |
| DC bus voltage | 600 V |
| Coolant fault threshold | < 0.5 L/min |
| Blade wear warn | > 30 µm |

---

## Project Structure

```
wafer-proc-sim/
├── fem/                C++ / CUDA — spindle, inverter, encoder, thermal
│   ├── _spindle_kernel.cpp
│   ├── _servo_inverter_kernel.cpp
│   ├── _encoder_kernel.cpp
│   ├── _frame_stiffness_kernel.cpp
│   ├── _heat_diffusion_kernel.cu   ← CUDA, RTX 4090
│   └── dicing_heat_sim.py          ← Python wrapper (CUDA + NumPy fallback)
├── ml/                 C++ — APC, GP, MPC
│   ├── _enkf_kernel.cpp
│   └── _gp_inference_kernel.cpp
├── pipeline/           C++ — motion, recipe, interlock, MPC
│   ├── _5axis_interpolation.cpp
│   ├── _recipe_sequencer.cpp
│   ├── _interlock_monitor.cpp
│   └── _mpc_kernel.cpp             ← GP-surrogate MPC optimizer
├── optimization/       C++ — NSGA-II
│   └── _nsga2_kernel.cpp
├── machine/            C++ — unified digital twin
│   └── _disco_machine.cpp
├── analysis/           Python — tolerance stack-up
├── plc/                IEC 61131-3 — PLC function blocks
│   ├── SpindleFB.st
│   ├── InterlockFB.st
│   ├── RecipeSeqFB.st
│   └── DicingController.st
├── rust/               Rust / PyO3 — memory-safe kernels
│   ├── Cargo.toml
│   └── src/lib.rs
├── telemetry/          Go — HTTP/WebSocket server
│   ├── main.go
│   └── go.mod
├── tests/              279 pytest tests
├── benchmark_all_kernels.py
└── build_all_kernels.sh
```

---

## License

MIT — see [LICENSE](LICENSE).
