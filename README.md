# wafer-proc-sim

**DISCO semiconductor dicing saw — digital twin & APC simulation suite**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-673%20passing-brightgreen)](#quick-start)
[![Languages](https://img.shields.io/badge/languages-Python%20%7C%20C%2B%2B%20%7C%20CUDA%20%7C%20Rust%20%7C%20Go%20%7C%20IEC%2061131--3-blue)](#language-coverage)

Physics-accurate simulation of a DISCO DFL7160-class dicing saw.
Hot loops in **C++ / CUDA** (pybind11), orchestration in **Python**,
PLC logic in **IEC 61131-3 Structured Text**, telemetry in **Go**, safety-critical kernels in **Rust**.

---

## 📂 Portfolio — start here

> An engineering portfolio by **Keisuke Nishioka (西岡 佳祐)**: computational mechanics turned into design decisions — FEA & material mechanics, optimization & uncertainty quantification, and CAE-surrogate / generative design. Every module is written from scratch (no black-box solvers) and validated against a closed-form or analytic reference.

**▶ Live dashboards:** **https://keisuke58.github.io/wafer-proc-sim/** ・ **▶ 会社別サマリ:** **[companies.html](https://keisuke58.github.io/wafer-proc-sim/companies.html)**

### 🗺 10社×専用ダッシュボード

同一リポジトリで10社それぞれのド真ん中の技術を「物理モデル→実装→定量評価→テスト→公開ページ」まで一気通貫で実装（自作モジュールのみ・結果JSONコミット済み・CI 673テスト緑）:

| 会社 | 領域 | 看板の数字 | ページ |
|------|------|-----------|--------|
| **DISCO**（本命） | 切る・削る・磨く×知能化 | ドレッシングCBM総コスト−55%、AE異常AUROC 1.0 | [dashboard](https://keisuke58.github.io/wafer-proc-sim/dashboard.html) ほか6面 |
| **Sony** | 鏡筒メカ＋センサ物理＋AF | STOP温度窓55→119°C、PTC変換ゲイン0.77% | [optomech](https://keisuke58.github.io/wafer-proc-sim/optomech.html) |
| **Keyence** | 変位計/画像寸法の信号鎖 | ロバスト抽出5.5µm vs 重心35µm、実動JSデモ | [keyence](https://keisuke58.github.io/wafer-proc-sim/keyence.html) |
| **Lasertec** | EUVマスク検査 | 転写性ΔCD格付け、フォトンバジェット7.2σ | [inspection](https://keisuke58.github.io/wafer-proc-sim/inspection.html) |
| **SCREEN** | 枚葉洗浄物理 | PRE@30nm 0→85%（メガソニック） | [screen](https://keisuke58.github.io/wafer-proc-sim/screen.html) |
| **Advantest** | テスト工学 | 注入RJを12.8%誤差で回収→1e-12外挿 | [advantest](https://keisuke58.github.io/wafer-proc-sim/advantest.html) |
| **キオクシア** | 3D NANDデバイス物理 | 耐久定格5322 P/E＝物理から導出 | [kioxia](https://keisuke58.github.io/wafer-proc-sim/kioxia.html) |
| **東京エレクトロン** | エッチ/ALD | 1ノブでBOW→垂直→TAPER、ドーズ∝AR^1.93 | [tel](https://keisuke58.github.io/wafer-proc-sim/tel.html) |
| **東京精密** | ISO粗さ・真円度 | λc透過率50.0%、Ra=2A/π閉ループ | [accretech](https://keisuke58.github.io/wafer-proc-sim/accretech.html) |
| **荏原製作所** | 真空ポンプ・CMP | N2ベントで1Pa到達8.6×短縮、密度CMP 373nm | [ebara](https://keisuke58.github.io/wafer-proc-sim/ebara.html) |

Two project families live in this repo:

### 1 · Semiconductor process — DISCO-class digital twin & metrology (`vision/`, `machine/`, this suite)
Dicing, back-grinding, laser processing, CMP, machine-systems and a metrology/optimization layer — the measure → process → decide loop.
- **Dicing / grind / laser / CMP** — kerf detection, TTV·BOW·SSD, ablation & stealth-dicing, Preston removal + EPD
- **Machine systems** — real-time S-curve motion + servo, digital-twin OEE, bump inspection, ADC (CNN → yield map)
- **Metrology & optimization** — Gage R&R calibration, phase-correlation registration, Bayesian recipe search (GP + EI), A\* + 2-opt transport
- Dashboards: [processing](https://keisuke58.github.io/wafer-proc-sim/dashboard.html) · [grinding](https://keisuke58.github.io/wafer-proc-sim/grind.html) · [laser](https://keisuke58.github.io/wafer-proc-sim/laser.html) · [systems](https://keisuke58.github.io/wafer-proc-sim/systems.html) · [metrology](https://keisuke58.github.io/wafer-proc-sim/metro.html)

### 2 · Optomechanics — camera lens-barrel design & simulation ([`optomech/`](optomech/))
The optics ⇄ mechanics boundary: tolerance stack-up, mechanism/control, structural·thermal FEA, wave-optics image quality, generative design, and parametric CAD. **Dashboard → [optomech.html](https://keisuke58.github.io/wafer-proc-sim/optomech.html) (14 cards).**
- **CAE** — beam + **axisymmetric-solid FE** (1st mode 10.6 kHz within 0.14 %; thermal growth matches αLΔT within 2.5 %), drop-shock (Newmark-β)
- **Wave-optics MTF** — mechanical error → wavefront → MTF (matches diffraction limit to 0.0003); MTF-based tolerancing
- **Control** — OIS servo loop shaping (PM 53° / GM 13 dB), disturbance rejection (3.4 stops), resonance notch (−4 → +5 dB)
- **Generative design** — PyTorch **GNN surrogate** (R² 0.99) + **Bayesian** search → lightest barrel meeting constraints
- **CAD** — dimension-driven revolved solid (**STL**) + **GD&T** drawing (**DXF**), tolerances derived from the boresight budget

**Research:** *Frontiers in Materials* (2025) — FEA × GNN for defect identification in perforated CFRP · GPU-accelerated Bayesian inference in JAX (Leibniz Universität Hannover).

---

## Architecture

### System overview

```
╔══════════════════════════════════════════════════════════════════════════════╗
║   wafer-proc-sim  ·  DISCO DFL7160  Digital Twin  &  APC Suite              ║
╚══════════════════════════════════════════════════════════════════════════════╝

 ┌────────────────────────────────────────────────────────────────────────┐
 │  PYTHON   notebooks · ml/ · analysis/ · pipeline/ · optimization/     │
 └────────────────────────┬───────────────────────────────────────────────┘
          pybind11 FFI    │                                      PyO3 FFI
 ┌────────────────────────▼───────────────────────────────────────────────┐
 │  NATIVE KERNELS                                                        │
 │                                                                        │
 │ ┌─ C++  (14 shared libs) ───────────────────────────────────────────┐ │
 │ │                                                                   │ │
 │ │  Electrical            Motion / Mechanical    ML / APC            │ │
 │ │  ┌──────────────┐      ┌─────────────────┐   ┌────────────────┐  │ │
 │ │  │ _spindle     │      │ _encoder   4×AB │   │ _enkf  EnKF    │  │ │
 │ │  │  PMSM+PI-FOC │      │ _5axis  traj.   │   │ _gp    RBF+EI  │  │ │
 │ │  │ _inverter    │      │ _frame_fem  FEM  │   │ _mpc   GP-MPC  │  │ │
 │ │  │  SVPWM       │      └─────────────────┘   │ _nsga2 827×    │  │ │
 │ │  └──────────────┘                             └────────────────┘  │ │
 │ │                                                                   │ │
 │ │  Safety / Sequencing                                              │ │
 │ │  ┌───────────────────────────────────────────────────────────┐   │ │
 │ │  │  _recipe  12-step FSM  ·  _interlock  IEC 61508  ·        │   │ │
 │ │  │  _motion  P-ctrl       ·  _state_machine                  │   │ │
 │ │  └───────────────────────────────────────────────────────────┘   │ │
 │ └───────────────────────────────┬───────────────────────────────────┘ │
 │                                 │ integrate                            │
 │ ┌─ DiscoMachine  (unified twin) ▼──────────────────────────────────┐  │
 │ │                                                                   │  │
 │ │   tick()  ──── every 10 µs ────────────────────────────────►    │  │
 │ │    │                                                              │  │
 │ │    ├─ SpindleState   ω, i_d, i_q  →  T_e [N·m]                  │  │
 │ │    ├─ InverterState  SVPWM sector  →  duty_a/b/c  →  P_sw [W]   │  │
 │ │    ├─ MotionState    err → v_cmd   →  pos_mm  →  encoder counts  │  │
 │ │    └─ E-STOP         overspeed / blade-depth / door  →  latch    │  │
 │ │                                                                   │  │
 │ │   simulate(N) → telemetry dict     get_state() → full snapshot   │  │
 │ └───────────────────────────────────────────────────────────────────┘  │
 │                                                                        │
 │ ┌─ CUDA  sm_89  RTX 4090 ────────┐  ┌─ Rust  PyO3 / maturin ───────┐ │
 │ │  ∂T/∂t = α∇²T + Q/ρc          │  │  SpindleKernel  (FOC)        │ │
 │ │  TILE=16 shared-mem tiling     │  │  nondominated_sort           │ │
 │ │  Neumann BC + blade source Q   │  │  rbf_kernel_matrix           │ │
 │ │  ~150× vs NumPy FD             │  │  IEC 61508 memory-safe       │ │
 │ └────────────────────────────────┘  └──────────────────────────────┘ │
 └────────────────────────────────────────────────────────────────────────┘
           ║                                       ║
           ║ same physics, different language       ║ streams state
           ▼                                       ▼
 ┌───────────────────────────────────┐   ┌────────────────────────────────┐
 │  IEC 61131-3  Structured Text     │   │  Go  Telemetry  :8080          │
 │                                   │   │                                │
 │  DicingController  ◄─ PROGRAM     │   │  GET  /health                  │
 │   ├─ SpindleFB    PMSM + PI-FOC   │   │  POST /simulate  →  JSON       │
 │   ├─ InterlockFB  IEC 61508 SIL-1 │   │  GET  /ws        →  10 Hz WS   │
 │   └─ RecipeSeqFB  CASE state FSM  │   │                                │
 │                                   │   │  Go → subprocess → C++ .so    │
 │  OpenPLC  ·  CODESYS  ·  TwinCAT  │   └────────────────────────────────┘
 └───────────────────────────────────┘
```

### DiscoMachine control loop

```
     Python call                C++ tick()  (10 µs / cycle)
    ─────────────               ─────────────────────────────────────────
    sim.set_target()  ──────►  [1] Speed error → i_q_ref  (FOC outer)
    sim.simulate(N)            [2] PI current ctrl  v_d, v_q
                               [3] dq Euler:  Δi_d, Δi_q, Δω  (PMSM)
                               [4] Clarke + SVPWM → duty_a/b/c, sector
                               [5] Dead-time comp + P_sw accumulation
                               [6] P-ctrl  err_xyz → v_cmd → Δpos
                               [7] 4× quadrature → encoder counts
                               [8] E-STOP check (6 sensors)
                               [9] Telemetry ring buffer append
    sim.get_state()   ◄──────  {ω, i_q, T_e, x, y, z, duties, mode}
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
| `tests/` | Python | **673 tests** — physics invariants + C++/Python parity + integration |

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
| **機械エンジニア** | FEM, stiffness, thermal PDE, 5-axis motion, vibration, warpage | `_frame_stiffness_kernel`, `_heat_diffusion_kernel.cu`, `_5axis_interpolation`, `grinding_warpage_2d/3d.py`, `stage_vibration_modal.py`, `dicing_blade_2d/3d.py`, `stage_cad_geometry.py` |
| **電気・制御エンジニア** | PMSM, FOC, SVPWM, PLC (IEC 61131-3) | `SpindleFB.st`, `_spindle_kernel`, `_servo_inverter_kernel`, `_encoder_kernel` |
| **プロセス研究** | APC, GP regression, Bayesian opt, EnKF | `_enkf_kernel`, `_gp_inference_kernel`, `_mpc_kernel`, `hybrid_process_gp.py`, `surrogate_grinding.py` |
| **R&D（先行研究）** | Laser/plasma physics, TAIKO®/KABRA digital twin, GaN, die strength | `kabra_thermal_2d.py`, `laser_groove_thermal_2d.py`, `plasma_bosch_model.py`, `gan_dicing_model.py`, `taiko_grinding_gp.py`, `bayesian_opt_grinding.py`, `die_strength_gp.py`, `hybrid_bonding_model.py` |
| **ソフトウェア開発** | C++, embedded, real-time, Python, Rust, Go | `machine/`, `pipeline/` 14 C++ kernels, `rust/` PyO3, `telemetry/` WebSocket |
| **生産技術** | Tolerance, yield, multi-objective opt | `analysis/tolerance_stack.py`, `_nsga2_kernel`, `validation/` |

---

---

## 応用情報技術者試験 (AP試験) カバレッジ

| # | AP試験 分野 | 重要度 | 実装ファイル | 具体的なトピック |
|---|------------|:------:|-------------|----------------|
| 1 | **組み込みシステム** | ★★★★★ | `embedded/motor_ctrl_sm.cpp` | PMSM FOC, 10kHz ISR, デジタル制御 |
| 2 | **割り込み・DMA** | ★★★★★ | `embedded/rtos_scheduler.cpp` | 割り込み駆動タスク起動, コンテキストスイッチ |
| 3 | **RTOS・タスク管理** | ★★★★★ | `embedded/rtos_scheduler.cpp` | Rate Monotonic, RM利用率上限 n(2^(1/n)−1), 応答時間解析 |
| 4 | **状態機械** | ★★★★★ | `pipeline/_state_machine.cpp`, `embedded/motor_ctrl_sm.cpp` | FSM, IDLE/CUTTING/FAULT遷移, IEC 61508 |
| 5 | **TCP/IP・ネットワーク (L3/L4)** | ★★★★ | `comms/ip_packet_kernel.cpp` | IPv4ヘッダ, インターネットチェックサム, TCP 3ウェイハンドシェイク, 状態遷移 |
| 6 | **ネットワーク (L2・産業プロトコル)** | ★★★★ | `comms/ethercat_frame_kernel.cpp`, `comms/modbus_rtu_kernel.cpp` | EtherCAT (IEC 61158), Modbus RTU, L2フレーム |
| 7 | **セキュリティ** | ★★★★ | `comms/hmac_auth.cpp` | SHA-256 (FIPS 180-4), HMAC-SHA256 (RFC 2104), チャレンジ・レスポンス, IEC 62443 OT |
| 8 | **情報理論・誤り検出** | ★★★★ | `comms/modbus_rtu_kernel.cpp` | CRC-16/IBM (多項式 x¹⁶+x¹⁵+x²+1), エラー検出率 |
| 9 | **アルゴリズム・データ構造** | ★★★★ | `optimization/_nsga2_kernel.cpp`, `ml/_enkf_kernel.cpp` | NSGA-II非支配ソート O(MN²), EnKFアンサンブル更新 |
| 10 | **品質管理 (SPC・Cpk)** | ★★★★ | `pipeline/spc_monitor.cpp` | X̄-R管理図, WECO 8ルール, Cp/Cpk/EWMA |
| 11 | **OS・メモリ管理** | ★★★ | `embedded/memory_pool.cpp` | 固定サイズプール O(1), スタックアロケータ, 断片化ゼロ, malloc禁止の理由 |
| 12 | **信頼性・RASIS** | ★★★ | `machine/reliability_kernel.cpp` | MTBF/MTTR/稼働率, 直列/並列システム, Weibull, OEE |
| 13 | **データベース (SQL)** | ★★★ | `data/process_db.py` | 3NF正規化, SELECT/JOIN/GROUP BY/VIEW, SQLite ACID |
| 14 | **論理設計** | ★★★ | `embedded/motor_ctrl_sm.cpp` | 状態符号化, フリップフロップ相当の遷移論理 |
| 15 | **UML・設計手法** | ★★ | Architecture 節 / `pipeline/_state_machine.cpp` | ステートマシン図, コンポーネント図相当 |

> AP試験全テクノロジ系主要分野を **DISCO DFL7160 ダイシングソーのデジタルツイン** に直接マッピング。  
> OSI L2 (EtherCAT/Modbus) → L3/L4 (IPv4/TCP) → L7 (HMAC認証) の全層実装。  
> ソフトウェア職種の技術面接では「応用情報で学んだ○○を実装した経験」として提示できる。

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
python -m pytest tests/ -q       # 673 tests, ~90 s
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
├── tests/              673 pytest tests
├── benchmark_all_kernels.py
└── build_all_kernels.sh
```

---

## License

MIT — see [LICENSE](LICENSE).
