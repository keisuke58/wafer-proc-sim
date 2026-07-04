# optomech-lens-sim

**C++17 optomechanical simulation toolkit for camera lens-barrel (鏡筒) design.**
Everything is written from scratch — the paraxial optics, the linear algebra, the
control loops and the shock-response integrator — with no external dependencies.
Each module has its own tests (`ctest`), is compiler-warning-free
(`-Wall -Wextra -Wpedantic`) and `cppcheck`-clean.

**Live dashboard:** https://keisuke58.github.io/wafer-proc-sim/optomech.html

The barrel is the structural spine of a camera lens: it holds each element to a
mechanical tolerance, moves the focus/zoom groups, survives drops, and (with OIS)
cancels hand shake. This toolkit models the four questions that decide that
design, and reports numbers a designer can act on.

| # | Module (`optm::…`) | What it computes |
|---|---|---|
| ① | **`tol`** (`tolerance.hpp`) | Optomechanical **tolerance stack-up**: propagates per-element decenter / tilt / spacing tolerances to the image plane by **Monte Carlo + RSS**, and ranks the **critical tolerance**. |
| ② | **`cam`** (`cam.hpp`) | **Autofocus mechanism**: a cycloidal **cam-barrel** follower law + a **voice-coil-motor** actuator under **PID + feedforward** control (settling / overshoot / steady-state error, with force saturation & anti-windup). |
| ③ | **`fea`** (`fea.hpp`) | **Barrel structural analysis**: thermal focus drift (CTE), first bending resonance (cantilever tube + tip lens mass), and **drop-shock** stress via a half-sine **shock-response** integration (DAF) → safety factor. |
| ④ | **`ois`** (`ois.hpp`) | **Optical image stabilization**: synthesizes multi-tone hand tremor, drives a 2nd-order shift actuator to cancel it, and reports the residual image motion in photographic **stops**. |

All four are built on a shared paraxial ray-transfer (**ABCD**) optical core
(`optics.hpp`).

## Results (from `apps/optm_demo`, this host)

- **Tolerance** — 4-element f/… lens, EFL ≈ 33 mm. Monte-Carlo image boresight
  **P99 = 15.3 µm** against a **20 µm** budget → **PASS**; the dominant
  contributor is **element-2 tilt (34 % of variance)** — the tolerance to tighten
  first.
- **Autofocus** — a 300 µm focus move settles in **3.9 ms** with model-based
  feedforward and **~0 µm** steady-state error; without feedforward the coil
  cannot fully overcome the spring, leaving a **23 µm** focus error.
- **Barrel FEA** — Al 6061 barrel: thermal drift **0.94 µm/°C**, first bending
  mode **10.6 kHz**, and a 1 m drop (≈1400 g, DAF 1.1) gives **4.2 MPa** bending
  stress → **safety factor 65**.
- **OIS** — a 35 mm lens: hand-shake image motion **83 µm RMS** falls to
  **8.9 µm RMS** with stabilization — **3.2 stops**.

## Build & run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure     # tolerance / cam / fea / ois

# Emit metrics as JSON + the three dashboard figures:
./build/optm_demo --tol tol.ppm --cam cam.ppm --ois ois.ppm > metrics.json
```

## Layout

```
include/optomech/   optics, tolerance, cam, fea, ois, ppm   (headers)
src/                implementations
apps/optm_demo.cpp  runs all modules, writes JSON + figures
tests/              one ctest per module
docs/               self-contained GitHub Pages dashboard + assets
```

## Notes on the physics

- **Optics** — thin-lens ABCD matrices; EFL/BFD from the system matrix; lateral
  image sensitivities from the downstream matrix × element power (decenter) and
  the downstream `B` term (tilt).
- **Tolerance** — RSS is the analytic first-order 2-D radial budget; Monte Carlo
  samples all sources together (Rayleigh-distributed image error) for the true
  P99.
- **Cam/VCM** — cycloidal motion law (finite acceleration everywhere); a
  saturating coil with anti-windup integral control (real VCMs are force-limited).
- **FEA** — closed-form CTE / cantilever-beam models; the drop-shock DAF is the
  maximax response of an undamped SDOF to a half-sine base pulse, integrated
  numerically.
- **OIS** — the actuator is a 2nd-order servo whose finite bandwidth sets how much
  high-frequency tremor leaks through.

## Finite-element barrel model (`fem/barrel_fem.py`, Python)

A genuine mesh-based **Euler-Bernoulli beam FE** model of the barrel (2 DOF/node),
complementing the closed-form C++ `fea` module. It assembles element
stiffness/consistent-mass matrices and runs three analyses a lens-barrel
designer actually performs:

- **Modal** — generalized eigenproblem for natural frequencies + mode shapes;
  the first mode (**10,639 Hz**) matches the closed-form cantilever within
  **0.14%**.
- **Thermal-structural coupling** — a radial temperature gradient imposes a
  thermal curvature; the FE tip deflection equals the analytic uniform-curvature
  result (**3.15 µm** per 5 °C gradient).
- **Drop-shock transient** — Newmark-β time integration under a half-sine base
  pulse → peak root bending stress and **safety factor 30.6** (a more realistic,
  transient answer than the closed-form estimate).

```bash
python3 fem/barrel_fem.py    # writes fem/figures/*.png + barrel_fem_results.json
```
Requires `numpy` + `matplotlib` (`scipy` used for the eigensolver if present).

## Generative design — GNN surrogate + Bayesian search (`ai/generative_design.py`, PyTorch)

The "生成AIを活用した設計改革" piece, on the applicant's own research stack
(**PyTorch / graph neural networks + Bayesian optimization**):

1. **CAE ground truth** — a zone-varying barrel is meshed into beam elements; a
   modal + mass + drop-shock evaluation gives the true performance.
2. **PyTorch GNN surrogate** — each design's FE mesh is a graph; a Graph
   Convolutional Network learns "design → performance" almost instantly
   (held-out **R² = 0.99 / 1.00 / 1.00** for first-mode / mass / safety factor).
3. **Bayesian generative search** — a Gaussian process with Expected Improvement
   uses the surrogate to *generate* the lightest barrel meeting the stiffness
   and shock constraints, verified against the real FE solver. The AI generated
   a **Mg-alloy thin-wall + ribbed** design (9.1 kHz, 17.2 g, SF 36).

```bash
pip install torch          # + numpy scipy scikit-learn matplotlib
python3 ai/generative_design.py
```

