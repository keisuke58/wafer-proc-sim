# wproc — C++ dicing-kerf image-inspection toolkit

A dependency-free **C++17 image-processing system** for inspecting wafer
dicing streets (kerfs): it measures kerf width to sub-pixel accuracy and detects
chipping defects along the cut walls, then emits a machine-readable report and an
annotated overlay image.

Built from scratch with **no OpenCV / no third-party libraries** — only the C++
standard library and CMake. The image algorithms (Gaussian smoothing, Sobel
gradients, Otsu thresholding, binary morphology, connected-component labeling)
are implemented directly, which keeps the build portable to any machine with a
C++ compiler and demonstrates the underlying computer-vision math end to end.

> **Visual explainer:** a student-friendly, annotated walkthrough of a FAIL-sample
> inspection (numbered callouts, color legend, measurement-vs-spec, and a
> plain-language primer) is published here:
> https://claude.ai/code/artifact/d3ef3e24-f807-4e54-958f-74b70b5598d3
> The same page is committed at [`docs/kerf_annotated.html`](docs/kerf_annotated.html).

## What it does

Given a grayscale scribe-line image (a dark kerf channel on a brighter silicon
substrate), the pipeline:

1. **Denoises** with a separable Gaussian blur.
2. **Locates the two kerf walls** from the signed Sobel-x column projection —
   the strongest negative slope is the left wall, the strongest positive slope
   the right wall — refined to sub-pixel precision by parabolic interpolation.
3. **Measures kerf width** in microns via a calibrated `um_per_px` scale.
4. **Detects chipping** by Otsu-thresholding the dark regions outside the kerf
   band, cleaning specks with morphological opening, and labeling connected
   components; each chip is measured for protrusion depth and area.
5. **Applies spec limits** (nominal width ± tolerance, max chip size) and returns
   a `pass` / `fail` verdict as JSON plus an annotated overlay.

## Architecture

```
include/wproc/         public headers (one module per concern)
  image.hpp            Image (float) / ColorImage (RGB) containers
  io.hpp               PGM/PPM (Netpbm) read & write
  filters.hpp          Gaussian blur, Sobel gradient
  threshold.hpp        Otsu threshold + binarization
  morphology.hpp       erode / dilate / open / close
  connected.hpp        connected-component labeling (union-find)
  draw.hpp             annotation primitives
  pipeline.hpp         extensible Stage / Pipeline abstraction
  kerf_inspect.hpp     domain inspector (walls, chips, spec verdict, JSON)
src/                   implementations
apps/
  gen_synthetic.cpp    synthetic test-image generator (repeatable, no data)
  kerf_inspect_main.cpp CLI front-end
tests/
  test_imgproc.cpp     unit tests (ctest)
```

The design is **extension-first**: new processing steps implement the `Stage`
interface (`pipeline.hpp`) and are appended to a `Pipeline` without touching the
core. Metrology scale and spec limits are all parameters (`KerfInspector::Params`).

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Requires only CMake ≥ 3.16 and a C++17 compiler.

## Run (end to end, no external data)

```bash
# 1. Generate a synthetic scribe-line image (40 µm kerf, small chip)
./build/gen_synthetic kerf.pgm --width-um 40 --chip-um 1.5

# 2. Inspect it against a spec, writing a JSON report + annotated overlay
./build/kerf_inspect kerf.pgm \
    --um-per-px 0.5 --nominal-um 40 --tol-um 5 --max-chip-um 5 \
    --annotate kerf_annotated.ppm
```

Example report:

```json
{
  "status": "ok",
  "found": true,
  "left_wall_px": 159.34,
  "right_wall_px": 240.62,
  "width_um": 40.64,
  "confidence": 1.0,
  "pass": true,
  "max_chip_um": 1.17,
  "chip_count": 1,
  "chips": [ { "id": 1, "side": "left", "protrusion_um": 1.17, "area_um2": 12.5 } ]
}
```

The process exit code is the verdict: **0 = pass**, **1 = fail / no kerf**,
so the tool drops straight into a shell or CI pass/fail gate.

## Inputs / outputs

- **Input:** 8-bit PGM (`P2` ASCII or `P5` binary).
- **Report:** JSON on stdout.
- **Overlay:** PPM (`P6`) — green wall lines; chip boxes colored yellow
  (in-spec) or red (over the max-chip limit), with a marker at each centroid.

## Optional backends (auto-detected by CMake)

All three are optional: the core library stays dependency-free, and each
backend is built only when its toolkit is found.

### Qt6 GUI viewer (`kerf_viewer`)

Interactive inspection: open a PGM, tune the spec parameters live, see the
annotated overlay, a PASS/FAIL verdict, and a per-chip table; save the
annotated PNG and JSON report. Built when Qt6 Widgets is found
(`WPROC_BUILD_GUI`, default ON).

```bash
./build/kerf_viewer kerf.pgm            # interactive
# headless self-test — renders off-screen, saves a screenshot, prints JSON:
QT_QPA_PLATFORM=offscreen ./build/kerf_viewer --selftest shot.png kerf.pgm
```

### OpenCV backend + parity test (`wproc_cv`)

`wproc::cv_backend::{gaussian_blur, sobel, otsu_threshold}` implement the same
operations with OpenCV (`WPROC_WITH_OPENCV`, default ON). The `cv_parity`
ctest cross-validates the from-scratch kernels against OpenCV on a noisy
synthetic kerf scene: Gaussian and Sobel agree to float rounding noise
(max |diff| ~1e-4 on a 0–255 scale) and Otsu picks the identical threshold.

### CUDA backend (`wproc_cuda`)

`wproc::cuda_backend::{gaussian_blur, sobel}` run the separable Gaussian and
3x3 Sobel on the GPU with the same border/radius conventions as the CPU path
(`WPROC_WITH_CUDA`, default ON; compiled only when `check_language(CUDA)`
finds a toolkit — machines without CUDA skip it with a status message).

## Performance, metrology & machine-vision layers

### Real-time filters + benchmark (`bench`)

`wproc::simd::{gaussian_blur, sobel}` add AVX2/FMA (8 floats/lane, scalar
borders) and row-tiled multithreaded variants, bit-parity with the scalar path
(`simd_parity` test). The `bench` app reports throughput and camera frame rate:

```bash
./build/bench --width 2048 --height 2048 --iters 7
```

Measured on a 4-core x86 (2048×2048): Gaussian **46.6 → 697 MPix/s** (15×,
139 fps @ 5 MP); Sobel **167 → 1326 MPix/s** (265 fps @ 5 MP).

### Metrology — repeatability & calibration (`wproc/metrology.hpp`)

- `metro::repeatability(...)` inspects N noisy realizations of a scene and
  reports kerf-width mean, **1-σ spread**, min/max, and CV% (Gage-R&R-style
  precision; measured CV ≈ 0.02 % on a synthetic kerf).
- `metro::estimate_um_per_px(...)` recovers the pixel scale from a periodic
  calibration target via the dominant autocorrelation period of the signed
  Sobel-x projection, making width measurements traceable.

### Machine-vision front-end — skew & multi-street (`wproc/street_detect.hpp`, OpenCV)

- `vision::estimate_skew_deg` / `deskew` — find the street orientation
  (Canny + Hough) and straighten the image; skew round-trips to ~0° after
  deskew.
- `vision::detect_streets` — locate every dark street centre across the field,
  turning the single-kerf inspector into a full-field front-end.

### FPGA — streaming 3×3 Sobel (`fpga/`)

A synthesizable line-buffer Verilog Sobel core (`sobel3x3.v`, one pixel/clock,
zero-padded stream) with a self-checking testbench that compares every output
bit-exact against an independent reference. Run with Icarus Verilog:

```bash
bash fpga/run_sim.sh    # iverilog + vvp; prints "TB PASS"
```

## CLI reference

`gen_synthetic <out.pgm> [--width-um W] [--chip-um D] [--noise S] [--seed N]`

`kerf_inspect <in.pgm> [--um-per-px U] [--nominal-um N] [--tol-um T]
[--max-chip-um C] [--blur S] [--annotate out.ppm] [--quiet]`
