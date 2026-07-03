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

## CLI reference

`gen_synthetic <out.pgm> [--width-um W] [--chip-um D] [--noise S] [--seed N]`

`kerf_inspect <in.pgm> [--um-per-px U] [--nominal-um N] [--tol-um T]
[--max-chip-um C] [--blur S] [--annotate out.ppm] [--quiet]`
