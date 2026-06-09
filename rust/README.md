# Rust kernels — wafer-proc-sim

Reimplements the C++ pybind11 hot loops in Rust (PyO3) for memory safety
and modern systems-programming demonstration.

## What's here

| Symbol | Description |
|--------|-------------|
| `SpindleKernel` | PMSM + FOC integration — same physics as `fem/_spindle_kernel.cpp` |
| `nondominated_sort` | NSGA-II Pareto sort — same algorithm as `optimization/_nsga2_kernel.cpp` |
| `rbf_kernel_matrix` | Symmetric RBF kernel — same formula as `ml/_gp_inference_kernel.cpp` |

## Build & install

```bash
# 1. Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 2. Install maturin (Rust→Python bridge, analogous to pybind11 for C++)
pip install maturin

# 3. Build and install into current venv
cd rust
maturin develop --release

# 4. Test
python -c "from wafer_proc_sim import SpindleKernel; k = SpindleKernel(); print(k.tick(30000, 10))"
```

## Architecture note

The Rust kernel uses **PyO3** (the Rust equivalent of pybind11):

```
Python layer (tests, notebooks)
        │
        │  PyO3 FFI  (same role as pybind11 in C++ kernels)
        ▼
Rust kernel (wafer_proc_sim.so)
  ├── SpindleKernel::tick()       — zero-copy, no GIL
  ├── nondominated_sort()         — pure safe Rust, no unsafe blocks
  └── rbf_kernel_matrix()         — auto-vectorised by rustc/LLVM
```

Rust guarantees no null-pointer dereferences, no data races, and no
memory leaks at compile time — properties relevant for embedded control
software where safety certification (IEC 61508) is required.
