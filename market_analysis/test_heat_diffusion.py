"""
Quick smoke-test for _heat_diffusion_kernel.cu
Run on Vancouver (RTX 4090):
    python fem/test_heat_diffusion.py
"""
import sys, os, numpy as np, time

# Support both: `python fem/test_heat_diffusion.py` (from repo root)
#           and: `python test_heat_diffusion.py`     (from fem/)
sys.path.insert(0, os.path.dirname(__file__))

try:
    from _heat_diffusion_kernel import heat_diffusion_solve, gpu_info
    CUDA = True
except ImportError:
    CUDA = False
    print("[!] _heat_diffusion_kernel.so not found — build with:")
    print("    bash build_all_kernels.sh heat_diffusion")

# ── NumPy reference (explicit FD) ─────────────────────────────────────────────

def heat_numpy(T, alpha, dx, dt, n_steps):
    r = alpha * dt / dx**2
    assert r <= 0.25, f"unstable: r={r:.4f}"
    for _ in range(n_steps):
        lap = (np.roll(T, 1, 0) + np.roll(T, -1, 0)
             + np.roll(T, 1, 1) + np.roll(T, -1, 1) - 4 * T)
        T = T + r * lap
    return T

# ── Test parameters ───────────────────────────────────────────────────────────

NY, NX  = 256, 256
# SiC thermal diffusivity: ~84 mm²/s = 84e-6 m²/s
ALPHA   = 84e-6      # m²/s
DX      = 20e-6      # 20 µm mesh
DT      = 0.5 * DX**2 / (4 * ALPHA)   # stability limit * 0.5
NSTEPS  = 1000

# Initial condition: hot spot in centre (Gaussian)
cy, cx = NY // 2, NX // 2
yy, xx = np.ogrid[:NY, :NX]
T0 = 25.0 + 300.0 * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (20**2)).astype(np.float64)

# ── GPU info ──────────────────────────────────────────────────────────────────

if CUDA:
    info = gpu_info()
    print(f"GPU: {info.get('name', 'N/A')}  "
          f"CC={info.get('compute_capability', '?')}  "
          f"VRAM={info.get('global_mem_GB', 0):.1f} GB  "
          f"SMs={info.get('sm_count', '?')}")

# ── Run CUDA ──────────────────────────────────────────────────────────────────

if CUDA:
    t0 = time.perf_counter()
    T_cuda = heat_diffusion_solve(T0, ALPHA, DX, DT, NSTEPS)
    t_cuda = time.perf_counter() - t0
    print(f"CUDA  : {t_cuda*1e3:.2f} ms  ({NSTEPS} steps, {NY}×{NX} grid)")

# ── Run NumPy ─────────────────────────────────────────────────────────────────

t0 = time.perf_counter()
T_np = heat_numpy(T0.copy(), ALPHA, DX, DT, NSTEPS)
t_np = time.perf_counter() - t0
print(f"NumPy : {t_np*1e3:.2f} ms  ({NSTEPS} steps, {NY}×{NX} grid)")

# ── Compare ───────────────────────────────────────────────────────────────────

if CUDA:
    max_err = float(np.max(np.abs(T_cuda - T_np)))
    speedup = t_np / t_cuda
    print(f"Max |CUDA - NumPy| = {max_err:.6f} °C  (expect < 1e-9 for identical FD)")
    print(f"Speedup            = {speedup:.1f}×")
    assert max_err < 1e-6, f"CUDA/NumPy mismatch: {max_err}"
    print("[✓] CUDA kernel matches NumPy reference")
