"""
Benchmark: scalar Python loop vs NumPy-vectorized vs numba-JIT for the
stealth-dicing hot loop.

This quantifies the "move only the hot loop to native code" argument: the
orchestration stays in Python, but evaluating the analytical model over a
parameter grid drops from a per-point Python/`math` loop to a single
C-speed array op (NumPy) — or a compiled kernel (numba), if installed.

Run:  python fem/bench_stealth_dicing.py [--n 200000]
"""

import argparse
import os
import sys
import time

import numpy as np

# Allow running directly (`python fem/bench_stealth_dicing.py`) by putting the
# repo root on the path so the `fem` package resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fem.laser_groove_thermal_2d import DEFAULT, stealth_dicing
from fem.laser_groove_vectorized import (
    _NUMBA_AVAILABLE,
    stealth_dicing_vec,
    stealth_dicing_chipping_fast,
)


def _make_grid(n):
    """n quasi-random points over a realistic process window."""
    rng = np.random.default_rng(0)
    P = rng.uniform(1.0, 30.0, n)
    V = rng.uniform(50.0, 500.0, n)
    D = rng.uniform(30.0, 300.0, n)
    return P, V, D


def _time(fn, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000,
                    help="number of parameter points to evaluate")
    args = ap.parse_args()
    n = args.n

    P, V, D = _make_grid(n)

    # ── 1. Scalar Python loop (the status quo inside sweeps / the optimizer) ──
    def scalar_loop():
        out = np.empty(n)
        for i in range(n):
            r = stealth_dicing(dict(
                DEFAULT,
                laser_power_W=P[i], scan_speed_mm_s=V[i], focal_depth_um=D[i],
                pulse_regime="ps", numerical_aperture=0.65, n_layers=2,
                n_passes=2,
            ))
            out[i] = r["surface_chipping_um"]
        return out

    # ── 2. NumPy vectorized (C under the hood, zero new dependencies) ────────
    def numpy_vec():
        return stealth_dicing_vec(
            laser_power_W=P, scan_speed_mm_s=V, focal_depth_um=D,
        )["surface_chipping_um"]

    # Scalar loop is expensive; use fewer repeats for large n.
    scalar_repeat = 1 if n > 50_000 else 3
    t_scalar = _time(scalar_loop, repeat=scalar_repeat)
    t_numpy = _time(numpy_vec)

    print(f"\nStealth-dicing hot-loop benchmark  (n = {n:,} points)")
    print("─" * 56)
    print(f"  scalar Python loop : {t_scalar*1e3:10.2f} ms   "
          f"({n/t_scalar:,.0f} pts/s)   1.0x")
    print(f"  numpy vectorized   : {t_numpy*1e3:10.2f} ms   "
          f"({n/t_numpy:,.0f} pts/s)   {t_scalar/t_numpy:,.1f}x")

    if _NUMBA_AVAILABLE:
        # Warm up the JIT (first call compiles), then time.
        stealth_dicing_chipping_fast(
            laser_power_W=P[:8], scan_speed_mm_s=V[:8], focal_depth_um=D[:8])

        def numba_fast():
            return stealth_dicing_chipping_fast(
                laser_power_W=P, scan_speed_mm_s=V, focal_depth_um=D)

        t_numba = _time(numba_fast)
        print(f"  numba @njit        : {t_numba*1e3:10.2f} ms   "
              f"({n/t_numba:,.0f} pts/s)   {t_scalar/t_numba:,.1f}x")
    else:
        print("  numba @njit        :  (not installed — pip install numba)")

    print("─" * 56)

    # Sanity: vectorized result must match the scalar loop (small subset).
    ref = scalar_loop_subset(P, V, D, k=256)
    got = stealth_dicing_vec(
        laser_power_W=P[:256], scan_speed_mm_s=V[:256], focal_depth_um=D[:256],
    )["surface_chipping_um"]
    max_err = float(np.max(np.abs(ref - got)))
    print(f"  parity check (256 pts): max abs error = {max_err:.2e} µm")


def scalar_loop_subset(P, V, D, k):
    out = np.empty(k)
    for i in range(k):
        out[i] = stealth_dicing(dict(
            DEFAULT, laser_power_W=P[i], scan_speed_mm_s=V[i],
            focal_depth_um=D[i], pulse_regime="ps", numerical_aperture=0.65,
            n_layers=2, n_passes=2,
        ))["surface_chipping_um"]
    return out


if __name__ == "__main__":
    main()
