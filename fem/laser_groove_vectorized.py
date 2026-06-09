"""
Vectorized hot-loop kernels for the laser stealth-dicing analytical model.

WHY THIS FILE EXISTS
────────────────────
The scalar `stealth_dicing()` in `laser_groove_thermal_2d.py` is a clean,
readable reference — but it operates on ONE parameter set per call using the
`math` module. The genuine hot loop in this project is calling that scalar
function thousands of times inside:

    • parametric sweeps (power × speed × focal-depth grids)
    • the Bayesian optimizer (each acquisition step re-evaluates the model)
    • the matplotlib sweep panels in `laser_groove_thermal_2d.py`

That per-point Python-call + `math.*` overhead — not the arithmetic — is the
bottleneck. This is the textbook "keep the orchestration in Python, move only
the hot loop to native code" pattern:

    stealth_dicing_vec()      → NumPy vectorized (C under the hood, 0 new deps)
    stealth_dicing_numba()    → numba @njit compiled kernel (optional, graceful
                                fallback to the NumPy path if numba is absent)

Both are numerically identical to the scalar reference (see
`tests/test_laser_vectorized.py`); only the execution strategy changes.

Run the benchmark:  python fem/bench_stealth_dicing.py
"""

from __future__ import annotations

import numpy as np

# Reuse the single source of truth for regime physics so the kernels can never
# drift from the scalar reference.
from fem.laser_groove_thermal_2d import LASER_REGIMES

# Physical constants (identical to the scalar `stealth_dicing`)
_WAVELENGTH_UM = 1.064          # Nd:YAG IR [µm]
_N_SIC = 2.65                   # SiC refractive index @ 1064 nm
_R_FOCAL_FLOOR_UM = 0.15        # physical diffraction floor ~150 nm
_N_PHOTON = 3                   # 3-photon MPI process (SiC bandgap 3.26 eV)


def stealth_dicing_vec(
    laser_power_W,
    scan_speed_mm_s,
    pulse_freq_kHz=100.0,
    n_passes=2,
    focal_depth_um=100.0,
    numerical_aperture=0.65,
    n_layers=2,
    pulse_regime="ps",
):
    """
    Vectorized stealth-dicing model. Every argument may be a scalar or an
    array-like; standard NumPy broadcasting applies. Returns a dict of arrays
    matching the keys of the scalar `stealth_dicing()` (minus the bookkeeping
    strings), evaluated element-wise.

    This is a drop-in replacement for the inner ``for`` loop of a parameter
    sweep: build the grids once, call this once, get the whole field back.
    """
    rp = LASER_REGIMES.get(pulse_regime, LASER_REGIMES["ps"])
    F_th_J_cm2 = rp["F_th_J_cm2"]
    HAZ_factor = rp["HAZ_factor"]

    P = np.asarray(laser_power_W, dtype=np.float64)
    v = np.asarray(scan_speed_mm_s, dtype=np.float64)
    f = np.asarray(pulse_freq_kHz, dtype=np.float64) * 1e3
    n_p = np.asarray(n_passes, dtype=np.float64)
    focal_z = np.asarray(focal_depth_um, dtype=np.float64)
    NA = np.asarray(numerical_aperture, dtype=np.float64)
    n_lay = np.asarray(n_layers, dtype=np.float64)

    # Diffraction-limited focal spot (Abbe, immersion-corrected) with floor.
    r_focal_um = np.maximum(0.61 * _WAVELENGTH_UM / (NA * _N_SIC), _R_FOCAL_FLOOR_UM)

    # Rayleigh length (confocal parameter / 2).
    z_R_um = np.pi * r_focal_um ** 2 / _WAVELENGTH_UM * _N_SIC

    # Peak fluence at focal plane [J/cm²].
    E_pulse_J = P / f
    r_cm = r_focal_um * 1e-4
    F_peak = E_pulse_J / (np.pi * r_cm ** 2)

    # MPI threshold: F_th scaled for the 3-photon process.
    F_th_MPI = F_th_J_cm2 * (4.0 ** _N_PHOTON) ** (1.0 / _N_PHOTON)

    valid = F_peak > F_th_MPI
    # F_peak / F_th_MPI is always finite & positive; where invalid we substitute
    # a placeholder > 1 so the log is safe, then mask those points to 0 below.
    ratio = np.where(valid, F_peak / F_th_MPI, 2.0)
    excess = np.log(np.maximum(ratio, 1.0 + 1e-9))

    pulse_pitch_um = v / f * 1e3
    overlap = np.maximum(0.05, 1.0 - pulse_pitch_um / (2.0 * r_focal_um))

    mod_h = z_R_um * 2.0 * np.sqrt(excess) * overlap * n_lay * n_p
    mod_h = np.minimum(mod_h, focal_z * 0.6)
    mod_h = np.where(valid, mod_h, 0.0)

    mod_w = 2.0 * r_focal_um * np.sqrt(excess) * (n_lay ** 0.25) * (n_p ** 0.2)
    mod_w = np.where(valid, mod_w, 0.0)

    # Crack angle: near-vertical for deep focus, shallower for shallow focus.
    m = np.minimum(1.0, focal_z / 100.0)
    crack_angle_deg = 75.0 * m + 45.0 * (1.0 - m)
    ca = np.maximum(crack_angle_deg, 30.0)

    chipping = np.minimum(mod_w * 0.25 / np.tan(np.radians(ca)), mod_w)
    chipping = np.where(mod_h > 0, chipping, 0.0)

    HAZ_depth_um = HAZ_factor * 0.05 * mod_h

    # Broadcast scalar-derived fields up to the output shape for a uniform dict.
    out_shape = mod_h.shape
    return {
        "modified_layer_h_um": mod_h,
        "modified_layer_w_um": mod_w,
        "surface_chipping_um": chipping,
        "HAZ_depth_um": HAZ_depth_um,
        "F_peak_J_cm2": np.broadcast_to(F_peak, out_shape).copy(),
        "r_focal_um": np.broadcast_to(r_focal_um, out_shape).copy(),
        "z_R_um": np.broadcast_to(z_R_um, out_shape).copy(),
    }


# ── Optional numba JIT kernel ────────────────────────────────────────────────
# This is the literal "drop the hot loop to compiled code" path. If numba is
# not installed the public wrapper transparently falls back to the NumPy
# vectorized kernel above — callers never need to know which path ran.
try:
    from numba import njit, prange

    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when numba absent
    _NUMBA_AVAILABLE = False


# ── Optional compiled C++ kernel (pybind11) ──────────────────────────────────
# The fastest / most "production-shaped" path: a real C++ kernel crossing the
# Python↔C++ FFI boundary the way DISCO's on-tool software stack is layered
# (Python orchestration on top, compiled C++ underneath). Build with
# `bash fem/build_cpp_kernel.sh`; if the compiled module is absent the wrapper
# falls back to numba, then NumPy — callers never need to know which ran.
try:
    from fem import _stealth_kernel as _cpp_kernel  # compiled .so, git-ignored

    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False


if _NUMBA_AVAILABLE:

    @njit(parallel=True, fastmath=True, cache=True)
    def _stealth_chipping_kernel(P, v, f_hz, n_p, focal_z, NA, n_lay,
                                 F_th_MPI):  # pragma: no cover - compiled
        """Compiled element-wise kernel returning surface chipping [µm]."""
        n = P.shape[0]
        out = np.empty(n, dtype=np.float64)
        wl = 1.064
        n_sic = 2.65
        for i in prange(n):
            r_focal = 0.61 * wl / (NA[i] * n_sic)
            if r_focal < 0.15:
                r_focal = 0.15
            z_R = np.pi * r_focal * r_focal / wl * n_sic
            E_pulse = P[i] / f_hz[i]
            r_cm = r_focal * 1e-4
            F_peak = E_pulse / (np.pi * r_cm * r_cm)
            if F_peak <= F_th_MPI:
                out[i] = 0.0
                continue
            excess = np.log(F_peak / F_th_MPI)
            pulse_pitch = v[i] / f_hz[i] * 1e3
            overlap = 1.0 - pulse_pitch / (2.0 * r_focal)
            if overlap < 0.05:
                overlap = 0.05
            mod_h = z_R * 2.0 * np.sqrt(excess) * overlap * n_lay[i] * n_p[i]
            cap = focal_z[i] * 0.6
            if mod_h > cap:
                mod_h = cap
            if mod_h <= 0.0:
                out[i] = 0.0
                continue
            mod_w = (2.0 * r_focal * np.sqrt(excess)
                     * n_lay[i] ** 0.25 * n_p[i] ** 0.2)
            mm = focal_z[i] / 100.0
            if mm > 1.0:
                mm = 1.0
            crack = 75.0 * mm + 45.0 * (1.0 - mm)
            if crack < 30.0:
                crack = 30.0
            chip = mod_w * 0.25 / np.tan(np.radians(crack))
            if chip > mod_w:
                chip = mod_w
            out[i] = chip
        return out


def stealth_dicing_chipping_fast(
    laser_power_W,
    scan_speed_mm_s,
    pulse_freq_kHz=100.0,
    n_passes=2,
    focal_depth_um=100.0,
    numerical_aperture=0.65,
    n_layers=2,
    pulse_regime="ps",
):
    """
    Fastest available path for the most-swept output: surface chipping [µm].

    Kernel preference: compiled C++ (pybind11) → numba @njit → NumPy
    vectorized. All three are numerically identical; only the backend differs.
    Inputs are broadcast to a common shape.
    """
    rp = LASER_REGIMES.get(pulse_regime, LASER_REGIMES["ps"])
    F_th_MPI = rp["F_th_J_cm2"] * (4.0 ** _N_PHOTON) ** (1.0 / _N_PHOTON)

    arrs = np.broadcast_arrays(
        np.asarray(laser_power_W, np.float64),
        np.asarray(scan_speed_mm_s, np.float64),
        np.asarray(pulse_freq_kHz, np.float64) * 1e3,
        np.asarray(n_passes, np.float64),
        np.asarray(focal_depth_um, np.float64),
        np.asarray(numerical_aperture, np.float64),
        np.asarray(n_layers, np.float64),
    )
    shape = arrs[0].shape
    flat = [np.ascontiguousarray(a).ravel() for a in arrs]

    if _CPP_AVAILABLE:
        P, v, f_hz, n_p, focal_z, NA, n_lay = flat
        out = _cpp_kernel.stealth_chipping(
            P, v, f_hz, n_p, focal_z, NA, n_lay, F_th_MPI)
        return out.reshape(shape)

    if _NUMBA_AVAILABLE:
        out = _stealth_chipping_kernel(*flat, F_th_MPI)
        return out.reshape(shape)

    return stealth_dicing_vec(
        laser_power_W, scan_speed_mm_s, pulse_freq_kHz, n_passes,
        focal_depth_um, numerical_aperture, n_layers, pulse_regime,
    )["surface_chipping_um"]
