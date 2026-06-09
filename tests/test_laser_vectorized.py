"""
Parity tests: the vectorized / numba kernels must be numerically identical to
the scalar reference `stealth_dicing()`. The optimization only changes HOW the
model runs, never WHAT it computes.
"""

import numpy as np
import pytest

from fem.laser_groove_thermal_2d import DEFAULT, stealth_dicing
from fem.laser_groove_vectorized import (
    stealth_dicing_vec,
    stealth_dicing_chipping_fast,
)

# Output keys produced by both the scalar and vectorized kernels.
_SHARED_KEYS = [
    "modified_layer_h_um",
    "modified_layer_w_um",
    "surface_chipping_um",
    "HAZ_depth_um",
    "F_peak_J_cm2",
    "r_focal_um",
    "z_R_um",
]


def _grid():
    """A representative power × speed × focal-depth grid, incl. sub-threshold."""
    powers = np.linspace(1.0, 30.0, 11)
    speeds = np.linspace(50.0, 500.0, 7)
    depths = np.linspace(30.0, 300.0, 5)
    P, V, D = np.meshgrid(powers, speeds, depths, indexing="ij")
    return P.ravel(), V.ravel(), D.ravel()


def test_vectorized_matches_scalar():
    P, V, D = _grid()

    vec = stealth_dicing_vec(
        laser_power_W=P,
        scan_speed_mm_s=V,
        pulse_freq_kHz=DEFAULT["pulse_freq_kHz"],
        n_passes=2,
        focal_depth_um=D,
        numerical_aperture=0.65,
        n_layers=2,
        pulse_regime="ps",
    )

    for i in range(P.size):
        p = dict(
            DEFAULT,
            laser_power_W=float(P[i]),
            scan_speed_mm_s=float(V[i]),
            n_passes=2,
            focal_depth_um=float(D[i]),
            pulse_regime="ps",
            numerical_aperture=0.65,
            n_layers=2,
        )
        ref = stealth_dicing(p)
        for k in _SHARED_KEYS:
            # Scalar rounds to 3–4 dp for reporting; compare on that tolerance.
            assert vec[k][i] == pytest.approx(ref[k], abs=2e-3), (
                f"mismatch at i={i}, key={k}: vec={vec[k][i]} ref={ref[k]}"
            )


def test_cpp_kernel_matches_scalar_if_built():
    """If the pybind11 C++ kernel is compiled, it must match the scalar ref."""
    pytest.importorskip("fem._stealth_kernel")
    from fem._stealth_kernel import stealth_chipping

    P, V, D = _grid()
    F_th_MPI = (
        __import__("fem.laser_groove_thermal_2d", fromlist=["LASER_REGIMES"])
        .LASER_REGIMES["ps"]["F_th_J_cm2"] * (4.0 ** 3) ** (1.0 / 3.0)
    )
    f_hz = np.full_like(P, DEFAULT["pulse_freq_kHz"] * 1e3)
    n_p = np.full_like(P, 2.0)
    NA = np.full_like(P, 0.65)
    n_lay = np.full_like(P, 2.0)
    cpp = stealth_chipping(P, V, f_hz, n_p, D, NA, n_lay, F_th_MPI)

    for i in range(P.size):
        ref = stealth_dicing(dict(
            DEFAULT, laser_power_W=float(P[i]), scan_speed_mm_s=float(V[i]),
            focal_depth_um=float(D[i]), pulse_regime="ps",
            numerical_aperture=0.65, n_layers=2, n_passes=2,
        ))
        assert cpp[i] == pytest.approx(ref["surface_chipping_um"], abs=2e-3)


def test_fast_path_matches_vectorized_chipping():
    P, V, D = _grid()
    fast = stealth_dicing_chipping_fast(
        laser_power_W=P, scan_speed_mm_s=V, focal_depth_um=D,
        pulse_freq_kHz=DEFAULT["pulse_freq_kHz"],
    )
    vec = stealth_dicing_vec(
        laser_power_W=P, scan_speed_mm_s=V, focal_depth_um=D,
        pulse_freq_kHz=DEFAULT["pulse_freq_kHz"],
    )["surface_chipping_um"]
    np.testing.assert_allclose(fast, vec, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("regime", ["ns", "ps", "fs"])
def test_all_regimes_parity(regime):
    P, V, D = _grid()
    vec = stealth_dicing_vec(
        laser_power_W=P, scan_speed_mm_s=V, focal_depth_um=D,
        pulse_freq_kHz=DEFAULT["pulse_freq_kHz"], pulse_regime=regime,
    )
    # Spot-check a handful of points against the scalar reference per regime.
    for i in (0, P.size // 3, P.size // 2, P.size - 1):
        ref = stealth_dicing(dict(
            DEFAULT, laser_power_W=float(P[i]), scan_speed_mm_s=float(V[i]),
            focal_depth_um=float(D[i]), pulse_regime=regime,
            numerical_aperture=0.65, n_layers=2, n_passes=2,
        ))
        assert vec["surface_chipping_um"][i] == pytest.approx(
            ref["surface_chipping_um"], abs=2e-3
        )
