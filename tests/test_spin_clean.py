"""Tests for cleaning/spin_clean — single-wafer wet cleaning physics."""
import numpy as np
import pytest

sc = pytest.importorskip("cleaning.spin_clean")


def test_force_scaling_laws():
    """Adhesion ∝ d, drag ∝ d² — the reason small particles are hard."""
    f1 = sc.adhesion_force(30e-9)
    f2 = sc.adhesion_force(60e-9)
    assert np.isclose(f2 / f1, 2.0, rtol=1e-9)
    tau = sc.wall_shear_spin()
    d1 = sc.drag_force(30e-9, tau)
    d2 = sc.drag_force(60e-9, tau)
    assert np.isclose(d2 / d1, 4.0, rtol=1e-9)


def test_pre_monotonic_and_megasonic_helps_small_particles():
    tau = sc.wall_shear_spin()
    spin = sc.removal_efficiency([30.0, 60.0, 100.0, 150.0], tau)
    assert np.all(np.diff(spin) > 0)
    mega30 = sc.removal_efficiency([30.0], tau, megasonic=True)[0]
    assert mega30 > spin[0] + 0.5          # transformative, not incremental


def test_full_undercut_releases_particle():
    tau = sc.wall_shear_spin()
    pre = sc.removal_efficiency([20.0], tau,
                                etch_nm=sc.A_CONTACT * 1e9 + 0.1)[0]
    assert pre == 1.0


def test_collapse_ar_ordering_and_quarter_power_law():
    ar_w = sc.max_aspect_ratio(sc.SIGMA_WATER)
    ar_i = sc.max_aspect_ratio(sc.SIGMA_IPA)
    ar_c = sc.max_aspect_ratio(sc.SIGMA_SCCO2)
    assert ar_c > ar_i > ar_w
    # AR_max ∝ σ^(-1/4)
    assert np.isclose(ar_i / ar_w,
                      (sc.SIGMA_WATER / sc.SIGMA_IPA) ** 0.25, rtol=1e-6)


def test_film_thinning_matches_analytic():
    t, ha, hn = sc.film_thinning()
    m = ha > 1e-3
    assert np.max(np.abs(hn[m] - ha[m]) / ha[m]) < 0.02
