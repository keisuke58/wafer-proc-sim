"""
Tests for taiko/deflection.py — TAIKO® physics-based plate-deflection model.

Checks:
  * analytical limits (h_thin = h_ring → uniform Kirchhoff plate;
    ring width → 0 → uniformly thinned wafer)
  * scaling laws (w ∝ 1/h³ at fixed q, w ∝ a⁴, self-weight w ∝ 1/h²)
  * sign / physical sanity (wafer sags downward, TAIKO® sags less)
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from taiko.deflection import (  # noqa: E402
    H_FULL_M,
    WAFER_RADIUS_M,
    parametric_study,
    solve_taiko_deflection,
    taiko_advantage,
    uniform_disk_max_deflection,
    uniform_disk_max_stress,
    uniform_disk_profile,
)

H_THIN = 50e-6
W_RING = 3e-3


# ── Analytical limits ──────────────────────────────────────────────────────────

class TestAnalyticalLimits:
    def test_uniform_limit_matches_kirchhoff(self):
        """h_thin = h_ring → numerical BVP must reproduce the analytical
        simply supported uniform-disk solution."""
        res = solve_taiko_deflection(H_FULL_M, W_RING, h_ring=H_FULL_M)
        w_ana = uniform_disk_max_deflection(H_FULL_M)
        assert res["w_max"] == pytest.approx(w_ana, rel=1e-3)

    def test_uniform_limit_full_profile(self):
        """Whole w(r) profile must match the analytical solution, not
        just the maximum."""
        res = solve_taiko_deflection(H_FULL_M, W_RING, h_ring=H_FULL_M)
        w_ana = uniform_disk_profile(res["r"], H_FULL_M)
        np.testing.assert_allclose(res["w"], w_ana, rtol=1e-3,
                                   atol=1e-3 * np.max(w_ana))

    def test_ring_width_to_zero_approaches_uniform(self):
        """As the ring narrows, the TAIKO® sag must converge to the
        uniformly thinned wafer (advantage → 1, monotonically)."""
        widths = [3e-3, 1e-3, 0.1e-3, 0.02e-3]
        advs = [taiko_advantage(H_THIN, w) for w in widths]
        # monotone: narrower ring → less benefit
        assert advs == sorted(advs, reverse=True)
        # asymptote: 20 µm ring is nearly no ring
        assert advs[-1] == pytest.approx(1.0, abs=0.2)

    def test_thin_equals_ring_gives_advantage_simply_supported(self):
        """If nothing is ground (h_thin = h_ring) TAIKO® ≡ uniform wafer,
        so the advantage ratio must be ≈ 1."""
        adv = taiko_advantage(H_FULL_M, W_RING, h_ring=H_FULL_M)
        assert adv == pytest.approx(1.0, rel=1e-3)


# ── Scaling laws ───────────────────────────────────────────────────────────────

class TestScaling:
    def test_sign_wafer_sags_downward(self):
        """Positive w convention = sag in gravity direction."""
        assert uniform_disk_max_deflection(H_THIN) > 0
        res = solve_taiko_deflection(H_THIN, W_RING)
        assert res["w_max"] > 0

    def test_w_scales_inverse_h_cubed_fixed_load(self):
        """At fixed pressure q, w ∝ 1/D ∝ 1/h³."""
        q = 1.0
        r = uniform_disk_max_deflection(H_THIN, q=q) / \
            uniform_disk_max_deflection(2 * H_THIN, q=q)
        assert r == pytest.approx(8.0, rel=1e-12)

    def test_w_scales_a_fourth(self):
        """w ∝ a⁴."""
        r = uniform_disk_max_deflection(H_THIN, a=2 * WAFER_RADIUS_M) / \
            uniform_disk_max_deflection(H_THIN, a=WAFER_RADIUS_M)
        assert r == pytest.approx(16.0, rel=1e-12)

    def test_self_weight_scales_inverse_h_squared(self):
        """Self-weight: q = ρgh, so w ∝ h/h³ = 1/h²."""
        r = uniform_disk_max_deflection(H_THIN) / \
            uniform_disk_max_deflection(2 * H_THIN)
        assert r == pytest.approx(4.0, rel=1e-12)

    def test_load_linearity_g_factor(self):
        """Linear theory: deflection scales linearly with g_factor."""
        w1 = solve_taiko_deflection(H_THIN, W_RING, g_factor=1.0)["w_max"]
        w10 = solve_taiko_deflection(H_THIN, W_RING, g_factor=10.0)["w_max"]
        assert w10 / w1 == pytest.approx(10.0, rel=1e-3)


# ── Physical sanity: TAIKO® beats uniform thinning ─────────────────────────────

class TestTaikoPhysics:
    def test_taiko_sags_less_than_uniform(self):
        """Core claim: the edge ring reduces self-weight sag."""
        w_uni = uniform_disk_max_deflection(H_THIN)
        w_tai = solve_taiko_deflection(H_THIN, W_RING)["w_max"]
        assert w_tai < w_uni

    def test_standard_taiko_advantage_near_clamped_limit(self):
        """50 µm + 3 mm ring: the stiff ring acts as a near-built-in
        clamp → advantage close to the simply-supported/clamped ratio
        (5+ν)/(1+ν) ≈ 4.1 for ν = 0.28."""
        adv = taiko_advantage(H_THIN, W_RING)
        assert 3.0 < adv < 5.0

    def test_taiko_reduces_bending_stress(self):
        """The ring also lowers peak bending stress (crack risk)."""
        res = solve_taiko_deflection(H_THIN, W_RING)
        assert res["sigma_max"] < uniform_disk_max_stress(H_THIN)

    def test_thinner_wafer_sags_more(self):
        w30 = solve_taiko_deflection(30e-6, W_RING)["w_max"]
        w100 = solve_taiko_deflection(100e-6, W_RING)["w_max"]
        assert w30 > w100

    def test_wider_ring_sags_less(self):
        w1 = solve_taiko_deflection(H_THIN, 1e-3)["w_max"]
        w5 = solve_taiko_deflection(H_THIN, 5e-3)["w_max"]
        assert w5 < w1

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            solve_taiko_deflection(800e-6, W_RING)   # h_thin > h_ring
        with pytest.raises(ValueError):
            solve_taiko_deflection(H_THIN, -1e-3)    # negative ring width


# ── Parametric study ───────────────────────────────────────────────────────────

class TestParametricStudy:
    def test_map_shapes_and_monotonicity(self):
        study = parametric_study(
            ring_widths=np.array([1e-3, 3e-3, 5e-3]),
            thin_thicknesses=np.array([30e-6, 50e-6, 100e-6]),
        )
        assert study["w_max"].shape == (3, 3)
        assert study["sigma_max"].shape == (3, 3)
        assert np.all(study["w_max"] > 0)
        assert np.all(study["advantage"] > 1.0)
        # sag decreases with thickness (rows) and with ring width (cols)
        assert np.all(np.diff(study["w_max"], axis=0) < 0)
        assert np.all(np.diff(study["w_max"], axis=1) < 0)
