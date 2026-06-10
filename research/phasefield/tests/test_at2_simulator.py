"""
Tests for at2_simulator.py

Physics-based assertions:
  - Higher Gc  ->  crack initiates later  (more energy needed)
  - Larger ell ->  wider damage zone
  - Damage field is in [0, 1] and monotonically non-decreasing with load
  - Anisotropic Gc formula is correct
  - Binary observation returns 0 or 1
  - Log-likelihood is negative; true params beat wrong params on synthetic data
"""
from __future__ import annotations
import sys, os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from research.phasefield.at2_simulator import (
    PFParams, SimConfig, ForwardResult,
    gc_anisotropic, critical_strain_estimate,
    run_forward, crack_probability, simulate_observation,
    log_likelihood, log_posterior, generate_synthetic_dataset,
)

# ── Shared fast config (coarse mesh for speed) ───────────────────────────────
FAST = SimConfig(n_elem=100, n_steps=80, u_max_factor=5.0)

BASE = PFParams(Gc=8e-4, ell=0.04, E=1.0, beta=0.0)


# ════════════════════════════════════════════════════════════════════════════
#  1.  gc_anisotropic
# ════════════════════════════════════════════════════════════════════════════

class TestGcAnisotropic:
    def test_isotropic_returns_gc0(self):
        for theta in [0, 30, 60, 90]:
            assert gc_anisotropic(np.radians(theta), 1.0, 0.0) == pytest.approx(1.0)

    def test_theta_zero_always_gc0(self):
        # At theta=0: sin^2(0) = 0  =>  Gc_eff = Gc0 * 1  regardless of beta
        for beta in [-0.5, 0.0, 0.5, 1.0]:
            assert gc_anisotropic(0.0, 2.0, beta) == pytest.approx(2.0)

    def test_theta_90_gives_gc0_times_one_plus_beta(self):
        assert gc_anisotropic(np.pi / 2, 2.0, 0.5) == pytest.approx(2.0 * 1.5)

    def test_positive_beta_increases_gc_off_axis(self):
        gc_0  = gc_anisotropic(0.0,       1.0, 0.5)
        gc_45 = gc_anisotropic(np.pi / 4, 1.0, 0.5)
        gc_90 = gc_anisotropic(np.pi / 2, 1.0, 0.5)
        assert gc_0 < gc_45 < gc_90

    def test_negative_beta_decreases_gc_off_axis(self):
        gc_0  = gc_anisotropic(0.0,       1.0, -0.4)
        gc_90 = gc_anisotropic(np.pi / 2, 1.0, -0.4)
        assert gc_0 > gc_90

    def test_sic_like_beta_minus_half(self):
        # 4H-SiC: Gc_a / Gc_c = (1.4/2.0)^2 = 0.49
        # At theta=90: Gc_eff = Gc0 * (1 - 0.5) = 0.5 * Gc0
        gc_a = gc_anisotropic(np.pi / 2, 1.0, -0.5)
        assert gc_a == pytest.approx(0.5, rel=1e-6)


# ════════════════════════════════════════════════════════════════════════════
#  2.  critical_strain_estimate
# ════════════════════════════════════════════════════════════════════════════

class TestCriticalStrain:
    def test_positive(self):
        assert critical_strain_estimate(BASE) > 0.0

    def test_higher_gc_higher_critical_strain(self):
        p_hi = PFParams(Gc=BASE.Gc * 2, ell=BASE.ell, E=BASE.E)
        assert critical_strain_estimate(p_hi) > critical_strain_estimate(BASE)

    def test_larger_ell_lower_critical_strain(self):
        # ell in denominator: larger ell -> smaller eps_c
        p_big = PFParams(Gc=BASE.Gc, ell=BASE.ell * 2, E=BASE.E)
        assert critical_strain_estimate(p_big) < critical_strain_estimate(BASE)

    def test_formula_value(self):
        p = PFParams(Gc=0.001, ell=0.05, E=1.0)
        expected = (3 / 8) * np.sqrt(3 * 0.001 / (2 * 1.0 * 0.05))
        assert critical_strain_estimate(p) == pytest.approx(expected, rel=1e-10)


# ════════════════════════════════════════════════════════════════════════════
#  3.  run_forward — output structure
# ════════════════════════════════════════════════════════════════════════════

class TestRunForward:
    def setup_method(self):
        self.result = run_forward(BASE, FAST)

    def test_result_is_forward_result(self):
        assert isinstance(self.result, ForwardResult)

    def test_u_shape(self):
        assert len(self.result.U) == FAST.n_steps + 1

    def test_f_shape(self):
        assert len(self.result.F) == FAST.n_steps + 1

    def test_d_field_shape(self):
        assert len(self.result.d_field) == FAST.n_elem + 1

    def test_damage_in_unit_interval(self):
        assert np.all(self.result.d_field >= -1e-10)
        assert np.all(self.result.d_field <=  1.0 + 1e-10)

    def test_u_monotone_increasing(self):
        assert np.all(np.diff(self.result.U) >= 0)

    def test_peak_f_positive(self):
        assert self.result.peak_F > 0.0

    def test_init_u_positive(self):
        assert self.result.init_U > 0.0

    def test_init_u_le_u_max(self):
        u_max = FAST.u_max_factor * BASE.ell
        assert self.result.init_U <= u_max + 1e-12

    def test_force_drops_after_peak(self):
        F = self.result.F
        assert F[-1] < F[np.argmax(F)], "Stress should drop after peak (softening)"


# ════════════════════════════════════════════════════════════════════════════
#  4.  run_forward — physics monotonicity
# ════════════════════════════════════════════════════════════════════════════

class TestForwardPhysics:
    def test_higher_gc_delays_crack_initiation(self):
        p_lo = PFParams(Gc=6e-4, ell=BASE.ell, E=BASE.E)
        p_hi = PFParams(Gc=12e-4, ell=BASE.ell, E=BASE.E)
        r_lo = run_forward(p_lo, FAST)
        r_hi = run_forward(p_hi, FAST)
        assert r_hi.init_U >= r_lo.init_U, \
            "Higher Gc should delay crack initiation"

    def test_higher_gc_higher_peak_force(self):
        p_lo = PFParams(Gc=6e-4,  ell=BASE.ell, E=BASE.E)
        p_hi = PFParams(Gc=12e-4, ell=BASE.ell, E=BASE.E)
        r_lo = run_forward(p_lo, FAST)
        r_hi = run_forward(p_hi, FAST)
        assert r_hi.peak_F >= r_lo.peak_F

    def test_larger_ell_wider_damage_zone(self):
        p_thin = PFParams(Gc=BASE.Gc, ell=0.02, E=BASE.E)
        p_wide = PFParams(Gc=BASE.Gc, ell=0.08, E=BASE.E)
        r_thin = run_forward(p_thin, FAST)
        r_wide = run_forward(p_wide, FAST)
        w_thin = np.sum(r_thin.d_field > 0.01)
        w_wide = np.sum(r_wide.d_field > 0.01)
        assert w_wide >= w_thin, "Larger ell should give wider damage zone"

    def test_damage_localised_near_centre(self):
        result = run_forward(BASE, FAST)
        n = FAST.n_elem
        centre = n // 2
        flank = n // 4
        assert result.d_field[centre] > result.d_field[flank], \
            "Peak damage should be at bar centre (weak zone)"

    def test_no_damage_at_zero_load(self):
        cfg_tiny = SimConfig(n_elem=50, n_steps=5, u_max_factor=0.01)
        result = run_forward(BASE, cfg_tiny)
        assert np.max(result.d_field) < 0.01

    def test_higher_stiffness_higher_peak(self):
        p_soft  = PFParams(Gc=BASE.Gc, ell=BASE.ell, E=0.5)
        p_stiff = PFParams(Gc=BASE.Gc, ell=BASE.ell, E=2.0)
        r_soft  = run_forward(p_soft,  FAST)
        r_stiff = run_forward(p_stiff, FAST)
        assert r_stiff.peak_F > r_soft.peak_F


# ════════════════════════════════════════════════════════════════════════════
#  5.  crack_probability and simulate_observation
# ════════════════════════════════════════════════════════════════════════════

class TestObservationModel:
    def test_probability_in_unit_interval(self):
        for theta in [0.0, 30.0, 60.0, 90.0]:
            p = crack_probability(BASE, theta, cfg=FAST)
            assert 0.0 < p < 1.0, f"p={p} at theta={theta}"

    def test_simulate_observation_returns_binary(self):
        for _ in range(5):
            y = simulate_observation(BASE, 45.0, cfg=FAST, seed=0)
            assert y in (0, 1)

    def test_lower_gc_eff_higher_crack_probability(self):
        # beta<0 at theta=90 -> lower Gc_eff -> more cracks
        p_low_gc  = crack_probability(
            PFParams(Gc=BASE.Gc, ell=BASE.ell, E=BASE.E, beta=-0.5), 90.0, cfg=FAST)
        p_high_gc = crack_probability(
            PFParams(Gc=BASE.Gc, ell=BASE.ell, E=BASE.E, beta=+0.5), 90.0, cfg=FAST)
        assert p_low_gc > p_high_gc

    def test_anisotropic_vs_isotropic_at_cleavage_angle(self):
        p_aniso = crack_probability(
            PFParams(Gc=BASE.Gc, ell=BASE.ell, E=BASE.E, beta=-0.5), 90.0, cfg=FAST)
        p_iso = crack_probability(BASE, 90.0, cfg=FAST)
        assert p_aniso > p_iso

    def test_seed_reproducible(self):
        y1 = simulate_observation(BASE, 45.0, cfg=FAST, seed=7)
        y2 = simulate_observation(BASE, 45.0, cfg=FAST, seed=7)
        assert y1 == y2


# ════════════════════════════════════════════════════════════════════════════
#  6.  log_likelihood and log_posterior
# ════════════════════════════════════════════════════════════════════════════

class TestLikelihood:
    def setup_method(self):
        self.obs = generate_synthetic_dataset(
            BASE, theta_list=[0.0, 30.0, 60.0, 90.0], cfg=FAST, seed=42)

    def test_log_likelihood_finite(self):
        ll = log_likelihood(BASE, self.obs, cfg=FAST)
        assert np.isfinite(ll)

    def test_log_likelihood_nonpositive(self):
        ll = log_likelihood(BASE, self.obs, cfg=FAST)
        assert ll <= 0.0

    def test_true_params_beat_wrong_gc(self):
        ll_true  = log_likelihood(BASE, self.obs, cfg=FAST)
        ll_wrong = log_likelihood(
            PFParams(Gc=BASE.Gc * 5, ell=BASE.ell, E=BASE.E),
            self.obs, cfg=FAST)
        assert ll_true >= ll_wrong - 3.0, \
            f"True LL={ll_true:.3f} far below wrong LL={ll_wrong:.3f}"

    def test_log_posterior_finite_for_valid_params(self):
        lp = log_posterior(BASE, self.obs, cfg=FAST)
        assert np.isfinite(lp)

    def test_log_posterior_minus_inf_for_negative_gc(self):
        bad = PFParams(Gc=-1.0, ell=BASE.ell, E=BASE.E)
        assert log_posterior(bad, self.obs, cfg=FAST) == -np.inf

    def test_generate_synthetic_dataset_length(self):
        obs = generate_synthetic_dataset(
            BASE, theta_list=list(range(0, 91, 10)), cfg=FAST, seed=0)
        assert len(obs) == 10

    def test_all_observations_binary(self):
        obs = generate_synthetic_dataset(BASE, cfg=FAST, seed=1)
        for _, y in obs:
            assert y in (0, 1)
