"""
Tests for:
  - machine/_disco_machine   (unified digital twin)
  - ml/_gp_inference_kernel  (RBF kernel, prediction, EI/UCB)
"""
from __future__ import annotations
import math, sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from machine import _disco_machine as _dm
    _HAS_DM = True
except ImportError:
    _HAS_DM = False

try:
    from ml import _gp_inference_kernel as _gp
    _HAS_GP = True
except ImportError:
    _HAS_GP = False


# ═══════════════════════════════════════════════════════════════════════════════
# DiscoMachine digital twin
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _HAS_DM, reason="_disco_machine not built")
class TestDiscoMachine:

    def _make(self, **kw):
        p = _dm.MachineParams()
        for k, v in kw.items():
            setattr(p, k, v)
        return _dm.DiscoMachine(p)

    # ── initial state ──────────────────────────────────────────────────────────

    def test_initial_mode_idle(self):
        m = self._make()
        assert m.mode_name() == "IDLE"

    def test_initial_clock_zero(self):
        m = self._make()
        assert m.clock_us == pytest.approx(0.0)

    def test_initial_estop_false(self):
        assert not self._make().e_stop

    # ── tick / simulate ────────────────────────────────────────────────────────

    def test_tick_advances_clock(self):
        m = self._make(dt_us=10.0)
        m.tick()
        assert m.clock_us == pytest.approx(10.0)

    def test_simulate_step_count(self):
        m = self._make()
        m.simulate(500)
        assert m.step_count == 500

    def test_simulate_returns_telemetry_keys(self):
        m = self._make()
        r = m.simulate(100)
        for k in ("omega_rad_s", "T_e_Nm", "x_mm", "y_mm", "z_mm"):
            assert k in r

    def test_telemetry_length_matches_steps(self):
        m = self._make()
        r = m.simulate(200)
        assert len(r["omega_rad_s"]) == 200

    # ── spindle physics ────────────────────────────────────────────────────────

    def test_spindle_omega_positive(self):
        m = self._make()
        m.simulate(1000)
        s = m.get_state()
        assert s["omega_rad_s"] > 0.0

    def test_spindle_rpm_conversion(self):
        m = self._make()
        m.simulate(100)
        s = m.get_state()
        assert abs(s["omega_rpm"] - s["omega_rad_s"] * 60 / (2*math.pi)) < 1e-6

    # ── inverter ───────────────────────────────────────────────────────────────

    def test_inverter_duties_in_range(self):
        m = self._make()
        m.simulate(100)
        s = m.get_state()
        for key in ("duty_a", "duty_b", "duty_c"):
            assert 0.0 <= s[key] <= 1.0

    def test_inverter_sector_valid(self):
        m = self._make()
        m.simulate(200)
        s = m.get_state()
        assert 1 <= s["inverter_sector"] <= 6

    def test_inverter_energy_accumulates(self):
        m = self._make()
        m.simulate(1000)
        assert m.get_state()["inverter_energy_J"] > 0.0

    # ── motion control ─────────────────────────────────────────────────────────

    def test_motion_moves_toward_target(self):
        m = self._make()
        m.set_target(50.0, 0.0, 0.0)
        m.simulate(5000)
        x = m.get_state()["x_mm"]
        assert x > 1.0, f"x={x:.4f} did not move toward target 50 mm"

    def test_motion_z_moves_toward_target(self):
        m = self._make()
        m.set_target(0.0, 0.0, -0.3)
        m.simulate(5000)
        z = m.get_state()["z_mm"]
        assert z < -0.05, f"z={z:.4f} did not move toward -0.3 mm"

    # ── E-STOP ─────────────────────────────────────────────────────────────────

    def test_estop_freezes_clock(self):
        m = self._make()
        m.simulate(100)
        m.engage_estop()
        t0 = m.clock_us
        m.tick(); m.tick()
        assert m.clock_us == pytest.approx(t0)

    def test_clear_estop_resumes(self):
        m = self._make()
        m.engage_estop()
        m.clear_estop()
        assert not m.e_stop
        assert m.mode_name() == "IDLE"

    def test_start_dicing_after_estop_raises(self):
        m = self._make()
        m.engage_estop()
        with pytest.raises(Exception):
            m.start_dicing()

    # ── reset ──────────────────────────────────────────────────────────────────

    def test_reset_clears_state(self):
        m = self._make()
        m.simulate(500)
        m.reset()
        assert m.step_count == 0
        assert m.clock_us == pytest.approx(0.0)

    def test_get_state_keys(self):
        m = self._make()
        m.simulate(10)
        s = m.get_state()
        for k in ("clock_us", "mode", "omega_rad_s", "T_e_Nm",
                  "x_mm", "y_mm", "z_mm", "duty_a", "inverter_sector"):
            assert k in s, f"missing key: {k}"


# ═══════════════════════════════════════════════════════════════════════════════
# GP inference kernel
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _HAS_GP, reason="_gp_inference_kernel not built")
class TestGPInferenceKernel:

    # ── rbf_kernel_matrix ─────────────────────────────────────────────────────

    def test_kernel_matrix_symmetric(self):
        rng = np.random.default_rng(0)
        X = np.ascontiguousarray(rng.standard_normal((10, 3)))
        ls = np.ones(3)
        K = np.array(_gp.rbf_kernel_matrix(X, ls, 1.0))
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_kernel_matrix_diagonal_equals_signal_var(self):
        rng = np.random.default_rng(1)
        X = np.ascontiguousarray(rng.standard_normal((8, 2)))
        sf = 2.5
        K = np.array(_gp.rbf_kernel_matrix(X, np.ones(2), sf))
        np.testing.assert_allclose(np.diag(K), sf, rtol=1e-10)

    def test_kernel_matrix_positive_semidefinite(self):
        rng = np.random.default_rng(2)
        X = np.ascontiguousarray(rng.standard_normal((12, 4)))
        K = np.array(_gp.rbf_kernel_matrix(X, np.ones(4) * 0.5, 1.0))
        eigs = np.linalg.eigvalsh(K)
        assert eigs.min() >= -1e-9, f"min eigenvalue {eigs.min():.2e} < 0"

    def test_kernel_matrix_matches_sklearn(self):
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
        rng = np.random.default_rng(3)
        X = np.ascontiguousarray(rng.standard_normal((6, 2)))
        ls = np.array([1.5, 2.0]); sf = 0.8
        K_cpp = np.array(_gp.rbf_kernel_matrix(X, ls, sf))
        # sklearn: C(sf) * RBF(ls) with anisotropic length scales
        # k(x,x') = sf * exp(-0.5 * sum ((xi-xj)/li)^2)
        K_py = sf * np.exp(-0.5 * np.sum(
            ((X[:,None,:] - X[None,:,:]) / ls)**2, axis=-1))
        np.testing.assert_allclose(K_cpp, K_py, rtol=1e-9)

    # ── rbf_kernel_vector ──────────────────────────────────────────────────────

    def test_kernel_vector_length(self):
        rng = np.random.default_rng(4)
        X = np.ascontiguousarray(rng.standard_normal((20, 3)))
        x = np.ascontiguousarray(rng.standard_normal(3))
        k = np.array(_gp.rbf_kernel_vector(x, X, np.ones(3), 1.0))
        assert k.shape == (20,)

    def test_kernel_vector_max_at_identical_point(self):
        """k(x, x) = signal_var (max possible value)."""
        X = np.ascontiguousarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
        x = np.ascontiguousarray(np.array([1.0, 2.0]))
        k = np.array(_gp.rbf_kernel_vector(x, X, np.ones(2), 1.0))
        assert abs(k[0] - 1.0) < 1e-10
        assert k[1] < k[0]

    # ── gp_predict_mean ────────────────────────────────────────────────────────

    def test_predict_mean_shape(self):
        rng = np.random.default_rng(5)
        Xtr = np.ascontiguousarray(rng.standard_normal((15, 2)))
        Xte = np.ascontiguousarray(rng.standard_normal((30, 2)))
        alpha = np.ascontiguousarray(rng.standard_normal(15))
        mu = np.array(_gp.gp_predict_mean(Xte, Xtr, alpha, np.ones(2), 1.0))
        assert mu.shape == (30,)

    def test_predict_mean_matches_manual(self):
        """C++ mean must match Python reference k_star @ alpha."""
        rng = np.random.default_rng(6)
        N, D = 10, 3
        Xtr   = np.ascontiguousarray(rng.standard_normal((N, D)))
        Xte   = np.ascontiguousarray(rng.standard_normal((5, D)))
        alpha = np.ascontiguousarray(rng.standard_normal(N))
        ls    = np.ascontiguousarray(np.array([1.0, 1.5, 0.8]))
        sf    = 1.2

        mu_cpp = np.array(_gp.gp_predict_mean(Xte, Xtr, alpha, ls, sf))

        # Python reference
        K_star = sf * np.exp(-0.5 * np.sum(
            ((Xte[:,None,:] - Xtr[None,:,:]) / ls)**2, axis=-1))
        mu_py  = K_star @ alpha

        np.testing.assert_allclose(mu_cpp, mu_py, rtol=1e-9)

    # ── gp_predict_full ────────────────────────────────────────────────────────

    def test_predict_full_std_nonnegative(self):
        rng = np.random.default_rng(7)
        N, D = 8, 2
        Xtr   = np.ascontiguousarray(rng.standard_normal((N, D)))
        Xte   = np.ascontiguousarray(rng.standard_normal((5, D)))
        ls    = np.ones(D); sf = 1.0; noise = 1e-3

        K = np.array(_gp.rbf_kernel_matrix(Xtr, ls, sf)) + noise * np.eye(N)
        L    = np.linalg.cholesky(K)
        L_inv = np.ascontiguousarray(np.linalg.inv(L))
        y     = np.ascontiguousarray(rng.standard_normal(N))
        alpha = np.ascontiguousarray(np.linalg.solve(K, y))

        r = _gp.gp_predict_full(Xte, Xtr, alpha, L_inv, ls, sf, noise)
        std = np.array(r["std"])
        assert np.all(std >= 0.0), f"negative std: {std.min():.4e}"

    def test_predict_full_std_small_at_training_points(self):
        """Posterior std should be ~0 at training points (noiseless GP)."""
        rng = np.random.default_rng(8)
        N, D = 5, 1
        Xtr = np.ascontiguousarray(rng.uniform(-2, 2, (N, D)))
        ls  = np.ones(D); sf = 1.0; noise = 1e-8

        K     = np.array(_gp.rbf_kernel_matrix(Xtr, ls, sf)) + noise * np.eye(N)
        L     = np.linalg.cholesky(K)
        L_inv = np.ascontiguousarray(np.linalg.inv(L))
        y     = np.ascontiguousarray(np.sin(Xtr[:, 0]))
        alpha = np.ascontiguousarray(np.linalg.solve(K, y))

        r   = _gp.gp_predict_full(Xtr, Xtr, alpha, L_inv, ls, sf, noise)
        std = np.array(r["std"])
        assert std.max() < 0.01, f"std at training points too large: {std.max():.4f}"

    # ── acquisition functions ──────────────────────────────────────────────────

    def test_ei_nonnegative(self):
        mu  = np.ascontiguousarray(np.array([0.5, 1.0, 1.5, 0.2]))
        std = np.ascontiguousarray(np.array([0.1, 0.2, 0.3, 0.05]))
        ei = np.array(_gp.expected_improvement(mu, std, f_best=0.8))
        assert np.all(ei >= 0.0)

    def test_ei_zero_below_fbest_with_zero_std(self):
        """EI = 0 when std ≈ 0 and mean < f_best."""
        mu  = np.ascontiguousarray(np.array([0.1]))
        std = np.ascontiguousarray(np.array([1e-15]))
        ei  = np.array(_gp.expected_improvement(mu, std, f_best=1.0))
        assert abs(ei[0]) < 1e-10

    def test_ucb_larger_for_larger_std(self):
        mu  = np.ascontiguousarray(np.array([1.0, 1.0]))
        std = np.ascontiguousarray(np.array([0.1, 0.5]))
        ucb = np.array(_gp.upper_confidence_bound(mu, std, beta=2.0))
        assert ucb[1] > ucb[0]

    def test_ucb_formula(self):
        mu  = np.ascontiguousarray(np.array([2.0]))
        std = np.ascontiguousarray(np.array([0.3]))
        ucb = np.array(_gp.upper_confidence_bound(mu, std, beta=4.0))
        assert abs(ucb[0] - (2.0 + math.sqrt(4.0) * 0.3)) < 1e-10
