"""
Tests for the EnKF analysis kernel (_enkf_kernel) and the
IEC 61131-3 state machine kernel (_state_machine).
"""

from __future__ import annotations

import math
import numpy as np
import pytest

# ── EnKF kernel ───────────────────────────────────────────────────────────────

try:
    from ml import _enkf_kernel as _cpp_enkf
    _HAS_ENKF = True
except ImportError:
    _HAS_ENKF = False


@pytest.mark.skipif(not _HAS_ENKF, reason="_enkf_kernel not built")
class TestEnkfKernel:

    def _simple_ensemble(self, N=30, n_state=2, seed=0):
        rng = np.random.default_rng(seed)
        return np.ascontiguousarray(
            rng.standard_normal((N, n_state)), dtype=np.float64)

    # ── enkf_analysis ─────────────────────────────────────────────────────────

    def test_analysis_matches_python(self):
        """C++ enkf_analysis must produce the same result as the Python reference."""
        N, n_state = 40, 3
        rng = np.random.default_rng(7)
        xf = np.ascontiguousarray(rng.standard_normal((N, n_state)))
        yf = rng.standard_normal(N) * 0.5 + 5.0
        v  = rng.standard_normal(N) * 0.3
        R  = 0.09
        y_obs = 5.2

        # C++ path
        xa_cpp = _cpp_enkf.enkf_analysis(xf, yf, y_obs, v, R)

        # Python reference (verbatim from ml/ekf_state_estimation.py)
        yf_mean = yf.mean()
        xf_mean = xf.mean(axis=0)
        dX = xf - xf_mean
        dY = yf - yf_mean
        Pxy = (dX.T @ dY) / (N - 1)
        Pyy = float((dY @ dY) / (N - 1)) + R
        K   = Pxy / Pyy
        innov = y_obs + v - yf
        xa_py = xf + np.outer(innov, K)

        np.testing.assert_allclose(xa_cpp, xa_py, atol=1e-10,
                                   err_msg="enkf_analysis C++ vs Python mismatch")

    def test_analysis_reduces_spread(self):
        """Analysis should reduce ensemble spread when obs is informative."""
        N = 100
        rng = np.random.default_rng(42)
        xf = np.ascontiguousarray(rng.standard_normal((N, 2)) * 2.0)
        yf = xf[:, 0] + rng.standard_normal(N) * 0.1
        v  = rng.standard_normal(N) * 0.3
        xa = _cpp_enkf.enkf_analysis(xf, yf, 0.0, v, 0.09)
        assert xa.std(axis=0)[0] < xf.std(axis=0)[0] * 0.9, \
            "analysis should reduce spread in direction correlated with obs"

    # ── blade_wear_obs ────────────────────────────────────────────────────────

    def test_blade_wear_obs_matches_scalar(self):
        """C++ vectorized obs must match scalar chip_forward for each member."""
        from pipeline.keyence_apc import (
            chip_forward, CHIP0_UM, FEED_NOMINAL, WEAR_SCALE, FEED_EXPONENT
        )
        rng = np.random.default_rng(3)
        ens = np.ascontiguousarray(
            np.column_stack([rng.uniform(0, 1.5, 50),
                             rng.uniform(0.5, 1.5, 50)]))
        feed = 2.5
        hx_cpp = _cpp_enkf.blade_wear_obs(
            ens, feed, CHIP0_UM, FEED_NOMINAL, WEAR_SCALE, FEED_EXPONENT)
        hx_py = np.array([chip_forward(feed, ens[i, 0], ens[i, 1])
                          for i in range(50)])
        np.testing.assert_allclose(hx_cpp, hx_py, rtol=1e-10)

    # ── three_state_obs ───────────────────────────────────────────────────────

    def test_three_state_obs_matches_scalar(self):
        """C++ three_state_obs must match scalar observation_model for each member."""
        from ml.ekf_state_estimation import observation_model, D_NOM, BLADE_W
        rng = np.random.default_rng(5)
        ens = np.ascontiguousarray(np.column_stack([
            rng.uniform(0.8, 1.2, 60),   # d_eff
            rng.uniform(0.5, 1.5, 60),   # alpha_w
            rng.uniform(0.5, 1.5, 60),   # k_ic
        ]))
        hx_cpp = _cpp_enkf.three_state_obs(ens, D_NOM, BLADE_W)
        hx_py  = np.array([observation_model(ens[i]) for i in range(60)])
        np.testing.assert_allclose(hx_cpp, hx_py, rtol=1e-10)

    # ── blade_wear_update (full round-trip) ───────────────────────────────────

    def test_blade_wear_update_parity(self):
        """Full blade_wear_update must match the Python BladeWearEnKF.update path."""
        import pipeline.keyence_apc as apc

        # Fix seed so both paths draw the same noise.
        rng_seed = 99
        n = 30

        # Python path (force fallback).
        orig = apc._CPP_ENKF
        apc._CPP_ENKF = False
        try:
            enkf_py = apc.BladeWearEnKF(sigma_meas=0.3, n_ensemble=n, seed=rng_seed)
            enkf_py.update(feed_applied=3.0, y_meas=5.1)
            ens_py = enkf_py.ens.copy()
        finally:
            apc._CPP_ENKF = orig

        # C++ path (same initial ensemble — recreate from seed).
        enkf_cpp = apc.BladeWearEnKF(sigma_meas=0.3, n_ensemble=n, seed=rng_seed)
        # Manually draw the same v the Python path used.
        rng_for_v = np.random.default_rng(rng_seed)
        # Re-init draws n*2 normals for the ensemble, then n for v.
        rng_for_v.standard_normal((n, 2))   # skip init noise
        v = rng_for_v.normal(0.0, math.sqrt(enkf_cpp.R), n)

        xf = np.ascontiguousarray(enkf_cpp.ens, np.float64)
        xa, _ = _cpp_enkf.blade_wear_update(
            xf, 3.0, 5.1, v, enkf_cpp.R,
            apc.CHIP0_UM, apc.FEED_NOMINAL, apc.WEAR_SCALE, apc.FEED_EXPONENT,
            0.0, 2.0, 0.5, 1.5,
        )
        np.testing.assert_allclose(xa, ens_py, atol=1e-9,
                                   err_msg="blade_wear_update C++ vs Python mismatch")


# ── State machine kernel ──────────────────────────────────────────────────────

try:
    from pipeline import _state_machine as _cpp_sm
    _HAS_SM = True
except ImportError:
    _HAS_SM = False


@pytest.mark.skipif(not _HAS_SM, reason="_state_machine not built")
class TestStateMachine:
    S = None  # set in first test (avoids import error at collection time)

    def _sm(self):
        return _cpp_sm.StateMachine()

    def test_initial_state_is_idle(self):
        sm = self._sm()
        assert sm.state == _cpp_sm.State.IDLE
        assert sm.state_name == "IDLE"

    def test_valid_transitions(self):
        sm = self._sm()
        assert sm.transition(_cpp_sm.State.SETUP)
        assert sm.state == _cpp_sm.State.SETUP
        assert sm.transition(_cpp_sm.State.DICING)
        assert sm.state == _cpp_sm.State.DICING
        assert sm.transition(_cpp_sm.State.INSPECT)
        assert sm.state == _cpp_sm.State.INSPECT
        assert sm.transition(_cpp_sm.State.DONE)
        assert sm.state == _cpp_sm.State.DONE
        assert sm.transition(_cpp_sm.State.IDLE)
        assert sm.state == _cpp_sm.State.IDLE

    def test_invalid_transition_returns_false(self):
        """Jumping IDLE → DICING (skipping SETUP) must fail without side-effect."""
        sm = self._sm()
        ok = sm.transition(_cpp_sm.State.DICING)
        assert not ok
        assert sm.state == _cpp_sm.State.IDLE, "state must not change on invalid transition"

    def test_estop_from_any_running_state(self):
        sm = self._sm()
        sm.transition(_cpp_sm.State.SETUP)
        sm.transition(_cpp_sm.State.DICING)
        assert sm.transition(_cpp_sm.State.E_STOP)
        assert sm.state == _cpp_sm.State.E_STOP

    def test_fault_recovery_to_idle(self):
        sm = self._sm()
        sm.transition(_cpp_sm.State.SETUP)
        sm.transition(_cpp_sm.State.FAULT)
        assert sm.state == _cpp_sm.State.FAULT
        assert sm.transition(_cpp_sm.State.IDLE)
        assert sm.state == _cpp_sm.State.IDLE

    def test_history_records_transitions(self):
        sm = self._sm()
        sm.transition(_cpp_sm.State.SETUP,  "recipe loaded")
        sm.transition(_cpp_sm.State.DICING, "blade engaged")
        h = sm.history
        assert len(h) == 2
        assert h[0].from_state == "IDLE"
        assert h[0].to_state   == "SETUP"
        assert h[0].reason     == "recipe loaded"
        assert h[1].from_state == "SETUP"
        assert h[1].to_state   == "DICING"

    def test_callbacks_fire_on_enter(self):
        sm = self._sm()
        fired = []
        sm.on_enter(_cpp_sm.State.DICING, lambda: fired.append("enter_dicing"))
        sm.transition(_cpp_sm.State.SETUP)
        assert "enter_dicing" not in fired
        sm.transition(_cpp_sm.State.DICING)
        assert "enter_dicing" in fired

    def test_callbacks_fire_on_exit(self):
        sm = self._sm()
        fired = []
        sm.on_exit(_cpp_sm.State.SETUP, lambda: fired.append("exit_setup"))
        sm.transition(_cpp_sm.State.SETUP)
        assert "exit_setup" not in fired
        sm.transition(_cpp_sm.State.DICING)
        assert "exit_setup" in fired

    def test_reset_always_returns_to_idle(self):
        sm = self._sm()
        sm.transition(_cpp_sm.State.SETUP)
        sm.transition(_cpp_sm.State.DICING)
        sm.reset()
        assert sm.state == _cpp_sm.State.IDLE

    def test_valid_next_correct(self):
        sm = self._sm()
        assert set(sm.valid_next()) == {"SETUP", "E_STOP"}
        sm.transition(_cpp_sm.State.SETUP)
        assert set(sm.valid_next()) == {"DICING", "FAULT", "E_STOP"}

    def test_parity_with_python_state_machine(self):
        """C++ and Python state machines must agree on transition outcomes."""
        from pipeline.disco_sw_stack import StateMachine, MachineState

        py_sm  = StateMachine()
        cpp_sm = self._sm()

        transitions = [
            (_cpp_sm.State.SETUP,   MachineState.SETUP),
            (_cpp_sm.State.DICING,  MachineState.DICING),
            (_cpp_sm.State.INSPECT, MachineState.INSPECT),
            (_cpp_sm.State.DONE,    MachineState.DONE),
            (_cpp_sm.State.IDLE,    MachineState.IDLE),
            (_cpp_sm.State.SETUP,   MachineState.SETUP),
            (_cpp_sm.State.FAULT,   MachineState.FAULT),
            (_cpp_sm.State.IDLE,    MachineState.IDLE),
        ]
        for cpp_t, py_t in transitions:
            cpp_ok = cpp_sm.transition(cpp_t)
            py_ok  = py_sm.transition(py_t)
            assert cpp_ok == py_ok, \
                f"transition result mismatch for {py_t.name}: C++={cpp_ok} Python={py_ok}"
            assert cpp_sm.state_name == py_sm.state.name, \
                f"state mismatch: C++={cpp_sm.state_name} Python={py_sm.state.name}"
