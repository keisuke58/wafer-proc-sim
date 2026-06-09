"""
Tests for the PMSM + FOC spindle motor kernel (_spindle_kernel).

Checks:
  - Numerical parity with the Python reference loop
  - Physics invariants (speed settling, torque sign, power > 0)
  - PMSMSimulator class: step() / simulate() / simulate_until_settled()
  - Speed ripple is non-zero under intermittent blade contact
"""

from __future__ import annotations

import math
import numpy as np
import pytest

try:
    from fem import _spindle_kernel as _cpp_sp
    _HAS_SPINDLE = True
except ImportError:
    _HAS_SPINDLE = False

# Default parameters matching spindle_motor_control.py
DEFAULT = dict(
    omega_ref_rpm = 30000.0,
    R_ohm         = 0.5,
    L_d_mH        = 1.2,
    L_q_mH        = 1.4,
    psi_f_Wb      = 0.08,
    p_pairs       = 2,
    J_kgm2        = 5e-5,
    B_Nms         = 1e-4,
    r_blade_mm    = 55.0,
    F_cut_N       = 15.0,
    Kp_i          = 5.0,
    Ki_i          = 200.0,
    dt_us         = 10.0,
    contact_duty  = 0.3,
)


def _python_loop(n_steps=5000, **kw):
    """Reference Python loop — direct port of spindle_motor_control.py."""
    p = DEFAULT.copy()
    p.update(kw)

    R      = p["R_ohm"]
    Ld     = p["L_d_mH"] * 1e-3
    Lq     = p["L_q_mH"] * 1e-3
    psi_f  = p["psi_f_Wb"]
    pp     = int(p["p_pairs"])
    J      = p["J_kgm2"]
    B      = p["B_Nms"]
    r      = p["r_blade_mm"] * 1e-3
    F_cut  = p["F_cut_N"]
    Kp_i   = p["Kp_i"]
    Ki_i   = p["Ki_i"]
    dt     = p["dt_us"] * 1e-6
    duty   = p["contact_duty"]
    omega_ref = p["omega_ref_rpm"] * 2 * math.pi / 60.0
    T_load_base = F_cut * r

    omega, i_d, i_q, int_iq = omega_ref * 0.9, 0.0, 0.0, 0.0
    omega_log = []
    for k in range(n_steps):
        t = k * dt
        in_contact = (t * p["omega_ref_rpm"] / 60.0 % 1.0) < duty
        T_l = T_load_base if in_contact else T_load_base * 0.1

        err_omega = omega_ref - omega
        i_q_ref   = Kp_i * err_omega * 0.01
        err_iq    = i_q_ref - i_q
        int_iq   += err_iq * dt
        v_q       = Kp_i * err_iq + Ki_i * int_iq
        v_d       = 0.0

        di_d = (v_d - R * i_d + omega * pp * Lq * i_q) / Ld
        di_q = (v_q - R * i_q - omega * pp * (Ld * i_d + psi_f)) / Lq
        i_d += di_d * dt
        i_q += di_q * dt

        T_e   = 1.5 * pp * (psi_f * i_q + (Ld - Lq) * i_d * i_q)
        domega = (T_e - T_l - B * omega) / J
        omega += domega * dt
        omega_log.append(omega)

    return np.array(omega_log)


@pytest.mark.skipif(not _HAS_SPINDLE, reason="_spindle_kernel not built")
class TestSpindleKernel:

    def _make_sim(self, **kw):
        p = DEFAULT.copy(); p.update(kw)
        return _cpp_sp.PMSMSimulator(**p)

    # ── parity ────────────────────────────────────────────────────────────────

    def test_simulate_matches_python_loop(self):
        """C++ simulate() trajectory must match the Python reference to float precision."""
        N = 5000
        sim = self._make_sim()
        r   = sim.simulate(N)
        cpp_omega = np.array(r["omega_rad_s"])
        py_omega  = _python_loop(N)

        assert len(cpp_omega) == N
        np.testing.assert_allclose(cpp_omega, py_omega, atol=1e-10,
                                   err_msg="omega trajectory mismatch")

    def test_simulate_foc_matches_python(self):
        """Standalone simulate_foc() must also match the Python loop."""
        N = 5000
        r = _cpp_sp.simulate_foc(**{k: DEFAULT[k] for k in DEFAULT}, n_steps=N)
        np.testing.assert_allclose(
            np.array(r["omega_rad_s"]), _python_loop(N), atol=1e-10)

    # ── physics ───────────────────────────────────────────────────────────────

    def test_speed_approaches_reference(self):
        """simulate_until_settled must reach within 5% of reference speed."""
        omega_ref = DEFAULT["omega_ref_rpm"] * 2 * math.pi / 60.0
        sim = self._make_sim()
        r   = sim.simulate_until_settled(tol=0.05, max_steps=200000)
        final_omega = float(np.array(r["omega_rad_s"])[-1])
        assert abs(final_omega - omega_ref) / omega_ref < 0.05, \
            f"speed {final_omega:.1f} rad/s too far from ref {omega_ref:.1f}"

    def test_torque_is_positive_under_load(self):
        """T_e must reach positive values during acceleration transient.

        The first ~60 steps have T_e < 0 because back-EMF >> v_q while i_q
        builds from zero.  Once i_q is established the motor drives positive
        torque.  We verify T_e peaks above zero within 100 steps.
        """
        sim = self._make_sim()
        r   = sim.simulate(100)
        T_e = np.array(r["T_e_Nm"])
        assert T_e.max() > 0.0, \
            f"T_e never went positive in 100 steps (max={T_e.max():.4f})"

    def test_power_positive(self):
        """Electrical power should be mostly positive (motoring, not regenerating)."""
        sim = self._make_sim()
        r   = sim.simulate(5000)
        P   = np.array(r["P_elec_W"])
        assert P.mean() > 0.0

    def test_speed_ripple_nonzero(self):
        """Speed ripple must be non-zero under intermittent blade contact."""
        sim = self._make_sim()
        r   = sim.simulate(5000)
        # Only examine settled portion.
        omega = np.array(r["omega_rad_s"])
        ripple = sim.speed_ripple_pct(omega[2000:])
        assert ripple > 0.0, "expected non-zero speed ripple from blade contact"

    def test_omega_rpm_conversion(self):
        """omega_rpm array must equal omega_rad_s * 60 / (2*pi)."""
        sim = self._make_sim()
        r   = sim.simulate(200)
        omega_rad = np.array(r["omega_rad_s"])
        omega_rpm = np.array(r["omega_rpm"])
        np.testing.assert_allclose(
            omega_rpm, omega_rad * 60.0 / (2 * math.pi), rtol=1e-10)

    # ── PMSMSimulator API ─────────────────────────────────────────────────────

    def test_step_increments_time(self):
        """Each step() call must advance t by dt."""
        dt_us = 10.0
        sim = self._make_sim(dt_us=dt_us)
        assert sim.t == pytest.approx(0.0)
        sim.step()
        assert sim.t == pytest.approx(dt_us * 1e-6)

    def test_reset_restores_initial_state(self):
        """reset() must restore omega to 0.9 * omega_ref and zero currents."""
        sim = self._make_sim()
        sim.simulate(500)
        sim.reset()
        omega_ref = DEFAULT["omega_ref_rpm"] * 2 * math.pi / 60.0
        assert sim.omega == pytest.approx(omega_ref * 0.9)
        assert sim.i_d   == pytest.approx(0.0)
        assert sim.i_q   == pytest.approx(0.0)
        assert sim.t     == pytest.approx(0.0)

    def test_simulate_until_settled(self):
        """simulate_until_settled must stop before max_steps if speed converges."""
        sim = self._make_sim()
        r   = sim.simulate_until_settled(tol=0.01, max_steps=100000)
        assert r["settled"], "expected settled=True within 100k steps"
        assert int(r["n_steps"]) < 100000

    def test_step_sequence_matches_simulate(self):
        """Manual step() loop must produce identical trajectory to simulate()."""
        N = 200
        # via simulate()
        sim_a = self._make_sim()
        r_a   = sim_a.simulate(N)

        # via step() loop
        sim_b = self._make_sim()
        omega_b = []
        for _ in range(N):
            s = sim_b.step()
            omega_b.append(s[0])

        np.testing.assert_allclose(
            np.array(r_a["omega_rad_s"]), np.array(omega_b), atol=1e-14)
