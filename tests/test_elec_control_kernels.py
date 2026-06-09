"""
Tests for:
  - fem/_servo_inverter_kernel  (SVPWM, dead-time, switching loss, InverterSimulator)
  - fem/_encoder_kernel         (4x quadrature decode, velocity, EncoderSimulator)
  - pipeline/_recipe_sequencer  (RecipeSequencer lifecycle)
  - pipeline/_interlock_monitor (InterlockMonitor threshold / E-STOP)
"""

from __future__ import annotations
import math, sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── imports ───────────────────────────────────────────────────────────────────

try:
    from fem import _servo_inverter_kernel as _inv
    _HAS_INV = True
except ImportError:
    _HAS_INV = False

try:
    from fem import _encoder_kernel as _enc
    _HAS_ENC = True
except ImportError:
    _HAS_ENC = False

try:
    from pipeline import _recipe_sequencer as _rseq
    _HAS_RSEQ = True
except ImportError:
    _HAS_RSEQ = False

try:
    from pipeline import _interlock_monitor as _ilk
    _HAS_ILK = True
except ImportError:
    _HAS_ILK = False


# ═══════════════════════════════════════════════════════════════════════════════
# Servo inverter (SVPWM)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _HAS_INV, reason="_servo_inverter_kernel not built")
class TestServoInverter:

    # ── svpwm ──────────────────────────────────────────────────────────────────

    def test_svpwm_duties_in_unit_interval(self):
        """All duty cycles must be in [0, 1] for any modulation index."""
        for v_mag in [0, 50, 150, 250, 340]:
            for angle in np.linspace(0, 2*math.pi, 12, endpoint=False):
                v_a = v_mag * math.cos(angle)
                v_b = v_mag * math.sin(angle)
                r = _inv.svpwm(v_a, v_b, 600.0)
                for key in ("duty_a", "duty_b", "duty_c"):
                    d = r[key]
                    assert 0.0 <= d <= 1.0, f"{key}={d:.4f} out of range"

    def test_svpwm_zero_voltage_midpoint(self):
        """Zero reference → all duties = 0.5 (mid-rail)."""
        r = _inv.svpwm(0.0, 0.0, 600.0)
        for key in ("duty_a", "duty_b", "duty_c"):
            assert abs(r[key] - 0.5) < 1e-9, f"{key} = {r[key]}"

    def test_svpwm_sector_rotates(self):
        """Rotating voltage vector must cycle through all 6 sectors."""
        v_dc = 600.0; v_mag = 200.0
        sectors = set()
        for angle in np.linspace(0, 2*math.pi, 60, endpoint=False):
            r = _inv.svpwm(v_mag * math.cos(angle), v_mag * math.sin(angle), v_dc)
            sectors.add(r["sector"])
        assert sectors == {1, 2, 3, 4, 5, 6}

    def test_svpwm_modulation_index_correct(self):
        """Modulation index = |v| / (v_dc/√3)."""
        v_dc = 600.0; v_mag = 200.0
        r = _inv.svpwm(v_mag, 0.0, v_dc)
        expected = v_mag / (v_dc / math.sqrt(3.0))
        assert abs(r["modulation_index"] - expected) < 1e-9

    def test_svpwm_negative_vdc_raises(self):
        with pytest.raises(Exception):
            _inv.svpwm(100.0, 0.0, -600.0)

    # ── dead-time compensation ─────────────────────────────────────────────────

    def test_dead_time_reduces_positive_current_duty(self):
        """Positive current → duty decreases by dead_ratio."""
        duties   = [0.6, 0.5, 0.4]
        currents = [5.0, 5.0, 5.0]
        t_dead, T_sw = 1.5, 66.7
        comp = _inv.dead_time_compensate(duties, currents, t_dead, T_sw)
        ratio = t_dead / T_sw
        for i in range(3):
            expected = duties[i] - ratio
            assert abs(comp[i] - expected) < 1e-9, f"phase {i}: {comp[i]} vs {expected}"

    def test_dead_time_increases_negative_current_duty(self):
        duties   = [0.5, 0.5, 0.5]
        currents = [-5.0, -5.0, -5.0]
        t_dead, T_sw = 1.5, 66.7
        comp = _inv.dead_time_compensate(duties, currents, t_dead, T_sw)
        ratio = t_dead / T_sw
        for i in range(3):
            assert abs(comp[i] - (0.5 + ratio)) < 1e-9

    def test_dead_time_clamps_to_unit(self):
        """Compensation must clamp result to [0, 1]."""
        comp = _inv.dead_time_compensate([0.01, 0.99, 0.5], [10.0, -10.0, 1.0],
                                          50.0, 66.7)
        for d in comp:
            assert 0.0 <= d <= 1.0

    # ── switching loss ─────────────────────────────────────────────────────────

    def test_switching_loss_positive(self):
        P = _inv.switching_loss(10.0, 600.0, 0.5, 0.3, 15.0)
        assert P > 0.0

    def test_switching_loss_zero_current(self):
        P = _inv.switching_loss(0.0, 600.0, 0.5, 0.3, 15.0)
        assert P == pytest.approx(0.0)

    def test_switching_loss_scales_with_frequency(self):
        P1 = _inv.switching_loss(10.0, 600.0, 0.5, 0.3, 10.0)
        P2 = _inv.switching_loss(10.0, 600.0, 0.5, 0.3, 20.0)
        assert abs(P2 - 2.0 * P1) < 1e-6

    # ── InverterSimulator ──────────────────────────────────────────────────────

    def test_inverter_simulate_output_shape(self):
        inv = _inv.InverterSimulator(v_dc=600.0)
        r = inv.simulate(100, 200.0, 0.0)
        for key in ("v_a_V", "v_b_V", "v_c_V", "duty_a", "duty_b", "duty_c",
                    "P_sw_W", "sector"):
            assert len(r[key]) == 100

    def test_inverter_energy_accumulates(self):
        inv = _inv.InverterSimulator()
        inv.set_currents(5.0, -3.0, -2.0)
        inv.simulate(200, 100.0, 50.0)
        assert inv.energy_loss_J > 0.0

    def test_inverter_reset_clears_energy(self):
        inv = _inv.InverterSimulator()
        inv.simulate(100, 100.0, 0.0)
        inv.reset()
        assert inv.energy_loss_J == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Encoder kernel
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _HAS_ENC, reason="_encoder_kernel not built")
class TestEncoderKernel:

    # ── decode_quadrature ──────────────────────────────────────────────────────

    def test_forward_rotation_positive_counts(self):
        """CW rotation (A leads B) → monotonically increasing counts."""
        N = 100
        enc = _enc.EncoderSimulator(ppr=100)
        theta = np.linspace(0, 2*math.pi, N)
        r = enc.simulate(theta.tolist(), dt_s=1e-4)
        counts = np.array(r["counts"])
        assert counts[-1] > counts[0], "expected positive count increment"

    def test_reverse_rotation_negative_counts(self):
        """CCW rotation → decreasing counts."""
        N = 100
        enc = _enc.EncoderSimulator(ppr=100)
        theta = np.linspace(0, -2*math.pi, N)
        r = enc.simulate(theta.tolist(), dt_s=1e-4)
        counts = np.array(r["counts"])
        assert counts[-1] < counts[0]

    def test_one_revolution_count(self):
        """One full revolution → counts increase by PPR*4."""
        ppr = 100
        enc = _enc.EncoderSimulator(ppr=ppr)
        # Use 4*ppr+1 points so every quadrature edge is captured
        N = ppr * 4 + 1
        theta = np.linspace(0, 2*math.pi, N)
        r = enc.simulate(theta.tolist(), dt_s=1e-5)
        counts = np.array(r["counts"])
        assert counts[-1] == ppr * 4, f"expected {ppr*4}, got {counts[-1]}"

    def test_position_quantization_error_lt_one_count(self):
        """Decoded position error < 1 quadrature count width."""
        ppr = 2500
        enc = _enc.EncoderSimulator(ppr=ppr)
        theta = np.linspace(0, 2*math.pi, ppr*4*2 + 1).tolist()
        r = enc.simulate(theta, dt_s=1e-5)
        err = np.array(r["err_rad"])
        res = enc.position_resolution_rad()
        assert np.all(np.abs(err) <= res * 1.01), \
            f"max error {np.abs(err).max():.2e} rad > resolution {res:.2e} rad"

    def test_velocity_sign_correct(self):
        """Positive rotation → positive rpm."""
        ppr = 100
        enc = _enc.EncoderSimulator(ppr=ppr)
        N = ppr * 4 * 5  # 5 revolutions
        theta = np.linspace(0, 5 * 2 * math.pi, N).tolist()
        r = enc.simulate(theta, dt_s=1e-4)
        vel = np.array(r["vel_rpm"])
        # After warm-up, velocity should be positive
        assert vel[ppr*4:].mean() > 0.0

    def test_position_resolution_formula(self):
        """Resolution = 2π / (PPR * 4)."""
        ppr = 2500
        enc = _enc.EncoderSimulator(ppr=ppr)
        expected = 2 * math.pi / (ppr * 4)
        assert abs(enc.position_resolution_rad() - expected) < 1e-12

    def test_encode_decode_ab_signals(self):
        """encode() at 0, π/2, π, 3π/2 should give distinct AB states."""
        enc = _enc.EncoderSimulator(ppr=1)
        states = set()
        for theta in [0.0, math.pi/2, math.pi, 3*math.pi/2]:
            ab = enc.encode(theta)
            states.add(tuple(ab))
        assert len(states) == 4, f"expected 4 distinct AB states, got {len(states)}"


# ═══════════════════════════════════════════════════════════════════════════════
# Recipe sequencer
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _HAS_RSEQ, reason="_recipe_sequencer not built")
class TestRecipeSequencer:

    def _seq(self):
        """Minimal 3-step recipe."""
        seq = _rseq.RecipeSequencer()
        seq.add_step("A", 1000.0, 5000.0)
        seq.add_step("B", 2000.0, 8000.0)
        seq.add_step("C",  500.0, 2000.0)
        return seq

    def test_initial_state_idle(self):
        assert self._seq().state_name() == "IDLE"

    def test_start_transitions_to_running(self):
        seq = self._seq(); seq.start()
        assert seq.state_name() == "RUNNING"

    def test_first_step_is_correct(self):
        seq = self._seq(); seq.start()
        assert seq.current_step_name() == "A"

    def test_complete_step_advances(self):
        seq = self._seq(); seq.start()
        seq.complete_step(True)
        assert seq.current_step_name() == "B"

    def test_all_steps_done_transitions_done(self):
        seq = self._seq(); seq.start()
        for _ in range(3):
            seq.complete_step(True)
        assert seq.state_name() == "DONE"

    def test_failed_step_transitions_fault(self):
        seq = self._seq(); seq.start()
        seq.complete_step(False)
        assert seq.state_name() == "FAULT"

    def test_timeout_triggers_fault(self):
        seq = self._seq(); seq.start()
        seq.tick(6000.0)  # exceeds A timeout of 5000 ms
        assert seq.state_name() == "FAULT"
        assert "timeout" in seq.fault_reason.lower()

    def test_history_records_completed_steps(self):
        seq = self._seq(); seq.start()
        seq.complete_step(True)
        seq.complete_step(True)
        h = seq.get_history()
        assert len(h) == 2
        assert h[0]["name"] == "A"
        assert h[0]["success"] == True
        assert h[1]["name"] == "B"

    def test_progress_pct_increases(self):
        seq = self._seq(); seq.start()
        p0 = seq.progress_pct()
        seq.complete_step(True)
        p1 = seq.progress_pct()
        assert p1 > p0

    def test_pause_resume(self):
        seq = self._seq(); seq.start()
        seq.pause()
        assert seq.state_name() == "PAUSED"
        seq.resume()
        assert seq.state_name() == "RUNNING"

    def test_abort_sets_fault(self):
        seq = self._seq(); seq.start()
        seq.abort("test abort")
        assert seq.state_name() == "FAULT"
        assert "test abort" in seq.fault_reason

    def test_reset_returns_to_idle(self):
        seq = self._seq(); seq.start()
        seq.complete_step(True)
        seq.reset()
        assert seq.state_name() == "IDLE"
        assert seq.steps_done() == 0

    def test_disco_dicing_recipe_12_steps(self):
        seq = _rseq.RecipeSequencer.disco_dicing_recipe()
        assert seq.steps_total() == 12

    def test_empty_recipe_start_raises(self):
        seq = _rseq.RecipeSequencer()
        with pytest.raises(Exception):
            seq.start()


# ═══════════════════════════════════════════════════════════════════════════════
# Interlock monitor
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _HAS_ILK, reason="_interlock_monitor not built")
class TestInterlockMonitor:

    def _mon(self):
        """Monitor with one sensor: hi_warn=80, hi_stop=90, hi_fault=100."""
        m = _ilk.InterlockMonitor()
        m.add_sensor("temp", hi_warn=80.0, hi_stop=90.0, hi_fault=100.0)
        return m

    def test_ok_when_in_range(self):
        m = self._mon()
        m.update("temp", 70.0)
        assert m.system_level_name() == "OK"
        assert m.is_safe_to_run()

    def test_warn_when_hi_warn_crossed(self):
        m = self._mon()
        m.update("temp", 85.0)
        assert m.system_level_name() == "WARN"
        assert m.is_safe_to_run()  # WARN does not stop machine

    def test_stop_when_hi_stop_crossed(self):
        m = self._mon()
        m.update("temp", 95.0)
        assert m.system_level_name() == "STOP"
        assert not m.is_safe_to_run()

    def test_fault_triggers_estop(self):
        m = self._mon()
        m.update("temp", 105.0)
        assert m.system_level_name() == "FAULT"
        assert m.e_stop_active
        assert not m.is_safe_to_run()

    def test_back_to_ok_clears_active_alarm(self):
        m = self._mon()
        m.update("temp", 85.0)
        assert len(m.get_active_alarms()) == 1
        m.update("temp", 70.0)
        assert len(m.get_active_alarms()) == 0

    def test_alarm_log_grows(self):
        m = self._mon()
        m.update("temp", 85.0)
        m.update("temp", 95.0)
        assert len(m.get_alarm_log()) >= 2

    def test_reset_clears_everything(self):
        m = self._mon()
        m.update("temp", 105.0)
        m.reset()
        assert m.system_level_name() == "OK"
        assert not m.e_stop_active
        assert len(m.get_active_alarms()) == 0

    def test_acknowledge_clears_specific_alarm(self):
        m = self._mon()
        m.add_sensor("pressure", hi_warn=5.0)
        m.update("temp", 85.0)
        m.update("pressure", 6.0)
        assert len(m.get_active_alarms()) == 2
        m.acknowledge_alarm("temp")
        assert len(m.get_active_alarms()) == 1

    def test_update_all(self):
        m = _ilk.InterlockMonitor()
        m.add_sensor("a", hi_fault=10.0)
        m.add_sensor("b", hi_warn=5.0)
        lv = m.update_all({"a": 5.0, "b": 6.0})
        assert m.get_sensor_level("a") == "OK"
        assert m.get_sensor_level("b") == "WARN"

    def test_unknown_sensor_raises(self):
        m = self._mon()
        with pytest.raises(Exception):
            m.update("nonexistent", 42.0)

    def test_disco_interlocks_factory(self):
        m = _ilk.InterlockMonitor.disco_dicing_interlocks()
        # Nominal conditions → all OK
        m.update_all({
            "spindle_rpm":      30000.0,
            "coolant_flow_lpm": 3.0,
            "door_closed":      1.0,
            "blade_depth_mm":  -0.2,
            "chuck_vacuum_kPa": 80.0,
            "blade_wear_um":    15.0,
        })
        assert m.system_level_name() == "OK"

    def test_disco_spindle_fault(self):
        """Spindle overspeed → FAULT."""
        m = _ilk.InterlockMonitor.disco_dicing_interlocks()
        m.update("spindle_rpm", 36000.0)
        assert m.system_level_name() == "FAULT"

    def test_disco_door_open_fault(self):
        """Door open → FAULT."""
        m = _ilk.InterlockMonitor.disco_dicing_interlocks()
        m.update("door_closed", 0.0)
        assert m.system_level_name() == "FAULT"
