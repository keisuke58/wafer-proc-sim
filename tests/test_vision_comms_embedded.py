"""
Smoke + correctness tests for the 4 new C++ kernels:
  vision/_kerf_detection_kernel   — Sobel edge / kerf measurement
  embedded/_motor_ctrl_sm_kernel  — PMSM FOC state machine
  comms/_modbus_rtu_kernel        — Modbus RTU CRC + slave
  comms/_ethercat_frame_kernel    — EtherCAT frame parse/build
"""
from __future__ import annotations
import numpy as np
import pytest

# ── vision: kerf detection ────────────────────────────────────────────────────

try:
    from vision._kerf_detection_kernel import detect_kerf_width, detect_blade_tip, measure_chipping
    _HAS_VISION = True
except ImportError:
    _HAS_VISION = False


def _make_kerf_image(H=64, W=128, left=40, right=80, noise=5.0, rng=None):
    rng = rng or np.random.default_rng(0)
    img = np.zeros((H, W), dtype=np.float32) + rng.normal(0, noise, (H, W)).astype(np.float32)
    img[:, left]  += 200.0   # left wall: bright edge
    img[:, right] += 200.0   # right wall: bright edge
    return np.clip(img, 0, 255)


@pytest.mark.skipif(not _HAS_VISION, reason="_kerf_detection_kernel not built")
class TestKerfDetection:
    def test_kerf_width_known(self):
        img = _make_kerf_image(left=30, right=70)
        # threshold in projection units (sum over 64 rows); noise floor ≈ 64*14 ≈ 900
        res = detect_kerf_width(img, threshold=5000.0, pixel_per_mm=10.0)
        assert res["status"] == "ok"
        # Sobel flanking peaks at c=29 and c=71 → span = 42 px / 10 ppmm = 4.2 mm
        assert 3.5 <= res["kerf_width_mm"] <= 5.0, f"Got {res['kerf_width_mm']}"

    def test_no_kerf_returns_negative(self):
        img = np.zeros((32, 64), dtype=np.float32)
        res = detect_kerf_width(img, threshold=10.0, pixel_per_mm=10.0)
        assert res["status"] == "no_kerf_found"
        assert res["kerf_width_mm"] == -1.0

    def test_blade_tip_detected(self):
        img = np.zeros((64, 64), dtype=np.float32)
        img[10, 20:40] = 220.0   # bright tip row
        res = detect_blade_tip(img, threshold=200.0)
        assert res["status"] == "ok"
        assert res["tip_y"] == pytest.approx(10.0, abs=1.0)

    def test_chipping_measure(self):
        img = np.full((32, 64), 128.0, dtype=np.float32)
        img[:, 32:38] = 10.0   # dark chipped region outside kerf walls at 30/40
        res = measure_chipping(img, kerf_left_px=30.0, kerf_right_px=40.0, pixel_per_mm=10.0)
        assert res["status"] == "ok"
        assert res["chipping_ratio"] >= 0.0


# ── embedded: PMSM state machine ─────────────────────────────────────────────

try:
    from embedded._motor_ctrl_sm_kernel import MotorControlSM
    _HAS_EMBEDDED = True
except ImportError:
    _HAS_EMBEDDED = False


@pytest.mark.skipif(not _HAS_EMBEDDED, reason="_motor_ctrl_sm_kernel not built")
class TestMotorControlSM:
    def test_ramp_to_target(self):
        # omega_max = Kt*iq_max / B = 0.05*10 / 0.001 = 500 rad/s
        # Use 300 rad/s (60% of max) to stay in linear range
        sm = MotorControlSM(dt=1e-4, ramp_rate=200.0)
        sm.power_on()
        # Burn through POWERON (t=0.05 s) + INIT → IDLE
        for _ in range(600): sm.isr_tick()
        omega_target = 300.0  # rad/s ≈ 2865 rpm
        sm.set_target_omega(omega_target)
        result = sm.run(60_000)  # 6 s (ramp 1.5 s + settle 4.5 s)
        final = result["final_omega_rad_s"]
        assert abs(final - omega_target) / omega_target < 0.05, \
            f"Speed error > 5%: final={final:.1f} target={omega_target}"

    def test_estop_zeroes_torque(self):
        sm = MotorControlSM()
        sm.power_on()
        for _ in range(600): sm.isr_tick()
        sm.set_target_omega(1000.0)
        sm.run(10_000)
        sm.emergency_stop()
        assert sm.get_state() == "ESTOP"

    def test_fault_clears(self):
        sm = MotorControlSM()
        sm.power_on()
        for _ in range(600): sm.isr_tick()
        sm.inject_fault()
        assert sm.get_state() == "FAULT"
        sm.clear_fault()
        assert sm.get_state() == "IDLE"

    def test_telemetry_shapes(self):
        sm = MotorControlSM()
        sm.power_on()
        N = 500
        result = sm.run(N)
        assert len(result["omega_rad_s"]) == N
        assert len(result["iq_A"]) == N


# ── comms: Modbus RTU ─────────────────────────────────────────────────────────

try:
    from comms._modbus_rtu_kernel import (
        crc16_modbus, encode_read_holding, encode_write_single,
        parse_frame, ModbusRTUSlave,
    )
    _HAS_MODBUS = True
except ImportError:
    _HAS_MODBUS = False


@pytest.mark.skipif(not _HAS_MODBUS, reason="_modbus_rtu_kernel not built")
class TestModbusRTU:
    def test_crc16_known_value(self):
        # Frame [01 03 00 00 00 01] → CRC bytes in frame = 84 0A (little-endian)
        # → integer value = 0x0A84
        frame = bytes([0x01, 0x03, 0x00, 0x00, 0x00, 0x01])
        crc = crc16_modbus(frame)
        assert crc == 0x0A84, f"CRC mismatch: 0x{crc:04X}"

    def test_encode_decode_roundtrip(self):
        raw = bytes(encode_read_holding(1, 0, 4))
        parsed = parse_frame(raw)
        assert parsed["slave_id"]  == 1
        assert parsed["func_code"] == 0x03
        assert parsed["crc_valid"] is True

    def test_slave_write_read(self):
        slave = ModbusRTUSlave(slave_id=1, n_regs=64)
        slave.write_register(10, 0xABCD)
        assert slave.read_register(10) == 0xABCD

    def test_slave_process_read_request(self):
        slave = ModbusRTUSlave(slave_id=1, n_regs=64)
        slave.write_register(0, 42)
        req = bytes(encode_read_holding(1, 0, 1))
        resp = slave.process_request(req)
        assert len(resp) >= 5
        # Response: 01 03 02 00 2A CRC CRC
        assert resp[3] == 0x00 and resp[4] == 42

    def test_slave_write_single_request(self):
        slave = ModbusRTUSlave(slave_id=1, n_regs=64)
        req = bytes(encode_write_single(1, 5, 999))
        slave.process_request(req)
        assert slave.read_register(5) == 999


# ── comms: EtherCAT ──────────────────────────────────────────────────────────

try:
    from comms._ethercat_frame_kernel import parse_ec_frame, build_ec_frame, cmd_name, EcSlave
    _HAS_EC = True
except ImportError:
    _HAS_EC = False


@pytest.mark.skipif(not _HAS_EC, reason="_ethercat_frame_kernel not built")
class TestEtherCAT:
    def test_build_parse_roundtrip(self):
        payload = bytes([0xAA, 0xBB, 0xCC, 0xDD])
        raw = build_ec_frame(0x04, 0x01, 0x00010130, payload, False)  # FPRD
        datagrams = parse_ec_frame(raw)
        assert len(datagrams) == 1
        dg = datagrams[0]
        assert dg["cmd"] == 0x04
        assert dg["idx"] == 0x01
        assert list(dg["data"]) == list(payload)

    def test_cmd_name(self):
        assert cmd_name(0x04) == "FPRD"
        assert cmd_name(0x05) == "FPWR"
        assert cmd_name(0x0C) == "LRW"

    def test_slave_fprd(self):
        slave = EcSlave(configured_addr=0x0001)
        slave.write_byte(0x0130, 0x08)   # write AL status
        dg_in = {"cmd": 0x04, "idx": 1, "addr": 0x01300001,
                 "data": [0x00], "wkc": 0}  # FPRD addr=0x0001, offset=0x0130
        dg_out = slave.process_datagram(dg_in)
        assert dg_out["wkc"] == 1
        assert dg_out["data"][0] == 0x08

    def test_slave_fpwr(self):
        slave = EcSlave(configured_addr=0x0002)
        dg_in = {"cmd": 0x05, "idx": 2, "addr": 0x01000002,
                 "data": [0x01], "wkc": 0}  # FPWR addr=0x0002, offset=0x0100
        dg_out = slave.process_datagram(dg_in)
        assert dg_out["wkc"] == 1
        assert slave.read_byte(0x0100) == 0x01
