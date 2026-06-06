"""
DISCO dicing saw — electrical & control system model
Covers the core エレキ scope: power electronics, servo drives, vision electronics, safety I/O.

Subsystems modelled
-------------------
1. DC bus  — 3-phase rectifier + DC-link capacitor sizing
2. Spindle drive  — extends spindle_motor_control.py; full inverter + PMSM
3. XY linear-motor stage — thrust force, current control, position loop (PID + notch)
4. Vision illumination — coaxial LED driver, strobing sync with spindle index
5. EMI budget — PWM switching noise, cable coupling, CE/RE margins
6. Safety interlock PLC — IEC 62061 SIL-2 watchdog logic

References
----------
Mohan et al. (2003) Power Electronics
DISCO DFD6361 spec: 60 000 rpm spindle, 400 W
THK LM guide catalog: linear motor stage (HSR series)
IEC 62061:2021 (Safety of machinery)
"""

import numpy as np

# ── 1. DC Bus ─────────────────────────────────────────────────────────────────

def dc_bus(V_line_rms: float = 200.0,  # line-to-line RMS [V]
           P_spindle_W: float = 400.0,
           P_stage_W: float = 200.0,
           f_ripple_Hz: float = 360.0,  # 6-pulse rectifier: 6 × 60 Hz
           ripple_spec_pct: float = 2.0
           ) -> dict:
    """
    Size the DC-link capacitor for a 6-pulse diode rectifier.

    V_dc = √2 × V_line × (3/π)  (ideal 6-pulse average)
    ΔV / V_dc = I_load / (C × f_ripple × V_dc)  → C_min
    """
    V_dc = np.sqrt(2) * V_line_rms * (3 / np.pi) * np.sqrt(3)  # ≈ 270 V for 200 V input
    I_load = (P_spindle_W + P_stage_W) / V_dc
    delta_V = V_dc * (ripple_spec_pct / 100)
    C_uF = (I_load / (f_ripple_Hz * delta_V)) * 1e6

    return {
        "V_dc_V": round(V_dc, 1),
        "I_load_A": round(I_load, 2),
        "C_min_uF": round(C_uF, 1),
        "ripple_spec_pct": ripple_spec_pct,
        "note": "6-pulse diode bridge; add 20% derating → C_design ≈ {:.0f} µF".format(C_uF * 1.2),
    }


# ── 2. Spindle inverter — thermal sizing ─────────────────────────────────────

def spindle_inverter(
    omega_max_rpm: float = 60_000.0,
    P_rated_W: float = 400.0,
    eta_inverter: float = 0.96,
    T_ambient_C: float = 40.0,
    R_th_jc: float = 0.4,   # IGBT junction-to-case [K/W]
    R_th_cs: float = 0.1,   # case-to-sink [K/W]
    R_th_sa: float = 1.2,   # sink-to-air [K/W]
    T_j_max_C: float = 150.0,
) -> dict:
    """
    Thermal margin check for the spindle IGBT module.
    P_loss = P_rated × (1 - η) / η
    T_j = T_amb + P_loss × (R_th_jc + R_th_cs + R_th_sa)
    """
    P_loss = P_rated_W * (1 - eta_inverter) / eta_inverter
    T_j = T_ambient_C + P_loss * (R_th_jc + R_th_cs + R_th_sa)
    margin_K = T_j_max_C - T_j

    return {
        "P_loss_W": round(P_loss, 1),
        "T_j_C": round(T_j, 1),
        "T_j_margin_K": round(margin_K, 1),
        "pass": margin_K > 20,  # 20 K derating guard
        "omega_max_rpm": omega_max_rpm,
    }


# ── 3. XY Linear Motor Stage ─────────────────────────────────────────────────

def xy_stage(
    mass_kg: float = 5.0,       # moving carriage + chuck table
    stroke_mm: float = 200.0,
    v_max_mms: float = 300.0,   # max feed for dicing [mm/s]
    a_max_ms2: float = 10.0,    # acceleration
    Kt_N_A: float = 40.0,       # thrust constant [N/A]
    R_coil_ohm: float = 2.5,
    L_coil_mH: float = 1.5,         # low-inductance linear motor for high BW
    f_bandwidth_Hz: float = 20.0,   # position-loop BW target for precision dicing (not servo robot)
    notch_freq_Hz: float | None = 120.0,  # resonance to suppress
) -> dict:
    """
    Linear motor force/current sizing and current-loop bandwidth check.

    Thrust:   F = m × a_max
    Peak current: I_peak = F / Kt
    Current-loop BW: ω_ci = R/L  (natural)  → achievable BW
    Position-loop BW: ω_cp ≈ ω_ci / 10  (rule of thumb)
    """
    F_peak_N = mass_kg * a_max_ms2
    I_peak_A = F_peak_N / Kt_N_A
    P_copper_W = I_peak_A**2 * R_coil_ohm

    omega_ci_rad = R_coil_ohm / (L_coil_mH * 1e-3)  # natural current-loop bandwidth
    f_ci_Hz = omega_ci_rad / (2 * np.pi)
    f_cp_Hz = f_ci_Hz / 10  # achievable position-loop BW

    result = {
        "F_peak_N": round(F_peak_N, 1),
        "I_peak_A": round(I_peak_A, 2),
        "P_copper_W": round(P_copper_W, 1),
        "f_current_loop_Hz": round(f_ci_Hz, 0),
        "f_position_loop_Hz": round(f_cp_Hz, 1),
        "bandwidth_spec_Hz": f_bandwidth_Hz,
        "bandwidth_ok": f_cp_Hz >= f_bandwidth_Hz,
    }
    if notch_freq_Hz is not None:
        # Notch filter: second-order, Q=5, at resonance
        Q = 5.0
        omega_n = 2 * np.pi * notch_freq_Hz
        attenuation_dB = -40 * np.log10(Q)  # simplified: peak attenuation
        result["notch_freq_Hz"] = notch_freq_Hz
        result["notch_attenuation_dB"] = round(attenuation_dB, 1)

    return result


# ── 4. Vision LED Illumination ─────────────────────────────────────────────────

def vision_illumination(
    camera_fps: float = 500.0,       # frames/s
    strobe_duty: float = 0.02,       # 2 % duty during exposure
    LED_peak_mA: float = 3000.0,     # peak drive current [mA]
    V_forward_V: float = 3.2,        # LED forward voltage
    n_LEDs: int = 48,
    spindle_rpm: float = 30_000.0,   # dicing speed for sync check
) -> dict:
    """
    Strobe sync: exposure must complete within one blade-slot gap.
    Slot gap Δt = 1 / (n_slots × rpm/60).  DISCO blades typ. 4-6 slots.
    LED average power vs. thermal limit.
    """
    exposure_us = (strobe_duty / camera_fps) * 1e6  # µs per frame
    n_slots = 4  # typical dicing blade
    slot_gap_us = 1e6 / (n_slots * spindle_rpm / 60)
    sync_ok = exposure_us < slot_gap_us

    P_avg_W = n_LEDs * (LED_peak_mA * 1e-3) * V_forward_V * strobe_duty
    P_peak_W = n_LEDs * (LED_peak_mA * 1e-3) * V_forward_V

    return {
        "exposure_us": round(exposure_us, 2),
        "slot_gap_us": round(slot_gap_us, 2),
        "strobe_sync_ok": sync_ok,
        "P_avg_W": round(P_avg_W, 2),
        "P_peak_W": round(P_peak_W, 1),
        "camera_fps": camera_fps,
    }


# ── 5. EMI Budget ─────────────────────────────────────────────────────────────

def emi_budget(
    V_dc_V: float = 270.0,
    f_sw_kHz: float = 10.0,      # PWM switching frequency [kHz]
    C_stray_pF: float = 2.0,     # stray capacitance motor cable [pF] (0.5 m double-shielded)
    I_switch_A: float = 5.0,     # peak switch current
    t_rise_ns: float = 500.0,    # IGBT rise time [ns] (soft-switched)
    cable_len_m: float = 0.5,    # short shielded cable inside cabinet
) -> dict:
    """
    Simplified conducted emission (CE) estimate at switching frequency.
    Common-mode noise voltage: V_cm ≈ V_dc × C_stray / C_total
    Differential peak: V_diff ≈ L_cable × dI/dt  (after LC filter)
    CISPR 11 Class A limit at 10 kHz: 79 dBµV (quasi-peak)
    Filter attenuation applied: 40 dB (LC input filter, typical).
    """
    # dI/dt during switching
    dI_dt = I_switch_A / (t_rise_ns * 1e-9)  # A/s
    L_cable_nH = 50 * cable_len_m  # 50 nH/m typical shielded cable
    V_diff_pk_unfiltered = L_cable_nH * 1e-9 * dI_dt    # V
    filter_attenuation = 1e-2  # 40 dB LC input filter
    V_diff_pk = V_diff_pk_unfiltered * filter_attenuation
    V_diff_dBuV = 20 * np.log10(max(V_diff_pk * 1e6, 1e-9))

    # Common-mode via stray cap, attenuated by CM choke (typ 40 dB at 10 kHz)
    C_total_pF = C_stray_pF + 2200  # + filter cap 2.2 nF
    cm_choke_atten = 1e-2           # 40 dB common-mode choke attenuation
    V_cm = V_dc_V * (C_stray_pF / C_total_pF) * cm_choke_atten
    V_cm_dBuV = 20 * np.log10(V_cm * 1e6) if V_cm > 0 else -np.inf

    CISPR11_A_dBuV = 79.0
    margin_diff = CISPR11_A_dBuV - V_diff_dBuV
    margin_cm   = CISPR11_A_dBuV - V_cm_dBuV

    return {
        "f_sw_kHz": f_sw_kHz,
        "V_diff_pk_V": round(V_diff_pk, 3),
        "V_diff_dBuV": round(V_diff_dBuV, 1),
        "V_cm_dBuV": round(V_cm_dBuV, 1),
        "CISPR11_A_limit_dBuV": CISPR11_A_dBuV,
        "margin_diff_dB": round(margin_diff, 1),
        "margin_cm_dB": round(margin_cm, 1),
        "CE_pass": margin_diff > 6 and margin_cm > 6,  # 6 dB guard
    }


# ── 6. Safety Interlock Logic (IEC 62061 SIL-2) ──────────────────────────────

def safety_plc(
    PFH_sensor: float = 1e-8,   # probability of failure per hour (SIL-2 sensor)
    PFH_logic: float = 5e-9,    # safety PLC
    PFH_actuator: float = 2e-8, # safety relay / drive STO
    mission_years: float = 20,
) -> dict:
    """
    IEC 62061 SIL-2 verification: PFH_total = sum of subsystem PFHs.
    SIL-2 target: 10^-7 ≤ PFH < 10^-6 (per IEC 62061 Table 3).

    Safety functions covered:
      SF1 — Emergency stop: cuts STO (Safe Torque Off) to all drives
      SF2 — Guard door: stops spindle before door can open (t_stop < 1 s)
      SF3 — Overspeed monitor: SLS (Safe Limited Speed) during teach
    """
    PFH_total = PFH_sensor + PFH_logic + PFH_actuator
    # IEC 62061 Table 3: SIL-2 → 10^-7 ≤ PFH < 10^-6
    #                    SIL-3 → 10^-8 ≤ PFH < 10^-7
    if PFH_total < 1e-8:
        sil = 4
    elif PFH_total < 1e-7:
        sil = 3
    elif PFH_total < 1e-6:
        sil = 2
    elif PFH_total < 1e-5:
        sil = 1
    else:
        sil = 0
    sil_ok = sil >= 2  # target SIL-2

    # Proof-test interval from RRF
    RRF = 1 / PFH_total
    PFD_avg = PFH_total * mission_years * 8760 / 2  # simplified

    return {
        "PFH_total": PFH_total,
        "SIL_achieved": sil,
        "SIL2_ok": sil_ok,
        "RRF": round(RRF),
        "PFD_avg": round(PFD_avg, 6),
        "safety_functions": ["SF1_EStop_STO", "SF2_GuardDoor_SLS", "SF3_Overspeed_SLS"],
        "proof_test_interval_yr": 1,
    }


# ── System summary ─────────────────────────────────────────────────────────────

def system_summary() -> dict:
    bus    = dc_bus()
    spndl  = spindle_inverter()
    stage  = xy_stage()
    vision = vision_illumination()
    emi    = emi_budget(V_dc_V=bus["V_dc_V"])
    safety = safety_plc()

    print("=" * 60)
    print("DISCO dicing saw — Electrical System Summary")
    print("=" * 60)

    print(f"\n[DC Bus]")
    print(f"  V_dc = {bus['V_dc_V']} V,  C_min = {bus['C_min_uF']} µF")
    print(f"  {bus['note']}")

    print(f"\n[Spindle Inverter]")
    st = "PASS" if spndl["pass"] else "FAIL"
    print(f"  P_loss = {spndl['P_loss_W']} W,  T_j = {spndl['T_j_C']} °C  [{st}]")
    print(f"  Margin = {spndl['T_j_margin_K']} K to T_j_max")

    print(f"\n[XY Stage (linear motor)]")
    bw_st = "OK" if stage["bandwidth_ok"] else "NG"
    print(f"  F_peak = {stage['F_peak_N']} N,  I_peak = {stage['I_peak_A']} A")
    print(f"  Position-loop BW = {stage['f_position_loop_Hz']} Hz  [{bw_st}]")
    if "notch_freq_Hz" in stage:
        print(f"  Notch @ {stage['notch_freq_Hz']} Hz  ({stage['notch_attenuation_dB']} dB)")

    print(f"\n[Vision Illumination]")
    sync_st = "OK" if vision["strobe_sync_ok"] else "NG — exposure > slot gap"
    print(f"  Exposure = {vision['exposure_us']} µs,  Slot gap = {vision['slot_gap_us']} µs  [{sync_st}]")
    print(f"  LED avg power = {vision['P_avg_W']} W  (peak {vision['P_peak_W']} W)")

    print(f"\n[EMI Budget (CISPR 11 Class A)]")
    emi_st = "PASS" if emi["CE_pass"] else "FAIL"
    print(f"  V_diff = {emi['V_diff_dBuV']} dBµV  (margin {emi['margin_diff_dB']} dB)")
    print(f"  V_cm   = {emi['V_cm_dBuV']} dBµV  (margin {emi['margin_cm_dB']} dB)  [{emi_st}]")

    print(f"\n[Safety PLC — IEC 62061]")
    sil_st = f"SIL-{safety['SIL_achieved']} {'OK' if safety['SIL2_ok'] else 'FAIL'}"
    print(f"  PFH_total = {safety['PFH_total']:.1e}  [{sil_st}]")
    print(f"  Functions: {', '.join(safety['safety_functions'])}")
    print()

    return {
        "dc_bus": bus, "spindle": spndl, "stage": stage,
        "vision": vision, "emi": emi, "safety": safety,
    }


if __name__ == "__main__":
    system_summary()
