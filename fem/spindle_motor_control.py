"""
Spindle motor electrical model — DISCO dicing saw spindle control
PMSM (Permanent Magnet Synchronous Motor) + FOC (Field-Oriented Control)

Electrical elements covered:
    - PMSM dq-axis model (torque/flux equations)
    - FOC with PI current controllers
    - Blade load torque estimation from cutting force
    - Spindle speed ripple under intermittent blade contact
    - Power draw vs. cutting condition

Physics:
    dq model:   v_d = R*i_d + L_d*di_d/dt - omega*L_q*i_q
                v_q = R*i_q + L_q*di_q/dt + omega*(L_d*i_d + psi_f)
    Torque:     T_e = 1.5 * p * (psi_f*i_q + (L_d-L_q)*i_d*i_q)
    Mechanics:  J*d_omega/dt = T_e - T_load - B*omega
    Load:       T_load = F_tangential * r_blade

Outputs:
    omega_rpm       spindle speed [rpm]
    T_e_Nm          electromagnetic torque [Nm]
    P_elec_W        electrical power draw [W]
    speed_ripple_pct speed ripple under blade contact [%]
    i_q_A           q-axis (torque) current [A]

References:
    Vas (1998) Sensorless Vector and Direct Torque Control
    DISCO DFD6361 spec: 60,000 rpm, 400 W spindle motor
"""

import sys
import math

DEFAULT = {
    "omega_ref_rpm":  30000.0,   # reference spindle speed [rpm]
    "R_ohm":            0.5,     # stator resistance [Ω]
    "L_d_mH":           1.2,     # d-axis inductance [mH]
    "L_q_mH":           1.4,     # q-axis inductance [mH]
    "psi_f_Wb":         0.08,    # permanent magnet flux linkage [Wb]
    "p_pairs":          2,       # pole pairs
    "J_kgm2":           5e-5,    # rotor + blade inertia [kg·m²]
    "B_Nms":            1e-4,    # viscous damping [N·m·s]
    "r_blade_mm":      55.0,     # blade radius [mm]
    "F_cut_N":         15.0,     # tangential cutting force [N]
    "Kp_i":             5.0,     # PI current controller proportional gain
    "Ki_i":           200.0,     # PI current controller integral gain
    "dt_us":            10.0,    # simulation timestep [µs]
    "t_sim_ms":         50.0,    # simulation duration [ms]
}

cfg = dict(DEFAULT)
_args = sys.argv[:]
for k in ("omega_ref_rpm", "R_ohm", "L_d_mH", "L_q_mH", "psi_f_Wb",
          "r_blade_mm", "F_cut_N", "Kp_i", "Ki_i", "dt_us", "t_sim_ms"):
    tag = "--" + k
    if tag in _args:
        cfg[k] = float(_args[_args.index(tag) + 1])

# ── Motor parameters ───────────────────────────────────────────────────────────
R      = cfg["R_ohm"]
Ld     = cfg["L_d_mH"] * 1e-3
Lq     = cfg["L_q_mH"] * 1e-3
psi_f  = cfg["psi_f_Wb"]
p      = int(cfg["p_pairs"])
J      = cfg["J_kgm2"]
B      = cfg["B_Nms"]
r      = cfg["r_blade_mm"] * 1e-3

omega_ref = cfg["omega_ref_rpm"] * 2 * math.pi / 60.0   # [rad/s]
T_load    = cfg["F_cut_N"] * r                           # load torque [Nm]

# ── Steady-state analytical solution ──────────────────────────────────────────
# At steady state d/dt = 0, i_d = 0 (MTPA for non-salient):
i_q_ss  = T_load / (1.5 * p * psi_f)
v_q_ss  = R * i_q_ss + omega_ref * p * psi_f
v_d_ss  = -omega_ref * p * Lq * i_q_ss
v_mag   = math.sqrt(v_q_ss**2 + v_d_ss**2)
T_e_ss  = 1.5 * p * psi_f * i_q_ss
P_elec  = 1.5 * (v_d_ss * 0.0 + v_q_ss * i_q_ss)   # i_d=0

# Speed ripple (due to intermittent blade-workpiece contact)
# Simplified: delta_omega / omega = T_ripple / (J * omega * f_contact)
n_teeth     = 1                         # blade acts as single-tooth cutter
f_contact   = cfg["omega_ref_rpm"] / 60.0 * n_teeth
delta_T     = T_load * 0.5             # 50% torque ripple assumption
delta_omega = delta_T / (J * (2 * math.pi * f_contact))
ripple_pct  = (delta_omega / omega_ref) * 100.0

# Simple Euler time-domain simulation (coarse)
dt    = cfg["dt_us"] * 1e-6
N     = int(cfg["t_sim_ms"] * 1e-3 / dt)
omega = omega_ref * 0.9    # start at 90%
i_q   = 0.0
i_d   = 0.0
int_iq = 0.0

omega_log = []
for k_step in range(min(N, 5000)):
    t = k_step * dt
    # Load: blade contacts every revolution
    in_contact = (t * cfg["omega_ref_rpm"] / 60.0 % 1.0) < 0.3
    T_l = T_load if in_contact else T_load * 0.1

    # Simple speed PI → i_q* reference
    err_omega = omega_ref - omega
    i_q_ref   = cfg["Kp_i"] * err_omega * 0.01

    # Current PI
    err_iq    = i_q_ref - i_q
    int_iq   += err_iq * dt
    v_q       = cfg["Kp_i"] * err_iq + cfg["Ki_i"] * int_iq
    v_d       = 0.0

    # dq dynamics
    di_d = (v_d - R * i_d + omega * p * Lq * i_q) / Ld
    di_q = (v_q - R * i_q - omega * p * (Ld * i_d + psi_f)) / Lq
    i_d += di_d * dt
    i_q += di_q * dt

    T_e   = 1.5 * p * (psi_f * i_q + (Ld - Lq) * i_d * i_q)
    domega = (T_e - T_l - B * omega) / J
    omega += domega * dt
    omega_log.append(omega)

omega_final = omega_log[-1] if omega_log else omega_ref
omega_min   = min(omega_log) if omega_log else omega_ref
omega_max   = max(omega_log) if omega_log else omega_ref
sim_ripple  = (omega_max - omega_min) / omega_ref * 100.0

print("=" * 60)
print("Spindle Motor FOC Model — PMSM")
print("=" * 60)
print("Target speed    : {:.0f} rpm  ({:.0f} rad/s)".format(cfg["omega_ref_rpm"], omega_ref))
print("Blade radius    : {:.1f} mm,  F_cut = {:.1f} N".format(cfg["r_blade_mm"], cfg["F_cut_N"]))
print("-" * 60)
print("Load torque     : {:.4f} Nm".format(T_load))
print("i_q (steady)    : {:.3f} A".format(i_q_ss))
print("V_dc needed     : {:.1f} V".format(v_mag * 1.15))   # 15% margin
print("P_electrical    : {:.1f} W".format(P_elec))
print("Speed ripple    : {:.3f} % (analytical)".format(ripple_pct))
print("Speed ripple    : {:.3f} % (sim, {:.1f} ms)".format(sim_ripple, cfg["t_sim_ms"]))
print("Final speed     : {:.0f} rpm".format(omega_final * 60 / (2 * math.pi)))
print("=" * 60)
if P_elec > 400:
    print("WARN: P > 400 W — exceeds typical DISCO spindle rating")
if sim_ripple > 1.0:
    print("WARN: speed ripple > 1% — increase J or tune PI gains")
