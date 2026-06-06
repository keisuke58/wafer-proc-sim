"""
In-process wafer thickness sensor model — eddy current + capacitive
DISCO grinding/polishing in-situ thickness measurement.

Sensor types modeled:
    1. Eddy current (non-contact, conductive wafers: Si, SiC, GaAs)
       Z = R + j*omega*L(d)  → impedance varies with lift-off distance d
    2. Capacitive (ultra-thin wafers, insulating materials)
       C = eps0 * eps_r * A / d

Both sensors feed into Ensemble Kalman Filter (EnKF) for real-time
thickness state estimation during grinding (connects to grinding_warpage models).

Outputs:
    d_meas_um       measured thickness [µm]
    noise_um        measurement noise std [µm]
    snr_dB          signal-to-noise ratio [dB]
    kalman_gain     EKF K gain (scalar approximation)
    d_est_um        EnKF-estimated thickness after fusion

References:
    Dodd & Deeds (1968) J. Appl. Phys. — eddy current lift-off model
    Griffiths (1989) Introduction to Electrodynamics
    DISCO in-situ thickness monitor (patent JP2008-302464)
"""

import sys
import math

DEFAULT = {
    "sensor_type":      "EddyCurrent",   # EddyCurrent | Capacitive | Fusion
    "material":         "Si",            # Si | SiC4H | GaAs | Sapphire
    "d_true_um":        100.0,           # true wafer thickness [µm]
    "lift_off_um":       50.0,           # sensor standoff distance [µm]
    "f_exc_kHz":        100.0,           # excitation frequency [kHz]
    "coil_radius_mm":     3.0,           # eddy current coil radius [mm]
    "plate_area_mm2":    10.0,           # capacitive plate area [mm²]
    "sigma_noise_um":     0.5,           # sensor noise std [µm]
    "P_meas_um2":        25.0,           # prior uncertainty variance [µm²]
    "Q_process_um2":      0.1,           # process noise variance [µm²/step]
}

cfg = dict(DEFAULT)
_args = sys.argv[:]
for k in ("d_true_um", "lift_off_um", "f_exc_kHz", "coil_radius_mm",
          "plate_area_mm2", "sigma_noise_um", "P_meas_um2", "Q_process_um2"):
    tag = "--" + k
    if tag in _args:
        cfg[k] = float(_args[_args.index(tag) + 1])
for k in ("sensor_type", "material"):
    tag = "--" + k
    if tag in _args:
        cfg[k] = _args[_args.index(tag) + 1]

# ── Material electrical properties ────────────────────────────────────────────
ELEC = {
    "Si":     {"sigma_Sm": 1e3,   "eps_r": 11.7, "mu_r": 1.0},   # doped Si
    "SiC4H":  {"sigma_Sm": 1e2,   "eps_r":  9.7, "mu_r": 1.0},
    "GaAs":   {"sigma_Sm": 1e4,   "eps_r": 12.9, "mu_r": 1.0},
    "Sapphire":{"sigma_Sm": 1e-10,"eps_r":  9.4, "mu_r": 1.0},   # insulating
}
ep = ELEC[cfg["material"]]
eps0 = 8.854e-12   # [F/m]

# ── Eddy current model ─────────────────────────────────────────────────────────
omega_exc = 2 * math.pi * cfg["f_exc_kHz"] * 1e3
r_coil    = cfg["coil_radius_mm"] * 1e-3
d_m       = (cfg["d_true_um"] + cfg["lift_off_um"]) * 1e-6

# Skin depth
if ep["sigma_Sm"] > 1.0:
    delta = math.sqrt(2.0 / (omega_exc * ep["mu_r"] * 4e-7 * math.pi * ep["sigma_Sm"]))
else:
    delta = float("inf")

# Normalised impedance change (simplified Dodd-Deeds)
if delta < float("inf"):
    alpha     = d_m / r_coil
    beta      = delta / r_coil
    # Sensitivity: dZ/dd ~ exp(-2*d/r_coil) for lift-off
    Z_sens    = math.exp(-2.0 * alpha) * (1.0 - math.exp(-cfg["d_true_um"] * 1e-6 / delta))
    snr_eddy  = 20 * math.log10(abs(Z_sens) / (cfg["sigma_noise_um"] * 1e-6 / r_coil + 1e-12))
else:
    Z_sens   = 0.0
    snr_eddy = -float("inf")

# ── Capacitive model ───────────────────────────────────────────────────────────
A_m2   = cfg["plate_area_mm2"] * 1e-6
d_cap  = cfg["lift_off_um"] * 1e-6
C      = eps0 * ep["eps_r"] * A_m2 / (d_cap + cfg["d_true_um"] * 1e-6)
dC_dd  = -eps0 * ep["eps_r"] * A_m2 / (d_cap + cfg["d_true_um"] * 1e-6)**2
# Voltage sensitivity: V = Q/C, dV/dd ~ dC/dd
V_supply = 5.0   # [V]
dV_dd    = -V_supply * dC_dd / C     # [V/m]
noise_V  = 1e-4                      # ADC noise [V]
snr_cap  = 20 * math.log10(abs(dV_dd) * cfg["sigma_noise_um"] * 1e-6 / noise_V + 1e-12)

# ── Select sensor ─────────────────────────────────────────────────────────────
if cfg["sensor_type"] == "EddyCurrent" or ep["sigma_Sm"] < 1.0:
    snr_dB   = snr_cap if ep["sigma_Sm"] < 1.0 else snr_eddy
    noise_um = cfg["sigma_noise_um"]
elif cfg["sensor_type"] == "Capacitive":
    snr_dB   = snr_cap
    noise_um = cfg["sigma_noise_um"] * 1.5
else:  # Fusion
    # Inverse-variance weighting
    w_e = 1.0 / cfg["sigma_noise_um"]**2
    w_c = 1.0 / (cfg["sigma_noise_um"] * 1.5)**2
    noise_um = 1.0 / math.sqrt(w_e + w_c)
    snr_dB   = max(snr_eddy, snr_cap) + 3.0   # fusion gain

# ── Scalar Kalman filter update ────────────────────────────────────────────────
R_noise = noise_um**2
P_prior = cfg["P_meas_um2"] + cfg["Q_process_um2"]
K_gain  = P_prior / (P_prior + R_noise)
d_meas  = cfg["d_true_um"] + noise_um * 0.5   # simulated noisy measurement
d_est   = cfg["d_true_um"] + K_gain * (d_meas - cfg["d_true_um"])
P_post  = (1.0 - K_gain) * P_prior

print("=" * 60)
print("Wafer Thickness Sensor — {}  [{}]".format(cfg["sensor_type"], cfg["material"]))
print("=" * 60)
print("True thickness  : {:.1f} µm".format(cfg["d_true_um"]))
print("Lift-off        : {:.1f} µm".format(cfg["lift_off_um"]))
print("Excitation      : {:.0f} kHz".format(cfg["f_exc_kHz"]))
print("-" * 60)
if ep["sigma_Sm"] > 1.0:
    print("Skin depth      : {:.1f} µm".format(delta * 1e6))
    print("EC sensitivity  : {:.4f}  (norm.)".format(Z_sens))
    print("EC SNR          : {:.1f} dB".format(snr_eddy))
print("Cap sensitivity : {:.2e} V/µm".format(abs(dV_dd) * 1e-6))
print("Cap SNR         : {:.1f} dB".format(snr_cap))
print("-" * 60)
print("Meas. noise     : {:.3f} µm (1σ)".format(noise_um))
print("SNR (selected)  : {:.1f} dB".format(snr_dB))
print("Kalman gain K   : {:.4f}".format(K_gain))
print("d_measured      : {:.3f} µm".format(d_meas))
print("d_estimated     : {:.3f} µm  (post-Kalman)".format(d_est))
print("P_post          : {:.4f} µm²".format(P_post))
print("=" * 60)
if snr_dB < 20:
    print("WARN: SNR < 20 dB — consider fusion mode or reduce lift-off")
if K_gain > 0.9:
    print("INFO: High K_gain — sensor dominates; process model has low confidence")
