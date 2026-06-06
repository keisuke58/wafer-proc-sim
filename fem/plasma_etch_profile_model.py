"""
Plasma dicing etch profile model — Bosch process + sidewall geometry
DISCO plasma dicing (Plasma-Therm partnership): deep reactive ion etching (DRIE)
for 2nm-node wafer singulation.

Physics:
    Etch rate:    ER = ER0 * (1 + beta * P_rf / P_ref) * f(aspect_ratio)
    ARDE (aspect-ratio-dependent etching):
                  ER(AR) = ER0 / (1 + (AR / AR_crit)^2)   [Lag model]
    Sidewall angle: theta = 90° - arctan(k_lat / k_vert)
    Undercut:     u = k_lat * t_etch
    Scallop depth: d_scallop = v_etch * t_passivate          [Bosch cycle]
    Mask selectivity: S = ER_Si / ER_mask
    Delamination risk (Low-k): risk = f(sigma_thermal, E_film, h_film)

Outputs:
    ER_um_min       etch rate [µm/min]
    sidewall_deg    sidewall angle from vertical [°]  (target < 1°)
    undercut_um     lateral undercut [µm]
    scallop_nm      Bosch scallop depth [nm]
    T_wafer_C       wafer temperature during etch [°C]
    delam_risk      delamination risk score 0-1 (Low-k BEOL)

References:
    Laermer & Schilp (1996) Bosch process patent
    Marty et al. (2005) J. Micromech. Microeng. — ARDE model
    Plasma-Therm Mosaic DSE — DISCO partnership (2026 2nm target)
    plasma_bosch_model.py in this repo — business/market model
"""

import sys
import math

DEFAULT = {
    "material":         "Si",      # Si | SiC4H | GaN_on_Si
    "process":          "Bosch",   # Bosch | CryoEtch | AtomicLayerEtch
    "trench_width_um":   8.0,      # dicing street width [µm]  (2nm node: ~8 µm)
    "wafer_thick_um":   775.0,     # wafer thickness [µm]
    "P_rf_W":           800.0,     # RF power [W]
    "P_chamber_mTorr":   30.0,     # chamber pressure [mTorr]
    "T_chuck_C":        -20.0,     # chuck (wafer) temperature [°C]
    "t_etch_s":           7.0,     # etch phase duration per Bosch cycle [s]
    "t_passivate_s":      3.0,     # passivation phase duration [s]
    "mask_material":    "SiO2",    # SiO2 | Photoresist | Metal
    "Low_k_BEOL":        True,     # Low-k dielectric present (delamination risk)
}

cfg = dict(DEFAULT)
_args = sys.argv[:]
for k in ("trench_width_um", "wafer_thick_um", "P_rf_W", "P_chamber_mTorr",
          "T_chuck_C", "t_etch_s", "t_passivate_s"):
    tag = "--" + k
    if tag in _args:
        cfg[k] = float(_args[_args.index(tag) + 1])
for k in ("material", "process", "mask_material"):
    tag = "--" + k
    if tag in _args:
        cfg[k] = _args[_args.index(tag) + 1]
if "--Low_k_BEOL" in _args:
    cfg["Low_k_BEOL"] = True

# ── Material etch parameters ───────────────────────────────────────────────────
ETCH = {
    "Si":        {"ER0": 15.0, "AR_crit": 20.0, "k_lat_frac": 0.02, "S_SiO2": 150.0, "S_PR": 80.0},
    "SiC4H":     {"ER0":  2.5, "AR_crit": 10.0, "k_lat_frac": 0.03, "S_SiO2":  30.0, "S_PR": 15.0},
    "GaN_on_Si": {"ER0":  3.0, "AR_crit": 12.0, "k_lat_frac": 0.025,"S_SiO2":  40.0, "S_PR": 20.0},
}
ep = ETCH[cfg["material"]]

# ── Etch profile calculations ──────────────────────────────────────────────────
# Aspect ratio at mid-etch
depth_mid    = cfg["wafer_thick_um"] / 2.0
AR_mid       = depth_mid / cfg["trench_width_um"]

# ARDE: etch rate reduction with increasing AR
ER_base      = ep["ER0"] * (1.0 + 0.3 * (cfg["P_rf_W"] / 800.0))  # RF scaling
ER_arde      = ER_base / (1.0 + (AR_mid / ep["AR_crit"])**2)        # ARDE lag

# Bosch cycle etch per cycle
v_cycle      = ER_arde / 60.0 * cfg["t_etch_s"]                     # [µm/cycle]
scallop_nm   = v_cycle * 1000.0 * (cfg["t_passivate_s"] / cfg["t_etch_s"]) * 0.3  # [nm]

# Sidewall angle & undercut
k_lat        = ER_arde * ep["k_lat_frac"]                            # lateral rate [µm/min]
t_total_min  = cfg["wafer_thick_um"] / ER_arde                       # total etch time [min]
undercut_um  = k_lat * t_total_min                                   # [µm]
sidewall_deg = math.degrees(math.atan(k_lat / ER_arde))              # [°]

# Mask selectivity
S_key = "S_" + cfg["mask_material"] if "S_" + cfg["mask_material"] in ep else "S_SiO2"
S     = ep[S_key]
mask_consumed_um = cfg["wafer_thick_um"] / S

# Wafer temperature (simplified: chuck + plasma heating)
T_wafer = cfg["T_chuck_C"] + cfg["P_rf_W"] * 0.01  # rough estimate

# Low-k delamination risk
if cfg["Low_k_BEOL"]:
    E_lowk     = 10e9    # Young's modulus Low-k [Pa]
    h_lowk     = 0.5e-6  # film thickness [m]
    sigma_th   = E_lowk * 3e-6 * abs(T_wafer - 25.0)  # thermal stress [Pa]
    G_c_lowk   = 4.0     # fracture energy [J/m²]  (low-k: 4-8 J/m²)
    G_applied  = sigma_th**2 * h_lowk / (2.0 * E_lowk)
    delam_risk = min(1.0, G_applied / G_c_lowk)
else:
    delam_risk = 0.0

print("=" * 60)
print("Plasma Etch Profile — {}  [{}]".format(cfg["material"], cfg["process"]))
print("=" * 60)
print("Street width    : {:.1f} µm,  Wafer: {:.0f} µm".format(cfg["trench_width_um"], cfg["wafer_thick_um"]))
print("RF power        : {:.0f} W,  Pressure: {:.0f} mTorr".format(cfg["P_rf_W"], cfg["P_chamber_mTorr"]))
print("Chuck temp      : {:.0f} °C".format(cfg["T_chuck_C"]))
print("-" * 60)
print("Etch rate (base): {:.2f} µm/min".format(ER_base))
print("Etch rate (ARDE): {:.2f} µm/min  (AR={:.1f})".format(ER_arde, AR_mid))
print("Total etch time : {:.1f} min".format(t_total_min))
print("-" * 60)
print("Scallop depth   : {:.1f} nm".format(scallop_nm))
print("Sidewall angle  : {:.2f}°  from vertical  (target < 1°)".format(sidewall_deg))
print("Undercut        : {:.2f} µm".format(undercut_um))
print("Mask consumed   : {:.2f} µm  (selectivity = {:.0f}:1)".format(mask_consumed_um, S))
print("Wafer temp      : {:.0f} °C".format(T_wafer))
if cfg["Low_k_BEOL"]:
    print("Delam risk (Low-k): {:.2f}  {}".format(
        delam_risk, "HIGH" if delam_risk > 0.3 else "LOW"))
print("=" * 60)
if sidewall_deg > 2.0:
    print("WARN: sidewall angle > 2° — increase RF or reduce pressure")
if scallop_nm > 100.0:
    print("WARN: scallop > 100 nm — reduce t_etch or increase t_passivate")
if delam_risk > 0.3:
    print("WARN: Low-k delamination risk HIGH — reduce chuck temp or RF power")
