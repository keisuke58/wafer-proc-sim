"""
DBG / SDBG (Dicing Before Grinding) process model
DISCO proprietary process: blade dice → backgrind → chip separation.

Process flow:
    DBG:  1) Half-cut blade dicing (groove depth = wafer_thick - grind_target)
          2) Backside grinding → wafer thins until grooves open → chips separate
    SDBG: 1) Stealth dicing (laser modified layer, no groove)
          2) Backside grinding → chips separate at modified layer

Physics:
    Chip strength improvement: Weibull statistics on die edge flaw distribution
    Groove stress concentration: K_t = 1 + 2*sqrt(a/r)  (Inglis)
    Grinding-induced fracture: GFR model (Malkin 1989)
    Die strength (Weibull): P_f = 1 - exp(-(sigma/sigma_0)^m)

Key advantage over conventional (Dice After Grind, DAG):
    - Chip edges NOT exposed to grinding → no grinding-induced edge damage
    - Die strength >> DAG (empirical: 2-4x improvement)
    - Enables ultra-thin chips < 50 µm

Outputs:
    groove_depth_um     required half-cut depth [µm]
    grind_stock_um      material to be removed by grinding [µm]
    K_t                 stress concentration factor at groove root
    sigma_chip_MPa      estimated die fracture strength [MPa]
    P_fail_1pct         stress at 1% failure probability [MPa]
    strength_ratio      DBG/DAG strength improvement ratio

References:
    Disco DBG patent: JP2003-197564
    Schoenfelder et al. (2007) Microelectronics Reliability — DBG die strength
    Malkin (1989) Grinding Technology — grinding fracture model
"""

import sys
import math

DEFAULT = {
    "material":           "Si",     # Si | SiC4H | GaAs
    "process":            "DBG",    # DBG | SDBG
    "wafer_thick_um":    775.0,     # initial wafer thickness [µm]
    "target_thick_um":    50.0,     # final chip thickness [µm]
    "blade_kerf_um":      30.0,     # blade kerf width [µm] (DBG only)
    "groove_radius_um":    5.0,     # groove root radius [µm] (DBG only)
    "die_size_mm":         5.0,     # die size [mm] (square)
    "weibull_m":          10.0,     # Weibull modulus (Si: 8-12)
    "sigma0_MPa":        500.0,     # Weibull characteristic strength [MPa]
    "dag_sigma_MPa":     200.0,     # reference DAG die strength [MPa] for ratio
}

cfg = dict(DEFAULT)
_args = sys.argv[:]
for k in ("wafer_thick_um", "target_thick_um", "blade_kerf_um", "groove_radius_um",
          "die_size_mm", "weibull_m", "sigma0_MPa", "dag_sigma_MPa"):
    tag = "--" + k
    if tag in _args:
        cfg[k] = float(_args[_args.index(tag) + 1])
for k in ("material", "process"):
    tag = "--" + k
    if tag in _args:
        cfg[k] = _args[_args.index(tag) + 1]

MATERIALS = {
    "Si":    {"E": 170e9, "K_Ic": 0.83, "H_GPa": 11.0},
    "SiC4H": {"E": 480e9, "K_Ic": 2.80, "H_GPa": 28.0},
    "GaAs":  {"E":  85e9, "K_Ic": 0.44, "H_GPa":  7.5},
}
mp = MATERIALS[cfg["material"]]

# ── DBG geometry ───────────────────────────────────────────────────────────────
grind_stock   = cfg["wafer_thick_um"] - cfg["target_thick_um"]   # [µm]
groove_depth  = grind_stock + 5.0                                  # slight overlap [µm]
groove_depth  = min(groove_depth, cfg["wafer_thick_um"] * 0.85)   # max 85% of thickness
remaining_web = cfg["wafer_thick_um"] - groove_depth               # [µm]

# Stress concentration at groove root (Inglis elliptical notch)
a_notch = cfg["blade_kerf_um"] / 2.0     # half-width of notch [µm]
r_notch = cfg["groove_radius_um"]         # root radius [µm]
K_t     = 1.0 + 2.0 * math.sqrt(a_notch / r_notch)

# Die strength model (Weibull)
m      = cfg["weibull_m"]
sig0   = cfg["sigma0_MPa"]
# DBG: edge damage limited to blade groove → smaller flaw population
# SDBG: no blade → even smaller flaw population
flaw_factor_dbg  = 0.6   # empirical: DBG edge flaws 60% of DAG
flaw_factor_sdbg = 0.4
flaw_factor = flaw_factor_dbg if cfg["process"] == "DBG" else flaw_factor_sdbg

sig0_eff = sig0 * (1.0 / flaw_factor) ** (1.0 / m)  # effective char. strength

# Characteristic strength and P_fail
def weibull_pf(sigma, sigma0, m):
    return 1.0 - math.exp(-(sigma / sigma0) ** m)

# Stress at 1% failure
sig_1pct = sig0_eff * (-math.log(1.0 - 0.01)) ** (1.0 / m)
# Mean die strength
sig_mean = sig0_eff * math.gamma(1.0 + 1.0 / m)

strength_ratio = sig_mean / cfg["dag_sigma_MPa"]

print("=" * 60)
print("DBG / SDBG Process Model — {}  [{}]".format(cfg["material"], cfg["process"]))
print("=" * 60)
print("Wafer thick     : {:.0f} µm → target: {:.0f} µm".format(
    cfg["wafer_thick_um"], cfg["target_thick_um"]))
if cfg["process"] == "DBG":
    print("Groove depth    : {:.1f} µm  (web remaining: {:.1f} µm)".format(
        groove_depth, remaining_web))
    print("Blade kerf      : {:.1f} µm,  root radius: {:.1f} µm".format(
        cfg["blade_kerf_um"], cfg["groove_radius_um"]))
    print("Stress conc K_t : {:.2f}".format(K_t))
print("Grind stock     : {:.1f} µm".format(grind_stock))
print("-" * 60)
print("Weibull m       : {:.1f},  sigma_0_eff = {:.0f} MPa".format(m, sig0_eff))
print("Mean die strength: {:.0f} MPa".format(sig_mean))
print("P_f=1% stress   : {:.0f} MPa".format(sig_1pct))
print("vs DAG ({:.0f} MPa): {:.1f}x improvement".format(cfg["dag_sigma_MPa"], strength_ratio))
print("=" * 60)
if remaining_web < 5.0 and cfg["process"] == "DBG":
    print("WARN: web < 5 µm — risk of premature chip separation during dicing")
if strength_ratio < 1.5:
    print("INFO: strength improvement < 1.5x — consider SDBG (laser) for better result")
else:
    print("OK: {:.1f}x strength improvement over DAG".format(strength_ratio))
