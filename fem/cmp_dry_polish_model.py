"""
CMP & Dry Polishing model — Preston equation + damage layer removal
DISCO polishing technology: post-grinding damage layer elimination.

Physics:
    CMP (Wet):  MRR = k_p * P * v_rel          [Preston 1927]
                Chemical term: MRR_chem = k_ox * exp(-Ea / kT)  [Arrhenius]
                Total: MRR = (k_p * P * v_rel) + MRR_chem
    Dry Polish: Abrasive-only; MRR = k_dry * P * v_rel  (no chemical term)
                Surface roughness evolution: Ra(t) = Ra0 * exp(-k_Ra * t)

Materials supported: SiC4H, Si, Sapphire, GaN, SiO2

Outputs:
    MRR_nm_min      material removal rate [nm/min]
    T_polish_min    time to remove damage layer [min]
    Ra_final_nm     estimated final surface roughness [nm]
    T_surface_C     surface temperature estimate [°C]

References:
    Preston (1927) J. Soc. Glass Technol.
    Shi & Sandhu (1995) — modified Preston for CMP
    Hotta et al. (2012) Ultraprecision CMP for SiC/GaN/Sapphire,
        Current Applied Physics 12, S1: SiC k_p = 3.2e-13 Pa⁻¹
    Wang et al. (2024) MRR model for SiC with agglomerated diamond,
        Precision Engineering: MRR up to 36 µm/h @ 27.6 kPa
"""

import sys
import math

# ── Defaults ───────────────────────────────────────────────────────────────────
DEFAULT = {
    "material":        "SiC4H",    # SiC4H | Si | Sapphire | GaN | SiO2
    "process":         "CMP",      # CMP | DryPolish
    "pressure_kPa":    27.6,       # polishing pressure [kPa]
    "v_rel_m_s":        1.5,       # relative velocity pad-wafer [m/s]
    "T_slurry_C":       25.0,      # slurry / ambient temperature [°C]
    "damage_depth_um":   2.0,      # post-grinding damage layer depth [µm]
    "Ra0_nm":           50.0,      # initial surface roughness Ra [nm]
    "Ra_target_nm":      0.5,      # target Ra [nm] (EPI-ready: <0.5 nm)
}

cfg = dict(DEFAULT)
_args = sys.argv[:]
for k in ("pressure_kPa", "v_rel_m_s", "T_slurry_C", "damage_depth_um", "Ra0_nm", "Ra_target_nm"):
    tag = "--" + k
    if tag in _args:
        cfg[k] = float(_args[_args.index(tag) + 1])
for k in ("material", "process"):
    tag = "--" + k
    if tag in _args:
        cfg[k] = _args[_args.index(tag) + 1]

# ── Material database ──────────────────────────────────────────────────────────
# k_p: Preston coefficient [Pa⁻¹], k_ox: chemical rate [nm/min], Ea: activation [eV]
# k_Ra: roughness decay constant [min⁻¹], k_dry: dry abrasive coeff [Pa⁻¹]
MATERIALS = {
    "SiC4H":   {"k_p": 3.2e-13, "k_ox": 0.05,  "Ea": 0.45, "k_Ra": 0.08, "k_dry": 1.5e-13, "MRR_ref_nm_min": 600.0},
    "Si":      {"k_p": 8.0e-13, "k_ox": 2.00,  "Ea": 0.30, "k_Ra": 0.20, "k_dry": 4.0e-13, "MRR_ref_nm_min": 300.0},
    "Sapphire":{"k_p": 1.8e-13, "k_ox": 0.03,  "Ea": 0.60, "k_Ra": 0.05, "k_dry": 0.8e-13, "MRR_ref_nm_min": 37.0},
    "GaN":     {"k_p": 0.5e-13, "k_ox": 0.30,  "Ea": 0.55, "k_Ra": 0.04, "k_dry": 0.3e-13, "MRR_ref_nm_min": 0.3},
    "SiO2":    {"k_p": 1.5e-12, "k_ox": 5.00,  "Ea": 0.25, "k_Ra": 0.40, "k_dry": 5.0e-13, "MRR_ref_nm_min": 200.0},
}
mp = MATERIALS[cfg["material"]]

# ── Compute MRR ────────────────────────────────────────────────────────────────
P   = cfg["pressure_kPa"] * 1e3        # [Pa]
v   = cfg["v_rel_m_s"]                 # [m/s]
T_K = cfg["T_slurry_C"] + 273.15
kB  = 8.617e-5                         # Boltzmann [eV/K]

if cfg["process"] == "CMP":
    MRR_mech = mp["k_p"] * P * v * 1e9 * 60.0        # [nm/min]
    MRR_chem = mp["k_ox"] * math.exp(-mp["Ea"] / (kB * T_K))
    MRR = MRR_mech + MRR_chem
else:  # DryPolish
    MRR = mp["k_dry"] * P * v * 1e9 * 60.0
    MRR_mech = MRR
    MRR_chem = 0.0

# Time to remove damage layer
T_polish = (cfg["damage_depth_um"] * 1e3) / MRR if MRR > 0 else float("inf")

# Roughness evolution: Ra(t) = Ra_target + (Ra0 - Ra_target)*exp(-k_Ra * t)
Ra0 = cfg["Ra0_nm"]
Ra_t = cfg["Ra_target_nm"]
if Ra0 > Ra_t and mp["k_Ra"] > 0:
    T_Ra = -math.log((Ra_t - Ra0 * 0.01) / (Ra_t - Ra0)) / mp["k_Ra"] \
           if Ra_t > Ra0 * 0.01 else float("inf")
    T_Ra = max(0.0, -math.log(max(1e-9, (Ra_t) / Ra0)) / mp["k_Ra"])
else:
    T_Ra = 0.0

T_total = max(T_polish, T_Ra)

# Surface temperature (frictional heating estimate)
mu_pad = 0.3
q_fric = mu_pad * P * v          # [W/m²]
k_w    = {"SiC4H": 490, "Si": 148, "Sapphire": 40, "GaN": 130, "SiO2": 1.4}[cfg["material"]]
T_rise = q_fric * 1e-3 / k_w     # rough estimate [°C]
T_surf = cfg["T_slurry_C"] + T_rise

print("=" * 60)
print("CMP / Dry Polishing Model — {}  [{}]".format(cfg["material"], cfg["process"]))
print("=" * 60)
print("Pressure        : {:.1f} kPa".format(cfg["pressure_kPa"]))
print("Relative vel.   : {:.2f} m/s".format(v))
print("Slurry temp     : {:.1f} °C".format(cfg["T_slurry_C"]))
print("-" * 60)
print("MRR (mech)      : {:.2f} nm/min".format(MRR_mech))
print("MRR (chem)      : {:.2f} nm/min".format(MRR_chem))
print("MRR (total)     : {:.2f} nm/min  [{:.2f} µm/h]".format(MRR, MRR * 60 / 1000))
print("Reference MRR   : {:.2f} nm/min  (literature)".format(mp["MRR_ref_nm_min"]))
print("-" * 60)
print("Damage layer    : {:.1f} µm  → removal time: {:.1f} min".format(cfg["damage_depth_um"], T_polish))
print("Ra: {:.1f} → {:.1f} nm  → Ra time: {:.1f} min".format(Ra0, Ra_t, T_Ra))
print("Total polish time: {:.1f} min".format(T_total))
print("Surface temp    : {:.1f} °C".format(T_surf))
print("=" * 60)
if cfg["material"] in ("SiC4H", "Sapphire", "GaN"):
    print("NOTE: Hard material — MRR is low. Consider:")
    print("  - Elevated pressure (> 30 kPa)")
    print("  - Diamond abrasive slurry (agglomerated 7-10 µm)")
    print("  - Higher temperature (> 60°C) to boost k_ox")
