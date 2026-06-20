"""
Material properties for wafer dicing/grinding simulations.

Sources:
  Si  : Hopcroft et al. (2010) J. Microelectromech. Syst.
  SiC : Zhao et al. (2021) Nanotechnology and Precision Engineering
        Wang et al. (2020) Int J Mach Tools Manuf (D-P params)
        Huang et al. (2021) Ceram Int (fracture params)
  GaN : Levinshtein et al. (2001) Properties of Advanced Semiconductor Materials
  Blade: typical values from dicing literature

Drucker-Prager parameters (ceramics machining):
  dp_friction_angle : β in degrees (linear D-P criterion)
  dp_cohesion_Pa    : initial yield stress at zero mean stress (compression hardening)
  dp_dilation_angle : ψ in degrees (0 = non-associated flow, standard for ceramics)
  K_dp              : yield stress ratio (triaxial tension / triaxial compression)
                      Use 1.0 for symmetric (simplification for ceramics)

Fracture strain table (triaxiality η = σ_m/σ̄):
  At η = +1/3 (uniaxial tension)  → very small ε_f (brittle)
  At η = -1/3 (uniaxial compress) → larger ε_f (ductile mode in contact zone)
  At η = -∞ (hydrostatic compress) → no fracture (ceramics survive)
"""

import math

def _sic_fracture_table(eps_f_tension, n_pts=7):
    """
    Build triaxiality-dependent fracture strain table for SiC.
    Based on Drucker-Prager ductile damage criterion.
    Table format: ((fracture_strain, triaxiality, strain_rate), ...)
    """
    # Ceramics: fracture strain increases strongly under compression
    # (ductile-brittle transition driven by pressure)
    entries = [
        (eps_f_tension * 50,  -2.0, 0.0),   # deep compression (no fracture)
        (eps_f_tension * 15,  -1.0, 0.0),   # high compression
        (eps_f_tension * 5,   -1/3, 0.0),   # uniaxial compression
        (eps_f_tension * 2,    0.0, 0.0),   # shear dominated
        (eps_f_tension,        1/3, 0.0),   # uniaxial tension ← key value
        (eps_f_tension * 0.5,  2/3, 0.0),   # biaxial tension
        (eps_f_tension * 0.25, 1.0, 0.0),   # triaxial tension (most brittle)
    ]
    return tuple(entries)


# ── Silicon (Si) ───────────────────────────────────────────────────────────────
Si = {
    "name":              "Silicon",
    # Grinding specific force coefficient Ct [N/m] for Ft/b = Ct*(vf/vs)^0.6*ap^0.4
    # Ref: Malkin & Guo (2008) Grinding Technology, Ch.3 (Si backgrind)
    "C_t_grinding":      1200.0,
    "density":           2330.0,      # kg/m³
    "E":                 130e9,       # Pa  [100] direction
    "nu":                0.28,
    "K_Ic":              0.83e6,      # Pa·m^0.5
    "H_v":               10.0e9,      # Pa  Vickers hardness (NOT fracture stress)
    # Actual strength values
    "sigma_tensile":     150e6,       # Pa  tensile fracture strength
    "sigma_compress":    1000e6,      # Pa  compressive strength
    "sigma_flex":        200e6,       # Pa  flexural strength
    # Drucker-Prager (ceramics machining, Wang 2020 adapted for Si)
    "dp_friction_angle": 55.0,        # degrees β
    "dp_cohesion_Pa":    500e6,       # Pa  initial yield stress under compression
    "dp_dilation_angle": 0.0,         # degrees ψ (non-associated)
    "K_dp":              1.0,
    "eps_fracture":      3e-4,        # fracture strain at η=1/3 (uniaxial tension)
    # Thermal
    "k_thermal":         148.0,       # W/(m·K)
    "alpha_thermal":     2.6e-6,      # 1/K
    "Cp":                700.0,       # J/(kg·K)
}
Si["fracture_table"] = _sic_fracture_table(Si["eps_fracture"])
Si["G_c"] = Si["K_Ic"]**2 / Si["E"]   # fracture energy [J/m²]


# ── Silicon Carbide (4H-SiC) ───────────────────────────────────────────────────
SiC = {
    "name":              "4H-SiC",
    # SiC is ~3× harder than Si → higher grinding forces
    "C_t_grinding":      3500.0,
    "density":           3210.0,      # kg/m³
    "E":                 448e9,       # Pa
    "nu":                0.21,
    "K_Ic":              2.8e6,       # Pa·m^0.5
    "H_v":               25.0e9,      # Pa  Vickers hardness (NOT fracture stress)
    # Actual strength values (Huang 2021, Zhang 2020)
    "sigma_tensile":     350e6,       # Pa  tensile fracture strength
    "sigma_compress":    3000e6,      # Pa  compressive strength (~10× tensile)
    "sigma_flex":        500e6,       # Pa  flexural strength
    # Drucker-Prager (Wang et al. 2020, calibrated for SiC machining)
    "dp_friction_angle": 68.8,        # degrees β
    "dp_cohesion_Pa":    1000e6,      # Pa  initial yield stress under zero pressure
    "dp_dilation_angle": 0.0,         # degrees ψ (non-associated flow)
    "K_dp":              1.0,
    "eps_fracture":      2e-4,        # fracture strain at η=1/3
    # Thermal
    "k_thermal":         370.0,       # W/(m·K)
    "alpha_thermal":     4.2e-6,      # 1/K
    "Cp":                750.0,       # J/(kg·K)
}
SiC["fracture_table"] = _sic_fracture_table(SiC["eps_fracture"])
SiC["G_c"] = SiC["K_Ic"]**2 / SiC["E"]


# ── Gallium Nitride (GaN) ──────────────────────────────────────────────────────
GaN = {
    "name":              "GaN",
    "C_t_grinding":      2000.0,
    "density":           6150.0,
    "E":                 295e9,
    "nu":                0.23,
    "K_Ic":              0.9e6,
    "H_v":               12.0e9,
    "sigma_tensile":     200e6,
    "sigma_compress":    1500e6,
    "sigma_flex":        280e6,
    "dp_friction_angle": 60.0,
    "dp_cohesion_Pa":    600e6,
    "dp_dilation_angle": 0.0,
    "K_dp":              1.0,
    "eps_fracture":      1.5e-4,
    "k_thermal":         130.0,
    "alpha_thermal":     5.6e-6,
    "Cp":                490.0,
}
GaN["fracture_table"] = _sic_fracture_table(GaN["eps_fracture"])
GaN["G_c"] = GaN["K_Ic"]**2 / GaN["E"]


# ── Diamond dicing blade (resin bond) ─────────────────────────────────────────
BladeResin = {
    "name":         "Resin-bond blade",
    "density":      3500.0,
    "E":            60e9,
    "nu":           0.25,
    "grit_size_um": 2.0,
    "concentration": 75,
}

# ── Damaged Si (ground surface layer) ─────────────────────────────────────────
# Microcrack network from grinding reduces effective stiffness.
# E_damaged / E_bulk ≈ 0.3–0.7 depending on grit/depth.
# Ref: Chen & Wolf (2003) Semicond. Sci. Technol. 18:261
# Default factor 0.5 used; override via run_config "E_damage_factor".
import copy as _copy
Si_damaged = _copy.copy(Si)
Si_damaged["name"]          = "Silicon_damaged"
Si_damaged["E"]             = Si["E"] * 0.5
Si_damaged["C_t_grinding"]  = Si["C_t_grinding"]  # unchanged

# ── Diamond (semiconductor grade, CVD) ────────────────────────────────────────
# Refs: Isberg et al. (2002) Science 297:1670 (carrier mobility)
#       Field et al. (1992) The Properties of Natural and Synthetic Diamond
#       Gaukroger et al. (2008) Diamond Relat. Mater. 17:262 (K_Ic)
Diamond = {
    "name":              "Diamond",
    "C_t_grinding":      8000.0,    # N/m — estimated (>>SiC, blade machining almost impossible)
    "density":           3515.0,    # kg/m³
    "E":                 1050e9,    # Pa — highest known solid
    "nu":                0.07,
    "K_Ic":              3.4e6,     # Pa·m^0.5 — Gaukroger 2008
    "H_v":               100.0e9,  # Pa — ~100 GPa, hardest known material
    "sigma_tensile":     2800e6,   # Pa — theoretical; practical ~1–3 GPa
    "sigma_compress":    110e9,    # Pa — compressive
    "sigma_flex":        1000e6,   # Pa
    "dp_friction_angle": 75.0,     # degrees β (higher than SiC)
    "dp_cohesion_Pa":    3000e6,   # Pa
    "dp_dilation_angle": 0.0,
    "K_dp":              1.0,
    "eps_fracture":      1e-4,     # very brittle
    # Thermal — exceptional
    "k_thermal":         2200.0,   # W/(m·K) — ~5× SiC, highest of all solids
    "alpha_thermal":     1.0e-6,   # 1/K — very low CTE
    "Cp":                502.0,    # J/(kg·K)
    # Optical — transparent to IR, UV-absorbing below 225nm
    "bandgap_eV":        5.47,     # indirect bandgap → transparent at 1064nm (IR)
    "laser_wavelength_nm_min": 193, # ArF excimer → first usable ablation wavelength
}
Diamond["G_c"] = Diamond["K_Ic"]**2 / Diamond["E"]
Diamond["fracture_table"] = _sic_fracture_table(Diamond["eps_fracture"])


# ── Gallium Oxide (β-Ga₂O₃) — next-gen ultra-wide bandgap ──────────────────
# Refs: Pearton et al. (2018) Appl. Phys. Rev. 5:011301
Ga2O3 = {
    "name":              "beta-Ga2O3",
    "C_t_grinding":      3500.0,
    "density":           5950.0,    # kg/m³
    "E":                 261e9,     # Pa
    "nu":                0.25,
    "K_Ic":              0.5e6,     # Pa·m^0.5 — very brittle
    "H_v":               12.0e9,
    "sigma_tensile":     100e6,
    "sigma_compress":    1500e6,
    "sigma_flex":        150e6,
    "dp_friction_angle": 60.0,
    "dp_cohesion_Pa":    800e6,
    "dp_dilation_angle": 0.0,
    "K_dp":              1.0,
    "eps_fracture":      2e-4,
    "k_thermal":         10.0,     # W/(m·K) — low (cleavage problem)
    "alpha_thermal":     5.0e-6,
    "Cp":                490.0,
    "bandgap_eV":        4.8,
}
Ga2O3["G_c"] = Ga2O3["K_Ic"]**2 / Ga2O3["E"]
Ga2O3["fracture_table"] = _sic_fracture_table(Ga2O3["eps_fracture"])


# ── Glass substrate (panel-level / glass-core packaging) ──────────────────────
# Refs: Corning/Schott panel glass; fused-silica/borosilicate typical.
# Low CTE (Si-matched), low E (warps more), brittle (low K_Ic, flaw-limited
# strength) → hard to thin and dice. Intel-pushed glass-core substrates.
Glass = {
    "name":              "Glass",
    "C_t_grinding":      1000.0,
    "density":           2230.0,      # kg/m³
    "E":                 74e9,        # Pa (lower than Si → more warp)
    "nu":                0.20,
    "K_Ic":              0.75e6,      # Pa·m^0.5 — brittle
    "H_v":               6.0e9,       # Pa
    "sigma_tensile":     50e6,        # Pa — flaw-limited
    "sigma_compress":    1100e6,      # Pa
    "sigma_flex":        70e6,        # Pa
    "dp_friction_angle": 55.0,
    "dp_cohesion_Pa":    400e6,
    "dp_dilation_angle": 0.0,
    "K_dp":              1.0,
    "eps_fracture":      1.5e-4,
    "k_thermal":         1.1,         # W/(m·K) — poor (laser heat localizes)
    "alpha_thermal":     3.2e-6,      # 1/K — Si-matched (the packaging appeal)
    "Cp":                830.0,
}
Glass["G_c"] = Glass["K_Ic"]**2 / Glass["E"]
Glass["fracture_table"] = _sic_fracture_table(Glass["eps_fracture"])


ALL_MATERIALS = {
    "Si": Si, "SiC": SiC, "GaN": GaN,
    "Diamond": Diamond, "Ga2O3": Ga2O3, "Glass": Glass,
    "Si_damaged": Si_damaged,
    "BladeResin": BladeResin,
}
