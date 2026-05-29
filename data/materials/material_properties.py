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

ALL_MATERIALS = {"Si": Si, "SiC": SiC, "GaN": GaN, "BladeResin": BladeResin}
