"""
Material properties for wafer dicing/grinding simulations.
Sources:
  - SiC: Zhao et al. (2021), Nanotechnology and Precision Engineering
  - Si:  Hopcroft et al. (2010), J. Microelectromech. Syst.
  - Blade bond (resin): typical values from dicing literature
"""

# ── Silicon (Si) ──────────────────────────────────────────────────
Si = {
    "name": "Silicon",
    "density":        2330.0,   # kg/m^3
    "E":              130e9,    # Pa (Young's modulus, [100] direction)
    "nu":             0.28,     # Poisson's ratio
    "K_Ic":          0.83e6,   # Pa·m^0.5 (fracture toughness)
    "H_v":           10.0e9,   # Pa (Vickers hardness)
    "sigma_y":       7.0e9,    # Pa (yield/fracture stress, brittle)
    "k_thermal":     148.0,    # W/(m·K)
    "alpha_thermal": 2.6e-6,   # 1/K (CTE)
}

# ── Silicon Carbide (SiC, 4H polytype) ───────────────────────────
SiC = {
    "name": "4H-SiC",
    "density":        3210.0,   # kg/m^3
    "E":              448e9,    # Pa
    "nu":             0.21,
    "K_Ic":          2.8e6,    # Pa·m^0.5
    "H_v":           25.0e9,   # Pa
    "sigma_y":       21.0e9,   # Pa (brittle fracture stress)
    "k_thermal":     370.0,    # W/(m·K)
    "alpha_thermal": 4.2e-6,   # 1/K
}

# ── Gallium Nitride (GaN) ─────────────────────────────────────────
GaN = {
    "name": "GaN",
    "density":        6150.0,
    "E":              295e9,
    "nu":             0.23,
    "K_Ic":          0.9e6,
    "H_v":           12.0e9,
    "sigma_y":       8.5e9,
    "k_thermal":     130.0,
    "alpha_thermal": 5.6e-6,
}

# ── Diamond dicing blade (resin bond, typical) ───────────────────
BladeResin = {
    "name": "Resin-bond blade",
    "density":        3500.0,
    "E":              60e9,
    "nu":             0.25,
    "grit_size_um":  2.0,      # µm (diamond grit)
    "concentration": 75,       # % diamond concentration
}

ALL_MATERIALS = {"Si": Si, "SiC": SiC, "GaN": GaN, "BladeResin": BladeResin}
