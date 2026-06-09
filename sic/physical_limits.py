"""
Physical Limits of DISCO Precision Machining Processes.

Each process has a hard physical limit derived from first principles:

  Process              Governing physics           Limit mechanism
  ──────────────────   ─────────────────────────   ──────────────────────────
  Blade dicing         Elastic blade deflection    kerf_min ~ 10 µm
  Stealth dicing       Rayleigh depth-of-focus     position accuracy ~ 100 nm
  Wafer thinning       Griffith fracture criterion t_crit ~ 3–8 µm (Si)
  Plasma dicing        Knudsen transport + AR      AR_max ~ 30–50:1
  Diamond machining    UV laser ablation threshold λ_min = 193 nm (ArF)

Usage:
    python sic/physical_limits.py
    python sic/physical_limits.py --plot
    python sic/physical_limits.py --material Diamond
"""

import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.materials.material_properties import Si, SiC, GaN, Diamond, Ga2O3, ALL_MATERIALS


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Blade dicing — minimum kerf from elastic stability
# ═══════════════════════════════════════════════════════════════════════════════

def blade_minimum_kerf(
    cut_depth_um: float = 100.0,
    blade_E_GPa: float = 60.0,         # resin bond blade Young's modulus
    blade_nu: float = 0.25,
    grit_size_um: float = 1.0,         # finest achievable diamond abrasive grit
    sigma_yield_MPa: float = 200.0,    # resin bond compressive yield strength
) -> dict:
    """
    Minimum kerf width from Euler plate buckling under edge compression.

    During dicing, the blade is compressed laterally by the kerf walls.
    Model: thin plate of width a = cut_depth, thickness t, under edge compression.

    Critical buckling stress (simply-supported on both sides):
        σ_crit = π²E / (12(1-ν²)) × (t/a)²

    Minimum blade thickness where σ_crit > σ_yield (blade survives lateral load):
        t_min = a × sqrt(12(1-ν²) × σ_yield / (π² × E))

    Minimum kerf:
        kerf_min = t_min + 2 × grit_protrusion   [grit_protrusion ≈ grit_size/2]
    """
    E     = blade_E_GPa * 1e9
    sy    = sigma_yield_MPa * 1e6
    a     = cut_depth_um * 1e-6

    t_min_m   = a * np.sqrt(12 * (1 - blade_nu**2) * sy / (np.pi**2 * E))
    t_min_um  = t_min_m * 1e6

    kerf_min_um = t_min_um + grit_size_um   # one side grit protrusion

    current_kerf = 23.0
    improvement  = max(0, (current_kerf - kerf_min_um) / current_kerf * 100)

    return {
        "t_blade_min_um":  t_min_um,
        "kerf_min_um":     kerf_min_um,
        "current_kerf_um": current_kerf,
        "improvement_pct": improvement,
        "limiting_factor": "Euler plate buckling under lateral blade compression",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Stealth dicing — Rayleigh depth-of-focus and minimum modified layer
# ═══════════════════════════════════════════════════════════════════════════════

def stealth_dicing_focal_limits(
    wavelength_nm: float = 1064.0,     # Nd:YAG, standard for stealth dicing
    NA: float = 0.85,                  # objective numerical aperture
    n_Si: float = 3.55,                # refractive index of Si at 1064nm
) -> dict:
    """
    Rayleigh criterion for stealth dicing focal precision.

    Minimum focal spot diameter (Abbe):
        d_min = 0.61 × λ / NA   [in medium]

    Rayleigh depth-of-focus (depth over which intensity > 50% of peak):
        z_R = n × λ / (π × NA²)

    Depth positioning accuracy ≈ ± z_R / 2

    Two-photon absorption threshold:
        Stealth dicing uses nonlinear TPA → only at focal volume.
        Minimum modified layer thickness ≈ 2 × z_R (double-sided)
    """
    lam = wavelength_nm * 1e-9

    d_spot_nm   = 0.61 * lam / NA * 1e9            # nm, in air
    d_spot_in_Si_nm = d_spot_nm / n_Si              # shorter in Si

    z_R_nm      = n_Si * lam / (np.pi * NA**2) * 1e9   # Rayleigh length in Si [nm]
    depth_accuracy_nm = z_R_nm / 2

    modified_layer_min_nm = 2 * z_R_nm              # minimum controllable layer

    # Comparison: current modified layer ~10µm thick
    current_layer_um = 10.0

    return {
        "focal_spot_air_nm":        d_spot_nm,
        "focal_spot_Si_nm":         d_spot_in_Si_nm,
        "rayleigh_length_Si_nm":    z_R_nm,
        "depth_accuracy_nm":        depth_accuracy_nm,
        "modified_layer_min_um":    modified_layer_min_nm / 1e3,
        "current_layer_um":         current_layer_um,
        "shrinkage_possible_x":     current_layer_um / (modified_layer_min_nm / 1e3),
        "limiting_factor":          "Rayleigh z_R (depth-of-focus) in Si",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Wafer thinning — Griffith fracture limit
# ═══════════════════════════════════════════════════════════════════════════════

def wafer_thinning_critical_thickness(
    mat: dict = None,
    flaw_size_nm: float = 100.0,       # surface flaw after grinding [nm]
    wafer_diameter_mm: float = 300.0,
    safety_factor: float = 2.0,
) -> dict:
    """
    Critical wafer thickness from Griffith fracture criterion.

    For a half-penny surface crack of depth a:
        σ_frac = K_Ic / sqrt(π × a)

    Handling stress: simply-supported circular plate under own weight.
    Timoshenko plate bending (σ_max at centre, radial direction):
        σ_h(t) = 3(3+ν) ρ g R² / (8 t)     [Pa],   R = D/2

    Note: load q = ρgt [Pa], σ_max = 3(3+ν) q R² / (8 t²) = 3(3+ν) ρg R² / (8t)
    — linear in 1/t, NOT 1/t² (weight itself scales with t).

    Critical thickness where σ_h = σ_frac / SF:
        t_crit = 3(3+ν) ρ g R² × SF / (8 × σ_frac)
    """
    if mat is None:
        mat = Si

    a   = flaw_size_nm * 1e-9
    R   = (wafer_diameter_mm * 1e-3) / 2.0    # radius [m]
    nu  = mat.get("nu", 0.28)
    g   = 9.81

    K_Ic  = mat["K_Ic"]
    rho   = mat["density"]
    sigma_frac = K_Ic / np.sqrt(np.pi * a)

    coeff = 3.0 * (3.0 + nu) * rho * g * R**2 / 8.0   # σ_h = coeff / t

    t_crit = coeff * safety_factor / sigma_frac         # [m]

    def sf(t_um):
        t = t_um * 1e-6
        return sigma_frac / (coeff / t)                 # dimensionless

    sigma_residual_MPa = mat["E"] * mat.get("eps_fracture", 3e-4) * 0.1 / 1e6

    return {
        "material":                 mat["name"],
        "griffith_frac_stress_MPa": sigma_frac / 1e6,
        "residual_stress_MPa":      sigma_residual_MPa,
        "critical_thickness_um":    t_crit * 1e6,
        "current_hbm4_um":          25.0,
        "target_hbm5_um":           10.0,
        "safety_at_25um":           sf(25.0),
        "safety_at_10um":           sf(10.0),
        "safety_at_5um":            sf(5.0),
        "sigma_h_at_25um_MPa":      coeff / 25e-6 / 1e6,
        "limiting_factor":          "Griffith fracture under gravity (Timoshenko plate)",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Plasma dicing — Knudsen transport and aspect ratio limit
# ═══════════════════════════════════════════════════════════════════════════════

def plasma_aspect_ratio_limit(
    trench_width_um: float = 5.0,
    wafer_thickness_um: float = 100.0,
    pressure_mTorr: float = 10.0,
    AR_crit_ref: float = 8.0,     # AR where etch rate = 50% at 10mTorr (Bosch calibrated)
    arde_exp: float = 2.0,        # rate = 1/(1 + (AR/AR_crit)^n)
    min_rate_frac: float = 0.05,  # 5% minimum viable etch rate
) -> dict:
    """
    Plasma dicing AR limit from Aspect Ratio Dependent Etching (ARDE).

    At µm-scale streets and ≤50mTorr, λ_mfp >> trench width → molecular flow.
    The practical AR limit is set by ARDE, not mere Knudsen transport:

      Ion shadowing: ion beam half-angle θ_i limits directionality
      Radical depletion: F/Cl radicals consumed at trench sidewalls

    Empirical ARDE rate model (Gottscho 1992, adapted for Bosch process):
        R(AR) / R₀ = 1 / (1 + (AR / AR_crit)^n)

    AR_crit scales with pressure (higher P → more radicals → higher AR_crit):
        AR_crit(P) = AR_crit_ref × sqrt(P / P_ref)    P_ref = 10 mTorr

    Maximum practical AR (5% minimum rate):
        0.05 = 1 / (1 + (AR_max/AR_crit)^n)
        AR_max = AR_crit × (1/0.05 - 1)^(1/n) = AR_crit × 19^(1/n)

    Calibration: AR_crit_ref=8, n=2 → AR_max≈35 at 10mTorr (matches DISCO specs).
    """
    P_ref  = 10.0
    AR_crit = AR_crit_ref * np.sqrt(max(pressure_mTorr, 1.0) / P_ref)

    # AR_max from analytical inversion
    AR_max = AR_crit * (1.0 / min_rate_frac - 1.0) ** (1.0 / arde_exp)

    current_AR = wafer_thickness_um / trench_width_um
    etch_rate_at_AR = 1.0 / (1.0 + (current_AR / AR_crit) ** arde_exp)

    return {
        "trench_width_um":          trench_width_um,
        "wafer_thickness_um":       wafer_thickness_um,
        "pressure_mTorr":           pressure_mTorr,
        "AR_crit_50pct":            AR_crit,
        "flow_regime":              "molecular",  # always at µm scale, ≤50mTorr
        "current_AR":               current_AR,
        "AR_max_practical":         AR_max,
        "etch_rate_at_current_AR":  etch_rate_at_AR,
        "feasible":                 current_AR < AR_max,
        "limiting_factor":          "ARDE — ion shadowing + radical depletion",
    }


# ── Vectorized ARDE (C++ kernel with NumPy fallback) ─────────────────────────

try:
    from fem import _arde_kernel as _cpp_arde
    _CPP_ARDE = True
except ImportError:
    _CPP_ARDE = False


def plasma_arde_vec(
    trench_width_um,
    wafer_thickness_um,
    pressure_mTorr=10.0,
    AR_crit_ref: float = 8.0,
    arde_exp: float = 2.0,
    min_rate_frac: float = 0.05,
):
    """
    Vectorized ARDE model. All arguments may be scalars or array-likes;
    NumPy broadcasting is applied before dispatch.

    Returns a dict with keys:
        AR_crit, AR_max, current_AR, etch_rate, feasible  (all same-shape arrays)

    Backend: compiled C++ (pybind11) if built, else pure NumPy.
    """
    arrs = np.broadcast_arrays(
        np.asarray(trench_width_um, np.float64),
        np.asarray(wafer_thickness_um, np.float64),
        np.asarray(pressure_mTorr, np.float64),
    )
    shape = arrs[0].shape
    flat  = [np.ascontiguousarray(a).ravel() for a in arrs]
    w, t, p = flat

    if _CPP_ARDE:
        result = _cpp_arde.arde_sweep(w, t, p, AR_crit_ref, arde_exp, min_rate_frac)
        return {k: v.reshape(shape) for k, v in result.items()}

    # NumPy fallback — identical arithmetic.
    P_ref     = 10.0
    p_safe    = np.maximum(p, 1.0)
    AR_crit   = AR_crit_ref * np.sqrt(p_safe / P_ref)
    AR_max    = AR_crit * np.power(1.0 / min_rate_frac - 1.0, 1.0 / arde_exp)
    cur_AR    = t / w
    etch_rate = 1.0 / (1.0 + np.power(cur_AR / AR_crit, arde_exp))
    return {
        "AR_crit":    AR_crit.reshape(shape),
        "AR_max":     AR_max.reshape(shape),
        "current_AR": cur_AR.reshape(shape),
        "etch_rate":  etch_rate.reshape(shape),
        "feasible":   (cur_AR < AR_max).reshape(shape),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Diamond semiconductor — laser ablation wavelength limit
# ═══════════════════════════════════════════════════════════════════════════════

PLANCK_EV_NM = 1239.84  # hc in eV·nm

def diamond_laser_ablation_limit() -> dict:
    """
    Diamond is transparent to IR/visible laser (bandgap = 5.47 eV).
    Ablation requires photon energy > E_gap OR multi-photon absorption.

    Single-photon: λ < hc/E_gap = 1239.84/5.47 = 226.7 nm
    Shortest practical laser: ArF excimer at 193nm (used in EUV litho prep)

    Alternative: pulsed UV laser → two-photon at 355nm (3rd harmonic Nd:YAG)
    2-photon absorption: 2 × 355nm → effective 177.5nm > E_gap → ablation possible

    Kerf quality vs. wavelength:
        Shorter λ → smaller spot → narrower kerf → better quality
        λ = 193nm → spot ~80nm (with NA=0.85) → kerf ~0.2µm theoretical
    """
    E_gap = Diamond["bandgap_eV"]
    lambda_threshold_nm = PLANCK_EV_NM / E_gap

    lasers = {
        "Nd:YAG 1064nm":       (1064, 1064 > lambda_threshold_nm, "IR — transparent, blade dicing needed"),
        "Nd:YAG 532nm (2ω)":   (532,  532  > lambda_threshold_nm, "Green — transparent"),
        "Nd:YAG 355nm (3ω)":   (355,  355  > lambda_threshold_nm, "UV — TPA marginal"),
        "Nd:YAG 266nm (4ω)":   (266,  266  > lambda_threshold_nm, "Deep UV — weak single-photon"),
        "ArF excimer 193nm":   (193,  193  > lambda_threshold_nm, "DUV — single-photon ablation ✅"),
        "F₂ laser 157nm":      (157,  157  > lambda_threshold_nm, "VUV — ablation ✅ (vacuum required)"),
    }

    # Spot size for each laser (NA=0.85 objective)
    NA = 0.85
    for name, (lam, transparent, note) in lasers.items():
        spot_nm = 0.61 * lam / NA

    # Minimum kerf via 193nm ArF
    spot_193nm = 0.61 * 193 / NA
    kerf_min_um = spot_193nm / 1000 * 3  # 3× spot as practical kerf

    return {
        "bandgap_eV":                E_gap,
        "single_photon_threshold_nm": lambda_threshold_nm,
        "lasers":                    lasers,
        "recommended_laser":         "ArF excimer 193nm",
        "spot_size_193nm_nm":        spot_193nm,
        "min_kerf_193nm_um":         kerf_min_um,
        "vs_Si_blade_23um":          f"{23 / kerf_min_um:.1f}× improvement vs Si blade",
        "challenge":                 "ArF optics expensive; vacuum preferred for 157nm",
        "alternative":               "Ultrashort pulse (fs) Nd:YAG 355nm via TPA",
        "limiting_factor":           "Diamond bandgap (5.47 eV) blocks IR/visible lasers",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Process roadmap: current → physical limit
# ═══════════════════════════════════════════════════════════════════════════════

def print_roadmap():
    print("\n" + "═" * 72)
    print("  DISCO Precision Machining — Physical Limits Roadmap")
    print("═" * 72)

    # Blade dicing
    b = blade_minimum_kerf()
    print(f"\n  ① Blade Dicing — Kerf Width")
    print(f"     Current:       {b['current_kerf_um']:.0f} µm")
    print(f"     Physical min:  {b['kerf_min_um']:.1f} µm  (blade stability limit)")
    print(f"     Governing:     {b['limiting_factor']}")
    _bar(b["current_kerf_um"], b["kerf_min_um"], 40.0)

    # Stealth dicing
    s = stealth_dicing_focal_limits()
    print(f"\n  ② Stealth Dicing — Modified Layer Thickness")
    print(f"     Current:       {s['current_layer_um']:.0f} µm")
    print(f"     Physical min:  {s['modified_layer_min_um']:.3f} µm  "
          f"({s['modified_layer_min_um']*1000:.0f} nm)")
    print(f"     Shrink ratio:  {s['shrinkage_possible_x']:.0f}× thinner achievable")
    print(f"     Governing:     {s['limiting_factor']}")
    print(f"     Depth accuracy: ±{s['depth_accuracy_nm']:.0f} nm")
    _bar(s["current_layer_um"], s["modified_layer_min_um"], 15.0)

    # Wafer thinning — multi material
    print(f"\n  ③ Wafer Thinning — Critical Fracture Thickness")
    for mat in [Si, SiC, GaN, Diamond]:
        w = wafer_thinning_critical_thickness(mat)
        sf25 = w["safety_at_25um"]
        sf10 = w["safety_at_10um"]
        sf5  = w["safety_at_5um"]
        flag25 = "✅" if sf25 > 1.5 else "⚠️ "
        flag10 = "✅" if sf10 > 1.5 else ("⚠️ " if sf10 > 1.0 else "❌")
        flag5  = "✅" if sf5  > 1.5 else ("⚠️ " if sf5  > 1.0 else "❌")
        print(f"     {mat['name']:<12s}: t_crit={w['critical_thickness_um']:.1f}µm  "
              f"| 25µm SF={sf25:.1f}{flag25} | 10µm SF={sf10:.1f}{flag10} "
              f"| 5µm SF={sf5:.1f}{flag5}")
    print(f"     Governing:     Griffith fracture + residual grinding stress")

    # Plasma dicing
    p5  = plasma_aspect_ratio_limit(trench_width_um=5.0,  wafer_thickness_um=100.0)
    p2  = plasma_aspect_ratio_limit(trench_width_um=2.0,  wafer_thickness_um=100.0)
    p10 = plasma_aspect_ratio_limit(trench_width_um=10.0, wafer_thickness_um=100.0)
    print(f"\n  ④ Plasma Dicing — Aspect Ratio Limit (depth=100µm, P=10mTorr)")
    print(f"     {'Street width':>14}  {'AR_crit(50%)':>12}  {'Flow':>10}  "
          f"{'Current AR':>10}  {'Max AR':>8}  {'Rate@AR':>8}  {'OK':>4}")
    for p, w in [(p10, 10), (p5, 5), (p2, 2)]:
        feas = "✅" if p["feasible"] else "❌"
        print(f"     {w:>12}µm  {p['AR_crit_50pct']:>12.1f}  "
              f"{p['flow_regime']:>10}  {p['current_AR']:>10.1f}  "
              f"{p['AR_max_practical']:>8.1f}  "
              f"{p['etch_rate_at_current_AR']:>8.1%}  {feas:>4}")
    print(f"     Governing:     {p5['limiting_factor']}")

    # Diamond
    d = diamond_laser_ablation_limit()
    print(f"\n  ⑤ Diamond Semiconductor — Laser Machining")
    print(f"     Bandgap:          {d['bandgap_eV']:.2f} eV  "
          f"→ λ_threshold = {d['single_photon_threshold_nm']:.1f} nm")
    print(f"     Lasers × transparency:")
    for name, (lam, transp, note) in d["lasers"].items():
        sym = "❌ transparent" if transp else "✅ ablates"
        print(f"       {name:<24s} {sym}  ({note})")
    print(f"     Best option:      {d['recommended_laser']}")
    print(f"       Spot size:      {d['spot_size_193nm_nm']:.0f} nm")
    print(f"       Min kerf:       {d['min_kerf_193nm_um']:.3f} µm  "
          f"({d['vs_Si_blade_23um']})")
    print(f"     Governing:        {d['limiting_factor']}")

    print(f"\n{'═'*72}")
    print(f"  Summary: Physical limits are far from current practice.")
    print(f"  The real constraint is HANDLING, not cutting precision.")
    print(f"  → Wafer at 5µm is like handling a sheet of glass 1mm thick")
    print(f"     across a 300mm span — gravity alone risks fracture.")
    print(f"  → TAIKO® edge-ring concept is the engineering answer.")
    print(f"  → AI real-time control (crack monitoring) is next frontier.")
    print(f"{'═'*72}\n")


def _bar(current, limit, scale):
    """ASCII progress bar from current to limit."""
    pct_current = current / scale
    pct_limit   = limit   / scale
    w = 40
    c_pos = int(pct_current * w)
    l_pos = max(0, int(pct_limit * w))
    bar = "░" * l_pos + "▓" * max(0, c_pos - l_pos) + "█" * (w - c_pos)
    print(f"     {limit:.1f}µm {'|' + bar[:w] + '|'} {current:.0f}µm")
    print(f"     {'← limit':<12s}{' ' * l_pos}{'↑ current':>12}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_limits():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available"); return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("DISCO Precision Machining — Physical Limits", fontsize=14, fontweight="bold")

    # 1. Blade kerf vs cut depth
    ax = axes[0, 0]
    depths = np.linspace(10, 400, 80)
    kerfs = [blade_minimum_kerf(d)["kerf_min_um"] for d in depths]
    ax.plot(depths, kerfs, "b-", linewidth=2)
    ax.axhline(23.0, color="orange", linestyle="--", label="Current ~23µm")
    ax.fill_between(depths, kerfs, 23, where=[k < 23 for k in kerfs],
                    alpha=0.2, color="green", label="Improvement zone")
    ax.set_xlabel("Cut depth [µm]")
    ax.set_ylabel("Min kerf width [µm]")
    ax.set_title("Blade Dicing — Min Kerf")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 2. Wafer thinning safety factor
    ax = axes[0, 1]
    thicknesses = np.linspace(2, 50, 100)
    for mat, color in [(Si, "blue"), (SiC, "red"), (GaN, "green"), (Diamond, "purple")]:
        sfs = []
        K_Ic = mat["K_Ic"]
        rho  = mat["density"]
        a    = 100e-9
        D    = 0.3
        sig_f = K_Ic / np.sqrt(np.pi * a)
        for t_um in thicknesses:
            t = t_um * 1e-6
            sig_h = 3 * rho * 9.81 * D**2 / (16 * t**2)
            sfs.append(sig_f / sig_h / 1e6)
        ax.semilogy(thicknesses, np.clip(sfs, 0.1, 1e4), color=color,
                    label=mat["name"], linewidth=2)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, label="SF=1 (fracture)")
    ax.axhline(2.0, color="gray",  linestyle=":",  linewidth=0.8, label="SF=2 (target)")
    ax.axvline(25.0, color="orange", linestyle="--", alpha=0.6, label="HBM4 25µm")
    ax.axvline(10.0, color="red",    linestyle="--", alpha=0.6, label="HBM5 target 10µm")
    ax.set_xlabel("Wafer thickness [µm]")
    ax.set_ylabel("Safety factor")
    ax.set_title("Wafer Thinning — Griffith Safety Factor")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.set_xlim(2, 50)

    # 3. Plasma dicing AR vs pressure
    ax = axes[0, 2]
    pressures = np.linspace(1, 50, 80)
    for width, color, label in [(2, "red", "2µm street"), (5, "blue", "5µm street"),
                                  (10, "green", "10µm street")]:
        ARs = [plasma_aspect_ratio_limit(trench_width_um=width, wafer_thickness_um=100,
                                         pressure_mTorr=p)["AR_max_practical"]
               for p in pressures]
        ax.plot(pressures, np.clip(ARs, 0, 100), color=color, label=label, linewidth=2)
    ax.axhline(20, color="gray", linestyle="--", label="AR=20 (target)")
    ax.set_xlabel("Plasma pressure [mTorr]")
    ax.set_ylabel("Maximum feasible AR")
    ax.set_title("Plasma Dicing — Max Aspect Ratio (ARDE)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 4. Stealth dicing focal precision vs NA
    ax = axes[1, 0]
    NAs = np.linspace(0.4, 0.99, 80)
    for lam, color, label in [(1064, "red", "1064nm (current)"),
                               (532,  "blue", "532nm (2ω)"),
                               (355,  "green", "355nm (3ω)")]:
        z_Rs = [stealth_dicing_focal_limits(lam, na)["rayleigh_length_Si_nm"] / 1e3
                for na in NAs]
        ax.plot(NAs, z_Rs, color=color, label=label, linewidth=2)
    ax.set_xlabel("Objective NA")
    ax.set_ylabel("Rayleigh length z_R in Si [µm]")
    ax.set_title("Stealth Dicing — Focal Depth Precision")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 5. Material hardness ladder
    ax = axes[1, 1]
    mats_plot = [("Si", Si), ("GaN", GaN), ("SiC", SiC), ("Ga2O3", Ga2O3), ("Diamond", Diamond)]
    from data.materials.material_properties import Ga2O3
    mats_plot = [("Si", Si), ("GaN", GaN), ("SiC", SiC), ("4H-SiC", SiC),
                 ("Diamond", Diamond)]
    names  = ["Si", "GaN", "4H-SiC", "Diamond"]
    hvs    = [Si["H_v"], GaN["H_v"], SiC["H_v"], Diamond["H_v"]]
    colors = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0"]
    bars   = ax.bar(names, [h/1e9 for h in hvs], color=colors, alpha=0.85)
    ax.set_ylabel("Vickers hardness [GPa]")
    ax.set_title("Material Hardness — Machining Difficulty")
    for bar, h in zip(bars, hvs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{h/1e9:.0f} GPa", ha="center", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # 6. Diamond laser transparency map
    ax = axes[1, 2]
    wavelengths = np.arange(100, 1200, 5)
    E_gap = Diamond["bandgap_eV"]
    photon_E = PLANCK_EV_NM / wavelengths
    absorbs = photon_E > E_gap
    ax.fill_between(wavelengths, 0, 1, where=absorbs,
                    alpha=0.3, color="green", label="Absorbing (ablation possible)")
    ax.fill_between(wavelengths, 0, 1, where=~absorbs,
                    alpha=0.3, color="gray",  label="Transparent (no single-photon)")
    for name, lam, col in [("ArF 193nm","#FF1744"), ("Nd:YAG 3ω 355nm","#FF9800"),
                             ("Nd:YAG 532nm","#4CAF50"), ("Nd:YAG 1064nm","#2196F3")]:
        lam_val = int(name.split(" ")[-1].replace("nm", ""))
        ax.axvline(lam_val, color=col, linestyle="--", linewidth=1.5,
                   label=f"{name}")
    ax.axvline(PLANCK_EV_NM / E_gap, color="black", linewidth=2,
               label=f"Threshold {PLANCK_EV_NM/E_gap:.0f}nm")
    ax.set_xlabel("Laser wavelength [nm]")
    ax.set_title(f"Diamond Laser Window (Eg={E_gap}eV)")
    ax.set_xlim(100, 1200); ax.set_ylim(0, 1)
    ax.set_yticks([]); ax.legend(fontsize=7)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "results", "physical_limits.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Literature validation — compare model vs. experimental data
# ═══════════════════════════════════════════════════════════════════════════════

def validate_against_literature() -> dict:
    """
    Cross-check wafer_thinning_critical_thickness() against HBM paper data.

    HBM paper (Materials 2024, PMC11595813) measures chip fracture strength
    after 3-point bending for three singulation methods on Si(111).

    We back-calculate the effective Griffith flaw size implied by each process,
    then re-run our model with that flaw size and compare the predicted fracture
    stress against the measured value.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from validation.experimental_data import (
        CHIP_STRENGTH_MPa, effective_flaw_size_um, gf_to_mpa
    )

    methods = ("stealth_dicing", "blade_dicing", "laser_grooving")
    thicknesses = (60, 90, 120)

    rows = []
    for method in methods:
        for t_um in thicknesses:
            sigma_exp = CHIP_STRENGTH_MPa.get((method, t_um))
            if sigma_exp is None:
                continue
            a_eff_um  = effective_flaw_size_um(method, t_um)
            # Model prediction at that flaw size
            w = wafer_thinning_critical_thickness(
                Si, flaw_size_nm=a_eff_um * 1e3, wafer_diameter_mm=4.0
            )
            sigma_model = w["griffith_frac_stress_MPa"]
            err_pct = (sigma_model - sigma_exp) / sigma_exp * 100

            rows.append({
                "method":       method,
                "thickness_um": t_um,
                "sigma_exp_MPa":   sigma_exp,
                "a_eff_um":        a_eff_um,
                "sigma_model_MPa": sigma_model,
                "error_pct":       err_pct,
            })

    return rows


def print_validation():
    rows = validate_against_literature()

    print("\n" + "═" * 72)
    print("  Literature Validation — Griffith model vs. HBM paper (PMC11595813)")
    print("  Si(111) chip 3PB strength: back-calc a_eff → re-predict σ_f")
    print("═" * 72)
    print(f"  {'Method':<18} {'t[µm]':>6} {'σ_exp':>8} {'a_eff':>8} "
          f"{'σ_model':>8} {'err%':>7}")
    print("  " + "-" * 60)
    for r in rows:
        flag = "✅" if abs(r["error_pct"]) < 5 else ("⚠️ " if abs(r["error_pct"]) < 15 else "❌")
        print(f"  {r['method']:<18} {r['thickness_um']:>6} "
              f"{r['sigma_exp_MPa']:>7.0f}M "
              f"{r['a_eff_um']:>7.2f}µ "
              f"{r['sigma_model_MPa']:>7.0f}M "
              f"{r['error_pct']:>+6.1f}%  {flag}")
    print()
    print("  Note: σ_model = K_Ic/√(πa_eff) — tautological at 60µm (by construction).")
    print("  Physical insight: a_eff encodes process damage depth:")
    print("    stealth_dicing : ~1–2 µm   (modified layer, minimal mechanical damage)")
    print("    blade_dicing   : ~8–10 µm  (subsurface microcracks from abrasion)")
    print("    laser_grooving : ~90 µm    (HAZ + thermal cracking)")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

PLANCK_EV_NM = 1239.84

def main():
    parser = argparse.ArgumentParser(description="Physical limits of DISCO processes")
    parser.add_argument("--plot",     action="store_true")
    parser.add_argument("--validate", action="store_true",
                        help="Cross-check against HBM paper data")
    parser.add_argument("--material", default="Si",
                        choices=["Si", "SiC", "GaN", "Diamond"])
    args = parser.parse_args()

    mat_map = {"Si": Si, "SiC": SiC, "GaN": GaN, "Diamond": Diamond}
    print_roadmap()

    w = wafer_thinning_critical_thickness(mat_map[args.material])
    print(f"  Detailed Griffith analysis: {args.material}")
    print(f"    σ_fracture (Griffith): {w['griffith_frac_stress_MPa']:.0f} MPa")
    print(f"    Critical thickness:    {w['critical_thickness_um']:.1f} µm")
    print(f"    Safety factor @ 25µm:  {w['safety_at_25um']:.2f}")
    print(f"    Safety factor @ 10µm:  {w['safety_at_10um']:.2f}")
    print(f"    Safety factor @  5µm:  {w['safety_at_5um']:.2f}")

    if args.validate:
        print_validation()

    if args.plot:
        _plot_limits()


if __name__ == "__main__":
    main()
