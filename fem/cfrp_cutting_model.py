"""
CFRP Cutting Physics Model — Multi-process
==========================================
Analytical models for 4 cutting processes applied to CFRP laminates.

Processes:
    1. Milling / Turning  — continuous cutting (Merchant + composite extension)
    2. Drilling           — Hocheng-Dharan delamination model
    3. Dicing/Slicing     — Disco-type blade cutting (connects to dicing_blade_2d.py)
    4. Waterjet (AWJ)     — abrasive waterjet, kerf geometry + roughness

Key outputs per process:
    - Cutting force  Fc, Ft  [N]
    - Delamination factor Fd  [-]
    - Surface roughness Ra    [µm]
    - Tool wear rate VB        [mm / (N·m)]
    - Specific cutting energy  Ks [N/mm²]

Physics references:
    Merchant (1945)  — shear plane model
    Hocheng & Dharan (1993) — drill delamination
    Zhang et al. (2001)     — fiber orientation effects on cutting force
    Wang & Zhang (2003)     — surface roughness vs θ
    Shanmugam et al. (2008) — AWJ kerf geometry

Usage:
    python fem/cfrp_cutting_model.py
    python fem/cfrp_cutting_model.py --process all --plot
    python fem/cfrp_cutting_model.py --process milling --theta-sweep
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import fsolve

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# CFRP material properties
# ════════════════════════════════════════════════════════════════════════════

class CFRPLaminate:
    """
    Unidirectional CFRP laminate (T300/Epoxy, Vf=0.60) material model.
    Properties vary with fiber orientation angle θ (deg).
    """
    # Constituent properties
    E_f   = 230e3    # MPa  fiber axial modulus
    E_m   =   3.5e3  # MPa  matrix modulus
    G_m   =   1.3e3  # MPa  matrix shear modulus
    Vf    =   0.60
    rho   =   1560   # kg/m³

    # Laminate (rule of mixtures)
    E11   = 138e3    # MPa
    E22   =   9.5e3  # MPa
    G12   =   5.5e3  # MPa
    nu12  =   0.30

    # Strength
    Xt    = 1500     # MPa  longitudinal tensile strength
    Xc    = 1200     # MPa  longitudinal compressive strength
    Yt    =   50     # MPa  transverse tensile strength
    Yc    =  200     # MPa  transverse compressive strength
    S12   =   70     # MPa  in-plane shear strength
    ILSS  =   60     # MPa  interlaminar shear strength

    # Fracture
    GIc   =  0.20    # kJ/m²  mode-I
    GIIc  =  0.80    # kJ/m²  mode-II

    # Thermal
    k_z   =   0.70   # W/mK  through-thickness
    Cp    = 900      # J/kgK

    @staticmethod
    def E_theta(theta_deg: float) -> float:
        """Young's modulus in cutting direction (off-axis)."""
        t  = np.radians(theta_deg)
        c, s = np.cos(t), np.sin(t)
        E11, E22, G12, nu12 = (CFRPLaminate.E11, CFRPLaminate.E22,
                               CFRPLaminate.G12, CFRPLaminate.nu12)
        nu21 = nu12 * E22 / E11
        inv_E = (c**4 / E11 + s**4 / E22 +
                 c**2 * s**2 * (1/G12 - 2*nu12/E11))
        return 1.0 / inv_E

    @staticmethod
    def strength_theta(theta_deg: float) -> float:
        """
        Tsai-Hill failure stress in cutting direction [MPa].
        """
        t    = np.radians(theta_deg)
        c, s = np.cos(t), np.sin(t)
        X, Y, S = CFRPLaminate.Xt, CFRPLaminate.Yt, CFRPLaminate.S12
        # Tsai-Hill: (σ1/X)² - σ1σ2/X² + (σ2/Y)² + (τ12/S)² = 1
        # For uniaxial σ in θ direction:
        # σ1 = σcos²θ, σ2 = σsin²θ, τ12 = -σsinθcosθ
        denom = (c**4/X**2 - c**2*s**2/X**2 +
                 s**4/Y**2 + c**2*s**2/S**2)
        return 1.0 / np.sqrt(max(denom, 1e-30))


MAT = CFRPLaminate()


# ════════════════════════════════════════════════════════════════════════════
# 1. Milling / Turning  (Merchant's orthogonal cutting + fiber effect)
# ════════════════════════════════════════════════════════════════════════════

def milling_forces(theta_deg: float,
                   h_mm: float = 0.1,
                   width_mm: float = 1.0,
                   rake_deg: float = 0.0,
                   friction_coeff: float = 0.35) -> dict:
    """
    Orthogonal cutting forces for CFRP at fiber angle θ.

    Extended Merchant model for composites (Zhang 2001):
    - Shear plane angle φ determined from energy minimization
    - Specific cutting pressure Ks modified by orientation factor K_θ

    Args:
        theta_deg   : fiber orientation relative to cutting direction [°]
        h_mm        : uncut chip thickness [mm]
        width_mm    : cutting width [mm]
        rake_deg    : tool rake angle [°]
        friction_coeff: tool-chip friction coefficient µ

    Returns:
        dict with Fc [N], Ft [N], Ra [µm], Ks [N/mm²], phi [°]
    """
    gamma  = np.radians(rake_deg)
    beta   = np.arctan(friction_coeff)          # friction angle

    # Effective shear strength at orientation θ
    tau_s  = MAT.strength_theta(theta_deg) / 3.0  # von Mises approx

    # Shear plane angle (Merchant)
    phi    = np.pi/4 - (beta - gamma) / 2.0
    phi    = max(phi, 0.05)

    # Fiber orientation factor K_θ (empirical, Zhang 2001)
    # K_θ = 1 at θ=90° (transverse), minimum near θ=15°
    theta_r = np.radians(theta_deg)
    K_theta = (1.0 + 0.5 * np.sin(theta_r) ** 2 +
               0.3 * np.cos(2 * theta_r))

    # Cutting force components (per unit width)
    area   = h_mm * width_mm
    Fc     = (tau_s * K_theta * area * np.cos(beta - gamma) /
              (np.sin(phi) * np.cos(phi + beta - gamma)))
    Ft     = Fc * np.sin(beta - gamma) / np.cos(beta - gamma)

    # Specific cutting energy
    Ks     = Fc / area

    # Surface roughness (Wang & Zhang 2003 empirical)
    # Ra peaks at θ=90° (fibers perpendicular to cut), minimum at θ=0°
    Ra     = 1.0 + 4.0 * np.sin(theta_r) ** 2 + 2.0 * np.cos(theta_r - np.pi/3) ** 2
    Ra    *= (h_mm / 0.1) ** 0.4    # feed effect

    return {
        "Fc_N":       float(Fc),
        "Ft_N":       float(Ft),
        "Ra_um":      float(Ra),
        "Ks_N_mm2":   float(Ks),
        "phi_deg":    float(np.degrees(phi)),
        "theta_deg":  theta_deg,
    }


def milling_theta_sweep(thetas: np.ndarray = None, **kwargs) -> dict:
    """Run milling_forces over a range of fiber angles."""
    if thetas is None:
        thetas = np.linspace(0, 180, 37)
    results = [milling_forces(t, **kwargs) for t in thetas]
    return {k: np.array([r[k] for r in results]) for k in results[0]}


# ════════════════════════════════════════════════════════════════════════════
# 2. Drilling  (Hocheng-Dharan delamination model)
# ════════════════════════════════════════════════════════════════════════════

def drilling_delamination(feed_mm_rev: float = 0.1,
                           spindle_rpm: float = 3000.0,
                           drill_D_mm: float = 6.0,
                           n_plies: int = 16,
                           ply_t_mm: float = 0.125) -> dict:
    """
    Hocheng-Dharan (1993) delamination model for CFRP drilling.

    Critical thrust force for delamination onset:
        Fcr = π √(8 GIc E h³ / (3(1-ν²)))

    where h = remaining uncut thickness when drill tip reaches delamination plane.

    Also computes:
    - Delamination factor Fd = D_del / D_drill
    - Peel-up (entry) and push-down (exit) delamination
    - Thrust force from empirical feed-force model
    """
    t_total = n_plies * ply_t_mm                # total laminate thickness [mm]
    E       = MAT.E22                           # MPa  (transverse, dominates)
    nu      = MAT.nu12
    GIc_J_m2 = MAT.GIc * 1e3                    # kJ/m² → J/m²
    E_Pa     = E * 1e6                           # MPa → Pa

    # Remaining thickness when drill is near exit
    h_arr   = np.linspace(0.1, t_total, 50)    # mm

    def F_critical(h):
        # Hocheng-Dharan: F_cr = π √(8 G_Ic E h³ / (3(1-ν²)))  [N]
        h_m = h * 1e-3                           # mm → m
        return np.pi * np.sqrt(8 * GIc_J_m2 * E_Pa * h_m**3 /
                                (3 * (1 - nu**2)))

    Fcr_arr = np.array([F_critical(h) for h in h_arr])

    # Empirical thrust force (Voss et al. 2019 type)
    # F_thrust = C_f * f^0.8 * D^0.4  (N)
    C_f   = 150    # empirical constant for CFRP (N / (mm/rev)^0.8 / mm^0.4)
    Fz    = C_f * feed_mm_rev ** 0.8 * drill_D_mm ** 0.4

    # At which depth does thrust exceed critical?
    # (critical is smallest near exit, i.e., small h)
    h_at_Fcr = None
    for h, fcr in zip(h_arr, Fcr_arr):
        if Fz > fcr:
            h_at_Fcr = h
            break

    if h_at_Tcr := h_at_Fcr:
        del_depth = t_total - h_at_Tcr
        D_del_est = drill_D_mm + 2 * del_depth * 0.5   # rough estimate
    else:
        del_depth = 0.0
        D_del_est = drill_D_mm

    Fd = D_del_est / drill_D_mm

    # Surface roughness of drilled hole
    v_c_m_s = np.pi * drill_D_mm * 1e-3 * spindle_rpm / 60
    Ra_hole  = 1.5 + 3.5 * feed_mm_rev ** 0.6 * (v_c_m_s / 2.0) ** (-0.3)

    return {
        "Fz_thrust_N":       float(Fz),
        "Fcr_exit_N":        float(Fcr_arr[0]),    # most critical (h→0)
        "Fd_delamination":   float(Fd),
        "delamination_depth_mm": float(del_depth),
        "Ra_hole_um":        float(Ra_hole),
        "v_c_m_s":           float(v_c_m_s),
        "h_sweep_mm":        h_arr,
        "Fcr_sweep_N":       Fcr_arr,
        "will_delaminate":   Fz > Fcr_arr[0],
    }


def drilling_feed_sweep(feeds: np.ndarray = None,
                         drill_D_mm: float = 6.0,
                         spindle_rpm: float = 3000.0) -> dict:
    """Sweep over feed rate to show delamination onset."""
    if feeds is None:
        feeds = np.linspace(0.01, 0.3, 50)
    results = [drilling_delamination(f, spindle_rpm, drill_D_mm) for f in feeds]
    return {
        "feeds":    feeds,
        "Fz":       np.array([r["Fz_thrust_N"] for r in results]),
        "Fcr":      np.array([r["Fcr_exit_N"]  for r in results]),
        "Fd":       np.array([r["Fd_delamination"] for r in results]),
        "Ra":       np.array([r["Ra_hole_um"]   for r in results]),
        "delam":    np.array([r["will_delaminate"] for r in results]),
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. Dicing / Slicing  (Disco-type blade, brittle fracture regime)
# ════════════════════════════════════════════════════════════════════════════

def dicing_blade(cut_depth_mm: float = 2.0,
                 blade_W_mm: float = 0.3,
                 feed_mm_s: float = 50.0,
                 spindle_rpm: float = 30000.0,
                 theta_deg: float = 0.0,
                 blade_grit_um: float = 10.0) -> dict:
    """
    Disco-type blade dicing of CFRP.

    Uses brittle fracture + abrasive grinding model:
    - Chipping from subsurface fracture (similar to SiC model)
    - Delamination from peel-up force
    - Kerf width = blade_W + 2 * lateral_crack

    Chipping model adapted from SiC dicing_blade_2d.py physics:
        chip ∝ (KIc / hardness)² × (depth/blade_W)^0.5
    """
    KIc     = np.sqrt(MAT.GIc * 1e3 * MAT.E22 * 1e6)   # Pa√m
    KIc_MPa = KIc * 1e-6                                 # MPa√m

    # Hardness proxy from strength
    H       = MAT.Xt / 3.0     # MPa

    # Peripheral velocity [m/s]
    D_blade = 52.0   # mm (typical Disco blade diameter)
    v_c     = np.pi * D_blade * 1e-3 * spindle_rpm / 60.0

    # Chip thickness per abrasive grain (undeformed chip)
    # ag = grain size [mm]
    ag      = blade_grit_um * 1e-3
    h_chip  = (feed_mm_s * 1e-3 / (60 * spindle_rpm)) * np.sqrt(
        cut_depth_mm / (D_blade / 2)
    ) * 1e3   # µm

    # Lateral crack length (Lawn-Evans model) → chipping
    chip_um = 4.0 * (KIc_MPa / H) ** (4/3) * (
        feed_mm_s / (v_c * 1000)
    ) ** (1/3) * cut_depth_mm ** (1/2) * 1e4

    # Fiber orientation effect (same as milling K_θ)
    theta_r   = np.radians(theta_deg)
    K_theta   = 1.0 + 0.6 * np.sin(theta_r) ** 2
    chip_um  *= K_theta

    # Delamination (peel-up force estimate)
    Fp        = 0.3 * chip_um * blade_W_mm   # rough estimate [N]
    F_del_cr  = MAT.ILSS * blade_W_mm * cut_depth_mm * 0.1   # [N]
    will_delam = Fp > F_del_cr

    # Kerf width
    kerf_mm   = blade_W_mm + 2 * chip_um * 1e-3

    # Surface roughness
    Ra_wall   = 0.5 + 2.0 * (feed_mm_s / v_c / 1e3) ** 0.4 * K_theta

    # Cutting force per unit width
    tau_eff   = MAT.strength_theta(theta_deg) / 3.0
    Fc_N_mm   = tau_eff * ag * cut_depth_mm / blade_W_mm

    return {
        "chip_um":          float(chip_um),
        "kerf_mm":          float(kerf_mm),
        "Ra_wall_um":       float(Ra_wall),
        "Fc_N_mm":          float(Fc_N_mm),
        "Fp_delam_N":       float(Fp),
        "F_del_cr_N":       float(F_del_cr),
        "will_delaminate":  bool(will_delam),
        "v_c_m_s":          float(v_c),
        "h_chip_um":        float(h_chip),
        "theta_deg":        theta_deg,
    }


def dicing_parametric(depths: np.ndarray = None,
                       feeds:  np.ndarray = None,
                       theta_deg: float = 0.0) -> dict:
    """2D parametric sweep: cut depth × feed rate."""
    if depths is None:
        depths = np.linspace(0.5, 5.0, 20)
    if feeds is None:
        feeds  = np.linspace(10.0, 200.0, 20)
    D, F = np.meshgrid(depths, feeds)
    chip = np.zeros_like(D)
    Ra   = np.zeros_like(D)
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            r = dicing_blade(D[i, j], feed_mm_s=F[i, j], theta_deg=theta_deg)
            chip[i, j] = r["chip_um"]
            Ra[i, j]   = r["Ra_wall_um"]
    return {"D": D, "F": F, "chip": chip, "Ra": Ra}


# ════════════════════════════════════════════════════════════════════════════
# 4. Abrasive Waterjet (AWJ)
# ════════════════════════════════════════════════════════════════════════════

def awj_cutting(pressure_MPa: float = 300.0,
                abrasive_flow_g_min: float = 300.0,
                feed_mm_min: float = 200.0,
                nozzle_D_mm: float = 0.8,
                standoff_mm: float = 3.0,
                thickness_mm: float = 5.0,
                theta_deg: float = 0.0) -> dict:
    """
    Abrasive Waterjet cutting of CFRP (Shanmugam 2008 model).

    Kerf geometry + taper + surface roughness from empirical correlations.

    Specific energy: e_s = P_jet * (nozzle area) / (feed * kerf)
    Depth of cut:  h = C_a * (P^1.25 * ma^0.687) / (f^0.563 * σ_u^0.799 * D_n^0.787)
    """
    # Jet hydraulic power
    rho_w   = 1000.0      # kg/m³
    v_jet   = np.sqrt(2 * pressure_MPa * 1e6 / rho_w)   # m/s  (Bernoulli)
    A_noz   = np.pi * (nozzle_D_mm * 1e-3) ** 2 / 4
    P_jet_W = 0.5 * rho_w * A_noz * v_jet ** 3

    # Material removal model (Hashish 1984 / Shanmugam 2008)
    sigma_u  = MAT.Xt   # MPa  (use fiber direction strength)
    ma_kg_s  = abrasive_flow_g_min / 60e3    # kg/s
    f_m_s    = feed_mm_min / 60e3            # m/s

    C_a      = 0.80    # empirical constant calibrated to CFRP AWJ literature
    h_cut    = (C_a * (pressure_MPa ** 1.25) *
                (abrasive_flow_g_min ** 0.687) /
                (feed_mm_min ** 0.563 *
                 sigma_u ** 0.799 *
                 nozzle_D_mm ** 0.787))   # mm

    # Kerf width (top and bottom due to jet divergence)
    taper_factor = 0.08 * standoff_mm / thickness_mm
    w_top        = nozzle_D_mm + 0.3 * standoff_mm * 0.1
    w_bot        = max(w_top * (1 - taper_factor * thickness_mm), 0.5 * w_top)
    taper_angle  = np.degrees(np.arctan((w_top - w_bot) / (2 * thickness_mm)))

    # Surface roughness (empirical)
    Ra_awj = (10 * (feed_mm_min / 100) ** 0.6 *
              (thickness_mm / 5) ** 0.4 *
              (pressure_MPa / 300) ** (-0.5))

    # Delamination: AWJ has very low tool thrust → minimal delamination
    Fd_awj = 1.0 + 0.01 * (feed_mm_min / 100)

    # Fiber orientation: AWJ mostly isotropic (jet erodes all directions)
    theta_r   = np.radians(theta_deg)
    K_theta   = 1.0 + 0.15 * np.sin(theta_r) ** 2   # weak anisotropy
    Ra_awj   *= K_theta

    # Cutting quality judgment
    cuts_through = h_cut >= thickness_mm

    return {
        "h_cut_mm":         float(h_cut),
        "cuts_through":     bool(cuts_through),
        "kerf_top_mm":      float(w_top),
        "kerf_bot_mm":      float(w_bot),
        "taper_deg":        float(taper_angle),
        "Ra_um":            float(Ra_awj),
        "Fd_delamination":  float(Fd_awj),
        "v_jet_m_s":        float(v_jet),
        "P_jet_kW":         float(P_jet_W * 1e-3),
        "theta_deg":        theta_deg,
    }


def awj_pressure_sweep(pressures: np.ndarray = None,
                        thickness_mm: float = 5.0) -> dict:
    """Sweep over jet pressure."""
    if pressures is None:
        pressures = np.linspace(100, 400, 40)
    results = [awj_cutting(p, thickness_mm=thickness_mm) for p in pressures]
    return {
        "pressure": pressures,
        "h_cut":    np.array([r["h_cut_mm"]    for r in results]),
        "Ra":       np.array([r["Ra_um"]        for r in results]),
        "taper":    np.array([r["taper_deg"]    for r in results]),
        "Fd":       np.array([r["Fd_delamination"] for r in results]),
    }


# ════════════════════════════════════════════════════════════════════════════
# Comparative process summary
# ════════════════════════════════════════════════════════════════════════════

def process_comparison(theta_deg: float = 0.0) -> dict:
    """
    Compare all 4 cutting processes at representative conditions.
    Returns summary dict for table / radar chart.
    """
    mill   = milling_forces(theta_deg, h_mm=0.1, width_mm=1.0)
    drill  = drilling_delamination()
    dicing = dicing_blade(cut_depth_mm=2.0, theta_deg=theta_deg)
    awj    = awj_cutting(theta_deg=theta_deg)

    return {
        "Milling":    {"Ra_um": mill["Ra_um"],    "Fd": 1.0,
                       "Fc_N": mill["Fc_N"],       "Ft_N": mill["Ft_N"],
                       "delam_risk": "Low"},
        "Drilling":   {"Ra_um": drill["Ra_hole_um"], "Fd": drill["Fd_delamination"],
                       "Fc_N": drill["Fz_thrust_N"], "Ft_N": 0.0,
                       "delam_risk": "High" if drill["will_delaminate"] else "Low"},
        "Dicing":     {"Ra_um": dicing["Ra_wall_um"], "Fd": 1.0 + dicing["chip_um"] * 0.01,
                       "Fc_N": dicing["Fc_N_mm"] * 0.3, "Ft_N": dicing["Fp_delam_N"],
                       "delam_risk": "Med" if dicing["will_delaminate"] else "Low"},
        "AWJ":        {"Ra_um": awj["Ra_um"],     "Fd": awj["Fd_delamination"],
                       "Fc_N": 0.0,               "Ft_N": 0.0,
                       "delam_risk": "Very Low"},
    }


# ════════════════════════════════════════════════════════════════════════════
# Visualisation
# ════════════════════════════════════════════════════════════════════════════

def plot_all_processes(save: bool = True) -> str:
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 3, hspace=0.42, wspace=0.32)

    thetas = np.linspace(0, 180, 37)
    colors = {"Milling": "#1f77b4", "Drilling": "#d62728",
              "Dicing": "#2ca02c",  "AWJ": "#ff7f0e"}

    # ── Panel 1: Milling force vs fiber angle ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    mill = milling_theta_sweep(thetas)
    ax1.plot(thetas, mill["Fc_N"],  color="#1f77b4", lw=2.0, label="Fc (cutting)")
    ax1.plot(thetas, mill["Ft_N"],  color="#1f77b4", lw=2.0, ls="--", label="Ft (thrust)")
    ax1.set_xlabel("Fiber angle θ [°]"); ax1.set_ylabel("Force [N]  (width=1mm)")
    ax1.set_title("(a) Milling — Force vs Fiber Angle")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.25)
    ax1.set_xticks([0, 45, 90, 135, 180])

    # ── Panel 2: Milling surface roughness vs θ ───────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(thetas, mill["Ra_um"], color="#ff7f0e", lw=2.0)
    ax2.axhline(3.2, color="k", ls=":", lw=1, label="Ra 3.2 µm limit")
    ax2.set_xlabel("Fiber angle θ [°]"); ax2.set_ylabel("Ra [µm]")
    ax2.set_title("(b) Milling — Surface Roughness vs θ")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.25)
    ax2.set_xticks([0, 45, 90, 135, 180])

    # ── Panel 3: Drilling delamination vs feed ────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    dr   = drilling_feed_sweep()
    ax3.plot(dr["feeds"], dr["Fz"],  color="#d62728", lw=2.0, label="Thrust Fz")
    ax3.plot(dr["feeds"], dr["Fcr"], color="#2166ac", lw=2.0, ls="--",
             label="Critical Fcr")
    onset = dr["feeds"][np.argmax(dr["delam"])] if dr["delam"].any() else None
    if onset:
        ax3.axvline(onset, color="#d62728", ls=":", lw=1.5,
                    label=f"Delam onset {onset:.3f} mm/rev")
    ax3.set_xlabel("Feed [mm/rev]"); ax3.set_ylabel("Force [N]")
    ax3.set_title("(c) Drilling — Delamination Onset")
    ax3.legend(fontsize=8); ax3.grid(alpha=0.25)

    # ── Panel 4: Drilling Fd vs feed ──────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(dr["feeds"], dr["Fd"], color="#d62728", lw=2.0)
    ax4.axhline(1.2, color="k", ls=":", lw=1, label="Fd=1.2 threshold")
    ax4.set_xlabel("Feed [mm/rev]"); ax4.set_ylabel("Delamination factor Fd [-]")
    ax4.set_title("(d) Drilling — Delamination Factor")
    ax4.legend(fontsize=8); ax4.grid(alpha=0.25)

    # ── Panel 5: Dicing chipping 2D map ──────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    dp   = dicing_parametric(theta_deg=0.0)
    im5  = ax5.contourf(dp["D"], dp["F"], dp["chip"], levels=20, cmap="YlOrRd")
    plt.colorbar(im5, ax=ax5, label="Chipping [µm]")
    ax5.contour(dp["D"], dp["F"], dp["chip"], levels=[50.0],
                colors="white", linewidths=1.5, linestyles="--")
    ax5.set_xlabel("Cut depth [mm]"); ax5.set_ylabel("Feed [mm/s]")
    ax5.set_title("(e) Dicing — Chipping Map (θ=0°)\n── 50µm limit")

    # ── Panel 6: Dicing — fiber angle effect ─────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    thetas2 = np.linspace(0, 180, 37)
    chip_th = [dicing_blade(theta_deg=t)["chip_um"] for t in thetas2]
    Ra_th   = [dicing_blade(theta_deg=t)["Ra_wall_um"] for t in thetas2]
    ax6.plot(thetas2, chip_th, color="#2ca02c", lw=2.0, label="Chipping [µm]")
    ax6b = ax6.twinx()
    ax6b.plot(thetas2, Ra_th, color="#9467bd", lw=2.0, ls="--", label="Ra [µm]")
    ax6b.set_ylabel("Ra [µm]", color="#9467bd")
    ax6.set_xlabel("Fiber angle θ [°]")
    ax6.set_ylabel("Chipping [µm]", color="#2ca02c")
    ax6.set_title("(f) Dicing — Fiber Angle Effect")
    ax6.set_xticks([0, 45, 90, 135, 180])
    lines1, labs1 = ax6.get_legend_handles_labels()
    lines2, labs2 = ax6b.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labs1 + labs2, fontsize=8)
    ax6.grid(alpha=0.25)

    # ── Panel 7: AWJ pressure sweep ───────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 0])
    aw   = awj_pressure_sweep()
    ax7.plot(aw["pressure"], aw["h_cut"], color="#ff7f0e", lw=2.0, label="Depth of cut")
    ax7.axhline(5.0, color="k", ls="--", lw=1, label="Target thickness 5mm")
    ax7.set_xlabel("Jet pressure [MPa]"); ax7.set_ylabel("Depth of cut [mm]")
    ax7.set_title("(g) AWJ — Pressure vs Depth of Cut")
    ax7.legend(fontsize=8); ax7.grid(alpha=0.25)

    # ── Panel 8: AWJ taper vs Ra ──────────────────────────────────────────────
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(aw["pressure"], aw["Ra"],   color="#ff7f0e", lw=2.0, label="Ra [µm]")
    ax8b = ax8.twinx()
    ax8b.plot(aw["pressure"], aw["taper"], color="#8c564b", lw=2.0, ls="--",
              label="Taper [°]")
    ax8b.set_ylabel("Taper angle [°]", color="#8c564b")
    ax8.set_xlabel("Jet pressure [MPa]")
    ax8.set_ylabel("Ra [µm]", color="#ff7f0e")
    ax8.set_title("(h) AWJ — Surface Quality")
    lines1, labs1 = ax8.get_legend_handles_labels()
    lines2, labs2 = ax8b.get_legend_handles_labels()
    ax8.legend(lines1 + lines2, labs1 + labs2, fontsize=8)
    ax8.grid(alpha=0.25)

    # ── Panel 9: Process comparison radar ────────────────────────────────────
    ax9 = fig.add_subplot(gs[2, 2], polar=True)
    comp = process_comparison()
    procs     = list(comp.keys())
    metrics   = ["Ra_um", "Fd", "Fc_N"]
    m_labels  = ["Ra [µm]", "Delam. Fd", "Fc [N]"]
    n_m       = len(metrics)
    angles    = np.linspace(0, 2*np.pi, n_m, endpoint=False).tolist()
    angles   += angles[:1]

    # Normalize metrics to [0,1]
    vals_all = {m: [comp[p][m] for p in procs] for m in metrics}
    maxv     = {m: max(max(vals_all[m]), 1e-9) for m in metrics}

    for proc in procs:
        vals = [comp[proc][m] / maxv[m] for m in metrics]
        vals += vals[:1]
        ax9.plot(angles, vals, lw=2, label=proc,
                 color=colors.get(proc, "gray"))
        ax9.fill(angles, vals, alpha=0.1, color=colors.get(proc, "gray"))

    ax9.set_xticks(angles[:-1])
    ax9.set_xticklabels(m_labels, fontsize=9)
    ax9.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax9.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7)
    ax9.set_title("(i) Process Comparison\n(normalized)", pad=15)
    ax9.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)

    fig.suptitle("CFRP Cutting — Multi-process Physics Model\n"
                 "T300/Epoxy (Vf=0.60)  |  Milling · Drilling · Dicing · AWJ",
                 fontsize=12, fontweight="bold")

    if save:
        path = os.path.join(OUT_DIR, "cfrp_cutting_model.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  [✓] Saved: {path}")
        plt.close()
        return path
    plt.show()
    return ""


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="CFRP Multi-process Cutting Model")
    ap.add_argument("--process", default="all",
                    choices=["all", "milling", "drilling", "dicing", "awj"])
    ap.add_argument("--theta",  type=float, default=0.0,
                    help="Fiber orientation angle [deg]")
    ap.add_argument("--plot",   action="store_true", default=True)
    args = ap.parse_args()

    print(f"\n{'═'*55}")
    print(f"  CFRP Cutting Model  (θ={args.theta}°)")
    print(f"{'═'*55}")

    if args.process in ("all", "milling"):
        m = milling_forces(args.theta)
        print(f"\n[Milling]  Fc={m['Fc_N']:.1f}N  Ft={m['Ft_N']:.1f}N  "
              f"Ra={m['Ra_um']:.2f}µm  Ks={m['Ks_N_mm2']:.0f}N/mm²")

    if args.process in ("all", "drilling"):
        d = drilling_delamination()
        print(f"\n[Drilling] Fz={d['Fz_thrust_N']:.1f}N  "
              f"Fcr={d['Fcr_exit_N']:.1f}N  "
              f"Fd={d['Fd_delamination']:.3f}  "
              f"Ra={d['Ra_hole_um']:.2f}µm  "
              f"Delam={'YES' if d['will_delaminate'] else 'NO'}")

    if args.process in ("all", "dicing"):
        dc = dicing_blade(theta_deg=args.theta)
        print(f"\n[Dicing]   chip={dc['chip_um']:.1f}µm  "
              f"kerf={dc['kerf_mm']:.3f}mm  "
              f"Ra={dc['Ra_wall_um']:.2f}µm  "
              f"Delam={'YES' if dc['will_delaminate'] else 'NO'}")

    if args.process in ("all", "awj"):
        aw = awj_cutting(theta_deg=args.theta)
        print(f"\n[AWJ]      h_cut={aw['h_cut_mm']:.2f}mm  "
              f"Ra={aw['Ra_um']:.2f}µm  "
              f"taper={aw['taper_deg']:.2f}°  "
              f"Vjet={aw['v_jet_m_s']:.0f}m/s  "
              f"cuts={'YES' if aw['cuts_through'] else 'NO'}")

    if args.plot:
        print("\n[*] Generating plots …")
        plot_all_processes(save=True)


if __name__ == "__main__":
    main()
