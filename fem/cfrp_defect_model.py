"""
CFRP Defect Physics Model — Multi-modal NDT Simulation
=======================================================
Analytical physics models for 5 NDT methods applied to CFRP laminates.

Defect types simulated:
    delamination  : flat crack between plies (most common)
    void          : spherical air pocket in matrix
    fiber_break   : local fiber rupture (stiffness reduction)
    impact        : cone-shaped subsurface damage zone
    matrix_crack  : distributed micro-cracks (porosity increase)

NDT methods:
    1. Ultrasonic Testing (UT)   — A-scan / C-scan
    2. Active Thermography (AT)  — pulsed heat, surface ΔT map
    3. Eddy Current Testing (ECT)— impedance change (conductive CF)
    4. Acoustic Emission (AE)    — crack-growth event signals
    5. X-ray (qualitative depth) — Beer-Lambert attenuation

Usage:
    python fem/cfrp_defect_model.py
    python fem/cfrp_defect_model.py --defect delamination --plot
    python fem/cfrp_defect_model.py --all
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# Material library
# ════════════════════════════════════════════════════════════════════════════

CFRP_PROPS = {
    # Mechanical (T300/Epoxy, Vf=0.60)
    "E11_GPa":   138.0,   # fiber direction
    "E22_GPa":     9.5,   # transverse
    "G12_GPa":     5.5,
    "nu12":        0.30,
    "rho_kg_m3": 1560.0,
    # Thermal
    "k_z_W_mK":    0.70,  # through-thickness conductivity
    "Cp_J_kgK":  900.0,
    "alpha_t":     4.4e-7,  # thermal diffusivity m²/s  (k/ρCp)
    # Acoustic
    "v_L_m_s":  2900.0,   # longitudinal wave, through thickness
    "Z_MRayl":     4.5,   # acoustic impedance  ρ·v_L  [MRayl]
    "atten_dB_mm": 3.0,   # ultrasonic attenuation [dB/mm]
    # Electrical
    "sigma_S_m":  5e3,    # in-plane conductivity S/m (carbon fiber contribution)
    "mu_r":        1.0,
    # Fracture
    "GIc_kJ_m2":   0.20,  # mode-I interlaminar fracture toughness
    "GIIc_kJ_m2":  0.80,
    "ILSS_MPa":   60.0,   # interlaminar shear strength
}

AIR_Z_MRayl = 0.000413   # acoustic impedance of air

PLY_THICKNESS_MM = 0.125  # standard prepreg ply thickness


# ════════════════════════════════════════════════════════════════════════════
# Defect geometry definitions
# ════════════════════════════════════════════════════════════════════════════

def defect_params(defect_type: str, depth_mm: float = 1.0) -> dict:
    """Return canonical defect geometry for each defect type."""
    base = {"type": defect_type, "depth_mm": depth_mm}
    if defect_type == "delamination":
        return {**base, "diameter_mm": 10.0, "thickness_mm": 0.05,
                "aspect_ratio": 200.0, "reflectivity": 0.97}
    if defect_type == "void":
        return {**base, "diameter_mm": 2.0, "thickness_mm": 2.0,
                "aspect_ratio": 1.0, "reflectivity": 0.90}
    if defect_type == "fiber_break":
        return {**base, "diameter_mm": 1.0, "thickness_mm": 0.125,
                "stiffness_reduction": 0.40, "reflectivity": 0.45}
    if defect_type == "impact":
        return {**base, "diameter_mm": 20.0, "thickness_mm": depth_mm,
                "porosity_increase": 0.08, "reflectivity": 0.65}
    if defect_type == "matrix_crack":
        return {**base, "diameter_mm": 5.0, "thickness_mm": 0.25,
                "crack_density_1_mm": 0.5, "reflectivity": 0.30}
    raise ValueError(f"Unknown defect: {defect_type}")


# ════════════════════════════════════════════════════════════════════════════
# 1. Ultrasonic Testing (UT) — A-scan + C-scan
# ════════════════════════════════════════════════════════════════════════════

def ut_ascan(depth_mm: float, defect: dict,
             f_MHz: float = 5.0,
             thickness_mm: float = 4.0,
             dt_ns: float = 0.5,
             n_points: int = 500) -> dict:
    """
    Simulate pulse-echo A-scan with a defect at given depth.

    Returns time vector [µs] and amplitude [normalized].
    """
    props   = CFRP_PROPS
    v       = props["v_L_m_s"] * 1e-3   # mm/µs
    alpha_a = props["atten_dB_mm"] / (20 * np.log10(np.e))  # Np/mm

    t_us    = np.arange(n_points) * dt_ns * 1e-3   # µs
    signal  = np.zeros(n_points)

    def gaussian_pulse(t, t0, freq_MHz, amp):
        sigma = 1.0 / (np.pi * freq_MHz)   # µs
        return amp * np.exp(-((t - t0) ** 2) / (2 * sigma ** 2)) * np.cos(
            2 * np.pi * freq_MHz * (t - t0)
        )

    # Back-wall echo (full thickness)
    t_bw  = 2 * thickness_mm / v
    A_bw  = np.exp(-alpha_a * thickness_mm)       # attenuation one-way
    R_bw  = (CFRP_PROPS["Z_MRayl"] - AIR_Z_MRayl) / (
             CFRP_PROPS["Z_MRayl"] + AIR_Z_MRayl)   # ≈ 1 at free surface
    signal += gaussian_pulse(t_us, t_bw, f_MHz, A_bw * abs(R_bw))

    # Defect echo
    t_def = 2 * depth_mm / v
    A_def = np.exp(-alpha_a * depth_mm)
    R_def = defect.get("reflectivity", 0.8)
    signal += gaussian_pulse(t_us, t_def, f_MHz, A_def * R_def * 0.8)

    # Entry echo (t=0 surface)
    signal += gaussian_pulse(t_us, 0.0, f_MHz, 1.0)

    return {"t_us": t_us, "amplitude": signal,
            "t_defect_us": t_def, "t_backwall_us": t_bw,
            "depth_mm": depth_mm, "defect_type": defect["type"]}


def ut_cscan(defect: dict,
             scan_mm: float = 40.0,
             n_px: int = 80,
             thickness_mm: float = 4.0) -> dict:
    """
    Simulate 2D C-scan (peak amplitude map) for a circular defect.

    Returns 2D amplitude array and coordinate grids.
    """
    props  = CFRP_PROPS
    v      = props["v_L_m_s"] * 1e-3
    alpha_a = props["atten_dB_mm"] / (20 * np.log10(np.e))

    x = np.linspace(-scan_mm / 2, scan_mm / 2, n_px)
    y = np.linspace(-scan_mm / 2, scan_mm / 2, n_px)
    X, Y = np.meshgrid(x, y)
    R_xy = np.sqrt(X ** 2 + Y ** 2)   # radial distance from defect centre

    d    = defect["depth_mm"]
    r_d  = defect["diameter_mm"] / 2.0

    # Amplitude from defect echo (within defect radius → full reflection)
    inside  = R_xy <= r_d
    A_depth = np.exp(-alpha_a * d)
    amp     = np.where(inside,
                       A_depth * defect.get("reflectivity", 0.8),
                       0.05 * A_depth)   # small scatter outside

    # Gaussian lateral spread (beam divergence)
    beam_sigma = d * 0.15                # beam spreading with depth
    amp = amp + defect.get("reflectivity", 0.8) * A_depth * np.exp(
        -R_xy ** 2 / (2 * beam_sigma ** 2 + 0.01)
    ) * 0.3
    amp = np.clip(amp, 0, 1)

    return {"X": X, "Y": Y, "amplitude": amp,
            "defect_radius_mm": r_d, "depth_mm": d,
            "defect_type": defect["type"]}


# ════════════════════════════════════════════════════════════════════════════
# 2. Active Thermography (AT) — pulsed flash, surface ΔT
# ════════════════════════════════════════════════════════════════════════════

def thermography_response(defect: dict,
                           t_max_s: float = 2.0,
                           n_t: int = 200,
                           Q_J_m2: float = 1e4) -> dict:
    """
    1D analytical pulsed thermography model.

    Healthy area: T(t) ∝ 1/√t  (semi-infinite half-space cooling)
    Over defect:  heat blocked at depth d → surface stays warmer

    ΔT(t) = T_defect(t) - T_healthy(t)
    """
    props = CFRP_PROPS
    alpha = props["alpha_t"]   # m²/s
    rho   = props["rho_kg_m3"]
    Cp    = props["Cp_J_kgK"]
    k     = props["k_z_W_mK"]

    t     = np.linspace(0.02, t_max_s, n_t)
    d     = defect["depth_mm"] * 1e-3   # convert to m

    # Healthy surface temperature (Dirac pulse heating, semi-infinite)
    T_healthy = Q_J_m2 / (rho * Cp * np.sqrt(np.pi * alpha * t))

    # Defect: effectively reduces thermal mass → surface stays hotter
    # Approximation: defect acts as insulating layer at depth d
    # Using image-source method: reflection from insulating boundary
    t_peak    = d ** 2 / (4 * alpha)   # peak ΔT time ∝ d²

    # Amplitude of ΔT depends on defect thermal resistance
    R_thermal = defect.get("thickness_mm", 0.1) * 1e-3 / 0.025  # kJ_air
    delta_T = (Q_J_m2 * R_thermal * d) / (
        rho * Cp * (4 * np.pi * alpha) ** 0.5 * (t + t_peak) ** 1.5
    ) * np.exp(-d ** 2 / (4 * alpha * (t + t_peak)))

    snr = np.max(delta_T) / (0.05 * np.max(T_healthy))  # rough SNR

    return {"t_s": t, "T_healthy": T_healthy,
            "delta_T": delta_T,
            "t_peak_s": float(t[np.argmax(delta_T)]),
            "max_delta_T_C": float(np.max(delta_T)),
            "depth_mm": defect["depth_mm"],
            "snr": snr,
            "detectable": snr > 1.0}


def thermography_cscan(defect: dict,
                       scan_mm: float = 40.0, n_px: int = 80,
                       t_eval_s: float = None) -> dict:
    """2D thermography C-scan at time t_eval (default: optimal contrast time)."""
    d_m   = defect["depth_mm"] * 1e-3
    alpha = CFRP_PROPS["alpha_t"]
    t_opt = d_m ** 2 / (4 * alpha)       # peak contrast time
    t_eval = t_eval_s if t_eval_s else t_opt

    x = np.linspace(-scan_mm / 2, scan_mm / 2, n_px)
    X, Y = np.meshgrid(x, x)
    R_xy = np.sqrt(X ** 2 + Y ** 2)
    r_d  = defect["diameter_mm"] / 2.0

    # Lateral heat diffusion blurs the defect boundary
    sigma_lat = np.sqrt(4 * alpha * t_eval) * 1e3  # mm
    inside    = R_xy <= r_d
    # Gaussian-blurred defect
    weight = np.exp(-((R_xy - r_d * (R_xy > r_d)) ** 2) / (2 * sigma_lat ** 2))
    delta_T = np.where(inside, 1.0, 0.0) * 0.7 + weight * 0.3

    return {"X": X, "Y": Y, "delta_T": delta_T,
            "t_eval_s": t_eval, "defect_type": defect["type"]}


# ════════════════════════════════════════════════════════════════════════════
# 3. Eddy Current Testing (ECT) — impedance change
# ════════════════════════════════════════════════════════════════════════════

def ect_impedance_scan(defect: dict,
                       f_kHz: float = 100.0,
                       coil_r_mm: float = 2.0,
                       scan_mm: float = 30.0,
                       n_px: int = 60) -> dict:
    """
    Simplified ECT model for CFRP.

    CFRP is weakly conductive due to carbon fibers.
    Defects (air gaps) locally reduce effective conductivity
    → measurable impedance change ΔZ/Z₀.

    Skin depth: δ = √(2/(ωμσ))
    """
    props  = CFRP_PROPS
    sigma  = props["sigma_S_m"]
    mu     = 4 * np.pi * 1e-7 * props["mu_r"]
    omega  = 2 * np.pi * f_kHz * 1e3

    skin_depth_mm = np.sqrt(2 / (omega * mu * sigma)) * 1e3

    x   = np.linspace(-scan_mm / 2, scan_mm / 2, n_px)
    X, Y = np.meshgrid(x, x)
    R_xy = np.sqrt(X ** 2 + Y ** 2)
    r_d  = defect["diameter_mm"] / 2.0
    d    = defect["depth_mm"]

    # Sensitivity decays exponentially with depth
    depth_factor = np.exp(-d / skin_depth_mm)

    # Impedance change: Gaussian profile over defect (coil size limited)
    sigma_coil = coil_r_mm + r_d / 2
    inside     = R_xy <= r_d + coil_r_mm
    dZ_real = depth_factor * np.where(
        inside,
        0.8 * np.exp(-R_xy ** 2 / (2 * sigma_coil ** 2)),
        0.0
    )
    dZ_imag = depth_factor * np.where(
        inside,
        0.4 * np.exp(-R_xy ** 2 / (2 * sigma_coil ** 2 * 1.3)),
        0.0
    )

    detectable = depth_factor * 0.8 > 0.05  # 5% threshold

    return {"X": X, "Y": Y,
            "dZ_real": dZ_real, "dZ_imag": dZ_imag,
            "dZ_mag": np.sqrt(dZ_real ** 2 + dZ_imag ** 2),
            "skin_depth_mm": skin_depth_mm,
            "depth_factor": depth_factor,
            "detectable": bool(detectable),
            "defect_type": defect["type"]}


# ════════════════════════════════════════════════════════════════════════════
# 4. Acoustic Emission (AE) — crack event signals
# ════════════════════════════════════════════════════════════════════════════

def ae_event_simulation(defect: dict,
                         n_cycles: int = 1000,
                         load_max_MPa: float = 400.0,
                         sensor_dist_mm: float = 50.0,
                         seed: int = 42) -> dict:
    """
    Simulate AE hit sequence during cyclic loading.

    AE activity increases with:
      - Crack area growth (Paris law)
      - Local stress intensity > K_th (threshold)

    Returns:
      - cycles: load cycle numbers of each AE hit
      - amplitude_dB: AE signal amplitude at sensor
      - b_value: Gutenberg-Richter b-value (slope of amplitude dist)
    """
    rng     = np.random.default_rng(seed)
    GIc     = CFRP_PROPS["GIc_kJ_m2"] * 1e3   # J/m²
    ILSS    = CFRP_PROPS["ILSS_MPa"] * 1e6     # Pa
    alpha_ae = 0.04                              # AE attenuation Np/mm

    # Defect growth rate (simplified Paris law)
    da_dN_base = 1e-6  # m/cycle baseline crack growth
    r_d0       = defect["diameter_mm"] / 2.0 * 1e-3  # initial crack radius (m)

    hits_cycles = []
    hits_amp    = []

    crack_r = r_d0
    for cyc in range(n_cycles):
        load  = load_max_MPa * np.sin(np.pi * cyc / n_cycles) ** 2  # ramp
        K_app = load * 1e6 * np.sqrt(np.pi * crack_r)               # Pa√m
        K_Ic  = np.sqrt(GIc * CFRP_PROPS["E22_GPa"] * 1e9)

        if K_app < 0.1 * K_Ic:
            continue

        # Probability of AE event per cycle
        p_hit = 0.05 * (K_app / K_Ic) ** 2.5
        p_hit = min(p_hit, 0.95)

        if rng.random() < p_hit:
            # AE source amplitude (proportional to crack area increment)
            da     = da_dN_base * (K_app / K_Ic) ** 4
            src_amp_dB = 60 + 20 * np.log10(da / da_dN_base + 1e-9)
            # Attenuation to sensor
            rec_amp_dB = src_amp_dB - alpha_ae * sensor_dist_mm * 20 * np.log10(
                np.e
            )
            rec_amp_dB = max(rec_amp_dB, 20.0)  # threshold 20 dB

            hits_cycles.append(cyc)
            hits_amp.append(rec_amp_dB)

        crack_r += da_dN_base * (K_app / K_Ic) ** 4

    # b-value (Gutenberg-Richter)
    if len(hits_amp) > 5:
        amps  = np.array(hits_amp)
        bins  = np.linspace(amps.min(), amps.max(), 10)
        counts = np.array([np.sum(amps >= b) for b in bins])
        valid  = counts > 0
        if valid.sum() > 3:
            try:
                b_val = float(-np.polyfit(bins[valid], np.log10(counts[valid] + 1e-9), 1)[0])
            except np.linalg.LinAlgError:
                b_val = 1.5
        else:
            b_val = 1.5
    else:
        b_val = np.nan

    return {"cycles": np.array(hits_cycles),
            "amplitude_dB": np.array(hits_amp),
            "n_hits": len(hits_cycles),
            "b_value": b_val,
            "final_crack_r_mm": crack_r * 1e3,
            "defect_type": defect["type"]}


# ════════════════════════════════════════════════════════════════════════════
# 5. X-ray Attenuation (qualitative depth map)
# ════════════════════════════════════════════════════════════════════════════

def xray_transmission(defect: dict,
                       thickness_mm: float = 4.0,
                       energy_keV: float = 100.0,
                       scan_mm: float = 40.0,
                       n_px: int = 80) -> dict:
    """
    Beer-Lambert X-ray transmission map.

    CFRP μ/ρ ≈ 0.17 cm²/g at 100 keV (carbon-dominated)
    Void/delamination: air replaces CFRP → less attenuation → brighter spot
    """
    rho   = CFRP_PROPS["rho_kg_m3"] * 1e-3   # g/cm³
    mu_rho = 0.17   # cm²/g  (mass attenuation coefficient at 100 keV)
    mu    = mu_rho * rho * 0.1   # 1/mm

    x   = np.linspace(-scan_mm / 2, scan_mm / 2, n_px)
    X, Y = np.meshgrid(x, x)
    R_xy = np.sqrt(X ** 2 + Y ** 2)
    r_d  = defect["diameter_mm"] / 2.0

    # Baseline transmission (healthy)
    I_healthy = np.exp(-mu * thickness_mm)

    # Over defect: void replaces CFRP → effective thickness reduced
    void_frac = np.where(R_xy <= r_d, 1.0, 0.0)
    # Gaussian blur (X-ray focal spot / scattering)
    from scipy.ndimage import gaussian_filter
    void_frac_blur = gaussian_filter(void_frac, sigma=2.0)

    defect_thickness = defect.get("thickness_mm", 0.1)
    I_defect = np.exp(-mu * (thickness_mm - void_frac_blur * defect_thickness))

    contrast = (I_defect - I_healthy) / I_healthy  # positive = brighter

    return {"X": X, "Y": Y,
            "transmission": I_defect,
            "contrast": contrast,
            "I_healthy": I_healthy,
            "mu_mm": mu,
            "defect_type": defect["type"]}


# ════════════════════════════════════════════════════════════════════════════
# Multi-modal summary
# ════════════════════════════════════════════════════════════════════════════

def multimodal_ndt(defect_type: str = "delamination",
                   depth_mm: float = 1.5) -> dict:
    """Run all 5 NDT methods on a given defect and return summary."""
    defect = defect_params(defect_type, depth_mm)

    ut   = ut_ascan(depth_mm, defect)
    thm  = thermography_response(defect)
    ect  = ect_impedance_scan(defect)
    ae   = ae_event_simulation(defect)
    xr   = xray_transmission(defect)

    return {
        "defect":  defect,
        "ut":      ut,
        "thermo":  thm,
        "ect":     ect,
        "ae":      ae,
        "xray":    xr,
    }


# ════════════════════════════════════════════════════════════════════════════
# Visualisation
# ════════════════════════════════════════════════════════════════════════════

def plot_multimodal(results: dict, save: bool = True) -> str:
    defect = results["defect"]
    ut     = results["ut"]
    thm    = results["thermo"]
    ae     = results["ae"]

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, hspace=0.40, wspace=0.35)

    # ── UT A-scan ────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ut["t_us"], ut["amplitude"], color="#1f77b4", lw=1.5)
    ax1.axvline(ut["t_defect_us"], color="#d62728", ls="--", lw=1.2,
                label=f"Defect ({defect['depth_mm']:.1f} mm)")
    ax1.axvline(ut["t_backwall_us"], color="#2ca02c", ls="--", lw=1.2,
                label="Back wall")
    ax1.set_xlabel("Time [µs]"); ax1.set_ylabel("Amplitude [norm.]")
    ax1.set_title(f"UT A-scan  ({defect['type']})")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.25)

    # ── UT C-scan ────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    cs   = ut_cscan(defect)
    im2  = ax2.contourf(cs["X"], cs["Y"], cs["amplitude"],
                        levels=20, cmap="hot_r")
    plt.colorbar(im2, ax=ax2, label="Amplitude")
    circ = plt.Circle((0, 0), cs["defect_radius_mm"],
                       fill=False, color="cyan", lw=1.5, ls="--")
    ax2.add_patch(circ)
    ax2.set_xlabel("X [mm]"); ax2.set_ylabel("Y [mm]")
    ax2.set_title("UT C-scan (peak amplitude map)")
    ax2.set_aspect("equal")

    # ── Thermography ΔT ──────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(thm["t_s"], thm["T_healthy"] / thm["T_healthy"].max(),
             color="#888", lw=1.3, label="Healthy (norm.)")
    ax3.plot(thm["t_s"], thm["delta_T"] / (thm["T_healthy"].max() + 1e-12),
             color="#ff7f0e", lw=2.0, label=f"ΔT defect  (det={thm['detectable']})")
    ax3.axvline(thm["t_peak_s"], color="#ff7f0e", ls=":", lw=1)
    ax3.set_xlabel("Time [s]"); ax3.set_ylabel("Normalized T / ΔT")
    ax3.set_title(f"Active Thermography  (d={defect['depth_mm']:.1f}mm)")
    ax3.legend(fontsize=8); ax3.grid(alpha=0.25)

    # ── ECT impedance map ────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ect_r = ect_impedance_scan(defect)
    im4   = ax4.contourf(ect_r["X"], ect_r["Y"], ect_r["dZ_mag"],
                         levels=20, cmap="plasma")
    plt.colorbar(im4, ax=ax4, label="|ΔZ| [norm.]")
    ax4.set_xlabel("X [mm]"); ax4.set_ylabel("Y [mm]")
    ax4.set_title(f"ECT  |ΔZ|  (skin δ={ect_r['skin_depth_mm']:.1f}mm)")
    ax4.set_aspect("equal")

    # ── AE amplitude history ──────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    if ae["n_hits"] > 0:
        ax5.scatter(ae["cycles"], ae["amplitude_dB"],
                    s=8, alpha=0.6, color="#9467bd")
        ax5.set_xlabel("Load cycle"); ax5.set_ylabel("AE amplitude [dB]")
        bv = ae["b_value"]
        bv_str = f"{bv:.2f}" if not np.isnan(bv) else "N/A"
        ax5.set_title(f"AE  hits={ae['n_hits']}  b={bv_str}")
    else:
        ax5.text(0.5, 0.5, "No AE events detected",
                 ha="center", va="center", transform=ax5.transAxes)
        ax5.set_title("Acoustic Emission")
    ax5.grid(alpha=0.25)

    # ── X-ray contrast map ────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    xr   = xray_transmission(defect)
    im6  = ax6.contourf(xr["X"], xr["Y"], xr["contrast"] * 100,
                        levels=20, cmap="RdYlGn")
    plt.colorbar(im6, ax=ax6, label="Contrast [%]")
    ax6.set_xlabel("X [mm]"); ax6.set_ylabel("Y [mm]")
    ax6.set_title("X-ray Transmission Contrast")
    ax6.set_aspect("equal")

    title = (f"CFRP Multi-modal NDT — {defect['type'].replace('_',' ').title()}"
             f"  depth={defect['depth_mm']:.1f} mm  "
             f"diam={defect['diameter_mm']:.0f} mm")
    fig.suptitle(title, fontsize=12, fontweight="bold")

    if save:
        path = os.path.join(OUT_DIR, f"cfrp_ndt_{defect['type']}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  [✓] Saved: {path}")
        plt.close()
        return path
    plt.show()
    return ""


def plot_defect_comparison(save: bool = True) -> str:
    """Compare detectability across all defect types and NDT methods."""
    defect_types = ["delamination", "void", "fiber_break", "impact", "matrix_crack"]
    depth = 1.5

    fig, axes = plt.subplots(len(defect_types), 3, figsize=(14, 16))
    cmaps = ["hot_r", "YlOrRd", "plasma"]
    titles = ["UT C-scan (amplitude)", "Thermography ΔT", "ECT |ΔZ|"]

    for row, dt in enumerate(defect_types):
        defect = defect_params(dt, depth)
        datasets = [
            ut_cscan(defect)["amplitude"],
            thermography_cscan(defect)["delta_T"],
            ect_impedance_scan(defect)["dZ_mag"],
        ]
        for col, (data, cmap, ttl) in enumerate(zip(datasets, cmaps, titles)):
            ax  = axes[row, col]
            n   = data.shape[0]
            ext = [-20, 20, -20, 20]
            im  = ax.imshow(data, cmap=cmap, extent=ext, origin="lower",
                            vmin=0, vmax=data.max() if data.max() > 0 else 1)
            ax.set_title(f"{dt.replace('_',' ')}\n{ttl}" if row == 0 else "", fontsize=8)
            ax.set_ylabel(dt.replace("_", "\n"), fontsize=8) if col == 0 else None
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"CFRP Defect Detectability Matrix  (depth={depth} mm)\n"
                 "Rows: defect type  |  Columns: NDT method",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()

    if save:
        path = os.path.join(OUT_DIR, "cfrp_defect_comparison.png")
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
    ap = argparse.ArgumentParser(description="CFRP Multi-modal NDT Simulator")
    ap.add_argument("--defect", default="delamination",
                    choices=["delamination", "void", "fiber_break",
                             "impact", "matrix_crack"],
                    help="Defect type to simulate")
    ap.add_argument("--depth", type=float, default=1.5,
                    help="Defect depth [mm]")
    ap.add_argument("--all",  action="store_true",
                    help="Run all defect types + comparison matrix")
    ap.add_argument("--plot", action="store_true", default=True)
    args = ap.parse_args()

    if args.all:
        print("[*] Running all defect types …")
        for dt in ["delamination", "void", "fiber_break", "impact", "matrix_crack"]:
            print(f"  {dt} …")
            res = multimodal_ndt(dt, args.depth)
            plot_multimodal(res, save=True)
        print("[*] Generating comparison matrix …")
        plot_defect_comparison(save=True)
    else:
        print(f"[*] {args.defect}  depth={args.depth}mm")
        res = multimodal_ndt(args.defect, args.depth)

        # Print summary
        thm = res["thermo"]
        ect = res["ect"]
        ae  = res["ae"]
        print(f"\n  UT:   defect ToF = {res['ut']['t_defect_us']:.2f} µs")
        print(f"  Thermo: ΔT_max = {thm['max_delta_T_C']:.4f} °C  "
              f"detectable={thm['detectable']}")
        print(f"  ECT:  skin depth = {ect['skin_depth_mm']:.2f} mm  "
              f"detectable={ect['detectable']}")
        bv = ae["b_value"]
        print(f"  AE:   {ae['n_hits']} hits  b-value={bv:.2f}" if not np.isnan(bv)
              else f"  AE:   {ae['n_hits']} hits")

        if args.plot:
            plot_multimodal(res, save=True)


if __name__ == "__main__":
    main()
