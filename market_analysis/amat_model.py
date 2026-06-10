"""
Applied Materials (AMAT) — CVD / PVD / CMP / Ion Implant Model
==========================================================
Covers Centura (PECVD), Endura (PVD), Reflexion GT (CMP), and VIISta (ion implant) platforms.

Key physics:
  - PECVD deposition rate: SiO₂/SiN/Low-k (TEOS/SiH₄/SiCOH precursors)
  - PVD sputtering: Sigmund-Thompson sputter yield
  - CMP material removal rate: Preston equation + pad conditioning
  - CMP dishing: isolated feature width dependence
  - Ion implant dopant profile: LSS/Pearson-IV (Rp + ΔRp)
  - Dopant activation: SPER (Solid Phase Epitaxy Regrowth) fraction

実行:
    python fem/amat_model.py
    python fem/amat_model.py --cmp --material Cu
    python fem/amat_model.py --implant --species B --energy 10

参考文献:
    Preston (1927) J. Soc. Glass Tech. — CMP Preston equation
    Sigmund (1969) Phys. Rev. — sputter yield model
    Ziegler et al. (1985) SRIM — ion range statistics (LSS)
    Bain & Ruber (1987) J. Appl. Phys. — SPER activation model
    Applied Materials product briefs (Centura, Endura, Reflexion, VIISta, 2023)
"""

import argparse
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import erf

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ════════════════════════════════════════════════════════════════════════════
# 1. Process Parameter Tables
# ════════════════════════════════════════════════════════════════════════════

PECVD_FILMS = {
    "SiO2_TEOS": {
        "precursor": "TEOS",
        "dep_rate_base_nm_min": 120.0,
        "stress_MPa": -200.0,      # compressive
        "WER_nm_min_BHF": 3.0,     # wet etch rate in BHF
        "k_dielectric": 4.0,
        "color": "#2166ac",
        "ref": "Applied Materials Centura brief (2023)",
    },
    "SiN_SiH4": {
        "precursor": "SiH4/NH3",
        "dep_rate_base_nm_min": 80.0,
        "stress_MPa": +400.0,      # tensile
        "WER_nm_min_BHF": 0.5,
        "k_dielectric": 7.5,
        "color": "#d62728",
        "ref": "Applied Materials Centura brief (2023)",
    },
    "SiCOH_lowk": {
        "precursor": "DEMS/porogen",
        "dep_rate_base_nm_min": 60.0,
        "stress_MPa": -100.0,
        "WER_nm_min_BHF": 12.0,
        "k_dielectric": 2.5,
        "color": "#ff7f0e",
        "ref": "Baklanov et al. (2012) Adv. Mater.",
    },
    "TiN_barrier": {
        "precursor": "TiCl4/NH3",
        "dep_rate_base_nm_min": 25.0,
        "stress_MPa": +800.0,
        "WER_nm_min_BHF": 0.05,
        "k_dielectric": 80.0,      # metallic
        "color": "#2ca02c",
        "ref": "Applied Materials Endura brief (2023)",
    },
}

# CMP materials: Preston coefficient Kp [Pa⁻¹·m⁻¹·s⁻¹]
CMP_MATERIALS = {
    "Cu":  {"Kp": 2.8e-13, "hardness_GPa": 1.3, "density_g_cm3": 8.96,
            "slurry": "H2O2/BTA", "color": "#d62728"},
    "SiO2":{"Kp": 6.0e-13, "hardness_GPa": 7.5, "density_g_cm3": 2.20,
            "slurry": "ceria", "color": "#2166ac"},
    "W":   {"Kp": 1.5e-13, "hardness_GPa": 4.0, "density_g_cm3": 19.3,
            "slurry": "H2O2/Fe", "color": "#ff7f0e"},
    "SiN": {"Kp": 3.5e-13, "hardness_GPa": 14.0, "density_g_cm3": 3.17,
            "slurry": "ceria", "color": "#2ca02c"},
}

# Ion implant: {species: (Rp_keV_nm_coeff, deltaRp_fraction, charge_Z)}
ION_PARAMS = {
    "B":   {"Rp_coeff": 3.8,  "dRp_frac": 0.35, "Z": 5,   "color": "#2166ac"},
    "P":   {"Rp_coeff": 1.5,  "dRp_frac": 0.28, "Z": 15,  "color": "#d62728"},
    "As":  {"Rp_coeff": 0.65, "dRp_frac": 0.18, "Z": 33,  "color": "#ff7f0e"},
    "In":  {"Rp_coeff": 0.40, "dRp_frac": 0.15, "Z": 49,  "color": "#2ca02c"},
    "Ge":  {"Rp_coeff": 0.55, "dRp_frac": 0.20, "Z": 32,  "color": "#9467bd"},
}

# ════════════════════════════════════════════════════════════════════════════
# 2. Core Physics Functions
# ════════════════════════════════════════════════════════════════════════════

def pecvd_dep_rate(film: str, T_C: float, power_W: float,
                   pressure_Torr: float = 5.0) -> dict:
    """
    PECVD deposition rate: empirical power-law + Arrhenius temperature correction.

    DR = DR_base * (P/P_ref)^0.5 * (pressure/Torr_ref)^0.3 * exp(-Ea/kT)
    where Ea ≈ 0.15 eV for TEOS PECVD in the 300-500°C range.
    """
    if film not in PECVD_FILMS:
        film = "SiO2_TEOS"
    p = PECVD_FILMS[film]

    Ea_eV = 0.15
    k_B_eV = 8.617e-5
    T_ref = 400.0 + 273.15
    T_K = T_C + 273.15
    T_factor = math.exp(-Ea_eV / k_B_eV * (1.0 / T_K - 1.0 / T_ref))
    P_factor = math.sqrt(power_W / 1000.0)
    press_factor = (pressure_Torr / 5.0) ** 0.3
    DR = p["dep_rate_base_nm_min"] * T_factor * P_factor * press_factor
    uniformity_3s_pct = 3.5 / P_factor   # higher power → better uniformity

    return {
        "film": film,
        "T_C": T_C,
        "power_W": power_W,
        "pressure_Torr": pressure_Torr,
        "dep_rate_nm_min": round(DR, 1),
        "stress_MPa": p["stress_MPa"],
        "k_dielectric": p["k_dielectric"],
        "uniformity_3sigma_pct": round(uniformity_3s_pct, 2),
        "WER_nm_min": p["WER_nm_min_BHF"],
    }


def pvd_sputter_yield(target_material: str, energy_eV: float,
                      angle_deg: float = 0.0) -> dict:
    """
    Sigmund-Thompson sputter yield.

    Y(E) = 0.042 * alpha * (M2/M1) * Sn(E) / U0
    Y(theta) = Y(0) / cos^f(theta),  f ≈ 1.5 for most metals

    参考: Sigmund (1969) Phys. Rev. 184, 383
    """
    # Target parameters: {material: (M_amu, U0_eV, alpha, color)}
    target_params = {
        "Ti": (47.9,  4.89, 0.28, "#2166ac"),
        "Cu": (63.5,  3.49, 0.34, "#d62728"),
        "W":  (183.8, 8.79, 0.40, "#ff7f0e"),
        "Al": (26.98, 3.36, 0.25, "#2ca02c"),
        "Ta": (180.9, 8.09, 0.42, "#9467bd"),
    }
    M_targ, U0, alpha, _ = target_params.get(target_material,
                                              target_params["Ti"])
    M_Ar = 39.95   # Ar sputter gas

    # Nuclear stopping Sn (Lindhard reduced energy)
    eps = 32.5 * M_Ar / (M_Ar + M_targ) * energy_eV / (M_targ * 14.1)
    Sn = 3.441 * math.sqrt(eps) * math.log(eps + 2.718) / (1.0 + 6.355 * math.sqrt(eps) + eps * (6.882 * math.sqrt(eps) - 1.708))
    Sn = max(Sn, 0.0)

    Y0 = 0.042 * alpha * (M_Ar / M_targ) * Sn / max(U0, 0.1)
    Y0 = max(Y0, 0.0)
    # Angular dependence
    theta_rad = math.radians(angle_deg)
    Y_angle = Y0 / max(math.cos(theta_rad), 1e-3) ** 1.5 if angle_deg < 80 else Y0

    dep_rate_nm_min = Y_angle * 100.0 / max(energy_eV / 500.0, 0.1)  # empirical scaling

    return {
        "target": target_material,
        "energy_eV": energy_eV,
        "angle_deg": angle_deg,
        "sputter_yield": round(Y_angle, 4),
        "dep_rate_nm_min": round(dep_rate_nm_min, 1),
        "U0_eV": U0,
    }


def cmp_mrr(material: str, pressure_kPa: float, velocity_m_s: float,
            pad_age_k_wafers: float = 0.0) -> dict:
    """
    Preston equation MRR + pad conditioning factor.

    MRR = Kp * P * V * pad_factor
    pad_factor = exp(-0.03 * pad_age_k_wafers)  (pad glazing)

    参考: Preston (1927) J. Soc. Glass Tech. 11, 247
    """
    if material not in CMP_MATERIALS:
        material = "SiO2"
    p = CMP_MATERIALS[material]
    pad_factor = math.exp(-0.03 * pad_age_k_wafers)
    MRR_m_s = p["Kp"] * pressure_kPa * 1e3 * velocity_m_s * pad_factor
    MRR_nm_min = MRR_m_s * 1e9 * 60.0

    return {
        "material": material,
        "pressure_kPa": pressure_kPa,
        "velocity_m_s": velocity_m_s,
        "pad_age_k_wafers": pad_age_k_wafers,
        "MRR_nm_min": round(MRR_nm_min, 1),
        "pad_factor": round(pad_factor, 4),
        "slurry": p["slurry"],
    }


def cmp_dishing(material: str, feature_w_um: float,
                pressure_kPa: float = 30.0) -> dict:
    """
    CMP dishing depth vs. isolated feature width.

    dishing_nm = dishing0 * (1 - exp(-feature_w / lambda_dish))
    where dishing0 ∝ compliance * sqrt(pressure), lambda_dish ≈ 20 µm.
    """
    mat_compliance = {"Cu": 80.0, "W": 30.0, "SiO2": 10.0, "SiN": 5.0}
    C = mat_compliance.get(material, 30.0)
    dishing0 = C * math.sqrt(pressure_kPa / 30.0)
    lambda_dish = 20.0  # µm
    dishing_nm = dishing0 * (1.0 - math.exp(-feature_w_um / lambda_dish))
    erosion_nm = dishing_nm * 0.15  # oxide erosion around features

    return {
        "material": material,
        "feature_w_um": feature_w_um,
        "dishing_nm": round(dishing_nm, 1),
        "erosion_nm": round(erosion_nm, 1),
        "within_spec": dishing_nm < 20.0,
    }


def ion_implant_profile(species: str, energy_keV: float,
                        dose_cm2: float = 1e14) -> dict:
    """
    Pearson-IV dopant depth profile (approximated as Gaussian here).
    Rp scales as energy^0.6 / sqrt(Z_target); ΔRp ≈ fraction of Rp.

    参考: Ziegler et al. (1985) SRIM — ion range statistics
    """
    if species not in ION_PARAMS:
        species = "B"
    p = ION_PARAMS[species]

    Rp_nm  = p["Rp_coeff"] * (energy_keV ** 0.6) * 10.0   # [nm]
    dRp_nm = Rp_nm * p["dRp_frac"]
    # Peak concentration: N_peak = dose / (sqrt(2π) * ΔRp) [cm⁻³]
    N_peak = dose_cm2 / (math.sqrt(2.0 * math.pi) * dRp_nm * 1e-7)

    # Profile at surface: N(0) ≈ N_peak * exp(-Rp²/(2*ΔRp²))
    N_surface = N_peak * math.exp(-0.5 * (Rp_nm / dRp_nm) ** 2)

    return {
        "species": species,
        "energy_keV": energy_keV,
        "dose_cm2": dose_cm2,
        "Rp_nm": round(Rp_nm, 1),
        "dRp_nm": round(dRp_nm, 1),
        "N_peak_cm3": round(N_peak, 3),
        "junction_nm_at_1e17": round(Rp_nm + dRp_nm * math.sqrt(2 * math.log(N_peak / 1e17)), 1)
                               if N_peak > 1e17 else None,
    }


def anneal_activation(dose_cm2: float, T_C: float, t_s: float = 1.0) -> dict:
    """
    SPER dopant activation fraction (Solid Phase Epitaxy Regrowth).

    f_act = 1 - exp(-v_SPER * t / d_amorphous)
    v_SPER = v0 * exp(-Ea / kT),  v0=3.7e9 nm/s, Ea=2.68 eV

    参考: Olson & Roth (1988) Mater. Sci. Rep. — SPER velocity in Si
    """
    k_B_eV = 8.617e-5
    Ea = 2.68   # eV (SPER activation energy)
    v0 = 3.7e9  # nm/s
    v_SPER = v0 * math.exp(-Ea / (k_B_eV * (T_C + 273.15)))
    d_amorphous = 30.0  # nm typical amorphous layer from implant
    f_act = 1.0 - math.exp(-v_SPER * t_s / max(d_amorphous, 0.1))
    f_act = min(f_act, 0.999)

    return {
        "T_C": T_C,
        "t_s": t_s,
        "v_SPER_nm_s": round(v_SPER, 3),
        "f_activation": round(f_act, 5),
        "activated_dose_cm2": round(dose_cm2 * f_act, 4),
        "full_activation": f_act > 0.99,
    }


def full_pipeline(node_nm: float = 3.0, mat_stack: str = "SiO2") -> dict:
    """
    AMAT complete process chain: PECVD → PVD barrier → CMP → implant → anneal.
    """
    dep   = pecvd_dep_rate("SiO2_TEOS", T_C=400, power_W=1000)
    pvd   = pvd_sputter_yield("Ti", energy_eV=500, angle_deg=0)
    mrr   = cmp_mrr(mat_stack, pressure_kPa=30, velocity_m_s=0.5)
    dish  = cmp_dishing(mat_stack, feature_w_um=max(node_nm * 5.0, 0.5))
    impl  = ion_implant_profile("B", energy_keV=max(node_nm * 3.0, 1.0))
    annl  = anneal_activation(dose_cm2=1e14, T_C=1100, t_s=60.0)

    return {
        "node_nm": node_nm,
        "pecvd_dep_rate_nm_min": dep["dep_rate_nm_min"],
        "pvd_sputter_yield": pvd["sputter_yield"],
        "cmp_mrr_nm_min": mrr["MRR_nm_min"],
        "dishing_nm": dish["dishing_nm"],
        "implant_Rp_nm": impl["Rp_nm"],
        "activation_fraction": annl["f_activation"],
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. Visualization
# ════════════════════════════════════════════════════════════════════════════

def plot_all(out_dir: str = OUT_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(17, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)
    ax = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    # ── Panel 0: PECVD dep rate vs. temperature ───────────────────────────
    T_arr = np.linspace(200, 600, 100)
    for film, col in [("SiO2_TEOS", "#2166ac"), ("SiN_SiH4", "#d62728"),
                      ("SiCOH_lowk", "#ff7f0e"), ("TiN_barrier", "#2ca02c")]:
        dr = [pecvd_dep_rate(film, T, 1000)["dep_rate_nm_min"] for T in T_arr]
        ax[0].plot(T_arr, dr, color=col, lw=2.2,
                   label=f"{film.split('_')[0]} k={PECVD_FILMS[film]['k_dielectric']:.1f}")
    ax[0].set_xlabel("Temperature [°C]")
    ax[0].set_ylabel("Dep Rate [nm/min]")
    ax[0].set_title("PECVD Deposition Rate vs. Temperature\n(AMAT Centura, 1 kW, 5 Torr)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.25)

    # ── Panel 1: Sputter yield vs. energy ────────────────────────────────
    E_arr = np.linspace(100, 1000, 100)
    for tgt, col in [("Ti", "#2166ac"), ("Cu", "#d62728"), ("W", "#ff7f0e"),
                     ("Al", "#2ca02c"), ("Ta", "#9467bd")]:
        sy = [pvd_sputter_yield(tgt, E)["sputter_yield"] for E in E_arr]
        ax[1].plot(E_arr, sy, color=col, lw=2.2, label=tgt)
    ax[1].set_xlabel("Ion Energy [eV]")
    ax[1].set_ylabel("Sputter Yield [atoms/ion]")
    ax[1].set_title("PVD Sputter Yield vs. Energy\n(Sigmund-Thompson, Ar⁺ on target)")
    ax[1].legend(fontsize=8.5)
    ax[1].grid(alpha=0.25)

    # ── Panel 2: CMP MRR vs. pressure ────────────────────────────────────
    P_arr = np.linspace(5, 60, 100)
    for mat, col in [("Cu", "#d62728"), ("SiO2", "#2166ac"), ("W", "#ff7f0e"), ("SiN", "#2ca02c")]:
        mrr = [cmp_mrr(mat, P, 0.5)["MRR_nm_min"] for P in P_arr]
        ax[2].plot(P_arr, mrr, color=col, lw=2.2, label=mat)
    ax[2].set_xlabel("Pressure [kPa]")
    ax[2].set_ylabel("MRR [nm/min]")
    ax[2].set_title("CMP Material Removal Rate\n(Preston Equation, v=0.5 m/s)")
    ax[2].legend(fontsize=8.5)
    ax[2].grid(alpha=0.25)

    # ── Panel 3: CMP dishing vs. feature width ────────────────────────────
    fw_arr = np.linspace(0.1, 100, 200)
    for mat, col in [("Cu", "#d62728"), ("W", "#ff7f0e"), ("SiO2", "#2166ac")]:
        dish = [cmp_dishing(mat, fw)["dishing_nm"] for fw in fw_arr]
        ax[3].plot(fw_arr, dish, color=col, lw=2.2, label=mat)
    ax[3].axhline(20.0, color="k", ls="--", lw=1.2, label="20 nm spec")
    ax[3].set_xlabel("Feature Width [µm]")
    ax[3].set_ylabel("Dishing Depth [nm]")
    ax[3].set_title("CMP Dishing vs. Feature Width\n(AMAT Reflexion GT, 30 kPa)")
    ax[3].legend(fontsize=8.5)
    ax[3].grid(alpha=0.25)
    ax[3].set_xlim(0, 100)

    # ── Panel 4: Ion implant depth profile ───────────────────────────────
    depth_arr = np.linspace(0, 100, 300)
    for sp, col in [("B", "#2166ac"), ("P", "#d62728"), ("As", "#ff7f0e"), ("In", "#2ca02c")]:
        res = ion_implant_profile(sp, energy_keV=20.0, dose_cm2=1e14)
        Rp, dRp = res["Rp_nm"], res["dRp_nm"]
        prof = np.exp(-0.5 * ((depth_arr - Rp) / dRp) ** 2) / (math.sqrt(2 * math.pi) * dRp)
        prof *= 1e14 * 1e-7  # convert to cm⁻³ (dose / √2π / ΔRp in nm)
        ax[4].semilogy(depth_arr, np.maximum(prof, 1e14), color=col, lw=2.2,
                       label=f"{sp} Rp={Rp:.0f}nm")
    ax[4].set_xlabel("Depth [nm]")
    ax[4].set_ylabel("Concentration [cm⁻³]")
    ax[4].set_title("Ion Implant Dopant Profile\n(Pearson-IV/Gaussian, 20 keV, 1×10¹⁴ cm⁻²)")
    ax[4].legend(fontsize=8)
    ax[4].set_ylim(1e14, 1e22)
    ax[4].grid(alpha=0.25)

    # ── Panel 5: Process summary table ───────────────────────────────────
    ax[5].axis("off")
    col_labels = ["Node\n[nm]", "PECVD\n[nm/min]", "PVD Yield\n[at/ion]",
                  "CMP MRR\n[nm/min]", "Dishing\n[nm]", "Activation\n[%]"]
    tdata = []
    for node in [2.0, 3.0, 5.0, 7.0]:
        res = full_pipeline(node_nm=node, mat_stack="Cu")
        tdata.append([f"{node:.0f}",
                      f"{res['pecvd_dep_rate_nm_min']:.0f}",
                      f"{res['pvd_sputter_yield']:.3f}",
                      f"{res['cmp_mrr_nm_min']:.0f}",
                      f"{res['dishing_nm']:.1f}",
                      f"{res['activation_fraction']*100:.1f}"])
    tbl = ax[5].table(cellText=tdata, colLabels=col_labels,
                      cellLoc="center", loc="center",
                      bbox=[0.0, 0.15, 1.0, 0.80])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#c0392b")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#fdf2f2")
    ax[5].set_title("AMAT Process Pipeline Summary\nCentura · Endura · Reflexion · VIISta")

    fig.suptitle(
        "Applied Materials (AMAT) — CVD / PVD / CMP / Ion Implant\n"
        "Centura PECVD · Endura PVD · Reflexion GT CMP · VIISta Implant",
        fontsize=12, fontweight="bold",
    )
    out = os.path.join(out_dir, "amat_model.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] AMAT model → {out}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="Applied Materials CVD/PVD/CMP/Implant model")
    ap.add_argument("--cmp",     action="store_true", help="Print CMP MRR summary")
    ap.add_argument("--implant", action="store_true", help="Print implant profile summary")
    ap.add_argument("--material", default="Cu",  choices=list(CMP_MATERIALS.keys()))
    ap.add_argument("--species",  default="B",   choices=list(ION_PARAMS.keys()))
    ap.add_argument("--energy",   type=float, default=20.0, help="Implant energy [keV]")
    ap.add_argument("--no-plot",  action="store_true")
    args = ap.parse_args()

    if args.cmp:
        print("\nCMP MRR Summary")
        print("=" * 40)
        for P in [10, 20, 30, 40, 50]:
            r = cmp_mrr(args.material, P, 0.5)
            print(f"  {args.material:6s}  P={P:2d} kPa  MRR={r['MRR_nm_min']:6.1f} nm/min  "
                  f"slurry={r['slurry']}")

    if args.implant:
        print("\nIon Implant Profile")
        print("=" * 40)
        r = ion_implant_profile(args.species, args.energy)
        for k, v in r.items():
            print(f"  {k:<30} {v}")

    if not args.no_plot:
        plot_all()


if __name__ == "__main__":
    main()
