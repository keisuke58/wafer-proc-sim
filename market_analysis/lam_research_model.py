"""
Lam Research — Plasma Etch & ALD Equipment Model
==========================================================
Covers Kiyo C (conductor etch), Sense.i (dielectric etch), and VECTOR (ALD) platforms.

Key physics:
  - ARDE (Aspect-Ratio Dependent Etching): neutral-ion flux balance, ion shadowing
  - CD bias vs. aspect ratio (tapered sidewall correction)
  - Etch selectivity: Si / SiO2 / SiN / W gas-chemistry table
  - Within-wafer uniformity: gas ring conductance model
  - ALD saturation: Langmuir adsorption isotherm (GPC vs. dose)

実行:
    python fem/lam_research_model.py
    python fem/lam_research_model.py --etch --material Si
    python fem/lam_research_model.py --ald --precursor TMA

参考文献:
    Gottscho et al. (1992) J. Vac. Sci. Technol. B — ARDE neutral-flux model
    Coburn & Winters (1979) J. Appl. Phys. — ion-neutral synergy
    Elam et al. (2002) Thin Solid Films — ALD Langmuir saturation
    Lam Research Corp. product briefs (Kiyo C, Sense.i, VECTOR, 2023)
"""

import argparse
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ════════════════════════════════════════════════════════════════════════════
# 1. Chamber & Material Parameter Tables
# ════════════════════════════════════════════════════════════════════════════

CHAMBERS = {
    "Kiyo_C": {
        "type": "conductor",
        "platform": "2300 Kiyo C",
        "freq_MHz": 27.12,
        "power_W_max": 5000,
        "pressure_mTorr_range": (5, 50),
        "gas_default": "Cl2/BCl3",
        "color": "#d62728",
        "ref": "Lam Kiyo C product brief (2023)",
    },
    "Sense_i": {
        "type": "dielectric",
        "platform": "2300 Sense.i",
        "freq_MHz": 60.0,
        "power_W_max": 3000,
        "pressure_mTorr_range": (10, 100),
        "gas_default": "CF4/CHF3/Ar",
        "color": "#2166ac",
        "ref": "Lam Sense.i product brief (2023)",
    },
    "VECTOR": {
        "type": "ALD",
        "platform": "VECTOR ALD",
        "freq_MHz": 13.56,
        "power_W_max": 500,
        "pressure_mTorr_range": (500, 2000),
        "gas_default": "TMA/H2O",
        "color": "#2ca02c",
        "ref": "Lam VECTOR ALD product brief (2023)",
    },
}

# Selectivity table: {(etch_mat, stop_mat, gas_chemistry): selectivity_ratio}
SELECTIVITY_TABLE = {
    ("Si",   "SiO2", "Cl2/BCl3"):   25.0,
    ("Si",   "SiN",  "Cl2/BCl3"):   18.0,
    ("W",    "TiN",  "SF6/N2"):     30.0,
    ("W",    "SiO2", "SF6/N2"):     20.0,
    ("SiO2", "Si",   "CF4/CHF3"):   10.0,
    ("SiO2", "SiN",  "CF4/CHF3"):    3.0,
    ("SiN",  "Si",   "CF4/O2"):     15.0,
    ("SiN",  "SiO2", "CF4/O2"):      5.0,
}

# ALD precursor parameters for Langmuir saturation model
ALD_PRECURSORS = {
    "TMA": {
        "full_name": "Trimethylaluminum (Al2O3)",
        "sigma_nm2": 0.18,      # adsorption cross-section [nm²/molecule]
        "GPC_max_A": 1.1,       # saturated GPC [Å/cycle]
        "T_window_C": (150, 300),
        "color": "#2166ac",
        "ref": "Elam et al. (2002) Thin Solid Films",
    },
    "TDMAT": {
        "full_name": "Tetrakis(dimethylamido)titanium (TiO2)",
        "sigma_nm2": 0.22,
        "GPC_max_A": 0.55,
        "T_window_C": (200, 300),
        "color": "#d62728",
        "ref": "Ritala et al. (1999) Chem. Mater.",
    },
    "HCDS": {
        "full_name": "Hexachlorodisilane (SiO2)",
        "sigma_nm2": 0.15,
        "GPC_max_A": 1.3,
        "T_window_C": (500, 700),
        "color": "#ff7f0e",
        "ref": "Klaus et al. (2000) Appl. Phys. Lett.",
    },
    "WF6": {
        "full_name": "Tungsten hexafluoride (W)",
        "sigma_nm2": 0.12,
        "GPC_max_A": 2.5,
        "T_window_C": (300, 400),
        "color": "#9467bd",
        "ref": "Kim et al. (2002) J. Electrochem. Soc.",
    },
}

# ════════════════════════════════════════════════════════════════════════════
# 2. Core Physics Functions
# ════════════════════════════════════════════════════════════════════════════

def etch_rate(gas: str, pressure_mTorr: float, power_W: float,
              aspect_ratio: float = 1.0) -> dict:
    """
    ARDE etch rate model: Gottscho neutral-flux balance + ion shadowing.

    AR-dependent suppression:  ER(AR) = ER0 * exp(-k_arde * AR) + ER_ion * f_ion(AR)
    where f_ion(AR) = 1/(1 + (AR/AR_crit)^2)  (ion visibility factor)

    参考: Gottscho et al. (1992) J. Vac. Sci. Technol. B 10, 2133
    """
    gas_params = {
        "Cl2/BCl3":  {"ER0_nm_min": 450.0, "k_arde": 0.06, "AR_crit": 8.0,  "ER_ion_frac": 0.45},
        "SF6/N2":    {"ER0_nm_min": 600.0, "k_arde": 0.05, "AR_crit": 10.0, "ER_ion_frac": 0.50},
        "CF4/CHF3":  {"ER0_nm_min": 300.0, "k_arde": 0.08, "AR_crit": 6.0,  "ER_ion_frac": 0.35},
        "CF4/O2":    {"ER0_nm_min": 350.0, "k_arde": 0.07, "AR_crit": 7.0,  "ER_ion_frac": 0.40},
        "HBr/Cl2":   {"ER0_nm_min": 400.0, "k_arde": 0.07, "AR_crit": 9.0,  "ER_ion_frac": 0.42},
    }
    p = gas_params.get(gas, gas_params["Cl2/BCl3"])

    # Pressure & power scaling (empirical: ER ∝ sqrt(P_rf) / sqrt(pressure))
    p_scale = math.sqrt(power_W / 2000.0) / math.sqrt(pressure_mTorr / 20.0)
    ER0 = p["ER0_nm_min"] * p_scale

    # Neutral flux suppression
    ER_neutral = ER0 * math.exp(-p["k_arde"] * max(aspect_ratio - 1.0, 0.0))

    # Ion flux (shadowed by aspect ratio)
    f_ion = 1.0 / (1.0 + (aspect_ratio / p["AR_crit"]) ** 2)
    ER_ion = ER0 * p["ER_ion_frac"] * f_ion

    ER_total = ER_neutral * (1.0 - p["ER_ion_frac"]) + ER_ion
    ard_suppression_pct = 100.0 * (1.0 - ER_total / max(ER0, 1e-9))

    return {
        "gas": gas,
        "pressure_mTorr": pressure_mTorr,
        "power_W": power_W,
        "aspect_ratio": aspect_ratio,
        "ER_nm_min": round(ER_total, 1),
        "ER_neutral_nm_min": round(ER_neutral * (1.0 - p["ER_ion_frac"]), 1),
        "ER_ion_nm_min": round(ER_ion, 1),
        "ard_suppression_pct": round(ard_suppression_pct, 1),
        "f_ion": round(f_ion, 4),
    }


def cd_bias(aspect_ratio: float, gas: str = "Cl2/BCl3",
            pressure_mTorr: float = 20.0) -> dict:
    """
    CD bias (lateral etch) vs. aspect ratio — tapered sidewall correction.

    Sidewall passivation limits lateral etch; model:
      CD_bias_nm = CD_bias0 * exp(-lambda_pass * AR)
    where lambda_pass is passivation decay constant (gas dependent).
    """
    gas_bias = {
        "Cl2/BCl3": {"CD_bias0_nm": 12.0, "lambda_pass": 0.15},
        "SF6/N2":   {"CD_bias0_nm": 18.0, "lambda_pass": 0.10},
        "CF4/CHF3": {"CD_bias0_nm":  8.0, "lambda_pass": 0.20},
        "CF4/O2":   {"CD_bias0_nm": 10.0, "lambda_pass": 0.18},
        "HBr/Cl2":  {"CD_bias0_nm":  7.0, "lambda_pass": 0.22},
    }
    p = gas_bias.get(gas, gas_bias["Cl2/BCl3"])
    # Pressure correction: higher pressure → more passivant → less bias
    p_corr = 1.0 - 0.01 * (pressure_mTorr - 20.0)
    cd_b = p["CD_bias0_nm"] * math.exp(-p["lambda_pass"] * aspect_ratio) * max(p_corr, 0.5)
    taper_angle_deg = math.degrees(math.atan(cd_b / max(aspect_ratio * 10.0, 1.0)))
    return {
        "aspect_ratio": aspect_ratio,
        "gas": gas,
        "CD_bias_nm": round(cd_b, 2),
        "taper_angle_deg": round(taper_angle_deg, 2),
        "profile": "anisotropic" if cd_b < 5.0 else "tapered",
    }


def selectivity(mat_etch: str, mat_stop: str, gas: str) -> float:
    """Return etch selectivity ratio from lookup table; interpolate with ±10% tolerance."""
    key = (mat_etch, mat_stop, gas)
    if key in SELECTIVITY_TABLE:
        return SELECTIVITY_TABLE[key]
    # Fallback: try reversed lookup
    rev_key = (mat_stop, mat_etch, gas)
    if rev_key in SELECTIVITY_TABLE:
        return round(1.0 / SELECTIVITY_TABLE[rev_key], 2)
    return 5.0  # conservative default


def uniformity_model(chamber: str, n_wafers: int = 1) -> dict:
    """
    Within-wafer etch rate non-uniformity (1σ %) from gas ring conductance model.

    Model: NU ∝ 1/sqrt(N_holes * conductance) + aging_factor * (n_wafers/10000)
    """
    chamber_params = {
        "Kiyo_C":  {"NU_base_pct": 1.2, "holes": 64,  "aging_coeff": 0.05},
        "Sense_i": {"NU_base_pct": 0.9, "holes": 80,  "aging_coeff": 0.04},
        "VECTOR":  {"NU_base_pct": 0.5, "holes": 128, "aging_coeff": 0.02},
    }
    p = chamber_params.get(chamber, chamber_params["Kiyo_C"])
    aging = p["aging_coeff"] * (n_wafers / 10000.0)
    nu = p["NU_base_pct"] + aging
    pm_3sigma = round(nu * 3.0, 2)
    return {
        "chamber": chamber,
        "n_wafers_processed": n_wafers,
        "NU_1sigma_pct": round(nu, 3),
        "NU_3sigma_pct": pm_3sigma,
        "pm_range_pct": f"±{pm_3sigma}",
        "spec_met": pm_3sigma < 5.0,
    }


def ald_saturation(precursor: str, T_C: float, dose_mTorr_s: float) -> dict:
    """
    ALD GPC from Langmuir adsorption saturation model.

    GPC(dose) = GPC_max * (1 - exp(-sigma * n_dose))
    where n_dose = dose_mTorr_s * k_conv (molecules/nm²)
    """
    if precursor not in ALD_PRECURSORS:
        precursor = "TMA"
    p = ALD_PRECURSORS[precursor]

    T_min, T_max = p["T_window_C"]
    in_window = T_min <= T_C <= T_max

    # Temperature correction: slight Arrhenius decline outside window
    if T_C < T_min:
        T_factor = math.exp(-0.02 * (T_min - T_C))
    elif T_C > T_max:
        T_factor = math.exp(-0.03 * (T_C - T_max))
    else:
        T_factor = 1.0

    # Convert dose: 1 mTorr·s ≈ 3.2e13 molecules/cm² → ~0.32 molecules/nm²
    n_dose = dose_mTorr_s * 0.32
    theta = 1.0 - math.exp(-p["sigma_nm2"] * n_dose)
    GPC = p["GPC_max_A"] * theta * T_factor

    return {
        "precursor": precursor,
        "T_C": T_C,
        "dose_mTorr_s": dose_mTorr_s,
        "theta_surface_coverage": round(theta, 4),
        "GPC_A_cycle": round(GPC, 3),
        "saturated": theta > 0.97,
        "in_ALD_window": in_window,
        "GPC_max_A": p["GPC_max_A"],
    }


def full_pipeline(feature_nm: float = 14.0, aspect_ratio: float = 10.0,
                  material: str = "Si", precursor: str = "TMA",
                  power_W: float = 2000.0, pressure_mTorr: float = 20.0) -> dict:
    """
    End-to-end etch + ALD liner pipeline:
      1. Conductor/dielectric etch (Kiyo C or Sense.i)
      2. CD bias correction
      3. Uniformity check
      4. ALD liner deposition (VECTOR)
    """
    gas_map = {"Si": "Cl2/BCl3", "W": "SF6/N2", "SiO2": "CF4/CHF3",
               "SiN": "CF4/O2", "default": "Cl2/BCl3"}
    gas = gas_map.get(material, gas_map["default"])

    er = etch_rate(gas, pressure_mTorr, power_W, aspect_ratio)
    cdb = cd_bias(aspect_ratio, gas, pressure_mTorr)
    depth_nm = feature_nm * aspect_ratio
    etch_time_s = depth_nm / er["ER_nm_min"] * 60.0
    chamber = "Kiyo_C" if material in ("Si", "W") else "Sense_i"
    uni = uniformity_model(chamber, n_wafers=5000)
    ald = ald_saturation(precursor, T_C=200, dose_mTorr_s=2.0)

    return {
        "feature_nm": feature_nm,
        "aspect_ratio": aspect_ratio,
        "material": material,
        "gas": gas,
        "ER_nm_min": er["ER_nm_min"],
        "ard_suppression_pct": er["ard_suppression_pct"],
        "CD_bias_nm": cdb["CD_bias_nm"],
        "taper_angle_deg": cdb["taper_angle_deg"],
        "etch_time_s": round(etch_time_s, 1),
        "NU_3sigma_pct": uni["NU_3sigma_pct"],
        "ALD_GPC_A_cycle": ald["GPC_A_cycle"],
        "ALD_saturated": ald["saturated"],
        "liner_cycles_for_2nm": round(20.0 / max(ald["GPC_A_cycle"], 0.1)),
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. Visualization
# ════════════════════════════════════════════════════════════════════════════

def plot_all(out_dir: str = OUT_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(17, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)
    ax = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    ar_arr = np.linspace(1, 25, 120)

    # ── Panel 0: ARDE etch rate vs. aspect ratio ──────────────────────────
    gas_list = [
        ("Cl2/BCl3", "#d62728", "Si (Cl₂/BCl₃)"),
        ("CF4/CHF3", "#2166ac", "SiO₂ (CF₄/CHF₃)"),
        ("SF6/N2",   "#ff7f0e", "W (SF₆/N₂)"),
        ("CF4/O2",   "#2ca02c", "SiN (CF₄/O₂)"),
    ]
    for gas, col, lbl in gas_list:
        er_arr = [etch_rate(gas, 20.0, 2000.0, ar)["ER_nm_min"] for ar in ar_arr]
        ax[0].plot(ar_arr, er_arr, color=col, lw=2.2, label=lbl)
    ax[0].set_xlabel("Aspect Ratio (depth/width)")
    ax[0].set_ylabel("Etch Rate [nm/min]")
    ax[0].set_title("ARDE: Etch Rate vs. Aspect Ratio\n(Lam Kiyo C / Sense.i, 2 kW, 20 mTorr)")
    ax[0].legend(fontsize=7.5)
    ax[0].grid(alpha=0.25)
    ax[0].set_xlim(1, 25)

    # ── Panel 1: CD bias vs. aspect ratio ────────────────────────────────
    for gas, col, lbl in gas_list:
        cb_arr = [cd_bias(ar, gas)["CD_bias_nm"] for ar in ar_arr]
        ax[1].plot(ar_arr, cb_arr, color=col, lw=2.2, label=lbl)
    ax[1].axhline(5.0, color="k", ls="--", lw=1.2, label="5 nm spec limit")
    ax[1].set_xlabel("Aspect Ratio")
    ax[1].set_ylabel("CD Bias [nm]")
    ax[1].set_title("CD Bias (Lateral Etch) vs. Aspect Ratio\n(Sidewall Passivation Model)")
    ax[1].legend(fontsize=7.5)
    ax[1].grid(alpha=0.25)

    # ── Panel 2: Etch selectivity bar chart ──────────────────────────────
    sel_cases = [
        ("Si/SiO₂\nCl₂/BCl₃",  selectivity("Si",   "SiO2", "Cl2/BCl3"),  "#d62728"),
        ("Si/SiN\nCl₂/BCl₃",   selectivity("Si",   "SiN",  "Cl2/BCl3"),  "#e08080"),
        ("W/TiN\nSF₆/N₂",      selectivity("W",    "TiN",  "SF6/N2"),    "#ff7f0e"),
        ("SiO₂/Si\nCF₄/CHF₃",  selectivity("SiO2", "Si",   "CF4/CHF3"), "#2166ac"),
        ("SiN/Si\nCF₄/O₂",     selectivity("SiN",  "Si",   "CF4/O2"),   "#2ca02c"),
    ]
    labels = [s[0] for s in sel_cases]
    values = [s[1] for s in sel_cases]
    colors = [s[2] for s in sel_cases]
    bars = ax[2].bar(range(len(labels)), values, color=colors, width=0.6, alpha=0.85)
    ax[2].set_xticks(range(len(labels)))
    ax[2].set_xticklabels(labels, fontsize=7.5)
    ax[2].set_ylabel("Selectivity Ratio")
    ax[2].set_title("Etch Selectivity Table\n(Lam Research Gas Chemistry)")
    ax[2].grid(axis="y", alpha=0.25)
    for b, v in zip(bars, values):
        ax[2].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, f"{v:.0f}×",
                   ha="center", va="bottom", fontsize=8)

    # ── Panel 3: Within-wafer uniformity vs. wafer count (aging) ─────────
    wfr_arr = np.linspace(0, 50000, 200)
    for cname, col in [("Kiyo_C", "#d62728"), ("Sense_i", "#2166ac"), ("VECTOR", "#2ca02c")]:
        nu_arr = [uniformity_model(cname, int(w))["NU_1sigma_pct"] for w in wfr_arr]
        ax[3].plot(wfr_arr / 1000, nu_arr, color=col, lw=2.2, label=cname.replace("_", " "))
    ax[3].axhline(5.0 / 3, color="k", ls="--", lw=1.2, label="1σ spec (3σ=5%)")
    ax[3].set_xlabel("Wafers Processed [k]")
    ax[3].set_ylabel("NU 1σ [%]")
    ax[3].set_title("WIW Uniformity vs. Chamber Aging\n(Gas Ring Conductance Model)")
    ax[3].legend(fontsize=8)
    ax[3].grid(alpha=0.25)

    # ── Panel 4: ALD saturation curves ───────────────────────────────────
    dose_arr = np.linspace(0.01, 5.0, 200)
    T_dep = 200.0
    for prec, col in [("TMA", "#2166ac"), ("TDMAT", "#d62728"), ("HCDS", "#ff7f0e"), ("WF6", "#9467bd")]:
        gpc_arr = [ald_saturation(prec, T_dep, d)["GPC_A_cycle"] for d in dose_arr]
        lbl = ALD_PRECURSORS[prec]["full_name"].split("(")[1].rstrip(")")
        ax[4].plot(dose_arr, gpc_arr, color=col, lw=2.2, label=lbl)
    ax[4].axvline(2.0, color="gray", ls=":", lw=1.2, label="Typical dose (2 mTorr·s)")
    ax[4].set_xlabel("Dose [mTorr·s]")
    ax[4].set_ylabel("GPC [Å/cycle]")
    ax[4].set_title("ALD Saturation Curves\n(Lam VECTOR, Langmuir Model, 200 °C)")
    ax[4].legend(fontsize=7.5)
    ax[4].grid(alpha=0.25)

    # ── Panel 5: Pipeline summary table ─────────────────────────────────
    ax[5].axis("off")
    materials = ["Si", "W", "SiO2", "SiN"]
    col_labels = ["Material", "AR", "ER\n[nm/min]", "ARDE\n[%]", "CD Bias\n[nm]", "Etch\nTime [s]"]
    tdata = []
    for mat in materials:
        res = full_pipeline(feature_nm=14, aspect_ratio=10, material=mat)
        tdata.append([mat, "10×",
                      f"{res['ER_nm_min']:.0f}",
                      f"{res['ard_suppression_pct']:.1f}",
                      f"{res['CD_bias_nm']:.2f}",
                      f"{res['etch_time_s']:.1f}"])
    tbl = ax[5].table(cellText=tdata, colLabels=col_labels,
                      cellLoc="center", loc="center",
                      bbox=[0.0, 0.15, 1.0, 0.80])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2166ac")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f0f4f8")
    ax[5].set_title("Etch Pipeline Summary (AR=10, 14 nm feature)\nLam Research — Kiyo C / Sense.i")

    fig.suptitle(
        "Lam Research — Plasma Etch & ALD Equipment Physics\n"
        "Kiyo C (conductor) · Sense.i (dielectric) · VECTOR (ALD)",
        fontsize=12, fontweight="bold",
    )
    out = os.path.join(out_dir, "lam_research.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Lam Research model → {out}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="Lam Research plasma etch & ALD model")
    ap.add_argument("--etch", action="store_true", help="Print etch pipeline summary")
    ap.add_argument("--ald",  action="store_true", help="Print ALD saturation summary")
    ap.add_argument("--material", default="Si",  choices=["Si", "W", "SiO2", "SiN"])
    ap.add_argument("--precursor", default="TMA", choices=list(ALD_PRECURSORS.keys()))
    ap.add_argument("--ar", type=float, default=10.0, help="Aspect ratio for etch pipeline")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    if args.etch:
        res = full_pipeline(feature_nm=14, aspect_ratio=args.ar, material=args.material)
        print("\nLam Research Etch Pipeline Summary")
        print("=" * 40)
        for k, v in res.items():
            print(f"  {k:<30} {v}")

    if args.ald:
        for dose in [0.5, 1.0, 2.0, 3.0, 5.0]:
            r = ald_saturation(args.precursor, T_C=200, dose_mTorr_s=dose)
            print(f"  dose={dose:.1f} mTorr·s  GPC={r['GPC_A_cycle']:.3f} Å  "
                  f"θ={r['theta_surface_coverage']:.3f}  sat={r['saturated']}")

    if not args.no_plot:
        plot_all()


if __name__ == "__main__":
    main()
