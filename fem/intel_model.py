"""
Intel — 18A RibbonFET + Advanced Packaging Model
==========================================================
Covers Intel 18A (RibbonFET + PowerVia), Intel 3, EMIB die-to-die interconnect,
and Foveros 3D stacking.

Key physics:
  - RibbonFET multi-nanosheet Id–Vd (Virtual Source model)
  - PowerVia backside power delivery network resistance
  - EMIB microbump bandwidth (SerDes + bump parasitics)
  - Foveros 3D stack junction temperature (thermal stack model)
  - Murphy yield model (node-dependent D0)

実行:
    python fem/intel_model.py
    python fem/intel_model.py --transistor --node 18A
    python fem/intel_model.py --packaging --type EMIB

参考文献:
    Natarajan et al. (2023) IEDM — Intel 18A RibbonFET / PowerVia
    Naous et al. (2021) ISSCC — EMIB interconnect bandwidth
    Elsherbini et al. (2018) ECTC — Foveros 3D thermal model
    Murphy (1964) Proc. IEEE — die yield model
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

# Physical constants
q_C   = 1.602e-19   # electron charge [C]
k_B   = 1.381e-23   # Boltzmann constant [J/K]
T_K   = 300.0       # room temperature [K]
eps0  = 8.854e-12   # permittivity of vacuum [F/m]

# ════════════════════════════════════════════════════════════════════════════
# 1. Process Node Parameter Tables
# ════════════════════════════════════════════════════════════════════════════

INTEL_NODES = {
    "Intel_18A": {
        "L_g_nm": 6.0,
        "W_ns_nm": 5.0,          # nanosheet width
        "n_sheet": 3,             # nanosheets per stack
        "t_ox_nm": 0.8,           # EOT
        "Vdd_V": 0.7,
        "density_MT_mm2": 200.0,  # MTr/mm²
        "D0_per_cm2": 0.10,       # baseline defect density
        "node_nm": 1.8,
        "color": "#2166ac",
        "ref": "Natarajan et al. IEDM 2023",
    },
    "Intel_3": {
        "L_g_nm": 18.0,
        "W_ns_nm": 7.0,
        "n_sheet": 3,
        "t_ox_nm": 1.2,
        "Vdd_V": 0.85,
        "density_MT_mm2": 100.0,
        "D0_per_cm2": 0.08,
        "node_nm": 3.0,
        "color": "#d62728",
        "ref": "Intel Process & Packaging Technology Overview 2023",
    },
    "Intel_7": {
        "L_g_nm": 34.0,
        "W_ns_nm": 0.0,           # FinFET (not nanosheet)
        "n_sheet": 0,
        "t_ox_nm": 1.5,
        "Vdd_V": 0.95,
        "density_MT_mm2": 60.0,
        "D0_per_cm2": 0.06,
        "node_nm": 7.0,
        "color": "#ff7f0e",
        "ref": "Intel Technology Briefing 2021",
    },
}

PACKAGE_TYPES = {
    "EMIB": {
        "bump_pitch_um": 55.0,
        "n_io_per_mm": 91.0,     # bumps/mm at die edge
        "freq_GHz": 8.0,
        "voltage_V": 0.9,
        "energy_pJ_bit": 0.5,
        "color": "#2166ac",
        "ref": "Naous et al. ISSCC 2021",
    },
    "Foveros": {
        "bump_pitch_um": 10.0,
        "n_io_per_mm": 1000.0,
        "freq_GHz": 4.0,
        "voltage_V": 0.8,
        "energy_pJ_bit": 0.2,
        "color": "#d62728",
        "ref": "Elsherbini et al. ECTC 2018",
    },
    "Foveros_Direct": {
        "bump_pitch_um": 3.0,
        "n_io_per_mm": 3300.0,
        "freq_GHz": 4.0,
        "voltage_V": 0.75,
        "energy_pJ_bit": 0.1,
        "color": "#2ca02c",
        "ref": "Intel Architecture Day 2021",
    },
    "ODI": {
        "bump_pitch_um": 36.0,
        "n_io_per_mm": 139.0,
        "freq_GHz": 16.0,
        "voltage_V": 0.9,
        "energy_pJ_bit": 1.0,
        "color": "#9467bd",
        "ref": "Intel ECTC 2023 (ODI/MDIO)",
    },
}

# ════════════════════════════════════════════════════════════════════════════
# 2. Core Physics Functions
# ════════════════════════════════════════════════════════════════════════════

def ribbonfet_ids(Vgs: float, Vds: float, node: str = "Intel_18A") -> dict:
    """
    Multi-nanosheet RibbonFET drain current using Virtual Source model.

    I_D = n_sheet * W_ns * (Q_i0 * v_T * F_S) * [1 - exp(-Vds/V_dsat)]
    where Q_i0 ∝ Cox*(Vgs-Vt), v_T = thermal velocity, F_S = source injection factor.

    参考: Jeong et al. (2013) IEEE TED — Virtual Source model for GAA FETs
    """
    p = INTEL_NODES.get(node, INTEL_NODES["Intel_18A"])
    L_g = p["L_g_nm"] * 1e-9
    W_ns = p["W_ns_nm"] * 1e-9
    n_s = max(p["n_sheet"], 1)
    t_ox = p["t_ox_nm"] * 1e-9
    Vdd = p["Vdd_V"]

    eps_ox = 3.9 * eps0
    Cox = eps_ox / t_ox           # [F/m²]
    Vt = 0.22                     # threshold voltage [V]
    v_T = 1.2e5                   # thermal velocity Si at 300K [m/s]
    mu_eff = 0.02                 # effective mobility [m²/Vs] (inversion layer)
    V_dsat = max(Vgs - Vt, 0.01) * 0.5   # saturation voltage

    Qinv = Cox * max(Vgs - Vt, 0.0)
    # Source charge injection (virtual source)
    F_S = mu_eff * Qinv / (v_T * L_g + mu_eff * Qinv / (v_T + 1e-15))
    I_sat = n_s * W_ns * Qinv * v_T * 0.5
    I_D = I_sat * (1.0 - math.exp(-max(Vds, 0.0) / max(V_dsat, 0.01)))
    I_D_uA_um = I_D / max(n_s * W_ns, 1e-15) * 1e6 / 1e-6   # [µA/µm] normalized

    return {
        "node": node,
        "Vgs_V": Vgs,
        "Vds_V": Vds,
        "n_sheet": n_s,
        "Id_uA": round(I_D * 1e6, 2),
        "Id_uA_um": round(I_D_uA_um, 1),
        "Vt_V": Vt,
        "on_state": Vgs >= Vt,
    }


def powervia_resistance(via_pitch_um: float = 0.48, via_d_nm: float = 220.0,
                        fill_material: str = "W") -> dict:
    """
    Backside PDN via resistance (PowerVia).

    R_via = rho * L / A  where L = die thickness, A = pi*(d/2)²
    R_total per mm² = R_via / (n_vias_per_mm2)

    参考: Natarajan et al. IEDM 2023
    """
    rho_map = {"W": 5.3e-8, "Cu": 1.7e-8, "Ru": 7.1e-8}  # [Ω·m]
    rho = rho_map.get(fill_material, 5.3e-8)

    d = via_d_nm * 1e-9          # via diameter [m]
    L = 775e-9                   # die thinning ~775 nm (backside)
    A = math.pi * (d / 2.0) ** 2
    R_single = rho * L / A       # single via resistance [Ω]

    n_vias_per_mm2 = 1.0 / (via_pitch_um * 1e-3) ** 2
    R_parallel = R_single / n_vias_per_mm2 * 1e-6  # [Ω·mm²] → [mΩ·mm²]

    return {
        "via_pitch_um": via_pitch_um,
        "via_d_nm": via_d_nm,
        "fill": fill_material,
        "R_single_ohm": round(R_single, 4),
        "n_vias_per_mm2": round(n_vias_per_mm2, 1),
        "R_pdn_mohm_mm2": round(R_parallel * 1000, 4),
        "IR_drop_mV_at_100mA_mm2": round(R_parallel * 1000 * 0.1, 3),
    }


def emib_bandwidth(pkg_type: str = "EMIB", die_edge_mm: float = 10.0) -> dict:
    """
    Die-to-die bandwidth via EMIB / Foveros bump interface.

    BW_total = n_io * freq * efficiency
    where n_io = n_io_per_mm * die_edge_mm * utilization
    """
    p = PACKAGE_TYPES.get(pkg_type, PACKAGE_TYPES["EMIB"])
    utilization = 0.80
    n_io = p["n_io_per_mm"] * die_edge_mm * utilization
    BW_Gbps = n_io * p["freq_GHz"]
    BW_TBs  = BW_Gbps / 8000.0          # [TB/s]
    power_W = n_io * p["freq_GHz"] * p["energy_pJ_bit"] * 1e-3 / 8.0

    return {
        "pkg_type": pkg_type,
        "bump_pitch_um": p["bump_pitch_um"],
        "n_io_total": round(n_io),
        "BW_Gbps": round(BW_Gbps, 1),
        "BW_TBs": round(BW_TBs, 3),
        "power_W": round(power_W, 2),
        "energy_pJ_bit": p["energy_pJ_bit"],
    }


def foveros_thermal(P_top_W: float = 15.0, A_die_mm2: float = 35.0,
                    TIM_k_W_mK: float = 8.0, bond_thickness_um: float = 10.0) -> dict:
    """
    3D stack junction temperature: top die + TIM + bottom die + package.

    T_j = T_ambient + R_th_total * P
    R_th = bond_layer / (k_TIM * A) + R_package (empirical 0.05 °C/W/mm²)

    参考: Elsherbini et al. ECTC 2018
    """
    A_m2 = A_die_mm2 * 1e-6
    R_bond = bond_thickness_um * 1e-6 / (TIM_k_W_mK * A_m2)  # [°C/W]
    R_pkg  = 0.08 / A_die_mm2                                  # empirical package spreading
    R_total = R_bond + R_pkg
    T_ambient = 35.0   # °C (ambient in thermal test fixture)
    T_j = T_ambient + R_total * P_top_W

    return {
        "P_top_W": P_top_W,
        "A_die_mm2": A_die_mm2,
        "TIM_k_W_mK": TIM_k_W_mK,
        "bond_thickness_um": bond_thickness_um,
        "R_bond_C_W": round(R_bond, 4),
        "R_total_C_W": round(R_total, 4),
        "T_junction_C": round(T_j, 1),
        "hotspot_ok": T_j < 100.0,
    }


def intel_yield(node: str = "Intel_18A", die_area_mm2: float = 100.0) -> dict:
    """
    Murphy yield model: Y = [(1 - exp(-D0*A)) / (D0*A)]²

    参考: Murphy (1964) Proc. IEEE — semiconductor yield estimation
    """
    p = INTEL_NODES.get(node, INTEL_NODES["Intel_18A"])
    D0 = p["D0_per_cm2"]
    A_cm2 = die_area_mm2 / 100.0
    val = (1.0 - math.exp(-D0 * A_cm2)) / (D0 * A_cm2 + 1e-15)
    Y = val ** 2
    dies_per_300 = int((math.pi * 150.0 ** 2 - math.pi * 150.0 * math.sqrt(2 * die_area_mm2))
                       / die_area_mm2)
    good_dies = int(dies_per_300 * Y)

    return {
        "node": node,
        "die_area_mm2": die_area_mm2,
        "D0_per_cm2": D0,
        "yield_pct": round(Y * 100, 1),
        "dies_per_300mm_wafer": dies_per_300,
        "good_dies_per_wafer": good_dies,
        "density_MT_mm2": p["density_MT_mm2"],
    }


def full_pipeline(node: str = "Intel_18A", pkg_type: str = "Foveros",
                  die_area_mm2: float = 80.0, P_top_W: float = 20.0) -> dict:
    """Complete Intel process + packaging performance summary."""
    trans = ribbonfet_ids(Vgs=0.7, Vds=0.65, node=node)
    pwr   = powervia_resistance(via_pitch_um=0.48, via_d_nm=220, fill_material="W")
    bw    = emib_bandwidth(pkg_type=pkg_type, die_edge_mm=10.0)
    thm   = foveros_thermal(P_top_W=P_top_W, A_die_mm2=die_area_mm2)
    yld   = intel_yield(node=node, die_area_mm2=die_area_mm2)

    return {
        "node": node,
        "pkg_type": pkg_type,
        "Id_uA_um": trans["Id_uA_um"],
        "R_pdn_mohm_mm2": pwr["R_pdn_mohm_mm2"],
        "BW_TBs": bw["BW_TBs"],
        "T_junction_C": thm["T_junction_C"],
        "yield_pct": yld["yield_pct"],
        "good_dies_per_wafer": yld["good_dies_per_wafer"],
        "density_MT_mm2": yld["density_MT_mm2"],
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. Visualization
# ════════════════════════════════════════════════════════════════════════════

def plot_all(out_dir: str = OUT_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(17, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)
    ax = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    # ── Panel 0: RibbonFET Id–Vd family curves ───────────────────────────
    Vds_arr = np.linspace(0, 0.7, 120)
    for node_name, col in [("Intel_18A", "#2166ac"), ("Intel_3", "#d62728"), ("Intel_7", "#ff7f0e")]:
        node_nm = INTEL_NODES[node_name]["node_nm"]
        for Vgs, ls in [(0.7, "-"), (0.5, "--"), (0.3, ":")]:
            ids = [ribbonfet_ids(Vgs, Vd, node_name)["Id_uA_um"] for Vd in Vds_arr]
            lbl = f"{node_name.replace('_',' ')} Vgs={Vgs:.1f}V" if ls == "-" else None
            ax[0].plot(Vds_arr, ids, color=col, ls=ls, lw=2.0, label=lbl)
    ax[0].set_xlabel("Vds [V]")
    ax[0].set_ylabel("Id [µA/µm]")
    ax[0].set_title("RibbonFET Id–Vd (Virtual Source)\n18A · Intel 3 · Intel 7")
    ax[0].legend(fontsize=7.5)
    ax[0].grid(alpha=0.25)

    # ── Panel 1: PowerVia PDN resistance vs. via pitch ───────────────────
    pitch_arr = np.linspace(0.3, 2.0, 100)
    for d_nm, col, lbl in [(220, "#2166ac", "W 220nm"), (300, "#d62728", "W 300nm"), (200, "#2ca02c", "Cu 200nm")]:
        mat = "Cu" if "Cu" in lbl else "W"
        r_arr = [max(powervia_resistance(p, d_nm, mat)["R_pdn_mohm_mm2"], 1e-6) for p in pitch_arr]
        ax[1].semilogy(pitch_arr, r_arr, color=col, lw=2.2, label=lbl)
    ax[1].axhline(0.5, color="k", ls="--", lw=1.2, label="0.5 mΩ·mm² spec")
    ax[1].set_xlabel("Via Pitch [µm]")
    ax[1].set_ylabel("R_PDN [mΩ·mm²]")
    ax[1].set_title("PowerVia PDN Resistance vs. Via Pitch\n(Backside Power Delivery)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.25)

    # ── Panel 2: Die-to-die bandwidth by package type ────────────────────
    pkg_list = list(PACKAGE_TYPES.keys())
    bw_vals = [emib_bandwidth(pk, die_edge_mm=10.0)["BW_TBs"] for pk in pkg_list]
    cols    = [PACKAGE_TYPES[pk]["color"] for pk in pkg_list]
    bars = ax[2].bar(range(len(pkg_list)), bw_vals, color=cols, width=0.6, alpha=0.85)
    ax[2].set_xticks(range(len(pkg_list)))
    ax[2].set_xticklabels(pkg_list, fontsize=8.5)
    ax[2].set_ylabel("Bandwidth [TB/s]")
    ax[2].set_title("Die-to-Die Bandwidth by Package\n(EMIB / Foveros / ODI, 10mm edge)")
    ax[2].grid(axis="y", alpha=0.25)
    for b, v in zip(bars, bw_vals):
        ax[2].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.002,
                   f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    # ── Panel 3: Foveros thermal: T_junction vs. top-die power ───────────
    P_arr = np.linspace(5, 80, 100)
    for A, col, lbl in [(35, "#2166ac", "A=35mm²"), (70, "#d62728", "A=70mm²"), (120, "#2ca02c", "A=120mm²")]:
        Tj_arr = [foveros_thermal(P, A)["T_junction_C"] for P in P_arr]
        ax[3].plot(P_arr, Tj_arr, color=col, lw=2.2, label=lbl)
    ax[3].axhline(100.0, color="k", ls="--", lw=1.2, label="100°C limit")
    ax[3].set_xlabel("Top Die Power [W]")
    ax[3].set_ylabel("T_junction [°C]")
    ax[3].set_title("Foveros 3D Stack Thermal\n(TIM k=8 W/m·K, T_amb=35°C)")
    ax[3].legend(fontsize=8)
    ax[3].grid(alpha=0.25)

    # ── Panel 4: Murphy yield vs. die area ───────────────────────────────
    area_arr = np.linspace(10, 400, 200)
    for node_name, col in [("Intel_18A", "#2166ac"), ("Intel_3", "#d62728"), ("Intel_7", "#ff7f0e")]:
        y_arr = [intel_yield(node_name, a)["yield_pct"] for a in area_arr]
        ax[4].plot(area_arr, y_arr, color=col, lw=2.2,
                   label=f"Intel {INTEL_NODES[node_name]['node_nm']}nm "
                         f"(D₀={INTEL_NODES[node_name]['D0_per_cm2']}/cm²)")
    ax[4].set_xlabel("Die Area [mm²]")
    ax[4].set_ylabel("Yield [%]")
    ax[4].set_title("Murphy Yield Model vs. Die Area\n(300mm wafer, defect density D₀)")
    ax[4].legend(fontsize=8)
    ax[4].grid(alpha=0.25)
    ax[4].set_ylim(0, 100)

    # ── Panel 5: Node comparison table ───────────────────────────────────
    ax[5].axis("off")
    col_labels = ["Node", "Lg [nm]", "Density\n[MT/mm²]", "Vdd [V]",
                  "Pkg", "BW [TB/s]", "Yield\n100mm²"]
    tdata = []
    for nn, pt in [("Intel_18A", "Foveros_Direct"), ("Intel_3", "Foveros"), ("Intel_7", "EMIB")]:
        res = full_pipeline(node=nn, pkg_type=pt, die_area_mm2=100.0, P_top_W=15.0)
        nd  = INTEL_NODES[nn]
        tdata.append([nn.replace("_", " "),
                      f"{nd['L_g_nm']:.0f}",
                      f"{nd['density_MT_mm2']:.0f}",
                      f"{nd['Vdd_V']:.2f}",
                      pt.replace("_", " "),
                      f"{res['BW_TBs']:.3f}",
                      f"{res['yield_pct']:.1f}%"])
    tbl = ax[5].table(cellText=tdata, colLabels=col_labels,
                      cellLoc="center", loc="center",
                      bbox=[0.0, 0.15, 1.0, 0.80])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2166ac")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f0f4f8")
    ax[5].set_title("Intel Process Node Comparison\n18A · Intel 3 · Intel 7")

    fig.suptitle(
        "Intel — 18A RibbonFET + Advanced Packaging Physics\n"
        "RibbonFET · PowerVia · EMIB · Foveros 3D · Murphy Yield",
        fontsize=12, fontweight="bold",
    )
    out = os.path.join(out_dir, "intel_process.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Intel model → {out}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="Intel 18A / Advanced Packaging model")
    ap.add_argument("--transistor", action="store_true", help="Print RibbonFET summary")
    ap.add_argument("--packaging",  action="store_true", help="Print packaging bandwidth")
    ap.add_argument("--node",    default="Intel_18A", choices=list(INTEL_NODES.keys()))
    ap.add_argument("--type",    default="Foveros",   choices=list(PACKAGE_TYPES.keys()))
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    if args.transistor:
        print("\nRibbonFET Summary")
        print("=" * 40)
        res = ribbonfet_ids(Vgs=0.7, Vds=0.65, node=args.node)
        for k, v in res.items():
            print(f"  {k:<30} {v}")

    if args.packaging:
        print("\nPackaging Bandwidth Summary")
        print("=" * 40)
        res = emib_bandwidth(pkg_type=args.type, die_edge_mm=10.0)
        for k, v in res.items():
            print(f"  {k:<30} {v}")

    if not args.no_plot:
        plot_all()


if __name__ == "__main__":
    main()
